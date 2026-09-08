import math
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import requests


class KRAFeatureEngineV2:
  """KRA AI Quant Feature Engineering & Probability Calibration Engine V2

  주요 기능:
  1. 베이지안 승률 평활화 (표본 수에 따른 신뢰도 보정)
  2. 필드 내 Z-Score 상대평가 (부담중량, 레이팅 등)
  3. 체중 변동 연속 비선형 가우시안 감점 (±10kg 단일 패널티 대체)
  4. 주로 상태-마필 특성 상호작용 (Interaction Feature)
  5. 엔트로피(Entropy) & Margin 기반 레이스 난이도 (PASS/A/B/C) 산출
  6. Temperature Scaling을 통한 Softmax Calibration
  """

  def __init__(self, service_key: str):
    self.service_key = service_key
    self.endpoints = {
        "entry_sheet": "https://apis.data.go.kr/B551015/API26_2/entrySheet_2",
        "track_info": "https://apis.data.go.kr/B551015/API189_1/Track_1",
        "horse_weight": (
            "https://apis.data.go.kr/B551015/API25_1/entryHorseWeightInfo_1"
        ),
        "jockey_result": (
            "https://apis.data.go.kr/B551015/jkyresult/getjkyresult"
        ),
        "total_record": (
            "https://apis.data.go.kr/B551015/API27_1/totalRecord_1"
        ),
        "race_result": "https://apis.data.go.kr/B551015/API156/raceRsutDtl",
    }

  def fetch_raw_api(self, url: str, params: dict) -> tuple:
    """공통 KRA API 호출 모듈"""
    full_params = {
        "ServiceKey": self.service_key,
        "serviceKey": self.service_key,
        "pageNo": "1",
        "numOfRows": "100",
        "_type": "xml",
        **params,
    }
    try:
      res = requests.get(url, params=full_params, timeout=6)
      if res.status_code != 200:
        return None, f"HTTP {res.status_code}"

      root = ET.fromstring(res.content)
      err_msg = root.findtext(".//errMsg") or root.findtext(".//returnAuthMsg")
      if err_msg:
        return None, f"API Auth Error: {err_msg}"

      res_code = root.findtext(".//resultCode")
      if res_code in ["00", "0", "NORMAL_SERVICE", "OK"]:
        items = root.findall(".//item")
        if items:
          return (
              pd.DataFrame([
                  {child.tag: child.text for child in item} for item in items
              ]),
              None,
          )
        return pd.DataFrame(), "No Data"
      return None, f"API Error [{res_code}]"
    except Exception as e:
      return None, f"Connection Failure: {str(e)}"

  # ------------------------------------------------------------------
  # Feature Engineering 핵심 수학/통계 연산
  # ------------------------------------------------------------------

  @staticmethod
  def bayesian_smoothed_win_rate(
      win_cnt: float, total_cnt: float, field_mean: float, m_confidence: float = 15.0
  ) -> float:
    """[① 베이지안 승률 평활화]

    표본 수가 적을 때(예: 2전 1승 = 50%) 승률이 과도하게 튀는 현상을 방지.
    m_confidence: 사전 신뢰 표본수 (기본값 15전)
    """
    if total_cnt <= 0:
      return field_mean
    return (win_cnt + m_confidence * field_mean) / (total_cnt + m_confidence)

  @staticmethod
  def compute_z_score(series: pd.Series) -> pd.Series:
    """[③ 상대평가 Z-Score]

    해당 경주 필드 내 평균과 표준편차를 기준으로 상대적 우위 산출.
    """
    std = series.std(ddof=0)
    if pd.isna(std) or std < 1e-6:
      return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std

  @staticmethod
  def gaussian_weight_penalty(diff_weight: float) -> float:
    """[③ 체중변화 연속 비선형 가우시안 패널티]

    ±10kg 단일 하드컷 대신 연속적인 패널티 함수 적용.
    0kg 근처는 패널티 없음, 변동폭이 커질수록 감점 가속.
    """
    sigma = 6.0  # 표준편차 6kg 기준
    # 0kg 변동 시 0.0, ±12kg 변동 시 약 -0.4 감점
    penalty = 1.0 - math.exp(-((diff_weight / sigma) ** 2) / 2.0)
    return -float(penalty) * 0.5

  @staticmethod
  def calculate_race_difficulty(probs: np.ndarray) -> tuple:
    """[④ 레이스 난이도 & Entropy 측정]

    TOP1 확률, Margin(1위-2위 격차), Shannon Entropy를 종합하여 경주 난이도 측정.
    """
    probs = np.clip(probs, 1e-12, 1.0)
    probs = probs / probs.sum()

    sorted_p = np.sort(probs)[::-1]
    top1_p = sorted_p[0]
    top2_p = sorted_p[1] if len(sorted_p) > 1 else 0.0
    gap = top1_p - top2_p

    # 샤논 엔트로피 (정보 무질서도: 클수록 초혼전)
    entropy = -np.sum(probs * np.log2(probs))
    max_entropy = np.log2(len(probs))
    norm_entropy = (
        entropy / max_entropy if max_entropy > 0 else 1.0
    )  # 0~1 정규화

    # 등급 결정
    if top1_p >= 0.25 and gap >= 0.07 and norm_entropy <= 0.85:
      difficulty_grade = "🟢 A급 (명확한 축마)"
      bet_recommend = True
    elif top1_p >= 0.18 and gap >= 0.03:
      difficulty_grade = "🟡 B급 (중혼전 - 엄격 선택)"
      bet_recommend = True
    else:
      difficulty_grade = "🔴 C급 (초혼전 / PASS 권장)"
      bet_recommend = False

    return difficulty_grade, bet_recommend, top1_p, gap, norm_entropy

  # ------------------------------------------------------------------
  # 메인 파이프라인 수신 & 정제
  # ------------------------------------------------------------------

  def build_feature_matrix(
      self,
      meet_code: str,
      target_date: str,
      selected_race: str,
      temperature: float = 1.5,
  ):
    """V2 퀀트 피처 매트릭스 생성 및 확률 도출 파이프라인

    temperature: Softmax 온도 파라미터 (스케일 과열 방지 및 Calibration 조절)
    """
    # 1. 원본 API 수신
    df_entry, err = self.fetch_raw_api(
        self.endpoints["entry_sheet"],
        {"meet": meet_code, "rc_date": target_date},
    )
    if err or df_entry is None or df_entry.empty:
      return None, err or "출전표 데이터 없음", None

    # 해당 경주 필터링
    if "rcNo" in df_entry.columns:
      df = df_entry[df_entry["rcNo"].astype(str) == str(selected_race)].copy()
    else:
      df = df_entry.copy()

    if df.empty:
      return None, f"{selected_race}경주 데이터를 찾을 수 없습니다.", None

    df_track, _ = self.fetch_raw_api(
        self.endpoints["track_info"],
        {
            "meet": meet_code,
            "rc_date_fr": target_date,
            "rc_date_to": target_date,
        },
    )
    df_weight, _ = self.fetch_raw_api(
        self.endpoints["horse_weight"],
        {"meet": meet_code, "rc_date": target_date},
    )
    df_jockey, _ = self.fetch_raw_api(
        self.endpoints["jockey_result"], {"meet": meet_code}
    )

    # 2. 보조 데이터 매핑 딕셔너리 구축
    jockey_win_y_map, jockey_qu_y_map = {}, {}
    if (
        df_jockey is not None
        and not df_jockey.empty
        and "jkName" in df_jockey.columns
    ):
      if "winRateyear" in df_jockey.columns:
        jockey_win_y_map = dict(
            zip(
                df_jockey["jkName"],
                pd.to_numeric(
                    df_jockey["winRateyear"].astype(str).str.replace("%", ""),
                    errors="coerce",
                ).fillna(0.0)
                / 100.0,
            )
        )
      if "quRateyear" in df_jockey.columns:
        jockey_qu_y_map = dict(
            zip(
                df_jockey["jkName"],
                pd.to_numeric(
                    df_jockey["quRateyear"].astype(str).str.replace("%", ""),
                    errors="coerce",
                ).fillna(0.0)
                / 100.0,
            )
        )

    weight_diff_map = {}
    if (
        df_weight is not None
        and not df_weight.empty
        and "chulNo" in df_weight.columns
    ):
      if "wgHrDiff" in df_weight.columns:
        weight_diff_map = dict(
            zip(
                df_weight["chulNo"].astype(str),
                pd.to_numeric(df_weight["wgHrDiff"], errors="coerce").fillna(
                    0.0
                ),
            )
        )

    water_percent = 4.0
    if (
        df_track is not None
        and not df_track.empty
        and "waterPercent" in df_track.columns
    ):
      try:
        water_percent = float(df_track.iloc[0].get("waterPercent", 4.0))
      except ValueError:
        water_percent = 4.0

    # 3. 데이터 원본 파싱 및 기본 수치 변환
    df["ord1CntY"] = pd.to_numeric(df.get("ord1CntY", 0), errors="coerce").fillna(
        0.0
    )
    df["rcCntY"] = pd.to_numeric(df.get("rcCntY", 0), errors="coerce").fillna(
        0.0
    )
    df["rating_num"] = pd.to_numeric(df.get("rating", 0), errors="coerce").fillna(
        0.0
    )
    df["budam_num"] = pd.to_numeric(
        df.get("wgBudam", df.get("handyCap", 0)), errors="coerce"
    ).fillna(0.0)

    # 필드 평균 마필 승률
    field_mean_hr_win = (
        df["ord1CntY"].sum() / max(df["rcCntY"].sum(), 1.0)
        if df["rcCntY"].sum() > 0
        else 0.10
    )

    # ------------------------------------------------------------------
    # 4. Feature Extraction & Normalization
    # ------------------------------------------------------------------

    # [Feature 1] 마필 승률 베이지안 평활화 (Bayesian Smoothed Win Rate)
    df["feat_hr_smoothed_win"] = [
        self.bayesian_smoothed_win_rate(
            r["ord1CntY"], r["rcCntY"], field_mean_hr_win, m_confidence=12.0
        )
        for _, r in df.iterrows()
    ]

    # [Feature 2] 기수 최근 1년 종합 성적 (Win 70% + Qu 30%)
    df["feat_jk_score"] = [
        (jockey_win_y_map.get(str(r.get("jkName", "")), 0.0) * 0.7)
        + (jockey_qu_y_map.get(str(r.get("jkName", "")), 0.0) * 0.3)
        for _, r in df.iterrows()
    ]

    # [Feature 3] 부담중량 Z-Score (상대평가: 무거울수록 마이너스)
    # Z-score가 +1.5이면 평균보다 1.5표준편차 무거우므로 불리
    df["feat_budam_z"] = -1.0 * self.compute_z_score(df["budam_num"])

    # [Feature 4] 레이팅 Z-Score (상대평가: 높을수록 유리)
    df["feat_rating_z"] = self.compute_z_score(df["rating_num"])

    # [Feature 5] 체중 변동 연속 비선형 가우시안 감점
    df["feat_weight_penalty"] = [
        self.gaussian_weight_penalty(
            weight_diff_map.get(str(r.get("chulNo", "")), 0.0)
        )
        for _, r in df.iterrows()
    ]

    # [Feature 6] 주로 함수율 상호작용 (Environment Interaction)
    # 주로가 젖어있을 때(함수율>=10%) 선입/추입 능력이 있는 High-Rating 마필에 보너스 가산
    df["feat_track_interaction"] = (
        0.25 * df["feat_rating_z"] if water_percent >= 10.0 else 0.0
    )

    # ------------------------------------------------------------------
    # 5. Composite Score & Calibrated Softmax
    # ------------------------------------------------------------------

    # 가중치 결합 (모든 Z-Score 및 Probability Feature 스케일 균형 조정)
    df["composite_logits"] = (
        (df["feat_hr_smoothed_win"] * 2.0)
        + (df["feat_jk_score"] * 1.8)
        + (df["feat_rating_z"] * 0.8)
        + (df["feat_budam_z"] * 0.5)
        + df["feat_weight_penalty"]
        + df["feat_track_interaction"]
    )

    # Temperature Scaling 적용 Softmax
    # temperature가 높을수록 확률 산출이 완화되어 과도한 마필 쏠림 방지 (Calibration)
    scaled_logits = df["composite_logits"] / temperature
    exp_logits = np.exp(scaled_logits - scaled_logits.max())
    df["AI_prob"] = exp_logits / exp_logits.sum()
    df["AI_승률(%)"] = (df["AI_prob"] * 100).round(1)

    # 예측 순위
    df["AI_예측순위"] = (
        df["AI_prob"].rank(ascending=False, method="min").astype(int)
    )
    sorted_df = df.sort_values(by="AI_예측순위").reset_index(drop=True)

    # ------------------------------------------------------------------
    # 6. Race Difficulty & Decision
    # ------------------------------------------------------------------
    grade, bet_rec, top1_p, gap, entropy = self.calculate_race_difficulty(
        sorted_df["AI_prob"].values
    )

    metrics_summary = {
        "difficulty_grade": grade,
        "bet_recommend": bet_rec,
        "top1_prob": round(top1_p * 100, 1),
        "gap_p": round(gap * 100, 1),
        "normalized_entropy": round(entropy, 3),
        "water_percent": water_percent,
        "total_horses": len(sorted_df),
    }

    return sorted_df, metrics_summary, None


# ======================================================================
# 3. 과거 경주 Walk-Forward 백테스트 모듈
# ======================================================================
def run_walk_forward_backtest(
    service_key: str, race_dates: list, meet_code: str = "1"
):
  """과거 날짜 시퀀스를 시간순으로 테스트하여 백테스팅 검증"""
  engine = KRAFeatureEngineV2(service_key)
  results = []

  print("\n" + "=" * 70)
  print("🚀 [Walk-Forward Backtest] V2 퀀트 파이프라인 과거 데이터 검증")
  print("=" * 70)

  for r_date in race_dates:
    for r_no in range(1, 11):  # 1~10경주 순회
      df_res, summary, err = engine.build_feature_matrix(
          meet_code, r_date, str(r_no)
      )
      if err or df_res is None or df_res.empty:
        continue

      top1 = df_res.iloc[0]
      results.append({
          "date": r_date,
          "race_no": r_no,
          "grade": summary["difficulty_grade"],
          "recommend": summary["bet_recommend"],
          "top1_horse": top1.get("hrName"),
          "top1_no": top1.get("chulNo"),
          "top1_prob": summary["top1_prob"],
          "gap": summary["gap_p"],
          "entropy": summary["normalized_entropy"],
      })

  df_bt = pd.DataFrame(results)
  print(f"✅ 총 {len(df_bt)}개 경주 백테스트 파이프라인 정제 완료\n")
  return df_bt


if __name__ == "__main__":
  # 테스트 실행 예시 (키 설정 후 사용)
  TEST_KEY = "YOUR_KRA_DECODING_SERVICE_KEY"
  engine = KRAFeatureEngineV2(TEST_KEY)
  df_result, summary_info, err_msg = engine.build_feature_matrix(
      meet_code="1", target_date="20240901", selected_race="1"
  )

  if err_msg:
    print("오류 발생:", err_msg)
  else:
    print("📊 [경주 난이도 측정 결과]")
    print(summary_info)
    print("\n📋 [상위 3마 퀀트 피처 분석]")
    print(
        df_result[[
            "AI_예측순위",
            "chulNo",
            "hrName",
            "jkName",
            "feat_hr_smoothed_win",
            "feat_budam_z",
            "feat_rating_z",
            "AI_승률(%)",
        ]].head(3)
    )
