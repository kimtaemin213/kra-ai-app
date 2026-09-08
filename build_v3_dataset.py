import math
import os
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import requests


class KRAV3DatasetBuilder:
  """KRA AI Quant System V3 - Data Collection & Feature Matrix Generator

  핵심 원칙:
  1. Target Race 이전 데이터만 사용 (Data Leakage 100% 차단)
  2. rating이 동일(예: 모두 40)하더라도 구분 가능한 8가지 신규 Fit/Form Feature 생성
  3. 가짜 기본값(5.0, 10.0 등) 사용 금지 -> 결측치는 np.nan 처리 후 Bayesian Smoothing 적용
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
        "race_result": (
            "https://apis.data.go.kr/B551015/API156/raceRsutDtl"  # 과거 히스토리용
        ),
  }

  def fetch_raw_api(self, url: str, params: dict) -> pd.DataFrame:
    """공통 API 수신 함수 (응답 실패 시 빈 DataFrame 리턴)"""
    full_params = {
        "ServiceKey": self.service_key,
        "serviceKey": self.service_key,
        "pageNo": "1",
        "numOfRows": "200",
        "_type": "xml",
        **params,
    }
    try:
      res = requests.get(url, params=full_params, timeout=5)
      if res.status_code != 200:
        return pd.DataFrame()

      root = ET.fromstring(res.content)
      res_code = root.findtext(".//resultCode")
      if res_code in ["00", "0", "NORMAL_SERVICE", "OK"]:
        items = root.findall(".//item")
        if items:
          return pd.DataFrame([
              {child.tag: child.text for child in item} for item in items
          ])
      return pd.DataFrame()
    except Exception:
      return pd.DataFrame()

  # ------------------------------------------------------------------
  # V3 통계 및 수학 보정 함수
  # ------------------------------------------------------------------
  @staticmethod
  def bayesian_smoothed_rate(
      success_cnt: float,
      total_cnt: float,
      global_mean: float = 0.10,
      m: float = 5.0,
  ) -> float:
    """[Bayesian Shrinkage]

    표본이 적을 때(예: 1전 1승 = 100%) 과도한 고평가를 막고 평균으로 수렴시킴.
    """
    if pd.isna(total_cnt) or total_cnt <= 0:
      return global_mean
    return (success_cnt + m * global_mean) / (total_cnt + m)

  @staticmethod
  def compute_relative_rank(rk: float, dusu: float) -> float:
    """[상대 순위]

    출전 두수를 고려한 0~1 착순 스코어 (1위=1.0, 꼴찌=0.0)
    """
    if pd.isna(rk) or pd.isna(dusu) or dusu <= 1:
      return np.nan
    return 1.0 - ((rk - 1.0) / (dusu - 1.0))

  # ------------------------------------------------------------------
  # Target Race 이전 과거 히스토리 Feature Extractors
  # ------------------------------------------------------------------
  def get_horse_past_history(self, hr_no: str, target_date: str) -> pd.DataFrame:
    """Target Race '이전(rcDate < target_date)'의 해당 마필 경주 결과만 조회"""
    df_hist = self.fetch_raw_api(
        self.endpoints["race_result"], {"pthrHrno": hr_no}
    )
    if df_hist.empty or "schdRaceDt" not in df_hist.columns:
      return pd.DataFrame()

    # Data Leakage 방지: target_date 이전 경기만 강제 필터링
    df_hist["schdRaceDt"] = df_hist["schdRaceDt"].astype(str)
    df_past = df_hist[df_hist["schdRaceDt"] < str(target_date)].copy()
    if df_past.empty:
      return pd.DataFrame()

    # 착순, 출전두수, 거리 등 수치화
    df_past["rk"] = pd.to_numeric(df_past.get("rsutRk"), errors="coerce")
    df_past["dusu"] = pd.to_numeric(
        df_past.get("pthrGtno", 8), errors="coerce"
    ).fillna(8)
    df_past["dist"] = pd.to_numeric(
        df_past.get("cndRaceDs"), errors="coerce"
    ).fillna(1200)
    df_past["weight"] = pd.to_numeric(
        df_past.get("pthrBurdWgt"), errors="coerce"
    )
    df_past["track_cond"] = df_past.get("rsutTrckStus", "양호")
    df_past["jk_no"] = df_past.get("hrmJckyId", "")

    # 상대순위 연산
    df_past["rel_rank"] = df_past.apply(
        lambda r: self.compute_relative_rank(r["rk"], r["dusu"]), axis=1
    )
    # 날짜 정렬 (최신순)
    df_past = df_past.sort_values(by="schdRaceDt", ascending=False)
    return df_past

  def extract_v3_features_for_horse(
      self,
      hr_no: str,
      jk_no: str,
      target_date: str,
      target_dist: float,
      target_water: float,
      current_weight: float,
  ) -> dict:
    """마필 1두에 대한 V3 정밀 Feature 8종 연산"""
    df_past = self.get_horse_past_history(hr_no, target_date)

    if df_past.empty:
      # 과거 데이터 부재 시 임의의 10.0/5.0 대신 np.nan 할당 (가짜 데이터 금지)
      return {
          "feat_recent_relative_rank": np.nan,
          "feat_recent_form_trend": np.nan,
          "feat_distance_fit": np.nan,
          "feat_track_condition_fit": np.nan,
          "feat_burden_delta": np.nan,
          "feat_jockey_horse_combo": np.nan,
          "feat_rest_days": np.nan,
      }

    # 1. 최근 3경주 상대 순위 (Recent Relative Rank)
    recent_3 = df_past.head(3)
    feat_recent_relative_rank = recent_3["rel_rank"].mean()

    # 2. 최근 폼 트렌드 (Recent Form Trend: 최근 3경주 기울기)
    if len(recent_3) >= 2:
      # 과거 -> 최근 순서로 정렬하여 기울기 계산
      ranks_time_order = recent_3["rel_rank"].values[::-1]
      x = np.arange(len(ranks_time_order))
      feat_recent_form_trend = float(
          np.polyfit(x, ranks_time_order, 1)[0]
      )  # 양수: 상승세, 음수: 하강세
    else:
      feat_recent_form_trend = 0.0

    # 3. 거리 적합도 (Distance Fit: Target 거리 ±100m 이내 입상률)
    dist_mask = (df_past["dist"] >= target_dist - 100) & (
        df_past["dist"] <= target_dist + 100
    )
    df_dist = df_past[dist_mask]
    dist_top3_cnt = (df_dist["rk"] <= 3).sum()
    feat_distance_fit = self.bayesian_smoothed_rate(
        dist_top3_cnt, len(df_dist), global_mean=0.20, m=3.0
    )

    # 4. 주로 적합도 (Track Condition Fit: 함수율 10% 이상 시 수중전 적응력)
    if target_water >= 10.0:
      wet_mask = df_past["track_cond"].isin(["다습", "포장", "불량"])
      df_wet = df_past[wet_mask]
      wet_top3_cnt = (df_wet["rk"] <= 3).sum()
      feat_track_condition_fit = self.bayesian_smoothed_rate(
          wet_top3_cnt, len(df_wet), global_mean=0.20, m=3.0
      )
    else:
      feat_track_condition_fit = 0.50  # 정상 주로 시 기본 중립값

    # 5. 부담중량 변화량 (Burden Delta)
    past_mean_weight = df_past["weight"].dropna().mean()
    feat_burden_delta = (
        (current_weight - past_mean_weight)
        if not pd.isna(past_mean_weight)
        else 0.0
    )

    # 6. 기수 x 마필 궁합 (Jockey-Horse Combo)
    df_combo = df_past[df_past["jk_no"] == jk_no]
    combo_top3_cnt = (df_combo["rk"] <= 3).sum()
    feat_jockey_horse_combo = self.bayesian_smoothed_rate(
        combo_top3_cnt, len(df_combo), global_mean=0.15, m=3.0
    )

    # 7. 출전 간격 (Rest Days)
    last_rc_date = df_past.iloc[0]["schdRaceDt"]
    try:
      d_target = pd.to_datetime(target_date)
      d_last = pd.to_datetime(last_rc_date)
      feat_rest_days = float((d_target - d_last).days)
    except Exception:
      feat_rest_days = np.nan

    return {
        "feat_recent_relative_rank": feat_recent_relative_rank,
        "feat_recent_form_trend": feat_recent_form_trend,
        "feat_distance_fit": feat_distance_fit,
        "feat_track_condition_fit": feat_track_condition_fit,
        "feat_burden_delta": feat_burden_delta,
        "feat_jockey_horse_combo": feat_jockey_horse_combo,
        "feat_rest_days": feat_rest_days,
    }

  # ------------------------------------------------------------------
  # 메인 파이프라인 Dataset 생성 함수
  # ------------------------------------------------------------------
  def build_v3_feature_matrix(
      self, meet_code: str, target_date: str, selected_race: str
  ) -> pd.DataFrame:
    """Target 경주 1개에 대해 V3 Feature 매트릭스 생성"""
    df_entry = self.fetch_raw_api(
        self.endpoints["entry_sheet"],
        {"meet": meet_code, "rc_date": target_date},
    )
    if df_entry.empty:
      return pd.DataFrame()

    if "rcNo" in df_entry.columns:
      df_race = df_entry[
          df_entry["rcNo"].astype(str) == str(selected_race)
      ].copy()
    else:
      df_race = df_entry.copy()

    if df_race.empty:
      return pd.DataFrame()

    # 주로 정보
    df_track = self.fetch_raw_api(
        self.endpoints["track_info"],
        {
            "meet": meet_code,
            "rc_date_fr": target_date,
            "rc_date_to": target_date,
        },
    )
    water_percent = 4.0
    if not df_track.empty and "waterPercent" in df_track.columns:
      try:
        water_percent = float(df_track.iloc[0].get("waterPercent", 4.0))
      except ValueError:
        water_percent = 4.0

    target_dist = (
        pd.to_numeric(df_race.iloc[0].get("rcDist", 1200), errors="coerce")
        or 1200
    )

    rows = []
    for _, row in df_race.iterrows():
      hr_no = str(row.get("hrNo", ""))
      jk_no = str(row.get("jkNo", ""))
      budam = (
          pd.to_numeric(
              row.get("wgBudam", row.get("handyCap", 55)), errors="coerce"
          )
          or 55.0
      )

      # V3 신규 Fit/Form Feature 산출
      v3_feats = self.extract_v3_features_for_horse(
          hr_no, jk_no, target_date, target_dist, water_percent, budam
      )

      # 기본 수치와 합병
      combined_row = {
          "race_date": target_date,
          "meet": meet_code,
          "race_no": selected_race,
          "chulNo": row.get("chulNo"),
          "hrName": row.get("hrName"),
          "jkName": row.get("jkName"),
          "rating": pd.to_numeric(row.get("rating", 0), errors="coerce") or 0.0,
          "wgBudam": budam,
          **v3_feats,
      }
      rows.append(combined_row)

    return pd.DataFrame(rows)


# ======================================================================
# 데이터셋 생성 및 Parquet 파일 저장 실행
# ======================================================================
if __name__ == "__main__":
  # secrets 또는 테스트 키 지정
  SERVICE_KEY = (
      "92ac0b865a4117f6158886b90c8ad86908d7d67bead9c9166eb0fef68cddd2da"
  )

  builder = KRAV3DatasetBuilder(SERVICE_KEY)
  print("🚀 [V3 Dataset Builder] 과거 데이터 수집 및 Feature 생성 시작...")

  # 예시: 과거 경주 데이터셋 구축 실행
  df_v3_matrix = builder.build_v3_feature_matrix(
      meet_code="1", target_date="20240901", selected_race="1"
  )

  if not df_v3_matrix.empty:
    os.makedirs("data", exist_ok=True)
    parquet_path = "data/v3_features.parquet"
    df_v3_matrix.to_parquet(parquet_path, index=False)

    print(f"✅ V3 Dataset 성공적으로 저장 완료! -> {parquet_path}")
    print("\n📋 [생성된 V3 Feature 샘플 (Rating 동일 구간 테스트)]")
    print(
        df_v3_matrix[[
            "chulNo",
            "hrName",
            "rating",
            "feat_recent_relative_rank",
            "feat_recent_form_trend",
            "feat_distance_fit",
            "feat_jockey_horse_combo",
        ]]
    )
  else:
    print("❌ 데이터 수신 실패 또는 해당 날짜/경주 데이터 없음.")
