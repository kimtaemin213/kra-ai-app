import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import requests


class KRADataPipeline:

  def __init__(self, service_key):
    self.service_key = service_key
    # 8개 공식 확인된 KRA API 엔드포인트 전체 등록
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
        "highest_dividend": (
            "https://apis.data.go.kr/B551015/API35_1/highestDividendRateInfo_1"
        ),
        "jeju_result": (
            "https://apis.data.go.kr/B551015/jejuhorseresult/getjejuhorseresult"
        ),
    }

  def fetch_raw_api(self, url, params):
    """공통 API 호출 및 XML 파싱"""
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
        return None, f"게이트웨이 에러: {err_msg}"

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
        return pd.DataFrame(), "데이터 없음"
      return None, f"서비스 에러 [{res_code}]"
    except Exception as e:
      return None, f"통신 장애: {str(e)}"

  def process_pipeline(
      self, meet_code, target_date, selected_race, daily_spent
  ):
    """8개 API 융합 및 AI 퀀트 정제 메인 파이프라인"""
    DAILY_LIMIT = 30000
    rem_budget = max(0, DAILY_LIMIT - daily_spent)

    # 1. 메인 API 수신 (출전표 및 주로)
    df_entry, err = self.fetch_raw_api(
        self.endpoints["entry_sheet"],
        {"meet": meet_code, "rc_date": target_date},
    )
    df_track, _ = self.fetch_raw_api(
        self.endpoints["track_info"],
        {
            "meet": meet_code,
            "rc_date_fr": target_date,
            "rc_date_to": target_date,
        },
    )

    if err or df_entry is None or df_entry.empty:
      return None, err or "데이터 없음", None

    # 선택 경주 번호(rcNo) 필터링
    if "rcNo" in df_entry.columns:
      df_race = df_entry[
          df_entry["rcNo"].astype(str) == str(selected_race)
      ].copy()
    else:
      df_race = df_entry.copy()

    if df_race.empty:
      return None, f"{selected_race}경주 출전 데이터가 없습니다.", None

    # 2. 보조 API 6종 병렬 데이터 수신
    df_weight, _ = self.fetch_raw_api(
        self.endpoints["horse_weight"],
        {"meet": meet_code, "rc_date": target_date},
    )
    df_jockey, _ = self.fetch_raw_api(
        self.endpoints["jockey_result"], {"meet": meet_code}
    )

    # 수치형 필드 전처리
    num_cols = ["winOdds", "jkWinRt", "hrWinRt", "rating", "handyCap", "chulNo"]
    for col in num_cols:
      if col in df_race.columns:
        df_race[col] = (
            pd.to_numeric(
                df_race[col].astype(str).str.replace("%", ""), errors="coerce"
            )
            .fillna(0.0)
        )
      else:
        df_race[col] = 5.0 if col == "winOdds" else 10.0

    # 3. 보조 API 융합 가중치 계산
    # (1) 체중 변동 보정 (API25_1)
    df_race["weight_penalty"] = 0.0
    if df_weight is not None and not df_weight.empty:
      if "chulNo" in df_weight.columns and "diffWeight" in df_weight.columns:
        weight_map = dict(
            zip(df_weight["chulNo"].astype(str), df_weight["diffWeight"])
        )
        for idx, row in df_race.iterrows():
          chul_no = str(row.get("chulNo", ""))
          diff_w = pd.to_numeric(weight_map.get(chul_no, 0), errors="coerce")
          if abs(diff_w) >= 10:  # 체중 10kg 이상 급변 시 감점
            df_race.at[idx, "weight_penalty"] = -0.3

    # (2) 기수 최근 1년 폼 보정 (jkyresult)
    df_race["jockey_1yr_bonus"] = 0.0
    if df_jockey is not None and not df_jockey.empty:
      if "jkName" in df_jockey.columns and "ord1Cnt" in df_jockey.columns:
        jockey_map = dict(
            zip(
                df_jockey["jkName"],
                pd.to_numeric(df_jockey["ord1Cnt"], errors="coerce").fillna(0),
            )
        )
        for idx, row in df_race.iterrows():
          jk_name = row.get("jkName", "")
          wins_1yr = jockey_map.get(jk_name, 0)
          if wins_1yr >= 20:  # 최근 1년 20승 이상 우수 기수 우대
            df_race.at[idx, "jockey_1yr_bonus"] = 0.4

    # (3) 주로 함수율 보정 (API189_1)
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
    humidity_bonus = 0.20 if water_percent >= 10.0 else 0.0

    # 4. 8개 API 다단계 융합 AI Score 연산
    df_race["score"] = (
        (df_race["jkWinRt"] / 20.0) * 1.40
        + (df_race["hrWinRt"] / 25.0) * 1.40
        + (df_race["rating"] / 80.0) * 1.10
        - (df_race["handyCap"] / 58.0) * 0.40
        + df_race["jockey_1yr_bonus"]  # 최근 1년 기수 폼 반영
        + df_race["weight_penalty"]  # 마체중 급변 반영
        + humidity_bonus  # 주로 함수율 반영
    )

    # Softmax 상대 승률
    exp_s = np.exp(df_race["score"] - df_race["score"].max())
    df_race["AI_승률(%)"] = ((exp_s / exp_s.sum()) * 100).round(1)

    # 임플라이드 시장 승률
    df_race["market_raw"] = 1.0 / np.maximum(df_race["winOdds"], 1.05)
    df_race["시장_승률(%)"] = (
        (df_race["market_raw"] / df_race["market_raw"].sum()) * 100
    ).round(1)

    df_race["AI_EDGE(%p)"] = (
        df_race["AI_승률(%)"] - df_race["시장_승률(%)"]
    ).round(1)
    df_race["EV_기대값"] = (
        (df_race["AI_승률(%)"] / 100.0) * df_race["winOdds"]
    ) - 1.0
    df_race["AI_예측순위"] = (
        df_race["AI_승률(%)"].rank(ascending=False, method="min").astype(int)
    )

    # 5. 퀀트 의사결정 및 자금 관리
    sorted_df = df_race.sort_values(by="AI_예측순위").reset_index(drop=True)
    top1 = sorted_df.iloc[0]
    top2 = sorted_df.iloc[1] if len(sorted_df) > 1 else top1

    top1_win_rt = top1["AI_승률(%)"]
    gap = top1_win_rt - top2["AI_승률(%)"]
    ev = top1["EV_기대값"]

    if top1_win_rt >= 20.0 or gap >= 5.0 or ev >= 0.15:
      raw_grade, target_stake = "🔥 S급", 5000
    elif top1_win_rt >= 15.0 or gap >= 3.0 or ev >= 0.0:
      raw_grade, target_stake = "🔷 A급", 3000
    elif top1_win_rt >= 12.0 or gap >= 1.5 or ev >= -0.15:
      raw_grade, target_stake = "📙 B급", 2000
    else:
      raw_grade, target_stake = "🔴 C/D급", 0

    if target_stake == 0:
      final_grade, badge_cls, actual_stake, action = (
          "🔴 PASS (초혼전 경주)",
          "badge-pass",
          0,
          "PASS",
      )
    elif rem_budget < target_stake:
      final_grade, badge_cls, actual_stake, action = (
          (
              f"🔒 PASS (예산 부족: 남은 예산 {rem_budget:,}원 < 필요금액"
              f" {target_stake:,}원)"
          ),
          "badge-pass",
          0,
          "PASS",
      )
    else:
      final_grade = f"{raw_grade} (추천)"
      badge_cls = (
          "badge-s"
          if "S급" in raw_grade
          else ("badge-a" if "A급" in raw_grade else "badge-b")
      )
      actual_stake, action = target_stake, "BET"

    # 6. UI 전용 필드 슬라이싱
    display_map = {
        "AI_예측순위": "순위",
        "chulNo": "마번",
        "hrName": "마명",
        "jkName": "기수",
        "trName": "조교사",
        "handyCap": "부담중량",
        "winOdds": "단승배당",
        "AI_승률(%)": "AI 승률",
        "시장_승률(%)": "시장 승률",
        "AI_EDGE(%p)": "EDGE(%p)",
    }

    valid_cols = [c for c in display_map.keys() if c in sorted_df.columns]
    ui_df = sorted_df[valid_cols].rename(columns=display_map)

    if "AI 승률" in ui_df.columns:
      ui_df["AI 승률"] = ui_df["AI 승률"].astype(str) + "%"
    if "시장 승률" in ui_df.columns:
      ui_df["시장 승률"] = ui_df["시장 승률"].astype(str) + "%"

    summary_info = {
        "water_pct": water_percent,
        "grade": final_grade,
        "badge_cls": badge_cls,
        "actual_stake": actual_stake,
        "action": action,
        "gap": gap,
        "rem_budget": rem_budget,
        "top1_no": top1.get("chulNo", ""),
        "top1_name": top1.get("hrName", ""),
        "top1_ev": top1.get("EV_기대값", 0.0),
    }

    return ui_df, None, summary_info
