import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import requests


class KRADataPipeline:

  def __init__(self, service_key):
    self.service_key = service_key
    self.endpoints = {
        "entry_sheet": "https://apis.data.go.kr/B551015/API26_2/entrySheet_2",
        "track_info": "https://apis.data.go.kr/B551015/API189_1/Track_1",
    }

  def fetch_raw_api(self, url, params):
    """API 수신 및 XML 파싱 함수"""
    full_params = {
        "ServiceKey": self.service_key,
        "serviceKey": self.service_key,
        "pageNo": "1",
        "numOfRows": "100",
        "_type": "xml",
        **params,
    }
    try:
      res = requests.get(url, params=full_params, timeout=8)
      if res.status_code != 200:
        return None, f"HTTP {res.status_code} Error"

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
        return pd.DataFrame(), "조회된 데이터가 없습니다."
      return None, f"서비스 에러 [{res_code}]"
    except Exception as e:
      return None, f"통신 장애: {str(e)}"

  def process_pipeline(
      self, meet_code, target_date, selected_race, daily_spent
  ):
    """실시간 수신 -> 정제 -> 퀀트연산 -> UI 가공 일괄 처리 파이프라인"""
    DAILY_LIMIT = 30000
    rem_budget = max(0, DAILY_LIMIT - daily_spent)

    # 1. 출전표 및 주로정보 실시간 수신
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

    # 2. Target 경주 번호(rcNo) 필터링
    if "rcNo" in df_entry.columns:
      df_race = df_entry[
          df_entry["rcNo"].astype(str) == str(selected_race)
      ].copy()
    else:
      df_race = df_entry.copy()

    if df_race.empty:
      return None, f"{selected_race}경주 출전 데이터가 없습니다.", None

    # 3. 수치형 데이터 전처리
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

    # 4. 주로 상태 (함수율) 파싱
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

    # 5. Softmax AI 승률 및 EV 기대값 연산
    humidity_bonus = 0.15 if water_percent >= 10.0 else 0.0
    df_race["score"] = (
        (df_race["jkWinRt"] / 25.0) * 1.20
        + (df_race["hrWinRt"] / 30.0) * 1.30
        + (df_race["rating"] / 100.0) * 1.05
        - (df_race["handyCap"] / 60.0) * 0.50
        + humidity_bonus
    )

    exp_s = np.exp(df_race["score"] - df_race["score"].max())
    df_race["AI_승률(%)"] = ((exp_s / exp_s.sum()) * 100).round(1)
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

    # 6. 의사결정 및 자금 차감 검사
    sorted_df = df_race.sort_values(by="AI_예측순위").reset_index(drop=True)
    top1 = sorted_df.iloc[0]
    top2 = sorted_df.iloc[1] if len(sorted_df) > 1 else top1
    gap = top1["AI_승률(%)"] - top2["AI_승률(%)"]
    ev = top1["EV_기대값"]

    if ev >= 0.40 and gap >= 10.0:
      raw_grade, target_stake = "🔥 S급", 5000
    elif ev >= 0.20 and gap >= 6.0:
      raw_grade, target_stake = "🔷 A급", 3000
    elif ev >= 0.08 and gap >= 4.0:
      raw_grade, target_stake = "📙 B급", 2000
    else:
      raw_grade, target_stake = "🔴 C/D급", 0

    if target_stake == 0:
      final_grade, badge_cls, actual_stake, action = (
          "🔴 PASS (조건 미달)",
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

    # 7. UI 전용 필수 컬럼 슬라이싱 & 가공
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
        "EV_기대값": "EV 기대값",
    }

    valid_cols = [c for c in display_map.keys() if c in sorted_df.columns]
    ui_df = sorted_df[valid_cols].rename(columns=display_map)

    ui_df["EV 기대값"] = (ui_df["EV 기대값"] * 100).round(1).astype(str) + "%"
    ui_df["AI 승률"] = ui_df["AI 승률"].astype(str) + "%"
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
