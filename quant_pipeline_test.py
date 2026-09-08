def run_ai_quant_decision(df_entry, df_track, daily_spent):
  DAILY_LIMIT = 30000
  rem_budget = max(0, DAILY_LIMIT - daily_spent)

  df = df_entry.copy()

  # 1. 수치형 컬럼 보정
  num_cols = ["winOdds", "jkWinRt", "hrWinRt", "rating", "handyCap", "chulNo"]
  for col in num_cols:
    if col in df.columns:
      df[col] = (
          pd.to_numeric(
              df[col].astype(str).str.replace("%", ""), errors="coerce"
          )
          .fillna(0.0)
      )
    else:
      df[col] = 5.0 if col == "winOdds" else 10.0

  # 2. 주로 상태 (함수율) 파싱
  water_percent = 4.0
  if not df_track.empty and "waterPercent" in df_track.columns:
    try:
      water_percent = float(df_track.iloc[0].get("waterPercent", 4.0))
    except ValueError:
      water_percent = 4.0

  # 3. AI 가중치 스코어링 (기수+마필 능력치 및 주로적성 반영)
  humidity_bonus = 0.20 if water_percent >= 10.0 else 0.0
  df["score"] = (
      (df["jkWinRt"] / 20.0) * 1.50  # 기수 최근 승률 가중치 강화
      + (df["hrWinRt"] / 25.0) * 1.50  # 마필 승률 가중치 강화
      + (df["rating"] / 80.0) * 1.20  # 레이팅 점수 반영
      - (df["handyCap"] / 58.0) * 0.40
      + humidity_bonus
  )

  # Softmax 확률 도출
  exp_s = np.exp(df["score"] - df["score"].max())
  df["AI_승률(%)"] = ((exp_s / exp_s.sum()) * 100).round(1)

  df["market_raw"] = 1.0 / np.maximum(df["winOdds"], 1.05)
  df["시장_승률(%)"] = ((df["market_raw"] / df["market_raw"].sum()) * 100).round(
      1
  )

  df["AI_EDGE(%p)"] = (df["AI_승률(%)"] - df["시장_승률(%)"]).round(1)
  df["EV_기대값"] = ((df["AI_승률(%)"] / 100.0) * df["winOdds"]) - 1.0
  df["AI_예측순위"] = (
      df["AI_승률(%)"].rank(ascending=False, method="min").astype(int)
  )

  sorted_df = df.sort_values(by="AI_예측순위").reset_index(drop=True)
  top1 = sorted_df.iloc[0]
  top2 = sorted_df.iloc[1] if len(sorted_df) > 1 else top1

  top1_win_rt = top1["AI_승률(%)"]
  gap = top1_win_rt - top2["AI_승률(%)"]
  ev = top1["EV_기대값"]

  # 4. 🔥 [현실화된 등급 판정 로직]
  # 1위 승률이 높거나, 2위와의 격차가 벌어지면 적극적으로 베팅 추천!
  if top1_win_rt >= 20.0 or gap >= 5.0 or ev >= 0.15:
    raw_grade, target_stake = "🔥 S급", 5000
  elif top1_win_rt >= 15.0 or gap >= 3.0 or ev >= 0.0:
    raw_grade, target_stake = "🔷 A급", 3000
  elif top1_win_rt >= 12.0 or gap >= 1.5 or ev >= -0.15:
    raw_grade, target_stake = "📙 B급", 2000
  else:
    raw_grade, target_stake = "🔴 C/D급", 0

  # 5. 잔여 예산 검사
  if target_stake == 0:
    final_grade, badge_cls, actual_stake, action = (
        "🔴 PASS (초혼전 경주)",
        "badge-pass",
        0,
        "PASS",
    )
  elif rem_budget < target_stake:
    final_grade, badge_cls, actual_stake, action = (
        f"🔒 PASS (예산 부족: 남은 예산 {rem_budget:,}원)",
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

  # UI 표 가공
  display_cols = {
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

  valid_cols = [c for c in display_cols.keys() if c in sorted_df.columns]
  ui_df = sorted_df[valid_cols].rename(columns=display_cols)

  if "AI 승률" in ui_df.columns:
    ui_df["AI 승률"] = ui_df["AI 승률"].astype(str) + "%"
  if "시장 승률" in ui_df.columns:
    ui_df["시장 승률"] = ui_df["시장 승률"].astype(str) + "%"

  return (
      ui_df,
      water_percent,
      final_grade,
      badge_cls,
      actual_stake,
      action,
      gap,
      rem_budget,
      top1,
  )
