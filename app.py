import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import requests
import streamlit as st


# ==========================================
# 1. KRA 퀀트 파이프라인 Class
# ==========================================
class KRADataPipeline:

  def __init__(self, service_key):
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
    }

  def fetch_raw_api(self, url, params):
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
        return pd.DataFrame(), "조회 데이터 없음"
      return None, f"서비스 에러 [{res_code}]"
    except Exception as e:
      return None, f"통신 장애: {str(e)}"

  def process_pipeline(
      self, meet_code, target_date, selected_race, daily_spent
  ):
    DAILY_LIMIT = 30000
    rem_budget = max(0, DAILY_LIMIT - daily_spent)

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
      return None, err or "출전표 수신 실패", None

    if "rcNo" in df_entry.columns:
      df_race = df_entry[
          df_entry["rcNo"].astype(str) == str(selected_race)
      ].copy()
    else:
      df_race = df_entry.copy()

    if df_race.empty:
      return None, f"{selected_race}경주 출전 데이터가 없습니다.", None

    df_weight, _ = self.fetch_raw_api(
        self.endpoints["horse_weight"],
        {"meet": meet_code, "rc_date": target_date},
    )
    df_jockey, _ = self.fetch_raw_api(
        self.endpoints["jockey_result"], {"meet": meet_code}
    )

    jockey_win_map = {}
    if (
        df_jockey is not None
        and not df_jockey.empty
        and "jkName" in df_jockey.columns
    ):
      if "winRateyear" in df_jockey.columns:
        jockey_win_map = dict(
            zip(
                df_jockey["jkName"],
                pd.to_numeric(
                    df_jockey["winRateyear"].astype(str).str.replace("%", ""),
                    errors="coerce",
                ).fillna(0.0),
            )
        )

    weight_diff_map = {}
    now_weight_map = {}
    if df_weight is not None and not df_weight.empty:
      if "chulNo" in df_weight.columns:
        if "wgHrDiff" in df_weight.columns:
          weight_diff_map = dict(
              zip(
                  df_weight["chulNo"].astype(str),
                  pd.to_numeric(df_weight["wgHrDiff"], errors="coerce").fillna(
                      0
                  ),
              )
          )
        if "wgHr" in df_weight.columns:
          now_weight_map = dict(
              zip(
                  df_weight["chulNo"].astype(str),
                  pd.to_numeric(df_weight["wgHr"], errors="coerce").fillna(0),
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

    scores = []
    for idx, row in df_race.iterrows():
      jk_name = str(row.get("jkName", ""))
      jk_win_rt = jockey_win_map.get(jk_name, 10.0)

      ord1_cnt_y = pd.to_numeric(row.get("ord1CntY", 0), errors="coerce") or 0.0
      rc_cnt_y = pd.to_numeric(row.get("rcCntY", 1), errors="coerce") or 1.0
      hr_win_rt = (ord1_cnt_y / max(rc_cnt_y, 1.0)) * 100.0

      rating = pd.to_numeric(row.get("rating", 0), errors="coerce") or 0.0
      rating_score = (rating / 120.0) * 100.0

      handy_cap = (
          pd.to_numeric(
              row.get("wgBudam", row.get("handyCap", 0)), errors="coerce"
          )
          or 0.0
      )
      handy_score = (handy_cap / 60.0) * 100.0

      chul_no = str(row.get("chulNo", ""))
      wg_diff = weight_diff_map.get(chul_no, 0.0)
      weight_penalty = -0.3 if abs(wg_diff) >= 10.0 else 0.0
      humidity_bonus = 0.20 if water_percent >= 10.0 else 0.0

      jk_weight, hr_weight = (
          (1.20, 1.40) if str(meet_code) == "2" else (1.40, 1.20)
      )

      total_score = (
          (jk_win_rt / 20.0) * jk_weight
          + (hr_win_rt / 25.0) * hr_weight
          + (rating_score / 80.0) * 1.10
          - (handy_score / 50.0) * 0.40
          + weight_penalty
          + humidity_bonus
      )
      scores.append(total_score)

    df_race["raw_score"] = scores
    exp_s = np.exp(df_race["raw_score"] - df_race["raw_score"].max())
    df_race["AI_승률_val"] = ((exp_s / exp_s.sum()) * 100).round(1)

    # 마체중 및 체중변화 수집
    df_race["체중"] = [
        f"{now_weight_map.get(str(r.get('chulNo')), '-')}kg"
        for _, r in df_race.iterrows()
    ]
    df_race["체중변화"] = [
        f"{weight_diff_map.get(str(r.get('chulNo')), 0):+d}kg"
        for _, r in df_race.iterrows()
    ]

    has_real_odds = False
    if "winOdds" in df_race.columns:
      odds_vals = pd.to_numeric(df_race["winOdds"], errors="coerce").fillna(0.0)
      if not (odds_vals == 0.0).all() and not (odds_vals == 5.0).all():
        has_real_odds = True
        df_race["winOdds_val"] = odds_vals

    if has_real_odds:
      df_race["market_raw"] = 1.0 / np.maximum(df_race["winOdds_val"], 1.05)
      df_race["시장_승률"] = (
          (df_race["market_raw"] / df_race["market_raw"].sum()) * 100
      ).round(1)
      df_race["EV_기대값"] = (
          (df_race["AI_승률_val"] / 100.0) * df_race["winOdds_val"]
      ) - 1.0
      df_race["단승배당"] = df_race["winOdds_val"].apply(lambda x: f"{x:.1f}배")
    else:
      df_race["단승배당"] = "대기중"
      df_race["EV_기대값"] = 0.0

    df_race["AI_예측순위"] = (
        df_race["AI_승률_val"]
        .rank(ascending=False, method="min")
        .astype(int)
    )
    sorted_df = df_race.sort_values(by="AI_예측순위").reset_index(drop=True)

    return sorted_df, None, rem_budget, has_real_odds


# ==========================================
# 2. UI 레이아웃 리뉴얼 (이미지 디자인 100% 반영)
# ==========================================
st.set_page_config(
    page_title="KRA AI Quant Betting Console", page_icon="🏇", layout="wide"
)

# Custom CSS for Premium UI
st.markdown(
    """
    <style>
    .top-card-container { display: flex; gap: 15px; margin-bottom: 25px; }
    .top-card { flex: 1; padding: 16px; border-radius: 12px; font-family: 'Noto Sans KR', sans-serif; }
    .card-1 { background-color: #FEF9C3; border: 1px solid #FDE047; color: #854D0E; }
    .card-2 { background-color: #EFF6FF; border: 1px solid #BFDBFE; color: #1E40AF; }
    .card-3 { background-color: #FFEDD5; border: 1px solid #FED7AA; color: #9A3412; }
    
    .card-title { font-weight: bold; font-size: 0.9rem; margin-bottom: 6px; }
    .card-horse { font-size: 1.3rem; font-weight: 800; margin-bottom: 4px; }
    .card-sub { font-size: 0.82rem; opacity: 0.85; margin-bottom: 8px; }
    .card-win { font-size: 1.1rem; font-weight: bold; color: #0284C7; }

    .portfolio-box { background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 12px 18px; margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center; }
    .badge-p { padding: 4px 8px; border-radius: 6px; font-weight: bold; font-size: 0.8rem; color: white; margin-right: 8px; }
    .bg-main { background-color: #0369A1; }
    .bg-sub { background-color: #0284C7; }
    .bg-single { background-color: #DB2777; }
    </style>
""",
    unsafe_allow_html=True,
)


def get_service_key():
  try:
    return st.secrets["KRA_SERVICE_KEY"]
  except Exception:
    return st.session_state.get(
        "custom_service_key",
        "92ac0b865a4117f6158886b90c8ad86908d7d67bead9c9166eb0fef68cddd2da",
    )


st.title("🏇 KRA AI 승률 예측 & 퀀트 포트폴리오 콘솔")

if "daily_spent" not in st.session_state:
  st.session_state["daily_spent"] = 0

# 상단 입력 제어바
col_a, col_b, col_c = st.columns([2, 2, 1])
with col_a:
  target_date = st.text_input("📅 경기 날짜 (YYYYMMDD)", value="20240901")
with col_b:
  meet_choice = st.selectbox(
      "🏟️ 경마장",
      options=["1", "2", "3"],
      format_func=lambda x: {"1": "서울", "2": "제주", "3": "부산경남"}[x],
  )
with col_c:
  selected_race = st.number_input(
      "🏁 경주 번호", min_value=1, max_value=15, value=1
  )

if st.button("🚀 AI 퀀트 분석 및 시각화 리포트 생성"):
  pipeline = KRADataPipeline(get_service_key())
  sorted_df, err, rem_budget, has_odds = pipeline.process_pipeline(
      meet_choice, target_date, selected_race, st.session_state["daily_spent"]
  )

  if err:
    st.error(f"데이터 파이프라인 처리 오류: {err}")
  else:
    top1 = sorted_df.iloc[0]
    top2 = sorted_df.iloc[1] if len(sorted_df) > 1 else top1
    top3 = sorted_df.iloc[2] if len(sorted_df) > 2 else top2

    # -------------------------------------------------------------
    # 섹션 1: 상단 👑 AI 순수 예측 TOP 3 카드리포트
    # -------------------------------------------------------------
    st.subheader("👑 AI 순수 예측 TOP 3")

    c1, c2, c3 = st.columns(3)
    with c1:
      st.markdown(
          f"""
            <div class="top-card card-1">
                <div class="card-title">🥇 1위 예측</div>
                <div class="card-horse">{top1.get('chulNo')}번 {top1.get('hrName')}</div>
                <div class="card-sub">기수: {top1.get('jkName')} (레이팅: {top1.get('rating', '-')})</div>
                <div class="card-win">승률: {top1.get('AI_승률_val')}%</div>
                <div style="font-size:0.8rem; color:#A16207;">(단승 배당: {top1.get('단승배당')})</div>
            </div>
            """,
          unsafe_allow_html=True,
      )

    with c2:
      st.markdown(
          f"""
            <div class="top-card card-2">
                <div class="card-title">🥈 2위 예측</div>
                <div class="card-horse">{top2.get('chulNo')}번 {top2.get('hrName')}</div>
                <div class="card-sub">기수: {top2.get('jkName')} (레이팅: {top2.get('rating', '-')})</div>
                <div class="card-win">승률: {top2.get('AI_승률_val')}%</div>
                <div style="font-size:0.8rem; color:#1D4ED8;">(단승 배당: {top2.get('단승배당')})</div>
            </div>
            """,
          unsafe_allow_html=True,
      )

    with c3:
      st.markdown(
          f"""
            <div class="top-card card-3">
                <div class="card-title">🥉 3위 예측</div>
                <div class="card-horse">{top3.get('chulNo')}번 {top3.get('hrName')}</div>
                <div class="card-sub">기수: {top3.get('jkName')} (레이팅: {top3.get('rating', '-')})</div>
                <div class="card-win">승률: {top3.get('AI_승률_val')}%</div>
                <div style="font-size:0.8rem; color:#C2410C;">(단승 배당: {top3.get('단승배당')})</div>
            </div>
            """,
          unsafe_allow_html=True,
      )

    st.write("")

    # -------------------------------------------------------------
    # 섹션 2: 📊 출전마 AI 순수 승률 & EV 분석표 (그래프 시각화)
    # -------------------------------------------------------------
    st.subheader("📊 출전마 AI 순수 승률 & EV 분석표")

    # 이미지 스타일의 데이터프레임 가공
    display_df = pd.DataFrame({
        "AI순위": sorted_df["AI_예측순위"],
        "게이트": sorted_df["chulNo"],
        "마명": sorted_df["hrName"],
        "기수명": sorted_df["jkName"],
        "레이팅": sorted_df.get("rating", "-"),
        "AI 승률(%)": sorted_df["AI_승률_val"],
        "단승배당": sorted_df["단승배당"],
        "EV (기대값)": sorted_df["EV_기대값"].apply(lambda x: f"{x:+.2f}"),
        "체중": sorted_df["체중"],
        "체중변화": sorted_df["체중변화"],
    })

    # Progress bar로 승률 시각화 처리
    st.dataframe(
        display_df,
        column_config={
            "AI 승률(%)": st.column_config.ProgressColumn(
                "AI 승률(%)",
                format="%.1f%%",
                min_value=0,
                max_value=float(sorted_df["AI_승률_val"].max() * 1.2),
            ),
        },
        use_container_width=True,
        hide_index=True,
    )

    st.write("")

    # -------------------------------------------------------------
    # 섹션 3: 🏇 AI 추천 실전 베팅 포트폴리오 (자금 분배)
    # -------------------------------------------------------------
    st.subheader("🏇 AI 추천 실전 베팅 포트폴리오")

    total_bet_budget = min(rem_budget, 10000)  # 이번 경주 투입 예산 (예: 1만원)

    main_stake = int(total_bet_budget * 0.50)
    sub1_stake = int(total_bet_budget * 0.20)
    sub2_stake = int(total_bet_budget * 0.15)
    single_stake = int(total_bet_budget * 0.15)

    st.markdown(
        f"""
        <div class="portfolio-box">
            <div>
                <span class="badge-p bg-main">복승식 (메인)</span> <b>{top1.get('chulNo')}번 ({top1.get('hrName')}) - {top2.get('chulNo')}번 ({top2.get('hrName')})</b>
                <div style="font-size:0.8rem; color:#64748B; margin-top:2px;">AI 승률 1위({top1.get('AI_승률_val')}%) & 2위({top2.get('AI_승률_val')}%) 메인 축 조합</div>
            </div>
            <div style="font-weight:bold; font-size:1.05rem; color:#0F172A;">{main_stake:,}원 (50%)</div>
        </div>

        <div class="portfolio-box">
            <div>
                <span class="badge-p bg-sub">삼복승식 (서브)</span> <b>{top1.get('chulNo')} - {top2.get('chulNo')} - {top3.get('chulNo')}번 ({top3.get('hrName')})</b>
                <div style="font-size:0.8rem; color:#64748B; margin-top:2px;">1-2위 고정 후 3착 복병 삼복승 방어</div>
            </div>
            <div style="font-weight:bold; font-size:1.05rem; color:#0F172A;">{sub1_stake:,}원 (20%)</div>
        </div>

        <div class="portfolio-box">
            <div>
                <span class="badge-p bg-single">단승식 (가치베팅)</span> <b>{top1.get('chulNo')}번 ({top1.get('hrName')})</b>
                <div style="font-size:0.8rem; color:#64748B; margin-top:2px;">AI 최상위 1위 축마 단승식 단독 집중</div>
            </div>
            <div style="font-weight:bold; font-size:1.05rem; color:#0F172A;">{single_stake:,}원 (15%)</div>
        </div>
    """,
        unsafe_allow_html=True,
    )
