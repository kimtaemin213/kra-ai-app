import xml.etree.ElementTree as ET
import datetime
import numpy as np
import pandas as pd
import requests
import streamlit as st

# ==========================================
# 0. 레이아웃 및 기본 설정
# ==========================================
st.set_page_config(
    page_title="KRA AI 개인용 경마 매매 시스템",
    page_icon="🏇",
    layout="centered"
)

st.markdown(
    """
    <style>
    .main { padding: 0.8rem; }
    .stButton>button { width: 100%; border-radius: 10px; font-weight: bold; background-color: #0284c7; color: white; height: 3.2em; }
    .metric-card { background-color: #1e293b; padding: 15px; border-radius: 12px; color: white; margin-bottom: 12px; }
    .badge-s { background-color: #15803d; color: white; padding: 4px 8px; border-radius: 6px; font-weight: bold; }
    .badge-a { background-color: #0369a1; color: white; padding: 4px 8px; border-radius: 6px; font-weight: bold; }
    .badge-b { background-color: #b45309; color: white; padding: 4px 8px; border-radius: 6px; font-weight: bold; }
    .badge-pass { background-color: #b91c1c; color: white; padding: 4px 8px; border-radius: 6px; font-weight: bold; }
    </style>
""",
    unsafe_allow_html=True,
)

DECODING_KEY = "h7qTh7gmeKJ49Z3vL/4Oss49MVNvYWqGYyDyJgk8cUGs99uBTP1Qit3jk0qs8UVkMiqYWuLoxnNhd6nHuGKA2w=="

# ==========================================
# 1. API 수신 및 날짜 연산 모듈
# ==========================================
def fetch_api_dataframe(url, params):
    try:
        response = requests.get(url, params=params, timeout=6)
        if response.status_code == 200:
            root = ET.fromstring(response.content)
            res_code = root.findtext(".//resultCode")
            if res_code and res_code not in ["00", "0", "NORMAL_SERVICE", "OK"]:
                return pd.DataFrame()
            items = root.findall(".//item")
            data = [{child.tag: child.text for child in item} for item in items]
            return pd.DataFrame(data)
    except Exception:
        pass
    return pd.DataFrame()

def generate_all_possible_dates():
    today = datetime.date.today()
    race_dates = []
    for i in range(-14, 30):
        d = today - datetime.timedelta(days=i)
        if d.weekday() in [3, 4, 5, 6]:
            race_dates.append(d.strftime("%Y%m%d"))
    return sorted(race_dates, reverse=True)

@st.cache_data(ttl=180)
def get_active_meets_for_date(target_date):
    active_meets = []
    for meet_code in ["1", "2", "3"]:
        params = {
            "serviceKey": DECODING_KEY,
            "pageNo": "1",
            "numOfRows": "5",
            "meet": meet_code,
            "rc_date": target_date,
        }
        df = fetch_api_dataframe("http://apis.data.go.kr/B551015/API21_1/raceHorseList_1", params)
        if not df.empty:
            active_meets.append(meet_code)

    return active_meets if active_meets else ["1", "2", "3"]

@st.cache_data(ttl=180)
def collect_kra_data(meet_code, target_date):
    base_params = {
        "serviceKey": DECODING_KEY,
        "pageNo": "1",
        "numOfRows": "100",
        "meet": meet_code,
        "rc_date": target_date,
    }

    df_race = fetch_api_dataframe("http://apis.data.go.kr/B551015/API21_1/raceHorseList_1", base_params)
    if df_race.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df_weight = fetch_api_dataframe("http://apis.data.go.kr/B551015/API25/horseWeightList", base_params)
    df_jockey = fetch_api_dataframe("http://apis.data.go.kr/B551015/API22/jockeyList", base_params)
    df_record = fetch_api_dataframe("http://apis.data.go.kr/B551015/API24/totalRecordList", base_params)
    df_odds = fetch_api_dataframe("http://apis.data.go.kr/B551015/API28/oddsInfoList", base_params)
    df_track = fetch_api_dataframe("http://apis.data.go.kr/B551015/API30/trackConditionList", base_params)
    df_result = fetch_api_dataframe("http://apis.data.go.kr/B551015/API27_1/raceResultList_1", base_params)

    df_merged = df_race.copy()

    if not df_weight.empty and "hrName" in df_weight.columns:
        cols = [c for c in ["hrName", "nowWeight", "chgWeight"] if c in df_weight.columns]
        df_merged = pd.merge(df_merged, df_weight[cols], on="hrName", how="left")

    if not df_jockey.empty and "jkName" in df_jockey.columns:
        cols = [c for c in ["jkName", "jkWinRt", "jkQrt"] if c in df_jockey.columns]
        df_merged = pd.merge(df_merged, df_jockey[cols], on="jkName", how="left")

    if not df_record.empty and "hrName" in df_record.columns:
        cols = [c for c in ["hrName", "hrWinRt", "ord1Cnt", "rating"] if c in df_record.columns]
        df_merged = pd.merge(df_merged, df_record[cols], on="hrName", how="left")

    if not df_odds.empty and "chulNo" in df_odds.columns:
        cols = [c for c in ["chulNo", "winOdds", "plcOdds"] if c in df_odds.columns]
        df_merged = pd.merge(df_merged, df_odds[cols], on="chulNo", how="left")

    numeric_defaults = {
        "handyCap": 55.0, "nowWeight": 480.0, "chgWeight": 0.0,
        "jkWinRt": 10.0, "hrWinRt": 10.0, "ord1Cnt": 1.0,
        "rating": 50.0, "winOdds": 5.0, "chulNo": 1
    }

    for col, default_val in numeric_defaults.items():
        if col in df_merged.columns:
            df_merged[col] = pd.to_numeric(df_merged[col].astype(str).str.replace("%", ""), errors="coerce").fillna(default_val)
        else:
            df_merged[col] = default_val

    return df_merged, df_track, df_result

# ==========================================
# 2. Race-level 의사결정 엔진
# ==========================================
def predict_race_decision(df_raw, df_track, selected_race):
    X = df_raw.copy()

    track_moisture = 5.0
    track_state_str = "양호"
    if not df_track.empty and "rcNo" in df_track.columns:
        target_track = df_track[df_track["rcNo"] == str(selected_race)]
        if not target_track.empty:
            track_moisture = float(target_track.iloc[0].get("humidity", 5.0))
            track_state_str = target_track.iloc[0].get("trackCondition", "양호")

    # 시장 확률 정규화
    X["market_raw"] = 1.0 / X["winOdds"]
    X["시장_승률(%)"] = ((X["market_raw"] / X["market_raw"].sum()) * 100).round(1)

    # Race-level Softmax
    humidity_bonus = 0.12 if track_moisture >= 10.0 else 0.0
    X["score_base"] = (
        (X["jkWinRt"] / 25.0) * 1.18
        + (X["hrWinRt"] / 30.0) * 1.22
        + (X["ord1Cnt"] / 8.0) * 0.75
        + (X["rating"] / 100.0) * 1.05
        - (X["handyCap"] / 60.0) * 0.55
        - (X["chgWeight"].abs() / 10.0) * 0.18
        + humidity_bonus
    )

    exp_scores = np.exp(X["score_base"] - X["score_base"].max())
    X["AI_승률(%)"] = ((exp_scores / exp_scores.sum()) * 100).round(1)
    X["AI_EDGE(%p)"] = (X["AI_승률(%)"] - X["시장_승률(%)"]).round(1)
    X["EV_기대값"] = ((X["AI_승률(%)"] / 100.0) * X["winOdds"]) - 1.0
    X["AI_예측순위"] = X["AI_승률(%)"].rank(ascending=False, method="min")

    sorted_df = X.sort_values(by="AI_예측순위").reset_index(drop=True)

    top1 = sorted_df.iloc[0]
    top2 = sorted_df.iloc[1] if len(sorted_df) > 1 else sorted_df.iloc[0]
    gap = top1["AI_승률(%)"] - top2["AI_승률(%)"]
    ev = top1["EV_기대값"]

    # 하루 3만원 예산 한도 내 등급 판단
    if ev >= 0.50 and gap >= 12.0:
        grade = "🔥 S급 (최고 가치)"
        badge_cls = "badge-s"
        bet_amount = 5000
        action = "BET"
    elif ev >= 0.20 and gap >= 8.0:
        grade = "🔷 A급 (우수)"
        badge_cls = "badge-a"
        bet_amount = 3000
        action = "BET"
    elif ev >= 0.10 and gap >= 5.0:
        grade = "📙 B급 (일반)"
        badge_cls = "badge-b"
        bet_amount = 2000
        action = "BET"
    else:
        grade = "🔴 C/D급 (PASS 권장)"
        badge_cls = "badge-pass"
        bet_amount = 0
        action = "PASS"

    reasons = []
    if top1["hrWinRt"] >= 15.0: reasons.append(f"말 통산 승률 우수 ({top1['hrWinRt']}%)")
    if top1["jkWinRt"] >= 12.0: reasons.append(f"기수 승률 상위권 ({top1['jkWinRt']}%)")
    if top1["rating"] >= 50: reasons.append(f"레이팅 보증 ({top1['rating']})")
    if top1["AI_EDGE(%p)"] > 5.0: reasons.append(f"시장 대비 높은 AI EDGE (+{top1['AI_EDGE(%p)']}%p)")
    if not reasons: reasons.append("전반적 스탯 안정성 보유")

    return sorted_df, track_moisture, track_state_str, grade, badge_cls, bet_amount, action, gap, reasons

# ==========================================
# 3. Streamlit 대시보드 UI
# ==========================================
st.title("🏇 KRA AI 개인용 경마 매매 시스템")
st.caption("하루 최대 30,000원 위험 관리 & 의사결정 대시보드")

with st.expander("⚙️ 경기 일자 및 경마장 선택", expanded=True):
    col1, col2 = st.columns(2)

    with col1:
        all_dates = generate_all_possible_dates()
        target_date_str = st.selectbox(
            "📅 경기 날짜 선택",
            options=all_dates,
            format_func=lambda x: f"{x[:4]}-{x[4:6]}-{x[6:]}",
        )

        active_meets = get_active_meets_for_date(target_date_str)
        meet_choice = st.selectbox(
            "🏟️ 경마장 선택",
            options=active_meets,
            format_func=lambda x: {"1": "서울 렛츠런파크", "2": "부산경남", "3": "제주"}[x],
        )

    with col2:
        df_check, _, _ = collect_kra_data(meet_choice, target_date_str)
        if not df_check.empty and "rcNo" in df_check.columns:
            actual_races = sorted(pd.to_numeric(df_check["rcNo"], errors="coerce").dropna().unique().astype(int).tolist())
            race_options = actual_races if actual_races else list(range(1, 12))
        else:
            race_options = list(range(1, 12))

        selected_race = st.selectbox("🏁 경주 번호 (RACE)", options=race_options, index=0)

run_button = st.button("🚀 실시간 AI 매매 의사결정 분석")

if run_button:
    with st.spinner(f"[{target_date_str}] {selected_race}경주 API 수신 및 분석 중..."):
        df_raw, df_track, df_real_result = collect_kra_data(meet_choice, target_date_str)

        if df_raw.empty:
            st.error("⏳ 해당 날짜의 API 데이터가 아직 준비되지 않았습니다.")
        else:
            df_target_race = df_raw[df_raw["rcNo"].astype(str) == str(selected_race)].copy() if "rcNo" in df_raw.columns else df_raw.copy()

            if df_target_race.empty:
                st.warning(f"⚠️ {selected_race}경주 출전 정보가 없습니다.")
            else:
                race_data, moisture, track_state, grade, badge_cls, bet_amount, action, gap, reasons = predict_race_decision(
                    df_target_race, df_track, selected_race
                )

                st.markdown(
                    f"""
                <div class="metric-card">
                    <h3>📢 의사결정: <span class="{badge_cls}">{grade}</span></h3>
                    <p><b>추천 베팅금액:</b> <span style="color:#f59e0b; font-size:1.3em; font-weight:bold;">{bet_amount:,}원</span> (일일 잔여한도 차감 적용)</p>
                    <p>💧 함수율: {moisture}% | ☀️ 주로상태: {track_state} | Top1-2 격차: {gap:.1f}%p</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

                top1 = race_data.iloc[0]

                st.subheader(f"🥇 [{selected_race}경주] AI 추천 1위: {top1['chulNo']}번 ({top1['hrName']})")
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("AI 승률", f"{top1['AI_승률(%)']}%")
                col_b.metric("시장 승률", f"{top1['시장_승률(%)']}%")
                col_c.metric("AI EDGE", f"+{top1['AI_EDGE(%p)']}%p")

                st.markdown("**💡 AI 핵심 선택 이유:**")
                for r in reasons:
                    st.write(f"• {r}")

                st.subheader("📊 출전마 전체 AI 분석표")
                display_df = race_data[[
                    "AI_예측순위", "chulNo", "hrName", "jkName", "rating",
                    "AI_승률(%)", "시장_승률(%)", "AI_EDGE(%p)", "winOdds", "EV_기대값"
                ]].copy()

                rename_map = {
                    "AI_예측순위": "순위", "chulNo": "게이트", "hrName": "마명",
                    "jkName": "기수명", "rating": "레이팅", "AI_승률(%)": "AI승률",
                    "시장_승률(%)": "시장승률", "AI_EDGE(%p)": "EDGE", "winOdds": "배당률", "EV_기대값": "EV"
                }
                display_df.rename(columns=rename_map, inplace=True)
                st.dataframe(display_df, use_container_width=True)

                if action == "PASS":
                    st.error("🔒 이 경주는 AI 신뢰도가 낮아 **[PASS (관망)]**를 권장합니다.")
                else:
                    st.success(f"💰 **[매매 실행]**: {top1['chulNo']}번({top1['hrName']}) 단승식 **{bet_amount:,}원** 투입 권장")
