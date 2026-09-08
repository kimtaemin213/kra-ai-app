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

# ==========================================
# 1. API 설정 및 문제의 엔드포인트 정의 (구버전)
# ==========================================
DECODING_KEY = "h7qTh7gmeKJ49Z3vL/4Oss49MVNvYWqGYyDyJgk8cUGs99uBTP1Qit3jk0qs8UVkMiqYWuLoxnNhd6nHuGKA2w=="
ENCODED_KEY = "h7qTh7gmeKJ49Z3vL%2F4Oss49MVNvYWqGYyDyJgk8cUGs99uBTP1Qit3jk0qs8UVkMiqYWuLoxnNhd6nHuGKA2w%3D%3D"

# ⚠️ 문제가 발생 중인 구버전 API 엔드포인트 맵
API_BASE_URL = "http://apis.data.go.kr/B551015"

ENDPOINTS = {
    "race_horse": f"{API_BASE_URL}/API21_1/raceHorseList_1",      # 출전표 (구버전 -> API214_1 개편 필요)
    "horse_weight": f"{API_BASE_URL}/API25/horseWeightList",       # 마체중
    "jockey": f"{API_BASE_URL}/API22/jockeyList",                  # 기수 (구버전 -> API12_1 개편 필요)
    "total_record": f"{API_BASE_URL}/API24/totalRecordList",      # 통산성적
    "odds": f"{API_BASE_URL}/API28/oddsInfoList",                  # 배당 (구버전 -> API160_1/API301 개편 필요)
    "track": f"{API_BASE_URL}/API30/trackConditionList",           # 주로/날씨 (구버전 -> API189_1 개편 필요)
    "race_result": f"{API_BASE_URL}/API27_1/raceResultList_1"      # 결과 (구버전 -> API299 개편 필요)
}

# ==========================================
# 2. 실제 HTTP 호출 및 파싱 모듈
# ==========================================
def fetch_api_dataframe(url, params):
    """
    KRA API 단일 호출 함수
    """
    try:
        # serviceKey를 params와 결합하여 호출
        full_params = {"serviceKey": DECODING_KEY, **params}
        response = requests.get(url, params=full_params, timeout=6)
        
        if response.status_code == 200:
            root = ET.fromstring(response.content)
            res_code = root.findtext(".//resultCode")
            
            if res_code in ["00", "0", "NORMAL_SERVICE", "OK"]:
                items = root.findall(".//item")
                if items:
                    data = [{child.tag: child.text for child in item} for item in items]
                    return pd.DataFrame(data)
    except Exception as e:
        st.error(f"API 호출 에러 발생: {url} -> {e}")
    return pd.DataFrame()

@st.cache_data(ttl=180)
def collect_kra_data(meet_code, target_date):
    """
    모든 API 데이터를 수집하고 Merge하는 파이프라인
    """
    base_params = {
        "pageNo": "1",
        "numOfRows": "100",
        "meet": meet_code,
        "rc_date": target_date,
    }

    # 1) 출전표 수신
    df_race = fetch_api_dataframe(ENDPOINTS["race_horse"], base_params)
    if df_race.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # 2) 부가 정보 수신
    df_weight = fetch_api_dataframe(ENDPOINTS["horse_weight"], base_params)
    df_jockey = fetch_api_dataframe(ENDPOINTS["jockey"], base_params)
    df_record = fetch_api_dataframe(ENDPOINTS["total_record"], base_params)
    df_odds = fetch_api_dataframe(ENDPOINTS["odds"], base_params)
    df_track = fetch_api_dataframe(ENDPOINTS["track"], base_params)
    df_result = fetch_api_dataframe(ENDPOINTS["race_result"], base_params)

    # 3) 데이터 병합 (Merge)
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

    # 수치형 컬럼 전처리 및 기본값 채우기
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
# 3. Race-level 의사결정 엔진
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

    # 등급 판단
    if ev >= 0.50 and gap >= 12.0:
        grade, badge_cls, bet_amount, action = "🔥 S급 (최고 가치)", "badge-s", 5000, "BET"
    elif ev >= 0.20 and gap >= 8.0:
        grade, badge_cls, bet_amount, action = "🔷 A급 (우수)", "badge-a", 3000, "BET"
    elif ev >= 0.10 and gap >= 5.0:
        grade, badge_cls, bet_amount, action = "📙 B급 (일반)", "badge-b", 2000, "BET"
    else:
        grade, badge_cls, bet_amount, action = "🔴 C/D급 (PASS 권장)", "badge-pass", 0, "PASS"

    return sorted_df, track_moisture, track_state_str, grade, badge_cls, bet_amount, action, gap

# ==========================================
# 4. Streamlit 메인 UI
# ==========================================
st.title("🏇 KRA AI 개인용 경마 매매 시스템")
st.caption("실시간 API 연동 및 의사결정 콘솔")

col1, col2 = st.columns(2)
with col1:
    target_date_str = st.text_input("📅 경기 날짜 (YYYYMMDD)", value="20260905")
    meet_choice = st.selectbox("🏟️ 경마장", options=["1", "2", "3"], format_func=lambda x: {"1": "서울", "2": "부산경남", "3": "제주"}[x])

with col2:
    selected_race = st.number_input("🏁 경주 번호", min_value=1, max_value=15, value=1)

if st.button("🚀 실시간 KRA API 데이터 분석"):
    with st.spinner("API 데이터 호출 중..."):
        df_raw, df_track, df_result = collect_kra_data(meet_choice, target_date_str)

        if df_raw.empty:
            st.error("❌ API 데이터 수신 실패! (엔드포인트 에러 또는 서버 응답 없음)")
        else:
            df_target = df_raw[df_raw["rcNo"].astype(str) == str(selected_race)].copy()
            
            if df_target.empty:
                st.warning(f"{selected_race}경주 데이터가 존재하지 않습니다.")
            else:
                race_data, moisture, track_state, grade, badge_cls, bet_amount, action, gap = predict_race_decision(
                    df_target, df_track, selected_race
                )

                st.markdown(
                    f"""
                <div class="metric-card">
                    <h3>📢 의사결정: <span class="{badge_cls}">{grade}</span></h3>
                    <p><b>추천 베팅금액:</b> <span style="color:#f59e0b; font-size:1.3em; font-weight:bold;">{bet_amount:,}원</span></p>
                    <p>💧 함수율: {moisture}% | ☀️ 주로상태: {track_state} | Top1-2 격차: {gap:.1f}%p</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

                st.subheader("📊 출전마 분석표")
                st.dataframe(race_data[["AI_예측순위", "chulNo", "hrName", "jkName", "AI_승률(%)", "시장_승률(%)", "AI_EDGE(%p)", "winOdds", "EV_기대값"]], use_container_width=True)
