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
    .error-box { background-color: #450a0a; border: 1px solid #f87171; padding: 12px; border-radius: 8px; color: #fca5a5; }
    </style>
""",
    unsafe_allow_html=True,
)

# ==========================================
# 1. 인증키 및 현행 API 엔드포인트 설정
# ==========================================
# Streamlit Secrets에서 인증키 수급 (하드코딩 제거)
# st.secrets["KRA_SERVICE_KEY"] 가 없으면 기본 안내 메시지 출력
def get_service_key():
    try:
        return st.secrets["KRA_SERVICE_KEY"]
    except Exception:
        # 테스트 및 안내용 로컬 세션 처리
        return st.session_state.get("custom_service_key", "")

# 🎯 공공데이터포털 현행 공식 API 엔드포인트 매핑 구조
API_BASE_URL = "http://apis.data.go.kr/B551015"

CURRENT_ENDPOINTS = {
    # 1. 경주마 명단 / 기본 정보 (공식: racehorselist/getracehorselist)
    "race_horse": f"{API_BASE_URL}/racehorselist/getracehorselist",
    
    # 2. 경주별 상세 성적표 (공식 현행 서비스: 경주일자, 경주번호, 착순, 기수, 마체중, 배당, 레이팅 포함)
    "race_detail_result": f"{API_BASE_URL}/API299/raceResultList_1",  
    
    # 3. 주로 / 날씨 / 함수율 (현행)
    "track_condition": f"{API_BASE_URL}/API189_1/trackConditionList"
}

# ==========================================
# 2. 투명한 API 호출 및 에러 디버깅 엔진
# ==========================================
def fetch_api_with_detailed_logging(url, params):
    """
    에러를 은폐하지 않고 HTTP Status 및 XML 에러 원문을 그대로 파싱하여 전달하는 함수
    """
    service_key = get_service_key()
    if not service_key:
        return None, "🔑 인증키가 설정되지 않았습니다. Secrets 키 또는 인증키를 입력해주세요."

    full_params = {"serviceKey": service_key, **params}
    
    try:
        response = requests.get(url, params=full_params, timeout=8)
        status_code = response.status_code
        req_url = response.url

        # HTTP Status 가 200이 아닌 경우
        if status_code != 200:
            error_msg = f"HTTP {status_code} Error\nURL: {req_url}\nResponse Body:\n{response.text[:1000]}"
            return None, error_msg

        # HTTP 200 OK 내 XML 응답 파싱
        try:
            root = ET.fromstring(response.content)
            
            # 1. 공공데이터 포털 서버 에러 헤더 체크
            err_msg = root.findtext(".//errMsg")
            return_auth_msg = root.findtext(".//returnAuthMsg")
            if err_msg or return_auth_msg:
                error_detail = f"🔴 API 게이트웨이 에러:\n- errMsg: {err_msg}\n- returnAuthMsg: {return_auth_msg}\n- URL: {req_url}"
                return None, error_detail

            # 2. 비즈니스 resultCode 체크
            res_code = root.findtext(".//resultCode")
            res_msg = root.findtext(".//resultMsg")

            if res_code in ["00", "0", "NORMAL_SERVICE", "OK"]:
                items = root.findall(".//item")
                if items:
                    data = [{child.tag: child.text for child in item} for item in items]
                    return pd.DataFrame(data), None
                else:
                    return pd.DataFrame(), f"🟡 [정상 응답] 조회된 데이터(item)가 없습니다. (날짜/경주번호 확인 필요)"
            else:
                return None, f"🔴 서비스 에러 [{res_code}]: {res_msg}"

        except ET.ParseError:
            return None, f"🔴 XML 파싱 에러 (응답이 XML 규격이 아닙니다):\n{response.text[:500]}"

    except requests.exceptions.Timeout:
        return None, "💥 통신 시간 초과 (Timeout 8초)"
    except Exception as e:
        return None, f"💥 네트워크 에러: {str(e)}"

# ==========================================
# 3. AI 의사결정 및 자금 관리 엔진 (DAILY_LIMIT 차감 적용)
# ==========================================
def evaluate_race_decision(df_target, daily_spent):
    DAILY_LIMIT = 30000
    rem_budget = max(0, DAILY_LIMIT - daily_spent)

    df = df_target.copy()

    # 수치 변환
    numeric_cols = ["winOdds", "jkWinRt", "hrWinRt", "rating", "handyCap"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace("%", ""), errors="coerce").fillna(0.0)
        else:
            df[col] = 1.0 if col == "winOdds" else 10.0

    # 시장 확률
    df["market_raw"] = 1.0 / np.maximum(df["winOdds"], 1.1)
    df["시장_승률(%)"] = ((df["market_raw"] / df["market_raw"].sum()) * 100).round(1)

    # AI Softmax Prob
    df["score"] = (
        (df["jkWinRt"] / 25.0) * 1.2 +
        (df["hrWinRt"] / 30.0) * 1.3 +
        (df["rating"] / 100.0) * 1.0 -
        (df["handyCap"] / 60.0) * 0.5
    )
    exp_s = np.exp(df["score"] - df["score"].max())
    df["AI_승률(%)"] = ((exp_s / exp_s.sum()) * 100).round(1)
    df["AI_EDGE(%p)"] = (df["AI_승률(%)"] - df["시장_승률(%)"]).round(1)
    df["EV_기대값"] = ((df["AI_승률(%)"] / 100.0) * df["winOdds"]) - 1.0
    df["AI_예측순위"] = df["AI_승률(%)"].rank(ascending=False, method="min").astype(int)

    sorted_df = df.sort_values(by="AI_예측순위").reset_index(drop=True)
    top1 = sorted_df.iloc[0]
    top2 = sorted_df.iloc[1] if len(sorted_df) > 1 else top1
    gap = top1["AI_승률(%)"] - top2["AI_승률(%)"]
    ev = top1["EV_기대값"]

    # 1차 등급 및 베팅 기본금 결정
    if ev >= 0.40 and gap >= 10.0:
        raw_grade, target_stake = "🔥 S급", 5000
    elif ev >= 0.20 and gap >= 6.0:
        raw_grade, target_stake = "🔷 A급", 3000
    elif ev >= 0.08 and gap >= 4.0:
        raw_grade, target_stake = "📙 B급", 2000
    else:
        raw_grade, target_stake = "🔴 C/D급", 0

    # 2차 잔여 예산 제어 (DAILY_LIMIT 차감검사)
    if target_stake == 0:
        action = "PASS"
        final_grade = "🔴 PASS (조건 미달)"
        badge_cls = "badge-pass"
        actual_stake = 0
    elif rem_budget < target_stake:
        action = "PASS"
        final_grade = f"🔒 PASS (예산 부족: 남은 예산 {rem_budget:,}원 < 필요금액 {target_stake:,}원)"
        badge_cls = "badge-pass"
        actual_stake = 0
    else:
        action = "BET"
        final_grade = f"{raw_grade} (추천)"
        badge_cls = "badge-s" if "S급" in raw_grade else ("badge-a" if "A급" in raw_grade else "badge-b")
        actual_stake = target_stake

    return sorted_df, final_grade, badge_cls, actual_stake, action, gap, rem_budget

# ==========================================
# 4. Streamlit UI
# ==========================================
st.title("🏇 KRA AI 개인용 경마 매매 시스템")
st.caption("공공데이터 포털 현행 API 단락 진단 및 퀀트 의사결정 콘솔")

# Secrets 설정 확인 및 로컬 키 입력
if not get_service_key():
    st.warning("⚠️ Secrets 설정에서 `KRA_SERVICE_KEY`가 발견되지 않았습니다.")
    user_key = st.text_input("🔑 재발급받은 공공데이터 포털 서비스 키(디코딩/인코딩 원본)를 입력하세요", type="password")
    if user_key:
        st.session_state["custom_service_key"] = user_key
        st.success("인증키가 세션에 임시 설정되었습니다.")

# 일일 누적 베팅금 상태 관리
if "daily_spent" not in st.session_state:
    st.session_state["daily_spent"] = 0

with st.sidebar:
    st.header("💰 일일 자금 관리 현황")
    st.metric("하루 한도 (DAILY_LIMIT)", "30,000원")
    st.metric("오늘 누적 베팅액", f"{st.session_state['daily_spent']:,}원")
    st.metric("남은 베팅 가능 예산", f"{30000 - st.session_state['daily_spent']:,}원")
    if st.button("🔄 자금 리셋 (새로운 날 시작)"):
        st.session_state["daily_spent"] = 0
        st.rerun()

col1, col2 = st.columns(2)
with col1:
    target_date_str = st.text_input("📅 경기 날짜 (YYYYMMDD)", value="20240901")
    meet_choice = st.selectbox("🏟️ 경마장", options=["1", "2", "3"], format_func=lambda x: {"1": "서울", "2": "부산경남", "3": "제주"}[x])

with col2:
    selected_race = st.number_input("🏁 경주 번호", min_value=1, max_value=15, value=1)
    target_api = st.selectbox("🔌 테스트할 API 선택", options=[
        ("race_detail_result", "경주별 상세 성적표 (API299)"),
        ("race_horse", "경주마 명단 (racehorselist)"),
        ("track_condition", "주로/날씨 정보 (API189_1)")
    ], format_func=lambda x: x[1])

if st.button("🚀 선택한 현행 API 단락 테스트 및 AI 분석"):
    api_key_name = target_api[0]
    api_url = CURRENT_ENDPOINTS[api_key_name]
    
    params = {
        "pageNo": "1",
        "numOfRows": "20",
        "meet": meet_choice,
        "rc_date": target_date_str,
        "rc_no": str(selected_race)
    }

    with st.spinner(f"[{CURRENT_ENDPOINTS[api_key_name]}] 호출 중..."):
        df_result, error_msg = fetch_api_with_detailed_logging(api_url, params)

        if error_msg:
            st.markdown(f"""
            <div class="error-box">
                <h4>❌ API 호출 실패 상세원인</h4>
                <pre>{error_msg}</pre>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.success("🟢 API 데이터 수신 완전 성공!")
            
            # AI 의사결정 및 자금 차감 검사
            res_df, grade, badge_cls, actual_stake, action, gap, rem_budget = evaluate_race_decision(
                df_result, st.session_state["daily_spent"]
            )
            top1 = res_df.iloc[0]

            st.markdown(f"""
            <div class="metric-card">
                <h3>의사결정: <span class="{badge_cls}">{grade}</span></h3>
                <p><b>실제 투입 권장금액:</b> <span style="color:#f59e0b; font-size:1.3em; font-weight:bold;">{actual_stake:,}원</span> (남은 한도: {rem_budget:,}원)</p>
                <p>Top1-2 승률 격차: {gap:.1f}%p | AI 승률: {top1['AI_승률(%)']}% | 시장 승률: {top1['시장_승률(%)']}%</p>
            </div>
            """, unsafe_allow_html=True)

            if action == "BET" and actual_stake > 0:
                if st.button(f"💵 {actual_stake:,}원 매매 실행 (일일 한도 차감)"):
                    st.session_state["daily_spent"] += actual_stake
                    st.success(f"{actual_stake:,}원 투입 완료! (오늘 누적 사용: {st.session_state['daily_spent']:,}원)")
                    st.rerun()

            st.subheader("📊 수신된 KRA 데이터 기반 AI 분석표")
            st.dataframe(res_df, use_container_width=True)
