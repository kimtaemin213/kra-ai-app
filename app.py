import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import requests
import streamlit as st

# ==========================================
# 0. UI 레이아웃 및 스타일 설정
# ==========================================
st.set_page_config(
    page_title="KRA AI 개인용 경마 매매 시스템",
    page_icon="🏇",
    layout="wide"  # 깔끔한 대시보드를 위해 wide 레이아웃 적용
)

st.markdown(
    """
    <style>
    .main { padding: 0.8rem; }
    .stButton>button { width: 100%; border-radius: 10px; font-weight: bold; background-color: #0284c7; color: white; height: 3.2em; }
    .metric-card { background-color: #1e293b; padding: 20px; border-radius: 12px; color: white; margin-bottom: 15px; }
    .badge-s { background-color: #15803d; color: white; padding: 4px 8px; border-radius: 6px; font-weight: bold; }
    .badge-a { background-color: #0369a1; color: white; padding: 4px 8px; border-radius: 6px; font-weight: bold; }
    .badge-b { background-color: #b45309; color: white; padding: 4px 8px; border-radius: 6px; font-weight: bold; }
    .badge-pass { background-color: #b91c1c; color: white; padding: 4px 8px; border-radius: 6px; font-weight: bold; }
    .error-box { background-color: #450a0a; border: 1px solid #f87171; padding: 12px; border-radius: 8px; color: #fca5a5; }
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


# ==========================================
# 1. KRA 공식 API 엔드포인트
# ==========================================
ENDPOINTS = {
    "entry_sheet": "https://apis.data.go.kr/B551015/API26_2/entrySheet_2",
    "track_info": "https://apis.data.go.kr/B551015/API189_1/Track_1",
}


# ==========================================
# 2. 실시간 API 수신 모듈
# ==========================================
def fetch_kra_data(url, extra_params):
    key = get_service_key()
    params = {
        "ServiceKey": key,
        "serviceKey": key,
        "pageNo": "1",
        "numOfRows": "100",
        "_type": "xml",
        **extra_params,
    }

    try:
        res = requests.get(url, params=params, timeout=8)
        if res.status_code != 200:
            return None, f"HTTP {res.status_code} Error:\n{res.text[:500]}"

        root = ET.fromstring(res.content)
        err_msg = root.findtext(".//errMsg") or root.findtext(".//returnAuthMsg")
        if err_msg:
            return None, f"🔴 게이트웨이 에러: {err_msg}"

        res_code = root.findtext(".//resultCode")
        res_msg = root.findtext(".//resultMsg")

        if res_code in ["00", "0", "NORMAL_SERVICE", "OK"]:
            items = root.findall(".//item")
            if items:
                data = [{child.tag: child.text for child in item} for item in items]
                return pd.DataFrame(data), None
            return pd.DataFrame(), "🟡 조건에 일치하는 데이터가 없습니다."
        return None, f"🔴 서비스 에러 [{res_code}]: {res_msg}"

    except Exception as e:
        return None, f"💥 통신 연결 장애: {str(e)}"


# ==========================================
# 3. AI Quant Decision Engine (불필요 컬럼 제거 및 UI 정제)
# ==========================================
def run_ai_quant_decision(df_entry, df_track, daily_spent):
    DAILY_LIMIT = 30000
    rem_budget = max(0, DAILY_LIMIT - daily_spent)

    df = df_entry.copy()

    # 수치형 컬럼 보정
    num_cols = ["winOdds", "jkWinRt", "hrWinRt", "rating", "handyCap", "chulNo", "rcNo"]
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

    # 주로 상태(함수율) 파싱
    water_percent = 4.0
    if not df_track.empty and "waterPercent" in df_track.columns:
        try:
            water_percent = float(df_track.iloc[0].get("waterPercent", 4.0))
        except ValueError:
            water_percent = 4.0

    # AI 연산 (Softmax 기반 승률 및 EV 추정)
    humidity_bonus = 0.15 if water_percent >= 10.0 else 0.0
    df["score"] = (
        (df["jkWinRt"] / 25.0) * 1.20
        + (df["hrWinRt"] / 30.0) * 1.30
        + (df["rating"] / 100.0) * 1.05
        - (df["handyCap"] / 60.0) * 0.50
        + humidity_bonus
    )

    exp_s = np.exp(df["score"] - df["score"].max())
    df["AI_승률(%)"] = ((exp_s / exp_s.sum()) * 100).round(1)

    df["market_raw"] = 1.0 / np.maximum(df["winOdds"], 1.05)
    df["시장_승률(%)"] = ((df["market_raw"] / df["market_raw"].sum()) * 100).round(1)

    df["AI_EDGE(%p)"] = (df["AI_승률(%)"] - df["시장_승률(%)"]).round(1)
    df["EV_기대값"] = ((df["AI_승률(%)"] / 100.0) * df["winOdds"]) - 1.0
    df["AI_예측순위"] = (
        df["AI_승률(%)"].rank(ascending=False, method="min").astype(int)
    )

    # 1등 및 상위마 분석
    sorted_df = df.sort_values(by="AI_예측순위").reset_index(drop=True)
    top1 = sorted_df.iloc[0]
    top2 = sorted_df.iloc[1] if len(sorted_df) > 1 else top1
    gap = top1["AI_승률(%)"] - top2["AI_승률(%)"]
    ev = top1["EV_기대값"]

    # 1차 등급판정
    if ev >= 0.40 and gap >= 10.0:
        raw_grade, target_stake = "🔥 S급", 5000
    elif ev >= 0.20 and gap >= 6.0:
        raw_grade, target_stake = "🔷 A급", 3000
    elif ev >= 0.08 and gap >= 4.0:
        raw_grade, target_stake = "📙 B급", 2000
    else:
        raw_grade, target_stake = "🔴 C/D급", 0

    # 2차 잔여 예산 제어
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

    # 🧹 [핵심] 쓸데없는 필드 모두 제거! 실전 매매 표 전용 컬럼 추출 및 한글화
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
        "EV_기대값": "EV 기대값",
    }

    # 존재하는 컬럼만 추출하여 정제
    valid_cols = [c for c in display_cols.keys() if c in sorted_df.columns]
    ui_df = sorted_df[valid_cols].rename(columns=display_cols)

    # 수치 포맷팅 (보기 편하게 %)
    if "EV 기대값" in ui_df.columns:
        ui_df["EV 기대값"] = (ui_df["EV 기대값"] * 100).round(1).astype(str) + "%"
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


# ==========================================
# 4. Streamlit UI
# ==========================================
st.title("🏇 KRA AI 개인용 경마 매매 시스템")
st.caption("실시간 마사회 API 데이터 정제 및 퀀트 매매 콘솔")

if "daily_spent" not in st.session_state:
    st.session_state["daily_spent"] = 0

with st.sidebar:
    st.header("💰 자금 관리 현황")
    st.metric("하루 한도 (DAILY_LIMIT)", "30,000원")
    st.metric("오늘 누적 사용액", f"{st.session_state['daily_spent']:,}원")
    st.metric("남은 베팅 가능 예산", f"{30000 - st.session_state['daily_spent']:,}원")
    if st.button("🔄 리셋 (새로운 날 시작)"):
        st.session_state["daily_spent"] = 0
        st.rerun()

col1, col2 = st.columns(2)
with col1:
    target_date = st.text_input("📅 경기 날짜 (YYYYMMDD)", value="20240901")
    meet_choice = st.selectbox(
        "🏟️ 경마장",
        options=["1", "2", "3"],
        format_func=lambda x: {"1": "서울", "2": "제주", "3": "부산경남"}[x],
    )

with col2:
    selected_race = st.number_input(
        "🏁 경주 번호", min_value=1, max_value=15, value=1
    )

if st.button("🚀 실시간 KRA 데이터 연동 및 AI 분석"):
    with st.spinner("마사회 서버와 실시간 데이터 수신 중..."):
        # 1. 출전표 (API26_2)
        df_entry, err_entry = fetch_kra_data(
            ENDPOINTS["entry_sheet"], {"meet": meet_choice, "rc_date": target_date}
        )
        # 2. 경주로 정보 (API189_1)
        df_track, _ = fetch_kra_data(
            ENDPOINTS["track_info"],
            {
                "meet": meet_choice,
                "rc_date_fr": target_date,
                "rc_date_to": target_date,
            },
        )

        if err_entry:
            st.markdown(
                f'<div class="error-box"><h4>❌ 출전표 API 수신 에러</h4><pre>{err_entry}</pre></div>',
                unsafe_allow_html=True,
            )
        elif df_entry.empty:
            st.warning("해당 조건의 출전표 데이터가 존재하지 않습니다.")
        else:
            # 특정 경주 번호 데이터만 필터링 (rcNo 기준)
            if "rcNo" in df_entry.columns:
                df_race_target = df_entry[df_entry["rcNo"].astype(str) == str(selected_race)].copy()
            else:
                df_race_target = df_entry.copy()

            if df_race_target.empty:
                st.warning(f"{selected_race}경주 출전 데이터가 없습니다.")
            else:
                st.success("🟢 KRA 실시간 출전표 수신 및 정제 완료!")

                (
                    ui_df,
                    water_pct,
                    grade,
                    badge_cls,
                    actual_stake,
                    action,
                    gap,
                    rem_budget,
                    top1,
                ) = run_ai_quant_decision(
                    df_race_target, df_track, st.session_state["daily_spent"]
                )

                st.markdown(
                    f"""
                <div class="metric-card">
                    <h3>의사결정: <span class="{badge_cls}">{grade}</span></h3>
                    <p><b>실제 추천 투입금:</b> <span style="color:#f59e0b; font-size:1.3em; font-weight:bold;">{actual_stake:,}원</span> (남은 한도: {rem_budget:,}원)</p>
                    <p>💧 주로 함수율: {water_pct}% | 🥇 1위 추천마: <b>{top1.get('chulNo','')}번 ({top1.get('hrName','')})</b> | Top1-2 격차: {gap:.1f}%p</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

                if action == "BET" and actual_stake > 0:
                    if st.button(f"💵 {actual_stake:,}원 매매 실행 (잔액 차감)"):
                        st.session_state["daily_spent"] += actual_stake
                        st.success(
                            f"{actual_stake:,}원 투입 완료! (오늘 총 사용:"
                            f" {st.session_state['daily_spent']:,}원)"
                        )
                        st.rerun()

                st.subheader(f"📊 {selected_race}경주 핵심 AI 퀀트 정제표")
                # 쓸데없는 컬럼 제거 후 정제된 깔끔한 UI DataFrame 출력
                st.dataframe(ui_df, use_container_width=True, hide_index=True)
