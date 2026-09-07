import numpy as np
import pandas as pd
import streamlit as st

# ==========================================
# 0. 모바일 레이아웃 및 페이지 기본 설정
# ==========================================
st.set_page_config(
    page_title="KRA AI 경마 예측 시스템 (테스트 모드)",
    page_icon="🏇",
    layout="centered",
)

st.markdown(
    """
    <style>
    .main { padding: 0.8rem; }
    .stButton>button { width: 100%; border-radius: 10px; font-weight: bold; background-color: #0284c7; color: white; height: 3.2em; }
    .metric-card { background-color: #1e293b; padding: 15px; border-radius: 12px; color: white; margin-bottom: 12px; }
    </style>
""",
    unsafe_allow_html=True,
)


# ==========================================
# 1. 2026년 9월 5일(토) 제주 1경주 시뮬레이션 데이터
# ==========================================
def load_last_week_jeju_data():
    race_data = [
        {
            "rcNo": "1",
            "chulNo": 1,
            "hrName": "제주의꿈",
            "jkName": "강득수",
            "handyCap": 54.0,
            "nowWeight": 310,
            "chgWeight": 2,
            "jkWinRt": 12.5,
            "hrWinRt": 18.2,
            "ord1Cnt": 3,
            "rating": 45,
            "winOdds": 4.2,
            "real_ord": 2,
        },
        {
            "rcNo": "1",
            "chulNo": 2,
            "hrName": "한라영웅",
            "jkName": "한영민",
            "handyCap": 56.5,
            "nowWeight": 325,
            "chgWeight": -3,
            "jkWinRt": 18.0,
            "hrWinRt": 28.5,
            "ord1Cnt": 5,
            "rating": 58,
            "winOdds": 2.1,
            "real_ord": 1,
        },
        {
            "rcNo": "1",
            "chulNo": 3,
            "hrName": "탐라태양",
            "jkName": "김준호",
            "handyCap": 53.0,
            "nowWeight": 298,
            "chgWeight": 0,
            "jkWinRt": 8.1,
            "hrWinRt": 10.0,
            "ord1Cnt": 1,
            "rating": 38,
            "winOdds": 11.5,
            "real_ord": 5,
        },
        {
            "rcNo": "1",
            "chulNo": 4,
            "hrName": "제주신화",
            "jkName": "문현진",
            "handyCap": 55.0,
            "nowWeight": 318,
            "chgWeight": 4,
            "jkWinRt": 14.2,
            "hrWinRt": 20.0,
            "ord1Cnt": 4,
            "rating": 50,
            "winOdds": 5.8,
            "real_ord": 3,
        },
        {
            "rcNo": "1",
            "chulNo": 5,
            "hrName": "오름돌풍",
            "jkName": "이재웅",
            "handyCap": 52.0,
            "nowWeight": 305,
            "chgWeight": -1,
            "jkWinRt": 6.5,
            "hrWinRt": 7.5,
            "ord1Cnt": 1,
            "rating": 32,
            "winOdds": 18.2,
            "real_ord": 6,
        },
        {
            "rcNo": "1",
            "chulNo": 6,
            "hrName": "번개호",
            "jkName": "정명일",
            "handyCap": 54.5,
            "nowWeight": 312,
            "chgWeight": 1,
            "jkWinRt": 10.4,
            "hrWinRt": 15.0,
            "ord1Cnt": 2,
            "rating": 42,
            "winOdds": 8.0,
            "real_ord": 4,
        },
        {
            "rcNo": "1",
            "chulNo": 7,
            "hrName": "삼다수",
            "jkName": "곽용남",
            "handyCap": 53.5,
            "nowWeight": 302,
            "chgWeight": -2,
            "jkWinRt": 5.0,
            "hrWinRt": 4.0,
            "ord1Cnt": 0,
            "rating": 28,
            "winOdds": 35.0,
            "real_ord": 7,
        },
    ]

    df_race = pd.DataFrame(race_data)
    track_moisture = 8.0
    track_state_str = "양호"

    return df_race, track_moisture, track_state_str


# ==========================================
# 2. AI 승률 연산 ENGINE
# ==========================================
def predict_pure_probabilities(df_raw, track_moisture, track_state_str):
    X = df_raw.copy()

    if track_moisture >= 10.0:
        moisture_penalty = 0.5
        speed_bonus = 1.2
    else:
        moisture_penalty = 0.4
        speed_bonus = 1.0

    market_expectation = (10.0 / X["winOdds"].clip(lower=1.1)).clip(upper=8.0)
    rating_score = (X["rating"] - 50).clip(lower=0) * 0.3

    X["score_base"] = (
        (X["jkWinRt"] * 1.2)
        + (X["hrWinRt"] * 1.2 * speed_bonus)
        + (X["ord1Cnt"] * 0.8)
        + rating_score
        + market_expectation
        - (X["handyCap"] * moisture_penalty)
        - (X["chgWeight"].abs() * 0.5)
    )

    exp_scores = np.exp(X["score_base"] / 25.0)
    sum_exp = exp_scores.sum()
    X["AI_승률(%)"] = ((exp_scores / sum_exp) * 100).round(1)
    X["AI_예측순위"] = X["AI_승률(%)"].rank(ascending=False, method="min")
    X["EV_기대값"] = ((X["AI_승률(%)"] / 100.0) * X["winOdds"]) - 1.0

    return (
        X.sort_values(by="AI_예측순위").reset_index(drop=True),
        track_moisture,
        track_state_str,
    )


# ==========================================
# 3. 실시간 UI 구성
# ==========================================
st.title("🏇 KRA AI 경마 예측 시스템")
st.caption("🧪 [지난주 데이터 테스트] 2026-09-05 제주 1경주 백테스트")

with st.expander("⚙️ 경주 일정 및 설정 (테스트 데이터)", expanded=True):
    col1, col2 = st.columns(2)

    with col1:
        target_date_str = st.selectbox(
            "📅 경기 날짜 선택", options=["2026-09-05 (지난주 토요일)"]
        )
        meet_choice = st.selectbox("🏟️ 경마장 선택", options=["제주 경마장"])

    with col2:
        selected_race = st.selectbox("🏁 경주 번호 (RACE)", options=["1경주"])
        total_budget = st.number_input(
            "💵 베팅 예산 (원)", min_value=10000, value=100000, step=10000
        )

run_button = st.button("🚀 지난주 경기 AI 승률 검증 실행")

if run_button:
    with st.spinner("9월 5일 제주 1경주 데이터 분석 및 대조 연산 중..."):
        df_raw, moisture, track_state = load_last_week_jeju_data()
        race_data, moisture, track_state = predict_pure_probabilities(
            df_raw, moisture, track_state
        )

        st.success("🟢 2026-09-05 제주 1경주 데이터 연산 완료!")
        st.markdown(
            f"""
        <div class="metric-card">
            <h4>💧 당일 함수율: {moisture}% | ☀️ 주로상태: {track_state}</h4>
            <p>분석 대상: <b>제주 1경주</b> | 설정 예산: <b>{total_budget:,}원</b></p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # TOP 3 메트릭 카드
        st.subheader("🥇 AI 순수 예측 TOP 3")
        top_cols = st.columns(3)
        badges = ["🥇 1위", "🥈 2위", "🥉 3위"]
        for idx in range(min(3, len(race_data))):
            row = race_data.iloc[idx]
            with top_cols[idx]:
                st.metric(
                    label=f"{badges[idx]} ({int(row['chulNo'])}번)",
                    value=f"{row['hrName']}",
                    delta=f"승률 {row['AI_승률(%)']}%",
                )
                st.caption(f"기수: {row['jkName']} | 배당: {row['winOdds']}배")

        # 전체 출전마 AI 분석표 및 실제 착순 대조
        st.subheader("📊 출전마 AI 승률 & 실제 착순 대조표")
        display_df = race_data[
            [
                "AI_예측순위",
                "chulNo",
                "hrName",
                "jkName",
                "rating",
                "AI_승률(%)",
                "winOdds",
                "EV_기대값",
                "nowWeight",
                "chgWeight",
                "real_ord",
            ]
        ].copy()

        rename_map = {
            "AI_예측순위": "AI순위",
            "chulNo": "게이트",
            "hrName": "마명",
            "jkName": "기수명",
            "rating": "레이팅",
            "AI_승률(%)": "AI승률(%)",
            "winOdds": "단승배당",
            "EV_기대값": "EV기대값",
            "nowWeight": "체중",
            "chgWeight": "체중변화",
            "real_ord": "실제착순",
        }
        display_df.rename(columns=rename_map, inplace=True)

        st.dataframe(display_df, use_container_width=True)

        # AI 추천 베팅 포트폴리오
        st.subheader("💰 AI 추천 실전 베팅 포트폴리오")
        top1 = race_data.iloc[0]
        top2 = race_data.iloc[1]
        st.warning(
            f"🎯 **[복승식 메인 (50%)]**: {int(top1['chulNo'])}번({top1['hrName']}) - {int(top2['chulNo'])}번({top2['hrName']}) | 추천액: **{int(total_budget*0.5):,}원**"
        )

        if len(race_data) >= 4:
            sub1, sub2 = race_data.iloc[2], race_data.iloc[3]
            st.info(
                f"🛡️ **[삼복승식 서브 (30%)]**: {int(top1['chulNo'])} - {int(top2['chulNo'])} - {int(sub1['chulNo'])} / {int(sub2['chulNo'])} | 추천액: **{int(total_budget*0.3):,}원**"
            )

        best_ev_row = race_data.sort_values(
            by="EV_기대값", ascending=False
        ).iloc[0]
        if best_ev_row["EV_기대값"] > 0:
            st.success(
                f"🔥 **[단승식 가치베팅 (20%)]**: {int(best_ev_row['chulNo'])}번({best_ev_row['hrName']}) | 기대수익률: +{best_ev_row['EV_기대값']*100:.1f}% | 추천액: **{int(total_budget*0.2):,}원**"
            )
        else:
            st.info(
                f"🔒 **[단승식 안전방어 (20%)]**: {int(top1['chulNo'])}번({top1['hrName']}) | 추천액: **{int(total_budget*0.2):,}원**"
            )
