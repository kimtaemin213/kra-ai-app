"""
import xml.etree.ElementTree as ET
import datetime
import numpy as np
import pandas as pd
import requests
import streamlit as st

# ==========================================
# 0. 모바일 레이아웃 및 페이지 기본 설정
# ==========================================
st.set_page_config(
    page_title="KRA AI 경마 예측 시스템", page_icon="🏇", layout="centered"
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

DECODING_KEY = "h7qTh7gmeKJ49Z3vL/4Oss49MVNvYWqGYyDyJgk8cUGs99uBTP1Qit3jk0qs8UVkMiqYWuLoxnNhd6nHuGKA2w=="


# ==========================================
# 1. API 데이터 수집 모듈
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
    """최근 14일 전부터 향후 30일까지의 목/금/토/일 날짜 생성"""
    today = datetime.date.today()
    race_dates = []
    for i in range(-14, 30):
        d = today - datetime.timedelta(days=i)
        if d.weekday() in [3, 4, 5, 6]:  # 목, 금, 토, 일
            race_dates.append(d.strftime("%Y%m%d"))
    return sorted(race_dates, reverse=True)


@st.cache_data(ttl=180)
def get_active_meets_for_date(target_date):
    """선택한 날짜에 실제 경주 데이터가 있는 경마장 코드만 추출"""
    active_meets = []
    meet_names = {"1": "서울 렛츠런파크", "2": "부산경남", "3": "제주"}

    for meet_code in ["1", "2", "3"]:
        params = {
            "serviceKey": DECODING_KEY,
            "pageNo": "1",
            "numOfRows": "5",
            "meet": meet_code,
            "rc_date": target_date,
        }
        df = fetch_api_dataframe(
            "http://apis.data.go.kr/B551015/API21_1/raceHorseList_1", params
        )
        if not df.empty:
            active_meets.append(meet_code)

    # API 미동기화 또는 데이터 조회 불가 시 기본 전체 제공
    if not active_meets:
        return ["1", "2", "3"]

    return active_meets


@st.cache_data(ttl=180)
def collect_kra_data(meet_code, target_date):
    base_params = {
        "serviceKey": DECODING_KEY,
        "pageNo": "1",
        "numOfRows": "100",
        "meet": meet_code,
        "rc_date": target_date,
    }

    df_race = fetch_api_dataframe(
        "http://apis.data.go.kr/B551015/API21_1/raceHorseList_1", base_params
    )
    if df_race.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df_weight = fetch_api_dataframe(
        "http://apis.data.go.kr/B551015/API25/horseWeightList", base_params
    )
    df_jockey = fetch_api_dataframe(
        "http://apis.data.go.kr/B551015/API22/jockeyList", base_params
    )
    df_record = fetch_api_dataframe(
        "http://apis.data.go.kr/B551015/API24/totalRecordList", base_params
    )
    df_odds = fetch_api_dataframe(
        "http://apis.data.go.kr/B551015/API28/oddsInfoList", base_params
    )
    df_track = fetch_api_dataframe(
        "http://apis.data.go.kr/B551015/API30/trackConditionList", base_params
    )
    df_result = fetch_api_dataframe(
        "http://apis.data.go.kr/B551015/API27_1/raceResultList_1", base_params
    )

    df_merged = df_race.copy()

    if not df_weight.empty and "hrName" in df_weight.columns:
        cols = [
            c
            for c in ["hrName", "nowWeight", "chgWeight"]
            if c in df_weight.columns
        ]
        df_merged = pd.merge(df_merged, df_weight[cols], on="hrName", how="left")

    if not df_jockey.empty and "jkName" in df_jockey.columns:
        cols = [
            c
            for c in ["jkName", "jkWinRt", "jkQrt"]
            if c in df_jockey.columns
        ]
        df_merged = pd.merge(df_merged, df_jockey[cols], on="jkName", how="left")

    if not df_record.empty and "hrName" in df_record.columns:
        cols = [
            c
            for c in ["hrName", "hrWinRt", "ord1Cnt", "rating"]
            if c in df_record.columns
        ]
        df_merged = pd.merge(df_merged, df_record[cols], on="hrName", how="left")

    if not df_odds.empty and "chulNo" in df_odds.columns:
        cols = [
            c for c in ["chulNo", "winOdds", "plcOdds"] if c in df_odds.columns
        ]
        df_merged = pd.merge(df_merged, df_odds[cols], on="chulNo", how="left")

    numeric_defaults = {
        "handyCap": 55.0,
        "nowWeight": 480.0,
        "chgWeight": 0.0,
        "jkWinRt": 10.0,
        "hrWinRt": 10.0,
        "ord1Cnt": 1.0,
        "rating": 50.0,
        "winOdds": 5.0,
        "chulNo": 1,
    }

    for col, default_val in numeric_defaults.items():
        if col in df_merged.columns:
            df_merged[col] = pd.to_numeric(
                df_merged[col].astype(str).str.replace("%", ""), errors="coerce"
            ).fillna(default_val)
        else:
            df_merged[col] = default_val

    return df_merged, df_track, df_result


# ==========================================
# 2. AI 승률 연산 ENGINE
# ==========================================
def predict_pure_probabilities(df_raw, df_track, selected_race):
    X = df_raw.copy()

    track_moisture = 5.0
    track_state_str = "양호"
    if not df_track.empty and "rcNo" in df_track.columns:
        target_track = df_track[df_track["rcNo"] == str(selected_race)]
        if not target_track.empty:
            track_moisture = float(target_track.iloc[0].get("humidity", 5.0))
            track_state_str = target_track.iloc[0].get(
                "trackCondition", "양호"
            )

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
# 3. 실시간 UI 구성 (날짜 선선택 구조)
# ==========================================
st.title("🏇 KRA AI 경마 예측 시스템")
st.caption("실제 경주 일정 자동 동기화 대시보드")

with st.expander("⚙️ 경주 일정 및 설정", expanded=True):
    col1, col2 = st.columns(2)

    with col1:
        # 1) 날짜를 먼저 선택
        all_dates = generate_all_possible_dates()
        target_date_str = st.selectbox(
            "📅 경기 날짜 선택",
            options=all_dates,
            format_func=lambda x: f"{x[:4]}-{x[4:6]}-{x[6:]}",
        )

        # 2) 선택한 날짜에 경기가 있는 경마장만 동적 추출
        active_meets = get_active_meets_for_date(target_date_str)
        meet_choice = st.selectbox(
            "🏟️ 경마장 선택 (해당일 개최지)",
            options=active_meets,
            format_func=lambda x: {
                "1": "서울 렛츠런파크",
                "2": "부산경남",
                "3": "제주",
            }[x],
        )

    with col2:
        # 3) 선택한 날짜/경마장의 실제 출전 경주 번호만 동적 추출
        df_check, _, _ = collect_kra_data(meet_choice, target_date_str)

        if not df_check.empty and "rcNo" in df_check.columns:
            actual_races = sorted(
                pd.to_numeric(df_check["rcNo"], errors="coerce")
                .dropna()
                .unique()
                .astype(int)
                .tolist()
            )
            race_options = actual_races if actual_races else list(range(1, 12))
        else:
            race_options = list(range(1, 12))

        selected_race = st.selectbox(
            "🏁 경주 번호 (RACE)", options=race_options, index=0
        )
        total_budget = st.number_input(
            "💵 베팅 예산 (원)", min_value=10000, value=100000, step=10000
        )

run_button = st.button("🚀 실시간 API 승률 분석")

if run_button:
    with st.spinner(
        f"[{target_date_str}] {selected_race}경주 API 데이터 수신 중..."
    ):
        df_raw, df_track, df_real_result = collect_kra_data(
            meet_choice, target_date_str
        )

        if df_raw.empty:
            st.error(
                "⏳ 해당 날짜의 API 수신 데이터가 없습니다.\n\n"
                "• 아직 출전표가 작성되지 않은 미래 날짜이거나, API 키 전산 동기화 중일 수 있습니다."
            )
        else:
            if "rcNo" in df_raw.columns:
                df_target_race = df_raw[
                    df_raw["rcNo"].astype(str) == str(selected_race)
                ].copy()
            else:
                df_target_race = df_raw.copy()

            if df_target_race.empty:
                st.warning(
                    f"⚠️ {target_date_str} 날짜에는 {selected_race}경주의 출전 정보가 없습니다."
                )
            else:
                race_data, moisture, track_state = predict_pure_probabilities(
                    df_target_race, df_track, selected_race
                )

                if (
                    not df_real_result.empty
                    and "ord" in df_real_result.columns
                    and "chulNo" in df_real_result.columns
                ):
                    target_res = df_real_result[
                        df_real_result["rcNo"].astype(str) == str(selected_race)
                    ].copy()
                    target_res["ord"] = pd.to_numeric(
                        target_res["ord"], errors="coerce"
                    )
                    target_res["chulNo"] = pd.to_numeric(
                        target_res["chulNo"], errors="coerce"
                    )
                    race_data = pd.merge(
                        race_data,
                        target_res[["chulNo", "ord"]],
                        on="chulNo",
                        how="left",
                    )
                    race_data.rename(columns={"ord": "실제 착순"}, inplace=True)
                else:
                    race_data["실제 착순"] = np.nan

                race_data = race_data.sort_values(
                    by="AI_예측순위"
                ).reset_index(drop=True)

                st.success(
                    f"🟢 [{target_date_str}] {meet_choice}번 경마장 {selected_race}경주 API 수신 완료"
                )
                st.markdown(
                    f"""
                <div class="metric-card">
                    <h4>💧 당일 함수율: {moisture}% | ☀️ 주로상태: {track_state}</h4>
                    <p>분석 대상: <b>{selected_race}경주</b> | 설정 예산: <b>{total_budget:,}원</b></p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

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
                        st.caption(
                            f"기수: {row['jkName']} | 배당: {row['winOdds']}배"
                        )

                st.subheader("📊 출전마 AI 승률 & 착순 대조표")
                display_cols = [
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
                    "실제 착순",
                ]
                existing_cols = [
                    c for c in display_cols if c in race_data.columns
                ]
                display_df = race_data[existing_cols].copy()

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
                    "실제 착순": "실제착순",
                }
                display_df.rename(columns=rename_map, inplace=True)

                st.dataframe(
                    display_df.style.background_gradient(
                        subset=["AI승률(%)"], cmap="Blues"
                    ).format(
                        {
                            "AI순위": "{:.0f}",
                            "게이트": "{:.0f}",
                            "레이팅": "{:.0f}",
                            "AI승률(%)": "{:.1f}%",
                            "단승배당": "{:.1f}배",
                            "EV기대값": "{:+.2f}",
                            "체중": "{:.0f}kg",
                            "체중변화": "{:+.0f}kg",
                            "실제착순": lambda x: (
                                f"{int(x)}위" if pd.notna(x) else "대기중"
                            ),
                        }
                    ),
                    use_container_width=True,
                )

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
                    '''
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
# 1. 2026년 8월 30일 서울 8경주 실제 형태 더미 데이터 생성
# ==========================================
def load_mock_kra_data():
    # 출전마 및 전적, 기수, 배당, 체중 실제 구조 데이터
    race_data = [
        {
            "rcNo": "8",
            "chulNo": 1,
            "hrName": "글로벌강타",
            "jkName": "문세영",
            "handyCap": 56.0,
            "nowWeight": 488,
            "chgWeight": 2,
            "jkWinRt": 21.5,
            "hrWinRt": 33.3,
            "ord1Cnt": 5,
            "rating": 68,
            "winOdds": 2.4,
            "real_ord": 1,
        },
        {
            "rcNo": "8",
            "chulNo": 2,
            "hrName": "번개의신",
            "jkName": "유승완",
            "handyCap": 54.0,
            "nowWeight": 465,
            "chgWeight": -4,
            "jkWinRt": 11.2,
            "hrWinRt": 15.0,
            "ord1Cnt": 2,
            "rating": 52,
            "winOdds": 8.7,
            "real_ord": 4,
        },
        {
            "rcNo": "8",
            "chulNo": 3,
            "hrName": "위너스맨",
            "jkName": "서승운",
            "handyCap": 57.5,
            "nowWeight": 510,
            "chgWeight": 0,
            "jkWinRt": 19.8,
            "hrWinRt": 40.0,
            "ord1Cnt": 8,
            "rating": 85,
            "winOdds": 3.1,
            "real_ord": 2,
        },
        {
            "rcNo": "8",
            "chulNo": 4,
            "hrName": "제주의빛",
            "jkName": "안혁수",
            "handyCap": 52.0,
            "nowWeight": 442,
            "chgWeight": 6,
            "jkWinRt": 8.5,
            "hrWinRt": 10.0,
            "ord1Cnt": 1,
            "rating": 45,
            "winOdds": 15.3,
            "real_ord": 6,
        },
        {
            "rcNo": "8",
            "chulNo": 5,
            "hrName": "황금돌풍",
            "jkName": "김용근",
            "handyCap": 55.0,
            "nowWeight": 492,
            "chgWeight": -2,
            "jkWinRt": 14.3,
            "hrWinRt": 22.2,
            "ord1Cnt": 4,
            "rating": 60,
            "winOdds": 5.8,
            "real_ord": 3,
        },
        {
            "rcNo": "8",
            "chulNo": 6,
            "hrName": "스피드킹",
            "jkName": "이혁",
            "handyCap": 53.5,
            "nowWeight": 478,
            "chgWeight": 1,
            "jkWinRt": 7.1,
            "hrWinRt": 8.3,
            "ord1Cnt": 1,
            "rating": 42,
            "winOdds": 22.0,
            "real_ord": 7,
        },
        {
            "rcNo": "8",
            "chulNo": 7,
            "hrName": "한라의태양",
            "jkName": "송재철",
            "handyCap": 54.5,
            "nowWeight": 485,
            "chgWeight": 3,
            "jkWinRt": 9.4,
            "hrWinRt": 12.5,
            "ord1Cnt": 2,
            "rating": 50,
            "winOdds": 12.4,
            "real_ord": 5,
        },
    ]

    df_race = pd.DataFrame(race_data)

    # 주로 상태 데이터 (8/30 당일 함유율 6% '양호')
    track_moisture = 6.0
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
# 3. 실시간 UI 구성 (테스트용)
# ==========================================
st.title("🏇 KRA AI 경마 예측 시스템")
st.caption("🧪 [실제 데이터 시뮬레이션] 2026-08-30 서울 8경주 테스트")

with st.expander("⚙️ 경주 일정 및 설정 (8월 마지막주 테스트)", expanded=True):
    col1, col2 = st.columns(2)

    with col1:
        target_date_str = st.selectbox(
            "📅 경기 날짜 선택", options=["2026-08-30 (8월 마지막주 일요일)"]
        )
        meet_choice = st.selectbox("🏟️ 경마장 선택", options=["서울 렛츠런파크"])

    with col2:
        selected_race = st.selectbox("🏁 경주 번호 (RACE)", options=["8경주"])
        total_budget = st.number_input(
            "💵 베팅 예산 (원)", min_value=10000, value=100000, step=10000
        )

run_button = st.button("🚀 실시간 AI 승률 분석 실행 (시뮬레이션)")

if run_button:
    with st.spinner("8월 30일 서울 8경주 데이터 AI 연산 중..."):
        df_raw, moisture, track_state = load_mock_kra_data()
        race_data, moisture, track_state = predict_pure_probabilities(
            df_raw, moisture, track_state
        )

        st.success("🟢 2026-08-30 서울 8경주 데이터 연산 완료!")
        st.markdown(
            f"""
        <div class="metric-card">
            <h4>💧 당일 함수율: {moisture}% | ☀️ 주로상태: {track_state}</h4>
            <p>분석 대상: <b>서울 8경주</b> | 설정 예산: <b>{total_budget:,}원</b></p>
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

        st.dataframe(
            display_df.style.background_gradient(
                subset=["AI승률(%)"], cmap="Blues"
            ).format(
                {
                    "AI순위": "{:.0f}",
                    "게이트": "{:.0f}",
                    "레이팅": "{:.0f}",
                    "AI승률(%)": "{:.1f}%",
                    "단승배당": "{:.1f}배",
                    "EV기대값": "{:+.2f}",
                    "체중": "{:.0f}kg",
                    "체중변화": "{:+.0f}kg",
                    "실제착순": "{:.0f}위",
                }
            ),
            use_container_width=True,
        )

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


