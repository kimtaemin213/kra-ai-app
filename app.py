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
# 1. API 데이터 수집 및 실제 경주일자 생성
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


def generate_actual_race_dates(meet_code):
    days_map = {
        "1": [3, 4, 5, 6],  # 목(3), 금(4), 토(5), 일(6)
        "2": [3, 4, 5, 6],
        "3": [3, 4, 5, 6],
    }
    target_days = days_map.get(meet_code, [3, 4, 5, 6])

    today = datetime.date.today()
    race_dates = []

    for i in range(-14, 60):
        d = today - datetime.timedelta(days=i)
        if d.weekday() in target_days:
            race_dates.append(d.strftime("%Y%m%d"))

    return sorted(race_dates, reverse=True)


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
# 3. 실시간 UI 구성 (실제 경주 번호 동적 추출)
# ==========================================
st.title("🏇 KRA AI 경마 예측 시스템")
st.caption("실제 경주 일정 자동 동기화 대시보드")

with st.expander("⚙️ 경주 일정 및 설정 (실제 경기 날짜만 표시)", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        meet_choice = st.selectbox(
            "경마장 선택",
            options=["1", "2", "3"],
            format_func=lambda x: {
                "1": "서울 렛츠런파크",
                "2": "부산경남",
                "3": "제주",
            }[x],
        )

        valid_dates = generate_actual_race_dates(meet_choice)
        target_date_str = st.selectbox(
            "📅 실제 경기 날짜 선택",
            options=valid_dates,
            format_func=lambda x: f"{x[:4]}-{x[4:6]}-{x[6:]}",
        )

    with col2:
        # 💡 해당 날짜의 전체 출전표 데이터를 가져와 실제 존재하는 경주 번호(rcNo) 목록만 자동 추출
        df_check, _, _ = collect_kra_data(meet_choice, target_date_str)

        if not df_check.empty and "rcNo" in df_check.columns:
            # API 데이터에서 실제 존재하는 경주 번호 추출 및 정렬 (예: 1~8 또는 1~11)
            actual_races = sorted(
                pd.to_numeric(df_check["rcNo"], errors="coerce")
                .dropna()
                .unique()
                .astype(int)
                .tolist()
            )
            race_options = actual_races if actual_races else list(range(1, 12))
        else:
            # API 키 동기화 전이거나 데이터가 없는 날짜일 경우 기본 범위 설정
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
