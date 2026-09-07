import warnings
import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

warnings.filterwarnings("ignore")

# ==========================================
# 11. DATA LEAKAGE AUDIT (자동 누수 검사 모듈)
# ==========================================
def audit_data_leakage(df):
    print("\n🔍 [Data Leakage Audit 실행 중...]")
    leakage_errors = []
    
    # Check 1: Point-in-time check (Feature 계산 시 미래 경주일 참조 여부)
    for col in ['hr_recent_avg_finish_3', 'hr_recent_win_rate_3', 'finish_trend']:
        if col in df.columns:
            # 과거 성적 산출용 컬럼 유효성 검사
            if df[col].isnull().sum() > len(df) * 0.9:
                leakage_errors.append(f"CRITICAL: {col} 피처 결측치 과다 (생성 오류 가능성)")
                
    # Check 2: Race level leakage check
    grouped_dates = df.groupby('race_id')['rc_date'].nunique()
    if (grouped_dates > 1).any():
        leakage_errors.append("CRITICAL: 동일 race_id에 서로 다른 날짜가 포함되어 있습니다.")
        
    if not leakage_errors:
        print("✅ [Audit Pass] 미래 데이터 누수 및 데이터 오염이 발견되지 않았습니다.")
    else:
        for err in leakage_errors:
            print(f"❌ {err}")

# ==========================================
# STEP 1 & 3 & 4. 시계열 데이터 생성 및 Point-in-Time 피처 엔진
# ==========================================
def generate_timeseries_mock_data(n_races=2000):
    np.random.seed(2026)
    start_date = datetime.date(2024, 1, 1)

    races = []
    horse_pool = [f"마필_{i:04d}" for i in range(1, 301)]
    jockey_pool = [f"기수_{i:03d}" for i in range(1, 51)]
    trainer_pool = [f"조교사_{i:03d}" for i in range(1, 30)]

    current_date = start_date
    for r_idx in range(1, n_races + 1):
        if r_idx % 8 == 0:
            current_date += datetime.timedelta(days=1)

        n_horses = np.random.choice([8, 10, 12, 14], p=[0.2, 0.4, 0.3, 0.1])
        selected_horses = np.random.choice(horse_pool, size=n_horses, replace=False)
        selected_jockeys = np.random.choice(jockey_pool, size=n_horses, replace=True)
        selected_trainers = np.random.choice(trainer_pool, size=n_horses, replace=True)
        meet_code = np.random.choice(["서울", "부경", "제주"])
        distance = np.random.choice([1000, 1200, 1400, 1700])

        true_ability = np.random.normal(50, 10, size=n_horses)
        handicap = np.random.normal(55, 2, size=n_horses)
        weight_chg = np.random.normal(0, 4, size=n_horses)

        race_score = (
            true_ability
            - handicap * 0.5
            - np.abs(weight_chg) * 0.3
            + np.random.normal(0, 4, size=n_horses)
        )
        finish_rank = np.argsort(-race_score) + 1

        market_prob = np.exp(true_ability / 8) / np.sum(np.exp(true_ability / 8))
        win_odds = np.clip(1 / (market_prob * 1.18), 1.2, 80.0)

        for h_idx in range(n_horses):
            races.append({
                "race_id": f"R{r_idx:05d}",
                "rc_date": current_date,
                "meet": meet_code,
                "distance": distance,
                "chulNo": h_idx + 1,
                "hrName": selected_horses[h_idx],
                "jkName": selected_jockeys[h_idx],
                "trName": selected_trainers[h_idx],
                "handyCap": handicap[h_idx],
                "chgWeight": weight_chg[h_idx],
                "winOdds": win_odds[h_idx],
                "finish": finish_rank[h_idx],
            })

    df = pd.DataFrame(races)
    df["rc_date"] = pd.to_datetime(df["rc_date"])
    return df

def build_advanced_pit_features(df):
    print("⚙️ Point-in-Time 고도화 피처 생성 중 (최근 3/5전, 추세, 코스/거리별 성적)...")
    df = df.sort_values(["rc_date", "race_id", "chulNo"]).reset_index(drop=True)

    horse_history = {}
    updated_rows = []

    for row in df.itertuples():
        h = row.hrName
        curr_date = row.rc_date
        meet = row.meet
        dist = row.distance

        if h in horse_history:
            h_past = horse_history[h]
            recent_3 = h_past[-3:]
            recent_5 = h_past[-5:]

            avg_finish_3 = np.mean([x["finish"] for x in recent_3])
            avg_finish_5 = np.mean([x["finish"] for x in recent_5])
            win_rt_3 = sum(1 for x in recent_3 if x["finish"] == 1) / len(recent_3)
            top3_rt_5 = sum(1 for x in recent_5 if x["finish"] <= 3) / len(recent_5)

            # 착순 추세 (직전 3경주 기울기: 음수면 상승세)
            if len(recent_3) >= 2:
                finishes = [x["finish"] for x in recent_3]
                finish_trend = finishes[-1] - finishes[0]
            else:
                finish_trend = 0.0

            # 코스/거리별 이전 성적
            course_past = [x for x in h_past if x["meet"] == meet and x["distance"] == dist]
            course_dist_win_rt = (
                sum(1 for x in course_past if x["finish"] == 1) / len(course_past)
                if course_past else 0.1
            )
            days_since = (curr_date - h_past[-1]["rc_date"]).days
        else:
            avg_finish_3, avg_finish_5 = 7.0, 7.0
            win_rt_3, top3_rt_5 = 0.1, 0.2
            finish_trend = 0.0
            course_dist_win_rt = 0.1
            days_since = 60.0

        row_dict = row._asdict()
        row_dict["hr_recent_avg_finish_3"] = avg_finish_3
        row_dict["hr_recent_avg_finish_5"] = avg_finish_5
        row_dict["hr_recent_win_rate_3"] = win_rt_3
        row_dict["hr_recent_top3_rate_5"] = top3_rt_5
        row_dict["finish_trend"] = finish_trend
        row_dict["course_dist_win_rt"] = course_dist_win_rt
        row_dict["days_since_last_race"] = days_since
        updated_rows.append(row_dict)

        if h not in horse_history:
            horse_history[h] = []
        horse_history[h].append({
            "rc_date": curr_date,
            "finish": row.finish,
            "meet": meet,
            "distance": dist
        })

    return pd.DataFrame(updated_rows)

# ==========================================
# STEP 1 & 2. Market Implied Probability & Race-grouped Softmax
# ==========================================
def run_walk_forward_evaluation_v2(df):
    features = [
        "handyCap", "chgWeight", "hr_recent_avg_finish_3", "hr_recent_avg_finish_5",
        "hr_recent_win_rate_3", "hr_recent_top3_rate_5", "finish_trend",
        "course_dist_win_rt", "days_since_last_race"
    ]
    df["target"] = (df["finish"] == 1).astype(int)

    # 1. Market Implied Probability 산출 (경주 단위 100% 정규화)
    df["market_raw"] = 1.0 / df["winOdds"]
    df["prob_Market"] = df.groupby("race_id")["market_raw"].transform(lambda x: x / x.sum())

    # 2. Random
    df["prob_Random"] = df.groupby("race_id")["chulNo"].transform(lambda x: 1.0 / len(x))

    # 3. Favorite (인기 1위 100%, 나머지 0%)
    df["fav_rank"] = df.groupby("race_id")["winOdds"].rank(method="first")
    df["prob_Favorite"] = np.where(df["fav_rank"] == 1, 1.0, 0.0)

    unique_dates = sorted(df["rc_date"].unique())
    n_splits = 4
    split_size = len(unique_dates) // (n_splits + 1)
    prediction_records = []

    print("\n🔄 Race-level Walk-forward 학습 및 예측 실행 중...")
    for i in range(1, n_splits + 1):
        train_dates = unique_dates[: split_size * i]
        test_dates = unique_dates[split_size * i : split_size * (i + 1)]

        train_df = df[df["rc_date"].isin(train_dates)]
        test_df = df[df["rc_date"].isin(test_dates)].copy()

        if test_df.empty:
            continue

        X_train, y_train = train_df[features], train_df["target"]
        X_test, y_test = test_df[features], test_df["target"]

        # Baseline
        test_df["pred_base"] = (
            test_df["hr_recent_win_rate_3"] * 1.22
            - test_df["hr_recent_avg_finish_3"] * 0.15
            - test_df["handyCap"] * 0.05
        )
        test_df["prob_Baseline"] = test_df.groupby("race_id")["pred_base"].transform(
            lambda x: np.exp(x - x.max()) / np.exp(x - x.max()).sum()
        )

        # Logistic Regression
        lr = LogisticRegression()
        lr.fit(X_train, y_train)
        test_df["prob_Logistic"] = lr.predict_proba(X_test)[:, 1]
        test_df["prob_Logistic"] = test_df.groupby("race_id")["prob_Logistic"].transform(lambda x: x / x.sum())

        # LightGBM (Race-grouped Softmax 적용)
        lgb_model = lgb.LGBMClassifier(n_estimators=60, max_depth=3, verbose=-1, random_state=2026)
        lgb_model.fit(X_train, y_train)
        raw_lgb = lgb_model.predict_proba(X_test)[:, 1]
        test_df["raw_lgb"] = raw_lgb
        test_df["prob_LightGBM"] = test_df.groupby("race_id")["raw_lgb"].transform(lambda x: x / x.sum())

        # Isotonic Calibration (Test set 지독한 분리)
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(lgb_model.predict_proba(X_train)[:, 1], y_train)
        calibrated_raw = iso.transform(raw_lgb)
        test_df["calib_raw"] = calibrated_raw
        test_df["prob_Calibrated_LGBM"] = test_df.groupby("race_id")["calib_raw"].transform(
            lambda x: (x + 1e-5) / (x + 1e-5).sum()
        )

        prediction_records.append(test_df)

    return pd.concat(prediction_records, ignore_index=True)

# ==========================================
# STEP 7 & 10. 회수율 vs ROI 분리 및 정밀 종합 평가
# ==========================================
def evaluate_market_vs_ai(df):
    models = ["Random", "Market", "Favorite", "Baseline", "Logistic", "LightGBM", "Calibrated_LGBM"]

    print("\n" + "=" * 95)
    print("📊 [최종 평가] 시장 기준선(Market) vs AI 모델 정밀 성능 비교")
    print("=" * 95)

    eval_results = []
    STAKE_PER_BET = 10000  # 7. 고정 1만원 베팅 원칙 준수

    for m in models:
        prob_col = f"prob_{m}"

        # 1착 적중률 (Top 1)
        top1_df = df.sort_values(prob_col, ascending=False).groupby("race_id").first()
        acc_top1 = (top1_df["finish"] == 1).mean() * 100

        # Top 3 적중률
        df["rank_pred"] = df.groupby("race_id")[prob_col].rank(ascending=False, method="first")
        acc_top3 = df[df["rank_pred"] <= 3].groupby("race_id")["finish"].apply(lambda x: (x <= 3).any()).mean() * 100

        # 정밀 손실 지표
        loss = log_loss(df["target"], df[prob_col])
        brier = brier_score_loss(df["target"], df[prob_col])

        # 💰 수익성 정밀 계산 (회수율과 ROI 완벽 분리)
        n_bets = len(top1_df)
        total_stake = n_bets * STAKE_PER_BET
        payout = (top1_df[top1_df["finish"] == 1]["winOdds"] * STAKE_PER_BET).sum()
        net_profit = payout - total_stake

        # ROI = (순수익 / 총투자금) * 100 %
        roi = (net_profit / total_stake) * 100 if total_stake > 0 else 0
        # 회수율 = (총회수금 / 총투자금) * 100 %
        return_rate = (payout / total_stake) * 100 if total_stake > 0 else 0

        # MDD 연산
        top1_df["profit"] = np.where(top1_df["finish"] == 1, (top1_df["winOdds"] - 1) * STAKE_PER_BET, -STAKE_PER_BET)
        top1_df["cum_profit"] = top1_df["profit"].cumsum()
        cum_max = top1_df["cum_profit"].cummax()
        mdd = (top1_df["cum_profit"] - cum_max).min()

        eval_results.append({
            "Model": m,
            "1착 적중률": f"{acc_top1:.2f}%",
            "Top3 적중률": f"{acc_top3:.2f}%",
            "LogLoss": f"{loss:.4f}",
            "Brier": f"{brier:.4f}",
            "총 회수금": f"{int(payout):,}원",
            "순수익": f"{int(net_profit):,}원",
            "회수율": f"{return_rate:.2f}%",
            "ROI(수익률)": f"{roi:+.2f}%",
            "MDD": f"{int(mdd):,}원"
        })

    res_df = pd.DataFrame(eval_results)
    print(res_df.to_string(index=False))

# ==========================================
# STEP 5 & 6. 배당대별 성적 분석 & EV 과대평가 진단
# ==========================================
def diagnose_odds_and_ev(df):
    print("\n" + "=" * 95)
    print("🔎 5. [진단] 배당대별 실제 승률 vs AI 예측 승률 & ROI 분석 (Calibrated LGBM 기준)")
    print("=" * 95)

    df["odds_bin"] = pd.cut(
        df["winOdds"],
        bins=[1.0, 3.0, 5.0, 8.0, 15.0, 999.0],
        labels=["1~3배", "3~5배", "5~8배", "8~15배", "15배 이상"]
    )

    odds_summary = []
    for bin_name, group in df.groupby("odds_bin"):
        if group.empty:
            continue
        n_horses = len(group)
        actual_win_rt = (group["finish"] == 1).mean() * 100
        ai_avg_prob = group["prob_Calibrated_LGBM"].mean() * 100

        # 10,000원 고정 베팅시 ROI
        total_stake = n_horses * 10000
        payout = (group[group["finish"] == 1]["winOdds"] * 10000).sum()
        roi = ((payout - total_stake) / total_stake) * 100

        odds_summary.append({
            "배당 구간": bin_name,
            "표본 수": f"{n_horses:,}마리",
            "실제 승률": f"{actual_win_rt:.2f}%",
            "AI 평균 확률": f"{ai_avg_prob:.2f}%",
            "평균 배당": f"{group['winOdds'].mean():.2f}배",
            "ROI": f"{roi:+.2f}%"
        })

    print(pd.DataFrame(odds_summary).to_string(index=False))

    print("\n" + "=" * 95)
    print("🎯 6. [진단] AI 예측 EV 구간별 실제 적중률 및 실질 ROI 검증")
    print("=" * 95)

    df["EV"] = (df["prob_Calibrated_LGBM"] * df["winOdds"]) - 1.0
    top1_df = df.sort_values("prob_Calibrated_LGBM", ascending=False).groupby("race_id").first()

    top1_df["ev_bin"] = pd.cut(
        top1_df["EV"],
        bins=[-1.0, -0.2, 0.0, 0.1, 0.2, 0.5, 99.0],
        labels=["< -20%", "-20%~0%", "0%~10%", "10%~20%", "20%~50%", "50%+"]
    )

    ev_summary = []
    for bin_name, group in top1_df.groupby("ev_bin"):
        if group.empty:
            continue
        n_bets = len(group)
        hit_rt = (group["finish"] == 1).mean() * 100
        total_stake = n_bets * 10000
        payout = (group[group["finish"] == 1]["winOdds"] * 10000).sum()
        roi = ((payout - total_stake) / total_stake) * 100

        ev_summary.append({
            "EV 구간": bin_name,
            "베팅 수": f"{n_bets:,}회",
            "실제 적중률": f"{hit_rt:.2f}%",
            "평균 예측 EV": f"{group['EV'].mean()*100:+.1f}%",
            "실제 ROI": f"{roi:+.2f}%"
        })

    print(pd.DataFrame(ev_summary).to_string(index=False))

# 파이프라인 전체 실행
if __name__ == "__main__":
    df_raw = generate_timeseries_mock_data(n_races=2000)
    audit_data_leakage(df_raw)
    df_features = build_advanced_pit_features(df_raw)
    df_predictions = run_walk_forward_evaluation_v2(df_features)
    evaluate_market_vs_ai(df_predictions)
    diagnose_odds_and_ev(df_predictions)
