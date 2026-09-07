import numpy as np
import pandas as pd
import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# STEP 1. API 수신 데이터 모사 시계열 데이터 생성기
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
        
        # 순수 능력치
        true_ability = np.random.normal(50, 10, size=n_horses)
        handicap = np.random.normal(55, 2, size=n_horses)
        weight_chg = np.random.normal(0, 4, size=n_horses)
        
        # 경기 결과 산출
        race_score = true_ability - handicap * 0.5 - np.abs(weight_chg) * 0.3 + np.random.normal(0, 5, size=n_horses)
        finish_rank = np.argsort(-race_score) + 1
        
        # 시장 배당률 형성
        market_prob = np.exp(true_ability / 10) / np.sum(np.exp(true_ability / 10))
        win_odds = np.clip(1 / (market_prob * 1.18), 1.2, 99.0)
        
        for h_idx in range(n_horses):
            races.append({
                "race_id": f"R{r_idx:05d}",
                "rc_date": current_date,
                "meet": np.random.choice(["1", "2", "3"]),
                "chulNo": h_idx + 1,
                "hrName": selected_horses[h_idx],
                "jkName": selected_jockeys[h_idx],
                "trName": selected_trainers[h_idx],
                "handyCap": handicap[h_idx],
                "chgWeight": weight_chg[h_idx],
                "winOdds": win_odds[h_idx],
                "finish": finish_rank[h_idx]
            })
            
    df = pd.DataFrame(races)
    df['rc_date'] = pd.to_datetime(df['rc_date'])
    return df

# ==========================================
# STEP 2 & STEP 12. 미래 정보 누수 완전 차단 피처 엔지니어링
# ==========================================
def build_leakage_free_features(df):
    print("⚙️ 시계열 데이터 가공 및 시계열 피처 생성 중 (누수 방지)...")
    df = df.sort_values(['rc_date', 'race_id', 'chulNo']).reset_index(drop=True)
    
    # 과거 이력 저장용 컬럼 초기화
    df['hr_recent_win_rt_3'] = 0.0
    df['hr_recent_avg_rank_5'] = 8.0
    df['days_since_last_race'] = 999.0
    df['jk_tr_combo_win_rt'] = 0.0
    
    # 누수 방지를 위한 마필/조합별 내역 사전
    horse_history = {}
    combo_history = {}
    
    updated_rows = []
    
    for row in df.itertuples():
        h = row.hrName
        combo = (row.jkName, row.trName)
        curr_date = row.rc_date
        
        # 1. 마필 피처 계산
        if h in horse_history:
            h_past = horse_history[h]
            # 최근 3경주 승률
            recent_3 = h_past[-3:]
            win_rt_3 = sum(1 for x in recent_3 if x['finish'] == 1) / len(recent_3)
            # 최근 5경주 평균 착순
            recent_5 = h_past[-5:]
            avg_rank_5 = sum(x['finish'] for x in recent_5) / len(recent_5)
            # 직전 경기 후 경과일
            days_since = (curr_date - h_past[-1]['rc_date']).days
        else:
            win_rt_3 = 0.1
            avg_rank_5 = 7.0
            days_since = 60.0
            
        # 2. 기수x조교사 조합 피처 계산
        if combo in combo_history:
            c_past = combo_history[combo]
            combo_win_rt = sum(1 for x in c_past if x['finish'] == 1) / len(c_past)
        else:
            combo_win_rt = 0.1
            
        row_dict = row._asdict()
        row_dict['hr_recent_win_rt_3'] = win_rt_3
        row_dict['hr_recent_avg_rank_5'] = avg_rank_5
        row_dict['days_since_last_race'] = days_since
        row_dict['jk_tr_combo_win_rt'] = combo_win_rt
        updated_rows.append(row_dict)
        
        # 경주 종료 후 이력 업데이트 (현재 경주는 이전 경주 피처에만 영향)
        if h not in horse_history:
            horse_history[h] = []
        horse_history[h].append({'rc_date': curr_date, 'finish': row.finish})
        
        if combo not in combo_history:
            combo_history[combo] = []
        combo_history[combo].append({'finish': row.finish})
        
    res_df = pd.DataFrame(updated_rows)
    return res_df

# ==========================================
# STEP 3 ~ STEP 6. Walk-forward 백테스트 & 모델 비교
# ==========================================
def run_walk_forward_evaluation(df):
    features = [
        'handyCap', 'chgWeight', 'hr_recent_win_rt_3', 
        'hr_recent_avg_rank_5', 'days_since_last_race', 'jk_tr_combo_win_rt'
    ]
    df['target'] = (df['finish'] == 1).astype(int)
    
    # 3개월 단위 롤링 윈도우 스플릿
    unique_dates = sorted(df['rc_date'].unique())
    n_splits = 4
    split_size = len(unique_dates) // (n_splits + 1)
    
    results = {'Baseline': [], 'Logistic': [], 'LightGBM': [], 'LGBM_Calibrated': []}
    prediction_records = []
    
    print("\n🔄 Walk-forward 백테스트 진행 중...")
    for i in range(1, n_splits + 1):
        train_dates = unique_dates[:split_size * i]
        test_dates = unique_dates[split_size * i : split_size * (i + 1)]
        
        train_df = df[df['rc_date'].isin(train_dates)]
        test_df = df[df['rc_date'].isin(test_dates)].copy()
        
        if test_df.empty:
            continue
            
        X_train, y_train = train_df[features], train_df['target']
        X_test, y_test = test_df[features], test_df['target']
        
        # 1. Baseline 수동 가중치 모델
        test_df['pred_base_raw'] = (
            test_df['hr_recent_win_rt_3'] * 1.22 + 
            (10 / test_df['hr_recent_avg_rank_5']) * 1.05 - 
            test_df['handyCap'] * 0.05
        )
        test_df['prob_Baseline'] = test_df.groupby('race_id')['pred_base_raw'].transform(lambda x: np.exp(x)/np.exp(x).sum())
        
        # 2. Logistic Regression
        lr = LogisticRegression()
        lr.fit(X_train, y_train)
        test_df['prob_Logistic'] = lr.predict_proba(X_test)[:, 1]
        test_df['prob_Logistic'] = test_df.groupby('race_id')['prob_Logistic'].transform(lambda x: x / x.sum())
        
        # 3. LightGBM
        lgb_model = lgb.LGBMClassifier(n_estimators=50, max_depth=3, verbose=-1, random_state=2026)
        lgb_model.fit(X_train, y_train)
        raw_lgb_prob = lgb_model.predict_proba(X_test)[:, 1]
        test_df['prob_LightGBM'] = raw_lgb_prob
        test_df['prob_LightGBM'] = test_df.groupby('race_id')['prob_LightGBM'].transform(lambda x: x / x.sum())
        
        # 4. Probability Calibration (Isotonic Regression)
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(lgb_model.predict_proba(X_train)[:, 1], y_train)
        calibrated_prob = iso.transform(raw_lgb_prob)
        test_df['prob_LGBM_Calibrated'] = calibrated_prob
        test_df['prob_LGBM_Calibrated'] = test_df.groupby('race_id')['prob_LGBM_Calibrated'].transform(lambda x: x / x.sum())
        
        prediction_records.append(test_df)
        
    full_test = pd.concat(prediction_records, ignore_index=True)
    return full_test

# ==========================================
# STEP 7 ~ STEP 10. 지표 산출, EV & Dynamic Betting 분석
# ==========================================
def evaluate_quant_performance(df):
    models = ['Baseline', 'Logistic', 'LightGBM', 'LGBM_Calibrated']
    
    print("\n" + "="*80)
    print("📊 1. 모델별 정밀 예측 및 확률 보정(Calibration) 평가지표")
    print("="*80)
    
    eval_metrics = []
    for m in models:
        prob_col = f'prob_{m}'
        # Top 1 예측 추출
        top1_df = df.sort_values(prob_col, ascending=False).groupby('race_id').first()
        acc_top1 = (top1_df['finish'] == 1).mean() * 100
        
        # Top 3 / Top 5
        df['rank_pred'] = df.groupby('race_id')[prob_col].rank(ascending=False, method='first')
        acc_top3 = df[df['rank_pred'] <= 3]['finish'].le(3).mean() * 100
        
        loss = log_loss(df['target'], df[prob_col])
        brier = brier_score_loss(df['target'], df[prob_col])
        auc = roc_auc_score(df['target'], df[prob_col])
        
        eval_metrics.append({
            'Model': m,
            '1착 적중률': f"{acc_top1:.2f}%",
            'Top3 적중률': f"{acc_top3:.2f}%",
            'Log Loss': f"{loss:.4f}",
            'Brier Score': f"{brier:.4f}",
            'ROC-AUC': f"{auc:.4f}"
        })
    print(pd.DataFrame(eval_metrics).to_string(index=False))
    
    # 2. EV 조건 및 Dynamic Betting 백테스트 (LGBM_Calibrated 기준)
    print("\n" + "="*80)
    print("💰 2. EV 임계값 및 Dynamic Betting 백테스트 (Calibrated LightGBM 기준)")
    print("="*80)
    
    df['EV'] = (df['prob_LGBM_Calibrated'] * df['winOdds']) - 1.0
    
    # 불확실성 산출 (Race Entropy & Top1-Top2 Gap)
    def calc_race_confidence(group):
        probs = np.sort(group['prob_LGBM_Calibrated'])[::-1]
        gap = probs[0] - probs[1] if len(probs) > 1 else probs[0]
        group['confidence_gap'] = gap
        return group
        
    df = df.groupby('race_id', group_keys=False).apply(calc_race_confidence)
    
    ev_thresholds = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]
    backtest_results = []
    
    for ev_th in ev_thresholds:
        # 각 경주 AI 승률 1위 선택
        top1_races = df.sort_values('prob_LGBM_Calibrated', ascending=False).groupby('race_id').first()
        
        # Confidence 판단 (Gap 0.1 이상만 PASS 없이 진행)
        valid_bets = top1_races[(top1_races['EV'] >= ev_th) & (top1_races['confidence_gap'] >= 0.08)].copy()
        
        if valid_bets.empty:
            continue
            
        # Flat Betting vs Dynamic Betting 비교
        # Dynamic: EV와 Confidence_gap에 비례하여 기본 1만원~최대 5만원 차등 베팅
        valid_bets['bet_flat'] = 10000
        valid_bets['bet_dynamic'] = np.clip(10000 * (1 + valid_bets['EV'] * 2 + valid_bets['confidence_gap'] * 2), 10000, 50000)
        
        # 수익 계산 (Dynamic 기준)
        valid_bets['is_win'] = (valid_bets['finish'] == 1)
        valid_bets['payout'] = np.where(valid_bets['is_win'], valid_bets['bet_dynamic'] * valid_bets['winOdds'], 0)
        valid_bets['profit'] = valid_bets['payout'] - valid_bets['bet_dynamic']
        
        total_bets = len(valid_bets)
        total_stake = valid_bets['bet_dynamic'].sum()
        total_profit = valid_bets['profit'].sum()
        roi = (total_profit / total_stake) * 100 if total_stake > 0 else 0
        hit_rate = valid_bets['is_win'].mean() * 100
        
        # MDD 연산
        valid_bets['cum_profit'] = valid_bets['profit'].cumsum()
        cum_max = valid_bets['cum_profit'].cummax()
        drawdown = valid_bets['cum_profit'] - cum_max
        mdd = drawdown.min()
        
        backtest_results.append({
            'EV 기준': f">={int(ev_th*100)}%",
            '베팅 수': f"{total_bets:,}회",
            '적중률': f"{hit_rate:.2f}%",
            '평균베팅금': f"{int(valid_bets['bet_dynamic'].mean()):,}원",
            '총 투자금': f"{int(total_stake):,}원",
            '순수익': f"{int(total_profit):,}원",
            'ROI': f"{roi:+.2f}%",
            'MDD': f"{int(mdd):,}원"
        })
        
    print(pd.DataFrame(backtest_results).to_string(index=False))

# 파이프라인 전체 메인 실행
if __name__ == "__main__":
    df_raw = generate_timeseries_mock_data(n_races=1500)
    df_features = build_leakage_free_features(df_raw)
    df_predictions = run_walk_forward_evaluation(df_features)
    evaluate_quant_performance(df_predictions)
