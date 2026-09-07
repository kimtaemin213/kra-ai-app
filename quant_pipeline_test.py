import warnings
import datetime
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ==========================================
# 8. 수학적으로 완벽히 공정한 Synthetic Market 생성기
# ==========================================
def generate_fair_synthetic_races(n_races=3000, margin=0.27):
    np.random.seed(2026)
    start_date = datetime.date(2024, 1, 1)
    takeout_rate = 1.0 - margin  # 정확히 0.73 (환급률 73%)

    races = []
    current_date = start_date

    for r_idx in range(1, n_races + 1):
        if r_idx % 8 == 0:
            current_date += datetime.timedelta(days=1)

        n_horses = np.random.choice([8, 10, 12, 14], p=[0.2, 0.4, 0.3, 0.1])
        
        # Dirichlet 분포로 합계가 정확히 1.0이 되는 승률 생성 (스케일 왜곡 방지)
        true_prob = np.random.dirichlet(np.ones(n_horses) * 2.0)

        # 승자 결정
        winner_idx = np.random.choice(n_horses, p=true_prob)
        finish_rank = np.ones(n_horses, dtype=int) * 2
        finish_rank[winner_idx] = 1

        # 이론적 공정 배당 (Clip 없이 정직하게 적용)
        win_odds = takeout_rate / true_prob

        for h_idx in range(n_horses):
            races.append({
                "race_id": f"R{r_idx:05d}",
                "rc_date": current_date,
                "chulNo": h_idx + 1,
                "hrName": f"마필_{h_idx+1:02d}",
                "true_prob": true_prob[h_idx],
                "winOdds": win_odds[h_idx],
                "finish": finish_rank[h_idx]
            })

    df = pd.DataFrame(races)
    df["rc_date"] = pd.to_datetime(df["rc_date"])
    return df

# ==========================================
# SANITY CHECK EXECUTION ENGINE
# ==========================================
def run_all_sanity_checks():
    print("=" * 90)
    print("🧪 퀀트 시뮬레이터 공정성 & 회계 정산 SANITY CHECK (10,000회 검증)")
    print("=" * 90)

    df = generate_fair_synthetic_races(n_races=3000)
    STAKE = 10000

    # TEST 1: Random Betting Monte Carlo
    print(f"\n[TEST 1] Random 선택 Monte Carlo Simulation (1,000회 반복)...")
    random_rois = []
    for _ in range(1000):
        random_picks = df.groupby("race_id").sample(n=1)
        tot_stake = len(random_picks) * STAKE
        hits = random_picks[random_picks["finish"] == 1]
        payout = (hits["winOdds"] * STAKE).sum()
        roi = ((payout - tot_stake) / tot_stake) * 100
        random_rois.append(roi)

    random_rois = np.array(random_rois)
    avg_roi = np.mean(random_rois)
    print(f"  • 평균 ROI : {avg_roi:+.2f}% (기대값: -27.00%)")

    # TEST 4: Finish Order Permutation (가중 무작위 셔플)
    print(f"\n[TEST 4] Finish Order Permutation (가중 무작위 셔플) 검사...")
    permuted_df = df.copy()
    
    def weighted_shuffle_race(group):
        group['finish'] = 2
        probs = group['true_prob'].values
        winner_pos = np.random.choice(len(group), p=probs)
        group.iloc[winner_pos, group.columns.get_loc('finish')] = 1
        return group

    permuted_df = permuted_df.groupby('race_id', group_keys=False).apply(weighted_shuffle_race)
    
    perm_picks = permuted_df.groupby("race_id").sample(n=1)
    tot_stk = len(perm_picks) * STAKE
    hits = perm_picks[perm_picks["finish"] == 1]
    pay = (hits["winOdds"] * STAKE).sum()
    perm_roi = ((pay - tot_stk) / tot_stk) * 100
    
    print(f"  • Permutation 후 Random ROI: {perm_roi:+.2f}%")
    if -32.0 <= perm_roi <= -22.0:
        print("  🟢 Pass: 수식 완전 정상화! 결과 셔플 시 -27% 부근 손실이 정확히 산출됩니다.")
    else:
        print(f"  🔴 Fail: 결과 수치({perm_roi:.2f}%) 재검증 필요.")

if __name__ == "__main__":
    run_all_sanity_checks()
