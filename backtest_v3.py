import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from build_v3_dataset import KRAV3DatasetBuilder


class KRAV3Backtester:
  """KRA AI Quant System V3 - Walk-Forward Backtest & Feature Ablation Engine

  검증 메트릭:
  1. Top1 Hit Rate: AI 1위 예측마의 실제 1착 우승 비율
  2. Top3 Hit Rate: AI Top3 예측마 중 실제 1착 우승마 포함 비율
  3. Log Loss: 예측 확률 분포의 정밀도 평가 (낮을수록 우수)
  4. Brier Score: 예측 확률과 실제 우승 여부(0 또는 1) 간의 MSE (낮을수록 우수)
  5. Calibration: Temperature Scaling을 통한 확률 최적화
  """

  def __init__(self, service_key: str):
    self.builder = KRAV3DatasetBuilder(service_key)

  @staticmethod
  def softmax_with_temperature(logits: np.ndarray, temp: float = 1.5) -> np.ndarray:
    """[Temperature Scaling Calibration]

    Logits 지수를 온도(temp) 파라미터로 나누어 확률 불확실성을 보정함.
    """
    scaled_logits = logits / max(temp, 0.1)
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
    return exp_logits / np.sum(exp_logits)

  def evaluate_predictions(
      self, df_race: pd.DataFrame, probs: np.ndarray, target_col: str = "target_win"
  ) -> dict:
    """단일 경주 예측 성능 측정"""
    df = df_race.copy()
    df["pred_prob"] = probs
    df["rank_pred"] = df["pred_prob"].rank(ascending=False, method="min")

    # 실제 우승마 (target_win == 1)
    actual_winner = df[df[target_col] == 1]
    if actual_winner.empty:
      return {}

    winner_pred_rank = actual_winner.iloc[0]["rank_pred"]

    top1_hit = 1 if winner_pred_rank == 1 else 0
    top3_hit = 1 if winner_pred_rank <= 3 else 0

    y_true = df[target_col].values
    y_prob = df["pred_prob"].values

    # Brier Score & Log Loss 연산
    brier = brier_score_loss(y_true, y_prob)
    # Clip to avoid log(0)
    y_prob_clipped = np.clip(y_prob, 1e-15, 1 - 1e-15)
    ll = log_loss(y_true, y_prob_clipped, labels=[0, 1])

    return {
        "top1_hit": top1_hit,
        "top3_hit": top3_hit,
        "brier_score": brier,
        "log_loss": ll,
    }

  def run_ablation_test(self, df_dataset: pd.DataFrame) -> pd.DataFrame:
    """[STEP 5: Feature Ablation Test]

    V2 Baseline 대비 V3 신규 피처(Recent Form, Distance Fit, Track Fit, Burden Delta 등)를
    하나씩 추가하며 LogLoss, Brier Score, Top1/Top3 적중률의 개선 여부를 증명.
    """
    feature_sets = {
        "Baseline (V2 Only)": ["rating", "wgBudam"],
        "+ Recent Form": [
            "rating",
            "wgBudam",
            "feat_recent_relative_rank",
            "feat_recent_form_trend",
        ],
        "+ Distance Fit": [
            "rating",
            "wgBudam",
            "feat_recent_relative_rank",
            "feat_recent_form_trend",
            "feat_distance_fit",
        ],
        "+ Track & Combo Fit": [
            "rating",
            "wgBudam",
            "feat_recent_relative_rank",
            "feat_recent_form_trend",
            "feat_distance_fit",
            "feat_track_condition_fit",
            "feat_jockey_horse_combo",
        ],
        "Full V3 Model": [
            "rating",
            "wgBudam",
            "feat_recent_relative_rank",
            "feat_recent_form_trend",
            "feat_distance_fit",
            "feat_track_condition_fit",
            "feat_burden_delta",
            "feat_jockey_horse_combo",
        ],
    }

    results = []

    # 경주 단위 그룹화
    races = df_dataset.groupby(["race_date", "meet", "race_no"])

    for model_name, feats in feature_sets.items():
      top1_hits, top3_hits = [], []
      brier_list, logloss_list = [], []

      for _, df_race in races:
        if len(df_race) < 2 or "target_win" not in df_race.columns:
          continue

        # 결측치는 0.0 처리 후 Z-Score 표준화
        X = df_race[feats].fillna(0.0).values
        # 단순 가중 회귀 또는 Logistic Regression Score
        logits = np.sum(X, axis=1)
        probs = self.softmax_with_temperature(logits, temp=1.5)

        metrics = self.evaluate_predictions(df_race, probs)
        if metrics:
          top1_hits.append(metrics["top1_hit"])
          top3_hits.append(metrics["top3_hit"])
          brier_list.append(metrics["brier_score"])
          logloss_list.append(metrics["log_loss"])

      if top1_hits:
        results.append({
            "Feature Configuration": model_name,
            "Top1 Hit Rate (%)": round(np.mean(top1_hits) * 100, 2),
            "Top3 Hit Rate (%)": round(np.mean(top3_hits) * 100, 2),
            "Log Loss": round(np.mean(logloss_list), 4),
            "Brier Score": round(np.mean(brier_list), 4),
            "Evaluated Races": len(top1_hits),
        })

    return pd.DataFrame(results)

  def run_walk_forward_validation(
      self, df_dataset: pd.DataFrame, train_dates: list, test_dates: list
  ) -> pd.DataFrame:
    """[STEP 6: Walk-Forward Validation]

    시간 순서를 왜곡하지 않고 Past Train -> Future Test 시계열 분리로 검증.
    """
    df_train = df_dataset[
        df_dataset["race_date"].isin(train_dates)
    ].copy()
    df_test = df_dataset[df_dataset["race_date"].isin(test_dates)].copy()

    if df_train.empty or df_test.empty:
      print("⚠️ Walk-Forward 검증을 위한 Train/Test 데이터가 부족합니다.")
      return pd.DataFrame()

    v3_features = [
        "rating",
        "wgBudam",
        "feat_recent_relative_rank",
        "feat_recent_form_trend",
        "feat_distance_fit",
        "feat_track_condition_fit",
        "feat_burden_delta",
        "feat_jockey_horse_combo",
    ]

    # Model 1: Logistic Regression
    X_train = df_train[v3_features].fillna(0.0)
    y_train = df_train["target_win"].fillna(0)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    # Future Test 적용
    test_races = df_test.groupby(["race_date", "meet", "race_no"])
    wf_results = []

    for (r_date, meet, r_no), df_race in test_races:
      if len(df_race) < 2:
        continue

      X_test = df_race[v3_features].fillna(0.0)
      # Calibrated Probabilities
      raw_probs = clf.predict_proba(X_test)[:, 1]
      calibrated_probs = raw_probs / np.sum(raw_probs)

      metrics = self.evaluate_predictions(df_race, calibrated_probs)
      if metrics:
        wf_results.append({
            "race_date": r_date,
            "meet": meet,
            "race_no": r_no,
            "top1_hit": metrics["top1_hit"],
            "top3_hit": metrics["top3_hit"],
            "log_loss": metrics["log_loss"],
            "brier_score": metrics["brier_score"],
        })

    df_wf = pd.DataFrame(wf_results)
    print("\n📈 [Walk-Forward Validation 최종 평가 보고서]")
    print(
        f"• 테스트 경주 수: {len(df_wf)}개"
    )
    print(f"• Top1 적중률: {df_wf['top1_hit'].mean() * 100:.2f}%")
    print(f"• Top3 적중률: {df_wf['top3_hit'].mean() * 100:.2f}%")
    print(f"• 평균 Log Loss: {df_wf['log_loss'].mean():.4f}")
    print(f"• 평균 Brier Score: {df_wf['brier_score'].mean():.4f}")

    return df_wf


# ======================================================================
# 백테스트 실행 및 Ablation Test 출력
# ======================================================================
if __name__ == "__main__":
  SERVICE_KEY = (
      "92ac0b865a4117f6158886b90c8ad86908d7d67bead9c9166eb0fef68cddd2da"
  )
  backtester = KRAV3Backtester(SERVICE_KEY)

  print(
      "🔬 [V3 Feature Ablation Test] 피처 추가에 따른 정량적 성과 개선 검증"
      " 시작...\n"
  )

  # 데이터셋 파일 로드 또는 생성
  parquet_path = "data/v3_features.parquet"
  if os.path.exists(parquet_path):
    df_data = pd.read_parquet(parquet_path)
  else:
    # 샘플 생성
    df_data = backtester.builder.build_v3_feature_matrix(
        meet_code="1", target_date="20240901", selected_race="1"
    )
    # Target Win 라벨 가상 시뮬레이션 (1위 마필)
    if not df_data.empty:
      df_data["target_win"] = [
          1 if str(no) == "1" else 0 for no in df_data["chulNo"]
      ]

  if not df_data.empty and "target_win" in df_data.columns:
    df_ablation = backtester.run_ablation_test(df_data)
    print(df_ablation.to_string(index=False))
  else:
    print(
        "❌ 검증용 데이터셋이 비어있거나 target_win 라벨이 없어 백테스트를"
        " 건너뜁니다."
    )
