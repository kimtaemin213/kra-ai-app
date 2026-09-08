import numpy as np
import pandas as pd
import streamlit as st
from build_v3_dataset import KRAV3DatasetBuilder

# ==========================================
# 1. 페이지 설정 및 커스텀 CSS
# ==========================================
st.set_page_config(
    page_title="KRA AI Quant System V3", page_icon="🏇", layout="wide"
)

st.markdown(
    """
    <style>
    .top-card { padding: 16px; border-radius: 12px; font-family: 'Noto Sans KR', sans-serif; }
    .card-1 { background-color: #FEF9C3; border: 1px solid #FDE047; color: #854D0E; }
    .card-2 { background-color: #EFF6FF; border: 1px solid #BFDBFE; color: #1E40AF; }
    .card-3 { background-color: #FFEDD5; border: 1px solid #FED7AA; color: #9A3412; }
    
    .card-title { font-weight: bold; font-size: 0.9rem; margin-bottom: 6px; }
    .card-horse { font-size: 1.3rem; font-weight: 800; margin-bottom: 4px; }
    .card-sub { font-size: 0.82rem; opacity: 0.85; margin-bottom: 8px; }
    .card-win { font-size: 1.1rem; font-weight: bold; color: #0284C7; }

    .portfolio-box { background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 12px 18px; margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center; }
    .badge-p { padding: 4px 8px; border-radius: 6px; font-weight: bold; font-size: 0.8rem; color: white; margin-right: 8px; }
    .bg-main { background-color: #0369A1; }
    .bg-sub { background-color: #0284C7; }
    .bg-single { background-color: #DB2777; }
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


def attach_rank_medals(rank_series):
  result = []
  for r in rank_series:
    if r == 1:
      result.append(f"🥇 {r}")
    elif r == 2:
      result.append(f"🥈 {r}")
    elif r == 3:
      result.append(f"🥉 {r}")
    else:
      result.append(f"{r}")
  return result


def attach_val_medals(df, val_col):
  ranks = df[val_col].rank(ascending=False, method="min")
  result = []
  for val, rank in zip(df[val_col], ranks):
    if rank == 1:
      result.append(f"🥇 {val}")
    elif rank == 2:
      result.append(f"🥈 {val}")
    elif rank == 3:
      result.append(f"🥉 {val}")
    else:
      result.append(f"{val}")
  return result


# ==========================================
# 2. V3 스코어링 & Softmax 확률 변환 함수
# ==========================================
def process_v3_scoring(df_matrix, temperature=1.5):
  """Rating이 모두 40으로 동일하더라도 V3 Fit/Form 피처로 우위를 가려내는 정밀 연산"""
  df = df_matrix.copy()

  # Z-score 계산 함수
  def z_score(s):
    std = s.std(ddof=0)
    return (s - s.mean()) / max(std, 1e-6) if not pd.isna(std) else s * 0.0

  # 수치형 피처 채우기 및 Z-Score 정규화
  r_z = z_score(df["rating"].fillna(0.0))
  budam_z = -1.0 * z_score(df["wgBudam"].fillna(55.0))  # 가벼울수록 플러스
  rel_rank_z = z_score(df["feat_recent_relative_rank"].fillna(0.5))
  form_trend_z = z_score(df["feat_recent_form_trend"].fillna(0.0))
  dist_fit_z = z_score(df["feat_distance_fit"].fillna(0.2))
  track_fit_z = z_score(df["feat_track_condition_fit"].fillna(0.5))
  combo_z = z_score(df["feat_jockey_horse_combo"].fillna(0.15))
  burden_delta_z = -1.0 * z_score(
      df["feat_burden_delta"].fillna(0.0)
  )  # 중량 증가시 마이너스

  # V3 종합 퀀트 스코어 산출
  df["v3_composite_logits"] = (
      (r_z * 0.5)
      + (budam_z * 0.4)
      + (rel_rank_z * 2.0)
      + (form_trend_z * 1.5)
      + (dist_fit_z * 1.8)
      + (track_fit_z * 1.0)
      + (combo_z * 1.2)
      + (burden_delta_z * 0.8)
  )

  # Temperature Scaling Softmax (Calibration)
  scaled_logits = df["v3_composite_logits"] / max(temperature, 0.1)
  exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
  df["AI_prob"] = exp_logits / np.sum(exp_logits)
  df["AI_승률(%)"] = (df["AI_prob"] * 100).round(1)

  df["AI_예측순위"] = (
      df["AI_prob"].rank(ascending=False, method="min").astype(int)
  )
  sorted_df = df.sort_values(by="AI_예측순위").reset_index(drop=True)

  # 경주 불확실성 (Shannon Entropy) 측정
  probs = sorted_df["AI_prob"].values
  top1_p = probs[0]
  top2_p = probs[1] if len(probs) > 1 else 0.0
  gap_p = top1_p - top2_p

  entropy = -np.sum(probs * np.log2(np.clip(probs, 1e-12, 1.0)))
  max_entropy = np.log2(len(probs))
  norm_entropy = entropy / max_entropy if max_entropy > 0 else 1.0

  if top1_p >= 0.25 and gap_p >= 0.07 and norm_entropy <= 0.85:
    grade, rec = "🟢 A급 (명확한 축마)", True
  elif top1_p >= 0.18 and gap_p >= 0.03:
    grade, rec = "🟡 B급 (중혼전 - 엄격 선택)", True
  else:
    grade, rec = "🔴 C급 (초혼전 / PASS 권장)", False

  summary = {
      "difficulty_grade": grade,
      "bet_recommend": rec,
      "top1_prob": float(round(top1_p * 100, 1)),
      "gap_p": float(round(gap_p * 100, 1)),
      "normalized_entropy": float(round(norm_entropy, 3)),
  }

  return sorted_df, summary


# ==========================================
# 3. Streamlit 대시보드 UI
# ==========================================
st.title("🏇 KRA AI 승률 예측 콘솔 V3")
st.caption(
    "Data-Driven Quantitative Engine V3: Relative Rank | Form Trend | Distance"
    " & Track Fit | Jockey-Horse Combo"
)

if "daily_spent" not in st.session_state:
  st.session_state["daily_spent"] = 0

col_a, col_b, col_c = st.columns([2, 2, 1])
with col_a:
  target_date = st.text_input("📅 경기 날짜 (YYYYMMDD)", value="20240901")
with col_b:
  meet_choice = st.selectbox(
      "🏟️ 경마장",
      options=["1", "2", "3"],
      format_func=lambda x: {"1": "서울", "2": "제주", "3": "부산경남"}[x],
  )
with col_c:
  selected_race = st.number_input(
      "🏁 경주 번호", min_value=1, max_value=15, value=1
  )

if st.button("🚀 V3 AI 퀀트 분석 실행"):
  with st.spinner("KRA 과거 히스토리 수집 및 Cutoff 피처 연산 중..."):
    try:
      builder = KRAV3DatasetBuilder(get_service_key())
      df_matrix = builder.build_v3_feature_matrix(
          meet_choice, target_date, str(selected_race)
      )
      if not df_matrix.empty:
        sorted_df, summary = process_v3_scoring(df_matrix)
        err = None
      else:
        sorted_df, summary, err = None, None, "출전표 데이터를 불러올 수 없습니다."
    except Exception as ex:
      sorted_df, summary, err = None, None, f"엔진 실행 예외: {str(ex)}"

  if err or sorted_df is None:
    st.error(f"파이프라인 처리 오류: {err}")
  else:
    # -------------------------------------------------------------
    # 섹션 1: 경주 난이도 및 불확실성 헤더
    # -------------------------------------------------------------
    status_color = "#22c55e" if summary["bet_recommend"] else "#ef4444"
    st.markdown(
        f"""
        <div style="background-color:#1e293b; padding:18px; border-radius:12px; color:white; margin-bottom:15px;">
            <h3 style="margin:0;">경주 난이도 판독: <span style="color:{status_color};">{summary['difficulty_grade']}</span></h3>
            <p style="margin-top:8px; color:#94a3b8; font-size:0.95rem;">
                🥇 Top1 확률: <b>{summary['top1_prob']}%</b> | 
                Top1-2 승률 격차: <b>{summary['gap_p']}%p</b> | 
                불확실성(Entropy): <b>{summary['normalized_entropy']}</b> (1.0에 가까울수록 초혼전)
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not summary["bet_recommend"]:
      st.warning(
          "🔒 [PASS 권장] 경주 난이도가 너무 높거나(C급 초혼전) AI 승률 상위마 간"
          " 격차가 적어 위험도가 높습니다."
      )

    top1 = sorted_df.iloc[0]
    top2 = sorted_df.iloc[1] if len(sorted_df) > 1 else top1
    top3 = sorted_df.iloc[2] if len(sorted_df) > 2 else top2

    # -------------------------------------------------------------
    # 섹션 2: 👑 AI 순수 예측 TOP 3 카드리포트
    # -------------------------------------------------------------
    st.subheader("👑 AI 순수 예측 TOP 3")
    c1, c2, c3 = st.columns(3)
    with c1:
      st.markdown(
          f"""
            <div class="top-card card-1">
                <div class="card-title">🥇 1위 예측</div>
                <div class="card-horse">{top1.get('chulNo')}번 {top1.get('hrName')}</div>
                <div class="card-sub">기수: {top1.get('jkName')} | Rating: {top1.get('rating', '-')}</div>
                <div class="card-win">AI 승률: {top1.get('AI_승률(%)')}%</div>
            </div>
            """,
          unsafe_allow_html=True,
      )

    with c2:
      st.markdown(
          f"""
            <div class="top-card card-2">
                <div class="card-title">🥈 2위 예측</div>
                <div class="card-horse">{top2.get('chulNo')}번 {top2.get('hrName')}</div>
                <div class="card-sub">기수: {top2.get('jkName')} | Rating: {top2.get('rating', '-')}</div>
                <div class="card-win">AI 승률: {top2.get('AI_승률(%)')}%</div>
            </div>
            """,
          unsafe_allow_html=True,
      )

    with c3:
      st.markdown(
          f"""
            <div class="top-card card-3">
                <div class="card-title">🥉 3위 예측</div>
                <div class="card-horse">{top3.get('chulNo')}번 {top3.get('hrName')}</div>
                <div class="card-sub">기수: {top3.get('jkName')} | Rating: {top3.get('rating', '-')}</div>
                <div class="card-win">AI 승률: {top3.get('AI_승률(%)')}%</div>
            </div>
            """,
          unsafe_allow_html=True,
      )

    st.write("")

    # -------------------------------------------------------------
    # 섹션 3: 📊 출전마 V3 퀀트 분석 메인 데이터프레임
    # -------------------------------------------------------------
    st.subheader("📊 출전마 V3 퀀트 분석표 (Rating 동률 보정)")

    ai_rank_medals = attach_rank_medals(sorted_df["AI_예측순위"])
    rating_medals = attach_val_medals(sorted_df, "rating")

    display_df = pd.DataFrame({
        "AI 순위": ai_rank_medals,
        "게이트": sorted_df["chulNo"],
        "마명": sorted_df["hrName"],
        "기수": sorted_df["jkName"],
        "부담중량": sorted_df["wgBudam"],
        "레이팅": rating_medals,
        "최근 상대순위": sorted_df["feat_recent_relative_rank"].apply(
            lambda x: f"{x*100:.1f}%" if not pd.isna(x) else "N/A"
        ),
        "폼 기울기": sorted_df["feat_form_trend"] if "feat_form_trend" in sorted_df else sorted_df["feat_recent_form_trend"].apply(
            lambda x: f"{x:+.2f}" if not pd.isna(x) else "0.00"
        ),
        "거리 적합도": sorted_df["feat_distance_fit"].apply(
            lambda x: f"{x*100:.1f}%" if not pd.isna(x) else "N/A"
        ),
        "기수x마필 궁합": sorted_df["feat_jockey_horse_combo"].apply(
            lambda x: f"{x*100:.1f}%" if not pd.isna(x) else "N/A"
        ),
        "AI 승률(%)": sorted_df["AI_승률(%)"],
    })

    st.dataframe(
        display_df,
        column_config={
            "AI 승률(%)": st.column_config.ProgressColumn(
                "AI 승률(%)",
                format="%.1f%%",
                min_value=0,
                max_value=float(sorted_df["AI_승률(%)"].max() * 1.2),
            ),
        },
        use_container_width=True,
        hide_index=True,
    )

    st.write("")

    # -------------------------------------------------------------
    # 섹션 4: 🔍 세부 퀀트 피처 펼쳐보기 (설명 가능 모델)
    # -------------------------------------------------------------
    with st.expander("🔍 1위 예측마 정밀 분석 리포트 (왜 1위인가?)"):
      st.markdown(f"### 🥇 **{top1.get('hrName')}** 상세 분석")
      col_x, col_y = st.columns(2)
      with col_x:
        st.write(
            f"• **최근 상대 순위 점수**: "
            f"{top1.get('feat_recent_relative_rank', 0)*100:.1f}% (출전두수 대비"
            " 상위 성적)"
        )
        st.write(
            f"• **최근 폼 기울기**: {top1.get('feat_recent_form_trend', 0):+.2f}"
            " (양수일수록 최근 성적 상승세)"
        )
        st.write(
            f"• **이번 거리 적합도**: {top1.get('feat_distance_fit', 0)*100:.1f}%"
            " (유사 거리 과거 입상률)"
        )
      with col_y:
        st.write(
            f"• **기수×마필 호흡 궁합**: "
            f"{top1.get('feat_jockey_horse_combo', 0)*100:.1f}% (동일 기수 탑승"
            " 입상률)"
        )
        st.write(
            f"• **부담중량 변동폭**: {top1.get('feat_burden_delta', 0):+.1f}kg"
            " (직전 평균 대비)"
        )
        st.write(
            f"• **휴식 일수**: {top1.get('feat_rest_days', 'N/A')}일"
            " (출전 주기)"
        )

    st.write("")

    # -------------------------------------------------------------
    # 섹션 5: 🏇 AI 추천 실전 베팅 포트폴리오
    # -------------------------------------------------------------
    st.subheader("🏇 AI 추천 실전 베팅 포트폴리오")
    rem_budget = max(0, 30000 - st.session_state["daily_spent"])
    total_bet_budget = min(rem_budget, 10000)

    main_stake = int(total_bet_budget * 0.50)
    sub1_stake = int(total_bet_budget * 0.20)
    single_stake = int(total_bet_budget * 0.15)

    st.markdown(
        f"""
        <div class="portfolio-box">
            <div>
                <span class="badge-p bg-main">복승식 (메인)</span> <b>{top1.get('chulNo')}번 ({top1.get('hrName')}) - {top2.get('chulNo')}번 ({top2.get('hrName')})</b>
                <div style="font-size:0.8rem; color:#64748B; margin-top:2px;">AI 승률 1위({top1.get('AI_승률(%)')}%) & 2위({top2.get('AI_승률(%)')}%) 메인 축 조합</div>
            </div>
            <div style="font-weight:bold; font-size:1.05rem; color:#0F172A;">{main_stake:,}원 (50%)</div>
        </div>

        <div class="portfolio-box">
            <div>
                <span class="badge-p bg-sub">삼복승식 (서브)</span> <b>{top1.get('chulNo')} - {top2.get('chulNo')} - {top3.get('chulNo')}번 ({top3.get('hrName')})</b>
                <div style="font-size:0.8rem; color:#64748B; margin-top:2px;">1-2위 고정 후 3착 복병({top3.get('hrName')}) 삼복승 방어</div>
            </div>
            <div style="font-weight:bold; font-size:1.05rem; color:#0F172A;">{sub1_stake:,}원 (20%)</div>
        </div>

        <div class="portfolio-box">
            <div>
                <span class="badge-p bg-single">단승식 (가치베팅)</span> <b>{top1.get('chulNo')}번 ({top1.get('hrName')})</b>
                <div style="font-size:0.8rem; color:#64748B; margin-top:2px;">AI 최상위 1위 단독 베팅</div>
            </div>
            <div style="font-weight:bold; font-size:1.05rem; color:#0F172A;">{single_stake:,}원 (15%)</div>
        </div>
    """,
        unsafe_allow_html=True,
    )
