import pandas as pd
import streamlit as st


def attach_rank_medals(rank_series):
  """순위(1, 2, 3 등)에 맞춰 정확히 🥇🥈🥉 이모지를 붙여주는 함수"""
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
  """지표 값(승률, 레이팅 등)이 높은 순서대로 1, 2, 3위에게 메달을 붙여주는 함수"""
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


st.set_page_config(
    page_title="KRA AI Quant Betting Console V2", page_icon="🏇", layout="wide"
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


st.title("🏇 KRA AI 승률 예측 & 퀀트 포트폴리오 콘솔 V2")
st.caption("Feature Engine V2: Bayesian Smoothing | Relative Z-Score | Shannon Entropy Difficulty")

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

if st.button("🚀 V2 AI 퀀트 분석 및 시각화 리포트 생성"):
  with st.spinner("KRA 8개 API 다단계 수신 및 V2 퀀트 분석 중..."):
    try:
      from quant_pipeline_test import KRAFeatureEngineV2

      engine = KRAFeatureEngineV2(get_service_key())
      sorted_df, summary, err = engine.build_feature_matrix(
          meet_choice, target_date, str(selected_race)
      )
    except Exception as ex:
      sorted_df, summary, err = None, None, f"엔진 실행 예외: {str(ex)}"

  if err or not isinstance(summary, dict) or sorted_df is None:
    st.error(f"파이프라인 처리 오류: {err or '데이터 수신 실패'}")
  else:
    status_color = "#22c55e" if summary.get("bet_recommend", False) else "#ef4444"
    st.markdown(
        f"""
        <div style="background-color:#1e293b; padding:18px; border-radius:12px; color:white; margin-bottom:15px;">
            <h3 style="margin:0;">경주 난이도 판독: <span style="color:{status_color};">{summary.get('difficulty_grade', 'N/A')}</span></h3>
            <p style="margin-top:8px; color:#94a3b8; font-size:0.95rem;">
                🥇 Top1 확률: <b>{summary.get('top1_prob', 0)}%</b> | 
                Top1-2 승률 격차: <b>{summary.get('gap_p', 0)}%p</b> | 
                불확실성(Entropy): <b>{summary.get('normalized_entropy', 0)}</b> (1.0에 가까울수록 초혼전) | 
                💧 주로 함수율: <b>{summary.get('water_percent', 0)}%</b>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not summary.get("bet_recommend", False):
      st.warning(
          "🔒 [PASS 권장] 경주 난이도가 너무 높거나(C급 초혼전) AI 승률 상위마 간"
          " 격차가 적어 위험도가 높습니다."
      )

    top1 = sorted_df.iloc[0]
    top2 = sorted_df.iloc[1] if len(sorted_df) > 1 else top1
    top3 = sorted_df.iloc[2] if len(sorted_df) > 2 else top2

    st.subheader("👑 AI 순수 예측 TOP 3")
    c1, c2, c3 = st.columns(3)
    with c1:
      st.markdown(
          f"""
            <div class="top-card card-1">
                <div class="card-title">🥇 1위 예측</div>
                <div class="card-horse">{top1.get('chulNo')}번 {top1.get('hrName')}</div>
                <div class="card-sub">기수: {top1.get('jkName')} | 부담중량: {top1.get('budam_num', '-')}kg</div>
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
                <div class="card-sub">기수: {top2.get('jkName')} | 부담중량: {top2.get('budam_num', '-')}kg</div>
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
                <div class="card-sub">기수: {top3.get('jkName')} | 부담중량: {top3.get('budam_num', '-')}kg</div>
                <div class="card-win">AI 승률: {top3.get('AI_승률(%)')}%</div>
            </div>
            """,
          unsafe_allow_html=True,
      )

    st.write("")

    st.subheader("📊 출전마 AI 순수 승률 & V2 퀀트 분석표")

    sorted_df["smoothed_win_pct"] = (
        sorted_df["feat_hr_smoothed_win"] * 100
    ).round(1)
    sorted_df["jk_score_pct"] = (sorted_df["feat_jk_score"] * 100).round(1)

    # 💡 [핵심 수정]: 순위와 수치 지표 메달 부착 정정
    ai_rank_medals = attach_rank_medals(sorted_df["AI_예측순위"])
    hr_win_medals = [
        f"{val}%" for val in attach_val_medals(sorted_df, "smoothed_win_pct")
    ]
    jk_win_medals = [
        f"{val}%" for val in attach_val_medals(sorted_df, "jk_score_pct")
    ]
    rating_medals = attach_val_medals(sorted_df, "rating_num")

    display_df = pd.DataFrame({
        "AI 순위": ai_rank_medals,
        "게이트": sorted_df["chulNo"],
        "마명": sorted_df["hrName"],
        "기수": sorted_df["jkName"],
        "부담중량": sorted_df["budam_num"],
        "레이팅": rating_medals,
        "평활화 마필승률": hr_win_medals,
        "기수 1년성적": jk_win_medals,
        "부담 상대Z": sorted_df["feat_budam_z"].apply(
            lambda x: f"{x:+.2f}σ"
        ),
        "체중 감점": sorted_df["feat_weight_penalty"].apply(
            lambda x: f"{x:.2f}"
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
