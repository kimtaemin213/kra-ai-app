import pandas as pd
import streamlit as st
from kra_pipeline_v2 import KRAFeatureEngineV2

st.set_page_config(
    page_title="KRA AI Quant System V2", page_icon="🏇", layout="wide"
)


def get_service_key():
  try:
    return st.secrets["KRA_SERVICE_KEY"]
  except Exception:
    return st.session_state.get(
        "custom_service_key",
        "92ac0b865a4117f6158886b90c8ad86908d7d67bead9c9166eb0fef68cddd2da",
    )


st.title("🏇 KRA AI Quant System V2 (Z-Score & Calibration 적용)")
st.caption("Bayesian Smoothing | Relative Z-Score | Shannon Entropy Difficulty")

col1, col2, col3 = st.columns([2, 2, 1])
with col1:
  target_date = st.text_input("📅 경주 날짜 (YYYYMMDD)", value="20240901")
with col2:
  meet_choice = st.selectbox(
      "🏟️ 경마장",
      options=["1", "2", "3"],
      format_func=lambda x: {"1": "서울", "2": "제주", "3": "부산경남"}[x],
  )
with col3:
  selected_race = st.number_input(
      "🏁 경주 번호", min_value=1, max_value=15, value=1
  )

if st.button("🚀 V2 퀀트 분석 엔진 가동"):
  engine = KRAFeatureEngineV2(get_service_key())
  df_res, summary, err = engine.build_feature_matrix(
      meet_choice, target_date, str(selected_race)
  )

  if err:
    st.error(f"파이프라인 장애: {err}")
  else:
    # 1. 경주 난이도 헤더
    status_color = "green" if summary["bet_recommend"] else "red"
    st.markdown(
        f"""
        <div style="background-color:#1e293b; padding:18px; border-radius:12px; color:white; margin-bottom:15px;">
            <h3 style="margin:0;">분석 결과: <span style="color:{status_color};">{summary['difficulty_grade']}</span></h3>
            <p style="margin-top:8px; color:#94a3b8;">
                🥇 Top1 확률: <b>{summary['top1_prob']}%</b> | 
                Top1-2 격차: <b>{summary['gap_p']}%p</b> | 
                불확실성(Entropy): <b>{summary['normalized_entropy']}</b> (1.0에 가까울수록 초혼전) | 
                주로 함수율: <b>{summary['water_percent']}%</b>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not summary["bet_recommend"]:
      st.warning(
          "🔒 [PASS 권장] 경주 난이도가 너무 높거나(C급 초혼전) AI 승률 상위마 간"
          " 격차가 적어 매매를 건너뜁니다."
      )

    # 2. 메인 퀀트 테이블 표출
    st.subheader("📊 V2 피처 융합 및 AI 승률 데이터프레임")

    display_df = pd.DataFrame({
        "AI 순위": df_res["AI_예측순위"],
        "마번": df_res["chulNo"],
        "마명": df_res["hrName"],
        "기수": df_res["jkName"],
        "평활화 마필승률": df_res["feat_hr_smoothed_win"].apply(
            lambda x: f"{x*100:.1f}%"
        ),
        "기수 1년성적": df_res["feat_jk_score"].apply(
            lambda x: f"{x*100:.1f}%"
        ),
        "부담중량 상대Z": df_res["feat_budam_z"].apply(
            lambda x: f"{x:+.2f}σ"
        ),
        "레이팅 상대Z": df_res["feat_rating_z"].apply(
            lambda x: f"{x:+.2f}σ"
        ),
        "체중 감점": df_res["feat_weight_penalty"].apply(
            lambda x: f"{x:.2f}"
        ),
        "AI 승률(%)": df_res["AI_승률(%)"],
    })

    st.dataframe(
        display_df,
        column_config={
            "AI 승률(%)": st.column_config.ProgressColumn(
                "AI 승률(%)",
                format="%.1f%%",
                min_value=0,
                max_value=float(df_res["AI_승률(%)"].max() * 1.2),
            ),
        },
        use_container_width=True,
        hide_index=True,
    )
