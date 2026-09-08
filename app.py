import streamlit as st
# 동일한 디렉토리에 위치한 파이프라인 Class 로드
from quant_pipeline_test import KRADataPipeline
# ==========================================
# 0. UI 기본 레이아웃 및 CSS 설정
# ==========================================
st.set_page_config(
    page_title="KRA AI 개인용 경마 매매 시스템",
    page_icon="🏇",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main { padding: 0.8rem; }
    .stButton>button { width: 100%; border-radius: 10px; font-weight: bold; background-color: #0284c7; color: white; height: 3.2em; }
    .metric-card { background-color: #1e293b; padding: 20px; border-radius: 12px; color: white; margin-bottom: 15px; }
    .badge-s { background-color: #15803d; color: white; padding: 4px 8px; border-radius: 6px; font-weight: bold; }
    .badge-a { background-color: #0369a1; color: white; padding: 4px 8px; border-radius: 6px; font-weight: bold; }
    .badge-b { background-color: #b45309; color: white; padding: 4px 8px; border-radius: 6px; font-weight: bold; }
    .badge-pass { background-color: #b91c1c; color: white; padding: 4px 8px; border-radius: 6px; font-weight: bold; }
    .error-box { background-color: #450a0a; border: 1px solid #f87171; padding: 12px; border-radius: 8px; color: #fca5a5; }
    </style>
""",
    unsafe_allow_html=True,
)


def get_service_key():
  """Secrets 또는 예비 키 수급"""
  try:
    return st.secrets["KRA_SERVICE_KEY"]
  except Exception:
    return st.session_state.get(
        "custom_service_key",
        "92ac0b865a4117f6158886b90c8ad86908d7d67bead9c9166eb0fef68cddd2da",
    )


# ==========================================
# 1. 메인 대시보드 UI
# ==========================================
st.title("🏇 KRA AI 개인용 경마 매매 시스템")
st.caption("실시간 마사회 API 정제 파이프라인 및 퀀트 의사결정 콘솔")

# 일일 누적 사용금액 세션 관리
if "daily_spent" not in st.session_state:
  st.session_state["daily_spent"] = 0

# 사이드바 예산 관리
with st.sidebar:
  st.header("💰 일일 자금 관리 현황")
  st.metric("하루 한도 (DAILY_LIMIT)", "30,000원")
  st.metric("오늘 누적 사용액", f"{st.session_state['daily_spent']:,}원")
  st.metric("남은 베팅 가능 예산", f"{30000 - st.session_state['daily_spent']:,}원")
  if st.button("🔄 리셋 (새로운 날 시작)"):
    st.session_state["daily_spent"] = 0
    st.rerun()

col1, col2 = st.columns(2)
with col1:
  target_date = st.text_input("📅 경기 날짜 (YYYYMMDD)", value="20240901")
  meet_choice = st.selectbox(
      "🏟️ 경마장",
      options=["1", "2", "3"],
      format_func=lambda x: {"1": "서울", "2": "제주", "3": "부산경남"}[x],
  )

with col2:
  selected_race = st.number_input(
      "🏁 경주 번호", min_value=1, max_value=15, value=1
  )

if st.button("🚀 실시간 KRA 데이터 연동 및 AI 분석"):
  with st.spinner("마사회 서버 통신 및 정제 파이프라인 가동 중..."):
    # 파이프라인 인스턴스 생성 및 일괄 연산 실행
    pipeline = KRADataPipeline(get_service_key())
    ui_df, err, summary = pipeline.process_pipeline(
        meet_choice, target_date, selected_race, st.session_state["daily_spent"]
    )

    if err:
      st.markdown(
          f'<div class="error-box"><h4>❌ 파이프라인 처리 오류</h4><pre>{err}</pre></div>',
          unsafe_allow_html=True,
      )
    else:
      st.success("🟢 KRA 실시간 출전표 수신 및 퀀트 정제 완료!")

      # 상단 퀀트 의사결정 요약 카드
      st.markdown(
          f"""
        <div class="metric-card">
            <h3>의사결정: <span class="{summary['badge_cls']}">{summary['grade']}</span></h3>
            <p><b>실제 추천 투입금:</b> <span style="color:#f59e0b; font-size:1.3em; font-weight:bold;">{summary['actual_stake']:,}원</span> (남은 한도: {summary['rem_budget']:,}원)</p>
            <p>💧 주로 함수율: {summary['water_pct']}% | 🥇 1위 추천마: <b>{summary['top1_no']}번 ({summary['top1_name']})</b> | Top1-2 승률 격차: {summary['gap']:.1f}%p</p>
        </div>
        """,
          unsafe_allow_html=True,
      )

      # 실제 매매 버튼 클릭 시 일일 한도 자동 차감
      if summary["action"] == "BET" and summary["actual_stake"] > 0:
        if st.button(
            f"💵 {summary['actual_stake']:,}원 매매 실행 (일일 한도 차감)"
        ):
          st.session_state["daily_spent"] += summary["actual_stake"]
          st.success(
              f"{summary['actual_stake']:,}원 투입 완료! (오늘 총 사용:"
              f" {st.session_state['daily_spent']:,}원)"
          )
          st.rerun()

      # 하단 핵심 정제표
      st.subheader(f"📊 {selected_race}경주 핵심 AI 퀀트 정제표")
      st.dataframe(ui_df, use_container_width=True, hide_index=True)
