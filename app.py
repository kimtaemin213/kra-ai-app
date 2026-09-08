import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import requests
import streamlit as st

# ==========================================
# 0. UI 레이아웃 및 스타일 설정
# ==========================================
st.set_page_config(
    page_title="KRA AI 개인용 경마 매매 시스템",
    page_icon="🏇",
    layout="centered",
)

st.markdown(
    """
    <style>
    .main { padding: 0.8rem; }
    .stButton>button { width: 100%; border-radius: 10px; font-weight: bold; background-color: #0284c7; color: white; height: 3.2em; }
    .metric-card { background-color: #1e293b; padding: 18px; border-radius: 12px; color: white; margin-bottom: 15px; }
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
  """Streamlit Secrets 키 수급 및 로컬 세션 예비 처리"""
  try:
    return st.secrets["KRA_SERVICE_KEY"]
  except Exception:
    return st.session_state.get(
        "custom_service_key",
        "92ac0b865a4117f6158886b90c8ad86908d7d67bead9c9166eb0fef68cddd2da",
    )


# ==========================================
# 1. 8개 공식 확인된 KRA API 엔드포인트 디렉토리
# ==========================================
ENDPOINTS = {
    "entry_sheet": "https://apis.data.go.kr/B551015/API26_2/entrySheet_2",  # 출전표 정보
    "track_info": "https://apis.data.go.kr/B551015/API189_1/Track_1",  # 경주로/날씨 정보
    "race_result_dtl": (
        "https://apis.data.go.kr/B551015/API156/raceRsutDtl"
    ),  # 경주결과 상세
    "horse_weight": (
        "https://apis.data.go.kr/B551015/API25_1/entryHorseWeightInfo_1"
    ),  # 출전마 체중
    "total_record": (
        "https://apis.data.go.kr/B551015/API27_1/totalRecord_1"
    ),  # 통산경주기록
    "jockey_result": (
        "https://apis.data.go.kr/B551015/jkyresult/getjkyresult"
    ),  # 기수 최근1년 성적
    "highest_dividend": (
        "https://apis.data.go.kr/B551015/API35_1/highestDividendRateInfo_1"
    ),  # 최고배당률
    "jeju_result": (
        "https://apis.data.go.kr/B551015/jejuhorseresult/getjejuhorseresult"
    ),  # 제주경주마 성적
}


# ==========================================
# 2. 실시간 API 수신 및 디버깅 모듈 (에러 원문 노출)
# ==========================================
def fetch_kra_data(url, extra_params):
  key = get_service_key()
  # 이중 인코딩 및 대소문자 이슈 방지를 위해 대소문자 키 모두 세팅
  params = {
      "ServiceKey": key,
      "serviceKey": key,
      "pageNo": "1",
      "numOfRows": "100",
      "_type": "xml",
      **extra_params,
  }

  try:
    res = requests.get(url, params=params, timeout=8)
    if res.status_code != 200:
      return None, f"HTTP {res.status_code} Error:\n{res.text[:500]}"

    root = ET.fromstring(res.content)

    # 게이트웨이 에러 체크
    err_msg = root.findtext(".//errMsg") or root.findtext(".//returnAuthMsg")
    if err_msg:
      return None, f"🔴 게이트웨이 에러: {err_msg}"

    # 비즈니스 결과 코드 체크
    res_code = root.findtext(".//resultCode")
    res_msg = root.findtext(".//resultMsg")

    if res_code in ["00", "0", "NORMAL_SERVICE", "OK"]:
      items = root.findall(".//item")
      if items:
        data = [{child.tag: child.text for child in item} for item in items]
        return pd.DataFrame(data), None
      return pd.DataFrame(), "🟡 조건에 일치하는 데이터가 없습니다."
    return None, f"🔴 서비스 에러 [{res_code}]: {res_msg}"

  except Exception as e:
    return None, f"💥 통신 연결 장애: {str(e)}"


# ==========================================
# 3. AI Quant Decision Engine (DAILY_LIMIT $30,000 차감)
# ==========================================
def run_ai_quant_decision(df_entry, df_track, daily_spent):
  DAILY_LIMIT = 30000
  rem_budget = max(0, DAILY_LIMIT - daily_spent)

  df = df_entry.copy()

  # 수치 전처리
  num_cols = ["winOdds", "jkWinRt", "hrWinRt", "rating", "handyCap"]
  for col in num_cols:
    if col in df.columns:
      df[col] = (
          pd.to_numeric(
              df[col].astype(str).str.replace("%", ""), errors="coerce"
          )
          .fillna(0.0)
      )
    else:
      df[col] = 5.0 if col == "winOdds" else 10.0

  # 주로 상태 (함수율) 보정
  water_percent = 4.0
  if not df_track.empty and "waterPercent" in df_track.columns:
    try:
      water_percent = float(df_track.iloc[0].get("waterPercent", 4.0))
    except ValueError:
      water_percent = 4.0

  # Softmax AI 확률 계산
  humidity_bonus = 0.15 if water_percent >= 10.0 else 0.0
  df["score"] = (
      (df["jkWinRt"] / 25.0) * 1.20
      + (df["hrWinRt"] / 30.0) * 1.30
      + (df["rating"] / 100.0) * 1.05
      - (df["handyCap"] / 60.0) * 0.50
      + humidity_bonus
  )

  exp_s = np.exp(df["score"] - df["score"].max())
  df["AI_승률(%)"] = ((exp_s / exp_s.sum()) * 100).round(1)

  # 시장 임플라이드 확률 정규화
  df["market_raw"] = 1.0 / np.maximum(df["winOdds"], 1.05)
  df["시장_승률(%)"] = ((df["market_raw"] / df["market_raw"].sum()) * 100).round(
      1
  )

  df["AI_EDGE(%p)"] = (df["AI_승률(%)"] - df["시장_승률(%)"]).round(1)
  df["EV_기대값"] = ((df["AI_승률(%)"] / 100.0) * df["winOdds"]) - 1.0
  df["AI_예측순위"] = (
      df["AI_승률(%)"].rank(ascending=False, method="min").astype(int)
  )

  sorted_df = df.sort_values(by="AI_예측순위").reset_index(drop=True)
  top1 = sorted_df.iloc[0]
  top2 = sorted_df.iloc[1] if len(sorted_df) > 1 else top1
  gap = top1["AI_승률(%)"] - top2["AI_승률(%)"]
  ev = top1["EV_기대값"]

  # 1차 퀀트 등급 분류
  if ev >= 0.40 and gap >= 10.0:
    raw_grade, target_stake = "🔥 S급", 5000
  elif ev >= 0.20 and gap >= 6.0:
    raw_grade, target_stake = "🔷 A급", 3000
  elif ev >= 0.08 and gap >= 4.0:
    raw_grade, target_stake = "📙 B급", 2000
  else:
    raw_grade, target_stake = "🔴 C/D급", 0

  # 2차 리스크 필터링 (잔여 예산 한도 차감 검사)
  if target_stake == 0:
    final_grade, badge_cls, actual_stake, action = (
        "🔴 PASS (조건 미달)",
        "badge-pass",
        0,
        "PASS",
    )
  elif rem_budget < target_stake:
    final_grade, badge_cls, actual_stake, action = (
        (
            f"🔒 PASS (예산 부족: 남은 예산 {rem_budget:,}원 < 필요금액"
            f" {target_stake:,}원)"
        ),
        "badge-pass",
        0,
        "PASS",
    )
  else:
    final_grade = f"{raw_grade} (추천)"
    badge_cls = (
        "badge-s"
        if "S급" in raw_grade
        else ("badge-a" if "A급" in raw_grade else "badge-b")
    )
    actual_stake, action = target_stake, "BET"

  return (
      sorted_df,
      water_percent,
      final_grade,
      badge_cls,
      actual_stake,
      action,
      gap,
      rem_budget,
  )


# ==========================================
# 4. Streamlit 메인 대시보드 UI
# ==========================================
st.title("🏇 KRA AI 개인용 경마 매매 시스템")
st.caption("실시간 마사회 공공데이터 API 연동 및 퀀트 의사결정 콘솔")

# 오늘 누적 사용 금액 세션 관리
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
  with st.spinner("마사회 서버와 실시간 데이터 수신 중..."):
    # 1. 출전표 (API26_2)
    df_entry, err_entry = fetch_kra_data(
        ENDPOINTS["entry_sheet"], {"meet": meet_choice, "rc_date": target_date}
    )
    # 2. 경주로/주로 정보 (API189_1)
    df_track, _ = fetch_kra_data(
        ENDPOINTS["track_info"],
        {
            "meet": meet_choice,
            "rc_date_fr": target_date,
            "rc_date_to": target_date,
        },
    )

    if err_entry:
      st.markdown(
          f'<div class="error-box"><h4>❌ 출전표 API 수신 에러</h4><pre>{err_entry}</pre></div>',
          unsafe_allow_html=True,
      )
    elif df_entry.empty:
      st.warning("해당 조건의 출전표 데이터가 존재하지 않습니다.")
    else:
      st.success("🟢 KRA 실시간 출전표 데이터 수신 성공!")

      # AI 의사결정 및 자금 차감 검사
      (
          res_df,
          water_pct,
          grade,
          badge_cls,
          actual_stake,
          action,
          gap,
          rem_budget,
      ) = run_ai_quant_decision(
          df_entry, df_track, st.session_state["daily_spent"]
      )
      top1 = res_df.iloc[0]

      st.markdown(
          f"""
        <div class="metric-card">
            <h3>의사결정: <span class="{badge_cls}">{grade}</span></h3>
            <p><b>실제 추천 투입금:</b> <span style="color:#f59e0b; font-size:1.3em; font-weight:bold;">{actual_stake:,}원</span> (남은 한도: {rem_budget:,}원)</p>
            <p>💧 함수율: {water_pct}% | Top1-2 승률 격차: {gap:.1f}%p | AI 승률: {top1['AI_승률(%)']}% | EV 기대값: {top1['EV_기대값']*100:+.1f}%</p>
        </div>
        """,
          unsafe_allow_html=True,
      )

      # 매매 버튼 클릭 시 예산 실시간 차감
      if action == "BET" and actual_stake > 0:
        if st.button(f"💵 {actual_stake:,}원 매매 실행 (일일 한도 차감)"):
          st.session_state["daily_spent"] += actual_stake
          st.success(
              f"{actual_stake:,}원 투입 완료! (오늘 총 사용:"
              f" {st.session_state['daily_spent']:,}원)"
          )
          st.rerun()

      st.subheader("📊 KRA 실시간 출전마 AI 퀀트 분석표")
      st.dataframe(res_df, use_container_width=True)
