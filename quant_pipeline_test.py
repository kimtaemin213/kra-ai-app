import math
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import requests


class KRAFeatureEngineV2:
  """KRA AI Quant Feature Engineering & Probability Calibration Engine V2"""

  def __init__(self, service_key: str):
    self.service_key = service_key
    self.endpoints = {
        "entry_sheet": "https://apis.data.go.kr/B551015/API26_2/entrySheet_2",
        "track_info": "https://apis.data.go.kr/B551015/API189_1/Track_1",
        "horse_weight": (
            "https://apis.data.go.kr/B551015/API25_1/entryHorseWeightInfo_1"
        ),
        "jockey_result": (
            "https://apis.data.go.kr/B551015/jkyresult/getjkyresult"
        ),
        "total_record": (
            "https://apis.data.go.kr/B551015/API27_1/totalRecord_1"
        ),
        "race_result": "https://apis.data.go.kr/B551015/API156/raceRsutDtl",
    }

  def fetch_raw_api(self, url: str, params: dict) -> tuple:
    full_params = {
        "ServiceKey": self.service_key,
        "serviceKey": self.service_key,
        "pageNo": "1",
        "numOfRows": "100",
        "_type": "xml",
        **params,
    }
    try:
      # timeout을 4초로 단축하여 무한 멈춤 방지
      res = requests.get(url, params=full_params, timeout=4)
      if res.status_code != 200:
        return None, f"HTTP {res.status_code}"

      root = ET.fromstring(res.content)
      err_msg = root.findtext(".//errMsg") or root.findtext(".//returnAuthMsg")
      if err_msg:
        return None, f"API 인증 오류: {err_msg}"

      res_code = root.findtext(".//resultCode")
      if res_code in ["00", "0", "NORMAL_SERVICE", "OK"]:
        items = root.findall(".//item")
        if items:
          return (
              pd.DataFrame([
                  {child.tag: child.text for child in item} for item in items
              ]),
              None,
          )
        return pd.DataFrame(), "데이터 없음"
      return None, f"서비스 에러 [{res_code}]"
    except Exception as e:
      return None, f"통신 장애/초과: {str(e)}"

  @staticmethod
  def bayesian_smoothed_win_rate(
      win_cnt: float, total_cnt: float, field_mean: float, m_confidence: float = 12.0
  ) -> float:
    if total_cnt <= 0:
      return field_mean
    return (win_cnt + m_confidence * field_mean) / (total_cnt + m_confidence)

  @staticmethod
  def compute_z_score(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if pd.isna(std) or std < 1e-6:
      return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std

  @staticmethod
  def gaussian_weight_penalty(diff_weight: float) -> float:
    sigma = 6.0
    penalty = 1.0 - math.exp(-((diff_weight / sigma) ** 2) / 2.0)
    return -float(penalty) * 0.5

  @staticmethod
  def calculate_race_difficulty(probs: np.ndarray) -> tuple:
    if len(probs) == 0:
      return "🔴 데이터 없음", False, 0.0, 0.0, 1.0

    probs = np.clip(probs, 1e-12, 1.0)
    probs = probs / probs.sum()

    sorted_p = np.sort(probs)[::-1]
    top1_p = sorted_p[0]
    top2_p = sorted_p[1] if len(sorted_p) > 1 else 0.0
    gap = top1_p - top2_p

    entropy = -np.sum(probs * np.log2(probs))
    max_entropy = np.log2(len(probs))
    norm_entropy = entropy / max_entropy if max_entropy > 0 else 1.0

    if top1_p >= 0.25 and gap >= 0.07 and norm_entropy <= 0.85:
      difficulty_grade = "🟢 A급 (명확한 축마)"
      bet_recommend = True
    elif top1_p >= 0.18 and gap >= 0.03:
      difficulty_grade = "🟡 B급 (중혼전 - 엄격 선택)"
      bet_recommend = True
    else:
      difficulty_grade = "🔴 C급 (초혼전 / PASS 권장)"
      bet_recommend = False

    return difficulty_grade, bet_recommend, top1_p, gap, norm_entropy

  def build_feature_matrix(
      self,
      meet_code: str,
      target_date: str,
      selected_race: str,
      temperature: float = 1.5,
  ):
    df_entry, err = self.fetch_raw_api(
        self.endpoints["entry_sheet"],
        {"meet": meet_code, "rc_date": target_date},
    )
    if err or df_entry is None or df_entry.empty:
      return None, None, err or "출전표 데이터 없음"

    if "rcNo" in df_entry.columns:
      df = df_entry[df_entry["rcNo"].astype(str) == str(selected_race)].copy()
    else:
      df = df_entry.copy()

    if df.empty:
      return None, None, f"{selected_race}경주 데이터를 찾을 수 없습니다."

    df_track, _ = self.fetch_raw_api(
        self.endpoints["track_info"],
        {
            "meet": meet_code,
            "rc_date_fr": target_date,
            "rc_date_to": target_date,
        },
    )
    df_weight, _ = self.fetch_raw_api(
        self.endpoints["horse_weight"],
        {"meet": meet_code, "rc_date": target_date},
    )
    df_jockey, _ = self.fetch_raw_api(
        self.endpoints["jockey_result"], {"meet": meet_code}
    )

    jockey_win_y_map, jockey_qu_y_map = {}, {}
    if (
        df_jockey is not None
        and not df_jockey.empty
        and "jkName" in df_jockey.columns
    ):
      if "winRateyear" in df_jockey.columns:
        jockey_win_y_map = dict(
            zip(
                df_jockey["jkName"],
                pd.to_numeric(
                    df_jockey["winRateyear"].astype(str).str.replace("%", ""),
                    errors="coerce",
                ).fillna(0.0)
                / 100.0,
            )
        )
      if "quRateyear" in df_jockey.columns:
        jockey_qu_y_map = dict(
            zip(
                df_jockey["jkName"],
                pd.to_numeric(
                    df_jockey["quRateyear"].astype(str).str.replace("%", ""),
                    errors="coerce",
                ).fillna(0.0)
                / 100.0,
            )
        )

    weight_diff_map = {}
    if (
        df_weight is not None
        and not df_weight.empty
        and "chulNo" in df_weight.columns
    ):
      if "wgHrDiff" in df_weight.columns:
        weight_diff_map = dict(
            zip(
                df_weight["chulNo"].astype(str),
                pd.to_numeric(df_weight["wgHrDiff"], errors="coerce").fillna(
                    0.0
                ),
            )
        )

    water_percent = 4.0
    if (
        df_track is not None
        and not df_track.empty
        and "waterPercent" in df_track.columns
    ):
      try:
        water_percent = float(df_track.iloc[0].get("waterPercent", 4.0))
      except ValueError:
        water_percent = 4.0

    df["ord1CntY"] = pd.to_numeric(df.get("ord1CntY", 0), errors="coerce").fillna(
        0.0
    )
    df["rcCntY"] = pd.to_numeric(df.get("rcCntY", 0), errors="coerce").fillna(
        0.0
    )
    df["rating_num"] = pd.to_numeric(df.get("rating", 0), errors="coerce").fillna(
        0.0
    )
    df["budam_num"] = pd.to_numeric(
        df.get("wgBudam", df.get("handyCap", 0)), errors="coerce"
    ).fillna(0.0)

    field_mean_hr_win = (
        df["ord1CntY"].sum() / max(df["rcCntY"].sum(), 1.0)
        if df["rcCntY"].sum() > 0
        else 0.10
    )

    df["feat_hr_smoothed_win"] = [
        self.bayesian_smoothed_win_rate(
            r["ord1CntY"], r["rcCntY"], field_mean_hr_win, m_confidence=12.0
        )
        for _, r in df.iterrows()
    ]
    df["feat_jk_score"] = [
        (jockey_win_y_map.get(str(r.get("jkName", "")), 0.0) * 0.7)
        + (jockey_qu_y_map.get(str(r.get("jkName", "")), 0.0) * 0.3)
        for _, r in df.iterrows()
    ]
    df["feat_budam_z"] = -1.0 * self.compute_z_score(df["budam_num"])
    df["feat_rating_z"] = self.compute_z_score(df["rating_num"])
    df["feat_weight_penalty"] = [
        self.gaussian_weight_penalty(
            weight_diff_map.get(str(r.get("chulNo", "")), 0.0)
        )
        for _, r in df.iterrows()
    ]
    df["feat_track_interaction"] = (
        0.25 * df["feat_rating_z"] if water_percent >= 10.0 else 0.0
    )

    df["composite_logits"] = (
        (df["feat_hr_smoothed_win"] * 2.0)
        + (df["feat_jk_score"] * 1.8)
        + (df["feat_rating_z"] * 0.8)
        + (df["feat_budam_z"] * 0.5)
        + df["feat_weight_penalty"]
        + df["feat_track_interaction"]
    )

    scaled_logits = df["composite_logits"] / temperature
    exp_logits = np.exp(scaled_logits - scaled_logits.max())
    df["AI_prob"] = exp_logits / exp_logits.sum()
    df["AI_승률(%)"] = (df["AI_prob"] * 100).round(1)

    df["AI_예측순위"] = (
        df["AI_prob"].rank(ascending=False, method="min").astype(int)
    )
    sorted_df = df.sort_values(by="AI_예측순위").reset_index(drop=True)

    grade, bet_rec, top1_p, gap, entropy = self.calculate_race_difficulty(
        sorted_df["AI_prob"].values
    )

    metrics_summary = {
        "difficulty_grade": grade,
        "bet_recommend": bool(bet_rec),
        "top1_prob": float(round(top1_p * 100, 1)),
        "gap_p": float(round(gap * 100, 1)),
        "normalized_entropy": float(round(entropy, 3)),
        "water_percent": float(water_percent),
        "total_horses": int(len(sorted_df)),
    }

    return sorted_df, metrics_summary, None
