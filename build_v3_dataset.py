import math
import os
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import requests


class KRAV3DatasetBuilder:
  """KRA AI Quant System V3 - Data Collection & Feature Matrix Generator"""

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
        "race_result": "https://apis.data.go.kr/B551015/API156/raceRsutDtl",
    }

  def fetch_raw_api(self, url: str, params: dict) -> pd.DataFrame:
    full_params = {
        "ServiceKey": self.service_key,
        "serviceKey": self.service_key,
        "pageNo": "1",
        "numOfRows": "200",
        "_type": "xml",
        **params,
    }
    try:
      res = requests.get(url, params=full_params, timeout=5)
      if res.status_code != 200:
        return pd.DataFrame()

      root = ET.fromstring(res.content)
      res_code = root.findtext(".//resultCode")
      if res_code in ["00", "0", "NORMAL_SERVICE", "OK"]:
        items = root.findall(".//item")
        if items:
          return pd.DataFrame([
              {child.tag: child.text for child in item} for item in items
          ])
      return pd.DataFrame()
    except Exception:
      return pd.DataFrame()

  @staticmethod
  def bayesian_smoothed_rate(
      success_cnt: float,
      total_cnt: float,
      global_mean: float = 0.10,
      m: float = 5.0,
  ) -> float:
    if pd.isna(total_cnt) or total_cnt <= 0:
      return global_mean
    return (success_cnt + m * global_mean) / (total_cnt + m)

  @staticmethod
  def compute_relative_rank(rk: float, dusu: float) -> float:
    if pd.isna(rk) or pd.isna(dusu) or dusu <= 1:
      return np.nan
    return 1.0 - ((rk - 1.0) / (dusu - 1.0))

  def get_horse_past_history(
      self, hr_no: str, hr_name: str, target_date: str
  ) -> pd.DataFrame:
    """마필 고유번호(hrNo) 또는 마필명(hrName)으로 과거 경주 파싱"""
    df_hist = pd.DataFrame()

    # 1. hrNo로 우선 조회
    if hr_no and str(hr_no).strip() != "":
      df_hist = self.fetch_raw_api(
          self.endpoints["race_result"], {"hrNo": str(hr_no).strip()}
      )
      if df_hist.empty:
        df_hist = self.fetch_raw_api(
            self.endpoints["race_result"], {"pthrHrno": str(hr_no).strip()}
        )

    # 2. hrNo 실패 시 마명(hrName)으로 fallback 조회
    if df_hist.empty and hr_name and str(hr_name).strip() != "":
      df_hist = self.fetch_raw_api(
          self.endpoints["race_result"], {"hrName": str(hr_name).strip()}
      )
      if df_hist.empty:
        df_hist = self.fetch_raw_api(
            self.endpoints["race_result"], {"pthrHrnm": str(hr_name).strip()}
        )

    if df_hist.empty:
      return pd.DataFrame()

    date_col = (
        "schdRaceDt"
        if "schdRaceDt" in df_hist.columns
        else ("rcDate" if "rcDate" in df_hist.columns else None)
    )
    if not date_col:
      return pd.DataFrame()

    df_hist[date_col] = df_hist[date_col].astype(str)
    df_past = df_hist[df_hist[date_col] < str(target_date)].copy()
    if df_past.empty:
      return pd.DataFrame()

    rk_col = "rsutRk" if "rsutRk" in df_past.columns else "ord"
    dusu_col = "pthrGtno" if "pthrGtno" in df_past.columns else "dusu"
    dist_col = "cndRaceDs" if "cndRaceDs" in df_past.columns else "rcDist"
    weight_col = (
        "pthrBurdWgt" if "pthrBurdWgt" in df_past.columns else "wgBudam"
    )

    df_past["rk"] = pd.to_numeric(df_past.get(rk_col), errors="coerce")
    df_past["dusu"] = pd.to_numeric(
        df_past.get(dusu_col, 8), errors="coerce"
    ).fillna(8)
    df_past["dist"] = pd.to_numeric(
        df_past.get(dist_col, 1200), errors="coerce"
    ).fillna(1200)
    df_past["weight"] = pd.to_numeric(df_past.get(weight_col), errors="coerce")
    df_past["track_cond"] = df_past.get("rsutTrckStus", "양호")
    df_past["jk_no"] = df_past.get("hrmJckyId", df_past.get("jkNo", ""))

    df_past["rel_rank"] = df_past.apply(
        lambda r: self.compute_relative_rank(r["rk"], r["dusu"]), axis=1
    )
    df_past = df_past.sort_values(by=date_col, ascending=False)
    return df_past

  def extract_v3_features_for_horse(
      self,
      hr_no: str,
      hr_name: str,
      jk_no: str,
      target_date: str,
      target_dist: float,
      target_water: float,
      current_weight: float,
  ) -> dict:
    res = {
        "feat_recent_relative_rank": np.nan,
        "feat_recent_form_trend": 0.00,
        "feat_distance_fit": np.nan,
        "feat_track_condition_fit": np.nan,
        "feat_burden_delta": 0.0,
        "feat_jockey_horse_combo": np.nan,
        "feat_rest_days": np.nan,
    }

    df_past = self.get_horse_past_history(hr_no, hr_name, target_date)
    if df_past.empty:
      return res

    recent_3 = df_past.head(3).dropna(subset=["rel_rank"])
    if not recent_3.empty:
      res["feat_recent_relative_rank"] = float(recent_3["rel_rank"].mean())

    if len(recent_3) >= 2:
      ranks_time_order = recent_3["rel_rank"].values[::-1]
      x = np.arange(len(ranks_time_order))
      slope = float(np.polyfit(x, ranks_time_order, 1)[0])
      res["feat_recent_form_trend"] = round(slope, 3)

    dist_mask = (df_past["dist"] >= target_dist - 100) & (
        df_past["dist"] <= target_dist + 100
    )
    df_dist = df_past[dist_mask]
    if not df_dist.empty:
      dist_top3_cnt = (df_dist["rk"] <= 3).sum()
      res["feat_distance_fit"] = self.bayesian_smoothed_rate(
          dist_top3_cnt, len(df_dist), global_mean=0.20, m=3.0
      )

    if target_water >= 10.0:
      wet_mask = df_past["track_cond"].isin(["다습", "포장", "불량"])
      df_wet = df_past[wet_mask]
      if not df_wet.empty:
        wet_top3_cnt = (df_wet["rk"] <= 3).sum()
        res["feat_track_condition_fit"] = self.bayesian_smoothed_rate(
            wet_top3_cnt, len(df_wet), global_mean=0.20, m=3.0
        )

    past_mean_weight = df_past["weight"].dropna().mean()
    if not pd.isna(past_mean_weight):
      res["feat_burden_delta"] = float(current_weight - past_mean_weight)

    if jk_no:
      df_combo = df_past[df_past["jk_no"] == jk_no]
      if not df_combo.empty:
        combo_top3_cnt = (df_combo["rk"] <= 3).sum()
        res["feat_jockey_horse_combo"] = self.bayesian_smoothed_rate(
            combo_top3_cnt, len(df_combo), global_mean=0.15, m=3.0
        )

    last_rc_date = df_past.iloc[0].get(
        "schdRaceDt", df_past.iloc[0].get("rcDate")
    )
    if last_rc_date:
      try:
        d_target = pd.to_datetime(target_date)
        d_last = pd.to_datetime(last_rc_date)
        res["feat_rest_days"] = float((d_target - d_last).days)
      except Exception:
        pass

    return res

  def build_v3_feature_matrix(
      self, meet_code: str, target_date: str, selected_race: str
  ) -> pd.DataFrame:
    df_entry = self.fetch_raw_api(
        self.endpoints["entry_sheet"],
        {"meet": meet_code, "rc_date": target_date},
    )
    if df_entry.empty:
      return pd.DataFrame()

    if "rcNo" in df_entry.columns:
      df_race = df_entry[
          df_entry["rcNo"].astype(str) == str(selected_race)
      ].copy()
    else:
      df_race = df_entry.copy()

    if df_race.empty:
      return pd.DataFrame()

    df_track = self.fetch_raw_api(
        self.endpoints["track_info"],
        {
            "meet": meet_code,
            "rc_date_fr": target_date,
            "rc_date_to": target_date,
        },
    )
    water_percent = 4.0
    if not df_track.empty and "waterPercent" in df_track.columns:
      try:
        water_percent = float(df_track.iloc[0].get("waterPercent", 4.0))
      except ValueError:
        water_percent = 4.0

    target_dist = (
        pd.to_numeric(df_race.iloc[0].get("rcDist", 1200), errors="coerce")
        or 1200
    )

    rows = []
    for _, row in df_race.iterrows():
      # hrNo 파싱 다중 검사
      hr_no = str(
          row.get("hrNo", row.get("pthrHrno", row.get("hr_no", "")))
      ).strip()
      hr_name = str(row.get("hrName", row.get("hr_name", ""))).strip()
      jk_no = str(row.get("jkNo", row.get("jk_no", ""))).strip()

      budam = (
          pd.to_numeric(
              row.get("wgBudam", row.get("handyCap", 55)), errors="coerce"
          )
          or 55.0
      )

      v3_feats = self.extract_v3_features_for_horse(
          hr_no, hr_name, jk_no, target_date, target_dist, water_percent, budam
      )

      combined_row = {
          "race_date": target_date,
          "meet": meet_code,
          "race_no": selected_race,
          "chulNo": row.get("chulNo"),
          "hrName": row.get("hrName"),
          "jkName": row.get("jkName"),
          "rating": pd.to_numeric(row.get("rating", 0), errors="coerce") or 0.0,
          "wgBudam": budam,
          **v3_feats,
      }
      rows.append(combined_row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
  SERVICE_KEY = (
      "92ac0b865a4117f6158886b90c8ad86908d7d67bead9c9166eb0fef68cddd2da"
  )
  builder = KRAV3DatasetBuilder(SERVICE_KEY)
  df_v3_matrix = builder.build_v3_feature_matrix(
      meet_code="1", target_date="20240901", selected_race="1"
  )
  if not df_v3_matrix.empty:
    os.makedirs("data", exist_ok=True)
    df_v3_matrix.to_parquet("data/v3_features.parquet", index=False)
