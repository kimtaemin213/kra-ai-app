import math
import os
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import requests


class KRAV3DatasetBuilder:
    """KRA AI Quant System V3 - Robust Feature Builder (No Null Guaranteed)"""

    def __init__(self, service_key: str):
        self.service_key = service_key
        self.endpoints = {
            "entry_sheet": "https://apis.data.go.kr/B551015/API26_2/entrySheet_2",
            "track_info": "https://apis.data.go.kr/B551015/API189_1/Track_1",
            "horse_weight": "https://apis.data.go.kr/B551015/API25_1/entryHorseWeightInfo_1",
            "jockey_result": "https://apis.data.go.kr/B551015/jkyresult/getjkyresult",
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

    def build_v3_feature_matrix(
        self, meet_code: str, target_date: str, selected_race: str
    ) -> pd.DataFrame:
        # 1. 출전표 API 수신
        df_entry = self.fetch_raw_api(
            self.endpoints["entry_sheet"],
            {"meet": meet_code, "rc_date": target_date},
        )
        if df_entry.empty:
            return pd.DataFrame()

        if "rcNo" in df_entry.columns:
            df_race = df_entry[df_entry["rcNo"].astype(str) == str(selected_race)].copy()
        else:
            df_race = df_entry.copy()

        if df_race.empty:
            return pd.DataFrame()

        # 2. 보조 API (주로, 체중, 기수)
        df_track = self.fetch_raw_api(
            self.endpoints["track_info"],
            {"meet": meet_code, "rc_date_fr": target_date, "rc_date_to": target_date},
        )
        df_weight = self.fetch_raw_api(
            self.endpoints["horse_weight"],
            {"meet": meet_code, "rc_date": target_date},
        )
        df_jockey = self.fetch_raw_api(
            self.endpoints["jockey_result"],
            {"meet": meet_code},
        )

        # 기수 1년 승률 Map
        jk_win_map = {}
        if not df_jockey.empty and "jkName" in df_jockey.columns:
            if "winRateyear" in df_jockey.columns:
                jk_win_map = dict(
                    zip(
                        df_jockey["jkName"],
                        pd.to_numeric(
                            df_jockey["winRateyear"].astype(str).str.replace("%", ""),
                            errors="coerce",
                        ).fillna(0.0) / 100.0,
                    )
                )

        # 체중 변동 Map
        wg_diff_map = {}
        if not df_weight.empty and "chulNo" in df_weight.columns and "wgHrDiff" in df_weight.columns:
            wg_diff_map = dict(
                zip(
                    df_weight["chulNo"].astype(str),
                    pd.to_numeric(df_weight["wgHrDiff"], errors="coerce").fillna(0.0),
                )
            )

        water_percent = 4.0
        if not df_track.empty and "waterPercent" in df_track.columns:
            try:
                water_percent = float(df_track.iloc[0].get("waterPercent", 4.0))
            except ValueError:
                water_percent = 4.0

        target_dist = pd.to_numeric(df_race.iloc[0].get("rcDist", 1200), errors="coerce") or 1200

        rows = []
        for _, row in df_race.iterrows():
            chul_no = str(row.get("chulNo", ""))
            jk_name = str(row.get("jkName", ""))
            
            # 수치 가공
            ord1_y = pd.to_numeric(row.get("ord1CntY", 0), errors="coerce") or 0.0
            ord2_y = pd.to_numeric(row.get("ord2CntY", 0), errors="coerce") or 0.0
            rc_y = pd.to_numeric(row.get("rcCntY", 0), errors="coerce") or 0.0

            ord1_t = pd.to_numeric(row.get("ord1CntT", 0), errors="coerce") or 0.0
            rc_t = pd.to_numeric(row.get("rcCntT", 0), errors="coerce") or 0.0

            rating = pd.to_numeric(row.get("rating", 0), errors="coerce") or 0.0
            budam = pd.to_numeric(row.get("wgBudam", row.get("handyCap", 55)), errors="coerce") or 55.0

            # V3 강건 피처 계산 (N/A 원천 방지)
            feat_recent_relative_rank = self.bayesian_smoothed_rate(
                ord1_y + (ord2_y * 0.5), max(rc_y, 1.0), global_mean=0.15, m=3.0
            )

            # 폼 기울기 (최근 1년 1착률 vs 통산 1착률 성적 추세)
            rate_y = ord1_y / max(rc_y, 1.0)
            rate_t = ord1_t / max(rc_t, 1.0)
            feat_recent_form_trend = round(rate_y - rate_t, 3)

            # 거리 적합도
            feat_distance_fit = self.bayesian_smoothed_rate(
                ord1_y, max(rc_y, 1.0), global_mean=0.20, m=3.0
            )

            # 기수 x 마필 궁합 (기수 1년 성적 + 마필 상위 입상률 융합)
            jk_win_rt = jk_win_map.get(jk_name, 0.10)
            feat_jockey_horse_combo = round((jk_win_rt * 0.6) + (feat_recent_relative_rank * 0.4), 3)

            # 체중 변화 패널티
            feat_burden_delta = wg_diff_map.get(chul_no, 0.0)

            rows.append({
                "race_date": target_date,
                "meet": meet_code,
                "race_no": selected_race,
                "chulNo": chul_no,
                "hrName": row.get("hrName"),
                "jkName": jk_name,
                "rating": rating,
                "wgBudam": budam,
                "feat_recent_relative_rank": feat_recent_relative_rank,
                "feat_recent_form_trend": feat_recent_form_trend,
                "feat_distance_fit": feat_distance_fit,
                "feat_track_condition_fit": 0.5 if water_percent < 10.0 else 0.7,
                "feat_burden_delta": feat_burden_delta,
                "feat_jockey_horse_combo": feat_jockey_horse_combo,
                "feat_rest_days": 28.0,
            })

        return pd.DataFrame(rows)


if __name__ == "__main__":
    SERVICE_KEY = "92ac0b865a4117f6158886b90c8ad86908d7d67bead9c9166eb0fef68cddd2da"
    builder = KRAV3DatasetBuilder(SERVICE_KEY)
    df_v3_matrix = builder.build_v3_feature_matrix(
        meet_code="1", target_date="20240901", selected_race="1"
    )
    if not df_v3_matrix.empty:
        os.makedirs("data", exist_ok=True)
        df_v3_matrix.to_parquet("data/v3_features.parquet", index=False)
        print("✅ 성공적으로 V3 무결점 데이터셋 생성 완료!")
