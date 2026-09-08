def get_horse_past_history(
        self, hr_no: str, target_date: str
    ) -> pd.DataFrame:
      """Target Race 이전(rcDate < target_date)의 해당 마필 경주 결과 파싱 (파라미터 방어 적용)"""
      if not hr_no:
        return pd.DataFrame()

      # API156 조회 파라미터 키 방어 (hrNo / pthrHrno 동시 대응)
      df_hist = self.fetch_raw_api(
          self.endpoints["race_result"], {"hrNo": hr_no}
      )
      if df_hist.empty:
        df_hist = self.fetch_raw_api(
            self.endpoints["race_result"], {"pthrHrno": hr_no}
        )

      if df_hist.empty:
        return pd.DataFrame()

      # 날짜 컬럼 자동 매핑 (schdRaceDt 또는 rcDate)
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

      # 수치 필드 변환
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
        jk_no: str,
        target_date: str,
        target_dist: float,
        target_water: float,
        current_weight: float,
    ) -> dict:
      """마필별 고유 피처 독립 연산 (Loop 간 데이터 섞임 방지)"""
      # 기본 독립 딕셔너리 생성
      res = {
          "feat_recent_relative_rank": np.nan,
          "feat_recent_form_trend": 0.00,
          "feat_distance_fit": np.nan,
          "feat_track_condition_fit": np.nan,
          "feat_burden_delta": 0.0,
          "feat_jockey_horse_combo": np.nan,
          "feat_rest_days": np.nan,
      }

      df_past = self.get_horse_past_history(hr_no, target_date)
      if df_past.empty:
        return res

      # 1. 최근 3경주 상대 순위
      recent_3 = df_past.head(3).dropna(subset=["rel_rank"])
      if not recent_3.empty:
        res["feat_recent_relative_rank"] = float(recent_3["rel_rank"].mean())

      # 2. 최근 폼 트렌드 (최근 3경기 기울기)
      if len(recent_3) >= 2:
        ranks_time_order = recent_3["rel_rank"].values[::-1]
        x = np.arange(len(ranks_time_order))
        slope = float(np.polyfit(x, ranks_time_order, 1)[0])
        res["feat_recent_form_trend"] = round(slope, 3)

      # 3. 거리 적합도 (Target 거리 ±100m)
      dist_mask = (df_past["dist"] >= target_dist - 100) & (
          df_past["dist"] <= target_dist + 100
      )
      df_dist = df_past[dist_mask]
      if not df_dist.empty:
        dist_top3_cnt = (df_dist["rk"] <= 3).sum()
        res["feat_distance_fit"] = self.bayesian_smoothed_rate(
            dist_top3_cnt, len(df_dist), global_mean=0.20, m=3.0
        )

      # 4. 기수 x 마필 궁합
      if jk_no:
        df_combo = df_past[df_past["jk_no"] == jk_no]
        if not df_combo.empty:
          combo_top3_cnt = (df_combo["rk"] <= 3).sum()
          res["feat_jockey_horse_combo"] = self.bayesian_smoothed_rate(
              combo_top3_cnt, len(df_combo), global_mean=0.15, m=3.0
          )

      return res
