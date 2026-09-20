"""시군구별 외국인 소비 프로파일 피처 생성."""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    AGE_SENIOR,
    AGE_YOUNG,
    BENCHMARK_KEYS,
    BUZ_GROUPS,
    FEATURE_COLS,
    MIN_FOREIGN_AMT,
)

BUZ_TO_GROUP = {b: g for g, items in BUZ_GROUPS.items() for b in items}


def region_totals(df: pd.DataFrame) -> pd.DataFrame:
    """지역별 전체/외국인 소비 합계와 외국인 비중."""
    g = df.groupby("REGION").agg(
        전체금액=("AMT", "sum"),
        전체건수=("CNT", "sum"),
    )
    f = (
        df[df["IS_FOREIGN"]]
        .groupby("REGION")
        .agg(외국인금액=("AMT", "sum"), 외국인건수=("CNT", "sum"))
    )
    out = g.join(f, how="left").fillna({"외국인금액": 0, "외국인건수": 0})
    out["외국인비중"] = out["외국인금액"] / out["전체금액"]
    out["건단가"] = np.where(
        out["외국인건수"] > 0, out["외국인금액"] / out["외국인건수"], np.nan
    )
    return out


def _share_matrix(fdf: pd.DataFrame, index: str, col: str) -> pd.DataFrame:
    piv = fdf.pivot_table(
        index="REGION", columns=col, values="AMT", aggfunc="sum", fill_value=0
    )
    return piv.div(piv.sum(axis=1), axis=0)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """외국인 소비 기준 업종/연령 구성비 + 외국인비중 + log건단가."""
    fdf = df[df["IS_FOREIGN"]].copy()
    fdf["BUZ_GROUP"] = fdf["TP_BUZ_NM"].map(BUZ_TO_GROUP)

    buz_share = _share_matrix(fdf, "REGION", "BUZ_GROUP")
    buz_share = buz_share.reindex(columns=list(BUZ_GROUPS), fill_value=0.0)
    buz_share.columns = [f"{c}비중" for c in buz_share.columns]

    age_share = _share_matrix(fdf, "REGION", "AGE_CD")
    young = [c for c in AGE_YOUNG if c in age_share.columns]
    senior = [c for c in AGE_SENIOR if c in age_share.columns]
    age_feat = pd.DataFrame(
        {
            "청년비중": age_share[young].sum(axis=1),
            "중장년비중": age_share[senior].sum(axis=1),
        }
    )

    totals = region_totals(df)
    feat = (
        totals.join(buz_share, how="left")
        .join(age_feat, how="left")
        .assign(log건단가=lambda d: np.log(d["건단가"]))
    )
    return feat


def split_targets(feat: pd.DataFrame):
    """분석 대상(외국인 10억 이상) / 학습 대상(제주 제외) / 벤치마크로 분리."""
    target = feat[feat["외국인금액"] >= MIN_FOREIGN_AMT].copy()
    bench = target[target.index.isin(BENCHMARK_KEYS)].copy()
    train = target[~target.index.isin(BENCHMARK_KEYS)].copy()
    missing = train[FEATURE_COLS].isna().any(axis=1)
    if missing.any():
        print(f"[warn] 피처 결측으로 제외: {train.index[missing].tolist()}")
        train = train[~missing]
    return target, train, bench
