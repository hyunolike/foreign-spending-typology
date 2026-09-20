"""계절성 보정 상대성장률.

5월에 전체 소비가 튀는 등 월별 변동이 있어 외국인 증감만으로는 해석이 어렵다.
같은 지역의 내국인 증감으로 나눠 지역 공통 요인을 상쇄한다.

    상대성장률 = (외국인 Q2/Q1) / (내국인 Q2/Q1) - 1
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import Q1_MONTHS, Q2_MONTHS

EPS = 1.0  # 0 나눗셈 방지 (원 단위라 영향 없음)


def quarter_amt(df: pd.DataFrame, foreign: bool) -> pd.DataFrame:
    sub = df[df["IS_FOREIGN"] == foreign]
    q1 = sub[sub["STRD_YYMM"].isin(Q1_MONTHS)].groupby("REGION")["AMT"].sum()
    q2 = sub[sub["STRD_YYMM"].isin(Q2_MONTHS)].groupby("REGION")["AMT"].sum()
    return pd.DataFrame({"Q1": q1, "Q2": q2}).fillna(0.0)


def relative_growth(df: pd.DataFrame) -> pd.DataFrame:
    f = quarter_amt(df, True).add_prefix("외국인_")
    d = quarter_amt(df, False).add_prefix("내국인_")
    out = f.join(d, how="outer").fillna(0.0)
    out["외국인성장률"] = out["외국인_Q2"] / (out["외국인_Q1"] + EPS) - 1
    out["내국인성장률"] = out["내국인_Q2"] / (out["내국인_Q1"] + EPS) - 1
    out["상대성장률"] = (out["외국인_Q2"] / (out["외국인_Q1"] + EPS)) / (
        out["내국인_Q2"] / (out["내국인_Q1"] + EPS)
    ) - 1
    return out.replace([np.inf, -np.inf], np.nan)


def monthly_index(df: pd.DataFrame, type_map: pd.Series) -> pd.DataFrame:
    """유형별 월별 상대지수 = (유형 외국인 소비 / 유형 내국인 소비)를 1월=100으로 지수화."""
    d = df.assign(TYPE=df["REGION"].map(type_map)).dropna(subset=["TYPE"])
    piv = d.pivot_table(
        index=["TYPE", "STRD_YYMM"], columns="IS_FOREIGN", values="AMT", aggfunc="sum"
    ).rename(columns={True: "외국인", False: "내국인"})
    ratio = (piv["외국인"] / piv["내국인"]).unstack(0)
    return ratio.div(ratio.iloc[0], axis=1) * 100
