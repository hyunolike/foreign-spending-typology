"""등록외국인 수로 설명되지 않는 외국인 소비 = 방문 수요 추정.

외국인 소비에는 거주 외국인의 생활 소비와 방문객 소비가 섞여 있다.
등록외국인 수(정주 규모)와 내국인 소비(상권 규모)를 통제한 뒤 남는 초과분을
**방문수요지수(VDI, Visit Demand Index)**로 본다.

    log(외국인소비) = a + b1·log(등록외국인수) + b2·log(내국인소비) + e
    VDI = e (회귀 잔차)

내국인 소비를 반드시 통제해야 한다. 빼면 잔차가 '관광'이 아니라 '상권 규모'를
잡아서 안산 단원·시흥 같은 산업단지가 상위로 올라온다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BENCH_LABEL = "제주(벤치마크)"


def _design(d: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [np.ones(len(d)), np.log(d["등록외국인수"]), np.log(d["내국인금액"])]
    )


def fit_vdi(table: pd.DataFrame):
    """벤치마크(제주)를 뺀 지역으로 적합하고, 제주에도 같은 계수를 적용한다."""
    t = table.copy()
    t["내국인금액"] = t["전체금액"] - t["외국인금액"]
    t["인당소비"] = t["외국인금액"] / t["등록외국인수"]

    train = t[t["유형"] != BENCH_LABEL]
    X = _design(train)
    y = np.log(train["외국인금액"]).values
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)

    pred_all = _design(t) @ beta
    t["VDI"] = np.log(t["외국인금액"]).values - pred_all

    resid_train = y - X @ beta
    r2 = 1 - (resid_train**2).sum() / ((y - y.mean()) ** 2).sum()
    stats = {
        "beta": beta.tolist(),
        "r2": float(r2),
        "resid_std": float(resid_train.std(ddof=3)),
        "n": int(len(train)),
    }
    return t, stats


def correlation_by_type(table: pd.DataFrame) -> pd.DataFrame:
    """유형별로 '외국인 소비가 등록외국인 수로 얼마나 설명되는지'."""
    rows = []
    for ty, g in table[table["유형"] != BENCH_LABEL].groupby("유형"):
        r = np.corrcoef(np.log(g["등록외국인수"]), np.log(g["외국인금액"]))[0, 1]
        rows.append(
            {
                "유형": ty,
                "시군구수": len(g),
                "상관계수": r,
                "설명력_R2": r**2,
                "인당소비_중앙값": g["인당소비"].median(),
                "VDI_평균": g["VDI"].mean(),
                "VDI_표준편차": g["VDI"].std(),
            }
        )
    return pd.DataFrame(rows).set_index("유형").sort_values("상관계수", ascending=False)
