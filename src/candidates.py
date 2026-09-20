"""잠재 상권 후보 도출.

도심 방문형 중심에 가까우면서(유사도), 외국인 소비가 상대적으로 빠르게 늘고(성장),
청년 비중이 높은(체류형 수요) 지역을 점수화한다.
"""
from __future__ import annotations

import pandas as pd

from config import TYPE_URBAN

W_SIMILARITY = 0.4
W_GROWTH = 0.4
W_YOUNG = 0.2
FLAGSHIP_TOP_N = 5  # 도심 방문형 내 외국인 소비 상위 = 이미 알려진 대표 상권


def flagship_regions(table: pd.DataFrame, n: int = FLAGSHIP_TOP_N) -> list[str]:
    urban = table[table["유형"] == TYPE_URBAN]
    return urban.nlargest(n, "외국인금액").index.tolist()


def build_candidates(table: pd.DataFrame, urban_cluster: int) -> pd.DataFrame:
    """도심 방문형 중심과의 거리가 해당 유형 내 75분위 이하인 지역을 후보로."""
    dist_col = f"거리_{urban_cluster}"
    urban_dist = table.loc[table["유형"] == TYPE_URBAN, dist_col]
    cutoff = urban_dist.quantile(0.75)

    flagships = flagship_regions(table)
    cand = table[(table[dist_col] <= cutoff) & (~table.index.isin(flagships))].copy()
    cand = cand.dropna(subset=["상대성장률"])

    cand["유사도순위"] = (-cand[dist_col]).rank(pct=True)
    cand["성장률순위"] = cand["상대성장률"].rank(pct=True)
    cand["청년순위"] = cand["청년비중"].rank(pct=True)
    cand["점수"] = (
        W_SIMILARITY * cand["유사도순위"]
        + W_GROWTH * cand["성장률순위"]
        + W_YOUNG * cand["청년순위"]
    )
    return cand.sort_values("점수", ascending=False), cutoff, flagships
