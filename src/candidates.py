"""잠재 상권 후보 도출.

두 가지 버전이 있다.

- `build_candidates`      제공 데이터만 사용 (유사도 + 성장률 + 청년비중)
- `build_candidates_external`  법무부 등록외국인 결합 후 (VDI + 성장률)

두 결과를 나란히 두면 외부 데이터가 무엇을 바로잡았는지가 드러나므로,
내부 전용 버전도 남겨 둔다.
"""
from __future__ import annotations

import numpy as np
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


# --- 외부 데이터(법무부 등록외국인) 결합 후 ------------------------------------
# 외부 데이터를 붙이자 내부 전용 점수의 전제 두 개가 모두 틀렸음이 드러났다.
#
# 1) 청년비중은 방문 수요가 아니라 '유학생 밀집'을 뽑는다. 기존 상위 후보의
#    1인당 외국인 소비는 35~53만원으로 도심 방문형 중앙값(69만원)에 못 미친다.
# 2) 도심 방문형 중심과의 유사도는 방향이 반대다. 75분위 컷오프가 잘라낸 쪽에
#    동성로(대구 중구)·광안리(부산 수영구)·이태원(용산)·홍대(마포)가 몰려 있다.
#    중심에 가까운 지역이 오히려 평균적인 대학가다.
#
# 그래서 외부 결합 버전은 유사도 컷오프를 걷어내고(유형 소속 판정에만 사용),
# 방문수요지수(VDI)와 성장률만으로 점수를 만든다.
W_VDI = 0.6
W_GROWTH_EXT = 0.4


def build_candidates_external(table: pd.DataFrame):
    """도심 방문형 전체에서 대표 상권만 제외하고 VDI·성장률로 점수화."""
    flagships = flagship_regions(table)
    cand = table[(table["유형"] == TYPE_URBAN) & (~table.index.isin(flagships))].copy()
    cand = cand.dropna(subset=["VDI", "상대성장률"])

    cand["VDI순위"] = cand["VDI"].rank(pct=True)
    cand["성장률순위"] = cand["상대성장률"].rank(pct=True)
    cand["점수_외부"] = W_VDI * cand["VDI순위"] + W_GROWTH_EXT * cand["성장률순위"]
    return cand.sort_values("점수_외부", ascending=False), flagships


def final_top5(cand: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """VDI가 양(+)인 후보, 즉 방문 수요의 근거가 실제로 있는 곳만 추린다.

    성장률만 높고 VDI가 음(-)인 곳(예: 홍성)은 '잠재 성장형'으로 남겨 두고
    최종 추천에서는 뺀다. 방문 수요가 아직 관측되지 않았기 때문이다.
    """
    return cand[cand["VDI"] > 0].head(n)


def label_quadrant(cand: pd.DataFrame) -> pd.Series:
    """VDI·성장률 사분면으로 후보 성격을 구분한다."""
    vdi_pos = cand["VDI"] > 0
    grow_pos = cand["상대성장률"] > 0
    return pd.Series(
        np.select(
            [vdi_pos & grow_pos, vdi_pos & ~grow_pos, ~vdi_pos & grow_pos],
            ["방문 수요 + 성장", "방문 수요 정체", "잠재 성장형"],
            default="관찰 대상",
        ),
        index=cand.index,
    )
