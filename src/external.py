"""외부 데이터 로더: 법무부 월별 등록외국인 시군구별 거주 현황.

원천(공공데이터포털 15100022)에 다음 이슈가 있어 그대로 결합하면 지역이 누락된다.

1. 시도 표기 체계가 월마다 바뀐다.
   2026년 1~5월은 구 명칭(강원도·전라북도·경기), 6월부터 신 명칭
   (강원특별자치도·전북특별자치도). 7월에는 전남·광주가 '전남광주통합특별시'로
   합쳐진다(본 분석은 1~6월만 사용하므로 해당 없음).
2. 목포시·여수시가 1~5월에 '전라북도'로 잘못 분류되어 있다(실제 전라남도).
   6월에는 전라남도로 바로잡혀 있다.
3. 화성시가 1월에는 통합('화성시'), 2월부터 4개 구로 분리 집계된다.
   1월 55,155명 ≒ 2월 4개 구 합계 55,159명으로 동일 모집단임이 확인된다.

1·2는 시군구명이 전국에서 유일한 경우 BC 데이터의 시도로 자동 교정해 해결하고,
3은 화성 4개 구에 한해 2~6월 평균을 쓴다(월 변동 1% 미만이라 영향이 없다).
"""
from __future__ import annotations

import pandas as pd

from config import DATA_DIR
from regions import normalize_sido

MOJ_CSV = DATA_DIR / "external" / "moj_registered_foreigners.csv"
MOJ_ENCODING = "cp949"

YEAR = 2026
MONTHS = range(1, 7)

# 법무부가 2월부터 분리 집계해 1월 값이 없는 지역
LATE_SPLIT_REGIONS = (
    "경기도 화성시 동탄구",
    "경기도 화성시 만세구",
    "경기도 화성시 병점구",
    "경기도 화성시 효행구",
)


def load_moj(path=MOJ_CSV) -> pd.DataFrame:
    d = pd.read_csv(path, encoding=MOJ_ENCODING)
    d = d[(d["년"] == YEAR) & (d["월"].isin(MONTHS))].copy()
    # '경기도　'처럼 전각 공백이 섞여 있어 strip이 필요하다
    d["시도"] = d["시도"].str.strip().map(normalize_sido)
    d["시군구"] = d["시군구"].str.strip()
    d["REGION"] = d["시도"] + " " + d["시군구"]
    return d[["월", "시도", "시군구", "REGION", "등록외국인수"]]


def fix_sido_by_name(d: pd.DataFrame, bc_regions: pd.Index, verbose: bool = True):
    """시도가 어긋난 행을, 시군구명이 BC 데이터에서 유일할 때만 교정한다.

    동명 시군구(중구·동구 등)는 교정하지 않고 그대로 두어 오매칭을 막는다.
    """
    bc = pd.DataFrame({"REGION": bc_regions})
    bc["시군구"] = bc["REGION"].str.split(" ", n=1).str[1]
    counts = bc["시군구"].value_counts()
    unique_map = bc[bc["시군구"].map(counts) == 1].set_index("시군구")["REGION"]

    out = d.copy()
    unmatched = ~out["REGION"].isin(bc_regions)
    fixable = unmatched & out["시군구"].isin(unique_map.index)
    fixed = out.loc[fixable, ["REGION", "시군구"]].drop_duplicates()
    out.loc[fixable, "REGION"] = out.loc[fixable, "시군구"].map(unique_map)

    if verbose and len(fixed):
        pairs = sorted({(r, unique_map[c]) for r, c in fixed.itertuples(index=False)})
        for before, after in pairs:
            print(f"    [시도 교정] {before} -> {after}")
    return out, len(fixed)


def registered_foreigners(bc_regions: pd.Index, verbose: bool = True) -> pd.Series:
    """BC 지역 키에 맞춘 2026.1~6 월평균 등록외국인 수."""
    d = load_moj()
    d, n_fixed = fix_sido_by_name(d, bc_regions, verbose=verbose)

    monthly = d.groupby(["REGION", "월"])["등록외국인수"].sum().unstack("월")
    # 화성 4개 구는 1월 값이 없으므로 관측된 달(2~6월)의 평균을 쓴다
    avg = monthly.mean(axis=1, skipna=True)

    if verbose:
        late = [r for r in LATE_SPLIT_REGIONS if r in monthly.index]
        if late:
            print(f"    [부분결측] {len(late)}곳은 2~6월 평균 사용 (법무부 1월 통합집계)")
        missing = [r for r in bc_regions if r not in avg.index]
        print(f"    매칭 {len(bc_regions) - len(missing)}/{len(bc_regions)}곳"
              f"{', 미매칭 ' + str(missing) if missing else ''}")
    return avg.rename("등록외국인수")
