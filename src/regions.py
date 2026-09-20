"""외부 데이터 결합을 위한 행정구역 이름 정규화.

제공 데이터에는 화성시가 2026년 신규 행정구역(동탄구/만세구/병점구/효행구)으로
들어와 있어, 통계청·행안부 등 외부 통계의 '화성시'와 그대로 결합되지 않는다.
외부 데이터를 붙일 때는 아래 매핑으로 시군구 단위를 맞춘 뒤 합산한다.
"""
from __future__ import annotations

import pandas as pd

# 제공 데이터의 신설 구 -> 외부 통계의 모(母) 시군구
SUBDIVIDED_TO_PARENT = {
    "경기도 동탄구": "경기도 화성시",
    "경기도 만세구": "경기도 화성시",
    "경기도 병점구": "경기도 화성시",
    "경기도 효행구": "경기도 화성시",
}

# 시도명 표기 차이 (외부 통계는 약칭을 쓰는 경우가 많음)
SIDO_ALIASES = {
    "서울": "서울특별시",
    "부산": "부산광역시",
    "대구": "대구광역시",
    "인천": "인천광역시",
    "광주": "광주광역시",
    "대전": "대전광역시",
    "울산": "울산광역시",
    "세종": "세종특별자치시",
    "세종시": "세종특별자치시",
    "경기": "경기도",
    "강원": "강원특별자치도",
    "강원도": "강원특별자치도",
    "충북": "충청북도",
    "충남": "충청남도",
    "전북": "전북특별자치도",
    "전라북도": "전북특별자치도",
    "전남": "전라남도",
    "경북": "경상북도",
    "경남": "경상남도",
    "제주": "제주특별자치도",
    "제주도": "제주특별자치도",
}


def normalize_sido(name: str) -> str:
    name = str(name).strip()
    return SIDO_ALIASES.get(name, name)


def make_key(sido: str, ccg: str) -> str:
    return f"{normalize_sido(sido)} {str(ccg).strip()}"


def to_parent_region(region: str) -> str:
    """화성시 신설 구 등 세분화된 코드를 외부 통계 기준 시군구로 되돌린다."""
    return SUBDIVIDED_TO_PARENT.get(region, region)


def add_join_key(df: pd.DataFrame, region_col: str = "REGION") -> pd.DataFrame:
    out = df.copy()
    out["JOIN_REGION"] = out[region_col].map(to_parent_region)
    return out
