"""원천 CSV 로드 및 전처리."""
from __future__ import annotations

import pandas as pd

from config import BUZ_MERGE_TO_KOREAN, FOREIGN_CODE, RAW_CSV

DTYPES = {
    "STRD_YYMM": "int32",
    "SIDO_NM": "string",
    "CCG_NM": "string",
    "GENDER_CD": "string",
    "AGE_CD": "string",
    "TP_BUZ_NO": "string",
    "TP_BUZ_NM": "string",
}


def load_raw(path=RAW_CSV) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=DTYPES)
    df = df.rename(columns={"amt": "AMT", "cnt": "CNT"})
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """업종명 공백 제거, 소표본 업종 병합, 지역 키 생성."""
    out = df.copy()

    # '편 의 점', '슈퍼 마켓', '제 과 점' 등 업종명 내 공백 제거
    out["TP_BUZ_NM"] = out["TP_BUZ_NM"].str.replace(r"\s+", "", regex=True)

    # 갈비전문점·한정식은 표본이 매우 적어 일반한식에 병합
    out["TP_BUZ_NM"] = out["TP_BUZ_NM"].replace(
        {b: "일반한식" for b in BUZ_MERGE_TO_KOREAN}
    )

    # 동명 시군구 구분을 위해 시도+시군구를 키로 사용
    out["REGION"] = out["SIDO_NM"].str.strip() + " " + out["CCG_NM"].str.strip()

    out["IS_FOREIGN"] = out["GENDER_CD"] == FOREIGN_CODE
    return out


def load() -> pd.DataFrame:
    return clean(load_raw())


def sanity_report(df: pd.DataFrame) -> dict:
    """제안서에 쓰는 기초 수치를 재계산해 반환 (하드코딩 금지용)."""
    total = float(df["AMT"].sum())
    foreign = float(df.loc[df["IS_FOREIGN"], "AMT"].sum())
    return {
        "rows": int(len(df)),
        "regions": int(df["REGION"].nunique()),
        "months": sorted(df["STRD_YYMM"].unique().tolist()),
        "total_amt": total,
        "foreign_amt": foreign,
        "foreign_share": foreign / total,
        "monthly_amt": (df.groupby("STRD_YYMM")["AMT"].sum()).to_dict(),
        "na_count": int(df.isna().sum().sum()),
        "dup_count": int(
            df.duplicated(
                ["STRD_YYMM", "REGION", "GENDER_CD", "AGE_CD", "TP_BUZ_NO"]
            ).sum()
        ),
    }
