"""프로젝트 공통 설정: 경로, 상수, 한글 폰트."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
TBL_DIR = OUT_DIR / "tables"

RAW_CSV = DATA_DIR / "ABP_CONTEST_DATA.csv"

RANDOM_STATE = 42

# 외국인 코드 (GENDER_CD)
FOREIGN_CODE = "3"

# 분석 대상 필터: 6개월 외국인 소비 하한 (원)
MIN_FOREIGN_AMT = 1_000_000_000

# 규모 이상치로 학습에서 제외하고 벤치마크로만 쓰는 지역
BENCHMARK_KEYS = ("제주특별자치도 제주시", "제주특별자치도 서귀포시")

# 유형 후보 k 탐색 범위 (k>=3 중 실루엣 최대값 채택)
K_RANGE = range(2, 7)
K_MIN_ACCEPT = 3

# 분기 정의 (계절성 보정 성장률)
Q1_MONTHS = (202601, 202602, 202603)
Q2_MONTHS = (202604, 202605, 202606)

# 업종 그룹 (공백 제거 후 기준)
BUZ_GROUPS = {
    "장보기": ["슈퍼마켓", "대형할인점"],
    "한식": ["일반한식"],  # 갈비전문점·한정식은 표본이 적어 일반한식에 병합
    "카페양식": ["서양음식", "제과점"],
    "중일분식": ["중국음식", "일식회집", "스넥"],
    "편의점": ["편의점"],
}

# 표본이 적어 일반한식으로 병합하는 업종
BUZ_MERGE_TO_KOREAN = ["갈비전문점", "한정식"]

AGE_YOUNG = ["1", "2"]   # 20대 이하 + 20대
AGE_SENIOR = ["5", "6"]  # 50대 + 60대 이상

FEATURE_COLS = [
    "장보기비중",
    "한식비중",
    "카페양식비중",
    "중일분식비중",
    "편의점비중",
    "청년비중",
    "중장년비중",
    "외국인비중",
    "log건단가",
]

TYPE_INDUSTRIAL = "산업단지 근로형"
TYPE_RESIDENT = "중장년 정주형"
TYPE_URBAN = "도심 방문형"


def setup_font() -> str:
    """matplotlib 한글 폰트 설정. 설치된 나눔/노토 계열을 자동 탐색."""
    import matplotlib
    import matplotlib.font_manager as fm
    from matplotlib import pyplot as plt

    candidates = [
        "NanumGothic", "NanumBarunGothic", "Noto Sans CJK KR", "Noto Sans KR",
        "Malgun Gothic", "AppleGothic",
    ]
    installed = {f.name for f in fm.fontManager.ttflist}
    chosen = next((c for c in candidates if c in installed), None)
    if chosen is None:  # 폰트가 없으면 경고만 남기고 기본값 사용
        print("[warn] 한글 폰트를 찾지 못했습니다. 차트의 한글이 깨질 수 있습니다.")
        chosen = matplotlib.rcParams["font.family"][0]
    plt.rcParams["font.family"] = chosen
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 130
    plt.rcParams["savefig.bbox"] = "tight"
    return chosen


def ensure_dirs() -> None:
    for d in (OUT_DIR, FIG_DIR, TBL_DIR):
        d.mkdir(parents=True, exist_ok=True)
