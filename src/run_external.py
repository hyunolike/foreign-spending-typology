"""외부 데이터(법무부 등록외국인) 결합 단계.

`run_all.py`를 먼저 실행해 outputs/tables/region_typology.csv 가 있어야 한다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

import candidates as cd
import external as ex
import residual as rs
import viz
from config import TBL_DIR, ensure_dirs, setup_font

TYPOLOGY_CSV = TBL_DIR / "region_typology.csv"


def main() -> None:
    ensure_dirs()
    setup_font()
    if not TYPOLOGY_CSV.exists():
        raise SystemExit(f"{TYPOLOGY_CSV} 가 없습니다. 먼저 `python src/run_all.py` 실행.")

    table = pd.read_csv(TYPOLOGY_CSV, index_col=0)

    # 1. 결합 ---------------------------------------------------------------
    print("[1] 법무부 등록외국인 결합")
    table = table.join(ex.registered_foreigners(table.index))

    # 2. 검증 ---------------------------------------------------------------
    table, stats = rs.fit_vdi(table)
    b = stats["beta"]
    print(f"[2] log(외국인소비) = {b[0]:.2f} + {b[1]:.3f}·log(등록외국인)"
          f" + {b[2]:.3f}·log(내국인소비)")
    print(f"    R2={stats['r2']:.3f}, n={stats['n']}, 잔차 s.d.={stats['resid_std']:.3f}")

    corr = rs.correlation_by_type(table)
    print("[3] 유형별 검증")
    print(corr[["시군구수", "상관계수", "설명력_R2", "VDI_평균"]].round(3).to_string())
    bench = table[table["유형"] == rs.BENCH_LABEL]
    print("    제주 VDI:", bench["VDI"].round(2).to_dict())
    viz.fig7_external_validation(table, corr)

    # 3. 최종 후보 -----------------------------------------------------------
    cand, flagships = cd.build_candidates_external(table)
    cand["성격"] = cd.label_quadrant(cand)
    top = cd.final_top5(cand)
    print(f"[4] 후보 {len(cand)}곳 (대표 상권 제외: {flagships})")
    print(cand["성격"].value_counts().to_string())
    print("\n[5] 최종 잠재 관광 상권 TOP 5")
    print(
        top.assign(인당만원=lambda d: (d["인당소비"] / 1e4).round(0))[
            ["점수_외부", "VDI", "상대성장률", "인당만원"]
        ].round(3).to_string()
    )
    viz.fig8_final_candidates(cand, top)

    # 4. 저장 ---------------------------------------------------------------
    table.to_csv(TBL_DIR / "region_typology_external.csv", encoding="utf-8-sig")
    cand.to_csv(TBL_DIR / "candidates_external.csv", encoding="utf-8-sig")
    corr.to_csv(TBL_DIR / "external_validation.csv", encoding="utf-8-sig")
    (TBL_DIR / "external_meta.json").write_text(
        json.dumps(
            {
                "regression": stats,
                "corr_by_type": corr["상관계수"].to_dict(),
                "per_capita_median": corr["인당소비_중앙값"].to_dict(),
                "jeju_vdi": bench["VDI"].to_dict(),
                "flagships": flagships,
                "n_candidates": int(len(cand)),
                "top5": top.index.tolist(),
            },
            ensure_ascii=False, indent=2, default=float,
        ),
        encoding="utf-8",
    )
    print(f"\n[6] 저장 완료: {TBL_DIR}")


if __name__ == "__main__":
    main()
