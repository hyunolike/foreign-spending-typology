"""전체 파이프라인 실행.

전처리 -> 문제 제기 -> 프로파일 -> k 선정 -> 유형 해석 -> 성장률 -> 후보 도출 -> 저장
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import candidates as cd
import cluster as cl
import features as ft
import growth as gr
import preprocess as pp
import viz
from config import BENCHMARK_KEYS, FEATURE_COLS, TBL_DIR, TYPE_URBAN, ensure_dirs, setup_font


def main() -> None:
    ensure_dirs()
    font = setup_font()
    print(f"[0] 폰트: {font}")

    # 1. 전처리 -------------------------------------------------------------
    df = pp.load()
    rep = pp.sanity_report(df)
    print(
        f"[1] {rep['rows']:,}행 / {rep['regions']}개 시군구 / 결측 {rep['na_count']} / "
        f"중복 {rep['dup_count']}"
    )
    print(
        f"    전체 {rep['total_amt']/1e12:.3f}조, 외국인 {rep['foreign_amt']/1e8:,.0f}억 "
        f"({rep['foreign_share']*100:.2f}%)"
    )

    # 2. 피처 + 대상 선정 ---------------------------------------------------
    feat = ft.build_features(df)
    target, train, bench = ft.split_targets(feat)
    print(f"[2] 분석 대상 {len(target)}곳 / 학습 {len(train)}곳 / 벤치마크 {len(bench)}곳")

    viz.fig1_top15(feat)

    # 3. k 선정 -------------------------------------------------------------
    scaler, X = cl.scale(train)
    scores = cl.select_k(X)
    k = cl.best_k(scores)
    ari = cl.stability(X, k)
    print("[3] 실루엣:", {int(r.k): round(r.silhouette, 3) for r in scores.itertuples()})
    print(f"    채택 k={k}, 시드 10회 평균 ARI={ari:.3f}")
    viz.fig2_select_k(scores, k)

    # 4. 군집 + 명명 --------------------------------------------------------
    scaler, X, km = cl.fit(train, k)
    labels = km.labels_
    names = cl.name_clusters(train, labels)
    train["cluster"] = labels
    train["유형"] = train["cluster"].map(names)
    urban_cluster = next(c for c, n in names.items() if n == TYPE_URBAN)

    prof = train.groupby("유형")[FEATURE_COLS].mean()
    prof_z = (prof - train[FEATURE_COLS].mean()) / train[FEATURE_COLS].std()
    viz.fig3_profile_heatmap(prof_z)

    all_rows = pd.concat([train.drop(columns=["cluster", "유형"]), bench])
    dist = cl.distance_to_centers(scaler, km, all_rows)
    dist.columns = [f"거리_{c}" for c in dist.columns]

    # 벤치마크도 학습과 동일한 스케일러를 통과시킨 뒤 투영한다
    bench_scaled = scaler.transform(bench[FEATURE_COLS])
    nearest = dist.min(axis=1)
    viz.fig4_pca(
        X,
        train["유형"],
        bench_scaled,
        bench.index.tolist(),
        nearest.loc[bench.index],
        float(nearest.loc[train.index].median()),
    )

    # 5. 계절성 보정 성장률 --------------------------------------------------
    g = gr.relative_growth(df)
    type_map = train["유형"]
    viz.fig5_monthly_index(gr.monthly_index(df, type_map))

    table = (
        all_rows.join(dist)
        .join(train["유형"])
        .join(g[["외국인_Q1", "외국인_Q2", "외국인성장률", "내국인성장률", "상대성장률"]])
    )
    table["유형"] = table["유형"].fillna("제주(벤치마크)")
    by_type = (
        table[table.index.isin(train.index)]
        .groupby("유형")
        .agg(
            시군구수=("외국인금액", "size"),
            외국인소비합=("외국인금액", "sum"),
            평균외국인비중=("외국인비중", "mean"),
            평균상대성장률=("상대성장률", "mean"),
            **{c: (c, "mean") for c in FEATURE_COLS},
        )
    )
    by_type["외국인소비비중"] = by_type["외국인소비합"] / rep["foreign_amt"]
    print("[5] 유형별 평균 상대성장률:")
    print((by_type[["시군구수", "외국인소비비중", "평균상대성장률"]] * 1).round(4))

    # 6. 후보 도출 ----------------------------------------------------------
    cand, cutoff, flagships = cd.build_candidates(table, urban_cluster)
    print(f"[6] 대표 상권 제외: {flagships}")
    print(f"    유사도 컷오프(75분위)={cutoff:.3f}, 후보 {len(cand)}곳")
    print(cand.head(10)[["유형", "점수", "상대성장률", "청년비중"]].round(4))
    viz.fig6_candidates(cand)

    # 7. 저장 ---------------------------------------------------------------
    table.to_csv(TBL_DIR / "region_typology.csv", encoding="utf-8-sig")
    cand.head(15).to_csv(TBL_DIR / "candidates_top15.csv", encoding="utf-8-sig")
    by_type.to_csv(TBL_DIR / "type_summary.csv", encoding="utf-8-sig")

    meta = {
        "rows": rep["rows"],
        "regions": rep["regions"],
        "total_amt": rep["total_amt"],
        "foreign_amt": rep["foreign_amt"],
        "foreign_share": rep["foreign_share"],
        "monthly_amt": {str(k_): v for k_, v in rep["monthly_amt"].items()},
        "n_target": len(target),
        "n_train": len(train),
        "silhouette": {int(r.k): r.silhouette for r in scores.itertuples()},
        "k": k,
        "ari_mean": ari,
        "cluster_names": {str(c): n for c, n in names.items()},
        "n_candidates": len(cand),
        "similarity_cutoff": float(cutoff),
        "flagships": flagships,
        "jeju_distance": {
            key: dist.loc[key].to_dict() for key in BENCHMARK_KEYS if key in dist.index
        },
    }
    (TBL_DIR / "run_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    print(f"[7] 저장 완료: {TBL_DIR}")


if __name__ == "__main__":
    main()
