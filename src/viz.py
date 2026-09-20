"""차트 생성.

색 규칙
- 범주형(3개 유형): 고정 순서 blue / orange / aqua (섞거나 순환하지 않음)
- 순차형(크기): 단일 색상 명도 램프
- 발산형(표준화 프로파일): 두 색 + 중립 회색 중간값
- 수치·라벨은 잉크 색만 사용하고, 색은 마크에만 쓴다.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from config import FEATURE_COLS, FIG_DIR, TYPE_INDUSTRIAL, TYPE_RESIDENT, TYPE_URBAN

SERIES = {TYPE_INDUSTRIAL: "#2a78d6", TYPE_RESIDENT: "#eb6834", TYPE_URBAN: "#1baf7a"}
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8880"
GRID = "#e4e2dd"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#e8f0fb", "#2a78d6", "#123a6b"])
DIV = LinearSegmentedColormap.from_list("div", ["#eb6834", "#ece9e3", "#2a78d6"])


def _frame(ax, title=None, subtitle=None):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, length=0, labelsize=9)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    # 제목/부제는 axes 높이가 아니라 포인트 오프셋으로 띄워 겹침을 막는다
    if subtitle:
        ax.annotate(subtitle, xy=(0, 1), xycoords="axes fraction",
                    xytext=(0, 7), textcoords="offset points",
                    va="bottom", ha="left", color=INK_2, fontsize=9)
    if title:
        ax.annotate(title, xy=(0, 1), xycoords="axes fraction",
                    xytext=(0, 24 if subtitle else 7), textcoords="offset points",
                    va="bottom", ha="left", color=INK, fontsize=12)


def save(fig, name: str) -> str:
    path = FIG_DIR / name
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return str(path)


def fig1_top15(totals: pd.DataFrame) -> str:
    """외국인 소비 금액 TOP15와 비중 TOP15 (문제 제기용)."""
    amt = totals.nlargest(15, "외국인금액")
    shr = totals[totals["외국인금액"] >= 1e9].nlargest(15, "외국인비중")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.2))
    for ax, data, col, fmt, title, sub in (
        (axes[0], amt, "외국인금액", lambda v: f"{v/1e8:,.0f}억",
         "외국인 소비 금액 TOP 15", "2026.1~6 · 단위 억원"),
        (axes[1], shr, "외국인비중", lambda v: f"{v*100:.1f}%",
         "외국인 소비 비중 TOP 15", "지역 전체 소비 대비 · 외국인 10억원 이상 지역"),
    ):
        y = np.arange(len(data))[::-1]
        vals = data[col].values
        ax.barh(y, vals, height=0.62, color="#2a78d6")
        ax.set_yticks(y, data.index, fontsize=9)
        for yy, v in zip(y, vals):
            ax.text(v, yy, "  " + fmt(v), va="center", fontsize=8.5, color=INK_2)
        ax.set_xlim(0, vals.max() * 1.18)
        ax.set_xticks([])
        _frame(ax, title, sub)
        ax.grid(False)
    fig.suptitle(
        "금액 상위는 관광지가 아니라 산업단지·근로자 밀집지가 차지한다",
        x=0.01, ha="left", color=INK, fontsize=13.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return save(fig, "fig1_foreign_top15.png")


def fig2_select_k(scores: pd.DataFrame, k: int) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, col, title, ylab in (
        (axes[0], "inertia", "군집 내 분산 (elbow)", "inertia"),
        (axes[1], "silhouette", "실루엣 점수", "silhouette"),
    ):
        ax.plot(scores["k"], scores[col], lw=2, color="#2a78d6",
                marker="o", ms=8, mfc="white", mew=2)
        ax.set_xticks(scores["k"])
        ax.set_xlabel("k", color=INK_2, fontsize=9)
        ax.set_ylabel(ylab, color=INK_2, fontsize=9)
        _frame(ax, title)
        ax.grid(axis="y", color=GRID, lw=0.8)
    sel = scores.loc[scores["k"] == k, "silhouette"].iloc[0]
    axes[1].scatter([k], [sel], s=150, color="#eb6834", zorder=5)
    axes[1].annotate(f"채택 k={k} ({sel:.3f})", (k, sel), textcoords="offset points",
                     xytext=(10, 10), color=INK, fontsize=9.5)
    fig.suptitle("k=3 채택 (k>=3 중 실루엣 최대)", x=0.01, ha="left", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return save(fig, "fig2_select_k.png")


def fig3_profile_heatmap(prof_z: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(10, 3.6))
    v = np.abs(prof_z.values).max()
    im = ax.imshow(prof_z.values, cmap=DIV, norm=TwoSlopeNorm(0, -v, v), aspect="auto")
    ax.set_xticks(range(prof_z.shape[1]), prof_z.columns, fontsize=9, rotation=20, ha="right")
    ax.set_yticks(range(prof_z.shape[0]), prof_z.index, fontsize=10)
    for i in range(prof_z.shape[0]):
        for j in range(prof_z.shape[1]):
            z = prof_z.values[i, j]
            ax.text(j, i, f"{z:+.1f}", ha="center", va="center", fontsize=8.5,
                    color="white" if abs(z) > v * 0.55 else INK)
    ax.tick_params(length=0, colors=INK_2)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, colors=INK_MUTED, labelsize=8)
    cb.set_label("표준화 점수 (전체 평균=0)", color=INK_2, fontsize=8.5)
    ax.annotate("유형별 외국인 소비 프로파일", xy=(0, 1), xycoords="axes fraction",
                xytext=(0, 10), textcoords="offset points", va="bottom", ha="left",
                color=INK, fontsize=12.5)
    fig.tight_layout()
    return save(fig, "fig3_type_profile.png")


def fig4_pca(X, labels_named: pd.Series, bench_scaled: np.ndarray,
             bench_names: list[str], bench_dist: pd.Series,
             typical_dist: float) -> str:
    """PCA 지도. 벤치마크(제주)도 학습과 동일한 스케일러를 거친 값으로 투영한다."""
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2, random_state=0).fit(X)
    P = pca.transform(X)
    fig, ax = plt.subplots(figsize=(9.4, 6.4))
    for name, color in SERIES.items():
        m = (labels_named == name).values
        ax.scatter(P[m, 0], P[m, 1], s=46, color=color, alpha=0.85,
                   edgecolor="white", lw=1.2, label=name, zorder=3)
    if len(bench_scaled):
        B = pca.transform(bench_scaled)
        ax.scatter(B[:, 0], B[:, 1], s=340, marker="*", color="#4a3aa7",
                   edgecolor="white", lw=1.4, label="제주 (학습 제외)", zorder=5)
        offsets = [(16, 10), (16, -18)]
        for i, (nm, (x, y)) in enumerate(zip(bench_names, B)):
            ax.annotate(f"{nm.split()[-1]} (거리 {bench_dist.iloc[i]:.1f})", (x, y),
                        textcoords="offset points", xytext=offsets[i % 2],
                        fontsize=9.5, color=INK, zorder=6,
                        arrowprops=dict(arrowstyle="-", color=INK_MUTED, lw=0.8))
    ev = pca.explained_variance_ratio_ * 100
    ax.set_xlabel(f"PC1 ({ev[0]:.0f}%)", color=INK_2, fontsize=9)
    ax.set_ylabel(f"PC2 ({ev[1]:.0f}%)", color=INK_2, fontsize=9)
    _frame(ax, "시군구 외국인 소비 프로파일 지도 (PCA)",
           f"2차원 투영은 분산의 {ev.sum():.0f}%만 담는다 · 9개 피처 전체 기준 "
           f"제주의 최근접 중심 거리는 {bench_dist.min():.1f}~{bench_dist.max():.1f}"
           f"로 일반 지역(중앙값 {typical_dist:.1f})의 3배")
    ax.grid(color=GRID, lw=0.8)
    leg = ax.legend(frameon=False, fontsize=9.5, loc="upper left",
                    bbox_to_anchor=(1.01, 1.0))
    for t in leg.get_texts():
        t.set_color(INK_2)
    fig.tight_layout()
    return save(fig, "fig4_pca_map.png")


def fig5_monthly_index(idx: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(9, 5.2))
    xs = np.arange(len(idx))
    for name in [TYPE_URBAN, TYPE_INDUSTRIAL, TYPE_RESIDENT]:
        if name not in idx.columns:
            continue
        ax.plot(xs, idx[name].values, lw=2.2, color=SERIES[name],
                marker="o", ms=7, mfc="white", mew=2, label=name)
        ax.annotate(f"{idx[name].iloc[-1]:.0f}", (xs[-1], idx[name].iloc[-1]),
                    textcoords="offset points", xytext=(9, -3),
                    fontsize=9.5, color=SERIES[name])
    ax.axhline(100, color=INK_MUTED, lw=1, ls="--")
    ax.set_xticks(xs, [f"{m % 100}월" for m in idx.index], fontsize=9)
    ax.set_ylabel("내국인 대비 외국인 소비 지수 (1월=100)", color=INK_2, fontsize=9)
    _frame(ax, "유형별 월별 상대지수", "지역 공통 요인(5월 소비 급증 등)을 내국인으로 보정")
    ax.grid(axis="y", color=GRID, lw=0.8)
    leg = ax.legend(frameon=False, fontsize=9.5)
    for t in leg.get_texts():
        t.set_color(INK_2)
    fig.tight_layout()
    return save(fig, "fig5_monthly_index.png")


def fig6_candidates(cand: pd.DataFrame, top_n: int = 10) -> str:
    """상위 후보는 번호 배지로 표시하고, 지역명은 오른쪽 목록으로 뺀다 (라벨 충돌 방지)."""
    fig, ax = plt.subplots(figsize=(10.6, 6.4))
    size = cand["외국인금액"] / cand["외국인금액"].max() * 380 + 40
    ax.scatter(cand["유사도순위"], cand["상대성장률"] * 100, s=size,
               color="#a9c8ec", edgecolor="white", lw=1.1, zorder=3,
               label="기타 후보")
    top = cand.head(top_n)
    ax.scatter(top["유사도순위"], top["상대성장률"] * 100,
               s=np.maximum(size.loc[top.index], 200), color="#2a78d6",
               edgecolor="white", lw=1.3, zorder=4, label=f"상위 {top_n} 후보")
    for i, (nm, r) in enumerate(top.iterrows(), start=1):
        ax.text(r["유사도순위"], r["상대성장률"] * 100, str(i), ha="center",
                va="center", fontsize=8.5, color="white", zorder=5)

    ax.axhline(0, color=INK_MUTED, lw=1, ls="--")
    ax.set_xlim(-0.06, 1.06)
    ax.set_xlabel("도심 방문형 유사도 (백분위)", color=INK_2, fontsize=9)
    ax.set_ylabel("계절성 보정 상대성장률 (%)", color=INK_2, fontsize=9)
    _frame(ax, "잠재 상권 후보", "원 크기 = 외국인 소비 금액 · 번호 = 종합점수 순위")
    ax.grid(color=GRID, lw=0.8)
    leg = ax.legend(frameon=False, fontsize=9, loc="lower left", ncols=2,
                    bbox_to_anchor=(0.0, -0.17))
    for t in leg.get_texts():
        t.set_color(INK_2)

    short = [
        nm.replace("특별자치도", "").replace("특별시", "").replace("광역시", "")
        for nm in top.index
    ]
    lines = [f"{i}. {nm}" for i, nm in enumerate(short, start=1)]
    ax.text(1.03, 1.0, "종합점수 상위 " + str(top_n), transform=ax.transAxes,
            va="top", ha="left", fontsize=10, color=INK)
    ax.text(1.03, 0.94, "\n".join(lines), transform=ax.transAxes,
            va="top", ha="left", fontsize=9, color=INK_2, linespacing=1.7)
    fig.tight_layout()
    return save(fig, "fig6_candidates.png")


def fig7_external_validation(table: pd.DataFrame, corr: pd.DataFrame,
                             bench_label: str = "제주(벤치마크)") -> str:
    """등록외국인 수로 외국인 소비가 얼마나 설명되는지를 유형별로 비교."""
    fig, ax = plt.subplots(figsize=(9.4, 6.4))
    tr = table[table["유형"] != bench_label]
    for name, color in SERIES.items():
        g = tr[tr["유형"] == name]
        if not len(g):
            continue
        x, y = np.log10(g["등록외국인수"]), np.log10(g["외국인금액"] / 1e8)
        ax.scatter(x, y, s=42, color=color, alpha=0.8, edgecolor="white", lw=1.1,
                   zorder=3, label=f"{name} (r={corr.loc[name, '상관계수']:+.2f})")
        b, a = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 20)
        ax.plot(xs, a + b * xs, color=color, lw=2, alpha=0.85, zorder=4)

    bench = table[table["유형"] == bench_label]
    if len(bench):
        ax.scatter(np.log10(bench["등록외국인수"]), np.log10(bench["외국인금액"] / 1e8),
                   s=340, marker="*", color="#4a3aa7", edgecolor="white", lw=1.4,
                   zorder=5, label="제주 (학습 제외)")
        for nm, r in bench.iterrows():
            ax.annotate(nm.split()[-1],
                        (np.log10(r["등록외국인수"]), np.log10(r["외국인금액"] / 1e8)),
                        textcoords="offset points", xytext=(14, -4),
                        fontsize=9.5, color=INK, zorder=6)

    ax.set_xlabel("등록외국인 수 (log10, 명)", color=INK_2, fontsize=9)
    ax.set_ylabel("외국인 소비 (log10, 억원)", color=INK_2, fontsize=9)
    _frame(ax, "외부 데이터 검증: 외국인 소비 vs 등록외국인 수",
           "근로형은 정주 인구로 거의 설명되지만(r=0.95) 도심 방문형은 절반도 설명되지 않는다")
    ax.grid(color=GRID, lw=0.8)
    leg = ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    for t_ in leg.get_texts():
        t_.set_color(INK_2)
    fig.tight_layout()
    return save(fig, "fig7_external_validation.png")


def fig8_final_candidates(cand: pd.DataFrame, top: pd.DataFrame) -> str:
    """VDI(방문 수요) × 상대성장률 사분면."""
    fig, ax = plt.subplots(figsize=(10.2, 6.6))
    ax.axhline(0, color=INK_MUTED, lw=1, ls="--", zorder=1)
    ax.axvline(0, color=INK_MUTED, lw=1, ls="--", zorder=1)

    rest = cand.drop(index=top.index, errors="ignore")
    size = lambda d: d["외국인금액"] / cand["외국인금액"].max() * 320 + 50
    ax.scatter(rest["VDI"], rest["상대성장률"] * 100, s=size(rest), color="#a9c8ec",
               edgecolor="white", lw=1.1, zorder=3, label="기타 후보")
    ax.scatter(top["VDI"], top["상대성장률"] * 100, s=size(top), color="#2a78d6",
               edgecolor="white", lw=1.3, zorder=4, label="최종 TOP 5")
    # 사분면 라벨이 들어갈 여백을 위쪽에 먼저 확보한다
    xl = ax.get_xlim()
    ax.set_xlim(xl[0] - 0.05, xl[1] + 0.12)
    yl = ax.get_ylim()
    ax.set_ylim(yl[0], yl[1] + (yl[1] - yl[0]) * 0.14)

    # 값이 비슷한 후보끼리 겹치지 않도록 위/아래를 번갈아 둔다
    for i, (nm, r) in enumerate(top.iterrows()):
        label = (nm.replace("특별자치도", "").replace("특별시", "")
                   .replace("광역시", "").replace("충청북도 ", ""))
        lo, hi = ax.get_xlim()
        right = r["VDI"] > lo + 0.75 * (hi - lo)  # 오른쪽 끝에서만 안쪽으로 붙인다
        ax.annotate(label, (r["VDI"], r["상대성장률"] * 100),
                    textcoords="offset points",
                    xytext=(-12 if right else 12, 9 if i % 2 == 0 else -17),
                    ha="right" if right else "left",
                    fontsize=9.5, color=INK, zorder=6)

    xl, yl = ax.get_xlim(), ax.get_ylim()
    ax.text(xl[1] - 0.01, yl[1], "방문 수요 + 성장 ", ha="right", va="top",
            fontsize=9, color=INK_MUTED)
    ax.text(xl[0] + 0.01, yl[1], " 잠재 성장형", ha="left", va="top",
            fontsize=9, color=INK_MUTED)
    ax.set_xlabel("방문수요지수 VDI (정주 인구·상권 규모 통제 후 초과 소비)",
                  color=INK_2, fontsize=9)
    ax.set_ylabel("계절성 보정 상대성장률 (%)", color=INK_2, fontsize=9)
    _frame(ax, "잠재 관광 상권 최종 후보",
           "도심 방문형 45곳 (대표 상권 5곳 제외) · 원 크기 = 외국인 소비 금액")
    ax.grid(color=GRID, lw=0.8)
    leg = ax.legend(frameon=False, fontsize=9.5, loc="lower right")
    for t_ in leg.get_texts():
        t_.set_color(INK_2)
    fig.tight_layout()
    return save(fig, "fig8_final_candidates.png")
