"""K-means 유형화: k 선정, 안정성 점검, 유형 명명."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from config import (
    FEATURE_COLS,
    K_MIN_ACCEPT,
    K_RANGE,
    RANDOM_STATE,
    TYPE_INDUSTRIAL,
    TYPE_RESIDENT,
    TYPE_URBAN,
)


def scale(train: pd.DataFrame):
    scaler = StandardScaler().fit(train[FEATURE_COLS])
    X = scaler.transform(train[FEATURE_COLS])
    return scaler, X


def select_k(X: np.ndarray) -> pd.DataFrame:
    rows = []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE).fit(X)
        rows.append(
            {
                "k": k,
                "inertia": km.inertia_,
                "silhouette": silhouette_score(X, km.labels_),
            }
        )
    return pd.DataFrame(rows)


def best_k(scores: pd.DataFrame) -> int:
    cand = scores[scores["k"] >= K_MIN_ACCEPT]
    return int(cand.loc[cand["silhouette"].idxmax(), "k"])


def stability(X: np.ndarray, k: int, n_seeds: int = 10) -> float:
    """시드 변경 시 군집 결과가 유지되는지 ARI 평균으로 확인."""
    base = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE).fit_predict(X)
    aris = [
        adjusted_rand_score(
            base, KMeans(n_clusters=k, n_init=20, random_state=s).fit_predict(X)
        )
        for s in range(n_seeds)
    ]
    return float(np.mean(aris))


def fit(train: pd.DataFrame, k: int):
    scaler, X = scale(train)
    km = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE).fit(X)
    return scaler, X, km


def name_clusters(train: pd.DataFrame, labels: np.ndarray) -> dict:
    """규칙 기반 명명: 카페양식 최고 -> 도심 방문형, 남은 군 중 장보기 최고 -> 근로형."""
    prof = train.assign(cluster=labels).groupby("cluster")[FEATURE_COLS].mean()
    urban = prof["카페양식비중"].idxmax()
    rest = prof.drop(index=urban)
    industrial = rest["장보기비중"].idxmax()
    names = {urban: TYPE_URBAN, industrial: TYPE_INDUSTRIAL}
    for c in prof.index:
        names.setdefault(c, TYPE_RESIDENT)
    return names


def distance_to_centers(scaler, km, feat: pd.DataFrame) -> pd.DataFrame:
    """임의 지역(제주 포함)에서 각 군집 중심까지의 표준화 거리."""
    Z = scaler.transform(feat[FEATURE_COLS])
    d = np.linalg.norm(Z[:, None, :] - km.cluster_centers_[None, :, :], axis=2)
    return pd.DataFrame(d, index=feat.index, columns=range(km.n_clusters))
