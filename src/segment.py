"""K-Means market segmentation of 2025 car models.

Data: data/cars_2025.csv (Kaggle "Cars Datasets 2025", ~1,200 models).
Clustering needs no target variable, so this works directly on the real specs.

Run:  python -m src.segment
Writes data/cars_2025_segments.csv and models/segments.json.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "cars_2025.csv"
MODELS = ROOT / "models"

SPEC_COLUMNS = {
    "HorsePower": "horsepower",
    "Torque": "torque_nm",
    "Total Speed": "top_speed_kmh",
    "Performance(0 - 100 )KM/H": "zero_to_100_s",
    "Cars Prices": "price_usd",
}
FEATURES = list(SPEC_COLUMNS.values())


def extract_max_numeric(val) -> float:
    """Pull the largest number out of messy text like '100 - 140 Nm' or '$1,100,000'."""
    if pd.isna(val):
        return np.nan
    numbers = re.findall(r"\d+\.?\d*", str(val).replace(",", ""))
    return max(float(n) for n in numbers) if numbers else np.nan


def load_data(path: Path = DATA) -> pd.DataFrame:
    raw = pd.read_csv(path, encoding="latin1")
    df = raw[["Company Names", "Cars Names", "Fuel Types"]].rename(
        columns={"Company Names": "make", "Cars Names": "model", "Fuel Types": "fuel_type"}
    )
    for src, dst in SPEC_COLUMNS.items():
        df[dst] = raw[src].apply(extract_max_numeric)
    df = df.dropna(subset=FEATURES)
    # Data-quality rules found during EDA:
    #  - 0-100 times under 1.5 s and prices under $1,000 are entry errors.
    #  - GMC Hummer EV (15,590) and Tesla Roadster (10,000) list "wheel torque", a marketing
    #    number roughly 10x larger than engine torque, so torque over 4,000 Nm is dropped.
    #  - Horsepower over 2,000 is an entry error (e.g. a Nissan Urvan van listed at 2,488 hp).
    df = df[
        (df["zero_to_100_s"] > 1.5)
        & (df["price_usd"] > 1000)
        & (df["torque_nm"] <= 4000)
        & (df["horsepower"] <= 2000)
    ]
    return df.reset_index(drop=True)


def fit_segments(df: pd.DataFrame, k_range=range(3, 8), random_state: int = 42):
    """Pick k by silhouette score. k starts at 3: k=2 scores about the same (0.53 vs 0.52)
    but only splits "expensive" from "not", which is too coarse to be useful."""
    # Prices span $10k to $3M+, so log-transform before scaling.
    X = df[FEATURES].copy()
    X["price_usd"] = np.log10(X["price_usd"])
    X_scaled = StandardScaler().fit_transform(X)

    scores = {}
    for k in k_range:
        labels = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit_predict(X_scaled)
        scores[k] = float(silhouette_score(X_scaled, labels))
    best_k = max(scores, key=scores.get)
    labels = KMeans(n_clusters=best_k, n_init=10, random_state=random_state).fit_predict(X_scaled)
    return labels, best_k, scores


def name_segments(df: pd.DataFrame) -> dict[int, str]:
    """Give each cluster a readable name from its median specs.

    Special profiles are named first (heavy-duty trucks, hypercars); the rest are ranked by
    median price.
    """
    med = df.groupby("segment_id")[FEATURES].median()
    names: dict[int, str] = {}
    for seg, row in med.iterrows():
        if row["torque_nm"] > 1500 and row["top_speed_kmh"] < 160:
            names[seg] = "Heavy-duty"
        elif row["price_usd"] >= 1_000_000:
            names[seg] = "Hypercar"
    tiers = ["Everyday", "Premium", "Sports & Luxury", "Supercar", "Ultra-luxury"]
    rest = med.drop(index=list(names)).sort_values("price_usd").index
    for i, seg in enumerate(rest):
        names[seg] = tiers[i] if i < len(tiers) else f"Segment {i + 1}"
    return names


def main() -> None:
    df = load_data()
    labels, best_k, scores = fit_segments(df)
    df["segment_id"] = labels
    df["segment"] = df["segment_id"].map(name_segments(df))

    summary = (
        df.groupby("segment")[FEATURES]
        .median()
        .assign(count=df.groupby("segment").size())
        .sort_values("price_usd")
    )
    print(f"Rows: {len(df)}  |  best k by silhouette: {best_k}")
    print({k: round(v, 3) for k, v in scores.items()})
    print(summary.round(1))

    df.to_csv(ROOT / "data" / "cars_2025_segments.csv", index=False)
    MODELS.mkdir(exist_ok=True)
    (MODELS / "segments.json").write_text(json.dumps(
        {"best_k": best_k, "silhouette": scores,
         "summary": summary.round(2).reset_index().to_dict(orient="records")}, indent=2))


if __name__ == "__main__":
    main()
