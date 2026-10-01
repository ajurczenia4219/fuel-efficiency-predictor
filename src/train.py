"""Train and compare regression models that predict real, measured fuel economy (MPG).

Data: data/car_data.csv (550 vehicles, model years 2014-2024, measured city/highway/combined MPG).

Run:  python -m src.train
Saves the best model to models/model.joblib and metrics to models/metrics.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "car_data.csv"
MODELS = ROOT / "models"

NUMERIC = ["cylinders", "displacement", "year"]
CATEGORICAL = ["class", "drive", "transmission"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "combination_mpg"


def load_data(path: Path = DATA) -> pd.DataFrame:
    """Load and clean the data.

    Electric vehicles (2 rows) are removed: MPG is not meaningful for them and they have
    no displacement/cylinders. Diesel (2 rows) is too rare to learn from, so the model is
    scoped to gasoline vehicles.
    """
    df = pd.read_csv(path)
    df = df[df["fuel_type"] == "gas"]
    df = df.dropna(subset=FEATURES + [TARGET])
    return df.reset_index(drop=True)


def build_pipeline(model) -> Pipeline:
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )
    return Pipeline([("pre", pre), ("model", model)])


def candidates() -> dict:
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05,
                                subsample=0.9, random_state=42),
    }


def evaluate(df: pd.DataFrame) -> dict:
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    cv = KFold(n_splits=10, shuffle=True, random_state=42)

    results, fold_errors = {}, {}
    for name, model in candidates().items():
        pipe = build_pipeline(model)
        errs = -cross_val_score(pipe, X_train, y_train, cv=cv, scoring="neg_mean_absolute_error")
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        fold_errors[name] = errs
        results[name] = {
            "cv_mae": round(float(errs.mean()), 3),
            "test_mae": round(float(mean_absolute_error(y_test, pred)), 3),
            "test_rmse": round(float(root_mean_squared_error(y_test, pred)), 3),
            "test_r2": round(float(r2_score(y_test, pred)), 3),
        }

    ranked = sorted(results, key=lambda n: results[n]["cv_mae"])
    best, runner_up = ranked[0], ranked[1]
    # Paired t-test on the same 10 CV folds: is the best model's error significantly lower?
    _, p = stats.ttest_rel(fold_errors[best], fold_errors[runner_up])
    baseline_mae = float((y_test - y_train.mean()).abs().mean())  # always-guess-the-average
    return {
        "best_model": best,
        "results": results,
        "significance": {"best": best, "vs": runner_up, "paired_t_test_p": round(float(p), 4)},
        "baseline_mae": round(baseline_mae, 3),
        "n_rows": int(len(df)),
    }


def main() -> None:
    df = load_data()
    report = evaluate(df)
    for name, r in report["results"].items():
        print(f"{name:18s} {r}")
    s = report["significance"]
    print(f"\nBaseline (predict the mean) MAE: {report['baseline_mae']}")
    print(f"Best: {s['best']} vs {s['vs']}: paired t-test p = {s['paired_t_test_p']}")

    final = build_pipeline(candidates()[report["best_model"]]).fit(df[FEATURES], df[TARGET])
    MODELS.mkdir(exist_ok=True)
    joblib.dump(final, MODELS / "model.joblib")
    (MODELS / "metrics.json").write_text(json.dumps(report, indent=2))
    print(f"Saved {report['best_model']} to models/model.joblib")


if __name__ == "__main__":
    main()
