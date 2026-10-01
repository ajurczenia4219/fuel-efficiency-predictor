"""Streamlit web app: predict a car's fuel economy and explore 2025 market segments.

Run locally:  streamlit run app.py
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from src import segment, train

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"
SEGMENTS_PATH = ROOT / "data" / "cars_2025_segments.csv"

st.set_page_config(page_title="Car Fuel Efficiency ML", page_icon="🚗", layout="wide")


@st.cache_resource
def get_model():
    if not MODEL_PATH.exists():  # first run on a fresh deploy: train once
        train.main()
    return joblib.load(MODEL_PATH), json.loads(METRICS_PATH.read_text())


@st.cache_data
def get_training_data():
    return train.load_data()


@st.cache_data
def get_segments():
    if not SEGMENTS_PATH.exists():
        segment.main()
    return pd.read_csv(SEGMENTS_PATH)


model, metrics = get_model()
data = get_training_data()

st.title("Car Fuel Efficiency ML")
st.caption("Predict combined MPG from vehicle specs, and explore how 2025 cars cluster into market segments.")

tab_predict, tab_models, tab_segments = st.tabs(["Predict MPG", "Model comparison", "Market segments"])

with tab_predict:
    col_in, col_out = st.columns([1, 1])
    with col_in:
        vclass = st.selectbox("Vehicle class", sorted(data["class"].unique()),
                              index=sorted(data["class"].unique()).index("compact car"))
        drives = sorted(data["drive"].unique())
        drive = st.selectbox("Drive", drives, index=drives.index("fwd"))
        transmission = st.radio("Transmission", ["a", "m"], horizontal=True,
                                format_func=lambda t: "Automatic" if t == "a" else "Manual")
        cylinders = st.select_slider("Cylinders", options=[3, 4, 5, 6, 8, 10, 12], value=4)
        displacement = st.slider("Engine displacement (L)", 1.0, 6.8, 2.0, 0.1)
        year = st.slider("Model year", 2014, 2024, 2022)

    row = pd.DataFrame([{"class": vclass, "drive": drive, "transmission": transmission,
                         "cylinders": cylinders, "displacement": displacement, "year": year}])
    mpg = float(model.predict(row[train.FEATURES])[0])
    mae = round(metrics["results"][metrics["best_model"]]["test_mae"], 1)

    with col_out:
        st.metric("Predicted combined MPG", f"{mpg:.1f}", help=f"Typical error: ±{mae} MPG")
        st.write(f"About **{235.215 / mpg:.1f} L/100 km**. Typical error on unseen cars: ±{mae} MPG.")
        similar = data[(data["class"] == vclass) & (data["cylinders"] == cylinders)]
        if len(similar):
            st.write(f"Real {vclass}s with {cylinders} cylinders in the data average "
                     f"**{similar['combination_mpg'].mean():.1f} MPG** (n={len(similar)}).")
        fig = px.histogram(data, x="combination_mpg", nbins=30,
                           labels={"combination_mpg": "Combined MPG"}, title="Where this car falls")
        fig.add_vline(x=mpg, line_dash="dash", annotation_text="Your car")
        st.plotly_chart(fig, width="stretch")

with tab_models:
    res = pd.DataFrame(metrics["results"]).T.rename(columns={
        "cv_mae": "CV MAE", "test_mae": "Test MAE", "test_rmse": "Test RMSE", "test_r2": "Test R²"})
    st.dataframe(res, width="stretch")
    sig = metrics["significance"]
    st.write(
        f"Trained on **{metrics['n_rows']} gasoline vehicles** with measured MPG. "
        f"Guessing the average for every car gives an MAE of {metrics['baseline_mae']} MPG, "
        f"so the best model cuts error by about "
        f"{100 * (1 - metrics['results'][metrics['best_model']]['test_mae'] / metrics['baseline_mae']):.0f}%."
    )
    verdict = "is" if sig["paired_t_test_p"] < 0.05 else "is **not**"
    st.write(f"A paired t-test across 10 cross-validation folds shows {sig['best']} {verdict} "
             f"significantly better than {sig['vs']} (p = {sig['paired_t_test_p']}).")

with tab_segments:
    seg = get_segments()
    st.write(f"K-Means on {len(seg):,} 2025 models using horsepower, torque, top speed, "
             "0-100 km/h time and price (log scale). k was chosen by silhouette score.")
    fig = px.scatter(seg, x="horsepower", y="price_usd", color="segment", log_y=True,
                     hover_data=["make", "model", "zero_to_100_s"],
                     labels={"price_usd": "Price (USD, log scale)", "horsepower": "Horsepower"})
    st.plotly_chart(fig, width="stretch")
    st.dataframe(
        seg.groupby("segment")[segment.FEATURES].median().round(1)
        .assign(models=seg.groupby("segment").size()).sort_values("price_usd"),
        width="stretch",
    )
