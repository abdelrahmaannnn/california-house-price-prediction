"""
California House Price Predictor - Gradio web app.

Trains two models on startup (a few seconds) and serves a small web UI:
  1. Univariate linear regression on median income, trained with
     gradient descent written from scratch: f(x) = w*x + b
  2. Multiple linear regression on all 8 features (scikit-learn)

Run locally:
    pip install -r requirements.txt
    python app.py
"""

import numpy as np
import gradio as gr
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURES = ["MedInc", "HouseAge", "AveRooms", "AveBedrms",
            "Population", "AveOccup", "Latitude", "Longitude"]
TARGET = "MedHouseVal"


# ---------------------------------------------------------------------------
# Model 1: univariate linear regression from scratch
# ---------------------------------------------------------------------------
def predict(x, w, b):
    """f_wb(x) = w * x + b"""
    return w * x + b


def compute_gradient(x, y, w, b):
    m = len(x)
    errors = predict(x, w, b) - y
    dj_dw = np.sum(errors * x) / m
    dj_db = np.sum(errors) / m
    return dj_dw, dj_db


def gradient_descent(x, y, w, b, alpha=0.01, num_iters=10_000):
    for _ in range(num_iters):
        dj_dw, dj_db = compute_gradient(x, y, w, b)
        w, b = w - alpha * dj_dw, b - alpha * dj_db   # simultaneous update
    return w, b


# ---------------------------------------------------------------------------
# Data + training
# ---------------------------------------------------------------------------
df = fetch_california_housing(as_frame=True).frame
df = df[df[TARGET] < 5.0].copy()          # drop capped prices (500k+ recorded as 500k)

x_train, _, y_train, _ = train_test_split(
    df["MedInc"].values, df[TARGET].values, test_size=0.2, random_state=42)
w_final, b_final = gradient_descent(x_train, y_train, 0.0, 0.0)

X_train, _, Y_train, _ = train_test_split(
    df[FEATURES].values, df[TARGET].values, test_size=0.2, random_state=42)
scaler = StandardScaler().fit(X_train)
multi = LinearRegression().fit(scaler.transform(X_train), Y_train)

print(f"Model 1 trained: f(x) = {w_final:.4f} * x + {b_final:.4f}")


# ---------------------------------------------------------------------------
# Prediction function used by the UI
# ---------------------------------------------------------------------------
def predict_price(med_inc, house_age, ave_rooms, ave_bedrms,
                  population, ave_occup, lat, lon):
    simple = predict(med_inc, w_final, b_final)

    row = np.array([[med_inc, house_age, ave_rooms, ave_bedrms,
                     population, ave_occup, lat, lon]])
    full = multi.predict(scaler.transform(row))[0]   # same scaling as training

    # prices are in units of 100k USD; clip to the training range
    simple = float(np.clip(simple, 0.15, 5.0))
    full = float(np.clip(full, 0.15, 5.0))
    return f"${simple * 100_000:,.0f}", f"${full * 100_000:,.0f}"


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def slider(col, label, step):
    lo, hi = df[col].quantile([0.01, 0.99])
    return gr.Slider(float(lo), float(hi), value=float(df[col].median()),
                     step=step, label=label)


with gr.Blocks(title="California House Price Predictor") as demo:
    gr.Markdown("# 🏠 California House Price Predictor\n"
                "Describe a neighborhood and press **Predict**. "
                "Income is in tens of thousands of USD (6 = 60,000 USD).")

    with gr.Row():
        with gr.Column():
            inc = slider("MedInc", "Median income (x 10k USD)", 0.1)
            age = slider("HouseAge", "Median house age (years)", 1)
            rooms = slider("AveRooms", "Average rooms per house", 0.1)
            beds = slider("AveBedrms", "Average bedrooms per house", 0.05)
        with gr.Column():
            pop = slider("Population", "Block population", 10)
            occ = slider("AveOccup", "Average people per house", 0.1)
            lat = slider("Latitude", "Latitude", 0.01)
            lon = slider("Longitude", "Longitude", 0.01)

    btn = gr.Button("Predict", variant="primary")

    with gr.Row():
        out_simple = gr.Textbox(label="Model 1: income only (gradient descent from scratch)")
        out_full = gr.Textbox(label="Model 2: all 8 features (scikit-learn)")

    inputs = [inc, age, rooms, beds, pop, occ, lat, lon]
    btn.click(predict_price, inputs=inputs, outputs=[out_simple, out_full])

    gr.Examples(
        examples=[
            [8.0, 30, 6.0, 1.0, 1000, 2.5, 37.77, -122.42],   # San Francisco
            [3.5, 25, 5.0, 1.1, 1500, 3.2, 34.05, -118.25],   # Los Angeles
            [2.5, 20, 5.5, 1.1, 1200, 3.0, 36.74, -119.79],   # Fresno (inland)
        ],
        inputs=inputs,
        label="Try an example",
    )

if __name__ == "__main__":
    demo.launch()
