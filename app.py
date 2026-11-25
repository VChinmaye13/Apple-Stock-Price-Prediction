from __future__ import annotations

import io
import itertools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Apple Stock Analytics | Model Dashboard",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = Path(__file__).resolve().parent
DATA_NAMES = ("P721 DATASET.csv", "P721 DATASET(4).csv")
FORECAST_HORIZON = 30
LOOKBACK_DEFAULT = 10
EPOCHS_DEFAULT = 15
RANDOM_SEED = 42
MODEL_NAMES = ("ARIMA", "SARIMA", "LSTM", "GRU")


def _clean_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean the uploaded/bundled OHLCV data."""
    raw = raw.copy()
    raw.columns = [str(c).strip() for c in raw.columns]
    required = {"Date", "Close"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"CSV is missing required column(s): {', '.join(sorted(missing))}")

    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
    numeric_columns = [c for c in ("Open", "High", "Low", "Close", "Adj Close", "Volume") if c in raw.columns]
    for column in numeric_columns:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")

    raw = raw.dropna(subset=["Date", "Close"])
    raw = raw.loc[raw["Close"] > 0]
    raw = raw.sort_values("Date").drop_duplicates("Date", keep="last").set_index("Date")
    if len(raw) < 120:
        raise ValueError(f"At least 120 valid observations are required; found {len(raw)}.")

    if "Volume" in raw.columns:
        raw["Volume"] = raw["Volume"].fillna(raw["Volume"].median()).clip(lower=0)
    return raw


@st.cache_data(show_spinner=False)
def load_data(uploaded_bytes: bytes | None = None) -> pd.DataFrame:
    if uploaded_bytes is not None:
        raw = pd.read_csv(io.BytesIO(uploaded_bytes))
    else:
        path = next((APP_DIR / name for name in DATA_NAMES if (APP_DIR / name).exists()), None)
        if path is None:
            raise FileNotFoundError(
                "P721 DATASET.csv was not found beside app.py. Add the dataset to the project folder "
                "or upload it using the sidebar."
            )
        raw = pd.read_csv(path)
    return _clean_frame(raw)


def feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Causal features: each row uses only that date and earlier observations."""
    close = frame["Close"].astype(float)
    if "Volume" in frame.columns:
        volume_log = np.log1p(frame["Volume"].astype(float).clip(lower=0))
    else:
        volume_log = pd.Series(0.0, index=frame.index)
    return pd.DataFrame(
        {
            "Close": close,
            "Volume_Log": volume_log,
            "MA_7": close.rolling(7, min_periods=1).mean(),
            "MA_30": close.rolling(30, min_periods=1).mean(),
        },
        index=frame.index,
    )


def calculate_metrics(actual, predicted, previous_close: float) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float).reshape(-1)
    predicted = np.asarray(predicted, dtype=float).reshape(-1)
    if len(actual) != len(predicted) or len(actual) == 0:
        raise ValueError("Actual and predicted values must have the same non-zero length.")
    mae = float(mean_absolute_error(actual, predicted))
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    mape = float(np.mean(np.abs((actual - predicted) / np.maximum(np.abs(actual), 1e-8))) * 100)
    r2 = float(r2_score(actual, predicted)) if len(actual) > 1 else float("nan")
    actual_path = np.r_[float(previous_close), actual]
    predicted_path = np.r_[float(previous_close), predicted]
    directional = float(
        np.mean(np.sign(np.diff(actual_path)) == np.sign(np.diff(predicted_path))) * 100
    )
    return {
        "RMSE": rmse,
        "MAE": mae,
        "MAPE (%)": mape,
        "R²": r2,
        "Directional Accuracy (%)": directional,
    }


def select_arima_order(train_close: pd.Series) -> tuple[int, int, int]:
    """Select ARIMA order using training-only AIC; test data remains untouched."""
    best_order = (1, 1, 1)
    best_aic = np.inf
    for order in itertools.product(range(3), (1,), range(3)):
        try:
            fitted = ARIMA(train_close.astype(float), order=order).fit()
            if np.isfinite(fitted.aic) and fitted.aic < best_aic:
                best_aic, best_order = float(fitted.aic), order
        except Exception:
            continue
    return best_order


def build_neural_sequences(scaled_features: np.ndarray, start_index: int, end_index: int, lookback: int):
    """Create sequences whose target rows fall in [start_index, end_index)."""
    X, y = [], []
    first_target = max(lookback, start_index)
    for target_index in range(first_target, end_index):
        X.append(scaled_features[target_index - lookback:target_index, :])
        y.append(scaled_features[target_index, 0])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


def build_neural_model(model_name: str, lookback: int, n_features: int):
    # Import lazily so the data-exploration pages still open if TensorFlow is unavailable.
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout

    tf.keras.utils.set_random_seed(RANDOM_SEED)
    recurrent_layer = LSTM if model_name == "LSTM" else GRU
    model = Sequential(
        [
            recurrent_layer(50, input_shape=(lookback, n_features), return_sequences=False),
            Dropout(0.2),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    callback = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    return model, callback


def fit_neural_model(
    model_name: str,
    frame: pd.DataFrame,
    train_end: int,
    lookback: int,
    epochs: int,
):
    """Fit on the first 80%; reserve the final 20% for testing."""
    features = feature_frame(frame)
    scaler = MinMaxScaler()
    scaler.fit(features.iloc[:train_end])
    scaled = scaler.transform(features).astype(np.float32)
    X_train, y_train = build_neural_sequences(scaled, lookback, train_end, lookback)
    if len(X_train) < 30:
        raise ValueError("Not enough training sequences for this dataset/lookback.")
    model, _ = build_neural_model(model_name, lookback, X_train.shape[2])
    model.fit(X_train, y_train, epochs=int(epochs), batch_size=32,
              shuffle=False, verbose=0)
    return model, scaler, int(epochs)


def fit_neural_on_all_data(model_name: str, frame: pd.DataFrame, lookback: int, epochs: int):
    """Fit a neural model on all available history for the future forecast."""
    features = feature_frame(frame)
    scaler = MinMaxScaler()
    scaler.fit(features)
    scaled = scaler.transform(features).astype(np.float32)
    X_all, y_all = build_neural_sequences(scaled, lookback, len(frame), lookback)
    if len(X_all) < 30:
        raise ValueError("Not enough sequences to fit the final forecasting model.")
    model, _ = build_neural_model(model_name, lookback, X_all.shape[2])
    model.fit(X_all, y_all, epochs=int(epochs), batch_size=32, shuffle=False, verbose=0)
    return model, scaler


def recursive_neural_forecast(
    model,
    scaler: MinMaxScaler,
    history_frame: pd.DataFrame,
    horizon: int,
    lookback: int,
) -> np.ndarray:
    """Fixed-origin recursive forecast; future volume is held at the last known value."""
    raw_features = feature_frame(history_frame).to_numpy(dtype=float).tolist()
    close_history = history_frame["Close"].astype(float).tolist()
    future_values: list[float] = []
    last_volume_log = float(raw_features[-1][1])

    for _ in range(horizon):
        current_scaled = scaler.transform(np.asarray(raw_features[-lookback:], dtype=float))
        x = np.asarray(current_scaled, dtype=np.float32).reshape(1, lookback, 4)
        predicted_scaled = float(np.asarray(model.predict(x, verbose=0)).reshape(-1)[0])
        dummy = np.zeros((1, 4), dtype=float)
        dummy[0, 0] = predicted_scaled
        predicted_close = float(scaler.inverse_transform(dummy)[0, 0])
        predicted_close = max(predicted_close, 0.01)

        close_history.append(predicted_close)
        ma7 = float(np.mean(close_history[-7:]))
        ma30 = float(np.mean(close_history[-30:]))
        raw_features.append([predicted_close, last_volume_log, ma7, ma30])
        future_values.append(predicted_close)

    return np.asarray(future_values, dtype=float)


def run_models(frame: pd.DataFrame, epochs: int, lookback: int):
    """Train/evaluate four models with chronological 80/20 train/test."""
    close = frame["Close"].astype(float)
    train_end = int(len(frame) * 0.80)
    train_frame = frame.iloc[:train_end].copy()
    test_frame = frame.iloc[train_end:].copy()
    train_close = train_frame["Close"].astype(float)
    test_close = test_frame["Close"].astype(float)
    if len(train_frame) <= lookback + 30 or len(test_frame) < 2:
        raise ValueError("The dataset is too short for the chronological 80/20 split and selected lookback.")

    metrics_rows, predictions, errors, fitted_models = [], {}, {}, {}
    best_order = select_arima_order(train_close)

    try:
        arima_fit = ARIMA(train_close, order=best_order).fit()
        arima_pred = np.asarray(arima_fit.forecast(steps=len(test_frame)), dtype=float)
        predictions[f"ARIMA {best_order}"] = pd.Series(arima_pred, index=test_frame.index, name=f"ARIMA {best_order}")
        metrics_rows.append({"Model": f"ARIMA {best_order}", **calculate_metrics(test_close, arima_pred, train_close.iloc[-1])})
        fitted_models["ARIMA"] = arima_fit
    except Exception as exc:
        errors["ARIMA"] = f"{type(exc).__name__}: {exc}"

    try:
        sarima_fit = SARIMAX(train_close, order=best_order, seasonal_order=(1, 1, 1, 5),
            enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        sarima_pred = np.asarray(sarima_fit.forecast(steps=len(test_frame)), dtype=float)
        predictions["SARIMA"] = pd.Series(sarima_pred, index=test_frame.index, name="SARIMA")
        metrics_rows.append({"Model": "SARIMA", **calculate_metrics(test_close, sarima_pred, train_close.iloc[-1])})
        fitted_models["SARIMA"] = sarima_fit
    except Exception as exc:
        errors["SARIMA"] = f"{type(exc).__name__}: {exc}"

    for model_name in ("LSTM", "GRU"):
        try:
            neural_model, scaler, selected_epochs = fit_neural_model(model_name, frame, train_end, lookback, epochs)
            neural_pred = recursive_neural_forecast(neural_model, scaler, train_frame, len(test_frame), lookback)
            predictions[model_name] = pd.Series(neural_pred, index=test_frame.index, name=model_name)
            metrics_rows.append({"Model": model_name, **calculate_metrics(test_close, neural_pred, train_close.iloc[-1])})
            fitted_models[model_name] = (neural_model, scaler, selected_epochs)
        except Exception as exc:
            errors[model_name] = f"{type(exc).__name__}: {exc}"

    if not metrics_rows:
        raise RuntimeError("None of the four models completed successfully. Check the training notes.")
    metrics = pd.DataFrame(metrics_rows).set_index("Model").sort_values("RMSE")
    best_model_label = str(metrics.index[0])
    best_model_key = "ARIMA" if best_model_label.startswith("ARIMA") else best_model_label

    future_dates = pd.bdate_range(start=frame.index[-1] + pd.offsets.BDay(1), periods=FORECAST_HORIZON)
    future_forecasts = {}
    full_close = close.astype(float)
    if "ARIMA" in fitted_models:
        try:
            fit = ARIMA(full_close, order=best_order).fit()
            vals = np.asarray(fit.forecast(steps=FORECAST_HORIZON), dtype=float)
            future_forecasts["ARIMA"] = pd.DataFrame({"Date": future_dates, "Predicted_Close_USD": vals})
        except Exception as exc:
            errors["ARIMA future forecast"] = f"{type(exc).__name__}: {exc}"
    if "SARIMA" in fitted_models:
        try:
            fit = SARIMAX(full_close, order=best_order, seasonal_order=(1, 1, 1, 5),
                enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
            vals = np.asarray(fit.forecast(steps=FORECAST_HORIZON), dtype=float)
            future_forecasts["SARIMA"] = pd.DataFrame({"Date": future_dates, "Predicted_Close_USD": vals})
        except Exception as exc:
            errors["SARIMA future forecast"] = f"{type(exc).__name__}: {exc}"
    for model_name in ("LSTM", "GRU"):
        if model_name not in fitted_models:
            continue
        try:
            selected_epochs = int(fitted_models[model_name][2])
            full_model, full_scaler = fit_neural_on_all_data(model_name, frame, lookback, selected_epochs)
            vals = recursive_neural_forecast(full_model, full_scaler, frame, FORECAST_HORIZON, lookback)
            future_forecasts[model_name] = pd.DataFrame({"Date": future_dates, "Predicted_Close_USD": vals})
        except Exception as exc:
            errors[f"{model_name} future forecast"] = f"{type(exc).__name__}: {exc}"

    prediction_frame = pd.DataFrame({"Actual_Close_USD": test_close})
    for name, series in predictions.items():
        prediction_frame[name] = series.reindex(test_frame.index)
    return {
        "metrics": metrics, "predictions": predictions, "test_actual": test_close,
        "forecast_options": future_forecasts, "prediction_frame": prediction_frame,
        "errors": errors, "best_order": best_order, "best_model": best_model_key,
        "train_end": train_end, "train_count": len(train_frame), "test_count": len(test_frame),
        "lookback": lookback, "epochs": epochs,
    }


def chart_layout(fig, height: int = 420):
    fig.update_layout(
        template="plotly_white",
        height=height,
        hovermode="x unified",
        margin=dict(l=18, r=18, t=58, b=18),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ----------------------------- Sidebar -----------------------------
with st.sidebar:
    st.title("🍎 Apple Analytics")
    st.caption("Stock prediction project workspace")
    st.divider()
    page = st.radio(
        "PROJECT NAVIGATION",
        [
            "Dashboard",
            "Dataset",
            "Model Training & Evaluation",
            "Stock Forecast",
            "About Project",
            "Reports & Downloads",
        ],
    )
    st.divider()
    st.subheader("Project controls")
    uploaded_file = st.file_uploader("Upload Apple historical data (CSV)", type=["csv"])
    epochs = st.slider("Neural-network training epochs", min_value=5, max_value=50, value=EPOCHS_DEFAULT, step=1)
    lookback = st.select_slider("Neural lookback window", options=[5, 10, 15, 20, 30], value=LOOKBACK_DEFAULT)
    st.caption("The project compares ARIMA, SARIMA, LSTM and GRU using a chronological 80/20 train/test split.")
    train_clicked = st.button("🚀 Train & evaluate four models", type="primary", use_container_width=True)
    st.divider()
    st.caption("Required CSV columns: Date and Close. Open, High, Low and Volume are used when available.")

try:
    data = load_data(uploaded_file.getvalue() if uploaded_file else None)
except Exception as exc:
    st.error(str(exc))
    st.stop()

signature = (
    str(data.index.min()), str(data.index.max()), len(data),
    float(data["Close"].iloc[-1]), int(epochs), int(lookback),
)
if st.session_state.get("apple_data_signature") != signature:
    st.session_state.pop("apple_model_results", None)

if train_clicked:
    with st.spinner("Training ARIMA, SARIMA, LSTM and GRU. Neural-network training may take several minutes..."):
        try:
            st.session_state["apple_model_results"] = run_models(data, epochs, lookback)
            st.session_state["apple_data_signature"] = signature
            st.success("All available model runs have finished.")
        except Exception as exc:
            st.error(f"Model evaluation could not be completed: {type(exc).__name__}: {exc}")

results = st.session_state.get("apple_model_results")
has_results = results is not None and st.session_state.get("apple_data_signature") == signature

close = data["Close"].astype(float)
latest_close = float(close.iloc[-1])
period_change = (latest_close / float(close.iloc[0]) - 1) * 100
returns = close.pct_change() * 100


def show_kpis():
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Observations", f"{len(data):,}")
    c2.metric("First available date", data.index.min().strftime("%d %b %Y"))
    c3.metric("Latest available close", f"${latest_close:,.2f}", data.index.max().strftime("%d %b %Y"))
    c4.metric("Period price change", f"{period_change:+.2f}%")


def show_model_comparison():
    if not has_results:
        st.info("Use **Train & evaluate four models** in the sidebar to generate the comparison.")
        return
    metrics = results["metrics"]
    st.dataframe(
        metrics.style.format({
            "RMSE": "{:.4f}", "MAE": "{:.4f}", "MAPE (%)": "{:.2f}%",
            "R²": "{:.4f}", "Directional Accuracy (%)": "{:.2f}%",
        }),
        use_container_width=True,
    )
    best_label = str(metrics.index[0])
    st.success(f"Best model on the held-out test set by RMSE: {best_label}")
    if results["errors"]:
        with st.expander("Training notes and model warnings"):
            for name, message in results["errors"].items():
                st.warning(f"{name}: {message}")


# ----------------------------- Main pages -----------------------------
st.title("Apple Stock Price Prediction")
st.caption("An interactive analytics workspace for historical price exploration, four-model evaluation and 30-session forecasting.")
st.info(
    "Academic demonstration only. The supplied dataset covers historical observations from 2012–2019. "
    "Forecasts are model estimates, not live market quotes or investment advice."
)

if page == "Dashboard":
    st.header("📊 Project Dashboard")
    st.write("A summary of the Apple historical stock dataset and the latest model evaluation.")
    show_kpis()
    st.subheader("Historical closing-price trend")
    fig = px.line(data, x=data.index, y="Close", labels={"x": "Date", "Close": "Closing price (USD)"})
    fig.update_traces(line=dict(width=2))
    fig.update_xaxes(rangeslider_visible=True)
    st.plotly_chart(chart_layout(fig, 470), use_container_width=True)
    left, right = st.columns(2)
    with left:
        st.subheader("Daily percentage returns")
        ret_fig = px.line(x=returns.index, y=returns.values, labels={"x": "Date", "y": "Daily return (%)"})
        ret_fig.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(chart_layout(ret_fig, 320), use_container_width=True)
    with right:
        st.subheader("Trading volume")
        if "Volume" in data.columns:
            vol_fig = px.area(data, x=data.index, y="Volume", labels={"x": "Date", "Volume": "Shares traded"})
            st.plotly_chart(chart_layout(vol_fig, 320), use_container_width=True)
        else:
            st.info("Volume is not available in this CSV.")
    st.subheader("Model run status")
    if has_results:
        st.success(f"Evaluation completed. Best test-set model by RMSE: {results['metrics'].index[0]}.")
    else:
        st.info("No model results are available yet. Use the training button in the sidebar.")

elif page == "Dataset":
    st.header("🗂️ Dataset Explorer")
    st.write("Inspect the supplied Apple historical stock-price data after basic cleaning.")
    show_kpis()
    c1, c2, c3 = st.columns(3)
    c1.metric("Missing cells after cleaning", f"{int(data.isna().sum().sum()):,}")
    c2.metric("Duplicate dates after cleaning", f"{int(data.index.duplicated().sum()):,}")
    c3.metric("Non-positive closing prices", f"{int((data['Close'] <= 0).sum()):,}")
    start_date, end_date = data.index.min().date(), data.index.max().date()
    date_range = st.date_input("Select date range", value=(start_date, end_date), min_value=start_date, max_value=end_date)
    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        view = data.loc[str(date_range[0]):str(date_range[1])]
    else:
        view = data
    st.caption(f"Showing {len(view):,} observations.")
    fig = px.line(view, x=view.index, y="Close", labels={"x": "Date", "Close": "Closing price (USD)"})
    st.plotly_chart(chart_layout(fig, 380), use_container_width=True)
    st.subheader("Descriptive statistics")
    stat_cols = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in view.columns]
    st.dataframe(view[stat_cols].describe().T, use_container_width=True)
    st.subheader("Cleaned data preview")
    st.dataframe(view.tail(100), use_container_width=True)
    st.download_button(
        "Download cleaned dataset (CSV)",
        view.to_csv(index_label="Date").encode("utf-8"),
        file_name="apple_cleaned_dataset.csv",
        mime="text/csv",
    )

elif page == "Model Training & Evaluation":
    st.header("🧠 Model Training & Evaluation")
    st.write("Train and compare the four project models using a chronological 80/20 train/test split.")
    c1, c2, c3, c4 = st.columns(4)
    split_point = int(len(data) * 0.80)
    training_data = data.iloc[:split_point]
    testing_data = data.iloc[split_point:]
    c1.metric("Training observations (80%)", f"{len(training_data):,}")
    c2.metric("Testing observations (20%)", f"{len(testing_data):,}")
    c3.metric("Training period", f"{training_data.index.min():%Y-%m} – {training_data.index.max():%Y-%m}")
    c4.metric("Testing period", f"{testing_data.index.min():%Y-%m} – {testing_data.index.max():%Y-%m}")
    st.info(
        "Models: ARIMA, SARIMA, LSTM and GRU. ARIMA order selection uses training-only AIC; statistical models fit the 80% training period. "
        "LSTM/GRU use Close, log-volume, 7-session moving average and 30-session moving average. "
        "Neural models fit on the first 80% and are evaluated on the untouched final 20%."
    )
    st.caption("Use the **Train & evaluate four models** button in the sidebar. Training may take several minutes.")
    st.caption("The split is chronological: earliest 80% training, latest 20% testing. No random shuffling is used.")
    show_model_comparison()
    if has_results:
        metrics = results["metrics"]
        st.subheader("Actual versus predicted test prices")
        compare = go.Figure()
        compare.add_trace(go.Scatter(x=results["test_actual"].index, y=results["test_actual"].values, name="Actual", mode="lines", line=dict(width=2.5, color="#222222")))
        for name, pred in results["predictions"].items():
            compare.add_trace(go.Scatter(x=pred.index, y=pred.values, name=name, mode="lines"))
        st.plotly_chart(chart_layout(compare, 470), use_container_width=True)
        rmse_fig = px.bar(metrics.reset_index(), x="Model", y="RMSE", title="Model comparison – RMSE (lower is better)")
        st.plotly_chart(chart_layout(rmse_fig, 340), use_container_width=True)
        st.subheader("ARIMA residual diagnostics")
        arima_name = next((name for name in results["predictions"] if name.startswith("ARIMA")), None)
        if arima_name:
            residuals = results["test_actual"] - results["predictions"][arima_name]
            r1, r2 = st.columns(2)
            with r1:
                residual_fig = px.line(x=residuals.index, y=residuals.values, labels={"x": "Date", "y": "Residual"}, title="ARIMA residuals over time")
                residual_fig.add_hline(y=0, line_dash="dash", line_color="red")
                st.plotly_chart(chart_layout(residual_fig, 320), use_container_width=True)
            with r2:
                hist_fig = px.histogram(x=residuals.values, nbins=30, labels={"x": "Residual", "y": "Frequency"}, title="ARIMA residual distribution")
                st.plotly_chart(chart_layout(hist_fig, 320), use_container_width=True)
        st.download_button("Download model metrics (CSV)", metrics.to_csv().encode("utf-8"), file_name="apple_model_metrics.csv", mime="text/csv")
        st.download_button("Download test predictions (CSV)", results["prediction_frame"].to_csv(index_label="Date").encode("utf-8"), file_name="apple_test_predictions.csv", mime="text/csv")

elif page == "Stock Forecast":
    st.header("🔮 Stock Forecast")
    st.write("Generate a 30-weekday forecast using one of the four evaluated models.")
    if not has_results:
        st.info("Train and evaluate the four models first using the sidebar button.")
    else:
        options = list(results["forecast_options"].keys())
        if not options:
            st.error("No future forecast could be generated. Check the training notes in Model Training & Evaluation.")
        else:
            preferred_model = results["best_model"]
            default_model = preferred_model if preferred_model in options else ("GRU" if "GRU" in options else options[0])
            selected_forecast_model = st.selectbox("Choose forecasting model", options, index=options.index(default_model))
            forecast_df = results["forecast_options"][selected_forecast_model]
            first_value = float(forecast_df["Predicted_Close_USD"].iloc[0])
            last_value = float(forecast_df["Predicted_Close_USD"].iloc[-1])
            avg_value = float(forecast_df["Predicted_Close_USD"].mean())
            a, b, c = st.columns(3)
            a.metric("First forecast close", f"${first_value:,.2f}")
            b.metric("30th forecast close", f"${last_value:,.2f}")
            c.metric("Average forecast close", f"${avg_value:,.2f}")
            hist = data["Close"].tail(100)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name="Historical close", mode="lines"))
            fig.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["Predicted_Close_USD"], name=f"{selected_forecast_model} forecast", mode="lines", line=dict(dash="dash", width=2.5)))
            fig.update_layout(title=f"Apple closing-price forecast – {selected_forecast_model}", xaxis_title="Date", yaxis_title="Closing price (USD)")
            st.plotly_chart(chart_layout(fig, 470), use_container_width=True)
            st.caption("The forecast is recursive. For LSTM/GRU, future volume is assumed to remain at the last observed log-volume level.")
            st.dataframe(forecast_df.round(2), use_container_width=True)
            st.download_button("Download 30-session forecast (CSV)", forecast_df.to_csv(index=False).encode("utf-8"), file_name=f"apple_{selected_forecast_model.lower()}_30_session_forecast.csv", mime="text/csv")

elif page == "About Project":
    st.header("ℹ️ About the Project")
    st.subheader("Apple Stock Price Prediction")
    st.write(
        "This academic project explores historical Apple stock closing prices and compares four forecasting approaches: "
        "ARIMA, SARIMA, LSTM and GRU. The supplied dataset contains historical daily market observations described as covering 2012–2019."
    )
    st.subheader("Project workflow")
    st.markdown(
        "1. Load and clean historical stock data.\n"
        "2. Explore closing prices, returns and trading volume.\n"
        "3. Create moving-average and log-volume features for neural models.\n"
        "4. Split observations chronologically into 80% training and 20% testing.\n"
        "5. Use training data for ARIMA order selection.\n"
        "6. Evaluate ARIMA, SARIMA, LSTM and GRU on the untouched final 20% test period.\n"
        "7. Forecast the next 30 weekdays and export the results."
    )
    st.subheader("Models used")
    st.markdown(
        "- **ARIMA:** autoregressive integrated moving-average statistical model.\n"
        "- **SARIMA:** seasonal ARIMA model with a 5-session seasonal period.\n"
        "- **LSTM:** recurrent neural network for sequential patterns.\n"
        "- **GRU:** gated recurrent neural network for sequential patterns."
    )
    st.warning("This is an educational forecasting demonstration. It does not provide live prices, trading signals or investment advice.")

elif page == "Reports & Downloads":
    st.header("📁 Reports & Downloads")
    st.write("Download cleaned data, model evaluation results and forecast outputs.")
    st.download_button("Download cleaned dataset (CSV)", data.to_csv(index_label="Date").encode("utf-8"), file_name="apple_cleaned_dataset.csv", mime="text/csv")
    if has_results:
        st.download_button("Download four-model metrics (CSV)", results["metrics"].to_csv().encode("utf-8"), file_name="apple_four_model_metrics.csv", mime="text/csv")
        st.download_button("Download test predictions (CSV)", results["prediction_frame"].to_csv(index_label="Date").encode("utf-8"), file_name="apple_four_model_test_predictions.csv", mime="text/csv")
        for model_name, forecast_df in results["forecast_options"].items():
            st.download_button(
                f"Download {model_name} 30-session forecast (CSV)",
                forecast_df.to_csv(index=False).encode("utf-8"),
                file_name=f"apple_{model_name.lower()}_30_session_forecast.csv",
                mime="text/csv",
                key=f"download_{model_name}",
            )
        summary = (
            "APPLE STOCK PRICE PREDICTION – PROJECT SUMMARY\n"
            f"Dataset period: {data.index.min():%Y-%m-%d} to {data.index.max():%Y-%m-%d}\n"
            f"Observations: {len(data)}\n"
            f"Best model by held-out RMSE: {results['metrics'].index[0]}\n\n"
            "FOUR-MODEL METRICS\n" + results["metrics"].to_string() +
            "\n\nNote: 30-weekday estimates are historical academic forecasts, not investment advice.\n"
        )
        st.download_button("Download project summary (TXT)", summary.encode("utf-8"), file_name="apple_project_summary.txt", mime="text/plain")
    else:
        st.info("Train the four models to enable evaluation and forecast downloads.")

st.divider()
st.caption(
    "Methodology: chronological 80/20 train/test split; training data supports ARIMA order selection; "
    "the final 20% is held out for test evaluation. LSTM and GRU use Close, log-volume, MA_7 and MA_30 features with chronological scaling and recursive test forecasting. "
    "Forecast horizon: 30 weekdays (exchange holidays are not excluded). Academic use only; not investment advice."
)
