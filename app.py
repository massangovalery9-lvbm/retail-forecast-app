import contextlib
import io
import logging
import warnings

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose


warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)


st.set_page_config(
    page_title="Retail Demand Forecasting",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def load_data(uploaded_file):
    """Read an uploaded CSV file into a dataframe."""
    uploaded_file.seek(0)
    return pd.read_csv(uploaded_file)


def validate_columns(df, date_column, sales_column):
    """Validate selected date and sales columns."""
    missing = [col for col in [date_column, sales_column] if col not in df.columns]
    if missing:
        return False, f"Missing required column(s): {', '.join(missing)}"
    return True, ""


def preprocess_data(df, date_column, sales_column, frequency):
    """Clean dates, sales values, missing data, and aggregate demand."""
    working_df = df[[date_column, sales_column]].copy()
    working_df.columns = ["date", "sales"]

    working_df["date"] = pd.to_datetime(working_df["date"], errors="coerce")
    working_df["sales"] = pd.to_numeric(working_df["sales"], errors="coerce")
    working_df = working_df.dropna(subset=["date"])

    if working_df.empty:
        raise ValueError("No valid dates were found. Please check the selected date column.")

    working_df = working_df.sort_values("date")
    working_df["sales"] = working_df["sales"].interpolate(method="linear").ffill().bfill()
    working_df["sales"] = working_df["sales"].fillna(0)

    freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "MS"}
    pandas_freq = freq_map[frequency]

    processed = (
        working_df.set_index("date")["sales"]
        .resample(pandas_freq)
        .sum()
        .reset_index()
    )
    processed["sales"] = processed["sales"].interpolate(method="linear").ffill().bfill()
    return processed, pandas_freq


def plot_trends(df):
    """Create an interactive time series trend chart."""
    rolling_window = min(7, max(2, len(df) // 8))
    chart_df = df.copy()
    chart_df["moving_average"] = chart_df["sales"].rolling(
        window=rolling_window, min_periods=1
    ).mean()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=chart_df["date"],
            y=chart_df["sales"],
            mode="lines+markers",
            name="Observed demand",
            line=dict(color="#2878B5", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=chart_df["date"],
            y=chart_df["moving_average"],
            mode="lines",
            name=f"{rolling_window}-period moving average",
            line=dict(color="#F18F01", width=3),
        )
    )
    fig.update_layout(
        title="Retail Demand Over Time",
        xaxis_title="Date",
        yaxis_title="Sales",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_seasonality(df, pandas_freq):
    """Create a seasonal decomposition chart with trend, seasonality, and residuals."""
    period = 12 if pandas_freq == "MS" else 52 if pandas_freq == "W" else 7
    if len(df) < period * 2:
        raise ValueError(
            f"Seasonality needs at least {period * 2} aggregated records for this frequency."
        )

    series = df.set_index("date")["sales"].asfreq(pandas_freq).interpolate()
    decomposition = seasonal_decompose(series, model="additive", period=period)

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("Observed", "Trend", "Seasonality", "Residuals"),
        vertical_spacing=0.07,
    )

    traces = [
        ("Observed", decomposition.observed, "#2878B5"),
        ("Trend", decomposition.trend, "#F18F01"),
        ("Seasonality", decomposition.seasonal, "#359C73"),
        ("Residuals", decomposition.resid, "#8E5EA2"),
    ]
    for row, (name, values, color) in enumerate(traces, start=1):
        fig.add_trace(
            go.Scatter(x=values.index, y=values, mode="lines", name=name, line=dict(color=color)),
            row=row,
            col=1,
        )

    fig.update_layout(
        height=760,
        showlegend=False,
        template="plotly_white",
        title="Seasonal Decomposition",
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Sales", row=1, col=1)
    fig.update_yaxes(title_text="Trend", row=2, col=1)
    fig.update_yaxes(title_text="Seasonal", row=3, col=1)
    fig.update_yaxes(title_text="Residual", row=4, col=1)
    return fig


def _quiet_model_fit(func, *args, **kwargs):
    """Run forecasting libraries without exposing backend logs to the terminal."""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return func(*args, **kwargs)


def forecast_prophet(df, periods, pandas_freq):
    """Generate a Prophet forecast with confidence intervals."""
    prophet_df = df.rename(columns={"date": "ds", "sales": "y"})
    model = Prophet(interval_width=0.95, daily_seasonality=False)
    _quiet_model_fit(model.fit, prophet_df)

    future = model.make_future_dataframe(periods=periods, freq=pandas_freq)
    forecast = _quiet_model_fit(model.predict, future)
    forecast = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].rename(
        columns={
            "ds": "date",
            "yhat": "forecast",
            "yhat_lower": "lower_bound",
            "yhat_upper": "upper_bound",
        }
    )
    return forecast.tail(periods), forecast


def forecast_arima(df, periods, pandas_freq):
    """Generate an ARIMA forecast with confidence intervals."""
    series = df.set_index("date")["sales"].asfreq(pandas_freq).interpolate().ffill().bfill()
    order = (1, 1, 1) if len(series) < 30 else (2, 1, 2)
    model = ARIMA(series, order=order)
    fitted_model = _quiet_model_fit(model.fit)
    prediction = fitted_model.get_forecast(steps=periods)
    mean_forecast = prediction.predicted_mean
    intervals = prediction.conf_int(alpha=0.05)

    forecast = pd.DataFrame(
        {
            "date": mean_forecast.index,
            "forecast": mean_forecast.values,
            "lower_bound": intervals.iloc[:, 0].values,
            "upper_bound": intervals.iloc[:, 1].values,
        }
    )
    return forecast, forecast


def plot_forecast(history_df, forecast_df, model_name):
    """Create an interactive forecast chart."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history_df["date"],
            y=history_df["sales"],
            mode="lines",
            name="Historical demand",
            line=dict(color="#2878B5", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_df["date"],
            y=forecast_df["forecast"],
            mode="lines+markers",
            name=f"{model_name} forecast",
            line=dict(color="#D1495B", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=pd.concat([forecast_df["date"], forecast_df["date"][::-1]]),
            y=pd.concat([forecast_df["upper_bound"], forecast_df["lower_bound"][::-1]]),
            fill="toself",
            fillcolor="rgba(209, 73, 91, 0.18)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name="95% confidence interval",
        )
    )
    fig.update_layout(
        title=f"{model_name} Retail Demand Forecast",
        xaxis_title="Date",
        yaxis_title="Sales",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def initialize_state():
    defaults = {
        "raw_data": None,
        "processed_data": None,
        "pandas_freq": None,
        "forecast_data": None,
        "forecast_model": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def main():
    initialize_state()

    st.title("Retail Demand Forecasting")
    st.caption("Interactive time series analysis with Prophet and ARIMA")

    with st.sidebar:
        st.header("Data Setup")
        uploaded_file = st.file_uploader("Upload retail demand CSV", type=["csv"])

        date_column = None
        sales_column = None
        raw_preview = None

        if uploaded_file is not None:
            try:
                raw_preview = load_data(uploaded_file)
                columns = list(raw_preview.columns)
                date_default = next(
                    (i for i, col in enumerate(columns) if col.lower() in {"date", "ds"}),
                    0,
                )
                sales_default = next(
                    (i for i, col in enumerate(columns) if col.lower() in {"sales", "demand", "y"}),
                    min(1, len(columns) - 1),
                )
                date_column = st.selectbox("Date column", columns, index=date_default)
                sales_column = st.selectbox("Sales column", columns, index=sales_default)
            except Exception as exc:
                st.error(f"Unable to read this CSV file: {exc}")

        frequency = st.selectbox("Aggregation frequency", ["Daily", "Weekly", "Monthly"], index=1)

        load_clicked = st.button("Load Data", type="primary", use_container_width=True)

        st.divider()
        st.header("Forecast Controls")
        model_choice = st.selectbox("Model choice", ["Prophet", "ARIMA"])
        horizon_unit = st.selectbox("Forecast horizon unit", ["Weeks", "Months"])
        horizon = st.slider("Forecast duration", min_value=1, max_value=52, value=12)
        generate_clicked = st.button("Generate Forecast", use_container_width=True)

    if raw_preview is not None:
        st.subheader("Dataset Preview")
        st.dataframe(raw_preview.head(20), use_container_width=True)

    if load_clicked:
        if raw_preview is None:
            st.warning("Please upload a CSV file before loading data.")
        else:
            is_valid, message = validate_columns(raw_preview, date_column, sales_column)
            if not is_valid:
                st.error(message)
            else:
                try:
                    processed, pandas_freq = preprocess_data(
                        raw_preview, date_column, sales_column, frequency
                    )
                    st.session_state.raw_data = raw_preview
                    st.session_state.processed_data = processed
                    st.session_state.pandas_freq = pandas_freq
                    st.session_state.forecast_data = None
                    st.success("Data loaded and preprocessed successfully.")
                except Exception as exc:
                    st.error(f"Data preprocessing failed: {exc}")

    processed_data = st.session_state.processed_data
    if processed_data is None:
        st.info("Upload a CSV file, select the date and sales columns, then click Load Data.")
        return

    total_demand = processed_data["sales"].sum()
    average_demand = processed_data["sales"].mean()
    latest_demand = processed_data["sales"].iloc[-1]
    date_min = processed_data["date"].min().date()
    date_max = processed_data["date"].max().date()

    metric_cols = st.columns(4)
    metric_cols[0].metric("Total Demand", f"{total_demand:,.0f}")
    metric_cols[1].metric("Average Demand", f"{average_demand:,.2f}")
    metric_cols[2].metric("Latest Demand", f"{latest_demand:,.0f}")
    metric_cols[3].metric("Date Range", f"{date_min} to {date_max}")

    st.subheader("Processed Data")
    st.dataframe(processed_data, use_container_width=True)

    st.subheader("Visual Analysis")
    trend_col, seasonality_col = st.columns(2)
    show_trend = trend_col.button("Show Trend", use_container_width=True)
    show_seasonality = seasonality_col.button("Show Seasonality", use_container_width=True)

    if show_trend:
        st.plotly_chart(plot_trends(processed_data), use_container_width=True)

    if show_seasonality:
        try:
            st.plotly_chart(
                plot_seasonality(processed_data, st.session_state.pandas_freq),
                use_container_width=True,
            )
        except Exception as exc:
            st.warning(str(exc))

    if generate_clicked:
        if len(processed_data) < 8:
            st.warning("Please provide at least 8 aggregated records before forecasting.")
        else:
            periods = horizon * 4 if horizon_unit == "Months" and st.session_state.pandas_freq == "W" else horizon
            if horizon_unit == "Months" and st.session_state.pandas_freq == "D":
                periods = horizon * 30
            if horizon_unit == "Weeks" and st.session_state.pandas_freq == "MS":
                periods = max(1, round(horizon / 4))

            try:
                with st.spinner(f"Generating {model_choice} forecast..."):
                    if model_choice == "Prophet":
                        future_forecast, _ = forecast_prophet(
                            processed_data, periods, st.session_state.pandas_freq
                        )
                    else:
                        future_forecast, _ = forecast_arima(
                            processed_data, periods, st.session_state.pandas_freq
                        )

                future_forecast = future_forecast.copy()
                future_forecast[["forecast", "lower_bound", "upper_bound"]] = future_forecast[
                    ["forecast", "lower_bound", "upper_bound"]
                ].clip(lower=0)
                st.session_state.forecast_data = future_forecast
                st.session_state.forecast_model = model_choice
                st.success("Forecast generated successfully.")
            except Exception as exc:
                st.error(f"Forecast generation failed: {exc}")

    if st.session_state.forecast_data is not None:
        forecast_data = st.session_state.forecast_data
        forecast_model = st.session_state.forecast_model

        st.subheader("Forecast Results")
        forecast_total = forecast_data["forecast"].sum()
        forecast_average = forecast_data["forecast"].mean()
        forecast_peak = forecast_data["forecast"].max()

        forecast_cols = st.columns(3)
        forecast_cols[0].metric("Predicted Total Demand", f"{forecast_total:,.0f}")
        forecast_cols[1].metric("Predicted Average Demand", f"{forecast_average:,.2f}")
        forecast_cols[2].metric("Predicted Peak Demand", f"{forecast_peak:,.0f}")

        st.plotly_chart(
            plot_forecast(processed_data, forecast_data, forecast_model),
            use_container_width=True,
        )

        display_forecast = forecast_data.rename(
            columns={
                "date": "Date",
                "forecast": "Predicted Sales",
                "lower_bound": "Lower Confidence Bound",
                "upper_bound": "Upper Confidence Bound",
            }
        )
        st.dataframe(display_forecast, use_container_width=True)


if __name__ == "__main__":
    main()
