from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from cnlib.base_strategy import BaseStrategy


COIN_LABELS = {
    "kapcoin-usd_train": "KAPCOIN / USD",
    "metucoin-usd_train": "METUCOIN / USD",
    "tamcoin-usd_train": "TAMCOIN / USD",
}

TIME_RANGES = {
    "1W": "Son 1 hafta",
    "1M": "Son 1 ay",
    "3M": "Son 3 ay",
    "6M": "Son 6 ay",
    "YTD": "Yil basi",
    "1Y": "Son 1 yil",
    "All": "Tum veri",
}

CANDLE_PERIODS = {
    "Gunluk": None,
    "Haftalik": "W",
    "Aylik": "ME",
    "Yillik": "YE",
}

REQUIRED_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]


class MarketDataLoader(BaseStrategy):
    def predict(self, data: dict) -> list[dict]:
        return [
            {"coin": coin, "signal": 0, "allocation": 0.0, "leverage": 1}
            for coin in data
        ]


@dataclass(frozen=True)
class MarketStats:
    last_price: float
    return_pct: float
    period_high: float
    period_low: float
    total_volume: float
    candle_count: int


def set_page_style() -> None:
    st.set_page_config(
        page_title="CNLIB Trade Dashboard",
        page_icon="",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        .stApp {
            background: #f7f8fb;
            color: #111827;
        }

        [data-testid="stSidebar"] {
            background: #101418;
            border-right: 1px solid #232a31;
        }

        [data-testid="stSidebar"] * {
            color: #f8fafc;
        }

        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 14px 16px;
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
        }

        [data-testid="stMetricLabel"] {
            color: #64748b;
        }

        [data-testid="stMetricValue"] {
            color: #111827;
            font-size: 1.55rem;
        }

        .terminal-bar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 16px;
            padding: 16px 18px;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            background: #ffffff;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.05);
            margin-bottom: 18px;
        }

        .terminal-title {
            font-size: 1.35rem;
            font-weight: 700;
            color: #0f172a;
            line-height: 1.2;
        }

        .terminal-subtitle {
            color: #64748b;
            font-size: 0.92rem;
            margin-top: 4px;
        }

        .market-badge {
            color: #0f766e;
            background: #ccfbf1;
            border: 1px solid #99f6e4;
            border-radius: 999px;
            font-weight: 700;
            padding: 8px 12px;
            white-space: nowrap;
        }

        div[data-testid="stPlotlyChart"] {
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            overflow: hidden;
            background: #ffffff;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.05);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def normalize_market_frame(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()

    if "Date" not in df.columns:
        df = df.reset_index()
        if "Date" not in df.columns:
            first_column = df.columns[0]
            df = df.rename(columns={first_column: "Date"})

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Eksik kolonlar: {', '.join(missing)}")

    df = df[REQUIRED_COLUMNS].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    numeric_columns = ["Open", "High", "Low", "Close", "Volume"]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=REQUIRED_COLUMNS)
    df = df.sort_values("Date").reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def load_market_data() -> dict[str, pd.DataFrame]:
    loader = MarketDataLoader()
    loader.get_data()

    market_data = {}
    for coin, frame in loader.coin_data.items():
        if coin in COIN_LABELS:
            market_data[coin] = normalize_market_frame(frame)

    if not market_data:
        raise RuntimeError("CNLIB icinden beklenen coin verisi okunamadi.")

    return market_data


def filter_time_range(df: pd.DataFrame, range_key: str) -> pd.DataFrame:
    if df.empty or range_key == "All":
        return df.copy()

    end_date = df["Date"].max()

    if range_key == "1W":
        start_date = end_date - pd.Timedelta(days=7)
    elif range_key == "1M":
        start_date = end_date - pd.DateOffset(months=1)
    elif range_key == "3M":
        start_date = end_date - pd.DateOffset(months=3)
    elif range_key == "6M":
        start_date = end_date - pd.DateOffset(months=6)
    elif range_key == "YTD":
        start_date = pd.Timestamp(year=end_date.year, month=1, day=1)
    elif range_key == "1Y":
        start_date = end_date - pd.DateOffset(years=1)
    else:
        start_date = df["Date"].min()

    filtered = df[df["Date"] >= start_date].copy()
    return filtered if not filtered.empty else df.tail(1).copy()


def resample_ohlcv(df: pd.DataFrame, rule: str | None) -> pd.DataFrame:
    if rule is None or df.empty:
        return df.copy()

    aggregated = (
        df.set_index("Date")
        .resample(rule)
        .agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }
        )
        .dropna()
        .reset_index()
    )
    return aggregated


def calculate_stats(df: pd.DataFrame) -> MarketStats:
    first_close = float(df["Close"].iloc[0])
    last_price = float(df["Close"].iloc[-1])
    return_pct = ((last_price / first_close) - 1) * 100 if first_close else 0.0

    return MarketStats(
        last_price=last_price,
        return_pct=return_pct,
        period_high=float(df["High"].max()),
        period_low=float(df["Low"].min()),
        total_volume=float(df["Volume"].sum()),
        candle_count=len(df),
    )


def format_price(value: float) -> str:
    return f"${value:,.2f}"


def format_pct(value: float) -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.2f}%"


def format_volume(value: float) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.2f}K"
    return f"{value:,.0f}"


def make_market_chart(df: pd.DataFrame, coin_label: str) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.72, 0.28],
    )

    fig.add_trace(
        go.Candlestick(
            x=df["Date"],
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Fiyat",
            increasing_line_color="#16a34a",
            increasing_fillcolor="#22c55e",
            decreasing_line_color="#dc2626",
            decreasing_fillcolor="#ef4444",
        ),
        row=1,
        col=1,
    )

    volume_colors = [
        "#16a34a" if close >= open_ else "#dc2626"
        for open_, close in zip(df["Open"], df["Close"])
    ]
    fig.add_trace(
        go.Bar(
            x=df["Date"],
            y=df["Volume"],
            name="Hacim",
            marker_color=volume_colors,
            opacity=0.65,
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title={
            "text": coin_label,
            "x": 0.02,
            "xanchor": "left",
            "font": {"size": 20, "color": "#111827"},
        },
        template="plotly_white",
        height=680,
        margin={"l": 32, "r": 24, "t": 56, "b": 28},
        hovermode="x unified",
        showlegend=False,
        xaxis_rangeslider_visible=False,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font={"family": "Arial, sans-serif", "color": "#111827"},
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="#eef2f7",
        linecolor="#e5e7eb",
        tickfont={"color": "#475569"},
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#eef2f7",
        linecolor="#e5e7eb",
        tickfont={"color": "#475569"},
        title_text="Fiyat",
        row=1,
        col=1,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#eef2f7",
        linecolor="#e5e7eb",
        tickfont={"color": "#475569"},
        title_text="Hacim",
        row=2,
        col=1,
    )
    return fig


def render_header(coin_label: str, date_min: pd.Timestamp, date_max: pd.Timestamp) -> None:
    st.markdown(
        f"""
        <div class="terminal-bar">
            <div>
                <div class="terminal-title">CNLIB Trade Dashboard</div>
                <div class="terminal-subtitle">
                    {date_min:%d.%m.%Y} - {date_max:%d.%m.%Y} arasi piyasa verisi
                </div>
            </div>
            <div class="market-badge">{coin_label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metrics(stats: MarketStats) -> None:
    col_price, col_return, col_high_low, col_volume, col_candles = st.columns(5)
    col_price.metric("Son fiyat", format_price(stats.last_price))
    col_return.metric("Donem getirisi", format_pct(stats.return_pct))
    col_high_low.metric(
        "Yuksek / Dusuk",
        f"{format_price(stats.period_high)} / {format_price(stats.period_low)}",
    )
    col_volume.metric("Toplam hacim", format_volume(stats.total_volume))
    col_candles.metric("Mum sayisi", f"{stats.candle_count:,}")


def main() -> None:
    set_page_style()

    with st.spinner("CNLIB market verisi yukleniyor..."):
        market_data = load_market_data()

    available_coins = [coin for coin in COIN_LABELS if coin in market_data]

    st.sidebar.title("Piyasa")
    selected_coin = st.sidebar.selectbox(
        "Coin",
        options=available_coins,
        format_func=lambda coin: COIN_LABELS.get(coin, coin),
    )
    selected_range = st.sidebar.radio(
        "Zaman araligi",
        options=list(TIME_RANGES.keys()),
        format_func=lambda key: f"{key} - {TIME_RANGES[key]}",
        horizontal=False,
    )
    selected_period = st.sidebar.radio(
        "Mum periyodu",
        options=list(CANDLE_PERIODS.keys()),
        horizontal=False,
    )

    source_df = market_data[selected_coin]
    filtered_df = filter_time_range(source_df, selected_range)
    chart_df = resample_ohlcv(filtered_df, CANDLE_PERIODS[selected_period])

    coin_label = COIN_LABELS.get(selected_coin, selected_coin)
    render_header(coin_label, source_df["Date"].min(), source_df["Date"].max())

    stats = calculate_stats(chart_df)
    render_metrics(stats)

    st.plotly_chart(
        make_market_chart(chart_df, f"{coin_label} - {selected_range} / {selected_period}"),
        use_container_width=True,
        config={
            "displaylogo": False,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
            "responsive": True,
        },
    )

    with st.expander("Son mumlar", expanded=False):
        display_df = chart_df.tail(12).sort_values("Date", ascending=False).copy()
        display_df["Date"] = display_df["Date"].dt.strftime("%d.%m.%Y")
        st.dataframe(display_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
