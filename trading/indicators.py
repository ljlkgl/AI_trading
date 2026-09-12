"""技术指标计算。

与 TradingAgents 采用的 stockstats 指标体系保持一致：
SMA50/200、EMA10、MACD(12,26,9)、RSI(14)、Bollinger(20,2)、ATR(14)、VWMA、MFI(14)。
纯 pandas/numpy 实现，避免对 stockstats 与 yfinance 的依赖。
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from trading.types import Candle, TechnicalSnapshot

logger = logging.getLogger(__name__)


def candles_to_df(candles: list[Candle]) -> pd.DataFrame:
    """将 K 线列表转为 DataFrame，列名与 TradingAgents 的 OHLCV 一致。"""
    df = pd.DataFrame(
        [
            {
                "Date": pd.to_datetime(c.open_time, unit="ms"),
                "Open": c.open,
                "High": c.high,
                "Low": c.low,
                "Close": c.close,
                "Volume": c.volume,
            }
            for c in candles
        ]
    )
    return df


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window).mean()


def ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False).mean()


def macd(close: pd.Series, fast=12, slow=26, signal=9):
    """返回 (macd, macd_signal, macd_hist)。与 stockstats 的 macd/macds/macdh 对齐。"""
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    dif = ema_fast - ema_slow
    dea = ema(dif, signal)
    hist = (dif - dea) * 2  # stockstats 的 macdh 是 2*(DIF-DEA)
    return dif, dea, hist


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    r = 100 - 100 / (1 + rs)
    return r.fillna(50.0)


def bollinger(close: pd.Series, window: int = 20, num_std: float = 2.0):
    """返回 (middle, upper, lower)。"""
    mid = sma(close, window)
    std = close.rolling(window=window).std(ddof=0)
    upper = mid + num_std * std
    lower = mid - num_std * std
    return mid, upper, lower


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / window, adjust=False).mean()


def vwma(df: pd.DataFrame, window: int = 20) -> pd.Series:
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    vol = df["Volume"]
    return (tp * vol).rolling(window=window).sum() / vol.rolling(window=window).sum()


def mfi(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Money Flow Index。"""
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    raw_mf = tp * df["Volume"]
    pos_mf = raw_mf.where(tp > tp.shift(1), 0.0)
    neg_mf = raw_mf.where(tp < tp.shift(1), 0.0)
    pos_sum = pos_mf.rolling(window=window).sum()
    neg_sum = neg_mf.rolling(window=window).sum()
    mfr = pos_sum / neg_sum.replace(0, np.nan)
    m = 100 - 100 / (1 + mfr)
    return m.fillna(50.0)


def _last(series: pd.Series) -> Optional[float]:
    if series is None or len(series) == 0:
        return None
    val = series.iloc[-1]
    if pd.isna(val):
        return None
    return float(val)


def compute_snapshot(
    symbol: str, interval: str, candles: list[Candle]
) -> TechnicalSnapshot:
    """对单个周期 K 线计算全套指标，返回最新值快照。"""
    df = candles_to_df(candles)
    if df.empty:
        raise ValueError(f"{symbol} {interval} 无K线数据")

    close = df["Close"]
    mid, upper, lower = bollinger(close)
    dif, dea, hist = macd(close)
    r = rsi(close)
    at = atr(df)
    vw = vwma(df)
    m = mfi(df)

    # 首根到最后K线的涨跌幅（不含当前未走完的一根影响过大时仅作参考）
    price_change_pct = 0.0
    if len(close) >= 2:
        price_change_pct = float((close.iloc[-1] / close.iloc[0] - 1) * 100)

    return TechnicalSnapshot(
        symbol=symbol,
        interval=interval,
        last_price=float(close.iloc[-1]),
        close_50_sma=_last(sma(close, 50)),
        close_200_sma=_last(sma(close, 200)),
        close_10_ema=_last(ema(close, 10)),
        macd=_last(dif),
        macds=_last(dea),
        macdh=_last(hist),
        rsi=_last(r),
        boll=_last(mid),
        boll_ub=_last(upper),
        boll_lb=_last(lower),
        atr=_last(at),
        vwma=_last(vw),
        mfi=_last(m),
        price_change_pct=price_change_pct,
    )


def format_snapshot_markdown(snapshots: list[TechnicalSnapshot]) -> str:
    """将多个周期的指标快照格式化为 markdown 表格（供 LLM 上下文）。"""
    if not snapshots:
        return ""
    lines = [f"# 技术指标快照：{snapshots[0].symbol}", ""]
    for snap in snapshots:
        lines.append(f"## 周期 {snap.interval}（最新收盘价 {snap.last_price:.6g}，"
                     f"区间涨跌 {snap.price_change_pct:+.2f}%）")
        rows = [
            ("close_50_sma", snap.close_50_sma, "50周期SMA"),
            ("close_200_sma", snap.close_200_sma, "200周期SMA"),
            ("close_10_ema", snap.close_10_ema, "10周期EMA"),
            ("macd", snap.macd, "MACD"),
            ("macds", snap.macds, "MACD信号"),
            ("macdh", snap.macdh, "MACD柱"),
            ("rsi", snap.rsi, "RSI(14)"),
            ("boll", snap.boll, "布林中轨"),
            ("boll_ub", snap.boll_ub, "布林上轨"),
            ("boll_lb", snap.boll_lb, "布林下轨"),
            ("atr", snap.atr, "ATR(14)"),
            ("vwma", snap.vwma, "VWMA(20)"),
            ("mfi", snap.mfi, "MFI(14)"),
        ]
        lines.append("| 指标 | 值 | 说明 |")
        lines.append("|---|---|---|")
        for name, val, desc in rows:
            val_str = f"{val:.6g}" if val is not None else "N/A"
            lines.append(f"| {name} | {val_str} | {desc} |")
        lines.append("")
    return "\n".join(lines)


def compute_ohlcv_csv(candles: list[Candle]) -> str:
    """将K线输出为 CSV 字符串（与 TradingAgents get_stock_data 输出风格一致）。"""
    df = candles_to_df(candles)
    return df.round(6).to_csv(index=False)
