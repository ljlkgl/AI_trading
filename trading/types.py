"""共享数据类型定义。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Candle:
    """K线数据（币安 klines 一行）。"""

    open_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: int
    quote_volume: float


@dataclass
class Position:
    """永续合约持仓。"""

    symbol: str
    position_side: str          # LONG / SHORT（单向持仓模式）
    position_amt: float         # 持仓数量（正=多，负=空）
    entry_price: float
    mark_price: float
    unrealized_pnl: float
    leverage: float
    isolated_margin: float = 0.0
    liquidation_price: float = 0.0


@dataclass
class AccountInfo:
    """账户快照。"""

    total_balance: float
    available_balance: float
    unrealized_pnl: float
    margin_balance: float
    positions: list[Position] = field(default_factory=list)


@dataclass
class SymbolInfo:
    """交易对信息（精度等）。"""

    symbol: str
    price_precision: int
    qty_precision: int
    min_qty: float
    min_notional: float
    price_tick: float
    qty_step: float


@dataclass
class TechnicalSnapshot:
    """单币种、单周期的技术指标快照。"""

    symbol: str
    interval: str
    last_price: float
    close_50_sma: Optional[float] = None
    close_200_sma: Optional[float] = None
    close_10_ema: Optional[float] = None
    macd: Optional[float] = None
    macds: Optional[float] = None
    macdh: Optional[float] = None
    rsi: Optional[float] = None
    boll: Optional[float] = None
    boll_ub: Optional[float] = None
    boll_lb: Optional[float] = None
    atr: Optional[float] = None
    vwma: Optional[float] = None
    mfi: Optional[float] = None
    price_change_pct: float = 0.0  # 该周期首根到当前K线的涨跌幅
