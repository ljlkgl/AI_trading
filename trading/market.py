"""市场数据服务：负责拉取行情、计算指标并格式化上下文。

对应 TradingAgents 中 get_stock_data / get_indicators /
get_verified_market_snapshot 三个工具职责，统一封装为一次调用。
"""
from __future__ import annotations

import logging
from typing import Optional

from trading.binance_client import BinanceClient
from trading.indicators import (
    candles_to_df,
    compute_snapshot,
    format_snapshot_markdown,
)
from trading.types import Candle, TechnicalSnapshot

logger = logging.getLogger(__name__)

# 多周期分析
DEFAULT_INTERVALS = ("15m", "1h", "4h", "1d")


class MarketDataService:
    """组合行情 + 指标，输出给分析师的上下文。"""

    def __init__(self, client: BinanceClient) -> None:
        self.client = client

    def fetch_klines(self, symbol: str, interval: str, limit: int) -> list[Candle]:
        return self.client.get_klines(symbol, interval, limit)

    def build_market_context(
        self,
        symbol: str,
        intervals: tuple[str, ...] = DEFAULT_INTERVALS,
        limit: int = 500,
    ) -> tuple[str, dict[str, list[TechnicalSnapshot]]]:
        """抓取多周期K线并计算指标，返回 (markdown 上下文, 快照字典)。"""
        snapshots_by_interval: dict[str, list[TechnicalSnapshot]] = {}
        lines: list[str] = [f"# {symbol} 永续合约行情与指标", ""]

        for interval in intervals:
            candles = self.fetch_klines(symbol, interval, limit)
            if not candles:
                lines.append(f"## {interval}: 无数据")
                continue
            snap = compute_snapshot(symbol, interval, candles)
            snapshots_by_interval.setdefault(interval, []).append(snap)
            lines.append(f"## 周期 {interval}（共 {len(candles)} 根K线，最新价 {snap.last_price:.6g}）")
            lines.append(format_snapshot_markdown([snap]))
            # 最近 5 根K线的原始 OHLCV（供模型参考近期价格行为）
            df = candles_to_df(candles)
            tail = df.tail(5).copy()
            # 仅对数值列取整（Date 为 datetime 类型，round 会告警）
            numeric_cols = tail.select_dtypes(include="number").columns
            tail[numeric_cols] = tail[numeric_cols].round(6)
            lines.append("最近5根K线(Open/High/Low/Close/Volume):")
            lines.append(tail.to_csv(index=False))
            lines.append("")

        # 24h 行情与资金费率
        try:
            t24 = self.client.get_24hr_ticker(symbol)
            lines.append("## 24小时行情")
            lines.append(
                f"- 最新价: {t24.get('lastPrice')}  涨跌幅: {t24.get('priceChangePercent')}%"
            )
            lines.append(
                f"- 最高: {t24.get('highPrice')}  最低: {t24.get('lowPrice')}"
            )
            lines.append(f"- 成交量: {t24.get('volume')} 合约  成交额: {t24.get('quoteVolume')} USDT")
        except Exception as exc:  # noqa: BLE001
            logger.warning("获取 %s 24h 行情失败: %s", symbol, exc)

        try:
            fr = self.client.get_funding_rate(symbol, limit=5)
            if fr:
                latest = fr[-1]
                lines.append("## 资金费率(最近5次)")
                for item in fr:
                    lines.append(
                        f"- {item['fundingTime']}: {item['fundingRate']}"
                    )
                lines.append(f"最新资金费率: {latest.get('fundingRate')}")
        except Exception as exc:  # noqa: BLE001
            logger.warning("获取 %s 资金费率失败: %s", symbol, exc)

        return "\n".join(lines), snapshots_by_interval

    def build_market_context_for_symbols(
        self, symbols: list[str], limit: int = 500
    ) -> str:
        """为多个币种构建整体市场上下文。"""
        blocks = []
        for sym in symbols:
            try:
                ctx, _ = self.build_market_context(sym, limit=limit)
                blocks.append(ctx)
            except Exception as exc:  # noqa: BLE001
                logger.warning("构建 %s 市场上下文失败: %s", sym, exc)
                blocks.append(f"# {sym}\n\n数据获取失败: {exc}")
        return "\n\n".join(blocks)
