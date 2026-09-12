"""市场分析师。

借鉴 TradingAgents 的 market_analyst + news_analyst 提示词逻辑：
基于多周期技术指标 + 新闻面，输出**结构化**分析报告（含逐标的支撑/压力区间、
入场/目标/止损、分化可能性，以及新闻定价的失效时间）。

相比早期版本，关键改进——分析师「有状态」：
- 上一轮观点经 AnalystStateStore 注入本轮的 user 上下文，模型必须对照并标注 bias_change，
  避免"每轮像第一次看市场"导致判断漂移/重复翻转；
- 结构化为可被系统读取的 JSON，供决策者精确消费（而非仅自由文本），
  并便于系统在方向翻转时记日志，暴露可能的入场级误判。
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from agents.llm import LLMClient
from agents.schemas import AnalystOutput

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a professional crypto perpetual futures market analyst.
You will be given a multi-timeframe technical indicator snapshot (15m/1h/4h/1d) with Bollinger
Bands, moving averages and K-line structure, recent news, current mark prices, AND your own
prior-round view (when available).

Your job: produce a NUANCED, ACTIONABLE analysis as structured JSON. Be specific and cite
actual indicator values and news headlines. Do NOT invent data not present.

For EVERY asset output:
- bias: LONG / SHORT / NEUTRAL, with confidence and the reason.
  confidence MUST be EXACTLY one of three tiers — use these rigid definitions:
  HIGH = trend + momentum + structure all clearly aligned, price near your ideal entry level,
         high conviction you can enter near support/resistance without chasing;
  MEDIUM = some conflict between timeframes or between signals, or price already moved away
         from the ideal entry (would require a chase) — direction still favored but not clean;
  LOW = weak / mixed signals, or you are not sure the level will hold — direction is a lean only.
  Do NOT invent other labels like "moderate"/"very high"/"strong". Keep the exact word then a
  brief reason, e.g. "confidence": "HIGH: ...".
- bias_change: relative to your previous-round view (UNCHANGED / FLIPPED / FROM_NEUTRAL /
  TO_NEUTRAL / NEW / REINFORCED). If you flip direction, you MUST explain the driver in reason;
  do not flip without a concrete reason.
- support_low/support_high: PRIMARY reference = Bollinger Bands and moving averages (e.g. the
  4h Bollinger lower/middle band, key SMA/EMA), confirmed by nearby K-line prior swing highs/lows
  (前高/前低). The indicator-established zones take precedence; K-line structure helps refine
  the zone width and confirm whether a level is "strong".
- entry_from/entry_to: The entry zone.
  PRIMARY: Around the prior swing high/low (breakout retest) — use LIMIT orders here.
  SECONDARY: If price is in the "mid-range" (between the prior swing point and current price)
  but shows a CLEAR REVERSAL PATTERN (e.g., 1h engulfing, hammer, pin bar, or RSI divergence),
  you may offer a small-entry zone there. Label this as "mid-range bounce" with a note
  suggesting reduced position sizing. If no clear entry signal exists, set null.
- stop_price: Place the stop order at the level that invalidates the entry thesis.
  For primary retest: just outside the structural swing point.
  For mid-range bounce: just outside the reversal candle's extreme.
- target_1/target_2: target_1 is the first take-profit (T1, close 50% there). target_2 is the
  second take-profit (T2, close 30% more there) — may be null if there is no clear second target.
  For LONG, targets must be beyond resistance; for SHORT, below support.
- divergence: give the two-sided outlook — explicitly state the observable/quantifiable condition
  under which the view flips (e.g. "if price closes above the resistance_high on rising volume,
  trend continues and target is raised; if it breaks below support_low with expanding volume, the
  move is a fake-out and bias turns short/neutral"). No vague filler.

For news (only news that matters to trading direction):
- headline, asset, event_time_utc, impact_direction (BULLISH/BEARISH/NEUTRAL).
- priced_in: Not / Partially / Fully priced in.
- remaining_space: if not fully priced, the unconsumed up/down space (e.g. "1-2% remaining").
- priced_in_by_utc: the absolute UTC time when you estimate the market will FULLY digest this event.
  Be concrete to the minute (e.g. 2026-08-18T12:30:00Z).
- window_hours: hours from now until priced_in_by_utc (0 if already past / negligible).
  This tells the decision maker whether "unpriced remaining space" is still actionable THIS round.

Finally: market_overview — a 1-3 sentence overall stance on BTC/ETH/SOL and what (if anything)
the desk should do next, respecting the standing rule of entering at support/resistance levels
and never chasing.

Output ONLY valid JSON matching this schema (no markdown fences):
{
  "market_overview": "...",
  "assets": [
    {
      "symbol": "BTCUSDT", "bias": "LONG", "confidence": "medium: ...",
      "bias_change": "UNCHANGED", "reason": "...",
      "support_low": 63000.0, "support_high": 64000.0,
      "resistance_low": 64600.0, "resistance_high": 65000.0,
      "entry_from": 63000.0, "entry_to": 64000.0,
      "target_1": 65000.0, "target_2": 66000.0,
      "stop_price": 62700.0,
      "divergence": "...", "asof_utc": "..."
    }
  ],
  "news_pricing": [
    {
      "headline": "...", "asset": "BTCUSDT", "event_time_utc": "...",
      "impact_direction": "BULLISH", "priced_in": "Partially priced in",
      "remaining_space": "1-2% remaining",
      "priced_in_by_utc": "2026-08-18T12:30:00Z",
      "window_hours": 1.2, "note": "..."
    }
  ]
}
"""


class MarketAnalyst:
    """市场技术分析师（含新闻面），结构化输出 + 跨轮状态。"""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def analyze(
        self,
        market_context: str = "",
        news_context: str = "",
        prior_context: str = "",
        current_prices: dict[str, float] | None = None,
    ) -> AnalystOutput:
        """返回结构化分析，供系统精确消费并持久化状态。

        market_context：多周期技术指标 + K线快照（布林/均线结构，主输入）。
        prior_context：上一轮观点的摘要（由 AnalystStateStore 渲染），用于对照标注 bias_change。
        current_prices：{symbol: mark_price}，供模型结合现价判断入场/追高。
        """
        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        user_parts = [f"当前时间（UTC）：{now_utc}\n\n"]
        sym_str = ", ".join(sorted(current_prices.keys())) if current_prices else ""
        if market_context and market_context.strip():
            user_parts.append(
                f"以下是 {sym_str or '本账户标的'} 的多周期技术指标与K线数据（布林带/均线/K线结构），"
                "请基于此输出结构化分析：\n\n"
            )
            user_parts.append(market_context)
        else:
            user_parts.append("（未提供行情数据，请根据已有知识分析）")
        if current_prices:
            px = "; ".join(f"{s}={v:.4g}" for s, v in current_prices.items())
            user_parts.append(f"\n\n当前标记价格：{px}\n")
        if news_context and news_context.strip():
            user_parts.extend(["\n\n新闻面信息：\n\n", news_context])
        if prior_context and prior_context.strip():
            user_parts.extend(["\n\n上一轮你自己的判断（请对照并如实标注 bias_change）：\n\n", prior_context])
        user_parts.append(
            f"\n\n请输出结构化分析 JSON："
            f"为每个资产标定 bias_change、基于布林带/均线/K线结构的支撑/压力区间、"
            f"入场区间（主回踩+辅山腰反弹）、target_1（50%止盈）/ target_2（再30%止盈）/ 止损价、"
            f"分化可能，以及各条相关新闻的定价状态与预计完全消化的时间点 priced_in_by_utc。"
        )
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": "".join(user_parts)},
        ]
        data = self.llm.chat_json(messages, temperature=0.3, label=f"市场分析师 {self.llm.model}")
        try:
            return AnalystOutput.model_validate(data)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "分析师输出校验失败: %s\n原始: %s",
                exc, json.dumps(data, ensure_ascii=False)[:800],
            )
            raise