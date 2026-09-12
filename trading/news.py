"""新闻数据服务。

提取自 TradingAgents 的 yfinance_news 逻辑：
- 标的新闻：yf.Ticker("<BASE>-USD").get_news()（加密货币用 -USD 形式）
- 宏观新闻：yf.Search(query, news_count) 按搜索词抓取
对币安交易对（BTCUSDT → BTC-USD）做符号映射。
"""
from __future__ import annotations

import contextlib
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# 币安交易对 → Yahoo 符号：去掉 USDT/USDC 后缀改为 -USD
_YAHOO_QUOTES = ("USDT", "USDC", "USD")


def to_yahoo_symbol(binance_symbol: str) -> str:
    """BTCUSDT -> BTC-USD, ETHUSDT -> ETH-USD, SOLUSDT -> SOL-USD。"""
    s = binance_symbol.strip().upper()
    for q in _YAHOO_QUOTES:
        if s.endswith(q):
            return f"{s[:-len(q)]}-USD"
    return s


class NewsService:
    """抓取标的与宏观新闻，输出 markdown 上下文。"""

    def __init__(self, article_limit: int = 15, global_limit: int = 15) -> None:
        self.article_limit = article_limit
        self.global_limit = global_limit
        self._yf = None

    @property
    def yf(self):
        """惰性导入 yfinance，避免未安装时报错。"""
        if self._yf is None:
            import yfinance as yf
            self._yf = yf
        return self._yf

    def get_symbol_news(self, binance_symbol: str, days: int = 3) -> str:
        """获取单个交易对的近期新闻。"""
        yahoo = to_yahoo_symbol(binance_symbol)
        try:
            stock = self.yf.Ticker(yahoo)
            news = stock.get_news(count=self.article_limit)
        except Exception as exc:  # noqa: BLE001
            logger.warning("获取 %s 新闻失败: %s", binance_symbol, exc)
            return f"（%s 新闻获取失败: %s）" % (binance_symbol, exc)

        if not news:
            return f"（{binance_symbol} 近期无新闻）"

        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        lines = [f"## {binance_symbol} 近期新闻（最近 {days} 天）"]
        kept = 0
        for article in news:
            data = self._extract_article(article)
            if data["pub_date"] is None or data["pub_date"] < cutoff:
                continue
            lines.append(f"- [{data['pub_date'].strftime('%m-%d %H:%M')}] {data['title']}（{data['publisher']}）")
            if data["summary"]:
                lines.append(f"  摘要: {data['summary'][:200]}")
            kept += 1
            if kept >= self.article_limit:
                break
        if kept == 0:
            return f"（{binance_symbol} 近 {days} 天无新闻）"
        return "\n".join(lines)

    def get_global_news(
        self, queries: Optional[list[str]] = None, days: int = 3
    ) -> str:
        """获取宏观加密市场新闻。"""
        queries = queries or [
            "Bitcoin price",
            "Ethereum crypto market",
            "crypto regulation SEC",
            "Federal Reserve interest rates",
            "Solana cryptocurrency",
            # 地缘政治 / 宏观环境关键词：美国政策与地缘冲突都会影响加密市场
            "America"
        ]
        all_news: list[dict] = []
        seen: set[str] = set()
        for query in queries:
            try:
                search = self.yf.Search(
                    query=query,
                    news_count=self.global_limit,
                    enable_fuzzy_query=True,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("搜索新闻失败 query=%s: %s", query, exc)
                continue
            for article in search.news or []:
                data = self._extract_article(article)
                if data["title"] and data["title"] not in seen:
                    seen.add(data["title"])
                    all_news.append(data)

        if not all_news:
            return "（未获取到宏观新闻）"

        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        lines = [f"## 宏观市场新闻（最近 {days} 天）"]
        kept = 0
        for data in sorted(all_news, key=lambda x: x["pub_date"] or datetime.min, reverse=True):
            if data["pub_date"] is not None and data["pub_date"] < cutoff:
                continue
            lines.append(f"- [{data['pub_date'].strftime('%m-%d %H:%M')}] {data['title']}（{data['publisher']}）")
            if data["summary"]:
                lines.append(f"  摘要: {data['summary'][:200]}")
            kept += 1
            if kept >= self.global_limit:
                break
        if kept == 0:
            return "（近几天无宏观新闻）"
        return "\n".join(lines)

    def build_news_context(self, symbols: list[str]) -> str:
        """组合所有标的 + 宏观新闻为一个上下文。"""
        blocks = ["# 新闻面信息"]
        for sym in symbols:
            blocks.append(self.get_symbol_news(sym))
        blocks.append(self.get_global_news())
        return "\n\n".join(blocks)

    @staticmethod
    def _extract_article(article: dict) -> dict:
        """从 yfinance 新闻格式中提取字段（兼容嵌套/扁平结构）。"""
        pub_date = None
        if "content" in article:
            content = article["content"]
            title = content.get("title", "No title")
            summary = content.get("summary", "")
            provider = content.get("provider", {})
            publisher = provider.get("displayName", "Unknown")
            pub_str = content.get("pubDate", "")
            if pub_str:
                with contextlib.suppress(ValueError, AttributeError):
                    pub_date = datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
        else:
            title = article.get("title", "No title")
            summary = article.get("summary", "")
            publisher = article.get("publisher", "Unknown")
            ts = article.get("providerPublishTime")
            if ts:
                with contextlib.suppress(ValueError, OSError, TypeError):
                    pub_date = datetime.fromtimestamp(ts, tz=timezone.utc)
        if pub_date is not None and pub_date.tzinfo is None:
            pub_date = pub_date.replace(tzinfo=timezone.utc)
        return {
            "title": title,
            "summary": summary,
            "publisher": publisher,
            "pub_date": pub_date,
        }
