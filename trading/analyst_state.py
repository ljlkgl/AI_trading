"""分析师状态库（AnalystStateStore）。

使市场分析师「有状态」：把上一轮的逐标观点（方向/支撑/压力/入场区/目标/止损/分化）
持久化，下一轮开始前以可读摘要注入分析师提示词，让它知道自己上轮说了什么；
分析结束返回结构化结果后再写回，供下一轮对比。跨越"报告永远积极、决策永远不动"的割裂，
并在翻转方向时记日志，便于发现入场级误判。

存储位置：state/analyst_state.json
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class AnalystStateStore:
    """上一轮分析师结构化观点的持久化存储。"""

    def __init__(self, path: Optional[Path | str] = None) -> None:
        self.path = Path(path) if path else (
            Path(__file__).resolve().parent.parent / "state" / "analyst_state.json"
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict = self._load()

    def _load(self) -> dict:
        if not self.path.exists():
            return {}
        try:
            data = json.load(self.path.open("r", encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("分析师状态文件损坏，重新初始化: %s", exc)
            return {}

    def save(self) -> None:
        # 紧凑 JSON（无缩进/空格），占用存储较少
        self.path.open("w", encoding="utf-8").write(
            json.dumps(self._data, ensure_ascii=False, separators=(",", ":"))
        )

    # ---------- 读写上一轮观点 ----------

    def save_views(
        self,
        market_overview: str,
        assets: list[dict],
        news_pricing: list[dict],
    ) -> None:
        """保存本轮分析师结构化观点，供下一轮对比与注入。"""
        self._data["last"] = {
            "asof_utc": datetime.now(timezone.utc).isoformat(),
            "market_overview": market_overview,
            "assets": assets,
            "news_pricing": news_pricing,
        }
        self.save()

    def has_prior(self) -> bool:
        return bool(self._data.get("last"))

    def prior_assets_by_symbol(self) -> dict[str, dict]:
        """上一轮逐标观点，key=symbol。"""
        last = self._data.get("last") or {}
        return {a.get("symbol"): a for a in last.get("assets") or []}

    def prior_news_pricing(self) -> list[dict]:
        last = self._data.get("last") or {}
        return last.get("news_pricing") or []

    def format_prior_context(self) -> str:
        """把上一轮观点渲染成注入分析师提示词的摘要（无记录则空串）。"""
        last = self._data.get("last")
        if not last:
            return ""
        lines = ["# 你上一轮的判断（务必对照，标注 bias_change）", ""]
        lines.append(f"上一轮生成时间: {last.get('asof_utc', '')}")
        assets = last.get("assets") or []
        for a in assets:
            if a.get("entry_from") is not None:
                entry = f"[{a.get('entry_from')}, {a.get('entry_to')}]"
            else:
                entry = "无清晰信号"
            t2 = a.get("target_2") if a.get("target_2") is not None else "无"
            lines.append(
                f"- {a.get('symbol')} bias={a.get('bias')} 信心={a.get('confidence')}"
                f" 支撑=[{a.get('support_low')}, {a.get('support_high')}]"
                f" 压力=[{a.get('resistance_low')}, {a.get('resistance_high')}]"
                f" 入场={entry}"
                f" T1目标={a.get('target_1')} T2目标={t2} 止损={a.get('stop_price')}"
            )
        np_ = last.get("news_pricing") or []
        if np_:
            lines.append("上一轮新闻定价:")
            for n in np_:
                lines.append(
                    f"  - {n.get('headline')}: {n.get('priced_in')}"
                    f" 预计完全消化@{n.get('priced_in_by_utc')}"
                )
        lines.append("")
        lines.append(
            "请在下面每个资产的 bias_change 中如实标注相对上轮的变化（UNCHANGED/FLIPPED/"
            "FROM_NEUTRAL/TO_NEUTRAL/NEW/REINFORCED）。若方向翻转，必须在 reason 里说明驱动因素，"
            "避免无依据翻转；同时更新 support/resistance/entry/target/stop 到本轮的当前区间。"
        )
        return "\n".join(lines)

    # ---------- 确定性翻转检测（系统层面，日志记录，避免"报告永远积极"） ----------

    def detect_flips(self) -> list[dict]:
        """对比上一轮与本轮方向，返回翻转明细。供调用方记日志。"""
        curr_assets = (self._data.get("last") or {}).get("assets") or []
        out: list[dict] = []
        for a in curr_assets:
            sym = a.get("symbol")
            b = a.get("bias")
            bc = a.get("bias_change")
            # 采用模型自评为主；若缺失再按系统饱和度兜底
            if not bc or bc in ("UNCHANGED", "REINFORCED", "NEW"):
                continue
            out.append({
                "symbol": sym,
                "bias": b,
                "bias_change": bc,
                "reason": a.get("reason", "")[:160],
            })
        return out