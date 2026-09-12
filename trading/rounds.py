"""轮次历史记录（RoundLog）。

每轮交易结束后将 result 精简后追加保存，供 HTTP 页面展示历史交易情况。
为节省存储：
- 只保留关键字段（账户/持仓/挂单/指令/执行结果/决策评估/风控拦截等），
  丢弃 market_report、thesis_context、news 等大文本；
- 紧凑 JSON（无缩进/空格）；
- 只保留最近 ROUNDS_KEEP 轮，避免文件无限增长。

存储位置：state/rounds.json
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# 只保留最近多少轮
ROUNDS_KEEP = 50


class RoundLog:
    """轮次历史记录的持久化存储（追加 + 截断 + 紧凑 JSON）。"""

    def __init__(self, path: Optional[Path] = None, keep: int = ROUNDS_KEEP) -> None:
        self.path = path or (Path(__file__).resolve().parent.parent / "state" / "rounds.json")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.keep = keep
        self._data: list[dict] = self._load()

    def _load(self) -> list[dict]:
        if not self.path.exists():
            return []
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("轮次历史文件损坏，重新初始化: %s", exc)
            return []

    def save(self) -> None:
        # 紧凑 JSON（无缩进/空格），占用存储较少
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, separators=(",", ":"))

    def append(self, result: dict[str, Any]) -> None:
        """追加一轮精简记录，并截断到最近 keep 轮。"""
        summary = self._summarize(result)
        self._data.append(summary)
        if len(self._data) > self.keep:
            self._data = self._data[-self.keep:]
        self.save()

    def all(self) -> list[dict]:
        return list(self._data)

    def count(self) -> int:
        return len(self._data)

    @staticmethod
    def _summarize(result: dict[str, Any]) -> dict[str, Any]:
        """从完整 result 中抽取关键字段，丢弃大文本以节省存储。"""
        keep_keys = (
            "timestamp", "symbols", "testnet", "dry_run",
            "account", "open_orders", "min_margin_context",
            "market_assessment", "risk_notes",
            "thesis_ops", "watch_conditions", "risk_blocked",
            "instructions_after_risk", "confirmations", "execution",
            "thesis_count", "reflection", "error", "analyst_state",
        )
        return {k: result.get(k) for k in keep_keys if k in result}


class RuntimeState:
    """运行时状态持久化（state/state.json），支持程序关闭再开启的恢复。

    核心字段 last_round_at：上一轮分析开始时刻（epoch 秒）。
    程序重启后据此计算距下一轮分析的剩余时间：若尚未到点则「接上等待」，
    而不是立即重新分析，从而延续修改前的决定与节奏。
    交易相关状态（挂单/持仓/操作理由列表/唤醒条件/轮次历史）本就各自持久化
    或存于交易所，重启后按真实账户自动对齐，天然兼容修改前的决定。
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = path or (
            Path(__file__).resolve().parent.parent / "state" / "state.json"
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("运行时状态文件损坏，重新初始化: %s", exc)
            return {}

    def save(self) -> None:
        # 紧凑 JSON（无缩进/空格），占用存储较少
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, separators=(",", ":"))

    def mark_round_started(self, now: Optional[float] = None) -> None:
        """记录本轮分析开始时刻并持久化（中途崩溃后下次启动也能接续节奏）。"""
        self._data["last_round_at"] = now if now is not None else time.time()
        self.save()

    def last_round_at(self) -> Optional[float]:
        """上一轮分析开始时刻（epoch 秒），无记录返回 None。"""
        return self._data.get("last_round_at")

    def set_feedback(self, text: str) -> None:
        """记录上一轮执行反馈（风控拦截 / 执行失败原因等），供下一轮决策者参考。

        持久化到本地：系统重启后仍保留，避免模型因不知道上轮指令为何未成交
        而重复犯错（例如开仓因保证金不足被拦截后，下轮仍按同样 margin 下单）。
        """
        if text:
            self._data["last_feedback"] = text
        else:
            self._data.pop("last_feedback", None)
        self.save()

    def last_feedback(self) -> str:
        """上一轮执行反馈文本；无记录返回空串。"""
        return self._data.get("last_feedback", "")
