"""唤醒条件存储（WatchStore）。

模型可在每轮决策中输出 watch_conditions（如 BTC 价格触及某价位时唤醒），
系统在正常循环等待期间按间隔轮询价格，任一条件满足即提前触发一轮分析。
条件为一次性：触发后清除，由下一轮决策重新设定（全量替换）。

存储位置：state/watch_triggers.json
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from agents.schemas import WakeCondition

logger = logging.getLogger(__name__)


class WatchStore:
    """唤醒条件的持久化存储。"""

    def __init__(
        self,
        path: Optional[Path] = None,
        max_age_hours: int = 24,
    ) -> None:
        self.path = path or (
            Path(__file__).resolve().parent.parent / "state" / "watch_triggers.json"
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_age_hours = max_age_hours
        self._data: list[dict] = self._load()

    def _load(self) -> list[dict]:
        if not self.path.exists():
            return []
        try:
            data = json.load(self.path.open("r", encoding="utf-8"))
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("唤醒条件文件损坏，重新初始化: %s", exc)
            return []

    def save(self) -> None:
        # 紧凑 JSON（无缩进/空格），占用存储较少
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, separators=(",", ":"))

    def replace(self, conditions: list[WakeCondition]) -> None:
        """用模型本轮输出的条件全量替换旧条件（空列表=清除全部）。"""
        now = datetime.now(timezone.utc)
        expires = (now + timedelta(hours=self.max_age_hours)).isoformat()
        self._data = [
            {**c.model_dump(), "created_at": now.isoformat(), "expires_at": expires}
            for c in conditions
        ]
        self.save()
        if conditions:
            logger.info(
                "已设置 %d 个唤醒条件（有效期 %dh）: %s",
                len(conditions), self.max_age_hours,
                ", ".join(
                    f"{c.symbol} {c.condition}@{c.value:g}" for c in conditions
                ),
            )
        else:
            logger.info("唤醒条件已清除（模型输出空列表）")

    def all(self) -> list[dict]:
        """返回有效条件，并清理已过期的。"""
        now = datetime.now(timezone.utc)
        fresh = []
        for t in self._data:
            expires = t.get("expires_at", "")
            if not expires or datetime.fromisoformat(expires) >= now:
                fresh.append(t)
        if len(fresh) != len(self._data):
            self._data = fresh
            self.save()
        return fresh

    def clear_all(self, reason: str = "") -> None:
        if self._data:
            logger.info("清除全部唤醒条件%s", f"（{reason}）" if reason else "")
            self._data = []
            self.save()

    def count(self) -> int:
        return len(self.all())
