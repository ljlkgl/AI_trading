"""自主经验库（Experience Library）。

对应 TradingAgents 的 TradingMemoryLog（复盘记忆）机制：
模型在严重亏损或发现自身不足时，可自主写入、修改、删除经验条目，
供后续轮次的决策参考。

存储位置：state/experience_library.json
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class ExperienceStore:
    """自主经验库的持久化存储。"""

    def __init__(self, path: Optional[Path] = None, max_items: Optional[int] = None) -> None:
        self.path = path or (
            Path(__file__).resolve().parent.parent / "state" / "experience_library.json"
        )
        # 条目硬上限：超出淘汰最旧，防止文件/内存无限增长
        self.max_items = max_items
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, dict] = self._load()
        # 启动时若已有状态文件超限，立即收敛（避免历史遗留文件一次性超量）
        if self._enforce_max():
            self.save()

    def _load(self) -> dict[str, dict]:
        if not self.path.exists():
            return {}
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("经验库文件损坏，重新初始化: %s", exc)
            return {}

    def save(self) -> None:
        # 紧凑 JSON（无缩进/空格），占用存储较少
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, separators=(",", ":"))

    # ---------- 增删改查 ----------

    def add(
        self,
        category: str,
        title: str,
        content: str,
    ) -> str:
        """新增一条经验，返回经验 id。"""
        eid = uuid.uuid4().hex[:12]
        now = datetime.now().isoformat()
        self._data[eid] = {
            "id": eid,
            "category": category,
            "title": title,
            "content": content,
            "created_at": now,
            "updated_at": now,
        }
        self._enforce_max()
        self.save()
        logger.info("经验库新增 #%s [%s] %s", eid, category, title)
        return eid

    def _enforce_max(self) -> int:
        """条目数超上限时淘汰最旧条目（刚写入的最新一条不淘汰），返回淘汰数量。"""
        if not self.max_items or len(self._data) <= self.max_items:
            return 0
        oldest_keys = sorted(
            self._data, key=lambda k: self._data[k].get("updated_at", "")
        )
        removed = 0
        for k in oldest_keys:
            if len(self._data) <= self.max_items:
                break
            self._data.pop(k, None)
            removed += 1
        if removed:
            logger.warning(
                "经验库超上限 %d 条，已淘汰 %d 条最旧条目", self.max_items, removed
            )
        return removed

    def update(self, eid: str, **fields) -> bool:
        """按字段更新经验（category/title/content 可改），返回是否成功。"""
        if eid not in self._data:
            logger.warning("经验库更新失败：id=%s 不存在", eid)
            return False
        for key in ("category", "title", "content"):
            if key in fields and fields[key] is not None:
                self._data[eid][key] = str(fields[key])
        self._data[eid]["updated_at"] = datetime.now().isoformat()
        self.save()
        logger.info("经验库更新 #%s", eid)
        return True

    def delete(self, eid: str) -> bool:
        """删除一条经验，返回是否成功。"""
        if eid not in self._data:
            logger.warning("经验库删除失败：id=%s 不存在", eid)
            return False
        del self._data[eid]
        self.save()
        logger.info("经验库删除 #%s", eid)
        return True

    def get(self, eid: str) -> Optional[dict]:
        return self._data.get(eid)

    def all(self) -> dict[str, dict]:
        return dict(self._data)

    def count(self) -> int:
        return len(self._data)

    # ---------- 上下文格式化 ----------

    def format_for_context(self, limit: int = 15) -> str:
        """将经验库格式化为 markdown 上下文（供决策者/反思者参考）。"""
        items = list(self._data.values())
        if not items:
            return "# 自主经验库\n\n（经验库为空，暂无历史经验参考）"
        # 按更新时间倒序
        items.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        lines = ["# 自主经验库（历史经验，供本轮参考）"]
        for item in items[:limit]:
            lines.append("")
            lines.append(
                f"- [id:{item['id']}] [{item.get('category', '未分类')}] "
                f"{item.get('title', '')}（更新于 {item.get('updated_at', '')[:16]}）"
            )
            content = item.get("content", "")
            if content:
                lines.append(f"  内容: {content[:300]}")
        if len(items) > limit:
            lines.append(f"\n（共 {len(items)} 条，仅展示最近 {limit} 条）")
        return "\n".join(lines)
