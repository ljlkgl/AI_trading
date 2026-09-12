"""操作理由（Thesis）列表存储。

维护一个「当前进行中的操作理由列表」：列表中的每条记录对应一个当前正在进行的
操作（开仓持仓 / 限价挂单 / 其它执行动作）及其完整理由。

生命周期：
- 启动某个操作（开仓 / 挂单）时 ADD 一条记录；
- 操作仍在进行（仓位未平完 / 挂单未成交或未撤销）时，记录保留；
- 操作周期结束（仓位被完全平掉 / 挂单撤销且不再续挂）时 DELETE 该记录。

模型拥有对该列表的完整操作权（ADD / UPDATE / DELETE）；系统也会在每轮结束
根据真实账户状态自动清理过期条目，防止上下文过大。

存储位置：state/theses.json
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# 允许的 thesis 记录字段
_VALID_FIELDS = {
    "symbol", "kind", "direction", "entry_price", "stop_loss",
    "take_profit", "thesis", "note", "parent_id", "order_id",
    "opened_at", "updated_at",
}


class ThesisStore:
    """操作理由列表的持久化存储（增/改/删 + 过期清理）。"""

    def __init__(
        self,
        path: Optional[Path] = None,
        max_age_hours: Optional[int] = None,
        max_items: Optional[int] = None,
    ) -> None:
        self.path = path or (Path(__file__).resolve().parent.parent / "state" / "theses.json")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # 无 max_age_hours 时不按时间自动过期（由账户状态清理兜底）
        self.max_age_hours = max_age_hours
        # 条目硬上限：超出按「非持仓优先、最旧优先」淘汰，保证上下文/存储有界
        self.max_items = max_items
        self._data: list[dict] = self._load()
        # 启动时若已有状态文件超限，立即收敛（避免历史遗留文件一次性超量进上下文）
        if self._enforce_max():
            self.save()
        self._seq = self._next_seq()

    # ---------- 基础存取 ----------

    def _load(self) -> list[dict]:
        if not self.path.exists():
            return []
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("理由列表状态文件损坏，重新初始化: %s", exc)
            return []

    def save(self) -> None:
        # 紧凑 JSON（无缩进/空格），占用存储较少
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, separators=(",", ":"))

    def _next_seq(self) -> int:
        n = 0
        for r in self._data:
            tid = r.get("id", "")
            if tid.startswith("th_"):
                try:
                    n = max(n, int(tid.split("_")[-1]))
                except ValueError:
                    pass
        return n

    # ---------- 列表操作 ----------

    def add(self, **fields: Any) -> str:
        """新增一条操作理由记录，返回分配的 id。

        必填：symbol、thesis（理由）。其余字段（kind/direction/entry_price/
        stop_loss/take_profit/note 等）可选；parent_id 用于建立父子层级
        （如开仓=父编号、其止盈止损=各为子编号，层级可两层以上）。
        """
        symbol = fields.get("symbol")
        thesis = fields.get("thesis")
        if not symbol or not thesis:
            raise ValueError("add 必须提供 symbol 与 thesis")
        parent_id = fields.get("parent_id")
        if parent_id and not any(r.get("id") == parent_id for r in self._data):
            logger.warning("理由列表 ADD 的 parent_id=%s 不存在，忽略层级挂载", parent_id)
            parent_id = None
        self._seq += 1
        record = {
            "id": f"th_{datetime.now().strftime('%Y%m%d')}_{self._seq:04d}",
            "symbol": symbol,
            "kind": fields.get("kind", "position"),
            "direction": fields.get("direction"),
            "entry_price": fields.get("entry_price"),
            "stop_loss": fields.get("stop_loss"),
            "take_profit": fields.get("take_profit"),
            "order_id": fields.get("order_id"),
            "thesis": thesis,
            "note": fields.get("note"),
            "parent_id": parent_id,
            "opened_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }
        self._data.append(record)
        self._enforce_max()
        self.save()
        logger.info("理由列表 ADD %s (%s %s%s)", record["id"], symbol, record["kind"],
                    f" 父={parent_id}" if parent_id else "")
        return record["id"]

    def _enforce_max(self) -> int:
        """条目数超上限时淘汰，保证上下文/存储有界。

        优先淘汰「非持仓」的最旧条目（挂单/其它类相对可弃），仍超出才淘汰最旧的
        持仓条目（安全阀，正常不会触发）。刚加入的记录不淘汰，让本轮操作生效。
        被淘汰节点的全部子孙一并级联删除，与 remove 的级联语义一致。
        """
        if not self.max_items or len(self._data) <= self.max_items:
            return 0
        over = len(self._data) - self.max_items
        to_remove: set[str] = set()
        # 1) 优先淘汰非 position 的最旧条目
        removable = [r for r in self._data[:-1] if r.get("kind") != "position"]
        removable.sort(key=lambda x: x.get("updated_at", ""))
        for r in removable:
            if over <= 0:
                break
            to_remove.add(r.get("id", ""))
            over -= 1
        # 2) 仍超出则淘汰最旧 position 条目（安全阀）
        if over > 0:
            pos = [r for r in self._data[:-1] if r.get("kind") == "position"]
            pos.sort(key=lambda x: x.get("updated_at", ""))
            for r in pos:
                if over <= 0:
                    break
                to_remove.add(r.get("id", ""))
                over -= 1
        if not to_remove:
            return 0
        # 级联：被淘汰节点的全部子孙一并删除（集合收敛，天然防环）
        expanded: set[str] = set(to_remove)
        changed = True
        while changed:
            changed = False
            for r in self._data:
                if r.get("parent_id") in expanded and r.get("id") not in expanded:
                    expanded.add(r["id"])
                    changed = True
        self._data = [r for r in self._data if r.get("id") not in expanded]
        logger.warning(
            "理由列表超上限 %d 条，已淘汰 %d 条最旧/非持仓条目（含级联子孙），上下文有界",
            self.max_items, len(expanded),
        )
        return len(expanded)

    def _descendant_ids(self, thesis_id: str) -> set[str]:
        """递归收集该节点的全部子孙 id（含多层层级，防环）。"""
        ids: set[str] = set()
        frontier = [thesis_id]
        while frontier:
            cur = frontier.pop()
            for r in self._data:
                if r.get("parent_id") == cur and r.get("id") not in ids:
                    ids.add(r["id"])
                    frontier.append(r["id"])
        return ids

    def update(self, thesis_id: str, **fields: Any) -> bool:
        """修改一条已有记录；只更新传入的合法字段，返回是否成功。

        parent_id 允许调整层级，但禁止形成循环（父不能是自身或其子孙）。
        """
        for r in self._data:
            if r.get("id") == thesis_id:
                if "parent_id" in fields and fields.get("parent_id"):
                    new_pid = fields["parent_id"]
                    # 若新父编号是目标节点自身或其子孙，则会把目标挂到自己的后代下形成环
                    if new_pid == thesis_id or new_pid in self._descendant_ids(thesis_id):
                        logger.warning(
                            "理由列表 UPDATE 拒绝：parent_id=%s 会形成循环（目标 %s）",
                            new_pid, thesis_id,
                        )
                        fields.pop("parent_id")
                for k, v in fields.items():
                    if k in _VALID_FIELDS:
                        r[k] = v
                r["updated_at"] = datetime.now().isoformat()
                self.save()
                logger.info("理由列表 UPDATE %s", thesis_id)
                return True
        logger.warning("理由列表 UPDATE 失败: id=%s 不存在", thesis_id)
        return False

    def remove(self, thesis_id: str) -> bool:
        """删除一条记录及其全部子孙记录（级联），返回是否删除了目标节点。"""
        targets = self._descendant_ids(thesis_id)
        targets.add(thesis_id)
        before = len(self._data)
        self._data = [r for r in self._data if r.get("id") not in targets]
        if len(self._data) != before:
            self.save()
            logger.info(
                "理由列表 DELETE %s（级联删除 %d 条子孙）", thesis_id, len(targets) - 1
            )
            return True
        logger.warning("理由列表 DELETE 失败: id=%s 不存在", thesis_id)
        return False

    def complete(self, thesis_id: str) -> bool:
        """结束一条操作（标注完成记号）：系统自动级联删除该编号及其全部子编号的理由。"""
        return self.remove(thesis_id)

    def get(self, thesis_id: str) -> Optional[dict]:
        for r in self._data:
            if r.get("id") == thesis_id:
                return r
        return None

    def all(self) -> list[dict]:
        return list(self._data)

    def by_symbol(self, symbol: str) -> list[dict]:
        return [r for r in self._data if r.get("symbol") == symbol]

    def count(self) -> int:
        return len(self._data)

    # ---------- 过期清理 ----------

    def prune_stale(
        self,
        account_positions: dict[str, Any],
        open_orders_by_symbol: Optional[dict[str, list]] = None,
        live_open_order_ids: Optional[set] = None,
    ) -> int:
        """自动清理过期条目，返回被清除的数量（含被级联删除的子孙节点）。

        规则（按操作类型判断其操作周期是否已结束）：
        1. kind=position（持仓理由）：该币种已无任何持仓（仓位被完全平掉）→ 删除；
           若记录了绑定单号 order_id，则持仓列表(live 持仓)中必须能对应到该币种，
           否则视为已平仓 → 删除；
        2. kind=limit_order（挂单理由）：
           - 若记录绑定单号 order_id：该单号必须仍处于未成交挂单列表(live_open_order_ids)
             中，或已成交转化为持仓（存在持仓）；否则（单已撤 / 成交后又平掉）→ 删除；
           - 若未记录单号（旧数据）：按原规则——该币种已无未成交 LIMIT 挂单 且 无持仓 → 删除；
        3. kind=other / 未知：无账户状态可核对，超时（超过 max_age_hours）→ 删除，
           防止上下文无限膨胀。

        关键：这里以「绑定单号是否还存在于挂单/持仓列表」为准清除，即使模型忘记主动
        COMPLETE/DELETE，只要其绑定的币安单号已不在存活列表中，系统也会自动清理，
        防止模型自身遗忘导致理由列表残留。

        任一节点被判过期时，其全部子孙节点（父子层级）也一并级联删除。
        """
        open_orders_by_symbol = open_orders_by_symbol or {}
        live_open_order_ids = live_open_order_ids or set()
        now = datetime.now().astimezone()

        def _has_open_limit(symbol: str) -> bool:
            for orders in open_orders_by_symbol.values():
                for o in orders:
                    if o.get("symbol") == symbol and (o.get("type") == "LIMIT"):
                        return True
            return False

        def _has_position(symbol: str) -> bool:
            return symbol in account_positions

        stale_ids: set[str] = set()
        for r in self._data:
            symbol = r.get("symbol", "")
            kind = r.get("kind", "position")
            bound_id = r.get("order_id")
            stale = False

            if kind == "position":
                # 持仓理由：按「持仓列表」判断——该币种仍存有持仓即有效
                if not _has_position(symbol):
                    stale = True
            elif kind == "limit_order":
                if bound_id is not None:
                    # 有绑定单号：必须仍在该币种未成交挂单中，或由该单成交转为持仓（暂不误删）
                    if (
                        str(bound_id) not in live_open_order_ids
                        and not _has_position(symbol)
                    ):
                        stale = True
                else:
                    # 旧数据无单号：退回原 symbol 级判断
                    if not _has_open_limit(symbol) and not _has_position(symbol):
                        stale = True
            else:  # other / 未知
                if self.max_age_hours:
                    opened = r.get("opened_at")
                    if opened:
                        try:
                            t = datetime.fromisoformat(opened)
                            if t.tzinfo is None:
                                t = t.astimezone()
                            if (now - t).total_seconds() > self.max_age_hours * 3600:
                                stale = True
                        except ValueError:
                            pass

            if stale:
                stale_ids.add(r.get("id", ""))

        if not stale_ids:
            return 0

        # 级联：被删除节点的全部子孙一并删除（集合收敛，天然防环）
        expanded: set[str] = set(stale_ids)
        changed = True
        while changed:
            changed = False
            for r in self._data:
                if r.get("parent_id") in expanded and r.get("id") not in expanded:
                    expanded.add(r["id"])
                    changed = True
        removed = len(expanded)
        self._data = [r for r in self._data if r.get("id") not in expanded]
        self.save()
        for rid in stale_ids:
            logger.info("理由列表自动清理过期条目 %s", rid)
        if removed > len(stale_ids):
            logger.info("理由列表级联清理子孙 %d 条", removed - len(stale_ids))
        return removed

    # ---------- 上下文渲染 ----------

    def render_context(self) -> str:
        """渲染「操作理由列表」上下文，供决策者读取与操作。

        每条记录给出 id（供 UPDATE/COMPLETE/DELETE 引用）、类型、方向、价格、理由与备注；
        父子层级以缩进展示（如开仓=父编号、其止盈止损=各为子编号，层级可两层以上）。
        """
        lines = ["# 操作理由列表（当前进行中的操作及其理由）"]
        lines.append(
            "- 每条记录 = 一个正在进行的操作（持仓 / 挂单 / 其它）及其理由；"
            "id 供你在 thesis_ops 中引用（UPDATE/COMPLETE/DELETE）"
        )
        lines.append(
            "- 记录可有父子层级（ADD 时带 parent_id，如开仓=父编号、其止盈止损=各为子编号，"
            "层级可两层以上，缩进表示层级）；COMPLETE 一个父编号时，"
            "系统会自动级联删除其全部子编号"
        )
        lines.append(
            "- 当一个操作周期结束（仓位被完全平掉 / 挂单撤销且不再续挂）时，"
            "请通过 thesis_ops 的 COMPLETE（推荐，级联清子树）或 DELETE 删除对应条目，"
            "避免列表无限膨胀"
        )
        if not self._data:
            lines.append("\n（当前无进行中的操作理由记录）")
            return "\n".join(lines)

        # 按父子层级构建树
        children: dict[str, list[dict]] = {}
        roots: list[dict] = []
        for r in self._data:
            pid = r.get("parent_id")
            if pid and pid != r.get("id"):
                children.setdefault(pid, []).append(r)
            else:
                roots.append(r)

        def emit(node: dict, depth: int, visited: set) -> None:
            if node.get("id") in visited:
                return
            visited.add(node.get("id"))
            indent = "  " * depth
            opened = (node.get("opened_at") or "")[:19].replace("T", " ")
            lines.append(
                f"\n{indent}### [{node.get('id')}] {node.get('symbol')} · {node.get('kind')}"
                + (f"（父: {node.get('parent_id')}）" if node.get("parent_id") else "（顶层）")
            )
            lines.append(
                f"{indent}- 方向: {node.get('direction') or 'N/A'}  "
                f"开仓/挂单价: {node.get('entry_price') or 'N/A'}  "
                f"止损: {node.get('stop_loss') or 'N/A'}  止盈: {node.get('take_profit') or 'N/A'}"
            )
            if node.get("order_id"):
                lines.append(f"{indent}- 绑定币安单号(order_id): {node.get('order_id')}")
            lines.append(f"{indent}- 开始时间: {opened}")
            lines.append(f"{indent}- 理由: {node.get('thesis') or 'N/A'}")
            if node.get("note"):
                lines.append(f"{indent}- 备注: {node['note']}")
            for child in children.get(node.get("id"), []):
                emit(child, depth + 1, visited)

        visited: set = set()
        for r in roots:
            emit(r, 0, visited)
        # 兜底：未被根遍历到的节点（孤儿 / 异常成环）也列出，避免遗漏
        for r in self._data:
            if r.get("id") not in visited:
                emit(r, 0, visited)
        return "\n".join(lines)

    def render_drift_check(self, account_positions: dict[str, Any]) -> str:
        """渲染「理由 vs 当前行情」偏离检查（供反思者对照 / 决策者复核）。

        对每个仍有持仓的 position 记录，给出相对开仓价的偏离程度，便于判断
        行情是否仍符合原始理由。
        """
        lines = ["# 操作理由偏离检查（原始理由 vs 当前行情）"]
        has_content = False
        for r in self._data:
            if r.get("kind") != "position":
                continue
            pos = account_positions.get(r.get("symbol", ""))
            if pos is None:
                continue
            has_content = True
            entry = r.get("entry_price")
            drift_txt = ""
            if isinstance(entry, (int, float)) and entry > 0 and pos.mark_price > 0:
                drift = (pos.mark_price / entry - 1) * 100
                drift_txt = f" 偏离 {drift:+.2f}%"
            lines.append(
                f"- [{r.get('id')}] {r.get('symbol')} {r.get('direction') or 'N/A'} "
                f"@ {entry or 'N/A'}{drift_txt} | 现持仓 {abs(pos.position_amt):.6g} | "
                f"理由: {(r.get('thesis') or 'N/A')[:80]}"
            )
        if not has_content:
            lines.append("（无现存持仓理由记录）")
        return "\n".join(lines)
