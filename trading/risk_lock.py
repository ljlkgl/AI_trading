"""止损漂移（Stop-Loss Drift）物理铁律：初始风险额度锁定 + 校验。

背景：AI 在多轮连续决策中可能“追跌”式下移止损（止损漂移），把单笔最大亏损越拉越大。
为此在开仓/挂单成交（PENDING→FILLED）那一刻，把该仓位的“绝对风险敞口”固化为不可变基线
（locked_risk_usdt），此后任何 SET_SL_TP 都将被强制校验：

    proposed_risk = |当前开仓均价 − 新止损价| × 当前持仓数量
    只要 proposed_risk > locked_risk_usdt，直接驳回（绝对亏损只许缩小、不许扩大）。

即：想扩大止损距离（亏更多点数）就必须减仓；想加仓就必须收紧止损。账户总风险被锁死。

USDT 本位永续合约面值=1，故 locked_risk = |entry − stop| × qty × 1。持久化到
state/risk_locks.json，按 symbol 隔离。
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def abs_risk_usdt(entry: float, stop: float, qty: float, contract_mult: float = 1.0) -> float:
    """绝对风险敞口（USDT 计价的最大亏损额）= |入场价−止损价| × 数量 × 合约面值。

    USDT 本位（正向）永续合约面值取 1；币本位需额外换算，此处默认 1。
    """
    if entry is None or stop is None or qty is None or qty <= 0:
        return 0.0
    return abs(float(entry) - float(stop)) * float(qty) * float(contract_mult)


class RiskLockStore:
    """按币种持久化「初始风险额度锁」。初始值一旦落盘永久冻结，不随后续 AI 输出更新。"""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = path or (Path(__file__).resolve().parent.parent / "state" / "risk_locks.json")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, dict] = self._load()

    def _load(self) -> dict[str, dict]:
        if not self.path.exists():
            return {}
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("风险额度锁文件损坏，重新初始化: %s", exc)
            return {}

    def save(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, separators=(",", ":"))

    def set(
        self,
        symbol: str,
        entry: float,
        stop: float,
        qty: float,
        contract_mult: float = 1.0,
    ) -> float:
        """写入/刷新某币种风险额度锁，返回锁定的绝对风险金额 locked_risk_usdt。

        仅在需要建立或更新初始基线时调用（开仓成交、挂单成交升级为持仓）。
        """
        locked = abs_risk_usdt(entry, stop, qty, contract_mult)
        self._data[symbol] = {
            "symbol": symbol,
            "entry_price": entry,
            "stop_price": stop,
            "qty": qty,
            "locked_risk_usdt": locked,
            "updated_at": datetime.now().isoformat(),
        }
        self.save()
        logger.info("锁定 %s 初始风险额度 locked_risk=%.6fU (entry=%s stop=%s qty=%s)",
                    symbol, locked, entry, stop, qty)
        return locked

    def get(self, symbol: str) -> Optional[dict]:
        return self._data.get(symbol)

    def clear(self, symbol: str) -> None:
        if symbol in self._data:
            del self._data[symbol]
            self.save()
            logger.info("清除 %s 风险额度锁（仓位已平）", symbol)

    def all_keys(self) -> list[str]:
        return list(self._data.keys())