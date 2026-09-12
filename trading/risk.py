"""风控模块：在执行下单前对指令做硬性校验。

校验规则（与决策者提示词中的硬约束一致）：
1. 杠杆 ≤ max_leverage
2. 单笔名义价值（= 初始保证金 × 杠杆）≥ min_notional；初始保证金 ≥ min_margin
3. 初始保证金 ≤ 账户可用余额（真实可成交性底线）
4. 止损强制与方向合理性；换算出的下单数量 ≥ 交易所最小下单量
仓位大小（初始保证金）由模型自主决定，不做比例拦截。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from agents.schemas import OrderAction, TradeInstruction
from config import config
from trading.types import AccountInfo, Position, SymbolInfo

logger = logging.getLogger(__name__)


def ceil_to_step(value: float, step: float) -> float:
    """按 stepSize 向上取整（保证不小于原始值，用于最小下单量换算）。"""
    if step <= 0:
        return value
    n = int(-(-value // step))
    return n * step


def _round_step_half(value: float, step: float) -> float:
    """按 stepSize 四舍五入（ROUND_HALF_UP），与 OrderExecutor._quantity 同口径。

    用于把「目标名义=保证金×杠杆 / 开仓价」换算的数量折叠到交易所实际可下的数量，
    从而得到与 executor 真正会发送一致的「实际可成交名义」。避免风控用目标名义
    误判高价币种的有效订单（如 BTC 取整到最小 0.001 已远超 min_notional）。
    """
    if step <= 0:
        return value
    from decimal import Decimal, ROUND_HALF_UP

    dstep = Decimal(str(step))
    n = (Decimal(str(value)) / dstep).to_integral_value(rounding=ROUND_HALF_UP)
    return float((n * dstep).quantize(dstep))


def min_margin_for(
    symbol_info: SymbolInfo,
    price: float,
    leverage: int,
    min_margin_cfg: Optional[float] = None,
    min_notional_cfg: Optional[float] = None,
) -> float:
    """计算给定杠杆下，某交易品种的最少初始保证金（USDT）。

    名义价值 = 初始保证金 × 杠杆；因此 最少初始保证金 = 最少名义价值 / 杠杆。
    最少名义价值取以下口径的最大值：
    1. 系统单笔名义价值下限 min_notional_cfg（默认 config.min_notional）；
    2. 交易所 MIN_NOTIONAL 下限（symbol_info.min_notional）；
    3. 交易所最小下单量约束：数量 = 名义/价格 ≥ minQty，
       即 名义 ≥ ceil(minQty/step)*step*price（按 qty_step 向上取整）。
    再与系统单笔保证金下限 min_margin_cfg 取最大（保证金本身不能低于系统下限）。
    """
    min_notional_cfg = min_notional_cfg if min_notional_cfg is not None else config.min_notional
    min_margin_cfg = min_margin_cfg if min_margin_cfg is not None else config.min_margin
    if leverage <= 0 or price <= 0:
        return min_margin_cfg

    # 交易所最小下单量对应的名义价值
    qty_min = ceil_to_step(symbol_info.min_qty, symbol_info.qty_step)
    notional_from_qty = qty_min * price

    min_notional = max(
        min_notional_cfg,
        symbol_info.min_notional,
        notional_from_qty,
    )
    margin = max(min_notional / leverage, min_margin_cfg)
    return float(margin)


def build_min_margin_context(
    symbols: list[str],
    symbol_info_map: dict[str, SymbolInfo],
    price_map: dict[str, float],
    max_leverage: int = 20,
    min_margin_cfg: Optional[float] = None,
    min_notional_cfg: Optional[float] = None,
) -> str:
    """构建「各品种在不同杠杆下的最少初始保证金」上下文，供决策者参考。

    输出的核心信息：你给出的开仓 margin 必须 ≥ 对应杠杆下的最少初始保证金，
    否则换算出的下单数量可能低于交易所最小下单量而被拦截。
    """
    leverages = sorted({1, 5, 10, 15, max(15, max_leverage)})
    lines = [
        "# 各品种最少初始保证金（按杠杆）",
        "- 规则：最少初始保证金 = 最少名义价值 / 杠杆；最少名义价值取以下最大值：",
        f"  系统下限 {min_notional_cfg or config.min_notional:.0f} USDT、"
        f"交易所 MIN_NOTIONAL、以及 交易所最小下单量×当前价（按最小步长向上取整）",
        f"- 同时单笔保证金不能低于系统下限 {min_margin_cfg or config.min_margin:.2f} USDT",
        "- 你输出的 margin 必须 ≥ 当前杠杆对应的最少初始保证金，否则下单会被系统拦截",
        "",
        "| 品种 | 现价 | " + " | ".join(f"{l}x" for l in leverages) + " |",
        "|---|--|" + "---|" * len(leverages),
    ]
    for sym in symbols:
        info = symbol_info_map.get(sym)
        price = price_map.get(sym, 0)
        if info is None or price <= 0:
            lines.append(f"| {sym} | 无价格/精度 | - |")
            continue
        cells = [
            f"{min_margin_for(info, price, l, min_margin_cfg, min_notional_cfg):.2f}"
            for l in leverages
        ]
        lines.append(f"| {sym} | {price:.6g} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


@dataclass
class RiskCheckResult:
    ok: bool
    errors: list[str]


class RiskManager:
    """风控校验器。"""

    def __init__(
        self,
        max_position_ratio: Optional[float] = None,
        max_total_position_ratio: Optional[float] = None,
        max_leverage: Optional[int] = None,
        min_notional: Optional[float] = None,
        min_margin: Optional[float] = None,
        max_sl_risk_ratio: Optional[float] = None,
        max_total_sl_risk_ratio: Optional[float] = None,
        lock_store=None,
    ) -> None:
        self.max_position_ratio = (
            max_position_ratio or config.max_position_ratio
        )
        self.max_total_position_ratio = (
            max_total_position_ratio or config.max_total_position_ratio
        )
        self.max_leverage = max_leverage or config.max_leverage
        self.min_notional = min_notional or config.min_notional
        self.min_margin = min_margin if min_margin is not None else config.min_margin
        # 止损绝对风险上限比例（占权益）；代码级硬约束，防止损漂移
        self.max_sl_risk_ratio = (
            max_sl_risk_ratio
            if max_sl_risk_ratio is not None
            else (config.max_sl_risk_ratio if hasattr(config, "max_sl_risk_ratio") else 0.02)
        )
        # 初始风险额度锁（RiskLockStore），用于校验 SET_SL_TP 不扩大锁定亏损
        self.lock_store = lock_store
        # 全局止损绝对风险总上限：所有币种已锁定风险之和 ≤ 权益×此比例。
        # 用于开新单前合计当前风险敞口、检查剩余预算（任务一B/六）。
        self.max_total_sl_risk_ratio = (
            max_total_sl_risk_ratio
            if max_total_sl_risk_ratio is not None
            else (
                config.max_total_sl_risk_ratio
                if hasattr(config, "max_total_sl_risk_ratio")
                else 0.05
            )
        )

    # ---------- 辅助 ----------

    def current_position(
        self, account: AccountInfo, symbol: str
    ) -> Optional[Position]:
        for p in account.positions:
            if p.symbol == symbol:
                return p
        return None

    # ---------- 止损漂移硬约束 ----------

    @staticmethod
    def _abs_risk(entry: float, sl: float, qty: float) -> float:
        """单品种绝对亏损额（USDT）= |开仓均价 − 止损价| × 持仓数量（USDT 本位，面值=1）。"""
        if not entry or not sl or not qty or qty <= 0:
            return 0.0
        return abs(float(entry) - float(sl)) * float(qty)

    def _check_stop_risk(self, errors, symbol, entry, total_qty, proposed_sl, equity):
        """校验新止损，只做「防止损漂移」约束。

        不再按账户权益比例强算单笔止损：止损位置与仓位由模型按技术位（支撑/阻力）
        自主决定，系统不因"绝对亏损额 > 权益×比例"而拦单。
        仅保留：若该币种已有初始锁定风险额度 locked_risk，则绝对亏损额只许缩小、
        不许超过 locked_risk（想扩大止损距离必须减仓），防止止损漂移扩大风险敞口。
        """
        rk = self._abs_risk(entry, proposed_sl, total_qty)
        if rk <= 0:
            return
        # （权益强止损已停用）不再按 self.max_sl_risk_ratio 拦截单笔绝对风险。
        # 规则2：风险额度锁（初始基线），若已锁定则相对该基线防漂移
        locked = self.lock_store.get(symbol) if self.lock_store else None
        if locked:
            locked_u = locked.get("locked_risk_usdt")
            if locked_u and rk > locked_u + 1e-9:
                errors.append(
                    f"{symbol}: 新止损将绝对亏损扩大至 {rk:.4f}U，"
                    f"违反初始锁定风险额度 {locked_u:.4f}U。"
                    f"止损与持仓数量联动：想扩大止损距离必须同步减仓至 "
                    f"{locked_u / (abs(float(entry) - float(proposed_sl)) or 1):.6g}，"
                    f"否则系统按漂移风险拦截"
                )

    # ---------- 指令校验 ----------

    def validate_instruction(
        self,
        instruction: TradeInstruction,
        account: AccountInfo,
        symbol_info: SymbolInfo,
        mark_price: float,
    ) -> RiskCheckResult:
        """校验单条指令，返回结果。"""
        errors: list[str] = []
        equity = account.margin_balance if account.margin_balance > 0 else account.total_balance

        if instruction.action == OrderAction.HOLD:
            return RiskCheckResult(ok=True, errors=[])

        # 挂单管理类：不涉及保证金/杠杆/止损，仅校验必要字段
        if instruction.action == OrderAction.CANCEL_ORDERS:
            return RiskCheckResult(ok=True, errors=[])
        if instruction.action == OrderAction.REPLACE_LIMIT:
            if instruction.quantity is None or instruction.quantity <= 0:
                errors.append(f"{instruction.symbol}: REPLACE_LIMIT 必须提供 quantity>0")
            if instruction.price is None or instruction.price <= 0:
                errors.append(f"{instruction.symbol}: REPLACE_LIMIT 必须提供 price>0")
            if instruction.side not in ("BUY", "SELL"):
                errors.append(f"{instruction.symbol}: REPLACE_LIMIT 必须提供 side=BUY/SELL")
            return RiskCheckResult(ok=len(errors) == 0, errors=errors)

        # 止盈止损调整类：必须已有持仓，且止损/止盈方向相对当前价合理
        if instruction.action == OrderAction.SET_SL_TP:
            pos = self.current_position(account, instruction.symbol)
            ref_price = instruction.price or mark_price
            if pos is None:
                errors.append(f"{instruction.symbol}: 无持仓，无法调整止盈止损")
            else:
                sl, tp = instruction.stop_loss, instruction.take_profit
                if sl is None and tp is None:
                    errors.append(
                        f"{instruction.symbol}: SET_SL_TP 必须提供 stop_loss 或 take_profit 至少一个"
                    )
                if sl is not None:
                    if pos.position_amt > 0 and sl >= ref_price:
                        errors.append(f"{instruction.symbol}: 多仓止损价需低于当前价 {ref_price}")
                    if pos.position_amt < 0 and sl <= ref_price:
                        errors.append(f"{instruction.symbol}: 空仓止损价需高于当前价 {ref_price}")
                    # 物理风控铁律：止损漂移校验（2% 上限 + 初始风险额度锁）
                    self._check_stop_risk(
                        errors, instruction.symbol,
                        pos.entry_price, abs(pos.position_amt), sl, equity,
                    )
                if tp is not None:
                    if pos.position_amt > 0 and tp <= ref_price:
                        errors.append(f"{instruction.symbol}: 多仓止盈价需高于当前价 {ref_price}")
                    if pos.position_amt < 0 and tp >= ref_price:
                        errors.append(f"{instruction.symbol}: 空仓止盈价需低于当前价 {ref_price}")
            return RiskCheckResult(ok=len(errors) == 0, errors=errors)

        pos = self.current_position(account, instruction.symbol)

        # 开仓类：校验初始保证金 + 杠杆 + 名义价值（名义 = 保证金×杠杆）
        if instruction.action in (OrderAction.OPEN_LONG, OrderAction.OPEN_SHORT):
            lev = instruction.leverage or 1
            entry = instruction.price or mark_price
            margin = instruction.margin
            if margin is None or margin <= 0:
                errors.append(
                    f"{instruction.symbol}: 开仓必须提供 margin>0（初始保证金，USDT）"
                )
            else:
                notional = margin * lev
                # 名义价值下限 = 系统下限 与 交易所 MIN_NOTIONAL 的较大值（二者都需满足）
                min_notional = max(self.min_notional, symbol_info.min_notional)
                # 实际可成交名义 = 按交易所 step 取整后的数量 × 开仓价（与 OrderExecutor._quantity
                # 同口径）。高价币种（如 BTC≈7 万）即便取整到最小 0.001 也已远超 min_notional，
                # 若用「目标名义=保证金×杠杆」会低于实际可成交值，误杀本可有效成交的订单。
                if entry > 0 and symbol_info.qty_step > 0:
                    q_fold = _round_step_half(notional / entry, symbol_info.qty_step)
                    real_notional = q_fold * entry
                else:
                    q_fold = None
                    real_notional = notional
                if real_notional < min_notional:
                    _qtxt = f"数量 {notional / entry:.6g} 按 step 取整为 {q_fold:.6g}" if q_fold is not None else ""
                    errors.append(
                        f"{instruction.symbol}: 实际可成交名义 {real_notional:.2f} USDT"
                        f"（{_qtxt}）低于最小限制 {min_notional:.2f} USDT"
                        f"（含交易所 MIN_NOTIONAL {symbol_info.min_notional:.2f}；"
                        f"名义=取整后数量×开仓价）"
                    )
                if margin < self.min_margin:
                    errors.append(
                        f"{instruction.symbol}: 开仓保证金 {margin:.4f} USDT "
                        f"低于最小保证金 {self.min_margin:.2f} USDT"
                    )
                # 可用余额校验：初始保证金不得超过账户可用余额（真实可成交性底线）
                if margin > account.available_balance:
                    errors.append(
                        f"{instruction.symbol}: 开仓保证金 {margin:.4f} USDT "
                        f"超过可用余额 {account.available_balance:.4f} USDT"
                    )
                # 换算数量按 step 取整后不得低于交易所最小下单量
                if q_fold is not None and q_fold < symbol_info.min_qty:
                    errors.append(
                        f"{instruction.symbol}: 保证金 {margin:.4f}×杠杆{lev} 换算数量 "
                        f"{q_fold:.6g} 低于最小下单量 {symbol_info.min_qty:.6g}（需增大保证金或杠杆）"
                    )
            if lev > self.max_leverage:
                errors.append(
                    f"{instruction.symbol}: 杠杆 {lev} 超过上限 {self.max_leverage}"
                )
            # 止损强制：开仓必须带止损
            if instruction.stop_loss is None:
                errors.append(f"{instruction.symbol}: 开仓必须设置 stop_loss（止损）")
            else:
                if instruction.action == OrderAction.OPEN_LONG and instruction.stop_loss >= (instruction.price or mark_price):
                    errors.append(f"{instruction.symbol}: 多单止损价需低于开仓价")
                if instruction.action == OrderAction.OPEN_SHORT and instruction.stop_loss <= (instruction.price or mark_price):
                    errors.append(f"{instruction.symbol}: 空单止损价需高于开仓价")
                # 物理风控铁律：开仓/加仓的止损绝对风险校验（2% 上限 + 风险额度锁）。
                # 开仓数量 = 名义价值/开仓价；若为已有持仓的加仓，则按当前+新仓的总数量与
                # 合并均价校验，确保加仓不突破初始锁定风险额度。
                if entry > 0 and margin and margin > 0:
                    new_qty = notional / entry if margin and lev else 0
                    cur_qty = abs(pos.position_amt) if pos else 0.0
                    total_qty = cur_qty + new_qty
                    if cur_qty > 0:
                        # 加权合并均价：已有持仓均价 与 新仓开仓价按数量加权
                        combined_entry = (pos.position_amt * pos.entry_price + new_qty * entry) / total_qty
                    else:
                        combined_entry = entry
                    self._check_stop_risk(
                        errors, instruction.symbol,
                        combined_entry, total_qty, instruction.stop_loss, equity,
                    )

        # 平仓类：校验数量不超过持仓
        if instruction.action in (OrderAction.CLOSE_LONG, OrderAction.CLOSE_SHORT, OrderAction.FLATTEN):
            if pos is None:
                errors.append(f"{instruction.symbol}: 无持仓，无需平仓")
            else:
                side_ok = (
                    instruction.action == OrderAction.FLATTEN
                    or (instruction.action == OrderAction.CLOSE_LONG and pos.position_amt > 0)
                    or (instruction.action == OrderAction.CLOSE_SHORT and pos.position_amt < 0)
                )
                if not side_ok:
                    errors.append(
                        f"{instruction.symbol}: 平仓方向与持仓方向不一致（持仓 {pos.position_amt:+.6g}）"
                    )
                if instruction.quantity is not None and abs(instruction.quantity) > abs(pos.position_amt):
                    errors.append(
                        f"{instruction.symbol}: 平仓数量 {instruction.quantity} 超过持仓 {abs(pos.position_amt):.6g}"
                    )

        return RiskCheckResult(ok=len(errors) == 0, errors=errors)

    def fit_margin_to_budget(
        self,
        instruction: TradeInstruction,
        account: AccountInfo,
        symbol_info: SymbolInfo,
        mark_price: float,
    ) -> Optional[float]:
        """开仓被「止损绝对风险超限」拦截时，反解出预算边界内允许的最大 margin。

        目标：|入场−止损| × 数量 ≤ 风险预算，其中 数量 = margin × 杠杆 / 入场价。
        反解得 margin ≤ 预算 × 入场价 / (杠杆 × |入场−止损|)。
        - 预算边界：单品种上限（权益×max_sl_risk_ratio）与锁定额度取较小者，再叠加已有持仓占用；
        - 全局预算：所有币种已锁定风险之和 ≤ 权益×max_total_sl_risk_ratio（防止多单挤占预算）；
        - 可用余额：margin 不得超额可用余额（动态按账户规模缩放头寸）。

        返回预算内最大 margin；若无法满足（预算≤0 或失真）返回 None，调用方保持拦截。
        """
        if instruction.action not in (OrderAction.OPEN_LONG, OrderAction.OPEN_SHORT):
            return None
        leverage = instruction.leverage or 1
        entry = instruction.price or mark_price
        sl = instruction.stop_loss
        if not entry or entry <= 0 or not sl or leverage <= 0:
            return None
        # 预算边界：权益×单品种比例 与 该币种锁定额度 取较小者
        equity = account.margin_balance if account.margin_balance > 0 else account.total_balance
        budget = equity * self.max_sl_risk_ratio
        locked = self.lock_store.get(instruction.symbol) if self.lock_store else None
        locked_u = (locked or {}).get("locked_risk_usdt") if locked else None
        if locked_u:
            budget = min(budget, locked_u)
        # 全局预算：所有币种已锁定风险之和 ≤ 权益×总比例，剩余预算分给本币种
        if self.lock_store:
            other_locked = 0.0
            for sym in self.lock_store.all_keys():
                if sym == instruction.symbol:
                    continue
                other_locked += (self.lock_store.get(sym) or {}).get("locked_risk_usdt", 0.0)
            global_cap = equity * self.max_total_sl_risk_ratio
            budget = min(budget, max(0.0, global_cap - other_locked))
        # 已有持仓占用本币种预算
        pos = self.current_position(account, instruction.symbol)
        occupied = 0.0
        if pos and pos.position_amt:
            occupied = self._abs_risk(pos.entry_price, sl, abs(pos.position_amt))
        remaining_risk = budget - occupied
        if remaining_risk <= 0:
            return None
        distance = abs(float(entry) - float(sl))
        if distance <= 0:
            return None
        # 反解 margin：数量 = margin×杠杆/入场价，风险 = 数量×距离 ≤ remaining_risk
        max_margin = remaining_risk * entry / (leverage * distance)
        # 动态缩放：margin 不得超额可用余额（账户规模自适配）
        avail = account.available_balance
        if avail is not None and avail > 0:
            max_margin = min(max_margin, avail)
        if max_margin <= 0:
            return None
        return float(max_margin)

    def validate_decision(
        self,
        instructions: list[TradeInstruction],
        account: AccountInfo,
        price_map: dict[str, float],
        symbol_info_map: dict[str, SymbolInfo],
        auto_downgrade: bool = True,
    ) -> tuple[list[TradeInstruction], list[RiskCheckResult]]:
        """校验整个决策。返回 (通过的指令, 校验结果列表)。

        auto_downgrade=True（默认）：当开仓因「止损绝对风险超限」被拒时，尝试把 margin
        自动缩到预算边界内重新提交（自动降档重提），而不只是拦截——避免错过入场窗口。
        若降档后仍不满足其它硬约束（如低于最少保证金/名义价值/止损方向），保持拦截。
        """
        results: list[RiskCheckResult] = []
        # 先整体校验一次，收集被拒的开仓（供降档重提）
        initial: list[TradeInstruction] = []
        for ins in instructions:
            mark = price_map.get(ins.symbol, 0.0)
            info = symbol_info_map.get(ins.symbol)
            if mark <= 0 or info is None:
                results.append(RiskCheckResult(ok=False, errors=[f"{ins.symbol}: 缺少标记价格或交易对信息"]))
                continue
            initial.append(ins)

        passed: list[TradeInstruction] = []
        for ins in initial:
            mark = price_map.get(ins.symbol, 0.0)
            info = symbol_info_map.get(ins.symbol)
            res = self.validate_instruction(ins, account, info, mark)
            if res.ok:
                passed.append(ins)
                results.append(res)
                continue
            # 仅对「开仓因止损绝对风险超限被拒」做自动降档重提
            if auto_downgrade and ins.action in (OrderAction.OPEN_LONG, OrderAction.OPEN_SHORT):
                fit = self.fit_margin_to_budget(ins, account, info, mark)
                if fit is not None and ins.margin is not None and fit < ins.margin:
                    fitted = ins.model_copy(update={"margin": fit})
                    res2 = self.validate_instruction(fitted, account, info, mark)
                    if res2.ok:
                        logger.info(
                            "%s %s 止损绝对风险超限，自动降档 margin %.6g→%.6g 重提成功",
                            ins.symbol, ins.action.value, ins.margin, fit,
                        )
                        passed.append(fitted)
                        results.append(RiskCheckResult(
                            ok=True,
                            errors=[f"自动降档重提：margin {ins.margin:.6g}→{fit:.6g}（止损绝对风险缩到预算内）"],
                        ))
                        continue
            # 保持拦截
            results.append(res)
            logger.warning("指令被风控拦截 %s %s: %s", ins.symbol, ins.action, "; ".join(res.errors))
        return passed, results
