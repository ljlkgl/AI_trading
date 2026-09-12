"""订单执行器：将 TradingDecision 中的指令映射为币安订单。

处理开仓（设置杠杆/保证金模式→下单→挂止损止盈）、平仓（reduceOnly）。
DRY_RUN 模式下只打印不真实下单。
"""
from __future__ import annotations

import logging
import time
import uuid
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Optional

from agents.schemas import OrderAction, OrderType, TradeInstruction
from config import config
from trading.binance_client import BinanceClient, BinanceError
from trading.risk import RiskManager
from trading.risk_lock import RiskLockStore
from trading.types import AccountInfo

logger = logging.getLogger(__name__)


def round_to_step(value: float, step: float) -> float:
    """按 stepSize 量化并去掉浮点噪声，避免精度错误。

    例：round_to_step(0.007776, 0.0001) -> 0.0078（而非 0.0078000000000000005）。
    用 Decimal 计算后再转 float，保证 str() 序列化时是干净的十进制表示。
    """
    if step <= 0:
        return value
    dstep = Decimal(str(step))
    n = (Decimal(str(value)) / dstep).to_integral_value(rounding=ROUND_HALF_UP)
    return float((n * dstep).quantize(dstep))


def round_price(value: float, tick: float) -> float:
    if tick <= 0:
        return value
    dstep = Decimal(str(tick))
    n = (Decimal(str(value)) / dstep).to_integral_value(rounding=ROUND_HALF_UP)
    return float((n * dstep).quantize(dstep))


class OrderExecutor:
    """执行交易决策。"""

    def __init__(
        self,
        client: BinanceClient,
        risk_manager: RiskManager,
        dry_run: Optional[bool] = None,
        hedge_mode: bool = False,
        lock_store=None,
    ) -> None:
        self.client = client
        self.risk = risk_manager
        self.dry_run = config.dry_run if dry_run is None else dry_run
        # 双向持仓模式(Hedge Mode)：下单需带 positionSide；单向模式为 None
        self.hedge_mode = hedge_mode
        # 初始风险额度锁：开仓成交瞬间固化绝对风险敞口（USDT 本位面值=1）。
        # 默认自建一个；调用方可传入共享实例（与风控共享，确保校验读到同一份锁）。
        self._locks = lock_store if lock_store is not None else RiskLockStore()

    def _lock_initial_risk(self, symbol: str, entry_price: float, qty: float, stop) -> None:
        """开仓一侧定初始风险额度锁：|入场价−止损价|×数量，落盘后永久冻结。

        仅当带止损时才锁（无止损开仓会被风控拦截，理论上不会到此）。
        """
        if stop is None or not entry_price or not qty or qty <= 0:
            return
        self._locks.set(symbol, entry_price, stop, abs(qty))

    # ---------- 工具 ----------

    def _ps(self, position_side: str) -> Optional[str]:
        """双向模式返回 positionSide（LONG/SHORT）；单向模式返回 None（不下传）。"""
        return position_side if self.hedge_mode else None

    def _pos_side(self, pos) -> str:
        """从持仓对象推导仓位方向：优先真实 positionSide，回退按数量符号。"""
        if pos.position_side in ("LONG", "SHORT"):
            return pos.position_side
        return "LONG" if pos.position_amt > 0 else "SHORT"

    @staticmethod
    def _prec_dec(precision, fallback_step) -> Decimal:
        """下单精度的 Decimal 步长：优先用交易所精度 precision，退化用 stepSize。"""
        if precision is not None and int(precision) >= 0:
            return Decimal(1).scaleb(-int(precision))
        return Decimal(str(fallback_step))

    def _quantity(self, symbol_info, qty: float) -> float:
        # 先按 stepSize 量化去除浮点噪声，再显式裁剪到交易所允许的小数位，
        # 双保险保证绝不超过 Precision 上限。
        d = Decimal(str(round_to_step(abs(qty), symbol_info.qty_step)))
        return float(d.quantize(self._prec_dec(symbol_info.qty_precision, symbol_info.qty_step)))

    def _price(self, symbol_info, price: float) -> float:
        d = Decimal(str(round_price(price, symbol_info.price_tick)))
        return float(d.quantize(self._prec_dec(symbol_info.price_precision, symbol_info.price_tick)))

    def _client_order_id(self, tag: str) -> str:
        return f"{tag}{uuid.uuid4().hex[:12]}"

    # ---------- 核心执行 ----------

    def execute(
        self,
        instructions: list[TradeInstruction],
        account: AccountInfo,
    ) -> list[dict]:
        """执行指令列表，返回执行结果日志列表。"""
        results: list[dict] = []
        for ins in instructions:
            try:
                result = self._execute_one(ins, account)
                results.append(result)
            except Exception as exc:  # noqa: BLE001
                logger.exception("执行指令失败 %s %s: %s", ins.symbol, ins.action, exc)
                results.append(
                    {
                        "symbol": ins.symbol,
                        "action": ins.action.value,
                        "status": "FAILED",
                        "error": str(exc),
                    }
                )
        return results

    def flatten_all(self, account: AccountInfo, reason: str = "LLM API 全部不可用") -> list[dict]:
        """紧急平仓：市价（reduceOnly）平掉账户全部持仓。

        用于所有 LLM API 均无法通讯时的风险撤退；DRY_RUN 下只打印不真实下单。
        """
        results: list[dict] = []
        for p in account.positions:
            try:
                side = "SELL" if p.position_amt > 0 else "BUY"
                qty = self._quantity(
                    self.client.get_symbol_info(p.symbol), abs(p.position_amt)
                )
                if qty <= 0:
                    results.append({
                        "symbol": p.symbol, "action": "EMERGENCY_FLATTEN",
                        "status": "REJECTED", "error": "数量无效",
                    })
                    continue
                if self.dry_run:
                    logger.warning(
                        "[DRY_RUN] 紧急平仓 %s %s qty=%s side=%s（%s）",
                        p.symbol, p.position_amt, qty, side, reason,
                    )
                    results.append({
                        "symbol": p.symbol, "action": "EMERGENCY_FLATTEN",
                        "status": "DRY_RUN", "side": side, "quantity": qty,
                        "reason": reason,
                    })
                    continue
                order = self.client.close_position(
                    symbol=p.symbol, side=side, quantity=qty,
                    position_side=self._ps(self._pos_side(p)),
                )
                self._locks.clear(p.symbol)
                logger.warning(
                    "紧急平仓 %s %s qty=%s side=%s -> %s（%s）",
                    p.symbol, p.position_amt, qty, side, order.get("status"), reason,
                )
                results.append({
                    "symbol": p.symbol, "action": "EMERGENCY_FLATTEN",
                    "status": "CLOSED", "side": side, "quantity": qty, "order": order,
                    "order_id": order.get("orderId"), "reason": reason,
                })
            except Exception as exc:  # noqa: BLE001
                logger.exception("紧急平仓失败 %s: %s", p.symbol, exc)
                results.append({
                    "symbol": p.symbol, "action": "EMERGENCY_FLATTEN",
                    "status": "FAILED", "error": str(exc),
                })
        if not account.positions:
            logger.info("紧急平仓：当前无持仓")
        return results

    def _execute_one(self, ins: TradeInstruction, account: AccountInfo) -> dict:
        symbol_info = self.client.get_symbol_info(ins.symbol)
        mark_price = self.client.get_ticker_price(ins.symbol)
        pos = next((p for p in account.positions if p.symbol == ins.symbol), None)

        base = {
            "symbol": ins.symbol,
            "action": ins.action.value,
            "dry_run": self.dry_run,
        }

        if ins.action == OrderAction.HOLD:
            base.update(status="SKIPPED", detail="持有不动")
            return base

        if ins.action in (OrderAction.OPEN_LONG, OrderAction.OPEN_SHORT):
            return self._open(ins, symbol_info, mark_price, base)
        if ins.action in (OrderAction.CLOSE_LONG, OrderAction.CLOSE_SHORT):
            return self._close(ins, pos, mark_price, base)
        if ins.action == OrderAction.FLATTEN:
            return self._flatten(ins, pos, mark_price, base)
        if ins.action == OrderAction.CANCEL_ORDERS:
            return self._cancel_orders(ins, base)
        if ins.action == OrderAction.REPLACE_LIMIT:
            return self._replace_limit(ins, symbol_info, base)
        if ins.action == OrderAction.SET_SL_TP:
            return self._set_sl_tp(ins, pos, symbol_info, base)
        raise ValueError(f"未知动作: {ins.action}")

    def _cancel_orders(self, ins: TradeInstruction, base: dict) -> dict:
        """撤销该币种全部未成交挂单（含主 LIMIT 单及其配套止盈止损 algo 条件单）。"""
        if self.dry_run:
            logger.info("[DRY_RUN] %s CANCEL_ORDERS（撤销全部未成交挂单，含 algo 止盈止损）", ins.symbol)
            base.update(status="DRY_RUN")
            return base
        try:
            cancelled = self._cancel_pending_limits(ins.symbol)
            unresolved = self._confirm_cancel(ins.symbol, cancelled)
            logger.info(
                "%s 已撤销挂单: %d 笔%s",
                ins.symbol, len(cancelled),
                f", 未确认消失 {len(unresolved)}:{unresolved}" if unresolved else "",
            )
            base.update(status="CANCELLED", cancelled_count=len(cancelled),
                        cancelled_ids=cancelled)
            if unresolved:
                logger.warning(
                    "%s 撤单后仍有 %d 笔未从交易所消失（最终一致或端点未清除，需跟进）: %s",
                    ins.symbol, len(unresolved), unresolved,
                )
                base["unconfirmed_cancel_ids"] = unresolved
        except BinanceError as exc:
            base.update(status="FAILED", error=str(exc))
        return base

    def _confirm_cancel(self, symbol: str, ids: list[Any]) -> list[Any]:
        """撤单后确认目标订单已从交易所消失；仍存在的重试撤销，返回仍未消失的 ID 列表。

        币安撤单是异步的（最终一致），仅信任返回码会把撤单当成已生效。这里对目标
        （含普通 orderId 与 algoId）做至多三轮回查，若仍在 open orders / algo 列表则
        用已修正的逻辑（普通 LIMIT + algo 条件单）再次撤销；三轮后仍未消失则上报，
        交由本轮/反思层知晓，而不是把 `CANCELLED` 当成绝对成功。
        """
        if not ids:
            return []
        pending = set(str(i) for i in ids)
        for _ in range(3):
            if not pending:
                break
            time.sleep(0.4)
            try:
                self._cancel_pending_limits(symbol)  # 重新撤销残留（普通单 + algo）
                open_ids = {str(o.get("orderId")) for o in self.client.get_open_orders(symbol)}
                algo_ids = {str(a.get("algoId")) for a in self.client.get_open_algo_orders(symbol)}
            except BinanceError:
                break
            still = pending & (open_ids | algo_ids)
            if not still:
                return []
            pending = still
        return sorted(pending)

    def _cancel_pending_limits(self, symbol: str) -> list[Any]:
        """撤销该币种全部未成交挂单，返回被撤销的 ID 列表（orderId 或 algoId）。

        包含两类：
        1. 普通 LIMIT 挂单（主挂单）；
        2. 配套的 algo 条件单（STOP_MARKET / TAKE_PROFIT_MARKET 止盈止损）。

        修复「撤主挂单后其止盈止损 algo 单未随同取消」的遗留单问题：主挂单一经撤销，
        挂靠它的 algo 条件单也必须一并取消，否则会在交易所残留幽灵 algo 单，与系统
        内存 / 账户实际状态脱节。
        """
        cancelled: list[Any] = []
        # 1）普通 LIMIT 挂单
        for order in self.client.get_open_orders(symbol):
            if order.get("type") == "LIMIT":
                self.client.cancel_order(symbol, order["orderId"])
                cancelled.append(order["orderId"])
        # 2）algo 条件单（止盈止损），主挂单撤销时应一并取消，避免残留
        for order in self.client.get_open_algo_orders(symbol):
            if order.get("algoStatus") == "NEW":
                o_id = order.get("algoId")
                if o_id not in cancelled:
                    self.client.cancel_algo_order(symbol, o_id)
                    cancelled.append(o_id)
        return cancelled

    def _replace_limit(self, ins: TradeInstruction, symbol_info, base: dict) -> dict:
        """更改挂单：先撤销该币种全部挂单（含配套 algo 止盈止损），再按新价格/数量重挂 LIMIT。

        落单前先把旧入口单及其 algo 单清空，避免与旧挂单/旧 algo 单并存冲突。
        """
        qty = self._quantity(symbol_info, ins.quantity or 0)
        price = self._price(symbol_info, ins.price)
        if qty <= 0:
            base.update(status="REJECTED", error="数量无效")
            return base
        side = ins.side
        if self.dry_run:
            logger.info(
                "[DRY_RUN] %s REPLACE_LIMIT：撤销全部挂单后挂 LIMIT %s qty=%s price=%s",
                ins.symbol, side, qty, price,
            )
            base.update(status="DRY_RUN", side=side, quantity=qty,
                        price=price, order_type="LIMIT")
            return base
        try:
            cancelled = self._cancel_pending_limits(ins.symbol)
            order = self.client.place_order(
                symbol=ins.symbol,
                side=side,
                order_type="LIMIT",
                quantity=qty,
                price=price,
                time_in_force="GTC",
                client_order_id=self._client_order_id("rp"),
                position_side=self._ps("LONG" if side == "BUY" else "SHORT"),
            )
            logger.info(
                "%s 已改单：撤销 %d 笔 LIMIT 挂单并新挂 LIMIT %s qty=%s price=%s",
                ins.symbol, len(cancelled), side, qty, price,
            )
            base.update(status="REPLACED", side=side, quantity=qty,
                        price=price, order_type="LIMIT", order=order,
                        order_id=order.get("orderId"), cancelled_ids=cancelled)
        except BinanceError as exc:
            base.update(status="FAILED", error=str(exc))
        return base

    def _set_sl_tp(self, ins: TradeInstruction, pos, symbol_info, base: dict) -> dict:
        """调整已持仓位的止盈/止损：先撤销旧保护单，再按新价重挂。

        stop_loss / take_profit 至少提供一个（缺省的一项保留由新挂单处理）；
        保护单按持仓全量（reduceOnly + 持仓数量）。
        """
        if pos is None:
            base.update(status="REJECTED", error="无持仓，无法调整止盈止损")
            return base
        if ins.stop_loss is None and ins.take_profit is None:
            base.update(status="REJECTED", error="stop_loss 与 take_profit 至少提供一个")
            return base
        protective_side = "SELL" if pos.position_amt > 0 else "BUY"
        qty = self._quantity(symbol_info, abs(pos.position_amt))
        if qty <= 0:
            base.update(status="REJECTED", error="持仓数量无效")
            return base

        if self.dry_run:
            logger.info(
                "[DRY_RUN] %s SET_SL_TP：调整止盈止损 sl=%s tp=%s side=%s",
                ins.symbol, ins.stop_loss, ins.take_profit, protective_side,
            )
            base.update(status="DRY_RUN", side=protective_side, quantity=qty,
                        stop_loss=ins.stop_loss, take_profit=ins.take_profit)
            return base

        try:
            cancelled = self._cancel_protective_orders(ins.symbol)
            placed = self._place_stop_loss_take_profit(
                ins, qty, protective_side,
                protective_position_side=self._ps(self._pos_side(pos)),
            )
            logger.info(
                "%s 已调整止盈止损：撤销 %d 笔旧保护单，重挂 %d 笔（sl=%s tp=%s）",
                ins.symbol, len(cancelled), len(placed),
                ins.stop_loss, ins.take_profit,
            )
            base.update(status="ADJUSTED", side=protective_side, quantity=qty,
                        cancelled_ids=cancelled, sl_tp_orders=placed,
                        stop_loss=ins.stop_loss, take_profit=ins.take_profit)
        except BinanceError as exc:
            base.update(status="FAILED", error=str(exc))
        return base

    def _cancel_protective_orders(self, symbol: str) -> list[Any]:
        """撤销该币种全部未成交的止损/止盈保护单，返回被撤销的 ID 列表（orderId 或 algoId）。"""
        cancelled: list[Any] = []
        # 普通订单（旧接口遗留）
        for order in self.client.get_open_orders(symbol):
            if order.get("type") in ("STOP_MARKET", "TAKE_PROFIT_MARKET"):
                self.client.cancel_order(symbol, order["orderId"])
                cancelled.append(order["orderId"])
        # 算法条件单（新接口）
        for order in self.client.get_open_algo_orders(symbol):
            o_type = order.get("orderType") or order.get("type")
            if o_type in ("STOP_MARKET", "TAKE_PROFIT_MARKET") and order.get("algoStatus") == "NEW":
                self.client.cancel_algo_order(symbol, order["algoId"])
                cancelled.append(order["algoId"])
        return cancelled

    def _open(
        self, ins: TradeInstruction, symbol_info, mark_price: float, base: dict
    ) -> dict:
        side = "BUY" if ins.action == OrderAction.OPEN_LONG else "SELL"
        leverage = ins.leverage if ins.leverage else 1
        # 入场价：限价单用挂单价，市价单用当前标记价（决定换算数量与风控口径）
        entry_price = ins.price if (ins.order_type == OrderType.LIMIT and ins.price) else mark_price
        if entry_price <= 0:
            base.update(status="REJECTED", error="无效入场价")
            return base
        # 由初始保证金换算数量：名义价值 = margin × 杠杆；数量 = 名义价值 / 开仓价
        margin = ins.margin
        if margin is None or margin <= 0:
            base.update(status="REJECTED", error="开仓必须提供 margin>0（初始保证金，USDT）")
            return base
        notional = margin * leverage
        qty = self._quantity(symbol_info, notional / entry_price)
        if qty <= 0:
            base.update(
                status="REJECTED",
                error="数量无效（margin×杠杆/开仓价 换算结果过小，需增大保证金或杠杆）",
            )
            return base
        price = None
        order_type = "MARKET"
        if ins.order_type == OrderType.LIMIT:
            if not ins.price:
                base.update(status="REJECTED", error="限价单缺少价格")
                return base
            order_type = "LIMIT"
            price = self._price(symbol_info, ins.price)

        if self.dry_run:
            logger.info(
                "[DRY_RUN] %s %s qty=%s price=%s lev=%s margin=%s notional=%s sl=%s tp=%s",
                ins.symbol, ins.action.value, qty, price, leverage, margin, notional,
                ins.stop_loss, ins.take_profit,
            )
            base.update(
                status="DRY_RUN",
                order_type=order_type,
                side=side,
                quantity=qty,
                price=price,
                margin=margin,
                notional=notional,
            )
            # DRY_RUN 也用计划入场价/数量/止损固化学风险额度基线（便于一致校验）
            self._lock_initial_risk(ins.symbol, entry_price, qty, ins.stop_loss)
            return base

        # 真实下单
        try:
            self.client.set_margin_type(ins.symbol, "ISOLATED")
        except BinanceError as exc:
            logger.warning("设置保证金模式失败(可能已是ISOLATED): %s", exc)
        try:
            self.client.set_leverage(ins.symbol, leverage)
        except BinanceError as exc:
            logger.warning("设置杠杆失败: %s", exc)

        def _place(q: float, p):
            return self.client.place_order(
                symbol=ins.symbol,
                side=side,
                order_type=order_type,
                quantity=q,
                price=p,
                time_in_force="GTC" if order_type == "LIMIT" else None,
                client_order_id=self._client_order_id("open"),
                position_side=self._ps("LONG" if ins.action == OrderAction.OPEN_LONG else "SHORT"),
            )

        # _quantity/_price 已按交易所 step/tick 取整，正常情况下不会超精度；此处仅作保险：
        # 当交易所返回「Precision is over the maximum」时（常见于交易所调整 step/tick 或
        # 精度缓存过期），强制刷新精度并重新取整后重试一次，避免一次有效的开仓被误拒。
        try:
            order = _place(qty, price)
        except BinanceError as exc:
            if "precision" not in str(exc).lower():
                raise
            logger.warning("%s 下单精度被拒(%s)，刷新精度并重新取整后重试一次", ins.symbol, exc)
            fresh = self.client.refresh_symbol_info(ins.symbol)
            order = _place(
                self._quantity(fresh, notional / entry_price),
                self._price(fresh, ins.price) if order_type == "LIMIT" else None,
            )
        result = dict(base)
        result.update(
            status="OPENED",
            order=order,
            order_id=order.get("orderId"),
            quantity=qty,
            price=price,
            order_type=order_type,
            margin=margin,
            notional=notional,
        )
        # PENDING→FILLED 瞬间固化绝对风险额基线：取实际成交均价与成交数量；
        # 若交易所未回传成交均价（如限价单刚挂未成交），回退到计划价/数量。
        self._lock_initial_risk(
            ins.symbol,
            entry_price=float(order.get("avgPrice") or entry_price),
            qty=float(order.get("executedQty") or qty),
            stop=ins.stop_loss,
        )
        # 挂止损止盈（LIMIT 单未成交时 reduceOnly 保护单可能被拒，降级不致命）
        # 保护单必须用「平仓方向」：多头→SELL、空头→BUY；若误用开仓方向（多头=BUY），
        # 触发价会相对现价立即触发（Order would immediately trigger）而被交易所拒绝
        if ins.stop_loss is not None or ins.take_profit is not None:
            protective_side = "SELL" if ins.action == OrderAction.OPEN_LONG else "BUY"
            try:
                sl_tp = self._place_stop_loss_take_profit(
                    ins, qty, protective_side,
                    protective_position_side=self._ps(
                        "LONG" if ins.action == OrderAction.OPEN_LONG else "SHORT"
                    ),
                )
                result["sl_tp_orders"] = sl_tp
            except BinanceError as exc:
                logger.warning(
                    "%s 挂保护单失败（限价单可能未成交）: %s", ins.symbol, exc
                )
                result["sl_tp_error"] = str(exc)
        return result

    def _place_stop_loss_take_profit(
        self,
        ins: TradeInstruction,
        qty: float,
        protective_side: str,
        protective_position_side: Optional[str] = None,
    ) -> list[dict]:
        """为仓位挂 STOP_MARKET / TAKE_PROFIT_MARKET 保护单（Algo Order API）。

        支持分批止盈：单个 take_profit 平全仓；take_profits 多档则每档按比例 qty×fraction 独立挂单。
        每档独立下单，个别失败（如交易所限「每仓位 1 TP + 1 SL」时第二档被拒）不阻断其余，
        失败信息记入 placed 对应项的 "fail" 字段，由调用方日志披露。

        protective_side：保护单方向（多头仓位→SELL 保护；空头仓位→BUY 保护）。
        protective_position_side：双向模式下保护单所属仓位方向 LONG/SHORT（单向为 None）。
        双向模式(Hedge Mode)禁传 reduceOnly（由 positionSide 指定仓位方向）；
        单向模式传 positionSide=BOTH + reduceOnly=True。
        """
        placed: list[dict] = []
        symbol_info = self.client.get_symbol_info(ins.symbol)
        if self.hedge_mode:
            algo_position_side = protective_position_side or "BOTH"
            reduce_only = None
        else:
            algo_position_side = "BOTH"
            reduce_only = True

        def _place(kind: str, otype: str, trigger: float, oqty: float, tag: str) -> None:
            try:
                order = self.client.place_algo_order(
                    symbol=ins.symbol,
                    side=protective_side,
                    order_type=otype,
                    quantity=oqty,
                    trigger_price=trigger,
                    position_side=algo_position_side,
                    client_algo_id=self._client_order_id(tag),
                    reduce_only=reduce_only,
                )
                placed.append({"kind": kind, "order": order,
                               "order_id": order.get("algoId")})
            except Exception as exc:  # noqa: BLE001
                placed.append({"kind": kind, "fail": str(exc)})

        if ins.stop_loss is not None:
            _place("stop_loss", "STOP_MARKET",
                   self._price(symbol_info, ins.stop_loss), qty, "sl")

        tps = ins.take_profits or []
        if tps:
            for i, lv in enumerate(tps, 1):
                oqty = self._quantity(symbol_info, qty * lv.fraction)
                if oqty <= 0:
                    placed.append({"kind": f"take_profit_{i}",
                                   "fail": "按比例折算后数量为0，忽略"})
                    continue
                _place(f"take_profit_{i}", "TAKE_PROFIT_MARKET",
                       self._price(symbol_info, lv.price), oqty, f"tp{i}")
        elif ins.take_profit is not None:
            _place("take_profit", "TAKE_PROFIT_MARKET",
                   self._price(symbol_info, ins.take_profit), qty, "tp")

        return placed

    def _close(
        self, ins: TradeInstruction, pos, mark_price: float, base: dict
    ) -> dict:
        if pos is None:
            base.update(status="REJECTED", error="无持仓")
            return base
        if ins.action == OrderAction.CLOSE_LONG and pos.position_amt <= 0:
            base.update(status="REJECTED", error="无多仓")
            return base
        if ins.action == OrderAction.CLOSE_SHORT and pos.position_amt >= 0:
            base.update(status="REJECTED", error="无空仓")
            return base

        close_side = "SELL" if pos.position_amt > 0 else "BUY"
        qty = self._quantity_for_close(ins, pos)
        return self._do_close(ins, pos, close_side, qty, base)

    def _flatten(self, ins: TradeInstruction, pos, mark_price: float, base: dict) -> dict:
        if pos is None:
            base.update(status="REJECTED", error="无持仓")
            return base
        close_side = "SELL" if pos.position_amt > 0 else "BUY"
        qty = abs(pos.position_amt)
        return self._do_close(ins, pos, close_side, qty, base)

    def _quantity_for_close(self, ins: TradeInstruction, pos) -> float:
        """平仓数量：优先指令值，缺省为全部持仓。"""
        if ins.quantity is not None and ins.quantity > 0:
            return min(abs(ins.quantity), abs(pos.position_amt))
        return abs(pos.position_amt)

    def _do_close(self, ins: TradeInstruction, pos, close_side: str, qty: float, base: dict) -> dict:
        symbol_info = self.client.get_symbol_info(ins.symbol)
        qty = self._quantity(symbol_info, qty)
        if qty <= 0:
            base.update(status="REJECTED", error="平仓数量无效")
            return base

        order_type = "MARKET"
        price = None
        if ins.order_type == OrderType.LIMIT and ins.price:
            order_type = "LIMIT"
            price = self._price(symbol_info, ins.price)

        # 全部平完则清除该币种风险额度锁（供下一轮开仓重新固化基线）
        fully_closed = qty >= abs(pos.position_amt)
        if fully_closed:
            self._locks.clear(ins.symbol)

        if self.dry_run:
            logger.info(
                "[DRY_RUN] %s %s qty=%s side=%s", ins.symbol, ins.action.value, qty, close_side
            )
            base.update(status="DRY_RUN", side=close_side, quantity=qty, order_type=order_type, price=price)
            return base

        order = self.client.close_position(
            symbol=ins.symbol,
            side=close_side,
            quantity=qty,
            order_type=order_type,
            price=price,
            position_side=self._ps(self._pos_side(pos)),
        )
        base.update(status="CLOSED", side=close_side, quantity=qty,
                    order=order, order_id=order.get("orderId"))
        return base
