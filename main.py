"""交易系统主入口。

用法：
  python main.py            # 循环模式（按 INTERVAL_MINUTES 轮询）
  python main.py --once     # 只执行一轮
  python main.py --interval 30   # 覆盖轮询间隔（分钟）

每一轮流程（顺序固定）：
1. 最先获取账户现状（余额/持仓/未实现盈亏）+ 未成交挂单（普通订单 + 算法条件单）
2. 提前取价与精度 → 构建「各品种最少初始保证金」上下文
3. 操作理由列表：自动清理过期条目（仓位完全平掉 / 挂单撤销且不再续挂），
   渲染当前进行中操作的理由列表
4. 拉取行情 + 指标 + 新闻 → 市场分析师报告
5. 决策者（接收市场报告 + 新闻 + 理由列表 + 最少保证金 + 经验库 + 账户现状）→ 结构化举措
6. 风控校验 → 逐条确认 → 执行
7. 应用模型对理由列表的操作（thesis_ops：ADD/UPDATE/DELETE）+ 自动同步补录/清理
8. 确认者复盘本轮结果（反思者职责已并入确认者），直接用行动写入/修改/删除经验库条目
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from typing import Optional

from agents.confirmer import Confirmer
from agents.decision_maker import DecisionMaker, format_account_context
from agents.llm import AllLLMUnavailable, LLMClient
from agents.market_analyst import MarketAnalyst
from agents.schemas import (
    ConfirmationAction, ExperienceAction, OrderAction,
    ThesisAction, TradingDecision, WakeCondition,
)
from config import config
from trading.analyst_state import AnalystStateStore
from trading.binance_client import BinanceClient
from trading.experience import ExperienceStore
from trading.executor import OrderExecutor
from trading.hypothesis import ThesisStore
from trading.market import MarketDataService
from trading.news import NewsService
from trading.risk import RiskManager, build_min_margin_context
from trading.risk_lock import RiskLockStore
from trading.rounds import RoundLog, RuntimeState
from trading.watch import WatchStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("trading_system")


def _format_account_ctx(account, open_orders_by_symbol=None) -> str:
    """账户现状 markdown（余额/持仓/未成交挂单），供确认者与反思者使用。"""
    return format_account_context(account, open_orders_by_symbol)


def _merge_open_orders(symbol: str, client: BinanceClient) -> list[dict]:
    """合并某币种普通订单与算法条件单（止盈止损），并统一为普通订单字段形状。

    2025-12-09 起 STOP_MARKET/TAKE_PROFIT_MARKET 走 /fapi/v1/algoOrder，
    查询/撤销需用 algoId；此处将算法单字段映射成 orderId/type/stopPrice 等
    统一形状，供决策者/确认者/结果展示直接使用。
    """
    orders: list[dict] = []
    try:
        orders = list(client.get_open_orders(symbol))
    except Exception as exc:  # noqa: BLE001
        logger.warning("获取 %s 普通挂单失败: %s", symbol, exc)
    try:
        for ao in client.get_open_algo_orders(symbol):
            orders.append({
                "orderId": ao.get("algoId"),
                "clientOrderId": ao.get("clientAlgoId"),
                "symbol": ao.get("symbol"),
                "side": ao.get("side"),
                "positionSide": ao.get("positionSide"),
                "type": ao.get("orderType") or ao.get("type"),
                "price": ao.get("price"),
                "stopPrice": ao.get("triggerPrice"),
                "origQty": ao.get("quantity"),
                "executedQty": "0",
                "reduceOnly": ao.get("reduceOnly"),
                "status": ao.get("algoStatus") or ao.get("status"),
                "is_algo": True,
            })
    except Exception as exc:  # noqa: BLE001
        logger.warning("获取 %s 条件单(Algo)失败: %s", symbol, exc)
    return orders


def _live_open_order_ids(open_orders_by_symbol: dict[str, list]) -> set[str]:
    """收集当前仍存活的未成交钉单 id（orderId / algoId），统一为 str 集合。

    用于操作理由（thesis）的「绑定钉单号存活」校验：若某条理由绑定的单号
    已不在存活钉单中且无对应持仓，则系统自动清除该理由（防模型遗忘清理）。
    """
    live: set[str] = set()
    for orders in open_orders_by_symbol.values():
        for o in orders:
            oid = o.get("orderId")
            if oid is not None:
                live.add(str(oid))
            cid = o.get("clientOrderId")
            if cid:
                live.add(str(cid))
    return live


class TradingSystem:
    """整合行情→分析→决策→风控→执行→假设记录的主流程。"""

    def __init__(self, interval_minutes: Optional[int] = None) -> None:
        self.interval_minutes = interval_minutes or config.interval_minutes
        self.client = BinanceClient(
            api_key=config.binance_api_key,
            api_secret=config.binance_api_secret,
            testnet=config.binance_testnet,
        )
        # 两档模型分工（仿照 TradingAgents 的 deep/quick 分工）：
        # - quick_think_llm（LLM_MODEL）：市场分析师、反思者 —— 快速任务
        # - deep_think_llm（LLM_DEEP_MODEL）：决策者/研究经理 —— 复杂推理
        # 两者都挂同一个备用 LLM（LLM_BACKUP_*）：主 API 失效时自动切换
        backup_llm = None
        if config.llm_backup_model:
            backup_llm = LLMClient(
                api_key=config.llm_backup_api_key or config.llm_api_key,
                base_url=config.llm_backup_base_url or config.llm_base_url,
                model=config.llm_backup_model,
            )
            logger.info("已启用备用 LLM: %s", backup_llm.model)
        self.llm = LLMClient(fallback=backup_llm)
        self.llm_deep = (
            LLMClient(model=config.llm_deep_model, fallback=backup_llm)
            if config.llm_deep_model
            else self.llm
        )
        self.market_data = MarketDataService(self.client)
        self.news_service = NewsService()
        self.market_analyst = MarketAnalyst(self.llm)
        self.decision_maker = DecisionMaker(self.llm_deep)
        # 执行前逐条确认者（用 quick 模型：单条指令的轻量复核）；
        # 反思者职责已并入确认者（每轮后由确认者复盘并直接维护经验库）
        self.confirmer = Confirmer(self.llm)
        # 共享风险额度锁：开仓固化基线、风控校验读到同一份、平仓清除
        self.risk_locks = RiskLockStore()
        self.risk = RiskManager(lock_store=self.risk_locks)
        # 启动时检查持仓模式：单向(One-way)则切换为双向(Hedge Mode)
        self.hedge_mode = False
        try:
            self.hedge_mode = self._init_hedge_mode()
        except Exception as exc:  # noqa: BLE001
            logger.warning("检查持仓模式失败（按单向模式运行）: %s", exc)
        self.executor = OrderExecutor(
            self.client, self.risk, hedge_mode=self.hedge_mode,
            lock_store=self.risk_locks,
        )
        self.theses = ThesisStore(
            max_age_hours=config.thesis_max_age_hours,
            max_items=config.thesis_max_items,
        )
        self.experiences = ExperienceStore(max_items=config.experience_max_items)
        self.watch_store = WatchStore(max_age_hours=config.watch_max_age_hours)
        self.round_log = RoundLog()
        self.runtime = RuntimeState()
        # 分析师跨轮状态：持久化上一轮观点，供下一轮对照标注 bias_change / 记翻转日志
        self.analyst_state = AnalystStateStore()
        self.symbols = config.symbols
        self._last_reflection: Optional[dict] = None
        # 本轮被「唤醒条件」提前唤醒时的触发快照（供决策者检查其它条件是否也接近触发）
        self._wake_snapshot: Optional[list[dict]] = None
        self._web_controller: Optional[WebServerController] = None
        if config.web_enabled:
            from web import WebServerController

            self._web_controller = WebServerController(
                host=config.web_host, port=config.web_port
            )
            self._web_controller.start()
        else:
            logger.info("状态面板未启用（WEB_ENABLED=false）")

    def _init_hedge_mode(self) -> bool:
        """检查账户持仓模式；单向则尝试切换为双向持仓。

        返回 True=双向模式（下单带 positionSide）；False=单向模式（切换失败或查询失败，
        系统按单向模式继续运行，不阻塞）。
        """
        try:
            dual = self.client.get_position_mode()
        except Exception as exc:  # noqa: BLE001
            logger.warning("查询持仓模式失败: %s", exc)
            return False
        if dual:
            logger.info("账户为双向持仓模式（Hedge Mode）")
            return True
        logger.warning("账户为单向持仓模式，尝试切换为双向持仓...")
        try:
            self.client.set_position_mode(True)
            logger.info("已切换为双向持仓模式（Hedge Mode）")
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "切换双向持仓失败（账户存在持仓或挂单时无法切换，需先平仓）: %s", exc
            )
            return False

    def run_once(self) -> dict:
        """执行一轮完整的 行情→分析→决策→风控→执行→假设记录。"""
        logger.info("===== 开始一轮交易决策 @ %s =====", datetime.now().isoformat())
        # 先持久化本轮开始时刻：程序中途关闭/崩溃后，下次启动可据此接续等待节奏
        try:
            self.runtime.mark_round_started()
        except Exception as exc:  # noqa: BLE001
            logger.warning("记录本轮开始时刻失败: %s", exc)
        result: dict = {
            "timestamp": datetime.now().isoformat(),
            "symbols": self.symbols,
            "testnet": config.binance_testnet,
            "dry_run": config.dry_run,
        }

        # 1. 最先获取账户现状（决策者必需）
        try:
            account = self.client.get_account()
            logger.info(
                "账户: 权益=%.2f 可用=%.2f 未实现盈亏=%.2f 持仓数=%d",
                account.margin_balance, account.available_balance,
                account.unrealized_pnl, len(account.positions),
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("获取账户失败: %s", exc)
            result["error"] = f"账户获取失败: {exc}"
            return result
        result["account"] = {
            "margin_balance": account.margin_balance,
            "available_balance": account.available_balance,
            "unrealized_pnl": account.unrealized_pnl,
            "positions": [
                {"symbol": p.symbol, "side": "LONG" if p.position_amt > 0 else "SHORT",
                 "qty": abs(p.position_amt), "entry": p.entry_price,
                 "mark": p.mark_price, "pnl": p.unrealized_pnl,
                 "leverage": p.leverage, "liq": p.liquidation_price}
                for p in account.positions
            ],
        }

        # 1.5 未成交挂单（决策者/确认者管理挂单、调止盈止损必需：含订单ID）
        # 普通订单 + 算法条件单（止盈止损）合并展示，统一字段形状
        open_orders_by_symbol: dict[str, list] = {}
        for sym in self.symbols:
            try:
                open_orders_by_symbol[sym] = _merge_open_orders(sym, self.client)
            except Exception as exc:  # noqa: BLE001
                logger.warning("获取 %s 未成交挂单失败: %s", sym, exc)
                open_orders_by_symbol[sym] = []
        result["open_orders"] = {
            sym: [
                {"orderId": o.get("orderId"), "side": o.get("side"), "type": o.get("type"),
                 "price": o.get("price"), "stopPrice": o.get("stopPrice"),
                 "qty": o.get("origQty"), "filled": o.get("executedQty"),
                 "reduceOnly": o.get("reduceOnly"), "status": o.get("status")}
                for o in orders
            ]
            for sym, orders in open_orders_by_symbol.items()
        }

        # 账户日志旁一并输出未成交挂单摘要，便于人工/日志一眼确认当前挂单状态
        pending_parts: list[str] = []
        pending_total = 0
        for sym in self.symbols:
            for o in open_orders_by_symbol.get(sym, []):
                pending_total += 1
                otype = o.get("type") or "-"
                oside = o.get("side") or "-"
                opx = o.get("price") or o.get("stopPrice") or "-"
                oqty = o.get("origQty") or "-"
                algo = "[algo]" if o.get("is_algo") else ""
                pending_parts.append(f"{sym} {oside} {otype}{algo} @{opx} x{oqty}")
        if pending_parts:
            logger.info("未成交挂单(%d): %s", pending_total, " | ".join(pending_parts))
        else:
            logger.info("未成交挂单: 无")

        # 2. 提前取价与精度（构建「最少初始保证金」上下文；后续风控校验复用）
        price_map: dict[str, float] = {}
        symbol_info_map = {}
        for sym in self.symbols:
            try:
                price_map[sym] = self.client.get_ticker_price(sym)
                symbol_info_map[sym] = self.client.get_symbol_info(sym)
            except Exception as exc:  # noqa: BLE001
                logger.warning("获取 %s 价格/精度失败: %s", sym, exc)
        min_margin_context = build_min_margin_context(
            self.symbols, symbol_info_map, price_map, config.max_leverage
        )
        result["min_margin_context"] = min_margin_context

        # 3. 操作理由列表：先自动清理过期条目（仓位已完全平掉 / 挂单撤销且不再续挂），
        #    再渲染当前进行中操作的理由列表，供决策者读取与操作（thesis_ops）
        account_positions = {p.symbol: p for p in account.positions}
        if not config.dry_run:
            try:
                pruned = self.theses.prune_stale(
                    account_positions,
                    open_orders_by_symbol,
                    live_open_order_ids=_live_open_order_ids(open_orders_by_symbol),
                )
                if pruned:
                    logger.info("操作理由列表已自动清理 %d 条过期条目", pruned)
            except Exception as exc:  # noqa: BLE001
                logger.warning("操作理由列表清理失败: %s", exc)
        thesis_context = self.theses.render_context()

        # 二-状态同步：HOLD/计划挂单状态核对。
        # 理由列表里 kind=limit_order 的条目标记了「计划挂单」，但交易所实际未成交挂单中
        # 可能根本没有对应订单（例如下单被风控拦截、或从未成功提交）。这类"计划 vs 实际"
        # 不一致会导致决策者继续假设挂单存在（如"维持 ETH 限价 1910 挂单"），造成回踩
        # 入场被静默丢失。这里逐条核对：绑定 order_id 缺失、或绑定单号已不在存活挂单中的
        # 挂单理由，一律注入告警到决策上下文，提示决策者核实/重挂/更新理由。
        try:
            live_ids_now = _live_open_order_ids(open_orders_by_symbol)
            ghost_orders = []
            for thesis in self.theses.all():
                if thesis.get("kind") != "limit_order":
                    continue
                if not thesis.get("symbol"):
                    continue
                bound_id = thesis.get("order_id")
                if not bound_id or str(bound_id) not in live_ids_now:
                    ghost_orders.append(thesis)
            if ghost_orders:
                alert = "\n\n## 计划挂单状态告警（理由列表 vs 交易所实际订单不一致）"
                alert += (
                    "\n以下 limit_order 理由在交易所未成交挂单中**找不到对应实际订单**"
                    "（无绑定单号，或绑定单号已不复存活）。决策者不应再假设这些挂单仍然存在；"
                    "请核实情况并决定：重新挂单 / 按当前行情调整 / 更新或删除该理由："
                )
                for t in ghost_orders:
                    alert += (
                        f"\n- [{t.get('id')}] {t.get('symbol')} @ "
                        f"{t.get('entry_price')}"
                        f"（方向: {t.get('direction')}；绑定单号: {t.get('order_id')}）"
                    )
                logger.warning(
                    "计划挂单状态核对：发现 %d 条理由在交易所无对应实际订单", len(ghost_orders)
                )
                thesis_context += alert
        except Exception as exc:  # noqa: BLE001
            logger.warning("计划挂单状态核对失败: %s", exc)

        result["thesis_context"] = thesis_context

        # 4. 市场上下文（多币种多周期）+ 新闻
        try:
            market_context = self.market_data.build_market_context_for_symbols(
                self.symbols, limit=config.klines_limit
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("获取行情失败: %s", exc)
            result["error"] = f"行情获取失败: {exc}"
            return result
        result["market_context_len"] = len(market_context)

        try:
            news_context = self.news_service.build_news_context(self.symbols)
        except Exception as exc:  # noqa: BLE001
            logger.warning("获取新闻失败（继续分析）: %s", exc)
            news_context = "（新闻获取失败，本轮忽略新闻面）"
        result["news_context_len"] = len(news_context)

        # 5. 市场分析师产出结构化报告（多周期指标+K线 + 新闻 + 跨轮状态 + 现价）
        try:
            analyst_out = self.market_analyst.analyze(
                market_context,  # 多周期指标快照（布林/均线/K线，主输入）
                news_context,
                prior_context=self.analyst_state.format_prior_context(),
                current_prices=price_map,
            )
            market_report = self._analyst_to_text(analyst_out)
            logger.info("市场分析报告已生成（%d 字符）", len(market_report))
        except AllLLMUnavailable as exc:
            return self._handle_llm_outage(exc, result)
        except Exception as exc:  # noqa: BLE001
            logger.error("市场分析失败: %s", exc)
            result["error"] = f"市场分析失败: {exc}"
            return result
        result["market_report"] = market_report
        result["analyst_state"] = self._analyst_state_bytes(analyst_out)

        # 5.5 分析师翻转检测 + 持久化状态（供下一轮对照）
        #     在方向翻转时醒目记录日志，暴露入场级判断的漂移，避免"报告永远积极"
        try:
            previous_assets = self.analyst_state.prior_assets_by_symbol()
            # 先写回本轮（detect_flips 读当前 last），再做差异说明
            self.analyst_state.save_views(
                analyst_out.market_overview,
                [a.model_dump() for a in analyst_out.assets],
                [n.model_dump() for n in analyst_out.news_pricing],
            )
            self._log_analyst_changes(previous_assets, analyst_out.assets)
        except Exception as exc:  # noqa: BLE001
            logger.warning("分析师状态持久化失败（不影响本轮）: %s", exc)

        # 6. 决策者输出结构化举措（含账户现状 + 操作理由列表 + 最少保证金 + 经验库）
        #     新闻定价时效（分析师给出 priced_in_by_utc / window_hours）拼进 news_context，
        #     让规则16真正判断"未定价剩余空间"窗口是否仍有效，减少滞后错过/误追。
        try:
            experience_context = self.experiences.format_for_context()
            news_ctx_for_decider = self._append_news_timing(news_context, analyst_out)
            wake_ctx = self._render_wake_context(price_map)
            decision = self.decision_maker.decide(
                market_report, news_ctx_for_decider, thesis_context, account,
                experience_context=experience_context,
                open_orders_by_symbol=open_orders_by_symbol,
                min_margin_context=min_margin_context,
                last_round_feedback=self.runtime.last_feedback(),
                current_watch_context=wake_ctx,
            )
            logger.info(
                "决策: %d 条指令, %d 条理由操作, 评估=%s",
                len(decision.instructions),
                len(decision.thesis_ops),
                decision.market_assessment[:120],
            )
        except AllLLMUnavailable as exc:
            return self._handle_llm_outage(exc, result)
        except Exception as exc:  # noqa: BLE001
            # 保守默认：决策解析失败（重试后仍失败）时，维持原状继续本轮，
            # 不产生任何指令、不触碰操作理由、保留现有唤醒条件——避免整轮被丢弃
            # 而"静默丢失"计划挂单或错过风险审视窗口。
            logger.error("决策失败（含重试），采用保守默认维持原状: %s", exc)
            existing_watch = self.watch_store.all()
            decision = TradingDecision(
                market_assessment=(
                    "决策解析失败，本轮保守维持原状（不产生任何新指令，保留现有挂单/持仓）。"
                ),
                instructions=[],
                thesis_ops=[],
                risk_notes=(
                    "决策解析失败已按保守策略处理：不做任何操作，保留现有挂单与持仓，"
                    "不修改操作理由，保存现有唤醒条件。请在下轮核验计划挂单是否存在。"
                ),
                watch_conditions=[
                    WakeCondition(**{k: v for k, v in w.items()
                                     if k in ("symbol", "condition", "value", "note",
                                              "volume_mult", "duration_seconds")})
                    for w in existing_watch
                ],
            )
            logger.warning("决策失败已生成保守兜底（HOLD 维持原状），原因: %s", exc)
        result["market_assessment"] = decision.market_assessment
        result["risk_notes"] = decision.risk_notes
        result["thesis_ops"] = [op.model_dump() for op in decision.thesis_ops]
        # 7. 更新唤醒条件（模型可在正常循环外设定价格触发，全量替换）
        result["watch_conditions"] = [
            c.model_dump() for c in decision.watch_conditions
        ]
        try:
            self.watch_store.replace(decision.watch_conditions)
        except Exception as exc:  # noqa: BLE001
            logger.warning("更新唤醒条件失败: %s", exc)

        # 8. 风控校验（价格/精度已在第 2 步获取，此处复用）
        passed, risk_results = self.risk.validate_decision(
            decision.instructions, account, price_map, symbol_info_map
        )
        # 指令执行顺序（确定性拆分）：同一币种先平仓/减仓(CLOSE/FLATTEN)，再调整
        # 止盈止损(SET_SL_TP)——SET_SL_TP 无数量参数、作用于当前全部持仓，只有先
        # 减仓它才会自动保护剩余仓位；避免「依赖减仓却先执行 SET_SL_TP」导致风控落空。
        # 其余指令（开仓/挂单管理）保持模型原相对顺序，不受影响。
        _RISK_ORDER = {
            OrderAction.CLOSE_LONG: 0,
            OrderAction.CLOSE_SHORT: 0,
            OrderAction.FLATTEN: 0,
            OrderAction.SET_SL_TP: 1,
        }
        passed = sorted(
            passed,
            key=lambda i: (_RISK_ORDER.get(i.action, 2),),
        )
        result["risk_blocked"] = [
            {"symbol": r.errors[0].split(":")[0], "errors": r.errors}
            for r in risk_results if not r.ok
        ]
        result["instructions_after_risk"] = [
            {
                "symbol": i.symbol, "action": i.action.value,
                "order_type": i.order_type.value if i.order_type else None, "price": i.price,
                "quantity": i.quantity, "margin": i.margin, "leverage": i.leverage,
                "stop_loss": i.stop_loss, "take_profit": i.take_profit,
                "reason": i.reason,
            }
            for i in passed
        ]

        # 9. 逐条确认后执行：每条指令执行前调用确认者复核，
        #    确认过程中模型可 PROCEED / SKIP / REPLACE（输出修正后的新指令）
        exec_results: list[dict] = []
        executed_instructions: list = []  # 实际执行的指令（含 REPLACE 修正后）
        confirmations: list[dict] = []
        if passed:
            account_snapshot = account
            for ins in passed:
                # 刷新最新价格（前几条执行可能已影响行情判断）
                try:
                    mark = self.client.get_ticker_price(ins.symbol)
                except Exception:  # noqa: BLE001
                    mark = price_map.get(ins.symbol, 0)
                # 刷新账户（真实模式下前几条执行会改变持仓/余额）
                try:
                    account_snapshot = self.client.get_account()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("刷新账户失败（沿用轮起始快照）: %s", exc)
                # 刷新该币种未成交挂单（前几条指令可能已撤销/新挂单）
                try:
                    open_orders_by_symbol[ins.symbol] = _merge_open_orders(ins.symbol, self.client)
                except Exception:  # noqa: BLE001
                    pass
                prior_summary = "\n".join(
                    f"- {r.get('symbol')} {r.get('action')} -> {r.get('status')}"
                    for r in exec_results
                )
                # 执行前逐条确认
                try:
                    conf = self.confirmer.confirm(
                        ins, mark, _format_account_ctx(account_snapshot, open_orders_by_symbol),
                        prior_summary, decision.market_assessment,
                    )
                except AllLLMUnavailable as exc:
                    return self._handle_llm_outage(exc, result)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("指令确认失败，跳过 %s: %s", ins.symbol, exc)
                    exec_results.append({
                        "symbol": ins.symbol, "action": ins.action.value,
                        "status": "SKIPPED", "error": f"确认失败: {exc}",
                    })
                    continue
                confirmations.append({
                    "symbol": ins.symbol, "decision": conf.decision.value,
                    "reason": conf.reason,
                })
                if conf.decision == ConfirmationAction.SKIP:
                    logger.info("确认者 SKIP %s: %s", ins.symbol, conf.reason[:120])
                    exec_results.append({
                        "symbol": ins.symbol, "action": ins.action.value,
                        "status": "SKIPPED", "error": conf.reason,
                    })
                    continue

                final_ins = ins
                if conf.decision == ConfirmationAction.REPLACE:
                    new_ins = conf.instruction
                    info = symbol_info_map.get(new_ins.symbol)
                    m = price_map.get(new_ins.symbol, 0)
                    if info is None or m <= 0:
                        logger.warning("REPLACE 缺少 %s 价格/精度信息，跳过", new_ins.symbol)
                        exec_results.append({
                            "symbol": new_ins.symbol, "action": "REPLACE",
                            "status": "SKIPPED", "error": "缺少价格/精度信息",
                        })
                        continue
                    r = self.risk.validate_instruction(new_ins, account_snapshot, info, m)
                    if not r.ok:
                        logger.warning(
                            "REPLACE 新指令未过风控，跳过 %s: %s",
                            new_ins.symbol, "; ".join(r.errors),
                        )
                        exec_results.append({
                            "symbol": new_ins.symbol, "action": "REPLACE",
                            "status": "SKIPPED", "error": "; ".join(r.errors),
                        })
                        continue
                    final_ins = new_ins
                    logger.info("确认者 REPLACE %s -> 新指令已过风控", new_ins.symbol)

                # 执行单条
                try:
                    r = self.executor.execute([final_ins], account_snapshot)[0]
                except Exception as exc:  # noqa: BLE001
                    r = {"symbol": final_ins.symbol, "action": final_ins.action.value,
                         "status": "FAILED", "error": str(exc)}
                exec_results.append(r)
                executed_instructions.append(final_ins)
                order_id = r.get("order_id")
                logger.info(
                    "执行结果 %s %s -> %s%s%s",
                    r.get("symbol"), r.get("action"), r.get("status"),
                    " (DRY_RUN)" if config.dry_run else "",
                    f" orderId={order_id}" if order_id else "",
                )
            result["confirmations"] = confirmations
        else:
            logger.info("无通过风控的指令，本轮不交易")
        result["execution"] = exec_results

        # 9.5 保存上一轮执行反馈（风控拦截 / 执行结果），供下一轮决策者参考并纠正
        #     否则模型永远不知道上轮指令为何未成交（如保证金不足被拦截），会重复犯错
        try:
            self.runtime.set_feedback(
                self._build_round_feedback(result.get("risk_blocked", []), exec_results)
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("保存上一轮执行反馈失败: %s", exc)

        # 10. 应用模型对「操作理由列表」的操作（模型拥有完整操作权：ADD/UPDATE/DELETE），
        #     并同步列表：为新开仓/新挂单补录理由、按真实账户清理过期条目
        #     本轮实际执行成功（含 DRY_RUN）的开仓币种，用于校验模型 ADD 的持仓理由是否成立
        executed_open_symbols = {
            r.get("symbol") for r in exec_results
            if r.get("action") in ("OPEN_LONG", "OPEN_SHORT")
            and r.get("status") in ("OPENED", "DRY_RUN")
        }
        try:
            self._apply_thesis_ops(
                decision.thesis_ops,
                executed_open_symbols=executed_open_symbols,
                account=account,
                open_orders_by_symbol=open_orders_by_symbol,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("应用理由列表操作失败: %s", exc)
        try:
            self._sync_theses(executed_instructions, exec_results, account, price_map)
        except Exception as exc:  # noqa: BLE001
            logger.error("同步操作理由列表失败: %s", exc)
        result["thesis_count"] = self.theses.count()

        # 11. 确认者复盘本轮（反思者职责已并入确认者），直接维护经验库（写入/修改/删除）
        self._reflect_and_apply(
            decision, exec_results, account, thesis_context, market_report,
            open_orders_by_symbol=open_orders_by_symbol,
            risk_blocked=result.get("risk_blocked", []),
        )
        if self._last_reflection:
            result["reflection"] = self._last_reflection

        logger.info("===== 本轮结束 =====")
        return result

    def _handle_llm_outage(self, exc, result: dict) -> dict:
        """所有 LLM API 均无法正常通讯：立即市价平掉全部持仓，本轮结束。

        由 LLMClient 在「主→备→按间隔确认多次仍失败」后抛出的
        AllLLMUnavailable 触发；平仓不依赖 LLM，仅用币安 API（reduceOnly）。
        """
        logger.error("所有 LLM API 均不可用，触发紧急平仓: %s", exc)
        try:
            account = self.client.get_account()
            emerg = self.executor.flatten_all(account, reason="LLM API 全部不可用")
            result["emergency_flatten"] = emerg
            for r in emerg:
                logger.warning(
                    "紧急平仓 %s %s -> %s",
                    r.get("symbol"), r.get("side"), r.get("status"),
                )
        except Exception as e:  # noqa: BLE001
            logger.error("紧急平仓失败: %s", e)
            result["emergency_flatten_error"] = str(e)
        result["error"] = f"所有 LLM API 均无法正常通讯，已执行紧急平仓: {exc}"
        return result

    def _emergency_flatten(self, reason: str) -> None:
        """仅执行紧急平仓（无 result 上下文时用，如反思环节）。"""
        try:
            account = self.client.get_account()
            self.executor.flatten_all(account, reason=reason)
        except Exception as exc:  # noqa: BLE001
            logger.error("紧急平仓失败: %s", exc)

    @staticmethod
    def _build_round_feedback(risk_blocked: list[dict], exec_results: list[dict]) -> str:
        """将上一轮的风控拦截与执行结果整理成决策者可见的反馈文本（供下一轮参考）。

        模型此前完全不知道自己的指令为何未成交（例如保证金不足被风控拦截），
        导致每轮重复犯同样的错。此反馈在每轮结束后持久化，下一轮决策时回传，
        让模型知道上轮哪些指令被拦截/失败及原因，从而修正本轮决策。
        """
        lines: list[str] = []
        for blk in risk_blocked:
            sym = blk.get("symbol", "?")
            errs = blk.get("errors") or []
            lines.append(f"- [{sym}] 上轮指令被风控拦截：{'；'.join(errs)}")
        for ex in exec_results:
            sym = ex.get("symbol", "?")
            act = ex.get("action", "?")
            status = ex.get("status", "?")
            if status in ("DRY_RUN", "OPENED", "CLOSED", "CANCELLED", "REPLACED",
                          "ADJUSTED", "FAILED", "REJECTED", "SKIPPED"):
                detail = ex.get("error") or ""
                lines.append(
                    f"- [{sym}] {act} -> {status}" + (f"：{detail}" if detail else "")
                )
        if not lines:
            return ""
        return (
            "# 上一轮执行反馈（系统上一轮对指令的处理结果；"
            "若你的指令曾被拦截或执行失败，请务必据此修正本轮决策）\n"
            + "\n".join(lines)
        )

    @staticmethod
    def _analyst_to_text(analyst_out) -> str:
        """把分析师结构化输出拼成决策者/反思者可读的 markdown 报告。"""
        blocks = ["# 市场分析报告", "## 市场总览", analyst_out.market_overview, ""]
        for a in analyst_out.assets:
            blocks.append(f"## {a.symbol} — {a.bias.value}（信心 {a.confidence}）")
            blocks.append(f"- 相对上轮: {a.bias_change.value}")
            blocks.append(f"- 依据: {a.reason}")
            blocks.append(
                f"- 支撑区: {a.support_low:g}–{a.support_high:g} | "
                f"压力区: {a.resistance_low:g}–{a.resistance_high:g}"
            )
            entry_str = (
                f"{a.entry_from:g}–{a.entry_to:g}" if a.entry_from is not None else "无清晰信号"
            )
            t2 = f"{a.target_2:g}" if a.target_2 is not None else "无"
            blocks.append(
                f"- 建议入场: {entry_str} | "
                f"T1目标: {a.target_1:g} | T2目标: {t2} | 止损: {a.stop_price:g}"
            )
            blocks.append(f"- 分化可能: {a.divergence}")
            blocks.append("")
        if analyst_out.news_pricing:
            blocks.append("## 新闻定价与时效")
            for n in analyst_out.news_pricing:
                blocks.append(
                    f"- [{n.impact_direction}] {n.headline} → {n.priced_in}"
                    f"（预计完全消化@{n.priced_in_by_utc}，窗口 {n.window_hours:g}h）"
                    f" {n.remaining_space}"
                )
            blocks.append("")
        return "\n".join(blocks).strip()

    @staticmethod
    def _analyst_state_bytes(analyst_out) -> dict:
        """精简的分析师状态摘要（供 result/展示，不含过重文本）。"""
        return {
            "market_overview": analyst_out.market_overview,
            "assets": [
                {
                    "symbol": a.symbol, "bias": a.bias.value,
                    "bias_change": a.bias_change.value, "confidence": a.confidence,
                    "support_low": a.support_low, "support_high": a.support_high,
                    "resistance_low": a.resistance_low, "resistance_high": a.resistance_high,
                }
                for a in analyst_out.assets
            ],
        }

    def _log_analyst_changes(self, previous_assets: dict[str, dict], curr_assets) -> None:
        """对比上轮与本轮观点，记日志：翻转时醒目告警，暴露入场级判断漂移。"""
        for cur in curr_assets:
            sym = cur.symbol
            prev = previous_assets.get(sym)
            bc = cur.bias_change.value
            if prev is None:
                logger.info("分析师 %s: 首次给出观点 bias=%s（NEW）", sym, cur.bias.value)
                continue
            prev_bias = prev.get("bias")
            cur_bias = cur.bias.value
            if prev_bias != cur_bias and prev_bias in ("LONG", "SHORT") and cur_bias in ("LONG", "SHORT"):
                logger.warning(
                    "分析师观点翻转 %s: %s -> %s（%s）。上一轮支撑/压力=[%s,%s]/[%s,%s]，"
                    "本轮=[%s,%s]/[%s,%s]。依据: %s",
                    sym, prev_bias, cur_bias, bc,
                    prev.get("support_low"), prev.get("support_high"),
                    prev.get("resistance_low"), prev.get("resistance_high"),
                    cur.support_low, cur.support_high,
                    cur.resistance_low, cur.resistance_high,
                    cur.reason[:160],
                )
            elif prev_bias != cur_bias:
                logger.info(
                    "分析师观点调整 %s: %s -> %s（%s）", sym, prev_bias, cur_bias, bc,
                )

    @staticmethod
    def _append_news_timing(news_context: str, analyst_out) -> str:
        """把新闻定价时效追加到新闻上下文，供决策者判断窗口。

        由代码层基于当前 UTC 时间算出每条新闻的「剩余小时数」，直接以纯数字塞给模型，
        避免模型自行把绝对 UTC 时间换算成小时（LLM 时间算术不可靠）。模型只做定性判断：
        剩余 ≤0 → 已消化；>0 且剩余空间明确 → 窗口仍开启。
        """
        if not analyst_out.news_pricing:
            return news_context
        now = datetime.now(timezone.utc)
        block = [
            "",
            "",
            "# 新闻定价时效（剩余小时数已由系统按当前 UTC 时间算好，直接看数字即可，无需换算）",
            "",
        ]
        for n in analyst_out.news_pricing:
            rh = None
            raw = getattr(n, "priced_in_by_utc", None)
            if raw:
                try:
                    t = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
                    rh = (t - now).total_seconds() / 3600.0
                except (ValueError, TypeError):
                    rh = None
            if rh is None:
                window_txt = "（时间解析失败，请以分析师原始时间自行判断）"
            elif rh <= 0:
                window_txt = f"剩余 {rh:.1f} 小时 → 已过期/已消化，不再追"
            else:
                window_txt = f"剩余 {rh:.1f} 小时 → 窗口仍开启"
            block.append(
                f"- [{n.impact_direction}] {n.headline}：{n.priced_in}"
                f"；剩余空间={n.remaining_space}；{window_txt}。"
            )
        block.append(
            "\n以上剩余小时数由系统实时计算：≤0 视为已消化不再追；"
            ">0 且剩余空间明确才可能利用，且仍须尊重技术面、于支撑/压力位入场、不追高。"
        )
        return news_context + "\n".join(block)

    def _reflect_and_apply(
        self,
        decision,
        exec_results: list[dict],
        account,
        thesis_context: str,
        market_report: str,
        open_orders_by_symbol: dict | None = None,
        risk_blocked: list[dict] | None = None,
    ) -> None:
        """由确认者复盘本轮（原反思者职责），并应用经验库操作。"""
        # 复盘执行结果，失败/降级不阻断主流程
        try:
            account_positions = {p.symbol: p for p in account.positions}
            prev_ctx = self.theses.render_drift_check(account_positions)
            decision_summary = json.dumps({
                "market_assessment": decision.market_assessment,
                "instructions": [
                    {"symbol": i.symbol, "action": i.action.value,
                     "quantity": i.quantity, "margin": i.margin, "leverage": i.leverage,
                     "stop_loss": i.stop_loss, "take_profit": i.take_profit,
                     "reason": i.reason}
                    for i in decision.instructions
                ],
            }, ensure_ascii=False, indent=2)
            reflection = self.confirmer.reflect(
                market_report=market_report,
                decision_summary=decision_summary,
                execution_results=exec_results,
                account_context=_format_account_ctx(account, open_orders_by_symbol),
                experience_context=self.experiences.format_for_context(),
                previous_context=prev_ctx,
                risk_blocked=risk_blocked,
            )
            self._last_reflection = {
                "self_assessment": reflection.self_assessment,
                "severe_loss": reflection.severe_loss,
            }
            logger.info("确认者-反思评估: %s", reflection.self_assessment[:120])
            for op in reflection.experience_ops:
                self._apply_experience_op(op)
        except AllLLMUnavailable as exc:
            # 反思环节 LLM 全部不可用：本轮刚执行过交易，立即紧急平仓撤退
            logger.error("反思环节所有 LLM 均不可用，执行紧急平仓: %s", exc)
            self._emergency_flatten("反思环节 LLM API 全部不可用")
        except Exception as exc:  # noqa: BLE001
            logger.warning("反思环节失败（不影响本轮交易）: %s", exc)

    def _apply_experience_op(self, op) -> None:
        """应用单条经验库操作。"""
        action = op.action
        try:
            if action == ExperienceAction.WRITE:
                if not op.title or not op.content:
                    logger.warning("WRITE 操作缺少标题或内容，已忽略")
                    return
                eid = self.experiences.add(
                    category=op.category or "市场观察",
                    title=op.title,
                    content=op.content,
                )
                logger.info("经验库 WRITE #%s", eid)
            elif action == ExperienceAction.UPDATE:
                ok = self.experiences.update(op.experience_id, **{
                    "category": op.category,
                    "title": op.title,
                    "content": op.content,
                })
                if not ok:
                    logger.warning("经验库 UPDATE 失败: id=%s 不存在", op.experience_id)
            elif action == ExperienceAction.DELETE:
                ok = self.experiences.delete(op.experience_id)
                if not ok:
                    logger.warning("经验库 DELETE 失败: id=%s 不存在", op.experience_id)
            elif action == ExperienceAction.NONE:
                pass
            else:
                logger.warning("未知经验库操作: %s", action)
        except Exception as exc:  # noqa: BLE001
            logger.error("应用经验库操作失败 %s: %s", action, exc)

    def _apply_thesis_ops(
        self, ops, executed_open_symbols=None, account=None, open_orders_by_symbol=None
    ) -> None:
        """应用模型对「操作理由列表」的操作（模型拥有完整操作权：ADD/UPDATE/DELETE）。

        系统侧防护（经验执行化：不依赖模型自觉读经验文本，执行层强制生效）：
        - kind=position：理由必须与该币种真实持仓、或本轮实际执行成功的开仓对应；
        - kind=limit_order：理由必须对应交易所中真实存活的 LIMIT 未成交挂单（含本轮
          刚执行成功的开仓挂单）。若挂单被风控拦截、从未提交，理由列表不得记录
          "幽灵挂单"，否则后续轮次会误以为该挂单仍存在（如"维持 ETH 限价挂单"）。

        凡指令被风控拦截/从未成交（无真实挂单、无真实持仓、也无本轮开仓成功），
        一律拒绝写入，避免把并不存在的操作记入理由列表误导后续轮次。
        """
        account_positions = {p.symbol for p in account.positions} if account else set()
        executed_open_symbols = executed_open_symbols or set()
        # 交易所中当前存活未成交的 LIMIT 挂单所覆盖的币种集合（用于校验挂单理由是否真实）
        open_orders_by_symbol = open_orders_by_symbol or {}
        live_limit_symbols = {
            o.get("symbol") for orders in open_orders_by_symbol.values()
            for o in orders if o.get("type") == "LIMIT" and o.get("symbol")
        }
        for op in ops:
            try:
                if op.action == ThesisAction.ADD:
                    kind = op.kind or "position"
                    if kind == "position":
                        if (
                            op.symbol not in account_positions
                            and op.symbol not in executed_open_symbols
                        ):
                            logger.warning(
                                "理由列表 ADD 拒绝：%s position 未实际开仓（指令被拦截或"
                                "未成交），不记录不存在的仓位",
                                op.symbol,
                            )
                            continue
                    elif kind == "limit_order":
                        # 该币种需存在真实存活 LIMIT 挂单，或本轮刚执行成功的开仓挂单
                        if (
                            op.symbol not in live_limit_symbols
                            and op.symbol not in executed_open_symbols
                        ):
                            logger.warning(
                                "理由列表 ADD 拒绝：%s limit_order 在交易所无真实存活"
                                " LIMIT 挂单（可能被风控拦截或从未提交），不记录幽灵挂单",
                                op.symbol,
                            )
                            continue
                    self.theses.add(
                        symbol=op.symbol,
                        kind=op.kind or "position",
                        parent_id=op.parent_id,
                        direction=op.direction,
                        entry_price=op.entry_price,
                        stop_loss=op.stop_loss,
                        take_profit=op.take_profit,
                        order_id=op.order_id,
                        thesis=op.thesis,
                        note=op.note,
                    )
                elif op.action == ThesisAction.UPDATE:
                    fields = {
                        k: v for k, v in {
                            "kind": op.kind, "direction": op.direction,
                            "entry_price": op.entry_price, "stop_loss": op.stop_loss,
                            "take_profit": op.take_profit, "order_id": op.order_id,
                            "thesis": op.thesis,
                            "note": op.note, "parent_id": op.parent_id,
                        }.items() if v is not None
                    }
                    if not self.theses.update(op.thesis_id, **fields):
                        logger.warning("理由列表 UPDATE 失败: id=%s 不存在", op.thesis_id)
                elif op.action == ThesisAction.COMPLETE:
                    # 结束该操作：标注完成记号，系统自动级联删除该编号及其全部子编号的理由
                    if not self.theses.complete(op.thesis_id):
                        logger.warning("理由列表 COMPLETE 失败: id=%s 不存在", op.thesis_id)
                elif op.action == ThesisAction.DELETE:
                    if not self.theses.remove(op.thesis_id):
                        logger.warning("理由列表 DELETE 失败: id=%s 不存在", op.thesis_id)
                elif op.action == ThesisAction.NONE:
                    pass
                else:
                    logger.warning("未知理由列表操作: %s", op.action)
            except Exception as exc:  # noqa: BLE001
                logger.error("应用理由列表操作失败 %s: %s", op.action, exc)

    def _sync_theses(
        self,
        passed: list,
        exec_results: list[dict],
        account,
        price_map: dict[str, float],
    ) -> None:
        """执行后同步操作理由列表。

        1. 自动补录：本轮成功执行的开仓/挂单，若该币种尚无理由记录则补一条
           （限价开仓记为 limit_order 理由，市价开仓记为 position 理由）；
        2. 真实模式下：挂单成交后自动将 limit_order 理由升级为 position 理由，
           并按真实账户状态清理过期条目（仓位完全平掉 / 挂单撤销且不再续挂）。
        """
        # 1. 自动补录
        instruction_map = {i.symbol: i for i in passed}
        for r in exec_results:
            symbol = r.get("symbol")
            ins = instruction_map.get(symbol)
            if ins is None:
                continue
            if r.get("action") not in ("OPEN_LONG", "OPEN_SHORT"):
                continue
            if r.get("status") not in ("DRY_RUN", "OPENED"):
                continue
            # 该币种已有理由记录（可能由模型 thesis_ops ADD 过），不重复补录
            if self.theses.by_symbol(symbol):
                continue
            direction = "LONG" if r.get("action") == "OPEN_LONG" else "SHORT"
            is_limit = r.get("order_type") == "LIMIT"
            entry = ins.price or price_map.get(symbol, 0)
            self.theses.add(
                symbol=symbol,
                kind="limit_order" if is_limit else "position",
                direction=direction,
                entry_price=entry,
                stop_loss=ins.stop_loss,
                take_profit=ins.take_profit,
                order_id=r.get("order_id"),
                thesis=ins.reason,
            )
            logger.info("已为 %s %s 自动补录操作理由(绑定单号=%s)",
                        symbol, "挂单" if is_limit else "开仓", r.get("order_id"))

        # 2. 真实模式：挂单成交→升级为持仓理由；并按真实账户清理过期条目
        #    注意：此处必须重新获取最新账户（而非执行前快照），否则刚开仓的仓位
        #    在 account 中看不到，导致 prune_stale 误判「仓位已平」而立即删除
        if not config.dry_run:
            try:
                fresh_account = self.client.get_account()
                account_positions = {p.symbol: p for p in fresh_account.positions}
            except Exception:
                account_positions = {p.symbol: p for p in account.positions}
            for th in self.theses.all():
                if th.get("kind") == "limit_order" and th.get("symbol") in account_positions:
                    pos = account_positions[th["symbol"]]
                    self.theses.update(
                        th["id"], kind="position",
                        direction="LONG" if pos.position_amt > 0 else "SHORT",
                        entry_price=pos.entry_price,
                    )
            try:
                open_orders_by_symbol = {
                    sym: _merge_open_orders(sym, self.client) for sym in self.symbols
                }
            except Exception:  # noqa: BLE001
                open_orders_by_symbol = {}
            removed = self.theses.prune_stale(
                account_positions,
                open_orders_by_symbol,
                live_open_order_ids=_live_open_order_ids(open_orders_by_symbol),
            )
            if removed:
                logger.info("操作理由列表自动清理 %d 条过期条目", removed)

    def run_loop(self) -> None:
        """循环模式：每 interval 分钟执行一轮；若模型设定唤醒条件，条件满足时提前执行。"""
        logger.info(
            "启动循环模式，间隔 %d 分钟（测试网=%s DRY_RUN=%s 条件唤醒=%s）",
            self.interval_minutes, config.binance_testnet, config.dry_run,
            config.watch_enabled,
        )
        # 程序恢复：若上一轮分析时间未到下一轮间隔，则接上等待（仅启动时执行一次）
        self._resume_wait_if_needed()
        while True:
            started = time.time()
            try:
                result = self.run_once()
                try:
                    self.round_log.append(result)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("记录轮次历史失败: %s", exc)
            except Exception as exc:  # noqa: BLE001
                logger.exception("循环中出现未处理异常: %s", exc)
            elapsed = time.time() - started
            sleep_sec = max(self.interval_minutes * 60 - elapsed, 10)
            if config.watch_enabled and self.watch_store.count() > 0:
                if self._wait_for_trigger_or_sleep(sleep_sec):
                    logger.info("唤醒条件触发，提前开始新一轮分析")
                    continue
                logger.info("等待 %d 秒无唤醒条件触发，按计划进入下一轮", sleep_sec)
            else:
                logger.info("下一轮将在 %.0f 秒后进行", sleep_sec)
                time.sleep(sleep_sec)

    def _resume_wait_if_needed(self) -> None:
        """程序关闭再开启后的恢复：接上上一轮的等待节奏。

        若上一轮分析开始时刻距今仍未到下一轮分析时间（interval 未到点），
        则继续等待剩余时长，而不是立即重新分析，从而延续修改前的决定；
        已到点或无记录则直接开始新一轮。等待期间仍响应模型设定的唤醒条件。
        """
        last = self.runtime.last_round_at()
        if last is None:
            logger.info("无上一轮记录，直接开始新一轮分析")
            return
        interval_sec = self.interval_minutes * 60
        remaining = interval_sec - (time.time() - last)
        if remaining <= 0:
            logger.info(
                "上一轮分析于 %.0f 秒前，已到下一轮分析时间，直接开始",
                interval_sec - remaining,
            )
            return
        logger.info(
            "程序恢复：上一轮分析于 %.0f 秒前，距下一轮还有 %.0f 秒，接上等待",
            interval_sec - remaining, remaining,
        )
        if config.watch_enabled and self.watch_store.count() > 0:
            if self._wait_for_trigger_or_sleep(remaining):
                logger.info("唤醒条件触发，提前开始新一轮分析")
        else:
            time.sleep(remaining)

    def _wait_for_trigger_or_sleep(self, sleep_sec: float) -> bool:
        """等待 sleep_sec 秒，期间按 WATCH_CHECK_INTERVAL 轮询唤醒条件；满足则提前返回 True。"""
        deadline = time.time() + sleep_sec
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                return False
            wait = min(max(config.watch_check_interval, 1), remaining)
            time.sleep(wait)
            if self._check_watch_triggers():
                return True

    def _check_watch_triggers(self) -> bool:
        """轮询当前价并核对唤醒条件；满足则统一唤醒一次。

        不再以「量能不足」为由跳过唤醒：价格触达阈值即计入待唤醒，是否放量不再作为门槛
        （由 duration_seconds 滤除单根 K 线的瞬时刺穿）。

        唤醒冷却：距上次唤醒不足 watch_wake_cooldown（默认 10 分钟）秒时，本次触发的
        条件只累积不立即唤醒，等冷却结束后与其它已触发条件一次性统一唤醒，避免短时间
        内因多个条件接连触发而频繁唤醒。

        触发真正唤醒前，把当前全部唤醒条件快照到 self._wake_snapshot（而不是直接丢弃），
        让被唤醒的这一轮决策者能看到「其它唤醒条件是否也已接近触发」，据此取消或调整，
        从而在代码冷却之外，从提示词层面进一步避免短时间多次唤醒。
        """
        triggered: list[tuple[dict, float]] = []
        for t in self.watch_store.all():
            try:
                price = self.client.get_ticker_price(t["symbol"])
            except Exception as exc:  # noqa: BLE001
                logger.warning("检查 %s 唤醒条件失败: %s", t["symbol"], exc)
                continue
            cond, val = t.get("condition"), t.get("value")
            hit = (cond == "price_above" and price >= val) or (
                cond == "price_below" and price <= val
            )
            if not hit:
                # 价格已回到阈值内：重置时间确认计时
                if "first_hit_at" in t:
                    del t["first_hit_at"]
                    self.watch_store.save()
                continue
            # 时间确认：价格需持续停留在阈值外 duration_seconds 秒（过滤瞬时刺穿）
            dur = int(t.get("duration_seconds") or 0)
            if dur > 0:
                now = time.time()
                first = t.get("first_hit_at")
                if first is None:
                    t["first_hit_at"] = now
                    self.watch_store.save()
                    continue  # 首次命中，等待后续轮询累计持续时间
                if now - float(first) < dur:
                    continue  # 尚未达到要求持续时间，暂不触发
            triggered.append((t, price))
        if not triggered:
            return False
        # 唤醒冷却：距上次唤醒不足 cooldown 秒则累计条件、暂不唤醒，待冷却后统一触发一次
        now = time.time()
        last = getattr(self, "_last_wakeup_at", 0.0)
        cooldown = float(config.watch_wake_cooldown)
        if now - last < cooldown:
            logger.info(
                "距上次唤醒仅 %.0fs，未到 %ds 冷却期，暂缓唤醒（已累积 %d 个触发待冷却后统一一次唤醒）",
                now - last, cooldown, len(triggered),
            )
            return False
        self._last_wakeup_at = now
        self._wake_snapshot = list(self.watch_store.all())
        detail = "; ".join(
            f"{t['symbol']} {t['condition']}@{t['value']:g}（现价 {price:.6g}）"
            for t, price in triggered
        )
        logger.warning("唤醒条件触发: %s", detail)
        self.watch_store.clear_all(reason="条件已触发")
        return True

    def _render_wake_context(self, price_map: dict[str, float]) -> str:
        """渲染「被唤醒条件提前唤醒」这一轮的开头要求与当前唤醒条件清单。

        只有当本轮确实是被唤醒条件触发（_wake_snapshot 非空）时才注入,并在渲染后
        清除快照（一次性）。内容：
        1. 提示词最前要求：先检查其它唤醒条件是否也已接近触发,有则取消/调整,
           避免短时间多次唤醒（与代码 10 分钟冷却互为补充）。
        2. 逐条列出唤醒条件与该币现价,给出距离与触发方向,便于模型判断谁接近、谁再等。
        """
        snapshot = getattr(self, "_wake_snapshot", None)
        if not snapshot:
            return ""
        self._wake_snapshot = None  # 一次性消费
        lines = [
            "【本轮为唤醒条件触发，请先处理唤醒清单】",
            "你这次是被下方某个「唤醒条件」（价格触达）而非正常定时循环提前唤醒的。",
            "系统已强制设定冷却：被唤醒后 10 分钟内无法再次唤醒，多个条件即便同时满足也统一",
            "在冷却结束后一次性唤醒。请在第一优先位置完成下述动作，再继续正常分析：",
            "1. 逐一对照下方「当前唤醒条件清单」，判断其中是否还有其它条件也已接近触发"
            "（现价距离其阈值很近、随时会再次触达）。",
            "2. 若有，请取消或调整它们（在下方输出的 watch_conditions 中，或删除、或把阈值拉远、"
            "或合并到同一个价位的触发），避免短期内在冷却结束后又因不同条件接连唤醒；",
            "只保留真正必要且与当前分析判断一致的唤醒条件。",
            "3. 若无接近触发的其它条件，维持精简的唤醒条件即可。",
            "",
            "当前唤醒条件清单（含现价对比，duration_seconds 为时间确认，满足才触发）：",
        ]
        for t in snapshot:
            sym = t.get("symbol", "?")
            cond = t.get("condition", "?")
            val = t.get("value")
            dur = int(t.get("duration_seconds") or 0)
            px = price_map.get(sym)
            px_str = f"{px:.6g}" if px is not None else "?"
            if px is not None:
                if val:
                    diff = px - float(val)
                    pct = abs(diff / float(val)) * 100
                    close = "（接近触发！）" if pct < 1.0 else ""
                    lines.append(
                        f"- {sym}: {cond}@{val:g}，现价 {px_str}（距阈值 {pct:.2f}%{close}）"
                        + (f"，需持续 {dur}s" if dur else "")
                    )
                else:
                    lines.append(f"- {sym}: {cond}@{val:g}，现价 {px_str}")
            else:
                lines.append(
                    f"- {sym}: {cond}@{val:g}，现价未知" + (f"，需持续 {dur}s" if dur else "")
                )
        return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="币安永续合约 AI 交易系统")
    parser.add_argument("--once", action="store_true", help="只执行一轮")
    parser.add_argument("--interval", type=int, default=None, help="循环间隔(分钟)")
    args = parser.parse_args()

    try:
        config.validate()
    except Exception as exc:  # noqa: BLE001
        logger.error("配置校验失败: %s", exc)
        raise SystemExit(1)

    system = TradingSystem(interval_minutes=args.interval)
    if args.once:
        result = system.run_once()
        try:
            system.round_log.append(result)
        except Exception as exc:  # noqa: BLE001
            logger.warning("记录轮次历史失败: %s", exc)
        # 简要输出决策摘要
        print("\n===== 决策摘要 =====")
        print("市场评估:", result.get("market_assessment", "N/A")[:300])
        for ins in result.get("instructions_after_risk", []):
            print(
                f"- {ins['symbol']} {ins['action']} "
                f"type={ins['order_type']} margin={ins.get('margin')} "
                f"qty={ins.get('quantity')} price={ins['price']} "
                f"lev={ins['leverage']} sl={ins['stop_loss']} tp={ins['take_profit']}"
            )
        if result.get("risk_blocked"):
            print("\n被风控拦截的指令:")
            for b in result["risk_blocked"]:
                print(f"- {b['symbol']}: {'; '.join(b['errors'])}")
        if result.get("reflection"):
            print("\n===== 自我反省 =====")
            refl = result["reflection"]
            print(f"严重亏损: {refl.get('severe_loss')}")
            print(f"评估: {refl.get('self_assessment', '')[:300]}")
        print(f"\n经验库条目数: {system.experiences.count()}（state/experience_library.json）")
    else:
        system.run_loop()


if __name__ == "__main__":
    main()
