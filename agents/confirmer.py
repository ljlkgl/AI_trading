"""执行前确认者（Confirmer）。

对应需求：模型输出完整 JSON 决策后，每执行一条指令前再调用模型确认一次，
确认过程中模型可输出新动作（REPLACE 修正指令），以便及时调整参数错误。

职责（已合并反思者）：
1. 执行前复核：对每条指令做二次把关。优先给出「修正建议」(REPLACE)，而非简单
   SKIP——当指令依赖的前置动作（如减仓）未完成、或参数需要调整时，必须产出
   可执行的替代指令，避免整条风控动作落空。
2. 轮次反思：每轮执行结束后复盘本轮决策与执行结果，直接用行动落实经验库维护
   （写入/修改/删除经验条目），不再依赖独立反思者模块。

使用 quick_think_llm（快速模型）：确认是单条指令的轻量复核，决策主体仍在
决策者（deep 模型）完成；本环节只做执行前的二次把关与轮次后的经验沉淀。
"""
from __future__ import annotations

import json
import logging
from typing import Optional

from agents.llm import LLMClient
from agents.schemas import InstructionConfirmation, Reflection, TradeInstruction
from config import config

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are the Confirmer of a crypto perpetual futures trading desk.
The Portfolio Manager has already produced the full decision JSON. Your ONLY job is a quick
execution-time review of EACH instruction right before it is actually placed on Binance.

Goal: catch parameter mistakes (price, quantity/margin, direction, stop-loss/take-profit too close to
current price, margin vs available balance, etc.) and correct them in time. You CORRECT, you do not
merely approve or reject.

Decide ONE of:
1. PROCEED: the instruction is reasonable as-is — execute it unchanged. (default when no clear problem)
2. REPLACE: the instruction needs a fix — provide the corrected COMPLETE instruction
   (same fields as a TradingDecision instruction). Common fixes: adjust stop_loss further from
   current price, reduce margin (for OPEN), correct a wrong direction, fix a LIMIT price,
   re-issue a protective instruction that a failed prerequisite would otherwise void.
3. SKIP: ONLY when the instruction is harmful AND you cannot produce any safe alternative.
   SKIP is the LAST resort, never the default. When you skip a protective/risk instruction
   (SET_SL_TP / CLOSE / FLATTEN), you MUST first try to REPLACE it with a safe fallback.

Rules:
- Do NOT modify instructions without a concrete reason tied to the data you see (price/account).
  When in doubt, PROCEED.
- For any OPEN action keep the hard constraints: margin>0 (initial margin in USDT) AND
  margin ≤ available_balance (see account snapshot) AND margin ≥ the minimum initial margin
  for the chosen leverage (see "各品种最少初始保证金" table if provided), leverage within cap,
  stop_loss mandatory & on the correct side (LONG: sl < price < tp; SHORT: sl > price > tp).
  For OPEN, quantity is derived by the system (quantity = margin × leverage / price), do NOT edit it.
- For CLOSE_LONG/CLOSE_SHORT (partial close allowed): if quantity is given it must be positive and
  must NOT exceed the current position size shown in the account snapshot; a quantity strictly smaller
  than the position means a PARTIAL close (e.g. close 50% = half the position). If quantity is omitted
  the whole position will be closed. Do NOT reject a CLOSE merely because it is partial.
- For REPLACE_LIMIT keep quantity>0 (coin quantity) and a valid LIMIT price.
- For REPLACE you MUST output the full corrected instruction JSON (including margin for OPEN).
- SET_SL_TP has NO quantity parameter BY DESIGN: the system always applies it to the FULL current
  position. If the reason mentions "先减仓" (reduce first) but the position is still full (the prior
  reduce failed or did not execute), STILL PROCEED when a position exists and the new stop_loss /
  take_profit are valid relative to the current price — protective levels on the full position are the
  minimum acceptable risk control and are far better than no protection. Only REPLACE if the LEVELS
  themselves are wrong (too close to price, or on the wrong side). NEVER SKIP a protective instruction
  merely because its reason referenced a prerequisite action that has not happened yet.
- Dependency handling: instructions execute in the order given; a CLOSE/FLATTEN before SET_SL_TP for
  the same symbol means the SET_SL_TP naturally covers the remaining (smaller) position. If a prior
  instruction FAILED, look at the CURRENT account snapshot provided and always leave a valid
  protective order in place for any open position.

Output ONLY valid JSON:
{
  "decision": "PROCEED" | "SKIP" | "REPLACE",
  "instruction": null | { ...corrected TradeInstruction... },
  "reason": "brief justification referencing concrete numbers"
}
No markdown fences, no extra text.
"""

_REFLECTION_PROMPT = """You are the Confirmer/Reflector of a crypto perpetual futures trading desk
(the Reflector role has been merged into you).
After each trading round you review the decision, execution and account outcome, then
decide how to update the autonomous experience library. You must ACT on lessons, not just note them.

Your job:
1. Honestly assess this round: was the decision well-grounded? did execution succeed?
   Did the account suffer a severe loss (e.g. single-round loss > 5% of equity, or a stop-loss triggered)?
2. If you identify a lesson (a loss, a mistake, an inefficiency, a repeating market pattern),
   WRITE it into the experience library with a concrete, reusable rule for the future.
3. If an existing experience is now outdated or wrong, UPDATE or DELETE it (you have full authority).
4. If a risk/protective action was SKIPped or FAILED this round (e.g. an order rejected by the exchange,
   a SET_SL_TP left unplaced, a stop-loss not set), WRITE an experience entry about the root cause so the
   desk does not repeat it.
5. Be concise and actionable. The library is read by future decision rounds, so write rules,
   not narratives.

Output strictly as JSON:
{
  "self_assessment": "...",
  "severe_loss": true/false,
  "experience_ops": [
    {"action": "WRITE", "category": "亏损教训", "title": "...", "content": "发生了什么|原因|以后如何避免"},
    {"action": "UPDATE", "experience_id": "abc123", "content": "新正文"},
    {"action": "DELETE", "experience_id": "abc123"}
  ]
}
Use action NONE with an empty ops list when no change is needed.
Output ONLY valid JSON. No markdown fences, no extra text.
"""


class Confirmer:
    """单条指令执行前确认 + 轮次反思（接管原反思者职责）。"""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def confirm(
        self,
        instruction: TradeInstruction,
        mark_price: float,
        account_context: str,
        prior_exec_summary: str,
        market_assessment: str = "",
    ) -> InstructionConfirmation:
        """确认一条指令，返回决定（可含替换指令）。"""
        user_parts = [
            "待确认指令（即将执行，请复核）：\n",
            json.dumps(instruction.model_dump(), ensure_ascii=False, indent=2),
            f"\n\n该币种当前标记价格: {mark_price:.6g}",
            "\n\n最新账户现状：\n",
            account_context,
        ]
        if prior_exec_summary and prior_exec_summary.strip():
            user_parts.extend(["\n\n本轮已执行指令结果（供参考是否重复/冲突）：\n", prior_exec_summary])
        if market_assessment:
            user_parts.extend(["\n\n决策者市场判断：\n", market_assessment[:300]])
        user_parts.append(
            "\n\n请输出确认决定（PROCEED / REPLACE / SKIP）。"
            "若需 REPLACE，请给出修正后的完整指令 JSON；"
            "请优先给出修正建议，不要轻易 SKIP。"
        )

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": "".join(user_parts)},
        ]
        data = self.llm.chat_json(
            messages,
            temperature=config.llm_temperature,
            label=f"确认者 {self.llm.model}（复核 {instruction.symbol} {instruction.action.value}）",
        )
        try:
            return InstructionConfirmation.model_validate(data)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "确认 JSON 校验失败: %s\n原始: %s",
                exc, json.dumps(data, ensure_ascii=False)[:800],
            )
            raise

    def reflect(
        self,
        market_report: str,
        decision_summary: str,
        execution_results: list[dict],
        account_context: str,
        experience_context: str,
        previous_context: Optional[str] = None,
        risk_blocked: Optional[list[dict]] = None,
    ) -> Reflection:
        """复盘本轮并输出经验库操作（原反思者职责，已并入确认者）。"""
        user_parts = [
            "市场分析报告（本轮）：\n", market_report[:2000],
            "\n\n本轮决策摘要：\n", decision_summary,
            "\n\n执行结果：\n",
            json.dumps(execution_results, ensure_ascii=False, indent=2)[:2000],
            "\n\n当前账户现状：\n", account_context,
            "\n\n当前自主经验库：\n", experience_context,
        ]
        if risk_blocked:
            user_parts.append(
                "\n\n本轮被风控拦截的指令"
                "（重要：若本轮未产生任何订单，请先确认是否因下述风控拦截所致，"
                "不要把它误判为「系统静默不执行」）：\n"
            )
            user_parts.append(
                json.dumps(risk_blocked, ensure_ascii=False, indent=2)[:1500]
            )
        if previous_context:
            user_parts.extend(["\n\n上轮持仓假设（供对照）：\n", previous_context])
        user_parts.append(
            "\n\n请输出对经验库的自主操作（可写/改/删），并给出自我评估。"
        )

        messages = [
            {"role": "system", "content": _REFLECTION_PROMPT},
            {"role": "user", "content": "".join(user_parts)},
        ]
        data = self.llm.chat_json(messages, temperature=0.3, label=f"确认者-反思 {self.llm.model}")
        try:
            reflection = Reflection.model_validate(data)
        except Exception as exc:  # noqa: BLE001
            logger.error("反思 JSON 校验失败: %s\n原始: %s", exc, json.dumps(data, ensure_ascii=False)[:800])
            raise
        return reflection
