"""反思者（Reflector）。

对应 TradingAgents 的 Reflector 职责：复盘本轮决策与执行结果，
自主决定是否向经验库写入 / 修改 / 删除经验条目，供以后交易参考。
"""
from __future__ import annotations

import json
import logging
from typing import Optional

from agents.llm import LLMClient
from agents.schemas import Reflection
from config import config

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are the Reflector of a crypto perpetual futures trading desk.
After each trading round you review the decision, execution and account outcome, then
decide how to update the autonomous experience library.

Your job:
1. Honestly assess this round: was the decision well-grounded? did execution succeed?
   Did the account suffer a severe loss (e.g. single-round loss > 5% of equity, or a stop-loss triggered)?
2. If you identify a lesson (a loss, a mistake, an inefficiency, a repeating market pattern),
   WRITE it into the experience library with a concrete, reusable rule for the future.
3. If an existing experience is now outdated or wrong, UPDATE or DELETE it (you have full authority).
4. Be concise and actionable. The library is read by future decision rounds, so write rules,
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


class Reflector:
    """每轮执行后运行，输出反思并更新经验库。"""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

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
        """复盘本轮并输出经验库操作。"""
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
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": "".join(user_parts)},
        ]
        data = self.llm.chat_json(messages, temperature=0.3, label=f"反思者 {self.llm.model}")
        try:
            reflection = Reflection.model_validate(data)
        except Exception as exc:  # noqa: BLE001
            logger.error("反思 JSON 校验失败: %s\n原始: %s", exc, json.dumps(data, ensure_ascii=False)[:800])
            raise
        return reflection
