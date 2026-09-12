"""LLM 客户端：OpenAI 兼容接口，支持结构化 JSON 输出。

复用 TradingAgents 的思路：模型输出必须遵循指定 JSON 格式，
这里通过 response_format=json_object + 强约束 prompt 实现，
并对输出做 pydantic 校验与重试。
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Optional

from openai import OpenAI

from config import config

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """LLM 调用或解析失败。"""


class AllLLMUnavailable(LLMError):
    """主 LLM 与备用 LLM 均无法正常通讯（已按间隔确认多次）。"""


class LLMClient:
    """轻量 OpenAI 兼容客户端。

    故障切换：
    1. 主 LLM 调用失败（含内部重试）→ 自动切到备用 LLM（fallback）；
    2. 备用也失败 → 按 confirm_interval 间隔尝试 confirm_attempts 次（每次先主后备）；
    3. 全部失败 → 抛 AllLLMUnavailable，由主流程触发紧急平仓。
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_retries: int = 2,
        reasoning_effort: Optional[str] = None,
        fallback: Optional["LLMClient"] = None,
        confirm_attempts: Optional[int] = None,
        confirm_interval: Optional[int] = None,
    ) -> None:
        self.api_key = api_key or config.llm_api_key
        self.base_url = base_url or config.llm_base_url or None
        self.model = model or config.llm_model
        self.temperature = temperature
        self.max_retries = max_retries
        # 推理强度三档（仿照 TradingAgents reasoning_effort）：low / medium / high，空=不设置
        effort = (reasoning_effort or config.llm_reasoning_effort).strip().lower()
        self.reasoning_effort = effort if effort in ("low", "medium", "high") else ""
        # 备用 LLM 与连通性确认参数
        self.fallback = fallback
        self.confirm_attempts = confirm_attempts or config.llm_emergency_attempts
        self.confirm_interval = (
            confirm_interval if confirm_interval is not None else config.llm_emergency_interval
        )
        if not self.api_key:
            raise LLMError("LLM_API_KEY 未配置")
        if not self.model:
            raise LLMError("LLM_MODEL 未配置")
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def chat(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        json_mode: bool = False,
        stream: bool = True,
        label: str = "",
    ) -> str:
        """发送对话，返回文本。

        主 LLM 优先；失败自动切备用；两者均失败则按间隔确认多次，
        仍不可用则抛 AllLLMUnavailable（由主流程触发紧急平仓）。
        """
        try:
            return self._chat_impl(messages, temperature, json_mode, stream, label)
        except LLMError as exc:
            if self.fallback is not None:
                try:
                    logger.warning(
                        "主 LLM(%s) 调用失败，切换到备用 LLM(%s): %s",
                        self.model, self.fallback.model, exc,
                    )
                    return self.fallback._chat_impl(
                        messages, temperature, json_mode, stream, label
                    )
                except LLMError as fb_exc:
                    exc = fb_exc
            # 主/备用均不可用 → 连通性确认
            logger.error(
                "主/备用 LLM 均调用失败，开始连通性确认（共 %d 次，每次间隔 %d 秒）",
                self.confirm_attempts, self.confirm_interval,
            )
            clients = (self, self.fallback) if self.fallback is not None else (self,)
            for i in range(self.confirm_attempts):
                time.sleep(self.confirm_interval)
                for client in clients:
                    try:
                        logger.warning(
                            "连通性确认第 %d/%d 次：尝试 %s",
                            i + 1, self.confirm_attempts, client.model,
                        )
                        return client._chat_impl(
                            messages, temperature, json_mode, stream, label
                        )
                    except LLMError:
                        continue
            raise AllLLMUnavailable(
                f"所有 LLM API 均无法正常通讯（已按 {self.confirm_interval}s 间隔"
                f"尝试 {self.confirm_attempts} 次）: {exc}"
            ) from exc

    def _chat_impl(
        self,
        messages: list[dict],
        temperature: Optional[float],
        json_mode: bool,
        stream: bool,
        label: str,
    ) -> str:
        """单次调用实现（不含故障切换），供 chat() 与备用 LLM 复用。"""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
        }
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if label:
            print(f"\n===== {label} =====", flush=True)
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                content = self._create_completion(kwargs, stream=stream, json_mode=json_mode)
                return content
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if "reasoning_effort" in kwargs:
                    # 部分兼容接口（如 SenseNova/部分 DeepSeek 网关）不支持该参数，
                    # 移除后降级重试一次，避免整轮失败
                    logger.warning(
                        "接口不支持 reasoning_effort 参数，已降级重试: %s", exc
                    )
                    kwargs.pop("reasoning_effort")
                    continue
                logger.warning("LLM 调用失败（第 %d 次）: %s", attempt + 1, exc)
        raise LLMError(f"LLM 调用重试后仍失败: {last_exc}")

    def _create_completion(
        self, kwargs: dict, stream: bool, json_mode: bool
    ) -> str:
        """执行一次 completion；流式时逐块打印。"""
        if stream:
            try:
                stream_resp = self.client.chat.completions.create(
                    **kwargs, stream=True
                )
            except Exception as exc:  # noqa: BLE001
                # 部分兼容接口不支持流式，回退到非流式
                logger.warning("流式请求失败，回退非流式: %s", exc)
                return self._create_completion(kwargs, stream=False, json_mode=json_mode)

            pieces: list[str] = []
            for chunk in stream_resp:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                token = getattr(delta, "content", None) or ""
                if token:
                    pieces.append(token)
                    print(token, end="", flush=True)
            print(flush=True)  # 结束换行
            content = "".join(pieces)
            if json_mode:
                content = _extract_json(content)
                json.loads(content)  # 校验合法 JSON
            return content

        resp = self.client.chat.completions.create(**kwargs)
        content = resp.choices[0].message.content or ""
        if json_mode:
            content = _extract_json(content)
            json.loads(content)
        return content

    def chat_json(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        label: str = "",
    ) -> dict:
        """返回解析后的 dict。"""
        content = self.chat(
            messages, temperature=temperature, json_mode=True, label=label
        )
        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            raise LLMError(f"模型输出非法 JSON: {exc}\n原文: {content[:500]}") from exc
        if not isinstance(data, dict):
            raise LLMError("模型输出 JSON 不是对象")
        return data


def _extract_json(text: str) -> str:
    """从模型文本中提取 JSON 对象（处理代码块包裹等情况）。"""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        return fence.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return text[start : end + 1]
    return text
