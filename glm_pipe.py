"""
title: GLM Power Pipe (Z.ai)
id: glm_power_pipe
author: Alexis Pietak (https://github.com/pietakio/)
version: 0.1.0
license: MIT
requirements: pydantic>=2.0.0, openai>=1.0.0
environment_variables:
    - ZAI_API_KEY (optional, can be set in Valves)

A minimal OpenWebUI pipe for Z.ai's GLM models (e.g. GLM-5.2).

GLM speaks an OpenAI-compatible API, so this pipe is deliberately small: it
just points the OpenAI Async client at the Z.ai base URL and injects GLM's
non-standard `thinking` parameter so reasoning effort is controllable from
the UI. Runs completely independently of the Anthropic Power Pipe — they
coexist with no conflict.

Supports:
- Streaming responses
- Thinking mode with configurable reasoning_effort (default: max)
- Reasoning stream rendered into an OpenWebUI <think> dropdown
- Per-user API key override
"""

import os
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional
from typing import Literal

from pydantic import BaseModel, Field

try:
    from openai import AsyncOpenAI
except ImportError:  # graceful degradation if SDK missing
    AsyncOpenAI = None  # type: ignore

logger = logging.getLogger("glm_power_pipe")


class Pipe:
    # Z.ai OpenAI-compatible endpoint. The OpenAI SDK appends /chat/completions.
    DEFAULT_BASE_URL = "https://api.z.ai/api/paas/v4/"

    # Models exposed by this pipe. Add entries here as Z.ai ships new models.
    MODELS = [
        {"id": "glm-5.2", "name": "GLM-5.2"},
    ]

    class Valves(BaseModel):
        ZAI_API_KEY: str = Field(
            default=os.getenv("ZAI_API_KEY", ""),
            description="Z.ai API key. Falls back to the ZAI_API_KEY environment variable.",
        )
        ZAI_BASE_URL: str = Field(
            default="",
            description="Custom base URL for the Z.ai API. Leave empty to use the default (https://api.z.ai/api/paas/v4/).",
        )
        MAX_TOKENS: int = Field(
            default=128000,
            ge=1,
            le=128000,
            description="Maximum output tokens. GLM-5.2 supports up to 128K output.",
        )
        REQUEST_TIMEOUT: int = Field(
            default=600,
            ge=10,
            le=3600,
            description="Request timeout in seconds. Reasoning at max effort can take a while.",
        )

    class UserValves(BaseModel):
        ZAI_API_KEY: str = Field(
            default="",
            description="Personal Z.ai API key. If set, overrides the admin-configured key.",
        )
        ENABLE_THINKING: bool = Field(
            default=True,
            description="Enable GLM thinking mode (chain-of-thought reasoning before the answer).",
        )
        REASONING_EFFORT: Literal["high", "max"] = Field(
            default="max",
            description="Reasoning effort when thinking is enabled. GLM-5.2 supports two levels: 'max' pushes reasoning hardest; 'high' balances quality and latency.",
        )
        TEMPERATURE: float = Field(
            default=1.0,
            ge=0.0,
            le=2.0,
            description="Sampling temperature.",
        )
        SHOW_REASONING: bool = Field(
            default=True,
            description="Render the reasoning stream in a collapsible <think> dropdown.",
        )

    def __init__(self):
        self.type = "manifold"
        self.id = "glm"
        self.valves = self.Valves()
        self.logger = logger

    def pipes(self) -> List[dict]:
        return [{"id": m["id"], "name": m["name"]} for m in self.MODELS]

    def _resolve_model_id(self, raw_model: str) -> str:
        """Strip the OpenWebUI manifold prefix (e.g. 'glm.glm-5.2' -> 'glm-5.2')."""
        if not raw_model:
            return self.MODELS[0]["id"]
        # OWUI joins pipe id and model id with a '.'; the model id itself contains
        # no dot, so take everything after the first '.' that follows the pipe id.
        prefix = f"{self.id}."
        if raw_model.startswith(prefix):
            return raw_model[len(prefix):]
        # Fallback: last path-ish segment
        return raw_model.split(".")[-1]

    def _client(self, api_key: str) -> "AsyncOpenAI":
        base_url = self.valves.ZAI_BASE_URL.strip() or self.DEFAULT_BASE_URL
        return AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=self.valves.REQUEST_TIMEOUT,
        )

    async def pipe(
        self,
        body: Dict[str, Any],
        __user__: Dict[str, Any],
        __event_emitter__: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
        __metadata__: Optional[Dict[str, Any]] = None,
        __task__: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        if AsyncOpenAI is None:
            return "Error: the `openai` package is not installed. Add `openai>=1.0.0` to the pipe requirements."

        async def emit_status(description: str, done: bool = False):
            if __event_emitter__:
                await __event_emitter__(
                    {"type": "status", "data": {"description": description, "done": done}}
                )

        # --- Resolve API key (UserValves override admin Valves) ---
        user_valves = __user__.get("valves") if __user__ else None
        user_key = getattr(user_valves, "ZAI_API_KEY", "") if user_valves else ""
        api_key = (user_key or "").strip() or self.valves.ZAI_API_KEY.strip()
        if not api_key:
            return "Error: No Z.ai API key configured. Set it in admin Valves or your personal UserValves."

        # --- Resolve per-user settings (fall back to UserValves defaults) ---
        uv = user_valves or self.UserValves()
        enable_thinking = getattr(uv, "ENABLE_THINKING", True)
        reasoning_effort = getattr(uv, "REASONING_EFFORT", "max")
        temperature = getattr(uv, "TEMPERATURE", 1.0)
        show_reasoning = getattr(uv, "SHOW_REASONING", True)

        model_id = self._resolve_model_id(body.get("model", ""))
        messages = body.get("messages", [])
        stream = body.get("stream", True)

        # GLM's non-standard params go through extra_body when using the OpenAI SDK.
        # Per the Z.ai docs, `thinking` and `reasoning_effort` are SIBLINGS, not nested:
        #   thinking={"type": "enabled"}, reasoning_effort="max"
        extra_body: Dict[str, Any] = {
            "thinking": {"type": "enabled" if enable_thinking else "disabled"}
        }
        if enable_thinking:
            extra_body["reasoning_effort"] = reasoning_effort

        request_kwargs: Dict[str, Any] = {
            "model": model_id,
            "messages": messages,
            "max_tokens": self.valves.MAX_TOKENS,
            "temperature": temperature,
            "stream": stream,
            "extra_body": extra_body,
        }

        client = self._client(api_key)

        # ----------------------- Non-streaming path -----------------------
        if not stream:
            try:
                resp = await client.chat.completions.create(**request_kwargs)
                choice = resp.choices[0].message
                reasoning = getattr(choice, "reasoning_content", None)
                content = choice.content or ""
                if reasoning and show_reasoning:
                    return f"<think>\n{reasoning}\n</think>\n\n{content}"
                return content
            except Exception as exc:
                logger.exception("GLM non-streaming request failed")
                await emit_status(f"Error: {exc}", done=True)
                return f"Error calling Z.ai: {exc}"

        # ------------------------- Streaming path --------------------------
        async def response_stream():
            in_think = False
            try:
                await emit_status("Contacting GLM…")
                response = await client.chat.completions.create(**request_kwargs)
                async for chunk in response:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta

                    # GLM streams chain-of-thought in `reasoning_content`.
                    reasoning = getattr(delta, "reasoning_content", None)
                    if reasoning and show_reasoning:
                        if not in_think:
                            yield "<think>\n"
                            in_think = True
                        yield reasoning

                    content = getattr(delta, "content", None)
                    if content:
                        if in_think:
                            yield "\n</think>\n\n"
                            in_think = False
                        yield content

                if in_think:  # close an unterminated think block
                    yield "\n</think>\n\n"
                await emit_status("", done=True)
            except Exception as exc:
                logger.exception("GLM streaming request failed")
                if in_think:
                    yield "\n</think>\n\n"
                await emit_status(f"Error: {exc}", done=True)
                yield f"\n\nError calling Z.ai: {exc}"

        return response_stream()
