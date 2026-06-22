"""
title: GLM (OpenRouter) Power Pipe
author: pietakio
version: 0.1.0
license: MIT
description: GLM 5.2 (and other approved models) via OpenRouter, hard-pinned to privacy-approved providers (Fireworks → Parasail), never Z.ai/China. Automatic Fireworks prompt caching; data_collection=deny.
requirements: openai>=1.0.0
required_open_webui_version: 0.9.0
"""

# =============================================================================
# GLM via OpenRouter — privacy-pinned manifold pipe for Open WebUI 0.9.x
# -----------------------------------------------------------------------------
# Why this exists:
#   GLM 5.2 is a Z.ai (China-hosted) model. To use it without exposing prompts
#   to Z.ai, we route through OpenRouter and HARD-PIN the request to approved
#   western providers (Fireworks first, Parasail as backup) with fallbacks
#   disabled, so a request fails closed rather than leaking to an unapproved
#   provider. We also set data_collection="deny" (only providers that do not
#   collect/train on your data).
#
# Caching:
#   Fireworks prompt caching is automatic and server-side (no cache_control
#   needed) and gives a real cost discount on the repeated prefix. There is no
#   configurable TTL (it is LRU/capacity-based, "minutes to hours"). We send
#   x-session-affinity (the chat id) to keep follow-up turns on the same
#   replica for better cache-hit reliability.
#
# Streaming (OWUI 0.9.x):
#   Content is streamed via the NATIVE {"type":"message"} event channel and the
#   full assembled text is RETURNED so OWUI persists it as the saved message.
#   (Mixing a raw chat:completion delta channel with native events wipes the
#   saved message on 0.9.x — learned the hard way on the Anthropic pipe.)
# =============================================================================

import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

try:
    from openai import AsyncOpenAI
except ImportError:  # graceful degradation if the SDK is missing
    AsyncOpenAI = None

logger = logging.getLogger("glm_openrouter_pipe")


class Pipe:
    class Valves(BaseModel):
        OPENROUTER_API_KEY: str = Field(
            default="",
            description="Your OpenRouter API key (sk-or-...).",
        )
        OPENROUTER_BASE_URL: str = Field(
            default="https://openrouter.ai/api/v1",
            description="OpenRouter OpenAI-compatible base URL.",
        )
        MODELS: str = Field(
            default="z-ai/glm-5.2 | GLM 5.2 (OpenRouter)",
            description=(
                "Models to expose, one per line as 'slug | Display Name'. "
                "Slug must be a valid OpenRouter model id (e.g. z-ai/glm-5.2)."
            ),
        )
        ALLOWED_PROVIDERS: str = Field(
            default="fireworks,parasail",
            description=(
                "Comma-separated OpenRouter provider slugs in PRIORITY order. "
                "With fallbacks disabled, only these are used — a request fails "
                "rather than routing to any other provider (e.g. Z.ai)."
            ),
        )
        ALLOW_FALLBACKS: bool = Field(
            default=False,
            description=(
                "If False (recommended), OpenRouter will NOT fall back to "
                "providers outside ALLOWED_PROVIDERS. Fail-closed for privacy."
            ),
        )
        DATA_COLLECTION_DENY: bool = Field(
            default=True,
            description="Only route to providers that do not collect/train on your data (provider.data_collection='deny').",
        )
        ZERO_DATA_RETENTION: bool = Field(
            default=False,
            description="Stricter: only route to Zero-Data-Retention endpoints (provider.zdr=true). Verify Fireworks/Parasail support it first, or requests will fail.",
        )
        SESSION_AFFINITY: bool = Field(
            default=True,
            description="Send x-session-affinity (chat id) so follow-up turns stick to the same replica — improves prompt-cache hit rate.",
        )
        MAX_TOKENS: int = Field(
            default=0,
            description="Max output tokens. 0 = leave unset (provider default).",
        )
        SHOW_USAGE_INFO: bool = Field(
            default=True,
            description="Show a status banner with the serving provider and token/cache usage after each response.",
        )
        HTTP_REFERER: str = Field(
            default="",
            description="Optional OpenRouter HTTP-Referer attribution header.",
        )
        X_TITLE: str = Field(
            default="Open WebUI",
            description="Optional OpenRouter X-Title attribution header.",
        )
        REQUEST_TIMEOUT: int = Field(
            default=600,
            description="Per-request timeout in seconds.",
        )

    def __init__(self):
        self.valves = self.Valves()

    # ------------------------------------------------------------------ models
    def _model_list(self) -> List[Dict[str, str]]:
        models: List[Dict[str, str]] = []
        for raw in self.valves.MODELS.splitlines():
            line = raw.strip()
            if not line:
                continue
            if "|" in line:
                slug, name = line.split("|", 1)
            else:
                slug, name = line, line
            slug, name = slug.strip(), name.strip()
            if slug:
                models.append({"id": slug, "name": name or slug})
        if not models:
            models = [{"id": "z-ai/glm-5.2", "name": "GLM 5.2 (OpenRouter)"}]
        return models

    def pipes(self) -> List[dict]:
        return self._model_list()

    def _resolve_model(self, requested: str) -> str:
        """Map OWUI's prefixed model id (e.g. 'plugin.z-ai/glm-5.2') back to a slug."""
        known = self._model_list()
        for m in known:
            if requested == m["id"] or requested.endswith("." + m["id"]) or requested.endswith(m["id"]):
                return m["id"]
        # Fallback: strip a single leading 'plugin.' prefix if present
        return known[0]["id"]

    # ------------------------------------------------------------- request bits
    def _provider_block(self) -> dict:
        order = [p.strip() for p in self.valves.ALLOWED_PROVIDERS.split(",") if p.strip()]
        block: dict[str, Any] = {
            "order": order,
            "allow_fallbacks": self.valves.ALLOW_FALLBACKS,
        }
        if self.valves.DATA_COLLECTION_DENY:
            block["data_collection"] = "deny"
        if self.valves.ZERO_DATA_RETENTION:
            block["zdr"] = True
        return block

    def _extra_headers(self, __metadata__: Optional[dict]) -> dict:
        headers: dict[str, str] = {}
        if self.valves.HTTP_REFERER:
            headers["HTTP-Referer"] = self.valves.HTTP_REFERER
        if self.valves.X_TITLE:
            headers["X-Title"] = self.valves.X_TITLE
        if self.valves.SESSION_AFFINITY and __metadata__:
            chat_id = __metadata__.get("chat_id")
            if chat_id:
                headers["x-session-affinity"] = str(chat_id)
        return headers

    def _client(self) -> "AsyncOpenAI":
        return AsyncOpenAI(
            api_key=self.valves.OPENROUTER_API_KEY,
            base_url=self.valves.OPENROUTER_BASE_URL.strip(),
            timeout=self.valves.REQUEST_TIMEOUT,
        )

    @staticmethod
    def _fmt(n: int) -> str:
        if n >= 1_000_000:
            return f"{n / 1_000_000:.1f}M"
        if n >= 1_000:
            return f"{n / 1_000:.1f}K"
        return str(n)

    # ------------------------------------------------------------------- entry
    async def pipe(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
        __metadata__: Optional[dict] = None,
        __task__: Optional[Any] = None,
        __request__: Optional[Any] = None,
    ):
        if AsyncOpenAI is None:
            return "Error: the 'openai' package is not installed in this Open WebUI environment."

        if not self.valves.OPENROUTER_API_KEY:
            return "Error: no OpenRouter API key configured. Set OPENROUTER_API_KEY in this function's Valves."

        model_id = self._resolve_model(body.get("model", ""))
        messages = body.get("messages", [])
        provider = self._provider_block()
        extra_body: dict[str, Any] = {"provider": provider}
        extra_headers = self._extra_headers(__metadata__)

        create_kwargs: dict[str, Any] = {
            "model": model_id,
            "messages": messages,
            "extra_body": extra_body,
            "extra_headers": extra_headers,
        }
        if self.valves.MAX_TOKENS and self.valves.MAX_TOKENS > 0:
            create_kwargs["max_tokens"] = self.valves.MAX_TOKENS

        client = self._client()

        # --- Task requests (title/tags/etc.): non-streaming, return plain text ---
        # Still provider-pinned so tasks never leak to an unapproved provider.
        if __task__:
            try:
                resp = await client.chat.completions.create(**create_kwargs)
                if resp.choices and resp.choices[0].message.content:
                    return resp.choices[0].message.content.strip()
                return ""
            except Exception as e:
                logger.debug(f"Task request failed: {e}")
                return ""

        # --- Normal streaming chat ---
        async def emit(event: dict) -> None:
            if __event_emitter__:
                await __event_emitter__(event)

        final: List[str] = []
        usage = None
        served_by: Optional[str] = None

        try:
            stream = await client.chat.completions.create(
                stream=True,
                stream_options={"include_usage": True},
                **create_kwargs,
            )
            async for chunk in stream:
                # Capture the serving provider (OpenRouter adds it as an extra field).
                if served_by is None:
                    served_by = getattr(chunk, "provider", None) or (
                        (getattr(chunk, "model_extra", None) or {}).get("provider")
                    )
                if getattr(chunk, "choices", None):
                    delta = chunk.choices[0].delta
                    piece = getattr(delta, "content", None) if delta else None
                    if piece:
                        final.append(piece)
                        await emit({"type": "message", "data": {"content": piece}})
                if getattr(chunk, "usage", None):
                    usage = chunk.usage
        except Exception as e:
            msg = f"\n\n⚠️ OpenRouter request failed: {e}"
            await emit({"type": "status", "data": {"description": "Request failed", "done": True}})
            return "".join(final) + msg

        # --- Finalization banner (served-by = privacy verification) ---
        if self.valves.SHOW_USAGE_INFO:
            parts: List[str] = ["✅ Complete"]
            if served_by:
                parts.append(f"served by {served_by}")
            if usage is not None:
                in_tok = getattr(usage, "prompt_tokens", 0) or 0
                out_tok = getattr(usage, "completion_tokens", 0) or 0
                cached = 0
                details = getattr(usage, "prompt_tokens_details", None)
                if details is not None:
                    cached = getattr(details, "cached_tokens", 0) or 0
                seg = f"in {self._fmt(in_tok)}"
                if cached:
                    seg += f" (cached {self._fmt(cached)})"
                seg += f" · out {self._fmt(out_tok)}"
                parts.append(seg)
            await emit(
                {"type": "status", "data": {"description": " · ".join(parts), "done": True}}
            )

        # Stream-finished signal (empty content) for the frontend.
        await emit(
            {
                "type": "chat:completion",
                "data": {"choices": [{"finish_reason": "stop", "delta": {"content": ""}}], "done": True},
            }
        )

        # Return the full assembled text so OWUI 0.9.x persists it as the saved message.
        return "".join(final)
