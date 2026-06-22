"""
title: OpenRouter Usage Banner
author: pietakio
version: 0.1.0
license: MIT
description: Adds a status banner (served-by provider · cache read · output tokens) to responses from OpenRouter models that are served by a DIRECT connection. Implemented as a Filter (inlet/outlet) so it decorates the existing model flow instead of replacing it.
required_open_webui_version: 0.9.0
"""

# =============================================================================
# OpenRouter Usage Banner — Open WebUI Filter
# -----------------------------------------------------------------------------
# Why a Filter (not a Pipe):
#   A Pipe REPLACES model serving. We want to keep using the direct OpenRouter
#   connection and merely DECORATE its responses with a banner. That is exactly
#   what a Filter does: inlet() runs before the request, outlet() runs after the
#   response — and outlet can emit a status event (the banner) without touching
#   how the model is served.
#
# What it shows:
#   "✅ Response complete · Served by: Fireworks · Cache read: 18.7K · out 356"
#   (cache read is the field you care about most — provider-side prompt-cache reads)
#
# Honest caveat about the data:
#   - inlet() asks the provider to return usage (stream_options.include_usage and
#     OpenRouter's usage.include), so token/cache numbers come back.
#   - cached_tokens lives in the STANDARD usage.prompt_tokens_details, so it is
#     the most likely field to survive Open WebUI's response normalization.
#   - "Served by" (the provider) is an OpenRouter NON-STANDARD field; OWUI may
#     strip it before the outlet sees it. If it doesn't appear, that's why — and
#     only a pipe (which reads OpenRouter's raw response) can guarantee it.
#
# If the banner comes up empty:
#   Flip DEBUG on. outlet() will log the body's key shape so we can see exactly
#   where (or whether) OWUI is putting the usage object, and adjust _find_usage.
# =============================================================================

import logging
from typing import Any, Awaitable, Callable, Optional, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger("openrouter_usage_banner")


class Filter:
    class Valves(BaseModel):
        priority: int = Field(
            default=0,
            description="Filter execution priority (lower runs earlier).",
        )
        REQUEST_USAGE: bool = Field(
            default=True,
            description=(
                "Inject stream_options.include_usage and OpenRouter's usage.include "
                "into the request so the provider returns token/cache usage. Turn off "
                "if it ever causes a request to be rejected."
            ),
        )
        SHOW_SERVED_BY: bool = Field(
            default=True,
            description="Show which provider served the request (may be stripped by OWUI on a direct connection).",
        )
        SHOW_CACHE_READ: bool = Field(
            default=True,
            description="Show cached (prompt-cache read) input tokens — the important one.",
        )
        SHOW_OUTPUT: bool = Field(
            default=True,
            description="Show output (completion) token count.",
        )
        SHOW_INPUT: bool = Field(
            default=False,
            description="Also show total prompt (input) tokens.",
        )
        DEBUG: bool = Field(
            default=False,
            description="Log the outlet body shape (and emit a notice if no usage is found) to help locate the usage fields.",
        )

    def __init__(self):
        self.valves = self.Valves()

    # ------------------------------------------------------------------ inlet
    async def inlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        """Ask the provider to return usage so the outlet has numbers to show."""
        if self.valves.REQUEST_USAGE and isinstance(body, dict):
            # Standard OpenAI streaming usage opt-in.
            stream_options = body.get("stream_options")
            if not isinstance(stream_options, dict):
                stream_options = {}
            stream_options["include_usage"] = True
            body["stream_options"] = stream_options

            # OpenRouter extension: returns usage (with cache breakdown) in the response.
            usage_req = body.get("usage")
            if not isinstance(usage_req, dict):
                usage_req = {}
            usage_req["include"] = True
            body["usage"] = usage_req
        return body

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _fmt(n: Any) -> str:
        try:
            n = int(n)
        except (TypeError, ValueError):
            return "0"
        if n >= 1_000_000:
            return f"{n / 1_000_000:.1f}M"
        if n >= 1_000:
            return f"{n / 1_000:.1f}K"
        return str(n)

    def _find_usage(self, body: dict) -> Tuple[Optional[dict], Optional[str]]:
        """Defensively locate the usage object and served-by provider.

        OWUI's outlet body shape varies by version/path, so check the likely
        spots: top level, then the last message and its 'info' block.
        """
        usage: Optional[dict] = None
        provider: Optional[str] = None

        if not isinstance(body, dict):
            return None, None

        # 1. Top-level
        if isinstance(body.get("usage"), dict):
            usage = body["usage"]
        provider = body.get("provider") or provider

        # 2. Last message + its info block
        messages = body.get("messages")
        if isinstance(messages, list) and messages:
            last = messages[-1]
            if isinstance(last, dict):
                if usage is None and isinstance(last.get("usage"), dict):
                    usage = last["usage"]
                provider = provider or last.get("provider")
                info = last.get("info")
                if isinstance(info, dict):
                    if usage is None and isinstance(info.get("usage"), dict):
                        usage = info["usage"]
                    provider = provider or info.get("provider")

        return usage, provider

    # ----------------------------------------------------------------- outlet
    async def outlet(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
    ) -> dict:
        if self.valves.DEBUG:
            try:
                logger.info(f"[usage-banner] outlet body keys: {list(body.keys())}")
                msgs = body.get("messages")
                if isinstance(msgs, list) and msgs and isinstance(msgs[-1], dict):
                    logger.info(
                        f"[usage-banner] last message keys: {list(msgs[-1].keys())}"
                    )
            except Exception as e:
                logger.info(f"[usage-banner] debug inspection failed: {e}")

        usage, provider = self._find_usage(body)

        parts = ["✅ Response complete"]
        if self.valves.SHOW_SERVED_BY and provider:
            parts.append(f"Served by: {provider}")

        if isinstance(usage, dict):
            details = usage.get("prompt_tokens_details")
            cached = 0
            if isinstance(details, dict):
                cached = details.get("cached_tokens", 0) or 0
            if self.valves.SHOW_INPUT:
                parts.append(f"in {self._fmt(usage.get('prompt_tokens', 0) or 0)}")
            if self.valves.SHOW_CACHE_READ:
                parts.append(f"Cache read: {self._fmt(cached)}")
            if self.valves.SHOW_OUTPUT:
                parts.append(f"out {self._fmt(usage.get('completion_tokens', 0) or 0)}")

        if __event_emitter__:
            if len(parts) > 1:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": " · ".join(parts), "done": True},
                    }
                )
            elif self.valves.DEBUG:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "⚠️ usage-banner: no usage/provider found in outlet body (enable DEBUG logs)",
                            "done": True,
                        },
                    }
                )

        return body
