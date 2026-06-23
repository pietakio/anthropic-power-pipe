"""
title: OpenRouter Anthropic Prompt Cache
author: pietakio
version: 0.2.0
license: MIT
description: Injects Anthropic cache_control breakpoints (1h TTL by default) into requests bound for Claude models served through a DIRECT OpenRouter (OpenAI-compatible) connection, so prompt caching actually engages. Implemented as an inlet Filter so it decorates the existing model flow instead of replacing it.
required_open_webui_version: 0.9.0
"""

# =============================================================================
# OpenRouter Anthropic Prompt Cache — Open WebUI Filter
# -----------------------------------------------------------------------------
# Why this exists:
#   Open WebUI's OpenAI connector sends Anthropic plain messages with NO
#   cache_control markers. Anthropic's prompt caching is opt-in per request —
#   no marker means nothing is ever cached. OpenRouter passes cache_control
#   through faithfully when it IS present (OpenAI multipart content format) and
#   handles the extended-cache beta plumbing on its side. So the fix is to add
#   the breakpoints here, in the inlet, before the request leaves OWUI.
#
# Why a Filter (not a Pipe):
#   A Pipe REPLACES model serving. We want to keep using the direct OpenRouter
#   connection and merely add cache_control to the outgoing request. inlet()
#   runs before the request is sent — exactly the hook we need.
#
# Pairs with the OpenRouter Usage Banner filter:
#   This filter CREATES the cache; the banner filter REPORTS the cache reads
#   (usage.prompt_tokens_details.cached_tokens). Run both — once this is live,
#   the banner's "Cache read" going non-zero on the second+ turn confirms it.
#
# Breakpoint strategy (Anthropic allows max 4 per request):
#   1. The system message  — the big stable win (caches tools+system prefix).
#   2. The last N messages — a rolling window so the conversation prefix stays
#      cached as it grows. A breakpoint caches everything BEFORE it, so marking
#      the latest turn caches system + entire history up to that point; next
#      turn that span becomes a cache READ.
#
# Honest caveat about minimums:
#   Anthropic silently won't cache a prefix below its minimum (4096 tokens for
#   Opus 4.x, 2048 for Sonnet 4.6). Short early conversations simply won't cache
#   — that's expected, not a failure. The banner's "Cache read" tells the truth.
#
# 1h vs 5m TTL:
#   Default here is "1h" because a human-paced chat routinely idles past the
#   5-minute default, which would expire the entry mid-think and waste the
#   write. A 1h write costs ~2x base (vs ~1.25x for 5m) and needs ~3 reads to
#   pay off, which a long conversation over a large system prompt clears easily.
# =============================================================================

import logging
from typing import Any, Awaitable, Callable, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger("openrouter_anthropic_cache")

_MAX_BREAKPOINTS = 4  # Anthropic hard limit per request.


class Filter:
    class Valves(BaseModel):
        priority: int = Field(
            default=0,
            description="Filter execution priority (lower runs earlier).",
        )
        ENABLED: bool = Field(
            default=True,
            description="Master switch. Turn off to send requests untouched.",
        )
        MODEL_MATCH: str = Field(
            default="anthropic,claude",
            description=(
                "Comma-separated substrings; cache_control is injected only when "
                "the model id contains one of these (case-insensitive). Keeps the "
                "filter from touching non-Anthropic models on the same connection, "
                "which would reject or ignore cache_control."
            ),
        )
        TTL: str = Field(
            default="1h",
            description=(
                "Cache lifetime: '1h' (1 hour) or '5m' (Anthropic's 5-minute "
                "default). '5m' / 'default' / empty sends no ttl field."
            ),
        )
        CACHE_SYSTEM: bool = Field(
            default=True,
            description="Put a breakpoint on the system message (caches the stable prefix).",
        )
        ROLLING_MESSAGE_BREAKPOINTS: int = Field(
            default=2,
            ge=0,
            le=3,
            description=(
                "How many of the most recent (non-system) messages to mark, so the "
                "growing conversation prefix stays cached across turns. system + this "
                "is capped at 4 total."
            ),
        )
        DEBUG: bool = Field(
            default=False,
            description=(
                "Surface what the filter did — the model id it saw, the TTL, and how "
                "many breakpoints it placed — as a status banner in the chat (and in "
                "the server log). The model id shown is the exact string the filter "
                "matched against."
            ),
        )

    def __init__(self):
        self.valves = self.Valves()

    # ---------------------------------------------------------------- helpers
    def _is_anthropic(self, model: str) -> bool:
        m = (model or "").lower()
        needles = [
            s.strip().lower() for s in self.valves.MODEL_MATCH.split(",") if s.strip()
        ]
        return any(n in m for n in needles)

    def _cache_control(self) -> dict:
        ttl = (self.valves.TTL or "").strip().lower()
        cc: dict = {"type": "ephemeral"}
        # Anthropic's default TTL is 5 minutes and takes no ttl field; only the
        # extended cache needs an explicit ttl.
        if ttl not in ("", "5m", "5min", "default"):
            cc["ttl"] = ttl
        return cc

    @staticmethod
    def _mark_message(message: dict, cc: dict) -> bool:
        """Attach cache_control to a message's content. Returns True if a
        breakpoint was placed. Uses a fresh dict per block so markers never
        share a mutable reference."""
        content = message.get("content")

        if isinstance(content, str):
            if not content.strip():
                return False
            message["content"] = [
                {"type": "text", "text": content, "cache_control": dict(cc)}
            ]
            return True

        if isinstance(content, list) and content:
            # Prefer the last text block (cache_control on text is always valid).
            for part in reversed(content):
                if isinstance(part, dict) and part.get("type") == "text":
                    part["cache_control"] = dict(cc)
                    return True
            # No text block (e.g. image-only turn) — fall back to the last block.
            last = content[-1]
            if isinstance(last, dict):
                last["cache_control"] = dict(cc)
                return True

        return False

    # ------------------------------------------------------------------ inlet
    async def inlet(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
    ) -> dict:
        if not self.valves.ENABLED or not isinstance(body, dict):
            return body

        model = str(body.get("model", ""))
        if not self._is_anthropic(model):
            # When debugging, say so in-chat — a silent skip here (model id didn't
            # match) is the easiest failure to misread as "the cache isn't working".
            if self.valves.DEBUG and __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": (
                                f"🪪 anthropic-cache: SKIPPED — model {model!r} "
                                f"didn't match MODEL_MATCH ({self.valves.MODEL_MATCH!r})"
                            ),
                            "done": True,
                        },
                    }
                )
            return body
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            return body

        cc = self._cache_control()

        # Collect target indices in priority order: system first, then most
        # recent messages. Trimming to the limit keeps system + the newest turns.
        targets: list[int] = []

        if self.valves.CACHE_SYSTEM:
            sys_idx = None
            for i, m in enumerate(messages):
                if isinstance(m, dict) and m.get("role") == "system":
                    sys_idx = i  # last system message bounds the whole system prefix
            if sys_idx is not None:
                targets.append(sys_idx)

        n = self.valves.ROLLING_MESSAGE_BREAKPOINTS
        if n > 0:
            count = 0
            for i in range(len(messages) - 1, -1, -1):
                m = messages[i]
                if not isinstance(m, dict) or m.get("role") == "system":
                    continue
                if i in targets:
                    continue
                targets.append(i)
                count += 1
                if count >= n:
                    break

        targets = targets[:_MAX_BREAKPOINTS]

        marked = 0
        for i in targets:
            if self._mark_message(messages[i], cc):
                marked += 1

        if self.valves.DEBUG:
            ttl_label = cc.get("ttl", "5m")
            indices = sorted(set(targets))
            logger.info(
                f"[anthropic-cache] model={model!r} ttl={ttl_label} "
                f"marked {marked} breakpoint(s) at {indices}"
            )
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": (
                                f"🪪 anthropic-cache: model {model!r} · ttl {ttl_label} · "
                                f"marked {marked} breakpoint(s) at {indices}"
                            ),
                            "done": True,
                        },
                    }
                )

        return body
