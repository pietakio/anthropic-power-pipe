"""
title: Anthropic Power Pipe
id: anthropic_power_pipe
author: Alexis Pietak (https://github.com/pietakio/)
github: https://github.com/pietakio/anthropic-power-pipe
original_author: Podden, Balaxxe, nbellochi
version: 0.1.0
license: MIT
requirements: pydantic>=2.0.0, anthropic>=0.75.0
environment_variables:
    - ANTHROPIC_API_KEY (required)

Supports:
- Uses Anthropic Python SDK
- File API with Skills and Code Execution
- Fetch Claude Models from API Endpoint
- Tool Call Loop (call multiple Tools in the same response)
- web_search Tool
- web_fetch Tool (URL content retrieval)
- citations for web_search
- Streaming responses
- Prompt caching (server-side) compatible with Openwebui Memory and RAG System
- Prompt Caching of System Prompts, Messages- and Tools Array (controllable via Valve)
- Comprehensive error
- Image processing
- Web_Search Toggle Action
- Fine Grained Tool Streaming
- Extended Thinking Toggle Action
- Code Execution Tool
- Vision
- Context Editing (clear tool results and thinking blocks)
- Tool Search (BM25/Regex)
- Native PDF Upload (visual PDF analysis with charts/images)
- Agent Skills (pptx, xlsx, docx, pdf and custom skills)
- Fast Mode (research preview) for Opus 4.6
- Memory Tool (integrated with OpenWebUI memory system)
- Programmatic Tool Calling (tools callable from code execution)
"""

import re
import os
import shutil
import base64
import traceback
import inspect
from datetime import datetime
from collections.abc import Awaitable
import asyncio
import html
import json
import logging
import time
from dataclasses import dataclass, field
from urllib.parse import quote, unquote
from typing import Any, Callable, List, Union, Dict, Optional
from pydantic import BaseModel, Field
from anthropic import (
    APIStatusError,
    AsyncAnthropic,
    RateLimitError,
    APIConnectionError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    PermissionDeniedError,
    NotFoundError,
    UnprocessableEntityError,
)

try:
    from anthropic import OverloadedError
except ImportError:
    # anthropic SDK < 0.45 doesn't have OverloadedError
    # Create a placeholder that will never match (handled via APIStatusError instead)
    class OverloadedError(Exception):
        pass
from typing import Literal
from fastapi import Request

logger = logging.getLogger(__name__)

# =============================================================================
# COMPILED REGEX PATTERNS
# Pre-compiled patterns for performance - avoids re-compiling on every call
# =============================================================================

# NOTE: Thinking blocks must NEVER be removed from assistant messages!
# Per Anthropic API docs:
# - During tool use loops: thinking blocks MUST be preserved unmodified in assistant content
# - Multi-turn: thinking blocks from prior turns CAN be omitted (API filters them),
#   but preserving them is preferred
# - The entire sequence of consecutive thinking blocks must match the original model output
# - signature field is critical and must be preserved exactly
# - Interleaved thinking (Claude 4): thinking blocks can appear BETWEEN tool calls
# Previously PATTERN_THINKING_BLOCK was defined here but never used - removed as dead code.

# Pattern to extract User Context from OpenWebUI Memory System in system prompts
# Matches everything after "\nUser Context:\n" to end of string
PATTERN_USER_CONTEXT = re.compile(r"\nUser Context:\n(.*)$", flags=re.DOTALL)

# Patterns for RAG template cleanup when all sources are native PDFs
PATTERN_RAG_TEMPLATE_WITH_CONTEXT = re.compile(
    r"###\s*Task:.*?<context>.*?</context>", flags=re.DOTALL | re.MULTILINE
)
PATTERN_RAG_TEMPLATE_FALLBACK = re.compile(
    r"###\s*Task:.*?$", flags=re.DOTALL | re.MULTILINE
)
PATTERN_EMPTY_CONTEXT = re.compile(r"<context>\s*</context>", flags=re.DOTALL)

# Pattern to find remaining source tags (for checking if all were removed)
PATTERN_SOURCE_TAGS = re.compile(r"<source[^>]*>.*?</source>", flags=re.DOTALL)

# RAG message detection: matches "### Task:...<context>...</context>" blocks
PATTERN_RAG_MESSAGE = re.compile(r"### Task:.*?<context>.*?</context>", re.DOTALL)

# Individual <source> tag with name attribute extraction
PATTERN_SOURCE_TAG = re.compile(
    r'<source[^>]*name="([^"]+)"[^>]*>.*?</source>\s*', re.DOTALL
)

# Empty <attached_files> blocks after file tag removal
PATTERN_EMPTY_ATTACHED = re.compile(
    r"<attached_files>\s*</attached_files>\s*", re.DOTALL
)

# Pattern to PARSE OpenWebUI <details type="tool_calls"> blocks from conversation history.
# Captures id, name, arguments, and result attributes for reconstruction as proper API blocks.
# These must be converted back to tool_use/tool_result structures for Claude to see its history.
PATTERN_TOOL_CALLS_PARSE = re.compile(
    r'<details\s+type="tool_calls"[^>]*'
    r'\s+id="([^"]*)"'          # Group 1: tool use ID
    r'[^>]*\s+name="([^"]*)"'   # Group 2: tool name
    r'[^>]*\s+arguments="([^"]*)"'  # Group 3: arguments (HTML-escaped JSON)
    r'(?:[^>]*\s+result="([^"]*)")?'  # Group 4: result (optional, HTML-escaped)
    r'[^>]*>.*?</details>',
    flags=re.DOTALL,
)

# Fallback pattern to strip any tool_calls blocks that don't match the parse pattern
# (e.g., in-progress tool calls without results)
PATTERN_TOOL_CALLS_STRIP = re.compile(
    r'\n?<details type="tool_calls"[^>]*>.*?</details>\n?',
    flags=re.DOTALL,
)

# Pattern to strip OpenWebUI <details type="code_interpreter"> blocks from conversation history.
PATTERN_CODE_INTERPRETER_DETAILS = re.compile(
    r'\n?<details type="code_interpreter"[^>]*>.*?</details>\n?',
    flags=re.DOTALL,
)

# Pattern to strip OpenWebUI <details type="reasoning"> blocks from conversation history.
# Thinking blocks from prior turns are auto-filtered by the API, and we can't reconstruct
# them without the signature. Strip them to avoid wasting tokens on unusable HTML text.
PATTERN_REASONING_DETAILS = re.compile(
    r'\n?<details type="reasoning"[^>]*>.*?</details>\n?',
    flags=re.DOTALL,
)

# Note: Some patterns are compiled dynamically at runtime because they depend
# on user-provided data (filenames, file IDs). See:
#   - _remove_specific_sources_from_rag_message() - dynamic filename pattern
#   - _remove_attached_files_tags() - dynamic file ID pattern

# =============================================================================
# IMPORTS
# =============================================================================

# Import OpenWebUI Models for auto-enabling native function calling
try:
    from open_webui.models.models import Models, ModelForm

    MODELS_AVAILABLE = True
except ImportError:
    Models = None
    ModelForm = None
    MODELS_AVAILABLE = False

# Import OpenWebUI builtin tools helper
try:
    from open_webui.utils.tools import get_builtin_tools

    BUILTIN_TOOLS_AVAILABLE = True
except ImportError:
    get_builtin_tools = None
    BUILTIN_TOOLS_AVAILABLE = False

# Import process_tool_result for Rich UI (HTMLResponse, embeds, files)
try:
    from open_webui.utils.middleware import process_tool_result

    PROCESS_TOOL_RESULT_AVAILABLE = True
except ImportError:
    process_tool_result = None
    PROCESS_TOOL_RESULT_AVAILABLE = False

# Import OpenWebUI Files and Storage for PDF native upload
try:
    from open_webui.models.files import Files
    from open_webui.storage.provider import Storage
    from pathlib import Path

    FILES_AVAILABLE = True
except ImportError:
    Files = None
    Storage = None
    Path = None
    FILES_AVAILABLE = False

# =============================================================================
# CONSTANTS
# =============================================================================

# Claude memory tool uses filesystem storage (no dependency on OpenWebUI Memories)
# Files stored under DATA_DIR/claude_memories/{user_id}/memories/
CLAUDE_MEMORY_DIR = os.path.join(
    os.environ.get("DATA_DIR", "data"), "claude_memories"
)


@dataclass
class PipeRenderStrategy:
    """Per-request rendering strategy toggles (no shared state across users)."""

    stream_reasoning_live: bool = True
    stream_code_execution_live: bool = False
    stream_tool_results_live: bool = False


@dataclass
class PipeRequestContext:
    """Request-scoped helpers/state. Must be instantiated inside pipe()."""

    pipe: Any
    event_emitter: Callable[[Dict[str, Any]], Awaitable[None]]
    render_strategy: PipeRenderStrategy = field(default_factory=PipeRenderStrategy)
    final_message: list[str] = field(default_factory=list)

    async def emit_event(self, event: dict) -> None:
        await self.pipe.emit_event(event, self.event_emitter)

    async def emit_delta(self, content: str) -> None:
        await self.emit_event(
            {
                "type": "chat:completion",
                "data": {"choices": [{"delta": {"content": content}}]},
            }
        )
        self.final_message.append(content)

    async def emit_replace(self, content: str) -> None:
        await self.emit_event({"type": "replace", "data": {"content": content}})
        self.final_message.clear()
        self.final_message.append(content)

    async def update_content_block(self, old_block: str, new_block: str) -> None:
        """Replace old_block in accumulated content with new_block, preserving surrounding text."""
        if old_block:
            text = self.text()
            idx = text.find(old_block)
            if idx != -1:
                text = text[:idx] + new_block + text[idx + len(old_block):]
                await self.emit_replace(text)
                return
        # First emit or old block not found — append and replace with full text
        self.final_message.append(new_block)
        await self.emit_replace(self.text())

    def text(self) -> str:
        return "".join(self.final_message)


class Pipe:
    API_VERSION = "2023-06-01"  # Current API version as of May 2025
    _DEFAULT_API_BASE = "https://api.anthropic.com"

    @property
    def MODEL_URL(self):
        base = self.valves.ANTHROPIC_BASE_URL.strip().rstrip("/") if hasattr(self, "valves") and self.valves.ANTHROPIC_BASE_URL.strip() else self._DEFAULT_API_BASE
        return f"{base}/v1/messages"

    # Capability overrides for fields NOT available from the /v1/models API.
    # The API now provides: max_tokens, max_input_tokens, capabilities (thinking, effort, vision, etc.)
    # These overrides only contain flags that must be derived from model identity.
    MODEL_CAPABILITY_OVERRIDES = {
        "claude-sonnet-4-20250514": {"supports_1m_context": True},
        "claude-sonnet-4-5-20250929": {"supports_1m_context": True},
        "claude-opus-4-6": {
            "supports_1m_context": True,
            "supports_dynamic_filtering": True,
            "supports_fast_mode": True,
        },
        "claude-sonnet-4-6": {
            "supports_1m_context": True,
            "supports_dynamic_filtering": True,
        },
        # Opus 4.7 breaking changes:
        # - No budget_tokens (adaptive thinking only)
        # - No sampling params (temperature/top_p/top_k)
        # - Thinking content omitted by default (need display: "summarized")
        # - New xhigh effort level
        "claude-opus-4-7": {
            "supports_1m_context": True,
            "supports_dynamic_filtering": True,
            "supports_fast_mode": True,
            "requires_adaptive_only": True,  # No budget_tokens, adaptive thinking only
            "supports_effort_xhigh": True,   # New xhigh effort level
            "no_sampling_params": True,      # temperature/top_p/top_k return 400 error
        },
    }

    # Cached model capabilities from API (populated by get_anthropic_models)
    _api_capabilities_cache: Dict[str, dict] = {}
    _api_capabilities_cache_ts: float = 0.0
    _API_CACHE_TTL = 86400  # 24 hours

    REQUEST_TIMEOUT = (
        300  # Default; overridden by valve REQUEST_TIMEOUT
    )
    THINKING_BUDGET_TOKENS = 4096  # Default thinking budget tokens (max 16K)
    TOOL_CALL_TIMEOUT = 120  # Default; overridden by valve TOOL_CALL_TIMEOUT

    # =========================================================================
    # MODEL INFO & INITIALIZATION
    # =========================================================================

    @classmethod
    def _parse_api_capabilities(cls, model) -> dict:
        """Parse capabilities from an Anthropic API ModelInfo object into our internal format."""
        caps = getattr(model, "capabilities", None)
        _sup = lambda obj, attr="supported": getattr(obj, attr, False) if obj else False

        thinking = getattr(caps, "thinking", None) if caps else None
        thinking_types = getattr(thinking, "types", None) if thinking else None
        effort = getattr(caps, "effort", None) if caps else None

        max_tokens = getattr(model, "max_tokens", 0) or 0
        max_input = getattr(model, "max_input_tokens", 0) or 0

        info = {
            "max_tokens": max_tokens if max_tokens > 0 else 4096,
            "context_length": max_input if max_input > 0 else 200000,
            "supports_thinking": _sup(thinking),
            "supports_adaptive_thinking": _sup(getattr(thinking_types, "adaptive", None)) if thinking_types else False,
            "supports_effort": _sup(effort),
            "supports_effort_max": _sup(getattr(effort, "max", None)) if effort else False,
            "supports_vision": _sup(getattr(caps, "image_input", None)) if caps else True,
            "supports_programmatic_calling": _sup(getattr(caps, "code_execution", None)) if caps else False,
            # All Claude 4+ models support memory
            "supports_memory": True,
            # Defaults for fields not in API — overridden by MODEL_CAPABILITY_OVERRIDES
            "supports_1m_context": False,
            "supports_dynamic_filtering": False,
            "supports_fast_mode": False,
            # Opus 4.7+ specific flags (defaults for older models)
            "requires_adaptive_only": False,  # True = no budget_tokens allowed
            "supports_effort_xhigh": False,   # True = xhigh effort level available
            "no_sampling_params": False,      # True = temperature/top_p/top_k not allowed
        }

        # Apply model-specific overrides for fields not available from API
        model_id = model.id if hasattr(model, "id") else ""
        overrides = cls.MODEL_CAPABILITY_OVERRIDES.get(model_id, {})
        info.update(overrides)

        return info

    @classmethod
    def get_model_info(cls, model_name: str) -> dict:
        """
        Get model capabilities by name. Reads from API cache first,
        falls back to safe defaults for unknown models.
        """
        if model_name in cls._api_capabilities_cache:
            return cls._api_capabilities_cache[model_name]

        # Return conservative defaults for unknown models
        return {
            "max_tokens": 4096,
            "context_length": 200000,
            "supports_thinking": True,
            "supports_1m_context": False,
            "supports_memory": False,
            "supports_vision": True,
            "supports_effort": False,
            "supports_programmatic_calling": False,
            "supports_dynamic_filtering": False,
            "supports_adaptive_thinking": False,
            "supports_effort_max": False,
            "supports_fast_mode": False,
            # Opus 4.7+ specific flags (conservative defaults)
            "requires_adaptive_only": False,
            "supports_effort_xhigh": False,
            "no_sampling_params": False,
        }

    class Valves(BaseModel):
        ANTHROPIC_API_KEY: str = "Your API Key Here"
        ANTHROPIC_BASE_URL: str = Field(
            default="",
            description="Custom base URL for the Anthropic API (e.g. for a proxy). Leave empty to use the default Anthropic endpoint (https://api.anthropic.com).",
        )
        ENABLE_FAST_MODE: bool = Field(
            default=False,
            description="Enable Fast Mode for Opus 4.6. Up to 2.5x faster output at higher costs",
        )
        # Not quite finished with testing yet, disabled for now
        # ENABLE_CLAUDE_MEMORY: bool = Field(
        #     default=False,
        #     description="Enable Claude memory tool (files stored per-user under data/claude_memories/)",
        # )
        ENABLE_INTERLEAVED_THINKING: bool = Field(
            default=True,
            description="Enable interleaved thinking. Claude can generate thinking blocks between tool calls instead of only at the end.",
        )
        WEB_SEARCH: bool = Field(
            default=True,
            description="Enable web search tool for Claude models. Use Anthropic Web Search Toggle Function for fine grained control",
        )
        WEB_FETCH: bool = Field(
            default=True,
            description="Enable web fetch tool for Claude models. Allows Claude to fetch and analyze content from URLs.",
        )
        MAX_TOOL_CALLS: int = Field(
            default=15,
            ge=1,
            le=9999,
            description="Maximum number of tool execution loops allowed per request. Each loop involves Claude generating tool calls, executing them, and feeding results back. Prevents infinite loops.",
        )
        MAX_RETRIES: int = Field(
            default=3,
            ge=0,
            le=50,
            description="Maximum number of retries for failed requests (due to rate limiting, transient errors or connection issues)",
        )
        CACHE_CONTROL: Literal[
            "cache disabled",
            "cache tools array only",
            "cache tools array and system prompt",
            "cache tools array, system prompt and messages",
        ] = Field(
            default="cache tools array, system prompt and messages",
            description="Cache control scope for prompts",
        )
        CACHE_TTL: Literal["5 minutes", "1 hour"] = Field(
            default="5 minutes",
            description="Cache time-to-live. 5 minutes is default Anthropic TTL; 1 hour requires no extra cost but keeps caches warm longer across conversations.",
        )
        WEB_SEARCH_USER_CITY: str = Field(
            default="",
            description="User's city for web search.",
        )
        WEB_SEARCH_USER_REGION: str = Field(
            default="",
            description="User's region/state for web search",
        )
        WEB_SEARCH_USER_COUNTRY: str = Field(
            default="",
            description="User's country code for web search",
        )
        WEB_SEARCH_USER_TIMEZONE: str = Field(
            default="",
            description="User's timezone for web search.",
        )
        ENABLE_PROGRAMMATIC_TOOL_CALLING: bool = Field(
            default=False,
            description="Enable programmatic tool calling. Claude can call tools from within code execution. Requires code execution to be active.",
        )
        ENABLE_TOOL_SEARCH: bool = Field(
            default=False,
            description="Enable tool search. Allows Claude to search for tools by name/description when many tools are available.",
        )
        TOOL_SEARCH_TYPE: Literal["regex", "bm25"] = Field(
            default="bm25",
            description="Type of tool search: 'regex' for pattern matching or 'bm25' for natural language search.",
        )
        TOOL_SEARCH_MAX_DESCRIPTION_LENGTH: int = Field(
            default=100,
            ge=10,
            le=10000,
            description="Maximum tool description length. Tools with longer JSON definitions will be deferred for lazy loading.",
        )
        TOOL_SEARCH_EXCLUDE_TOOLS: List[str] = Field(
            default=["web_search", "web_fetch", "code_execution_20250825", "code_execution_20260120"],
            description="Tools to exclude from defer_loading when tool search is enabled. These tools will always be loaded immediately.",
        )
        CONTEXT_EDITING_STRATEGY: Literal[
            "none", "clear_tool_results", "clear_thinking", "clear_both"
        ] = Field(
            default="none",
            description="Context editing strategy: none (disabled), clear_tool_results, clear_thinking, or clear_both.",
        )
        CONTEXT_EDITING_THINKING_KEEP: int = Field(
            default=5,
            ge=0,
            le=9999,
            description="How many thinking blocks to keep",
        )
        CONTEXT_EDITING_TOOL_TRIGGER: int = Field(
            default=50000,
            ge=1000,
            le=500000,
            description="Token count threshold that triggers tool result clearing.",
        )
        CONTEXT_EDITING_TOOL_KEEP: int = Field(
            default=5,
            ge=0,
            le=100,
            description="Number of recent tool results to preserve when clearing.",
        )
        CONTEXT_EDITING_TOOL_CLEAR_AT_LEAST: int = Field(
            default=10000,
            ge=0,
            le=100000,
            description="Minimum tokens to clear when triggered (helps with cache optimization).",
        )
        CONTEXT_EDITING_TOOL_CLEAR_TOOL_INPUT: bool = Field(
            default=False,
            description="Also clear tool input parameters when clearing tool results.",
        )
        DATA_RESIDENCY: Literal["global", "us"] = Field(
            default="global",
            description='Data residency for API requests. 1.1x Token Cost for "us".',
        )
        REQUEST_TIMEOUT: int = Field(
            default=300,
            ge=30,
            le=1800,
            description="Request timeout in seconds for Anthropic API calls. Increase if using slow local rerankers or large context (e.g. 600 for Top-K 15+).",
        )
        TOOL_CALL_TIMEOUT: int = Field(
            default=30,
            ge=10,
            le=600,
            description="Timeout in seconds for individual tool call execution.",
        )

    class UserValves(BaseModel):
        ANTHROPIC_API_KEY: str = Field(
            default="",
            description="Personal Anthropic API key. If set, overrides the admin-configured key.",
        )
        ENABLE_THINKING: bool = Field(
            default=False,
            description="Enable Extended Thinking",
        )
        THINKING_BUDGET_TOKENS: int = Field(
            default=8192,
            ge=1024,
            le=64000,
            description="Thinking budget tokens",
        )
        EFFORT: Literal["low", "medium", "high", "max", "xhigh"] = Field(
            default="high",
            description="Effort level for this user. 'xhigh' is Opus 4.7+ only. Also controllable with OpenWebUI's reasoning_effort parameter.",
        )
        USE_PDF_NATIVE_UPLOAD: bool = Field(
            default=True,
            description="Upload PDFs as native base64 documents instead of RAG text extraction. Enables visual PDF analysis (charts, images, layouts). Only applies to 'Use Full Document' mode.",
        )
        SHOW_TOKEN_COUNT: bool = Field(
            default=False,
            description="Show Context Window Progress",
        )
        SHOW_CACHE_INFO: bool = Field(
            default=False,
            description="Show cache status (write/read tokens) after each response. Immediately reveals when RAG, memory, or tool changes invalidate the cache.",
        )
        WEB_SEARCH_MAX_USES: int = Field(
            default=5,
            ge=1,
            le=20,
            description="Maximum number of web searches",
        )
        WEB_FETCH_MAX_USES: int = Field(
            default=5,
            ge=1,
            le=20,
            description="Maximum number of web fetch requests per conversation turn",
        )
        WEB_SEARCH_USER_CITY: str = Field(
            default="",
            description="User's city for web search.",
        )
        WEB_SEARCH_USER_REGION: str = Field(
            default="",
            description="User's region/state for web search",
        )
        WEB_SEARCH_USER_COUNTRY: str = Field(
            default="",
            description="User's country code for web search",
        )
        WEB_SEARCH_USER_TIMEZONE: str = Field(
            default="",
            description="User's timezone for web search.",
        )
        ENABLE_DYNAMIC_FILTERING: bool = Field(
            default=True,
            description="Use dynamic filtering for web search/fetch on supported models (4.6+). Much slower (~60s vs ~7s) but produces higher quality results by orchestrating multiple searches/fetches and filtering content via code execution. Trades speed for context efficiency.",
        )
        # Files API and Skills Settings
        USE_FILES_API: bool = Field(
            default=False,
            description="Upload files to Anthropic Files API for code execution access. Overrides native PDF upload. Requires code execution.",
        )
        SKILLS: List[str] = Field(
            default=[],
            description="Anthropic Skills to use (e.g., 'pptx', 'xlsx', 'docx', 'pdf' or custom skill IDs). Skills are validated against the API.",
        )
        DEBUG_MODE: bool = Field(
            default=False,
            description="Enable debug mode with verbose logging and additional status events.",
        )

    def __init__(self):
        self.type = "manifold"
        self.id = "anthropic"
        self.valves = self.Valves()
        self.logger = logger
        self._validated_skills_cache: Dict[str, Dict[str, Optional[Dict[str, Any]]]] = (
            {}
        )

    async def get_anthropic_models(self) -> List[dict]:
        """
        Fetches the current list of Anthropic models using the official Anthropic Python SDK.
        Parses capabilities from the API response and caches them.
        Returns OpenWebUI model dicts.
        """
        # Return cached result if still fresh
        if (
            self._api_capabilities_cache
            and time.time() - self._api_capabilities_cache_ts < self._API_CACHE_TTL
        ):
            models = []
            for name, info in self._api_capabilities_cache.items():
                models.append(self._build_openwebui_model_entry(name, info))
            return models

        from anthropic import AsyncAnthropic

        models = []
        new_cache: Dict[str, dict] = {}
        try:
            api_key = self.valves.ANTHROPIC_API_KEY
            base_url = self.valves.ANTHROPIC_BASE_URL.strip() or None
            client = AsyncAnthropic(api_key=api_key, **({"base_url": base_url} if base_url else {}))
            async for m in client.models.list():
                name = m.id
                display_name = getattr(m, "display_name", name)

                # Parse capabilities directly from API response
                info = self._parse_api_capabilities(m)
                info["_display_name"] = display_name
                new_cache[name] = info

                entry = self._build_openwebui_model_entry(name, info, display_name)
                models.append(entry)

            # Update class-level cache
            Pipe._api_capabilities_cache = new_cache
            Pipe._api_capabilities_cache_ts = time.time()
            logger.info(f"Cached capabilities for {len(new_cache)} models from API")
            return models
        except Exception as e:
            logging.warning(
                f"Could not fetch models from SDK/API: {e}"
            )
            # If we have stale cache, use it
            if self._api_capabilities_cache:
                logging.info("Using stale capability cache as fallback")
                for name, info in self._api_capabilities_cache.items():
                    models.append(self._build_openwebui_model_entry(name, info))
                return models
            # No cache available — return empty (API key likely invalid)
            return models

    @staticmethod
    def _build_openwebui_model_entry(
        name: str, info: dict, display_name: str = ""
    ) -> dict:
        return {
            "id": f"anthropic/{name}",
            "name": display_name or name,
            "context_length": info["context_length"],
            "supports_vision": info["supports_vision"],
            "supports_thinking": info["supports_thinking"],
            "is_hybrid_model": info["supports_thinking"],
            "max_output_tokens": info["max_tokens"],
            "info": {
                "meta": {
                    "capabilities": {
                        "status_updates": True
                    }
                }
            },
        }

    async def pipes(self) -> List[dict]:
        return await self.get_anthropic_models()

    # =========================================================================
    # PDF & FILE HANDLING
    # =========================================================================

    def _get_pdf_base64_from_file_id(self, file_id: str) -> Optional[tuple[str, str]]:
        """
        Read a PDF file from storage and return base64 encoded data.

        Args:
            file_id: The OpenWebUI file ID

        Returns:
            tuple[str, str]: (base64_data, filename) or None if not available
        """
        if not FILES_AVAILABLE:
            logger.warning("Files/Storage modules not available for PDF native upload")
            return None

        try:
            file = Files.get_file_by_id(file_id)
            if not file:
                logger.warning(f"File not found: {file_id}")
                return None

            # Check if it's a PDF
            content_type = file.meta.get("content_type", "")
            filename = file.meta.get("name", file.filename)

            if content_type != "application/pdf" and not filename.lower().endswith(
                ".pdf"
            ):
                logger.debug(f"File {file_id} is not a PDF: {content_type}")
                return None

            # Get file path from storage
            file_path = Storage.get_file(file.path)
            file_path = Path(file_path)

            if not file_path.is_file():
                logger.warning(f"PDF file not found on disk: {file_path}")
                return None

            # Read and encode the PDF
            with open(file_path, "rb") as pdf_file:
                pdf_data = pdf_file.read()
                encoded_data = base64.b64encode(pdf_data).decode("utf-8")

            # Check size limits (Anthropic has 32MB request limit, be conservative)
            MAX_PDF_SIZE = 25 * 1024 * 1024  # 25 MB
            if len(pdf_data) > MAX_PDF_SIZE:
                logger.warning(
                    f"PDF too large for native upload: {len(pdf_data)} bytes"
                )
                return None

            logger.debug(
                f"Successfully encoded PDF: {filename} ({len(pdf_data)} bytes)"
            )
            return (encoded_data, filename)

        except Exception as e:
            logger.error(f"Error reading PDF file {file_id}: {e}")
            return None

    def _get_full_context_pdfs(
        self,
        __files__: Optional[List[Dict[str, Any]]],
        previous_marker_metadata: List[str],
    ) -> tuple[List[Dict[str, Any]], List[str]]:
        """
        Extract PDFs from __files__ that should be uploaded as native documents.

        Args:
            __files__: List of file objects from OpenWebUI
            previous_marker_metadata: List of metadata strings from previous messages

        Returns:
            tuple: (List of document blocks for Anthropic API, List of metadata markers for processed PDFs)
        """
        pdf_documents = []
        markers = []

        if not __files__ or not FILES_AVAILABLE:
            return pdf_documents, markers

        for file in __files__:
            # Only process files with 'full' context (not RAG chunks)
            if file.get("type") != "file" or file.get("context") != "full":
                continue

            file_id = file.get("id")
            if not file_id:
                continue

            # Check if it's a PDF
            file_name = file.get("name", "")
            if not file_name.lower().endswith(".pdf"):
                continue

            # Check if this file was already processed (by checking file_id in metadata)
            if any(file_id in metadata for metadata in previous_marker_metadata):
                logger.debug(f"Skipping already processed PDF: {file_name}")
                continue

            # Get base64 encoded PDF
            result = self._get_pdf_base64_from_file_id(file_id)
            if result:
                encoded_data, filename = result
                pdf_documents.append(
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": encoded_data,
                        },
                        "title": filename,
                    }
                )
                markers.append(
                    self._create_metadata_marker("pdf", f"{file_id}:{filename}")
                )

        return pdf_documents, markers

    # =========================================================================
    # RAG (RETRIEVAL-AUGMENTED GENERATION) HANDLING
    # =========================================================================

    def _extract_rag_from_system_message(
        self, system_messages: List[Dict[str, Any]]
    ) -> Optional[str]:
        """
        Extract RAG content from system messages (when RAG_SYSTEM_CONTEXT is True).
        Returns the RAG message string if found, None otherwise.

        This handles the case where OpenWebUI puts RAG content in system prompts
        instead of user messages based on RAG_SYSTEM_CONTEXT env var.

        Args:
            system_messages: List of system message blocks

        Returns:
            str: Extracted RAG content including template and context tags, or None
        """
        for block in system_messages:
            if block.get("type") != "text":
                continue

            text = block.get("text", "")

            # Check if this contains RAG template markers
            if "### Task:" not in text or "<context>" not in text:
                continue

            # Extract the RAG portion (from ### Task: to end of </context>)
            match = PATTERN_RAG_MESSAGE.search(text)
            if match:
                logger.debug(
                    f"📋 RAG: Found RAG content in system message ({len(match.group(0))} chars)"
                )
                return match.group(0)

        return None

    def _remove_rag_from_system_messages(
        self, system_messages: List[Dict[str, Any]]
    ) -> None:
        """
        Remove RAG content from system messages in-place.

        Args:
            system_messages: List of system message blocks to modify
        """
        for i, block in enumerate(system_messages):
            if block.get("type") != "text":
                continue

            text = block.get("text", "")
            match = PATTERN_RAG_MESSAGE.search(text)

            if match:
                # Remove RAG content
                cleaned = text[: match.start()] + text[match.end() :]
                system_messages[i]["text"] = cleaned.strip()
                logger.debug(f"📋 RAG: Removed RAG from system message")
                break

    def _remove_rag_message(
        self,
        processed_messages: List[Dict[str, Any]],
    ) -> None:
        """
        Removes the last RAG message from processed_messages in place.
        Args:
            processed_messages: List of messages to process
        """

        # Find the last user message
        for i in range(len(processed_messages) - 1, -1, -1):
            msg = processed_messages[i]
            if msg.get("role") != "user":
                continue

            content = msg.get("content")
            if not isinstance(content, list):
                continue

            modified = False
            new_content: List[Dict[str, Any]] = []

            # Preserve original block order; only trim RAG portions inside text blocks
            for block in content:
                if block.get("type") == "text":
                    text = block.get("text", "")
                    m = PATTERN_RAG_MESSAGE.search(text)
                    if m:
                        start, end = m.span()
                        trimmed = text[:start] + text[end:]
                        # If trimmed text still has content, keep it
                        if trimmed.strip():
                            new_block = dict(block)
                            new_block["text"] = trimmed
                            new_content.append(new_block)
                        # Mark that we modified this message and continue preserving other blocks
                        modified = True
                        continue

                # Non-text blocks or text blocks without a match are preserved as-is
                new_content.append(block)

            if modified:
                processed_messages[i]["content"] = new_content
                # Only operate on the last user message that contains RAG content
                return

    def _remove_sources_from_rag(
        self, rag_content: str, filenames_to_remove: List[str]
    ) -> str:
        """
        Remove specific <source> tags from RAG content by filename.

        Args:
            rag_content: RAG message with <context> and <source> tags
            filenames_to_remove: List of filenames to remove from RAG sources

        Returns:
            str: RAG content with specified sources removed, or empty string if all sources removed
        """
        if not filenames_to_remove:
            return rag_content

        # Remove each source tag that matches the filenames
        modified = rag_content
        for filename in filenames_to_remove:
            # Match source tags with this filename in the name attribute
            # Need to escape the filename for regex but match it exactly
            pattern = re.compile(
                rf'<source[^>]*name="{re.escape(filename)}"[^>]*>.*?</source>\s*',
                re.DOTALL,
            )
            modified = pattern.sub("", modified)

        # Check if all sources were removed (only <context></context> or empty context remains)
        if PATTERN_EMPTY_CONTEXT.search(modified) or not PATTERN_SOURCE_TAGS.search(
            modified
        ):
            # All sources removed - remove entire RAG template
            logger.debug(f"📋 RAG: All sources removed, clearing entire RAG message")
            return ""

        logger.debug(
            f"📋 RAG: Removed {len(filenames_to_remove)} source(s) from RAG content"
        )
        return modified

    def _remove_specific_sources_from_rag_message(
        self,
        processed_messages: List[Dict[str, Any]],
        filenames_to_remove: List[str],
    ) -> None:
        """
        Remove specific sources from RAG messages by filename.
        Only removes the sources matching the given filenames, keeps other sources.
        If all sources are removed, the entire RAG template is removed.

        Args:
            processed_messages: List of messages to process
            filenames_to_remove: List of filenames whose sources should be removed from RAG
        """
        if not filenames_to_remove:
            return

        # Find the last user message with RAG content
        for i in range(len(processed_messages) - 1, -1, -1):
            msg = processed_messages[i]
            if msg.get("role") != "user":
                continue

            content = msg.get("content")
            if not isinstance(content, list):
                continue

            modified = False
            new_content: List[Dict[str, Any]] = []

            for block in content:
                if block.get("type") != "text":
                    new_content.append(block)
                    continue

                text = block.get("text", "")
                match = PATTERN_RAG_MESSAGE.search(text)

                if not match:
                    new_content.append(block)
                    continue

                # Found RAG content - extract and modify it
                rag_content = match.group(0)
                modified_rag = self._remove_sources_from_rag(
                    rag_content, filenames_to_remove
                )

                start, end = match.span()
                if not modified_rag:
                    # All sources removed - remove entire RAG block
                    new_text = text[:start] + text[end:]
                    logger.debug(
                        f"📋 RAG: Removed entire RAG block (all sources matched)"
                    )
                else:
                    # Some sources remain - update with modified RAG
                    new_text = text[:start] + modified_rag + text[end:]
                    logger.debug(
                        f"📋 RAG: Kept partial RAG content (some sources remain)"
                    )

                # Strip whitespace to prevent cache invalidation from leftover newlines
                new_text = new_text.strip()
                if new_text:
                    new_block = dict(block)
                    new_block["text"] = new_text
                    new_content.append(new_block)

                modified = True

            if modified:
                processed_messages[i]["content"] = new_content
                return  # Only process the first matching user message

    def _remove_attached_files_tags(
        self,
        processed_messages: List[Dict[str, Any]],
        file_ids_to_remove: List[str],
    ) -> None:
        """
        Remove <attached_files> or individual <file> tags for files that were processed natively.

        These tags are removed because:
        - Native PDF uploads are embedded as base64 in the message
        - Files API uploads are referenced via file_id
        - Code execution cannot access files by OpenWebUI URL

        Args:
            processed_messages: List of messages to process
            file_ids_to_remove: List of OpenWebUI file IDs whose attached_files tags should be removed
        """
        if not file_ids_to_remove:
            return

        # Build pattern to match individual file tags with specific IDs
        escaped_ids = [re.escape(fid) for fid in file_ids_to_remove]
        file_tag_pattern = re.compile(
            r'<file[^>]*url="(' + "|".join(escaped_ids) + r')"[^>]*/?>\s*', re.DOTALL
        )

        # Pattern to match empty attached_files blocks after removal
        # PATTERN_EMPTY_ATTACHED is pre-compiled at module level

        for msg in processed_messages:
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if block.get("type") != "text":
                    continue
                text = block.get("text", "")
                original_len = len(text)
                # First remove matching file tags
                text = file_tag_pattern.sub("", text)
                # Then remove empty attached_files blocks
                text = PATTERN_EMPTY_ATTACHED.sub("", text)
                if len(text) != original_len:
                    block["text"] = text
                    logger.debug(
                        f"📋 Removed attached_files tags for {len(file_ids_to_remove)} file(s)"
                    )

    # =========================================================================
    # FILES API (UPLOAD, DOWNLOAD, DEDUPLICATION)
    # =========================================================================

    async def _generate_file_download_link(
        self,
        file_id: str,
        api_key: str,
        user_id: str,
    ) -> str:
        """Download file from Anthropic Files API, save to OpenWebUI, return markdown link."""
        try:
            from anthropic import AsyncAnthropic
            import hashlib
            import uuid

            base_url = self.valves.ANTHROPIC_BASE_URL.strip() or None
            client = AsyncAnthropic(api_key=api_key, **({"base_url": base_url} if base_url else {}))

            # Get file metadata first
            file_meta = await client.beta.files.retrieve_metadata(file_id=file_id)
            filename = getattr(file_meta, "filename", file_id) or file_id

            # Download file content
            response = await client.beta.files.download(file_id=file_id)
            content = response.read()

            # Save to OpenWebUI storage
            owui_file_id = str(uuid.uuid4())
            storage_filename = f"code_exec_{owui_file_id}_{filename}"
            file_path = Storage.upload_file(content, storage_filename)

            # Create OpenWebUI file record
            file_hash = hashlib.sha256(content).hexdigest()
            Files.insert_new_file(
                user_id=user_id,
                form_data=type("FileForm", (), {
                    "model_dump": lambda self_: {
                        "id": owui_file_id,
                        "hash": file_hash,
                        "filename": filename,
                        "path": file_path,
                        "data": {},
                        "meta": {
                            "content_type": getattr(file_meta, "mime_type", "application/octet-stream"),
                            "size": len(content),
                            "source": "anthropic_code_execution",
                            "anthropic_file_id": file_id,
                        },
                    }
                })(),
            )

            # Return markdown download link
            base_url = os.environ.get("WEBUI_URL", "")
            download_url = f"{base_url}/api/v1/files/{owui_file_id}/content"
            return f"[📥 {filename}]({download_url})"

        except Exception as e:
            logger.error(f"Failed to download file {file_id}: {e}")
            return f"⚠️ Failed to download file {file_id}"

    async def _process_files_api_data(
        self,
        __files__: Optional[List[Dict[str, Any]]],
        __event_emitter__: Callable[[Dict[str, Any]], Awaitable[None]],
        processed_messages: List[Dict[str, Any]],
    ) -> tuple[Dict[int, List[Dict[str, Any]]], List[str]]:
        """
        Process files for Anthropic Files API using container_upload.

        Uploads files to Anthropic and caches the file_id in OpenWebUI file metadata.
        Tracks which user message each file belongs to for correct positioning.

        Returns:
            tuple: (
                Dict mapping user_msg_number → list of container_upload blocks,
                List of filenames that were processed (for RAG source removal)
            )
        """
        blocks_by_user_msg: Dict[int, List[Dict[str, Any]]] = {}
        processed_filenames: List[str] = []

        if not __files__ or not FILES_AVAILABLE:
            return blocks_by_user_msg, processed_filenames

        import io

        # Count user messages to determine "current" position for new files
        user_msg_count = sum(1 for m in processed_messages if m["role"] == "user")
        current_user_msg_num = max(0, user_msg_count - 1)  # 0-based

        client = None
        try:
            from anthropic import AsyncAnthropic
            base_url = self.valves.ANTHROPIC_BASE_URL.strip() or None
            client = AsyncAnthropic(api_key=self.valves.ANTHROPIC_API_KEY, **({"base_url": base_url} if base_url else {}))
        except ImportError:
            logger.warning("Anthropic SDK not available for file upload")
            return blocks_by_user_msg, processed_filenames

        for file in __files__:
            # Skip non-file entries (RAG chunks, knowledge base refs, etc.)
            if (
                file.get("type") != "file"
                or file.get("context") != "full"
                or file.get("collection_name")
                or file.get("docs")
            ):
                continue

            file_id_owui = file.get("id")
            file_name = file.get("name", "unknown")
            if not file_id_owui:
                continue

            # Skip images — they use Vision (base64/URL), not Files API
            content_type = file.get("content_type", "")
            if not content_type:
                # Fallback: check OpenWebUI file meta for content_type
                file_record_check = Files.get_file_by_id(file_id_owui)
                if file_record_check and file_record_check.meta:
                    content_type = file_record_check.meta.get("content_type", "")
            if content_type and content_type.startswith("image/"):
                logger.debug(f"Skipping image file for Files API: {file_name} ({content_type})")
                continue

            # Look up OpenWebUI file record for cached anthropic_file_id
            file_record = Files.get_file_by_id(file_id_owui)
            if not file_record:
                logger.warning(f"File not found in DB: {file_id_owui}")
                continue

            meta = file_record.meta or {}
            anthropic_file_id = meta.get("anthropic_file_id")
            msg_num = meta.get("anthropic_file_msg_idx")

            if anthropic_file_id:
                # Cached — reuse without re-uploading
                if msg_num is None:
                    msg_num = current_user_msg_num
                logger.debug(f"♻️ Reusing cached file {file_name} → {anthropic_file_id} (msg {msg_num})")
            else:
                # New file — upload to Anthropic
                try:
                    file_path = Storage.get_file(file_record.path)
                    if not file_path or not Path(file_path).is_file():
                        logger.warning(f"File not on disk: {file_id_owui}")
                        continue

                    with open(file_path, "rb") as f:
                        file_content = f.read()

                    await self.emit_event(
                        {
                            "type": "status",
                            "data": {"description": f"☁️ Uploading {file_name}...", "done": False},
                        },
                        __event_emitter__,
                    )

                    upload_result = await client.beta.files.upload(
                        file=(file_name, io.BytesIO(file_content)),
                    )
                    anthropic_file_id = upload_result.id
                    msg_num = current_user_msg_num

                    # Cache in OpenWebUI file metadata
                    Files.update_file_metadata_by_id(file_id_owui, {
                        "anthropic_file_id": anthropic_file_id,
                        "anthropic_file_msg_idx": msg_num,
                    })

                    logger.info(f"☁️ Uploaded {file_name} → {anthropic_file_id} (msg {msg_num})")

                    await self.emit_event(
                        {
                            "type": "status",
                            "data": {"description": f"☁️ Uploaded {file_name}", "done": True},
                        },
                        __event_emitter__,
                    )
                except Exception as e:
                    logger.error(f"Failed to upload {file_name}: {e}")
                    await self.emit_event(
                        {
                            "type": "notification",
                            "data": {"type": "warning", "content": f"Failed to upload {file_name}: {str(e)[:100]}"},
                        },
                        __event_emitter__,
                    )
                    continue

            # Group container_upload block by user message number
            if msg_num not in blocks_by_user_msg:
                blocks_by_user_msg[msg_num] = []
            blocks_by_user_msg[msg_num].append({
                "type": "container_upload",
                "file_id": anthropic_file_id,
            })
            processed_filenames.append(file_name)

        return blocks_by_user_msg, processed_filenames

    # =========================================================================
    # CACHE CONTROL
    # =========================================================================

    def _cache_control_marker(self) -> dict:
        """Return the cache_control dict based on the CACHE_TTL valve setting."""
        marker = {"type": "ephemeral"}
        if self.valves.CACHE_TTL == "1 hour":
            marker["ttl"] = "1h"
        return marker

    def _apply_cache_control(self, payload: dict, is_tool_loop: bool = False) -> None:
        """Apply cache_control breakpoints to the payload right before sending to the API.

        Called once before the initial request and once before each tool loop iteration.
        Strips all existing cache_control markers first, then applies fresh ones
        based on the current payload state and valve configuration.

        Anthropic rules:
        - Max 4 breakpoints, hierarchy: tools → system → messages
        - Cache prefixes are cumulative (hash depends on all prior blocks)
        - Never add cache_control to thinking/redacted_thinking blocks (API rejects extra fields)
        - 20-block lookback window from each explicit breakpoint
        - Minimum cacheable: 1024-4096 tokens depending on model
        - Tool_result blocks CAN have cache_control (unless programmatic calling)
        """
        cache_level = self.valves.CACHE_CONTROL
        if cache_level == "cache disabled":
            return

        # --- Step 1: Strip all existing cache_control from entire payload ---
        for tool in payload.get("tools", []):
            tool.pop("cache_control", None)
        for block in payload.get("system", []):
            block.pop("cache_control", None)
        for msg in payload.get("messages", []):
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        block.pop("cache_control", None)

        # --- Step 2: Cache tools (breakpoint 1) ---
        # Always cache tools at every non-disabled level — tools rarely change
        # and having a separate breakpoint ensures cache hits even when system/messages change.
        cache_marker = self._cache_control_marker()

        tools = payload.get("tools", [])
        if tools:
            # Find last non-deferred tool for the breakpoint
            placed = False
            for i in range(len(tools) - 1, -1, -1):
                if not tools[i].get("defer_loading", False):
                    tools[i]["cache_control"] = cache_marker
                    placed = True
                    break
            if not placed:
                # All deferred — cache the last one anyway
                tools[-1]["cache_control"] = cache_marker

        if cache_level == "cache tools array only":
            return

        # --- Step 3: Cache system prompt (breakpoint 2) ---
        system = payload.get("system", [])
        if system:
            # Find last text block with content
            for i in range(len(system) - 1, -1, -1):
                block = system[i]
                if block.get("type") == "text" and block.get("text", "").strip():
                    block["cache_control"] = cache_marker
                    break

        if cache_level == "cache tools array and system prompt":
            return

        # --- Step 4: Cache messages (breakpoint 3) ---
        # "cache tools array, system prompt and messages"
        messages = payload.get("messages", [])
        if not messages:
            return

        if is_tool_loop:
            # During tool loops: cache the last tool_result in the newest user message.
            # This caches the entire conversation (tools + system + all messages up to here)
            # so the next iteration gets a cache hit on everything.
            # EXCEPTION: Programmatic tool calling — API rejects cache_control on
            # tool_result blocks routed through code_execution.
            if self.valves.ENABLE_PROGRAMMATIC_TOOL_CALLING:
                # With programmatic calling, cache the last assistant message block instead
                # (thinking blocks excluded — find last text or tool_use block)
                for msg in reversed(messages):
                    if msg.get("role") == "assistant":
                        self._place_cache_on_last_cacheable_block(msg.get("content", []))
                        break
            else:
                # Standard tool loop: cache the last user message block (tool_result)
                for msg in reversed(messages):
                    if msg.get("role") == "user":
                        content = msg.get("content", [])
                        if content:
                            # tool_result blocks are cacheable
                            content[-1]["cache_control"] = cache_marker
                        break
        else:
            # Initial request: cache the last stable user message
            self._cache_last_stable_message(messages)

    def _place_cache_on_last_cacheable_block(self, content_blocks: list) -> None:
        """Add cache_control to the last block that isn't thinking/redacted_thinking
        or a tool_use called by code execution (API rejects cache_control on those)."""
        if not content_blocks:
            return
        for i in range(len(content_blocks) - 1, -1, -1):
            block = content_blocks[i]
            if isinstance(block, dict):
                btype = block.get("type")
                if btype in ("thinking", "redacted_thinking"):
                    continue
                # tool_use blocks called by code_execution cannot have cache_control
                if btype == "tool_use" and block.get("caller"):
                    continue
                block["cache_control"] = self._cache_control_marker()
                return

    def _cache_last_stable_message(self, messages: list) -> None:
        """Place cache breakpoint on the last stable message, avoiding RAG content
        and thinking/redacted_thinking blocks.

        RAG content changes per request (injected by OpenWebUI), so caching it
        would create a new cache entry every time, wasting cache writes.
        When RAG is detected in the last message, we cache the second-to-last instead.
        """
        if not messages:
            return

        last_msg = messages[-1]
        last_content = last_msg.get("content", [])

        # Detect RAG content in last message
        has_rag = False
        if isinstance(last_content, list):
            for block in last_content:
                if block.get("type") == "text":
                    text = block.get("text", "")
                    if "<context>" in text or ("### Task:" in text and "<source" in text):
                        has_rag = True
                        break

        target_idx = -2 if (has_rag and len(messages) >= 2) else -1
        target_msg = messages[target_idx]
        target_content = target_msg.get("content", [])

        self._place_cache_on_last_cacheable_block(target_content)

    # =========================================================================
    # PAYLOAD BUILDING & MESSAGE/TOOL CONVERSION
    # =========================================================================

    async def _create_payload(
        self,
        body: Dict,
        __metadata__: dict[str, Any],
        __user__: Dict[str, Any],
        __tools__: Optional[Dict[str, Dict[str, Any]]],
        __event_emitter__: Callable[[Dict[str, Any]], Awaitable[None]],
        __files__: Optional[List[Dict[str, Any]]] = None,
    ) -> tuple[dict, dict, List[str]]:

        ## General payload creation
        actual_model_name = body["model"].split("/")[-1]
        model_info = self.get_model_info(actual_model_name)
        max_tokens_limit = model_info["max_tokens"]
        requested_max_tokens = body.get("max_tokens", max_tokens_limit)
        max_tokens = min(requested_max_tokens, max_tokens_limit)
        payload: dict[str, Any] = {
            "model": actual_model_name,
            "max_tokens": max_tokens,
            "stream": body.get("stream", True),
            "metadata": body.get("metadata", {}),
        }
        # Sampling parameters: Opus 4.7+ returns 400 error for non-default values
        if not model_info.get("no_sampling_params"):
            if body.get("temperature") is not None:
                payload["temperature"] = float(body.get("temperature", 0))
            if body.get("top_k") is not None:
                payload["top_k"] = float(body.get("top_k", 0))
            if body.get("top_p") is not None:
                payload["top_p"] = float(body.get("top_p", 0))
        elif any(body.get(k) is not None for k in ["temperature", "top_k", "top_p"]):
            logger.debug("Skipping sampling parameters (temperature/top_k/top_p) for Opus 4.7+")

        # Add data residency if set to US (1.1x token cost)
        if self.valves.DATA_RESIDENCY == "us":
            payload["inference_geo"] = "us"

        # Add Fast Mode if enabled and model supports it (Opus 4.6 only)
        if self.valves.ENABLE_FAST_MODE and model_info.get("supports_fast_mode", False):
            payload["speed"] = "fast"
            logger.debug("Fast Mode enabled for this request")
            
        # Handle "Effort" parameter (maps from OpenWebUI's reasoning_effort or user valves)
        # Effort works differently based on model capabilities
        effort_config = None
        effective_effort = None

        if model_info["supports_effort"]:
            # Determine effective effort level
            # Priority: body.reasoning_effort > user valve
            body_effort = body.get("reasoning_effort")
            valve_effort = __user__["valves"].EFFORT

            if body_effort in ["low", "medium", "high", "max", "xhigh"]:
                requested_effort = body_effort
            else:
                requested_effort = valve_effort

            # Clamp effort levels based on model capabilities:
            # - "xhigh" requires supports_effort_xhigh (Opus 4.7+)
            # - "max" requires supports_effort_max (Opus 4.6+)
            if requested_effort == "xhigh":
                if model_info.get("supports_effort_xhigh"):
                    effective_effort = "xhigh"
                elif model_info.get("supports_effort_max"):
                    effective_effort = "max"  # Clamp to max for Opus 4.6
                else:
                    effective_effort = "high"  # Clamp to high for older models
            elif requested_effort == "max":
                if model_info.get("supports_effort_max"):
                    effective_effort = "max"
                else:
                    effective_effort = "high"  # Clamp to high for older models
            else:
                effective_effort = requested_effort

            effort_config = {"effort": effective_effort}
            logger.debug(f"Effort level set to: {effective_effort}")

        # Handle Thinking
        enable_thinking = __user__["valves"].ENABLE_THINKING or __metadata__.get(
            "anthropic_thinking", False
        )
        if enable_thinking and model_info["supports_thinking"]:
            # Opus 4.7+ (requires_adaptive_only): adaptive thinking only, with display for visibility
            # Thinking content is omitted by default in 4.7, so we set display: "summarized"
            if model_info.get("requires_adaptive_only"):
                payload["thinking"] = {"type": "adaptive", "display": "summarized"}
                logger.debug("Using adaptive thinking with display: summarized (Opus 4.7+)")
            # Opus 4.6 (supports adaptive thinking): adaptive without display override
            elif model_info.get("supports_adaptive_thinking"):
                payload["thinking"] = {"type": "adaptive"}
                logger.debug("Using adaptive thinking (Opus 4.6)")
            # Older models: use budget_tokens
            else:
                user_budget = __user__["valves"].THINKING_BUDGET_TOKENS
                max_tokens = min(
                    body.get("max_tokens", model_info["max_tokens"]),
                    model_info["max_tokens"],
                )
                context_limit = model_info.get("context_length", 200000)

                # For Claude 4 models with interleaved thinking+tools, allow up to context window
                if model_info.get("supports_thinking") and model_info.get(
                    "supports_programmatic_calling"
                ):
                    thinking_budget = min(user_budget, context_limit)
                else:
                    # budget_tokens must be < max_tokens
                    thinking_budget = (
                        min(user_budget, max_tokens - 1) if max_tokens > 1 else 1
                    )
                payload["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": thinking_budget,
                }
                logger.debug(
                    f"Using manual thinking with budget_tokens: {thinking_budget}, effort: {effective_effort}"
                )

        # Check if user has memory system enabled
        user_has_memory_system_enabled = False
        try:
            user_has_memory_system_enabled = (
                __user__.get("settings", {}).get("ui", {}).get("memory", False)
            )
        except (AttributeError, TypeError):
            pass
        logger.debug(f"Memory system enabled: {user_has_memory_system_enabled}")

        raw_messages = body.get("messages", []) or []

        system_messages, processed_messages, previous_marker_metadata = (
            self._convert_messages_to_claude_format(
                raw_messages, user_has_memory_system_enabled
            )
        )
        new_marker_metadata = ""

        # Extract container_id from previous metadata markers for multi-turn container reuse
        previous_container_id = None
        for metadata_entry in previous_marker_metadata:
            # Format: "N:container_id:ENCODED_VALUE"
            parts = metadata_entry.split(":", 2)
            if len(parts) >= 3 and parts[1] == "container_id":
                previous_container_id = unquote(parts[2])
                logger.debug(f"📦 Restored container_id from marker: {previous_container_id}")

        # Track if Files API uploaded any files (for auto-enabling code execution)
        has_files_api_uploads = False

        if __files__:
            use_files_api = __user__["valves"].USE_FILES_API

            if use_files_api:
                # Files API overrules native PDF upload — all files go as container_upload
                blocks_by_user_msg, uploaded_filenames = await self._process_files_api_data(
                    __files__, __event_emitter__, processed_messages
                )
                if blocks_by_user_msg:
                    has_files_api_uploads = True
                    # Insert container_upload blocks at the correct user messages
                    user_msg_num = 0
                    for i, msg in enumerate(processed_messages):
                        if msg["role"] == "user" and user_msg_num in blocks_by_user_msg:
                            # Ensure content is a list
                            if isinstance(msg["content"], str):
                                msg["content"] = [{"type": "text", "text": msg["content"]}]
                            msg["content"] = blocks_by_user_msg[user_msg_num] + msg["content"]
                        if msg["role"] == "user":
                            user_msg_num += 1

                    # Remove RAG sources for uploaded files
                    if uploaded_filenames:
                        logger.debug(f"📋 RAG: Removing {len(uploaded_filenames)} file source(s) from RAG")
                        self._remove_specific_sources_from_rag_message(processed_messages, uploaded_filenames)

            elif __user__["valves"].USE_PDF_NATIVE_UPLOAD:
                # Native PDF upload (base64 document blocks) — only PDFs
                pdf_documents_content_blocks, new_marker_metadata = (
                    self._get_full_context_pdfs(__files__, previous_marker_metadata)
                )
                if pdf_documents_content_blocks:
                    processed_messages[0]["content"] = (
                        pdf_documents_content_blocks + processed_messages[0]["content"]
                    )

                    # Remove RAG sources for files that were uploaded natively
                    native_pdf_filenames = []
                    for file in __files__:
                        if (
                            file.get("type") == "file"
                            and file.get("context") == "full"
                            and file.get("name", "").lower().endswith(".pdf")
                        ):
                            file_id = file.get("id")
                            filename = file.get("name")
                            if file_id and filename and not any(
                                file_id in metadata
                                for metadata in previous_marker_metadata
                            ):
                                native_pdf_filenames.append(filename)

                    if native_pdf_filenames:
                        logger.debug(
                            f"📋 RAG: Removing {len(native_pdf_filenames)} native PDF source(s) from RAG"
                        )
                        self._remove_specific_sources_from_rag_message(
                            processed_messages, native_pdf_filenames
                        )

        ## Tools Handling
        # Correct Order for Caching: Tools, System, Messages
        tools_list, api_tool_names = self._convert_tools_to_claude_format(
            __tools__, body, actual_model_name, __user__, __metadata__
        )

        activate_code_execution = __metadata__.get(
            "activate_code_execution_tool", False
        )

        # Auto-enable code execution when Files API uploaded files (container_upload needs it)
        if has_files_api_uploads:
            activate_code_execution = True

        # Auto-enable code execution when programmatic tool calling is active
        # (programmatic calling requires code execution to orchestrate tool calls)
        if (
            self.valves.ENABLE_PROGRAMMATIC_TOOL_CALLING
            and model_info.get("supports_programmatic_calling", False)
            and tools_list  # Only when there are tools to call programmatically
        ):
            activate_code_execution = True

        # Check if any dynamic filtering web tools (20260209) are in tools_list.
        # These tools cause the API to AUTO-INJECT code_execution internally.
        # We must NOT add code_execution_20250825 manually when these are present —
        # doing so triggers: "Auto-injecting tools would conflict with existing tool names"
        # However, code_execution_20260120 (programmatic) CAN coexist because we provide
        # it explicitly and the API won't auto-inject a second code_execution.
        has_dynamic_filtering_tools = any(
            t.get("type", "").endswith("_20260209") for t in tools_list
        )
        has_code_execution = any(
            t.get("name") == "code_execution" for t in tools_list
        )

        # Determine which code_execution version to add
        use_programmatic_code_exec = (
            self.valves.ENABLE_PROGRAMMATIC_TOOL_CALLING
            and model_info.get("supports_programmatic_calling", False)
        )

        if activate_code_execution and not has_code_execution:
            if use_programmatic_code_exec:
                # Always add code_execution_20260120 for programmatic calling,
                # even alongside dynamic filtering tools (it supersedes the auto-injected one)
                code_exec_type = "code_execution_20260120"
                tools_list.insert(0, {"type": code_exec_type, "name": "code_execution"})
                has_code_execution = True
            elif not has_dynamic_filtering_tools:
                # Only add code_execution_20250825 if no dynamic filtering
                # (dynamic filtering auto-injects its own code_execution)
                code_exec_type = "code_execution_20250825"
                tools_list.insert(0, {"type": code_exec_type, "name": "code_execution"})
                has_code_execution = True
            # else: dynamic filtering tools present, no programmatic → let API auto-inject

        # Create Headers - check UserValves API key first
        user_valves = __user__.get("valves") if __user__ else None
        user_api_key = getattr(user_valves, "ANTHROPIC_API_KEY", "") if user_valves else ""
        api_key = user_api_key.strip() if user_api_key and user_api_key.strip() else self.valves.ANTHROPIC_API_KEY

        headers = {
            "x-api-key": api_key,
            "anthropic-version": self.API_VERSION,
            "content-type": "application/json",
        }

        beta_headers: list[str] = []

        # Enable prompt caching if not disabled
        if self.valves.CACHE_CONTROL != "cache disabled":
            beta_headers.append("prompt-caching-2024-07-31")

        # Add code-execution beta header ONLY when we explicitly added code_execution to tools.
        # Do NOT add when using dynamic filtering v20260209 web tools — those auto-inject
        # code_execution internally and the beta header would cause a second injection → duplicate error.
        if has_code_execution:
            # code_execution_20260120 doesn't need the old beta header
            code_exec_is_new = any(
                t.get("type") == "code_execution_20260120" for t in tools_list
            )
            if not code_exec_is_new:
                beta_headers.append("code-execution-2025-08-25")
            if activate_code_execution:
                beta_headers.append("files-api-2025-04-14")
        if (
            self.valves.ENABLE_INTERLEAVED_THINKING
            and model_info["supports_thinking"]
            and not model_info["supports_adaptive_thinking"]
        ):
            beta_headers.append("interleaved-thinking-2025-05-14")

        # Add web_fetch beta header when using the older version (20250910)
        # The newer 20260209 version doesn't need a beta header
        uses_old_web_fetch = any(
            t.get("type") == "web_fetch_20250910" for t in tools_list
        )
        if self.valves.WEB_FETCH and uses_old_web_fetch:
            beta_headers.append("web-fetch-2025-09-10")

        # Add Files API beta header when files were uploaded but code_execution
        # wasn't otherwise activated (standalone file upload scenario)
        if has_files_api_uploads and "files-api-2025-04-14" not in beta_headers:
            beta_headers.append("files-api-2025-04-14")

        # Skills Integration
        if activate_code_execution and __user__["valves"].SKILLS:
            beta_headers.append("skills-2025-10-02")

            # Validate skills (cached to avoid API calls on every turn)
            validated_skills = await self._validate_and_get_skills(
                __user__["valves"].SKILLS,
                self.valves.ANTHROPIC_API_KEY,
                __event_emitter__,
            )
            payload["container"] = validated_skills
            logger.debug(f"🔧 Added {len(validated_skills)} skills")
        elif previous_container_id:
            # Reuse container from previous turn for code execution state continuity
            payload["container"] = previous_container_id
            logger.info(f"📦 Reusing container from previous turn: {previous_container_id}")

        # Add advanced tool use beta (for programmatic calling and tool search)
        if self.valves.ENABLE_TOOL_SEARCH or self.valves.ENABLE_PROGRAMMATIC_TOOL_CALLING:
            beta_headers.append("advanced-tool-use-2025-11-20")

        # Add context editing strategies if enabled
        context_editing_strategy = self.valves.CONTEXT_EDITING_STRATEGY
        if context_editing_strategy != "none":
            if "context-management-2025-06-27" not in beta_headers:
                beta_headers.append("context-management-2025-06-27")

            # Build context_management array for payload
            # IMPORTANT: clear_thinking must be FIRST if present (API requirement)
            context_management = []

            # Add clear_thinking FIRST if needed
            if (
                context_editing_strategy in ["clear_thinking", "clear_both"]
                and enable_thinking
                and model_info["supports_thinking"]
            ):
                clear_thinking = {
                    "type": "clear_thinking_20251015",
                    "keep": {
                        "type": "thinking_turns",
                        "value": self.valves.CONTEXT_EDITING_THINKING_KEEP,
                    },
                }
                context_management.append(clear_thinking)

            # Add clear_tool_uses SECOND
            if (
                context_editing_strategy in ["clear_tool_results", "clear_both"]
                and len(tools_list) > 2
            ):
                clear_tool_uses = {
                    "type": "clear_tool_uses_20250919",
                    "trigger": {
                        "type": "input_tokens",
                        "value": self.valves.CONTEXT_EDITING_TOOL_TRIGGER,
                    },
                    "keep": {
                        "type": "tool_uses",
                        "value": self.valves.CONTEXT_EDITING_TOOL_KEEP,
                    },
                }
                if self.valves.CONTEXT_EDITING_TOOL_CLEAR_AT_LEAST > 0:
                    clear_tool_uses["clear_at_least"] = {
                        "type": "input_tokens",
                        "value": self.valves.CONTEXT_EDITING_TOOL_CLEAR_AT_LEAST,
                    }
                if self.valves.CONTEXT_EDITING_TOOL_CLEAR_TOOL_INPUT:
                    clear_tool_uses["clear_tool_inputs"] = True
                context_management.append(clear_tool_uses)

            if context_management:
                payload["context_management"] = {"edits": context_management}

        # Add effort beta header and output_config if effort is configured
        if model_info["supports_effort"] and effort_config:
            beta_headers.append("effort-2025-11-24")
            payload["output_config"] = effort_config

        # Add Fast Mode beta header if enabled and model supports it
        if self.valves.ENABLE_FAST_MODE and model_info.get("supports_fast_mode", False):
            beta_headers.append("fast-mode-2026-02-01")

        if beta_headers and len(beta_headers) > 0:
            headers["anthropic-beta"] = ",".join(beta_headers)
            # Add betas list to payload for beta.messages.stream
            payload["betas"] = beta_headers

            ## Tool Choice Handling
            if __metadata__.get("web_search_enforced"):
                # Check if web_search is actually in the tools list
                has_web_search = any(t.get("name") == "web_search" for t in tools_list)
                if has_web_search:
                    if "thinking" not in payload:
                        # No thinking active - enforce web_search
                        payload["tool_choice"] = {"type": "tool", "name": "web_search"}
                        logger.debug("Enforcing web_search via tool_choice")
                    else:
                        # Thinking is active - cannot enforce web_search, but it's still available
                        payload["tool_choice"] = {"type": "auto"}
                        logger.debug(
                            "Thinking active - web_search added but not enforced (tool_choice=auto)"
                        )
                else:
                    # No enforcement - use auto tool choice
                    payload["tool_choice"] = {"type": "auto"}

        # API tool_choice passthrough (outside beta_headers block)
        # If no tool_choice was set by web_search enforcement, pass through from body
        if "tool_choice" not in payload and body.get("tool_choice"):
            api_tc = body["tool_choice"]
            if isinstance(api_tc, dict) and "function" in api_tc:
                # OpenAI format: {"type": "function", "function": {"name": "X"}}
                payload["tool_choice"] = {
                    "type": "tool",
                    "name": api_tc["function"]["name"],
                }
            elif isinstance(api_tc, str):
                # OpenAI string format: "auto", "none", "required"
                mapping = {"auto": "auto", "none": "none", "required": "any"}
                payload["tool_choice"] = {"type": mapping.get(api_tc, api_tc)}
            else:
                # Already in Anthropic format or other dict format
                payload["tool_choice"] = api_tc
            logger.debug(f"API tool_choice passthrough: {payload['tool_choice']}")

        payload["tools"] = tools_list

        # Processing Messages and Caching
        if system_messages and len(system_messages) > 0:
            payload["system"] = system_messages

        payload["messages"] = processed_messages

        return payload, headers, new_marker_metadata, api_tool_names

    def _convert_messages_to_claude_format(
        self, raw_messages, user_has_memory_system_enabled: bool = False
    ) -> tuple[list[dict], list[dict], list[str]]:
        processed_messages: list[Dict[str, Any]] = []
        extracted_memories = None
        previous_marker_metadata: list[str] = []
        system_messages = []
        if raw_messages is None or len(raw_messages) == 0:
            return system_messages, processed_messages, previous_marker_metadata

        for i, msg in enumerate(raw_messages):
            role = msg.get("role")
            raw_content = msg.get("content")

            # Check if this is an assistant message with tool call history in HTML
            # This needs to be handled specially to reconstruct proper API message sequence
            if (
                role == "assistant"
                and isinstance(raw_content, str)
                and '<details type="tool_calls"' in raw_content
            ):
                # Parse tool history from HTML
                tool_use_blocks, tool_result_blocks, text_before, text_after = (
                    self._parse_tool_history_from_html(raw_content)
                )

                if tool_use_blocks:
                    # Reconstruct proper message sequence for tool history
                    logger.debug(
                        f"🔧 Reconstructing tool history: {len(tool_use_blocks)} tool_use, "
                        f"{len(tool_result_blocks)} tool_result blocks"
                    )

                    # 1. Assistant message with text before + tool_use blocks
                    assistant_content = []
                    if text_before:
                        # Clean the text (strip reasoning/code_interpreter blocks)
                        cleaned_before = PATTERN_REASONING_DETAILS.sub("", text_before)
                        cleaned_before = PATTERN_CODE_INTERPRETER_DETAILS.sub(
                            "", cleaned_before
                        ).strip()
                        if cleaned_before:
                            assistant_content.append(
                                {"type": "text", "text": cleaned_before}
                            )
                    assistant_content.extend(tool_use_blocks)

                    if assistant_content:
                        wrapped_msg = {"role": "assistant", "content": assistant_content}
                        extracted_metadata = self._extract_metadata_marker_from_message(
                            wrapped_msg
                        )
                        if extracted_metadata:
                            previous_marker_metadata.extend(extracted_metadata)
                        processed_messages.append(wrapped_msg)

                    # 2. User message with tool_result blocks
                    if tool_result_blocks:
                        processed_messages.append(
                            {"role": "user", "content": tool_result_blocks}
                        )

                    # 3. If there's text after, create another assistant message
                    if text_after:
                        # Clean the text
                        cleaned_after = PATTERN_REASONING_DETAILS.sub("", text_after)
                        cleaned_after = PATTERN_CODE_INTERPRETER_DETAILS.sub(
                            "", cleaned_after
                        ).strip()
                        if cleaned_after:
                            wrapped_msg = {
                                "role": "assistant",
                                "content": [{"type": "text", "text": cleaned_after}],
                            }
                            extracted_metadata = (
                                self._extract_metadata_marker_from_message(wrapped_msg)
                            )
                            if extracted_metadata:
                                previous_marker_metadata.extend(extracted_metadata)
                            processed_messages.append(wrapped_msg)

                    continue  # Skip normal processing - we've handled this message

            # Normal processing for messages without tool history HTML
            claude_message = self._convert_content_to_claude_format(raw_content, role=role)
            if not claude_message:
                continue
            if role == "system":
                for block in claude_message:
                    text = block["text"]

                    # Only extract memory if user has memory system enabled
                    if user_has_memory_system_enabled:
                        # Extract and remove User Context
                        cleaned_text, extracted_memories = (
                            self._extract_and_remove_memories(text)
                        )

                        if extracted_memories:
                            logger.debug(
                                f"✓ Extracted User Context: {extracted_memories[:100]}..."
                            )
                            logger.debug(
                                f"✓ System prompt after removal (last 200 chars): ...{cleaned_text[-200:]}"
                            )

                        # Update block with cleaned text
                        block["text"] = cleaned_text

                    # Only add non-empty blocks to system (cache_control will be added later to last block only)
                    if block["text"].strip():
                        system_messages.append(block)
            else:
                # Wrap as dict so _extract_metadata_marker_from_message can check role
                # and modify content blocks in-place to strip markers
                wrapped_msg = {"role": role, "content": claude_message}
                extracted_metadata = self._extract_metadata_marker_from_message(
                    wrapped_msg
                )
                if extracted_metadata:
                    previous_marker_metadata.extend(extracted_metadata)

                processed_messages.append(wrapped_msg)

                if (
                    user_has_memory_system_enabled
                    and i == len(raw_messages) - 1
                    and role == "user"
                    and extracted_memories
                ):
                    # Append marker metadata and memories back to last message
                    processed_messages[-1]["content"].append(
                        {
                            "type": "text",
                            "text": f"\n\n---\n**IMPORTANT:** The following is NOT part of the user's message, but context from a memory system to help answer the user's questions:\n\n{extracted_memories}",
                        }
                    )
        return system_messages, processed_messages, previous_marker_metadata

    def _convert_tools_to_claude_format(
        self,
        __tools__,
        body: Dict[str, Any],
        actual_model_name: str,
        __user__: Dict[str, Any],
        __metadata__: dict[str, Any],
    ) -> tuple[List[dict], set]:
        """
        Convert OpenWebUI tools format to Claude API format.

        Extracts tool specs from TWO sources:
        1. body.tools - Built-in tools (OpenAI format specs only, no callables)
        2. __tools__ - User tools (specs + callables for execution)

        Args:
            __tools__: Dict of user tools with callables from OpenWebUI
            body: Request body containing body.tools (built-in tool specs)
            actual_model_name: Model name for capability checking
            __user__: User dict for valve overrides
            __metadata__: Metadata dict for checking enforcement flags
        Returns:
            tuple: (Tools in Claude API format, set of API-provided tool names without callables)
        """
        claude_tools = []
        tool_names_seen = set()  # Track unique tool names
        api_tool_names = set()  # Track tools from body.tools (no callable, API passthrough)

        # Names reserved for Anthropic server-side tools (skip if found in body.tools)
        anthropic_server_tool_names = {"web_search", "web_fetch", "memory"}

        # Extract built-in tools from body.tools (OpenAI format)
        body_tools = body.get("tools", [])
        if body_tools:
            logger.debug(f"Found {len(body_tools)} built-in tools in body.tools")
            for tool_entry in body_tools:
                if tool_entry.get("type") == "function":
                    func = tool_entry.get("function", {})
                    name = func.get("name")
                    if not name or name in tool_names_seen:
                        continue

                    # Skip tools that will be handled by Anthropic server-side tools
                    if name in anthropic_server_tool_names:
                        logger.info(f"Skipping body tool '{name}' — handled by Anthropic server tool")
                        continue

                    # Convert OpenAI format to Claude format
                    claude_tool = {
                        "name": name,
                        "description": func.get("description", f"Tool: {name}"),
                        "input_schema": func.get(
                            "parameters", {"type": "object", "properties": {}}
                        ),
                    }
                    claude_tools.append(claude_tool)
                    tool_names_seen.add(name)
                    # Track as API-provided tool (no callable — for passthrough)
                    if not (__tools__ and name in __tools__ and __tools__[name].get("callable")):
                        api_tool_names.add(name)

        # Log user tools from __tools__
        if __tools__ and logger.isEnabledFor(logging.DEBUG):
            # Only attempt serialization if DEBUG is enabled
            try:
                logger.debug(
                    f"Converting {len(__tools__)} user tools: {json.dumps(__tools__, indent=2)}"
                )
            except (TypeError, ValueError):
                # Log tool names only if full serialization fails
                tool_names = list(__tools__.keys())[:10]
                logger.debug(
                    f"Converting {len(__tools__)} user tools (names): {tool_names}{'...' if len(__tools__) > 10 else ''}"
                )
        elif not __tools__:
            logger.debug("No user tools to convert")

        # Add web search tool if enabled OR if metadata enforces it (even if valve is disabled)
        web_search_enabled = self.valves.WEB_SEARCH or __metadata__.get(
            "web_search_enforced", False
        )
        if web_search_enabled:
            # Get user location values with fallback to global valves
            city = (
                __user__["valves"].WEB_SEARCH_USER_CITY
                or self.valves.WEB_SEARCH_USER_CITY
            )
            region = (
                __user__["valves"].WEB_SEARCH_USER_REGION
                or self.valves.WEB_SEARCH_USER_REGION
            )
            country = (
                __user__["valves"].WEB_SEARCH_USER_COUNTRY
                or self.valves.WEB_SEARCH_USER_COUNTRY
            )
            timezone = (
                __user__["valves"].WEB_SEARCH_USER_TIMEZONE
                or self.valves.WEB_SEARCH_USER_TIMEZONE
            )

            # Build web search tool config
            # web_search_20260209 has dynamic filtering (code execution post-processes results)
            # web_search_20250305 works on all models without dynamic filtering
            model_info_ws = self.get_model_info(actual_model_name)
            use_dynamic = __user__["valves"].ENABLE_DYNAMIC_FILTERING
            if use_dynamic and model_info_ws.get("supports_dynamic_filtering", False):
                web_search_type = "web_search_20260209"
            else:
                web_search_type = "web_search_20250305"
            web_search_tool = {
                "type": web_search_type,
                "name": "web_search",
            }
            # max_uses is only supported on web_search_20250305 (non-dynamic filtering)
            # Dynamic filtering versions (20260209) don't document max_uses support
            if web_search_type == "web_search_20250305":
                web_search_tool["max_uses"] = __user__["valves"].WEB_SEARCH_MAX_USES

            # Only add user_location if at least one field has a value.
            # Only include non-empty fields to avoid Anthropic API validation errors
            # (e.g. country must be ISO 3166-1 alpha-2, can't be empty string)
            if city or region or country or timezone:
                loc: dict = {"type": "approximate"}
                if city:
                    loc["city"] = city
                if region:
                    loc["region"] = region
                if country:
                    loc["country"] = country
                if timezone:
                    loc["timezone"] = timezone
                web_search_tool["user_location"] = loc

            claude_tools.append(web_search_tool)
            tool_names_seen.add("web_search")
            logger.debug(f"Added web_search tool: {web_search_type}")

        # Add web_fetch tool if enabled
        # web_fetch_20260209 has dynamic filtering (requires code execution)
        # web_fetch_20250910 works on all models without dynamic filtering
        model_info = self.get_model_info(actual_model_name)
        if self.valves.WEB_FETCH:
            use_dynamic_fetch = __user__["valves"].ENABLE_DYNAMIC_FILTERING
            if use_dynamic_fetch and model_info.get("supports_dynamic_filtering", False):
                web_fetch_type = "web_fetch_20260209"
            else:
                web_fetch_type = "web_fetch_20250910"
            web_fetch_tool = {
                "type": web_fetch_type,
                "name": "web_fetch",
            }
            # max_uses is only supported on web_fetch_20250910 (non-dynamic filtering)
            # Dynamic filtering versions (20260209) don't document max_uses support
            if web_fetch_type == "web_fetch_20250910":
                web_fetch_tool["max_uses"] = __user__["valves"].WEB_FETCH_MAX_USES
            claude_tools.append(web_fetch_tool)
            tool_names_seen.add("web_fetch")
            logger.debug(f"Added web_fetch tool: {web_fetch_type}")

        # Add Claude Memory tool if enabled and supported by model
        # if self.valves.ENABLE_CLAUDE_MEMORY and model_info.get("supports_memory", False):
        #     claude_tools.append(
        #         {
        #             "type": "memory_20250818",
        #             "name": "memory"
        #         }
        #     )
        #     tool_names_seen.add("memory")

        # Process user tools from __tools__ (these have callables for execution)
        if __tools__ and len(__tools__) > 0:
            for tool_name, tool_data in __tools__.items():
                if not isinstance(tool_data, dict) or "spec" not in tool_data:
                    logger.debug(f"Skipping invalid tool: {tool_name} - missing spec")
                    continue

                spec = tool_data["spec"]

                # Extract basic tool info
                name = spec.get("name", tool_name)

                # Skip if tool name already exists
                if name in tool_names_seen:
                    continue

                # Skip if toolname starts with _ or __
                if name.startswith("_"):
                    logger.debug(f"Skipping private tool: {name}")
                    continue

                description = spec.get("description", f"Tool: {name}")
                parameters = spec.get("parameters", {})

                # Convert OpenWebUI parameters to Claude input_schema format
                # OpenWebUI parameters are typically already in JSON Schema format
                input_schema = {
                    "type": "object",
                    "properties": parameters.get("properties", {}),
                }

                # Add required fields if they exist
                if "required" in parameters:
                    input_schema["required"] = parameters["required"]

                # Create Claude tool format
                claude_tool = {
                    "name": name,
                    "description": description,
                    "input_schema": input_schema,
                }

                claude_tools.append(claude_tool)
                tool_names_seen.add(name)

        # Check if programmatic tool calling is active for this model
        # When active, tools must NOT be deferred (defer_loading) because
        # deferred tools loaded via tool_search may bypass allowed_callers enforcement
        is_programmatic_active = False
        if self.valves.ENABLE_PROGRAMMATIC_TOOL_CALLING:
            model_info_ptc = self.get_model_info(actual_model_name)
            is_programmatic_active = model_info_ptc.get("supports_programmatic_calling", False)

        for claude_tool in claude_tools:
            # Check if tool should be deferred for tool search
            # IMPORTANT: Skip deferring when programmatic tool calling is active
            if self.valves.ENABLE_TOOL_SEARCH and not is_programmatic_active:
                # Skip deferring if tool is in exclusion list
                name = claude_tool["name"]
                if name not in self.valves.TOOL_SEARCH_EXCLUDE_TOOLS:
                    # Calculate tool definition size (JSON representation)
                    tool_json = json.dumps(claude_tool)
                    tool_len = len(tool_json)
                    if len(tool_json) > self.valves.TOOL_SEARCH_MAX_DESCRIPTION_LENGTH:
                        claude_tool["defer_loading"] = True
                    else:
                        logger.debug(f"Tool '{name}' will be loaded normally")

            # Add allowed_callers for programmatic tool calling (only if model supports it)
            # When enabled, tools can be called from code execution
            # With code_execution_20260120 explicitly in the tools list, we can safely
            # add allowed_callers even alongside dynamic filtering tools (20260209) —
            # the explicit code_execution_20260120 supersedes auto-injection.
            if self.valves.ENABLE_PROGRAMMATIC_TOOL_CALLING:
                model_info = self.get_model_info(actual_model_name)
                if model_info.get("supports_programmatic_calling", False):
                    # Only add to user-defined tools (not server tools like web_search, web_fetch, memory)
                    if "type" not in claude_tool:  # Server tools have a "type" field
                        claude_tool["allowed_callers"] = ["code_execution_20260120"]

            # Enable fine-grained tool streaming for user-defined tools
            # Streams tool input JSON without buffering, reducing latency for large inputs
            # GA on all models, no beta header required
            if "type" not in claude_tool:  # Only user-defined tools (not server tools)
                claude_tool["eager_input_streaming"] = True

        if any(tool.get("defer_loading", False) for tool in claude_tools):
            if self.valves.TOOL_SEARCH_TYPE == "regex":
                tool_search_tool = {
                    "type": "tool_search_tool_regex_20251119",
                    "name": "tool_search_tool_regex",
                }
            else:  # bm25 (default)
                tool_search_tool = {
                    "type": "tool_search_tool_bm25_20251119",
                    "name": "tool_search_tool_bm25",
                }
            claude_tools.insert(0, tool_search_tool)

        logger.debug(f"Total tools converted: {len(claude_tools)}")
        for t in claude_tools:
            flags = []
            if t.get("defer_loading"):
                flags.append("DEFERRED")
            if t.get("allowed_callers"):
                flags.append(f"callers={t['allowed_callers']}")
            if t.get("type"):
                flags.append(f"type={t['type']}")
            if t.get("eager_input_streaming"):
                flags.append("eager_stream")
            logger.info(f"  🔧 Tool: {t.get('name')} [{', '.join(flags) or 'normal'}]")

        return claude_tools, api_tool_names

    def _parse_tool_history_from_html(
        self, text: str
    ) -> tuple[list[dict], list[dict], str, str]:
        """
        Parse tool call HTML blocks from assistant message text and reconstruct
        proper Anthropic API structures.

        Returns:
            tuple of (tool_use_blocks, tool_result_blocks, text_before, text_after)
            - tool_use_blocks: List of {"type": "tool_use", "id": ..., "name": ..., "input": ...}
            - tool_result_blocks: List of {"type": "tool_result", "tool_use_id": ..., "content": ...}
            - text_before: Text before the first tool_call block
            - text_after: Text after the last tool_call block
        """
        tool_use_blocks = []
        tool_result_blocks = []

        # Find all tool_calls blocks with their positions
        matches = list(PATTERN_TOOL_CALLS_PARSE.finditer(text))

        if not matches:
            # No tool calls found - return empty lists and full text as "after"
            # (will be cleaned up by _convert_content_to_claude_format)
            return [], [], "", text

        # Track positions for text splitting
        first_match_start = matches[0].start()
        last_match_end = matches[-1].end()

        # Extract text before first tool and after last tool
        text_before = text[:first_match_start].strip()
        text_after = text[last_match_end:].strip()

        # Parse each tool_calls block
        for match in matches:
            tool_id = match.group(1)
            tool_name = match.group(2)
            arguments_escaped = match.group(3)
            result_escaped = match.group(4)  # May be None if no result

            # Unescape HTML entities in arguments
            arguments_str = html.unescape(arguments_escaped) if arguments_escaped else ""

            # Parse arguments JSON
            try:
                if arguments_str.strip():
                    tool_input = json.loads(arguments_str)
                else:
                    tool_input = {}
            except json.JSONDecodeError:
                logger.debug(f"Failed to parse tool arguments JSON: {arguments_str[:100]}")
                tool_input = {}

            # Create tool_use block
            tool_use_blocks.append({
                "type": "tool_use",
                "id": tool_id,
                "name": tool_name,
                "input": tool_input,
            })

            # Create tool_result block if we have a result
            if result_escaped is not None:
                result_content = html.unescape(result_escaped)
                tool_result_blocks.append({
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": result_content,
                })

        return tool_use_blocks, tool_result_blocks, text_before, text_after

    def _convert_content_to_claude_format(
        self, content: Union[str, List[dict], None], role: str = "user"
    ) -> List[dict]:
        """
        Process content from OpenWebUI format to Claude API format.
        Handles text, images, PDFs, tool_calls, and tool_results according to
        Anthropic API documentation.
        Filters out empty text blocks to prevent API errors.
        """
        if content is None:
            return []

        if isinstance(content, str):
            # Strip OpenWebUI UI-rendering artifacts from conversation history.
            # NOTE: tool_calls blocks are parsed and reconstructed in _convert_messages_to_claude_format
            # before this method is called. Here we just clean up any remaining UI artifacts.
            if role == "assistant":
                # Strip any remaining tool_calls blocks (e.g., malformed or in-progress ones)
                content = PATTERN_TOOL_CALLS_STRIP.sub("", content)
                # Strip code_interpreter blocks (display-only UI artifact)
                content = PATTERN_CODE_INTERPRETER_DETAILS.sub("", content)
                # Strip thinking/reasoning blocks from prior turns.
                # We can't reconstruct them as proper API thinking blocks (missing signature),
                # and the API auto-filters prior turn thinking anyway. Keeping them as HTML text
                # just wastes tokens on unusable content.
                content = PATTERN_REASONING_DETAILS.sub("", content)

            # Only return non-empty text blocks
            if content.strip():
                return [{"type": "text", "text": content}]
            else:
                return []

        processed_content = []
        for item in content:
            if item.get("type") == "text":
                text_content = item.get("text", "")
                # Only add non-empty text blocks (Anthropic API requirement)
                if text_content.strip():
                    processed_content.append({"type": "text", "text": text_content})

            elif item.get("type") == "image_url":
                image_url = item.get("image_url", {}).get("url", "")

                if image_url.startswith("data:image"):
                    # Handle base64 encoded image data
                    try:
                        header, encoded = image_url.split(",", 1)
                        mime_type = header.split(":")[1].split(";")[0]

                        # Validate supported image formats according to Anthropic docs
                        supported_formats = [
                            "image/jpeg",
                            "image/png",
                            "image/gif",
                            "image/webp",
                        ]

                        if mime_type not in supported_formats:
                            logger.debug(f" Unsupported image mime type: {mime_type}")
                            processed_content.append(
                                {
                                    "type": "text",
                                    "text": f"[Image type {mime_type} not supported. Supported formats: JPEG, PNG, GIF, WebP]",
                                }
                            )
                            continue

                        # Check image size - API has 32MB request limit, but be conservative
                        MAX_IMAGE_SIZE = 25 * 1024 * 1024  # 25 MB (conservative)
                        try:
                            decoded_bytes = base64.b64decode(encoded)
                            if len(decoded_bytes) > MAX_IMAGE_SIZE:
                                logger.debug(
                                    f" Image too large: {len(decoded_bytes)} bytes"
                                )
                                processed_content.append(
                                    {
                                        "type": "text",
                                        "text": f"[Image too large for Anthropic API. Max size: 25MB, received: {len(decoded_bytes)//1024//1024}MB]",
                                    }
                                )
                                continue
                        except Exception as decode_ex:
                            logger.debug(f" Image base64 decode failed: {decode_ex}")
                            processed_content.append(
                                {
                                    "type": "text",
                                    "text": "[Image data could not be decoded - invalid base64 format]",
                                }
                            )
                            continue

                        processed_content.append(
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": mime_type,
                                    "data": encoded,
                                },
                            }
                        )

                    except ValueError as e:
                        logger.debug(f"Error parsing image data URL: {e}")
                        processed_content.append(
                            {
                                "type": "text",
                                "text": "[Error processing image - invalid data URL format]",
                            }
                        )
                    except Exception as e:
                        logger.debug(f"Unexpected error processing image: {e}")
                        processed_content.append(
                            {
                                "type": "text",
                                "text": "[Unexpected error processing image]",
                            }
                        )
                else:
                    # For image URLs (not base64), Claude API supports URL references
                    if image_url.startswith(("http://", "https://")):
                        processed_content.append(
                            {
                                "type": "image",
                                "source": {"type": "url", "url": image_url},
                            }
                        )
                    else:
                        processed_content.append(
                            {
                                "type": "text",
                                "text": f"[Invalid image URL format: {image_url}. Only HTTP/HTTPS URLs are supported]",
                            }
                        )

            elif item.get("type") == "tool_calls":
                converted_calls = self._process_tool_calls(item)
                processed_content.extend(converted_calls)

            elif item.get("type") == "tool_results":
                converted_results = self._process_tool_results(item)
                processed_content.extend(converted_results)

            else:
                logger.debug(
                    f" Unknown content type: {item.get('type')}, converting to text"
                )
                processed_content.append(
                    {
                        "type": "text",
                        "text": f"[Unsupported content type: {item.get('type')}]",
                    }
                )

        return processed_content

    def _process_tool_calls(self, tool_calls_item):
        """Convert OpenWebUI tool_calls format to Claude tool_use format."""
        claude_tool_uses = []
        if "tool_calls" in tool_calls_item:
            for tool_call in tool_calls_item["tool_calls"]:
                if tool_call.get("type") == "function" and "function" in tool_call:
                    function_def = tool_call["function"]
                    claude_tool_uses.append({
                        "type": "tool_use",
                        "id": tool_call.get("id", ""),
                        "name": function_def.get("name", ""),
                        "input": function_def.get("arguments", {}),
                    })
        return claude_tool_uses

    def _process_tool_results(self, tool_results_item):
        """Convert OpenWebUI tool_results format to Claude tool_result format."""
        claude_tool_results = []
        if "results" in tool_results_item:
            for result_item in tool_results_item["results"]:
                if "call" in result_item and "result" in result_item:
                    tool_call = result_item["call"]
                    tool_use_id = tool_call.get("id", "")
                    if tool_use_id:
                        claude_tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": str(result_item["result"]),
                        })
        return claude_tool_results

    def _safe_json(self, obj: Any) -> Any:
        """Recursively convert obj to JSON-serializable form."""

        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        if isinstance(obj, dict):
            return {k: self._safe_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._safe_json(v) for v in obj]
        if hasattr(obj, "dict"):
            try:
                return self._safe_json(obj.dict())
            except Exception:
                pass
        if hasattr(obj, "model_dump"):
            try:
                return self._safe_json(obj.model_dump())
            except Exception:
                pass
        return f"<UNSERIALIZABLE {type(obj).__name__}>"

    async def _emit_debug_citation(
        self,
        request_ctx: PipeRequestContext,
        name: str,
        value: Any,
    ) -> None:
        serial = self._safe_json(value)
        await request_ctx.emit_event(
            {
                "type": "citation",
                "data": {
                    "document": [json.dumps(serial, indent=2)],
                    "metadata": [{"source": name}],
                    "source": {"name": name},
                },
            }
        )

    def _handle_message_start_usage(
        self,
        event: Any,
        *,
        include_usage: bool,
        total_usage: Optional[dict[str, int]],
        stream_output_tokens: int,
    ) -> int:
        """Handle message_start usage accounting and return updated stream output tokens."""

        message = getattr(event, "message", None)
        if not message:
            return stream_output_tokens

        request_id = getattr(message, "id", None)
        logger.debug(f" Message started with ID: {request_id}")

        if not include_usage or total_usage is None:
            return stream_output_tokens

        usage = getattr(message, "usage", {})
        if not usage:
            return stream_output_tokens

        input_tokens = getattr(usage, "input_tokens", 0)
        current_output_tokens = getattr(usage, "output_tokens", 0)

        total_usage["input_tokens"] += input_tokens
        diff = current_output_tokens - stream_output_tokens
        total_usage["output_tokens"] += diff
        stream_output_tokens = current_output_tokens
        total_usage["total_tokens"] = input_tokens + current_output_tokens

        if self.valves.CACHE_CONTROL != "cache disabled":
            cache_creation_input_tokens = getattr(usage, "cache_creation_input_tokens", 0) or 0
            cache_read_input_tokens = getattr(usage, "cache_read_input_tokens", 0) or 0
            total_usage["cache_creation_input_tokens"] += cache_creation_input_tokens
            total_usage["cache_read_input_tokens"] = cache_read_input_tokens
            total_usage["total_tokens"] += cache_creation_input_tokens + cache_read_input_tokens
            logger.debug(
                f" Usage stats: input={input_tokens}, output={current_output_tokens}, "
                f"cache_creation={cache_creation_input_tokens}, cache_read={cache_read_input_tokens}"
            )
        else:
            logger.debug(f" Usage stats: input={input_tokens}, output={current_output_tokens}")
        logger.debug(f" Accumulated usage: {total_usage}")

        return stream_output_tokens

    async def _handle_stream_exception(
        self,
        exc: Exception,
        *,
        retry_attempts: int,
        request_ctx: PipeRequestContext,
    ) -> tuple[bool, int, str]:
        """Central stream exception policy.

        Returns: (should_retry, updated_retry_attempts, response_suffix)
        """

        max_retries = self.valves.MAX_RETRIES

        non_retry_map: dict[type[Exception], str] = {
            RateLimitError: f"\n\n⚠️ Rate limit exceeded - maximum retries ({max_retries}) reached. Please try again later.",
            AuthenticationError: f"\n\nError: API key issues. Reason: {getattr(exc, 'message', str(exc))}",
            PermissionDeniedError: f"\n\nError: Permission denied. Reason: {getattr(exc, 'message', str(exc))}",
            NotFoundError: f"\n\nError: Resource not found. Reason: {getattr(exc, 'message', str(exc))}",
            BadRequestError: f"\n\nError: Invalid request format. Reason: {getattr(exc, 'message', str(exc))}",
            UnprocessableEntityError: f"\n\nError: Unprocessable entity. Reason: {getattr(exc, 'message', str(exc))}",
        }

        for error_type, suffix in non_retry_map.items():
            if isinstance(exc, error_type):
                await self.handle_errors(exc, request_ctx.event_emitter)
                return (False, retry_attempts, suffix)

        retryable_with_status: list[tuple[type[Exception], str, str]] = [
            (OverloadedError, "⏳ API overloaded, retrying...", "🔧 API overloaded"),
            (InternalServerError, "⏳ Server error, retrying...", "🔧 Server error"),
            (APIConnectionError, "🌐 Connection error, retrying...", "🌐 Network connection failed"),
        ]

        for error_type, status_label, fail_label in retryable_with_status:
            if isinstance(exc, error_type):
                retry_attempts += 1
                if retry_attempts <= max_retries:
                    await request_ctx.emit_event(
                        {
                            "type": "status",
                            "data": {
                                "description": f"{status_label} ({retry_attempts}/{max_retries})",
                                "done": False,
                            },
                        }
                    )
                    return (True, retry_attempts, "")

                await self.handle_errors(exc, request_ctx.event_emitter)
                if isinstance(exc, APIConnectionError):
                    return (
                        False,
                        retry_attempts,
                        f"\n\n{fail_label} after {max_retries} attempts. Please check your connection.",
                    )
                return (
                    False,
                    retry_attempts,
                    f"\n\n{fail_label} - maximum retries ({max_retries}) reached. Please try again later.",
                )

        if isinstance(exc, APIStatusError):
            error_body = getattr(exc, "body", None) or {}
            error_info = error_body.get("error", {}) if isinstance(error_body, dict) else {}
            is_overloaded = error_info.get("type") == "overloaded_error"

            if is_overloaded and retry_attempts < max_retries:
                retry_attempts += 1
                await request_ctx.emit_event(
                    {
                        "type": "status",
                        "data": {
                            "description": f"⏳ API overloaded (streaming), retrying... ({retry_attempts}/{max_retries})",
                            "done": False,
                        },
                    }
                )
                return (True, retry_attempts, "")

            await self.handle_errors(exc, request_ctx.event_emitter)
            if is_overloaded:
                return (
                    False,
                    retry_attempts,
                    f"\n\n🔧 API overloaded (streaming) - maximum retries ({max_retries}) reached. Please try again later.",
                )
            return (
                False,
                retry_attempts,
                f"\n\nError: Anthropic API error. Reason: {getattr(exc, 'message', str(exc))}",
            )

        await self.handle_errors(exc, request_ctx.event_emitter)
        return (
            False,
            retry_attempts,
            f"\n\nError: {type(exc).__name__} occurred. Reason: {exc}",
        )

    async def _apply_sdk_stop_reason_fallback(
        self,
        *,
        sdk_final_message: Any,
        conversation_ended: bool,
        has_pending_tool_calls: bool,
        tool_calls: list[dict[str, Any]],
        tool_loop_iteration: int,
        payload_for_stream: dict[str, Any],
        stream_event_counts: dict[str, int],
        request_ctx: PipeRequestContext,
    ) -> tuple[bool, bool, list[dict[str, Any]]]:
        """Apply fallback stop-reason logic when message_delta was missing."""

        if not sdk_final_message or conversation_ended or has_pending_tool_calls:
            return conversation_ended, has_pending_tool_calls, tool_calls

        sdk_stop = getattr(sdk_final_message, "stop_reason", None)
        sdk_content = getattr(sdk_final_message, "content", [])

        if sdk_stop:
            logger.info(f"📍 Fallback stop_reason from SDK message: {sdk_stop}")
            if sdk_stop == "end_turn":
                conversation_ended = True
            elif sdk_stop == "tool_use":
                has_pending_tool_calls = True
                if not tool_calls:
                    for block in sdk_content:
                        if getattr(block, "type", None) == "tool_use":
                            logger.warning(
                                f"📍 Rebuilding tool_call from SDK: {getattr(block, 'name', '?')}"
                            )
                            tool_calls.append(
                                {
                                    "type": "tool_result",
                                    "tool_use_id": getattr(block, "id", ""),
                                    "content": "Error: tool call was not processed during streaming",
                                    "is_error": True,
                                }
                            )
            elif sdk_stop == "pause_turn":
                has_pending_tool_calls = True
                await request_ctx.emit_event(
                    {
                        "type": "status",
                        "data": {
                            "description": "⏳ Long-running turn paused, continuing...",
                            "done": False,
                        },
                    }
                )
            elif sdk_stop in (
                "max_tokens",
                "refusal",
                "stop_sequence",
                "model_context_window_exceeded",
            ):
                conversation_ended = True
                if sdk_stop == "max_tokens":
                    await request_ctx.emit_delta("\n\n⚠️ Maximum token limit reached.")
                elif sdk_stop == "model_context_window_exceeded":
                    await request_ctx.emit_delta("\n\n⚠️ Context window exceeded.")
        elif not sdk_content:
            logger.warning(
                f"⚠️ Empty API response (no stop_reason, no content). "
                f"Container: {payload_for_stream.get('container', 'NONE')}. "
                f"Events: {stream_event_counts}. Treating as end_turn."
            )
            conversation_ended = True
            if tool_loop_iteration > 1:
                await request_ctx.emit_delta(
                    "\n\n⚠️ Code execution continuation returned empty response. "
                    "The container may have timed out."
                )
        else:
            # stop_reason is None but content exists (e.g. thinking + server_tool blocks
            # without any text). This typically happens when the API is overloaded and
            # returns a truncated stream after 200 OK. Anthropic warns:
            # "When receiving a streaming response via SSE, it's possible that an error
            # can occur after returning a 200 response."
            # We leave conversation_ended=False here so the main loop's safety-break
            # section can detect this and trigger an auto-retry.
            block_types = [getattr(b, "type", "?") for b in sdk_content]
            has_text = any(
                getattr(b, "type", None) == "text"
                and len(getattr(b, "text", "") or "") > 0
                for b in sdk_content
            )
            logger.warning(
                f"⚠️ Truncated stream: no stop_reason but content present. "
                f"Blocks: {block_types}. has_text={has_text}. "
                f"Container: {payload_for_stream.get('container', 'NONE')}. "
                f"Events: {stream_event_counts}."
            )
            # Don't set conversation_ended — let the safety-break handle retry logic

        return conversation_ended, has_pending_tool_calls, tool_calls

    # =========================================================================
    # MAIN ENTRY POINT
    # =========================================================================

    async def pipe(
        self,
        body: dict[str, Any],
        __user__: Dict[str, Any],
        __event_emitter__: Callable[[Dict[str, Any]], Awaitable[None]],
        __metadata__: dict[str, Any] = {},
        __tools__: Optional[Dict[str, Dict[str, Any]]] = None,
        __files__: Optional[Dict[str, Any]] = None,
        __task__: Optional[dict[str, Any]] = None,
        __task_body__: Optional[dict[str, Any]] = None,
        __request__: Optional[Any] = None,
    ):
        """
        OpenWebUI Claude streaming pipe with integrated streaming logic.
        """
        # =========================================================================
        # PHASE 1: RESPONSE ACCUMULATION STATE
        # =========================================================================
        request_ctx = PipeRequestContext(pipe=self, event_emitter=__event_emitter__)
        final_message = request_ctx.final_message
        emit_event_local = request_ctx.emit_event
        emit_message_delta = request_ctx.emit_delta
        emit_message_replace = request_ctx.emit_replace
        update_content_block = request_ctx.update_content_block
        final_text = request_ctx.text


        try:
            # =========================================================================
            # PHASE 2: VALIDATION & SETUP
            # =========================================================================

            # Debug: Log all Valves and UserValves settings
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Valves: {self.valves.model_dump()}")
                user_valves = __user__.get("valves")
                if user_valves and hasattr(user_valves, "model_dump"):
                    logger.debug(f"UserValves: {user_valves.model_dump()}")
                elif user_valves:
                    logger.debug(f"UserValves: {user_valves}")

            # Get API key - check UserValves first, then fall back to admin valve
            user_valves = __user__.get("valves")
            user_api_key = getattr(user_valves, "ANTHROPIC_API_KEY", "") if user_valves else ""
            api_key = user_api_key.strip() if user_api_key and user_api_key.strip() else self.valves.ANTHROPIC_API_KEY
            if not api_key or api_key == "Your API Key Here":
                error_msg = "Error: No API key configured. Set it in admin Valves or your personal UserValves."
                logger.error(f"{error_msg}")
                await emit_event_local(
                    {
                        "type": "status",
                        "data": {
                            "description": "No API Key Set!",
                            "done": False,
                        },
                    }
                )
                return error_msg

            # STEP 1: Detect if task model (generate title, tags, follow-ups etc.), handle it separately
            if __task__:
                return await self._run_task_model_request(body)

            # STEP 2: Await tools if needed
            if inspect.isawaitable(__tools__):
                __tools__ = await __tools__

            # STEP 2.5: Get builtin tools from OpenWebUI (for tools from body.tools)
            builtin_tools = {}
            if BUILTIN_TOOLS_AVAILABLE and __request__:
                try:
                    # Determine if memory feature is enabled
                    memory_enabled = (
                        __user__.get("settings", {}).get("ui", {}).get("memory", False)
                        if __user__
                        else False
                    )
                    # Resolve skill IDs for view_skill builtin tool
                    skill_ids = []
                    try:
                        openwebui_model_id = __metadata__.get("model_id") or body.get("model", "")
                        if openwebui_model_id and MODELS_AVAILABLE:
                            owui_model = Models.get_model_by_id(openwebui_model_id)
                            if owui_model:
                                # ModelModel has .meta (ModelMeta pydantic model), not .info
                                meta = owui_model.meta
                                if meta:
                                    meta_dict = meta.model_dump() if hasattr(meta, "model_dump") else (meta if isinstance(meta, dict) else {})
                                    model_skill_ids = set(meta_dict.get("skillIds", []))
                                else:
                                    model_skill_ids = set()
                                logger.debug(f"Model {openwebui_model_id} skill IDs: {model_skill_ids}")
                                if model_skill_ids:
                                    from open_webui.models.skills import Skills as SkillsModel

                                    user_id = __user__.get("id", "") if __user__ else ""
                                    accessible = {
                                        s.id
                                        for s in SkillsModel.get_skills_by_user_id(user_id, "read")
                                    }
                                    logger.debug(f"Accessible skills for user: {accessible}")
                                    skill_ids = [
                                        sid
                                        for sid in model_skill_ids
                                        if sid in accessible
                                        and (s := SkillsModel.get_skill_by_id(sid))
                                        and s.is_active
                                    ]
                                    logger.debug(f"Resolved skill_ids: {skill_ids}")
                    except Exception as e:
                        logger.debug(f"Could not resolve skill IDs: {e}")

                    builtin_tools = get_builtin_tools(
                        __request__,
                        {
                            "__user__": __user__,
                            "__event_emitter__": __event_emitter__,
                            "__chat_id__": (
                                __metadata__.get("chat_id") if __metadata__ else None
                            ),
                            "__message_id__": (
                                __metadata__.get("message_id") if __metadata__ else None
                            ),
                            "__skill_ids__": skill_ids,
                        },
                        features={"memory": memory_enabled},
                        model={},
                    )
                    logger.debug(
                        f"Loaded {len(builtin_tools)} builtin tools: {list(builtin_tools.keys())}"
                    )
                except Exception as e:
                    logger.warning(f"Could not load builtin tools: {e}")
                    builtin_tools = {}

            # Merge external tools from metadata (Open Terminal, external tool servers)
            # These have callables for execution but are not in __tools__ or builtin_tools
            metadata_tools = __metadata__.get("tools", {}) if __metadata__ else {}
            if metadata_tools:
                for t_name, t_data in metadata_tools.items():
                    if t_name not in builtin_tools and (not __tools__ or t_name not in __tools__):
                        if isinstance(t_data, dict) and t_data.get("callable"):
                            builtin_tools[t_name] = t_data
                if builtin_tools:
                    logger.debug(
                        f"After metadata merge, builtin_tools: {list(builtin_tools.keys())}"
                    )

            # STEP 3: Auto-enable native function calling if tools are present
            # This prevents OpenWebUI's function_calling task system from being triggered
            if __tools__ and MODELS_AVAILABLE:
                try:
                    # Get the OpenWebUI model ID from metadata
                    openwebui_model_id = (
                        __metadata__.get("model_id") if __metadata__ else None
                    )
                    if not openwebui_model_id and body and "model" in body:
                        openwebui_model_id = body["model"]

                    if openwebui_model_id:
                        model = Models.get_model_by_id(openwebui_model_id)
                        if model:
                            params = dict(model.params or {})
                            if params.get("function_calling") != "native":
                                logger.debug(
                                    f"Auto-enabling native function calling for model: {openwebui_model_id}"
                                )

                                # Notify user
                                await emit_event_local(
                                    {
                                        "type": "notification",
                                        "data": {
                                            "type": "info",
                                            "content": f"Enabling native function calling for model: {openwebui_model_id}. Please re-run your query.",
                                        },
                                    }
                                )

                                params["function_calling"] = "native"
                                form_data = model.model_dump()
                                form_data["params"] = params
                                Models.update_model_by_id(
                                    openwebui_model_id, ModelForm(**form_data)
                                )
                except Exception as e:
                    logger.warning(
                        f"Could not auto-enable native function calling: {e}"
                    )

            payload, headers, new_marker_metadata, api_tool_names = await self._create_payload(
                body, __metadata__, __user__, __tools__, __event_emitter__, __files__
            )

            if __user__["valves"].DEBUG_MODE:
                await self._emit_debug_citation(request_ctx, "Payload", payload)
                await self._emit_debug_citation(request_ctx, "Headers", headers)
                await self._emit_debug_citation(request_ctx, "body", body)
                await self._emit_debug_citation(request_ctx, "__metadata__", __metadata__ or {})
                await self._emit_debug_citation(request_ctx, "__user__", __user__)
                await self._emit_debug_citation(request_ctx, "__files__", __files__ or [])
                await self._emit_debug_citation(request_ctx, "__tools__", __tools__ or {})
                await self._emit_debug_citation(request_ctx, "new_marker_metadata", new_marker_metadata or {})
                await self._emit_debug_citation(request_ctx, "Valves: ", self.valves.__dict__)
                await self._emit_debug_citation(request_ctx, "UserValves: ", __user__["valves"].__dict__)
                if __task__:
                    await self._emit_debug_citation(request_ctx, "__task__", __task_body__)

            # =========================================================================
            # PHASE 3: STREAMING STATE INITIALIZATION
            # =========================================================================
            api_key = headers.get("x-api-key", self.valves.ANTHROPIC_API_KEY)
            # Use UserValves API key if available (override header-level key too)
            if user_api_key and user_api_key.strip():
                api_key = user_api_key.strip()
                logger.debug("Using user-provided API key from UserValves")
            request_timeout = self.valves.REQUEST_TIMEOUT
            base_url = self.valves.ANTHROPIC_BASE_URL.strip() or None
            client = AsyncAnthropic(api_key=api_key, default_headers=headers, timeout=request_timeout, **({"base_url": base_url} if base_url else {}))
            payload_for_stream = {k: v for k, v in payload.items() if k != "stream"}
            include_usage = (
                __user__["valves"].SHOW_TOKEN_COUNT or
                __user__["valves"].SHOW_CACHE_INFO or
                body.get("stream_options", {}).get("include_usage", False)
            )
            if include_usage:
                total_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
                if self.valves.CACHE_CONTROL != "cache disabled":
                    total_usage["cache_creation_input_tokens"] = 0
                    total_usage["cache_read_input_tokens"] = 0

            # Stream configuration from valves
            token_buffer_size = getattr(self.valves, "TOKEN_BUFFER_SIZE", 1)
            max_function_calls = self.valves.MAX_TOOL_CALLS

            # Thinking mode state
            is_model_thinking = False
            thinking_message = ""
            thinking_start_time = None  # Track when thinking started for duration calc
            thinking_stream_start_idx = -1  # Position in final_message where thinking content starts
            thinking_last_block = ""  # Tracks current formatted thinking block for content-preserving replace

            # SDK-accumulated message: captured after each stream completes
            # Replaces manual api_assistant_blocks/thinking_blocks accumulation
            sdk_final_message = None

            # Tool execution state
            current_block_type = None  # Track current block type for stop events
            has_pending_tool_calls = False
            tools_buffer = ""
            tool_input_buffer = ""  # Accumulate just the input JSON for live streaming
            tool_calls = []
            running_tool_tasks = []  # Async tasks for executing tools immediately
            tool_call_data_list = []  # Store tool metadata for result matching
            api_tool_passthrough = False  # Flag for API tool passthrough mode
            tool_progress_blocks = {}  # Map tool_id → current in-progress block text for replacement
            # Note: tool_use_blocks and current_tool_caller removed - SDK preserves these in accumulated message

            # Server tool state (web_search, code_execution)
            active_server_tool_name = None
            active_server_tool_id = None
            server_tool_input_buffer = ""  # Accumulate server tool input JSON
            text_editor_file_content = ""  # Accumulate file_text for text_editor
            text_editor_file_path = ""  # Track file path for text_editor
            text_editor_command = ""  # Track text_editor command (create/view/edit)
            bash_execution_command = ""  # Track bash command for code execution
            code_execution_code = ""  # Track code from programmatic code_execution
            in_code_execution = False  # Whether we're currently in a code_execution flow
            code_exec_is_web_filtering = False  # True when code_execution is just dynamic filtering for web tools
            code_exec_tool_calls_info = []  # Accumulate tool call info for unified display
            code_exec_stream_start_idx = -1  # Position in final_message where code exec content starts
            code_exec_last_block = ""  # Tracks current formatted code block for content-preserving replace
            code_exec_current_code = ""  # Accumulated code for live streaming inside details
            code_exec_current_lang = "python"  # Language for live streaming
            code_exec_start_time = 0.0  # time.time() when code execution started
            last_code_language = (
                "bash"  # Track language of last code block for output association
            )
            last_code_content = ""  # Buffer code content for combining with output

            # Dynamic filtering detection:
            # If code_execution was NOT explicitly added to tools (no code_execution_20250825 or
            # code_execution_20260120 in payload), then any code_execution in the stream is from
            # dynamic filtering auto-injection → suppress UI.
            # If code_execution WAS explicitly added, code_exec blocks could be real code → show UI.
            payload_tools = payload.get("tools", [])
            has_explicit_code_execution = any(
                t.get("name") == "code_execution" for t in payload_tools
            )
            code_exec_has_user_tools = False  # Tracks if user tools were called in current code_exec
            code_exec_had_web_tools = False  # Tracks if web_search/web_fetch happened inside code_exec

            # Web search citation state
            current_search_query = ""  # Track the current web search query
            citation_counter = 0  # Track citation numbers for inline citations
            pending_citation_markers = []  # Deferred markers (web_search citations arrive before text)
            citations_list = []  # Store citations for reference list

            # Loop control state
            conversation_ended = False
            retry_attempts = 0
            current_function_calls = 0
            first_text_emitted = False  # Track if we've emitted "Responding..." status

            # Response chunk state
            chunk = ""
            chunk_count = 0

            await emit_event_local(
                {
                    "type": "status",
                    "data": {
                        "description": "Waiting for response...",
                        "done": False,
                        "hidden": False,
                    },
                }
            )

            # =========================================================================
            # PHASE 4: MAIN STREAMING LOOP
            # Continues until conversation ends or max tool calls reached
            # =========================================================================
            tool_loop_iteration = 0
            while (
                current_function_calls < max_function_calls
                and not conversation_ended
                and retry_attempts <= self.valves.MAX_RETRIES
            ):
                tool_loop_iteration += 1
                # Reset per-iteration state
                stream_output_tokens = 0

                try:
                    # Verbose tool loop logging
                    msg_summary = []
                    for msg in payload_for_stream.get("messages", []):
                        role = msg.get("role", "?")
                        content = msg.get("content", "")
                        if isinstance(content, list):
                            block_types = [b.get("type", "?") for b in content]
                            # Count thinking blocks and check signatures
                            thinking_info = []
                            for b in content:
                                if b.get("type") == "thinking":
                                    sig_len = len(b.get("signature", ""))
                                    think_len = len(b.get("thinking", ""))
                                    thinking_info.append(f"thinking({think_len}c,sig={sig_len}c)")
                                elif b.get("type") == "redacted_thinking":
                                    thinking_info.append(f"redacted(data={len(b.get('data', ''))}c)")
                            detail = ",".join(block_types)
                            if thinking_info:
                                detail += f" [{'; '.join(thinking_info)}]"
                            msg_summary.append(f"{role}:[{detail}]")
                        elif isinstance(content, str):
                            msg_summary.append(f"{role}:text({len(content)}c)")
                        else:
                            msg_summary.append(f"{role}:?")
                    logger.info(
                        f"🔄 Tool loop iter {tool_loop_iteration} | "
                        f"calls so far: {current_function_calls}/{max_function_calls} | "
                        f"messages: {len(payload_for_stream.get('messages', []))} | "
                        f"container: {payload_for_stream.get('container', 'NONE')} | "
                        f"msg_flow: {' → '.join(msg_summary[-6:]) if msg_summary else 'empty'}"
                    )

                    # Debug: comprehensive payload summary before API call
                    if logger.isEnabledFor(logging.DEBUG):
                        _tools = payload_for_stream.get("tools", [])
                        _tool_names = [t.get("name", t.get("type", "?")) for t in _tools]
                        _sys = payload_for_stream.get("system", [])
                        _sys_len = sum(len(s.get("text", "")) for s in _sys) if isinstance(_sys, list) else len(str(_sys))
                        _msgs = payload_for_stream.get("messages", [])
                        # Last message detail
                        _last_msg = _msgs[-1] if _msgs else {}
                        _last_role = _last_msg.get("role", "?")
                        _last_content = _last_msg.get("content", "")
                        if isinstance(_last_content, list):
                            _last_detail = [b.get("type", "?") for b in _last_content]
                            # For tool_result, show tool IDs
                            _tool_result_ids = [b.get("tool_use_id", "")[:20] for b in _last_content if b.get("type") == "tool_result"]
                            _last_summary = f"[{','.join(_last_detail)}]" + (f" tool_ids={_tool_result_ids}" if _tool_result_ids else "")
                        elif isinstance(_last_content, str):
                            _last_summary = f"text({len(_last_content)}c)"
                        else:
                            _last_summary = "?"
                        logger.debug(
                            f"📤 API payload: model={payload_for_stream.get('model', '?')} | "
                            f"tools={len(_tools)}:{_tool_names} | "
                            f"system={_sys_len}c | "
                            f"container={payload_for_stream.get('container', 'NONE')} | "
                            f"max_tokens={payload_for_stream.get('max_tokens', '?')} | "
                            f"thinking={payload_for_stream.get('thinking', {}).get('type', 'off')} | "
                            f"last_msg={_last_role}:{_last_summary}"
                        )

                    stream_event_counts = {}  # Track event types for diagnostics
                    # Apply cache breakpoints right before sending to API
                    self._apply_cache_control(payload_for_stream, is_tool_loop=(tool_loop_iteration > 1))
                    async with client.beta.messages.stream(
                        **payload_for_stream
                    ) as stream:
                        async for event in stream:
                            event_type = getattr(event, "type", None)
                            stream_event_counts[event_type] = stream_event_counts.get(event_type, 0) + 1
                            logger.debug(f"Received stream event: {event_type} | counts: {stream_event_counts} | payload: {event}")
                            if event_type == "message_start":
                                # Note: Container ID is not in message_start for streaming;
                                # it arrives in message_delta.
                                stream_output_tokens = self._handle_message_start_usage(
                                    event,
                                    include_usage=include_usage,
                                    total_usage=total_usage if include_usage else None,
                                    stream_output_tokens=stream_output_tokens,
                                )

                            # ---------------------------------------------------------
                            # EVENT: content_block_start
                            # Handles start of: text, thinking, tool_use, server_tool_use,
                            # bash_code_execution_tool_result, text_editor_code_execution_tool_result,
                            # web_search_tool_result, tool_search_tool_result, context_cleared
                            # ---------------------------------------------------------
                            elif event_type == "content_block_start":
                                content_block = getattr(event, "content_block", None)
                                content_type = getattr(content_block, "type", None)
                                current_block_type = content_type
                                if not content_block:
                                    continue
                                if content_type == "text":
                                    chunk += content_block.text or ""
                                elif content_type == "thinking":
                                    is_model_thinking = True
                                    thinking_start_time = time.time()
                                    thinking_message = ""
                                    thinking_stream_start_idx = len(final_message)
                                elif content_type == "redacted_thinking":
                                    is_model_thinking = True
                                elif content_type == "tool_use":
                                    tool_name = getattr(
                                        content_block, "name", "unknown"
                                    )
                                    logger.debug(
                                        f"🔧 Tool use block started: {tool_name}"
                                    )
                                    if in_code_execution and code_exec_is_web_filtering:
                                        # User tool called from within code_execution → NOT just web filtering
                                        code_exec_is_web_filtering = False
                                        code_exec_has_user_tools = True

                                    # For programmatic tool calls, the API may provide
                                    # the full input at content_block_start (no input_json_delta events)
                                    initial_input = getattr(content_block, "input", None) or {}
                                    tool_id_at_start = getattr(content_block, "id", "")
                                    if initial_input:
                                        # Input is pre-populated (programmatic call) - include it directly
                                        logger.debug(f"🔧 Tool input pre-populated at start: {json.dumps(initial_input)[:200]}")
                                        tools_buffer = json.dumps({
                                            "type": content_block.type,
                                            "id": content_block.id,
                                            "name": content_block.name,
                                            "input": initial_input,
                                        })
                                    else:
                                        # Standard streaming: input arrives via input_json_delta
                                        tools_buffer = (
                                            "{"
                                            f'"type": "{content_block.type}", '
                                            f'"id": "{content_block.id}", '
                                            f'"name": "{content_block.name}", '
                                            f'"input": '
                                        )

                                    # Show in-progress tool call block immediately (not for programmatic/code_exec)
                                    if not in_code_execution:
                                        in_progress_block = self._format_tool_result_block(
                                            tool_id_at_start, tool_name, initial_input or {}, "", done=False
                                        )
                                        tool_progress_blocks[tool_id_at_start] = in_progress_block
                                        final_message.append(in_progress_block)
                                        await emit_message_replace(final_text())
                                    # Reset input buffer for streaming tool arguments
                                    tool_input_buffer = ""
                                elif content_type == "server_tool_use":
                                    # Track active server tool (web_search, code_execution)
                                    # No need for tools_buffer - server handles execution
                                    active_server_tool_name = getattr(
                                        content_block, "name", ""
                                    )
                                    active_server_tool_id = getattr(
                                        content_block, "id", ""
                                    )
                                    server_tool_input_buffer = (
                                        ""  # Reset buffer for new tool
                                    )
                                    # Note: server_tool_use blocks are preserved by SDK accumulated message

                                    logger.debug(
                                        f"Server tool started: {active_server_tool_name} (ID: {active_server_tool_id})"
                                    )

                                    if active_server_tool_name in ("web_search", "web_fetch"):
                                        # Track that web tools were used inside code_execution
                                        # (confirms it's dynamic filtering, not programmatic code)
                                        if in_code_execution:
                                            code_exec_had_web_tools = True

                                    if active_server_tool_name == "web_search":
                                        await emit_event_local(
                                            {
                                                "type": "status",
                                                "data": {
                                                    "description": "🔍 Searching the web...",
                                                    "done": False,
                                                },
                                            }
                                        )
                                    elif active_server_tool_name == "web_fetch":
                                        await emit_event_local(
                                            {
                                                "type": "status",
                                                "data": {
                                                    "description": "🌐 Fetching URL...",
                                                    "done": False,
                                                },
                                            }
                                        )
                                    elif active_server_tool_name == "code_execution":
                                        # Finalize any previous live-streamed block before starting new one
                                        if code_exec_current_code:
                                            duration = time.time() - code_exec_start_time if code_exec_start_time else None
                                            block = self._format_code_execution_block(
                                                code_exec_current_code, code_exec_current_lang,
                                                done=True, duration=duration,
                                            )
                                            await update_content_block(code_exec_last_block, block)
                                            code_exec_last_block = ""

                                        in_code_execution = True
                                        # Start assuming web filtering — confirmed when web_search/web_fetch appears inside
                                        # Flipped to False if user tools (tool_use blocks) are called
                                        code_exec_is_web_filtering = True
                                        code_exec_has_user_tools = False
                                        code_exec_had_web_tools = False
                                        code_exec_tool_calls_info = []
                                        code_exec_stream_start_idx = len(final_message)
                                        code_exec_current_code = ""
                                        code_exec_current_lang = "python"
                                        code_exec_start_time = time.time()
                                        # Don't emit "Running code" yet — defer until we know it's not web filtering
                                    elif active_server_tool_name in ("bash_code_execution", "text_editor_code_execution"):
                                        # Finalize any previous live-streamed block before starting new one
                                        if code_exec_current_code:
                                            duration = time.time() - code_exec_start_time if code_exec_start_time else None
                                            block = self._format_code_execution_block(
                                                code_exec_current_code, code_exec_current_lang,
                                                done=True, duration=duration,
                                            )
                                            await update_content_block(code_exec_last_block, block)
                                            code_exec_last_block = ""

                                        code_exec_current_code = ""
                                        code_exec_current_lang = "bash" if active_server_tool_name == "bash_code_execution" else "python"
                                        code_exec_start_time = time.time()
                                elif content_type == "bash_code_execution_tool_result":
                                    logger.debug(
                                        f"Processing bash_code_execution_tool_result: {content_block}"
                                    )
                                    result_block = getattr(
                                        content_block, "content", None
                                    )
                                    if result_block:
                                        # Check for server tool errors (unavailable, execution_time_exceeded, container_expired, etc.)
                                        result_block_type = getattr(result_block, "type", "")
                                        if result_block_type == "bash_code_execution_tool_result_error":
                                            error_code = getattr(result_block, "error_code", "unknown")
                                            error_msg = f"⚠️ Code execution error: {error_code}"
                                            logger.warning(f"bash_code_execution error: {error_code}")
                                            await emit_message_delta(f"{error_msg}")
                                            last_code_content = ""
                                            continue

                                        stdout = getattr(result_block, "stdout", "")
                                        stderr = getattr(result_block, "stderr", "")
                                        return_code = getattr(
                                            result_block, "return_code", None
                                        )

                                        # Handle file outputs from code execution
                                        download_links = []
                                        files_output = getattr(
                                            result_block, "content", []
                                        )
                                        if files_output:
                                            logger.debug(
                                                f"Found {len(files_output)} file outputs"
                                            )
                                            for file_obj in files_output:
                                                logger.debug(
                                                    f" Processing file object: {file_obj}"
                                                )
                                                # Files in bash results have file_id
                                                file_id = getattr(
                                                    file_obj, "file_id", None
                                                )
                                                if file_id:
                                                    # Generate download link - download file and save to Open-WebUI
                                                    download_link = await self._generate_file_download_link(
                                                        file_id=file_id,
                                                        api_key=api_key,
                                                        user_id=__user__.get(
                                                            "id", "unknown"
                                                        ),
                                                    )
                                                    download_links.append(download_link)

                                        if (
                                            stdout
                                            or stderr
                                            or return_code is not None
                                            or download_links
                                        ):
                                            if code_exec_is_web_filtering and code_exec_had_web_tools:
                                                logger.debug("Suppressed bash code execution block (web filtering)")
                                            else:
                                                # Update the live-streamed code block with final result
                                                duration = time.time() - code_exec_start_time if code_exec_start_time else None
                                                block = self._format_code_execution_block(
                                                    last_code_content, "bash",
                                                    done=True, duration=duration,
                                                    stdout=stdout, stderr=stderr,
                                                    return_code=return_code,
                                                    download_links=download_links,
                                                )
                                                await update_content_block(code_exec_last_block, block)
                                                code_exec_last_block = ""
                                            last_code_content = ""
                                elif content_type == "text_editor_code_execution_tool_result":
                                    logger.debug(
                                        f"Processing text_editor_code_execution_tool_result: {content_block}"
                                    )
                                    result_block = getattr(
                                        content_block, "content", None
                                    )
                                    if result_block:
                                        result_type = getattr(result_block, "type", "")
                                        logger.debug(
                                            f"Text editor result type: {result_type}"
                                        )

                                        # Check for server tool errors (file_not_found, string_not_found, unavailable, etc.)
                                        if result_type == "text_editor_code_execution_tool_result_error":
                                            error_code = getattr(result_block, "error_code", "unknown")
                                            error_msg = f"⚠️ Text editor error: {error_code}"
                                            logger.warning(f"text_editor_code_execution error: {error_code}")
                                            await emit_message_delta(f"{error_msg}")
                                            last_code_content = ""
                                            continue

                                        if code_exec_is_web_filtering and code_exec_had_web_tools:
                                            # Dynamic filtering confirmed: suppress text editor UI
                                            logger.debug("Suppressed text editor block (web filtering)")
                                            last_code_content = ""
                                        else:
                                            # Handle create/update results
                                            if (
                                                result_type
                                                == "text_editor_code_execution_create_result"
                                            ):
                                                if last_code_content and last_code_language == "__inline_text__":
                                                    msg = f"\n\n{last_code_content}\n\n"
                                                    await emit_message_delta(msg)
                                                    last_code_content = ""
                                                    last_code_language = ""
                                                elif last_code_content:
                                                    # Replace live-streamed block with final version
                                                    duration = time.time() - code_exec_start_time if code_exec_start_time else None
                                                    block = self._format_code_execution_block(
                                                        last_code_content, last_code_language or "python",
                                                        done=True, duration=duration,
                                                    )
                                                    await update_content_block(code_exec_last_block, block)
                                                    code_exec_last_block = ""
                                                    last_code_content = ""

                                            elif (
                                                result_type
                                                == "text_editor_code_execution_view_result"
                                            ):
                                                content = getattr(
                                                    result_block, "content", ""
                                                )
                                                if content:
                                                    msg = f"\n<details>\n<summary>📄 File Content</summary>\n\n```\n{content}\n```\n</details>\n"
                                                    await emit_message_delta(msg)
                                elif content_type == "code_execution_tool_result":
                                    logger.debug(
                                        "Processing code_execution_tool_result"
                                    )
                                    result_block = getattr(content_block, "content", None)
                                    stdout = ""
                                    stderr = ""
                                    return_code = None
                                    if result_block:
                                        # Check for server tool errors
                                        result_block_type = (
                                            result_block.get("type", "") if isinstance(result_block, dict)
                                            else getattr(result_block, "type", "")
                                        )
                                        if result_block_type == "code_execution_tool_result_error":
                                            error_code = (
                                                result_block.get("error_code", "unknown") if isinstance(result_block, dict)
                                                else getattr(result_block, "error_code", "unknown")
                                            )
                                            error_msg = f"⚠️ Code execution error: {error_code}"
                                            logger.warning(f"code_execution error: {error_code}")
                                            await emit_message_delta(f"{error_msg}")
                                            last_code_content = ""
                                            in_code_execution = False
                                            code_exec_is_web_filtering = False
                                            continue

                                        # Handle both dict (legacy) and object (new API) formats
                                        if isinstance(result_block, dict):
                                            stdout = result_block.get("stdout", "")
                                            stderr = result_block.get("stderr", "")
                                            return_code = result_block.get("return_code", None)
                                        else:
                                            stdout = getattr(result_block, "stdout", "")
                                            stderr = getattr(result_block, "stderr", "")
                                            return_code = getattr(result_block, "return_code", None)

                                    if code_exec_is_web_filtering and code_exec_had_web_tools:
                                        logger.debug("Suppressed code_execution_tool_result (web filtering)")
                                        last_code_content = ""
                                    elif stdout or stderr or return_code is not None or code_exec_tool_calls_info:
                                        # Update the live-streamed code block with final result
                                        duration = time.time() - code_exec_start_time if code_exec_start_time else None
                                        code_to_show = last_code_content or code_exec_current_code
                                        block = self._format_code_execution_block(
                                            code_to_show, "python",
                                            done=True, duration=duration,
                                            stdout=stdout, stderr=stderr,
                                            return_code=return_code,
                                            tool_calls_info=code_exec_tool_calls_info,
                                        )
                                        await update_content_block(code_exec_last_block, block)
                                        code_exec_last_block = ""
                                        last_code_content = ""

                                    # Emit "complete" status only if it wasn't web filtering
                                    was_web_filtering = code_exec_is_web_filtering and code_exec_had_web_tools

                                    # Reset code execution state
                                    in_code_execution = False
                                    code_exec_is_web_filtering = False
                                    code_exec_has_user_tools = False
                                    code_exec_had_web_tools = False
                                    code_exec_tool_calls_info = []
                                    code_exec_stream_start_idx = -1

                                    if not was_web_filtering:
                                        pass  # code_interpreter block handles "Analyzed" state visually
                                elif content_type == "web_search_tool_result":
                                    logger.debug(
                                        f" Processing web search result event: {event}"
                                    )
                                    content_items = getattr(
                                        content_block, "content", None
                                    )
                                    # Check for error: content is a single error object (not a list)
                                    error_code = None
                                    if content_items and not isinstance(content_items, list):
                                        content_inner_type = getattr(content_items, "type", "")
                                        if content_inner_type == "web_search_tool_result_error":
                                            error_code = getattr(content_items, "error_code", "unknown")
                                    if error_code:
                                        error_msg = f"⚠️ Web search error: {error_code}"
                                        logger.warning(f"web_search error: {error_code}")
                                        await emit_event_local(
                                            {
                                                "type": "status",
                                                "data": {
                                                    "description": error_msg,
                                                    "done": True,
                                                },
                                            }
                                        )
                                    elif content_items and isinstance(content_items, list) and len(content_items) > 0:
                                            # Extract first result title for status
                                            first_result = (
                                                content_items[0]
                                                if content_items
                                                else None
                                            )
                                            result_title = (
                                                getattr(first_result, "title", "")
                                                if first_result
                                                else ""
                                            )
                                            result_count = len(content_items)

                                            if result_title and result_count > 0:
                                                status_desc = f"Found {result_count} results - {result_title}"
                                                if result_count > 1:
                                                    status_desc += (
                                                        f" +{result_count-1} more"
                                                    )
                                            else:
                                                status_desc = "Web Search Complete"

                                            await emit_event_local(
                                                {
                                                    "type": "status",
                                                    "data": {
                                                        "description": status_desc,
                                                        "done": True,
                                                    },
                                                }
                                            )
                                elif content_type == "web_fetch_tool_result":
                                    logger.debug("Processing web_fetch_tool_result")
                                    # Check for errors (error is in content.error_code)
                                    result_content = getattr(content_block, "content", None)
                                    error_code = None
                                    if result_content:
                                        content_type_inner = getattr(result_content, "type", "")
                                        if content_type_inner == "web_fetch_tool_error":
                                            error_code = getattr(result_content, "error_code", "unknown")
                                    if error_code:
                                        await emit_event_local(
                                            {
                                                "type": "status",
                                                "data": {
                                                    "description": f"🌐 Fetch failed: {error_code}",
                                                    "done": True,
                                                },
                                            }
                                        )
                                    else:
                                        await emit_event_local(
                                            {
                                                "type": "status",
                                                "data": {
                                                    "description": "🌐 URL fetched",
                                                    "done": True,
                                                },
                                            }
                                        )
                                elif content_type == "tool_search_tool_result":
                                    logger.debug(
                                        f" Processing tool search result event: {event}"
                                    )
                                    # Serialize the tool_search_tool_result block for API preservation
                                    # These blocks MUST be included verbatim in the next API call
                                    tool_use_id = getattr(content_block, "tool_use_id", "")
                                    content_obj = getattr(content_block, "content", None)
                                    tool_refs = []
                                    if content_obj:
                                        refs = (
                                            getattr(content_obj, "tool_references", [])
                                            if hasattr(content_obj, "tool_references")
                                            else content_obj.get("tool_references", [])
                                        )
                                        for ref in refs:
                                            tool_refs.append({
                                                "type": "tool_reference",
                                                "tool_name": (
                                                    getattr(ref, "tool_name", "")
                                                    if hasattr(ref, "tool_name")
                                                    else ref.get("tool_name", "")
                                                ),
                                            })
                                    # SDK preserves tool_search_tool_result blocks in accumulated message
                                    # They get stripped as response-only types in _convert_sdk_message_to_api_blocks
                                    # Only show status events if tool search valve is enabled
                                    if self.valves.ENABLE_TOOL_SEARCH:
                                        # Access nested structure: content_block.content.tool_references
                                        content = getattr(
                                            content_block, "content", None
                                        )
                                        tool_references = []
                                        if content:
                                            # Handle both dict and object access patterns
                                            if hasattr(content, "tool_references"):
                                                tool_references = getattr(
                                                    content, "tool_references", []
                                                )
                                            elif isinstance(content, dict):
                                                tool_references = content.get(
                                                    "tool_references", []
                                                )

                                        logger.debug(
                                            f"Tool search result - found {len(tool_references)} tool references"
                                        )

                                        if tool_references and len(tool_references) > 0:
                                            # Extract tool names from references
                                            tool_names = []
                                            for ref in tool_references[:5]:
                                                if hasattr(ref, "tool_name"):
                                                    tool_names.append(
                                                        getattr(
                                                            ref, "tool_name", "unknown"
                                                        )
                                                    )
                                                elif isinstance(ref, dict):
                                                    tool_names.append(
                                                        ref.get("tool_name", "unknown")
                                                    )

                                            status_desc = f"🔍 Found {len(tool_references)} tool(s): {', '.join(tool_names)}"
                                            logger.debug(
                                                f"Tool search success: {status_desc}"
                                            )
                                            await emit_event_local(
                                                {
                                                    "type": "status",
                                                    "data": {
                                                        "description": status_desc,
                                                        "done": True,
                                                    },
                                                }
                                            )
                                        else:
                                            logger.debug(
                                                "Tool search returned no results"
                                            )
                                            await emit_event_local(
                                                {
                                                    "type": "status",
                                                    "data": {
                                                        "description": "🔍 Tool search: No matching tools found",
                                                        "done": True,
                                                    },
                                                }
                                            )
                                elif content_type == "context_cleared":
                                    cleared_info = getattr(content_block, "cleared", {})
                                    cleared_type = (
                                        getattr(cleared_info, "type", "unknown")
                                        if hasattr(cleared_info, "type")
                                        else cleared_info.get("type", "unknown")
                                    )
                                    cleared_tokens = (
                                        getattr(cleared_info, "tokens_cleared", 0)
                                        if hasattr(cleared_info, "tokens_cleared")
                                        else cleared_info.get("tokens_cleared", 0)
                                    )

                                    if cleared_type == "tool_uses":
                                        status_desc = f"🧹 Cleared tool results: ~{cleared_tokens:,} tokens removed"
                                    elif cleared_type == "thinking":
                                        status_desc = f"🧹 Cleared thinking blocks: ~{cleared_tokens:,} tokens removed"
                                    else:
                                        status_desc = f"🧹 Context cleared: ~{cleared_tokens:,} tokens removed"

                                    await emit_event_local(
                                        {
                                            "type": "status",
                                            "data": {
                                                "description": status_desc,
                                                "done": True,
                                            },
                                        }
                                    )
                                    logger.debug(
                                        f"Context cleared: type={cleared_type}, tokens={cleared_tokens}"
                                    )

                            # ---------------------------------------------------------
                            # EVENT: content_block_delta
                            # Handles streaming deltas for: thinking, text, tool_use input,
                            # server tool input, citations
                            # ---------------------------------------------------------
                            elif event_type == "content_block_delta":
                                delta = getattr(event, "delta", None)
                                delta_type = getattr(delta, "type", None)
                                if delta_type == "thinking_delta":
                                    thinking_text = getattr(delta, "thinking", "")
                                    thinking_message += thinking_text
                                    # Stream thinking inside <details> wrapper for native collapsible UI
                                    if thinking_text:
                                        formatted = self._format_thinking_block(thinking_message, duration=None)
                                        await update_content_block(thinking_last_block, formatted)
                                        thinking_last_block = formatted
                                elif delta_type == "signature_delta":
                                    # No manual tracking needed
                                    pass
                                elif delta_type == "text_delta":
                                    text_delta = getattr(delta, "text", "")

                                    # Emit "Responding..." status on first text delta (only once)
                                    if (
                                        not first_text_emitted
                                        and not is_model_thinking
                                        and not active_server_tool_name
                                    ):
                                        await emit_event_local(
                                            {
                                                "type": "status",
                                                "data": {
                                                    "description": "Responding...",
                                                    "done": False,
                                                },
                                            }
                                        )
                                        first_text_emitted = True

                                    chunk += text_delta
                                    chunk_count += 1
                                elif delta_type == "input_json_delta":
                                    partial = getattr(delta, "partial_json", "")

                                    # Handle server tool input separately from client tools
                                    if active_server_tool_name:
                                        # Server tool (web_search, code_execution) - accumulate and extract query
                                        server_tool_input_buffer += partial

                                        if active_server_tool_name == "web_search":
                                            try:
                                                # Try to parse the accumulated JSON to extract query
                                                parsed = json.loads(
                                                    server_tool_input_buffer
                                                )
                                                if "query" in parsed:
                                                    new_query = parsed["query"]
                                                    logger.debug(
                                                        f"Web search query complete: '{new_query}'"
                                                    )

                                                    # Emit status only once when we get the complete query
                                                    if (
                                                        new_query
                                                        and new_query
                                                        != current_search_query
                                                    ):
                                                        current_search_query = (
                                                            new_query
                                                        )
                                                        await emit_event_local(
                                                            {
                                                                "type": "status",
                                                                "data": {
                                                                    "description": f"🔍 Searching for: {current_search_query}",
                                                                    "done": False,
                                                                },
                                                            }
                                                        )
                                            except Exception as e:
                                                logger.debug(
                                                    f"Web search query extraction error: {e}"
                                                )
                                        elif active_server_tool_name == "web_fetch":
                                            try:
                                                parsed = json.loads(server_tool_input_buffer)
                                                if "url" in parsed:
                                                    fetch_url = parsed["url"]
                                                    # Show truncated URL in status
                                                    display_url = fetch_url[:60] + "..." if len(fetch_url) > 60 else fetch_url
                                                    await emit_event_local(
                                                        {
                                                            "type": "status",
                                                            "data": {
                                                                "description": f"🌐 Fetching: {display_url}",
                                                                "done": False,
                                                            },
                                                        }
                                                    )
                                            except Exception:
                                                pass
                                        elif (
                                            active_server_tool_name
                                            == "code_execution"
                                        ):
                                            # Code execution (programmatic tool calling) - extract code for display
                                            try:
                                                parsed = json.loads(server_tool_input_buffer)
                                                if "code" in parsed:
                                                    code_execution_code = parsed["code"]
                                                    # Live stream code inside details block
                                                    if not code_exec_is_web_filtering or not code_exec_had_web_tools:
                                                        code_exec_current_code = code_execution_code
                                                        code_exec_current_lang = parsed.get("language", "python")
                                                        block = self._format_code_execution_block(
                                                            code_exec_current_code, code_exec_current_lang,
                                                        )
                                                        await update_content_block(code_exec_last_block, block)
                                                        code_exec_last_block = block
                                            except (json.JSONDecodeError, KeyError):
                                                pass
                                        elif (
                                            active_server_tool_name
                                            == "bash_code_execution"
                                        ):
                                            # Bash code execution - extract command (the actual code)
                                            try:
                                                parsed = json.loads(
                                                    server_tool_input_buffer
                                                )
                                                if "command" in parsed:
                                                    bash_execution_command = parsed[
                                                        "command"
                                                    ]
                                                    # Live stream bash code inside details block
                                                    if not code_exec_is_web_filtering or not code_exec_had_web_tools:
                                                        code_exec_current_code = bash_execution_command
                                                        code_exec_current_lang = "bash"
                                                        block = self._format_code_execution_block(
                                                            code_exec_current_code, code_exec_current_lang,
                                                        )
                                                        await update_content_block(code_exec_last_block, block)
                                                        code_exec_last_block = block
                                                    logger.debug(
                                                        f"Bash execution command: {bash_execution_command[:100]}..."
                                                    )
                                            except Exception as e:
                                                logger.debug(
                                                    f"Bash execution input extraction error: {e}"
                                                )
                                        elif (
                                            active_server_tool_name
                                            == "text_editor_code_execution"
                                        ):
                                            # Text editor input - extract command and file_text
                                            try:
                                                parsed = json.loads(
                                                    server_tool_input_buffer
                                                )
                                                if "command" in parsed:
                                                    text_editor_command = parsed[
                                                        "command"
                                                    ]
                                                if "path" in parsed:
                                                    text_editor_file_path = parsed[
                                                        "path"
                                                    ]
                                                if "file_text" in parsed:
                                                    text_editor_file_content = (
                                                        parsed["file_text"]
                                                    )
                                                    # Live stream file content inside details block
                                                    if text_editor_command == "create" and text_editor_file_content:
                                                        TEXT_EXTENSIONS = {".md", ".txt", ".csv", ".json", ".xml", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".log", ".rst", ".html", ".htm", ".css"}
                                                        file_ext = os.path.splitext(text_editor_file_path)[1].lower() if text_editor_file_path else ""
                                                        if file_ext not in TEXT_EXTENSIONS:
                                                            EXT_TO_LANG = {".py": "python", ".js": "javascript", ".ts": "typescript", ".sh": "bash", ".sql": "sql"}
                                                            lang = EXT_TO_LANG.get(file_ext, "python")
                                                            block = self._format_code_execution_block(text_editor_file_content, lang)
                                                            await update_content_block(code_exec_last_block, block)
                                                            code_exec_last_block = block
                                            except Exception as e:
                                                logger.debug(
                                                    f"Text editor input extraction error: {e}"
                                                )
                                        elif active_server_tool_name in [
                                            "tool_search_tool_regex",
                                            "tool_search_tool_bm25",
                                        ]:
                                            # Tool search input - extract and show query
                                            try:
                                                parsed = json.loads(
                                                    server_tool_input_buffer
                                                )
                                                if "query" in parsed:
                                                    search_query = parsed["query"]
                                                    logger.debug(
                                                        f"Tool search query: '{search_query}'"
                                                    )
                                                    await emit_event_local(
                                                        {
                                                            "type": "status",
                                                            "data": {
                                                                "description": f"🔍 Searching tools: {search_query}",
                                                                "done": False,
                                                            },
                                                        }
                                                    )
                                            except Exception as e:
                                                logger.debug(
                                                    f"Tool search query extraction error: {e}"
                                                )
                                    else:
                                        # Client-side tool - accumulate in tools_buffer
                                        tools_buffer += partial
                                        tool_input_buffer += partial

                                        # Live stream tool input to UI (like thinking blocks stream text)
                                        if not in_code_execution and tool_id_at_start in tool_progress_blocks:
                                            parsed_input = self._try_parse_partial_json(tool_input_buffer)
                                            if parsed_input is not None:
                                                old_block = tool_progress_blocks[tool_id_at_start]
                                                new_block = self._format_tool_result_block(
                                                    tool_id_at_start, tool_name, parsed_input, "", done=False
                                                )
                                                # Replace old in-progress text with updated text
                                                text = final_text()
                                                text = text.replace(old_block, new_block, 1)
                                                tool_progress_blocks[tool_id_at_start] = new_block
                                                final_message.clear()
                                                final_message.append(text)
                                                await request_ctx.emit_event({"type": "replace", "data": {"content": text}})
                                elif delta_type == "citations_delta":
                                        # Web search citations arrive BEFORE the text they cite.
                                        # Emit marker for PREVIOUS citation when a new one arrives
                                        # (so marker appears AFTER the cited text, not before it).
                                        if pending_citation_markers:
                                            citation_str = "".join(f"[{n}]" for n in pending_citation_markers)
                                            chunk += citation_str
                                            pending_citation_markers = []
                                        citation_counter += 1
                                        pending_citation_markers.append(citation_counter)

                                        # Process and store citation
                                        await self.handle_citation(
                                            event, __event_emitter__, citation_counter
                                        )

                            # ---------------------------------------------------------
                            # EVENT: content_block_stop
                            # Finalizes: thinking blocks, tool_use blocks, server tools
                            # Triggers async tool execution for client-side tools
                            # ---------------------------------------------------------
                            elif event_type == "content_block_stop":
                                content_block = getattr(event, "content_block", None)
                                content_type = (
                                    getattr(content_block, "type", None)
                                    if content_block
                                    else current_block_type  # Fallback: raw SDK stop events lack content_block
                                )
                                event_name = getattr(event, "name", "")

                                if content_type == "text":
                                    # Flush any remaining deferred citation markers
                                    if pending_citation_markers:
                                        chunk += "".join(f"[{n}]" for n in pending_citation_markers)
                                        pending_citation_markers = []
                                    if chunk:
                                        if not chunk.endswith("\n"):
                                            chunk += "\n"
                                        await emit_message_delta(chunk)
                                        chunk = ""
                                        chunk_count = 0
                                    elif final_message and not final_text().endswith("\n"):
                                        # Chunk was already flushed by throttle — ensure \n separator
                                        await emit_message_delta("\n")
                                elif content_type == "server_tool_use":
                                    logger.debug(
                                        f"Server tool block stopped: {active_server_tool_name}"
                                    )

                                    # Code is now streamed live inside <details> during input_json_delta
                                    # Just set last_code_content/language for result handlers
                                    if (
                                        active_server_tool_name == "bash_code_execution"
                                        and bash_execution_command
                                    ):
                                        last_code_language = "bash"
                                        last_code_content = bash_execution_command
                                    elif (
                                        active_server_tool_name
                                        == "text_editor_code_execution"
                                        and text_editor_command == "create"
                                        and text_editor_file_content
                                    ):
                                        TEXT_EXTENSIONS = {".md", ".txt", ".csv", ".json", ".xml", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".log", ".rst", ".html", ".htm", ".css"}
                                        file_ext = os.path.splitext(text_editor_file_path)[1].lower() if text_editor_file_path else ""
                                        if file_ext in TEXT_EXTENSIONS:
                                            last_code_content = text_editor_file_content
                                            last_code_language = "__inline_text__"
                                        else:
                                            EXT_TO_LANG = {".py": "python", ".js": "javascript", ".ts": "typescript", ".sh": "bash", ".sql": "sql", ".r": "r", ".rb": "ruby", ".java": "java", ".c": "c", ".cpp": "cpp", ".go": "go", ".rs": "rust"}
                                            last_code_language = EXT_TO_LANG.get(file_ext, "python")
                                            last_code_content = text_editor_file_content
                                    elif (
                                        active_server_tool_name == "code_execution"
                                        and code_execution_code
                                    ):
                                        last_code_language = "python"
                                        last_code_content = code_execution_code
                                    elif active_server_tool_name == "web_fetch":
                                        pass

                                    active_server_tool_name = None
                                    active_server_tool_id = None
                                    server_tool_input_buffer = ""
                                    text_editor_file_content = ""
                                    text_editor_file_path = ""
                                    text_editor_command = ""
                                    bash_execution_command = ""
                                    code_execution_code = ""
                                elif content_type == "tool_use" and tools_buffer:
                                    # Check if it's valid JSON already, if not close it
                                    try:
                                        json.loads(tools_buffer)
                                        # Already valid JSON, no need to close
                                        logger.debug(
                                            f" tools_buffer already valid JSON: {tools_buffer}"
                                        )
                                    except json.JSONDecodeError:
                                        # Check if input is empty (ends with "input": )
                                        if tools_buffer.rstrip().endswith(
                                            '"input":'
                                        ) or tools_buffer.rstrip().endswith(
                                            '"input": '
                                        ):
                                            # Add empty object for input
                                            tools_buffer += " {}"
                                            logger.debug(
                                                f" Added empty input object: {tools_buffer}"
                                            )
                                        # Invalid JSON, need to close the main object
                                        tools_buffer += "}"
                                        logger.debug(
                                            f" Closed tools_buffer in content_block_stop: {tools_buffer}"
                                        )

                                    # Parse and store this tool_use block
                                    logger.debug(f"Parsed tool call: {tools_buffer}")

                                    # Parse and start tool execution immediately!
                                    try:
                                        tool_call_data = json.loads(tools_buffer)
                                        tool_name = tool_call_data.get("name", "")
                                        tool_input = tool_call_data.get("input", {})
                                        tool_id = tool_call_data.get("id", "")

                                        # Store tool_use block for assistant message
                                        # Note: tool_use block with caller field is preserved
                                        # by SDK accumulated message automatically

                                        # Look up tool in __tools__ first (user tools with callable)
                                        tool = (
                                            __tools__.get(tool_name)
                                            if __tools__
                                            else None
                                        )
                                        if tool and tool.get("callable"):
                                            # User tool with callable - execute directly
                                            tool_call_data_list.append(tool_call_data)

                                            # Note: We only emit status events here, NOT done=false details blocks.
                                            # done=false blocks would pile up in the message since we use delta streaming,
                                            # not full message replacement like OpenWebUI's native pipes.

                                            # Start execution immediately as async task
                                            args = (
                                                tool_input
                                                if isinstance(tool_input, dict)
                                                else {}
                                            )
                                            task = asyncio.create_task(
                                                tool["callable"](**args)
                                            )
                                            running_tool_tasks.append(task)

                                            logger.debug(
                                                f"🚀 Started immediate execution for user tool '%s' (task #%d)",
                                                tool_name,
                                                len(running_tool_tasks),
                                            )
                                        # elif tool_name == "memory" and self.valves.ENABLE_CLAUDE_MEMORY:
                                        #     # Memory tool - bridge to OpenWebUI memory system
                                        #     tool_call_data_list.append(tool_call_data)

                                        #     # Execute memory tool operation
                                        #     mem_user_id = __user__.get("id", "") if __user__ else ""
                                        #     task = asyncio.create_task(
                                        #         self._handle_memory_tool(
                                        #             command=tool_input.get("command", "view"),
                                        #             path=tool_input.get("path", "/memories"),
                                        #             user_id=mem_user_id,
                                        #             file_text=tool_input.get("file_text", ""),
                                        #             old_str=tool_input.get("old_str", ""),
                                        #             new_str=tool_input.get("new_str", ""),
                                        #             insert_line=tool_input.get("insert_line", 0),
                                        #             new_path=tool_input.get("new_path", ""),
                                        #         )
                                        #     )
                                        #     running_tool_tasks.append(task)

                                        #     logger.debug(
                                        #         f"🧠 Started memory tool execution: {tool_input.get('command', 'view')} {tool_input.get('path', '/memories')}"
                                        #     )
                                        elif tool_name in builtin_tools and builtin_tools[tool_name].get("callable"):
                                            # Builtin tool from OpenWebUI - execute with proper context
                                            tool_call_data_list.append(tool_call_data)

                                            args = (
                                                tool_input
                                                if isinstance(tool_input, dict)
                                                else {}
                                            )
                                            task = asyncio.create_task(
                                                builtin_tools[tool_name]["callable"](
                                                    **args
                                                )
                                            )
                                            running_tool_tasks.append(task)

                                            logger.debug(
                                                f"🚀 Started immediate execution for builtin tool '%s' (task #%d)",
                                                tool_name,
                                                len(running_tool_tasks),
                                            )
                                        elif tool_name in api_tool_names:
                                            # API-provided tool (no callable) — passthrough mode
                                            # Return the tool input as JSON in the response content
                                            logger.info(
                                                f"🔄 API tool passthrough for '{tool_name}': returning tool input as response"
                                            )
                                            passthrough_json = json.dumps(
                                                tool_input, ensure_ascii=False
                                            )
                                            await emit_message_delta(passthrough_json)
                                            api_tool_passthrough = True
                                        else:
                                            # Unknown tool - add error result
                                            logger.warning(
                                                f"Tool '{tool_name}' not found in __tools__ or builtin_tools"
                                            )
                                            tool_call_data_list.append(tool_call_data)

                                            # Capture tool_name in default arg to avoid closure issue
                                            async def error_result(tn=tool_name):
                                                return json.dumps(
                                                    {
                                                        "error": f"Tool '{tn}' is not available. "
                                                        f"It may require server context or is not configured."
                                                    },
                                                    ensure_ascii=False,
                                                )

                                            task = asyncio.create_task(error_result())
                                            running_tool_tasks.append(task)
                                    except Exception as e:
                                        logger.error(
                                            f"Failed to start tool execution: {e}"
                                        )

                                    # Reset buffer for next tool
                                    tools_buffer = ""
                                elif is_model_thinking and content_type in ("thinking", "redacted_thinking"):
                                    if content_type == "thinking" and thinking_message:
                                        duration = time.time() - (thinking_start_time or time.time())
                                        formatted = self._format_thinking_block(thinking_message, duration)
                                        await update_content_block(thinking_last_block, formatted)
                                        thinking_last_block = ""
                                        logger.debug(f"Finalized thinking block ({len(thinking_message)} chars, {duration:.1f}s)")
                                    elif content_type == "redacted_thinking":
                                        logger.debug("Redacted thinking block completed (preserved by SDK)")
                                    is_model_thinking = False
                                    thinking_message = ""
                                    thinking_stream_start_idx = -1
                                # Reset tracked type
                                current_block_type = None

                            # ---------------------------------------------------------
                            # EVENT: message_delta
                            # Updates output token counts, handles stop_reason
                            # Flushes buffered chunks
                            # ---------------------------------------------------------
                            elif event_type == "message_delta":
                                if include_usage:
                                    usage = getattr(event, "usage", None)
                                    if usage:
                                        current_output_tokens = getattr(
                                            usage, "output_tokens", 0
                                        )
                                        diff = (
                                            current_output_tokens - stream_output_tokens
                                        )
                                        total_usage["output_tokens"] += diff
                                        stream_output_tokens = current_output_tokens
                                delta = getattr(event, "delta", None)
                                code_execution_container_id = getattr(delta, "container", None)
                                if code_execution_container_id:
                                    delta_container_id = getattr(code_execution_container_id, "id", None) if hasattr(code_execution_container_id, "id") else (code_execution_container_id.get("id") if isinstance(code_execution_container_id, dict) else str(code_execution_container_id))
                                    if delta_container_id:
                                        current_container_id = payload_for_stream.get("container")
                                        if current_container_id != delta_container_id:
                                            chunk += self._create_metadata_marker(
                                                "container_id",
                                                delta_container_id,
                                                messagenum=len(
                                                    payload_for_stream.get("messages", [])
                                                ),
                                            )
                                            logger.debug(
                                                f"📦 Container ID from message_delta: {delta_container_id}"
                                            )
                                        payload_for_stream["container"] = delta_container_id

                                stop_reason = getattr(delta, "stop_reason", None)
                                if stop_reason:
                                    logger.debug(f"📍 stop_reason received: {stop_reason}")
                                if stop_reason == "tool_use":
                                    # Emit any remaining text chunk before tool results
                                    if chunk:
                                        if not chunk.endswith("\n"):
                                            chunk += "\n"
                                        await emit_message_delta(chunk)
                                        chunk = ""
                                        chunk_count = 0

                                    # API tool passthrough — skip tool loop, return directly
                                    if api_tool_passthrough and not running_tool_tasks:
                                        logger.info(
                                            "🔄 API tool passthrough complete — skipping tool loop"
                                        )
                                        conversation_ended = True
                                        break

                                    # Wait for all running tool tasks to complete
                                    if running_tool_tasks:
                                        logger.debug(
                                            f"⏳ Waiting for %d tool tasks to complete...",
                                            len(running_tool_tasks),
                                        )

                                        try:
                                            results = await asyncio.gather(
                                                *running_tool_tasks
                                            )
                                            logger.debug(
                                                f"✅ All %d tool tasks completed",
                                                len(results),
                                            )

                                            # Build tool_result messages and emit to UI
                                            for tool_call_data, tool_result in zip(
                                                tool_call_data_list, results
                                            ):
                                                tool_use_id = tool_call_data.get(
                                                    "id", ""
                                                )
                                                tool_name = tool_call_data.get(
                                                    "name", ""
                                                )
                                                tool_input = tool_call_data.get(
                                                    "input", {}
                                                )

                                                # Process tool result through OpenWebUI's handler
                                                # for Rich UI (HTMLResponse, embeds, files)
                                                tool_result_embeds = []
                                                tool_result_files = []
                                                if PROCESS_TOOL_RESULT_AVAILABLE and __request__:
                                                    try:
                                                        tool_result, tool_result_files, tool_result_embeds = (
                                                            process_tool_result(
                                                                __request__,
                                                                tool_name,
                                                                tool_result,
                                                                "pipe",
                                                                metadata=__metadata__,
                                                                user=__user__,
                                                            )
                                                        )
                                                    except Exception as e:
                                                        logger.warning(f"process_tool_result failed for '{tool_name}': {e}")

                                                # Emit files event if tool produced files
                                                if tool_result_files and __event_emitter__:
                                                    await __event_emitter__(
                                                        {
                                                            "type": "files",
                                                            "data": {"files": tool_result_files},
                                                        }
                                                    )

                                                # Determine if error
                                                is_error = isinstance(
                                                    tool_result, str
                                                ) and (
                                                    tool_result.startswith("Error:")
                                                    or tool_result.startswith("Error executing tool")
                                                )

                                                # Build result block for API
                                                # Ensure result is valid JSON string (not Python repr with single quotes)
                                                if isinstance(tool_result, str):
                                                    result_str = tool_result
                                                else:
                                                    try:
                                                        result_str = json.dumps(tool_result, ensure_ascii=False)
                                                    except (TypeError, ValueError):
                                                        result_str = str(tool_result)
                                                result_block = {
                                                    "type": "tool_result",
                                                    "tool_use_id": tool_use_id,
                                                    "content": result_str,
                                                }
                                                if is_error:
                                                    result_block["is_error"] = True
                                                tool_calls.append(result_block)

                                                if in_code_execution:
                                                    # Accumulate for unified code execution display
                                                    code_exec_tool_calls_info.append({
                                                        "name": tool_name,
                                                        "input": tool_input,
                                                        "result": result_str,
                                                        "is_error": is_error,
                                                    })
                                                else:
                                                    # Replace the in-progress block with completed version
                                                    completed = self._format_tool_result_block(
                                                        tool_use_id, tool_name, tool_input,
                                                        str(tool_result), is_error=is_error, done=True,
                                                        embeds=tool_result_embeds, files=tool_result_files,
                                                    )
                                                    old_block = tool_progress_blocks.pop(tool_use_id, None)
                                                    if old_block:
                                                        text = final_text()
                                                        text = text.replace(old_block, completed, 1)
                                                        final_message.clear()
                                                        final_message.append(text)
                                                        await request_ctx.emit_event({"type": "replace", "data": {"content": text}})
                                                    else:
                                                        # Fallback: append if placeholder not found
                                                        final_message.append(completed)
                                                        await emit_message_replace(final_text())

                                            logger.debug(
                                                f"Emitted {len(results)} tool result(s)"
                                            )
                                        except Exception as ex:
                                            logger.error(
                                                f"❌ Tool execution failed: %s", ex
                                            )
                                            # Create error results and update in-progress blocks
                                            for (
                                                tool_call_data
                                            ) in tool_call_data_list:
                                                tool_use_id = tool_call_data.get(
                                                    "id", ""
                                                )
                                                tool_name = tool_call_data.get(
                                                    "name", "unknown"
                                                )
                                                error_result = f"Error executing tool '{tool_name}': {str(ex)}"
                                                tool_calls.append(
                                                    {
                                                        "type": "tool_result",
                                                        "tool_use_id": tool_use_id,
                                                        "content": error_result,
                                                        "is_error": True,
                                                    }
                                                )
                                                # Replace in-progress block with error block
                                                tool_input = tool_call_data.get("input", {})
                                                completed = self._format_tool_result_block(
                                                    tool_use_id, tool_name, tool_input,
                                                    error_result, is_error=True, done=True
                                                )
                                                old_block = tool_progress_blocks.pop(tool_use_id, None)
                                                if old_block:
                                                    text = final_text()
                                                    text = text.replace(old_block, completed, 1)
                                                    final_message.clear()
                                                    final_message.append(text)
                                                    await request_ctx.emit_event({"type": "replace", "data": {"content": text}})

                                    logger.debug(
                                        f" Tool use detected, collected {len(tool_calls)} tool results:\nTool_Call JSON: {tool_calls}"
                                    )

                                    # Reset for next iteration
                                    running_tool_tasks = []
                                    tool_call_data_list = []
                                    tool_progress_blocks = {}
                                    api_tool_passthrough = False
                                    has_pending_tool_calls = True
                                elif stop_reason == "max_tokens":
                                    chunk += "Claude has Reached the maximum token limit!"
                                elif stop_reason == "end_turn":
                                    conversation_ended = True
                                elif stop_reason == "pause_turn":
                                    # API paused a long-running turn — auto-continue
                                    has_pending_tool_calls = True  # reuses tool loop mechanism
                                    # tool_calls stays empty → PHASE 5 detects pause_turn
                                    await emit_event_local(
                                        {
                                            "type": "status",
                                            "data": {
                                                "description": "⏳ Long-running turn paused, continuing...",
                                                "done": False,
                                            },
                                        }
                                    )
                                elif stop_reason == "refusal":
                                    chunk += "Claude has refused to answer based on its content policies."
                                    conversation_ended = True
                                elif stop_reason == "stop_sequence":
                                    chunk += "Claude stopped generating based on stop sequence."
                                    conversation_ended = True
                                elif stop_reason == "model_context_window_exceeded":
                                    chunk += "Claude has reached the maximum context window for this model."
                                    conversation_ended = True

                            # ---------------------------------------------------------
                            # EVENT: message_stop
                            # Stream complete for this turn
                            # ---------------------------------------------------------
                            elif event_type == "message_stop":
                                pass  # No deferred blocks to flush

                            # ---------------------------------------------------------
                            # EVENT: message_error
                            # Handle stream-level errors
                            # ---------------------------------------------------------
                            elif event_type == "message_error":
                                error = getattr(event, "error", None)
                                if error:
                                    # Handle stream errors through handle_errors method
                                    error_details = f"Stream Error: {getattr(error, 'message', str(error))}"
                                    if hasattr(error, "type"):
                                        error_details = f"Stream Error ({error.type}): {getattr(error, 'message', str(error))}"

                                    # Create a mock exception for consistent error handling
                                    stream_error = Exception(error_details)
                                    await self.handle_errors(
                                        stream_error, __event_emitter__
                                    )
                                    return (
                                        final_text()
                                        + f"\n\nAn error occurred: {error_details}"
                                    )

                            if chunk_count > token_buffer_size:
                                if chunk:
                                    await emit_message_delta(chunk)
                                    chunk = ""
                                    chunk_count = 0

                        # Capture SDK accumulated message after stream is fully consumed
                        # This replaces manual api_assistant_blocks/thinking_blocks accumulation
                        sdk_final_message = stream.current_message_snapshot

                    # Log stream event diagnostics
                    logger.debug(f"📊 Stream events: {stream_event_counts}")

                    conversation_ended, has_pending_tool_calls, tool_calls = await self._apply_sdk_stop_reason_fallback(
                        sdk_final_message=sdk_final_message,
                        conversation_ended=conversation_ended,
                        has_pending_tool_calls=has_pending_tool_calls,
                        tool_calls=tool_calls,
                        tool_loop_iteration=tool_loop_iteration,
                        payload_for_stream=payload_for_stream,
                        stream_event_counts=stream_event_counts,
                        request_ctx=request_ctx,
                    )

                    if chunk:
                        await emit_message_delta(chunk)
                        chunk = ""
                        chunk_count = 0

                    # ---------------------------------------------------------
                    # PHASE 5: TOOL EXECUTION LOOP
                    # After stream ends, if tools were called:
                    # 1. Check max tool call limit
                    # 2. Build assistant message with thinking + text + tool_use blocks
                    # 3. Execute tools and collect results
                    # 4. Add tool_result blocks as user message
                    # 5. Loop back to API for continuation
                    # ---------------------------------------------------------
                    if has_pending_tool_calls and tool_calls:
                        # Log tool call details
                        tool_names = [tc.get("name", tc.get("tool_use_id", "?")) for tc in tool_calls]
                        sdk_block_types = [getattr(b, "type", "?") for b in sdk_final_message.content] if sdk_final_message else []
                        logger.info(
                            f"🔧 Tool loop iter {tool_loop_iteration} complete | "
                            f"{len(tool_calls)} tool results: {tool_names} | "
                            f"SDK blocks: {sdk_block_types}"
                        )
                        # Check if we've reached the max tool call limit
                        # Count actual tool results (not loop iterations) for accurate tracking
                        num_tool_results = sum(1 for tc in tool_calls if tc.get("type") == "tool_result")
                        current_function_calls += num_tool_results
                        if current_function_calls >= max_function_calls:
                            await emit_event_local(
                                {
                                    "type": "status",
                                    "data": {
                                        "description": f"⚠️ Maximum tool call limit ({max_function_calls}) reached. Stopping tool execution.",
                                        "done": True,
                                    },
                                }
                            )
                            await emit_event_local(
                                {
                                    "type": "notification",
                                    "data": {
                                        "type": "warning",
                                        "content": f"Tool call limit ({max_function_calls}) reached. Increase MAX_TOOL_CALLS in valves if needed.",
                                    },
                                }
                            )
                            await emit_message_delta(
                                f"\n\n⚠️ **Tool call limit reached** ({current_function_calls}/{max_function_calls}). Some tool results may not have been processed. You can increase the limit in the model's valve settings."
                            )
                            break

                        # Tools were already executed during stream (in message_delta)
                        # tool_calls now contains tool_result blocks ready for API
                        # UI output was already emitted during message_delta

                        # Build assistant message from SDK accumulated message
                        # SDK correctly handles: signature accumulation, block ordering,
                        # caller field preservation, input JSON assembly
                        if sdk_final_message:
                            assistant_content = self._convert_sdk_message_to_api_blocks(sdk_final_message)
                            logger.debug(
                                f"Built assistant_content from SDK message: "
                                f"{[b.get('type') for b in assistant_content]}"
                            )
                        else:
                            # Fallback: build from final_message text
                            assistant_content = []
                            final_message_snapshot = final_text()
                            if final_message_snapshot.strip():
                                assistant_content.append({"type": "text", "text": final_message_snapshot})
                            logger.warning("No SDK message available, using text fallback")

                        if assistant_content:
                            # Log detailed block analysis for debugging
                            for i, block in enumerate(assistant_content):
                                btype = block.get("type", "?")
                                if btype == "thinking":
                                    logger.debug(
                                        f"  assistant_content[{i}]: thinking "
                                        f"({len(block.get('thinking', ''))}c, "
                                        f"sig={len(block.get('signature', ''))}c)"
                                    )
                                elif btype == "redacted_thinking":
                                    logger.debug(
                                        f"  assistant_content[{i}]: redacted_thinking "
                                        f"(data={len(block.get('data', ''))}c)"
                                    )
                                elif btype == "tool_use":
                                    logger.debug(
                                        f"  assistant_content[{i}]: tool_use "
                                        f"name={block.get('name')}, id={block.get('id')}"
                                    )
                                elif btype == "text":
                                    logger.debug(
                                        f"  assistant_content[{i}]: text ({len(block.get('text', ''))}c)"
                                    )
                                else:
                                    logger.debug(f"  assistant_content[{i}]: {btype}")

                            payload_for_stream["messages"].append(
                                {"role": "assistant", "content": assistant_content}
                            )

                        # Safety: ensure every tool_use in assistant has a tool_result
                        tool_use_ids_in_assistant = {
                            b.get("id") for b in assistant_content
                            if b.get("type") == "tool_use"
                        }
                        tool_result_ids = {
                            b.get("tool_use_id") for b in tool_calls
                            if b.get("type") == "tool_result"
                        }
                        missing_ids = tool_use_ids_in_assistant - tool_result_ids
                        for missing_id in missing_ids:
                            logger.warning(f"⚠️ Missing tool_result for tool_use {missing_id}, adding error result")
                            tool_calls.append({
                                "type": "tool_result",
                                "tool_use_id": missing_id,
                                "content": "Error: tool execution failed - no result was produced",
                                "is_error": True,
                            })

                        # Add user message with tool results (tool_calls already contains tool_result blocks)
                        user_content = tool_calls.copy()
                        if user_content:
                            # Optimization: Move cache_control to the end for multi-step tool loops
                            # This ensures we cache the tool results for the next iteration
                            # IMPORTANT: Skip when programmatic tool calling is active - Anthropic rejects
                            payload_for_stream["messages"].append(
                                {"role": "user", "content": user_content}
                            )
                            # Debug log tool results with content sizes
                            if logger.isEnabledFor(logging.DEBUG):
                                for b in user_content:
                                    if b.get("type") == "tool_result":
                                        _content = b.get("content", "")
                                        _clen = len(_content) if isinstance(_content, str) else len(json.dumps(_content, default=str))
                                        logger.debug(
                                            f"📤 tool_result: id={b.get('tool_use_id', '?')[:25]} | "
                                            f"is_error={b.get('is_error', False)} | "
                                            f"content_size={_clen}c"
                                        )

                        # Ensure we added at least one message, otherwise break the loop
                        if not assistant_content and not user_content:
                            logger.debug(
                                f"🔧 No valid content to add, ending conversation"
                            )
                            break

                        # Check if we're approaching the limit BEFORE next iteration
                        # (current_function_calls already updated above with actual tool result count)
                        remaining = max_function_calls - current_function_calls
                        if remaining <= 0:
                            # Hard limit reached - this shouldn't happen as we check above, but safety first
                            break
                        elif remaining == 1:
                            # Only 1 call left - warn Claude this is the final chance
                            await emit_event_local(
                                {
                                    "type": "status",
                                    "data": {
                                        "description": f"⚠️ Final tool call available ({current_function_calls}/{max_function_calls} used)",
                                        "done": False,
                                    },
                                }
                            )
                            await asyncio.sleep(0.05)

                            # Add system message to warn Claude
                            # Skip when programmatic tool calling is active - only tool_result blocks allowed
                            if not self.valves.ENABLE_PROGRAMMATIC_TOOL_CALLING:
                                payload_for_stream["messages"].append(
                                    {
                                        "role": "user",
                                        "content": [
                                            {
                                                "type": "text",
                                                "text": f"⚠️ SYSTEM WARNING: Tool call limit nearly reached ({current_function_calls}/{max_function_calls} used). You have 1 tool call remaining. After the next tool use, the conversation will be automatically terminated. Please provide a comprehensive text response instead of calling more tools, and suggest the user continue manually if needed.",
                                            }
                                        ],
                                    }
                                )
                        elif remaining <= 5:
                            # Approaching limit - inform both user and Claude
                            await emit_event_local(
                                {
                                    "type": "status",
                                    "data": {
                                        "description": f"⚠️ {remaining} tool call(s) remaining ({current_function_calls}/{max_function_calls} used)",
                                        "done": False,
                                    },
                                }
                            )
                            await asyncio.sleep(0.05)

                            # Notify Claude about remaining calls so it can plan accordingly
                            if not self.valves.ENABLE_PROGRAMMATIC_TOOL_CALLING:
                                payload_for_stream["messages"].append(
                                    {
                                        "role": "user",
                                        "content": [
                                            {
                                                "type": "text",
                                                "text": f"[SYSTEM: {remaining} tool call(s) remaining out of {max_function_calls}. Plan your remaining tool calls carefully.]",
                                            }
                                        ],
                                    }
                                )

                        has_pending_tool_calls = False
                        tool_calls = []
                        sdk_final_message = None  # Reset for next iteration
                        chunk = ""
                        chunk_count = 0
                        current_search_query = (
                            ""  # Reset search query for next iteration
                        )
                        citation_counter = (
                            0  # Reset citation counter for next iteration
                        )
                        pending_citation_markers = []  # Reset pending citations
                        continue

                    # pause_turn continuation: API paused a long-running turn,
                    # send the response back as-is to let Claude continue
                    elif has_pending_tool_calls and not tool_calls:
                        logger.info(
                            f"⏸️ pause_turn continuation (iter {tool_loop_iteration})"
                        )
                        if sdk_final_message:
                            assistant_content = self._convert_sdk_message_to_api_blocks(sdk_final_message)
                            if assistant_content:
                                payload_for_stream["messages"].append(
                                    {"role": "assistant", "content": assistant_content}
                                )
                        has_pending_tool_calls = False
                        sdk_final_message = None
                        chunk = ""
                        chunk_count = 0
                        current_search_query = ""
                        citation_counter = 0
                        pending_citation_markers = []
                        continue

                    # SAFETY / TRUNCATED STREAM RETRY:
                    # If we reach here, the stream completed but no tool loop
                    # continuation was triggered and conversation_ended is False.
                    # This typically means a truncated stream (200 OK but no stop_reason).
                    # Auto-retry with the same payload instead of silently breaking.
                    if not conversation_ended:
                        retry_attempts += 1
                        if retry_attempts <= self.valves.MAX_RETRIES:
                            # Determine what happened for logging
                            sdk_block_types = (
                                [getattr(b, "type", "?") for b in getattr(sdk_final_message, "content", [])]
                                if sdk_final_message else []
                            )
                            logger.warning(
                                f"⚠️ Truncated stream (no stop_reason, no tool handling). "
                                f"SDK blocks: {sdk_block_types}. "
                                f"Auto-retrying ({retry_attempts}/{self.valves.MAX_RETRIES})..."
                            )
                            await request_ctx.emit_event(
                                {
                                    "type": "status",
                                    "data": {
                                        "description": f"⚠️ Stream abgebrochen, Retry ({retry_attempts}/{self.valves.MAX_RETRIES})...",
                                        "done": False,
                                    },
                                }
                            )
                            # Reset streaming state for retry — clear any partial content
                            # from this truncated iteration so we get a clean response
                            final_message.clear()
                            sdk_final_message = None
                            chunk = ""
                            chunk_count = 0
                            current_search_query = ""
                            citation_counter = 0
                            pending_citation_markers = []
                            citations_list = []
                            first_text_emitted = False
                            # Reset thinking state
                            thinking_message = ""
                            thinking_start_time = None
                            thinking_stream_start_idx = -1
                            thinking_last_block = ""
                            # Reset server tool state
                            active_server_tool_name = None
                            active_server_tool_id = None
                            server_tool_input_buffer = ""
                            in_code_execution = False
                            code_exec_current_code = ""
                            code_exec_last_block = ""
                            current_block_type = None
                            # payload_for_stream stays unchanged → same messages, same tools
                            # Cache from previous messages is preserved server-side
                            continue
                        else:
                            logger.error(
                                f"❌ Truncated stream: max retries ({self.valves.MAX_RETRIES}) exhausted. "
                                f"Returning error to user."
                            )
                            await request_ctx.emit_delta(
                                "\n\n⚠️ Die Anthropic API hat den Stream mehrfach abgebrochen "
                                f"({self.valves.MAX_RETRIES} Versuche). Bitte versuche es erneut."
                            )
                    break

                # ---------------------------------------------------------
                # PHASE 6: ERROR HANDLING
                # Catches and handles Anthropic API errors with retry logic:
                # - RateLimitError (429): Retryable, backoff
                # - AuthenticationError (401): API key issues
                # - InternalServerError (500, 529): Retryable
                # - APIConnectionError: Network issues, retryable
                # ---------------------------------------------------------
                except Exception as e:
                    # Finalize any open live code_exec block before handling error
                    if code_exec_current_code:
                        duration = time.time() - code_exec_start_time if code_exec_start_time else None
                        block = self._format_code_execution_block(
                            code_exec_current_code, code_exec_current_lang,
                            done=True, duration=duration,
                        )
                        await update_content_block(code_exec_last_block, block)
                        code_exec_last_block = ""
                        code_exec_current_code = ""
                    should_retry, retry_attempts, response_suffix = await self._handle_stream_exception(
                        e,
                        retry_attempts=retry_attempts,
                        request_ctx=request_ctx,
                    )
                    if should_retry:
                        continue
                    if response_suffix:
                        return final_text() + response_suffix
                    return final_text()
        except Exception as e:
            await self.handle_errors(e, __event_emitter__)
            return final_text()

        # ---------------------------------------------------------
        # PHASE 7: FINALIZATION
        # After successful completion:
        # - Build final status with token count display
        # - Emit completion status event
        # - Emit chat:completion event with usage stats
        # - Return final message text
        # ---------------------------------------------------------

        final_status = "✅ Response Complete"
        # ============ Token Count Display ============
        if include_usage and total_usage:
            def format_num(n: int) -> str:
                if n >= 1_000_000:
                    return f"{n/1_000_000:.1f}M"
                if n >= 1_000:
                    return f"{n/1_000:.1f}K"
                return str(n)

            # Context window progress bar
            if __user__["valves"].SHOW_TOKEN_COUNT:
                # Use total_tokens from total_usage which now represents the last turn (Context Size)
                total_tokens = total_usage.get("total_tokens", 0)

                # Determine context window based on model capability and valve setting
                model_info = self.get_model_info(body["model"].split("/")[-1])
                is_1m = model_info.get("supports_1m_context", False)
                context_window = 1_000_000 if is_1m else 200_000
                context_label = "1M" if is_1m else "200k"
                percentage = min((total_tokens / context_window) * 100, 100)

                # Progress bar (10 segments)
                filled = int(percentage / 10)
                bar = "█" * filled + "░" * (10 - filled)

                final_status += (
                    f" [{bar}] {format_num(total_tokens)}/{context_label} ({percentage:.1f}%)"
                )

            # Cache status display
            if (
                __user__["valves"].SHOW_CACHE_INFO and
                self.valves.CACHE_CONTROL != "cache disabled"
            ):
                ttl_label = "1hr" if self.valves.CACHE_TTL == "1 hour" else "5min"
                cache_write = total_usage.get("cache_creation_input_tokens", 0)
                cache_read = total_usage.get("cache_read_input_tokens", 0)
                final_status += (
                    f" | Cache write ({ttl_label}): {format_num(cache_write)}"
                    f" | Cache read: {format_num(cache_read)}"
                )

        # Consolidate: emit a final replace with the complete message so OpenWebUI
        # has the authoritative content (replaces any partial delta/replace state).
        consolidated = final_text()
        if consolidated:
            await emit_event_local(
                {"type": "replace", "data": {"content": consolidated}}
            )

        await emit_event_local(
            {
                "type": "status",
                "data": {
                    "description": final_status,
                },
            }
        )
        
        logger.debug(f"Final status emitted: {final_status}")
        logger.debug(f"Total usage: {total_usage}")
        logger.debug(f"Final Message: {consolidated}")
        
        # Emit chat:completion done event so frontend knows streaming finished
        # (triggers TTS finish, usage display, etc.)
        done_data: dict = {"choices": [{"finish_reason": "stop", "delta": {"content": ""}}], "done": True}
        if include_usage and total_usage:
            done_data["usage"] = total_usage
        await emit_event_local({"type": "chat:completion", "data": done_data})

        if __user__["valves"].DEBUG_MODE:
            # DEBUG: content already streamed via emit_event_local; skipping duplicate
            pass

    # =========================================================================
    # TASK MODEL (TITLE, TAGS, FOLLOW-UPS)
    # =========================================================================

    async def _run_task_model_request(
        self,
        body: dict[str, Any],
    ) -> str:
        """
        Handle task model requests (title generation, tags, follow-ups etc.) by making a
        non-streaming request to Anthropic API and returning only the text response.

        Task models should return plain text without any JSON formatting or status updates
        mixed into the response.
        """
        try:
            # Extract model and messages from body
            actual_model_name = body["model"].split("/")[-1]
            messages = body.get("messages", [])

            # Build simple payload for task request (non-streaming)
            task_payload = {
                "model": actual_model_name,
                "max_tokens": body.get("max_tokens", 4096),
                "messages": self._process_messages_for_task(messages),
                "stream": False,
            }

            logger.debug(f"Task payload: {json.dumps(task_payload, indent=2)}")

            # Make synchronous request to Anthropic API
            # For task requests, we don't have __user__ context, so use default key
            api_key = self.valves.ANTHROPIC_API_KEY
            base_url = self.valves.ANTHROPIC_BASE_URL.strip() or None
            client = AsyncAnthropic(api_key=api_key, **({"base_url": base_url} if base_url else {}))

            response = await client.messages.create(**task_payload)

            # Extract text from response
            text_parts = []
            for content_block in response.content:
                if content_block.type == "text":
                    text_parts.append(content_block.text)

            # Join without adding line breaks - preserve original formatting
            result = "".join(text_parts).strip()

            logger.debug(f"Task response: {result}")

            return result

        except Exception as e:
            logger.debug(f"Task model error: {e}")
            return ""

    def _process_messages_for_task(self, messages: List[dict]) -> List[dict]:
        """
        Process messages for task requests - convert to simple Anthropic format.
        Task requests don't need complex content processing.
        """
        processed = []
        for msg in messages:
            role = msg.get("role")
            if role == "system":
                continue  # System messages handled separately

            content = msg.get("content", "")
            if isinstance(content, str):
                processed.append({"role": role, "content": content})
            elif isinstance(content, list):
                # Extract text from content blocks
                text_parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                if text_parts:
                    processed.append({"role": role, "content": " ".join(text_parts)})

        return processed

    # =========================================================================
    # MEMORY SYSTEM (FILESYSTEM-BACKED CLAUDE MEMORY TOOL)
    # =========================================================================

    def _get_user_memory_dir(self, user_id: str) -> str:
        """Get the memory directory for a specific user, creating it if needed."""
        user_dir = os.path.join(CLAUDE_MEMORY_DIR, user_id, "memories")
        os.makedirs(user_dir, exist_ok=True)
        return user_dir

    def _resolve_memory_path(self, user_id: str, clean_path: str) -> str:
        """Resolve a memory path to an absolute filesystem path, ensuring it stays within the user's memory dir."""
        user_dir = self._get_user_memory_dir(user_id)
        resolved = os.path.realpath(os.path.join(user_dir, clean_path))
        # Prevent path traversal
        if not resolved.startswith(os.path.realpath(user_dir)):
            raise ValueError("Path traversal not allowed")
        return resolved

    async def _handle_memory_tool(
        self,
        command: str,
        path: str,
        user_id: str,
        file_text: str = "",
        old_str: str = "",
        new_str: str = "",
        insert_line: int = 0,
        new_path: str = "",
    ) -> str:
        """
        Handle Claude memory tool operations using filesystem storage.

        Memory files are stored as actual files on disk under:
            {DATA_DIR}/claude_memories/{user_id}/memories/

        This matches Anthropic's file/path model directly — each user
        gets their own directory, files support subdirectories, and all
        operations (view/create/str_replace/insert/delete/rename) are
        natural filesystem operations with no embedding overhead.

        Args:
            command: Memory operation (view, create, str_replace, insert, delete, rename)
            path: File path in memory namespace
            user_id: OpenWebUI user ID
            file_text: Content for create operations
            old_str: String to find for str_replace
            new_str: Replacement string for str_replace
            insert_line: Line number for insert operations
            new_path: New path for rename operations

        Returns:
            String result of the operation
        """
        try:
            # Normalize path - strip leading /memories/
            clean_path = path.strip("/")
            if clean_path.startswith("memories/"):
                clean_path = clean_path[len("memories/"):]
            elif clean_path == "memories":
                clean_path = ""

            user_dir = self._get_user_memory_dir(user_id)

            if command == "view":
                if not clean_path:
                    # Directory listing
                    if not os.path.exists(user_dir):
                        return "/memories/ (empty directory)\n\nNo memory files stored yet. Use 'create' to add memory files."

                    entries = []
                    for root, dirs, files in os.walk(user_dir):
                        rel_root = os.path.relpath(root, user_dir)
                        for f in sorted(files):
                            rel_path = f if rel_root == "." else os.path.join(rel_root, f)
                            full_path = os.path.join(root, f)
                            try:
                                size = os.path.getsize(full_path)
                                entries.append(f"  {rel_path}  ({size} bytes)")
                            except OSError:
                                entries.append(f"  {rel_path}")

                    lines = ["/memories/"]
                    if entries:
                        lines.extend(entries)
                    else:
                        lines.append("  (no files yet)")
                    return "\n".join(lines)
                else:
                    # View specific file
                    file_path = self._resolve_memory_path(user_id, clean_path)
                    if os.path.isdir(file_path):
                        # Subdirectory listing
                        entries = []
                        for item in sorted(os.listdir(file_path)):
                            full = os.path.join(file_path, item)
                            if os.path.isfile(full):
                                size = os.path.getsize(full)
                                entries.append(f"  {item}  ({size} bytes)")
                            elif os.path.isdir(full):
                                entries.append(f"  {item}/")
                        return f"/memories/{clean_path}/\n" + "\n".join(entries) if entries else f"/memories/{clean_path}/ (empty)"
                    elif os.path.isfile(file_path):
                        with open(file_path, "r", encoding="utf-8") as f:
                            return f.read() or "(empty file)"
                    else:
                        return f"Error: File '{clean_path}' not found in /memories/"

            elif command == "create":
                if not file_text:
                    return "Error: file_text is required for create command"
                if not clean_path:
                    return "Error: path with filename is required (e.g., /memories/notes.md)"

                file_path = self._resolve_memory_path(user_id, clean_path)
                # Create parent directories if needed
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                existed = os.path.exists(file_path)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(file_text)
                return f"{'Updated' if existed else 'Created'} file: /memories/{clean_path}"

            elif command == "str_replace":
                if not clean_path:
                    return "Error: path is required for str_replace command"
                file_path = self._resolve_memory_path(user_id, clean_path)
                if not os.path.isfile(file_path):
                    return f"Error: File '{clean_path}' not found"

                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                if old_str not in content:
                    return "Error: old_str not found in file content"
                content = content.replace(old_str, new_str, 1)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                return f"Updated file: /memories/{clean_path}"

            elif command == "insert":
                if not clean_path:
                    return "Error: path is required for insert command"
                file_path = self._resolve_memory_path(user_id, clean_path)
                if not os.path.isfile(file_path):
                    return f"Error: File '{clean_path}' not found"

                with open(file_path, "r", encoding="utf-8") as f:
                    lines = f.read().split("\n")
                insert_idx = max(0, min(insert_line, len(lines)))
                lines.insert(insert_idx, new_str)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(lines))
                return f"Inserted text at line {insert_line} in /memories/{clean_path}"

            elif command == "delete":
                if not clean_path:
                    return "Error: path is required for delete command"
                file_path = self._resolve_memory_path(user_id, clean_path)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    return f"Deleted file: /memories/{clean_path}"
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                    return f"Deleted directory: /memories/{clean_path}"
                return f"Error: File '{clean_path}' not found or could not be deleted"

            elif command == "rename":
                if not clean_path or not new_path:
                    return "Error: path and new_path are required for rename command"

                src_path = self._resolve_memory_path(user_id, clean_path)
                new_clean = new_path.strip("/")
                if new_clean.startswith("memories/"):
                    new_clean = new_clean[len("memories/"):]
                if not new_clean:
                    return "Error: new_path must include a filename (e.g., /memories/new_name.md)"

                dst_path = self._resolve_memory_path(user_id, new_clean)
                if not os.path.exists(src_path):
                    return f"Error: File '{clean_path}' not found"
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                os.rename(src_path, dst_path)
                return f"Renamed /memories/{clean_path} to /memories/{new_clean}"

            else:
                return f"Error: Unknown memory command '{command}'"

        except ValueError as e:
            return f"Error: {str(e)}"
        except Exception as e:
            logger.error(f"Memory tool error: {e}")
            return f"Error: {str(e)}"

    # =========================================================================
    # ERROR HANDLING
    # =========================================================================

    async def handle_errors(
        self,
        exception,
        __event_emitter__: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
    ):
        # Determine specific error message based on exception type
        if isinstance(exception, RateLimitError):
            error_msg = "Rate limit exceeded. Please wait before making more requests."
            user_msg = "⚠️ Rate limit reached. Please try again in a moment."
        elif isinstance(exception, AuthenticationError):
            error_msg = "Authentication failed. Please check your API key."
            user_msg = (
                "🔑 Invalid API key. Please verify your Anthropic API key is correct."
            )
        elif isinstance(exception, PermissionDeniedError):
            error_msg = (
                "Permission denied. Your API key may not have access to this resource."
            )
            user_msg = "🚫 Access denied. Your API key doesn't have permission for this request."
        elif isinstance(exception, NotFoundError):
            error_msg = (
                "Resource not found. The requested model or endpoint may not exist."
            )
            user_msg = "❓ Resource not found. Please check if the model is available."
        elif isinstance(exception, BadRequestError):
            error_msg = f"Bad request: {str(exception)}"
            user_msg = (
                "📝 Invalid request format. Please check your input and try again."
            )
        elif isinstance(exception, UnprocessableEntityError):
            error_msg = f"Unprocessable entity: {str(exception)}"
            user_msg = "📄 Request format issue. Please check your message structure and try again."
        elif isinstance(exception, InternalServerError):
            error_msg = "Anthropic server error. Please try again later."
            user_msg = (
                "🔧 Server temporarily unavailable. Please try again in a few moments."
            )
        elif isinstance(exception, APIConnectionError):
            error_msg = (
                "Network connection error. Please check your internet connection."
            )
            user_msg = "🌐 Connection error. Please check your network and try again."
        elif isinstance(exception, APIStatusError):
            status_code = getattr(exception, "status_code", "Unknown")
            error_msg = f"API Error ({status_code}): {str(exception)}"
            user_msg = (
                f"⚡ API Error ({status_code}). Please try again or contact support."
            )
        else:
            error_msg = f"Unexpected error: {str(exception)}"
            user_msg = "💥 An unexpected error occurred. Please try again."

        logger.error(f"Exception: {error_msg}")
        # Add request ID if available for debugging
        if isinstance(exception, APIStatusError) and hasattr(exception, "response"):
            try:
                request_id = exception.response.headers.get("request-id")
                if request_id:
                    logger.info(f"Request ID: %s", request_id)
            except Exception:
                pass  # Ignore if we can't get request ID

        await self.emit_event(
            {
                "type": "notification",
                "data": {
                    "type": "error",
                    "content": user_msg,
                },
            },
            __event_emitter__,
        )

        tb = traceback.format_exc()

        await self.emit_event(
            {
                "type": "source",
                "data": {
                    "source": {"name": "Anthropic Error", "url": None},
                    "document": [tb],
                    "metadata": [
                        {
                            "source": "anthropic api",
                            "type": "error",
                            "date_accessed": datetime.utcnow().isoformat(),
                        }
                    ],
                },
            },
            __event_emitter__,
        )
        await self.emit_event(
            {
                "type": "status",
                "data": {
                    "description": "❌ Response with Errors",
                    "done": True,
                },
            },
            __event_emitter__,
        )

    # =========================================================================
    # TOOL EXECUTION
    # =========================================================================

    async def _run_tool_callable(
        self,
        tool_callable: Callable[..., Awaitable[Any]],
        args: Dict[str, Any],
        tool_name: str,
    ) -> Any:
        try:
            return await asyncio.wait_for(
                tool_callable(**args),
                timeout=self.valves.TOOL_CALL_TIMEOUT,
            )
        except asyncio.TimeoutError:
            message = f"Error: Tool '{tool_name}' timed out after {self.valves.TOOL_CALL_TIMEOUT} seconds"
            self.logger.debug(message)
            return message
        except Exception as exc:
            self.logger.debug(f"Tool '%s' failed", tool_name, exc_info=exc)
            return f"Error: Tool '{tool_name}' failed: {exc}"

    # =========================================================================
    # TEXT PROCESSING & MEMORY EXTRACTION
    # =========================================================================

    def _extract_and_remove_memories(self, text: str) -> tuple[str, Optional[str]]:
        """
        Extract User Context from Openwebui Memory System from system prompt and remove it.
        Takes everything after "\nUser Context:\n" until end of string.

        Returns:
            tuple[str, Optional[str]]: (cleaned_text, extracted_context)
            - cleaned_text: Original text with User Context removed (stripped)
            - extracted_context: The extracted User Context block with label, or None if not found

        Uses pre-compiled PATTERN_USER_CONTEXT for performance.
        """
        match = PATTERN_USER_CONTEXT.search(text)

        if match:
            context_content = match.group(1).strip()
            extracted_context = (
                f"User Context:\n{context_content}" if context_content else None
            )
            # Remove "\nUser Context:\n" and everything after it
            cleaned_text = text[: match.start()].strip()
            return cleaned_text, extracted_context

        # No User Context found
        return text.strip(), None

    # =========================================================================
    # CITATIONS & EVENT EMISSION
    # =========================================================================

    async def handle_citation(self, event, __event_emitter__, citation_counter=None):
        """
        Handle web search citation events from Anthropic API and emit appropriate source events to OpenWebUI.

        Args:
            event: The citation event from Anthropic (content_block_delta with citations_delta)
            __event_emitter__: OpenWebUI event emitter function
            citation_counter: Optional citation number for inline citations
        """
        try:
            logger.debug(
                f" Processing citation event type: {getattr(event, 'type', 'unknown')}"
            )

            # Extract citation from delta within content_block_delta event
            delta = getattr(event, "delta", None)
            citation = None

            if delta and hasattr(delta, "citation"):
                citation = delta.citation
            elif hasattr(event, "citation"):
                # Fallback: direct citation in event
                citation = event.citation

            if not citation:
                logger.debug(f"No citation data found in event")
                return

            logger.debug(f" Citation data found: {citation}")

            # Only handle web search result citations
            citation_type = getattr(citation, "type", "")
            if citation_type != "web_search_result_location":
                logger.debug(f" Skipping non-web-search citation type: {citation_type}")
                return

            # Extract web search citation information
            url = getattr(citation, "url", "")
            title = getattr(citation, "title", "Unknown Source")
            cited_text = getattr(citation, "cited_text", "")

            # CRITICAL: metadata.source is used by OpenWebUI as the grouping ID
            # Must be unique for each citation to prevent Citation merging
            metadata = {
                "source": f"{url}#{citation_counter}",
                "date_accessed": datetime.now().isoformat(),
                "name": f"[{citation_counter}]",
            }

            source_data = {
                "source": {
                    "name": title,
                    "url": url,
                    "id": f"{citation_counter}",  # Unique source ID
                },
                "document": [cited_text],
                "metadata": [metadata],
            }

            # Emit the source event
            await self.emit_event(
                {"type": "source", "data": source_data}, __event_emitter__
            )

        except Exception as e:
            logger.error(f"Error handling citation: {str(e)}")
            await self.handle_errors(e, __event_emitter__)

    async def emit_event(
        self,
        event: Dict[str, Any],
        __event_emitter__: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
    ) -> None:
        """
        Safely emit an event, handling None __event_emitter__ (e.g., in Channel contexts).

        In OpenWebUI Channels, when models are mentioned, __event_emitter__ is None
        because the channel context doesn't provide a socket connection for status updates.
        This helper prevents 'NoneType' object is not callable errors.
        """
        if __event_emitter__ is None:
            return
        try:
            await __event_emitter__(event)
        except Exception as e:
            logger.warning(f"Event emitter failed: {e}")

    # =========================================================================
    # SDK MESSAGE CONVERSION HELPER
    # Converts SDK BetaMessage content blocks to API-compatible dicts
    # =========================================================================
    # CRITICAL: All blocks must be preserved to maintain thinking block positions.
    # The SDK (and Anthropic's tool runner) keeps ALL blocks as-is when sending
    # assistant content back during tool loops. Stripping server_tool_use or
    # *_tool_result shifts thinking block indices, causing:
    #   "thinking blocks cannot be modified"
    # Only strip truly structural meta-events (context_cleared, compaction).
    #
    # Thinking + redacted_thinking get strict key sanitization to prevent
    # cache_control or other extra fields from causing API errors.

    # Block types that must be strictly sanitized (extra keys cause API errors)
    _SANITIZE_BLOCK_KEYS = {
        "thinking": {"type", "thinking", "signature"},  # signature MUST be preserved exactly
        "redacted_thinking": {"type", "data"},           # opaque data, pass through unchanged
    }

    # Block types to skip entirely (structural meta-events, not real content)
    _SKIP_BLOCK_TYPES = frozenset({"context_cleared", "compaction"})

    def _convert_sdk_message_to_api_blocks(self, message) -> list:
        """Convert SDK accumulated BetaMessage content to API-compatible block dicts.

        Mirrors the SDK's own tool runner behavior: keeps ALL content blocks
        (including server_tool_use, *_tool_result) to preserve thinking block
        positions. Only skips structural meta-events (context_cleared, compaction).

        Strict key sanitization is applied ONLY to thinking/redacted_thinking
        blocks (to prevent cache_control from being sent). All other blocks
        are passed through with minimal processing.
        """
        blocks = []
        for block in message.content:
            block_dict = block.model_dump(exclude_none=True)
            block_type = block_dict.get("type", "")

            # Skip structural meta-events (not real content blocks)
            if block_type in self._SKIP_BLOCK_TYPES:
                continue

            # Thinking/redacted_thinking: strict key sanitization
            sanitize_keys = self._SANITIZE_BLOCK_KEYS.get(block_type)
            if sanitize_keys is not None:
                blocks.append({k: v for k, v in block_dict.items() if k in sanitize_keys})
                continue

            # Text blocks: strip citations (response-only presentation data)
            if block_type == "text":
                block_dict.pop("citations", None)
                blocks.append(block_dict)
                continue

            # tool_use blocks: strip "direct" caller (API rejects it),
            # but preserve programmatic caller (needed for code_execution routing)
            if block_type == "tool_use":
                caller = block_dict.get("caller")
                if caller and caller.get("type") == "direct":
                    block_dict.pop("caller", None)
                blocks.append(block_dict)
                continue

            # All other blocks (server_tool_use, *_tool_result, etc.):
            # pass through as-is to preserve thinking block positions
            blocks.append(block_dict)

        return blocks

    # =========================================================================
    # IMMEDIATE BLOCK FORMATTING HELPERS
    # These format individual blocks immediately when they finish streaming
    # =========================================================================
    def _format_thinking_block(
        self, content: str, duration: Optional[float] = None
    ) -> str:
        """Format a thinking block with OpenWebUI native <details type='reasoning'> format.

        This produces the same format that OpenWebUI's built-in pipes use,
        enabling proper spinner, localized text, and collapsible behavior.
        """
        # Escape content and add > prefix per line (OpenWebUI quota block style)
        escaped_lines = "\n".join(
            f"> {html.escape(line)}" if not line.startswith(">") else html.escape(line)
            for line in content.splitlines()
        )

        if duration is not None:
            duration_int = int(duration)
            return (
                f'<details type="reasoning" done="true" duration="{duration_int}">\n'
                f"<summary>Thought for {duration_int} seconds</summary>\n"
                f"{escaped_lines}\n"
                f"</details>\n"
            )
        else:
            return (
                f'<details type="reasoning" done="false">\n'
                f"<summary>Thinking…</summary>\n"
                f"{escaped_lines}\n"
                f"</details>\n"
            )

    def _format_code_block(
        self,
        content: str,
        language: str = "python",
        stdout: Optional[str] = None,
        stderr: Optional[str] = None,
        return_code: Optional[int] = None,
        download_links: Optional[list] = None,
    ) -> str:
        """Format a code execution block with <details> wrapper."""
        label = "Bash Command" if language == "bash" else "Python Script"
        exit_info = f" (exit: {return_code})" if return_code is not None else ""

        result = (
            f"\n<details open>\n"
            f"<summary>🔧 {label}{exit_info}</summary>\n\n"
            f"**Code:**\n"
            f"```{language}\n{content}\n```\n\n"
        )

        if download_links or stdout or stderr:
            result += "**Output:**\n"
            if download_links:
                result += "\n".join(download_links) + "\n\n"
            if stdout:
                result += f"```\n{stdout}\n```\n"
            if stderr:
                result += f"\n**Errors:**\n```\n{stderr}\n```\n"

        result += "</details>\n"
        return result

    @staticmethod
    def _try_parse_partial_json(buffer: str):
        """Try to parse partial JSON by attempting various closing strategies.

        During input_json_delta streaming, the input JSON arrives incrementally.
        This attempts to close the partial JSON to extract a parseable value
        for live UI updates. Returns parsed dict/list/value on success, None on failure.
        """
        if not buffer or not buffer.strip():
            return None
        # Try as-is first (might already be complete)
        for suffix in ("", "}", '"}', '"}}', "]}"):
            try:
                return json.loads(buffer + suffix)
            except (json.JSONDecodeError, ValueError):
                continue
        return None

    def _format_tool_result_block(
        self,
        tool_call_id: str,
        tool_name: str,
        tool_input: dict,
        tool_output: str,
        is_error: bool = False,
        done: bool = True,
        embeds: list = None,
        files: list = None,
    ) -> str:
        """Format a tool result block with OpenWebUI native <details type='tool_calls'> format.

        This produces the same format that OpenWebUI's built-in pipes use,
        enabling proper spinner, localized text, and collapsible behavior.

        Args:
            done: If True, shows "Tool Executed". If False, shows "Executing..." with spinner.
            embeds: List of embed content (HTML strings, URLs) from process_tool_result.
            files: List of file dicts from process_tool_result.
        """
        # Escape arguments for HTML attribute
        escaped_args = (
            html.escape(json.dumps(tool_input, ensure_ascii=False))
            if tool_input
            else ""
        )

        done_str = "true" if done else "false"
        summary = "Tool Executed" if done else "Executing..."
        error_attr = ' error="true"' if is_error and done else ""

        if done:
            # Escape result for HTML attribute
            try:
                if isinstance(tool_output, str):
                    try:
                        parsed = json.loads(tool_output)
                        escaped_result = html.escape(
                            json.dumps(parsed, ensure_ascii=False)
                        )
                    except (json.JSONDecodeError, ValueError):
                        escaped_result = html.escape(
                            json.dumps(tool_output, ensure_ascii=False)
                        )
                else:
                    escaped_result = html.escape(
                        json.dumps(tool_output, ensure_ascii=False)
                    )
            except Exception:
                escaped_result = html.escape(
                    json.dumps(str(tool_output), ensure_ascii=False)
                )

            return (
                f'<details type="tool_calls" done="{done_str}" id="{html.escape(tool_call_id)}" name="{html.escape(tool_name)}" '
                f'arguments="{escaped_args}" result="{escaped_result}" '
                f'files="{html.escape(json.dumps(files)) if files else ""}" '
                f'embeds="{html.escape(json.dumps(embeds)) if embeds else ""}"{error_attr}>\n'
                f"<summary>{summary}</summary>\n"
                f"</details>\n"
            )
        else:
            # In-progress tool call - no result yet
            return (
                f'<details type="tool_calls" done="{done_str}" id="{html.escape(tool_call_id)}" name="{html.escape(tool_name)}" '
                f'arguments="{escaped_args}">\n'
                f"<summary>{summary}</summary>\n"
                f"</details>\n"
            )

    def _format_code_execution_block(
        self,
        code: str,
        language: str = "bash",
        stdout: str = "",
        stderr: str = "",
        return_code: int = None,
        download_links: list = None,
        tool_calls_info: list = None,
    ) -> str:
        """Format a code execution block with output as a collapsible <details> block.
        
        Args:
            tool_calls_info: List of dicts with {name, input, result, is_error} for programmatic tool calls
        """
        # Build summary with tool call count
        tool_count = len(tool_calls_info) if tool_calls_info else 0
        summary_suffix = f" — {tool_count} tool call{'s' if tool_count != 1 else ''}" if tool_count else ""
        
        parts = []
        parts.append(f"\n<details>\n<summary>💻 Code Execution ({language}){summary_suffix}</summary>\n")
        if code:
            parts.append(f"\n```{language}\n{code}\n```\n")
        if tool_calls_info:
            parts.append("\n🔧 **Tool Calls:**\n")
            parts.append("| Tool | Arguments | Result |\n")
            parts.append("|------|-----------|--------|\n")
            for tc in tool_calls_info:
                name = tc.get("name", "?")
                # Format input as compact string
                inp = tc.get("input", {})
                if isinstance(inp, dict):
                    inp_str = ", ".join(f"{k}={v}" for k, v in inp.items())
                else:
                    inp_str = str(inp)
                # Format result - truncate if too long
                result = tc.get("result", "")
                try:
                    parsed_result = json.loads(result) if isinstance(result, str) else result
                    if isinstance(parsed_result, dict) and "result" in parsed_result:
                        result_str = str(parsed_result["result"])
                    else:
                        result_str = str(parsed_result)
                except (json.JSONDecodeError, ValueError):
                    result_str = str(result)
                if len(result_str) > 100:
                    result_str = result_str[:97] + "..."
                error_marker = " ❌" if tc.get("is_error") else ""
                parts.append(f"| {name} | {inp_str} | {result_str}{error_marker} |\n")
            parts.append("\n")
        if stdout:
            parts.append(f"**Output:**\n```\n{stdout}\n```\n")
        if stderr:
            parts.append(f"\n**Errors:**\n```\n{stderr}\n```\n")
        if return_code is not None and return_code != 0:
            parts.append(f"\n**Return code:** {return_code}\n")
        if download_links:
            parts.append("\n**Files:**\n")
            for link in download_links:
                parts.append(f"- {link}\n")
        parts.append("</details>\n")
        return "".join(parts)

    def _format_code_execution_block(
        self,
        code: str,
        language: str = "python",
        done: bool = False,
        duration: float = None,
        stdout: str = "",
        stderr: str = "",
        return_code: int = None,
        download_links: list = None,
        tool_calls_info: list = None,
    ) -> str:
        """Format code execution as <details type="code_interpreter"> matching OpenWebUI native format.

        Uses the same HTML structure as OpenWebUI's built-in code_interpreter,
        giving us spinner, Analyzing.../Analyzed transitions, and output display for free.
        """
        done_str = "true" if done else "false"
        summary = "Analyzed" if done else "Analyzing…"

        # Build display content (code block inside details body)
        display = f"```{language}\n{code}\n```" if code else ""

        # Build output JSON for the output attribute
        # CodeBlock.svelte expects {stdout, stderr, result} keys
        output_data = {}
        if stdout:
            output_data["stdout"] = stdout
        if stderr:
            output_data["stderr"] = stderr
        # Build a result summary for tool calls and other info
        result_parts = []
        if return_code is not None and return_code != 0:
            result_parts.append(f"Exit code: {return_code}")
        if tool_calls_info:
            for tc in tool_calls_info:
                name = tc.get("name", "?")
                res = tc.get("result", "")[:200]
                error = " ❌" if tc.get("is_error") else ""
                result_parts.append(f"🔧 {name}: {res}{error}")
        if download_links:
            result_parts.append("Files: " + ", ".join(download_links))
        if result_parts:
            output_data["result"] = "\n".join(result_parts)

        # Build attributes
        attrs = f'type="code_interpreter" done="{done_str}"'
        if duration is not None and done:
            attrs += f' duration="{duration:.1f}"'
        if output_data:
            output_json = json.dumps(output_data, ensure_ascii=False)
            attrs += f' output="{html.escape(output_json)}"'

        return f"<details {attrs}>\n<summary>{summary}</summary>\n{display}\n</details>\n"

    async def _emit_code_execution_source(
        self,
        emit_event_local: Callable,
        code: str,
        language: str,
        stdout: str = "",
        stderr: str = "",
        return_code: int = None,
        download_links: list = None,
        tool_calls_info: list = None,
    ) -> None:
        """Emit code execution output as a source/citation event for the citation panel."""
        output_parts = []
        if stdout:
            output_parts.append(f"stdout:\n{stdout}")
        if stderr:
            output_parts.append(f"stderr:\n{stderr}")
        if return_code is not None and return_code != 0:
            output_parts.append(f"Return code: {return_code}")
        if download_links:
            output_parts.append("Files:\n" + "\n".join(download_links))

        output_text = "\n\n".join(output_parts) if output_parts else "(no output)"

        # Build a concise code summary for the source name
        code_preview = code[:80].replace("\n", " ").strip() + "..." if code and len(code) > 80 else (code or "").replace("\n", " ").strip()
        source_name = f"💻 {language}: {code_preview}" if code_preview else f"💻 Code Execution ({language})"

        source_data = {
            "source": {
                "name": source_name,
            },
            "document": [output_text],
            "metadata": [
                {
                    "source": f"code_execution_{language}_{id(code)}",
                    "name": source_name,
                }
            ],
        }

        await emit_event_local({"type": "source", "data": source_data})

    # =========================================================================
    # SKILLS VALIDATION AND CONTAINER BUILDING
    # =========================================================================

    async def _validate_and_get_skills(
        self,
        skill_names: List[str],
        api_key: str,
        __event_emitter__: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Validate user-specified skill names against the Anthropic List Skills API.

        Skills can be specified as:
        - Anthropic skills: Short names like "pptx", "xlsx", "docx", "pdf"
        - Custom skills: Full IDs like "skill_01AbCdEfGhIjKlMnOpQrStUv"

        Validation results are cached per API key to avoid repeated API calls.

        Args:
            skill_names: List of skill names/IDs from user's SKILLS valve
            api_key: Anthropic API key
            __event_emitter__: Optional event emitter for status updates

        Returns:
            List of validated skill configurations for the container parameter
        """
        if not skill_names:
            return []

        # Initialize cache for this API key if needed
        if api_key not in self._validated_skills_cache:
            self._validated_skills_cache[api_key] = {}

        cache = self._validated_skills_cache[api_key]

        # Check which skills need validation
        skills_to_validate = [s for s in skill_names if s not in cache]

        # If we have skills to validate, fetch from API
        if skills_to_validate:
            logger.debug(
                f"🔧 Validating {len(skills_to_validate)} skills via API: {skills_to_validate}"
            )

            if __event_emitter__:
                await self.emit_event(
                    {
                        "type": "status",
                        "data": {
                            "description": "🔧 Validating Skills...",
                            "done": False,
                            "hidden": True,
                        },
                    },
                    __event_emitter__,
                )

            try:
                from anthropic import AsyncAnthropic

                base_url = self.valves.ANTHROPIC_BASE_URL.strip() or None
                client = AsyncAnthropic(api_key=api_key, **({"base_url": base_url} if base_url else {}))

                # Fetch all available skills
                available_skills = {}

                # Fetch Anthropic skills
                try:
                    anthropic_skills = await client.beta.skills.list(
                        source="anthropic", betas=["skills-2025-10-02"]
                    )
                    for skill in anthropic_skills.data:
                        # Store by both id and display_title for flexible matching
                        available_skills[skill.id] = {
                            "id": skill.id,
                            "type": "anthropic",
                            "source": "anthropic",
                            "display_title": getattr(skill, "display_title", skill.id),
                            "latest_version": getattr(
                                skill, "latest_version", "latest"
                            ),
                        }
                        # Also index by lowercase for case-insensitive matching
                        available_skills[skill.id.lower()] = available_skills[skill.id]
                except Exception as e:
                    logger.warning(f"Failed to fetch Anthropic skills: {e}")

                # Fetch custom skills
                try:
                    custom_skills = await client.beta.skills.list(
                        source="custom", betas=["skills-2025-10-02"]
                    )
                    for skill in custom_skills.data:
                        available_skills[skill.id] = {
                            "id": skill.id,
                            "type": "custom",
                            "source": "custom",
                            "display_title": getattr(skill, "display_title", skill.id),
                            "latest_version": getattr(
                                skill, "latest_version", "latest"
                            ),
                        }
                except Exception as e:
                    logger.warning(f"Failed to fetch custom skills: {e}")

                logger.debug(f"🔧 Found {len(available_skills)} available skills")

                # Validate each skill
                for skill_name in skills_to_validate:
                    skill_lower = skill_name.lower().strip()

                    # Try exact match first
                    if skill_name in available_skills:
                        cache[skill_name] = available_skills[skill_name]
                        logger.debug(f"✓ Validated skill '{skill_name}' (exact match)")
                    # Try lowercase match
                    elif skill_lower in available_skills:
                        cache[skill_name] = available_skills[skill_lower]
                        logger.debug(
                            f"✓ Validated skill '{skill_name}' (case-insensitive match)"
                        )
                    else:
                        # Mark as invalid
                        cache[skill_name] = None
                        logger.warning(
                            f"✗ Invalid skill '{skill_name}' - not found in available skills"
                        )

            except Exception as e:
                logger.error(f"Failed to validate skills: {e}")
                # Mark all as failed validation
                for skill_name in skills_to_validate:
                    cache[skill_name] = None

        # Build the validated skills list
        validated_skills = []
        invalid_skills = []

        for skill_name in skill_names:
            skill_info = cache.get(skill_name)
            if skill_info:
                validated_skills.append(
                    {
                        "type": skill_info["type"],
                        "skill_id": skill_info["id"],
                        "version": "latest",
                    }
                )
            else:
                invalid_skills.append(skill_name)

        if invalid_skills and __event_emitter__:
            await self.emit_event(
                {
                    "type": "notification",
                    "data": {
                        "type": "warning",
                        "content": f"⚠️ Invalid skills ignored: {', '.join(invalid_skills)}",
                    },
                },
                __event_emitter__,
            )

        logger.debug(f"🔧 Returning {len(validated_skills)} validated skills")
        return validated_skills

    # =========================================================================
    # METADATA PERSISTENCE SYSTEM
    # Stores metadata in empty markdown links that OpenWebUI doesn't render
    #
    # NEW COMPACT FORMAT for message-level file tracking:
    # [](anthropic:m=1:fid1,fid2|3:fid3;p=1:pid1|2:pid2;c=container_xyz;u=file.csv:aid1,doc.pdf:aid2)
    #
    # Keys:
    #   m = Files API: msg_idx:file_id,file_id|msg_idx:file_id
    #   p = Native PDFs: msg_idx:openwebui_id,openwebui_id|msg_idx:openwebui_id
    #   c = Container ID (single, reused across conversation)
    #   u = Uploaded file mapping: filename:anthropic_id,filename:anthropic_id
    #
    # CRITICAL: Only the LAST assistant message persists between requests.
    # We must accumulate ALL state in EVERY response.
    # =========================================================================

    METADATA_PATTERN = re.compile(r"\[\]\(anthropic:([^)]+)\)")

    def _create_metadata_marker(self, id: str, value: str, messagenum: int = 0) -> str:
        # URL-encode to handle special characters
        encoded_value = quote(value, safe="")
        return f" [](anthropic:{messagenum}:{id}:{encoded_value}) "

    def _extract_metadata_marker_from_message(self, message) -> List[str]:
        """
        Extract Anthropic metadata from the LAST assistant message in conversation.
        """
        metadata: List[str] = []
        if not isinstance(message, dict):
            return metadata
        if message.get("role") == "assistant":
            text = None
            content = message.get("content")
            if isinstance(content, list):
                # Join all text blocks for searching, but also update blocks in-place
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        block_text = block.get("text", "")
                        matches = self.METADATA_PATTERN.findall(block_text)
                        for match in matches:
                            metadata.append(match)
                        # Remove all metadata markers from this block
                        cleaned_text = self.METADATA_PATTERN.sub("", block_text)
                        block["text"] = cleaned_text
            elif isinstance(content, str):
                matches = self.METADATA_PATTERN.findall(content)
                for match in matches:
                    metadata.append(match)
                # Remove all metadata markers from the string
                message["content"] = self.METADATA_PATTERN.sub("", content)
        return metadata
