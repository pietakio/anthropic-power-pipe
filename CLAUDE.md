# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Anthropic Power Pipe is an OpenWebUI Function (pipe) that provides comprehensive Anthropic Claude API integration. It's a single-file Python module (~6000 lines) that implements Claude's full feature set for Open WebUI.

**Requirements**: `pydantic>=2.0.0`, `anthropic>=0.75.0`

## Architecture

### Core Structure

The entire implementation lives in `anthropic_power_pipe.py`:

- **`Pipe` class** (line 253): Main OpenWebUI manifold pipe with:
  - `Valves`: Admin-level configuration (API key, caching, tool limits, context editing)
  - `UserValves`: Per-user settings (thinking toggle, effort level, skills, PDF handling)
  - `pipe()`: Main entry point (line 2693) orchestrating the request lifecycle

- **`PipeRequestContext`** (line 211): Request-scoped state for streaming and event emission
- **`PipeRenderStrategy`** (line 202): Per-request rendering toggles

### Request Flow

1. **Validation & Setup** - API key validation, tool resolution, model capability lookup
2. **Message Processing** - Convert OpenWebUI messages to Anthropic format, handle PDFs/images
3. **Payload Creation** - Build API request with caching, tools, thinking config
4. **Stream Processing** - Handle Anthropic SDK streaming events, accumulate response
5. **Tool Loop** - Execute tool calls, feed results back, repeat until complete

### Key Subsystems

- **Prompt Caching**: 4-level cache (tools, system, messages) with RAG/Memory awareness
- **Context Editing**: Auto-clear old tool results and thinking blocks based on token thresholds
- **Tool Search**: BM25/Regex deferred loading for large toolsets
- **Native PDF Upload**: Visual PDF analysis bypassing RAG extraction
- **Code Execution**: Files API integration with persistent container state
- **Programmatic Tool Calling**: Tools callable from within code execution

### Model Capabilities

Capabilities are fetched from the Anthropic `/v1/models` API and cached for 24 hours. Model-specific overrides in `MODEL_CAPABILITY_OVERRIDES` handle flags not available from the API (1M context, dynamic filtering, fast mode).

## Development Notes

- The pipe integrates with OpenWebUI's internal modules (Files, Storage, Models, Skills) via try/except imports for graceful degradation
- Regex patterns are pre-compiled at module level (lines 82-138) for performance
- Thinking blocks must never be removed from assistant messages during tool loops (per Anthropic API requirements)
- The `_api_capabilities_cache` is class-level to share across requests
