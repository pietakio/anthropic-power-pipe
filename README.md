# 🚀 Anthropic Power Pipe for Open WebUI

> **Claude Feature parity for Open WebUI - streaming, prompt caching, tool use,
> extended thinking, code execution with file support, programmatic tool
> calling, agent skills, and more.**

---

## 📖 Overview

A comprehensive Anthropic API integration for OpenWeb UI built on the
**Anthropic Python SDK**. Claude API feature-complete while preserving
compatibility with native OpenWeb UI tooling. Let Claude orchestrate complex
multi-task workflows for your use case, including:

- *"Grab my Jira tasks and send a summary on Slack"* — token-efficient single
  request complete with programmatic tool calling and parallel execution.
- *"Check out this finance report - Extract the important information and
  distill it down into a nice PowerPoint Presentation"* — code execution with
  Files API Upload and Download Support.
- *"What's the meaning of life?"* — Extended Thinking consumes up to 64k Tokens
  while web search queries the internet before responding with the optimal
  answer. Interleaved Thinking permits Claude to contemplate its own reasoning
  path while formulating that response.

### 🎯 Key Highlights

| Feature | Description |
|---------|-------------|
| 🔧 **Programmatic Tool Calling** | Claude orchestrates tools through code execution — multi-tool workflows in one go |
| ⚡ **Parallel Execution** | Independent tools execute simultaneously |
| 💾 **Prompt Caching** | 4-level cache for system prompts, tools, and messages; compatible with RAG & Memory |
| 🧠 **Extended Thinking** | Classic budget, adaptive, and interleaved thinking with live streaming |
| 💻 **Code Execution** | Sandboxed Python with persistent container state, file upload/download |
| 🌐 **Web Search & Fetch** | Dynamic filtering, inline citations, URL content analysis |
| 🔍 **Tool Search** | BM25/Regex deferred loading for large toolsets |
| 🧹 **Context Editing** | Auto-clear old tool results and thinking blocks |
| 📊 **1M Token Context** | Extended context for Opus 4.6, Sonnet 4/4.5 |
| 🎨 **Agent Skills** | pptx, xlsx, docx, pdf & custom skills via Files API |

## ✨ Features

### Core

| Feature | Description |
|---------|-------------|
| **Anthropic Python SDK** | Official SDK for streaming and message accumulation |
| **Model Auto-Discovery** | Fetches available models from your API key |
| **Streaming** | Fine-grained tool streaming with eager input streaming (GA) |
| **Tool Call Loop** | Multiple tools per response cycle with configurable limit |
| **Parallel Tool Execution** | Local tools run concurrently |
| **Error Handling** | Retry logic for rate limits and transient errors |
| **Task Support** | Title, tag, and follow-up generation |
| **Notes & Channels** | Full OpenWebUI integration |
| **Token Count** | Toggleable context window progress bar per response |

### Claude API Features

| Feature | Description |
|---------|-------------|
| **Vision** | Image analysis (JPEG, PNG, GIF, WebP) |
| **Native PDF Upload** | Visual PDF analysis bypassing RAG extraction |
| **Citations** | Correctly positioned streaming citations from web search |
| **Extended Thinking** | Classic `budget_tokens` and adaptive thinking (Opus 4.6) |
| **Interleaved Thinking** | Claude thinks between tool calls |
| **Live Thinking Streaming** | Real-time streaming, collapsing into `<details>` blocks |
| **Files API (standalone)** | Native file handling for code execution |
| **Web Search** | `web_search_20260209` with dynamic filtering and location-aware results |
| **Web Fetch** | URL content retrieval and analysis |
| **Code Execution** | Persistent container state, unified display (code + tool calls + output) |
| **Programmatic Tool Calling** | Tools callable from within code execution; multi-tool coordination |
| **Tool Search** | BM25/Regex deferred loading for hundreds/thousands of tools |
| **Context Editing** | Auto-clear tool results & thinking blocks with token-count triggers |
| **Agent Skills** | pptx, xlsx, docx, pdf & custom skills, validated via List Skills API |
| **Data Residency** | `inference_geo` parameter (global or US) |
| **Fast Mode** | Up to 2.5× faster output for Opus 4.6 |
| **1M Token Context** | Opus 4.6, Sonnet 4/4.5 (Tier 4 required) |
| **Effort Parameter** | low / medium / high / max (GA) |
| **Prompt Caching** | 4-level cache: tools, system prompt, messages; RAG/Memory aware |

## 📦 Installation

<!-- FIXME: Uncomment after posting to openwebui.com, please. *sigh*
### Option 1: Install from OpenWebUI Community (Recommended)

| Component | Link |
|-----------|------|
| **Main Pipe** | [anthropic_pipe](https://openwebui.com/f/podden/anthropic_pipe) |
| **Thinking Toggle** | [anthropic_pipe_thinking_toggle](https://openwebui.com/f/podden/anthropic_pipe_thinking_toggle) |
| **Companion Filter** | [anthropic_manifold_companion](https://openwebui.com/f/podden/anthropic_manifold_companion) |
-->

### Option 1: Manual Installation

1. On GitHub:
   1. Browse to the [raw source code for Anthropic Power
      Pipe](https://raw.githubusercontent.com/pietakio/anthropic-power-pipe/refs/heads/main/anthropic_power_pipe.py).
   1. Copy the contents to your system clipboard (e.g., `<Ctrl-a> <Ctrl-c>`).
1. In OpenWeb UI:
   1. **Admin Settings** → **Functions** → **"+ New Function"**
   1. Paste the contents from your system clipboard (e.g., `<Ctrl-v>`).
   1. Define an arbitrary name, ID, and description.

## 🔧 Configuration

1. **Set API Key** in the pipe's Valves
1. **Configure Models** (Admin Settings → Models):
   - Activate Thinking and Companion Filter for each Claude model or globally
   - Activate `web_search` and `code_interpreter` capabilities
   - Optional: Add usage to see token consumption
   - Set **Function Calling** to `Native` in Advanced Parameters
   - **Set Valves and UserValves.** Experiment with the Settings to find values
     to your liking.
1. **Start chatting!**

### Valves (Global / Admin Settings)

| Valve | Default | Description |
|-------|---------|-------------|
| `ANTHROPIC_API_KEY` | — | Your Anthropic API key (required) |
| `ENABLE_FAST_MODE` | `false` | Fast Mode for Opus 4.6 (up to 2.5× faster, higher cost) |
| `ENABLE_1M_CONTEXT` | `false` | 1M token context window (Tier 4 required) |
| `ENABLE_INTERLEAVED_THINKING` | `true` | Claude thinks between tool calls |
| `WEB_SEARCH` | `true` | Enable web search tool |
| `WEB_FETCH` | `true` | Enable web fetch tool (URL content retrieval) |
| `MAX_TOOL_CALLS` | `15` | Max tool execution loops per request (1–50) |
| `MAX_RETRIES` | `3` | Max retries for failed requests (0–50) |
| `CACHE_CONTROL` | `cache tools array, system prompt and messages` | Prompt caching scope (see below) |
| `ENABLE_PROGRAMMATIC_TOOL_CALLING` | `false` | Tools callable from within code execution |
| `ENABLE_TOOL_SEARCH` | `false` | BM25/Regex tool search for large toolsets |
| `TOOL_SEARCH_TYPE` | `bm25` | `regex` or `bm25` |
| `TOOL_SEARCH_MAX_DESCRIPTION_LENGTH` | `100` | Tools with longer JSON defs are deferred |
| `TOOL_SEARCH_EXCLUDE_TOOLS` | `[web_search, web_fetch, ...]` | Always-loaded tools |
| `CONTEXT_EDITING_STRATEGY` | `none` | `none` / `clear_tool_results` / `clear_thinking` / `clear_both` |
| `CONTEXT_EDITING_THINKING_KEEP` | `5` | Thinking blocks to keep |
| `CONTEXT_EDITING_TOOL_TRIGGER` | `50000` | Token threshold for clearing tool results |
| `CONTEXT_EDITING_TOOL_KEEP` | `5` | Recent tool results to preserve |
| `CONTEXT_EDITING_TOOL_CLEAR_AT_LEAST` | `10000` | Minimum tokens to clear |
| `CONTEXT_EDITING_TOOL_CLEAR_TOOL_INPUT` | `false` | Also clear tool input parameters |
| `DATA_RESIDENCY` | `global` | `global` or `us` (1.1× cost for US) |
| `WEB_SEARCH_USER_*` | — | Default location for web searches (city, region, country, timezone) |

#### Cache Control Options

| Option | Description |
|--------|-------------|
| `cache disabled` | No caching |
| `cache tools array only` | Cache tool definitions |
| `cache tools array and system prompt` | Cache tools + system prompt |
| `cache tools array, system prompt and messages` | Full caching (recommended) |

> 💡 **RAG & Memory**: The pipe is aware of your settings and your intention,
> for example if you're attaching a PDF document with full context mode with
> NATIVE_PDF_UPLOAD active, it removed the RAG Promt and Sources entirely. If
> there's additional knowledge added, it strips the PDF RAG sources from RAG and
> moves the caching point to the previous messages as the last message is now
> always changing. It also extracts Memories from the System Promt and add them
> to the last user message when the Memory System is active to avoid cache
> misses. If you're encountering problems, feel free to open an issue!

### UserValves (Per-User Settings)

| Valve | Default | Range | Description |
|-------|---------|-------|-------------|
| `ENABLE_THINKING` | `false` | — | Enable Extended Thinking |
| `THINKING_BUDGET_TOKENS` | `8192` | 1024–64000 | Token budget for thinking |
| `EFFORT` | `high` | low/medium/high/max | Effort level (also controllable via OpenWebUI's `reasoning_effort`) |
| `USE_PDF_NATIVE_UPLOAD` | `true` | — | Visual PDF analysis instead of RAG extraction |
| `SHOW_TOKEN_COUNT` | `false` | — | Show context window progress bar |
| `WEB_SEARCH_MAX_USES` | `5` | 1–20 | Max web searches per turn |
| `WEB_FETCH_MAX_USES` | `5` | 1–20 | Max web fetch requests per turn |
| `WEB_SEARCH_USER_*` | — | — | Override global location settings |
| `SKILLS` | `[]` | — | Skills to activate (e.g., `pptx`, `xlsx`, `docx`, `pdf`, or custom IDs) |
| `DEBUG_MODE` | `false` | — | Logs some internal and external parameters as citation to send me for debugging ;) |

### Toggle Filters & Companion

| Filter | Purpose |
|--------|---------|
| **Thinking Toggle** | 🧠 Enable thinking for the next message |
| **Companion Filter** | 🔀 Intercepts OpenWebUI's built-in `web_search` / `code_interpreter` buttons and routes them to native Anthropic tools |

## 🤝 Contributing

Bug reports and feature requests are welcome! Feel free to [open an
issue](https://github.com/Podden/openwebui_anthropic_api_manifold_pipe/issues)
if you encounter any problems.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file
for details.

## 🙏 Acknowledgments

- Built for [Open WebUI](https://github.com/open-webui/open-webui)
- Powered by [Anthropic Claude](https://www.anthropic.com/)
- Huge shout-outs to
- [Podden](https://github.com/Podden/openwebui_anthropic_api_manifold_pipe),
  Balaxxe, and nbellochi for their original OpenWeb UI pipes.
