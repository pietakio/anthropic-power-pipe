# Bug: Tool call history lost on subsequent turns

## Summary

When Claude makes a tool call, the tool interaction works correctly on that turn. However, on **subsequent turns**, the tool call evidence is stripped from the conversation history sent to the Anthropic API. Claude can no longer see that it made the tool call, leading it to conclude it hallucinated the action — and often redundantly re-executing it.

## Impact

- Claude believes it hallucinated successful tool calls and re-executes them, wasting tokens
- Claude cannot build on previous tool results across turns
- Undermines Claude's ability to maintain coherent multi-step workflows
- Particularly damaging for Obsidian vault workflows where Claude is instructed to avoid redundant reads

## Root cause

Open WebUI stores assistant messages (including tool calls) as flat HTML text using `<details>` blocks. The pipe correctly handles tool calls **during the live turn** (the tool loop works), but when reconstructing conversation history for subsequent API requests, the `<details type="tool_calls">` HTML is **stripped rather than parsed back into structured Anthropic API blocks**.

## Evidence

### What Open WebUI stores in the body (correct data is present)

The body's message history contains tool call data embedded in HTML attributes:

```html
<details type="tool_calls" done="true" 
  id="toolu_01RWubFTMZtjNUDHgwDnELi2" 
  name="tool_list_directory_post" 
  arguments="" 
  result="{&quot;dirs&quot;: [&quot;Solace&quot;, &quot;Threshold&quot;, &quot;Verse&quot;, &quot;Workshop&quot;], &quot;files&quot;: [&quot;Welcome.md&quot;, &quot;hello-from-claude.md&quot;]}" 
  files="" embeds="">
<summary>Tool Executed</summary>
</details>
```

All necessary fields are present: `id`, `name`, `arguments`, `result`.

### What the pipe sends to Anthropic (tool call is gone)

The payload's messages array contains only a flat text block — the tool call HTML has been removed and **not reconstructed** as structured blocks:

```json
{
  "role": "assistant",
  "content": [
    {
      "type": "text",
      "text": "...Sure thing! Let me take a look....Here's what's in the vault root:\n\nFolders:\n- Solace/..."
    }
  ]
}
```

The entire tool interaction has vanished from the API payload.

### What Anthropic's API expects

The Anthropic Messages API represents tool interactions as structured content blocks across multiple messages:

```json
{
  "role": "assistant",
  "content": [
    {"type": "text", "text": "Sure thing! Let me take a look."},
    {
      "type": "tool_use",
      "id": "toolu_01RWubFTMZtjNUDHgwDnELi2",
      "name": "tool_list_directory_post",
      "input": {}
    }
  ]
},
{
  "role": "user",
  "content": [
    {
      "type": "tool_result",
      "tool_use_id": "toolu_01RWubFTMZtjNUDHgwDnELi2",
      "content": "{\"dirs\":[\"Solace\",\"Threshold\",\"Verse\",\"Workshop\"],\"files\":[\"Welcome.md\",\"hello-from-claude.md\"]}"
    }
  ]
},
{
  "role": "assistant",
  "content": [
    {"type": "text", "text": "Here's what's in the vault root:\n\nFolders:\n- Solace/..."}
  ]
}
```

## Suggested fix

When the pipe transforms the body's message history into the Anthropic API payload, it should:

1. **Parse `<details type="tool_calls">` blocks** from assistant message text
2. **Split the assistant message** at tool call boundaries into: text before → `tool_use` block → (interleaved thinking if present) → text after
3. **Insert a `tool_result` message** (with `role: "user"`) between the assistant's `tool_use` and subsequent content, using the `result` attribute from the HTML
4. **Reconstruct `arguments`** into the `input` field of the `tool_use` block (parse from the `arguments` attribute)

The HTML attributes already contain everything needed: `id`, `name`, `arguments`, and `result`. It's a parsing/reconstruction task, not a data loss problem.

## How to reproduce

1. Start a conversation with Claude using the pipe
2. Ask Claude to make any tool call (e.g., "list files in my Obsidian vault")
3. After Claude responds with the tool result, send a follow-up message
4. Enable `DEBUG_MODE` and check the **Payload** source — look at the `messages` array
5. Observe that the assistant's previous tool call is absent from the structured payload

## Environment

- Pipe version: v0.8.12 (tested)
- Model: claude-opus-4-6
- Thinking: adaptive, effort high
- Context editing: disabled (set to `none`)
- Tools: MCP vault tools (Obsidian), OpenTerminal, OpenWebUI built-in tools

## Notes

- The `<details type="reasoning">` blocks have the same flat-text treatment, but thinking blocks are less critical since the API handles thinking history differently (auto-stripping previous thinking). Tool calls, however, **must** be structured for Claude to recognize them as confirmed actions.
- This issue affects all tool types, not just vault tools.
- The live tool loop within a single turn works correctly — this is specifically about **history reconstruction across turns**.
