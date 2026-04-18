"""
Tests for tool call history reconstruction from OpenWebUI HTML format.
Covers _parse_tool_history_from_html and the full message conversion pipeline.
"""
import html
import json
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from anthropic_power_pipe import Pipe


@pytest.fixture
def pipe():
    return Pipe()


def make_tool_html(tool_id, name, arguments, result=None, done=True):
    """Helper to generate OpenWebUI tool_calls HTML."""
    done_str = "true" if done else "false"
    args_escaped = html.escape(json.dumps(arguments) if isinstance(arguments, dict) else arguments)
    tag = f'<details type="tool_calls" done="{done_str}" id="{tool_id}" name="{name}" arguments="{args_escaped}"'
    if result is not None:
        result_str = result if isinstance(result, str) else json.dumps(result)
        tag += f' result="{html.escape(result_str)}"'
    tag += ' files="" embeds="">\n<summary>Tool Executed</summary>\n</details>'
    return tag


# ---------------------------------------------------------------------------
# _parse_tool_history_from_html
# ---------------------------------------------------------------------------

class TestParseToolHistoryFromHtml:
    def test_no_tool_calls_returns_empty(self, pipe):
        tool_use, tool_result, before, after = pipe._parse_tool_history_from_html("Just plain text.")
        assert tool_use == []
        assert tool_result == []
        assert before == ""
        assert after == "Just plain text."

    def test_single_tool_with_result(self, pipe):
        html_block = make_tool_html("toolu_01", "list_files", {}, result={"files": ["a.md"]})
        text = f"Let me check.\n{html_block}\nHere are the files."
        tool_use, tool_result, before, after = pipe._parse_tool_history_from_html(text)

        assert len(tool_use) == 1
        assert tool_use[0] == {"type": "tool_use", "id": "toolu_01", "name": "list_files", "input": {}}

        assert len(tool_result) == 1
        assert tool_result[0]["type"] == "tool_result"
        assert tool_result[0]["tool_use_id"] == "toolu_01"
        assert json.loads(tool_result[0]["content"]) == {"files": ["a.md"]}

        assert before.strip() == "Let me check."
        assert after.strip() == "Here are the files."

    def test_single_tool_without_result(self, pipe):
        html_block = make_tool_html("toolu_01", "search", {"query": "test"}, result=None, done=False)
        tool_use, tool_result, before, after = pipe._parse_tool_history_from_html(html_block)

        assert len(tool_use) == 1
        assert tool_use[0]["input"] == {"query": "test"}
        assert tool_result == []  # No result attribute

    def test_multiple_tools_batched(self, pipe):
        t1 = make_tool_html("toolu_01", "read_file", {"path": "/a.txt"}, result="contents of a")
        t2 = make_tool_html("toolu_02", "read_file", {"path": "/b.txt"}, result="contents of b")
        text = f"Reading both files.\n{t1}\n{t2}\nDone."
        tool_use, tool_result, before, after = pipe._parse_tool_history_from_html(text)

        assert len(tool_use) == 2
        assert tool_use[0]["id"] == "toolu_01"
        assert tool_use[1]["id"] == "toolu_02"
        assert len(tool_result) == 2
        assert before.strip() == "Reading both files."
        assert after.strip() == "Done."

    def test_json_arguments_parsed(self, pipe):
        args = {"query": "python async", "max_results": 5}
        html_block = make_tool_html("toolu_01", "web_search", args, result="results")
        tool_use, _, _, _ = pipe._parse_tool_history_from_html(html_block)
        assert tool_use[0]["input"] == args

    def test_html_entities_in_result_unescaped(self, pipe):
        result = {"key": "value with <tags> & \"quotes\""}
        html_block = make_tool_html("toolu_01", "tool", {}, result=result)
        _, tool_result, _, _ = pipe._parse_tool_history_from_html(html_block)
        parsed = json.loads(tool_result[0]["content"])
        assert parsed == result

    def test_empty_arguments_becomes_empty_dict(self, pipe):
        html_block = make_tool_html("toolu_01", "no_args_tool", "", result="ok")
        tool_use, _, _, _ = pipe._parse_tool_history_from_html(html_block)
        assert tool_use[0]["input"] == {}


# ---------------------------------------------------------------------------
# _convert_messages_to_claude_format — tool history reconstruction
# ---------------------------------------------------------------------------

class TestConvertMessagesToolHistory:
    def test_assistant_with_tool_call_expands_to_three_messages(self, pipe):
        tool_html = make_tool_html("toolu_01", "list_files", {}, result='{"files": []}')
        raw = [
            {"role": "user", "content": "List my files."},
            {"role": "assistant", "content": f"Sure!\n{tool_html}\nHere you go."},
        ]
        _, msgs, _ = pipe._convert_messages_to_claude_format(raw)

        # Should produce: user, assistant(tool_use), user(tool_result), assistant(text_after)
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"
        assert any(b["type"] == "tool_use" for b in msgs[1]["content"])
        assert msgs[2]["role"] == "user"
        assert any(b["type"] == "tool_result" for b in msgs[2]["content"])
        assert msgs[3]["role"] == "assistant"
        assert msgs[3]["content"][0]["type"] == "text"
        assert "Here you go" in msgs[3]["content"][0]["text"]

    def test_text_before_tool_is_preserved(self, pipe):
        tool_html = make_tool_html("toolu_01", "tool", {}, result="result")
        raw = [{"role": "assistant", "content": f"Pre-text.\n{tool_html}\nPost-text."}]
        _, msgs, _ = pipe._convert_messages_to_claude_format(raw)

        assistant_blocks = msgs[0]["content"]
        text_blocks = [b for b in assistant_blocks if b["type"] == "text"]
        assert any("Pre-text" in b["text"] for b in text_blocks)

    def test_tool_use_id_matches_tool_result(self, pipe):
        tool_html = make_tool_html("toolu_abc123", "my_tool", {}, result="ok")
        raw = [{"role": "assistant", "content": tool_html}]
        _, msgs, _ = pipe._convert_messages_to_claude_format(raw)

        tool_use_id = msgs[0]["content"][0]["id"]
        tool_result_id = msgs[1]["content"][0]["tool_use_id"]
        assert tool_use_id == tool_result_id == "toolu_abc123"

    def test_multiple_tools_produce_single_user_message(self, pipe):
        t1 = make_tool_html("toolu_01", "tool_a", {}, result="r1")
        t2 = make_tool_html("toolu_02", "tool_b", {}, result="r2")
        raw = [{"role": "assistant", "content": f"Using both.\n{t1}\n{t2}\nDone."}]
        _, msgs, _ = pipe._convert_messages_to_claude_format(raw)

        # assistant(tool_use x2), user(tool_result x2), assistant(text)
        assert msgs[0]["role"] == "assistant"
        assert len([b for b in msgs[0]["content"] if b["type"] == "tool_use"]) == 2
        assert msgs[1]["role"] == "user"
        assert len(msgs[1]["content"]) == 2

    def test_message_without_tool_calls_unchanged(self, pipe):
        raw = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        _, msgs, _ = pipe._convert_messages_to_claude_format(raw)
        assert len(msgs) == 2
        assert msgs[1]["content"][0]["text"] == "Hi there!"


# ---------------------------------------------------------------------------
# Thinking block handling
# ---------------------------------------------------------------------------

REASONING_HTML = (
    '<details type="reasoning" done="true" duration="5">\n'
    '<summary>Thought for 5 seconds</summary>\n'
    '> I should check the files first.\n'
    '</details>\n'
)


class TestThinkingBlockHandling:
    def test_thinking_stripped_by_default(self, pipe):
        raw = [{"role": "assistant", "content": f"{REASONING_HTML}Here is my answer."}]
        _, msgs, _ = pipe._convert_messages_to_claude_format(raw, preserve_thinking_in_history=False)
        text = msgs[0]["content"][0]["text"]
        assert "reasoning" not in text
        assert "Here is my answer" in text

    def test_thinking_preserved_when_enabled(self, pipe):
        raw = [{"role": "assistant", "content": f"{REASONING_HTML}Here is my answer."}]
        _, msgs, _ = pipe._convert_messages_to_claude_format(raw, preserve_thinking_in_history=True)
        text = msgs[0]["content"][0]["text"]
        assert 'type="reasoning"' in text
        assert "Here is my answer" in text

    def test_thinking_stripped_inside_tool_history_by_default(self, pipe):
        tool_html = make_tool_html("toolu_01", "tool", {}, result="result")
        content = f"{REASONING_HTML}Pre.\n{tool_html}\n{REASONING_HTML}Post."
        raw = [{"role": "assistant", "content": content}]
        _, msgs, _ = pipe._convert_messages_to_claude_format(raw, preserve_thinking_in_history=False)

        all_text = " ".join(
            b.get("text", "") for m in msgs for b in m.get("content", []) if isinstance(b, dict)
        )
        assert "reasoning" not in all_text

    def test_thinking_preserved_inside_tool_history(self, pipe):
        tool_html = make_tool_html("toolu_01", "tool", {}, result="result")
        content = f"{REASONING_HTML}Pre.\n{tool_html}\nPost."
        raw = [{"role": "assistant", "content": content}]
        _, msgs, _ = pipe._convert_messages_to_claude_format(raw, preserve_thinking_in_history=True)

        all_text = " ".join(
            b.get("text", "") for m in msgs for b in m.get("content", []) if isinstance(b, dict)
        )
        assert 'type="reasoning"' in all_text
