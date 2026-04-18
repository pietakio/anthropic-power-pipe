"""
Tests for prompt cache control marker placement.
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from anthropic_power_pipe import Pipe


@pytest.fixture
def pipe():
    p = Pipe()
    # Use default 5-minute TTL
    p.valves.CACHE_CONTROL = "cache tools array, system prompt and messages"
    p.valves.CACHE_TTL = "5 minutes"
    return p


def make_payload(tools=None, system=None, messages=None):
    return {
        "tools": tools or [],
        "system": system or [],
        "messages": messages or [],
    }


class TestCacheControlMarkerPlacement:
    def test_cache_marker_on_last_non_deferred_tool(self, pipe):
        payload = make_payload(tools=[
            {"name": "tool_a"},
            {"name": "tool_b"},
        ])
        pipe._apply_cache_control(payload)
        assert "cache_control" not in payload["tools"][0]
        assert payload["tools"][1].get("cache_control") == {"type": "ephemeral"}

    def test_deferred_tool_skipped_for_cache_marker(self, pipe):
        payload = make_payload(tools=[
            {"name": "tool_a"},
            {"name": "tool_b", "defer_loading": True},
        ])
        pipe._apply_cache_control(payload)
        assert payload["tools"][0].get("cache_control") == {"type": "ephemeral"}
        assert "cache_control" not in payload["tools"][1]

    def test_system_prompt_gets_cache_marker(self, pipe):
        payload = make_payload(system=[{"type": "text", "text": "You are helpful."}])
        pipe._apply_cache_control(payload)
        assert payload["system"][0].get("cache_control") == {"type": "ephemeral"}

    def test_empty_system_block_skipped(self, pipe):
        payload = make_payload(system=[
            {"type": "text", "text": ""},
            {"type": "text", "text": "Non-empty"},
        ])
        pipe._apply_cache_control(payload)
        assert "cache_control" not in payload["system"][0]
        assert payload["system"][1].get("cache_control") == {"type": "ephemeral"}

    def test_last_stable_user_message_gets_cache_marker(self, pipe):
        payload = make_payload(messages=[
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Hi"}]},
            {"role": "user", "content": [{"type": "text", "text": "How are you?"}]},
        ])
        pipe._apply_cache_control(payload)
        last_user = payload["messages"][-1]
        assert last_user["content"][-1].get("cache_control") == {"type": "ephemeral"}

    def test_thinking_blocks_never_get_cache_marker(self, pipe):
        payload = make_payload(messages=[
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "...", "signature": "sig"},
                {"type": "text", "text": "answer"},
            ]},
            {"role": "user", "content": [{"type": "text", "text": "ok"}]},
        ])
        pipe._apply_cache_control(payload)
        thinking_block = payload["messages"][0]["content"][0]
        assert "cache_control" not in thinking_block

    def test_existing_cache_markers_stripped_before_reapplying(self, pipe):
        payload = make_payload(
            tools=[{"name": "tool_a", "cache_control": {"type": "ephemeral"}}],
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": "Hi", "cache_control": {"type": "ephemeral"}}
                ]}
            ]
        )
        pipe._apply_cache_control(payload)
        # After stripping and reapplying, only the last tool should have it
        assert payload["tools"][0].get("cache_control") == {"type": "ephemeral"}
        # User message should also still have it (it's the last stable message)
        assert payload["messages"][-1]["content"][-1].get("cache_control") == {"type": "ephemeral"}

    def test_cache_disabled_adds_no_markers(self, pipe):
        pipe.valves.CACHE_CONTROL = "cache disabled"
        payload = make_payload(
            tools=[{"name": "tool_a"}],
            system=[{"type": "text", "text": "System"}],
            messages=[{"role": "user", "content": [{"type": "text", "text": "Hi"}]}],
        )
        pipe._apply_cache_control(payload)
        assert "cache_control" not in payload["tools"][0]
        assert "cache_control" not in payload["system"][0]
        assert "cache_control" not in payload["messages"][0]["content"][0]

    def test_1hour_ttl_adds_ttl_field(self, pipe):
        pipe.valves.CACHE_TTL = "1 hour"
        payload = make_payload(tools=[{"name": "tool_a"}])
        pipe._apply_cache_control(payload)
        marker = payload["tools"][0].get("cache_control", {})
        assert marker.get("ttl") == "1h"
