"""
Tests for model capability detection, override logic, and effort clamping.
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from anthropic_power_pipe import Pipe


@pytest.fixture
def pipe():
    return Pipe()


# ---------------------------------------------------------------------------
# MODEL_CAPABILITY_OVERRIDES — prefix matching
# ---------------------------------------------------------------------------

class TestModelCapabilityOverrides:
    def test_exact_match_opus_4_7(self, pipe):
        info = pipe.get_model_info("claude-opus-4-7")
        assert info["requires_adaptive_only"] is True
        assert info["supports_effort_xhigh"] is True
        assert info["no_sampling_params"] is True

    def test_versioned_id_prefix_match_opus_4_7(self, pipe):
        # API may return date-suffixed IDs like claude-opus-4-7-20260415
        # Force cache miss so prefix matching in fallback is exercised
        Pipe._api_capabilities_cache = {}
        info = pipe.get_model_info("claude-opus-4-7-20260415")
        assert info["requires_adaptive_only"] is True
        assert info["supports_effort_xhigh"] is True
        assert info["no_sampling_params"] is True

    def test_exact_match_opus_4_6(self, pipe):
        Pipe._api_capabilities_cache = {}
        info = pipe.get_model_info("claude-opus-4-6")
        assert info["requires_adaptive_only"] is False
        assert info["supports_fast_mode"] is True
        assert info["supports_1m_context"] is True

    def test_versioned_id_prefix_match_opus_4_6(self, pipe):
        Pipe._api_capabilities_cache = {}
        info = pipe.get_model_info("claude-opus-4-6-20260101")
        assert info["supports_fast_mode"] is True
        assert info["requires_adaptive_only"] is False

    def test_unknown_model_returns_safe_defaults(self, pipe):
        Pipe._api_capabilities_cache = {}
        info = pipe.get_model_info("claude-unknown-model")
        assert info["requires_adaptive_only"] is False
        assert info["supports_effort_xhigh"] is False
        assert info["no_sampling_params"] is False

    def test_sonnet_4_6_gets_1m_context(self, pipe):
        Pipe._api_capabilities_cache = {}
        info = pipe.get_model_info("claude-sonnet-4-6")
        assert info["supports_1m_context"] is True
        assert info["requires_adaptive_only"] is False


# ---------------------------------------------------------------------------
# Effort clamping logic
# ---------------------------------------------------------------------------

def clamp_effort(requested, model_info):
    """Mirrors the clamping logic in _create_payload."""
    if requested == "xhigh":
        if model_info.get("supports_effort_xhigh"):
            return "xhigh"
        elif model_info.get("supports_effort_max"):
            return "max"
        else:
            return "high"
    elif requested == "max":
        if model_info.get("supports_effort_max"):
            return "max"
        else:
            return "high"
    return requested


class TestEffortClamping:
    def setup_method(self):
        Pipe._api_capabilities_cache = {}

    def test_xhigh_on_opus_4_7_stays_xhigh(self):
        info = Pipe.get_model_info("claude-opus-4-7")
        assert clamp_effort("xhigh", info) == "xhigh"

    def test_xhigh_on_opus_4_6_clamps_to_max(self):
        # Opus 4.6 supports_effort_max but not xhigh
        info = {**Pipe.MODEL_CAPABILITY_OVERRIDES.get("claude-opus-4-6", {}),
                "supports_effort_max": True, "supports_effort_xhigh": False}
        assert clamp_effort("xhigh", info) == "max"

    def test_xhigh_on_older_model_clamps_to_high(self):
        info = {"supports_effort_max": False, "supports_effort_xhigh": False}
        assert clamp_effort("xhigh", info) == "high"

    def test_max_on_opus_4_6_stays_max(self):
        info = {"supports_effort_max": True, "supports_effort_xhigh": False}
        assert clamp_effort("max", info) == "max"

    def test_max_on_older_model_clamps_to_high(self):
        info = {"supports_effort_max": False, "supports_effort_xhigh": False}
        assert clamp_effort("max", info) == "high"

    @pytest.mark.parametrize("level", ["low", "medium", "high"])
    def test_standard_levels_pass_through_unchanged(self, level):
        info = {"supports_effort_max": False, "supports_effort_xhigh": False}
        assert clamp_effort(level, info) == level
