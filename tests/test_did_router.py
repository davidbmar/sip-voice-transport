"""Tests for DIDRouter."""

import pytest
from sip_voice_transport.did_router import DIDRouter


@pytest.fixture
def sample_config(tmp_path):
    """Create a temporary config file."""
    config = tmp_path / "sip_config.yaml"
    config.write_text("""
dids:
  "+15125551234":
    user_id: david
    display_name: "Test Mac"
    llm_model: "llama3.2:7b"
    system_prompt: "You are a test assistant."
    tts_voice: "en-us-ryan-medium"
  "+15125555678":
    user_id: alice
    display_name: "Alice's Mac"
""")
    return str(config)


class TestDIDRouter:
    def test_lookup_existing(self, sample_config):
        router = DIDRouter(sample_config)
        config = router.lookup("+15125551234")
        assert config is not None
        assert config.user_id == "david"
        assert config.display_name == "Test Mac"

    def test_lookup_missing(self, sample_config):
        router = DIDRouter(sample_config)
        assert router.lookup("+19999999999") is None

    def test_normalize_formats(self, sample_config):
        router = DIDRouter(sample_config)
        # These should all find the same DID
        assert router.lookup("+1 (512) 555-1234") is not None
        assert router.lookup("15125551234") is not None
        assert router.lookup("+15125551234") is not None

    def test_dids_list(self, sample_config):
        router = DIDRouter(sample_config)
        assert len(router.dids) == 2

    def test_missing_config_file(self, tmp_path):
        router = DIDRouter(str(tmp_path / "nonexistent.yaml"))
        assert len(router.dids) == 0
