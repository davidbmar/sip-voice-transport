"""Tests for SipConfig."""

from sip_voice_transport.config import SipConfig


class TestSipConfig:
    def test_defaults(self):
        config = SipConfig()
        assert config.provider == "telnyx"
        assert config.server_port == 8000
        assert config.server_host == "127.0.0.1"

    def test_from_yaml_missing_file(self, tmp_path):
        config = SipConfig.from_yaml(str(tmp_path / "nonexistent.yaml"))
        assert config.provider == "telnyx"

    def test_from_yaml_with_data(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
provider: twilio
server_host: "0.0.0.0"
server_port: 9000
public_ws_url: "wss://example.com/sip/media-stream"
log_level: DEBUG
provider_config:
  telnyx:
    api_key: "test-telnyx-key"
    app_id: "test-app-id"
  twilio:
    account_sid: "ACtest123"
    auth_token: "token456"
""")
        config = SipConfig.from_yaml(str(config_file))
        assert config.provider == "twilio"
        assert config.server_host == "0.0.0.0"
        assert config.server_port == 9000
        assert config.public_ws_url == "wss://example.com/sip/media-stream"
        assert config.telnyx.api_key == "test-telnyx-key"
        assert config.twilio.account_sid == "ACtest123"

    def test_env_var_expansion(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TEST_TELNYX_KEY", "env-key-value")
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
provider_config:
  telnyx:
    api_key: "${TEST_TELNYX_KEY}"
""")
        config = SipConfig.from_yaml(str(config_file))
        assert config.telnyx.api_key == "env-key-value"

    def test_env_var_fallback_to_env(self, monkeypatch):
        """When no YAML file, telnyx api_key falls back to TELNYX_API_KEY env var."""
        monkeypatch.setenv("TELNYX_API_KEY", "from-env")
        config = SipConfig.from_yaml("/nonexistent/path.yaml")
        # Falls back to defaults since file doesn't exist
        assert config.telnyx.api_key == ""
