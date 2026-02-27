"""Configuration loader for sip-voice-transport."""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ProviderConfig:
    """Credentials for a telephony provider."""
    api_key: str = ""
    account_sid: str = ""    # Twilio-specific
    auth_token: str = ""     # Twilio-specific
    app_id: str = ""         # Telnyx TeXML app ID


@dataclass
class SipConfig:
    """Top-level configuration for the SIP voice transport."""
    provider: str = "telnyx"                             # "telnyx" or "twilio"
    telnyx: ProviderConfig = field(default_factory=ProviderConfig)
    twilio: ProviderConfig = field(default_factory=ProviderConfig)
    server_host: str = "127.0.0.1"
    server_port: int = 8000
    public_ws_url: str = ""                              # wss://... URL
    log_level: str = "INFO"

    @classmethod
    def from_yaml(cls, path: str = "~/.vibecode/sip_config.yaml") -> "SipConfig":
        """Load config from YAML file, with environment variable expansion.

        Environment variables in the format ${VAR_NAME} are expanded.
        """
        config_path = Path(path).expanduser()
        if not config_path.exists():
            return cls()

        with open(config_path) as f:
            raw = f.read()

        # Expand ${ENV_VAR} references
        def expand_env(match):
            var_name = match.group(1)
            return os.environ.get(var_name, match.group(0))

        raw = re.sub(r'\$\{(\w+)\}', expand_env, raw)
        data = yaml.safe_load(raw) or {}

        telnyx_data = data.get("provider_config", {}).get("telnyx", {})
        twilio_data = data.get("provider_config", {}).get("twilio", {})

        return cls(
            provider=data.get("provider", "telnyx"),
            telnyx=ProviderConfig(
                api_key=telnyx_data.get("api_key", os.environ.get("TELNYX_API_KEY", "")),
                app_id=telnyx_data.get("app_id", ""),
            ),
            twilio=ProviderConfig(
                account_sid=twilio_data.get("account_sid", os.environ.get("TWILIO_ACCOUNT_SID", "")),
                auth_token=twilio_data.get("auth_token", os.environ.get("TWILIO_AUTH_TOKEN", "")),
            ),
            server_host=data.get("server_host", "127.0.0.1"),
            server_port=data.get("server_port", 8000),
            public_ws_url=data.get("public_ws_url", ""),
            log_level=data.get("log_level", "INFO"),
        )
