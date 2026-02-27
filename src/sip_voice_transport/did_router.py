"""DID routing — maps inbound phone numbers to user configurations.

Reads from a YAML config file. Each DID maps to a user, LLM model,
system prompt, and TTS voice configuration.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from sip_voice_transport.exceptions import DIDNotFoundError, DIDConfigError

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = "~/.vibecode/sip_config.yaml"


@dataclass
class DIDConfig:
    """Configuration for a single DID."""
    did: str                            # E.164 phone number (e.g., "+15125551234")
    user_id: str                        # Owner identifier
    display_name: str = ""              # Human-readable name
    llm_model: str = "llama3.2:7b"      # Ollama model name
    system_prompt: str = "You are a helpful voice assistant. Keep responses concise."
    tts_voice: str = "en-us-ryan-medium"  # Piper voice identifier
    extra: dict = field(default_factory=dict)


class DIDRouter:
    """Maps inbound DIDs to user configurations.

    Loads routing config from a YAML file at startup. Supports hot-reload
    by calling reload().
    """

    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH):
        self._config_path = Path(config_path).expanduser()
        self._routes: dict[str, DIDConfig] = {}
        if self._config_path.exists():
            self._load()
        else:
            logger.warning("DID config not found at %s — no routes loaded", self._config_path)

    def lookup(self, did: str) -> Optional[DIDConfig]:
        """Look up configuration for a DID.

        Args:
            did: Phone number in any format. Will be normalized to E.164.

        Returns:
            DIDConfig if found, None if not.
        """
        normalized = self._normalize(did)
        return self._routes.get(normalized)

    def lookup_or_raise(self, did: str) -> DIDConfig:
        """Look up configuration for a DID, raising if not found.

        Args:
            did: Phone number in any format.

        Returns:
            DIDConfig for the DID.

        Raises:
            DIDNotFoundError: If the DID is not configured.
        """
        config = self.lookup(did)
        if config is None:
            raise DIDNotFoundError(did)
        return config

    def reload(self) -> None:
        """Reload configuration from disk."""
        self._routes.clear()
        self._load()
        logger.info("DID router reloaded: %d routes", len(self._routes))

    @property
    def dids(self) -> list[str]:
        """List all configured DIDs."""
        return list(self._routes.keys())

    def _load(self) -> None:
        """Load DID routing config from YAML."""
        try:
            with open(self._config_path) as f:
                data = yaml.safe_load(f) or {}
        except Exception as e:
            raise DIDConfigError(f"Failed to load DID config from {self._config_path}: {e}") from e

        dids_data = data.get("dids", {})
        for did_str, config_data in dids_data.items():
            normalized = self._normalize(did_str)
            if not isinstance(config_data, dict):
                logger.warning("Skipping invalid DID config for %s", did_str)
                continue

            self._routes[normalized] = DIDConfig(
                did=normalized,
                user_id=config_data.get("user_id", "unknown"),
                display_name=config_data.get("display_name", ""),
                llm_model=config_data.get("llm_model", "llama3.2:7b"),
                system_prompt=config_data.get("system_prompt", "You are a helpful voice assistant."),
                tts_voice=config_data.get("tts_voice", "en-us-ryan-medium"),
                extra={k: v for k, v in config_data.items()
                       if k not in ("user_id", "display_name", "llm_model", "system_prompt", "tts_voice")},
            )

        logger.info("DID router loaded: %d routes from %s", len(self._routes), self._config_path)

    @staticmethod
    def _normalize(did: str) -> str:
        """Normalize a phone number to E.164 format.

        Strips whitespace, dashes, parens. Ensures leading +.
        Does NOT validate country codes — just basic cleanup.

        Examples:
            "+1 (512) 555-1234" → "+15125551234"
            "15125551234" → "+15125551234"
            "+15125551234" → "+15125551234"
        """
        # Remove all non-digit characters except leading +
        cleaned = re.sub(r"[^\d+]", "", did.strip())
        if not cleaned.startswith("+"):
            cleaned = "+" + cleaned
        return cleaned
