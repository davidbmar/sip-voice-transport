# sip-voice-transport — Implementation Spec for Claude Code

> **Purpose:** This document is a complete specification for building the `sip-voice-transport`
> Python package from scratch. It is designed to be read by Claude Code (or any AI coding agent)
> and executed step by step. Every file, interface, dependency, and behavior is specified.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Dependencies & pyproject.toml](#3-dependencies--pyprojecttoml)
4. [Core Interfaces & ABCs](#4-core-interfaces--abcs)
5. [Audio Codec Module](#5-audio-codec-module)
6. [Provider Implementations](#6-provider-implementations)
7. [SIP Transport (Provider-Agnostic)](#7-sip-transport-provider-agnostic)
8. [DID Router](#8-did-router)
9. [Webhook Handler](#9-webhook-handler)
10. [Mac-Local Utilities](#10-mac-local-utilities)
11. [CLI Entry Point](#11-cli-entry-point)
12. [Example Voice Server](#12-example-voice-server)
13. [Configuration](#13-configuration)
14. [Tests](#14-tests)
15. [README](#15-readme)
16. [Build Order](#16-build-order)

---

## 1. Project Overview

### What This Is

A Python package that bridges PSTN phone calls into a local voice AI pipeline running on a Mac.
Callers dial a phone number (DID) → audio streams over WebSocket from a telephony provider
(Telnyx or Twilio) → through a Cloudflare tunnel → to a FastAPI server on the Mac → where
STT/LLM/TTS processes the audio and responds.

### Key Design Principles

1. **Provider-agnostic.** A `BaseSipProvider` ABC abstracts Telnyx vs Twilio. Swapping
   providers is a config change, not a code change.
2. **Transport-compatible.** `SipTransport` implements a `BaseTransport` ABC that is
   identical to what a WebRTC transport would implement. Application code doesn't know
   or care which transport it's using.
3. **Canonical audio format.** All audio crossing the transport boundary is 16kHz, 16-bit
   signed integer, mono, little-endian PCM. Codec conversion and resampling happen inside
   the transport layer.
4. **Mac-local.** Designed to run as a daemon on macOS. No cloud compute required beyond
   the telephony provider.
5. **Minimal dependencies.** Core uses only `fastapi`, `uvicorn`, `websockets`. Provider
   SDKs are optional extras.

### Terminology

- **DID** — Direct Inward Dialing number. A phone number assigned to a customer.
- **Provider** — Telephony service (Telnyx or Twilio) that bridges PSTN ↔ WebSocket.
- **TeXML** — Telnyx's XML call control language (compatible with TwiML).
- **TwiML** — Twilio's XML call control language.
- **Canonical format** — 16kHz, 16-bit signed LE PCM, mono. The internal audio contract.
- **L16** — Linear 16-bit PCM codec. Telnyx supports this at 16kHz.
- **mulaw / μ-law** — Logarithmic audio codec used by Twilio. 8kHz, 8-bit.

---

## 2. Repository Structure

```
sip-voice-transport/
├── src/
│   └── sip_voice_transport/
│       ├── __init__.py                    # Public API exports + __version__
│       ├── base_transport.py              # BaseTransport ABC
│       ├── base_sip_provider.py           # BaseSipProvider ABC
│       ├── transport.py                   # SipTransport(BaseTransport)
│       ├── audio_codec.py                 # AudioCodec: encode/decode/resample
│       ├── did_router.py                  # DIDConfig + DIDRouter
│       ├── webhook.py                     # SipWebhookHandler (FastAPI routes)
│       ├── config.py                      # SipConfig dataclass + loader
│       ├── exceptions.py                  # Exception hierarchy
│       ├── mac_utils.py                   # SleepInhibitor, process management
│       ├── providers/
│       │   ├── __init__.py                # Provider registry + get_provider()
│       │   ├── telnyx_provider.py         # TelnyxProvider(BaseSipProvider)
│       │   └── twilio_provider.py         # TwilioProvider(BaseSipProvider)
│       └── cli/
│           └── app.py                     # CLI entry point: `sip-voice-server`
├── tests/
│   ├── __init__.py
│   ├── conftest.py                        # Shared fixtures
│   ├── test_audio_codec.py
│   ├── test_did_router.py
│   ├── test_telnyx_provider.py
│   ├── test_twilio_provider.py
│   ├── test_transport.py
│   ├── test_webhook.py
│   └── test_config.py
├── examples/
│   └── mac_voice_server.py                # Complete working example
├── docs/
│   └── design.md                          # Architecture design document
├── .env.example
├── .gitignore
├── pyproject.toml
├── README.md
└── LICENSE                                # MIT
```

Create ALL of these files. Every file is specified in detail below.

---

## 3. Dependencies & pyproject.toml

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "sip-voice-transport"
version = "0.1.0"
description = "PSTN phone call transport for Mac-local voice AI — supports Telnyx and Twilio"
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.11"
authors = [
    {name = "David Mar"},
]
keywords = ["sip", "voip", "telephony", "voice-ai", "telnyx", "twilio", "webrtc", "transport"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Topic :: Communications :: Telephony",
    "Topic :: Multimedia :: Sound/Audio",
]

dependencies = [
    "fastapi>=0.104",
    "uvicorn[standard]>=0.24",
    "websockets>=12.0",
    "pyyaml>=6.0",
    "numpy>=1.24",
]

[project.optional-dependencies]
telnyx = ["telnyx>=2.0"]
twilio = ["twilio>=9.0"]
audio = ["scipy>=1.10", "noisereduce>=3.0"]
cli = ["rich>=13.0", "click>=8.0"]
dev = [
    "pytest>=7.0",
    "pytest-asyncio>=0.21",
    "pytest-cov>=4.0",
    "ruff>=0.1",
]
all = ["sip-voice-transport[telnyx,twilio,audio,cli,dev]"]

[project.scripts]
sip-voice-server = "sip_voice_transport.cli.app:main"

[project.urls]
Repository = "https://github.com/davidbmar/sip-voice-transport"

[tool.hatch.build.targets.wheel]
packages = ["src/sip_voice_transport"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.ruff]
target-version = "py311"
line-length = 100
```

---

## 4. Core Interfaces & ABCs

### 4.1 `src/sip_voice_transport/base_transport.py`

This is the contract that ALL transports (SIP, WebRTC, future) implement. The voice
engine only depends on this interface.

```python
"""Base transport ABC — the contract between transport and engine layers.

All audio crossing this boundary uses the CANONICAL FORMAT:
  - Sample rate: 16000 Hz
  - Bit depth: 16-bit signed integer
  - Channels: mono
  - Byte order: little-endian
  - Container: raw PCM bytes
"""

from abc import ABC, abstractmethod
from typing import AsyncGenerator


CANONICAL_SAMPLE_RATE = 16000
CANONICAL_SAMPLE_WIDTH = 2  # 16-bit = 2 bytes
CANONICAL_CHANNELS = 1


class BaseTransport(ABC):
    """Abstract base for all audio transport types (SIP, WebRTC, etc.)."""

    @abstractmethod
    async def receive_audio(self) -> AsyncGenerator[bytes, None]:
        """Yield audio chunks in canonical format (16kHz, 16-bit, mono PCM).

        Each chunk represents a small segment of audio (typically 20-100ms).
        The generator completes when the connection ends.

        Yields:
            bytes: Raw PCM audio in canonical format.
        """
        ...

    @abstractmethod
    async def send_audio(self, audio: bytes) -> None:
        """Send audio to the remote end.

        Args:
            audio: Raw PCM audio in canonical format (16kHz, 16-bit, mono).
        """
        ...

    @abstractmethod
    async def send_mark(self, name: str) -> None:
        """Send a playback mark for synchronization.

        The transport should notify when audio preceding this mark has finished playing.

        Args:
            name: Identifier for this mark.
        """
        ...

    @abstractmethod
    async def clear_audio(self) -> None:
        """Clear all buffered outbound audio (for barge-in / interruption)."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Clean up the transport connection."""
        ...

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Whether the transport is currently connected."""
        ...

    @property
    @abstractmethod
    def metadata(self) -> dict:
        """Transport metadata.

        Returns a dict with at minimum:
            - transport: str — transport type ("sip", "webrtc")
            - Additional provider-specific metadata.

        For SIP transports, also includes:
            - provider: str — "telnyx" or "twilio"
            - caller_id: str — caller's phone number (E.164)
            - did: str — DID that was called (E.164)
            - call_sid: str — provider's call identifier
        """
        ...
```

### 4.2 `src/sip_voice_transport/base_sip_provider.py`

This abstracts the differences between Telnyx and Twilio's WebSocket protocols.

```python
"""Base SIP provider ABC — abstracts provider-specific WebSocket protocol details.

Each provider (Telnyx, Twilio) sends audio in a different format and framing.
This ABC hides those differences from the SipTransport layer.
"""

from abc import ABC, abstractmethod
from typing import AsyncGenerator
from fastapi import WebSocket


class BaseSipProvider(ABC):
    """Abstract base for SIP telephony providers."""

    @abstractmethod
    async def connect(self, websocket: WebSocket, first_message: dict) -> None:
        """Initialize the provider connection.

        Called after the WebSocket is accepted and the first message has been
        read (used for provider detection). The provider should process the
        first message and any subsequent handshake messages (e.g., Twilio's
        'connected' + 'start' sequence).

        Args:
            websocket: The accepted FastAPI WebSocket connection.
            first_message: The first JSON message received (already consumed).
        """
        ...

    @abstractmethod
    async def receive_audio_frames(self) -> AsyncGenerator[bytes, None]:
        """Yield decoded audio frames from the provider's WebSocket.

        Returns audio in the PROVIDER'S NATIVE FORMAT — not canonical format.
        The SipTransport layer handles conversion to canonical format.

        For Telnyx with L16: yields 16kHz 16-bit PCM bytes.
        For Twilio: yields 8kHz mulaw bytes.

        Yields:
            bytes: Raw audio in provider's native format.
        """
        ...

    @abstractmethod
    async def send_audio_frame(self, audio: bytes) -> None:
        """Encode and send an audio frame in the provider's expected format.

        The audio is in the PROVIDER'S NATIVE FORMAT (already converted from
        canonical by the SipTransport layer).

        Args:
            audio: Raw audio bytes in provider's native format.
        """
        ...

    @abstractmethod
    async def send_mark(self, name: str) -> None:
        """Send a playback mark event.

        Args:
            name: Mark identifier.
        """
        ...

    @abstractmethod
    async def clear_audio(self) -> None:
        """Clear buffered outbound audio."""
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Clean up the WebSocket connection."""
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Provider identifier string (e.g., 'telnyx', 'twilio')."""
        ...

    @property
    @abstractmethod
    def native_sample_rate(self) -> int:
        """Provider's native audio sample rate in Hz.

        Telnyx L16: 16000
        Telnyx mulaw: 8000
        Twilio: 8000
        """
        ...

    @property
    @abstractmethod
    def native_encoding(self) -> str:
        """Provider's native audio encoding.

        One of: 'l16' (linear 16-bit PCM) or 'mulaw' (μ-law).
        """
        ...

    @property
    @abstractmethod
    def stream_id(self) -> str | None:
        """Provider-specific stream/call identifier. None if not yet connected."""
        ...

    @property
    @abstractmethod
    def call_metadata(self) -> dict:
        """Call information extracted from the provider's handshake.

        Returns dict with:
            - caller_id: str — E.164 phone number of caller
            - did: str — E.164 phone number that was called
            - call_sid: str — provider's unique call identifier
            - stream_id: str — provider's stream identifier
        """
        ...
```

### 4.3 `src/sip_voice_transport/exceptions.py`

```python
"""Exception hierarchy for sip-voice-transport."""


class SipTransportError(Exception):
    """Base exception for all SIP transport errors."""
    pass


class ProviderConnectionError(SipTransportError):
    """Failed to establish or maintain provider WebSocket connection."""
    pass


class AudioCodecError(SipTransportError):
    """Error during audio encoding, decoding, or resampling."""
    pass


class DIDNotFoundError(SipTransportError):
    """The called DID is not configured in the routing table."""

    def __init__(self, did: str):
        self.did = did
        super().__init__(f"DID not found in routing table: {did}")


class DIDConfigError(SipTransportError):
    """Error loading or parsing DID configuration."""
    pass


class ProviderDetectionError(SipTransportError):
    """Could not determine the telephony provider from the WebSocket message."""
    pass


class CallRejectedError(SipTransportError):
    """The call was intentionally rejected (e.g., DID not configured)."""
    pass
```

### 4.4 `src/sip_voice_transport/__init__.py`

```python
"""sip-voice-transport — PSTN phone call transport for Mac-local voice AI.

Supports Telnyx and Twilio as telephony providers.
"""

__version__ = "0.1.0"

from sip_voice_transport.base_transport import (
    BaseTransport,
    CANONICAL_SAMPLE_RATE,
    CANONICAL_SAMPLE_WIDTH,
    CANONICAL_CHANNELS,
)
from sip_voice_transport.base_sip_provider import BaseSipProvider
from sip_voice_transport.transport import SipTransport
from sip_voice_transport.audio_codec import AudioCodec
from sip_voice_transport.did_router import DIDConfig, DIDRouter
from sip_voice_transport.webhook import SipWebhookHandler
from sip_voice_transport.config import SipConfig
from sip_voice_transport.exceptions import (
    SipTransportError,
    ProviderConnectionError,
    AudioCodecError,
    DIDNotFoundError,
    DIDConfigError,
    ProviderDetectionError,
    CallRejectedError,
)

__all__ = [
    "BaseTransport",
    "BaseSipProvider",
    "SipTransport",
    "AudioCodec",
    "DIDConfig",
    "DIDRouter",
    "SipWebhookHandler",
    "SipConfig",
    "SipTransportError",
    "ProviderConnectionError",
    "AudioCodecError",
    "DIDNotFoundError",
    "DIDConfigError",
    "ProviderDetectionError",
    "CallRejectedError",
    "CANONICAL_SAMPLE_RATE",
    "CANONICAL_SAMPLE_WIDTH",
    "CANONICAL_CHANNELS",
]
```

---

## 5. Audio Codec Module

### `src/sip_voice_transport/audio_codec.py`

This is critical. It handles all audio format conversion between providers' native
formats and the canonical format.

```python
"""Audio codec utilities for converting between provider and canonical formats.

CANONICAL FORMAT: 16kHz, 16-bit signed LE PCM, mono.

Conversion paths:
  Telnyx L16/16kHz  → canonical: NO-OP (same format)
  Telnyx mulaw/8kHz → canonical: mulaw→PCM + resample 8k→16k
  Twilio mulaw/8kHz → canonical: mulaw→PCM + resample 8k→16k
"""

import base64
import struct
import logging
from typing import Optional

import numpy as np

from sip_voice_transport.exceptions import AudioCodecError

logger = logging.getLogger(__name__)


# μ-law decompression table (ITU-T G.711)
# This avoids dependency on audioop (deprecated in Python 3.11, removed in 3.13)
_MULAW_DECODE_TABLE: Optional[np.ndarray] = None


def _get_mulaw_decode_table() -> np.ndarray:
    """Lazily build the μ-law to linear PCM lookup table."""
    global _MULAW_DECODE_TABLE
    if _MULAW_DECODE_TABLE is None:
        table = np.zeros(256, dtype=np.int16)
        for i in range(256):
            val = ~i
            sign = val & 0x80
            exponent = (val >> 4) & 0x07
            mantissa = val & 0x0F
            sample = (mantissa << 3) + 0x84
            sample <<= exponent
            sample -= 0x84
            table[i] = -sample if sign else sample
        _MULAW_DECODE_TABLE = table
    return _MULAW_DECODE_TABLE


def _get_mulaw_encode_table() -> np.ndarray:
    """Build linear PCM to μ-law lookup via the decode table inverse."""
    decode = _get_mulaw_decode_table()
    # Build by finding closest match for each possible 16-bit value
    # For efficiency, use the standard algorithm instead of table lookup
    # This function is called per-frame so we use vectorized numpy
    pass  # See mulaw_encode below for vectorized implementation


class AudioCodec:
    """Handles audio format conversion between provider and canonical formats.

    Usage:
        codec = AudioCodec(source_rate=8000, source_encoding="mulaw")
        canonical_pcm = codec.decode_to_canonical(provider_bytes)
        provider_bytes = codec.encode_from_canonical(canonical_pcm)
    """

    def __init__(
        self,
        source_rate: int,
        source_encoding: str,
        target_rate: int = 16000,
        target_encoding: str = "l16",
    ):
        """
        Args:
            source_rate: Provider's native sample rate (8000 or 16000).
            source_encoding: Provider's native encoding ('l16' or 'mulaw').
            target_rate: Canonical sample rate. Always 16000.
            target_encoding: Canonical encoding. Always 'l16'.
        """
        self.source_rate = source_rate
        self.source_encoding = source_encoding.lower()
        self.target_rate = target_rate
        self.target_encoding = target_encoding.lower()

        self._needs_resample = (source_rate != target_rate)
        self._needs_decode = (self.source_encoding != self.target_encoding)
        self._is_noop = (not self._needs_resample and not self._needs_decode)

        if self._is_noop:
            logger.info(
                "AudioCodec: source matches canonical (%dHz %s) — no conversion needed",
                source_rate, source_encoding,
            )

    def decode_to_canonical(self, raw: bytes) -> bytes:
        """Convert provider-format audio to canonical 16kHz 16-bit PCM.

        Args:
            raw: Audio bytes in provider's native format.

        Returns:
            Audio bytes in canonical format (16kHz, 16-bit, mono PCM).
        """
        if self._is_noop:
            return raw

        try:
            pcm = raw
            if self.source_encoding == "mulaw":
                pcm = mulaw_decode(raw)
            if self._needs_resample:
                pcm = resample(pcm, self.source_rate, self.target_rate)
            return pcm
        except Exception as e:
            raise AudioCodecError(f"Failed to decode audio: {e}") from e

    def encode_from_canonical(self, pcm: bytes) -> bytes:
        """Convert canonical 16kHz 16-bit PCM to provider's format.

        Args:
            pcm: Audio bytes in canonical format.

        Returns:
            Audio bytes in provider's native format.
        """
        if self._is_noop:
            return pcm

        try:
            out = pcm
            if self._needs_resample:
                out = resample(out, self.target_rate, self.source_rate)
            if self.source_encoding == "mulaw":
                out = mulaw_encode(out)
            return out
        except Exception as e:
            raise AudioCodecError(f"Failed to encode audio: {e}") from e


def mulaw_decode(data: bytes) -> bytes:
    """Decode μ-law encoded audio to 16-bit linear PCM.

    This implements ITU-T G.711 μ-law decoding without using the deprecated
    `audioop` module.

    Args:
        data: μ-law encoded bytes (1 byte per sample).

    Returns:
        16-bit linear PCM bytes (2 bytes per sample, little-endian).
    """
    table = _get_mulaw_decode_table()
    samples = np.frombuffer(data, dtype=np.uint8)
    pcm = table[samples]
    return pcm.tobytes()


def mulaw_encode(pcm_data: bytes) -> bytes:
    """Encode 16-bit linear PCM to μ-law.

    Implements ITU-T G.711 μ-law compression without `audioop`.

    Args:
        pcm_data: 16-bit signed PCM bytes (little-endian).

    Returns:
        μ-law encoded bytes (1 byte per sample).
    """
    MULAW_MAX = 0x1FFF
    MULAW_BIAS = 0x84

    samples = np.frombuffer(pcm_data, dtype=np.int16).astype(np.int32)

    # Get sign
    sign = np.where(samples < 0, 0x80, 0).astype(np.uint8)
    samples = np.abs(samples)

    # Clip to max
    samples = np.minimum(samples, MULAW_MAX)

    # Add bias
    samples = samples + MULAW_BIAS

    # Find exponent and mantissa
    exponent = np.zeros(len(samples), dtype=np.uint8)
    for exp in range(7, -1, -1):
        mask = 1 << (exp + 3)
        exponent = np.where((samples >= mask) & (exponent == 0), exp, exponent)

    mantissa = ((samples >> (exponent + 3)) & 0x0F).astype(np.uint8)
    mulaw_byte = ~(sign | (exponent << 4) | mantissa)

    return mulaw_byte.astype(np.uint8).tobytes()


def resample(pcm_data: bytes, from_rate: int, to_rate: int) -> bytes:
    """Resample 16-bit PCM audio.

    Uses linear interpolation for speed. For higher quality, install scipy
    and this will use scipy.signal.resample instead.

    Args:
        pcm_data: 16-bit signed PCM bytes.
        from_rate: Source sample rate in Hz.
        to_rate: Target sample rate in Hz.

    Returns:
        Resampled 16-bit PCM bytes.
    """
    if from_rate == to_rate:
        return pcm_data

    samples = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float64)
    num_output = int(len(samples) * to_rate / from_rate)

    try:
        from scipy.signal import resample as scipy_resample
        resampled = scipy_resample(samples, num_output)
    except ImportError:
        # Fallback: linear interpolation
        x_old = np.linspace(0, 1, len(samples))
        x_new = np.linspace(0, 1, num_output)
        resampled = np.interp(x_new, x_old, samples)

    # Clip to int16 range and convert
    resampled = np.clip(resampled, -32768, 32767).astype(np.int16)
    return resampled.tobytes()


def base64_decode_audio(payload: str) -> bytes:
    """Decode a base64 audio payload from a provider WebSocket message.

    Args:
        payload: Base64 encoded string.

    Returns:
        Raw audio bytes.
    """
    return base64.b64decode(payload)


def base64_encode_audio(audio: bytes) -> str:
    """Encode audio bytes to base64 for sending over a provider WebSocket.

    Args:
        audio: Raw audio bytes.

    Returns:
        Base64 encoded string.
    """
    return base64.b64encode(audio).decode("ascii")
```

---

## 6. Provider Implementations

### 6.1 `src/sip_voice_transport/providers/__init__.py`

```python
"""SIP provider registry.

Providers are lazily imported to avoid requiring all provider SDKs.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sip_voice_transport.base_sip_provider import BaseSipProvider

_PROVIDERS: dict[str, str] = {
    "telnyx": "sip_voice_transport.providers.telnyx_provider:TelnyxProvider",
    "twilio": "sip_voice_transport.providers.twilio_provider:TwilioProvider",
}


def get_provider(name: str) -> "BaseSipProvider":
    """Get a provider instance by name.

    Args:
        name: Provider name ('telnyx' or 'twilio').

    Returns:
        An unconnected provider instance.

    Raises:
        ValueError: If the provider name is not recognized.
        ImportError: If the provider's dependencies are not installed.
    """
    name = name.lower()
    if name not in _PROVIDERS:
        raise ValueError(
            f"Unknown provider: {name!r}. Available: {list(_PROVIDERS.keys())}"
        )

    module_path, class_name = _PROVIDERS[name].rsplit(":", 1)
    import importlib
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls()


def detect_provider_from_message(message: dict) -> "BaseSipProvider":
    """Detect the provider from the first WebSocket message.

    Twilio sends: {"event": "connected", "protocol": "Call", "version": "1.0.0"}
    Telnyx sends: {"event": "connected", ...} without "protocol" field

    Args:
        message: The first JSON message from the WebSocket.

    Returns:
        An unconnected provider instance.

    Raises:
        ProviderDetectionError: If the provider cannot be determined.
    """
    from sip_voice_transport.exceptions import ProviderDetectionError

    if message.get("protocol") == "Call":
        return get_provider("twilio")
    elif message.get("event") == "connected":
        return get_provider("telnyx")
    else:
        raise ProviderDetectionError(
            f"Cannot determine provider from message: {message}"
        )
```

### 6.2 `src/sip_voice_transport/providers/telnyx_provider.py`

```python
"""Telnyx SIP provider implementation.

Telnyx WebSocket message flow:
  1. Server accepts WebSocket
  2. Telnyx sends: {"event": "connected", ...}
  3. Telnyx sends: {"event": "start", "stream_id": "...", ...}
  4. Telnyx sends: {"event": "media", "media": {"payload": "<base64>"}, ...}
     (repeated for each audio frame)
  5. Telnyx sends: {"event": "stop", ...}

Bidirectional audio:
  Server sends: {"event": "media", "stream_id": "...", "media": {"payload": "<base64>"}}

Mark events:
  Server sends: {"event": "mark", "stream_id": "...", "mark": {"name": "..."}}
  Telnyx echoes back when audio before mark finishes playing.

DTMF:
  Telnyx sends: {"event": "dtmf", "dtmf": {"digit": "1"}}

Codec configuration:
  When streaming is started via TeXML with bidirectional_codec="L16",
  audio is 16kHz 16-bit PCM. This is the recommended configuration as
  it matches the canonical format (no conversion needed).
"""

import json
import logging
from typing import AsyncGenerator

from fastapi import WebSocket, WebSocketDisconnect

from sip_voice_transport.base_sip_provider import BaseSipProvider
from sip_voice_transport.audio_codec import base64_decode_audio, base64_encode_audio
from sip_voice_transport.exceptions import ProviderConnectionError

logger = logging.getLogger(__name__)


class TelnyxProvider(BaseSipProvider):
    """Telnyx telephony provider."""

    def __init__(self):
        self._websocket: WebSocket | None = None
        self._stream_id: str | None = None
        self._call_control_id: str | None = None
        self._caller_id: str = ""
        self._did: str = ""
        self._call_sid: str = ""
        self._connected: bool = False
        # Default to L16/16kHz — updated from start message if different
        self._sample_rate: int = 16000
        self._encoding: str = "l16"

    async def connect(self, websocket: WebSocket, first_message: dict) -> None:
        """Process Telnyx handshake: connected event, then start event."""
        self._websocket = websocket

        # First message is the 'connected' event (already read by webhook handler)
        if first_message.get("event") != "connected":
            raise ProviderConnectionError(
                f"Expected 'connected' event, got: {first_message.get('event')}"
            )
        logger.info("Telnyx: connected event received")

        # Read the 'start' event which contains stream metadata
        start_msg = await websocket.receive_json()
        if start_msg.get("event") != "start":
            raise ProviderConnectionError(
                f"Expected 'start' event, got: {start_msg.get('event')}"
            )

        start_data = start_msg.get("start", {})
        self._stream_id = start_msg.get("stream_id")
        self._call_control_id = start_data.get("call_control_id", "")
        self._call_sid = start_data.get("call_control_id", "")

        # Extract caller info if available in custom parameters or headers
        # Telnyx includes these in the start message's custom_parameters
        custom = start_data.get("custom_parameters", {})
        self._caller_id = custom.get("caller_id", "")
        self._did = custom.get("did", "")

        # Detect encoding from start message if present
        media_format = start_data.get("media_format", {})
        if media_format:
            encoding = media_format.get("encoding", "").lower()
            if "mulaw" in encoding or "ulaw" in encoding:
                self._encoding = "mulaw"
                self._sample_rate = int(media_format.get("sample_rate", 8000))
            elif "l16" in encoding or "pcm" in encoding:
                self._encoding = "l16"
                self._sample_rate = int(media_format.get("sample_rate", 16000))

        self._connected = True
        logger.info(
            "Telnyx: stream started — id=%s, encoding=%s, rate=%d",
            self._stream_id, self._encoding, self._sample_rate,
        )

    async def receive_audio_frames(self) -> AsyncGenerator[bytes, None]:
        """Yield decoded audio from Telnyx media events."""
        if not self._websocket:
            raise ProviderConnectionError("Not connected")

        try:
            while True:
                message = await self._websocket.receive_json()
                event = message.get("event")

                if event == "media":
                    payload = message.get("media", {}).get("payload")
                    if payload:
                        yield base64_decode_audio(payload)

                elif event == "dtmf":
                    digit = message.get("dtmf", {}).get("digit")
                    logger.info("Telnyx: DTMF digit received: %s", digit)
                    # TODO: emit DTMF event to application layer

                elif event == "mark":
                    name = message.get("mark", {}).get("name")
                    logger.debug("Telnyx: mark received: %s", name)
                    # TODO: emit mark event to application layer

                elif event == "stop":
                    logger.info("Telnyx: stream stopped")
                    self._connected = False
                    break

                else:
                    logger.debug("Telnyx: unknown event: %s", event)

        except WebSocketDisconnect:
            logger.info("Telnyx: WebSocket disconnected")
            self._connected = False

    async def send_audio_frame(self, audio: bytes) -> None:
        """Send audio frame to Telnyx."""
        if not self._websocket or not self._stream_id:
            raise ProviderConnectionError("Not connected")

        message = {
            "event": "media",
            "stream_id": self._stream_id,
            "media": {
                "payload": base64_encode_audio(audio),
            },
        }
        await self._websocket.send_json(message)

    async def send_mark(self, name: str) -> None:
        if not self._websocket or not self._stream_id:
            raise ProviderConnectionError("Not connected")

        message = {
            "event": "mark",
            "stream_id": self._stream_id,
            "mark": {"name": name},
        }
        await self._websocket.send_json(message)

    async def clear_audio(self) -> None:
        if not self._websocket or not self._stream_id:
            raise ProviderConnectionError("Not connected")

        message = {
            "event": "clear",
            "stream_id": self._stream_id,
        }
        await self._websocket.send_json(message)

    async def disconnect(self) -> None:
        self._connected = False
        if self._websocket:
            try:
                await self._websocket.close()
            except Exception:
                pass
            self._websocket = None

    @property
    def provider_name(self) -> str:
        return "telnyx"

    @property
    def native_sample_rate(self) -> int:
        return self._sample_rate

    @property
    def native_encoding(self) -> str:
        return self._encoding

    @property
    def stream_id(self) -> str | None:
        return self._stream_id

    @property
    def call_metadata(self) -> dict:
        return {
            "caller_id": self._caller_id,
            "did": self._did,
            "call_sid": self._call_sid,
            "stream_id": self._stream_id or "",
            "call_control_id": self._call_control_id or "",
        }
```

### 6.3 `src/sip_voice_transport/providers/twilio_provider.py`

```python
"""Twilio SIP provider implementation.

Twilio WebSocket message flow:
  1. Server accepts WebSocket
  2. Twilio sends: {"event": "connected", "protocol": "Call", "version": "1.0.0"}
  3. Twilio sends: {"event": "start", "start": {"streamSid": "...", "callSid": "...",
     "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": 8000, "channels": 1}}}
  4. Twilio sends: {"event": "media", "media": {"payload": "<base64_mulaw>"}}
     (repeated for each audio frame)
  5. Twilio sends: {"event": "stop"}

Bidirectional audio (requires <Connect><Stream>):
  Server sends: {"event": "media", "streamSid": "...", "media": {"payload": "<base64_mulaw>"}}

Mark events:
  Server sends: {"event": "mark", "streamSid": "...", "mark": {"name": "..."}}

Clear (interrupt):
  Server sends: {"event": "clear", "streamSid": "..."}

IMPORTANT: Twilio bidirectional audio MUST be audio/x-mulaw at 8000 Hz.
There is no way to configure a different codec. This means we always need
mulaw↔PCM conversion and 8kHz↔16kHz resampling for Twilio.
"""

import logging
from typing import AsyncGenerator

from fastapi import WebSocket, WebSocketDisconnect

from sip_voice_transport.base_sip_provider import BaseSipProvider
from sip_voice_transport.audio_codec import base64_decode_audio, base64_encode_audio
from sip_voice_transport.exceptions import ProviderConnectionError

logger = logging.getLogger(__name__)


class TwilioProvider(BaseSipProvider):
    """Twilio telephony provider."""

    def __init__(self):
        self._websocket: WebSocket | None = None
        self._stream_sid: str | None = None
        self._call_sid: str = ""
        self._account_sid: str = ""
        self._caller_id: str = ""
        self._did: str = ""
        self._connected: bool = False
        self._custom_parameters: dict = {}

    async def connect(self, websocket: WebSocket, first_message: dict) -> None:
        """Process Twilio handshake: connected + start events."""
        self._websocket = websocket

        # First message is 'connected'
        if first_message.get("event") != "connected":
            raise ProviderConnectionError(
                f"Expected 'connected' event, got: {first_message.get('event')}"
            )
        logger.info("Twilio: connected event received (protocol=%s)", first_message.get("protocol"))

        # Read 'start' event
        start_msg = await websocket.receive_json()
        if start_msg.get("event") != "start":
            raise ProviderConnectionError(
                f"Expected 'start' event, got: {start_msg.get('event')}"
            )

        start_data = start_msg.get("start", {})
        self._stream_sid = start_data.get("streamSid") or start_msg.get("streamSid")
        self._call_sid = start_data.get("callSid", "")
        self._account_sid = start_data.get("accountSid", "")
        self._custom_parameters = start_data.get("customParameters", {})

        # Extract DID from custom parameters (passed via TwiML <Parameter>)
        self._did = self._custom_parameters.get("did", "")
        self._caller_id = self._custom_parameters.get("caller_id", "")

        self._connected = True
        logger.info(
            "Twilio: stream started — streamSid=%s, callSid=%s",
            self._stream_sid, self._call_sid,
        )

    async def receive_audio_frames(self) -> AsyncGenerator[bytes, None]:
        """Yield decoded audio from Twilio media events.

        Audio is mulaw at 8kHz. The SipTransport layer handles conversion.
        """
        if not self._websocket:
            raise ProviderConnectionError("Not connected")

        try:
            while True:
                message = await self._websocket.receive_json()
                event = message.get("event")

                if event == "media":
                    payload = message.get("media", {}).get("payload")
                    if payload:
                        yield base64_decode_audio(payload)

                elif event == "dtmf":
                    digit = message.get("dtmf", {}).get("digit")
                    logger.info("Twilio: DTMF digit received: %s", digit)

                elif event == "mark":
                    name = message.get("mark", {}).get("name")
                    logger.debug("Twilio: mark received: %s", name)

                elif event == "stop":
                    logger.info("Twilio: stream stopped")
                    self._connected = False
                    break

                else:
                    logger.debug("Twilio: unknown event: %s", event)

        except WebSocketDisconnect:
            logger.info("Twilio: WebSocket disconnected")
            self._connected = False

    async def send_audio_frame(self, audio: bytes) -> None:
        """Send mulaw audio frame to Twilio."""
        if not self._websocket or not self._stream_sid:
            raise ProviderConnectionError("Not connected")

        message = {
            "event": "media",
            "streamSid": self._stream_sid,
            "media": {
                "payload": base64_encode_audio(audio),
            },
        }
        await self._websocket.send_json(message)

    async def send_mark(self, name: str) -> None:
        if not self._websocket or not self._stream_sid:
            raise ProviderConnectionError("Not connected")

        message = {
            "event": "mark",
            "streamSid": self._stream_sid,
            "mark": {"name": name},
        }
        await self._websocket.send_json(message)

    async def clear_audio(self) -> None:
        if not self._websocket or not self._stream_sid:
            raise ProviderConnectionError("Not connected")

        message = {
            "event": "clear",
            "streamSid": self._stream_sid,
        }
        await self._websocket.send_json(message)

    async def disconnect(self) -> None:
        self._connected = False
        if self._websocket:
            try:
                await self._websocket.close()
            except Exception:
                pass
            self._websocket = None

    @property
    def provider_name(self) -> str:
        return "twilio"

    @property
    def native_sample_rate(self) -> int:
        return 8000  # Twilio is always 8kHz mulaw

    @property
    def native_encoding(self) -> str:
        return "mulaw"  # Twilio is always mulaw

    @property
    def stream_id(self) -> str | None:
        return self._stream_sid

    @property
    def call_metadata(self) -> dict:
        return {
            "caller_id": self._caller_id,
            "did": self._did,
            "call_sid": self._call_sid,
            "stream_id": self._stream_sid or "",
            "account_sid": self._account_sid,
        }
```

---

## 7. SIP Transport (Provider-Agnostic)

### `src/sip_voice_transport/transport.py`

```python
"""Provider-agnostic SIP transport.

Wraps a BaseSipProvider and handles:
  - Audio format conversion (provider native ↔ canonical 16kHz PCM)
  - Uniform interface for the voice engine (BaseTransport)

The voice engine never sees provider-specific details.
"""

import logging
from typing import AsyncGenerator

from sip_voice_transport.base_transport import BaseTransport
from sip_voice_transport.base_sip_provider import BaseSipProvider
from sip_voice_transport.audio_codec import AudioCodec

logger = logging.getLogger(__name__)


class SipTransport(BaseTransport):
    """SIP transport — adapts any BaseSipProvider to the BaseTransport interface.

    Usage:
        provider = TelnyxProvider()
        await provider.connect(websocket, first_message)
        transport = SipTransport(provider)

        # Now use as BaseTransport
        async for audio_chunk in transport.receive_audio():
            # audio_chunk is canonical 16kHz PCM
            ...
        await transport.send_audio(tts_pcm_bytes)
    """

    def __init__(self, provider: BaseSipProvider):
        self._provider = provider
        self._codec = AudioCodec(
            source_rate=provider.native_sample_rate,
            source_encoding=provider.native_encoding,
        )
        logger.info(
            "SipTransport initialized: provider=%s, native=%dHz/%s, codec=%s",
            provider.provider_name,
            provider.native_sample_rate,
            provider.native_encoding,
            "noop" if self._codec._is_noop else "converting",
        )

    async def receive_audio(self) -> AsyncGenerator[bytes, None]:
        """Yield canonical 16kHz PCM audio chunks from the phone call."""
        async for frame in self._provider.receive_audio_frames():
            yield self._codec.decode_to_canonical(frame)

    async def send_audio(self, audio: bytes) -> None:
        """Send canonical 16kHz PCM audio to the phone call."""
        encoded = self._codec.encode_from_canonical(audio)
        await self._provider.send_audio_frame(encoded)

    async def send_mark(self, name: str) -> None:
        await self._provider.send_mark(name)

    async def clear_audio(self) -> None:
        await self._provider.clear_audio()

    async def close(self) -> None:
        await self._provider.disconnect()

    @property
    def is_connected(self) -> bool:
        return self._provider.stream_id is not None

    @property
    def metadata(self) -> dict:
        return {
            "transport": "sip",
            "provider": self._provider.provider_name,
            **self._provider.call_metadata,
        }
```

---

## 8. DID Router

### `src/sip_voice_transport/did_router.py`

```python
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
```

---

## 9. Webhook Handler

### `src/sip_voice_transport/webhook.py`

```python
"""FastAPI webhook handler for SIP providers.

Registers three routes:
  POST /sip/telnyx/answer  — Telnyx inbound call webhook, returns TeXML
  POST /sip/twilio/answer  — Twilio inbound call webhook, returns TwiML
  WSS  /sip/media-stream   — WebSocket endpoint for audio streaming (both providers)

The webhook handler creates provider instances, wraps them in SipTransport,
and hands them off to the application's on_session callback.
"""

import asyncio
import logging
from typing import Callable, Awaitable, Optional

from fastapi import FastAPI, WebSocket, Request, WebSocketDisconnect
from fastapi.responses import Response

from sip_voice_transport.did_router import DIDRouter, DIDConfig
from sip_voice_transport.transport import SipTransport
from sip_voice_transport.providers import detect_provider_from_message
from sip_voice_transport.exceptions import (
    ProviderDetectionError,
    ProviderConnectionError,
    DIDNotFoundError,
)

logger = logging.getLogger(__name__)

# Type for the application callback
OnSessionCallback = Callable[[SipTransport, Optional[DIDConfig]], Awaitable[None]]


class SipWebhookHandler:
    """Registers FastAPI routes for SIP provider webhooks.

    Usage:
        app = FastAPI()
        router = DIDRouter()

        async def on_session(transport: SipTransport, config: DIDConfig | None):
            # Your voice AI logic here
            async for audio in transport.receive_audio():
                ...

        handler = SipWebhookHandler(
            did_router=router,
            on_session=on_session,
            public_ws_url="wss://voice.example.com/sip/media-stream",
        )
        handler.register(app)
    """

    def __init__(
        self,
        did_router: DIDRouter,
        on_session: OnSessionCallback,
        public_ws_url: str = "wss://localhost:8000/sip/media-stream",
    ):
        self.did_router = did_router
        self.on_session = on_session
        self.public_ws_url = public_ws_url

    def register(self, app: FastAPI) -> None:
        """Register all SIP routes on a FastAPI application."""

        @app.post("/sip/telnyx/answer")
        async def telnyx_answer(request: Request):
            """Handle Telnyx inbound call webhook. Returns TeXML."""
            form = await request.form()
            did = str(form.get("To", ""))
            caller = str(form.get("From", ""))
            call_sid = str(form.get("CallSid", ""))

            logger.info("Telnyx inbound call: %s → %s (CallSid=%s)", caller, did, call_sid)

            config = self.did_router.lookup(did)
            if not config:
                logger.warning("Telnyx: DID not configured: %s", did)
                return Response(
                    content=self._texml_reject("This number is not configured."),
                    media_type="application/xml",
                )

            return Response(
                content=self._texml_stream(caller, did),
                media_type="application/xml",
            )

        @app.post("/sip/twilio/answer")
        async def twilio_answer(request: Request):
            """Handle Twilio inbound call webhook. Returns TwiML."""
            form = await request.form()
            did = str(form.get("To", ""))
            caller = str(form.get("From", ""))
            call_sid = str(form.get("CallSid", ""))

            logger.info("Twilio inbound call: %s → %s (CallSid=%s)", caller, did, call_sid)

            config = self.did_router.lookup(did)
            if not config:
                logger.warning("Twilio: DID not configured: %s", did)
                return Response(
                    content=self._twiml_reject("This number is not configured."),
                    media_type="application/xml",
                )

            return Response(
                content=self._twiml_stream(caller, did),
                media_type="application/xml",
            )

        @app.websocket("/sip/media-stream")
        async def media_stream(websocket: WebSocket):
            """Handle audio streaming WebSocket from either provider."""
            await websocket.accept()
            logger.info("SIP media stream WebSocket connected")

            try:
                # Read first message to detect provider
                first_message = await websocket.receive_json()
                provider = detect_provider_from_message(first_message)

                # Let provider handle its handshake
                await provider.connect(websocket, first_message)

                # Look up DID config
                did = provider.call_metadata.get("did", "")
                config = self.did_router.lookup(did) if did else None

                # Wrap in transport and hand off to application
                transport = SipTransport(provider)
                logger.info(
                    "SIP session started: provider=%s, did=%s, caller=%s",
                    provider.provider_name,
                    provider.call_metadata.get("did"),
                    provider.call_metadata.get("caller_id"),
                )

                await self.on_session(transport, config)

            except ProviderDetectionError as e:
                logger.error("Failed to detect provider: %s", e)
            except ProviderConnectionError as e:
                logger.error("Provider connection error: %s", e)
            except WebSocketDisconnect:
                logger.info("SIP media stream WebSocket disconnected")
            except Exception as e:
                logger.exception("Unexpected error in SIP media stream: %s", e)
            finally:
                logger.info("SIP media stream ended")

    def _texml_stream(self, caller: str, did: str) -> str:
        """Generate TeXML to start bidirectional streaming with L16/16kHz."""
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Start>
    <Stream url="{self.public_ws_url}"
            track="inbound_track"
            bidirectionalMode="rtp"
            bidirectionalCodec="L16">
      <Parameter name="caller_id" value="{caller}"/>
      <Parameter name="did" value="{did}"/>
    </Stream>
  </Start>
  <Pause length="3600"/>
</Response>"""

    def _texml_reject(self, message: str) -> str:
        """Generate TeXML to reject a call with a spoken message."""
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say>{message}</Say>
  <Hangup/>
</Response>"""

    def _twiml_stream(self, caller: str, did: str) -> str:
        """Generate TwiML to start bidirectional streaming."""
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="{self.public_ws_url}">
      <Parameter name="caller_id" value="{caller}"/>
      <Parameter name="did" value="{did}"/>
    </Stream>
  </Connect>
</Response>"""

    def _twiml_reject(self, message: str) -> str:
        """Generate TwiML to reject a call with a spoken message."""
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say>{message}</Say>
  <Hangup/>
</Response>"""
```

---

## 10. Mac-Local Utilities

### `src/sip_voice_transport/mac_utils.py`

```python
"""macOS-specific utilities for running the voice server as a daemon."""

import logging
import platform
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)


class SleepInhibitor:
    """Prevent Mac from sleeping during active phone calls.

    Uses macOS `caffeinate` command. Safe to use on non-Mac platforms
    (methods become no-ops).
    """

    def __init__(self):
        self._process: Optional[subprocess.Popen] = None
        self._is_mac = platform.system() == "Darwin"

    def acquire(self) -> None:
        """Prevent sleep. Call when a phone call starts."""
        if not self._is_mac:
            return
        if self._process is not None:
            return  # Already acquired

        try:
            self._process = subprocess.Popen(
                ["caffeinate", "-d", "-i", "-s"],  # display + idle + system sleep
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            logger.debug("Sleep inhibitor acquired (pid=%d)", self._process.pid)
        except FileNotFoundError:
            logger.warning("caffeinate not found — sleep prevention unavailable")

    def release(self) -> None:
        """Allow sleep again. Call when a phone call ends."""
        if self._process is not None:
            self._process.terminate()
            self._process.wait()
            logger.debug("Sleep inhibitor released")
            self._process = None

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *args):
        self.release()


def check_ollama_running(host: str = "localhost", port: int = 11434) -> bool:
    """Check if Ollama is running and reachable.

    Args:
        host: Ollama host.
        port: Ollama port.

    Returns:
        True if Ollama is reachable.
    """
    import socket
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except (ConnectionRefusedError, socket.timeout, OSError):
        return False
```

---

## 11. CLI Entry Point

### `src/sip_voice_transport/cli/app.py`

```python
"""CLI entry point for the SIP voice server.

Usage:
    sip-voice-server                          # Start with defaults
    sip-voice-server --config path/to/config  # Custom config path
    sip-voice-server --port 8080              # Custom port
    sip-voice-server --provider telnyx        # Override provider
"""

import asyncio
import logging
import sys
from pathlib import Path

import click
import uvicorn
from fastapi import FastAPI

from sip_voice_transport.config import SipConfig
from sip_voice_transport.did_router import DIDRouter
from sip_voice_transport.webhook import SipWebhookHandler
from sip_voice_transport.transport import SipTransport
from sip_voice_transport.did_router import DIDConfig
from sip_voice_transport.mac_utils import SleepInhibitor, check_ollama_running


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def default_on_session(transport: SipTransport, config: DIDConfig | None) -> None:
    """Default session handler — logs audio and echoes back.

    This is the built-in handler for testing connectivity. Replace with
    your actual voice AI pipeline.
    """
    sleep = SleepInhibitor()
    sleep.acquire()

    try:
        logger.info("Call session started: %s", transport.metadata)
        chunk_count = 0
        async for audio_chunk in transport.receive_audio():
            chunk_count += 1
            if chunk_count % 50 == 0:  # Log every ~1 second (assuming 20ms chunks)
                logger.info("Received %d audio chunks (%d bytes each)", chunk_count, len(audio_chunk))

            # Echo mode: send audio back to the caller
            # Replace this with your STT → LLM → TTS pipeline
            await transport.send_audio(audio_chunk)

    except Exception as e:
        logger.exception("Error in call session: %s", e)
    finally:
        await transport.close()
        sleep.release()
        logger.info("Call session ended after %d chunks", chunk_count)


@click.command()
@click.option("--config", "-c", default="~/.vibecode/sip_config.yaml", help="Path to SIP config YAML")
@click.option("--port", "-p", default=8000, type=int, help="Port for the FastAPI server")
@click.option("--host", default="127.0.0.1", help="Host to bind to")
@click.option("--ws-url", default=None, help="Public WebSocket URL (e.g., wss://voice.example.com/sip/media-stream)")
@click.option("--log-level", default="INFO", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]))
def main(config: str, port: int, host: str, ws_url: str | None, log_level: str):
    """Start the SIP voice transport server."""
    logging.getLogger().setLevel(getattr(logging, log_level))

    # Load config
    config_path = Path(config).expanduser()
    logger.info("Loading config from %s", config_path)

    router = DIDRouter(str(config_path))
    logger.info("Loaded %d DID routes: %s", len(router.dids), router.dids)

    # Check Ollama
    if check_ollama_running():
        logger.info("Ollama is running")
    else:
        logger.warning("Ollama is NOT running — voice AI responses will fail")

    # Determine WebSocket URL
    if ws_url is None:
        ws_url = f"ws{'s' if port == 443 else ''}://{host}:{port}/sip/media-stream"
    logger.info("Public WebSocket URL: %s", ws_url)

    # Create FastAPI app
    app = FastAPI(title="SIP Voice Transport", version="0.1.0")

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "dids": router.dids,
            "ollama": check_ollama_running(),
        }

    # Register SIP routes
    handler = SipWebhookHandler(
        did_router=router,
        on_session=default_on_session,
        public_ws_url=ws_url,
    )
    handler.register(app)

    logger.info("Starting SIP voice server on %s:%d", host, port)
    uvicorn.run(app, host=host, port=port, log_level=log_level.lower())


if __name__ == "__main__":
    main()
```

---

## 12. Example Voice Server

### `examples/mac_voice_server.py`

```python
"""Example: Complete Mac-local voice server with STT → LLM → TTS.

This example shows how to wire sip-voice-transport with:
  - Whisper (faster-whisper) for STT
  - Ollama for LLM
  - Piper for TTS

Prerequisites:
  pip install sip-voice-transport[all]
  pip install faster-whisper
  # Install Ollama: https://ollama.com
  # Install Piper: pip install piper-tts

Usage:
  python examples/mac_voice_server.py --config ~/.vibecode/sip_config.yaml
"""

import asyncio
import io
import logging
from typing import Optional

import numpy as np
import uvicorn
from fastapi import FastAPI

from sip_voice_transport import (
    SipTransport,
    SipWebhookHandler,
    DIDRouter,
    DIDConfig,
    CANONICAL_SAMPLE_RATE,
)
from sip_voice_transport.mac_utils import SleepInhibitor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configure these ---
CONFIG_PATH = "~/.vibecode/sip_config.yaml"
PUBLIC_WS_URL = "wss://voice.yourdomain.com/sip/media-stream"  # Your CF tunnel URL
WHISPER_MODEL = "base"         # or "small", "medium"
OLLAMA_MODEL = "llama3.2:7b"   # or from DIDConfig
OLLAMA_URL = "http://localhost:11434"
# -----------------------


async def on_session(transport: SipTransport, config: Optional[DIDConfig]) -> None:
    """Handle a phone call: STT → LLM → TTS pipeline."""
    sleep = SleepInhibitor()
    sleep.acquire()

    try:
        logger.info("Call started: %s", transport.metadata)

        # Collect audio until silence, then transcribe
        audio_buffer = bytearray()
        silence_threshold = 500  # RMS threshold for silence detection
        silence_chunks = 0
        max_silence_chunks = 25  # ~500ms of silence at 20ms chunks

        async for chunk in transport.receive_audio():
            # Calculate RMS energy
            samples = np.frombuffer(chunk, dtype=np.int16)
            rms = np.sqrt(np.mean(samples.astype(np.float64) ** 2))

            if rms < silence_threshold:
                silence_chunks += 1
            else:
                silence_chunks = 0

            audio_buffer.extend(chunk)

            # If we have enough silence after speech, process the utterance
            if silence_chunks >= max_silence_chunks and len(audio_buffer) > CANONICAL_SAMPLE_RATE:
                # --- STT ---
                text = await transcribe(bytes(audio_buffer))
                audio_buffer.clear()
                silence_chunks = 0

                if not text or not text.strip():
                    continue

                logger.info("Transcribed: %s", text)

                # --- LLM ---
                system_prompt = config.system_prompt if config else "You are a helpful assistant."
                model = config.llm_model if config else OLLAMA_MODEL
                response_text = await llm_chat(text, system_prompt, model)
                logger.info("LLM response: %s", response_text)

                # --- TTS ---
                tts_audio = await synthesize(response_text)
                await transport.send_audio(tts_audio)

    except Exception as e:
        logger.exception("Error in call session: %s", e)
    finally:
        await transport.close()
        sleep.release()
        logger.info("Call ended")


async def transcribe(audio: bytes) -> str:
    """Transcribe audio using faster-whisper.

    Replace this with your preferred STT implementation.
    """
    try:
        from faster_whisper import WhisperModel
        # NOTE: In production, load the model once at startup, not per-call
        model = WhisperModel(WHISPER_MODEL, device="auto", compute_type="auto")
        audio_array = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _ = model.transcribe(audio_array, language="en")
        return " ".join(segment.text for segment in segments).strip()
    except ImportError:
        logger.error("faster-whisper not installed. pip install faster-whisper")
        return ""


async def llm_chat(user_message: str, system_prompt: str, model: str) -> str:
    """Chat with Ollama.

    Replace this with your preferred LLM implementation.
    """
    import httpx
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    "stream": False,
                },
                timeout=30.0,
            )
            response.raise_for_status()
            return response.json().get("message", {}).get("content", "")
    except Exception as e:
        logger.error("Ollama error: %s", e)
        return "I'm sorry, I encountered an error processing your request."


async def synthesize(text: str) -> bytes:
    """Synthesize speech using Piper TTS.

    Returns 16kHz 16-bit PCM audio.
    Replace this with your preferred TTS implementation.
    """
    try:
        # Piper TTS integration — adjust path to your piper model
        import subprocess
        result = subprocess.run(
            ["piper", "--model", "en_US-ryan-medium", "--output-raw"],
            input=text.encode(),
            capture_output=True,
            timeout=10,
        )
        return result.stdout
    except FileNotFoundError:
        logger.error("Piper TTS not installed")
        return b""
    except Exception as e:
        logger.error("TTS error: %s", e)
        return b""


# --- App setup ---
app = FastAPI(title="Mac Voice Server")
router = DIDRouter(CONFIG_PATH)
handler = SipWebhookHandler(
    did_router=router,
    on_session=on_session,
    public_ws_url=PUBLIC_WS_URL,
)
handler.register(app)


@app.get("/health")
async def health():
    return {"status": "ok", "dids": router.dids}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
```

---

## 13. Configuration

### `src/sip_voice_transport/config.py`

```python
"""Configuration loader for sip-voice-transport."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

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
        import re
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
```

### `.env.example`

```bash
# SIP Voice Transport — Environment Variables

# Provider selection: "telnyx" or "twilio"
SEARCH_PROVIDER=telnyx

# Telnyx credentials (get from Mission Control Portal)
TELNYX_API_KEY=KEY01234567890
# TELNYX_APP_ID=1494404757140276705

# Twilio credentials (get from Twilio Console)
# TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# TWILIO_AUTH_TOKEN=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Server config
# SIP_SERVER_PORT=8000
# SIP_PUBLIC_WS_URL=wss://voice.example.com/sip/media-stream
```

---

## 14. Tests

### `tests/conftest.py`

```python
"""Shared test fixtures."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from fastapi import WebSocket


@pytest.fixture
def mock_websocket():
    """Create a mock FastAPI WebSocket."""
    ws = AsyncMock(spec=WebSocket)
    ws.receive_json = AsyncMock()
    ws.send_json = AsyncMock()
    ws.accept = AsyncMock()
    ws.close = AsyncMock()
    return ws


@pytest.fixture
def telnyx_connected_msg():
    return {"event": "connected"}


@pytest.fixture
def telnyx_start_msg():
    return {
        "event": "start",
        "stream_id": "test-stream-123",
        "start": {
            "call_control_id": "test-call-456",
            "custom_parameters": {
                "caller_id": "+15125559999",
                "did": "+15125551234",
            },
            "media_format": {
                "encoding": "audio/l16",
                "sample_rate": 16000,
            },
        },
    }


@pytest.fixture
def twilio_connected_msg():
    return {"event": "connected", "protocol": "Call", "version": "1.0.0"}


@pytest.fixture
def twilio_start_msg():
    return {
        "event": "start",
        "sequenceNumber": "1",
        "start": {
            "streamSid": "MZtest123",
            "callSid": "CAtest456",
            "accountSid": "ACtest789",
            "tracks": ["inbound"],
            "mediaFormat": {
                "encoding": "audio/x-mulaw",
                "sampleRate": 8000,
                "channels": 1,
            },
            "customParameters": {
                "caller_id": "+15125559999",
                "did": "+15125551234",
            },
        },
        "streamSid": "MZtest123",
    }
```

### `tests/test_audio_codec.py`

```python
"""Tests for AudioCodec, mulaw encode/decode, and resampling."""

import numpy as np
import pytest

from sip_voice_transport.audio_codec import (
    AudioCodec,
    mulaw_decode,
    mulaw_encode,
    resample,
    base64_decode_audio,
    base64_encode_audio,
)


class TestMulawCodec:
    def test_mulaw_decode_produces_correct_length(self):
        """mulaw decode: 100 bytes in → 200 bytes out (1 byte → 2 bytes per sample)."""
        mulaw_data = bytes(range(100))
        pcm = mulaw_decode(mulaw_data)
        assert len(pcm) == 200

    def test_mulaw_roundtrip(self):
        """Encode then decode should produce approximately the same signal."""
        # Generate a simple sine wave as 16-bit PCM
        t = np.linspace(0, 0.01, 160, dtype=np.float64)
        signal = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)
        pcm_bytes = signal.tobytes()

        encoded = mulaw_encode(pcm_bytes)
        decoded = mulaw_decode(encoded)

        original = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float64)
        recovered = np.frombuffer(decoded, dtype=np.int16).astype(np.float64)

        # mulaw is lossy — allow ~2% error
        max_error = np.max(np.abs(original - recovered))
        assert max_error < 1000, f"Max roundtrip error too high: {max_error}"

    def test_mulaw_silence(self):
        """Silence (zeros) should roundtrip reasonably."""
        silence = np.zeros(100, dtype=np.int16).tobytes()
        encoded = mulaw_encode(silence)
        decoded = mulaw_decode(encoded)
        samples = np.frombuffer(decoded, dtype=np.int16)
        assert np.max(np.abs(samples)) < 200  # Near silence


class TestResample:
    def test_resample_8k_to_16k_doubles_length(self):
        """Upsampling 8kHz → 16kHz should roughly double the number of samples."""
        pcm_8k = np.zeros(800, dtype=np.int16).tobytes()  # 100ms at 8kHz
        pcm_16k = resample(pcm_8k, 8000, 16000)
        # Should have ~1600 samples = 3200 bytes
        assert abs(len(pcm_16k) - 3200) < 8  # Allow small rounding

    def test_resample_16k_to_8k_halves_length(self):
        pcm_16k = np.zeros(1600, dtype=np.int16).tobytes()
        pcm_8k = resample(pcm_16k, 16000, 8000)
        assert abs(len(pcm_8k) - 1600) < 8

    def test_resample_same_rate_is_noop(self):
        original = np.arange(100, dtype=np.int16).tobytes()
        result = resample(original, 16000, 16000)
        assert result == original


class TestAudioCodec:
    def test_noop_for_l16_16k(self):
        """Telnyx L16/16kHz → canonical should be a no-op."""
        codec = AudioCodec(source_rate=16000, source_encoding="l16")
        assert codec._is_noop is True
        data = b"\x01\x02\x03\x04"
        assert codec.decode_to_canonical(data) == data
        assert codec.encode_from_canonical(data) == data

    def test_mulaw_8k_requires_conversion(self):
        """Twilio mulaw/8kHz → canonical requires both decode and resample."""
        codec = AudioCodec(source_rate=8000, source_encoding="mulaw")
        assert codec._is_noop is False
        assert codec._needs_resample is True
        assert codec._needs_decode is True


class TestBase64:
    def test_roundtrip(self):
        data = b"\x00\x01\x02\xff"
        encoded = base64_encode_audio(data)
        decoded = base64_decode_audio(encoded)
        assert decoded == data
```

### `tests/test_did_router.py`

```python
"""Tests for DIDRouter."""

import pytest
from pathlib import Path
from sip_voice_transport.did_router import DIDRouter, DIDConfig


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
```

### `tests/test_telnyx_provider.py`

```python
"""Tests for TelnyxProvider."""

import pytest
from unittest.mock import AsyncMock
from fastapi import WebSocket

from sip_voice_transport.providers.telnyx_provider import TelnyxProvider
from sip_voice_transport.exceptions import ProviderConnectionError


class TestTelnyxProvider:
    @pytest.mark.asyncio
    async def test_connect_success(self, mock_websocket, telnyx_connected_msg, telnyx_start_msg):
        mock_websocket.receive_json = AsyncMock(return_value=telnyx_start_msg)

        provider = TelnyxProvider()
        await provider.connect(mock_websocket, telnyx_connected_msg)

        assert provider.stream_id == "test-stream-123"
        assert provider.provider_name == "telnyx"
        assert provider.call_metadata["did"] == "+15125551234"

    @pytest.mark.asyncio
    async def test_connect_wrong_first_event(self, mock_websocket):
        provider = TelnyxProvider()
        with pytest.raises(ProviderConnectionError):
            await provider.connect(mock_websocket, {"event": "wrong"})

    @pytest.mark.asyncio
    async def test_send_audio_frame(self, mock_websocket, telnyx_connected_msg, telnyx_start_msg):
        mock_websocket.receive_json = AsyncMock(return_value=telnyx_start_msg)

        provider = TelnyxProvider()
        await provider.connect(mock_websocket, telnyx_connected_msg)
        await provider.send_audio_frame(b"\x00\x01")

        mock_websocket.send_json.assert_called_once()
        sent = mock_websocket.send_json.call_args[0][0]
        assert sent["event"] == "media"
        assert sent["stream_id"] == "test-stream-123"

    def test_native_format_defaults(self):
        provider = TelnyxProvider()
        assert provider.native_sample_rate == 16000
        assert provider.native_encoding == "l16"
```

### `tests/test_twilio_provider.py`

```python
"""Tests for TwilioProvider."""

import pytest
from unittest.mock import AsyncMock

from sip_voice_transport.providers.twilio_provider import TwilioProvider
from sip_voice_transport.exceptions import ProviderConnectionError


class TestTwilioProvider:
    @pytest.mark.asyncio
    async def test_connect_success(self, mock_websocket, twilio_connected_msg, twilio_start_msg):
        mock_websocket.receive_json = AsyncMock(return_value=twilio_start_msg)

        provider = TwilioProvider()
        await provider.connect(mock_websocket, twilio_connected_msg)

        assert provider.stream_id == "MZtest123"
        assert provider.provider_name == "twilio"
        assert provider.native_sample_rate == 8000
        assert provider.native_encoding == "mulaw"

    @pytest.mark.asyncio
    async def test_send_uses_stream_sid(self, mock_websocket, twilio_connected_msg, twilio_start_msg):
        mock_websocket.receive_json = AsyncMock(return_value=twilio_start_msg)

        provider = TwilioProvider()
        await provider.connect(mock_websocket, twilio_connected_msg)
        await provider.send_audio_frame(b"\x00\x01")

        sent = mock_websocket.send_json.call_args[0][0]
        assert sent["streamSid"] == "MZtest123"
```

Add similar pattern tests for `test_transport.py`, `test_webhook.py`, `test_config.py`.

---

## 15. README

Create a README.md modeled on search-tool-provider's style. Key sections:

1. One-liner description + code snippet
2. Quick Start (install, create config, run)
3. Architecture diagram (ASCII)
4. Provider comparison table
5. Configuration reference
6. Example voice server
7. API Reference (BaseTransport, BaseSipProvider)
8. Development setup
9. License (MIT)

---

## 16. Build Order

**Instruct Claude Code to build in this exact order:**

1. Create repo structure (all directories)
2. `pyproject.toml`
3. `exceptions.py`
4. `base_transport.py`
5. `base_sip_provider.py`
6. `audio_codec.py` + `tests/test_audio_codec.py` → run tests
7. `did_router.py` + `tests/test_did_router.py` → run tests
8. `config.py` + `tests/test_config.py` → run tests
9. `providers/__init__.py`
10. `providers/telnyx_provider.py` + `tests/test_telnyx_provider.py` → run tests
11. `providers/twilio_provider.py` + `tests/test_twilio_provider.py` → run tests
12. `transport.py` + `tests/test_transport.py` → run tests
13. `webhook.py` + `tests/test_webhook.py` → run tests
14. `mac_utils.py`
15. `cli/app.py`
16. `__init__.py`
17. `examples/mac_voice_server.py`
18. `.env.example`, `.gitignore`, `LICENSE`, `README.md`
19. Run full test suite: `python -m pytest tests/ -v`
20. Run ruff: `ruff check src/ tests/`

**After each file, run its associated tests before moving to the next file.**
**If tests fail, fix before proceeding.**
