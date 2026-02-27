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
