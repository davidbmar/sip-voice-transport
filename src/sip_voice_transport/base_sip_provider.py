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
