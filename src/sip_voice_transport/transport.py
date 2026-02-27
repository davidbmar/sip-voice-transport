"""Provider-agnostic SIP transport.

Wraps a BaseSipProvider and handles:
  - Audio format conversion (provider native <-> canonical 16kHz PCM)
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
