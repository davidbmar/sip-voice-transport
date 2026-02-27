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
        if self._websocket is None:
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

                elif event == "mark":
                    name = message.get("mark", {}).get("name")
                    logger.debug("Telnyx: mark received: %s", name)

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
        if self._websocket is None or self._stream_id is None:
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
        if self._websocket is None or self._stream_id is None:
            raise ProviderConnectionError("Not connected")

        message = {
            "event": "mark",
            "stream_id": self._stream_id,
            "mark": {"name": name},
        }
        await self._websocket.send_json(message)

    async def clear_audio(self) -> None:
        if self._websocket is None or self._stream_id is None:
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
