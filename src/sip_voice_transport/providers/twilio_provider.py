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
mulaw<->PCM conversion and 8kHz<->16kHz resampling for Twilio.
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
        if self._websocket is None or self._stream_sid is None:
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
        if self._websocket is None or self._stream_sid is None:
            raise ProviderConnectionError("Not connected")

        message = {
            "event": "mark",
            "streamSid": self._stream_sid,
            "mark": {"name": name},
        }
        await self._websocket.send_json(message)

    async def clear_audio(self) -> None:
        if self._websocket is None or self._stream_sid is None:
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
