"""FastAPI webhook handler for SIP providers.

Registers three routes:
  POST /sip/telnyx/answer  — Telnyx inbound call webhook, returns TeXML
  POST /sip/twilio/answer  — Twilio inbound call webhook, returns TwiML
  WSS  /sip/media-stream   — WebSocket endpoint for audio streaming (both providers)

The webhook handler creates provider instances, wraps them in SipTransport,
and hands them off to the application's on_session callback.
"""

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
