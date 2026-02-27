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
