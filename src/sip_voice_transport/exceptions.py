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
