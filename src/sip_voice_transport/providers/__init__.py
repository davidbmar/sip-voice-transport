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
