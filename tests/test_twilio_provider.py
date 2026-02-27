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

    @pytest.mark.asyncio
    async def test_connect_wrong_first_event(self, mock_websocket):
        provider = TwilioProvider()
        with pytest.raises(ProviderConnectionError):
            await provider.connect(mock_websocket, {"event": "wrong"})

    @pytest.mark.asyncio
    async def test_call_metadata(self, mock_websocket, twilio_connected_msg, twilio_start_msg):
        mock_websocket.receive_json = AsyncMock(return_value=twilio_start_msg)

        provider = TwilioProvider()
        await provider.connect(mock_websocket, twilio_connected_msg)

        meta = provider.call_metadata
        assert meta["call_sid"] == "CAtest456"
        assert meta["did"] == "+15125551234"
        assert meta["caller_id"] == "+15125559999"
        assert meta["account_sid"] == "ACtest789"
