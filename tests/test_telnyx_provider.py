"""Tests for TelnyxProvider."""

import pytest
from unittest.mock import AsyncMock

from sip_voice_transport.providers.telnyx_provider import TelnyxProvider
from sip_voice_transport.exceptions import ProviderConnectionError


class TestTelnyxProvider:
    @pytest.mark.asyncio
    async def test_connect_success(self, mock_websocket, telnyx_connected_msg, telnyx_start_msg):
        mock_websocket.receive_json = AsyncMock(return_value=telnyx_start_msg)

        provider = TelnyxProvider()
        await provider.connect(mock_websocket, telnyx_connected_msg)

        assert provider.stream_id == "test-stream-123"
        assert provider.provider_name == "telnyx"
        assert provider.call_metadata["did"] == "+15125551234"

    @pytest.mark.asyncio
    async def test_connect_wrong_first_event(self, mock_websocket):
        provider = TelnyxProvider()
        with pytest.raises(ProviderConnectionError):
            await provider.connect(mock_websocket, {"event": "wrong"})

    @pytest.mark.asyncio
    async def test_send_audio_frame(self, mock_websocket, telnyx_connected_msg, telnyx_start_msg):
        mock_websocket.receive_json = AsyncMock(return_value=telnyx_start_msg)

        provider = TelnyxProvider()
        await provider.connect(mock_websocket, telnyx_connected_msg)
        await provider.send_audio_frame(b"\x00\x01")

        mock_websocket.send_json.assert_called_once()
        sent = mock_websocket.send_json.call_args[0][0]
        assert sent["event"] == "media"
        assert sent["stream_id"] == "test-stream-123"

    def test_native_format_defaults(self):
        provider = TelnyxProvider()
        assert provider.native_sample_rate == 16000
        assert provider.native_encoding == "l16"
