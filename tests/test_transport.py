"""Tests for SipTransport."""

import pytest
from unittest.mock import AsyncMock, PropertyMock

from sip_voice_transport.transport import SipTransport


def make_mock_provider(name="telnyx", sample_rate=16000, encoding="l16", stream_id="test-stream"):
    """Create a mock BaseSipProvider."""
    provider = AsyncMock()
    type(provider).provider_name = PropertyMock(return_value=name)
    type(provider).native_sample_rate = PropertyMock(return_value=sample_rate)
    type(provider).native_encoding = PropertyMock(return_value=encoding)
    type(provider).stream_id = PropertyMock(return_value=stream_id)
    type(provider).call_metadata = PropertyMock(return_value={
        "caller_id": "+15125559999",
        "did": "+15125551234",
        "call_sid": "test-call-123",
        "stream_id": stream_id,
    })
    return provider


class TestSipTransport:
    def test_metadata_includes_transport_type(self):
        provider = make_mock_provider()
        transport = SipTransport(provider)
        assert transport.metadata["transport"] == "sip"
        assert transport.metadata["provider"] == "telnyx"

    def test_is_connected_when_stream_id_exists(self):
        provider = make_mock_provider(stream_id="abc")
        transport = SipTransport(provider)
        assert transport.is_connected is True

    def test_is_not_connected_when_stream_id_none(self):
        provider = make_mock_provider(stream_id=None)
        transport = SipTransport(provider)
        assert transport.is_connected is False

    @pytest.mark.asyncio
    async def test_send_audio_delegates_to_provider(self):
        provider = make_mock_provider()
        transport = SipTransport(provider)

        await transport.send_audio(b"\x00\x01\x02\x03")
        provider.send_audio_frame.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_mark_delegates(self):
        provider = make_mock_provider()
        transport = SipTransport(provider)

        await transport.send_mark("mark-1")
        provider.send_mark.assert_called_once_with("mark-1")

    @pytest.mark.asyncio
    async def test_clear_audio_delegates(self):
        provider = make_mock_provider()
        transport = SipTransport(provider)

        await transport.clear_audio()
        provider.clear_audio.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_delegates(self):
        provider = make_mock_provider()
        transport = SipTransport(provider)

        await transport.close()
        provider.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_receive_audio_yields_canonical(self):
        """For L16/16kHz provider, audio passes through unchanged."""
        provider = make_mock_provider()

        async def fake_frames():
            yield b"\x01\x02\x03\x04"
            yield b"\x05\x06\x07\x08"

        provider.receive_audio_frames = fake_frames
        transport = SipTransport(provider)

        chunks = []
        async for chunk in transport.receive_audio():
            chunks.append(chunk)

        assert len(chunks) == 2
        assert chunks[0] == b"\x01\x02\x03\x04"

    def test_telnyx_l16_codec_is_noop(self):
        provider = make_mock_provider(name="telnyx", sample_rate=16000, encoding="l16")
        transport = SipTransport(provider)
        assert transport._codec._is_noop is True

    def test_twilio_mulaw_codec_converts(self):
        provider = make_mock_provider(name="twilio", sample_rate=8000, encoding="mulaw")
        transport = SipTransport(provider)
        assert transport._codec._is_noop is False
