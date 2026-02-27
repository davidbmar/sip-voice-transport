"""Shared test fixtures."""

import pytest
from unittest.mock import AsyncMock
from fastapi import WebSocket


@pytest.fixture
def mock_websocket():
    """Create a mock FastAPI WebSocket."""
    ws = AsyncMock(spec=WebSocket)
    ws.receive_json = AsyncMock()
    ws.send_json = AsyncMock()
    ws.accept = AsyncMock()
    ws.close = AsyncMock()
    return ws


@pytest.fixture
def telnyx_connected_msg():
    return {"event": "connected"}


@pytest.fixture
def telnyx_start_msg():
    return {
        "event": "start",
        "stream_id": "test-stream-123",
        "start": {
            "call_control_id": "test-call-456",
            "custom_parameters": {
                "caller_id": "+15125559999",
                "did": "+15125551234",
            },
            "media_format": {
                "encoding": "audio/l16",
                "sample_rate": 16000,
            },
        },
    }


@pytest.fixture
def twilio_connected_msg():
    return {"event": "connected", "protocol": "Call", "version": "1.0.0"}


@pytest.fixture
def twilio_start_msg():
    return {
        "event": "start",
        "sequenceNumber": "1",
        "start": {
            "streamSid": "MZtest123",
            "callSid": "CAtest456",
            "accountSid": "ACtest789",
            "tracks": ["inbound"],
            "mediaFormat": {
                "encoding": "audio/x-mulaw",
                "sampleRate": 8000,
                "channels": 1,
            },
            "customParameters": {
                "caller_id": "+15125559999",
                "did": "+15125551234",
            },
        },
        "streamSid": "MZtest123",
    }
