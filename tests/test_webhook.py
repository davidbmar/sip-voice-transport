"""Tests for SipWebhookHandler."""

import pytest
from unittest.mock import AsyncMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from sip_voice_transport.webhook import SipWebhookHandler
from sip_voice_transport.did_router import DIDRouter


@pytest.fixture
def sample_config(tmp_path):
    config = tmp_path / "sip_config.yaml"
    config.write_text("""
dids:
  "+15125551234":
    user_id: david
    display_name: "Test Mac"
""")
    return str(config)


@pytest.fixture
def app_with_handler(sample_config):
    app = FastAPI()
    router = DIDRouter(sample_config)

    on_session = AsyncMock()

    handler = SipWebhookHandler(
        did_router=router,
        on_session=on_session,
        public_ws_url="wss://test.example.com/sip/media-stream",
    )
    handler.register(app)
    return app, on_session


class TestTelnyxWebhook:
    def test_telnyx_answer_configured_did(self, app_with_handler):
        app, _ = app_with_handler
        client = TestClient(app)

        response = client.post(
            "/sip/telnyx/answer",
            data={"To": "+15125551234", "From": "+15125559999", "CallSid": "test123"},
        )
        assert response.status_code == 200
        assert "application/xml" in response.headers["content-type"]
        assert "<Stream" in response.text
        assert "wss://test.example.com/sip/media-stream" in response.text
        assert "L16" in response.text

    def test_telnyx_answer_unconfigured_did(self, app_with_handler):
        app, _ = app_with_handler
        client = TestClient(app)

        response = client.post(
            "/sip/telnyx/answer",
            data={"To": "+19999999999", "From": "+15125559999", "CallSid": "test123"},
        )
        assert response.status_code == 200
        assert "<Hangup" in response.text


class TestTwilioWebhook:
    def test_twilio_answer_configured_did(self, app_with_handler):
        app, _ = app_with_handler
        client = TestClient(app)

        response = client.post(
            "/sip/twilio/answer",
            data={"To": "+15125551234", "From": "+15125559999", "CallSid": "test456"},
        )
        assert response.status_code == 200
        assert "<Connect>" in response.text
        assert "<Stream" in response.text

    def test_twilio_answer_unconfigured_did(self, app_with_handler):
        app, _ = app_with_handler
        client = TestClient(app)

        response = client.post(
            "/sip/twilio/answer",
            data={"To": "+19999999999", "From": "+15125559999", "CallSid": "test456"},
        )
        assert response.status_code == 200
        assert "<Hangup" in response.text


class TestTeXMLGeneration:
    def test_texml_contains_caller_params(self, app_with_handler):
        app, _ = app_with_handler
        client = TestClient(app)

        response = client.post(
            "/sip/telnyx/answer",
            data={"To": "+15125551234", "From": "+15125559999", "CallSid": "test"},
        )
        assert 'name="caller_id"' in response.text
        assert 'name="did"' in response.text
