"""CLI entry point for the SIP voice server.

Usage:
    sip-voice-server                          # Start with defaults
    sip-voice-server --config path/to/config  # Custom config path
    sip-voice-server --port 8080              # Custom port
    sip-voice-server --provider telnyx        # Override provider
"""

import logging
import signal
import subprocess
import os
from pathlib import Path

import click
import uvicorn
from fastapi import FastAPI

from sip_voice_transport.did_router import DIDRouter
from sip_voice_transport.webhook import SipWebhookHandler
from sip_voice_transport.transport import SipTransport
from sip_voice_transport.did_router import DIDConfig
from sip_voice_transport.mac_utils import SleepInhibitor, check_ollama_running
from sip_voice_transport.dashboard import register_dashboard, track_call_start, track_call_end, track_audio


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _kill_stale_server(port: int) -> None:
    """Kill any existing process on the port before binding."""
    try:
        pids = subprocess.check_output(["lsof", "-ti", f":{port}"], text=True).strip()
        for pid in pids.splitlines():
            pid = int(pid)
            if pid != os.getpid():
                os.kill(pid, signal.SIGTERM)
                logger.info("Killed stale process %d on port %d", pid, port)
    except (subprocess.CalledProcessError, OSError):
        pass


async def default_on_session(transport: SipTransport, config: DIDConfig | None) -> None:
    """Default session handler — logs audio and echoes back.

    This is the built-in handler for testing connectivity. Replace with
    your actual voice AI pipeline.
    """
    sleep = SleepInhibitor()
    sleep.acquire()

    meta = transport.metadata
    stream_id = meta.get("stream_id", "unknown")
    track_call_start(stream_id, meta.get("provider", ""), meta.get("caller_id", ""), meta.get("did", ""))

    try:
        logger.info("Call session started: %s", meta)
        chunk_count = 0
        async for audio_chunk in transport.receive_audio():
            chunk_count += 1
            track_audio(stream_id, "received")
            if chunk_count % 50 == 0:  # Log every ~1 second (assuming 20ms chunks)
                logger.info("Received %d audio chunks (%d bytes each)", chunk_count, len(audio_chunk))

            # Echo mode: send audio back to the caller
            await transport.send_audio(audio_chunk)
            track_audio(stream_id, "sent")

    except Exception as e:
        logger.exception("Error in call session: %s", e)
    finally:
        track_call_end(stream_id)
        await transport.close()
        sleep.release()
        logger.info("Call session ended after %d chunks", chunk_count)


@click.command()
@click.option("--config", "-c", default="~/.vibecode/sip_config.yaml", help="Path to SIP config YAML")
@click.option("--port", "-p", default=8000, type=int, help="Port for the FastAPI server")
@click.option("--host", default="127.0.0.1", help="Host to bind to")
@click.option("--ws-url", default=None, help="Public WebSocket URL (e.g., wss://voice.example.com/sip/media-stream)")
@click.option("--log-level", default="INFO", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]))
def main(config: str, port: int, host: str, ws_url: str | None, log_level: str):
    """Start the SIP voice transport server."""
    logging.getLogger().setLevel(getattr(logging, log_level))

    # Kill any stale process on the port
    _kill_stale_server(port)

    # Load config
    config_path = Path(config).expanduser()
    logger.info("Loading config from %s", config_path)

    router = DIDRouter(str(config_path))
    logger.info("Loaded %d DID routes: %s", len(router.dids), router.dids)

    # Check Ollama
    if check_ollama_running():
        logger.info("Ollama is running")
    else:
        logger.warning("Ollama is NOT running — voice AI responses will fail")

    # Determine WebSocket URL
    if ws_url is None:
        ws_url = f"ws{'s' if port == 443 else ''}://{host}:{port}/sip/media-stream"
    logger.info("Public WebSocket URL: %s", ws_url)

    # Create FastAPI app
    app = FastAPI(title="SIP Voice Transport", version="0.1.0")

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "dids": router.dids,
            "ollama": check_ollama_running(),
        }

    # Register SIP routes
    handler = SipWebhookHandler(
        did_router=router,
        on_session=default_on_session,
        public_ws_url=ws_url,
    )
    handler.register(app)

    # Register dashboard UI
    register_dashboard(app, router)

    # Graceful shutdown
    def handle_signal(signum, frame):
        logger.info("Received signal %d, shutting down...", signum)
        raise SystemExit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    logger.info("Starting SIP voice server on %s:%d", host, port)
    uvicorn.run(app, host=host, port=port, log_level=log_level.lower())


if __name__ == "__main__":
    main()
