"""Example: Complete Mac-local voice server with STT -> LLM -> TTS.

This example shows how to wire sip-voice-transport with:
  - Whisper (faster-whisper) for STT
  - Ollama for LLM
  - Piper for TTS

Prerequisites:
  pip install sip-voice-transport[all]
  pip install faster-whisper
  # Install Ollama: https://ollama.com
  # Install Piper: pip install piper-tts

Usage:
  python examples/mac_voice_server.py --config ~/.vibecode/sip_config.yaml
"""

import asyncio
import logging
from typing import Optional

import numpy as np
import uvicorn
from fastapi import FastAPI

from sip_voice_transport import (
    SipTransport,
    SipWebhookHandler,
    DIDRouter,
    DIDConfig,
    CANONICAL_SAMPLE_RATE,
)
from sip_voice_transport.mac_utils import SleepInhibitor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configure these ---
CONFIG_PATH = "~/.vibecode/sip_config.yaml"
PUBLIC_WS_URL = "wss://voice.yourdomain.com/sip/media-stream"  # Your CF tunnel URL
WHISPER_MODEL = "base"         # or "small", "medium"
OLLAMA_MODEL = "llama3.2:7b"   # or from DIDConfig
OLLAMA_URL = "http://localhost:11434"
# -----------------------


async def on_session(transport: SipTransport, config: Optional[DIDConfig]) -> None:
    """Handle a phone call: STT -> LLM -> TTS pipeline."""
    sleep = SleepInhibitor()
    sleep.acquire()

    try:
        logger.info("Call started: %s", transport.metadata)

        audio_buffer = bytearray()
        silence_threshold = 500
        silence_chunks = 0
        max_silence_chunks = 25  # ~500ms of silence at 20ms chunks

        async for chunk in transport.receive_audio():
            samples = np.frombuffer(chunk, dtype=np.int16)
            rms = np.sqrt(np.mean(samples.astype(np.float64) ** 2))

            if rms < silence_threshold:
                silence_chunks += 1
            else:
                silence_chunks = 0

            audio_buffer.extend(chunk)

            if silence_chunks >= max_silence_chunks and len(audio_buffer) > CANONICAL_SAMPLE_RATE:
                text = await transcribe(bytes(audio_buffer))
                audio_buffer.clear()
                silence_chunks = 0

                if not text or not text.strip():
                    continue

                logger.info("Transcribed: %s", text)

                system_prompt = config.system_prompt if config else "You are a helpful assistant."
                model = config.llm_model if config else OLLAMA_MODEL
                response_text = await llm_chat(text, system_prompt, model)
                logger.info("LLM response: %s", response_text)

                tts_audio = await synthesize(response_text)
                await transport.send_audio(tts_audio)

    except Exception as e:
        logger.exception("Error in call session: %s", e)
    finally:
        await transport.close()
        sleep.release()
        logger.info("Call ended")


async def transcribe(audio: bytes) -> str:
    """Transcribe audio using faster-whisper."""
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel(WHISPER_MODEL, device="auto", compute_type="auto")
        audio_array = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _ = model.transcribe(audio_array, language="en")
        return " ".join(segment.text for segment in segments).strip()
    except ImportError:
        logger.error("faster-whisper not installed. pip install faster-whisper")
        return ""


async def llm_chat(user_message: str, system_prompt: str, model: str) -> str:
    """Chat with Ollama."""
    import httpx
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    "stream": False,
                },
                timeout=30.0,
            )
            response.raise_for_status()
            return response.json().get("message", {}).get("content", "")
    except Exception as e:
        logger.error("Ollama error: %s", e)
        return "I'm sorry, I encountered an error processing your request."


async def synthesize(text: str) -> bytes:
    """Synthesize speech using Piper TTS. Returns 16kHz 16-bit PCM audio."""
    try:
        import subprocess
        result = subprocess.run(
            ["piper", "--model", "en_US-ryan-medium", "--output-raw"],
            input=text.encode(),
            capture_output=True,
            timeout=10,
        )
        return result.stdout
    except FileNotFoundError:
        logger.error("Piper TTS not installed")
        return b""
    except Exception as e:
        logger.error("TTS error: %s", e)
        return b""


# --- App setup ---
app = FastAPI(title="Mac Voice Server")
router = DIDRouter(CONFIG_PATH)
handler = SipWebhookHandler(
    did_router=router,
    on_session=on_session,
    public_ws_url=PUBLIC_WS_URL,
)
handler.register(app)


@app.get("/health")
async def health():
    return {"status": "ok", "dids": router.dids}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
