# sip-voice-transport

PSTN phone call transport for Mac-local voice AI. Connect real phone calls to your local AI pipeline over Telnyx or Twilio.

```
Phone Call → Telnyx/Twilio → WebSocket → sip-voice-transport → Your AI Pipeline
                                              ↕
                                    16kHz PCM audio (canonical)
```

## What It Does

This package handles the telephony plumbing so you can focus on voice AI logic. It:

- **Receives inbound calls** via Telnyx or Twilio webhooks
- **Streams audio** bidirectionally over WebSocket
- **Converts codecs** automatically (Twilio's mulaw/8kHz ↔ canonical 16kHz PCM)
- **Routes DIDs** to per-number configs (LLM model, system prompt, TTS voice)
- **Prevents Mac sleep** during active calls via `caffeinate`

Your code just works with a clean async iterator:

```python
async for audio_chunk in transport.receive_audio():
    # audio_chunk is always 16kHz, 16-bit, mono PCM
    transcription = await stt(audio_chunk)
    response = await llm(transcription)
    tts_audio = await tts(response)
    await transport.send_audio(tts_audio)
```

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                    Your Application                   │
│         (STT → LLM → TTS pipeline, etc.)             │
└──────────────▲───────────────────────│───────────────┘
               │  receive_audio()      │  send_audio()
               │  (16kHz PCM)          ▼  (16kHz PCM)
┌──────────────│───────────────────────▼───────────────┐
│                   SipTransport                        │
│            (BaseTransport interface)                  │
│         Automatic codec conversion here               │
└──────────────▲───────────────────────│───────────────┘
               │                       │
┌──────────────│───────────────────────▼───────────────┐
│              BaseSipProvider (ABC)                     │
│    ┌─────────────────┐  ┌─────────────────────┐      │
│    │ TelnyxProvider   │  │   TwilioProvider    │      │
│    │ L16/16kHz (noop) │  │ mulaw/8kHz (convert)│      │
│    └─────────────────┘  └─────────────────────┘      │
└──────────────▲───────────────────────│───────────────┘
               │  WebSocket            │  WebSocket
┌──────────────│───────────────────────▼───────────────┐
│              SipWebhookHandler (FastAPI)               │
│  POST /sip/telnyx/answer  → TeXML                     │
│  POST /sip/twilio/answer  → TwiML                     │
│  WSS  /sip/media-stream   → Audio streaming           │
└──────────────────────────────────────────────────────┘
```

## Quickstart

### 1. Install

Requires **Python 3.11+**. Check your version with `python3 --version`.

```bash
# Create a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install the package
pip install -e ".[cli,dev]"
```

> **Note:** macOS ships with Python 3.9 which is too old. Use `brew install python@3.13` or similar to get a supported version.

### 2. Create a DID config

Create `~/.vibecode/sip_config.yaml`:

```yaml
dids:
  "+15125551234":          # Your Telnyx phone number
    user_id: david
    display_name: "My Mac"
    llm_model: "llama3.2:7b"
    system_prompt: "You are a helpful voice assistant."
    tts_voice: "en-us-ryan-medium"
```

### 3. Start the server

```bash
sip-voice-server --port 8765
```

### 4. Test with the simulator

```bash
python test_call.py --port 8765
```

This simulates a complete Telnyx inbound call: webhook POST, WebSocket handshake, and bidirectional audio streaming.

### 5. Go live with Telnyx

```bash
# Expose your server to the internet
cloudflared tunnel --url http://localhost:8765

# Set the tunnel URL as your Telnyx webhook:
#   Voice → Connection: webhook
#   URL: https://<tunnel-id>.cfargotunnel.com/sip/telnyx/answer
```

Call your Telnyx number — you'll hear your audio echoed back (default handler). Replace with your voice AI pipeline to make it useful.

## Provider Comparison

| Feature | Telnyx | Twilio |
|---------|--------|--------|
| Native codec | L16 (16kHz PCM) | mulaw (8kHz) |
| Conversion needed | None (matches canonical) | mulaw↔PCM + 8k↔16k resample |
| Latency | Lower (no conversion) | Higher (codec conversion) |
| Bidirectional audio | Via TeXML `bidirectionalMode="rtp"` | Via TwiML `<Connect><Stream>` |
| Cost | Generally lower | Generally higher |
| Recommendation | **Preferred** for this use case | Supported but less optimal |

## Configuration Reference

### DID Config (`sip_config.yaml`)

```yaml
dids:
  "+15125551234":                           # E.164 phone number
    user_id: david                          # Unique user identifier
    display_name: "David's Mac"             # Human-readable name
    llm_model: "llama3.2:7b"               # Ollama model name
    system_prompt: "You are a helpful..."   # System prompt for LLM
    tts_voice: "en-us-ryan-medium"          # Piper TTS voice
    # Any extra keys are preserved in config.extra dict

  "+15125555678":
    user_id: alice
    display_name: "Alice's Mac"
    llm_model: "llama3.2:7b"
    system_prompt: "You are Alice's assistant."
```

### Server Config (`SipConfig`)

Loaded from YAML with `${ENV_VAR}` expansion:

```yaml
provider: telnyx                             # Default provider
server_host: "127.0.0.1"
server_port: 8000
public_ws_url: "wss://voice.example.com/sip/media-stream"
log_level: INFO
provider_config:
  telnyx:
    api_key: "${TELNYX_API_KEY}"
    app_id: "your-app-id"
  twilio:
    account_sid: "${TWILIO_ACCOUNT_SID}"
    auth_token: "${TWILIO_AUTH_TOKEN}"
```

## CLI Reference

```
sip-voice-server [OPTIONS]

Options:
  -c, --config PATH        Path to SIP config YAML [default: ~/.vibecode/sip_config.yaml]
  -p, --port INTEGER       Server port [default: 8000]
  --host TEXT               Bind host [default: 127.0.0.1]
  --ws-url TEXT             Public WebSocket URL (auto-generated if not set)
  --log-level [DEBUG|INFO|WARNING|ERROR]  [default: INFO]
  --help                   Show this message and exit
```

## API Reference

### Core Classes

**`SipTransport`** — the main interface your code uses:

```python
transport = SipTransport(provider)

async for chunk in transport.receive_audio():  # 16kHz PCM bytes
    ...
await transport.send_audio(pcm_bytes)          # Send 16kHz PCM
await transport.send_mark("utterance-1")       # Mark for playback tracking
await transport.clear_audio()                  # Clear queued audio (barge-in)
await transport.close()                        # Hang up

transport.is_connected  # bool
transport.metadata      # {"provider": "telnyx", "caller_id": "+1...", "did": "+1..."}
```

**`SipWebhookHandler`** — registers routes on your FastAPI app:

```python
handler = SipWebhookHandler(
    did_router=DIDRouter("config.yaml"),
    on_session=my_callback,                    # async def(transport, config)
    public_ws_url="wss://example.com/sip/media-stream",
)
handler.register(app)
```

**`DIDRouter`** — maps phone numbers to configs:

```python
router = DIDRouter("sip_config.yaml")
config = router.lookup("+15125551234")   # Returns DIDConfig or None
config = router.lookup_or_raise("+1...")  # Raises DIDNotFoundError
router.reload()                           # Hot-reload YAML
```

**`AudioCodec`** — handles format conversion (used internally by SipTransport):

```python
codec = AudioCodec(source_rate=8000, source_encoding="mulaw")
canonical_pcm = codec.decode_to_canonical(provider_bytes)
provider_bytes = codec.encode_from_canonical(canonical_pcm)
codec._is_noop  # True for Telnyx L16/16kHz
```

### Constants

```python
from sip_voice_transport import CANONICAL_SAMPLE_RATE   # 16000
from sip_voice_transport import CANONICAL_SAMPLE_WIDTH  # 2 (16-bit)
from sip_voice_transport import CANONICAL_CHANNELS      # 1 (mono)
```

### Exceptions

All inherit from `SipTransportError`:

| Exception | When |
|-----------|------|
| `ProviderConnectionError` | WebSocket connection or handshake fails |
| `ProviderDetectionError` | Can't identify Telnyx vs Twilio from message |
| `AudioCodecError` | Audio encoding/decoding fails |
| `DIDNotFoundError` | Phone number not in config |
| `DIDConfigError` | YAML config is malformed |
| `CallRejectedError` | Call rejected by application logic |

## Writing a Voice AI Pipeline

See [`examples/mac_voice_server.py`](examples/mac_voice_server.py) for a complete STT → LLM → TTS example using Whisper, Ollama, and Piper.

The key pattern:

```python
async def on_session(transport: SipTransport, config: DIDConfig | None):
    """Called for each inbound call."""
    async for chunk in transport.receive_audio():
        # 1. Buffer audio until silence detected
        # 2. Transcribe with Whisper
        # 3. Send to Ollama for response
        # 4. Synthesize speech with Piper
        # 5. Send back to caller
        await transport.send_audio(tts_bytes)
```

## Testing

```bash
# Run unit tests
pytest

# Run with coverage
pytest --cov=sip_voice_transport

# Lint
ruff check src/ tests/
```

## Project Structure

```
src/sip_voice_transport/
├── __init__.py              # Public API exports
├── base_transport.py        # BaseTransport ABC + canonical constants
├── base_sip_provider.py     # BaseSipProvider ABC
├── transport.py             # SipTransport (wraps provider + codec)
├── audio_codec.py           # PCM/mulaw codec, resampling
├── webhook.py               # FastAPI route handler
├── did_router.py            # YAML DID config loader
├── config.py                # SipConfig dataclass
├── mac_utils.py             # Sleep inhibitor, Ollama check
├── exceptions.py            # Exception hierarchy
├── providers/
│   ├── __init__.py          # Provider registry + auto-detection
│   ├── telnyx_provider.py   # Telnyx WebSocket protocol
│   └── twilio_provider.py   # Twilio WebSocket protocol
└── cli/
    └── app.py               # Click CLI entry point
```

## License

MIT
