#!/usr/bin/env python3
"""Simulate a Telnyx inbound call against a running sip-voice-transport server.

This script does exactly what Telnyx would do when a phone call arrives:
  1. POST to /sip/telnyx/answer to trigger the webhook (like Telnyx HTTP callback)
  2. Open a WebSocket to /sip/media-stream
  3. Send the Telnyx handshake (connected → start)
  4. Stream audio frames and receive echo responses
  5. Send stop event and disconnect

Usage:
    # Start the server first:
    sip-voice-server --port 8765

    # Then in another terminal:
    python test_call.py
    python test_call.py --port 8765 --did "+15125551234" --caller "+15125559999"
    python test_call.py --audio-file recording.raw  # Send a real audio file
"""

import argparse
import asyncio
import base64
import json
import struct
import sys
import time

import httpx
import websockets


def generate_sine_tone(freq_hz: int = 440, duration_s: float = 0.02, sample_rate: int = 16000) -> bytes:
    """Generate a sine tone as 16-bit PCM at the given sample rate."""
    num_samples = int(sample_rate * duration_s)
    samples = []
    for i in range(num_samples):
        t = i / sample_rate
        value = int(16000 * __import__("math").sin(2 * __import__("math").pi * freq_hz * t))
        samples.append(struct.pack("<h", max(-32768, min(32767, value))))
    return b"".join(samples)


async def test_webhook(base_url: str, did: str, caller: str) -> bool:
    """Step 1: POST to the Telnyx answer webhook and verify TeXML response."""
    url = f"{base_url}/sip/telnyx/answer"
    print(f"\n{'='*60}")
    print(f"STEP 1: POST {url}")
    print(f"  From: {caller}  →  To: {did}")
    print(f"{'='*60}")

    async with httpx.AsyncClient() as client:
        response = await client.post(
            url,
            data={"From": caller, "To": did, "CallSid": "test-call-001"},
        )

    print(f"  Status: {response.status_code}")
    print(f"  Content-Type: {response.headers.get('content-type')}")
    print(f"  Body:\n{_indent(response.text)}")

    if response.status_code == 200 and "<Stream" in response.text:
        print("  ✓ Webhook returned TeXML with <Stream> — call would be connected")
        return True
    elif "<Hangup" in response.text:
        print("  ✗ Webhook returned <Hangup> — DID not configured")
        print(f"    Make sure {did} is in your sip_config.yaml")
        return False
    else:
        print("  ✗ Unexpected response")
        return False


async def test_websocket(
    ws_url: str,
    did: str,
    caller: str,
    num_frames: int,
    audio_file: str | None,
) -> bool:
    """Step 2: Open WebSocket, do Telnyx handshake, stream audio."""
    print(f"\n{'='*60}")
    print(f"STEP 2: WebSocket {ws_url}")
    print("  Simulating Telnyx media stream")
    print(f"{'='*60}")

    stream_id = f"test-stream-{int(time.time())}"

    try:
        async with websockets.connect(ws_url) as ws:
            # --- Telnyx Handshake ---

            # 1. Send "connected" event
            connected_msg = {"event": "connected", "version": "1.0.0"}
            await ws.send(json.dumps(connected_msg))
            print("  → Sent: connected")

            # 2. Send "start" event with stream metadata
            start_msg = {
                "event": "start",
                "stream_id": stream_id,
                "start": {
                    "call_control_id": "test-cc-001",
                    "custom_parameters": {
                        "caller_id": caller,
                        "did": did,
                    },
                    "media_format": {
                        "encoding": "audio/L16",
                        "sample_rate": "16000",
                        "channels": "1",
                    },
                },
            }
            await ws.send(json.dumps(start_msg))
            print(f"  → Sent: start (stream_id={stream_id})")
            print("  Handshake complete — streaming audio...\n")

            # --- Stream Audio ---
            if audio_file:
                await _stream_from_file(ws, stream_id, audio_file)
            else:
                await _stream_generated(ws, stream_id, num_frames)

            # --- Send Stop ---
            stop_msg = {"event": "stop", "stream_id": stream_id}
            await ws.send(json.dumps(stop_msg))
            print("\n  → Sent: stop")
            print("  ✓ WebSocket session completed successfully")
            return True

    except websockets.exceptions.ConnectionClosedError as e:
        print(f"  ✗ WebSocket closed unexpectedly: {e}")
        return False
    except ConnectionRefusedError:
        print("  ✗ Connection refused — is the server running?")
        return False


async def _stream_generated(ws, stream_id: str, num_frames: int):
    """Send generated sine tones and receive echo responses."""
    sent = 0
    received = 0
    total_bytes_sent = 0
    total_bytes_received = 0

    for i in range(num_frames):
        # Generate a 20ms audio frame (440 Hz tone)
        audio = generate_sine_tone(freq_hz=440, duration_s=0.02)
        payload = base64.b64encode(audio).decode()

        media_msg = {
            "event": "media",
            "stream_id": stream_id,
            "media": {"payload": payload},
        }
        await ws.send(json.dumps(media_msg))
        sent += 1
        total_bytes_sent += len(audio)

        # Try to read echo response (non-blocking with short timeout)
        try:
            response = await asyncio.wait_for(ws.recv(), timeout=0.05)
            msg = json.loads(response)
            if msg.get("event") == "media":
                received += 1
                resp_audio = base64.b64decode(msg["media"]["payload"])
                total_bytes_received += len(resp_audio)
        except asyncio.TimeoutError:
            pass

        # Progress every 50 frames (~1 second)
        if (i + 1) % 50 == 0:
            print(f"  [{i+1:4d}/{num_frames}] sent={sent} recv={received} "
                  f"({total_bytes_sent:,}B → {total_bytes_received:,}B)")

    # Drain remaining responses
    drain_start = time.time()
    while time.time() - drain_start < 1.0:
        try:
            response = await asyncio.wait_for(ws.recv(), timeout=0.1)
            msg = json.loads(response)
            if msg.get("event") == "media":
                received += 1
                resp_audio = base64.b64decode(msg["media"]["payload"])
                total_bytes_received += len(resp_audio)
        except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosedError):
            break

    print("\n  Summary:")
    print(f"    Frames sent:     {sent}")
    print(f"    Frames received: {received}")
    print(f"    Bytes sent:      {total_bytes_sent:,}")
    print(f"    Bytes received:  {total_bytes_received:,}")

    if received > 0:
        print(f"    ✓ Echo mode working — server echoed {received}/{sent} frames")
    else:
        print("    ⚠ No echo responses received (server may use a non-echo handler)")


async def _stream_from_file(ws, stream_id: str, audio_file: str):
    """Stream audio from a raw PCM file."""
    print(f"  Streaming from file: {audio_file}")

    with open(audio_file, "rb") as f:
        raw = f.read()

    frame_size = 640  # 20ms at 16kHz 16-bit = 640 bytes
    num_frames = len(raw) // frame_size
    print(f"  File size: {len(raw):,} bytes ({num_frames} frames, {num_frames * 0.02:.1f}s)")

    sent = 0
    received = 0

    for i in range(num_frames):
        chunk = raw[i * frame_size : (i + 1) * frame_size]
        payload = base64.b64encode(chunk).decode()

        media_msg = {
            "event": "media",
            "stream_id": stream_id,
            "media": {"payload": payload},
        }
        await ws.send(json.dumps(media_msg))
        sent += 1

        try:
            response = await asyncio.wait_for(ws.recv(), timeout=0.05)
            msg = json.loads(response)
            if msg.get("event") == "media":
                received += 1
        except asyncio.TimeoutError:
            pass

        if (i + 1) % 50 == 0:
            print(f"  [{i+1:4d}/{num_frames}] sent={sent} recv={received}")

        # Pace to real-time
        await asyncio.sleep(0.02)

    print(f"\n  Summary: sent={sent} frames, received={received} echo frames")


async def test_health(base_url: str) -> bool:
    """Step 0: Check server health."""
    url = f"{base_url}/health"
    print(f"\n{'='*60}")
    print(f"STEP 0: GET {url}")
    print(f"{'='*60}")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=5.0)
        data = response.json()
        print(f"  Status: {data.get('status')}")
        print(f"  DIDs:   {data.get('dids')}")
        print(f"  Ollama: {data.get('ollama')}")
        print("  ✓ Server is healthy")
        return True
    except httpx.ConnectError:
        print(f"  ✗ Cannot connect to {url}")
        print("    Start the server: sip-voice-server --port <port>")
        return False
    except Exception as e:
        print(f"  ✗ Health check failed: {e}")
        return False


def _indent(text: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + line for line in text.strip().splitlines())


async def main():
    parser = argparse.ArgumentParser(description="Simulate a Telnyx inbound call")
    parser.add_argument("--host", default="127.0.0.1", help="Server host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="Server port (default: 8765)")
    parser.add_argument("--did", default="+15125551234", help="Called number (DID)")
    parser.add_argument("--caller", default="+15125559999", help="Caller ID")
    parser.add_argument("--frames", type=int, default=150, help="Number of audio frames to send (~3s at 50fps)")
    parser.add_argument("--audio-file", default=None, help="Raw 16kHz 16-bit PCM file to stream")
    parser.add_argument("--skip-webhook", action="store_true", help="Skip webhook test, only test WebSocket")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    ws_url = f"ws://{args.host}:{args.port}/sip/media-stream"

    print("╔══════════════════════════════════════════════════════════╗")
    print("║         sip-voice-transport: Simulated Call Test        ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║  Server:  {base_url:<46s}  ║")
    print(f"║  DID:     {args.did:<46s}  ║")
    print(f"║  Caller:  {args.caller:<46s}  ║")
    print(f"║  Frames:  {args.frames:<46d}  ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # Step 0: Health check
    if not await test_health(base_url):
        sys.exit(1)

    # Step 1: Webhook test
    if not args.skip_webhook:
        if not await test_webhook(base_url, args.did, args.caller):
            print("\nWebhook test failed. Continuing with WebSocket test anyway...\n")

    # Step 2: WebSocket + audio streaming
    success = await test_websocket(ws_url, args.did, args.caller, args.frames, args.audio_file)

    # Results
    print(f"\n{'='*60}")
    if success:
        print("ALL TESTS PASSED")
        print(f"{'='*60}")
        print("\nNext steps:")
        print("  1. Set up Cloudflare tunnel: cloudflared tunnel --url http://localhost:8765")
        print("  2. Configure Telnyx webhook URL to point to your tunnel")
        print("  3. Call your DID from a real phone!")
    else:
        print("SOME TESTS FAILED")
        print(f"{'='*60}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
