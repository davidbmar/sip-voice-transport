"""Tests for AudioCodec, mulaw encode/decode, and resampling."""

import numpy as np

from sip_voice_transport.audio_codec import (
    AudioCodec,
    mulaw_decode,
    mulaw_encode,
    resample,
    base64_decode_audio,
    base64_encode_audio,
)


class TestMulawCodec:
    def test_mulaw_decode_produces_correct_length(self):
        """mulaw decode: 100 bytes in → 200 bytes out (1 byte → 2 bytes per sample)."""
        mulaw_data = bytes(range(100))
        pcm = mulaw_decode(mulaw_data)
        assert len(pcm) == 200

    def test_mulaw_roundtrip(self):
        """Encode then decode should produce approximately the same signal."""
        # Generate a simple sine wave as 16-bit PCM
        t = np.linspace(0, 0.01, 160, dtype=np.float64)
        signal = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)
        pcm_bytes = signal.tobytes()

        encoded = mulaw_encode(pcm_bytes)
        decoded = mulaw_decode(encoded)

        original = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float64)
        recovered = np.frombuffer(decoded, dtype=np.int16).astype(np.float64)

        # mulaw is lossy — allow ~2% error
        max_error = np.max(np.abs(original - recovered))
        assert max_error < 1000, f"Max roundtrip error too high: {max_error}"

    def test_mulaw_silence(self):
        """Silence (zeros) should roundtrip reasonably."""
        silence = np.zeros(100, dtype=np.int16).tobytes()
        encoded = mulaw_encode(silence)
        decoded = mulaw_decode(encoded)
        samples = np.frombuffer(decoded, dtype=np.int16)
        assert np.max(np.abs(samples)) < 200  # Near silence


class TestResample:
    def test_resample_8k_to_16k_doubles_length(self):
        """Upsampling 8kHz → 16kHz should roughly double the number of samples."""
        pcm_8k = np.zeros(800, dtype=np.int16).tobytes()  # 100ms at 8kHz
        pcm_16k = resample(pcm_8k, 8000, 16000)
        # Should have ~1600 samples = 3200 bytes
        assert abs(len(pcm_16k) - 3200) < 8  # Allow small rounding

    def test_resample_16k_to_8k_halves_length(self):
        pcm_16k = np.zeros(1600, dtype=np.int16).tobytes()
        pcm_8k = resample(pcm_16k, 16000, 8000)
        assert abs(len(pcm_8k) - 1600) < 8

    def test_resample_same_rate_is_noop(self):
        original = np.arange(100, dtype=np.int16).tobytes()
        result = resample(original, 16000, 16000)
        assert result == original


class TestAudioCodec:
    def test_noop_for_l16_16k(self):
        """Telnyx L16/16kHz → canonical should be a no-op."""
        codec = AudioCodec(source_rate=16000, source_encoding="l16")
        assert codec._is_noop is True
        data = b"\x01\x02\x03\x04"
        assert codec.decode_to_canonical(data) == data
        assert codec.encode_from_canonical(data) == data

    def test_mulaw_8k_requires_conversion(self):
        """Twilio mulaw/8kHz → canonical requires both decode and resample."""
        codec = AudioCodec(source_rate=8000, source_encoding="mulaw")
        assert codec._is_noop is False
        assert codec._needs_resample is True
        assert codec._needs_decode is True


class TestBase64:
    def test_roundtrip(self):
        data = b"\x00\x01\x02\xff"
        encoded = base64_encode_audio(data)
        decoded = base64_decode_audio(encoded)
        assert decoded == data
