"""Audio codec utilities for converting between provider and canonical formats.

CANONICAL FORMAT: 16kHz, 16-bit signed LE PCM, mono.

Conversion paths:
  Telnyx L16/16kHz  → canonical: NO-OP (same format)
  Telnyx mulaw/8kHz → canonical: mulaw→PCM + resample 8k→16k
  Twilio mulaw/8kHz → canonical: mulaw→PCM + resample 8k→16k
"""

import base64
import logging
from typing import Optional

import numpy as np

from sip_voice_transport.exceptions import AudioCodecError

logger = logging.getLogger(__name__)


# μ-law decompression table (ITU-T G.711)
# This avoids dependency on audioop (deprecated in Python 3.11, removed in 3.13)
_MULAW_DECODE_TABLE: Optional[np.ndarray] = None


def _get_mulaw_decode_table() -> np.ndarray:
    """Lazily build the μ-law to linear PCM lookup table."""
    global _MULAW_DECODE_TABLE
    if _MULAW_DECODE_TABLE is None:
        table = np.zeros(256, dtype=np.int16)
        for i in range(256):
            val = ~i
            sign = val & 0x80
            exponent = (val >> 4) & 0x07
            mantissa = val & 0x0F
            sample = (mantissa << 3) + 0x84
            sample <<= exponent
            sample -= 0x84
            table[i] = -sample if sign else sample
        _MULAW_DECODE_TABLE = table
    return _MULAW_DECODE_TABLE


class AudioCodec:
    """Handles audio format conversion between provider and canonical formats.

    Usage:
        codec = AudioCodec(source_rate=8000, source_encoding="mulaw")
        canonical_pcm = codec.decode_to_canonical(provider_bytes)
        provider_bytes = codec.encode_from_canonical(canonical_pcm)
    """

    def __init__(
        self,
        source_rate: int,
        source_encoding: str,
        target_rate: int = 16000,
        target_encoding: str = "l16",
    ):
        """
        Args:
            source_rate: Provider's native sample rate (8000 or 16000).
            source_encoding: Provider's native encoding ('l16' or 'mulaw').
            target_rate: Canonical sample rate. Always 16000.
            target_encoding: Canonical encoding. Always 'l16'.
        """
        self.source_rate = source_rate
        self.source_encoding = source_encoding.lower()
        self.target_rate = target_rate
        self.target_encoding = target_encoding.lower()

        self._needs_resample = (source_rate != target_rate)
        self._needs_decode = (self.source_encoding != self.target_encoding)
        self._is_noop = (not self._needs_resample and not self._needs_decode)

        if self._is_noop:
            logger.info(
                "AudioCodec: source matches canonical (%dHz %s) — no conversion needed",
                source_rate, source_encoding,
            )

    def decode_to_canonical(self, raw: bytes) -> bytes:
        """Convert provider-format audio to canonical 16kHz 16-bit PCM.

        Args:
            raw: Audio bytes in provider's native format.

        Returns:
            Audio bytes in canonical format (16kHz, 16-bit, mono PCM).
        """
        if self._is_noop:
            return raw

        try:
            pcm = raw
            if self.source_encoding == "mulaw":
                pcm = mulaw_decode(raw)
            if self._needs_resample:
                pcm = resample(pcm, self.source_rate, self.target_rate)
            return pcm
        except Exception as e:
            raise AudioCodecError(f"Failed to decode audio: {e}") from e

    def encode_from_canonical(self, pcm: bytes) -> bytes:
        """Convert canonical 16kHz 16-bit PCM to provider's format.

        Args:
            pcm: Audio bytes in canonical format.

        Returns:
            Audio bytes in provider's native format.
        """
        if self._is_noop:
            return pcm

        try:
            out = pcm
            if self._needs_resample:
                out = resample(out, self.target_rate, self.source_rate)
            if self.source_encoding == "mulaw":
                out = mulaw_encode(out)
            return out
        except Exception as e:
            raise AudioCodecError(f"Failed to encode audio: {e}") from e


def mulaw_decode(data: bytes) -> bytes:
    """Decode μ-law encoded audio to 16-bit linear PCM.

    This implements ITU-T G.711 μ-law decoding without using the deprecated
    `audioop` module.

    Args:
        data: μ-law encoded bytes (1 byte per sample).

    Returns:
        16-bit linear PCM bytes (2 bytes per sample, little-endian).
    """
    table = _get_mulaw_decode_table()
    samples = np.frombuffer(data, dtype=np.uint8)
    pcm = table[samples]
    return pcm.tobytes()


def mulaw_encode(pcm_data: bytes) -> bytes:
    """Encode 16-bit linear PCM to μ-law.

    Implements ITU-T G.711 μ-law compression without `audioop`.

    Args:
        pcm_data: 16-bit signed PCM bytes (little-endian).

    Returns:
        μ-law encoded bytes (1 byte per sample).
    """
    MULAW_CLIP = 32635
    MULAW_BIAS = 0x84

    # Standard G.711 exponent lookup table, indexed by (biased_sample >> 7) & 0xFF
    _EXP_LUT = np.array([
        0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
        5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
        5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
        6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
        6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
        6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
        6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
        7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
        7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
        7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
        7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
        7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
        7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
        7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
        7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    ], dtype=np.uint8)

    samples = np.frombuffer(pcm_data, dtype=np.int16).astype(np.int32)

    # Get sign
    sign = np.where(samples < 0, 0x80, 0).astype(np.uint8)
    samples = np.abs(samples)

    # Clip to max
    samples = np.minimum(samples, MULAW_CLIP)

    # Add bias
    samples = samples + MULAW_BIAS

    # Find exponent via lookup table
    exponent = _EXP_LUT[((samples >> 7) & 0xFF).astype(np.uint8)]

    # Extract mantissa
    mantissa = ((samples >> (exponent.astype(np.int32) + 3)) & 0x0F).astype(np.uint8)

    # Combine and complement
    mulaw_byte = ~(sign | (exponent << 4) | mantissa)

    return mulaw_byte.astype(np.uint8).tobytes()


def resample(pcm_data: bytes, from_rate: int, to_rate: int) -> bytes:
    """Resample 16-bit PCM audio.

    Uses linear interpolation for speed. For higher quality, install scipy
    and this will use scipy.signal.resample instead.

    Args:
        pcm_data: 16-bit signed PCM bytes.
        from_rate: Source sample rate in Hz.
        to_rate: Target sample rate in Hz.

    Returns:
        Resampled 16-bit PCM bytes.
    """
    if from_rate == to_rate:
        return pcm_data

    samples = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float64)
    num_output = int(len(samples) * to_rate / from_rate)

    try:
        from scipy.signal import resample as scipy_resample
        resampled = scipy_resample(samples, num_output)
    except ImportError:
        # Fallback: linear interpolation
        x_old = np.linspace(0, 1, len(samples))
        x_new = np.linspace(0, 1, num_output)
        resampled = np.interp(x_new, x_old, samples)

    # Clip to int16 range and convert
    resampled = np.clip(resampled, -32768, 32767).astype(np.int16)
    return resampled.tobytes()


def base64_decode_audio(payload: str) -> bytes:
    """Decode a base64 audio payload from a provider WebSocket message.

    Args:
        payload: Base64 encoded string.

    Returns:
        Raw audio bytes.
    """
    return base64.b64decode(payload)


def base64_encode_audio(audio: bytes) -> str:
    """Encode audio bytes to base64 for sending over a provider WebSocket.

    Args:
        audio: Raw audio bytes.

    Returns:
        Base64 encoded string.
    """
    return base64.b64encode(audio).decode("ascii")
