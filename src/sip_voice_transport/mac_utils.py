"""macOS-specific utilities for running the voice server as a daemon."""

import logging
import platform
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)


class SleepInhibitor:
    """Prevent Mac from sleeping during active phone calls.

    Uses macOS `caffeinate` command. Safe to use on non-Mac platforms
    (methods become no-ops).
    """

    def __init__(self):
        self._process: Optional[subprocess.Popen] = None
        self._is_mac = platform.system() == "Darwin"

    def acquire(self) -> None:
        """Prevent sleep. Call when a phone call starts."""
        if not self._is_mac:
            return
        if self._process is not None:
            return  # Already acquired

        try:
            self._process = subprocess.Popen(
                ["caffeinate", "-d", "-i", "-s"],  # display + idle + system sleep
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            logger.debug("Sleep inhibitor acquired (pid=%d)", self._process.pid)
        except FileNotFoundError:
            logger.warning("caffeinate not found — sleep prevention unavailable")

    def release(self) -> None:
        """Allow sleep again. Call when a phone call ends."""
        if self._process is not None:
            self._process.terminate()
            self._process.wait()
            logger.debug("Sleep inhibitor released")
            self._process = None

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *args):
        self.release()


def check_ollama_running(host: str = "localhost", port: int = 11434) -> bool:
    """Check if Ollama is running and reachable.

    Args:
        host: Ollama host.
        port: Ollama port.

    Returns:
        True if Ollama is reachable.
    """
    import socket
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except (ConnectionRefusedError, socket.timeout, OSError):
        return False
