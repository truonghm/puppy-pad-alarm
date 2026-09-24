"""Phone notification and local dog deterrent playback."""

from __future__ import annotations

import os
from contextlib import ExitStack
from pathlib import Path

import httpx
import pygame

PUSHOVER_URL = "https://api.pushover.net/1/messages.json"


def send_pushover(
    pad_color: str, snapshot: Path | None, *, reminder: bool = False
) -> None:
    """Send an event or reminder message with available image evidence.

    Raises:
        RuntimeError: Required credentials are missing or Pushover rejects the message.
        httpx.HTTPError: The request could not be delivered.
    """
    token = os.environ.get("PUSHOVER_TOKEN")
    user = os.environ.get("PUSHOVER_USER")
    if not token or not user:
        raise RuntimeError(
            "Set PUSHOVER_TOKEN and PUSHOVER_USER to enable phone notifications"
        )
    message = (
        f"Possible poop is still on the {pad_color} pad."
        if reminder
        else f"Possible poop detected on the {pad_color} pad."
    )
    with ExitStack() as stack:
        files = None
        if snapshot is not None and snapshot.exists():
            image = stack.enter_context(snapshot.open("rb"))
            files = {"attachment": (snapshot.name, image, "image/jpeg")}
        response = httpx.post(
            PUSHOVER_URL,
            data={
                "token": token,
                "user": user,
                "title": "Puppy pad reminder" if reminder else "Puppy pad check",
                "message": message,
            },
            files=files,
            timeout=10,
        )
    response.raise_for_status()
    result = response.json()
    if result.get("status") != 1:
        raise RuntimeError(
            f"Pushover rejected the message: {result.get('errors', result)}"
        )


class Deterrent:
    """Play the supplied sound once for each detected event."""

    def __init__(self, sound: Path) -> None:
        self.sound = sound
        self.ready = False

    def play(self) -> None:
        """Start local playback."""
        if not self.ready:
            pygame.mixer.init()
            self.ready = True
        pygame.mixer.music.load(str(self.sound))
        pygame.mixer.music.play()

    def stop(self) -> None:
        """Release local playback when the application exits."""
        if self.ready:
            pygame.mixer.music.stop()
