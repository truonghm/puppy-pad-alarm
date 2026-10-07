"""Phone notifications for possible detections."""

from __future__ import annotations

import logging
import os
import time
from contextlib import ExitStack
from pathlib import Path

import httpx

PUSHOVER_URL = "https://api.pushover.net/1/messages.json"


def send_pushover(snapshot: Path | None) -> None:
    """Send a normal-priority event message with available image evidence.

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
    data = {
        "token": token,
        "user": user,
        "title": "Puppy pad check",
        "message": "Possible poop detected in the selected pad area.",
        "priority": "0",
    }
    def post_once() -> httpx.Response:
        """Open the optional snapshot separately for each send attempt."""
        with ExitStack() as stack:
            files = None
            if snapshot is not None and snapshot.exists():
                image = stack.enter_context(snapshot.open("rb"))
                files = {"attachment": (snapshot.name, image, "image/jpeg")}
            return httpx.post(
                PUSHOVER_URL,
                data=data,
                files=files,
                timeout=httpx.Timeout(30.0),
            )

    try:
        response = post_once()
    except (httpx.ConnectError, httpx.ConnectTimeout) as error:
        logging.getLogger("puppy_pad_alarm").warning(
            "Pushover connection failed (%s); retrying once in 5 seconds", error
        )
        time.sleep(5)
        response = post_once()
    response.raise_for_status()
    result = response.json()
    if result.get("status") != 1:
        raise RuntimeError(
            f"Pushover rejected the message: {result.get('errors', result)}"
        )
