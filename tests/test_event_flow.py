"""User-visible detection, reminder, and restart behavior."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from puppy_pad_alarm.app import Application
from puppy_pad_alarm.config import Settings, save_settings
from puppy_pad_alarm.state import PadState, Phase
from puppy_pad_alarm.vision import Candidate


def test_hidden_object_waits_and_latches_once() -> None:
    """A hidden pad cannot confirm poop, and a later visit cannot alert again."""
    settings = Settings()
    state = PadState()
    state.baseline_set()
    candidate = Candidate((100.0, 100.0), np.array([[[100, 100]]], np.int32), 80)
    assert (
        state.update(
            dog_present=True,
            visible=False,
            candidates=[],
            reason="Dog hides pad",
            now=0.0,
            settings=settings,
        )
        is None
    )
    assert (
        state.update(
            dog_present=False,
            visible=False,
            candidates=[],
            reason="Dog hides pad",
            now=0.2,
            settings=settings,
        )
        is None
    )
    for moment in (0.3, 0.6):
        assert (
            state.update(
                dog_present=False,
                visible=True,
                candidates=[candidate],
                reason=None,
                now=moment,
                settings=settings,
            )
            is None
        )
    assert (
        state.update(
            dog_present=False,
            visible=True,
            candidates=[candidate],
            reason=None,
            now=0.9,
            settings=settings,
        )
        is candidate
    )
    assert state.phase == Phase.ALARM_LATCHED
    assert (
        state.update(
            dog_present=True,
            visible=True,
            candidates=[candidate],
            reason=None,
            now=1.2,
            settings=settings,
        )
        is None
    )


def test_occlusion_resets_partial_detection() -> None:
    """A hidden interval cannot complete a partial detection."""
    settings = Settings()
    state = PadState()
    state.baseline_set()
    candidate = Candidate((100.0, 100.0), np.array([[[100, 100]]], np.int32), 80)
    state.update(
        dog_present=True,
        visible=True,
        candidates=[candidate],
        reason=None,
        now=0.0,
        settings=settings,
    )
    state.update(
        dog_present=True,
        visible=True,
        candidates=[candidate],
        reason=None,
        now=0.3,
        settings=settings,
    )
    state.update(
        dog_present=True,
        visible=False,
        candidates=[],
        reason="Dog hides pad",
        now=0.4,
        settings=settings,
    )
    assert (
        state.update(
            dog_present=False,
            visible=True,
            candidates=[candidate],
            reason=None,
            now=0.7,
            settings=settings,
        )
        is None
    )
    assert (
        state.update(
            dog_present=False,
            visible=True,
            candidates=[candidate],
            reason=None,
            now=1.0,
            settings=settings,
        )
        is None
    )
    assert (
        state.update(
            dog_present=False,
            visible=True,
            candidates=[candidate],
            reason=None,
            now=1.3,
            settings=settings,
        )
        is candidate
    )


def test_reminders_stop_after_manual_clean() -> None:
    """A latched pad reminds on schedule until the user marks it clean."""
    state = PadState(phase=Phase.ALARM_LATCHED, detected_at=1_000.0)
    assert state.reminder_due(1_059.0) is None
    assert state.reminder_due(1_060.0) == 0
    assert state.reminder_due(1_060.0) is None
    assert state.reminder_due(1_600.0) == 9
    assert state.reminder_due(2_200.0) == 10
    assert state.reminder_due(19_600.0) == 39
    assert state.reminder_due(20_201.0) is None
    state.baseline_set()
    assert state.phase == Phase.READY
    assert state.reminder_due(30_000.0) is None


def test_restart_keeps_unclean_pad_and_reminder_progress(tmp_path: Path) -> None:
    """Restarting does not clear an alert or repeat an elapsed reminder."""
    settings_path = tmp_path / "config.yaml"
    save_settings(
        settings_path, Settings(output_dir=str(tmp_path), save_event_video=False)
    )
    baseline = np.full((240, 320, 3), 230, np.uint8)
    assert cv2.imwrite(str(tmp_path / "baseline_blue.png"), baseline)
    (tmp_path / "state.json").write_text(
        json.dumps(
            {
                "blue": {
                    "phase": "ALARM_LATCHED",
                    "detected_at": 1_000.0,
                    "last_reminder_index": 4,
                }
            }
        )
    )

    restarted = Application(settings_path)
    restored = restarted.states["blue"]
    assert restored.phase == Phase.ALARM_LATCHED
    assert restored.reminder_due(1_300.0) is None
    assert restored.reminder_due(1_360.0) == 5
    restarted.delivery.shutdown(wait=True)
