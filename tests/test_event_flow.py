"""User-visible detection and restart behavior."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from puppy_pad_alarm.app import Application, notification_due
from puppy_pad_alarm.config import Settings, save_settings
from puppy_pad_alarm.state import AreaState, Phase
from puppy_pad_alarm.vision import Candidate


def test_hidden_object_waits_and_latches_once() -> None:
    """A hidden pad cannot confirm poop, and one visit confirms only once."""
    settings = Settings()
    state = AreaState()
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
    assert state.phase == Phase.ALARM_LATCHED
    assert (
        state.update(
            dog_present=False,
            visible=True,
            candidates=[candidate],
            reason=None,
            now=1.5,
            settings=settings,
        )
        is None
    )
    assert state.phase == Phase.READY
    assert (
        state.update(
            dog_present=False,
            visible=True,
            candidates=[candidate],
            reason=None,
            now=1.6,
            settings=settings,
        )
        is None
    )
    assert state.phase == Phase.READY
    for moment in (2.0, 2.3):
        assert (
            state.update(
                dog_present=True,
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
            dog_present=True,
            visible=True,
            candidates=[candidate],
            reason=None,
            now=2.6,
            settings=settings,
        )
        is candidate
    )


def test_notification_cooldown_between_visits() -> None:
    """A later visit can alert only after ten minutes."""
    assert notification_due(None, 1_000.0)
    assert not notification_due(1_000.0, 1_599.9)
    assert notification_due(1_000.0, 1_600.0)


def test_occlusion_resets_partial_detection() -> None:
    """A hidden interval cannot complete a partial detection."""
    settings = Settings()
    state = AreaState()
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


def test_manual_clean_rearms_latched_pad() -> None:
    """A new clean baseline also clears the current visit state."""
    state = AreaState(phase=Phase.ALARM_LATCHED, detected_at=1_000.0)
    assert state.phase == Phase.ALARM_LATCHED
    state.baseline_set()
    assert state.phase == Phase.READY
    assert state.detected_at is None


def test_restart_keeps_unclean_pad_latched(tmp_path: Path) -> None:
    """Restarting retains the current visit until the dog leaves."""
    settings_path = tmp_path / "config.yaml"
    save_settings(
        settings_path, Settings(output_dir=str(tmp_path), search_region=[0, 0, 320, 240], save_event_video=False)
    )
    baseline = np.full((240, 320, 3), 230, np.uint8)
    assert cv2.imwrite(str(tmp_path / "baseline_region.png"), baseline)
    (tmp_path / "state.json").write_text(
        json.dumps(
            {"region": [0, 0, 320, 240], "baseline_region": [0, 0, 320, 240], "phase": "ALARM_LATCHED", "detected_at": 1_000.0}
        )
    )

    restarted = Application(settings_path)
    restored = restarted.state
    assert restored.phase == Phase.ALARM_LATCHED
    assert restored.detected_at == 1_000.0
    assert not notification_due(restored.detected_at, 1_500.0)
    restarted.delivery.shutdown(wait=True)
