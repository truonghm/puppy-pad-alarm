"""Camera-name selection without camera hardware."""

from __future__ import annotations

import pytest

from puppy_pad_alarm.app import named_camera_index


def test_logitech_c270_is_selected_by_name() -> None:
    """A variant Windows camera name still selects the C270."""
    devices = ["Integrated Camera", "Logitech HD Webcam C270"]
    assert named_camera_index(devices, "C270 HD WEBCAM") == 1


def test_missing_c270_does_not_select_laptop_camera() -> None:
    """A missing C270 reports the available cameras instead of using index zero."""
    with pytest.raises(RuntimeError, match="0: Integrated Camera"):
        named_camera_index(["Integrated Camera"], "C270 HD WEBCAM")
