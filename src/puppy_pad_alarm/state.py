"""Per-pad event state and candidate persistence."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .config import Settings
from .vision import Candidate


class Phase(StrEnum):
    """Visible state of one pad."""

    NEEDS_BASELINE = "NEEDS_BASELINE"
    READY = "READY"
    VISIT_ACTIVE = "VISIT_ACTIVE"
    CHECKING = "CHECKING"
    ALARM_LATCHED = "ALARM_LATCHED"


@dataclass
class PadState:
    """Track one pad independently of its image position."""

    phase: Phase = Phase.NEEDS_BASELINE
    candidate: Candidate | None = None
    first_seen: float = 0.0
    seen_frames: int = 0
    checking_since: float = 0.0
    inspected_frames: int = 0
    visit_confidence: float = 0.0
    detected_at: float | None = None
    snapshot_name: str | None = None
    object_center: tuple[float, float] | None = None
    dog_near: bool = False
    dog_away_since: float | None = None
    last_deterrent_at: float = 0.0
    reason: str = "No clean baseline"

    def baseline_set(self) -> None:
        """Rearm this pad after the user marks it clean."""
        self.phase = Phase.READY
        self.candidate = None
        self.first_seen = 0.0
        self.seen_frames = 0
        self.inspected_frames = 0
        self.visit_confidence = 0.0
        self.detected_at = None
        self.snapshot_name = None
        self.object_center = None
        self.dog_near = False
        self.dog_away_since = None
        self.last_deterrent_at = 0.0
        self.reason = ""

    def update(
        self,
        *,
        dog_present: bool,
        visible: bool,
        candidates: list[Candidate],
        reason: str | None,
        now: float,
        settings: Settings,
    ) -> Candidate | None:
        """Advance the visit and persistence rules; return a new confirmed region."""
        if self.phase in (Phase.NEEDS_BASELINE, Phase.ALARM_LATCHED):
            return None
        if dog_present:
            self.phase = Phase.VISIT_ACTIVE
            self.inspected_frames = 0
        elif self.phase == Phase.VISIT_ACTIVE:
            self.phase = Phase.CHECKING
            self.checking_since = now
            self.inspected_frames = 0
        if self.phase == Phase.READY:
            self.reason = reason or ("Pad is hidden" if not visible else "")
            return None
        if not visible or reason:
            self.reason = reason or "Pad is hidden; result pending"
            self.candidate = None
            self.seen_frames = 0
            return None
        self.reason = ""
        if self.phase == Phase.CHECKING:
            self.inspected_frames += 1
        match = self._match(candidates, settings)
        if match is None:
            self.candidate = None
            self.seen_frames = 0
        elif self.candidate is None:
            self.candidate = match
            self.first_seen = now
            self.seen_frames = 1
        else:
            self.candidate = match
            self.seen_frames += 1
            if (
                self.seen_frames >= settings.min_persistence_frames
                and now - self.first_seen >= settings.min_persistence_s
            ):
                self.phase = Phase.ALARM_LATCHED
                self.reason = "Possible poop detected"
                return match
        if (
            self.phase == Phase.CHECKING
            and not candidates
            and self.inspected_frames >= settings.clear_frames
            and now - self.checking_since >= settings.checking_timeout_s
        ):
            self.phase = Phase.READY
            self.visit_confidence = 0.0
        return None

    def _match(
        self, candidates: list[Candidate], settings: Settings
    ) -> Candidate | None:
        """Find the closest stable proposal or start with the largest one."""
        if not candidates:
            return None
        if self.candidate is None:
            return max(candidates, key=lambda item: item.area)
        previous = self.candidate.center
        matches = [
            item
            for item in candidates
            if (item.center[0] - previous[0]) ** 2 + (item.center[1] - previous[1]) ** 2
            <= settings.stable_distance_px**2
        ]
        return max(matches, key=lambda item: item.area) if matches else None
