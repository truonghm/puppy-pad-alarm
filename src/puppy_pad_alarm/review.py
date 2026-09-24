"""Optional advisory image review contract; never used in alert decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np


@dataclass(frozen=True)
class ReviewInput:
    """Pad-relative images to inspect when local rules are uncertain."""

    pad_color: str
    baseline: np.ndarray
    current: np.ndarray
    uncertainty_reason: str


@dataclass(frozen=True)
class ReviewResult:
    """An advisory result that a future UI can show to the user."""

    verdict: Literal["possible_object", "no_object", "uncertain"]
    explanation: str


class VisionReviewer(Protocol):
    """A future image service can implement this interface."""

    def review(self, request: ReviewInput) -> ReviewResult:
        """Return a second opinion without changing the local alarm state."""
        ...
