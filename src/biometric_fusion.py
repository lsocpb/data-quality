from __future__ import annotations

from dataclasses import dataclass

from src.face_eigenfaces import FaceVerificationResult
from src.keystroke_identity import VerificationDecision


@dataclass(frozen=True)
class BiometricFusionResult:
    claimed_user: str
    accepted: bool
    face_matched: bool
    keystroke_matched: bool
    face_confidence: float
    keystroke_score: float
    fusion_strategy: str


def fuse_and(
    face_result: FaceVerificationResult,
    keystroke_result: VerificationDecision,
) -> BiometricFusionResult:
    """Full-agreement fusion: both modalities must match to accept."""
    return BiometricFusionResult(
        claimed_user=str(face_result.claimed_user),
        accepted=face_result.matched and keystroke_result.matched,
        face_matched=face_result.matched,
        keystroke_matched=keystroke_result.matched,
        face_confidence=face_result.confidence,
        keystroke_score=keystroke_result.score,
        fusion_strategy="and",
    )


def fuse_weighted(
    face_result: FaceVerificationResult,
    keystroke_result: VerificationDecision,
    *,
    face_weight: float = 0.5,
) -> BiometricFusionResult:
    """
    Weighted fusion: normalized scores are combined.
    Face score = confidence / threshold (lower = better, capped at 1).
    Keystroke score = score / threshold (lower = better, capped at 1).
    Weighted sum < 1 → accept.
    """
    face_norm = min(face_result.confidence / face_result.threshold, 2.0)
    keystroke_norm = min(keystroke_result.score / keystroke_result.threshold, 2.0)
    keystroke_weight = 1.0 - face_weight
    combined = face_weight * face_norm + keystroke_weight * keystroke_norm

    return BiometricFusionResult(
        claimed_user=str(face_result.claimed_user),
        accepted=combined < 1.0,
        face_matched=face_result.matched,
        keystroke_matched=keystroke_result.matched,
        face_confidence=face_result.confidence,
        keystroke_score=keystroke_result.score,
        fusion_strategy=f"weighted(face={face_weight:.2f})",
    )
