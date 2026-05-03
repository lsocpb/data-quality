"""Data models for analysis results"""

from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class AnalysisResult:
    """Result of keystroke analysis (identification)"""
    user_id: str
    sample_number: int
    raw_event_count: int
    features: Dict[str, float]
    neighbors: List[tuple] = field(default_factory=list)  # (user_id, distance, sample_number)
    predicted_user_id: str = ""
    confidence_score: float = 0.0
    explanation: str = ""
    metric: str = "euclidean"  # Distance metric used
    k: int = 5  # Number of neighbors
    timing_ms: Dict[str, float] = field(default_factory=dict)  # {step_name: duration_ms}


@dataclass
class VerificationResult(AnalysisResult):
    """Result of keystroke verification (is claimed identity correct?)"""
    verification_match: bool = False
    threshold: float = 0.6
    claimed_user_id: str = ""
