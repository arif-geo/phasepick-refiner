"""Small data objects shared by the scientific classes."""

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd


def timestamp_text(value: Any) -> str:
    """Use one unambiguous UTC representation in intermediate products."""
    timestamp = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(timestamp):
        return ""
    return timestamp.strftime("%Y-%m-%d %H:%M:%S.%f+00:00")


@dataclass
class MasterSelection:
    """One selected reference event for a station-cluster pair."""

    cluster_id: str
    station_id: str
    station: str
    cluster_event_count: int
    observed_pair_count: int
    selection_status: str
    master_event_id: str = ""
    p_pick_time: str = ""
    s_pick_time: str = ""
    reviewed_p_pick_time: str = ""
    reviewed_s_pick_time: str = ""
    p_pick_edited: bool = False
    s_pick_edited: bool = False
    p_score: float | None = None
    s_score: float | None = None
    snr: float | None = None
    median_p_offset_seconds: float | None = None
    median_s_offset_seconds: float | None = None
    median_sp_seconds: float | None = None
    master_quality_score: float | None = None

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CorrelationHit:
    """Best waveform match for one phase."""

    pick_time: pd.Timestamp
    coefficient: float
    shift_seconds: float
    component: str


@dataclass
class PhasePickProposal:
    """One accepted automatic or manually reviewed phase pick."""

    event_id: str
    station_id: str
    station: str
    phase: str
    chosen_time: pd.Timestamp
    source: str
    original_time: pd.Timestamp | None = None
    search_center_time: pd.Timestamp | None = None
    search_center_source: str = ""
    cluster_id: str = ""
    master_event_id: str = ""
    cc: float | None = None
    shift_seconds: float | None = None
    cc_component: str = ""
    cc_threshold: float | None = None
    phase_score: float | None = None
    snr: float | None = None

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        for column in [
            "chosen_time",
            "original_time",
            "search_center_time",
        ]:
            record[column] = timestamp_text(record[column])
        return record


@dataclass
class ValidationReport:
    """Input problems and useful dataset counts."""

    errors: list[str]
    warnings: list[str]
    statistics: dict[str, int | float | str]

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def format(self) -> str:
        lines = ["Input validation"]
        for name, value in self.statistics.items():
            lines.append(f"  {name}: {value}")
        if self.warnings:
            lines.append("Warnings")
            lines.extend(f"  - {message}" for message in self.warnings)
        if self.errors:
            lines.append("Errors")
            lines.extend(f"  - {message}" for message in self.errors)
        else:
            lines.append("  status: ready")
        return "\n".join(lines)


@dataclass
class RefinementResult:
    """Accepted pick proposals plus an audit row for every CC attempt."""

    proposals: list[PhasePickProposal]
    attempts: pd.DataFrame

    def proposal_table(self) -> pd.DataFrame:
        return pd.DataFrame(
            [proposal.to_record() for proposal in self.proposals]
        )
