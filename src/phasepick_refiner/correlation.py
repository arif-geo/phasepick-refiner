"""Paired P/S waveform cross-correlation refinement."""

import numpy as np
import pandas as pd
from obspy import Trace, UTCDateTime
from obspy.signal.cross_correlation import correlate_template

from .configuration import ProjectConfiguration
from .data import PickDataset, short_station_name
from .models import (
    CorrelationHit,
    PhasePickProposal,
    RefinementResult,
)
from .waveforms import WaveformArchive


class CrossCorrelationRefiner:
    """Refine existing picks and detect missing pairs from selected masters."""

    horizontal_components = ("E", "N", "1", "2")

    def __init__(
        self,
        configuration: ProjectConfiguration,
        dataset: PickDataset,
        waveform_archive: WaveformArchive,
    ):
        self.configuration = configuration
        self.dataset = dataset
        self.waveform_archive = waveform_archive

    def refine(self, master_table: pd.DataFrame) -> RefinementResult:
        accepted_proposals: list[PhasePickProposal] = []
        attempt_records: list[dict[str, object]] = []

        usable_masters = master_table[
            master_table["master_event_id"].fillna("").astype(str).ne("")
        ]
        for position, master in enumerate(
            usable_masters.itertuples(index=False), start=1
        ):
            print(
                f"[CC {position}/{len(usable_masters)}] "
                f"cluster {master.cluster_id} at {master.station}"
            )
            proposals, attempts = self._refine_station_cluster(master)
            accepted_proposals.extend(proposals)
            attempt_records.extend(attempts)

        attempt_table = pd.DataFrame(attempt_records)
        output_file = self.configuration.output_settings.attempt_file
        output_file.parent.mkdir(parents=True, exist_ok=True)
        attempt_table.to_csv(output_file, index=False)
        return RefinementResult(accepted_proposals, attempt_table)

    def _refine_station_cluster(
        self, master: object
    ) -> tuple[list[PhasePickProposal], list[dict[str, object]]]:
        cluster_id = str(master.cluster_id)
        station_id = str(master.station_id)
        master_event_id = str(master.master_event_id)
        p_master_time = self._reviewed_or_original_time(
            master.reviewed_p_pick_time, master.p_pick_time
        )
        s_master_time = self._reviewed_or_original_time(
            master.reviewed_s_pick_time, master.s_pick_time
        )

        master_components = self.waveform_archive.station_components(
            master_event_id, station_id
        )
        p_templates = self._make_phase_templates(
            master_components, "P", p_master_time
        )
        s_templates = self._make_phase_templates(
            master_components, "S", s_master_time
        )
        if not p_templates or not s_templates:
            return [], [
                self._station_cluster_failure(
                    master,
                    "could not cut P and S templates from the selected master",
                )
            ]

        proposals: list[PhasePickProposal] = []
        attempts: list[dict[str, object]] = []
        for event_id in self.dataset.events_in_cluster(cluster_id):
            if event_id not in self.waveform_archive.event_ids:
                attempts.append(
                    self._attempt_record(
                        master, event_id, status="no waveform file"
                    )
                )
                continue
            if (
                event_id == master_event_id
                and not self.configuration.correlation_settings.include_master_in_cc
            ):
                attempts.append(
                    self._attempt_record(
                        master, event_id, status="master excluded from CC"
                    )
                )
                continue

            origin_time = self.dataset.origin_time(event_id)
            if origin_time is None:
                attempts.append(
                    self._attempt_record(
                        master, event_id, status="no catalog origin time"
                    )
                )
                continue

            existing_pick = self.dataset.best_pick_row(event_id, station_id)
            p_center, p_center_source, p_score, p_snr = (
                self._search_center(
                    existing_pick,
                    "P",
                    origin_time,
                    float(master.median_p_offset_seconds),
                )
            )
            s_center, s_center_source, s_score, s_snr = (
                self._search_center(
                    existing_pick,
                    "S",
                    origin_time,
                    float(master.median_s_offset_seconds),
                )
            )

            target_components = self.waveform_archive.station_components(
                event_id, station_id
            )
            p_hit = self._correlate_one_phase(
                target_components, p_templates, "P", p_center
            )
            s_hit = self._correlate_one_phase(
                target_components, s_templates, "S", s_center
            )
            status = self._acceptance_status(
                p_hit, s_hit, float(master.median_sp_seconds)
            )

            attempts.append(
                self._attempt_record(
                    master,
                    event_id,
                    status=status,
                    p_center=p_center,
                    p_center_source=p_center_source,
                    p_hit=p_hit,
                    s_center=s_center,
                    s_center_source=s_center_source,
                    s_hit=s_hit,
                )
            )
            if status != "accepted":
                continue

            proposals.extend(
                [
                    self._proposal(
                        event_id,
                        station_id,
                        cluster_id,
                        master_event_id,
                        "P",
                        p_center,
                        p_center_source,
                        p_hit,
                        p_score,
                        p_snr,
                    ),
                    self._proposal(
                        event_id,
                        station_id,
                        cluster_id,
                        master_event_id,
                        "S",
                        s_center,
                        s_center_source,
                        s_hit,
                        s_score,
                        s_snr,
                    ),
                ]
            )
        return proposals, attempts

    def _make_phase_templates(
        self,
        components: dict[str, Trace],
        phase: str,
        pick_time: pd.Timestamp,
    ) -> dict[str, np.ndarray]:
        if phase == "P":
            desired_components = ("Z",)
            time_window = (
                self.configuration.correlation_settings
                .p_template_window_seconds
            )
        else:
            desired_components = self.horizontal_components
            time_window = (
                self.configuration.correlation_settings
                .s_template_window_seconds
            )

        templates: dict[str, np.ndarray] = {}
        for component in desired_components:
            trace = components.get(component)
            if trace is None:
                continue
            waveform = self._cut_exact_window(trace, pick_time, time_window)
            if waveform is not None:
                templates[component] = waveform
        return templates

    def _correlate_one_phase(
        self,
        target_components: dict[str, Trace],
        templates: dict[str, np.ndarray],
        phase: str,
        search_center: pd.Timestamp,
    ) -> CorrelationHit | None:
        """Find the highest normalized CC on matching components."""
        if phase == "P":
            time_window = (
                self.configuration.correlation_settings
                .p_template_window_seconds
            )
        else:
            time_window = (
                self.configuration.correlation_settings
                .s_template_window_seconds
            )

        best_hit: CorrelationHit | None = None
        for component, template in templates.items():
            target_trace = target_components.get(component)
            if target_trace is None:
                continue

            hit = self._sliding_cc(
                target_trace,
                search_center,
                template,
                time_window,
            )
            if hit is None:
                continue
            hit.component = component
            if best_hit is None or hit.coefficient > best_hit.coefficient:
                best_hit = hit
        return best_hit

    def _sliding_cc(
        self,
        trace: Trace,
        search_center: pd.Timestamp,
        template: np.ndarray,
        template_window: tuple[float, float],
    ) -> CorrelationHit | None:
        """Slide a fixed template through a wider target-data segment."""
        sample_interval = float(trace.stats.delta)
        template_sample_count = len(template)
        search_half_width = (
            self.configuration.correlation_settings.search_half_width_seconds
        )
        extra_search_samples = int(
            round(2 * search_half_width / sample_interval)
        )
        segment_sample_count = template_sample_count + extra_search_samples
        if template_sample_count <= 1 or segment_sample_count < template_sample_count:
            return None

        segment_start = search_center + pd.Timedelta(
            seconds=template_window[0] - search_half_width
        )
        segment = self._cut_sample_count(
            trace,
            segment_start,
            segment_sample_count,
        )
        if segment is None:
            return None

        # "full" performs moving-window normalized CC, so amplitudes do not
        # need to match between the master and target event.
        cc_values = correlate_template(
            segment,
            template,
            mode="valid",
            normalize="full",
            demean=True,
        )
        cc_values = np.asarray(cc_values, dtype=float)
        cc_values[~np.isfinite(cc_values)] = -np.inf
        if len(cc_values) == 0 or np.all(np.isneginf(cc_values)):
            return None

        best_index = int(np.argmax(cc_values))
        best_coefficient = float(cc_values[best_index])
        shift_seconds = -search_half_width + best_index * sample_interval
        pick_time = search_center + pd.Timedelta(seconds=shift_seconds)
        return CorrelationHit(
            pick_time=pick_time,
            coefficient=best_coefficient,
            shift_seconds=shift_seconds,
            component="",
        )

    def _cut_exact_window(
        self,
        trace: Trace,
        center_time: pd.Timestamp,
        time_window: tuple[float, float],
    ) -> np.ndarray | None:
        sample_interval = float(trace.stats.delta)
        sample_count = int(
            round((time_window[1] - time_window[0]) / sample_interval)
        )
        start_time = center_time + pd.Timedelta(seconds=time_window[0])
        return self._cut_sample_count(trace, start_time, sample_count)

    @staticmethod
    def _cut_sample_count(
        trace: Trace,
        start_time: pd.Timestamp,
        sample_count: int,
    ) -> np.ndarray | None:
        if sample_count <= 1:
            return None

        sample_interval = float(trace.stats.delta)
        start_utc = UTCDateTime(start_time.to_pydatetime(warn=False))
        end_utc = start_utc + (sample_count - 1) * sample_interval
        cut_trace = trace.slice(
            starttime=start_utc,
            endtime=end_utc,
            nearest_sample=True,
        )
        if len(cut_trace.data) != sample_count:
            return None

        waveform = np.asarray(cut_trace.data, dtype=float).copy()
        if not np.all(np.isfinite(waveform)):
            return None
        waveform -= waveform.mean()
        if np.linalg.norm(waveform) <= 0:
            return None
        return waveform

    def _search_center(
        self,
        existing_pick: pd.Series | None,
        phase: str,
        origin_time: pd.Timestamp,
        median_offset_seconds: float,
    ) -> tuple[pd.Timestamp, str, float | None, float | None]:
        pick_column = "_p_pick_time" if phase == "P" else "_s_pick_time"
        score_column = "_p_score" if phase == "P" else "_s_score"
        if existing_pick is not None and pd.notna(existing_pick[pick_column]):
            return (
                existing_pick[pick_column],
                "ML",
                self._number_or_none(existing_pick[score_column]),
                self._number_or_none(existing_pick["_snr"]),
            )

        predicted_time = origin_time + pd.Timedelta(
            seconds=median_offset_seconds
        )
        return predicted_time, "cluster median", None, None

    def _acceptance_status(
        self,
        p_hit: CorrelationHit | None,
        s_hit: CorrelationHit | None,
        expected_sp_seconds: float,
    ) -> str:
        if p_hit is None:
            return "no usable P channel or correlation"
        if s_hit is None:
            return "no matching horizontal channel or correlation"

        threshold = self.configuration.correlation_settings.cc_threshold
        if p_hit.coefficient < threshold:
            return "P below CC threshold"
        if s_hit.coefficient < threshold:
            return "S below CC threshold"

        corrected_sp_seconds = (
            s_hit.pick_time - p_hit.pick_time
        ).total_seconds()
        if corrected_sp_seconds <= 0:
            return "corrected S is not after P"
        tolerance = (
            self.configuration.correlation_settings
            .sp_acceptance_tolerance_seconds
        )
        if abs(corrected_sp_seconds - expected_sp_seconds) > tolerance:
            return "corrected S-P outside tolerance"
        return "accepted"

    def _proposal(
        self,
        event_id: str,
        station_id: str,
        cluster_id: str,
        master_event_id: str,
        phase: str,
        search_center: pd.Timestamp,
        search_center_source: str,
        hit: CorrelationHit | None,
        phase_score: float | None,
        snr: float | None,
    ) -> PhasePickProposal:
        if hit is None:
            raise ValueError("Accepted proposal cannot have an empty CC hit")
        original_time = (
            search_center if search_center_source == "ML" else None
        )
        return PhasePickProposal(
            event_id=event_id,
            station_id=station_id,
            station=short_station_name(station_id),
            phase=phase,
            chosen_time=hit.pick_time,
            source="CC",
            original_time=original_time,
            search_center_time=search_center,
            search_center_source=search_center_source,
            cluster_id=cluster_id,
            master_event_id=master_event_id,
            cc=hit.coefficient,
            shift_seconds=hit.shift_seconds,
            cc_component=hit.component,
            cc_threshold=(
                self.configuration.correlation_settings.cc_threshold
            ),
            phase_score=phase_score,
            snr=snr,
        )

    def _attempt_record(
        self,
        master: object,
        event_id: str,
        status: str,
        p_center: pd.Timestamp | None = None,
        p_center_source: str = "",
        p_hit: CorrelationHit | None = None,
        s_center: pd.Timestamp | None = None,
        s_center_source: str = "",
        s_hit: CorrelationHit | None = None,
    ) -> dict[str, object]:
        corrected_sp = np.nan
        if p_hit is not None and s_hit is not None:
            corrected_sp = (s_hit.pick_time - p_hit.pick_time).total_seconds()
        return {
            "cluster_id": str(master.cluster_id),
            "station_id": str(master.station_id),
            "station": str(master.station),
            "master_event_id": str(master.master_event_id),
            "event_id": str(event_id),
            "status": status,
            "p_search_center": p_center,
            "p_search_source": p_center_source,
            "p_cc_time": p_hit.pick_time if p_hit else pd.NaT,
            "p_cc": p_hit.coefficient if p_hit else np.nan,
            "p_shift_seconds": p_hit.shift_seconds if p_hit else np.nan,
            "p_component": p_hit.component if p_hit else "",
            "s_search_center": s_center,
            "s_search_source": s_center_source,
            "s_cc_time": s_hit.pick_time if s_hit else pd.NaT,
            "s_cc": s_hit.coefficient if s_hit else np.nan,
            "s_shift_seconds": s_hit.shift_seconds if s_hit else np.nan,
            "s_component": s_hit.component if s_hit else "",
            "corrected_sp_seconds": corrected_sp,
            "cc_threshold": (
                self.configuration.correlation_settings.cc_threshold
            ),
        }

    @staticmethod
    def _station_cluster_failure(
        master: object, status: str
    ) -> dict[str, object]:
        return {
            "cluster_id": str(master.cluster_id),
            "station_id": str(master.station_id),
            "station": str(master.station),
            "master_event_id": str(master.master_event_id),
            "event_id": "",
            "status": status,
        }

    @staticmethod
    def _reviewed_or_original_time(
        reviewed_time: object, original_time: object
    ) -> pd.Timestamp:
        reviewed = pd.to_datetime(reviewed_time, utc=True, errors="coerce")
        if pd.notna(reviewed):
            return reviewed
        original = pd.to_datetime(original_time, utc=True, errors="coerce")
        if pd.isna(original):
            raise ValueError("Selected master has no usable phase time")
        return original

    @staticmethod
    def _number_or_none(value: object) -> float | None:
        return None if pd.isna(value) else float(value)

