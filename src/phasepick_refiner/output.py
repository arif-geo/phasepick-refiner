"""Schema-preserving refined-pick and provenance output."""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from .configuration import ProjectConfiguration
from .data import PickDataset, normalized_identifier, short_station_name
from .models import PhasePickProposal, RefinementResult, timestamp_text


class PickOutputWriter:
    """Apply accepted proposals without changing the user's table convention.

    The refined pick CSV has exactly the original column names and order. A
    separate provenance CSV records where each final phase came from and keeps
    CC metadata that may not belong in the user's normal processing schema.
    """

    source_priority = {"O": 0, "CC": 1, "C": 2}

    def __init__(
        self,
        configuration: ProjectConfiguration,
        dataset: PickDataset,
    ):
        self.configuration = configuration
        self.dataset = dataset

    def write(
        self,
        refinement_result: RefinementResult,
        master_table: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        proposals = list(refinement_result.proposals)
        proposals.extend(self._reviewed_master_proposals(master_table))
        chosen_proposals = self._choose_highest_priority_proposals(proposals)

        refined_picks = self.dataset.raw_picks.copy()
        original_column_order = list(refined_picks.columns)
        row_for_key = self._best_existing_row_indices()

        # Add at most one new row for each previously unpicked event/station.
        new_rows: dict[tuple[str, str], dict[str, object]] = {}
        for event_station_phase, proposal in chosen_proposals.items():
            event_id, station_id, phase = event_station_phase
            event_station_key = (event_id, station_id)

            if event_station_key in row_for_key:
                row_index = row_for_key[event_station_key]
            else:
                if event_station_key not in new_rows:
                    new_rows[event_station_key] = self._empty_user_row(
                        event_id, station_id
                    )
                self._put_phase_time_in_row(
                    new_rows[event_station_key],
                    phase,
                    proposal.chosen_time,
                )
                continue

            phase_column = self._phase_column(phase)
            refined_picks.at[row_index, phase_column] = (
                self._format_output_time(proposal.chosen_time)
            )

        if new_rows:
            new_row_table = pd.DataFrame(
                new_rows.values(), columns=original_column_order
            )
            # Pandas 2.2 warns about a future dtype inference change when a
            # newly added row has blank optional fields. The CSV values and
            # column order are intentional, so suppress only that warning.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=(
                        "The behavior of DataFrame concatenation with empty "
                        "or all-NA entries is deprecated"
                    ),
                    category=FutureWarning,
                )
                refined_picks = pd.concat(
                    [refined_picks, new_row_table],
                    ignore_index=True,
                )

        # Reindex explicitly: generated metadata never leaks into the main CSV.
        refined_picks = refined_picks.reindex(columns=original_column_order)
        provenance = self._build_provenance(
            refined_picks, chosen_proposals
        )

        output_settings = self.configuration.output_settings
        output_settings.directory.mkdir(parents=True, exist_ok=True)
        refined_picks.to_csv(output_settings.refined_pick_file, index=False)
        provenance.to_csv(output_settings.provenance_file, index=False)
        return refined_picks, provenance

    def _reviewed_master_proposals(
        self, master_table: pd.DataFrame
    ) -> list[PhasePickProposal]:
        proposals: list[PhasePickProposal] = []
        for master in master_table.itertuples(index=False):
            event_id = str(master.master_event_id)
            station_id = str(master.station_id)
            if not event_id or event_id.lower() == "nan":
                continue

            original_pick = self.dataset.best_pick_row(event_id, station_id)
            for phase in ["P", "S"]:
                edited = bool(getattr(master, f"{phase.lower()}_pick_edited"))
                if not edited:
                    continue
                reviewed_time = pd.to_datetime(
                    getattr(master, f"reviewed_{phase.lower()}_pick_time"),
                    utc=True,
                    errors="coerce",
                )
                if pd.isna(reviewed_time):
                    continue

                original_time = None
                phase_score = None
                snr = None
                if original_pick is not None:
                    original_time = original_pick[
                        f"_{phase.lower()}_pick_time"
                    ]
                    phase_score = self._number_or_none(
                        original_pick[f"_{phase.lower()}_score"]
                    )
                    snr = self._number_or_none(original_pick["_snr"])

                proposals.append(
                    PhasePickProposal(
                        event_id=event_id,
                        station_id=station_id,
                        station=short_station_name(station_id),
                        phase=phase,
                        chosen_time=reviewed_time,
                        source="C",
                        original_time=original_time,
                        search_center_time=original_time,
                        search_center_source="manual master review",
                        cluster_id=str(master.cluster_id),
                        master_event_id=event_id,
                        phase_score=phase_score,
                        snr=snr,
                    )
                )
        return proposals

    def _choose_highest_priority_proposals(
        self, proposals: list[PhasePickProposal]
    ) -> dict[tuple[str, str, str], PhasePickProposal]:
        chosen: dict[tuple[str, str, str], PhasePickProposal] = {}
        for proposal in proposals:
            key = (
                normalized_identifier(proposal.event_id),
                str(proposal.station_id).strip(),
                str(proposal.phase).upper(),
            )
            previous = chosen.get(key)
            if previous is None:
                chosen[key] = proposal
                continue

            new_priority = self.source_priority.get(proposal.source, -1)
            old_priority = self.source_priority.get(previous.source, -1)
            if new_priority > old_priority:
                chosen[key] = proposal
            elif new_priority == old_priority:
                new_cc = -np.inf if proposal.cc is None else proposal.cc
                old_cc = -np.inf if previous.cc is None else previous.cc
                if new_cc > old_cc:
                    chosen[key] = proposal
        return chosen

    def _best_existing_row_indices(
        self,
    ) -> dict[tuple[str, str], int]:
        indices: dict[tuple[str, str], int] = {}
        unique_keys = self.dataset.picks[
            ["_event_id", "_station_id"]
        ].drop_duplicates()
        for event_id, station_id in unique_keys.to_numpy():
            best_row = self.dataset.best_pick_row(
                event_id, station_id
            )
            if best_row is not None:
                indices[(str(event_id), str(station_id))] = int(
                    best_row.name
                )
        return indices

    def _empty_user_row(
        self, event_id: str, station_id: str
    ) -> dict[str, object]:
        columns = self.configuration.pick_columns
        row = {column: pd.NA for column in self.dataset.raw_picks.columns}

        # Event-level fields such as waveform filename can be copied from any
        # existing station row for the same event.
        matching_event_rows = self.dataset.picks[
            self.dataset.picks["_event_id"].eq(event_id)
        ]
        if not matching_event_rows.empty:
            source_index = matching_event_rows.index[0]
            source_row = self.dataset.raw_picks.loc[source_index]
            for column in (
                self.configuration.output_settings
                .event_level_columns_for_new_rows
            ):
                if column in row:
                    row[column] = source_row[column]

        row[columns.event_id] = event_id
        row[columns.station_id] = station_id
        waveform_column = columns.waveform_filename
        if waveform_column and waveform_column in row and pd.isna(
            row[waveform_column]
        ):
            separator = (
                self.configuration.waveform_settings.event_id_separator
            )
            matching_files = sorted(
                self.configuration.input_paths.waveform_directory.glob(
                    f"{event_id}{separator}*"
                )
            )
            if matching_files:
                row[waveform_column] = matching_files[0].name
        return row

    def _put_phase_time_in_row(
        self,
        row: dict[str, object],
        phase: str,
        pick_time: pd.Timestamp,
    ) -> None:
        row[self._phase_column(phase)] = self._format_output_time(pick_time)

    def _build_provenance(
        self,
        refined_picks: pd.DataFrame,
        chosen_proposals: dict[
            tuple[str, str, str], PhasePickProposal
        ],
    ) -> pd.DataFrame:
        columns = self.configuration.pick_columns
        provenance_records: list[dict[str, object]] = []

        working = refined_picks.copy()
        working["_event_id"] = working[columns.event_id].map(
            normalized_identifier
        )
        working["_station_id"] = (
            working[columns.station_id].fillna("").astype(str).str.strip()
        )
        working["_p_time"] = pd.to_datetime(
            working[columns.p_pick_time],
            utc=True,
            errors="coerce",
            format="mixed",
        )
        working["_s_time"] = pd.to_datetime(
            working[columns.s_pick_time],
            utc=True,
            errors="coerce",
            format="mixed",
        )

        # One provenance row per event/station/phase, even if the input had
        # duplicate event/station rows.
        for (event_id, station_id), group in working.groupby(
            ["_event_id", "_station_id"], sort=True
        ):
            original_pick = self.dataset.best_pick_row(event_id, station_id)
            for phase, time_column in [("P", "_p_time"), ("S", "_s_time")]:
                available_times = group[time_column].dropna()
                if available_times.empty:
                    continue
                final_time = available_times.iloc[-1]
                proposal = chosen_proposals.get(
                    (str(event_id), str(station_id), phase)
                )
                provenance_records.append(
                    self._provenance_record(
                        str(event_id),
                        str(station_id),
                        phase,
                        final_time,
                        original_pick,
                        proposal,
                    )
                )

        provenance_columns = [
            "event_id",
            "station_id",
            "station",
            "phase",
            "source",
            "original_time",
            "chosen_time",
            "search_center_time",
            "search_center_source",
            "cc",
            "shift_seconds",
            "cc_component",
            "master_event_id",
            "cc_threshold",
            "cluster_id",
            "phase_score",
            "snr",
        ]
        return pd.DataFrame(
            provenance_records, columns=provenance_columns
        ).sort_values(
            ["event_id", "station_id", "phase"]
        ).reset_index(drop=True)

    def _provenance_record(
        self,
        event_id: str,
        station_id: str,
        phase: str,
        final_time: pd.Timestamp,
        original_pick: pd.Series | None,
        proposal: PhasePickProposal | None,
    ) -> dict[str, object]:
        original_time = pd.NaT
        phase_score = np.nan
        snr = np.nan
        if original_pick is not None:
            original_time = original_pick[
                f"_{phase.lower()}_pick_time"
            ]
            phase_score = original_pick[f"_{phase.lower()}_score"]
            snr = original_pick["_snr"]

        if proposal is None:
            return {
                "event_id": event_id,
                "station_id": station_id,
                "station": short_station_name(station_id),
                "phase": phase,
                "source": "O",
                "original_time": timestamp_text(original_time),
                "chosen_time": timestamp_text(final_time),
                "search_center_time": "",
                "search_center_source": "",
                "cc": np.nan,
                "shift_seconds": np.nan,
                "cc_component": "",
                "master_event_id": "",
                "cc_threshold": np.nan,
                "cluster_id": self.dataset.event_to_cluster.get(event_id, ""),
                "phase_score": phase_score,
                "snr": snr,
            }

        return {
            "event_id": event_id,
            "station_id": station_id,
            "station": proposal.station,
            "phase": phase,
            "source": proposal.source,
            "original_time": timestamp_text(original_time),
            "chosen_time": timestamp_text(proposal.chosen_time),
            "search_center_time": timestamp_text(
                proposal.search_center_time
            ),
            "search_center_source": proposal.search_center_source,
            "cc": proposal.cc,
            "shift_seconds": proposal.shift_seconds,
            "cc_component": proposal.cc_component,
            "master_event_id": proposal.master_event_id,
            "cc_threshold": proposal.cc_threshold,
            "cluster_id": proposal.cluster_id,
            "phase_score": (
                proposal.phase_score
                if proposal.phase_score is not None
                else phase_score
            ),
            "snr": proposal.snr if proposal.snr is not None else snr,
        }

    def _phase_column(self, phase: str) -> str:
        if phase.upper() == "P":
            return self.configuration.pick_columns.p_pick_time
        if phase.upper() == "S":
            return self.configuration.pick_columns.s_pick_time
        raise ValueError(f"Unknown phase: {phase}")

    def _format_output_time(self, value: object) -> str:
        timestamp = pd.to_datetime(value, utc=True, errors="coerce")
        if pd.isna(timestamp):
            return ""
        return timestamp.strftime(
            self.configuration.output_settings.output_time_format
        )

    @staticmethod
    def _number_or_none(value: object) -> float | None:
        return None if pd.isna(value) else float(value)
