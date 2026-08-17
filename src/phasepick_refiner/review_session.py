"""State and scientific rules for the master-review GUI."""

import math
import warnings

import pandas as pd
from obspy import Trace

from .configuration import ProjectConfiguration
from .data import PickDataset, natural_sort_key, short_station_name
from .masters import MasterSelector
from .models import MasterSelection, timestamp_text
from .waveforms import WaveformArchive


class MasterReviewSession:
    """Keep GUI state separate from Qt widgets and Matplotlib drawing."""

    def __init__(
        self,
        configuration: ProjectConfiguration,
        dataset: PickDataset,
        waveform_archive: WaveformArchive,
    ):
        self.configuration = configuration
        self.dataset = dataset
        self.waveform_archive = waveform_archive
        self.master_selector = MasterSelector(configuration, dataset)
        self.master_table = self.master_selector.load()
        self.master_table_at_open = self.master_table.copy(deep=True)

        self.cluster_id = ""
        self.event_id = ""
        self.station_groups: dict[str, dict[str, Trace]] = {}
        self.station_ids: list[str] = []
        self.current_page = 0
        self.selected_station_id = ""
        self.pending_manual_picks: dict[
            tuple[str, str, str], dict[str, pd.Timestamp]
        ] = {}

    @property
    def stations_per_page(self) -> int:
        return self.configuration.viewer_settings.stations_per_page

    def cluster_ids(self) -> list[str]:
        selected = self.selected_master_rows()
        return sorted(
            selected["cluster_id"].astype(str).unique(),
            key=natural_sort_key,
        )

    def selected_master_rows(self) -> pd.DataFrame:
        master_event_ids = (
            self.master_table["master_event_id"]
            .fillna("")
            .astype(str)
            .str.strip()
        )
        configured_cluster_ids = set(
            self.dataset.selected_cluster_ids()
        )
        return self.master_table[
            master_event_ids.ne("")
            & master_event_ids.ne("nan")
            & self.master_table["cluster_id"]
            .astype(str)
            .isin(configured_cluster_ids)
        ].copy()

    def master_events(self, cluster_id: str) -> list[str]:
        selected = self.selected_master_rows()
        cluster_rows = selected[
            selected["cluster_id"].astype(str).eq(str(cluster_id))
        ]
        return sorted(
            cluster_rows["master_event_id"].astype(str).unique(),
            key=natural_sort_key,
        )

    def load_event(self, cluster_id: str, event_id: str) -> None:
        self.cluster_id = str(cluster_id)
        self.event_id = str(event_id)
        waveform_station_groups = (
            self.waveform_archive.event_station_components(
                event_id, apply_filter=False
            )
        )
        # The station inventory is authoritative. Event files sometimes carry
        # extra channels or temporary stations that should not enter review.
        self.station_groups = {
            station_id: components
            for station_id, components in waveform_station_groups.items()
            if self.dataset.station_is_catalogued(station_id)
        }
        self.station_ids = sorted(
            self.station_groups,
            key=self._station_distance_sort_key,
        )
        self.current_page = 0
        self.selected_station_id = (
            self.station_ids[0] if self.station_ids else ""
        )

    def _station_distance_sort_key(
        self, station_id: str
    ) -> tuple[bool, float, str, str]:
        """Put nearby stations first and unknown distances at the end."""
        distance_km = self.dataset.station_distance_km(
            self.event_id, station_id
        )
        return (
            distance_km is None,
            distance_km if distance_km is not None else float("inf"),
            short_station_name(station_id).lower(),
            station_id.lower(),
        )

    def station_distance_km(self, station_id: str) -> float | None:
        return self.dataset.station_distance_km(
            self.event_id, station_id
        )

    def page_count(self) -> int:
        if not self.station_ids:
            return 1
        return max(
            1,
            math.ceil(len(self.station_ids) / self.stations_per_page),
        )

    def stations_on_page(self) -> list[str]:
        start = self.current_page * self.stations_per_page
        end = start + self.stations_per_page
        return self.station_ids[start:end]

    def change_page(self, step: int) -> None:
        self.current_page = (
            self.current_page + step
        ) % self.page_count()
        page_stations = self.stations_on_page()
        self.selected_station_id = (
            page_stations[0] if page_stations else ""
        )

    def master_row_index(self, station_id: str) -> int | None:
        matches = self.master_table[
            self.master_table["cluster_id"]
            .astype(str)
            .eq(self.cluster_id)
            & self.master_table["station_id"]
            .astype(str)
            .eq(str(station_id))
        ]
        if matches.empty:
            return None
        return int(matches.index[0])

    def master_row(self, station_id: str) -> pd.Series | None:
        row_index = self.master_row_index(station_id)
        if row_index is None:
            return None
        return self.master_table.loc[row_index]

    def is_local_master(self, station_id: str) -> bool:
        row = self.master_row(station_id)
        if row is None:
            return False
        return str(row["master_event_id"]) == self.event_id

    def original_pick(self, station_id: str) -> pd.Series | None:
        return self.dataset.best_pick_row(self.event_id, station_id)

    def pending_picks(
        self, station_id: str
    ) -> dict[str, pd.Timestamp]:
        key = (self.cluster_id, self.event_id, station_id)
        return self.pending_manual_picks.get(key, {})

    def record_manual_pick(
        self,
        station_id: str,
        phase: str,
        reviewed_time: pd.Timestamp,
    ) -> str:
        row_index = self.master_row_index(station_id)

        # Editing an existing local master can be recorded immediately.
        if (
            row_index is not None
            and str(self.master_table.at[row_index, "master_event_id"])
            == self.event_id
        ):
            phase_name = phase.lower()
            self.master_table.at[
                row_index, f"reviewed_{phase_name}_pick_time"
            ] = timestamp_text(reviewed_time)
            self.master_table.at[
                row_index, f"{phase_name}_pick_edited"
            ] = True
            return (
                f"Updated {phase} for local master {self.event_id} at "
                f"{short_station_name(station_id)}"
            )

        # A different event becomes the local master only when both P and S
        # exist. An original ML pick may supply the unedited companion phase.
        pending_key = (self.cluster_id, self.event_id, station_id)
        pending = self.pending_manual_picks.setdefault(pending_key, {})
        pending[phase] = reviewed_time
        original_pick = self.original_pick(station_id)
        p_time = self._pending_or_original_time(
            pending, original_pick, "P"
        )
        s_time = self._pending_or_original_time(
            pending, original_pick, "S"
        )
        if p_time is not None and s_time is not None:
            if s_time <= p_time:
                return (
                    f"S must be after P at "
                    f"{short_station_name(station_id)}; repick the "
                    "incorrect phase"
                )
            self._promote_station_master(
                station_id,
                p_time,
                s_time,
                pending,
                original_pick,
            )
            self.pending_manual_picks.pop(pending_key, None)
            return (
                f"Set event {self.event_id} as the local master at "
                f"{short_station_name(station_id)}"
            )

        missing_phase = "S" if p_time is not None else "P"
        return (
            f"Stored {phase} at {short_station_name(station_id)}; "
            f"pick {missing_phase} to create a complete local master"
        )

    @staticmethod
    def _pending_or_original_time(
        pending: dict[str, pd.Timestamp],
        original_pick: pd.Series | None,
        phase: str,
    ) -> pd.Timestamp | None:
        if phase in pending:
            return pending[phase]
        if original_pick is None:
            return None
        original_time = original_pick[
            f"_{phase.lower()}_pick_time"
        ]
        if pd.isna(original_time):
            return None
        return original_time

    def _promote_station_master(
        self,
        station_id: str,
        p_time: pd.Timestamp,
        s_time: pd.Timestamp,
        manual_phases: dict[str, pd.Timestamp],
        original_pick: pd.Series | None,
    ) -> None:
        origin_time = self.dataset.origin_time(self.event_id)
        if origin_time is None:
            return

        p_original = (
            original_pick["_p_pick_time"]
            if original_pick is not None
            else pd.NaT
        )
        s_original = (
            original_pick["_s_pick_time"]
            if original_pick is not None
            else pd.NaT
        )
        cluster_picks = self.dataset.picks_for_cluster(self.cluster_id)
        station_picks = cluster_picks[
            cluster_picks["_station_id"].eq(station_id)
        ]
        observed_pair_count = int(
            (
                station_picks["_p_pick_time"].notna()
                & station_picks["_s_pick_time"].notna()
            ).sum()
        )
        p_offset = (p_time - origin_time).total_seconds()
        s_offset = (s_time - origin_time).total_seconds()

        selection = MasterSelection(
            cluster_id=self.cluster_id,
            station_id=station_id,
            station=short_station_name(station_id),
            cluster_event_count=len(
                self.dataset.events_in_cluster(self.cluster_id)
            ),
            observed_pair_count=observed_pair_count,
            selection_status="selected manually in reviewer",
            master_event_id=self.event_id,
            p_pick_time=timestamp_text(p_original),
            s_pick_time=timestamp_text(s_original),
            reviewed_p_pick_time=timestamp_text(p_time),
            reviewed_s_pick_time=timestamp_text(s_time),
            p_pick_edited="P" in manual_phases,
            s_pick_edited="S" in manual_phases,
            p_score=self._pick_number(original_pick, "_p_score"),
            s_score=self._pick_number(original_pick, "_s_score"),
            snr=self._pick_number(original_pick, "_snr"),
            median_p_offset_seconds=p_offset,
            median_s_offset_seconds=s_offset,
            median_sp_seconds=(s_time - p_time).total_seconds(),
            master_quality_score=None,
        )
        record = selection.to_record()
        existing_index = self.master_row_index(station_id)
        if existing_index is None:
            self._append_master_record(record)
        else:
            for column, value in record.items():
                self.master_table.at[existing_index, column] = value

    @staticmethod
    def _pick_number(
        pick_row: pd.Series | None, column: str
    ) -> float | None:
        if pick_row is None or pd.isna(pick_row[column]):
            return None
        return float(pick_row[column])

    def reset_selected_station(self) -> str:
        station_id = self.selected_station_id
        if not station_id:
            return "Select a station row first"

        current_index = self.master_row_index(station_id)
        original_matches = self.master_table_at_open[
            self.master_table_at_open["cluster_id"]
            .astype(str)
            .eq(self.cluster_id)
            & self.master_table_at_open["station_id"]
            .astype(str)
            .eq(station_id)
        ]
        if original_matches.empty:
            if current_index is not None:
                self.master_table = self.master_table.drop(
                    index=current_index
                )
        else:
            original_record = original_matches.iloc[0]
            if current_index is None:
                self._append_master_record(original_record.to_dict())
            else:
                for column in self.master_table.columns:
                    self.master_table.at[current_index, column] = (
                        original_record[column]
                    )

        pending_keys = [
            key
            for key in self.pending_manual_picks
            if key[0] == self.cluster_id and key[2] == station_id
        ]
        for key in pending_keys:
            self.pending_manual_picks.pop(key, None)
        return (
            f"Reset {short_station_name(station_id)} to its state "
            "when the reviewer opened"
        )

    def _append_master_record(
        self, record: dict[str, object]
    ) -> None:
        new_row = pd.DataFrame(
            [record], columns=self.master_table.columns
        )
        # Blank optional metadata is intentional in a new manual-master row.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=(
                    "The behavior of DataFrame concatenation with empty "
                    "or all-NA entries is deprecated"
                ),
                category=FutureWarning,
            )
            self.master_table = pd.concat(
                [self.master_table, new_row],
                ignore_index=True,
            )

    def station_status(self, station_id: str) -> str:
        if self.is_local_master(station_id):
            return "this event is the local master"
        if self.original_pick(station_id) is not None:
            return "ML pick exists; another event is the local master"
        if (
            self.station_distance_km(station_id) is not None
            and self.dataset.event_depth_km(self.event_id) is not None
        ):
            return "waveform only; theoretical arrivals available"
        master_row = self.master_row(station_id)
        if (
            master_row is not None
            and pd.notna(master_row["median_p_offset_seconds"])
        ):
            return "no ML pick; cluster timing estimate available"
        return "waveform only; no timing model"

    def save(self) -> tuple[object, int, int]:
        output_file = self.master_selector.save(self.master_table)
        edited_phase_count = int(
            self.master_table["p_pick_edited"].fillna(False).astype(bool).sum()
            + self.master_table["s_pick_edited"].fillna(False).astype(bool).sum()
        )
        pending_count = sum(
            len(phases) for phases in self.pending_manual_picks.values()
        )
        return output_file, edited_phase_count, pending_count
