"""Selection and persistence of station-local P/S master events."""

from pathlib import Path

import numpy as np
import pandas as pd

from .configuration import ProjectConfiguration
from .data import PickDataset, short_station_name
from .models import MasterSelection, timestamp_text


class MasterSelector:
    """Choose one common P/S master for every station-cluster pair."""

    def __init__(
        self,
        configuration: ProjectConfiguration,
        dataset: PickDataset,
    ):
        self.configuration = configuration
        self.dataset = dataset

    def select_all(self) -> pd.DataFrame:
        cluster_ids = self.dataset.selected_cluster_ids()
        master_table = self._select_clusters(cluster_ids)
        self.save(master_table)
        return master_table

    def _select_clusters(
        self, cluster_ids: list[str]
    ) -> pd.DataFrame:
        """Select station-local masters for the supplied cluster IDs."""
        selections: list[dict[str, object]] = []

        for position, cluster_id in enumerate(cluster_ids, start=1):
            cluster_event_ids = self.dataset.events_in_cluster(cluster_id)
            print(
                f"[masters {position}/{len(cluster_ids)}] "
                f"cluster {cluster_id}: {len(cluster_event_ids)} events"
            )
            for station_id, station_picks in (
                self.dataset.station_groups_for_cluster(cluster_id)
            ):
                selection = self.select_one(
                    cluster_id,
                    station_id,
                    station_picks,
                )
                selections.append(selection.to_record())

        master_table = pd.DataFrame(
            selections,
            columns=list(MasterSelection.__dataclass_fields__),
        )
        return master_table

    def ensure_configured_clusters(self) -> pd.DataFrame:
        """Append newly configured clusters without replacing reviewed rows."""
        master_file = self.configuration.output_settings.master_file
        if not master_file.exists():
            return self.select_all()

        existing_table = self.load()
        existing_cluster_ids = set(
            existing_table["cluster_id"].astype(str)
        )

        # Clusters with no pick rows cannot produce a station-local master or
        # appear in the reviewer, so they do not need repeated selection runs.
        clusters_with_picks = {
            self.dataset.event_to_cluster.get(str(event_id))
            for event_id in self.dataset.picks["_event_id"]
        }
        desired_cluster_ids = [
            cluster_id
            for cluster_id in self.dataset.selected_cluster_ids()
            if cluster_id in clusters_with_picks
        ]
        missing_cluster_ids = [
            cluster_id
            for cluster_id in desired_cluster_ids
            if cluster_id not in existing_cluster_ids
        ]
        if not missing_cluster_ids:
            return existing_table

        print(
            "Extending saved master selections with "
            f"{len(missing_cluster_ids)} newly configured clusters"
        )
        added_table = self._select_clusters(missing_cluster_ids)
        if added_table.empty:
            return existing_table

        # Existing rows win on duplicate station-cluster keys. They may carry
        # manual picks that must survive a broader configuration run.
        combined_table = pd.concat(
            [existing_table, added_table],
            ignore_index=True,
        )
        combined_table = combined_table.drop_duplicates(
            ["cluster_id", "station_id"],
            keep="first",
        )
        self.save(combined_table)
        return combined_table

    def select_one(
        self,
        cluster_id: str,
        station_id: str,
        station_picks: pd.DataFrame,
    ) -> MasterSelection:
        cluster_event_count = len(self.dataset.events_in_cluster(cluster_id))
        pair_table = self._build_pair_table(station_picks)

        selection = MasterSelection(
            cluster_id=str(cluster_id),
            station_id=str(station_id),
            station=short_station_name(station_id),
            cluster_event_count=cluster_event_count,
            observed_pair_count=len(pair_table),
            selection_status="no usable ML P/S pairs",
        )
        if pair_table.empty:
            return selection

        settings = self.configuration.master_settings
        if len(pair_table) == 1:
            candidate = pair_table.iloc[0]
            if not settings.allow_single_high_confidence_pair:
                selection.selection_status = "single P/S pair disabled"
                return selection
            if not self._single_pair_is_confident(candidate):
                selection.selection_status = "single P/S pair below confidence"
                return selection

            median_p_offset = float(candidate["_p_offset_seconds"])
            median_s_offset = float(candidate["_s_offset_seconds"])
            median_sp = float(candidate["_sp_seconds"])
            selection_status = "selected high-confidence single pair"
        elif len(pair_table) < settings.minimum_pick_pairs:
            selection.selection_status = (
                f"fewer than {settings.minimum_pick_pairs} usable ML P/S pairs"
            )
            return selection
        else:
            median_p_offset = float(pair_table["_p_offset_seconds"].median())
            median_s_offset = float(pair_table["_s_offset_seconds"].median())
            median_sp = float(pair_table["_sp_seconds"].median())

            # Favor high score/SNR picks whose S-P closely matches the cluster.
            consistent_candidates = pair_table[
                (pair_table["_sp_seconds"] - median_sp).abs()
                <= settings.sp_template_tolerance_seconds
            ].copy()
            if consistent_candidates.empty:
                consistent_candidates = pair_table.copy()
            consistent_candidates["_master_quality"] = (
                consistent_candidates["_quality"]
                - (consistent_candidates["_sp_seconds"] - median_sp).abs()
            )
            candidate = consistent_candidates.sort_values(
                "_master_quality", ascending=False
            ).iloc[0]
            selection_status = "selected"

        master_quality = candidate.get("_master_quality", candidate["_quality"])
        selection.selection_status = selection_status
        selection.master_event_id = str(candidate["_event_id"])
        selection.p_pick_time = timestamp_text(candidate["_p_pick_time"])
        selection.s_pick_time = timestamp_text(candidate["_s_pick_time"])
        selection.reviewed_p_pick_time = selection.p_pick_time
        selection.reviewed_s_pick_time = selection.s_pick_time
        selection.p_score = self._number_or_none(candidate["_p_score"])
        selection.s_score = self._number_or_none(candidate["_s_score"])
        selection.snr = self._number_or_none(candidate["_snr"])
        selection.median_p_offset_seconds = median_p_offset
        selection.median_s_offset_seconds = median_s_offset
        selection.median_sp_seconds = median_sp
        selection.master_quality_score = self._number_or_none(master_quality)
        return selection

    def _build_pair_table(
        self, station_picks: pd.DataFrame
    ) -> pd.DataFrame:
        rows = station_picks.copy()
        rows["_origin_time"] = rows["_event_id"].map(
            lambda event_id: self.dataset.origin_time(event_id)
        )
        rows = rows.dropna(
            subset=["_origin_time", "_p_pick_time", "_s_pick_time"]
        )

        rows["_p_offset_seconds"] = (
            rows["_p_pick_time"] - rows["_origin_time"]
        ).dt.total_seconds()
        rows["_s_offset_seconds"] = (
            rows["_s_pick_time"] - rows["_origin_time"]
        ).dt.total_seconds()
        rows["_sp_seconds"] = (
            rows["_s_pick_time"] - rows["_p_pick_time"]
        ).dt.total_seconds()
        rows = rows[rows["_sp_seconds"] > 0].copy()

        settings = self.configuration.master_settings
        rows["_quality"] = (
            rows["_p_score"].fillna(0)
            + rows["_s_score"].fillna(0)
            + settings.snr_weight
            * np.log1p(rows["_snr"].clip(lower=0).fillna(0))
        )
        return (
            rows.sort_values("_quality", ascending=False)
            .drop_duplicates("_event_id", keep="first")
            .copy()
        )

    def _single_pair_is_confident(self, candidate: pd.Series) -> bool:
        settings = self.configuration.master_settings
        p_score = self._number_or_negative_infinity(candidate["_p_score"])
        s_score = self._number_or_negative_infinity(candidate["_s_score"])
        snr = self._number_or_negative_infinity(candidate["_snr"])
        return (
            p_score >= settings.single_pair_minimum_p_score
            and s_score >= settings.single_pair_minimum_s_score
            and snr >= settings.single_pair_minimum_snr
        )

    @staticmethod
    def _number_or_none(value: object) -> float | None:
        return None if pd.isna(value) else float(value)

    @staticmethod
    def _number_or_negative_infinity(value: object) -> float:
        return -np.inf if pd.isna(value) else float(value)

    def save(self, master_table: pd.DataFrame) -> Path:
        output_file = self.configuration.output_settings.master_file
        output_file.parent.mkdir(parents=True, exist_ok=True)
        master_table.to_csv(output_file, index=False)
        return output_file

    def load(self, filename: str | Path | None = None) -> pd.DataFrame:
        master_file = (
            Path(filename)
            if filename is not None
            else self.configuration.output_settings.master_file
        )
        if not master_file.exists():
            raise FileNotFoundError(
                f"Master-selection file does not exist: {master_file}\n"
                "Run the select-masters command first."
            )
        master_table = pd.read_csv(
            master_file,
            dtype={
                "cluster_id": str,
                "station_id": str,
                "master_event_id": str,
            },
            low_memory=False,
        )
        for column in ["p_pick_edited", "s_pick_edited"]:
            if column not in master_table:
                master_table[column] = False
            else:
                master_table[column] = (
                    master_table[column]
                    .fillna(False)
                    .astype(str)
                    .str.lower()
                    .isin(["true", "1", "yes"])
                )
        return master_table
