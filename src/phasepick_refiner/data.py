"""Input loading, validation, and schema-aware data access."""

from collections.abc import Iterator
import json
from pathlib import Path
import re

import numpy as np
from obspy.geodetics.base import gps2dist_azimuth
import pandas as pd

from .configuration import ProjectConfiguration
from .models import ValidationReport


def normalized_identifier(value: object) -> str:
    """Normalize IDs while preserving non-numeric catalog identifiers."""
    text = str(value).strip()
    if re.fullmatch(r"[+-]?\d+\.0", text):
        return text[:-2]
    return text


def short_station_name(station_id: object) -> str:
    """Return STA from the common NET.STA.LOC.CHA station identifier."""
    parts = str(station_id).strip().split(".")
    if len(parts) >= 2 and parts[1]:
        return parts[1]
    return str(station_id).strip()


def clean_cluster_id(cluster_id: object) -> str:
    text = str(cluster_id).strip()
    if text.startswith("group_"):
        return text.split("_", 1)[1]
    return text


def natural_sort_key(value: object) -> tuple[object, ...]:
    """Sort cluster 10 after cluster 9 instead of after cluster 1."""
    pieces = re.split(r"(\d+)", str(value))
    return tuple(
        int(piece) if piece.isdigit() else piece.lower()
        for piece in pieces
    )


class PickDataset:
    """Loaded project data with both raw and normalized pick representations.

    `raw_picks` keeps the user's original columns for the final output.
    `picks` contains parsed times and numeric quality values for calculations.
    Keeping both forms in one object makes it difficult to accidentally lose
    the user's schema while doing scientific processing.
    """

    def __init__(self, configuration: ProjectConfiguration):
        self.configuration = configuration
        self.raw_picks = pd.DataFrame()
        self.picks = pd.DataFrame()
        self.catalog = pd.DataFrame()
        self.stations = pd.DataFrame()
        self.cluster_to_events: dict[str, list[str]] = {}
        self.event_to_cluster: dict[str, str] = {}
        self._best_pick_indices: dict[tuple[str, str], int] = {}
        self._station_lookup: dict[tuple[str, str], pd.Series] = {}

    def load(self) -> "PickDataset":
        self._check_input_files_exist()
        self._load_pick_table()
        self._load_catalog()
        self._load_clusters()
        self._load_station_table()
        return self

    def _check_input_files_exist(self) -> None:
        input_paths = self.configuration.input_paths
        required_paths = [
            input_paths.pick_file,
            input_paths.catalog_file,
            input_paths.cluster_file,
            input_paths.waveform_directory,
        ]
        missing_paths = [path for path in required_paths if not path.exists()]
        if missing_paths:
            formatted = "\n".join(f"  {path}" for path in missing_paths)
            raise FileNotFoundError(f"Required input paths are missing:\n{formatted}")
        if not input_paths.waveform_directory.is_dir():
            raise NotADirectoryError(input_paths.waveform_directory)

    def _load_pick_table(self) -> None:
        columns = self.configuration.pick_columns
        self.raw_picks = pd.read_csv(
            self.configuration.input_paths.pick_file,
            dtype={columns.event_id: str, columns.station_id: str},
            low_memory=False,
        )
        missing_columns = set(columns.required_columns()) - set(
            self.raw_picks.columns
        )
        if missing_columns:
            raise ValueError(
                "Pick table is missing required columns: "
                + ", ".join(sorted(missing_columns))
            )

        working = self.raw_picks.copy()
        working["_event_id"] = working[columns.event_id].map(
            normalized_identifier
        )
        working["_station_id"] = (
            working[columns.station_id].fillna("").astype(str).str.strip()
        )
        working["_station"] = working["_station_id"].map(short_station_name)
        working["_p_pick_time"] = pd.to_datetime(
            working[columns.p_pick_time],
            utc=True,
            errors="coerce",
            format="mixed",
        )
        working["_s_pick_time"] = pd.to_datetime(
            working[columns.s_pick_time],
            utc=True,
            errors="coerce",
            format="mixed",
        )

        optional_numeric_columns = {
            "_p_score": columns.p_score,
            "_s_score": columns.s_score,
            "_snr": columns.snr,
        }
        for internal_name, user_column_name in optional_numeric_columns.items():
            if user_column_name and user_column_name in working.columns:
                working[internal_name] = pd.to_numeric(
                    working[user_column_name], errors="coerce"
                )
            else:
                working[internal_name] = np.nan

        self.picks = working
        self._build_best_pick_index()

    def _build_best_pick_index(self) -> None:
        """Rank duplicate rows once instead of rescanning for every lookup."""
        ranked = self.picks.copy()
        ranked["_phase_count"] = (
            ranked["_p_pick_time"].notna().astype(int)
            + ranked["_s_pick_time"].notna().astype(int)
        )
        ranked["_quality"] = (
            ranked["_p_score"].fillna(0)
            + ranked["_s_score"].fillna(0)
            + self.configuration.master_settings.snr_weight
            * np.log1p(ranked["_snr"].clip(lower=0).fillna(0))
        )
        best_rows = ranked.sort_values(
            ["_phase_count", "_quality"], ascending=False
        ).drop_duplicates(["_event_id", "_station_id"], keep="first")
        self._best_pick_indices = {
            (str(row["_event_id"]), str(row["_station_id"])): int(index)
            for index, row in best_rows.iterrows()
        }

    def _load_catalog(self) -> None:
        columns = self.configuration.catalog_columns
        catalog = pd.read_csv(
            self.configuration.input_paths.catalog_file,
            dtype={columns.event_id: str},
            low_memory=False,
        )
        missing_columns = set(columns.required_columns()) - set(catalog.columns)
        if missing_columns:
            raise ValueError(
                "Catalog is missing required columns: "
                + ", ".join(sorted(missing_columns))
            )

        catalog["_event_id"] = catalog[columns.event_id].map(
            normalized_identifier
        )
        catalog["_origin_time"] = pd.to_datetime(
            catalog[columns.origin_time],
            utc=True,
            errors="coerce",
            format="mixed",
        )
        catalog = catalog.drop_duplicates("_event_id", keep="last")
        self.catalog = catalog.set_index("_event_id", drop=False)

    def _load_clusters(self) -> None:
        with self.configuration.input_paths.cluster_file.open(
            encoding="utf-8"
        ) as file:
            raw_clusters = json.load(file)
        if not isinstance(raw_clusters, dict):
            raise ValueError(
                "Cluster JSON must map each cluster ID to a list of event IDs"
            )

        cluster_to_events: dict[str, list[str]] = {}
        event_to_cluster: dict[str, str] = {}
        for raw_cluster_id, raw_event_ids in raw_clusters.items():
            if not isinstance(raw_event_ids, list):
                raise ValueError(
                    f"Cluster {raw_cluster_id!r} does not contain an event list"
                )
            cluster_id = clean_cluster_id(raw_cluster_id)
            event_ids = [
                normalized_identifier(event_id)
                for event_id in raw_event_ids
            ]
            cluster_to_events[cluster_id] = event_ids
            for event_id in event_ids:
                event_to_cluster[event_id] = cluster_id

        self.cluster_to_events = cluster_to_events
        self.event_to_cluster = event_to_cluster

    def _load_station_table(self) -> None:
        station_file = self.configuration.input_paths.station_file
        if station_file is None:
            self.stations = pd.DataFrame()
            self._station_lookup = {}
            return
        if not station_file.exists():
            raise FileNotFoundError(station_file)
        stations = pd.read_csv(station_file, low_memory=False)
        columns = self.configuration.station_columns
        required_columns = {
            columns.network,
            columns.station,
            columns.latitude,
            columns.longitude,
        }
        missing_columns = required_columns - set(stations.columns)
        if missing_columns:
            raise ValueError(
                "Station table is missing required columns: "
                + ", ".join(sorted(missing_columns))
            )

        # Match waveform IDs to the inventory by NET.STA. Location and channel
        # codes do not identify a different geographic station.
        stations["_network"] = (
            stations[columns.network].fillna("").astype(str).str.strip()
        )
        stations["_station"] = (
            stations[columns.station].fillna("").astype(str).str.strip()
        )
        stations = stations.drop_duplicates(
            ["_network", "_station"], keep="first"
        )
        self.stations = stations
        self._station_lookup = {
            (str(row["_network"]), str(row["_station"])): row
            for _, row in stations.iterrows()
        }

    def selected_cluster_ids(self) -> list[str]:
        cluster_ids = sorted(self.cluster_to_events, key=natural_sort_key)
        requested = self.configuration.run_settings.selected_cluster_ids
        if requested:
            requested_set = set(requested)
            cluster_ids = [
                cluster_id for cluster_id in cluster_ids
                if cluster_id in requested_set
            ]
        maximum_clusters = self.configuration.run_settings.maximum_clusters
        if maximum_clusters is not None:
            cluster_ids = cluster_ids[: int(maximum_clusters)]
        return cluster_ids

    def events_in_cluster(self, cluster_id: str) -> list[str]:
        return list(self.cluster_to_events.get(str(cluster_id), []))

    def picks_for_cluster(self, cluster_id: str) -> pd.DataFrame:
        event_ids = set(self.events_in_cluster(cluster_id))
        return self.picks[self.picks["_event_id"].isin(event_ids)].copy()

    def station_groups_for_cluster(
        self, cluster_id: str
    ) -> Iterator[tuple[str, pd.DataFrame]]:
        cluster_picks = self.picks_for_cluster(cluster_id)
        valid_station_rows = cluster_picks[
            cluster_picks["_station_id"].ne("")
        ]
        for station_id, station_picks in valid_station_rows.groupby(
            "_station_id", sort=True
        ):
            yield str(station_id), station_picks.copy()

    def origin_time(self, event_id: object) -> pd.Timestamp | None:
        key = normalized_identifier(event_id)
        if key not in self.catalog.index:
            return None
        origin_time = self.catalog.at[key, "_origin_time"]
        if pd.isna(origin_time):
            return None
        return origin_time

    @staticmethod
    def station_key(station_id: object) -> tuple[str, str]:
        """Return the NET.STA inventory key from a waveform station ID."""
        parts = str(station_id).strip().split(".")
        if len(parts) >= 2:
            return parts[0], parts[1]
        return "", str(station_id).strip()

    def station_is_catalogued(self, station_id: object) -> bool:
        """Accept all stations only when no station catalog was supplied."""
        if self.configuration.input_paths.station_file is None:
            return True
        return self.station_key(station_id) in self._station_lookup

    def station_distance_km(
        self, event_id: object, station_id: object
    ) -> float | None:
        """Calculate epicentral distance using catalog coordinates."""
        event_key = normalized_identifier(event_id)
        station_row = self._station_lookup.get(
            self.station_key(station_id)
        )
        if event_key not in self.catalog.index or station_row is None:
            return None

        catalog_columns = self.configuration.catalog_columns
        station_columns = self.configuration.station_columns
        if (
            not catalog_columns.latitude
            or not catalog_columns.longitude
        ):
            return None

        event_row = self.catalog.loc[event_key]
        coordinate_values = [
            event_row.get(catalog_columns.latitude),
            event_row.get(catalog_columns.longitude),
            station_row.get(station_columns.latitude),
            station_row.get(station_columns.longitude),
        ]
        coordinates = pd.to_numeric(
            pd.Series(coordinate_values), errors="coerce"
        )
        if coordinates.isna().any():
            return None

        event_latitude, event_longitude, station_latitude, station_longitude = (
            coordinates.astype(float).tolist()
        )
        distance_metres, _, _ = gps2dist_azimuth(
            event_latitude,
            event_longitude,
            station_latitude,
            station_longitude,
        )
        return float(distance_metres) / 1000.0

    def event_depth_km(self, event_id: object) -> float | None:
        """Return a non-negative source depth for travel-time prediction."""
        depth_column = self.configuration.catalog_columns.depth
        event_key = normalized_identifier(event_id)
        if not depth_column or event_key not in self.catalog.index:
            return None
        depth = pd.to_numeric(
            pd.Series([self.catalog.at[event_key, depth_column]]),
            errors="coerce",
        ).iloc[0]
        if pd.isna(depth):
            return None
        return max(0.0, float(depth))

    def best_pick_row(
        self, event_id: object, station_id: object
    ) -> pd.Series | None:
        event_key = normalized_identifier(event_id)
        station_key = str(station_id).strip()
        row_index = self._best_pick_indices.get((event_key, station_key))
        if row_index is None:
            return None
        return self.picks.loc[row_index]

    def validate(self, waveform_event_ids: set[str] | None = None) -> ValidationReport:
        errors: list[str] = []
        warnings: list[str] = []

        duplicate_pick_rows = int(
            self.picks.duplicated(
                ["_event_id", "_station_id"], keep=False
            ).sum()
        )
        if duplicate_pick_rows:
            warnings.append(
                f"{duplicate_pick_rows} pick rows share an event/station key; "
                "the strongest complete row will be used for calculations"
            )

        optional_mappings = {
            "P score": self.configuration.pick_columns.p_score,
            "S score": self.configuration.pick_columns.s_score,
            "SNR": self.configuration.pick_columns.snr,
            "waveform filename": (
                self.configuration.pick_columns.waveform_filename
            ),
        }
        for meaning, column_name in optional_mappings.items():
            if column_name and column_name not in self.raw_picks.columns:
                warnings.append(
                    f"Optional {meaning} column is mapped to "
                    f"{column_name!r}, but that column is absent"
                )

        missing_event_ids = int(self.picks["_event_id"].eq("").sum())
        missing_station_ids = int(self.picks["_station_id"].eq("").sum())
        invalid_catalog_origins = int(self.catalog["_origin_time"].isna().sum())
        if missing_event_ids:
            errors.append(f"{missing_event_ids} pick rows have no event ID")
        if missing_station_ids:
            errors.append(f"{missing_station_ids} pick rows have no station ID")
        if invalid_catalog_origins:
            errors.append(
                f"{invalid_catalog_origins} catalog events have invalid origin times"
            )

        pick_event_ids = set(self.picks["_event_id"])
        catalog_event_ids = set(self.catalog.index)
        clustered_event_ids = set(self.event_to_cluster)
        picks_without_catalog = len(pick_event_ids - catalog_event_ids)
        picks_without_cluster = len(pick_event_ids - clustered_event_ids)
        if picks_without_catalog:
            warnings.append(
                f"{picks_without_catalog} picked events are absent from the catalog"
            )
        if picks_without_cluster:
            warnings.append(
                f"{picks_without_cluster} picked events are absent from clusters"
            )

        statistics: dict[str, int | float | str] = {
            "pick rows": len(self.picks),
            "P picks": int(self.picks["_p_pick_time"].notna().sum()),
            "S picks": int(self.picks["_s_pick_time"].notna().sum()),
            "catalog events": len(self.catalog),
            "clustered events": len(clustered_event_ids),
            "clusters selected": len(self.selected_cluster_ids()),
        }
        if waveform_event_ids is not None:
            usable_event_ids = (
                catalog_event_ids & clustered_event_ids & waveform_event_ids
            )
            statistics["waveform events"] = len(waveform_event_ids)
            statistics["usable events"] = len(usable_event_ids)
            missing_waveforms = len(
                (catalog_event_ids & clustered_event_ids) - waveform_event_ids
            )
            if missing_waveforms:
                warnings.append(
                    f"{missing_waveforms} catalog/cluster events have no waveform file"
                )

        return ValidationReport(
            errors=errors,
            warnings=warnings,
            statistics=statistics,
        )
