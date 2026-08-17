"""Readable configuration objects for one phase-pick refinement project."""

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any


def _resolved_path(value: str | None, configuration_directory: Path) -> Path | None:
    """Resolve relative paths next to the configuration file."""
    if value is None or str(value).strip() == "":
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = configuration_directory / path
    return path.resolve()


def _tuple_of_floats(values: list[float], setting_name: str) -> tuple[float, float]:
    if len(values) != 2:
        raise ValueError(f"{setting_name} must contain exactly two numbers")
    return float(values[0]), float(values[1])


@dataclass(frozen=True)
class InputPaths:
    """Files supplied by the user."""

    pick_file: Path
    catalog_file: Path
    cluster_file: Path
    waveform_directory: Path
    station_file: Path | None = None


@dataclass(frozen=True)
class PickColumnNames:
    """Meanings of columns in the user's pick table."""

    event_id: str
    station_id: str
    p_pick_time: str
    s_pick_time: str
    p_score: str | None = None
    s_score: str | None = None
    snr: str | None = None
    waveform_filename: str | None = None

    def required_columns(self) -> list[str]:
        return [
            self.event_id,
            self.station_id,
            self.p_pick_time,
            self.s_pick_time,
        ]


@dataclass(frozen=True)
class CatalogColumnNames:
    """Meanings of columns in the event catalog."""

    event_id: str
    origin_time: str
    latitude: str | None = None
    longitude: str | None = None
    depth: str | None = None
    magnitude: str | None = None

    def required_columns(self) -> list[str]:
        return [self.event_id, self.origin_time]


@dataclass(frozen=True)
class StationColumnNames:
    """Optional station-inventory column names."""

    network: str = "network"
    station: str = "station"
    latitude: str = "latitude"
    longitude: str = "longitude"
    elevation: str = "elevation"


@dataclass(frozen=True)
class WaveformSettings:
    """How event files and channels are found."""

    file_glob: str = "*.mseed"
    event_id_separator: str = "_"
    stream_cache_size: int = 4


@dataclass(frozen=True)
class MasterSelectionSettings:
    """Rules used to select one P/S master for each station-cluster."""

    minimum_pick_pairs: int = 2
    snr_weight: float = 0.25
    sp_template_tolerance_seconds: float = 0.75
    allow_single_high_confidence_pair: bool = True
    single_pair_minimum_p_score: float = 0.75
    single_pair_minimum_s_score: float = 0.55
    single_pair_minimum_snr: float = 5.0


@dataclass(frozen=True)
class CorrelationSettings:
    """Waveform windows, filtering, and acceptance rules."""

    cc_threshold: float = 0.70
    filter_frequency_hz: tuple[float, float] = (2.0, 10.0)
    p_template_window_seconds: tuple[float, float] = (-1.0, 1.0)
    s_template_window_seconds: tuple[float, float] = (-1.0, 2.0)
    search_half_width_seconds: float = 1.0
    sp_acceptance_tolerance_seconds: float = 0.75
    include_master_in_cc: bool = False


@dataclass(frozen=True)
class ViewerSettings:
    """Initial layout and display values for the master-review window."""

    x_limits_seconds: tuple[float, float] = (0.0, 30.0)
    stations_per_page: int = 5
    default_gain: float = 1.0
    taup_model: str = "iasp91"


@dataclass(frozen=True)
class OutputSettings:
    """Names and schema rules for generated products."""

    directory: Path
    refined_pick_filename: str = "phasepicks_refined.csv"
    provenance_filename: str = "phasepicks_refined_sources.csv"
    master_filename: str = "master_selections.csv"
    attempt_filename: str = "cc_attempts.csv"
    report_directory_name: str = "report"
    output_time_format: str = "%Y-%m-%d %H:%M:%S.%f+00:00"
    event_level_columns_for_new_rows: tuple[str, ...] = ()

    def path_for(self, filename: str) -> Path:
        return self.directory / filename

    @property
    def refined_pick_file(self) -> Path:
        return self.path_for(self.refined_pick_filename)

    @property
    def provenance_file(self) -> Path:
        return self.path_for(self.provenance_filename)

    @property
    def master_file(self) -> Path:
        return self.path_for(self.master_filename)

    @property
    def attempt_file(self) -> Path:
        return self.path_for(self.attempt_filename)

    @property
    def report_directory(self) -> Path:
        return self.directory / self.report_directory_name


@dataclass(frozen=True)
class RunSettings:
    """Optional limits used for testing or selected-cluster runs."""

    selected_cluster_ids: tuple[str, ...] = ()
    maximum_clusters: int | None = None


@dataclass(frozen=True)
class ProjectConfiguration:
    """One object containing every setting needed by the program.

    Passing this object between classes is safer and easier to read than
    passing a long collection of unrelated paths and numbers to every
    function.
    """

    input_paths: InputPaths
    pick_columns: PickColumnNames
    catalog_columns: CatalogColumnNames
    station_columns: StationColumnNames
    waveform_settings: WaveformSettings
    master_settings: MasterSelectionSettings
    correlation_settings: CorrelationSettings
    viewer_settings: ViewerSettings
    output_settings: OutputSettings
    run_settings: RunSettings = field(default_factory=RunSettings)
    configuration_file: Path | None = None

    @classmethod
    def from_file(cls, filename: str | Path) -> "ProjectConfiguration":
        configuration_file = Path(filename).expanduser().resolve()
        raw_configuration = cls._read_mapping(configuration_file)
        configuration_directory = configuration_file.parent

        input_values = dict(raw_configuration.get("inputs", {}))
        column_values = dict(raw_configuration.get("columns", {}))
        pick_column_values = dict(column_values.get("picks", {}))
        catalog_column_values = dict(column_values.get("catalog", {}))
        station_column_values = dict(column_values.get("stations", {}))
        waveform_values = dict(raw_configuration.get("waveforms", {}))
        master_values = dict(raw_configuration.get("master_selection", {}))
        correlation_values = dict(raw_configuration.get("correlation", {}))
        viewer_values = dict(raw_configuration.get("viewer", {}))
        output_values = dict(raw_configuration.get("output", {}))
        run_values = dict(raw_configuration.get("run", {}))

        required_input_names = [
            "pick_file",
            "catalog_file",
            "cluster_file",
            "waveform_directory",
        ]
        missing_input_names = [
            name for name in required_input_names if not input_values.get(name)
        ]
        if missing_input_names:
            raise ValueError(
                "Configuration is missing input paths: "
                + ", ".join(missing_input_names)
            )

        required_pick_names = [
            "event_id",
            "station_id",
            "p_pick_time",
            "s_pick_time",
        ]
        missing_pick_names = [
            name for name in required_pick_names if not pick_column_values.get(name)
        ]
        if missing_pick_names:
            raise ValueError(
                "Configuration is missing pick-column mappings: "
                + ", ".join(missing_pick_names)
            )

        required_catalog_names = ["event_id", "origin_time"]
        missing_catalog_names = [
            name for name in required_catalog_names
            if not catalog_column_values.get(name)
        ]
        if missing_catalog_names:
            raise ValueError(
                "Configuration is missing catalog-column mappings: "
                + ", ".join(missing_catalog_names)
            )

        output_directory = _resolved_path(
            output_values.pop("directory", "output"),
            configuration_directory,
        )
        if output_directory is None:
            raise ValueError("Output directory cannot be empty")

        input_paths = InputPaths(
            pick_file=_resolved_path(
                input_values["pick_file"], configuration_directory
            ),
            catalog_file=_resolved_path(
                input_values["catalog_file"], configuration_directory
            ),
            cluster_file=_resolved_path(
                input_values["cluster_file"], configuration_directory
            ),
            waveform_directory=_resolved_path(
                input_values["waveform_directory"], configuration_directory
            ),
            station_file=_resolved_path(
                input_values.get("station_file"), configuration_directory
            ),
        )

        pick_columns = PickColumnNames(**pick_column_values)
        catalog_columns = CatalogColumnNames(**catalog_column_values)
        station_columns = StationColumnNames(**station_column_values)
        waveform_settings = WaveformSettings(**waveform_values)
        master_settings = MasterSelectionSettings(**master_values)

        filter_frequency_hz = _tuple_of_floats(
            correlation_values.pop("filter_frequency_hz", [2.0, 10.0]),
            "filter_frequency_hz",
        )
        p_template_window_seconds = _tuple_of_floats(
            correlation_values.pop(
                "p_template_window_seconds", [-1.0, 1.0]
            ),
            "p_template_window_seconds",
        )
        s_template_window_seconds = _tuple_of_floats(
            correlation_values.pop(
                "s_template_window_seconds", [-1.0, 2.0]
            ),
            "s_template_window_seconds",
        )
        correlation_settings = CorrelationSettings(
            filter_frequency_hz=filter_frequency_hz,
            p_template_window_seconds=p_template_window_seconds,
            s_template_window_seconds=s_template_window_seconds,
            **correlation_values,
        )

        x_limits_seconds = _tuple_of_floats(
            viewer_values.pop("x_limits_seconds", [0.0, 30.0]),
            "x_limits_seconds",
        )
        viewer_settings = ViewerSettings(
            x_limits_seconds=x_limits_seconds,
            **viewer_values,
        )

        event_level_columns = tuple(
            output_values.pop("event_level_columns_for_new_rows", [])
        )
        output_settings = OutputSettings(
            directory=output_directory,
            event_level_columns_for_new_rows=event_level_columns,
            **output_values,
        )
        run_settings = RunSettings(
            selected_cluster_ids=tuple(
                str(value) for value in run_values.get(
                    "selected_cluster_ids", []
                )
            ),
            maximum_clusters=run_values.get("maximum_clusters"),
        )

        configuration = cls(
            input_paths=input_paths,
            pick_columns=pick_columns,
            catalog_columns=catalog_columns,
            station_columns=station_columns,
            waveform_settings=waveform_settings,
            master_settings=master_settings,
            correlation_settings=correlation_settings,
            viewer_settings=viewer_settings,
            output_settings=output_settings,
            run_settings=run_settings,
            configuration_file=configuration_file,
        )
        configuration.validate_values()
        return configuration

    @staticmethod
    def _read_mapping(filename: Path) -> dict[str, Any]:
        suffix = filename.suffix.lower()
        if suffix == ".json":
            with filename.open(encoding="utf-8") as file:
                return json.load(file)
        if suffix in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as error:
                raise RuntimeError(
                    "YAML configuration requires PyYAML. Install the optional "
                    "dependency or use the supplied JSON configuration."
                ) from error
            with filename.open(encoding="utf-8") as file:
                loaded = yaml.safe_load(file)
            return loaded or {}
        raise ValueError("Configuration must be a .json, .yaml, or .yml file")

    def validate_values(self) -> None:
        low_frequency, high_frequency = (
            self.correlation_settings.filter_frequency_hz
        )
        if low_frequency <= 0 or high_frequency <= low_frequency:
            raise ValueError(
                "Filter frequencies must satisfy 0 < low < high"
            )
        if not 0 <= self.correlation_settings.cc_threshold <= 1:
            raise ValueError("CC threshold must be between 0 and 1")
        if self.master_settings.minimum_pick_pairs < 1:
            raise ValueError("minimum_pick_pairs must be at least 1")
        if self.waveform_settings.stream_cache_size < 0:
            raise ValueError("stream_cache_size cannot be negative")
        view_start, view_end = self.viewer_settings.x_limits_seconds
        if view_end <= view_start:
            raise ValueError(
                "Viewer X limits must satisfy start < end"
            )
        if self.viewer_settings.stations_per_page < 1:
            raise ValueError("stations_per_page must be at least 1")
        if self.viewer_settings.default_gain <= 0:
            raise ValueError("default_gain must be positive")
        if not self.viewer_settings.taup_model.strip():
            raise ValueError("taup_model cannot be empty")

    def semantic_column_report(self) -> str:
        """Return a user-facing list of required and optional columns."""
        pick_columns = self.pick_columns
        catalog_columns = self.catalog_columns
        lines = [
            "Required pick-table columns:",
            f"  event ID:       {pick_columns.event_id}",
            f"  station ID:     {pick_columns.station_id}",
            f"  P arrival time: {pick_columns.p_pick_time}",
            f"  S arrival time: {pick_columns.s_pick_time}",
            "",
            "Optional pick-table columns:",
            f"  P score:        {pick_columns.p_score or '(not mapped)'}",
            f"  S score:        {pick_columns.s_score or '(not mapped)'}",
            f"  SNR:            {pick_columns.snr or '(not mapped)'}",
            (
                "  waveform file:  "
                f"{pick_columns.waveform_filename or '(not mapped)'}"
            ),
            "",
            "Required catalog columns:",
            f"  event ID:       {catalog_columns.event_id}",
            f"  origin time:    {catalog_columns.origin_time}",
        ]
        return "\n".join(lines)
