"""Matplotlib rendering for one five-station waveform page."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from obspy import Trace
from obspy.geodetics.base import kilometer2degrees
from obspy.taup import TauPyModel

from matplotlib.lines import Line2D

from .configuration import ProjectConfiguration
from .data import PickDataset, short_station_name
from .review_session import MasterReviewSession


@dataclass
class WaveformPageResult:
    """Plot geometry needed to map a mouse click back to a station."""

    station_y_ranges: list[tuple[float, float, str]]


class WaveformPagePlotter:
    """Draw fixed three-component station bands and arrival markers."""

    trace_colors = {
        "Z": "#222222",
        "N": "#187795",
        "E": "#D97824",
        "1": "#187795",
        "2": "#D97824",
    }
    phase_colors = {"P": "#C83E4D", "S": "#187795"}
    station_backgrounds = ("#F3F5F6", "#FFFFFF")

    def __init__(
        self,
        configuration: ProjectConfiguration,
        dataset: PickDataset,
        session: MasterReviewSession,
    ):
        self.configuration = configuration
        self.dataset = dataset
        self.session = session
        # TauP supplies a portable rough estimate when project-specific local
        # travel-time tables are unavailable in this standalone package.
        self.travel_time_model = TauPyModel(
            model=configuration.viewer_settings.taup_model
        )
        self.travel_time_cache: dict[
            tuple[str, str, str], float | None
        ] = {}

    def draw(
        self,
        axis: object,
        low_frequency: float,
        high_frequency: float,
        gain: float,
        x_minimum: float,
        x_maximum: float,
    ) -> WaveformPageResult:
        origin_time = self.dataset.origin_time(self.session.event_id)
        if origin_time is None:
            axis.text(
                0.5,
                0.5,
                "No catalog origin time",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            return WaveformPageResult([])

        page_stations = self.session.stations_on_page()
        total_rows = self.session.stations_per_page * 3
        y_positions: list[float] = []
        y_labels: list[str] = []
        station_y_ranges: list[tuple[float, float, str]] = []

        for station_position, station_id in enumerate(page_stations):
            station_top = total_rows - 1 - station_position * 3
            station_bottom = station_top - 2
            lower_edge = station_bottom - 0.5
            upper_edge = station_top + 0.5
            station_y_ranges.append(
                (lower_edge, upper_edge, station_id)
            )
            self._draw_station_background(
                axis,
                station_position,
                station_id,
                lower_edge,
                upper_edge,
            )

            components = self.session.station_groups.get(station_id, {})
            component_slots = self.component_slots(components)
            master_symbol = (
                "*" if self.session.is_local_master(station_id) else ""
            )
            station_name = short_station_name(station_id)
            for component_offset, component in enumerate(component_slots):
                y_position = station_top - component_offset
                y_positions.append(y_position)
                y_labels.append(
                    f"{station_name}{master_symbol} {component}"
                )
                trace = components.get(component)
                if trace is None:
                    axis.text(
                        x_minimum + 0.01 * (x_maximum - x_minimum),
                        y_position,
                        "channel unavailable",
                        color="#9299A1",
                        fontsize=8,
                        va="center",
                    )
                    continue
                self._draw_trace(
                    axis,
                    trace,
                    component,
                    origin_time,
                    y_position,
                    low_frequency,
                    high_frequency,
                    gain,
                )

            self._draw_station_arrivals(
                axis,
                station_id,
                origin_time,
                lower_edge,
                upper_edge,
            )

        if not page_stations:
            axis.text(
                0.5,
                0.5,
                "No usable station waveforms in this event file",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )

        axis.set_xlim(x_minimum, x_maximum)
        axis.set_ylim(-0.5, total_rows - 0.5)
        axis.set_yticks(y_positions, y_labels)
        axis.set_xlabel("Seconds from catalog origin time")
        axis.set_ylabel("Station and component")
        axis.grid(axis="x", alpha=0.16, zorder=0)
        axis.set_title(
            f"Cluster {self.session.cluster_id} | "
            f"master event {self.session.event_id} | "
            f"station page {self.session.current_page + 1}/"
            f"{self.session.page_count()}",
            loc="left",
        )
        axis.legend(
            handles=self.legend_handles(),
            loc="lower right",
            bbox_to_anchor=(1.0, 1.005),
            frameon=True,
            framealpha=0.78,
            facecolor="white",
            edgecolor="#A8AFB5",
            ncol=5,
            fontsize=8,
        )
        return WaveformPageResult(station_y_ranges)

    def _draw_station_background(
        self,
        axis: object,
        station_position: int,
        station_id: str,
        lower_edge: float,
        upper_edge: float,
    ) -> None:
        is_selected = (
            station_id == self.session.selected_station_id
        )
        color = (
            "#DCEAF7"
            if is_selected
            else self.station_backgrounds[station_position % 2]
        )
        axis.axhspan(lower_edge, upper_edge, color=color, zorder=0)

    @staticmethod
    def component_slots(
        components: dict[str, Trace]
    ) -> tuple[str, str, str]:
        if "N" in components or "E" in components:
            return "Z", "N", "E"
        if "1" in components or "2" in components:
            return "Z", "1", "2"
        return "Z", "N", "E"

    def _draw_trace(
        self,
        axis: object,
        original_trace: Trace,
        component: str,
        origin_time: pd.Timestamp,
        y_position: float,
        low_frequency: float,
        high_frequency: float,
        gain: float,
    ) -> None:
        trace = self._filtered_trace(
            original_trace, low_frequency, high_frequency
        )
        if trace is None or len(trace.data) == 0:
            return

        relative_seconds = (
            float(trace.stats.starttime.timestamp) - origin_time.timestamp()
            + np.arange(len(trace.data)) * float(trace.stats.delta)
        )
        waveform = np.asarray(trace.data, dtype=float)
        maximum_amplitude = float(np.max(np.abs(waveform)))
        if maximum_amplitude > 0:
            waveform = 0.42 * gain * waveform / maximum_amplitude
            # Clip gain inside its row so strong S energy cannot cover another
            # component or station. This affects display only.
            waveform = np.clip(waveform, -0.48, 0.48)

        axis.plot(
            relative_seconds,
            waveform + y_position,
            color=self.trace_colors.get(component, "#444444"),
            linewidth=0.75,
            zorder=2,
        )

    @staticmethod
    def _filtered_trace(
        original_trace: Trace,
        low_frequency: float,
        high_frequency: float,
    ) -> Trace | None:
        nyquist_frequency = 0.5 / float(original_trace.stats.delta)
        high_frequency = min(
            high_frequency, 0.95 * nyquist_frequency
        )
        if high_frequency <= low_frequency:
            return None
        try:
            trace = original_trace.copy()
            trace.data = np.nan_to_num(
                np.asarray(trace.data, dtype=float)
            )
            trace.detrend("demean")
            trace.taper(max_percentage=0.05, type="cosine")
            trace.filter(
                "bandpass",
                freqmin=low_frequency,
                freqmax=high_frequency,
                corners=4,
                zerophase=False,
            )
            return trace
        except Exception:
            return None

    def _draw_station_arrivals(
        self,
        axis: object,
        station_id: str,
        origin_time: pd.Timestamp,
        lower_y: float,
        upper_y: float,
    ) -> None:
        original_pick = self.session.original_pick(station_id)
        master_row = self.session.master_row(station_id)
        pending = self.session.pending_picks(station_id)

        for phase in ["P", "S"]:
            phase_name = phase.lower()
            original_time = pd.NaT
            if original_pick is not None:
                original_time = original_pick[
                    f"_{phase_name}_pick_time"
                ]

            reviewed_time = pd.NaT
            reviewed = False
            if (
                master_row is not None
                and str(master_row["master_event_id"])
                == self.session.event_id
                and bool(master_row[f"{phase_name}_pick_edited"])
            ):
                reviewed_time = pd.to_datetime(
                    master_row[f"reviewed_{phase_name}_pick_time"],
                    utc=True,
                    errors="coerce",
                )
                reviewed = pd.notna(reviewed_time)

            pending_time = pending.get(phase)
            if pd.notna(original_time):
                self._draw_arrival_marker(
                    axis,
                    origin_time,
                    original_time,
                    phase,
                    "original",
                    lower_y,
                    upper_y,
                )
            if reviewed:
                self._draw_arrival_marker(
                    axis,
                    origin_time,
                    reviewed_time,
                    phase,
                    "reviewed",
                    lower_y,
                    upper_y,
                )
            if pending_time is not None:
                self._draw_arrival_marker(
                    axis,
                    origin_time,
                    pending_time,
                    phase,
                    "reviewed",
                    lower_y,
                    upper_y,
                )

            has_pick = (
                pd.notna(original_time)
                or reviewed
                or pending_time is not None
            )
            if not has_pick:
                travel_time_seconds = (
                    self._theoretical_travel_time_seconds(
                        station_id, phase
                    )
                )
                if travel_time_seconds is not None:
                    theoretical_time = origin_time + pd.Timedelta(
                        seconds=travel_time_seconds
                    )
                    self._draw_arrival_marker(
                        axis,
                        origin_time,
                        theoretical_time,
                        phase,
                        "theoretical",
                        lower_y,
                        upper_y,
                    )

    def _theoretical_travel_time_seconds(
        self, station_id: str, phase: str
    ) -> float | None:
        """Return the earliest requested TauP phase for this event/station."""
        cache_key = (self.session.event_id, station_id, phase)
        if cache_key in self.travel_time_cache:
            return self.travel_time_cache[cache_key]

        distance_km = self.session.station_distance_km(station_id)
        source_depth_km = self.dataset.event_depth_km(
            self.session.event_id
        )
        if distance_km is None or source_depth_km is None:
            self.travel_time_cache[cache_key] = None
            return None

        phase_names = (
            ["p", "P", "Pg", "Pn"]
            if phase == "P"
            else ["s", "S", "Sg", "Sn"]
        )
        arrivals = self.travel_time_model.get_travel_times(
            source_depth_in_km=source_depth_km,
            distance_in_degree=kilometer2degrees(distance_km),
            phase_list=phase_names,
        )
        travel_time = (
            min(float(arrival.time) for arrival in arrivals)
            if arrivals
            else None
        )
        self.travel_time_cache[cache_key] = travel_time
        return travel_time

    def _draw_arrival_marker(
        self,
        axis: object,
        origin_time: pd.Timestamp,
        pick_time: object,
        phase: str,
        source: str,
        lower_y: float,
        upper_y: float,
    ) -> None:
        parsed_time = pd.to_datetime(
            pick_time, utc=True, errors="coerce"
        )
        if pd.isna(parsed_time):
            return
        relative_seconds = (parsed_time - origin_time).total_seconds()
        line_styles = {
            "original": "--",
            "theoretical": ":",
            "reviewed": "-",
        }
        axis.plot(
            [relative_seconds, relative_seconds],
            [lower_y, upper_y],
            color=self.phase_colors[phase],
            linestyle=line_styles[source],
            linewidth=1.35,
            alpha=0.9,
            zorder=4,
        )

    def legend_handles(self) -> list[Line2D]:
        return [
            Line2D([0], [0], color=self.phase_colors["P"], label="P"),
            Line2D([0], [0], color=self.phase_colors["S"], label="S"),
            Line2D(
                [0], [0], color="#333333", linestyle="--", label="ML"
            ),
            Line2D(
                [0],
                [0],
                color="#333333",
                linestyle=":",
                label=(
                    "Theory "
                    f"({self.configuration.viewer_settings.taup_model})"
                ),
            ),
            Line2D(
                [0],
                [0],
                color="#333333",
                linestyle="-",
                label="Reviewed",
            ),
        ]
