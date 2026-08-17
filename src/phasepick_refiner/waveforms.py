"""Event-waveform indexing and station/component access."""

from collections import OrderedDict
from pathlib import Path

import numpy as np
from obspy import Stream, Trace, read

from .configuration import ProjectConfiguration
from .data import normalized_identifier


class WaveformArchive:
    """Read eventwise MiniSEED files and expose station components.

    The small in-memory cache matters during correlation because every event is
    visited more than once for its P and S channels. The rest of the program
    does not need to know whether a stream came from disk or from this cache.
    """

    usable_components = ("E", "N", "1", "2", "Z")

    def __init__(self, configuration: ProjectConfiguration):
        self.configuration = configuration
        self.waveform_files: dict[str, Path] = {}
        self._stream_cache: OrderedDict[str, Stream] = OrderedDict()

    def build_index(self) -> "WaveformArchive":
        waveform_directory = self.configuration.input_paths.waveform_directory
        waveform_settings = self.configuration.waveform_settings

        for waveform_file in sorted(
            waveform_directory.glob(waveform_settings.file_glob)
        ):
            event_id = waveform_file.name.split(
                waveform_settings.event_id_separator, 1
            )[0]
            event_id = normalized_identifier(event_id)
            self.waveform_files.setdefault(event_id, waveform_file)
        return self

    @property
    def event_ids(self) -> set[str]:
        return set(self.waveform_files)

    def file_for_event(self, event_id: object) -> Path | None:
        return self.waveform_files.get(normalized_identifier(event_id))

    def stream_for_event(self, event_id: object) -> Stream | None:
        """Return an event stream, reusing recently read MiniSEED files."""
        event_key = normalized_identifier(event_id)
        if event_key in self._stream_cache:
            stream = self._stream_cache.pop(event_key)
            self._stream_cache[event_key] = stream
            return stream

        waveform_file = self.file_for_event(event_key)
        if waveform_file is None:
            return None

        try:
            stream = read(str(waveform_file))
        except Exception:
            return None

        cache_size = self.configuration.waveform_settings.stream_cache_size
        if cache_size > 0:
            self._stream_cache[event_key] = stream
            while len(self._stream_cache) > cache_size:
                self._stream_cache.popitem(last=False)
        return stream

    def station_components(
        self,
        event_id: object,
        station_id: object,
        apply_filter: bool = True,
    ) -> dict[str, Trace]:
        """Return the first usable trace for each component at one station."""
        station_parts = str(station_id).strip().split(".")
        if len(station_parts) != 4:
            return {}
        network, station, location, channel_prefix = station_parts

        waveform_file = self.file_for_event(event_id)
        if waveform_file is None:
            return {}

        # MiniSEED supports source-name selection while reading. This avoids
        # loading every station in an event file when CC needs only one.
        source_name = (
            f"{network}_{station}_{location}_{channel_prefix}*"
        )
        try:
            selected_stream = read(
                str(waveform_file), sourcename=source_name
            )
        except Exception:
            return {}

        components: dict[str, Trace] = {}
        for trace in selected_stream:
            component = str(trace.stats.channel)[-1].upper()
            if component not in self.usable_components:
                continue
            if component in components:
                continue

            prepared_trace = self._prepare_trace(trace) if apply_filter else trace.copy()
            if prepared_trace is not None:
                components[component] = prepared_trace
        return components

    def event_station_components(
        self,
        event_id: object,
        apply_filter: bool = False,
    ) -> dict[str, dict[str, Trace]]:
        """Group every usable event trace by station and component.

        The returned station IDs use the same NET.STA.LOC.PREFIX convention as
        the pick table, for example ``BK.RBOW.00.HH`` or ``NC.KCT..HH``.
        """
        stream = self.stream_for_event(event_id)
        if stream is None:
            return {}

        station_groups: dict[str, dict[str, Trace]] = {}
        for original_trace in stream:
            channel = str(original_trace.stats.channel)
            if len(channel) < 2:
                continue
            component = channel[-1].upper()
            if component not in self.usable_components:
                continue

            station_id = ".".join(
                [
                    str(original_trace.stats.network),
                    str(original_trace.stats.station),
                    str(original_trace.stats.location),
                    channel[:-1],
                ]
            )
            station_components = station_groups.setdefault(station_id, {})
            if component in station_components:
                continue

            trace = (
                self._prepare_trace(original_trace)
                if apply_filter
                else original_trace.copy()
            )
            if trace is not None:
                station_components[component] = trace

        return dict(sorted(station_groups.items()))

    def _prepare_trace(self, original_trace: Trace) -> Trace | None:
        """Copy, clean, demean, and band-pass one trace."""
        low_frequency, high_frequency = (
            self.configuration.correlation_settings.filter_frequency_hz
        )
        nyquist_frequency = 0.5 / float(original_trace.stats.delta)
        usable_high_frequency = min(high_frequency, 0.95 * nyquist_frequency)
        if usable_high_frequency <= low_frequency:
            return None

        try:
            trace = original_trace.copy()
            trace.data = np.nan_to_num(
                np.asarray(trace.data, dtype=float),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            trace.detrend("demean")
            trace.filter(
                "bandpass",
                freqmin=low_frequency,
                freqmax=usable_high_frequency,
                corners=4,
                zerophase=False,
            )
            return trace
        except Exception:
            return None
