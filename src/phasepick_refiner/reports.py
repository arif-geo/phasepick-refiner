"""Tables, figures, and a plain-language refinement summary."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .configuration import ProjectConfiguration
from .data import PickDataset, normalized_identifier, short_station_name


class ReportGenerator:
    """Compare the original and refined pick tables."""

    colors = {
        "original": "#5B6770",
        "refined": "#187795",
        "O": "#5B6770",
        "CC": "#E28F2D",
        "C": "#4E9F6F",
    }

    def __init__(
        self,
        configuration: ProjectConfiguration,
        dataset: PickDataset,
    ):
        self.configuration = configuration
        self.dataset = dataset

    def generate(
        self,
        refined_picks: pd.DataFrame | None = None,
        provenance: pd.DataFrame | None = None,
    ) -> dict[str, pd.DataFrame]:
        output_settings = self.configuration.output_settings
        if refined_picks is None:
            refined_picks = pd.read_csv(
                output_settings.refined_pick_file,
                dtype={
                    self.configuration.pick_columns.event_id: str,
                    self.configuration.pick_columns.station_id: str,
                },
                low_memory=False,
            )
        if provenance is None:
            provenance = pd.read_csv(
                output_settings.provenance_file,
                dtype={
                    "event_id": str,
                    "station_id": str,
                    "phase": str,
                    "source": str,
                },
                low_memory=False,
            )

        report_directory = output_settings.report_directory
        report_directory.mkdir(parents=True, exist_ok=True)

        comparison = self._build_comparison(refined_picks, provenance)
        event_summary = self._event_summary(comparison)
        station_summary = self._station_summary(comparison)
        cluster_summary = self._cluster_summary(event_summary)

        comparison.to_csv(
            report_directory / "phase_comparison.csv", index=False
        )
        event_summary.to_csv(
            report_directory / "event_pick_counts.csv", index=False
        )
        station_summary.to_csv(
            report_directory / "station_pick_counts.csv", index=False
        )
        cluster_summary.to_csv(
            report_directory / "cluster_pick_counts.csv", index=False
        )

        self._overall_figure(comparison, report_directory)
        self._event_figure(event_summary, report_directory)
        self._station_figure(station_summary, report_directory)
        self._time_shift_figure(comparison, report_directory)
        self._cluster_figure(cluster_summary, report_directory)
        self._write_summary(
            comparison,
            event_summary,
            station_summary,
            report_directory,
        )
        return {
            "comparison": comparison,
            "events": event_summary,
            "stations": station_summary,
            "clusters": cluster_summary,
        }

    def _phase_records(
        self, pick_table: pd.DataFrame, time_name: str
    ) -> pd.DataFrame:
        columns = self.configuration.pick_columns
        working = pick_table.copy()
        working["event_id"] = working[columns.event_id].map(
            normalized_identifier
        )
        working["station_id"] = (
            working[columns.station_id].fillna("").astype(str).str.strip()
        )

        p_records = working[
            ["event_id", "station_id", columns.p_pick_time]
        ].rename(columns={columns.p_pick_time: time_name})
        p_records["phase"] = "P"
        s_records = working[
            ["event_id", "station_id", columns.s_pick_time]
        ].rename(columns={columns.s_pick_time: time_name})
        s_records["phase"] = "S"

        records = pd.concat([p_records, s_records], ignore_index=True)
        records[time_name] = pd.to_datetime(
            records[time_name],
            utc=True,
            errors="coerce",
            format="mixed",
        )
        records = records.sort_values(time_name).drop_duplicates(
            ["event_id", "station_id", "phase"], keep="last"
        )
        return records

    def _build_comparison(
        self,
        refined_picks: pd.DataFrame,
        provenance: pd.DataFrame,
    ) -> pd.DataFrame:
        original = self._phase_records(
            self.dataset.raw_picks, "original_time"
        )
        refined = self._phase_records(refined_picks, "refined_time")
        keys = ["event_id", "station_id", "phase"]
        comparison = original.merge(refined, how="outer", on=keys)

        source_columns = [
            "source",
            "cc",
            "shift_seconds",
            "cc_component",
            "master_event_id",
            "cc_threshold",
            "cluster_id",
        ]
        available_columns = keys + [
            column for column in source_columns if column in provenance
        ]
        source_table = provenance[available_columns].copy()
        source_table["event_id"] = source_table["event_id"].map(
            normalized_identifier
        )
        source_table["station_id"] = (
            source_table["station_id"].fillna("").astype(str).str.strip()
        )
        source_table["phase"] = (
            source_table["phase"].fillna("").astype(str).str.upper()
        )
        source_table = source_table.drop_duplicates(keys, keep="last")
        comparison = comparison.merge(source_table, how="left", on=keys)

        comparison["station"] = comparison["station_id"].map(
            short_station_name
        )
        comparison["original_pick"] = comparison["original_time"].notna()
        comparison["refined_pick"] = comparison["refined_time"].notna()
        comparison["source"] = comparison["source"].fillna("")
        comparison.loc[
            comparison["refined_pick"] & comparison["source"].eq(""),
            "source",
        ] = "O"
        comparison["time_shift_seconds"] = (
            comparison["refined_time"] - comparison["original_time"]
        ).dt.total_seconds()
        comparison["change"] = np.select(
            [
                comparison["refined_pick"] & ~comparison["original_pick"],
                comparison["refined_pick"] & comparison["original_pick"],
                ~comparison["refined_pick"] & comparison["original_pick"],
            ],
            ["new", "retained", "lost"],
            default="no pick",
        )
        return comparison.sort_values(keys).reset_index(drop=True)

    def _event_summary(self, comparison: pd.DataFrame) -> pd.DataFrame:
        all_event_ids = sorted(
            set(self.dataset.catalog.index) | set(comparison["event_id"])
        )
        summary = pd.DataFrame({"event_id": all_event_ids})
        for label, indicator in [
            ("original", "original_pick"),
            ("refined", "refined_pick"),
        ]:
            picked = comparison[comparison[indicator]]
            counts = (
                picked.groupby(["event_id", "phase"])
                .size()
                .unstack(fill_value=0)
            )
            summary[f"{label}_p_picks"] = (
                summary["event_id"]
                .map(counts.get("P", pd.Series(dtype=int)))
                .fillna(0)
                .astype(int)
            )
            summary[f"{label}_s_picks"] = (
                summary["event_id"]
                .map(counts.get("S", pd.Series(dtype=int)))
                .fillna(0)
                .astype(int)
            )
            summary[f"{label}_phase_picks"] = (
                summary[f"{label}_p_picks"]
                + summary[f"{label}_s_picks"]
            )

            station_phase_counts = picked.groupby(
                ["event_id", "station_id"]
            )["phase"].nunique()
            station_counts = station_phase_counts.groupby("event_id").size()
            pair_counts = (
                station_phase_counts.eq(2).groupby("event_id").sum()
            )
            summary[f"{label}_stations"] = (
                summary["event_id"].map(station_counts).fillna(0).astype(int)
            )
            summary[f"{label}_ps_pairs"] = (
                summary["event_id"].map(pair_counts).fillna(0).astype(int)
            )

        for metric in [
            "p_picks",
            "s_picks",
            "phase_picks",
            "stations",
            "ps_pairs",
        ]:
            summary[f"delta_{metric}"] = (
                summary[f"refined_{metric}"]
                - summary[f"original_{metric}"]
            )
        summary["cluster_id"] = summary["event_id"].map(
            self.dataset.event_to_cluster
        )
        return summary

    @staticmethod
    def _station_summary(comparison: pd.DataFrame) -> pd.DataFrame:
        stations = sorted(comparison["station"].dropna().unique())
        summary = pd.DataFrame({"station": stations})
        for label, indicator in [
            ("original", "original_pick"),
            ("refined", "refined_pick"),
        ]:
            picked = comparison[comparison[indicator]]
            counts = (
                picked.groupby(["station", "phase"])
                .size()
                .unstack(fill_value=0)
            )
            for phase, lower_phase in [("P", "p"), ("S", "s")]:
                summary[f"{label}_{lower_phase}_picks"] = (
                    summary["station"]
                    .map(counts.get(phase, pd.Series(dtype=int)))
                    .fillna(0)
                    .astype(int)
                )
            summary[f"{label}_phase_picks"] = (
                summary[f"{label}_p_picks"]
                + summary[f"{label}_s_picks"]
            )
        for metric in ["p_picks", "s_picks", "phase_picks"]:
            summary[f"delta_{metric}"] = (
                summary[f"refined_{metric}"]
                - summary[f"original_{metric}"]
            )
        return summary.sort_values(
            ["delta_phase_picks", "station"], ascending=[False, True]
        ).reset_index(drop=True)

    @staticmethod
    def _cluster_summary(event_summary: pd.DataFrame) -> pd.DataFrame:
        clustered = event_summary.dropna(subset=["cluster_id"])
        if clustered.empty:
            return pd.DataFrame(
                columns=[
                    "cluster_id",
                    "original_phase_picks",
                    "refined_phase_picks",
                    "delta_phase_picks",
                ]
            )
        value_columns = [
            "original_phase_picks",
            "refined_phase_picks",
            "original_ps_pairs",
            "refined_ps_pairs",
        ]
        summary = clustered.groupby(
            "cluster_id", as_index=False
        )[value_columns].sum()
        summary["delta_phase_picks"] = (
            summary["refined_phase_picks"]
            - summary["original_phase_picks"]
        )
        summary["delta_ps_pairs"] = (
            summary["refined_ps_pairs"] - summary["original_ps_pairs"]
        )
        return summary.sort_values(
            "delta_phase_picks", ascending=False
        ).reset_index(drop=True)

    def _overall_figure(
        self, comparison: pd.DataFrame, report_directory: object
    ) -> None:
        phases = ["P", "S"]
        x_positions = np.arange(2)
        original_counts = [
            int(
                (
                    comparison["original_pick"]
                    & comparison["phase"].eq(phase)
                ).sum()
            )
            for phase in phases
        ]
        refined_counts = [
            int(
                (
                    comparison["refined_pick"]
                    & comparison["phase"].eq(phase)
                ).sum()
            )
            for phase in phases
        ]

        figure, axes = plt.subplots(
            1, 2, figsize=(11, 4.2), constrained_layout=True
        )
        axes[0].bar(
            x_positions - 0.18,
            original_counts,
            width=0.36,
            color=self.colors["original"],
            label="Original ML",
        )
        axes[0].bar(
            x_positions + 0.18,
            refined_counts,
            width=0.36,
            color=self.colors["refined"],
            label="Refined",
        )
        axes[0].set_xticks(x_positions, phases)
        axes[0].set_ylabel("Phase picks")
        axes[0].set_title("Total phase picks")
        axes[0].legend(frameon=False)

        source_counts = (
            comparison[comparison["refined_pick"]]
            .groupby(["phase", "source"])
            .size()
            .unstack(fill_value=0)
        )
        bottom = np.zeros(2)
        for source in ["O", "CC", "C"]:
            if source not in source_counts:
                continue
            values = np.array(
                [
                    source_counts.at[phase, source]
                    if phase in source_counts.index
                    else 0
                    for phase in phases
                ]
            )
            axes[1].bar(
                x_positions,
                values,
                bottom=bottom,
                color=self.colors[source],
                label=source,
            )
            bottom += values
        axes[1].set_xticks(x_positions, phases)
        axes[1].set_ylabel("Refined phase picks")
        axes[1].set_title("Final-pick provenance")
        axes[1].legend(frameon=False)
        figure.savefig(
            report_directory / "01_overall_pick_counts.png", dpi=180
        )
        plt.close(figure)

    def _event_figure(
        self, event_summary: pd.DataFrame, report_directory: object
    ) -> None:
        figure, axes = plt.subplots(
            1, 2, figsize=(11, 4.2), constrained_layout=True
        )
        maximum_count = int(
            max(
                event_summary["original_phase_picks"].max(),
                event_summary["refined_phase_picks"].max(),
            )
        )
        bins = np.arange(-0.5, maximum_count + 1.5, 1)
        axes[0].hist(
            event_summary["original_phase_picks"],
            bins=bins,
            alpha=0.65,
            color=self.colors["original"],
            label="Original ML",
        )
        axes[0].hist(
            event_summary["refined_phase_picks"],
            bins=bins,
            histtype="step",
            linewidth=2,
            color=self.colors["refined"],
            label="Refined",
        )
        axes[0].set_xlabel("P + S picks per event")
        axes[0].set_ylabel("Events")
        axes[0].set_title("Per-event pick distribution")
        axes[0].legend(frameon=False)

        delta_counts = (
            event_summary["delta_phase_picks"]
            .value_counts()
            .sort_index()
        )
        axes[1].bar(
            delta_counts.index,
            delta_counts.values,
            color=self.colors["C"],
        )
        axes[1].axvline(0, color="black", linewidth=0.8)
        axes[1].set_xlabel("Refined minus original picks per event")
        axes[1].set_ylabel("Events")
        axes[1].set_title("Per-event pick gain")
        figure.savefig(
            report_directory / "02_event_pick_gains.png", dpi=180
        )
        plt.close(figure)

    def _station_figure(
        self, station_summary: pd.DataFrame, report_directory: object
    ) -> None:
        station_plot = station_summary.sort_values("station")
        y_positions = np.arange(len(station_plot))
        figure, axes = plt.subplots(
            1,
            2,
            figsize=(11, max(4.5, len(station_plot) * 0.30)),
            sharey=True,
            constrained_layout=True,
        )
        for axis, phase, phase_label in zip(
            axes, ["p", "s"], ["P", "S"]
        ):
            axis.barh(
                y_positions - 0.18,
                station_plot[f"original_{phase}_picks"],
                height=0.36,
                color=self.colors["original"],
                label="Original ML",
            )
            axis.barh(
                y_positions + 0.18,
                station_plot[f"refined_{phase}_picks"],
                height=0.36,
                color=self.colors["refined"],
                label="Refined",
            )
            axis.set_xlabel(f"{phase_label} picks")
            axis.set_title(f"Station-level {phase_label} picks")
        axes[0].set_yticks(y_positions, station_plot["station"])
        axes[0].legend(frameon=False)
        figure.savefig(
            report_directory / "03_station_pick_gains.png", dpi=180
        )
        plt.close(figure)

    def _time_shift_figure(
        self, comparison: pd.DataFrame, report_directory: object
    ) -> None:
        figure, axes = plt.subplots(
            1, 2, figsize=(11, 4.2), sharey=True, constrained_layout=True
        )
        for axis, phase in zip(axes, ["P", "S"]):
            selected = comparison[
                comparison["phase"].eq(phase)
                & comparison["original_pick"]
                & comparison["refined_pick"]
            ]
            changed_shifts = selected.loc[
                selected["source"].isin(["CC", "C"]),
                "time_shift_seconds",
            ].dropna()
            limit = (
                min(
                    5.0,
                    max(
                        0.25,
                        float(np.percentile(np.abs(changed_shifts), 99)),
                    ),
                )
                if len(changed_shifts)
                else 1.0
            )
            for source in ["CC", "C"]:
                shifts = selected.loc[
                    selected["source"].eq(source), "time_shift_seconds"
                ].dropna()
                if len(shifts):
                    axis.hist(
                        shifts,
                        bins=50,
                        range=(-limit, limit),
                        histtype="step",
                        linewidth=1.5,
                        color=self.colors[source],
                        label=f"{source} ({len(shifts)})",
                    )
            axis.axvline(0, color="black", linewidth=0.8)
            axis.set_xlabel("Refined minus original time (s)")
            axis.set_title(f"{phase}-pick time changes")
            axis.legend(frameon=False)
        axes[0].set_ylabel("Phase picks")
        figure.savefig(
            report_directory / "04_pick_time_changes.png", dpi=180
        )
        plt.close(figure)

    def _cluster_figure(
        self, cluster_summary: pd.DataFrame, report_directory: object
    ) -> None:
        positive = cluster_summary[
            cluster_summary["delta_phase_picks"] > 0
        ].head(20)
        if positive.empty:
            return
        positive = positive.iloc[::-1]
        figure, axis = plt.subplots(
            figsize=(8.5, max(4.5, len(positive) * 0.30)),
            constrained_layout=True,
        )
        axis.barh(
            positive["cluster_id"].astype(str),
            positive["delta_phase_picks"],
            color=self.colors["CC"],
        )
        axis.set_xlabel("Refined minus original P + S picks")
        axis.set_ylabel("Cluster")
        axis.set_title("Top clusters by pick gain")
        figure.savefig(
            report_directory / "05_cluster_pick_gains.png", dpi=180
        )
        plt.close(figure)

    @staticmethod
    def _write_summary(
        comparison: pd.DataFrame,
        event_summary: pd.DataFrame,
        station_summary: pd.DataFrame,
        report_directory: object,
    ) -> None:
        original_count = int(comparison["original_pick"].sum())
        refined_count = int(comparison["refined_pick"].sum())
        original_pairs = int(event_summary["original_ps_pairs"].sum())
        refined_pairs = int(event_summary["refined_ps_pairs"].sum())
        percent_gain = (
            100.0 * (refined_count - original_count) / original_count
            if original_count
            else np.nan
        )
        source_counts = (
            comparison[comparison["refined_pick"]]["source"]
            .value_counts()
            .to_dict()
        )
        lines = [
            "Phase-pick refinement summary",
            "",
            f"Original ML phase picks:       {original_count}",
            f"Refined phase picks:           {refined_count}",
            f"Net phase-pick gain:           {refined_count - original_count:+d}",
            f"Relative phase-pick gain:      {percent_gain:.2f}%",
            f"Original complete P/S pairs:   {original_pairs}",
            f"Refined complete P/S pairs:    {refined_pairs}",
            f"Net complete P/S-pair gain:    {refined_pairs - original_pairs:+d}",
            (
                "Events with positive gain:    "
                f"{int((event_summary['delta_phase_picks'] > 0).sum())}"
            ),
            (
                "Stations with positive gain:  "
                f"{int((station_summary['delta_phase_picks'] > 0).sum())}"
            ),
            "",
            "Final source counts:",
            f"  Original ML (O): {int(source_counts.get('O', 0))}",
            f"  Automatic CC:    {int(source_counts.get('CC', 0))}",
            f"  Manual master:   {int(source_counts.get('C', 0))}",
        ]
        (report_directory / "summary.txt").write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )
