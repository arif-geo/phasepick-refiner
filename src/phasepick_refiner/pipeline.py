"""High-level project object that coordinates the smaller scientific objects."""

from pathlib import Path

import pandas as pd

from .configuration import ProjectConfiguration
from .correlation import CrossCorrelationRefiner
from .data import PickDataset
from .masters import MasterSelector
from .models import RefinementResult, ValidationReport
from .output import PickOutputWriter
from .waveforms import WaveformArchive


class PhasePickRefinementProject:
    """One complete phase-pick refinement project.

    This object owns the loaded dataset and waveform archive. Its methods read
    like the actual workflow: validate, select masters, refine, and report.
    """

    def __init__(self, configuration: ProjectConfiguration):
        self.configuration = configuration
        self.dataset = PickDataset(configuration)
        self.waveform_archive = WaveformArchive(configuration)
        self._inputs_loaded = False

    @classmethod
    def from_configuration_file(
        cls, configuration_file: str | Path
    ) -> "PhasePickRefinementProject":
        configuration = ProjectConfiguration.from_file(configuration_file)
        return cls(configuration)

    def load_inputs(self) -> "PhasePickRefinementProject":
        if not self._inputs_loaded:
            self.dataset.load()
            self.waveform_archive.build_index()
            self._inputs_loaded = True
        return self

    def validate(self) -> ValidationReport:
        self.load_inputs()
        return self.dataset.validate(self.waveform_archive.event_ids)

    def select_masters(self) -> pd.DataFrame:
        self.load_inputs()
        selector = MasterSelector(self.configuration, self.dataset)
        master_table = selector.select_all()
        self._print_master_summary(master_table)
        return master_table

    def refine(
        self, master_table: pd.DataFrame | None = None
    ) -> tuple[RefinementResult, pd.DataFrame, pd.DataFrame]:
        self.load_inputs()
        selector = MasterSelector(self.configuration, self.dataset)
        if master_table is None:
            if self.configuration.output_settings.master_file.exists():
                master_table = selector.load()
            else:
                master_table = selector.select_all()

        refiner = CrossCorrelationRefiner(
            self.configuration,
            self.dataset,
            self.waveform_archive,
        )
        result = refiner.refine(master_table)
        writer = PickOutputWriter(self.configuration, self.dataset)
        refined_picks, provenance = writer.write(result, master_table)
        self._print_refinement_summary(result, provenance)
        return result, refined_picks, provenance

    def generate_report(
        self,
        refined_picks: pd.DataFrame | None = None,
        provenance: pd.DataFrame | None = None,
    ) -> dict[str, pd.DataFrame]:
        self.load_inputs()
        # Matplotlib is imported only for reporting, so commands such as
        # "help" and "validate" start quickly and do not create font caches.
        from .reports import ReportGenerator

        generator = ReportGenerator(self.configuration, self.dataset)
        report_tables = generator.generate(refined_picks, provenance)
        print(
            "Report written to "
            f"{self.configuration.output_settings.report_directory}"
        )
        return report_tables

    def run_all(self) -> None:
        validation = self.validate()
        print(validation.format())
        if not validation.is_valid:
            raise RuntimeError("Input validation failed")

        master_table = self.select_masters()
        _, refined_picks, provenance = self.refine(master_table)
        self.generate_report(refined_picks, provenance)

    def review_masters(self) -> None:
        """Open the optional master-only review window."""
        self.load_inputs()
        selector = MasterSelector(self.configuration, self.dataset)
        # Expanding a test configuration to all clusters should not require
        # deleting the master table or losing already reviewed manual picks.
        selector.ensure_configured_clusters()

        from .review_gui import open_master_review_window

        open_master_review_window(
            self.configuration,
            self.dataset,
            self.waveform_archive,
        )

    @staticmethod
    def _print_master_summary(master_table: pd.DataFrame) -> None:
        selected = master_table[
            master_table["master_event_id"].fillna("").astype(str).ne("")
        ]
        single_pair_count = int(
            selected["selection_status"]
            .eq("selected high-confidence single pair")
            .sum()
        )
        print("Master selection")
        print(f"  station-cluster pairs considered: {len(master_table)}")
        print(f"  masters selected:                 {len(selected)}")
        print(f"  high-confidence single masters:  {single_pair_count}")
        print(
            "  skipped pairs:                    "
            f"{len(master_table) - len(selected)}"
        )

    @staticmethod
    def _print_refinement_summary(
        result: RefinementResult,
        provenance: pd.DataFrame,
    ) -> None:
        attempts = result.attempts
        accepted_event_pairs = 0
        if not attempts.empty and "status" in attempts:
            accepted_event_pairs = int(attempts["status"].eq("accepted").sum())
        source_counts = provenance["source"].value_counts()
        print("Cross-correlation refinement")
        print(f"  accepted event/station P/S pairs: {accepted_event_pairs}")
        print(
            "  automatic CC phase picks:          "
            f"{int(source_counts.get('CC', 0))}"
        )
        print(
            "  manually reviewed master picks:    "
            f"{int(source_counts.get('C', 0))}"
        )
        print(
            "  original ML phase picks retained:  "
            f"{int(source_counts.get('O', 0))}"
        )
