"""Small command-line entry point without an argument-parser framework."""

from pathlib import Path
import sys


HELP_TEXT = """\
PhasePick Refiner

Usage:
  python run_phasepick_refiner.py COMMAND [CONFIG_FILE]

Commands:
  columns          Show the configured required and optional column names
  validate         Check paths, columns, IDs, and waveform coverage
  select-masters   Choose one P/S master per station-cluster pair
  review-masters   Open the optional master-only PyQt review window
  refine           Run paired P/S CC using the saved master table
  report           Rebuild reports from existing refined output
  all              Validate, select masters, refine picks, and make reports
  help             Show this message

CONFIG_FILE defaults to config.json in the current directory.
Paths and scientific settings are edited in that configuration file.
"""


def main(arguments: list[str] | None = None) -> int:
    command_line = list(sys.argv[1:] if arguments is None else arguments)
    if not command_line or command_line[0] in {"help", "-h", "--help"}:
        print(HELP_TEXT)
        return 0

    command = command_line[0].lower()
    configuration_file = Path(
        command_line[1] if len(command_line) > 1 else "config.json"
    )
    if command not in {
        "columns",
        "validate",
        "select-masters",
        "review-masters",
        "refine",
        "report",
        "all",
    }:
        print(f"Unknown command: {command}\n")
        print(HELP_TEXT)
        return 2
    if not configuration_file.exists():
        print(f"Configuration file not found: {configuration_file}")
        print("Start by copying config.example.json to config.json and edit it.")
        return 2

    try:
        # Scientific modules are imported only after help/config checks. This
        # keeps the lightweight commands fast and avoids loading ObsPy early.
        from .configuration import ProjectConfiguration
        from .pipeline import PhasePickRefinementProject

        configuration = ProjectConfiguration.from_file(configuration_file)
        if command == "columns":
            print(configuration.semantic_column_report())
            return 0

        project = PhasePickRefinementProject(configuration)
        if command == "validate":
            report = project.validate()
            print(report.format())
            return 0 if report.is_valid else 1
        if command == "select-masters":
            project.select_masters()
        elif command == "review-masters":
            project.review_masters()
        elif command == "refine":
            project.refine()
        elif command == "report":
            project.generate_report()
        elif command == "all":
            project.run_all()
        return 0
    except Exception as error:
        print(f"ERROR: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
