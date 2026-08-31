"""Validate and run the park-performance pipeline by analysis area."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from project_config import ConfigError, available_areas, load_config, project_root
from validate_inputs import run_validation


STAGES = [
    ("01", "01_network.py", "shared"),
    ("02", "02_parks.py", "shared"),
    ("03", "03_population.py", "area"),
    ("04", "04_reachability.py", "area"),
    ("05", "05_supply.py", "area"),
    ("06", "06_experience.py", "area"),
    ("07", "07_divergence.py", "area"),
    ("08", "08_equity.py", "area"),
    ("09", "09_statistics.py", "area"),
    ("10", "10_figures.py", "area"),
    ("11", "11_vancouver_extension.py", "vancouver"),
]

STAGE_INPUTS = {
    "01": {"census_gaf", "da_boundaries"},
    "02": {"parks_source"},
    "03": {"census_gaf", "da_boundaries", "db_boundaries"},
    "04": {"entrances"},
    "05": {"parks"},
    "06": {"google_ratings_validated"},
    "08": {"census_profile", "canale"},
}


def selected_areas(value: str) -> list[str]:
    cities = [area for area in available_areas() if area != "metro"]
    return cities if value == "all" else [value]


def print_area_list() -> None:
    print("Available analysis areas:")
    for area in available_areas():
        config = load_config(area)
        label = config["analysis_area"]["name"]
        print(f"  {area:<18} {label}")
    print("  all                All six municipalities (metro excluded)")


def selected_stages(stage_ids: list[str] | None) -> list[tuple[str, str, str]]:
    if not stage_ids:
        return STAGES
    wanted = set(stage_ids)
    return [stage for stage in STAGES if stage[0] in wanted]


def stage_commands(
    area_value: str,
    stage_ids: list[str] | None = None,
    force: bool = False,
) -> list[tuple[str, str, Path, list[str]]]:
    root = project_root()
    commands = []
    areas = selected_areas(area_value)

    for stage, filename, scope in selected_stages(stage_ids):
        script = root / "scripts" / filename
        if scope == "shared":
            targets = ["metro"] if area_value == "all" else areas
        elif scope == "vancouver":
            targets = ["vancouver"] if area_value in {"all", "vancouver"} else []
        else:
            targets = areas

        for area in targets:
            command = [sys.executable, str(script), "--area", area]
            if force:
                command.append("--force")
            commands.append((stage, area, script, command))

    return commands


def print_dry_run(
    area_value: str,
    stage_ids: list[str] | None = None,
    force: bool = False,
) -> None:
    print(f"\nPipeline plan: {area_value}")
    for stage, area, script, command in stage_commands(area_value, stage_ids, force):
        status = "ready" if script.exists() else "not yet migrated"
        print(f"  {stage}  {area:<18} {script.name:<22} {status}")


def required_inputs(stage_ids: list[str] | None) -> set[str] | None:
    if not stage_ids:
        return None

    inputs: set[str] = set()
    for stage in stage_ids:
        inputs.update(STAGE_INPUTS.get(stage, set()))
    # An explicitly selected derived stage may require no additional raw input.
    # Preserve an empty set so validation checks configuration only instead of
    # falling back to every project input.
    return inputs


def run_plan(
    area_value: str,
    stage_ids: list[str] | None = None,
    force: bool = False,
) -> bool:
    commands = stage_commands(area_value, stage_ids, force)
    missing = sorted({script.name for _, _, script, _ in commands if not script.exists()})
    if missing:
        print("\nRequested pipeline stages have not all been migrated yet:")
        for filename in missing:
            print(f"  - {filename}")
        print("Use --dry-run to review migration status or select a migrated stage.")
        return False

    input_keys = required_inputs(stage_ids)
    validation_areas = list(dict.fromkeys(area for _, area, _, _ in commands))
    for area in validation_areas:
        print(f"\nValidating {area} before execution...")
        if not run_validation(area, input_keys=input_keys):
            print(f"\nPipeline stopped: input validation failed for {area}.")
            return False

    for stage, area, script, command in commands:
        print(f"\nRunning stage {stage} for {area}: {script.name}")
        completed = subprocess.run(command, cwd=project_root(), check=False)
        if completed.returncode != 0:
            print(f"Pipeline stopped: stage {stage} returned {completed.returncode}.")
            return False

    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--area",
        choices=available_areas() + ["all"],
        help="One municipality, all six municipalities, or metro",
    )
    parser.add_argument("--list-areas", action="store_true", help="List configured areas")
    parser.add_argument("--check-only", action="store_true", help="Validate inputs and stop")
    parser.add_argument("--dry-run", action="store_true", help="Show stage readiness and stop")
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=[stage for stage, _, _ in STAGES],
        help="Run only the selected stage numbers, for example --stages 01 02",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow selected stages to rebuild existing artifacts",
    )
    return parser


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    parser = build_parser()
    args = parser.parse_args()

    if args.list_areas:
        print_area_list()
        return 0

    if not args.area:
        parser.error("--area is required unless --list-areas is used")

    try:
        areas = selected_areas(args.area)

        if args.dry_run:
            print_dry_run(args.area, args.stages, args.force)
            return 0

        if args.check_only:
            return 0 if all(run_validation(area) for area in areas) else 1

        return 0 if run_plan(args.area, args.stages, args.force) else 1
    except ConfigError as error:
        print(f"Configuration error: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
