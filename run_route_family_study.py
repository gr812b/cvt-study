from __future__ import annotations

import argparse
from pathlib import Path

from cvt_track_study.studies.route_family import run_route_family_study_project


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one CVT study independently on every route-family bundle."
    )
    parser.add_argument("project")
    parser.add_argument("study")
    parser.add_argument("--family-manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--replicates", type=int)
    args = parser.parse_args()
    output = run_route_family_study_project(
        args.project,
        study=args.study,
        family_manifest_path=args.family_manifest,
        output_directory=args.output,
        replicates_override=args.replicates,
    )
    print(output)


if __name__ == "__main__":
    main()
