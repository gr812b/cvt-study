from __future__ import annotations

import argparse
from pathlib import Path

from cvt_track_study.track.robustness_family import (
    run_route_family_track_robustness_project,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run track robustness independently for every supported route."
    )
    parser.add_argument("project")
    parser.add_argument("--study", default="track_robustness")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    output = run_route_family_track_robustness_project(
        args.project,
        study=args.study,
        output_directory=args.output,
        workers=args.workers,
        progress=not args.no_progress,
    )
    print(output)


if __name__ == "__main__":
    main()
