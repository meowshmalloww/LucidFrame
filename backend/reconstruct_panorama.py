"""Re-run only LucidFrame's panorama-to-splat stage.

This is intentionally separate from the upload API so a generated panorama can
be reconstructed again while tuning geometry, without re-running diffusion.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sharp360_wrapper import reconstruct_sharp360
from splat_compiler import compile_splat


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input equirectangular panorama")
    parser.add_argument("output_dir", type=Path, help="Directory for PLY, report, and final.splat")
    parser.add_argument(
        "--quality",
        choices=("balanced", "detail"),
        default="balanced",
        help="SHARP-360 reconstruction profile",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    gaussians = reconstruct_sharp360(args.input, args.output_dir, args.quality)
    destination = args.output_dir / "final.splat"
    compile_splat(gaussians, destination)
    print(f"Wrote {gaussians.count:,} Gaussians to {destination}")


if __name__ == "__main__":
    main()
