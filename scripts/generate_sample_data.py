from __future__ import annotations

import sys
from pathlib import Path
import argparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cuda_image_processing.sample_data import generate_sample_assets


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic lane-detection sample data.")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild sample files even if they already exist.")
    args = parser.parse_args()

    paths = generate_sample_assets(overwrite=args.overwrite)
    print(f"Sample image written to {paths.image_path}")
    print(f"Sample video written to {paths.video_path}")


if __name__ == "__main__":
    main()
