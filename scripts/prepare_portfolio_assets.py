from __future__ import annotations

import sys
from pathlib import Path
import argparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cuda_image_processing.io_utils import DATA_DIR, read_image
from cuda_image_processing.portfolio import write_portfolio_assets
from cuda_image_processing.sample_data import generate_sample_assets


def main() -> None:
    parser = argparse.ArgumentParser(description="Build committed portfolio screenshots and demo media.")
    parser.add_argument("--refresh-sample-data", action="store_true", help="Rebuild the synthetic sample image and video first.")
    args = parser.parse_args()

    sample_paths = generate_sample_assets(overwrite=args.refresh_sample_data)
    frame = read_image(sample_paths.image_path)
    hero_path, stages_path, video_path = write_portfolio_assets(frame, sample_paths.video_path)
    print(f"Hero image: {hero_path}")
    print(f"Stage collage: {stages_path}")
    print(f"Demo video: {video_path}")


if __name__ == "__main__":
    main()
