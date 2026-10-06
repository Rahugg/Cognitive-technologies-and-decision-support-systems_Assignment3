import argparse
import os
from pathlib import Path

import kagglehub

DATASETS = {
    "fer2013": "msambare/fer2013",
    "ckplus": "shawon10/ckplus",
    "rafdb": "shuvoalok/raf-db-dataset",
}


def main():
    parser = argparse.ArgumentParser(description="Download public FER datasets with KaggleHub")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    args = parser.parse_args()
    target = Path(os.getenv("DATA_DIR", "/workspace/data")) / "raw"
    target.mkdir(parents=True, exist_ok=True)
    if os.getenv("KAGGLE_API_TOKEN"):
        os.environ["KAGGLE_API_TOKEN"] = os.environ["KAGGLE_API_TOKEN"]
    for name in args.datasets:
        print(f"Downloading {name} ({DATASETS[name]})")
        downloaded = Path(kagglehub.dataset_download(DATASETS[name]))
        destination = target / name
        if destination.exists():
            print(f"Already present: {destination}")
            continue
        # KaggleHub cache can be mounted or copied by Docker; copy to stable project path.
        import shutil
        shutil.copytree(downloaded, destination)
        print(f"Saved {name} to {destination}")


if __name__ == "__main__":
    main()
