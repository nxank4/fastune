"""
Download wine quality datasets from UCI repository.
"""

import os
import urllib.request
from pathlib import Path


def download_wine_data():
    """Download wine quality datasets."""
    # Create data directory in parent folder
    data_dir = Path("../data")
    data_dir.mkdir(exist_ok=True)

    datasets = {
        "winequality-red.csv": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
        "winequality-white.csv": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv",
    }

    print("📥 Downloading wine quality datasets...")

    for filename, url in datasets.items():
        filepath = data_dir / filename
        if not filepath.exists():
            print(f"   Downloading {filename}...")
            try:
                urllib.request.urlretrieve(url, filepath)
                print(f"   ✅ Downloaded {filename}")
            except Exception as e:
                print(f"   ❌ Failed to download {filename}: {e}")
        else:
            print(f"   ✅ {filename} already exists")

    print("✅ Data download complete!")


if __name__ == "__main__":
    download_wine_data()
