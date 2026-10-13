"""Download the Olist dataset from Kaggle into data/ (needs ~/.kaggle/kaggle.json)."""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import DATA_DIR  # noqa: E402

DATASET = "olistbr/brazilian-ecommerce"


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    subprocess.run(
        ["kaggle", "datasets", "download", DATASET, "-p", str(DATA_DIR), "--unzip"],
        check=True,
    )
    print("CSVs:", sorted(p.name for p in DATA_DIR.glob("*.csv")))


if __name__ == "__main__":
    main()
