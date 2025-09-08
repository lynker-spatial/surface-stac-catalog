# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "polars",
#     "pystac",
# ]
# ///

from pathlib import Path
from dataclasses import dataclass
import csv

@dataclass
class STACMetadata:
    domain: str
    region: str
    source: str
    resolution: str
    has_topo: bool
    has_bathymetry: bool
    horizontal_crs: str
    vertical_datum: str
    vertical_datum_conversion: str
    priority: int
    source_url: str
    asset_urls: list[str]

def ensure_metadata_csv() -> Path:
    path = Path("data/metadata.csv")
    if not path.exists():
        raise FileNotFoundError(f"'{path}' does not exist")
    return path

def read_metadata_csv() -> list[STACMetadata]:
    csv_file = ensure_metadata_csv()
    with csv_file.open("rt") as f:
        return [STACMetadata(*row) for row in csv.reader(f)]


def main() -> None:
    meta = read_metadata_csv()
    print(meta)
    # print("Hello from generate_catalog.py!")


if __name__ == "__main__":
    main()
