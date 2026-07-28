"""Download and unpack the PoseBusters paper data from Zenodo.

Two files, ~53 MB total. The archive holds the benchmark *inputs* (protein PDB,
crystal ligand SDF and a generated start conformer for each of 513 complexes);
the CSV holds the paper's own check results for every pose. The predicted poses
themselves were never published, which is why this project analyses the
published results rather than re-running `bust` on predictions.
"""

from __future__ import annotations

import logging
import urllib.request
import zipfile

from . import paths

log = logging.getLogger(__name__)

RECORD = "https://zenodo.org/records/8278563/files"
FILES = {
    "posebusters_paper_data.zip": paths.RAW / "posebusters_paper_data.zip",
    "posebusters_paper_results.csv": paths.RESULTS_CSV,
}


def download() -> None:
    paths.RAW.mkdir(parents=True, exist_ok=True)
    for name, target in FILES.items():
        if target.exists():
            log.info("have %s (%.1f MB)", target.name, target.stat().st_size / 1e6)
            continue
        url = f"{RECORD}/{name}?download=1"
        log.info("fetching %s", url)
        urllib.request.urlopen(url)  # noqa: S310 - fixed https Zenodo URL
        urllib.request.urlretrieve(url, target)  # noqa: S310
        log.info("  -> %s (%.1f MB)", target, target.stat().st_size / 1e6)


def extract() -> None:
    archive = FILES["posebusters_paper_data.zip"]
    for dataset, directory in paths.SET_DIRS.items():
        if directory.is_dir() and any(directory.iterdir()):
            log.info("have %s (%d complexes)", dataset, len(list(directory.iterdir())))
            return
    log.info("extracting %s", archive.name)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(paths.EXTRACTED)
    for dataset, directory in paths.SET_DIRS.items():
        log.info("  %s: %d complexes", dataset, len(list(directory.iterdir())))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    download()
    extract()


if __name__ == "__main__":
    main()
