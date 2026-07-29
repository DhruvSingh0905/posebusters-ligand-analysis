"""Print an HTML document to PDF with headless Chrome.

Chrome is the only renderer assumed present on a Mac or a CI image with a
browser. Its absence is not fatal: the HTML is the primary artifact and prints
from any browser, so a missing binary degrades to a warning rather than
failing the run.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)

CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "google-chrome",
    "chromium",
    "chromium-browser",
]


def find_chrome() -> str | None:
    for candidate in CANDIDATES:
        if Path(candidate).is_file():
            return candidate
        found = shutil.which(candidate)
        if found:
            return found
    return None


def to_pdf(html: Path, pdf: Path | None = None, timeout: int = 120) -> Path | None:
    """Render `html` to PDF. Returns the path, or None if no browser was found."""
    pdf = pdf or html.with_suffix(".pdf")
    chrome = find_chrome()
    if chrome is None:
        log.warning(
            "no Chrome/Chromium found - skipping %s. The HTML is complete; "
            "print it from any browser to produce the PDF.", pdf.name
        )
        return None

    result = subprocess.run(
        [chrome, "--headless", "--disable-gpu", "--no-pdf-header-footer",
         f"--print-to-pdf={pdf}", html.resolve().as_uri()],
        capture_output=True, text=True, timeout=timeout,
    )
    if not pdf.exists():
        log.error("chrome exited %d without writing %s: %s",
                  result.returncode, pdf.name, result.stderr[-300:])
        return None
    log.info("wrote %s (%.0f kB)", pdf, pdf.stat().st_size / 1000)
    return pdf
