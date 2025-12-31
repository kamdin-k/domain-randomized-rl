from __future__ import annotations
import csv
from pathlib import Path
from typing import Any

class CSVLogger:
    """
    Minimal CSV logger for synthetic rollouts.
    Writes one row per step.
    """
    def __init__(self, path: str | Path, fieldnames: list[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = fieldnames
        self._file = self.path.open("w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames)
        self._writer.writeheader()

    def log(self, row: dict[str, Any]) -> None:
        # ensure consistent columns (missing keys become blank)
        self._writer.writerow({k: row.get(k, "") for k in self.fieldnames})

    def close(self) -> None:
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
