"""Feature preprocessing from PMML transformation files.

Supports the PMML NormContinuous element (piecewise-linear normalization),
which is what KNIME's Normalizer node exports (min-max and z-score are both
two-point linear maps). Values outside the declared range are linearly
extrapolated from the end segments (PMML outliers="asIs" default).

Configured per model under scoring.preprocessing:

  scoring:
    preprocessing:
      - path: "path/to/normalizer.pmml"

Each transform is applied in order to the feature frame after shuffling and
before scoring, replacing the referenced columns in place — so shuffling
operates on raw values (preserving each variable's true marginal
distribution) while the model sees the inputs it was trained on.
"""
from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("input_shuffling_v2")


def _local(tag: str) -> str:
    return tag.split("}")[-1]


class PMMLNormalizer:
    """Piecewise-linear per-column normalization parsed from a PMML file."""

    def __init__(self, path: str | Path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Preprocessing PMML not found: {path}")
        self.path = path
        self.fields: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        root = ET.parse(path).getroot()
        for derived in root.iter():
            if _local(derived.tag) != "DerivedField":
                continue
            for child in derived:
                if _local(child.tag) != "NormContinuous":
                    continue
                field = child.attrib["field"]
                pts = sorted(
                    (float(ln.attrib["orig"]), float(ln.attrib["norm"]))
                    for ln in child if _local(ln.tag) == "LinearNorm")
                if len(pts) < 2:
                    continue
                orig = np.array([p[0] for p in pts])
                norm = np.array([p[1] for p in pts])
                self.fields[field] = (orig, norm)

        if not self.fields:
            raise ValueError(
                f"No NormContinuous transformations found in {path} — is this a "
                "normalizer PMML (e.g. from KNIME's Normalizer node)?")
        logger.info(f"Loaded {len(self.fields)} column normalizations from {path.name}")

    @staticmethod
    def _piecewise_linear(x: np.ndarray, orig: np.ndarray, norm: np.ndarray) -> np.ndarray:
        out = np.interp(x, orig, norm)
        # np.interp clamps; PMML default extrapolates from the end segments
        lo_slope = (norm[1] - norm[0]) / (orig[1] - orig[0]) if orig[1] > orig[0] else 0.0
        hi_slope = (norm[-1] - norm[-2]) / (orig[-1] - orig[-2]) if orig[-1] > orig[-2] else 0.0
        below = x < orig[0]
        above = x > orig[-1]
        out = np.where(below, norm[0] + (x - orig[0]) * lo_slope, out)
        out = np.where(above, norm[-1] + (x - orig[-1]) * hi_slope, out)
        return out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for field, (orig, norm) in self.fields.items():
            if field in out.columns:
                out[field] = self._piecewise_linear(
                    out[field].to_numpy(dtype=float), orig, norm)
        return out


def load_preprocessing(cfg: dict) -> list[PMMLNormalizer]:
    """Build the ordered transform chain from scoring.preprocessing."""
    entries = cfg["scoring"].get("preprocessing") or []
    transforms = []
    for entry in entries:
        path = entry["path"] if isinstance(entry, dict) else entry
        transforms.append(PMMLNormalizer(path))
    return transforms
