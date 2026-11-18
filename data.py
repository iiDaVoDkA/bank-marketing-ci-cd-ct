"""Retrieve a prepared release, record its DVC identity and validate its splits."""

import csv
from pathlib import Path
import os
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import urlopen

import numpy as np
import pandas as pd
import yaml

VERSIONS = ("v1", "v2", "v3")
DEFAULT_BASE_URL = "https://wut2026.blob.core.windows.net/dvc/processed-datasets-ci-cd-ct"
SPLITS = ("train", "evaluation")

CATEGORIES = {
    "job": "admin. blue-collar entrepreneur housemaid management retired self-employed "
           "services student technician unemployed unknown",
    "marital": "divorced married single unknown",
    "education": "basic.4y basic.6y basic.9y high.school illiterate professional.course "
                 "university.degree unknown",
    "default": "yes no unknown",
    "housing": "yes no unknown",
    "loan": "yes no unknown",
    "contact": "cellular telephone",
    "month": "jan feb mar apr may jun jul aug sep oct nov dec",
    "day_of_week": "mon tue wed thu fri",
    "poutcome": "failure nonexistent success",
}

# Use plausible domain limits, leaving room for new observations.
INTEGER_LIMITS = {
    "source_row": (1, None), "age": (16, 120), "campaign": (1, None),
    "pdays": (0, 999), "previous": (0, None),
}
CONTINUOUS = ("emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed")
FEATURES = (
    "age", "job", "marital", "education", "default", "housing", "loan", "contact", "month",
    "day_of_week", "campaign", "pdays", "previous", "poutcome", *CONTINUOUS,
)
COLUMNS = ("source_row", *FEATURES, "y")


class DataError(ValueError):
    """An input failed acquisition or validation."""


def fetch_and_track(version, data_root, manifest_root):
    """Copy both files from one release and compute DVC hashes before validation."""
    source = os.getenv("DATA_DIR")
    base = os.getenv("DATA_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    sas = os.getenv("AZURE_SAS_TOKEN", "").strip().lstrip("?")
    if not source:
        url = urlsplit(base)
        if url.scheme != "https" or not url.hostname or url.username or url.query or url.fragment:
            raise DataError("DATA_BASE_URL must be an HTTPS blob path without credentials or a query")
        if not sas:
            raise DataError("Set AZURE_SAS_TOKEN to read the prepared dataset from Azure")

    data_root.mkdir(parents=True, exist_ok=True)
    manifest_root.mkdir(parents=True, exist_ok=True)
    paths = {split: data_root / f"{split}.csv" for split in SPLITS}
    # Use a small isolated DVC workspace so the student's Git tree stays clean.
    with TemporaryDirectory(prefix="bank-dvc-") as tmp:
        workspace = Path(tmp)
        for split, path in paths.items():
            target = workspace / path.name
            try:
                if source:
                    shutil.copyfile(Path(source) / version / path.name, target)
                else:
                    with urlopen(f"{base}/{version}/{path.name}?{sas}", timeout=45) as response:
                        with target.open("wb") as output:
                            shutil.copyfileobj(response, output)
            except HTTPError as exc:
                raise DataError(f"Could not fetch {version}/{path.name} (HTTP {exc.code})") from None
            except (OSError, URLError):
                raise DataError(f"Could not read {version}/{path.name}") from None

        dvc = [sys.executable, "-m", "dvc"]
        env = {
            **os.environ, "DVC_NO_ANALYTICS": "true",
            "DVC_SITE_CACHE_DIR": str(workspace / "site-cache"),
        }
        try:
            subprocess.run([*dvc, "init", "--no-scm", "-q"], cwd=workspace, env=env, check=True)
            subprocess.run(
                [*dvc, "add", "-q", *(path.name for path in paths.values())],
                cwd=workspace, env=env, check=True,
            )
        except subprocess.CalledProcessError:
            raise DataError("DVC could not track the prepared CSVs") from None
        fingerprints = {}
        for split, path in paths.items():
            manifest = workspace / f"{path.name}.dvc"
            fingerprints[split] = yaml.safe_load(manifest.read_text())["outs"][0]["md5"]
            shutil.copyfile(manifest, manifest_root / manifest.name)
            shutil.copyfile(workspace / path.name, path)
    return paths, fingerprints


def _validate_values(frame, name):
    if frame.empty or frame.isna().any().any():
        raise DataError(f"{name}: empty dataset or missing value")

    for col, choices in CATEGORIES.items():
        if not frame[col].isin(choices.split()).all():
            raise DataError(f"{name}: invalid category in {col}")

    for col in (*INTEGER_LIMITS, *CONTINUOUS):
        values = pd.to_numeric(frame[col], errors="coerce")
        if not np.isfinite(values).all():
            raise DataError(f"{name}: missing or non-finite numeric value in {col}")
        if col in INTEGER_LIMITS:
            lower, upper = INTEGER_LIMITS[col]
            if ((values % 1 != 0) | (values < lower)).any() or (
                upper is not None and (values > upper).any()
            ):
                raise DataError(f"{name}: invalid integer or range in {col}")
            values = values.astype("int64")
        else:
            values = values.astype("float64")
        frame[col] = values


def read_split(path):
    """Require the published header and normalize types before scikit-learn sees the data."""
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = csv.reader(handle, strict=True)
            if tuple(next(rows, [])) != COLUMNS:
                raise DataError(f"{path.name}: wrong CSV schema")
            if any(len(row) != len(COLUMNS) for row in rows):
                raise DataError(f"{path.name}: inconsistent number of CSV fields")
        frame = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
    except (OSError, UnicodeError, csv.Error, pd.errors.ParserError):
        raise DataError(f"{path.name}: unreadable CSV") from None

    _validate_values(frame, path.name)
    if set(frame["y"]) != {"yes", "no"}:
        raise DataError(f"{path.name}: y must contain both classes, yes and no")
    if not frame["source_row"].is_monotonic_increasing or not frame["source_row"].is_unique:
        raise DataError(f"{path.name}: source_row must be unique and in chronological order")
    return frame


def validate_splits(paths):
    train = read_split(paths["train"])
    evaluation = read_split(paths["evaluation"])
    if train["source_row"].max() >= evaluation["source_row"].min():
        raise DataError("Training observations must precede the evaluation window")
    return train, evaluation
