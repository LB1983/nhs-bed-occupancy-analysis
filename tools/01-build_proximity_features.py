#!/usr/bin/env python
"""
Build trust proximity features using HQ lat/lon file (robust to Excel/Windows quirks)
===================================================================================

This version is designed to avoid the two most common problems with Excel-export CSVs:
1) Non-UTF8 encoding (cp1252/latin-1)
2) Non-breaking spaces (0xA0) and odd spacing in column headers (e.g. 'ODS\xa0\xa0code')

Input
-----
01-data-raw/geo/trust_hq_locations.csv

Your headers (as provided):
- ODS  code
- Organisation name
- Region
- ICB
- Postcode
- Description
- Grid Reference
- X (easting)
- Y (northing)
- Latitude
- Longitude

We only require:
- ODS  code  -> org_code
- Latitude  -> latitude
- Longitude -> longitude

Output
------
02-data-interim/geo/trust_proximity_features.csv

Features
--------
- nearest_trust_km
- mean_distance_to_5_nearest_km
- trusts_within_10km / 25km / 50km

Run (repo root)
---------------
python 07-tools/01-build_proximity_features.py
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

INPUT_PATH = Path("01-data-raw/geo/trust_hq_locations.csv")
OUT_DIR = Path("02-data-interim/geo")
OUT_FILE = OUT_DIR / "trust_proximity_features.csv"

RADII_KM = [10, 25, 50]


def read_csv_robust(path: Path) -> pd.DataFrame:
    """Read CSV using a sequence of likely Windows/Excel encodings."""
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError as e:
            last_err = e
    raise last_err  # type: ignore[misc]


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip NBSPs and normalise whitespace in headers."""
    df = df.copy()
    df.columns = (
        pd.Index(df.columns)
        .astype(str)
        .str.replace("\xa0", " ", regex=False)  # NBSP -> space
        .str.replace(r"\s+", " ", regex=True)   # collapse multiple spaces
        .str.strip()
    )
    return df


def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorised haversine distance (km)."""
    r = 6371.0
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return r * c


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_PATH}")

    df = read_csv_robust(INPUT_PATH)
    df = normalise_columns(df)

    # After normalisation, your 'ODS  code' becomes 'ODS code'
    required = ["ODS code", "Latitude", "Longitude"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            "Missing required columns after normalising headers: " + ", ".join(missing)
            + "\nFound columns: " + ", ".join(df.columns)
        )

    df = df.rename(columns={
        "ODS code": "org_code",
        "Latitude": "latitude",
        "Longitude": "longitude",
    })

    df["org_code"] = df["org_code"].astype(str).str.strip().str.upper()
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    df = df.dropna(subset=["latitude", "longitude"]).copy()

    # De-duplicate org_code (keep first)
    df = df.sort_values(["org_code"]).drop_duplicates(subset=["org_code"], keep="first")

    lat = df["latitude"].to_numpy()
    lon = df["longitude"].to_numpy()

    n = len(df)
    dist = np.full((n, n), np.nan, dtype=float)

    for i in range(n):
        dist[i, :] = haversine_km(lat[i], lon[i], lat, lon)

    np.fill_diagonal(dist, np.nan)

    nearest = np.nanmin(dist, axis=1)
    nearest5_mean = np.nanmean(np.sort(dist, axis=1)[:, :5], axis=1)

    features = pd.DataFrame({
        "org_code": df["org_code"].values,
        "nearest_trust_km": nearest,
        "mean_distance_to_5_nearest_km": nearest5_mean,
    })

    for r in RADII_KM:
        features[f"trusts_within_{r}km"] = np.nansum(dist <= r, axis=1).astype(int)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    features.to_csv(OUT_FILE, index=False)

    print("✅ Saved proximity features:", OUT_FILE)
    print("Trusts processed:", len(features))
    print("Nearest trust km (median):", float(np.nanmedian(nearest)))


if __name__ == "__main__":
    main()
