"""Build demand-grounded Hangzhou BEVTSPD cases from official LaDe-D parquet.

Real: AOI centroids from courier-day / region-day slices.
Assumed (declared, NOT from LaDe): depot = AOI centroid, peripheral candidate
BSS layout (P-style in local km), full drone eligibility, Zhu energy params.

Does NOT invent truck/drone routes. Writes CSV coords + JSON meta for Julia freeze.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PARQUET = None
OUT = ROOT / "data" / "lade_generated"
OFFICIAL_SHA = "00b6b4dca471cadee796a3fc7f07dcd919499ec58459ea67db784dcb06089e75"
LNG_LO, LNG_HI = 119.0, 121.0
LAT_LO, LAT_HI = 29.5, 31.0


def assert_sha(path: Path) -> None:
    dig = hashlib.sha256(path.read_bytes()).hexdigest()
    assert dig == OFFICIAL_SHA, dig


def aoi_centroids(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.dropna(subset=["aoi_id", "lng", "lat"])
        .groupby("aoi_id", as_index=False)
        .agg(lng=("lng", "median"), lat=("lat", "median"), parcels=("order_id", "count"))
    )


def to_local_km(lng: np.ndarray, lat: np.ndarray, lng0: float, lat0: float):
    x = (lng - lng0) * 111.32 * np.cos(np.deg2rad(lat0))
    y = (lat - lat0) * 110.57
    return x, y


def peripheral_stations(k: int, seed: int, lim: float = 20.0) -> np.ndarray:
    """Synthetic P-style candidate BSS in local km; NOT real Hangzhou facilities."""
    rng = np.random.default_rng(seed)
    x = np.zeros((k, 2))
    for j in range(k):
        if rng.random() < 0.25:
            x[j] = 40.0 * rng.random(2) - 20.0
        else:
            ang = 2 * np.pi * rng.random()
            rad = 14.0 + 5.0 * rng.random()
            x[j, 0] = rad * np.cos(ang)
            x[j, 1] = rad * np.sin(ang)
        x[j] = np.clip(x[j], -lim, lim)
    return x


def station_count_for(n: int) -> int:
    if n <= 40:
        return 5
    if n <= 80:
        return 6
    return 8


def pick_near(series: pd.Series, target: float):
    idx = (series - target).abs().idxmin()
    return idx, int(series.loc[idx])


def main() -> None:
    assert_sha(PARQUET)
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(
        PARQUET, columns=["order_id", "region_id", "courier_id", "lng", "lat", "aoi_id", "ds"]
    )
    assert len(df) == 1_861_600
    df["lng"] = pd.to_numeric(df["lng"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    core = df.loc[
        np.isfinite(df["lng"])
        & np.isfinite(df["lat"])
        & df["lng"].between(LNG_LO, LNG_HI)
        & df["lat"].between(LAT_LO, LAT_HI)
    ].copy()

    r_aoi = core.groupby(["region_id", "ds"], observed=True)["aoi_id"].nunique()
    c_aoi = core.groupby(["courier_id", "ds"], observed=True)["aoi_id"].nunique()
    rg = core.groupby(["region_id", "ds"], observed=True).size()
    cg = core.groupby(["courier_id", "ds"], observed=True).size()

    # Target AOI counts in the pre-registered 30–100 priority band (+ one P99 stress)
    targets = [
        ("region", 35),
        ("region", 55),
        ("region", 80),
        ("region", 100),
        ("region", 120),
        ("courier", 22),
        ("courier", 30),
        ("courier", 45),
    ]

    used = set()
    cases = []
    for kind, target in targets:
        series = r_aoi if kind == "region" else c_aoi
        # restrict to band when possible
        band = series[(series >= max(15, target - 15)) & (series <= target + 20)]
        if band.empty:
            band = series
        key, n_aoi = pick_near(band, float(target))
        if key in used:
            # next nearest unused
            rem = band.drop(labels=[k for k in used if k in band.index], errors="ignore")
            if rem.empty:
                rem = series.drop(labels=[k for k in used if k in series.index], errors="ignore")
            key, n_aoi = pick_near(rem, float(target))
        used.add(key)

        if kind == "region":
            region_id, ds = key
            sub = core.loc[(core["region_id"] == region_id) & (core["ds"] == ds)]
            case_id = f"HZ_region{region_id}_ds{ds}_aoi{n_aoi}"
            n_parcels = int(rg.loc[key])
        else:
            courier_id, ds = key
            sub = core.loc[(core["courier_id"] == courier_id) & (core["ds"] == ds)]
            case_id = f"HZ_courier{courier_id}_ds{ds}_aoi{n_aoi}"
            n_parcels = int(cg.loc[key])

        stops = aoi_centroids(sub)
        n = len(stops)
        if n < 15:
            continue
        lng0 = float(stops["lng"].mean())
        lat0 = float(stops["lat"].mean())
        x, y = to_local_km(stops["lng"].to_numpy(), stops["lat"].to_numpy(), lng0, lat0)
        customers = np.column_stack([x, y])
        # Center depot at origin (AOI mean) — ASSUMED
        customers = customers - customers.mean(axis=0)
        k = station_count_for(n)
        # seed derived from case identity for reproducibility
        seed = int(hashlib.sha256(case_id.encode()).hexdigest()[:8], 16) % 10_000_000
        stations = peripheral_stations(k, seed)

        cust_csv = OUT / f"{case_id}_customers.csv"
        st_csv = OUT / f"{case_id}_stations.csv"
        np.savetxt(cust_csv, customers, delimiter=",", header="x_km,y_km", comments="")
        np.savetxt(st_csv, stations, delimiter=",", header="x_km,y_km", comments="")

        span = float(np.hypot(np.ptp(customers[:, 0]), np.ptp(customers[:, 1])))
        meta = {
            "case_id": case_id,
            "kind": kind,
            "source": "LaDe-D Hangzhou official parquet",
            "source_sha256": OFFICIAL_SHA,
            "n_customers_aoi": n,
            "n_parcels": n_parcels,
            "k_stations_synthetic": k,
            "station_layout": "synthetic_peripheral_P_style_local_km",
            "depot": "ASSUMED_at_AOI_centroid_origin",
            "drone_eligibility": "ASSUMED_all_AOI_serviceable",
            "bss_reality": "NOT real Hangzhou swap stations",
            "projection": "equirectangular_local_km_around_AOI_mean",
            "origin_lng_lat": [lng0, lat0],
            "approx_customer_span_km": span,
            "key": [int(x) for x in key],
            "ds": int(key[1]),
            "customers_csv": cust_csv.name,
            "stations_csv": st_csv.name,
            "geometry_seed_for_stations": seed,
        }
        (OUT / f"{case_id}_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        cases.append(meta)
        print(f"CASE {case_id} n={n} parcels={n_parcels} k={k} span≈{span:.1f}km")

    manifest = {
        "benchmark_id": "LADE_HZ_DEMAND_GROUNDED_SIMULATED_OPS_V1",
        "count": len(cases),
        "cases": cases,
        "warning": "Demand from LaDe; depot/BSS/drone ops are simulated assumptions.",
    }
    (OUT / "LADE_HZ_CASES_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    # flat CSV index for Julia
    rows = ["case_id,kind,n,k,n_parcels,customers_csv,stations_csv,span_km"]
    for c in cases:
        rows.append(
            f"{c['case_id']},{c['kind']},{c['n_customers_aoi']},{c['k_stations_synthetic']},"
            f"{c['n_parcels']},{c['customers_csv']},{c['stations_csv']},{c['approx_customer_span_km']:.4f}"
        )
    (OUT / "lade_hz_manifest.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"WROTE {len(cases)} cases -> {OUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    PARQUET, OUT = args.parquet, args.out
    main()
