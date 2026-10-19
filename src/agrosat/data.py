"""Sentinel-2 L2A search, inventory and loading for the AgroSat area of interest.

Pipeline position: STAC search -> inventory checks -> lazy datacube on the UTM 34N grid.
Cloud masking and monthly composites live in a later module (Day 3).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import pystac
import xarray as xr

from agrosat import config

_MGRS_IN_ID = re.compile(r"_(\d{2}[C-X][A-Z]{2})_")


# --------------------------------------------------------------------------- search


def search_items(
    bbox: Sequence[float] = config.AOI_BBOX,
    datetime: str = config.SEASON_2021,
    max_cloud_pct: float = config.MAX_SCENE_CLOUD_PCT,
    stac_url: str = config.STAC_URL,
) -> pystac.ItemCollection:
    """Return Sentinel-2 L2A items intersecting ``bbox`` within ``datetime``.

    ``eo:cloud_cover`` is computed for the whole 110 x 110 km MGRS tile, not for our AOI,
    so this filter is only a coarse pre-selection. Pixel-level masking with SCL is still
    required later.
    """
    from pystac_client import Client  # imported lazily so tests run without network

    catalog = Client.open(stac_url)
    search = catalog.search(
        collections=[config.COLLECTION],
        bbox=list(bbox),
        datetime=datetime,
        query={"eo:cloud_cover": {"lt": max_cloud_pct}},
    )
    return search.item_collection()


# ------------------------------------------------------------------------ inventory


def mgrs_tile(item: pystac.Item) -> str:
    """Return the MGRS tile code (e.g. ``34TDR``) of a Sentinel-2 item."""
    props = item.properties
    parts = (props.get("mgrs:utm_zone"), props.get("mgrs:latitude_band"),
             props.get("mgrs:grid_square"))
    if all(part is not None for part in parts):
        zone, band, square = parts
        return f"{int(zone):02d}{band}{square}"
    match = _MGRS_IN_ID.search(item.id)
    return match.group(1) if match else "unknown"


def processing_baseline(item: pystac.Item) -> float | None:
    """Return the ESA processing baseline as a float (``"04.00"`` -> ``4.0``)."""
    raw = item.properties.get("s2:processing_baseline")
    try:
        return float(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def needs_offset_correction(item: pystac.Item) -> bool:
    """True if this item's digital numbers still carry the -1000 BOA offset.

    From processing baseline 04.00 (25 Jan 2022) ESA stores L2A as
    ``DN = reflectance * 10000 + 1000``. Earth Search flags items whose offset was already
    removed with ``earthsearch:boa_offset_applied``; we only correct when the baseline is
    new *and* the provider has not done it for us. Getting this wrong biases every
    band by +0.1 reflectance and silently shifts NDVI, which would break 2021 -> 2025
    model transfer.
    """
    if item.properties.get("earthsearch:boa_offset_applied") is True:
        return False
    baseline = processing_baseline(item)
    return baseline is not None and baseline >= config.OFFSET_BASELINE


def items_to_frame(items: Sequence[pystac.Item]) -> pd.DataFrame:
    """Flatten STAC items into a tidy table, one row per item."""
    rows: list[dict[str, Any]] = []
    for item in items:
        if item.datetime is None:
            continue
        ts = pd.Timestamp(item.datetime)
        ts = ts.tz_convert(None) if ts.tzinfo else ts
        rows.append(
            {
                "id": item.id,
                "datetime": ts,
                "date": ts.normalize(),
                "month": ts.month,
                "mgrs_tile": mgrs_tile(item),
                "cloud_pct": item.properties.get("eo:cloud_cover"),
                "baseline": processing_baseline(item),
                "needs_offset_fix": needs_offset_correction(item),
            }
        )
    frame = pd.DataFrame(rows)
    return frame.sort_values("datetime").reset_index(drop=True) if not frame.empty else frame


def scenes_per_month(frame: pd.DataFrame) -> pd.DataFrame:
    """Count acquisition dates per month and MGRS tile.

    A month with 0-1 dates is a warning sign for the monthly median composite: it will be
    dominated by a single acquisition and may be mostly cloud-masked.
    """
    if frame.empty:
        return pd.DataFrame()
    table = (
        frame.groupby(["month", "mgrs_tile"])["date"].nunique().unstack(fill_value=0)
    )
    table["unique_dates"] = frame.groupby("month")["date"].nunique()
    return table


def clearest_item(items: Sequence[pystac.Item], month: int) -> pystac.Item:
    """Return the item with the lowest scene cloud cover acquired in ``month``."""
    candidates = [it for it in items if it.datetime and it.datetime.month == month]
    if not candidates:
        raise ValueError(f"No items found for month {month}.")
    return min(candidates, key=lambda it: it.properties.get("eo:cloud_cover", 100.0))


# -------------------------------------------------------------------------- loading


def load_cube(
    items: Sequence[pystac.Item],
    bbox: Sequence[float] = config.AOI_BBOX,
    bands: Sequence[str] = (*config.REFLECTANCE_BANDS, config.SCL_BAND),
    crs: str = config.TARGET_CRS,
    resolution: float = config.RESOLUTION_M,
    chunk_px: int = 2048,
) -> xr.Dataset:
    """Lazily load items as a dask-backed datacube on the target UTM grid.

    * ``groupby="solar_day"`` mosaics neighbouring MGRS tiles from the same overpass into
      one time step, so the AOI is not split across tiles.
    * Reflectance bands use bilinear resampling (only the 20 m SWIR band is actually
      upsampled). SCL is categorical and must use nearest neighbour: bilinear would
      invent classes like 3.5 between "shadow" and "vegetation".
    * odc-stac snaps the output grid to multiples of ``resolution``, and Sentinel-2 tile
      origins are multiples of 10 m, so our pixels coincide with native 10 m pixels.
    """
    import odc.stac

    return odc.stac.load(
        items,
        bands=list(bands),
        bbox=list(bbox),
        crs=crs,
        resolution=resolution,
        chunks={"x": chunk_px, "y": chunk_px},
        groupby="solar_day",
        resampling={"*": "bilinear", config.SCL_BAND: "nearest"},
    )


def to_reflectance(
    dataset: xr.Dataset,
    offset_by_date: pd.Series,
    bands: Sequence[str] = config.REFLECTANCE_BANDS,
) -> xr.Dataset:
    """Convert digital numbers to surface reflectance (0-1) and harmonize baselines.

    ``offset_by_date`` maps each acquisition date to whether the -1000 offset must be
    removed (see :func:`needs_offset_correction`). DN 0 is Sentinel-2 nodata -> NaN.
    """
    dates = pd.DatetimeIndex(dataset["time"].values).normalize()
    lookup = offset_by_date.groupby(level=0).any()
    flags = np.array([bool(lookup.get(d, False)) for d in dates])
    offset = xr.DataArray(
        np.where(flags, config.BOA_ADD_OFFSET, 0), coords={"time": dataset["time"]}, dims="time"
    )
    out = dataset.copy()
    for band in bands:
        dn = dataset[band]
        refl = ((dn + offset) / 10_000.0).clip(min=0.0).astype("float32")
        out[band] = refl.where(dn != 0)
    return out


# ------------------------------------------------------------------------- previews


def rgb_stretch(
    red: np.ndarray, green: np.ndarray, blue: np.ndarray, percentiles: tuple[float, float] = (2, 98)
) -> np.ndarray:
    """Stack bands into an ``(H, W, 3)`` float image scaled to 0-1 by percentile stretch."""
    stack = np.stack([red, green, blue], axis=-1).astype("float64")
    low, high = np.nanpercentile(stack, percentiles)
    scaled = (stack - low) / max(high - low, 1e-9)
    return np.nan_to_num(np.clip(scaled, 0.0, 1.0))
