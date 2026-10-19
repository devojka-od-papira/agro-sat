"""Offline tests for agrosat.data, using synthetic STAC items and local GeoTIFFs."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pystac
import pytest
import rasterio
from rasterio.transform import from_origin

from agrosat import config
from agrosat.data import (
    clearest_item,
    items_to_frame,
    load_cube,
    mgrs_tile,
    needs_offset_correction,
    rgb_stretch,
    scenes_per_month,
    to_reflectance,
)

# A 10 km x 10 km block in UTM 34N near Novi Sad, origin on the 10 m Sentinel-2 grid.
ORIGIN_X, ORIGIN_Y = 410_000.0, 5_020_000.0
SIZE_10M = 1000


def make_item(
    item_id: str,
    when: datetime,
    cloud: float = 5.0,
    baseline: str = "03.00",
    offset_applied: bool | None = None,
    assets: dict[str, pystac.Asset] | None = None,
) -> pystac.Item:
    props: dict = {"eo:cloud_cover": cloud, "s2:processing_baseline": baseline,
                   "mgrs:utm_zone": 34, "mgrs:latitude_band": "T", "mgrs:grid_square": "DR"}
    if offset_applied is not None:
        props["earthsearch:boa_offset_applied"] = offset_applied
    item = pystac.Item(item_id, geometry=None, bbox=None, datetime=when, properties=props)
    for key, asset in (assets or {}).items():
        item.add_asset(key, asset)
    return item


# ------------------------------------------------------------------ metadata logic


@pytest.mark.parametrize(
    ("baseline", "applied", "expected"),
    [
        ("03.00", None, False),   # 2021: no offset in the data
        ("04.00", None, True),    # 2022+: raw ESA DNs carry +1000
        ("05.09", None, True),
        ("05.09", True, False),   # provider already removed it
        (None, None, False),
    ],
)
def test_needs_offset_correction(baseline: str | None, applied: bool | None,
                                 expected: bool) -> None:
    item = make_item("x", datetime(2025, 7, 1, tzinfo=UTC), baseline=baseline,
                     offset_applied=applied)
    assert needs_offset_correction(item) is expected


def test_mgrs_tile_from_properties_and_id() -> None:
    assert mgrs_tile(make_item("x", datetime(2021, 7, 1, tzinfo=UTC))) == "34TDR"
    bare = pystac.Item("S2B_34TCR_20210710_0_L2A", None, None,
                       datetime(2021, 7, 10, tzinfo=UTC), {})
    assert mgrs_tile(bare) == "34TCR"


def test_inventory_counts_unique_dates_per_month() -> None:
    items = [
        make_item("a", datetime(2021, 6, 5, 9, 40, tzinfo=UTC)),
        make_item("b", datetime(2021, 7, 5, 9, 40, tzinfo=UTC)),
        make_item("c", datetime(2021, 7, 20, 9, 40, tzinfo=UTC)),
    ]
    table = scenes_per_month(items_to_frame(items))
    assert table.loc[6, "unique_dates"] == 1
    assert table.loc[7, "unique_dates"] == 2


def test_clearest_item_picks_lowest_cloud() -> None:
    items = [
        make_item("cloudy", datetime(2021, 7, 5, tzinfo=UTC), cloud=25),
        make_item("clear", datetime(2021, 7, 20, tzinfo=UTC), cloud=2),
        make_item("june", datetime(2021, 6, 1, tzinfo=UTC), cloud=0),
    ]
    assert clearest_item(items, 7).id == "clear"
    with pytest.raises(ValueError):
        clearest_item(items, 12)


def test_rgb_stretch_shape_and_range() -> None:
    band = np.arange(100, dtype=float).reshape(10, 10)
    image = rgb_stretch(band, band, band)
    assert image.shape == (10, 10, 3)
    assert image.min() == 0.0 and image.max() == 1.0


# ----------------------------------------------------- loading on the real grid logic


def _write_band(path: Path, value: int, res: int) -> pystac.Asset:
    size = SIZE_10M * 10 // res
    data = np.full((size, size), value, dtype="uint16")
    data[0, 0] = 0  # one nodata pixel
    transform = from_origin(ORIGIN_X, ORIGIN_Y, res, res)
    with rasterio.open(path, "w", driver="GTiff", width=size, height=size, count=1,
                       dtype="uint16", crs=config.TARGET_CRS, transform=transform,
                       nodata=0) as dst:
        dst.write(data, 1)
    asset = pystac.Asset(str(path), media_type=pystac.MediaType.GEOTIFF, roles=["data"])
    asset.extra_fields.update({
        "proj:epsg": 32634,
        "proj:shape": [size, size],
        "proj:transform": list(transform)[:6],
        "raster:bands": [{"nodata": 0, "data_type": "uint16"}],
    })
    return asset


@pytest.fixture()
def local_items(tmp_path: Path) -> list[pystac.Item]:
    """Two dates: 2021 (baseline 03.00) and 2025 (baseline 05.11, offset not removed)."""
    items = []
    for name, when, baseline, dn in [
        ("S2A_34TDR_20210715_0_L2A", datetime(2021, 7, 15, 9, 40, tzinfo=UTC), "03.00", 1500),
        ("S2B_34TDR_20250715_0_L2A", datetime(2025, 7, 15, 9, 40, tzinfo=UTC), "05.11", 2500),
    ]:
        assets = {band: _write_band(tmp_path / f"{name}_{band}.tif", dn, 10)
                  for band in ("blue", "green", "red", "nir")}
        assets["swir16"] = _write_band(tmp_path / f"{name}_swir16.tif", dn, 20)
        assets["scl"] = _write_band(tmp_path / f"{name}_scl.tif", 4, 20)  # 4 = vegetation
        item = make_item(name, when, baseline=baseline, assets=assets)
        item.geometry, item.bbox = _footprint_wgs84()
        items.append(item)
    return items


def _footprint_wgs84() -> tuple[dict, list[float]]:
    """Footprint of the synthetic rasters in lon/lat, as real STAC items carry."""
    from rasterio.warp import transform_bounds
    from shapely.geometry import box, mapping

    extent = SIZE_10M * 10
    bounds = transform_bounds(config.TARGET_CRS, "EPSG:4326", ORIGIN_X, ORIGIN_Y - extent,
                              ORIGIN_X + extent, ORIGIN_Y)
    return mapping(box(*bounds)), list(bounds)


def test_load_cube_grid_and_harmonization(local_items: list[pystac.Item]) -> None:
    # bbox well inside the synthetic rasters (approx. 19.9-19.95 E, 45.31-45.34 N)
    bbox = (19.88, 45.27, 19.93, 45.30)
    cube = load_cube(local_items, bbox=bbox, chunk_px=256)

    assert cube.odc.crs == config.TARGET_CRS
    assert cube.odc.geobox.resolution.x == 10
    # Pixel edges must sit on the native 10 m Sentinel-2 grid (no half-pixel shift).
    edge_x = float(cube.x[0]) - 5
    assert edge_x % 10 == 0
    assert cube.sizes["time"] == 2

    # SCL stays categorical after 20 m -> 10 m (nearest neighbour).
    assert set(np.unique(cube.scl.values)) <= {0, 4}

    flags = items_to_frame(local_items).set_index("date")["needs_offset_fix"]
    refl = to_reflectance(cube, flags).compute()
    red = refl.red.isel(y=slice(1, None)).median(dim=("x", "y")).values
    np.testing.assert_allclose(red, [0.15, 0.15], atol=1e-6)  # 2025 DN 2500 -> 0.15 after fix


def test_to_reflectance_marks_nodata() -> None:
    import xarray as xr
    times = pd.to_datetime(["2021-07-15"])
    data = np.array([[[0, 1000]]], dtype="uint16")
    ds = xr.Dataset({"red": (("time", "y", "x"), data)},
                    coords={"time": times, "y": [0], "x": [0, 1]})
    out = to_reflectance(ds, pd.Series([False], index=times), bands=("red",))
    assert np.isnan(out.red.values[0, 0, 0])
    assert out.red.values[0, 0, 1] == pytest.approx(0.1)


def test_cli_save_preview(local_items: list[pystac.Item], tmp_path: Path) -> None:
    from agrosat.cli import save_preview

    out = tmp_path / "preview.png"
    save_preview(local_items, month=7, resolution=60, out=out)
    assert out.stat().st_size > 10_000
