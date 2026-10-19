"""Command-line entry point: inventory the 2021 season and save a July RGB preview.

Usage::

    agrosat-inventory                      # inventory + preview at 20 m
    agrosat-inventory --no-preview         # inventory only (seconds, metadata only)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from agrosat import config  # noqa: E402
from agrosat.data import (  # noqa: E402
    clearest_item,
    items_to_frame,
    load_cube,
    rgb_stretch,
    scenes_per_month,
    search_items,
    to_reflectance,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--datetime", default=config.SEASON_2021)
    parser.add_argument("--max-cloud", type=float, default=config.MAX_SCENE_CLOUD_PCT)
    parser.add_argument("--preview-month", type=int, default=7)
    parser.add_argument("--preview-resolution", type=float, default=20.0,
                        help="Metres. 20 m keeps the preview download small.")
    parser.add_argument("--out", type=Path, default=Path("docs/img/july_2021_rgb.png"))
    parser.add_argument("--no-preview", action="store_true")
    return parser.parse_args()


def save_preview(items: list, month: int, resolution: float, out: Path) -> None:
    """Load the clearest scene of ``month`` and save a true-colour PNG."""
    item = clearest_item(items, month)
    same_day = [it for it in items if it.datetime.date() == item.datetime.date()]
    cube = load_cube(same_day, bands=("red", "green", "blue"), resolution=resolution)
    flags = items_to_frame(same_day).set_index("date")["needs_offset_fix"]
    refl = to_reflectance(cube, flags, bands=("red", "green", "blue")).isel(time=0).compute()
    image = rgb_stretch(refl.red.values, refl.green.values, refl.blue.values)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(image, extent=(float(refl.x.min()), float(refl.x.max()),
                             float(refl.y.min()), float(refl.y.max())))
    ax.set_title(f"Sentinel-2 true colour, {item.datetime:%Y-%m-%d}, Novi Sad (UTM 34N)")
    ax.set_xlabel("Easting [m]")
    ax.set_ylabel("Northing [m]")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Preview saved to {out} ({image.shape[1]} x {image.shape[0]} px)")


def main() -> None:
    args = _parse_args()
    items = list(search_items(datetime=args.datetime, max_cloud_pct=args.max_cloud))
    frame = items_to_frame(items)
    print(f"Found {len(items)} items, {frame['date'].nunique()} unique acquisition dates.\n")
    print("Acquisition dates per month and MGRS tile:")
    print(scenes_per_month(frame).to_string(), "\n")
    print("Processing baselines:", sorted(frame["baseline"].dropna().unique()))
    print("Items needing -1000 offset fix:", int(frame["needs_offset_fix"].sum()))
    if not args.no_preview:
        save_preview(items, args.preview_month, args.preview_resolution, args.out)


if __name__ == "__main__":
    main()
