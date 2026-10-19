# AgroSat Vojvodina

**AI crop mapping and drought monitoring from Sentinel-2 imagery, with an LLM agent you can ask questions in English.**

> Status: 🚧 Week 1 of 3 (data pipeline). Public demo planned for **9 November 2026**.

AgroSat maps **maize**, **winter cereals** and **other** land at 10 m resolution around Novi Sad, Serbia, using a multi-temporal Sentinel-2 stack. It also flags drought stress as a monthly NDVI anomaly. An LLM agent wraps the model as a tool, so you can ask *"How much maize was there north of Novi Sad in 2025, and is it under drought stress?"* and get an answer with a map.

## Architecture

```
Sentinel-2 L2A (Earth Search STAC)
  → Preprocessing (SCL cloud mask, monthly composites, UTM 34N, NDVI)
  → Model (LightGBM baseline → Prithvi-EO-2.0 fine-tune)   [+ ESA WorldCereal 2021 labels]
  → FastAPI service (prediction per region, NDVI anomaly)
  → LLM agent + web UI (Gradio on Hugging Face Spaces)
```

## Quickstart

```bash
git clone https://github.com/<your-username>/agro-sat.git && cd agro-sat
uv venv --python 3.11 && source .venv/bin/activate && uv pip install -e ".[dev,viz]"
agrosat-inventory            # lists 2021 scenes and saves docs/img/july_2021_rgb.png
```

## Area and grid

| Item | Value |
| --- | --- |
| AOI (lon/lat, WGS84) | `19.60, 45.15, 20.10, 45.45`, about 40 × 33 km around Novi Sad |
| Target CRS | UTM zone 34N, `EPSG:32634` |
| Resolution | 10 m |
| Season | March–October 2021 (labels year), then 2025/2026 for inference |
| Labels | ESA WorldCereal 2021 (maize, winter cereals), reprojected with nearest neighbour |

## Geospatial design decisions

**Why UTM 34N?** The AOI lies in the middle of UTM zone 34 (18°E–24°E), the native CRS of the Sentinel-2 MGRS tiles covering it. Loading in the native projection means the 10 m bands are never reprojected, so no resampling blur and no area distortion when computing hectares.

**Grid alignment.** The output grid is snapped to multiples of 10 m. Sentinel-2 tile origins are also multiples of 10 m, so every output pixel coincides with a native pixel (verified in `tests/test_data.py`).

**Resampling.** The 20 m SWIR band is upsampled bilinearly. The Scene Classification Layer (SCL) is categorical, so it uses **nearest neighbour**; bilinear interpolation would invent non-existent classes on boundaries between cloud and vegetation.

**Tile mosaicking.** The AOI spans more than one MGRS tile. Loading with `groupby="solar_day"` merges tiles from the same overpass into one time step.

**Scene cloud filter vs. pixel mask.** `eo:cloud_cover < 30` filters on the whole 110 × 110 km tile, not on the AOI. It is only a coarse pre-filter; pixel-level masking with SCL classes 0, 1, 3, 8, 9 and 10 follows.

### ⚠️ The Sentinel-2 reflectance offset (2022 onward)

From processing baseline **04.00** (25 January 2022) ESA stores L2A values as `DN = reflectance × 10000 + 1000`. Mixing 2021 and 2025 data without removing that offset shifts every band by +0.1 reflectance and biases NDVI, which silently breaks a model trained on 2021 and applied to 2025.

`agrosat.data.needs_offset_correction()` corrects an item only if its baseline is ≥ 04.00 **and** the provider has not already removed the offset (`earthsearch:boa_offset_applied`). `to_reflectance()` then converts all dates to 0–1 reflectance on the same scale.

## Repository layout

```
src/agrosat/   config.py (AOI, grid, bands) · data.py (search, inventory, load, reflectance) · cli.py
notebooks/     01_data.ipynb: inventory, July true-colour preview
tests/         offline tests on synthetic STAC items and GeoTIFFs (no network needed)
docs/img/      figures for this README
data/          local rasters and zarr stores (git-ignored)
```

## Roadmap

- [x] Week 1: STAC search, inventory, offset handling, lazy datacube
- [ ] Week 1: SCL cloud mask, monthly composites, WorldCereal labels, spatial split, LightGBM baseline
- [ ] Week 2: Prithvi-EO-2.0 fine-tune vs. baseline (target: +5 mIoU on a spatially separate test region)
- [ ] Week 3: FastAPI, LLM agent, Gradio demo on Hugging Face Spaces

## Data and credits

Contains modified Copernicus Sentinel data (2021–2026), accessed via [Element 84 Earth Search](https://element84.com/earth-search/). Labels: [ESA WorldCereal](https://esa-worldcereal.org/). Model: [Prithvi-EO-2.0](https://huggingface.co/ibm-nasa-geospatial) by IBM and NASA.

## License

MIT © 2026 Bojana Bojić
