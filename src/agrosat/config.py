"""Project-wide constants: area of interest, grid definition and data source.

Keeping these in one place guarantees that every step (imagery, labels,
predictions) lands on exactly the same pixel grid.
"""

from typing import Final

# Area of interest around Novi Sad, (min_lon, min_lat, max_lon, max_lat) in WGS84.
AOI_BBOX: Final[tuple[float, float, float, float]] = (19.60, 45.15, 20.10, 45.45)

# Target grid. The AOI sits in the middle of UTM zone 34 (18°E–24°E), which is also
# the native CRS of the Sentinel-2 MGRS tiles covering it. Loading in EPSG:32634 at
# 10 m therefore avoids any reprojection of the 10 m bands.
TARGET_CRS: Final[str] = "EPSG:32634"
RESOLUTION_M: Final[int] = 10

# Data source: Element 84 Earth Search (Sentinel-2 L2A as Cloud-Optimized GeoTIFFs on AWS).
STAC_URL: Final[str] = "https://earth-search.aws.element84.com/v1"
COLLECTION: Final[str] = "sentinel-2-l2a"

# Training season for the 2021 WorldCereal labels.
SEASON_2021: Final[str] = "2021-03-01/2021-10-31"
MAX_SCENE_CLOUD_PCT: Final[float] = 30.0

# Earth Search asset names. B02, B03, B04, B08 are 10 m; B11 (swir16) is 20 m; SCL is 20 m.
REFLECTANCE_BANDS: Final[tuple[str, ...]] = ("blue", "green", "red", "nir", "swir16")
SCL_BAND: Final[str] = "scl"

# Scene Classification Layer classes treated as invalid:
# 0 no data, 1 saturated/defective, 3 cloud shadow, 8 cloud medium prob.,
# 9 cloud high prob., 10 thin cirrus.
SCL_INVALID_CLASSES: Final[tuple[int, ...]] = (0, 1, 3, 8, 9, 10)

# Processing baseline 04.00 (from 25 Jan 2022) adds BOA_ADD_OFFSET = -1000 to L2A digital numbers.
OFFSET_BASELINE: Final[float] = 4.0
BOA_ADD_OFFSET: Final[int] = -1000
