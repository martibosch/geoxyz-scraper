"""Scraper for XYZ tile-based images.

From https://github.com/andolg/satellite-imagery-downloader
"""

import math
import os
import threading
from collections.abc import Mapping, Sequence
from os import path

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
import requests
import yaml
from pyregeon import RegionMixin, RegionType
from rasterio import transform
from shapely import geometry
from tqdm.auto import tqdm

from geoxyz_scraper import settings

PathType = str | os.PathLike


# Mercator projection
# https://developers.google.com/maps/documentation/javascript/examples/map-coordinates
def _project_with_scale(lat, lon, scale):
    # TODO: DRY with get_tile_gdf
    siny = np.sin(lat * np.pi / 180)
    siny = min(max(siny, -0.9999), 0.9999)
    x = scale * (0.5 + lon / 360)
    y = scale * (0.5 - np.log((1 + siny) / (1 - siny)) / (4 * np.pi))
    return x, y


def _download_tile(url, headers, channels):
    response = requests.get(url, headers=headers)
    arr = np.asarray(bytearray(response.content), dtype=np.uint8)

    if channels == 3:
        return cv2.imdecode(arr, 1)
    return cv2.imdecode(arr, -1)


def download_image(
    west: float,
    south: float,
    east: float,
    north: float,
    zoom: int,
    url: str,
    headers: dict,
    tile_size: int = 256,
    channels: int = 3,
) -> np.ndarray:
    """
    Download a map region as a `numpy.ndarray` with shape depending on `channels`.

    Parameters
    ----------
    west, south, east, north : float
        The bounding box of the region to download, in WGS84 coordinates (EPSG:4326).
    zoom : int
        The zoom level for the tiles.
    url : str
        The URL template for the tiles, e.g.,
        "https://tile.openstreetmap.org/{z}/{x}/{y}.png".
    headers : dict
        Headers to include in the request, e.g., {"User-Agent": "MyApp/1.0"}.
    tile_size : int
        The size of the tiles in pixels (default is 256).
    channels : int
        The number of channels in the image (default is 3 for RGB).

    Returns
    -------
    img_arr : np.ndarray
        Image array.
    """
    # get number of tiles for this zoom level
    n = 2.0**zoom

    # find the pixel coordinates and tile coordinates of the corners
    tl_proj_x, tl_proj_y = _project_with_scale(north, west, n)
    br_proj_x, br_proj_y = _project_with_scale(south, east, n)

    tl_pixel_x = int(tl_proj_x * tile_size)
    tl_pixel_y = int(tl_proj_y * tile_size)
    br_pixel_x = int(br_proj_x * tile_size)
    br_pixel_y = int(br_proj_y * tile_size)

    tl_tile_x = int(tl_proj_x)
    tl_tile_y = int(tl_proj_y)
    br_tile_x = int(br_proj_x)
    br_tile_y = int(br_proj_y)

    img_w = abs(tl_pixel_x - br_pixel_x)
    img_h = br_pixel_y - tl_pixel_y
    img = np.zeros((img_h, img_w, channels), np.uint8)

    def build_row(tile_y):
        for tile_x in range(tl_tile_x, br_tile_x + 1):
            tile = _download_tile(
                url.format(x=tile_x, y=tile_y, z=zoom), headers, channels
            )

            if tile is not None:
                # Find the pixel coordinates of the new tile relative to the image
                tl_rel_x = tile_x * tile_size - tl_pixel_x
                tl_rel_y = tile_y * tile_size - tl_pixel_y
                br_rel_x = tl_rel_x + tile_size
                br_rel_y = tl_rel_y + tile_size

                # Define where the tile will be placed on the image
                img_x_l = max(0, tl_rel_x)
                img_x_r = min(img_w + 1, br_rel_x)
                img_y_l = max(0, tl_rel_y)
                img_y_r = min(img_h + 1, br_rel_y)

                # Define how border tiles will be cropped
                cr_x_l = max(0, -tl_rel_x)
                cr_x_r = tile_size + min(0, img_w - br_rel_x)
                cr_y_l = max(0, -tl_rel_y)
                cr_y_r = tile_size + min(0, img_h - br_rel_y)

                img[img_y_l:img_y_r, img_x_l:img_x_r] = tile[
                    cr_y_l:cr_y_r, cr_x_l:cr_x_r
                ]

    threads = []
    for tile_y in range(tl_tile_y, br_tile_y + 1):
        thread = threading.Thread(target=build_row, args=[tile_y])
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    return img


def get_tile_gdf(region_gdf: RegionType, zoom: int) -> gpd.GeoDataFrame:
    """
    Generate a geo-data frame of XYZ tiles for the region at a given zoom level.

    Parameters
    ----------
    west, south, east, north : float
        The bounding box of the region in WGS84 coordinates (EPSG:4326).
    zoom : int
        The zoom level for the tiles.

    Returns
    -------
    tile_gdf : geopandas.GeoDataFrame
        A geo-data frame containing the XY tile bounds for the specified zoom level
        and their respective geometry as polygons in the WGS84 coordinate system.
    """
    # get region bounds (region is in WGS84)
    west, south, east, north = region_gdf["geometry"].to_crs(epsg=4236).iloc[0].bounds

    # get number of tiles for this zoom level
    n = 2.0**zoom

    # convert bounds to tile coordinates
    def deg2num(lat_deg, lon_deg):
        lat_rad = math.radians(lat_deg)
        xtile = int((lon_deg + 180.0) / 360.0 * n)
        ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        return (xtile, ytile)

    # convert tile bounds to WGS84
    def num2deg(xtile, ytile):
        lon_deg = xtile / n * 360.0 - 180.0
        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
        lat_deg = math.degrees(lat_rad)
        return (lon_deg, lat_deg)

    # convert a west, south, east, north row to a box
    def row_to_box(row):
        """Convert one row of tile bounds to a box (west, south, east, north)."""
        west, north = num2deg(row["west"], row["south"])
        east, south = num2deg(row["east"], row["north"])
        return geometry.box(west, south, east, north)

    # get tile range
    # top-left/north-west
    x_min, y_max = deg2num(north, west)
    # bottom-right/south-east
    x_max, y_min = deg2num(south, east)
    # generate a grid of size using numpy meshgrid
    # TODO: DRY with pyregeon.generate_regular_grid_gser?
    tile_x, tile_y = np.meshgrid(
        np.arange(x_min, x_max + 1),
        np.arange(y_max, y_min + 1),
        indexing="xy",
    )

    flat_tile_x = tile_x.flatten()
    flat_tile_y = tile_y.flatten()
    tile_df = pd.DataFrame(
        {
            "west": flat_tile_x,
            "south": flat_tile_y - 1,
            "east": flat_tile_x + 1,
            "north": flat_tile_y,
        }
    )

    # filter out invalid tiles (i.e., < 0 or >= 2^zoom)
    tile_df = tile_df[
        (tile_df["west"] >= 0)
        | (tile_df["south"] >= 0)
        | (tile_df["east"] < n)
        | (tile_df["north"] < n)
    ]

    # convert to WGS84 geo-series
    tile_gdf = gpd.GeoDataFrame(
        tile_df, geometry=tile_df.apply(row_to_box, axis="columns"), crs="epsg:4326"
    )
    # filter to the region extent
    return tile_gdf[tile_gdf.intersects(region_gdf["geometry"].iloc[0])]


class XYZScraper(RegionMixin):
    """Scraper for XYZ tile-based images.

    Parameters
    ----------
    region : str, Sequence, GeoSeries, GeoDataFrame, PathLike, or IO
        The region to process. This can be either:
        -  A string with a place name (Nominatim query) to geocode.
        -  A sequence with the west, south, east and north bounds. In such a case,a CRS
           must be provided.
        -  A geometric object, e.g., shapely geometry, or a sequence of geometric
           objects (polygon or multi-polygon). In such a case, the value is passed as
           the `data` argument of the GeoSeries constructor, and needs to be in the same
           CRS as the one provided through the `crs` argument.
        -  A geopandas geo-series or geo-data frame.
        -  A filename or URL, a file-like object opened in binary ('rb') mode, or a Path
           object that will be passed to `geopandas.read_file`.
    config: mapping or file-like
        Configuration, as a key-value mapping or as a path to a YAML file. The "url" key
        is required.
    tiling_zoom: int, default 18
        Zoom level for the tiling of the region, not to be confused with the zoom level
        for the image quality.
    """

    def __init__(
        self,
        region: RegionType,
        config: Mapping | PathType,
        *,
        tiling_zoom: int | None = None,
    ):
        """Initialize the XYZScraper with a region."""
        self.region = region
        # ACHTUNG: set the CRS AFTER the region is set, otherwise, if providing a naive
        # geometry, pyregeon will consider that it is in `self.crs`
        # TODO: inspect fix to pyregeon so that the crs of the region (when it is not a
        # naive geometry) is properly considered
        self.crs = "epsg:4326"
        # ensure region is in WGS84
        self.region = self.region.to_crs(self.crs)

        # get the tiling
        if tiling_zoom is None:
            tiling_zoom = settings.tiling_zoom
        self.tile_gdf = get_tile_gdf(self.region, tiling_zoom)

        # get parameters form the config file
        if isinstance(config, PathType):
            with open(config) as src:
                config = yaml.safe_load(src)

        self.url = config.get("url")
        self.headers = config.get("headers", {})
        self.channels = config.get("channels", 3)
        self.tile_size = config.get("tile_size", 256)

    def download_tiles(
        self,
        dst_dir: PathType,
        *,
        tile_ids: Sequence | None = None,
        img_zoom: int | None = None,
        driver: str | None = None,
        dtype: str | None = None,
    ):
        """
        Download tiles.

        Parameters
        ----------
        dst_dir : path-like
            The directory where to save the downloaded tiles.
        tile_ids : list-like or None, default None
            The list of tile IDs (indices of `self.tile_gdf`) to download. If None, all
            tiles will be downloaded.
        img_zoom : int, default None
            The zoom level for the images to download. If None, the value from
            `settings.IMG_ZOOM` will be used.
        driver : str, default None
            The image driver to use for saving the images. If None, the value from
            `settings.IMG_DRIVER` will be used.
        dtype : str, default None
            The data type to use for the images. If None, the value from
            `settings.IMG_DTYPE` will be used.
        """
        if driver is None:
            driver = settings.IMG_DRIVER
        if dtype is None:
            dtype = settings.IMG_DTYPE
        profile = {
            "driver": driver,
            "dtype": dtype,
            "count": self.channels,
            "crs": self.tile_gdf.crs,
        }
        if tile_ids is not None:
            # filter the tiles to download
            tile_gdf = self.tile_gdf.iloc[tile_ids]
        else:
            # use all tiles
            tile_gdf = self.tile_gdf
        for img_filename, tile_geom in tqdm(
            zip(tile_gdf["filename"], tile_gdf["geometry"]),
            total=len(tile_gdf.index),
        ):
            # download the image
            img = download_image(
                *tile_geom.bounds,
                img_zoom,
                self.url,
                self.headers,
                self.tile_size,
                self.channels,
            )
            # convert from BGR (OpenCV default) to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # rearrange dimensions to (channels, height, width)
            img = img.transpose((2, 0, 1))  # HWC to CHW
            # update profile with the image shape and transform
            profile.update(
                {
                    "width": img.shape[2],
                    "height": img.shape[1],
                    "transform": transform.from_bounds(
                        *tile_geom.bounds, img.shape[2], img.shape[1]
                    ),
                }
            )
            with rio.open(path.join(dst_dir, img_filename), "w", **profile) as dst:
                dst.write(img)
