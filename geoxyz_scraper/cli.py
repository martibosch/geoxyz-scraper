"""Command line interface (CLI)."""

import fire
import geopandas as gpd

from geoxyz_scraper import scraper, urban


def cli(
    region_filepath: str,
    config_filepath: str,
    dst_dir: str,
    *,
    img_zoom: int | None = None,
    tiling_zoom: int | None = None,
    driver: str | None = None,
    dtype: str | None = None,
    filter_urban: bool = True,
    min_bldg_area: float | None = None,
    min_road_length: float | None = None,
    min_road_intersections: float | None = None,
):
    """Command line interface (CLI) for geoxyz-scraper."""
    # if region_filepath is not None:
    # TODO: support list-like bounds
    region = gpd.read_file(region_filepath)

    s = scraper.XYZScraper(region, config_filepath, tiling_zoom=tiling_zoom)
    if filter_urban:
        tile_ids = urban.get_urban_tiles(
            s.tile_gdf["geometry"],
            min_bldg_area=min_bldg_area,
            min_road_length=min_road_length,
            min_road_intersections=min_road_intersections,
        )
    else:
        tile_ids = None

    print(f"Downloading images to: {dst_dir}")
    s.download_tiles(
        dst_dir, tile_ids=tile_ids, img_zoom=img_zoom, driver=driver, dtype=dtype
    )


def main():
    """Entrypoint."""
    fire.Fire(cli)
