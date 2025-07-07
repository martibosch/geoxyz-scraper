"""Get urban tiles."""

import geopandas as gpd
import numpy as np
import osmnx as ox

from geoxyz_scraper import settings


def get_urban_tiles(
    tile_gser: gpd.GeoSeries,
    *,
    min_bldg_area: float | None = None,
    min_road_length: float | None = None,
    min_road_intersections: float | None = None,
    tile_id_col: str = "tile_id",
) -> np.ndarray:
    """Get urban tiles based on OSM building area, road length, and road intersections.

    Parameters
    ----------
    tile_gser : geopandas.GeoSeries
        Tile geometries, as a pandas geo-series
    min_bldg_area : float, optional
        Minimum building area in square meters to consider a tile urban. If no value is
        provided, the default from `settings.MIN_BLDG_AREA` is used.
    min_road_length : float, optional
        Minimum road length in meters to consider a tile urban. If no value is
        provided, the default from `settings.MIN_ROAD_LENGTH` is used.
    min_road_intersections : int, optional
        Minimum number of road intersections to consider a tile urban. If no value is
        provided, the default from `settings.MIN_ROAD_INTERSECTIONS` is used.
    tile_id_col : str, optional
        Name of the column in `tile_gser` that contains the tile IDs, by default
        "tile_id"

    Returns
    -------
    urban_tiles : np.ndarray
        Array of tile IDs that are considered urban based on the criteria.
    """
    # process parameter arguments
    if min_bldg_area is None:
        min_bldg_area = settings.MIN_BLDG_AREA
    if min_road_length is None:
        min_road_length = settings.MIN_ROAD_LENGTH
    if min_road_intersections is None:
        min_road_intersections = settings.MIN_ROAD_INTERSECTIONS

    tile_gser = tile_gser.rename_axis(index=tile_id_col).reset_index()
    # get spatial extent
    extent_geom = tile_gser.to_crs(ox.settings.default_crs).union_all()

    # buildings
    bldg_gdf = ox.features_from_polygon(extent_geom, tags={"building": True})

    tile_bldg_gdf = tile_gser.overlay(
        bldg_gdf.loc[["way", "relation"]].to_crs(tile_gser.crs)
    )
    bldg_area_ser = (
        tile_bldg_gdf.assign(**{"area": ox.projection.project_gdf(tile_bldg_gdf).area})
        .groupby(tile_id_col)["area"]
        .sum()
        .fillna(0)
    )

    # roads
    # get road length and intersections per grid cell
    nodes_gdf, edges_gdf = ox.graph_to_gdfs(
        ox.convert.to_undirected(ox.graph_from_polygon(extent_geom, retain_all=True))
    )

    # road length
    tile_edges_gdf = tile_gser.overlay(
        edges_gdf.to_crs(tile_gser.crs), keep_geom_type=False
    )
    road_length_ser = (
        tile_edges_gdf.assign(
            **{"length": ox.projection.project_gdf(tile_edges_gdf).length}
        )
        .groupby(tile_id_col)["length"]
        .sum()
        .fillna(0)
    )

    # road intersections
    tile_nodes_gdf = tile_gser.sjoin(
        nodes_gdf.to_crs(tile_gser.crs),
        how="left",
        predicate="contains",
    )
    node_count_ser = tile_nodes_gdf[tile_id_col].value_counts(sort=False)

    return (
        tile_gser.set_index(tile_id_col)
        .index[
            (bldg_area_ser > min_bldg_area)
            | (road_length_ser > min_road_length)
            | (node_count_ser > min_road_intersections)
        ]
        .values
    )
