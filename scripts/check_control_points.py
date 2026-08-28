#!/usr/bin/env python3

"""
FIRIS - Control Point Diagnostic

This script DOES NOT modify FLI.

Purpose:
1. Read Fars Province boundary.
2. Extract representative boundary control points.
3. Convert geographic coordinates to FLI row/column
   using the current raster transform.
4. Report the current spatial relationship.

The result is diagnostic only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import rasterio
from rasterio.transform import rowcol


def parse_args():

    parser = argparse.ArgumentParser(
        description="Generate FIRIS boundary control points"
    )

    parser.add_argument(
        "--raster",
        required=True,
        type=Path
    )

    parser.add_argument(
        "--boundary",
        required=True,
        type=Path
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path
    )

    return parser.parse_args()


def collect_coordinates(geometry):

    coordinates = []

    if not geometry:
        return coordinates

    geometry_type = geometry.get(
        "type"
    )

    coords = geometry.get(
        "coordinates"
    )

    def walk(value):

        if (
            isinstance(value, list)
            and len(value) >= 2
            and isinstance(value[0], (int, float))
            and isinstance(value[1], (int, float))
        ):

            coordinates.append(
                (
                    float(value[0]),
                    float(value[1])
                )
            )

            return

        if isinstance(value, list):

            for item in value:
                walk(item)

    walk(coords)

    return coordinates


def load_boundary_points(
    boundary_path
):

    with boundary_path.open(
        "r",
        encoding="utf-8"
    ) as file:

        geojson = json.load(file)

    features = geojson.get(
        "features",
        []
    )

    points = []

    for feature in features:

        geometry = feature.get(
            "geometry"
        )

        points.extend(
            collect_coordinates(
                geometry
            )
        )

    if not points:

        raise RuntimeError(
            "No boundary coordinates found."
        )

    return points


def select_control_points(points):

    # --------------------------------------------------------
    # Find geographic extremes.
    # --------------------------------------------------------

    west = min(
        points,
        key=lambda p: p[0]
    )

    east = max(
        points,
        key=lambda p: p[0]
    )

    south = min(
        points,
        key=lambda p: p[1]
    )

    north = max(
        points,
        key=lambda p: p[1]
    )

    # --------------------------------------------------------
    # Select additional representative points.
    #
    # This avoids relying only on the four extremes.
    # --------------------------------------------------------

    sorted_by_x = sorted(
        points,
        key=lambda p: p[0]
    )

    sorted_by_y = sorted(
        points,
        key=lambda p: p[1]
    )

    west25 = sorted_by_x[
        len(sorted_by_x) // 4
    ]

    east25 = sorted_by_x[
        3 * len(sorted_by_x) // 4
    ]

    south25 = sorted_by_y[
        len(sorted_by_y) // 4
    ]

    north25 = sorted_by_y[
        3 * len(sorted_by_y) // 4
    ]

    selected = {

        "WEST_EXTREME": west,

        "EAST_EXTREME": east,

        "SOUTH_EXTREME": south,

        "NORTH_EXTREME": north,

        "WEST_INTERMEDIATE": west25,

        "EAST_INTERMEDIATE": east25,

        "SOUTH_INTERMEDIATE": south25,

        "NORTH_INTERMEDIATE": north25
    }

    return selected


def main():

    args = parse_args()

    raster_path = args.raster

    boundary_path = args.boundary

    output_path = args.output

    if not raster_path.is_file():

        raise FileNotFoundError(
            f"Raster not found: {raster_path}"
        )

    if not boundary_path.is_file():

        raise FileNotFoundError(
            f"Boundary not found: {boundary_path}"
        )

    points = load_boundary_points(
        boundary_path
    )

    control_points = select_control_points(
        points
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Raster geometry
    # --------------------------------------------------------

    with rasterio.open(
        raster_path
    ) as src:

        print("")
        print("=" * 70)
        print("FIRIS CONTROL POINT DIAGNOSTIC")
        print("=" * 70)

        print("")
        print("RASTER")
        print("------")

        print(
            f"CRS       : {src.crs}"
        )

        print(
            f"Size      : "
            f"{src.width} x {src.height}"
        )

        print(
            f"Resolution: {src.res}"
        )

        print(
            f"Bounds    : {src.bounds}"
        )

        print("")
        print("CONTROL POINTS")
        print("--------------")

        results = []

        for name, point in control_points.items():

            lon = point[0]
            lat = point[1]

            row, col = rowcol(
                src.transform,
                lon,
                lat
            )

            x, y = src.transform * (
                col + 0.5,
                row + 0.5
            )

            inside = (
                0 <= row < src.height
                and
                0 <= col < src.width
            )

            item = {

                "name":
                    name,

                "longitude":
                    lon,

                "latitude":
                    lat,

                "row":
                    int(row),

                "col":
                    int(col),

                "pixel_center_longitude":
                    float(x),

                "pixel_center_latitude":
                    float(y),

                "inside_raster":
                    bool(inside)
            }

            results.append(
                item
            )

            print("")
            print(
                f"{name}"
            )

            print(
                f"  Lon/Lat : "
                f"{lon:.8f}, {lat:.8f}"
            )

            print(
                f"  Row/Col : "
                f"{row}, {col}"
            )

            print(
                f"  Raster XY: "
                f"{x:.8f}, {y:.8f}"
            )

            print(
                f"  Inside  : {inside}"
            )

        # ----------------------------------------------------
        # Boundary extent
        # ----------------------------------------------------

        boundary_min_x = min(
            p[0]
            for p in points
        )

        boundary_max_x = max(
            p[0]
            for p in points
        )

        boundary_min_y = min(
            p[1]
            for p in points
        )

        boundary_max_y = max(
            p[1]
            for p in points
        )

        print("")
        print("BOUNDARY EXTENT")
        print("---------------")

        print(
            f"West : {boundary_min_x:.8f}"
        )

        print(
            f"East : {boundary_max_x:.8f}"
        )

        print(
            f"South: {boundary_min_y:.8f}"
        )

        print(
            f"North: {boundary_max_y:.8f}"
        )

        print("")
        print("RASTER EXTENT")

        print(
            f"West : {src.bounds.left:.8f}"
        )

        print(
            f"East : {src.bounds.right:.8f}"
        )

        print(
            f"South: {src.bounds.bottom:.8f}"
        )

        print(
            f"North: {src.bounds.top:.8f}"
        )

        report = {

            "raster": str(
                raster_path
            ),

            "boundary": str(
                boundary_path
            ),

            "crs": str(
                src.crs
            ),

            "raster_bounds": {

                "west":
                    float(src.bounds.left),

                "east":
                    float(src.bounds.right),

                "south":
                    float(src.bounds.bottom),

                "north":
                    float(src.bounds.top)
            },

            "boundary_bounds": {

                "west":
                    boundary_min_x,

                "east":
                    boundary_max_x,

                "south":
                    boundary_min_y,

                "north":
                    boundary_max_y
            },

            "control_points":
                results
        }

    output_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print("")
    print("=" * 70)

    print(
        f"CONTROL POINT REPORT: {output_path}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
