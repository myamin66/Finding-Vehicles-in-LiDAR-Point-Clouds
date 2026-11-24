"""
point_processing.py

Steps 1 and 2 of the pipeline:
  1. keep the area around our car, remove our own car, downsample
  2. estimate the local road height and split the points into
     ground / not ground
"""

import numpy as np
from scipy.ndimage import generic_filter, minimum_filter

import settings


def crop_area(points):
    """Keep points inside the square around us and outside our own car."""
    x, y = points[:, 0], points[:, 1]
    # Both masks are True/False per point; the result keeps the points
    # that are in the square AND not on our own car.
    half = settings.AREA_HALF_SIZE
    inside_area = (np.abs(x) < half) & (np.abs(y) < half)

    on_own_car = (
        (x > settings.EGO_BOX_X[0]) & (x < settings.EGO_BOX_X[1])
        & (y > settings.EGO_BOX_Y[0]) & (y < settings.EGO_BOX_Y[1])
    )
    return points[inside_area & ~on_own_car]


def voxel_downsample(points, voxel_size):
    """
    Cut space into cubes of `voxel_size` and replace all points inside one
    cube by their average. Returns fewer points with the same columns.
    """
    cube_index = np.floor(points[:, :3] / voxel_size).astype(np.int64)
    # Turn the 3 cube indices into one number per cube (like a house
    # number), which is much faster to sort than rows of 3 numbers.
    cube_index -= cube_index.min(axis=0)
    size = cube_index.max(axis=0) + 1
    cube_key = (cube_index[:, 0] * size[1] + cube_index[:, 1]) * size[2] + cube_index[:, 2]
    # Give every occupied cube a number; `cube_of_point` says which cube
    # each point belongs to.
    _, cube_of_point = np.unique(cube_key, return_inverse=True)
    cube_of_point = cube_of_point.ravel()
    n_cubes = cube_of_point.max() + 1

    # Add up all points of each cube and divide by how many there were.
    # This averages x, y, z and the intensity column at the same time.
    sums = np.zeros((n_cubes, points.shape[1]))
    np.add.at(sums, cube_of_point, points)
    counts = np.bincount(cube_of_point, minlength=n_cubes)
    return sums / counts[:, None]


def ground_height_map(points):
    """
    Estimates the road height on a grid of GROUND_CELL_SIZE cells:
      1. lowest point in every cell
      2. a cell much lower than its neighbours is a stray point: use the
         neighbours' median instead
      3. lowest value of each cell and its 8 neighbours (a cell fully
         covered by a car has no road points, its neighbours usually do)
      4. cells without any points get the median ground height
    Returns the grid (2D array of heights).
    """
    cell, n_cells = settings.GROUND_CELL_SIZE, grid_size()
    rows, columns = cell_index(points)

    # np.minimum.at goes through all points and keeps the smallest z of
    # each cell. Cells without any points stay at infinity.
    lowest = np.full((n_cells, n_cells), np.inf)
    valid = points[:, 2] > settings.LOWEST_VALID_Z
    np.minimum.at(lowest, (rows[valid], columns[valid]), points[valid, 2])

    # A single stray point below the road (a reflection) would become the
    # ground for 9 cells in the next step and turn the real road there into
    # "objects". So a cell that is much lower than the median of its
    # neighbours gets that median instead.
    with_gaps = np.where(np.isfinite(lowest), lowest, np.nan)
    neighbour_median = generic_filter(with_gaps, median_of_known, size=3, mode="nearest")
    too_low = lowest < neighbour_median - settings.MAX_DIP_BELOW_NEIGHBOURS
    lowest[too_low] = neighbour_median[too_low]

    # Every cell takes the lowest value of itself and its 8 neighbours.
    ground = minimum_filter(lowest, size=3, mode="nearest")
    # Cells that still have no value (nothing seen there, e.g. behind a
    # wall) get the typical ground height of this scan.
    known = np.isfinite(ground)
    ground[~known] = np.median(ground[known]) if known.any() else 0.0
    return ground


def median_of_known(values):
    """Median of the cells that have points (NaN = empty cell)."""
    known = values[~np.isnan(values)]
    return np.median(known) if len(known) else np.nan


def grid_size():
    return int(np.ceil(2 * settings.AREA_HALF_SIZE / settings.GROUND_CELL_SIZE))


def cell_index(points):
    """Grid row and column of every point."""
    shifted = (points[:, :2] + settings.AREA_HALF_SIZE) / settings.GROUND_CELL_SIZE
    index = np.clip(np.floor(shifted).astype(int), 0, grid_size() - 1)
    return index[:, 0], index[:, 1]


def split_ground(points):
    """
    Returns (is_ground, height, ground_map) for the points. `height` is the
    height above the local ground and is used again later (vehicle height).
    """
    ground_map = ground_height_map(points)
    # Height of each point above the ground of the cell it falls in.
    rows, columns = cell_index(points)
    height = points[:, 2] - ground_map[rows, columns]
    return height < settings.GROUND_MARGIN, height, ground_map
