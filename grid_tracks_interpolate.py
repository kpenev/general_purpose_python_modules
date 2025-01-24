"""Define a function for multi-D interpolation of tracks on a grid."""

from functools import reduce

import numpy


def grid_tracks_interpolate(interpolate_to, quantities, grid, data):
    """
    Multi-D linear interpolation of tracks of varying length on a grid.

    Performs linear interpolation for data consisting of potentially variable
    length tracks (list of quantities vs independent variable) available on a
    strict grid of other variables. For example, CMD data consists of stellar
    properties vs initial stellar mass on a grid of [Fe/H] and age.

    Args:
        interpolate_to(dict):    The values of the independent variables at
            which the interpolated value is desired. Keys should be either in
            ``grid`` (see below) or the name of the indepnedent variable of the
            tracks.

        quantities([str]):    A list of the quantities for which to find an
            interpolated value.

        grid(((str, []), ...)):    The names of the indepnedent interpolation
            variables and the values of these variables at the grid points.

        data([numpy field array or dict-like]):    The tracks to interpolate.
            Each entry in the list is the track at one of the grid points with
            earlier grid entry changing slower than later ones.
    """

    def evaluate_track(track):
        """Interpolate the given track to the specified mass."""

        assert len(interpolate_to) == 1
        name, value = next(iter(interpolate_to.items()))
        return tuple(
            numpy.interp(
                value,
                track[name],
                track[q],
            )
            for q in quantities
        )

    interpolate_to = interpolate_to.copy()
    var_name, var_grid = grid[0]
    target = interpolate_to.pop(var_name)
    if target < var_grid[0] or target > var_grid[-1]:
        raise ValueError(
            f"Requested {var_name} ({target}) is outside the available "
            "interpolation range: "
            f"{var_grid[0]} < {var_name} < {var_grid[-1]}"
        )
    above_ind = numpy.searchsorted(var_grid, target)
    data_step = reduce(lambda s, g: s * g[1].size, grid[1:], 1)
    if var_grid[above_ind] == target:
        if len(grid) == 1:
            return evaluate_track(data[above_ind])
        return grid_tracks_interpolate(
            interpolate_to,
            quantities,
            grid[1:],
            data[above_ind * data_step : (above_ind + 1) * data_step],
        )
    sub_grid = grid[1:]
    if sub_grid:
        closest_values = tuple(
            grid_tracks_interpolate(
                interpolate_to,
                quantities,
                sub_grid,
                data[ind * data_step : (ind + 1) * data_step],
            )
            for ind in [above_ind - 1, above_ind]
        )
    else:
        closest_values = (
            evaluate_track(data[above_ind - 1]),
            evaluate_track(data[above_ind]),
        )
    return tuple(
        numpy.interp(
            target, var_grid[above_ind - 1 : above_ind + 1], var_values
        )
        for var_values in zip(*closest_values)
    )