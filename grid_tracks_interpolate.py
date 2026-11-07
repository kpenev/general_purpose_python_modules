"""Define a function for multi-D interpolation of tracks on a grid."""

import logging
from functools import reduce

import numpy

_logger = logging.getLogger(__name__)


def _interpolate(interpolate_to, quantities, grid, data, clip):
    """
    Implement `grid_tracks_interpolate()`.

    If ``clip`` is True, the distance outside the tracks is appended to the
    quantities of each track, so it gets interpolated like them.
    """

    def evaluate_track(track):
        """Interpolate the given track to the specified mass."""

        assert len(interpolate_to) == 1
        name, value = next(iter(interpolate_to.items()))
        if value == "range":
            assert quantities is None
            return track[name][0], track[name][-1]
        excess = max(track[name][0] - value, value - track[name][-1], 0.0)
        if excess > 0 and not clip:
            raise ValueError(
                f"Interpolation only defined for {track[name][0]} <= {name} <= "
                f"{track[name][-1]}. Attempting to evaluate at {name} = "
                f"{value}",
                name,
            )
        # Outside the track numpy.interp returns the end values (the clipping).
        result = tuple(
            numpy.interp(
                value,
                track[name],
                track[q],
            )
            for q in quantities
        )
        return result + (excess,) if clip else result

    track_var = (set(interpolate_to.keys()) - set(g[0] for g in grid)).pop()
    interpolate_to = interpolate_to.copy()
    var_name, var_grid = grid[0]
    target = interpolate_to.pop(var_name)
    if not numpy.isfinite(target):
        raise ValueError(
            f"Requested interpolation to non-finite {var_name} = {target}"
        )
    if target < var_grid[0] or target > var_grid[-1]:
        raise ValueError(
            f"Requested {var_name} ({target}) is outside the available "
            "interpolation range: "
            f"{var_grid[0]} < {var_name} < {var_grid[-1]}",
            var_name,
        )
    above_ind = numpy.searchsorted(var_grid, target)
    data_step = reduce(lambda s, g: s * g[1].size, grid[1:], 1)
    if var_grid[above_ind] == target:
        if len(grid) == 1:
            return evaluate_track(data[above_ind])
        return _interpolate(
            interpolate_to,
            quantities,
            grid[1:],
            data[above_ind * data_step : (above_ind + 1) * data_step],
            clip,
        )
    sub_grid = grid[1:]
    if sub_grid:
        closest_values = tuple(
            _interpolate(
                interpolate_to,
                quantities,
                sub_grid,
                data[ind * data_step : (ind + 1) * data_step],
                clip,
            )
            for ind in [above_ind - 1, above_ind]
        )
    else:
        closest_values = (
            evaluate_track(data[above_ind - 1]),
            evaluate_track(data[above_ind]),
        )
    if interpolate_to[track_var] == "range":
        return (
            max(closest_values[0][0], closest_values[1][0]),
            min(closest_values[0][1], closest_values[1][1]),
        )
    return numpy.array(
        [
            numpy.interp(
                target, var_grid[above_ind - 1 : above_ind + 1], var_values
            )
            for var_values in zip(*closest_values)
        ]
    )


def grid_tracks_interpolate(interpolate_to, quantities, grid, data, clip=False):
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
            tracks. If the value of mass is set to 'range', instead of
            interpolating, the renge over which interpolation is defined is
            returned. In the latter case, quantities should be ``None``.

        quantities([str]):    A list of the quantities for which to find an
            interpolated value.

        grid(((str, []), ...)):    The names of the indepnedent interpolation
            variables and the values of these variables at the grid points.

        data([numpy field array or dict-like]):    The tracks to interpolate.
            Each entry in the list is the track at one of the grid points with
            earlier grid entry changing slower than later ones.

        clip(bool):    If False, a track variable value outside any of the
            tracks used raises ValueError. If True, each track is instead
            evaluated at the value clipped to its range. Not allowed with
            'range'.

    Returns:
        The interpolated quantities (or the range, see ``interpolate_to``). If
        ``clip`` is True, a 2-tuple of those and how far outside the tracks the
        track variable is (interpolated the same way). Both are continuous, and
        the distance is positive exactly where ValueError would be raised
        without ``clip``.
    """

    if not clip:
        return _interpolate(interpolate_to, quantities, grid, data, False)
    assert quantities is not None
    values = _interpolate(interpolate_to, quantities, grid, data, True)
    return values[:-1], values[-1]
