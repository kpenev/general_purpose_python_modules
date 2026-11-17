"""Define a function for multi-D interpolation of tracks on a grid."""

import logging
from functools import reduce

import numpy

_logger = logging.getLogger(__name__)

interpolation_methods = ("linear", "pchip")


def _pchip_evaluate(nodes, values, bracket, target):
    """
    Evaluate the PCHIP interpolant through the given nodes at one point.

    Gives the same result as
    ``scipy.interpolate.PchipInterpolator(nodes, values)(target)``, computed
    directly from the nodes that matter. The piece between two nodes depends
    only on the values at the two nodes on either side of it, so passing those
    (fewer at the ends of the data) gives the same result as passing all.

    Args:
        nodes(array):    Two or more strictly increasing values of the
            independent variable.

        values(array):    The values at the nodes, shape (number of nodes,
            number of quantities).

        bracket(int):    The index of the node starting the interval that
            contains ``target``. Nodes other than ``bracket`` and
            ``bracket + 1`` must be the immediate neighbours of the interval,
            or it must be at the end of the data.

        target(float):    Where to evaluate, between ``nodes[bracket]`` and
            ``nodes[bracket + 1]``.

    Returns:
        array:    The interpolated value of each quantity.
    """

    widths = nodes[1:] - nodes[:-1]
    slopes = (values[1:] - values[:-1]) / widths[:, None]

    def node_derivative(index):
        """Return the PCHIP derivative at the given node as SciPy does."""

        if nodes.size == 2:
            return slopes[0]
        if 0 < index < nodes.size - 1:
            # Weighted harmonic mean of the adjacent slopes (Fritsch-Butland),
            # zero where they differ in sign or either is zero.
            previous_slope, next_slope = slopes[index - 1], slopes[index]
            previous_weight = 2.0 * widths[index] + widths[index - 1]
            next_weight = widths[index] + 2.0 * widths[index - 1]
            with numpy.errstate(divide="ignore", invalid="ignore"):
                derivative = 1.0 / (
                    (
                        previous_weight / previous_slope
                        + next_weight / next_slope
                    )
                    / (previous_weight + next_weight)
                )
            return numpy.where(
                (numpy.sign(previous_slope) != numpy.sign(next_slope))
                | (previous_slope == 0)
                | (next_slope == 0),
                0.0,
                derivative,
            )
        # One-sided three-point estimate at an end, limited to preserve shape.
        if index == 0:
            near_width, far_width = widths[0], widths[1]
            near_slope, far_slope = slopes[0], slopes[1]
        else:
            near_width, far_width = widths[-1], widths[-2]
            near_slope, far_slope = slopes[-1], slopes[-2]
        derivative = (
            (2.0 * near_width + far_width) * near_slope - near_width * far_slope
        ) / (near_width + far_width)
        derivative = numpy.where(
            numpy.sign(derivative) != numpy.sign(near_slope), 0.0, derivative
        )
        return numpy.where(
            (numpy.sign(near_slope) != numpy.sign(far_slope))
            & (numpy.abs(derivative) > 3.0 * numpy.abs(near_slope)),
            3.0 * near_slope,
            derivative,
        )

    left_derivative = node_derivative(bracket)
    right_derivative = node_derivative(bracket + 1)
    width = widths[bracket]
    slope = slopes[bracket]
    cubic = (left_derivative + right_derivative - 2.0 * slope) / width
    quadratic = (slope - left_derivative) / width - cubic
    cubic = cubic / width
    offset = target - nodes[bracket]
    return values[bracket] + offset * (
        left_derivative + offset * (quadratic + offset * cubic)
    )


def _evaluate_track_pchip(track, name, value, quantities):
    """
    PCHIP interpolation of the given quantities along a track.

    Repeated values of the track variable (jumps in the tabulated quantities,
    e.g. between evolutionary phases) are breaks: no interval reaches across
    one. Within the track's range this gives the same result as SciPy's
    PchipInterpolator applied to each unbroken piece. Exactly at a repeated
    value the result is that of either side; which one is unspecified.

    Args:
        track(numpy field array or dict-like):    The track.

        name(str):    The track variable.

        value(float):    Where to evaluate, within the track's range.

        quantities([str]):    The quantities to interpolate.

    Returns:
        tuple:    The interpolated value of each quantity.
    """

    track_values = track[name]
    if value >= track_values[-1]:
        return tuple(track[quantity][-1] for quantity in quantities)
    bracket = numpy.searchsorted(track_values, value, side="right") - 1
    first = bracket
    if bracket > 0 and track_values[bracket - 1] < track_values[bracket]:
        first = bracket - 1
    last = bracket + 1
    if (
        bracket + 2 < len(track_values)
        and track_values[bracket + 2] > track_values[bracket + 1]
    ):
        last = bracket + 2
    return tuple(
        _pchip_evaluate(
            track_values[first : last + 1],
            numpy.array(
                [track[quantity][first : last + 1] for quantity in quantities]
            ).T,
            bracket - first,
            value,
        )
    )


def _interpolate(
    interpolate_to, quantities, grid, data, *, clip, method, strict=True
):
    """
    Implement `grid_tracks_interpolate()`.

    If ``clip`` is True, the distance outside the tracks is appended to the
    quantities of each track, so it gets interpolated like them.

    If ``strict`` is False, the tracks are evaluated at the track variable
    clipped to their range even if ``clip`` is False, instead of raising. PCHIP
    uses this for the grid nodes beyond the two bracketing the target: they
    only shape the interpolant between those two, so they do not limit where
    interpolation is defined.
    """

    def evaluate_track(track, track_strict):
        """Interpolate the given track to the specified mass."""

        assert len(interpolate_to) == 1
        name, value = next(iter(interpolate_to.items()))
        if value == "range":
            assert quantities is None
            return track[name][0], track[name][-1]
        excess = max(track[name][0] - value, value - track[name][-1], 0.0)
        if excess > 0 and track_strict and not clip:
            raise ValueError(
                f"Interpolation only defined for {track[name][0]} <= {name} <= "
                f"{track[name][-1]}. Attempting to evaluate at {name} = "
                f"{value}",
                name,
            )
        if method == "linear":
            # Outside the track numpy.interp returns the end values (the
            # clipping).
            result = tuple(
                numpy.interp(
                    value,
                    track[name],
                    track[q],
                )
                for q in quantities
            )
        else:
            result = _evaluate_track_pchip(
                track,
                name,
                min(max(value, track[name][0]), track[name][-1]),
                quantities,
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
    sub_grid = grid[1:]

    def evaluate_node(ind, node_strict):
        """Interpolate everything inside the given node of this variable."""

        if not sub_grid:
            return evaluate_track(data[ind], node_strict)
        return _interpolate(
            interpolate_to,
            quantities,
            sub_grid,
            data[ind * data_step : (ind + 1) * data_step],
            clip=clip,
            method=method,
            strict=node_strict,
        )

    if var_grid[above_ind] == target:
        return evaluate_node(above_ind, strict)
    if method == "linear" or interpolate_to[track_var] == "range":
        closest_values = (
            evaluate_node(above_ind - 1, strict),
            evaluate_node(above_ind, strict),
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
    # PCHIP: up to two nodes on each side of the target.
    first = max(above_ind - 2, 0)
    stop = min(above_ind + 2, var_grid.size)
    return _pchip_evaluate(
        var_grid[first:stop],
        numpy.array(
            [
                evaluate_node(ind, strict and above_ind - 1 <= ind <= above_ind)
                for ind in range(first, stop)
            ]
        ),
        above_ind - 1 - first,
        target,
    )


def grid_tracks_interpolate(
    interpolate_to, quantities, grid, data, clip=False, method="linear"
):
    """
    Multi-D interpolation of tracks of varying length on a grid.

    Performs interpolation for data consisting of potentially variable
    length tracks (list of quantities vs independent variable) available on a
    strict grid of other variables. For example, CMD data consists of stellar
    properties vs initial stellar mass on a grid of [Fe/H] and age.

    The grid variables are interpolated one at a time, slowest changing first:
    for each node used, everything inside it is interpolated down to the
    tracks, and then the results are interpolated between the nodes.

    With ``method="linear"`` every step is linear between the two nodes
    bracketing the target (and between the two track points bracketing the
    track variable).

    With ``method="pchip"`` every step is a piecewise cubic Hermite
    interpolation (PCHIP, as in SciPy's PchipInterpolator), using up to two
    nodes on each side of the target (fewer at the ends of the grid or the
    tracks). Each step stays between the values at the two bracketing nodes,
    so there is no overshoot, and the result has a continuous first
    derivative, except where the values turn around at a node (there the
    derivative is set to zero, so they turn around at the node). Along the
    tracks, repeated values of the track variable are treated as breaks (the
    tabulated quantities jump there, as linear interpolation also gives).
    The nodes beyond the bracketing two use their tracks clipped to their range
    when needed, so where interpolation is defined, the clipping and the
    'range' are determined by the bracketing nodes alone, as for linear.

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

        method(str):    How to interpolate: "linear" or "pchip" (see above).

    Returns:
        The interpolated quantities (or the range, see ``interpolate_to``). If
        ``clip`` is True, a 2-tuple of those and how far outside the tracks the
        track variable is (interpolated the same way). Both are continuous, and
        the distance is positive exactly where ValueError would be raised
        without ``clip``.
    """

    if method not in interpolation_methods:
        raise ValueError(
            f"Unknown interpolation method {method!r}, expected one of "
            f"{interpolation_methods}"
        )
    if not clip:
        return _interpolate(
            interpolate_to, quantities, grid, data, clip=False, method=method
        )
    assert quantities is not None
    values = _interpolate(
        interpolate_to, quantities, grid, data, clip=True, method=method
    )
    return values[:-1], values[-1]
