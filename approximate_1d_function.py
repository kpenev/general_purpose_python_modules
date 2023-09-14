"""Automatically tuned approximation of a scalar function of 1 argument."""

import logging

from matplotlib import pyplot
import numpy
from scipy.interpolate import InterpolatedUnivariateSpline

_logger = logging.getLogger(__name__)

def select_mismatches(calculated_values,
                      interpolated_values,
                      tolerance,
                      grid_refine_limit):
    """Return the grid indices to refine."""

    residuals = numpy.absolute(calculated_values - interpolated_values)
    _logger.debug('Max residual: %s', repr(residuals.max()))

    if grid_refine_limit:
        tolerance = max(
            tolerance,
            numpy.mean(
                numpy.partition(
                    residuals.flatten(),
                    (
                        -grid_refine_limit - 1,
                        -grid_refine_limit
                    )
                )[
                    -grid_refine_limit - 1
                    :
                    (-grid_refine_limit + 1) or residuals.size
                ]
            )
        )

    result = numpy.nonzero(residuals > tolerance)

    _logger.debug('Selected %d cells to refine', result[0].size)

    return result


def get_new_grid_points(mismatch_indices, current_grid, min_grid_step):
    """
    Find the new values to add per the given mismatch indices.

    Args:
        mismatch_indices(numpy.ndarray):    The indices within the current grid
            that need to be refined.

        current_grid(numpy.ndarray):    The current grid.

        min_grid_step(float):   The minimum allowed step between grid points.

    Returns:
        numpy.ndarray(float):
            The new grid points to add.

        numpy.ndarray(int):
            The number of current grid points that are smaller than each of the
            new grid points.
    """

    if mismatch_indices.size == 0:
        return None

    below_indices = numpy.unique(
        numpy.concatenate((
            (
                mismatch_indices
                if mismatch_indices[-1] < current_grid.size - 1 else
                mismatch_indices[:-1]
            ),
            (
                mismatch_indices
                if mismatch_indices[0] > 0 else
                mismatch_indices[1:]
            ) - 1
        ))
    )
    valid = (
        (current_grid[below_indices + 1] - current_grid[below_indices])
        >
        2.0 * min_grid_step
    )
    below_indices = below_indices[valid]
    return (
        0.5 * (current_grid[below_indices]
               +
               current_grid[below_indices + 1]),
        below_indices + 1
    )


def insert_entries(current, new, num_before, destination=None):
    """
    Set destination to new entries inserted among current.

    Args:
        current:    The current array to add entries to. Not modified.

        new:    The new entries to add.

        num_before:    The number of current entries to precede each
            entry in new.

        destination:    The array to fill. Sholud already be
            pre-allocated and is completely overwritten.

    Returns:
        destination
    """

    if destination is None:
        destination = numpy.empty(current.size + new.size)

    current_start = 0
    for new_index, (current_end, new_entry) in enumerate(zip(num_before,
                                                             new)):
        destination[
            current_start + new_index
            :
            current_end + new_index
        ] = current[current_start : current_end]

        destination[current_end + new_index] = new_entry

        current_start = current_end

    destination[current_start + len(new) : ] = current[current_start : ]
    return destination


def approximate_1d_function(func,
                            support,
                            *,
                            min_grid_points=100,
                            tolerance=1e-6,
                            min_grid_step=None,
                            grid_refine_limit=0,
                            grid_only=False,
                            **spline_options):
    """
    Approxmiate the given function.

    Args:
        func(callable):   The function to approximate.

        support(iterable):    The range over which to approximate the
            function.

        min_grid_points(int):   The minimum number of grid points to use.

        tolerance(float):    The interpolation grid resolution is increased
            until the inteprolaiton at mid points of grid cells is within
            the given tolerance of the true function value.

        min_grid_step(float):    The smallest step allowed between grid
            points.

        grid_refine_limit(int):    At each step, only the `grid_refine_limit`
            most discrepant cells are sub-divided. A value of `0` results in no
            limit (i.e. all cells not satisfying specified tolerance are
            sub-divided).

        spline_options:    Any additional keyword arguments to pass to
            `InterpolatedUnivariateSpline` (used to create the
            interpolaiton).
    """

    if min_grid_step is None:
        min_grid_step = 1e-12 * (support[1] - support[0])

    x_values = numpy.linspace(*support, min_grid_points)
    y_values = func(x_values)
    while True:
        interpolated_values = InterpolatedUnivariateSpline(
            x_values[ : : 2],
            y_values[ : : 2],
            **spline_options
        )(
            x_values[1 : : 2]
        )

        mismatch_indices = 2 * select_mismatches(
            y_values[1 : : 2],
            interpolated_values,
            tolerance,
            grid_refine_limit
        )[0] + 1
        if mismatch_indices.size == 0:
            break

        new_x_values, num_below = get_new_grid_points(
            mismatch_indices,
            x_values,
            min_grid_step
        )
        _logger.debug('Adding %d new grid points', new_x_values.size)

        if new_x_values.size == 0:
            break

        new_y_values = func(new_x_values)
        x_values = insert_entries(x_values, new_x_values, num_below)
        y_values = insert_entries(y_values, new_y_values, num_below)
        _logger.debug('New grid size: %d', x_values.size)

    if grid_only:
        return x_values, y_values

    return InterpolatedUnivariateSpline(x_values, y_values, **spline_options)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    approx_sin = approximate_1d_function(numpy.sin,
                                         (-10.0, 10.0),
                                         min_grid_points = 17)
    plot_x = numpy.linspace(-10.0, 10.0, 1000)
    pyplot.plot(plot_x, numpy.sin(plot_x), label='sin')
    pyplot.plot(plot_x, approx_sin(plot_x), label='approx sin')
    pyplot.show()
