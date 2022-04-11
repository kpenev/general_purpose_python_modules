import numpy
from matplotlib import pyplot
from corner import corner

def make_corner_plot(plot_data_frame, corner_plot_fname='', **corner_kwargs):
    """Create a corner plot of the given pandas DataFrame and save/display."""

    plot_ranges = [
        (
            numpy.min(values[numpy.isfinite(values)]),
            numpy.max(values[numpy.isfinite(values)])
        )
        for _, values in plot_data_frame.items()
    ]
    corner(plot_data_frame,
           range=plot_ranges,
           **corner_kwargs)
    if corner_plot_fname:
        pyplot.savefig(corner_plot_fname)
    else:
        pyplot.show()
