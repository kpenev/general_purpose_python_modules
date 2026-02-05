import numpy
from matplotlib import pyplot
from corner import corner


def make_corner_plot(
    plot_data_frame, corner_plot_fname="show", **corner_kwargs
):
    """Create a corner plot of the given pandas DataFrame and save/display."""

    finite = None
    for _, values in plot_data_frame.items():
        if finite is None:
            finite = numpy.isfinite(values)
        else:
            finite &= numpy.isfinite(values)

    plot_data_frame = plot_data_frame[finite]

    if "range" not in corner_kwargs:
        corner_kwargs["range"] = [
            (
                numpy.min(values[numpy.isfinite(values)]),
                numpy.max(values[numpy.isfinite(values)]),
            )
            for _, values in plot_data_frame.items()
        ]
        print("Setting range to:", corner_kwargs["range"])
    if "labels" not in corner_kwargs:
        corner_kwargs["labels"] = plot_data_frame.columns
    figure = corner(plot_data_frame, **corner_kwargs)
    if corner_plot_fname == "show":
        pyplot.show()
        return None
    if corner_plot_fname:
        pyplot.savefig(corner_plot_fname)
        return None
    assert corner_plot_fname is None
    return figure
