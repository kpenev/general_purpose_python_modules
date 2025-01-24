#!/usr/bin/env python3
"""Define a class that interpolates within a CMD isochrone grid."""

import os.path

from matplotlib import pyplot
import numpy
from astropy import units as u, constants as c

from general_purpose_python_modules import grid_tracks_interpolate


class IsochroneFileIterator:
    """
    Iterate over the sections of the isochrone grid file with line generators.

    Attributes:
        _isochrone:    The opened isochrone file to iterate over.

        _line:    The last line read from the file.

        header:    The collection of comment lines in the beginning of the
            isochrone file.
    """

    def __init__(self, isochrone_fname):
        """Create the iterator."""

        self._line = ""
        self.header = []
        self._isochrone_fname = isochrone_fname
        self._isochrone = None

    def __enter__(self):
        """Just return self."""

        self._isochrone = open(self._isochrone_fname, "r", encoding="utf-8")
        while True:
            self._line = self._isochrone.readline()
            assert self._line[0] == "#"
            if self._line.startswith("# Zini"):
                break
            self.header.append(self._line)

        return self

    def __iter__(self):
        """Just return self."""

        return self

    def __next__(self):
        """Return a generator to the next section in the isochrone grid."""

        def section_line_generator():
            """Generator over lines of a section of the isochrone grid file."""

            assert self._line[0] == "#"

            yield self._line
            self._line = self._isochrone.readline()
            while self._line == "" or self._line[0] != "#":
                if self._line == "":
                    return
                yield self._line
                self._line = self._isochrone.readline()

        while self._line and self._line != "#isochrone terminated\n":
            return section_line_generator()
        raise StopIteration()

    def __exit__(self, *_):
        """Close the underlying isochrone file."""

        self._isochrone.close()


class CMDInterpolator:
    """
    Use interpolation to estimate quantities at any mass/[Fe/H] from a CMD grid.

    Attributes:
        header([str]):    The comment lines in the beginning of the isochrone
            file.

        isochrone_fname(str):    The name of the file from which interpolation
            data was read.

        grid([str, []):    The names of the indepnedent interpolation variables
            and the values of these variables at the grid points.

        _data([numpy field array]):    A the data contained in the isochrone
            file downloaded from the CMD interface, organized as a list of
            numpy field arrays, one for each section (corresponding to a
            single combination of [Fe/H] and age)
    """

    def _get_interpolation_grid(self):
        """Sanity check on the input data and prepare the interpolation."""

        for section_index, section_data in enumerate(self._data):
            for quantity in ["MH", "logAge"]:
                assert numpy.unique(section_data[quantity]).size == 1
            section_label = {
                quantity: float(section_data[quantity][0])
                for quantity in ["MH", "logAge"]
            }

            invalid = (section_data["Mini"][1:] - section_data["Mini"][:-1]) < 0
            if invalid.any():
                message = (
                    "Non-monotonic initial mass in isochrone "
                    f"{self.isochrone_fname}: "
                )
                for bad_index in numpy.nonzero(invalid):
                    message += (
                        f"m[{bad_index}] = {section_data['Mini'][bad_index]}, "
                        f"m[{bad_index + 1}] = "
                        f"{section_data['Mini'][bad_index + 1]}"
                    )
                raise ValueError(message)

            if section_index == 0:
                first_label = section_label
            elif section_index == 1:
                # False positive
                # pylint: disable=unsubscriptable-object

                grid = tuple(
                    (
                        quantity,
                        [first_label[quantity]],
                    )
                    for quantity in (
                        ["MH", "logAge"]
                        if section_label["MH"] == first_label["MH"]
                        else ["logAge", "MH"]
                    )
                )
                if (first_label["MH"] == section_label["MH"]) == (
                    first_label["logAge"] == section_label["logAge"]
                ):
                    raise ValueError(
                        "Age and metallicity of input data must be arranged on "
                        "a grid."
                    )
                # pylint: enable=unsubscriptable-object
            if section_index >= 1:
                # False positive
                # pylint: disable=used-before-assignment
                if section_label[grid[0][0]] == grid[0][1][0]:
                    grid[1][1].append(section_label[grid[1][0]])
                else:
                    if (
                        section_label[grid[1][0]]
                        != grid[1][1][section_index % len(grid[1][1])]
                    ):
                        raise ValueError(
                            "Age and metallicity of input data must be arranged"
                            f" on a grid. Section {section_index} {grid[1][0]} "
                            "should be "
                            f"{grid[1][1][section_index % len(grid[1][1])]} "
                            f"instead of {section_label[grid[1][0]]}."
                        )
                    if section_label[grid[0][0]] != grid[0][1][-1]:
                        grid[0][1].append(section_label[grid[0][0]])
                # pylint: enable=used-before-assignment
        return grid

    def __init__(self, isochrone_fname):
        """Interpolate within the given isochrone grid."""

        with IsochroneFileIterator(isochrone_fname) as isochrone:
            self._data = [
                numpy.genfromtxt(section, names=True) for section in isochrone
            ]
            self.header = isochrone.header

        self.isochrone_fname = isochrone_fname
        self.grid = tuple(
            (quantity, numpy.array(values))
            for quantity, values in self._get_interpolation_grid()
            if len(values) > 1
        )

    def __call__(self, quantities, **interpolate_to):
        """
        Estimate the given quantities for a star of given initial mass & [Fe/H].

        Args:
            quantities(iterable of str):    The quantities to estimate. Should
                be a subset of the columns in the CMD isochrone file.

            interpolate_to(dict):    The value of [Z/H], logAge and Mini of the
                star for which to estimate the given quantities. Values for
                quantities that had only a single value in the input data are
                ignored.

        Return:
            tuple of floats:
                The values of the quantities estimated using linear
                interpolation in all dimensions.
        """

        return grid_tracks_interpolate(
            interpolate_to, quantities, self.grid, self._data
        )


def plot_isochrone(cmd_fname):
    """Plot an isochrone read from the CMD interface."""

    interpolator = CMDInterpolator(cmd_fname)
    print(f"Grid: {interpolator.grid!r}")
    print(
        "4.7Gyr Sun Teff = "
        + repr(
            10.0
            ** interpolator(
                ("logTe",), Mini=1.0, MH=0.0, logAge=9.0 + numpy.log10(4.7)
            )[0]
        )
    )

    log_age = numpy.linspace(0.0, 1.0, 1000)
    track_values = [
        interpolator(
            ("logTe", "logL", "logg", "Mass"),
            Mini=1.0,
            MH=0.0,
            logAge=9.0 + logt,
        )
        for logt in log_age
    ]
    track = {
        "logTe": numpy.array([v[0] for v in track_values]),
        "logL": numpy.array([v[1] for v in track_values]),
        "logg": numpy.array([v[2] for v in track_values]),
        "Mass": numpy.array([v[3] for v in track_values]),
    }

    pyplot.plot(-track["logTe"], track["logL"], "-k")
    pyplot.xlabel("-Teff")
    pyplot.ylabel("L")
    pyplot.show()

    pyplot.plot(
        log_age,
        numpy.sqrt(
            c.G
            * track["Mass"]
            * c.M_sun
            / (10 ** track["logg"] * u.cm / u.s**2)
        ).to_value(u.R_sun),
        "-r",
    )
    pyplot.plot(
        log_age,
        (
            10.0 ** (track["logL"] / 2.0 - 2.0 * track["logTe"])
            / (2.0 * u.K**2)
            * numpy.sqrt(c.L_sun / (numpy.pi * c.sigma_sb))
        ).to_value(u.R_sun),
        "-g",
    )
    pyplot.show()

    pyplot.plot(
        log_age,
        (
            (
                10.0 ** (track["logL"] / 2.0 - 2.0 * track["logTe"])
                / (2.0 * u.K**2)
                * numpy.sqrt(c.L_sun / (numpy.pi * c.sigma_sb))
            )
            / numpy.sqrt(
                c.G
                * track["Mass"]
                * c.M_sun
                / (10 ** track["logg"] * u.cm / u.s**2)
            )
        ).to_value(""),
        "-k",
    )
    pyplot.show()


if __name__ == "__main__":
    plot_isochrone(
        # os.path.expanduser(
        #    "~/projects/git/CircularizationDissipationConstraints/data/"
        #    "CMD_2.5Gyr_isochrone.dat"
        # )
        os.path.expanduser(
            "~/projects/git/general_purpose_python_modules/cmd_utils/"
            "downloaded.ssv"
        )
    )
