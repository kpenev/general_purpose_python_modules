#!/usr/bin/env python3
"""Define a class that interpolates within a CMD isochrone grid."""

import os.path

from matplotlib import pyplot
import numpy
import scipy.interpolate


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

        self._isochrone = open(isochrone_fname, "r", encoding="utf-8")
        self._line = ""
        self.header = []
        while True:
            self._line = self._isochrone.readline()
            assert self._line[0] == "#"
            if self._line.startswith("# Zini"):
                break
            self.header.append(self._line)

    def __enter__(self):
        """Just return self."""

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

        data([numpy field array]):    A the data contained in the isochrone
            file downloaded from the CMD interface, organized as a list of
            numpy field arrays, one for each section (corresponding to a
            single [Fe/H] value)
    """

    def _get_interpolation_grid(self):
        """Sanity check on the input data and prepare the interpolation."""

        for section_index, section_data in enumerate(self.data):
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

    @staticmethod
    def _interpolate(quantity, grid, data, initial_mass, track_properties):
        """Used to recursively implement `__call__()`."""

        def evaluate_track(track):
            """Interpolate the given track to the specified mass."""

            return numpy.interp(initial_mass, track["Mini"], track[quantity])

        var_name, var_grid = grid[0]
        print(f"{var_name} grid: {var_grid}")
        target = track_properties.pop(var_name)
        if target < var_grid[0] or target > var_grid[-1]:
            raise ValueError(
                f"Requested {var_name} ({target}) is outside the available "
                "interpolation range: "
                f"{var_grid[0]} < {var_name} < {var_grid[-1]}"
            )
        above_ind = numpy.searchsorted(var_grid, target)
        if var_grid[above_ind] == target:
            if len(grid) == 1:
                return evaluate_track(data[above_ind])
            return CMDInterpolator._interpolate(
                quantity,
                grid[1:],
                data[
                    above_ind
                    * grid[1][1].size : (above_ind + 1)
                    * grid[1][1].size
                ],
                initial_mass,
                track_properties,
            )
        sub_grid = grid[1:]
        if sub_grid:
            closest_values = tuple(
                CMDInterpolator._interpolate(
                    quantity,
                    sub_grid,
                    data[ind * grid[1][1].size : (ind + 1) * grid[1][1].size],
                    initial_mass,
                    track_properties,
                )
                for ind in [above_ind - 1, above_ind]
            )
        else:
            closest_values = (
                evaluate_track(data[above_ind - 1]),
                evaluate_track(data[above_ind]),
            )
        print(
            f"Interpolating {quantity} to {var_name} = {target}: "
            f"x = {var_grid[above_ind - 1 : above_ind + 1]} "
            f"y = {closest_values}"
        )
        return numpy.interp(
            target, var_grid[above_ind - 1 : above_ind + 1], closest_values
        )

    def __init__(self, isochrone_fname):
        """Interpolate within the given isochrone grid."""

        with IsochroneFileIterator(isochrone_fname) as isochrone:
            self.data = [
                numpy.genfromtxt(section, names=True) for section in isochrone
            ]
            self.header = isochrone.header

        self.isochrone_fname = isochrone_fname
        self.grid = tuple(
            (quantity, numpy.array(values))
            for quantity, values in self._get_interpolation_grid()
            if len(values) > 1
        )

    def __call__(self, quantity, initial_mass, **track_properties):
        """
        Estimate the given quantity for a star of given initial mass & [Fe/H].

        Args:
            quantity(str):    The quantity to estimate. Should be one of the
                columns in the CMD isochrone file.

            initial_mass(float):    The initial mass of the star for which to
                estimate the given quantity in solar masses.

            track_properties:    The value of [Z/H] and/or logAge of the star
                for which to estimate the given quantity. Values for quantities
                that had only a single value in the input data are ignored.

        Return:
            float:
                The value of the quantity estimated using linear interpolation
                in all dimensions.
        """

        return self._interpolate(
            quantity, self.grid, self.data, initial_mass, track_properties
        )

    def interpolate(self, quantity):
        """Return an interplolator over the given quantity."""

        return lambda initial_mass, feh: self.get_interpolated(
            quantity, initial_mass, feh
        )


def plot_isochrone(cmd_fname):
    """Plot an isochrone read from the CMD interface."""

    interpolator = CMDInterpolator(cmd_fname)
    print(f"Grid: {interpolator.grid!r}")
    print(
        "4.7Gyr Sun Teff = "
        + repr(
            10.0
            ** interpolator("logTe", 1.0, MH=0.0, logAge=9.0 + numpy.log10(4.7))
        )
    )

    log_age = numpy.linspace(0.0, 1.0, 1000)
    track = {
        quantity: numpy.array(
            [
                interpolator(quantity, 1.0, MH=0.0, logAge=9.0 + logt)
                for logt in log_age
            ]
        )
        for quantity in ["logTe", "logL"]
    }

    pyplot.plot(-track["logTe"], track["logL"], "-k")
    pyplot.xlabel("-Teff")
    pyplot.ylabel("L")
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
