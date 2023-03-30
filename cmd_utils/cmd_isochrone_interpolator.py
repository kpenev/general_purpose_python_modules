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

        self._isochrone = open(isochrone_fname, 'r')
        self._line = ''
        self.header = []
        while True:
            self._line = self._isochrone.readline()
            assert self._line[0] == '#'
            if self._line.startswith('# Zini'):
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

            assert self._line[0] == '#'

            yield self._line
            self._line = self._isochrone.readline()
            while self._line == '' or self._line[0] != '#':
                if self._line == '':
                    return
                yield self._line
                self._line = self._isochrone.readline()

        while self._line and self._line != '#isochrone terminated\n':
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

    def __init__(self, isochrone_fname):
        """Interpolate within the given isochrone grid."""

        with IsochroneFileIterator(isochrone_fname) as isochrone:
            self.data = [numpy.genfromtxt(section, names=True)
                         for section in isochrone]
            self.header = isochrone.header

        for section_data in self.data:
            assert numpy.unique(section_data['MH']).size == 1
            invalid = (
                (section_data['Mini'][1:] - section_data['Mini'][:-1])
                <
                0
            )
            if invalid.any():
                message = ('Non-monotonic initial mass in isochrone %s: '
                           %
                           isochrone_fname)
                for bad_index in numpy.nonzero(invalid):
                    message += (
                        'm[%d] = %s, m[%d] = %s'
                        %
                        (
                            bad_index,
                            section_data['Mini'][bad_index],
                            bad_index + 1,
                            section_data['Mini'][bad_index + 1]
                        )
                    )
                raise RuntimeError(message)
            assert (
                (section_data['Mini'][1:] - section_data['Mini'][:-1])
                >=
                0
            ).all()

        self.isochrone_fname = isochrone_fname

    def get_interpolated(self, quantity, initial_mass, feh):
        """
        Estimate the given quantity for a star of given initial mass & [Fe/H].

        Args:
            quantity(str):    The quantity to estimate. Should be one of the
                columns in the CMD isochrone file.

            initial_mass(float):    The initial mass of the star for which to
                estimate the given quantity in solar masses.

            feh(float):    The metallicity ([Fe/H]) of the star for which to
                estimate the given quantity

        Return:
            float:
                The value of the quantity estimated using linear interpolation
                against mass for each [Fe/H] value and then against [Fe/H].
        """

        interp_x = [section_data['MH'][0] for section_data in self.data]
        interp_y = [
            scipy.interpolate.interp1d(
                section_data['Mini'],
                section_data[quantity],
                bounds_error=False,
                assume_sorted=True
            )(
                initial_mass
            )
            for section_data in self.data
        ]
        if len(interp_x) == 1:
            return interp_y[0]

        return scipy.interpolate.interp1d(interp_x,
                                          interp_y,
                                          bounds_error=True,
                                          assume_sorted=True,
                                          axis=0)(feh)

    def interpolate(self, quantity):
        """Return an interplolator over the given quantity."""

        return lambda initial_mass, feh: self.get_interpolated(quantity,
                                                               initial_mass,
                                                               feh)

def plot_isochrone(cmd_fname):
    """Plot an isochrone read from the CMD interface."""

    interpolator = CMDInterpolator(cmd_fname)
    print('2.5Gyr Sun Teff = '
          +
          repr(10.0**interpolator.get_interpolated('logTe',
                                                   numpy.linspace(0.5, 1.0, 10),
                                                   [-0.1, 0.0, 0.1])))

    for section_data in interpolator.data:
        if abs(section_data['MH'][0]) <= 1e-5:
            pyplot.plot(section_data['Mini'],
                        section_data['Vmag'],
                        '-k')
    pyplot.xlabel('V')
    pyplot.ylabel('B-V')
    pyplot.show()

if __name__ == '__main__':
    plot_isochrone(
        os.path.expanduser(
            '~/projects/git/CircularizationDissipationConstraints/data/'
            'CMD_2.5Gyr_isochrone.dat'
        )
    )
