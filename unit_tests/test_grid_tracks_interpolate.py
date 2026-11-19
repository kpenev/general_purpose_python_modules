#!/usr/bin/env python3

"""Unit tests of grid_tracks_interpolate(), linear and PCHIP."""

import unittest

import numpy
from scipy.interpolate import PchipInterpolator

import grid_tracks_interpolate as module
from grid_tracks_interpolate import grid_tracks_interpolate

metallicity_nodes = numpy.array([-1.0, -0.6, -0.5, 0.0, 0.3, 0.7])
log_age_nodes = numpy.array([8.0, 8.4, 8.5, 8.9, 9.3, 9.4, 10.0])
grid = (("MH", metallicity_nodes), ("logAge", log_age_nodes))
quantities = ("wavy", "rising")
# Every track covers masses below this, so no clipping happens below it.
common_max_mass = 1.4


def max_mass(log_age):
    """The end of the track at the given age (tracks get shorter)."""

    return 3.0 - 0.8 * (log_age - 8.0)


def smooth_quantities(metallicity, log_age, mass):
    """Smooth test quantities: one turns around, one rises with mass."""

    return {
        "wavy": numpy.sin(2.0 * mass + metallicity)
        * numpy.exp(-0.3 * (log_age - 9.0)),
        "rising": mass**2 + 0.5 * metallicity - 0.2 * log_age,
    }


def linear_quantities(metallicity, log_age, mass):
    """Test quantities linear in all variables."""

    return {
        "wavy": 2.0 * mass + 3.0 * metallicity - log_age,
        "rising": -mass + 0.5 * metallicity + 2.0 * log_age,
    }


def make_data(rng, function, with_jumps):
    """
    Tabulate the function on the grid as tracks of varying length.

    If ``with_jumps``, each track repeats one mass, with the quantities offset
    from there on (as at a change of evolutionary phase).
    """

    data = []
    for metallicity in metallicity_nodes:
        for log_age in log_age_nodes:
            track_end = max_mass(log_age)
            masses = numpy.sort(
                numpy.concatenate(
                    [[0.1, track_end], rng.uniform(0.1, track_end, 40)]
                )
            )
            offsets = numpy.zeros(masses.size)
            if with_jumps:
                jump = numpy.searchsorted(masses, 0.6 * track_end)
                masses = numpy.insert(masses, jump, masses[jump])
                offsets = numpy.where(
                    numpy.arange(masses.size) > jump, 0.3, 0.0
                )
            track = numpy.empty(
                masses.size,
                dtype=[("Mini", float)]
                + [(name, float) for name in quantities],
            )
            track["Mini"] = masses
            values = function(metallicity, log_age, masses)
            for name in quantities:
                track[name] = values[name] + offsets
            data.append(track)
    return data


def interpolate(data, method, clip=False, **interpolate_to):
    """Shorthand for interpolating the test quantities."""

    return grid_tracks_interpolate(
        interpolate_to, quantities, grid, data, clip=clip, method=method
    )


def get_track(data, metallicity_index, log_age_index):
    """Return the track at the given grid node."""

    return data[metallicity_index * log_age_nodes.size + log_age_index]


class TestGridTracksInterpolate(unittest.TestCase):
    """Tests of linear and PCHIP interpolation of tracks on a grid."""

    def setUp(self):
        """Create the test data."""

        self.rng = numpy.random.default_rng(20261117)
        self.data = make_data(self.rng, smooth_quantities, with_jumps=True)

    def random_target(self, max_mass_factor=1.0):
        """Return a random target within the grid."""

        log_age = self.rng.uniform(log_age_nodes[0], log_age_nodes[-1])
        return {
            "MH": self.rng.uniform(metallicity_nodes[0], metallicity_nodes[-1]),
            "logAge": log_age,
            "Mini": self.rng.uniform(0.1, max_mass(log_age) * max_mass_factor),
        }

    def test_unknown_method(self):
        """An unknown method is an error."""

        with self.assertRaises(ValueError):
            interpolate(self.data, "cubic", MH=0.1, logAge=9.0, Mini=1.0)

    def test_pchip_evaluate_matches_scipy(self):
        """PCHIP on a local stencil equals SciPy's on all nodes."""

        for trial in range(300):
            size = self.rng.integers(2, 10)
            nodes = numpy.cumsum(self.rng.uniform(0.1, 2.0, size))
            values = self.rng.normal(size=(size, 3))
            if trial % 3 == 0:
                values[self.rng.integers(0, size, 2)] = 0.0
            if trial % 5 == 0:
                values = numpy.cumsum(numpy.abs(values), axis=0)
            reference = PchipInterpolator(nodes, values, axis=0)
            for target in self.rng.uniform(nodes[0], nodes[-1], 10):
                bracket = min(
                    numpy.searchsorted(nodes, target, side="right") - 1,
                    size - 2,
                )
                first = max(bracket - 1, 0)
                stop = min(bracket + 3, size)
                numpy.testing.assert_allclose(
                    module._pchip_evaluate(
                        nodes[first:stop],
                        values[first:stop],
                        bracket - first,
                        target,
                    ),
                    reference(target),
                    rtol=1e-13,
                    atol=1e-13,
                )

    def test_track_matches_scipy_per_piece(self):
        """Along a track PCHIP equals SciPy's on each unbroken piece."""

        for track in self.data[::5]:
            masses = track["Mini"]
            jump = numpy.flatnonzero(numpy.diff(masses) == 0)[0] + 1
            # Only the interval touching the jump would change if the
            # interpolation reached across it (through the derivative at the
            # piece's end), so each piece is always sampled there, close to
            # the jump, besides the random samples.
            near_jump = {
                (0, jump): masses[jump - 2]
                + 0.9 * (masses[jump - 1] - masses[jump - 2]),
                (jump, masses.size): masses[jump]
                + 0.1 * (masses[jump + 1] - masses[jump]),
            }
            for (start, stop), near_jump_mass in near_jump.items():
                reference = PchipInterpolator(
                    masses[start:stop],
                    numpy.array(
                        [track[name][start:stop] for name in quantities]
                    ).T,
                    axis=0,
                )
                for mass in numpy.append(
                    self.rng.uniform(masses[start], masses[stop - 1], 20),
                    near_jump_mass,
                ):
                    numpy.testing.assert_allclose(
                        module._evaluate_track_pchip(
                            track, "Mini", mass, quantities
                        ),
                        reference(mass),
                        rtol=1e-13,
                        atol=1e-13,
                    )

    def test_linear_data_reproduced(self):
        """Both methods are exact for data linear in all variables."""

        data = make_data(self.rng, linear_quantities, with_jumps=False)
        for _ in range(200):
            target = self.random_target()
            target["Mini"] = self.rng.uniform(0.1, common_max_mass)
            expected = linear_quantities(
                target["MH"], target["logAge"], target["Mini"]
            )
            for method in module.interpolation_methods:
                numpy.testing.assert_allclose(
                    interpolate(data, method, **target),
                    [expected[name] for name in quantities],
                    rtol=1e-12,
                    atol=1e-12,
                )

    def test_nodes_reproduced_exactly(self):
        """At grid nodes and track points PCHIP returns the tabulated values."""

        for metallicity_index, metallicity in enumerate(metallicity_nodes):
            for log_age_index, log_age in enumerate(log_age_nodes):
                track = get_track(self.data, metallicity_index, log_age_index)
                for point in range(0, track.size, 7):
                    # At a repeated mass the result is unspecified (either
                    # side's value, see _evaluate_track_pchip()).
                    if (track["Mini"] == track["Mini"][point]).sum() > 1:
                        continue
                    numpy.testing.assert_array_equal(
                        interpolate(
                            self.data,
                            "pchip",
                            MH=metallicity,
                            logAge=log_age,
                            Mini=track["Mini"][point],
                        ),
                        [track[name][point] for name in quantities],
                    )

    def test_no_overshoot(self):
        """PCHIP stays within the values of the four bracketing tracks."""

        for _ in range(300):
            target = self.random_target()
            metallicity_above = numpy.searchsorted(
                metallicity_nodes, target["MH"]
            )
            log_age_above = numpy.searchsorted(log_age_nodes, target["logAge"])
            corners = []
            for metallicity_index in (metallicity_above - 1, metallicity_above):
                for log_age_index in (log_age_above - 1, log_age_above):
                    track = get_track(
                        self.data, metallicity_index, log_age_index
                    )
                    corners.append(
                        module._evaluate_track_pchip(
                            track,
                            "Mini",
                            min(target["Mini"], track["Mini"][-1]),
                            quantities,
                        )
                    )
            corners = numpy.array(corners)
            result = numpy.array(
                interpolate(self.data, "pchip", clip=True, **target)[0]
            )
            self.assertTrue((result >= corners.min(axis=0) - 1e-14).all())
            self.assertTrue((result <= corners.max(axis=0) + 1e-14).all())

    def slope_jumps(self, method, variable, node, step, target):
        """Relative change of the slope across a node, per quantity."""

        values = {}
        for offset in (-2, -1, 0, 1, 2):
            target[variable] = node + offset * step
            values[offset] = numpy.array(
                interpolate(self.data, method, **target)
            )
        left = (3 * values[0] - 4 * values[-1] + values[-2]) / (2 * step)
        right = (-3 * values[0] + 4 * values[1] - values[2]) / (2 * step)
        return numpy.abs(right - left) / numpy.maximum(
            numpy.maximum(numpy.abs(left), numpy.abs(right)), 1e-3
        )

    def test_derivative_continuous(self):
        """PCHIP's slope is continuous across nodes; linear's is not."""

        cases = []
        for metallicity in metallicity_nodes[1:-1]:
            cases.append(
                ("MH", metallicity, 1e-6, {"logAge": 8.73, "Mini": 0.83})
            )
        for log_age in log_age_nodes[1:-1]:
            cases.append(("logAge", log_age, 1e-6, {"MH": -0.21, "Mini": 1.17}))
        track = get_track(self.data, 2, 3)
        for point in range(1, 20, 3):
            masses = track["Mini"]
            cases.append(
                (
                    "Mini",
                    masses[point],
                    1e-4
                    * min(
                        masses[point] - masses[point - 1],
                        masses[point + 1] - masses[point],
                    ),
                    {"MH": metallicity_nodes[2], "logAge": log_age_nodes[3]},
                )
            )
        largest_linear_jump = 0.0
        for variable, node, step, target in cases:
            pchip_jumps = self.slope_jumps(
                "pchip", variable, node, step, dict(target)
            )
            self.assertLess(pchip_jumps.max(), 1e-4, (variable, node))
            largest_linear_jump = max(
                largest_linear_jump,
                self.slope_jumps(
                    "linear", variable, node, step, dict(target)
                ).max(),
            )
        # The check would detect a kink.
        self.assertGreater(largest_linear_jump, 1e-2)

    def test_clip_errors_and_range_as_linear(self):
        """PCHIP raises, clips and gives the range exactly as linear does."""

        for _ in range(300):
            target = self.random_target(max_mass_factor=1.1)
            outcomes = {}
            for method in module.interpolation_methods:
                try:
                    interpolate(self.data, method, **target)
                    raises = False
                except ValueError:
                    raises = True
                values, excess = interpolate(
                    self.data, method, clip=True, **target
                )
                self.assertTrue(numpy.isfinite(values).all())
                self.assertEqual(raises, excess > 0)
                outcomes[method] = raises
                range_target = dict(target, Mini="range")
                outcomes[method, "range"] = grid_tracks_interpolate(
                    range_target, None, grid, self.data, method=method
                )
            self.assertEqual(outcomes["linear"], outcomes["pchip"])
            self.assertEqual(
                outcomes["linear", "range"], outcomes["pchip", "range"]
            )


if __name__ == "__main__":
    unittest.main()
