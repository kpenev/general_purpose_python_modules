#!/usr/bin/env python3

"""Unit tests of the memory mapped binary copy of CMD isochrone grids."""

import multiprocessing
import os
import os.path
import tempfile
import time
import unittest

import numpy

from general_purpose_python_modules.cmd_utils import (
    cmd_isochrone_interpolator as cmd,
)

columns = ("Zini", "MH", "logAge", "Mini", "Mass", "logL")


def write_isochrone_file(fname, offset=0.0):
    """Write a small grid (2 [M/H] x 3 ages) in the CMD text format."""

    with open(fname, "w", encoding="utf-8") as isochrone:
        isochrone.write("# File generated for a unit test\n")
        isochrone.write("# Kind of output: isochrone tables\n")
        for metallicity in (-0.5, 0.0):
            for log_age in (8.0, 9.0, 10.0):
                isochrone.write("# " + "  ".join(columns) + "\n")
                for mass in numpy.linspace(0.1, 2.0 - 0.3 * (log_age - 8), 5):
                    isochrone.write(
                        f"0.0152 {metallicity} {log_age} {mass:.10f} "
                        f"{mass * 0.99:.5f} {offset + 4.0 * mass:.5f}\n"
                    )
        isochrone.write("#isochrone terminated\n")


def count_builds(directory):
    """
    Make each binary copy creation leave a line in a log file.

    Each creation is also made to take 0.5 s (like parsing a real grid), so
    processes starting together would overlap without the lock.
    """

    original_write = cmd._write_binary

    def logged_write(*args):
        with open(
            os.path.join(directory, "builds.log"), "a", encoding="utf-8"
        ) as log:
            log.write(f"{os.getpid()}\n")
        time.sleep(0.5)
        original_write(*args)

    cmd._write_binary = logged_write
    return original_write


def get_builds(directory):
    """Return how many times a binary copy was created."""

    log_fname = os.path.join(directory, "builds.log")
    if not os.path.exists(log_fname):
        return 0
    with open(log_fname, "r", encoding="utf-8") as log:
        return len(log.readlines())


def load_in_subprocess(text_fname, binary_dir):
    """Load the grid (target of the concurrent loading test)."""

    cmd.load_isochrone_sections(text_fname, binary_dir)


class TestBinaryIsochrones(unittest.TestCase):
    """Tests of creating, reusing and refreshing the binary copy."""

    def setUp(self):
        """Create the text grid in a temporary directory."""

        self._directory = tempfile.TemporaryDirectory()
        self.text_fname = os.path.join(self._directory.name, "grid.ssv")
        self.binary_dir = os.path.join(self._directory.name, "binary")
        write_isochrone_file(self.text_fname)
        self._original_write = count_builds(self._directory.name)

    def tearDown(self):
        """Undo the build counting and remove the files."""

        cmd._write_binary = self._original_write
        self._directory.cleanup()

    def assert_matches_text(self, header, sections):
        """The loaded grid must equal the text file parsed directly."""

        text_header, text_sections = cmd._read_text_sections(self.text_fname)
        self.assertEqual(header, text_header)
        self.assertEqual(len(sections), len(text_sections))
        for section, text_section in zip(sections, text_sections):
            self.assertEqual(section.dtype, text_section.dtype)
            numpy.testing.assert_array_equal(section, text_section)

    def test_matches_text_and_is_reused(self):
        """Created once with values exactly as parsed, then reused."""

        for _ in range(2):
            self.assert_matches_text(
                *cmd.load_isochrone_sections(self.text_fname, self.binary_dir)
            )
        self.assertEqual(get_builds(self._directory.name), 1)

    def test_rebuilt_when_text_changes(self):
        """A changed text file (size or modification time) is reloaded."""

        cmd.load_isochrone_sections(self.text_fname, self.binary_dir)
        write_isochrone_file(self.text_fname, offset=1.0)
        stat = os.stat(self.text_fname)
        os.utime(
            self.text_fname, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9)
        )
        self.assert_matches_text(
            *cmd.load_isochrone_sections(self.text_fname, self.binary_dir)
        )
        self.assertEqual(get_builds(self._directory.name), 2)

    def test_read_only(self):
        """The grid cannot be modified by accident."""

        sections = cmd.load_isochrone_sections(
            self.text_fname, self.binary_dir
        )[1]
        with self.assertRaises(ValueError):
            sections[0]["Mini"][0] = 1.0

    def test_default_directory(self):
        """Without binary_dir, the copy is next to the text file."""

        cmd.load_isochrone_sections(self.text_fname)
        self.assertTrue(
            os.path.exists(
                os.path.join(self._directory.name, "grid.ssv.grid.npy")
            )
        )
        self.assertFalse(os.path.exists(self.binary_dir))

    def test_concurrent_loading_builds_once(self):
        """Processes starting together create the copy only once."""

        context = multiprocessing.get_context("fork")
        processes = [
            context.Process(
                target=load_in_subprocess,
                args=(self.text_fname, self.binary_dir),
            )
            for _ in range(4)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join()
            self.assertEqual(process.exitcode, 0)
        self.assertEqual(get_builds(self._directory.name), 1)

    def test_interpolator(self):
        """CMDInterpolator interpolates from the binary copy."""

        interpolator = cmd.CMDInterpolator(
            self.text_fname, binary_dir=self.binary_dir
        )
        self.assertTrue(
            os.path.exists(os.path.join(self.binary_dir, "grid.ssv.grid.npy"))
        )
        # logL = 4 Mini on every isochrone, so also when interpolated.
        numpy.testing.assert_allclose(
            interpolator(("logL",), MH=-0.2, logAge=8.7, Mini=1.234),
            [4.0 * 1.234],
            rtol=1e-5,
        )


if __name__ == "__main__":
    unittest.main()
