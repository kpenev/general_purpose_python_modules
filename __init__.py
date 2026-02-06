"""A collection of tools and utilities used by multiple projects."""

import os

from .multi_pickle import MultiPickle
from .kde import KDEDistribution
from .split_normal_distribution import split_normal
from .grid_tracks_interpolate import grid_tracks_interpolate

try:
    from .reproduce_system import find_evolution
except ImportError:
    print("Failed to import find_evolution")


def ensure_directory(fname):
    """Make sure the directory containing the given name exists."""

    dirname = os.path.dirname(fname)
    if dirname and not os.path.exists(dirname):
        try:
            os.makedirs(dirname)
        except FileExistsError:
            if not os.path.isdir(dirname):
                raise
