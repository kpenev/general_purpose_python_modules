"""Module for working with pickling results tied to configarations."""

from contextlib import contextmanager
from pickle import Pickler, Unpickler
import logging
from os import path
import shutil
from tempfile import TemporaryDirectory

import numpy
from scipy.stats import rv_continuous
from scipy.stats.distributions import rv_frozen

@contextmanager
def _manage_pickle_file(pickle_file, mode='r'):
    """Manage pickle file from either opened file or filename."""

    assert mode in 'arw'
    close = False
    if isinstance(pickle_file, str):
        opened_f = open(pickle_file, mode + 'b')
        close = True
    else:
        opened_f = pickle_file

    try:
        yield Unpickler(opened_f) if mode == 'r' else Pickler(opened_f)
    finally:
        if close:
            opened_f.close()


class MultiPickle:
    """Class for managing collections of results tied to configuration."""

    def _get_pickle_config(self, config):
        """
        Return configuration ready to pickle or compare to pickled.

        Args:
            config:    See same name argument to `check_for_pickled()`

        Returns:
            dict:
                Dictionary of the configuration to be pickled or compared to
                what is pickled.
        """

        if isinstance(config, dict):
            result = dict(config)
        else:
            result = dict(vars(config))

        for arg in self._ignore_config:
            if arg in result:
                del result[arg]
        return result


    def _compare_config(self, check_config, pickled_config):
        """Return True iff the two configurations match."""

        for check_arg, check_value in check_config.items():
            if check_arg not in pickled_config:
                return False
            pickled_value = pickled_config[check_arg]
            if isinstance(check_value, (rv_continuous, rv_frozen)):
                assert isinstance(pickled_value, (rv_continuous, rv_frozen))
                if (
                        pickled_value.kwds != check_value.kwds
                        or
                        pickled_value.args != check_value.args
                ):
                    self._logger.debug(
                        'Config %s distributions do not match. Pickled %s, %s '
                        'vs target %s, %s',
                        repr(check_arg),
                        repr(pickled_value.args),
                        repr(pickled_value.kwds),
                        repr(check_value.args),
                        repr(check_value.kwds)
                    )
                    return False
            elif isinstance(check_value, numpy.ndarray):
                assert isinstance(pickled_value, numpy.ndarray)
                if not (pickled_value == check_value).all():
                    self._logger.debug(
                        'Config %s arrays do not match. Pickled %s target %s',
                        repr(check_arg),
                        repr(pickled_value),
                        repr(check_value)
                    )
                    return False
            elif pickled_value != check_value:
                self._logger.debug(
                    'Config %s do not match. Pickled %s target %s',
                    repr(check_arg),
                    repr(pickled_value),
                    repr(check_value)
                )
                return False
        return True


    def __init__(self,
                 pickle_fname,
                 ignore_config_args=(),
                 auto_create=True):
        """
        Get ready to manage pickles in the given file.

        Args:
            pickle_fname(str):    The filename of the pickle file to manage.

            moder('r', 'w', or 'a'):    Whether the file should be opened for
                reading, writing (reset), or appending.

            ignore_config_args(iterable):    List of configuration arguments
                that do not affect the result and hence should not be checked or
                pickled.

            auto_create(bool):    Should the file be created if it does not
                exist when opening for writing is attempted?
        """

        self._ignore_config = ignore_config_args
        self._pickle_fname = pickle_fname
        self._auto_create = auto_create
        self._logger = logging.getLogger(__name__)


    def check_for_pickled(self, config):
        """
        If the given configuration already has pickled results return those.

        Args:
            config:    The configuration for which a result is needed. Usually a
                command line parser namespace directly as parsed.

            pickle_file:   Either an already opened pickle file (mode should be
                'rb'), or a filaname that will be opened.

        Returns:
            If the given configuration is found in the pickle file, returns
            whatever has been pickled with it, otherwise None.
        """

        if not path.exists(self._pickle_fname):
            self._logger.debug(
                'Checking for pickled result in non-existent pickle file %s.',
                repr(self._pickle_fname)
            )
            return None

        check_config = self._get_pickle_config(config)
        try:
            with _manage_pickle_file(self._pickle_fname) as unpickler:
                while True:
                    pickled_config = unpickler.load()
                    nobjects = unpickler.load()
                    if self._compare_config(check_config, pickled_config):
                        self._logger.debug('Found matching pickled results.')
                        return tuple(unpickler.load() for _ in range(nobjects))
                    self._logger.debug(
                        'Skipping over %s pickled objects.',
                        repr(nobjects)
                    )
                    for _ in range(nobjects):
                        self._logger.debug('Skipping from %s (%s)',
                                           repr(unpickler),
                                           type(unpickler))
                        unpickler.load()
        except EOFError:
            self._logger.debug('None of the pickled results match specfied '
                               'configuration.')
        return None


    def discard_result(self, config):
        """Discard all entries in the pickle file matching configuration."""

        if not path.exists(self._pickle_fname):
            self._logger.debug(
                'Attempt to discard entries from non-existent pickle file %s.',
                repr(self._pickle_fname)
            )
            return None

        check_config = self._get_pickle_config(config)
        with TemporaryDirectory() as temp_dir:
            shutil.move(self._pickle_fname, temp_dir)
            temp_pickle_fname = path.join(temp_dir,
                                          path.basename(self._pickle_fname))
            try:
                with _manage_pickle_file(temp_pickle_fname, 'r') as original,\
                        _manage_pickle_file(self._pickle_fname, 'w') as updated:
                    while True:
                        pickled_config = original.load()
                        nobjects = original.load()
                        if self._compare_config(check_config, pickled_config):
                            self._logger.debug(
                                'Discarding matching pickled results.'
                            )
                            for _ in range(nobjects):
                                original.load()
                        else:
                            updated.dump(pickled_config)
                            updated.dump(nobjects)
                            for _ in range(nobjects):
                                updated.dump(original.load())


            except EOFError:
                self._logger.debug('None of the pickled results match specfied '
                                   'configuration.')
        return None



    def add_result(self, config, *result):
        """
        Add a new result to the managed pickle file.

        Args:
            config:    The configuration for which this result applies.

            result:    Whatever needs to be saved.

        Returns:
            None
        """

        with _manage_pickle_file(self._pickle_fname, 'a') as pickler:
            pickler.dump(self._get_pickle_config(config))
            pickler.dump(len(result))
            for entry in result:
                pickler.dump(entry)
