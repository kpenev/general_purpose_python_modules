"""Functions used by multiple bayesian sampling scripts."""

import os
import os.path
from datetime import datetime
import logging

import git

def get_code_version_str():
    """Return a string identifying the version of the code being used."""

    repository = git.Repo(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.abspath(__file__)
                )
            )
        )
    )
    head_sha = repository.commit().hexsha
    if repository.is_dirty():
        return head_sha + ':dirty'
    return head_sha

def setup_process(fname_datetime_format = '%Y%m%d%H%M%S',
                  system = "test_system",
                  std_out_err_fname = 'sampling_output/%(system)s_%(now)s_%(pid)d.outerr',
                  logging_fname = 'sampling_output/%(system)s_%(now)s_%(pid)d.log',
                  logging_verbosity = 'debug',
                  logging_message_format = ('%(levelname)s %(asctime)s %(name)s: %(message)s | '
                          '%(pathname)s.%(funcName)s:%(lineno)d'), # default_logging_format (bayesian.basic_util)
                  logging_datetime_format = None,
                  task='calculate'):
    """
    Logging and I/O setup for the current processes.

    Args:
        fname_datetime_format(str):    The format string for the current time

        system(str):    The name of the system being sampled.

        std_out_err_fname(str):    The format string for the name of the file
            to which stdout and stderr will be redirected. 

        logging_fname(str):    The format string for the name of the file to
            which logging messages will be written.

        logging_verbosity(str):    The verbosity level for logging.
            Options: 'debug', 'info', 'warning', 'error', 'critical'
        
        logging_message_format(str):    The format string for logging messages.

        logging_datetime_format(str):    The format string for the datetime
            component of logging messages.
        
        task(str):    The task being performed by the current process.
    """

    def ensure_directory(fname):
        """Make sure the directory containing the given name exists."""

        dirname = os.path.dirname(fname)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

    fname_substitutions = dict(
        now=datetime.now().strftime(fname_datetime_format),
        system=system,
        pid=os.getpid(),
        task=task
    )

    std_out_err_fname = std_out_err_fname % fname_substitutions
    ensure_directory(std_out_err_fname)

    io_destination = os.open(
        std_out_err_fname,
        os.O_WRONLY | os.O_TRUNC | os.O_CREAT | os.O_DSYNC,
        mode=0o666
    )
    os.dup2(io_destination, 1)
    os.dup2(io_destination, 2)

    logging_fname = logging_fname % fname_substitutions
    ensure_directory(logging_fname)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        handler.close()
    logging_config = dict(
        filename=logging_fname,
        level=getattr(logging, logging_verbosity.upper()),
        format=logging_message_format,
    )
    if logging_datetime_format is not None:
        logging_config['datefmt'] = logging_datetime_format
    logging.basicConfig(**logging_config)
