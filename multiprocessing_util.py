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

def setup_process(**config):
    """
    Logging and I/O setup for the current processes.

    Args (keyword only):
        std_out_err_fname(str):    Format string for the standard output/error
            file name with substitutions including any keyword arguments passed
            to this function, ``now`` which gets replaced by current date/time,
            ``pid`` which gets replaced by the process ID, ``task`` which
            gets the value ``'calculate'`` by default but can be overwritten
            here.

        logging_fname(str):    Format string for the logging file name (see
            ``std_out_err_fname``).

        fname_datetime_format(str):    The format for the date and time string
            to be inserted in the file names.

        logging_message_format(str):    The format for the logging messages (see
            logging module documentation)

        logging_verbosity(str):    The verbosity of logging (see logging module
            documentation)

        All other keyword arguments are used to substitute into the format
            strings for the filenames.

    Returns:
        None
    """

    def ensure_directory(fname):
        """Make sure the directory containing the given name exists."""

        dirname = os.path.dirname(fname)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname)

    if 'task' not in config:
        config['task'] = 'calculate'
    config.update(
        now=datetime.now().strftime(config['fname_datetime_format']),
        pid=os.getpid()
    )

    if config.get('std_out_err_fname') is not None:
        print('Config: ' + repr(config))
        std_out_err_fname = config['std_out_err_fname'].format_map(config)
        ensure_directory(std_out_err_fname)

        io_destination = os.open(
            std_out_err_fname,
            os.O_WRONLY | os.O_TRUNC | os.O_CREAT | os.O_DSYNC,
            mode=0o666
        )
        os.dup2(io_destination, 1)
        os.dup2(io_destination, 2)

    logging_fname = config['logging_fname'].format_map(config)
    ensure_directory(logging_fname)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        handler.close()
    logging_config = dict(
        filename=logging_fname,
        level=getattr(
            logging,
            config.get('logging_verbosity', config.get('verbose')).upper()
        ),
        format=config['logging_message_format']
    )
    if config.get('logging_datetime_format') is not None:
        logging_config['datefmt'] = config['logging_datetime_format']
    logging.basicConfig(**logging_config)

def setup_process_map(config):
    """Like `setup_process`, but more convenient for `multiprocessing.Pool`."""

    setup_process(**config)
