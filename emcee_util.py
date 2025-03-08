"""Collection of convenience functions for working with EMCEE."""

import logging

import h5py
import numpy

_logger = logging.getLogger(__name__)


def save_initial_position(
    position,
    samples_fname,
    *,
    chain_name="mcmc",
    nwalkers=1,
    log_prob_result=None,
    index=None,
):
    """Add a good starting position found for the given chain."""

    with h5py.File(samples_fname, "a") as samples_file:
        if "starting_positions" not in samples_file[chain_name]:
            samples_file[chain_name].create_group("starting_positions")
        destination = samples_file[chain_name]["starting_positions"]
        if "num_positions_found" not in destination.attrs:
            destination.attrs["num_positions_found"] = 0
            assert "positions" not in destination
            destination.create_dataset(
                "positions",
                (nwalkers,) + position.shape,
                maxshape=(None, len(position)),
                dtype=numpy.float64,
            )
            if log_prob_result is not None:
                assert "log_prob_results" not in destination
                destination.create_dataset(
                    "log_prob_results",
                    (nwalkers, len(log_prob_result)),
                    maxshape=(None, len(log_prob_result)),
                    dtype=numpy.float64,
                )
            if index is not None:
                assert "defined" not in destination
                destination.create_dataset("defined", (nwalkers,), dtype=bool)

        assert destination["positions"].shape[1] == (len(position))

        if index is None:
            index = destination.attrs["num_positions_found"]
            assert destination["positions"].shape[0] >= index
            if destination["positions"].shape[0] == index:
                for dset_name in ["positions", "log_prob_results"]:
                    destination[dset_name].resize(index + nwalkers, axis=0)
        else:
            assert destination["positions"].shape[0] == nwalkers
            assert not destination["defined"][index]
            destination["defined"][index] = True
        destination["positions"][index, :] = position
        destination.attrs["num_positions_found"] += 1

        if log_prob_result is not None:
            assert (
                destination["positions"].shape[0]
                == destination["log_prob_results"].shape[0]
            )
            assert destination["log_prob_results"].shape[1] == len(
                log_prob_result
            )
            destination["log_prob_results"][index, :] = log_prob_result


def load_initial_positions(
    samples_fname,
    *,
    num_params=None,
    num_walkers=None,
    blobs_dtype=None,
    chain_name="mcmc",
):
    """Load previously saved initial positions."""

    if num_walkers is not None:
        assert num_params is not None
        starting_positions = numpy.empty((num_walkers, num_params), dtype=float)
    starting_log_prob = None
    starting_blobs = None

    with h5py.File(samples_fname, "r") as samples_file:
        if "starting_positions" in samples_file[chain_name]:
            position_group = samples_file[chain_name]["starting_positions"]

            if "defined" in position_group:
                positions_found = position_group["defined"][:]
            else:
                positions_found = numpy.zeros(num_walkers, dtype=bool)
                positions_found[
                    : position_group.attrs["num_positions_found"]
                ] = True

            if num_walkers is not None:
                starting_positions[positions_found, :] = position_group[
                    "positions"
                ][positions_found, :]
            else:
                starting_positions = position_group["positions"][
                    positions_found, :
                ]
            if "log_prob_results" in position_group:
                log_prob_results = position_group["log_prob_results"][
                    positions_found
                ]
                starting_log_prob = numpy.empty(num_walkers, dtype=float)

                if len(log_prob_results.shape) == 1:
                    starting_log_prob = log_prob_results
                else:
                    starting_blobs = numpy.empty(num_walkers, dtype=blobs_dtype)
                    starting_log_prob[positions_found] = log_prob_results[:, 0]
                    starting_blobs[positions_found] = [
                        tuple(row) for row in log_prob_results[:, 1:]
                    ]

        else:
            positions_found = 0

    _logger.info(
        "Loaded %d/%d starting positions from a previous run.",
        positions_found,
        num_walkers or starting_positions.shape[0],
    )
    if num_walkers is None:
        return starting_positions
    if starting_log_prob is None:
        return (starting_positions, positions_found.sum())
    if starting_blobs is None:
        return (starting_positions, starting_log_prob, positions_found.sum())
    return (
        starting_positions,
        starting_log_prob,
        starting_blobs,
        positions_found.sum(),
    )
