#!/usr/bin/env python3

"""Find tidal evolution reproducing the present state of a system."""

from glob import glob
import os.path
from types import SimpleNamespace
import logging
import pickle

#from scipy import optimize
import scipy
import numpy
from astropy import units
from configargparse import ArgumentParser, DefaultsFormatter

from stellar_evolution.library_interface import MESAInterpolator
from stellar_evolution.manager import StellarEvolutionManager
from orbital_evolution.transformations import phase_lag

from orbital_evolution.command_line_util import\
    add_binary_config,\
    add_evolution_config,\
    set_up_library

from general_purpose_python_modules.solve_for_initial_values import \
    InitialValueFinder

from general_purpose_python_modules.spin_calculation import \
    SpinPeriod

from multiprocessing import Pool
#import multiprocessing as mp
from functools import partial
from multiprocessing_util import setup_process

def add_dissipation_cmdline(parser, lgq_suffixes=('primary', 'secondary')):
    """
    Add argumets to a command line parser to define tidal dissipation.

    Args:
        parser:    The command line parser to add the argument to. Assumed it is
            an instance of argpars.ArgumentParser.

        lgq_suffixes:    All arguments are named --lgQ-<suffix>, with suffix
            iterating over this argument. This way multiple bodies can have
            argument to specify their dissipation.

    Returns:
        None, but add arguments to the parser.
    """

    for component in lgq_suffixes:
        prefix = '--lgQ-' + component

        parser.add_argument(
            prefix,
            type=float,
            default='6.0',
            help='The value of log10(Q*) to assume, at the reference tidal and '
            'spin periods if --lgQ-%(suffix)s-wtide-dependence and/or '
            '--lgQ-%(suffix)s-wspin-dependence is specified. '
            'Default: %%(default)s.' % dict(suffix=component)
        )
        parser.add_argument(
            prefix + '-wtide-dependence',
            nargs='+',
            type=float,
            default=[],
            metavar=('<powerlaw index> <break frequency> <powerlaw index>',
                     '<break frequency> <powerlaw index>'),
            help='Pass this argument to make lgQ depend on tidal period. At '
            'least three arguments must be passed: 1) the powerlaw index for '
            'tidal frequencies below the first break, 2) the frequency '
            '[rad/day] where the first break occurs and 3) the powerlaw index '
            'after the first break. Additional arguments must come in pairs, '
            'specifying more frequencies where breaks occur and the powerlaw '
            'indices for frequencies higher than the break.'
        )
        parser.add_argument(
            prefix + '-wspin-dependence',
            nargs='+',
            type=float,
            default=[],
            metavar=('<powerlaw index> <break frequency> <powerlaw index>',
                     '<break frequency> <powerlaw index>'),
            help='Pass this argument to make lgQ depend on spin '
            'period. At least three arguments must be passed: 1) the powerlaw '
            'index for tidal frequencies below the first break, 2) the '
            'frequency [rad/day] where the first break occurs and 3) the '
            'powerlaw index after the first break. Additional arguments must '
            'come in pairs, specifying more frequencies where breaks occur and'
            ' the powerlaw indices for frequencies higher than the break.'
        )

def get_poet_dissipation_from_cmdline(cmdline_args,
                                      lgq_suffixes=('primary', 'secondary')):
    """
    Return keyword arguments setting the dissipation as specified on cmdline.

    Args:
        cmdline_args:    The parsed command line which should have included the
            arguments added by add_dissipation_cmdline().

    Returns:
        dict:
            Dictionary with keys given by lgq_suffixes, and values dictionaries
            of keyword arguments to pass to the POET set_dissipation methods for
            stars and planets.
    """

    default_breaks = None
    default_powers = numpy.array([0.0])

    result = dict()
    for component in lgq_suffixes:
        lgq = getattr(cmdline_args, 'lgQ_' + component)
        dependence = dict(
            tidal=numpy.array(
                getattr(cmdline_args,
                        'lgQ_' + component + '_wtide_dependence')
            ),
            spin=numpy.array(
                getattr(cmdline_args,
                        'lgQ_' + component + '_wspin_dependence')
            )
        )
        if numpy.isfinite(lgq):
            result[component] = dict()
            result[component]['reference_phase_lag'] = phase_lag(lgq)
            for dep_name in ['tidal', 'spin']:
                result[
                    component
                ][
                    '_'.join((dep_name, 'frequency', 'breaks'))
                ] = (
                    numpy.copy(dependence[dep_name][1 : : 2])
                    if dependence[dep_name].size > 0 else
                    default_breaks
                )
                result[
                    component
                ][
                    '_'.join((dep_name, 'frequency', 'powers'))
                ] = (
                    numpy.copy(dependence[dep_name][: : 2][:])
                    if dependence[dep_name].size > 0 else
                    default_powers
                )
        else:
            result[component] = None

    return result


def parse_command_line():
    """Return command line arguments configuration the system to reproduce."""

    parser = ArgumentParser(
        description=__doc__,
        default_config_files=['system.params'],
        formatter_class=DefaultsFormatter,
        ignore_unknown_config_file_keys=False
    )
    parser.add_argument(
        '--config', '-c',
        is_config_file=True,
        help='Config file to use instead of default.'
    )
    parser.add_argument(
        '--create-config',
        default=None,
        help='Filename to create a config file where all options are set per '
        'what is currently parsed.'
    )

    add_binary_config(parser,
                      skip=('initial_orbital_period', 'dissipation', 'Wdisk'),
                      require_secondary=True)
    parser.add_argument(
        '--disk-lock-period', '--Pdisk',
        type=float,
        default=7.0,
        help='The fixed spin period of the surface of the primary until '
        'the disk dissipates.'
    )

    add_dissipation_cmdline(parser)
    add_evolution_config(parser)

    parser.add_argument(
        '--orbital-period', '--Porb',
        type=float,
        default=5.0,
        help='The orbital period to reproduce at the final age in days.'
    )
    parser.add_argument(
        '--orbital-period-tolerance', '--porb-tol',
        type=float,
        default=1e-6,
        help='The tolerance to which to find the initial orbital preiod when '
        'trying to match the final one.'
    )
    parser.add_argument(
        '--period-search-factor',
        type=float,
        default=2.0,
        help='The factor by which to change the initial period guess while '
        'searching for a range surrounding the known present day orbital '
        'period.'
    )
    parser.add_argument(
        '--scaled-period-guess',
        type=float,
        default=1.0,
        help='The search for initial period to bracket the observed final '
        'period will start from this value multiplied by the final orbital '
        'period.'
    )
    parser.add_argument(
        '--logging-verbosity',
        choices=['debug', 'info', 'warning', 'error', 'critical'],
        default='info',
        help='The lowest level logging messages to issue.'
    )
    parser.add_argument(
        '--output-pickle', '--output', '-o',
        default='found_evolution.pkl',
        help='Filename to which to append a pickle of the found evolution.'
    )
    parser.add_argument(
        '--eccentricity',
        type=float,
        default=0.8,
        help='Eccentricity of present-day system.'
    )

    result = parser.parse_args()
    if result.create_config:
        print('Creating config file: ' + repr(result.create_config))
        parser.write_config_file(result,
                                 [result.create_config],
                                 exit_after=True)
    logging.basicConfig(
        level=getattr(logging, result.logging_verbosity.upper())
    )
    return result


def get_interpolator(stellar_evolution_interpolator_dir,
                     track_path):
    """Return the stellar evolution interpolator to use."""

    manager = StellarEvolutionManager(
        stellar_evolution_interpolator_dir
    )
    if not list(manager.get_suite_tracks()):
        manager.register_track_collection(
            track_fnames=glob(
                os.path.join(track_path, '*.csv')
            )
        )

    interpolator_args = dict(num_threads=1)
    interpolator_args['new_interp_name'] = 'custom'
    interpolator_args['nodes'] = {
        q: 0 for q in MESAInterpolator.quantity_list
    }
    interpolator_args['smoothing'] = {
        q: float('nan') for q in MESAInterpolator.quantity_list
    }
    return manager.get_interpolator(**interpolator_args)

def check_if_secondary_is_star(system):
    """True iff the secondary in the system should be evolved as a star."""

    return (
        getattr(system, 'Msecondary', system.secondary_mass)
        >
        0.05 * units.M_sun
    )



#Unable to come up with refoctoring which does not decrease readability.
#pylint: disable=too-many-locals
def find_evolution(system,
                   interpolator,
                   dissipation,
                   *,
                   max_age=None,
                   initial_porb=27.3 * units.day,
                   initial_eccentricity=0.0,
                   initial_obliquity=0.0,
                   disk_period=None,
                   disk_dissipation_age=2e-3 * units.Gyr,
                   primary_wind_strength=0.17,
                   primary_wind_saturation=2.78,
                   primary_core_envelope_coupling_timescale=0.05,
                   secondary_wind_strength=0.0,
                   secondary_wind_saturation=100.0,
                   secondary_core_envelope_coupling_timescale=0.05,
                   secondary_disk_period=None,
                   orbital_period_tolerance=1e-6,
                   eccentricity_tolerance=1e-6,
                   obliquity_tolerance=1e-6,
                   period_search_factor=2.0,
                   scaled_period_guess=1.0,
                   eccentricity_upper_limit=0.8,
                   solve=True,
                   max_iterations=49,
                   secondary_is_star=None,
                   **extra_evolve_args):
    """
    Find the evolution of the given system.

    Args:
        system:    The system parameters. Usually parsed using
            read_hatsouth_info.

        interpolator:    See interpolator argument to
            EccentricitySolverCallable.__init__().

        dissipation:    See dissipation argument to
            EccentricitySolverCallable.__init__().

        primary_lgq:    The log10 of the tidal quality factor to assume for the
            primary.

        secondary_lgq:    The log10 of the tidal quality factor to assume for
            the secondary.

        max_age:    The maximum age up to which to calculate the evolution. If
            not specified, defaults to star's lifetime.
        
        initial_porb:    The initial orbital period to start the evolution with
            if we are not attempting to solve for anything.

        initial_eccentricity:    The initial eccentricity to start the evolution
            with. If set to the string 'solve' an attempt is made to find an
            initial eccentricity to reproduce the present day value given in the
            system.

        initial_obliquity:    See same name argument to
            PeriodSolverWrapper.__init__().

        disk_period:    The spin period of the primary star's surface convective
            zone until the secondary appears. If not specified, defaults to
            `system.Pprimary` or the period implied by `system.Vsini`.

        disk_dissipation_age:    The age at which the secondary appears and the
            primary's spin is released.

        secondary_wind_strength:    The wind strength parameter of the
            secondary.

        secondary_disk_period:    The period to which the secondary's surface
            spin is locked until the binary forms. If None, defaults to
            `disk_period`.

        orbital_period_tolerance:    The tolerance to which to find the initial
            orbital period when trying to match the final one.

        eccentricity_tolerance:    The tolerance to which to find the initial
            eccentricity when trying to match the final one.

        obliquity_tolerance:    The tolerance to which to find the initial
            obliquity when trying to match the final one.

        period_search_factor:    See same name argument to
            :meth:`InitialConditionSolver.__init__`

        scaled_period_guess:    See same name argument to
            :meth:`InitialConditionSolver.__init__`

        eccentricity_upper_limit:    The maximum initial eccentricity to try.
            TODO: Currently unused.

        solve:    If False, no attempt is made to find initial orbital period
            and/or eccentricity. Instead, the system parameters are assumed to
            be initial values.
        
        max_iterations:    The maximum number of iterations the solver should
            perform.

        secondary_is_star(bool or None):    Should the secondary object in the
            system be created as a star? If None, this is automatically
            determined based on the object's mass (large mass => star).

        extra_evolve_args:    Any extra arguments to pass to Binary.evolve().

    Returns:
        A structure with attributes containing the evolutions of orbital and
        stellar quantities. See EccentricitySolverCallable._format_evolution()
        for details.
    """

    logger=logging.getLogger(__name__)
    #laststep = [0.5,0.5,True]
    laststep = [scipy.nan,scipy.nan,scipy.nan]
    earlierstep = [scipy.nan,scipy.nan,scipy.nan]

    def errfunc(variable_conditions,
                fixed_conditions,
                initial_secondary_angmom,
                orbital_period_tolerance,
                eccentricity_tolerance,
                obliquity_tolerance,
                solve_type):
        """
        Returns differences between initial and found orbital period,
        eccentricity, and (potentially, if implemented) obliquity, given
        some specified initial conditions.
        """

        if solve_type == "porb":
            error_out = scipy.nan
            porb_i = variable_conditions
            ecc_i  = fixed_conditions[0]
            obliq_i = fixed_conditions[1]
        else:
            error_out = [scipy.nan,scipy.nan]
            if solve_type == "ecc":
                porb_i = variable_conditions[0]
                ecc_i  = variable_conditions[1]
                obliq_i = fixed_conditions
            elif solve_type == "obliq":
                porb_i = variable_conditions[0]
                ecc_i  = fixed_conditions
                obliq_i = variable_conditions[1]
            else:
                raise ValueError("Invalid solve type",0)
        
        initial_conditions = [porb_i,ecc_i,obliq_i]

        porb_true = system.orbital_period
        ecc_true  = system.eccentricity
        #obliq_true = system.obliquity

        # Sanity check
        if (ecc_i < 0 or ecc_i >= 1) or (porb_i < 0 or porb_i > 100) or (obliq_i < 0 or obliq_i >180):
            logger.warning('Invalid Initial Values')
            return error_out

        evolution = value_finder.try_system(initial_conditions,initial_secondary_angmom,max_age)
        porb_found = evolution.orbital_period[-1]
        ecc_found = evolution.eccentricity[-1]
        obliq_found = scipy.nan

        logger.debug('Found porb, ecc, obliq: %f, %f, %f',porb_found,ecc_found,obliq_found)
        logger.debug('Target porb, ecc, obliq: %f, %f, OBLIQUITY NOT YET HANDLED',porb_true.to_value("day"),ecc_true)

        porb_diff = porb_found-porb_true.to_value("day")
        ecc_diff = ecc_found-ecc_true

        logger.debug('porb tolerance vs porb diff: %f, %f',orbital_period_tolerance,porb_diff)
        logger.debug('ecc tolerance vs ecc diff: %f, %f',eccentricity_tolerance,ecc_diff)

        if numpy.logical_or(numpy.isnan(porb_found),(numpy.isnan(ecc_found))):
            logger.warning('Binary system was destroyed')
            return error_out

        if solve_type == "porb":
            difference = porb_diff
            check = orbital_period_tolerance
        elif solve_type == "ecc":
            difference = [porb_diff,ecc_diff]
            check = [orbital_period_tolerance,eccentricity_tolerance]
        else:
            difference = [porb_diff,obliq_found-obliq_i]
            check = [orbital_period_tolerance,obliquity_tolerance]
        
        if numpy.all(numpy.abs(difference) <= check):
            if (not numpy.isnan(earlierstep[0])) and solve_type == "ecc":
                A = numpy.matrix([
                                    [earlierstep[0],earlierstep[2],1],
                                    [laststep[0],laststep[2],1],
                                    [ecc_i,porb_found,1]
                                ])
                B = numpy.matrix([earlierstep[1],laststep[1],ecc_found])
                fit,residual,rnk,s = scipy.linalg.lstsq(A,B.T)
                ehat_prime = [fit[0],ecc_i]
                print('A is ',A)
                print('B is ',B)
            else:
                ehat_prime = [scipy.nan,scipy.nan]
            print('ehat_prime is ',ehat_prime)
            raise ValueError("solver and errfunc() have found initial values with acceptable results",1,evolution,ehat_prime)
        else:
            earlierstep[0] = laststep[0]
            earlierstep[1] = laststep[1]
            earlierstep[2] = laststep[2]
            laststep[0] = ecc_i
            laststep[1] = ecc_found
            laststep[2] = porb_found
            print(difference)
            return difference
        #TODO make sure last three points aren't all on a line?
    def get_period_range(initial_eccentricity):
        """
        Returns a range of initial orbital periods within which the
        difference between the found and correct final periods crosses
        zero.
        """

        period_search_factor = 1.1
        max_porb_initial = 50.0
        porb_min, porb_max = scipy.nan, scipy.nan
        porb_initial = system.orbital_period.to_value("day")
        #TODO better initial obliq
        porb = value_finder.try_system([porb_initial,initial_eccentricity,3],initial_secondary_angmom,max_age).orbital_period[-1]
        if scipy.isnan(porb):
            porb=0.0
        porb_error = porb - system.orbital_period.to_value("day")
        guess_porb_error = porb_error
        step = (period_search_factor if guess_porb_error < 0
                else 1.0 / period_search_factor)

        while (
                porb_error * guess_porb_error > 0
                and
                porb_initial < max_porb_initial
        ):
            if porb_error < 0:
                porb_min = porb_initial
            else:
                porb_max = porb_initial
            porb_initial *= step
            logger.debug(
                (
                    'Before evolution:'
                    '\n\tporb_error = %s'
                    '\n\tguess_porb_error = %s'
                    '\n\tporb_initial = %s'
                    '\n\tporb_min = %s'
                    '\n\tporb_max = %s'
                    '\n\tstep = %s'
                ),
                repr(porb_error),
                repr(guess_porb_error),
                repr(porb_initial),
                porb_min,
                porb_max,
                step
            )
            #TODO better initial obliq
            porb = value_finder.try_system([porb_initial,initial_eccentricity,3],
                                                    initial_secondary_angmom,
                                                    max_age).orbital_period[-1]
            logger.debug('After evolution: porb = %s', repr(porb))
            if scipy.isnan(porb):
                porb=0.0
            porb_error = porb - system.orbital_period.to_value("day")

        if porb==0.0:
            logger.exception("porb is 0")
            raise ValueError("porb is 0",0)

        if porb_error < 0:
            porb_min = porb_initial
        else:
            porb_max = porb_initial
            if porb_error == 0:
                porb_min = porb_initial

        logger.info(
            'For Pdisk = %s, orbital period range: %s < Porb < %s',
            repr(disk_period),
            repr(porb_min),
            repr(porb_max)
        )

        return porb_min, porb_max

    # Sanity check eccentricity and obliquity
    # This is redundant as it's also checked by POET
    if isinstance(initial_eccentricity, str):
        if initial_eccentricity != 'solve':
            logger.error('Invalid initial eccentricity %f',initial_eccentricity)
            raise ValueError("Invalid initial eccentricity")
        elif solve == False:
            logger.error('If we are not solving, initial eccentricity must be a numerical value')
            raise ValueError("Invalid initial eccentricity")
    elif initial_eccentricity < 0 or initial_eccentricity >= 1:
        logger.error('Invalid initial eccentricity %f',initial_eccentricity)
        raise ValueError("Invalid initial eccentricity")
    
    if isinstance(initial_obliquity, str):
        if initial_obliquity != 'solve':
            logger.error('Invalid initial obliquity %f',initial_obliquity)
            raise ValueError("Invalid initial obliquity")
        elif solve == False:
            logger.error('If we are not solving, initial obliquity must be a numerical value')
            raise ValueError("Invalid initial obliquity")
    elif initial_obliquity < 0 or initial_obliquity > numpy.pi:
        logger.error('Invalid initial obliquity %f',initial_obliquity)
        raise ValueError("Invalid initial obliquity")
    
    #False positive
    #pylint: disable=no-member
    if disk_period is None:
        if hasattr(system, 'Pprimary'):
            disk_period = system.Pprimary
        else:
            disk_period = (2.0 * numpy.pi * system.Rsecondary
                           /
                           system.Vsini)

    if secondary_is_star is None:
        secondary_is_star = check_if_secondary_is_star(system)
    if secondary_is_star == False:
        raise ValueError("Code assumes in several places that the secondary is a star.")
    if 'max_time_step' not in extra_evolve_args:
        extra_evolve_args['max_time_step'] = 1e-3
    #pylint: enable=no-member
    
    value_finder = InitialValueFinder(
        system=system,
        interpolator=interpolator,
        #False positive
        #pylint: disable=no-member
        current_age=system.age,
        #pylint: enable=no-member
        disk_period=disk_period,
        initial_obliquity=initial_obliquity,
        disk_dissipation_age=disk_dissipation_age,
        dissipation=dissipation,
        secondary_star=secondary_is_star,
        primary_wind_strength=primary_wind_strength,
        primary_wind_saturation=primary_wind_saturation,
        primary_core_envelope_coupling_timescale=(
            primary_core_envelope_coupling_timescale
        ),
        secondary_wind_strength=secondary_wind_strength,
        secondary_wind_saturation=secondary_wind_saturation,
        secondary_core_envelope_coupling_timescale=(
            secondary_core_envelope_coupling_timescale
        ),
        secondary_disk_period=secondary_disk_period,
        orbital_period_tolerance=orbital_period_tolerance,
        period_search_factor=period_search_factor,
        scaled_period_guess=scaled_period_guess,
        **extra_evolve_args
    )

    initial_guess = [system.orbital_period.to_value("day"),system.eccentricity,3]  #TODO make obliq reflect system
    initial_secondary_angmom = numpy.array(value_finder.get_secondary_initial_angmom())
    #initial_eccentricity='solve'
    error=SimpleNamespace(eccentricity=[scipy.nan],orbital_period=[scipy.nan])
    if solve:
        try:
            if initial_eccentricity == 'solve':
                scipy.optimize.root(
                    errfunc,
                    [initial_guess[0],initial_guess[1]], #TODO: get rid of initial_guess, use independent variables and whatnot
                    method='lm',
                    options={'xtol':0,
                            'ftol':0,
                            'maxiter':max_iterations},
                    args=(initial_guess[2],
                            initial_secondary_angmom,
                            orbital_period_tolerance,
                            eccentricity_tolerance,
                            obliquity_tolerance,
                            "ecc")
                )
            elif initial_obliquity == 'solve':
                scipy.optimize.root(
                    errfunc,
                    [initial_guess[0],initial_guess[2]],
                    method='lm',
                    options={'xtol':0,
                            'ftol':0,
                            'maxiter':max_iterations},
                    args=(initial_guess[1],
                            initial_secondary_angmom,
                            orbital_period_tolerance,
                            eccentricity_tolerance,
                            obliquity_tolerance,
                            "obliq")
                )
            else:
                # Just solving for period
                porb_min, porb_max = get_period_range(initial_eccentricity)
                initial_guess[1] = initial_eccentricity

                scipy.optimize.brentq(
                    errfunc,
                    porb_min,
                    porb_max,
                    xtol=orbital_period_tolerance/10,
                    rtol=orbital_period_tolerance/10,
                    maxiter=max_iterations,
                    args=([initial_guess[1],initial_guess[2]],
                            initial_secondary_angmom,
                            orbital_period_tolerance,
                            eccentricity_tolerance,
                            obliquity_tolerance,
                            "porb")
                )
        except ValueError as err:
            if err.args[1] == 1: # This means we actually completed successfully
                return err.args[2],err.args[3]
            else:
                logger.exception('Solver Crashed')
                raise
        except RuntimeError as err: # TODO: have a first variable appropriately formatted for all the evolution things
            logger.exception('Solver failed to converge. Error: %s',err)
            logger.exception('Here is what we are returning: %s, %s',error,scipy.nan)
            return error,[scipy.nan,scipy.nan]
    else:
        try:
            return value_finder.try_system(
                [initial_porb.to_value("day"),initial_eccentricity,initial_obliquity],
                initial_secondary_angmom,
                max_age
            )
        except:
            logger.exception('Something went wrong while trying to evolve the system with the given parameters.')
            raise

    # If we get to this point, we tried to solve but didn't get a solution
    # for some reason.
    logger.error("Solver failed to converge.")
    return error,[scipy.nan,scipy.nan]
#pylint: enable=too-many-locals

def test_find_evolution_parallel(test_set,**kwargs):

    initial_eccentricity = test_set[0]
    initial_obliquity = test_set[1]
    solve = test_set[2]
    secondary_is_star = test_set[3]

    logger=logging.getLogger(__name__)
    logger.warning("We are starting a new round in this process.")
    logger.warning("Test set: %s",test_set)

    # Update kwargs with the current combination of parameters.
    kwargs['initial_eccentricity'] = initial_eccentricity
    kwargs['initial_obliquity'] = initial_obliquity
    kwargs['solve'] = solve
    kwargs['secondary_is_star'] = secondary_is_star

    # Run the function.
    try:
        resultA = find_evolution(**kwargs)
        if hasattr(resultA, 'orbital_period') and hasattr(resultA, 'eccentricity'):
            result = resultA.orbital_period[-1],resultA.eccentricity[-1]
        else:
            result = resultA
        logger.warning("Result: %s",result)
    except Exception as e:
        logger.warning("Crash: %s",e)
        result = e
    except:
        logger.warning("Oops, looks like that one crashed in a weirder way than normal!")
        result = "CRASH"

    return (test_set,result)

def test_find_evolution(**kwargs):
    """
    Test a bespoke selection of values for the find_evolution() function.
    """

    initial_eccentricity_values = ['solve', -0.1, 0.0, kwargs['system'].eccentricity, 0.99, 1.0]
    initial_obliquity_values = ['solve', -1.0*numpy.pi/180, 0.0*numpy.pi/180, 90*numpy.pi/180, 180*numpy.pi/180, 181*numpy.pi/180]
    solve_values = [True, False]
    secondary_is_star_values = [None, True]#, False]

    all_iteration_sets = []

    for initial_eccentricity in initial_eccentricity_values:
        for initial_obliquity in initial_obliquity_values:
            for solve in solve_values:
                for secondary_is_star in secondary_is_star_values:
                    all_iteration_sets.append([initial_eccentricity,initial_obliquity,solve,secondary_is_star])

    processNum = 6
    test_it = partial(test_find_evolution_parallel,**kwargs)
    output=numpy.array(())
    with Pool(processNum,setup_process) as p:
        output=p.map(test_it,all_iteration_sets)
    
    # Save the output to a file.
    numpy.savetxt("test_find_evolution_output_3.txt",output,fmt="%s")

def test_ehatprime(e,**kwargs):
    logger=logging.getLogger(__name__)
    kwargs['initial_eccentricity'] = e
    try:
        evolution = find_evolution(**kwargs)[0]
        print('evolution is ',evolution)
    except:
        logger=logging.getLogger(__name__)
        logger.exception('Something went wrong while trying to evolve the system with the given parameters.')
        logger.exception('Here is what those parameters are: %s',kwargs)
        raise
    result_ef = evolution.eccentricity[-1]
    return result_ef

if __name__ == '__main__':
    config = parse_command_line()

    kwargs = dict(
        system=SimpleNamespace(
            primary_mass=config.primary_mass * units.M_sun,
            secondary_mass=config.secondary_mass * units.M_sun,
            feh=config.metallicity,
            orbital_period=config.orbital_period * units.day,
            age=config.final_age * units.Gyr,
            eccentricity=config.eccentricity
        ),
        interpolator=set_up_library(config),
        dissipation=get_poet_dissipation_from_cmdline(config),
        max_age=config.final_age * units.Gyr,
        disk_period=config.disk_lock_period * units.day,
        disk_dissipation_age=config.disk_dissipation_age * units.Gyr,
        primary_core_envelope_coupling_timescale=(
            config.primary_diff_rot_coupling_timescale * units.Gyr
        ),
        secondary_core_envelope_coupling_timescale=(
            config.primary_diff_rot_coupling_timescale
            if config.secondary_diff_rot_coupling_timescale is None else
            config.secondary_diff_rot_coupling_timescale
        ) * units.Gyr,
        secondary_wind_strength=(
            config.primary_wind_strength
            if config.secondary_wind_strength is None else
            config.secondary_wind_strength
        ),
        primary_wind_saturation=config.primary_wind_saturation_frequency,
        secondary_wind_saturation=(
            config.primary_wind_saturation_frequency
            if config.secondary_wind_saturation_frequency is None else
            config.secondary_wind_saturation_frequency
        ),
        required_ages=None,
        eccentricity_expansion_fname=(
            config.eccentricity_expansion_fname.encode('ascii')
        ),
        timeout=config.max_evolution_runtime
    )
    for param in ['initial_eccentricity',
                  'initial_obliquity',
                  'primary_wind_strength',
                  'orbital_period_tolerance',
                  'period_search_factor',
                  'scaled_period_guess',
                  'max_time_step',
                  'precision']:
        kwargs[param] = getattr(config, param)

    testing = "Plot"
    if testing == "False":
        evolution = find_evolution(**kwargs)
        with open(config.output_pickle, 'ab') as outf:
            pickle.dump(evolution, outf)
    elif testing == "Plot":
        # set of e_is to do first
        e_min = kwargs['system'].eccentricity - 0.3
        if e_min < 0.0:
            e_min = 0.0
        e_max = kwargs['system'].eccentricity + 0.3
        if e_max > 0.85:
            e_max = 0.85
        ei_values = numpy.linspace(e_min,e_max,100)
        print('ei_values, ',ei_values)
        #raise ValueError("We're not done yet!")
        eccf_list = numpy.array(())
        output=numpy.array(())
        test_it = partial(test_ehatprime,**kwargs)

        processNum = 16
        with Pool(processNum,setup_process) as p:
            output=p.map(test_it,ei_values)
        #output=test_it(e_min)
        
        print('Full output is ', output)
        for i in range(len(output)):
            eccf_list = numpy.append(eccf_list,output[i])
        new_eccf_list = eccf_list[::10]
        print('eccf_list is ',eccf_list)
        print('new_eccf_list is ',new_eccf_list)

        # Now we're going to feed all of these into the solver
        kwargs['initial_eccentricity'] = 'solve'
        points = numpy.array(()) #TODO This part should also be parallel processed
        ehat_prime_values = numpy.array(())
        for e in new_eccf_list:
            kwargs['system'].eccentricity = e
            evolution,ehat_prime = find_evolution(**kwargs)
            ecc_found = evolution.eccentricity[-1]
            ecc_i = ehat_prime[1]
            points = numpy.append(points,[ecc_i,ecc_found])
            ehat_prime_values = numpy.append(ehat_prime_values,ehat_prime[0])
        print(points)
        print(ehat_prime_values)
        print("Finished!")
    else:
        test_find_evolution(**kwargs)
