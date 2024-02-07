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
from astropy import units, constants
from configargparse import ArgumentParser, DefaultsFormatter

from stellar_evolution.library_interface import MESAInterpolator
from stellar_evolution.manager import StellarEvolutionManager
from orbital_evolution.transformations import phase_lag
from basic_utils import calc_orbital_angular_momentum,calc_semimajor,calc_orbital_frequency

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

def triangle_area(A, B):
        return (1/2) * (A[0]*(B[1] - B[2] ) + A[1]*(B[2] - B[0] ) + A[2]*(B[0] - B[1]))

#Unable to come up with refoctoring which does not decrease readability.
#pylint: disable=too-many-locals
def find_evolution(system,
                   interpolator,
                   dissipation,
                   *,
                   initial_porb=27.3 * units.day,
                   initial_eccentricity=0.0,
                   initial_obliquity=0.0,
                   disk_period=None,
                   disk_dissipation_age=2e-3 * units.Gyr,
                   primary_wind_strength=0.17,
                   primary_wind_saturation=2.78,
                   primary_core_envelope_coupling_timescale=0.05 * units.Gyr,
                   secondary_wind_strength=0.0,
                   secondary_wind_saturation=100.0,
                   secondary_core_envelope_coupling_timescale=0.05 * units.Gyr,
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
                   carepackage = None,
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
    laststeps = []
    past_diffs = [scipy.nan,scipy.nan,scipy.nan]

    def errfunc(variable_conditions,
                fixed_conditions,
                initial_secondary_angmom,
                orbital_period_tolerance,
                eccentricity_tolerance,
                obliquity_tolerance,
                solve_type,
                search_porb = None):
        """
        Returns differences between initial and found orbital period,
        eccentricity, and (potentially, if implemented) obliquity, given
        some specified initial conditions.
        """
        def report_target():
            if solve_type == "porb":
                logger.debug('Target porb: %f',porb_true)
                print('Target porb: ',porb_true)
            elif solve_type == "ecc":
                logger.debug('Target porb, ecc: %f, %f',porb_true,ecc_true)
                print('Target porb, ecc: ',porb_true,ecc_true)
            else: #TODO
                logger.debug('Target porb, obliq: %f, OBLIQUITY NOT YET HANDLED',porb_true)
                print('Target porb, obliq: ',porb_true,' OBLIQUITY NOT YET HANDLED')
        
        def get_initial_values():
            dL = scipy.nan
            if solve_type == "porb":
                error_out = scipy.nan
                porb_i = variable_conditions
                ecc_i  = fixed_conditions[0]
                obliq_i = fixed_conditions[1]
                thetype = '1d'
            else:
                error_out = [scipy.nan,scipy.nan]
                if solve_type == "ecc":
                    thetype = '2d'
                    dL = variable_conditions[0]
                    ecc_i  = variable_conditions[1]
                    obliq_i = fixed_conditions
                    print(system.primary_mass.to(units.M_sun).value,system.secondary_mass.to(units.M_sun).value,ecc_true,porb_true,ecc_i,dL)
                    # If ecc_i is weird we're going to catch it later; for now,
                    # keep the error in porb the same as the previous step
                    porb_i = laststeps[-1][2] if (ecc_i < 0 or ecc_i > 0.8) else \
                        unchange_variables(system.primary_mass.to(units.M_sun).value,system.secondary_mass.to(units.M_sun).value,ecc_true,porb_true,ecc_i,dL)[0]
                    error_out = [(porb_i.real-porb_true)*porb_sign,(ecc_i-ecc_true)*ecc_sign]
                elif solve_type == "obliq": #TODO: change of variables?
                    porb_i = variable_conditions[0]
                    ecc_i  = fixed_conditions
                    obliq_i = variable_conditions[1]
                else:
                    raise ValueError("Invalid solve type",0)
            return dL,porb_i,ecc_i,obliq_i,thetype,error_out
        def sanity_checks():
            error_out = None
            if (numpy.isnan(ecc_i) or numpy.iscomplex(ecc_i) or 
                numpy.isnan(porb_i) or numpy.iscomplex(porb_i) or 
                numpy.isnan(obliq_i) or numpy.iscomplex(obliq_i)): # TODO: Obliquity not implemented
                logger.warning('ecc_i, porb_i, or obliq_i is nan or imaginary: %f,%f,%f',
                            ecc_i,porb_i,obliq_i)
                porb_wrong = porb_sign * numpy.inf
                ecc_wrong = ecc_sign * numpy.inf
                obliq_wrong = numpy.inf
                wrong = porb_wrong if solve_type == "porb" else [porb_wrong,ecc_wrong] if solve_type == "ecc" else [porb_wrong,obliq_i-3.0]
                logger.debug('Returning the following value(s) to the solver: %s',repr(wrong))
                print('Returning the following values to the solver: ',wrong)
                report_target()
                return wrong
            if (ecc_i < 0 or ecc_i > 0.8) or (porb_i < 0) or (obliq_i < 0 or obliq_i > 180):
                logger.warning('Invalid Initial Value(s)')
                logger.warning('Initial porb, ecc, obliq: %f, %f, %f',porb_i,ecc_i,obliq_i)
                print('Invalid Initial Value(s)')
                print('Initial porb, ecc, obliq: ',porb_i,ecc_i,obliq_i)
                invalid = (porb_i.real-porb_true)*porb_sign if solve_type == "porb" else \
                            error_out if solve_type == "ecc" else \
                            [porb_i.real-porb_true,obliq_i-3.0]
                logger.debug('Returning the following value(s) to the solver: %s',repr(invalid))
                print('Returning the following values to the solver: ',invalid)
                report_target()
                return invalid
            return error_out

        def find_ehat_prime():
            def generate_points():
                point_a_period = porb_found + delta_p
                print('point_a_period is ',point_a_period)
                print('Solving for initial period that results in the new final period, so that we can find the new final eccentricity.')
                second_point = solve_for_point(ecc_i,point_a_period,obliq_i)
                print('second_point is ',second_point)

                point_b_ecc = ecc_i - (alpha*delta_p)
                print('point_b_ecc is ',point_b_ecc)
                if point_b_ecc < 0:
                    print('point_b_ecc is negative. Setting to 0. Final eccentricity must also be zero.')
                    point_b_ecc = 0
                    third_point = [0,0,porb_found,scipy.nan]
                else:
                    third_point = solve_for_point(point_b_ecc,porb_found,obliq_i)
                print('third_point is ',third_point)
                return second_point,third_point
            def search_for_points():
                second_point = []
                third_point = []
                breakloop = False
                i = 0
                while not breakloop and i < len(acceptable_points):
                    # For all acceptable points after point i (j)
                    for j in range(i+1,len(acceptable_points)):
                        # If the resulting triangle area (in ei vs pf) is acceptable
                        print('Checking triangle area for ',acceptable_points[i],' and ',acceptable_points[j])
                        tri_area = triangle_area([ecc_i,acceptable_points[i][0],acceptable_points[j][0]],[porb_found,acceptable_points[i][2],acceptable_points[j][2]])
                        print('Triangle area is ',tri_area)
                        if (tri_max > numpy.abs(tri_area) > tri_min
                        ):
                            # Use the results for the two other points
                            print('Triangle area is acceptable.')
                            second_point = acceptable_points[i]
                            print('second_point is ',second_point)
                            third_point = acceptable_points[j]
                            print('third_point is ',third_point)
                            # Break out of the loop
                            breakloop = True
                            break
                    i += 1
                return second_point,third_point
            def find_perpendicular_point():
                ei_two = second_point[0]
                pf_two = second_point[2]
                distance = numpy.sqrt((alpha*(ecc_i-ei_two))**2 + (porb_found-pf_two)**2)
                print('distance is ',distance)
                del_e = alpha*(ei_two-ecc_i)
                del_p = pf_two-porb_found
                print('Delta e is ',del_e)
                print('Delta p is ',del_p)
                # Always choose the new target coordinates with largest p_f
                rot_sign = -1 if del_e > 0 else 1
                ei_three = ecc_i + rot_sign * del_p / alpha
                pf_three = porb_found - rot_sign * del_e
                if ei_three < 0:
                    ei_three = 0
                    print('ei_three is negative. Setting to 0. Final eccentricity must also be zero.')
                    third_point = [0,0,porb_found,scipy.nan]
                else:
                    if ei_three > 0.8:
                        print('ei_three is ',ei_three,' which is too large. Setting to 0.8.')
                        ei_three = 0.8
                    print('ei_three is ',ei_three,' and pf_three is ',pf_three)
                    print('New distance is ',numpy.sqrt((alpha*(ei_three-ecc_i))**2 + (pf_three-porb_found)**2))
                    # Solve for initial period that results in the new final period, so that we can
                    # find the new final eccentricity
                    print('Solving for initial period that results in the new final period, so that we can find the new final eccentricity.')
                    third_point = solve_for_point(ei_three,pf_three,obliq_i)
                print('third_point is ',third_point)
                return third_point
            def calculate_ehat_prime():
                A = [
                        [ecc_i,porb_found,1],
                        [second_point[0],second_point[2],1],
                        [third_point[0],third_point[2],1]
                    ]
                B = [ecc_found,second_point[1],third_point[1]]
                print('A is ',A)
                print('B is ',B)
                tri_area = triangle_area([row[0] for row in A],[row[1] for row in A])
                print('Triangle area is ',tri_area)
                #if not (tri_max > numpy.abs(tri_area) > tri_min):
                #    print('Unable to find two points that avoid a degenerate solution.')             TODO
                #    raise ValueError("Unable to find two points that avoid a degenerate solution.",0)TODO

                # Perform least squares fit to find ehat_prime
                print('Performing least squares fit to find ehat_prime...')
                A = numpy.matrix(A)
                B = numpy.matrix(B)
                fit = scipy.linalg.lstsq(A,B.T)[0]
                ehat_prime = [fit[0],ecc_i]
                print('A is ',A)
                print('B is ',B)
                print('ehat_prime is ',ehat_prime)
                return ehat_prime
            delta_p = 0.01*porb_found   #TODO: proper logger statements
            alpha = 0.5
            tri_max = 0.25
            tri_min = 1e-8
            max_dist = 0.1
            min_dist = 0.001
            second_point = []
            third_point = []
            acceptable_points = []
            print('There are ',len(laststeps),' point(s) in laststeps.')
            # For all points before the most recent point
            for i in range(len(laststeps)):
                # Get the distance between the most recent point and the current point in ei,pf space
                distance = numpy.sqrt((alpha*(ecc_i-laststeps[i][0]))**2 + (porb_found-laststeps[i][2])**2)
                print('Distance between ',laststeps[i],' and ',[ecc_i,ecc_found,porb_found,porb_i],' is ',distance)
                # If the point is neither too close to nor too far from the most recent point
                if max_dist > distance > min_dist:
                    # Keep it
                    acceptable_points.append(laststeps[i])
            print('Removed ',len(laststeps)-len(acceptable_points),' point(s).')
            # If no points remain after filtering
            if len(acceptable_points) == 0:
                # Find two points in ei,pf space, and solve to find the initial period which
                # results in the new final period, so that we can use the new final eccentricity
                # We want larger period and smaller eccentricity vs. the most recent point
                print('No acceptable points remain after filtering.')
                second_point,third_point = generate_points()
            # Otherwise, points did remain, so let's work with that
            else:
                # For all acceptable points before the most recent point i
                print('Acceptable points remain after filtering.')
                acceptable_points.reverse()
                print('Acceptable points are ',acceptable_points)
                second_point,third_point = search_for_points()
                # If we don't have two identified points
                print('second_point is ',second_point)
                print('third_point is ',third_point)
                if len(second_point) == 0 or len(third_point) == 0:
                    print('Unable to find two points that avoid a degenerate solution.')
                    # Grab most recent acceptable point
                    second_point = acceptable_points[0]
                    print('second_point is ',second_point)
                    # Find another point perpendicular to the line between the most recent acceptable point and
                    # the most recent point in ei,pf space
                    third_point = find_perpendicular_point()
            # Final check to make sure the triangle area is acceptable
            print('second_point is ',second_point)
            print('third_point is ',third_point)
            return calculate_ehat_prime()

        porb_true = search_porb if search_porb is not None else system.orbital_period.to_value("day")
        ecc_true  = system.eccentricity
        #obliq_true = system.obliquity

        thetype = None

        porb_sign = past_diffs[0] if not numpy.isnan(past_diffs[0]) else 1.0
        ecc_sign = past_diffs[1] if not numpy.isnan(past_diffs[1]) else 1.0
        logger.debug('porb_sign: %f',porb_sign)
        logger.debug('ecc_sign: %f',ecc_sign)

        dL,porb_i,ecc_i,obliq_i,thetype,error_out = get_initial_values()
            
        logger.debug('Here are the input values the solver is trying.')
        logger.debug('porb_i: %f',porb_i)
        logger.debug('ecc_i: %f',ecc_i)
        logger.debug('obliq_i: %f',obliq_i)
        logger.debug('dL: %f',dL)
        print('Here are the input values the solver is trying.')
        print('porb_i: ',porb_i)
        print('ecc_i: ',ecc_i)
        print('obliq_i: ',obliq_i)
        print('dL: ',dL)
        
        initial_conditions = [porb_i,ecc_i,obliq_i]

        # Sanity checks
        error = sanity_checks()
        if not error is None:
            return error
        
        try:
            evolution = value_finder.try_system(initial_conditions,initial_secondary_angmom,
                                                thetype,
                                                carepackage)
        except AssertionError as err:
            logger.exception('AssertionError: %s',err)
            print('AssertionError: ',err)
            logger.debug('Returning the following values to the solver: %s',repr(error_out))
            print('Returning the following values to the solver: ',error_out)
            return error_out
        except:
            logger.exception('Something unknown went wrong while trying to evolve the system with the given parameters: %s',repr(initial_conditions))
            raise
        porb_found = evolution.orbital_period[-1]
        ecc_found = evolution.eccentricity[-1]
        obliq_found = scipy.nan

        logger.debug('Found porb, ecc, obliq: %f, %f, %f',porb_found,ecc_found,obliq_found)
        logger.debug('Target porb, ecc, obliq: %f, %f, OBLIQUITY NOT YET HANDLED',porb_true,ecc_true)

        porb_diff = porb_found-porb_true
        ecc_diff = ecc_found-ecc_true
        past_diffs[0] = porb_diff / numpy.abs(porb_diff)
        past_diffs[1] = ecc_diff / numpy.abs(ecc_diff)

        logger.debug('porb tolerance vs porb diff: %f, %f',orbital_period_tolerance,porb_diff)
        logger.debug('ecc tolerance vs ecc diff: %f, %f',eccentricity_tolerance,ecc_diff)

        if numpy.isnan(porb_found):
            logger.warning('porb_found is nan')
            porb_found = 0.0
        if numpy.isnan(ecc_found):
            logger.warning('ecc_found is nan')
            ecc_found = 0.0

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
            if len(laststeps) >= 3 and solve_type == "ecc":
                print('laststeps is ',laststeps)
                ehat_prime = find_ehat_prime()
            else:
                ehat_prime = [scipy.nan,scipy.nan]
            print('ehat_prime is ',ehat_prime)
            raise ValueError("solver and errfunc() have found initial values with acceptable results",1,evolution,ehat_prime)
        else:
            laststeps.append([ecc_i,ecc_found,porb_found,porb_i])
            print('difference: ',difference)
            return difference

    def solve_for_point(ecc,porb,obliq):
        porb_min, porb_max = get_period_range(ecc,porb)
        print('porb_min is ',porb_min)
        print('porb_max is ',porb_max)
        try:
            scipy.optimize.brentq(
                errfunc,
                porb_min,
                porb_max,
                xtol=orbital_period_tolerance/100,
                rtol=orbital_period_tolerance/100,
                maxiter=max_iterations,
                args=([ecc,obliq],
                        initial_secondary_angmom,
                        orbital_period_tolerance,
                        eccentricity_tolerance,
                        obliquity_tolerance,
                        "porb",
                        porb),
                full_output=True
            )
        except ValueError as err:
            try:
                length_of_args = len(err.args)
            except:
                # If that doesn't work then it's not one of our custom errors, so something weird happened
                logger.exception('err.args had no len()')
                length_of_args = 0
            if length_of_args >= 2 and err.args[1] == 1: # Assume it's one of our errors, and that we actually completed successfully
                # Use the results for the requested point
                return [ecc,err.args[2].eccentricity[-1],porb,err.args[2].orbital_period[0]]
            else:
                logger.exception('Solver crashed. Error: %s',err)
                raise
        except:
            logger.exception('Solver crashed.')
            raise
        logger.error("Solver failed to converge.")
        raise ValueError("Solver failed to converge.",0)

    def get_period_range(initial_eccentricity,search_porb = None):
        """
        Returns a range of initial orbital periods within which the
        difference between the found and correct final periods crosses
        zero.
        """

        period_search_factor = 1.1
        max_porb_initial = 200.0
        porb_min, porb_max = scipy.nan, scipy.nan
        porb_correct = search_porb if search_porb is not None else system.orbital_period.to_value("day")
        porb_initial = porb_correct * 3
        obliq_i = 0.0
        try:
            porb = value_finder.try_system([porb_initial,initial_eccentricity,obliq_i],initial_secondary_angmom,
                                           '1d',
                                           carepackage).orbital_period[-1]
        except AssertionError as err:
            logger.exception('AssertionError: %s',err)
            porb=0.0
        except:
            logger.exception('Something unknown went wrong while trying to evolve the system with the given parameters.')
            raise
        if numpy.isnan(porb):
            porb=0.0
        porb_error = porb - porb_correct
        guess_porb_error = porb_error
        step = (period_search_factor if guess_porb_error < 0
                else 1.0 / period_search_factor)

        logger.debug('Relevant parameters before starting period range loop:')
        logger.debug('porb_error: %f',porb_error)
        logger.debug('guess_porb_error: %f',guess_porb_error)
        logger.debug('porb_initial: %f',porb_initial)
        logger.debug('porb_min: %f',porb_min)
        logger.debug('porb_max: %f',porb_max)
        logger.debug('step: %f',step)
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
            try:
                porb = value_finder.try_system([porb_initial,initial_eccentricity,obliq_i],
                                                        initial_secondary_angmom,
                                                        '1d',
                                                        carepackage).orbital_period[-1]
            except AssertionError as err:
                logger.exception('AssertionError: %s',err)
                porb=0.0
            except:
                logger.exception('Something unknown went wrong while trying to evolve the system with the given parameters.')
                raise
            logger.debug('After evolution: porb = %s', repr(porb))
            if numpy.isnan(porb):
                porb=0.0
            porb_error = porb - porb_correct

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
        
        if numpy.isnan(porb_min) or numpy.isnan(porb_max):
            logger.exception("porb_min or porb_max is nan")
            raise ValueError("porb_min or porb_max is nan",0)

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

    initial_guess = [system.orbital_period.to_value("day"),system.eccentricity,0.0]  #TODO make obliq reflect system
    logger.debug('Old version of initial guess: %s',repr(initial_guess))
    #initial_guess = [10,0.3,3]
    newe = 0.5#system.eccentricity*2
    #if newe>0.8:
    #    newe=0.8
    initial_guess = [system.orbital_period.to_value("day")*2,newe,0.0]
    logger.debug('New version of initial guess: %s',repr(initial_guess))
    initial_secondary_angmom = numpy.array(value_finder.get_secondary_initial_angmom())
    #initial_eccentricity='solve'
    error=SimpleNamespace(eccentricity=[scipy.nan],orbital_period=[scipy.nan])
    if solve:
        try:
            if initial_eccentricity == 'solve': #TODO: make the order of p and e in (un)change_variables() match their order in other places
                print(system.primary_mass.to(units.M_sun).value,system.secondary_mass.to(units.M_sun).value,system.eccentricity,system.orbital_period.to_value("day"),initial_guess[1],initial_guess[0])
                pguess,eguess = change_variables(system.primary_mass.to(units.M_sun).value,system.secondary_mass.to(units.M_sun).value,system.eccentricity,system.orbital_period.to_value("day"),initial_guess[1],initial_guess[0])
                initial_guess[0] = 0#pguess
                initial_guess[1] = eguess
                print(scipy.optimize.root(
                    errfunc,
                    [initial_guess[0],initial_guess[1]], #TODO: get rid of initial_guess, use independent variables and whatnot
                    method='hybr',
                    options={'xtol':0,
                            'ftol':0,
                            'maxiter':max_iterations},
                    args=(initial_guess[2],
                            initial_secondary_angmom,
                            orbital_period_tolerance,
                            eccentricity_tolerance,
                            obliquity_tolerance,
                            "ecc")
                ) )
            elif initial_obliquity == 'solve':
                scipy.optimize.root(
                    errfunc,
                    [initial_guess[0],initial_guess[2]],
                    method='hybr',
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

                print( scipy.optimize.brentq(
                    errfunc,
                    porb_min,
                    porb_max,
                    xtol=orbital_period_tolerance/100,
                    rtol=orbital_period_tolerance/100,
                    maxiter=max_iterations,
                    args=([initial_guess[1],initial_guess[2]],
                            initial_secondary_angmom,
                            orbital_period_tolerance,
                            eccentricity_tolerance,
                            obliquity_tolerance,
                            "porb"),
                    full_output=True
                ) )
        except ValueError as err:
            try:
                length_of_args = len(err.args)
            except:
                # If that doesn't work then it's not one of our custom errors, so something weird happened
                logger.exception('err.args had no len()')
                length_of_args = 0
            if length_of_args >= 2: # Assume it's one of our errors
                if err.args[1] == 1: # This means we actually completed successfully
                    return err.args[2],err.args[3]
            logger.exception('Solver crashed. Error: %s',err)
            return error,[scipy.nan,scipy.nan]
        except RuntimeError as err: # TODO: have a first variable appropriately formatted for all the evolution things
            logger.exception('Solver failed to converge. Error: %s',err)
            return error,[scipy.nan,scipy.nan]
    else:
        try:
            return value_finder.try_system(
                [initial_porb.to_value("day"),initial_eccentricity,initial_obliquity],
                initial_secondary_angmom
            ),[scipy.nan,scipy.nan]
        except:
            logger.exception('Something went wrong while trying to evolve the system with the given parameters.')
            raise

    # If we get to this point, we tried to solve but didn't get a solution
    # for some reason.
    logger.error("Solver failed to converge.")
    return error,[scipy.nan,scipy.nan]
#pylint: enable=too-many-locals

def change_variables(m1,m2,ef,pf,ei,pi): #TODO: documentation
    semimajorF = calc_semimajor(m1,m2,pf)
    angmomF=calc_orbital_angular_momentum(m1,m2,semimajorF,ef)

    semimajorI = calc_semimajor(m1,m2,pi)
    angmomI=calc_orbital_angular_momentum(m1,m2,semimajorI,ei)

    dL = angmomF-angmomI

    return dL,ei

def unchange_variables(m1,m2,ef,pf,ei,dL):
    semimajorF = calc_semimajor(m1,m2,pf)
    angmomF=calc_orbital_angular_momentum(m1,m2,semimajorF,ef)

    angmomI = angmomF-dL
    pi = angular_momentum_to_p(m1,m2,ei,angmomI)

    return pi,ei

def angular_momentum_to_p(m1, m2, eccentricity, angmom):
    denom = (constants.G.to(units.R_sun**3 / (units.M_sun * units.day**2)) * (m1 + m2) * units.M_sun)
    semimajor = ((angmom * (m1 + m2) * (units.R_sun**2 / units.day) / (m1 * m2 * (1.0 - eccentricity**2)**0.5)))**2 / denom
    semimajor = semimajor.to(units.R_sun).value

    p = semimajor_to_period(m1, m2, semimajor)

    return p

def semimajor_to_period(m1, m2, semimajor):
    denom = constants.G.to(units.m**3 / (units.M_sun * units.day**2)) * (m1 + m2) * units.M_sun
    orbital_period = ( ( (semimajor * units.R_sun).to(units.m)**3.0 * (4.0 * numpy.pi**2) / denom )**(1.0 / 2.0) ).to(units.day).value
    return orbital_period

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

def test_ehatprime_2(e,**kwargs):
    kwargs['system'].eccentricity = e
    evolution,ehat_prime = find_evolution(**kwargs)
    ecc_found = evolution.eccentricity[-1]
    ecc_i = ehat_prime[1]
    return [ecc_i,ecc_found],ehat_prime[0]

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

    testing = "False"
    if testing == "False": #TODO: calculation time limit
        evolution = find_evolution(**kwargs)
        with open(config.output_pickle, 'ab') as outf:
            pickle.dump(evolution, outf)
    elif testing == "Plot":
        # set of e_is to do first
        #e_min = kwargs['system'].eccentricity - 0.3
        #if e_min < 0.0:
        #    e_min = 0.0
        #e_max = kwargs['system'].eccentricity + 0.3
        #if e_max > 0.85:
        #    e_max = 0.85
        e_min = 0.0
        e_max = 0.8
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
            if not numpy.isnan(output[i]):
                eccf_list = numpy.append(eccf_list,output[i])
        new_eccf_list = eccf_list[::10]
        #new_eccf_list = numpy.array(([0.0,0.00051171,0.00102366,0.00153612,0.00204937,0.00256379,0.00307979,0.00359789,0.00411873,0.00464744]))
        #new_eccf_list = numpy.array(([0.0,0.00095875,0.00192265,0.00289841,0.0041469,0.00589061, 0.0083251,0.0117726,0.0175089,0.02718144]))
        ####new_eccf_list = numpy.array(([0.,0.07298025,0.14444735,0.21265355,0.2856158,0.3490391,0.40744892,0.45674613,0.50566563]))
        #new_eccf_list = numpy.array(([0.0,0.07127826,0.14076185,0.20645556,0.26625803,0.32296499,0.37638409,0.42099214,0.46187062,0.498873]))
        #new_eccf_list = numpy.array(([0.0,0.06978789,0.13766322,0.20162655,0.26790725,0.32907037,0.38097794,0.42695883,0.4657867,0.50031818]))
        #new_eccf_list = numpy.array(([0.0,0.00095875,0.00192265,0.00289841,0.0041469,0.00589061,0.0083251,0.0117726,0.0175089,0.02718144]))
        print('eccf_list is ',eccf_list)
        print('new_eccf_list is ',new_eccf_list)

        # Now we're going to feed all of these into the solver
        kwargs['initial_eccentricity'] = 'solve'
        points = numpy.array(()) #TODO This part should also be parallel processed
        ehat_prime_values = numpy.array(())
        output=numpy.array(())
        test_it_2 = partial(test_ehatprime_2,**kwargs)
        processNum = 10
        with Pool(processNum,setup_process) as p:
            output=p.map(test_it_2,new_eccf_list)
        print('Full output is ', output)
        print('output[0] is ',output[0])
        print('output[1] is ',output[1])
        for i in range(len(output)):
            points = numpy.append(points,output[i][0])
            ehat_prime_values = numpy.append(ehat_prime_values,output[i][1])
        print(points)
        print(ehat_prime_values)
        print("Finished!")
    elif testing == "AAA":
        e_initial = 0.8
        kwargs['initial_eccentricity'] = e_initial
        Q = 5.0
        print('kwargs is ',kwargs)
        print('Current Q is ',Q)
        kwargs['dissipation']['primary']['reference_phase_lag'] = phase_lag(Q)
        kwargs['dissipation']['secondary']['reference_phase_lag'] = phase_lag(Q)
        print('given dissipation is ',kwargs['dissipation'])
        final_e=find_evolution(solve=False,**kwargs)[0].eccentricity[-1]
        while numpy.isnan(final_e) or final_e < 0.2:
            Q = Q + 0.5
            print('Current Q is ',Q)
            kwargs['dissipation']['primary']['reference_phase_lag'] = phase_lag(Q)
            kwargs['dissipation']['secondary']['reference_phase_lag'] = phase_lag(Q)
            print('given dissipation is ',kwargs['dissipation'])
            final_e=find_evolution(solve=False,**kwargs)[0].eccentricity[-1]
            print('Resulting e is ',final_e)
        print('Final Q is ',Q)
        print('final_e is ',final_e)
    else:
        test_find_evolution(**kwargs)
