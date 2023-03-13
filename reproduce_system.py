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
                   period_search_factor=2.0,
                   scaled_period_guess=1.0,
                   eccentricity_upper_limit=0.8,
                   solve=True,
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

        initial_eccentricity:    The initial eccentricity to star the evolution
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
            orbital preiod when trying to match the final one.

        period_search_factor:    See same name argument to
            :meth:`InitialConditionSolver.__init__`

        scaled_period_guess:    See same name argument to
            :meth:`InitialConditionSolver.__init__`

        eccentricity_upper_limit:    The maximum initial eccentricity to try.

        solve:    If False, no attempt is made to find initial orbital period
            and/or eccentricity. Instead, the system parameters are assumed to
            be initial values.

        secondary_is_star(bool or None):    Should the secondary object in the
            system be created as a star? If None, this is automatically
            determined based on the object's mass (large mass => star).

        extra_evolve_args:    Any extra arguments to pass to Binary.evolve().

    Returns:
        A structure with attributes containing the evolutions of orbital and
        stellar quantities. See EccentricitySolverCallable._format_evolution()
        for details.
    """

    def errfunc(initial_conditions,
                #targets,
                value_finder):
        """Returns differences for all three values."""

        #porb_i =
        #ecc_i  = initial_eccentricity
        #obliq_i = initial_obliquity

        porb_true = system.Porb
        ecc_true  = system.eccentricity
        obliq_true = system.obliquity #targets[2]

        # Sanity check
        if ecc_i < 0 or ecc_i >= 1:
            #complain
        if porb_probs:
            #complain
        if pbliq_pvib:
            #complain
        
        #more stuff

        porb_found,ecc_found,obliq_found = value_finder.try_system(initial_conditions)

        if numpy.logical_or(numpy.isnan(porb_found),(numpy.isnan(ecc_found))):
            self.logger.error('Binary system was destroyed')
            raise ValueError

        #binary.delete()

        return porb_found-porb_true,ecc_found-ecc_true,obliq_found-obliq_true

    def errfunc_e(other_initial_conditions,
                  obliq_i,
                  #targets,
                  value_finder):
        initial_conditions = [other_initial_conditions[0],other_initial_conditions[1],obliq_i]
        final_conditions = errfunc(initial_conditions,value_finder)
        return final_conditions[0],final_conditions[1]
    
    def errfunc_o(other_initial_conditions,
                  ecc_i,
                  #targets,
                  value_finder):
        initial_conditions = [other_initial_conditions[0],ecc_i,other_initial_conditions[1]]
        final_conditions = errfunc(initial_conditions,value_finder)
        return final_conditions[0],final_conditions[2]

    def errfunc_p(porb_i,
                  other_initial_conditions,
                  #targets,
                  value_finder):
        """For 1D solving"""
        initial_conditions = porb_i,other_initial_conditions[0],other_initial_conditions[1]

        return errfunc(initial_conditions,value_finder)[0]

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
    initial_guess = [10,0.5,3]  #TODO
    #targets = 
    if solve:
        try:
            initial_secondary_angmom = value_finder.get_secondary_initial_angmom()
            if initial_eccentricity == 'solve':
                initial_porb,initial_eccentricity=scipy.optimize.root(errfunc_e,
                                                                      [initial_guess[0],initial_guess[1]],
                                                                      method='lm',
                                                                      options={'xtol':1e-6,
                                                                               'ftol':1e-6,
                                                                               'maxiter':20},
                                                                      args=(initial_guess[2],
                                                                            #targets,
                                                                            value_finder)
                )
            elif initial_obliquity == 'solve':
                initial_porb,initial_obliquity=scipy.optimize.root(errfunc_o,
                                                                   [initial_guess[0],initial_guess[2]],
                                                                   method='lm',
                                                                   options={'xtol':1e-6,
                                                                            'ftol':1e-6,
                                                                            'maxiter':20},
                                                                   args=(initial_guess[1],
                                                                         #targets,
                                                                         value_finder)
                )
            else:
                # Just solving for period
                initial_porb = optimize.brentq(
                    errfunc_p,
                    system.eccentricity,
                    eccentricity_upper_limit,
                    xtol=1e-2,
                    rtol=1e-2,
                    args=([initial_guess[1],initial_guess[2]],
                          #targets,
                          value_finder)
                )
        except:
            self.logger.exception('Solver Crashed') #to be adapted
            self.spin=scipy.nan
            return scipy.nan,scipy.nan
    else:
        initial_porb = system.Porb

    return value_finder.get_found_evolution(
        porb_initial=initial_porb,
        initial_eccentricity=initial_eccentricity,
        initial_obliquity=initial_obliquity,
        max_age=max_age
    )
#pylint: enable=too-many-locals

if __name__ == '__main__':
    config = parse_command_line()

    kwargs = dict(
        system=SimpleNamespace(
            primary_mass=config.primary_mass * units.M_sun,
            secondary_mass=config.secondary_mass * units.M_sun,
            feh=config.metallicity,
            orbital_period=config.orbital_period * units.day,
            age=config.final_age * units.Gyr
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

    evolution = find_evolution(**kwargs)
    with open(config.output_pickle, 'ab') as outf:
        pickle.dump(evolution, outf)
