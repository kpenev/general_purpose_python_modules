#!/usr/bin/env python3

"""Find tidal evolution reproducing the present state of a system."""

from glob import glob
import os.path

import scipy
from astropy import units


from stellar_evolution.library_interface import MESAInterpolator
from stellar_evolution.manager import StellarEvolutionManager
from orbital_evolution.binary import Binary
from orbital_evolution.transformations import phase_lag
from orbital_evolution.star_interface import EvolvingStar
from orbital_evolution.planet_interface import LockedPlanet
from orbital_evolution.initial_condition_solver import InitialConditionSolver
from basic_utils import Structure

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
    default_powers = scipy.array([0.0])

    result = dict()
    for component in lgq_suffixes:
        lgq = getattr(cmdline_args, 'lgQ_' + component)
        dependence = dict(
            tidal=scipy.array(
                getattr(cmdline_args,
                        'lgQ_' + component + '_wtide_dependence')
            ),
            spin=scipy.array(
                getattr(cmdline_args,
                        'lgQ_' + component + '_wspin_dependence')
            )
        )
        if scipy.isfinite(lgq):
            result[component] = dict()
            result[component]['reference_phase_lag'] = phase_lag(lgq)
            for dep_name in ['tidal', 'spin']:
                result[
                    component
                ][
                    '_'.join((dep_name, 'frequency', 'breaks'))
                ] = (
                    scipy.copy(dependence[dep_name][1 : : 2])
                    if dependence[dep_name].size > 0 else
                    default_breaks
                )
                result[
                    component
                ][
                    '_'.join((dep_name, 'frequency', 'powers'))
                ] = (
                    scipy.copy(dependence[dep_name][: : 2][:])
                    if dependence[dep_name].size > 0 else
                    default_powers
                )
        else:
            result[component] = None

    return result

def create_planet(mass,
                  radius,
                  dissipation=None):
    """
    Return the configured planet in the given system.

    Args:
        mass:    The mass of the planets, along with astropy units.

        radius:    The radius of the planets, along with astropy units.

        dissipation:    If None, no dissipation is set. Otherwise, should be a
            dictionary of keyword arguments to pass to
            LockedPlanet.set_dissipation().

    """

    planet = LockedPlanet(
        #False positive
        #pylint: disable=no-member
        mass=mass.to(units.M_sun).value,
        radius=radius.to(units.R_sun).value
        #pylint: enable=no-member
    )
    if dissipation is not None:
        planet.set_dissipation(**dissipation)
    return planet

def create_star(mass,
                feh,
                interpolator,
                dissipation=None,
                *,
                wind_strength=0.17,
                wind_saturation_frequency=2.78,
                diff_rot_coupling_timescale=5e-3,
                interpolation_age=None):
    """
    Create the star to use in the evolution.

    Args:
        mass:    The mass of the star to create, along with astropy units.

        feh:    The [Fe/H] value of the star to create.

        interpolator:    POET stellar evolution interpolator giving the
            evolution of the star's properties.

        dissipation:    If None, no dissipation is set. Otherwise, a dictionary
            of keyword arguments to pass to EvolvingStar.set_dissipation().

        wind_strength:    See same name argument to EvolvingStar.__init__()

        wind_saturation_frequency:    See same name argument to
            EvolvingStar.__init__()

        diff_rot_coupling_timescale:    See same name argument to
            EvolvingStar.__init__()

        interpolation_age:    The age at which to initialize the interpolation
            for the star. If `None`, the core formation age is used.

    Returns:
        EvolvingStar:
            The star in the system useable for calculating obital evolution.
    """

    #False positive
    #pylint: disable=no-member
    star = EvolvingStar(
        mass=mass.to_value(units.M_sun),
        metallicity=feh,
        wind_strength=wind_strength,
        wind_saturation_frequency=wind_saturation_frequency,
        diff_rot_coupling_timescale=(
            diff_rot_coupling_timescale.to_value(units.Gyr)
        ),
        interpolator=interpolator
    )
    #pylint: enable=no-member
    print('Core formation age = ' + repr(star.core_formation_age()))
    star.select_interpolation_region(star.core_formation_age()
                                     if interpolation_age is None else
                                     interpolation_age)
    if dissipation is not None:
        star.set_dissipation(zone_index=0, **dissipation)
    return star

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

#This serves the purpose of a single function to pass to a solver.
#pylint: disable=too-few-public-methods
class PeriodSolverWrapper:
    """Provide methods to pass to a solver for initial ecc. or obliquity."""

    def _create_primary(self):
        """Create the primary object in the system."""

        mprimary = getattr(self.system,
                           'Mprimary',
                           self.system.primary_mass)

        return create_star(
            mprimary,
            self.system.feh,
            self.interpolator['primary'],
            self.configuration['dissipation']['primary'],
            wind_strength=self.configuration['primary_wind_strength'],
            wind_saturation_frequency=(
                self.configuration['primary_wind_saturation']
            ),
            diff_rot_coupling_timescale=(
                self.configuration['primary_core_envelope_coupling_timescale']
            )
        )


    def _create_secondary(self):
        """Create the two objects comprising the system to evolve."""

        msecondary = getattr(self.system,
                             'Msecondary',
                             self.system.secondary_mass)

        if self.secondary_star:
            return create_star(
                msecondary,
                self.system.feh,
                self.interpolator['secondary'],
                self.configuration['dissipation']['secondary'],
                wind_strength=self.configuration['secondary_wind_strength'],
                wind_saturation_frequency=(
                    self.configuration['secondary_wind_saturation']
                ),
                diff_rot_coupling_timescale=self.configuration[
                    'secondary_core_envelope_coupling_timescale'
                ],
                interpolation_age=self.configuration['disk_dissipation_age']
            )

        return create_planet(
            msecondary,
            getattr(self.system,
                    'Rsecondary',
                    self.system.secondary_radius),
            self.configuration['dissipation']['secondary']
        )

    def _get_secondary_initial_angmom(self):
        """Return the angular momentum of the secondary when binary forms."""

        if not self.secondary_star:
            return (0.0, 0.0)

        secondary = self._create_secondary()
        mock_companion = create_planet(1.0, 1.0)
        binary = Binary(
            primary=secondary,
            secondary=mock_companion,
            initial_orbital_period=1.0,
            initial_eccentricity=0.0,
            initial_inclination=0.0,
            disk_lock_frequency=(
                2.0 * scipy.pi
                /
                self.configuration['secondary_disk_period']
            ),
            disk_dissipation_age=(
                2.0
                *
                self.configuration['disk_dissipation_age']
            )
        )
        binary.configure(age=secondary.core_formation_age(),
                         semimajor=float('nan'),
                         eccentricity=float('nan'),
                         spin_angmom=scipy.array([0.0]),
                         inclination=None,
                         periapsis=None,
                         evolution_mode='LOCKED_SURFACE_SPIN')

        secondary.detect_stellar_wind_saturation()

        binary.evolve(
            self.configuration['disk_dissipation_age'],
            self.configuration['max_timestep'],
            1e-6,
            None
        )
        final_state = binary.final_state()

        secondary.delete()
        mock_companion.delete()
        binary.delete()

        return final_state.envelope_angmom, final_state.core_angmom

    def _create_system(self,
                       primary,
                       secondary,
                       *,
                       porb_initial,
                       initial_eccentricity,
                       initial_obliquity):
        """
        Create the system to evolve from the two bodies (primary & secondary).

        Args:
            primary:    The primary in the system. Usually created by calling
                create_star().

            planet:    The secondary in the system. Usually created by calling
                create_star() or create_planet().

            porb_initial:    Initial orbital period in days.

            initial_eccentricity:    The initial eccentricity of the system.

            initial_obliquity:     The initial obliquity to assume for the
                system in rad.

        Returns:
            Binary:
                The binary system ready to evolve.
        """

        #False positive
        #pylint: disable=no-member
        binary = Binary(
            primary=primary,
            secondary=secondary,
            initial_orbital_period=porb_initial,
            initial_eccentricity=initial_eccentricity,
            initial_inclination=initial_obliquity,
            disk_lock_frequency=(2.0 * scipy.pi
                                 /
                                 self.target_state.Pdisk),
            disk_dissipation_age=self.configuration['disk_dissipation_age'],
            secondary_formation_age=self.target_state.planet_formation_age
        )
        #pylint: enable=no-member
        binary.configure(age=primary.core_formation_age(),
                         semimajor=float('nan'),
                         eccentricity=float('nan'),
                         spin_angmom=scipy.array([0.0]),
                         inclination=None,
                         periapsis=None,
                         evolution_mode='LOCKED_SURFACE_SPIN')
        if isinstance(secondary, EvolvingStar):
            initial_obliquity = scipy.array([0.0])
            initial_periapsis = scipy.array([0.0])
        else:
            initial_obliquity = None
            initial_periapsis = None
        secondary.configure(
            #False positive
            #pylint: disable=no-member
            age=self.target_state.planet_formation_age,
            #pylint: enable=no-member
            companion_mass=primary.mass,
            #False positive
            #pylint: disable=no-member
            semimajor=binary.semimajor(porb_initial),
            #pylint: enable=no-member
            eccentricity=initial_eccentricity,
            spin_angmom=(
                scipy.array([0.01, 0.01])
                if isinstance(secondary, EvolvingStar) else
                scipy.array([0.0])
            ),
            inclination=initial_obliquity,
            periapsis=initial_periapsis,
            locked_surface=False,
            zero_outer_inclination=True,
            zero_outer_periapsis=True
        )
        primary.detect_stellar_wind_saturation()
        if isinstance(secondary, EvolvingStar):
            secondary.detect_stellar_wind_saturation()
        return binary

    @staticmethod
    def _format_evolution(binary, interpolators, secondary_star):
        """
        Return pre-calculated evolution augmented with stellar quantities.

        Args:
            binary:    The binary used to calculate the evolution to format.

            interpolators:


        The returned evolution contains the full evolution produced by
        Binary.get_evolution(), as well as the evolution of the following
        quantities:

            * **orbital_period**: the orbital period

            * **(primary/secondary)_radius: The radius of the primary/secondary
              star.

            * **(primary/secondary)_lum: The luminosity of the primary/secondary
              star.

            * **(primary/secondary)_(iconv/irad): The convective/radiative
              zone moment of inertia of the primary/secondary star.
        """

        evolution = binary.get_evolution()
        #False positive
        #pylint: disable=no-member
        evolution.orbital_period = binary.orbital_period(evolution.semimajor)

        components_to_get = ['primary']
        if secondary_star:
            components_to_get.append('secondary')

        for component in components_to_get:

            if (
                    len(interpolators[component].track_masses) == 1
                    and
                    len(interpolators[component].track_feh) == 1
            ):
                star_params = dict(
                    mass=interpolators[component].track_masses[0],
                    feh=interpolators[component].track_feh[0]
                )
            else:
                star_params = dict(
                    mass=getattr(binary, component).mass,
                    feh=getattr(binary, component).metallicity
                )

            for quantity_name in ['radius', 'lum', 'iconv', 'irad']:
                print('Start params: ' + repr(star_params))
                quantity = interpolators[component](
                    quantity_name,
                    **star_params
                )
                print(quantity_name + ' age range: ' + repr((quantity.min_age,
                                                             quantity.max_age)))
                values = scipy.full(evolution.age.shape, scipy.nan)

                #TODO: determine age range properly (requires C/C++ code
                #modifications)
                valid_ages = scipy.logical_and(
                    evolution.age > quantity.min_age * 2.0,
                    evolution.age < quantity.max_age / 2.0
                )
                if quantity_name in ['iconv', 'irad']:
                    values[valid_ages] = getattr(
                        getattr(binary, component),
                        (
                            ('envelope' if quantity_name == 'iconv' else 'core')
                            +
                            '_inertia'
                        )
                    )(evolution.age[valid_ages])
                else:
                    values[valid_ages] = quantity(evolution.age[valid_ages])
                setattr(evolution,
                        component + '_' + quantity_name,
                        values)

        return evolution

    def _get_final_state(self, initial_eccentricity, initial_obliquity):
        """Return the final state of the system given initial conditions."""

        find_ic = InitialConditionSolver(
            disk_dissipation_age=self.configuration['disk_dissipation_age'],
            evolution_max_time_step=self.configuration['max_timestep'],
            initial_eccentricity=initial_eccentricity,
            initial_inclination=initial_obliquity,
            orbital_period_tolerance=(
                self.configuration['orbital_period_tolerance']
            ),
            max_porb_initial=1000.0
        )
        primary = self._create_primary()
        secondary = self._create_secondary()
        self.porb_initial, self.psurf = find_ic(self.target_state,
                                                primary,
                                                secondary)
        #False positive
        #pylint: disable=no-member
        self.porb_initial *= units.day
        self.psurf *= units.day

        result = find_ic.binary.final_state()

        primary.delete()
        secondary.delete()
        find_ic.binary.delete()

        return result


    def __init__(self,
                 system,
                 interpolator,
                 *,
                 current_age,
                 disk_period,
                 initial_obliquity,
                 disk_dissipation_age,
                 max_timestep,
                 dissipation,
                 primary_wind_strength,
                 primary_wind_saturation,
                 primary_core_envelope_coupling_timescale,
                 secondary_wind_strength,
                 secondary_wind_saturation,
                 secondary_core_envelope_coupling_timescale,
                 secondary_disk_period=None,
                 secondary_star=False,
                 orbital_period_tolerance=1e-6):
        """
        Get ready for the solver.

        Args:
            system:    The parameters of the system we are trying to reproduce.

            interpolator:    The stellar evolution interpolator to use, could
                also be a pair of interpolators, one to use for the primary and
                one for the secondary.

            current_age:    The present day age of the system (when the target
                state is to be reproduced).

            disk_period: The period at which the primaly will initially spin.

            initial_obliquity:    The initial inclination of all zones
                of the primary relative to the orbit in which the secondary
                forms.

            disk_dissipation_age:    The age at which the disk dissipates and
                the secondary forms.

            max_timestep:    The maximum timestep the evolution is allowed to
                take.

            dissipation:    A dictionary containing the dissipation argument to
                pass to create_star() or create_planet() when creating the
                primary and secondary in the system. It should have two keys:
                `'primary'` and `'secondary'`, each containing the argument for
                the corresponding component.

            primary_wind_strength:    The normilazation constant defining how
                fast angular momentum is lost by the primary star.

            primary_wind_saturation:    The angular velocity above which the
                rate of angular momentum loss switches from being proportional
                to the cube of the angular velocity to linear for the primary
                star.

            primary_core_envelope_coupling_timescale:    The timescale over
                which the core and envelope of the primary converge toward solid
                body rotation.

            secondary_wind_strength:    Analogous to primary_wind_strength but
                for the secondary.

            secondary_wind_saturation:    Analogous to primary_wind_saturation
                but for the secondary.

            secondary_core_envelope_coupling_timescale:    Analogous to
                primary_core_envelope_coupling_timescale but for the secondary.

            secondary_disk_period:    The period to which the surface spin of
                the secondary is locked before the binary forms. If None, the
                primary disk period is used.

            secondary_star:    True iff the secondary object is a star.

            orbital_period_tolerance:    How precisely do we need to match the
                present day orbital period (relative error).

        Returns:
            None
        """

        self.target_state = Structure(
            #False positive
            #pylint: disable=no-member
            age=current_age.to(units.Gyr).value,
            Porb=getattr(system,
                         'Porb',
                         system.orbital_period).to(units.day).value,
            Pdisk=disk_period.to(units.day).value,
            planet_formation_age=disk_dissipation_age.to(units.Gyr).value
            #pylint: enable=no-member
        )
        print('For system:\n\t' + repr(system))
        print('Targeting:\n\t' + self.target_state.format('\n\t'))

        self.system = system
        if isinstance(interpolator, MESAInterpolator):
            self.interpolator = dict(primary=interpolator,
                                     secondary=interpolator)
        else:
            self.interpolator = dict(primary=interpolator[0],
                                     secondary=interpolator[1])
        self.configuration = dict(
            #False positive
            #pylint: disable=no-member
            disk_dissipation_age=disk_dissipation_age.to(units.Gyr).value,
            max_timestep=max_timestep.to(units.Gyr).value,
            #pylint: enable=no-member
            orbital_period_tolerance=orbital_period_tolerance,
            dissipation=dissipation,
            initial_obliquity=initial_obliquity,
            primary_wind_strength=primary_wind_strength,
            primary_wind_saturation=primary_wind_saturation,
            primary_core_envelope_coupling_timescale=(
                primary_core_envelope_coupling_timescale
            ),
            secondary_core_envelope_coupling_timescale=(
                secondary_core_envelope_coupling_timescale
            ),
            secondary_wind_strength=secondary_wind_strength,
            secondary_wind_saturation=secondary_wind_saturation,
            secondary_disk_period=(secondary_disk_period
                                   or
                                   disk_period).to_value(units.day)

        )
        self.porb_initial = None
        self.psurf = None
        self.secondary_star = secondary_star

    def eccentricity_difference(self, initial_eccentricity):
        """
        Return the discrepancy in eccentricity for the given initial value.

        An evolution is found which reproduces the present day orbital period of
        the system, starting with the given initial eccentricity and the result
        of this function is the difference between the present day eccentricity
        predicted by that evolution and the measured value supplied at
        construction through the system argument. In addition, the initial
        orbital period and (initial or final, depending on which is not
        specified) stellar spin period are stored in the
        :attr:porb_initial and :attr:psurf attributes.

        Args:
            initial_eccentricity(float):    The initial eccentricity with which
                the secondary forms.

        Returns:
            float:
                The difference between the predicted and measured values of the
                eccentricity.
        """

        final_eccentricity = self._get_final_state(
            initial_eccentricity=initial_eccentricity,
            initial_obliquity=self.system.obliquity
        ).eccentricity
        #pylint: enable=no-member
        print('Final eccentricity: ' + repr(final_eccentricity))

        return final_eccentricity - self.system.eccentricity

    def obliquity_difference(self, initial_obliquity):
        """Same as eccentricity_difference, but for obliquity."""

        final_obliquity = self._get_final_state(
            initial_eccentricity=self.system.eccentricity,
            initial_obliquity=initial_obliquity
        ).envelope_inclination
        #pylint: enable=no-member
        print('Final obliquity: ' + repr(final_obliquity))

        return final_obliquity - self.system.obliquity

    def get_found_evolution(self,
                            initial_eccentricity,
                            initial_obliquity,
                            max_age=None,
                            **evolve_kwargs):
        """
        Return the evolution matching the current system configuration.

        Args:
            initial_eccentricity:    The initial eccentricity found to reproduce
                the current state.

            max_age:    The age up to which to calculate the evolution. If None
                (default), the star lifetime is used.

            evolve_kwargs:    Additional keyword arguments to pass to
                Binary.evolve()

        Returns:
            See EccentricitySolverCallable._format_evolution().
        """

        if self.porb_initial is None:
            self._get_final_state(initial_eccentricity, initial_obliquity)
        primary = self._create_primary()
        secondary = self._create_secondary()

        binary = self._create_system(
            primary,
            secondary,
            #False positive
            #pylint: disable=no-member
            porb_initial=self.porb_initial.to(units.day).value,
            #pylint: enable=no-member
            initial_eccentricity=initial_eccentricity,
            initial_obliquity=initial_obliquity
        )
        if max_age is None:
            if isinstance(primary, EvolvingStar):
                max_age = primary.lifetime()
            else:
                max_age = (self.system.age).to(units.Gyr).value
        else:
            max_age = max_age.to(units.Gyr).value

        if 'required_ages' not in evolve_kwargs:
            evolve_kwargs['required_ages'] = None
        binary.evolve(
            #False positive
            #pylint: disable=no-member
            max_age,
            self.configuration['max_timestep'],
            #pylint: enable=no-member
            1e-6,
            **evolve_kwargs
        )

        result = self._format_evolution(binary,
                                        self.interpolator,
                                        self.secondary_star)

        primary.delete()
        secondary.delete()
        binary.delete()

        return result

def check_if_secondary_is_star(system):
    """True iff the secondary in the system should be evolved as a star."""

    return (
        getattr(system, 'Msecondary', system.secondary_mass)
        >
        0.05 * units.M_sun
    )

#pylint: enable=too-few-public-methods

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

    #False positive
    #pylint: disable=no-member
    if disk_period is None:
        if hasattr(system, 'Pprimary'):
            disk_period = system.Pprimary
        else:
            disk_period = (2.0 * scipy.pi * system.Rsecondary
                           /
                           system.Vsini)

    if secondary_is_star is None:
        secondary_is_star = check_if_secondary_is_star(system)
    max_timestep = 1e-3 * units.Gyr
    #pylint: enable=no-member
    period_solver = PeriodSolverWrapper(
        system=system,
        interpolator=interpolator,
        #False positive
        #pylint: disable=no-member
        current_age=system.age,
        #pylint: enable=no-member
        disk_period=disk_period,
        initial_obliquity=initial_obliquity,
        disk_dissipation_age=disk_dissipation_age,
        max_timestep=max_timestep,
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
        orbital_period_tolerance=orbital_period_tolerance
    )
    if solve:
        if initial_eccentricity == 'solve':
            initial_eccentricity = scipy.optimize.brentq(
                period_solver.eccentricity_difference,
                system.eccentricity,
                0.5,
                xtol=1e-2,
                rtol=1e-2
            )
        elif initial_obliquity == 'solve':
            initial_obliquity = scipy.optimize.brentq(
                period_solver.obliquity_difference,
                0,
                scipy.pi,
                xtol=1e-3,
                rtol=1e-3
            )
    else:
        period_solver.porb_initial = system.Porb

    return period_solver.get_found_evolution(
        initial_eccentricity=initial_eccentricity,
        initial_obliquity=initial_obliquity,
        max_age=max_age,
        **extra_evolve_args
    )
#pylint: enable=too-many-locals
