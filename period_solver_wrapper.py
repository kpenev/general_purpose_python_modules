"""Define a class allowing to nest initial period solver in another solver."""

import logging

import numpy
from astropy import units

from stellar_evolution.library_interface import MESAInterpolator
from orbital_evolution.binary import Binary
from orbital_evolution.star_interface import EvolvingStar
from orbital_evolution.planet_interface import LockedPlanet
from orbital_evolution.initial_condition_solver import InitialConditionSolver

from basic_utils import Structure

#TODO: see if this can be simplified
#pylint: disable=too-many-instance-attributes
class PeriodSolverWrapper:
    """Provide methods to pass to a solver for initial ecc. or obliquity."""

    @staticmethod
    def _create_planet(mass,
                       radius,
                       dissipation=None):
        """
        Return the configured planet in the given system.

        Args:
            mass:    The mass of the planets, along with astropy units.

            radius:    The radius of the planets, along with astropy units.

            dissipation:    If None, no dissipation is set. Otherwise, should be
                a dictionary of keyword arguments to pass to
                `LockedPlanet.set_dissipation`.

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

    @staticmethod
    def _create_star(mass,
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

            dissipation:    If None, no dissipation is set. Otherwise, a
                dictionary of keyword arguments to pass to
                `EvolvingStar.set_dissipation()`.

            wind_strength:    See same name argument to EvolvingStar.__init__()

            wind_saturation_frequency:    See same name argument to
                EvolvingStar.__init__()

            diff_rot_coupling_timescale:    See same name argument to
                EvolvingStar.__init__()

            interpolation_age:    The age at which to initialize the
                interpolation for the star. If `None`, the core formation age is
                used.

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
        star.select_interpolation_region(star.core_formation_age()
                                         if interpolation_age is None else
                                         interpolation_age)
        if dissipation is not None:
            star.set_dissipation(zone_index=0, **dissipation)
        return star



    def _create_primary(self):
        """Create the primary object in the system."""

        mprimary = getattr(self.system,
                           'Mprimary',
                           self.system.primary_mass)

        if self.primary_star:
            return self._create_star(
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
        return self._create_planet(
            mprimary,
            getattr(self.system,
                    'Rprimary',
                    self.system.primary_radius),
            self.configuration['dissipation']['primary']
        )


    def _create_secondary(self):
        """Create the two objects comprising the system to evolve."""

        msecondary = getattr(self.system,
                             'Msecondary',
                             self.system.secondary_mass)

        if self.secondary_star:
            return self._create_star(
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

        return self._create_planet(
            msecondary,
            getattr(self.system,
                    'Rsecondary',
                    self.system.secondary_radius),
            self.configuration['dissipation']['secondary']
        )

    def _create_system(self,
                       primary,
                       secondary,
                       *,
                       porb_initial,
                       initial_eccentricity,
                       initial_obliquity,
                       initial_secondary_angmom):
        """
        Create the system to evolve from the two bodies (primary & secondary).

        Args:
            primary:    The primary in the system. Usually created by calling
                :meth:`_create_star`.

            planet:    The secondary in the system. Usually created by calling
                :meth:`_create_star` or :meth:`_create_planet`.

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
            disk_lock_frequency=(2.0 * numpy.pi
                                 /
                                 self.target_state.Pdisk),
            disk_dissipation_age=self.configuration['disk_dissipation_age'],
            secondary_formation_age=self.target_state.planet_formation_age
        )
        #pylint: enable=no-member
        binary.configure(age=primary.core_formation_age(),
                         semimajor=float('nan'),
                         eccentricity=float('nan'),
                         spin_angmom=numpy.array([0.0]),
                         inclination=None,
                         periapsis=None,
                         evolution_mode='LOCKED_SURFACE_SPIN')
        if isinstance(secondary, EvolvingStar):
            initial_obliquity = numpy.array([0.0])
            initial_periapsis = numpy.array([0.0])
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
            spin_angmom=initial_secondary_angmom,
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
                quantity = interpolators[component](
                    quantity_name,
                    **star_params
                )
                values = numpy.full(evolution.age.shape, numpy.nan)

                #TODO: determine age range properly (requires C/C++ code
                #modifications)
                valid_ages = numpy.logical_and(
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

    def _match_target_state(self,
                            *,
                            initial_eccentricity,
                            initial_obliquity,
                            initial_secondary_angmom,
                            evolve_kwargs,
                            full_evolution=False):
        """Return the final state or evolution of the system matching target."""

        find_ic = InitialConditionSolver(
            evolution_max_time_step=self.configuration['max_timestep'],
            evolution_precision=evolve_kwargs.get('precision', 1e-6),
            evolution_timeout=evolve_kwargs.get('timeout', 0),
            initial_eccentricity=initial_eccentricity,
            initial_inclination=initial_obliquity,
            initial_secondary_angmom=initial_secondary_angmom,
            max_porb_initial=1000.0,
            **{
                param: self.configuration[param]
                for param in ['disk_dissipation_age',
                              'orbital_period_tolerance',
                              'period_search_factor',
                              'scaled_period_guess']
            }
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

        if full_evolution:
            result = self._format_evolution(find_ic.binary,
                                            self.interpolator,
                                            self.secondary_star)
        else:
            result = find_ic.binary.final_state()

        primary.delete()
        secondary.delete()
        find_ic.binary.delete()

        return result

    def _get_combined_evolve_args(self, overwrite_args):
        """Return `self._extra_evolve_args` overwritten by `overwrite_args`."""

        combined_evolve_kwargs = dict(self._extra_evolve_args)
        combined_evolve_kwargs.update(overwrite_args)
        return combined_evolve_kwargs

    #TODO: cleanup
    #pylint: disable=too-many-locals
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
                 primary_star=True,
                 secondary_star=False,
                 orbital_period_tolerance=1e-6,
                 period_search_factor=2.0,
                 scaled_period_guess=1.0,
                 **extra_evolve_args):
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
                pass to :meth:`_create_star` or :meth:`_create_planet` when
                creating the primary and secondary in the system. It should have
                two keys: `'primary'` and `'secondary'`, each containing the
                argument for the corresponding component.

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

            primary_star:    True iff the primary object is a star.

            secondary_star:    True iff the secondary object is a star.

            orbital_period_tolerance:    How precisely do we need to match the
                present day orbital period (relative error).

            period_search_factor:    See same name argument to
                :meth:`InitialConditionSolver.__init__`

            scaled_period_guess:    See same name argument to
                :meth:`InitialConditionSolver.__init__`

            extra_evolve_args:    Additional arguments to pass to binary.evolve.

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
        logging.getLogger(__name__).info(
            'For system:\n\t%s'
            '\nTargeting:\n\t%s',
            repr(system),
            self.target_state.format('\n\t')
        )

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
                                   disk_period).to_value(units.day),
            period_search_factor=period_search_factor,
            scaled_period_guess=scaled_period_guess
        )
        self.porb_initial = None
        self.psurf = None
        self.secondary_star = secondary_star
        self.primary_star = primary_star

        if 'precision' not in extra_evolve_args:
            extra_evolve_args['precision'] = 1e-6
        if 'required_ages' not in extra_evolve_args:
            extra_evolve_args['required_ages'] = None
        self._extra_evolve_args = extra_evolve_args
    #pylint: disable=too-many-locals


    def get_secondary_initial_angmom(self, **evolve_kwargs):
        """Return the angular momentum of the secondary when binary forms."""

        if not self.secondary_star:
            return (0.0, 0.0)

        secondary = self._create_secondary()
        mock_companion = self._create_planet(1.0 * units.M_jup,
                                             1.0 * units.R_jup)
        binary = Binary(
            primary=secondary,
            secondary=mock_companion,
            initial_orbital_period=1.0,
            initial_eccentricity=0.0,
            initial_inclination=0.0,
            disk_lock_frequency=(
                2.0 * numpy.pi
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
                         spin_angmom=numpy.array([0.0]),
                         inclination=None,
                         periapsis=None,
                         evolution_mode='LOCKED_SURFACE_SPIN')

        secondary.detect_stellar_wind_saturation()

        binary.evolve(
            final_age=self.configuration['disk_dissipation_age'],
            max_time_step=self.configuration['max_timestep'],
            **self._get_combined_evolve_args(evolve_kwargs)
        )
        final_state = binary.final_state()

        secondary.delete()
        mock_companion.delete()
        binary.delete()

        return final_state.envelope_angmom, final_state.core_angmom


    def eccentricity_difference(self,
                                initial_eccentricity,
                                initial_secondary_angmom,
                                **evolve_kwargs):
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

            evolve_kwargs:    Any additional keyword argument to pass to binary
                evolve.

        Returns:
            float:
                The difference between the predicted and measured values of the
                eccentricity.
        """

        final_eccentricity = self._match_target_state(
            initial_eccentricity=initial_eccentricity,
            initial_obliquity=self.system.obliquity,
            initial_secondary_angmom=initial_secondary_angmom,
            evolve_kwargs=self._get_combined_evolve_args(evolve_kwargs)
        ).eccentricity
        #pylint: enable=no-member

        return final_eccentricity - self.system.eccentricity

    def obliquity_difference(self,
                             initial_obliquity,
                             initial_secondary_angmom,
                             **evolve_kwargs):
        """Same as eccentricity_difference, but for obliquity."""

        final_obliquity = self._match_target_state(
            initial_eccentricity=self.system.eccentricity,
            initial_obliquity=initial_obliquity,
            initial_secondary_angmom=initial_secondary_angmom,
            evolve_kwargs=self._get_combined_evolve_args(evolve_kwargs)
        ).envelope_inclination
        #pylint: enable=no-member

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

        evolve_kwargs = self._get_combined_evolve_args(evolve_kwargs)
        initial_secondary_angmom = self.get_secondary_initial_angmom(
            **evolve_kwargs
        )
        if self.porb_initial is None:
            return self._match_target_state(
                initial_eccentricity=initial_eccentricity,
                initial_obliquity=initial_obliquity,
                initial_secondary_angmom=initial_secondary_angmom,
                evolve_kwargs=evolve_kwargs,
                full_evolution=True
            )
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
            initial_obliquity=initial_obliquity,
            initial_secondary_angmom=initial_secondary_angmom
        )
        if max_age is None:
            if isinstance(primary, EvolvingStar):
                max_age = primary.lifetime()
            else:
                max_age = (self.system.age).to(units.Gyr).value
        else:
            max_age = max_age.to(units.Gyr).value

        binary.evolve(
            #False positive
            #pylint: disable=no-member
            max_age,
            self.configuration['max_timestep'],
            #pylint: enable=no-member
            **evolve_kwargs
        )

        result = self._format_evolution(binary,
                                        self.interpolator,
                                        self.secondary_star)

        primary.delete()
        secondary.delete()
        binary.delete()

        return result
#pylint: enable=too-many-instance-attributes
