#!/usr/bin/env python3 -u
#with thanks to Ruskin

#imports
import numpy
import scipy
from astropy import units, constants
import time
import logging

from types import SimpleNamespace

from stellar_evolution.library_interface import MESAInterpolator
from orbital_evolution.binary import Binary
from orbital_evolution.transformations import phase_lag
from orbital_evolution.star_interface import EvolvingStar
from orbital_evolution.planet_interface import LockedPlanet

from POET.solver import poet_solver

class InitialValueFinder:
    """TODO add proper documentation."""

    @staticmethod
    def _create_planet(mass,
                      radius,
                      dissipation=None):
        """
        Return the configured planet in the given system. TODO

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
        logger = logging.getLogger(__name__)
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
        logger.debug('By the way, the interpolation age is %s', repr(interpolation_age))
        if dissipation is not None:
            star.set_dissipation(zone_index=0, **dissipation)
        return star

    def _create_primary(self):
        """Create the primary object in the system."""

        mprimary = getattr(self.system,
                           'Mprimary',
                           self.system.primary_mass)
        
        logger = logging.getLogger(__name__)
        logger.debug('The dissipation we are setting for the primary is %s', repr(self.configuration['dissipation']['primary']))

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


    def _create_secondary(self,startFromDiskDissipationAge=True):
        """Create the two objects comprising the system to evolve."""

        msecondary = getattr(self.system,
                             'Msecondary',
                             self.system.secondary_mass)

        logger = logging.getLogger(__name__)
        logger.debug('The dissipation we are setting for the secondary is %s', repr(self.configuration['dissipation']['secondary']))

        interpolation_age = self.configuration['disk_dissipation_age'] if startFromDiskDissipationAge else None
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
                interpolation_age=interpolation_age
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

        logger = logging.getLogger(__name__)

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
            spin_angmom=numpy.array(initial_secondary_angmom),
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
                    evolution.age < quantity.max_age
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
    
    def _get_combined_evolve_args(self, overwrite_args):
        """Return `self._extra_evolve_args` overwritten by `overwrite_args`."""

        combined_evolve_kwargs = dict(self._extra_evolve_args)
        combined_evolve_kwargs.update(overwrite_args)
        return combined_evolve_kwargs

    #pylint: disable=too-many-locals
    def __init__(self,
                 system,
                 interpolator,
                 *,
                 current_age,
                 disk_period,
                 initial_obliquity,
                 disk_dissipation_age,
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

        porb = getattr(system, 'Porb', None)
        if porb is None:
            porb = system.orbital_period
        self.target_state = SimpleNamespace(
            #False positive
            #pylint: disable=no-member
            age=current_age.to(units.Gyr).value,
            Porb=porb.to(units.day).value,
            Pdisk=disk_period.to(units.day).value,
            planet_formation_age=disk_dissipation_age.to(units.Gyr).value,
            evolution_max_time_step=extra_evolve_args['max_time_step'],
            evolution_precision=extra_evolve_args['precision']
            #pylint: enable=no-member
        )
        logging.getLogger(__name__).info(
            'For system:\n\t%s'
            '\nTargeting:\n\t%s',
            repr(system),
            repr(self.target_state)
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
        #self.porb_initial = None
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

        logger = logging.getLogger(__name__)

        secondary = self._create_secondary(False)

        if not self.secondary_star:
            return (
                (
                    2.0 * numpy.pi / self.configuration['secondary_disk_period']
                    *
                    secondary.inertia()
                )
                ,
            )

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
            **self._get_combined_evolve_args(evolve_kwargs)
        )
        final_state = binary.final_state()
        logger.debug(
            'Initial angmom getter final state: %s, ',
            repr(final_state)
        )

        result = final_state.envelope_angmom, final_state.core_angmom

        core_inertia = secondary.core_inertia(final_state.age)
        logging.getLogger(__name__).debug(
            'Initial secondary angular momenta: %s -> w = %s, %s -> w = %s '
            '(secondary core fomation  age: %s)',
            repr(result[0]),
            repr(result[0] / secondary.envelope_inertia(final_state.age)),
            repr(result[1]),
            repr(numpy.nan if core_inertia == 0 else result[1] / core_inertia),
            repr(secondary.core_formation_age())
        )

        secondary.delete()
        mock_companion.delete()
        binary.delete()

        return result
    
    def try_system(self,initial_conditions,initial_secondary_angmom,
                   type = None,
                   carepackage = None,):
        """
        Return the evolution matching the given system configuration.

        Args:
            initial_conditions:    The initial period, eccentricity, and
                obliquity to try reproducing the system with.

            initial_secondary_angmom:    The initial angular momentum of the
                secondary.

            max_age:    The age up to which to calculate the evolution. If None
                (default), the star lifetime is used.

        Returns:
            See EccentricitySolverCallable._format_evolution().
        """

        initial_orbital_period=initial_conditions[0]
        initial_eccentricity=initial_conditions[1]
        initial_obliquity=initial_conditions[2]

        logger = logging.getLogger(__name__)

        primary = self._create_primary()
        secondary = self._create_secondary()
        if (primary.core_inertia(self.configuration['disk_dissipation_age']) == 0
            or
            secondary.core_inertia(self.configuration['disk_dissipation_age']) == 0
            ):
            logger.warning(
                'Primary or secondary core inertia is zero at current disk dissipation age: %s, ',
                repr(self.configuration['disk_dissipation_age'])
            )
            self.configuration['disk_dissipation_age'] = 0.02
        if not primary.core_inertia(self.configuration['disk_dissipation_age']) > 0:
            logger.error(
                'Reported primary core inertia at disk dissipation age: %s, ',
                repr(primary.core_inertia(self.configuration['disk_dissipation_age']))
            )
            raise ValueError("Primary core inertia is zero. Primary has not formed.",0)
        if not secondary.core_inertia(self.configuration['disk_dissipation_age']) > 0:
            logger.error(
                'Reported secondary core inertia at disk dissipation age: %s, ',
                repr(secondary.core_inertia(self.configuration['disk_dissipation_age']))
            )
            raise ValueError("Secondary core inertia is zero. Secondary has not formed.",0)

        binary=self._create_system(
            primary,
            secondary,
            #False positive
            #pylint: disable=no-member
            porb_initial=initial_orbital_period,
            #pylint: enable=no-member
            initial_eccentricity=initial_eccentricity,
            initial_obliquity=initial_obliquity,
            initial_secondary_angmom=initial_secondary_angmom
        )

        #if max_age is None:
        max_age = self.target_state.age
        #else:
        #    max_age = max_age.to(units.Gyr).value

        binary.evolve(
            max_age,
            self.target_state.evolution_max_time_step,
            self.target_state.evolution_precision,
            None,
            timeout=3600
            )

        final_state=binary.final_state()
        logger.debug('Final state age: %s, ',
                                          repr(final_state.age))
        logger.debug('Target state age: %s, ',
                                            repr(self.target_state.age))
        logger.debug('Initial eccentricity: %s, ',
                                            repr(initial_eccentricity))
        logger.debug('Final eccentricity: %s, ',
                                            repr(final_state.eccentricity))
        logger.debug('Initial period: %s, ',
                                            repr(initial_orbital_period))

        final_orbital_period = binary.orbital_period(final_state.semimajor)
        result = SimpleNamespace()
        setattr(result, 'orbital_period', numpy.array([final_orbital_period]))
        setattr(result, 'eccentricity', numpy.array([final_state.eccentricity]))

        logger.debug('Final period: %s, ',
                                            repr(result.orbital_period[-1]))
        try:
            assert(final_state.age==self.target_state.age)
        except AssertionError:
            # Save the parameters and evolution to an astropy fits file. Parameters in header data.
            import os
            import datetime
            from astropy.table import Table

            filename = 'failed_solutions'

            # Make it clear which system the file is for, if possible
            if carepackage is not None:
                filename = filename + f'/{carepackage["system_name"]}'
            
            # Create the directory if it doesn't exist
            os.makedirs(filename, exist_ok=True)

            # Create the filename
            now = datetime.datetime.now()
            filename = filename + f'/solution_{now.strftime("%Y-%m-%d_%H-%M-%S")}.fits'

            evolution = self._format_evolution(binary,
                                           self.interpolator,
                                           self.secondary_star)

            # Create the table
            table = Table({
                'age': evolution.age,
                'porb': evolution.orbital_period,
                'eccentricity': evolution.eccentricity,
                'primary_radius': evolution.primary_radius,
                'primary_lum': evolution.primary_lum,
                'primary_iconv': evolution.primary_iconv,
                'primary_irad': evolution.primary_irad,
                'secondary_radius': evolution.secondary_radius,
                'secondary_lum': evolution.secondary_lum,
                'secondary_iconv': evolution.secondary_iconv,
                'secondary_irad': evolution.secondary_irad,
                'primary_L_env': evolution.primary_envelope_angmom,
                'primary_L_core': evolution.primary_core_angmom,
                'secondary_L_env': evolution.secondary_envelope_angmom,
                'secondary_L_core': evolution.secondary_core_angmom
            })

            # Create the header
            for key, value in self.configuration['dissipation']['primary'].items():
                name = key[::2][:8]
                table.meta[name] = str(value)
            table.meta['didiage'] = self.configuration['disk_dissipation_age']
            table.meta['p_dlp'] = self.target_state.Pdisk
            table.meta['p_windst'] = self.configuration['primary_wind_strength']
            table.meta['p_windsa'] = self.configuration['primary_wind_saturation']
            table.meta['p_cect'] = self.configuration['primary_core_envelope_coupling_timescale'].to_value(units.Gyr)
            table.meta['ecc_i'] = initial_eccentricity
            table.meta['s_dlp'] = self.configuration['secondary_disk_period']
            table.meta['s_windst'] = self.configuration['secondary_wind_strength']
            table.meta['s_windsa'] = self.configuration['secondary_wind_saturation']
            table.meta['s_cect'] = self.configuration['secondary_core_envelope_coupling_timescale'].to_value(units.Gyr)
            table.meta['age'] = self.target_state.age
            table.meta['feh'] = self.system.feh
            table.meta['porb'] = self.target_state.Porb
            table.meta['p_mass'] = self.system.primary_mass.to_value(units.M_sun)
            table.meta['s_mass'] = self.system.secondary_mass.to_value(units.M_sun)
            table.meta['p_rad'] = self.system.Rprimary.to_value(units.R_sun)
            table.meta['s_rad'] = self.system.Rsecondary.to_value(units.R_sun)
            table.meta['p_i'] = initial_orbital_period
            table.meta['evo_maxt'] = self.target_state.evolution_max_time_step
            table.meta['evo_prec'] = self.target_state.evolution_precision
            table.meta['orb_ptol'] = self.configuration['orbital_period_tolerance']
            table.meta['pformage'] = self.target_state.planet_formation_age

            # Save the file
            table.write(filename, overwrite=True)

            # Raise the error
            raise AssertionError(f"Final age does not match target age. See {filename} for details.")
        finally:
            # Clean up
            primary.delete()
            secondary.delete()
            binary.delete()

        # Set up AI stuff
        if type is not None and carepackage is not None:
            # Save all the x parameters into a table
            x_vals_list = [
                carepackage['lgQ_min'],
                carepackage['lgQ_break_period'].to_value(units.day),
                carepackage['lgQ_powerlaw'],
                self.target_state.age,
                self.system.feh,
                result.orbital_period[-1],
                self.system.primary_mass.to_value(units.M_sun),
                self.system.secondary_mass.to_value(units.M_sun),
                self.system.Rprimary.to_value(units.R_sun),
                self.system.Rsecondary.to_value(units.R_sun)
            ]
            logger.debug('x_vals_list = %s, ', repr(x_vals_list))

            params = {
                "type": 'blank',
                "epochs": 350,
                "batch_size": 50,
                "verbose": 2,
                "retrain": False,
                "threshold": 2000,
                "path_to_store": carepackage['path'],
                "version": carepackage['system_name'],
                "features": [True, True, True, True, True, True, True, True, True, True]
            }

            def save_data(param_type,params,x_train,y_train):
                if numpy.isnan(x_train).any() or numpy.isnan(y_train):
                    logger.warning('NaNs in the data. Not saving.')
                    return
                params['type'] = param_type
                model = poet_solver.POET_IC_Solver(**params)
                if carepackage['lock'] is not None:
                    logger.debug('Getting parallel processing lock.')
                    carepackage['lock'].acquire()
                    logger.debug('Got parallel processing lock.')
                try:
                    model.store_data(X_train=x_train, y_train=y_train)
                    # length = model.data_length()
                    # if length > params['threshold'] and length % 200 == 0:
                    #     model.just_fit()
                except Exception as e:
                    logger.error('Error in storing data: %s', repr(e))
                    raise e
                finally:
                    if carepackage['lock'] is not None:
                        logger.debug('Releasing parallel processing lock.')
                        carepackage['lock'].release()
                        logger.debug('Released parallel processing lock.')

            if type == '1d':
                X_train = numpy.array(x_vals_list)
                logger.debug('X_train = %s, ', repr(X_train))
                save_data('1d_period',params,X_train,initial_orbital_period)
            elif type == '2d':
                x_vals_list.append(final_state.eccentricity)
                logger.debug('NOW x_vals_list = %s, ', repr(x_vals_list))
                params['features'].append(True)
                X_train = numpy.array(x_vals_list)
                logger.debug('X_train = %s, ', repr(X_train))
                save_data('2d_period',params,X_train,initial_orbital_period)
                save_data('2d_eccentricity',params,X_train,initial_eccentricity)

        return result