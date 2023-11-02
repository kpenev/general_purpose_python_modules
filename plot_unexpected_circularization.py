#!/usr/bin/env python3

import pickle

from matplotlib import pyplot
import numpy

if __name__ == '__main__':
    available_ei = ['0.4', '0.5', '0.6', '0.7', '0.8']
    for ei_str in available_ei:
        with open(f'unexpected_circ_max_ef_0.2_ei_{ei_str}.pkl', 'rb') as pklf:
            evolution = pickle.load(pklf)

        pyplot.loglog(
            evolution.age,
            2.0 * numpy.pi / evolution.orbital_period,
            label=f'$\Omega_{{orb}}(e_i={ei_str})$'
        )
        pyplot.loglog(
            evolution.age,
            evolution.primary_envelope_angmom / evolution.primary_iconv,
            label=f'$\Omega_\star(e_i={ei_str})$'
        )

    pyplot.legend()
    pyplot.show()
