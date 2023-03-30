#!/usr/bin/env python3

"""Automate downloading CMD isochrone data."""

import requests
from bs4 import BeautifulSoup
from astropy import units as u


#TODO: decite retrun format: file vs interpolator
def query_cmd(age, feh=0.0, visual_extinction=0.0, output_fname=None):
    """
    Query the CMD database for a single isochrone with the given parameters.

    Args:
        visual_extinction(float):    The visual extinciton to query in
            magnitudes.

        age(astropy time):    The age to query for the isochrone, along with
            appropriate units.

        feh(float):    The [Fe/H] value to query.

        output_fname(str or None):    The file name to use for saving the
            downloaded isochrone. If `None`, the isochrone is returned as a
            string.

    Returns:
        str or None:
            If output_fname is specified, returns None, else returns the
    """

    cmd_url = 'http://stev.oapd.inaf.it/cgi-bin'
    age_str = '%.5f' % age.to_value('yr')
    feh_str = '%.5f' % feh
    response = requests.post(
        cmd_url + '/cmd',
        {
            'cmd_version': '3.3',
            'track_parsec': 'parsec_CAF09_v1.2S',
            'track_colibri': 'parsec_CAF09_v1.2S_S35',
            'track_postagb': 'no',
            'n_inTPC': '10',
            'eta_reimers': '0.2',
            'kind_interp': '1',
            'kind_postagb': '-1',
            'photsys_file': 'tab_mag_odfnew/tab_mag_ubvrijhk.dat',
            'photsys_version': 'YBC',
            'dust_sourceM': 'dpmod60alox40',
            'dust_sourceC': 'AMCSIC15',
            'kind_mag': '2',
            'kind_dust': '0',
            'extinction_av': '%.3f' % visual_extinction,
            'extinction_coeff': 'constant',
            'extinction_curve': 'cardelli',
            'imf_file': 'tab_imf/imf_kroupa_orig.dat',
            'isoc_isagelog': '0',
            'isoc_agelow': age_str,
            'isoc_ageupp': age_str,
            'isoc_dage': '0.0',
            'isoc_dlage': '0.0',
            'isoc_ismetlog': '1',
            'isoc_zlow': '0.0152',
            'isoc_zupp': '0.03',
            'isoc_dz': '0.0',
            'isoc_metlow': feh_str,
            'isoc_metupp': feh_str,
            'isoc_dmet': '0.0',
            'output_kind': '0',
            'output_evstage': '1',
            'lf_maginf': '-15',
            'lf_magsup': '20',
            'lf_deltamag': '0.5',
            'sim_mtot': '1.0e4',
            'output_gzip': '0',
            'submit_form': 'Submit',
            '.cgifields': ['photsys_version',
                           'isoc_ismetlog',
                           'dust_sourceC',
                           'isoc_isagelog',
                           'track_colibri',
                           'output_gzip',
                           'track_parsec',
                           'dust_sourceM',
                           'output_kind',
                           'extinction_coeff',
                           'track_postagb',
                           'extinction_curve']
        }
    )
    bs_response = BeautifulSoup(response.text, 'html.parser')
    downloaded = False
    for link in bs_response.find_all('a'):
        link_url = link.get('href')
        if link_url.endswith('.dat'):
            assert not downloaded
            data_url = '%s/%s' % (cmd_url, link_url)
            print('Downloading: ' + data_url)
            if output_fname:
                with open(output_fname, 'wb') as destination:
                    destination.write(
                        requests.get(data_url, allow_redirects=True).content
                    )
            else:
                result = requests.get(data_url, allow_redirects=True).content
            print('Done')
            downloaded = True
    assert downloaded
    return result

if __name__ == '__main__':
    print(query_cmd(age=1.0 * u.Gyr).decode())
