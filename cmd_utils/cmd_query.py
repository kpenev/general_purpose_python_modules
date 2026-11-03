#!/usr/bin/env python3

"""Automate downloading CMD isochrone data."""

import io
import logging
import ssl
import tempfile
from urllib.parse import urlparse

import certifi
import requests
from bs4 import BeautifulSoup
from astropy import units as u
from cryptography import x509
from cryptography.hazmat.primitives.serialization import Encoding
from cryptography.x509.oid import AuthorityInformationAccessOID

_cmd_url = "https://stev.oapd.inaf.it/cgi-bin"

_logger = logging.getLogger(__name__)


def _get_ca_bundle(host, timeout):
    """
    Return a CA bundle that also contains the intermediate the host omits.

    The CMD server sends only its own certificate, not the intermediate that
    signed it, so verification against the standard roots fails. Download the
    intermediate from the address listed in the server certificate, check that
    a trusted root signed it and that it signed the server certificate, then
    return a bundle of the standard roots plus that intermediate.
    """

    server_cert = x509.load_pem_x509_certificate(
        ssl.get_server_certificate((host, 443), timeout=timeout).encode()
    )
    issuer_urls = [
        description.access_location.value
        for description in server_cert.extensions.get_extension_for_class(
            x509.AuthorityInformationAccess
        ).value
        if description.access_method == AuthorityInformationAccessOID.CA_ISSUERS
    ]
    intermediate = x509.load_der_x509_certificate(
        requests.get(issuer_urls[0], timeout=timeout).content
    )
    with open(certifi.where(), "rb") as roots_file:
        roots_pem = roots_file.read()
    roots = [
        root
        for root in x509.load_pem_x509_certificates(roots_pem)
        if root.subject == intermediate.issuer
    ]
    if not roots:
        raise ssl.SSLError(
            f"No trusted root found for {intermediate.issuer.rfc4514_string()}"
        )
    intermediate.verify_directly_issued_by(roots[0])
    server_cert.verify_directly_issued_by(intermediate)

    with tempfile.NamedTemporaryFile(
        "wb", prefix="cmd_ca_bundle_", suffix=".pem", delete=False
    ) as bundle:
        bundle.write(roots_pem)
        bundle.write(b"\n" + intermediate.public_bytes(Encoding.PEM))
    _logger.info(
        "Using CA bundle %s with intermediate %s",
        bundle.name,
        intermediate.subject.rfc4514_string(),
    )
    return bundle.name


def _get_session(timeout):
    """Return a requests session able to verify the CMD server."""

    session = requests.Session()
    try:
        session.head(_cmd_url, timeout=timeout)
    except requests.exceptions.SSLError:
        session.verify = _get_ca_bundle(urlparse(_cmd_url).hostname, timeout)
    return session


def _format_query_range(value, units=None):
    """Return properly formatted min, max, step for given value."""

    try:
        assert len(value) == 3
    except TypeError:
        value = (value, value, 0.0)
    return tuple(
        f"{entry.to_value(units) if units else entry:.5f}" for entry in value
    )


def _grid_points(value, units=None):
    """
    Return the grid values CMD uses for a single value or (min, max, step).

    CMD uses floor((max - min) / step) + 1 points at min + i * step. For
    example, it reports 30 metallicities for -2.1 to 0.8 in steps of 0.1, but
    returns only 0.5 and 0.6 for 0.5 to 0.7 in steps of 0.1.
    """

    try:
        assert len(value) == 3
    except TypeError:
        value = (value, value, 0.0)
    low, high, step = (
        entry.to_value(units) if units else entry for entry in value
    )
    if step == 0:
        return [low]
    return [
        low + index * step for index in range(int((high - low) // step) + 1)
    ]


def _submit_query(
    *,
    age=(
        6.0 * u.yr * u.dex,
        10.0 * u.yr * u.dex,
        0.1 * u.yr * u.dex,
    ),
    feh=0.0,
    visual_extinction=0.0,
    photsys="ubvrijhk",
    cmd_version=None,
    session,
    timeout,
):
    """Submit the query to the CMD web interface."""

    feh = _format_query_range(feh)
    try:
        age = _format_query_range(age, u.yr * u.dex)
        log_age = "1"
    except u.UnitConversionError:
        age = _format_query_range(age, u.yr)
        log_age = "0"
    cmd_url = _cmd_url + "/cmd" + (f"_{cmd_version}" if cmd_version else "")
    print(f"Submitting form at: {cmd_url}")
    query_values = {
        "cmd_version": cmd_version,
        "track_omegai": "0.00",
        "track_parsec": "parsec_CAF09_v1.2S",
        "track_colibri": "parsec_CAF09_v1.2S_S_LMC_08_web",
        "track_postagb": "no",
        "n_inTPC": "10",
        "eta_reimers": "0.2",
        "kind_interp": "1",
        "kind_postagb": "-1",
        "photsys_file": f"YBC_tab_mag_odfnew/tab_mag_{photsys}.dat",
        "photsys_version": "YBCnewVega",
        "dust_sourceM": "dpmod60alox40",
        "dust_sourceC": "AMCSIC15",
        "kind_mag": "2",
        "kind_dust": "0",
        "extinction_av": f"{visual_extinction:.3f}",
        "extinction_coeff": "constant",
        "extinction_curve": "cardelli",
        "kind_LPV": "3",
        "imf_file": "tab_imf/imf_kroupa_orig.dat",
        "isoc_isagelog": log_age,
        "isoc_agelow": age[0],
        "isoc_ageupp": age[1],
        "isoc_dage": age[2],
        "isoc_lagelow": age[0],
        "isoc_lageupp": age[1],
        "isoc_dlage": age[2],
        "isoc_ismetlog": "1",
        "isoc_zlow": "0.0152",
        "isoc_zupp": "0.03",
        "isoc_dz": "0.0",
        "isoc_metlow": feh[0],
        "isoc_metupp": feh[1],
        "isoc_dmet": feh[2],
        "output_kind": "0",
        "output_evstage": "1",
        "lf_maginf": "-15",
        "lf_magsup": "20",
        "lf_deltamag": "0.5",
        "sim_mtot": "1.0e4",
        "output_gzip": "0",
        "submit_form": "Submit",
        ".cgifields": [
            "photsys_version",
            "isoc_ismetlog",
            "dust_sourceC",
            "isoc_isagelog",
            "track_colibri",
            "output_gzip",
            "track_parsec",
            "dust_sourceM",
            "output_kind",
            "extinction_coeff",
            "track_postagb",
            "extinction_curve",
        ],
    }
    print(
        "Query:\n\t"
        + "\n\t".join(f"{k}: {v!r}" for k, v in query_values.items())
    )
    response = session.post(
        cmd_url,
        query_values,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.text


def _download_single(session, timeout, **query_args):
    """Submit one query (within the server limit) and return the file."""

    bs_response = BeautifulSoup(
        _submit_query(session=session, timeout=timeout, **query_args),
        "html.parser",
    )
    data_urls = [
        f"{_cmd_url}/{link.get('href')}"
        for link in bs_response.find_all("a")
        if (link.get("href") or "").endswith(".dat")
    ]
    if len(data_urls) != 1:
        errors = [
            message.get_text(" ", strip=True)
            for message in bs_response.find_all(class_="errorwarning")
        ]
        raise RuntimeError(
            f"CMD returned {len(data_urls)} data files instead of 1: "
            f"{' '.join(errors) or 'no error message found'}"
        )
    print("Downloading: " + data_urls[0])
    download = session.get(data_urls[0], allow_redirects=True, timeout=timeout)
    download.raise_for_status()
    print("Done")
    return download.content


def _append_isochrones(destination, content, keep_header, sort_by=None):
    """
    Append the isochrones in ``content`` to ``destination``.

    The comment lines before the first column header are written only if
    ``keep_header`` is True, and the ``#isochrone terminated`` line is
    dropped, so several downloads can be joined into one file.

    If ``sort_by`` names a column (e.g. ``"Mini"``), the rows of each isochrone
    are stably sorted by that column. CMD occasionally returns rows out of
    order, which breaks interpolating along that column.
    """

    def flush(rows):
        """Write the rows of one isochrone, sorted if requested."""

        if sort_by is not None:
            rows = sorted(rows, key=lambda row: float(row.split()[sort_column]))
        destination.writelines(rows)

    in_header = True
    sort_column = None
    rows = []
    for line in content.splitlines(keepends=True):
        if not line.startswith(b"#"):
            rows.append(line)
            continue
        flush(rows)
        rows = []
        if line.startswith(b"# Zini"):
            in_header = False
            if sort_by is not None:
                sort_column = line[1:].split().index(sort_by.encode())
        if line.startswith(b"#isochrone terminated"):
            continue
        if in_header and not keep_header:
            continue
        destination.write(line)
    flush(rows)


def query_cmd(
    output_fname=None,
    timeout=600.0,
    *,
    age=(
        6.0 * u.yr * u.dex,
        10.0 * u.yr * u.dex,
        0.1 * u.yr * u.dex,
    ),
    feh=0.0,
    max_isochrones=400,
    sort_by=None,
    **query_args,
):
    """
    Query the CMD database for isochrones with the given parameters.

    The CMD server refuses requests for more than 400 isochrones. Larger grids
    are downloaded one metallicity at a time (and in consecutive age ranges if
    a single metallicity is still too many), then joined into one file with a
    single header and a single ``#isochrone terminated`` line.

    Args:
        output_fname(str or None):    The file name to use for saving the
            downloaded isochrones. If `None`, they are returned as bytes.

        timeout(float):    The number of seconds to allow for each query to
            complete.

        age(astropy time or 3-tuple):    The age, or (min, max, step) range of
            ages, to query, along with appropriate units.

        feh(float or 3-tuple):    The [M/H] value, or (min, max, step) range,
            to query.

        max_isochrones(int):    The largest number of isochrones to request
            in a single query.

        sort_by(str or None):    If not None, the column (e.g. ``"Mini"``) by
            which to stably sort the rows of each isochrone. CMD occasionally
            returns rows out of order, which breaks interpolation along that
            column.

        query_args:    Passed on to the CMD form, e.g. ``photsys`` (the part of
            the CMD file name between ``tab_mag_`` and ``.dat``, such as
            ``"TESS_2mass"``), ``visual_extinction``, and ``cmd_version``.

    Returns:
        bytes or None:
            If output_fname is specified, returns None, else returns the
            downloaded file content.
    """

    try:
        _format_query_range(age, u.yr * u.dex)
        age_units = u.yr * u.dex
    except u.UnitConversionError:
        age_units = u.yr
    age_values = _grid_points(age, age_units)
    feh_values = _grid_points(feh)

    if len(age_values) * len(feh_values) <= max_isochrones:
        chunks = [(age, feh)]
    else:
        if len(age_values) <= max_isochrones:
            age_ranges = [age]
        else:
            # Half a step past the last age, so CMD's floor() keeps it.
            age_ranges = [
                (
                    age_values[start] * age_units,
                    age_values[min(start + max_isochrones, len(age_values)) - 1]
                    * age_units
                    + age[2] / 2,
                    age[2],
                )
                for start in range(0, len(age_values), max_isochrones)
            ]
        chunks = [
            (age_range, (feh_value, feh_value, 0.0))
            for feh_value in feh_values
            for age_range in age_ranges
        ]
        print(
            f"Requesting {len(age_values) * len(feh_values)} isochrones in "
            f"{len(chunks)} queries."
        )

    session = _get_session(timeout)
    with (
        open(output_fname, "wb") if output_fname else io.BytesIO()
    ) as destination:
        for chunk_index, (chunk_age, chunk_feh) in enumerate(chunks):
            _append_isochrones(
                destination,
                _download_single(
                    session,
                    timeout,
                    age=chunk_age,
                    feh=chunk_feh,
                    **query_args,
                ),
                keep_header=chunk_index == 0,
                sort_by=sort_by,
            )
        destination.write(b"#isochrone terminated\n")
        if output_fname:
            return None
        return destination.getvalue()


if __name__ == "__main__":
    query_cmd(
        age=(1.0 * u.Myr, 13.0 * u.Gyr, 0.1 * u.Gyr),
        feh=(-0.2, 0.2, 0.2),
        cmd_version="3.7",
        output_fname="downloaded.ssv",
    )
