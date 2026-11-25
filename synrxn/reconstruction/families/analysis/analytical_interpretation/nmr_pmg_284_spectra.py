# SPDX-License-Identifier: LGPL-3.0-or-later
"""Build the PMG-284 table from the Magnetstein NMR spectra."""
from __future__ import annotations

import gzip
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile

from synrxn.reconstruction.families._selection import table_main
from synrxn.reconstruction.families.suite import physical_sha
from synrxn.reconstruction.paths import REPO, physical_path

TRAJECTORY = 'PMG_284'
SOURCE = Path('Data/raw/coverage/process/magnetstein_reactions')
SPECTRA = SOURCE / 'data/PMG_284/preprocessed_mixture.csv'
CALIBRATION = SOURCE / 'config.json'
CALIBRATION_SHA256 = 'aa13d317bd8cf7070f88b08b679e3da4596de12c4ce81c51ce2b0fc6e30cd60e'
SPECTRA_SHA256 = '6dc1d1ba2b734bf5e6cfd40670e7d8b11b24799f264b0e3cd2c3f45370ea6d70'
CALIBRATION_URL = (
    'https://raw.githubusercontent.com/BDomzal/magnetstein_x_chemical_reactions/'
    'a764ab5f3050ec99d8141b8e83f74cf51728f60a/config.json'
)
ARCHIVE_URL = 'https://zenodo.org/api/records/14814657/files/data.zip/content'
ARCHIVE_BYTES = 2264466137


def _download_spectra(target):
    """Read only the PMG-284 member of the pinned archive."""
    from synrxn.reconstruction.families.process._source_tables import RemoteZipFile

    remote = RemoteZipFile(ARCHIVE_URL, ARCHIVE_BYTES)
    with remote, remote.session, ZipFile(remote) as archive:
        with archive.open(SPECTRA.relative_to(SOURCE).as_posix()) as native, target.open('wb') as output:
            shutil.copyfileobj(native, output, length=8 * 1024 * 1024)


def acquire(repo: Path, cache: Path, offline: bool) -> str:
    """Recover the two pinned inputs from local storage or their upstream sources."""
    fetched = False
    for name, checksum in ((CALIBRATION, CALIBRATION_SHA256), (SPECTRA, SPECTRA_SHA256)):
        target = physical_path(name, repo)
        if target.is_file():
            if physical_sha(target) != checksum:
                raise ValueError(f'Cached native input has the wrong checksum: {target}')
            continue
        cached = cache / name.relative_to('Data')
        if not cached.is_file() and offline:
            raise ValueError(f'Missing PMG-284 input in offline cache: {cached}')
        target.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix='.pmg-284-', dir=target.parent) as temporary:
            staged = Path(temporary) / name.name
            if cached.is_file():
                shutil.copyfile(cached, staged)
            else:
                try:
                    if name == CALIBRATION:
                        request = Request(CALIBRATION_URL, headers={'User-Agent': 'SynRXN-data/2.0'})
                        with urlopen(request, timeout=60) as response, staged.open('wb') as output:
                            shutil.copyfileobj(response, output)
                    else:
                        _download_spectra(staged)
                except (OSError, ValueError, BadZipFile) as error:
                    raise ValueError(f'Upstream PMG-284 acquisition failed. Supply --cache DIR: {error}') from error
                fetched = True
            if physical_sha(staged) != checksum:
                raise ValueError(f'PMG-284 input does not match the pinned checksum: {name}')
            staged.replace(target)
    return 'upstream and local inputs' if fetched else 'cached native inputs'


def integrate_spectrum(ppm, intensity, component_windows):
    """Integrate the observed bins independently within each component window."""
    import numpy as np

    ppm, intensity = np.asarray(ppm), np.asarray(intensity)
    trapezoid = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
    areas = []
    for windows in component_windows:
        total = 0.0
        for lower, upper in windows:
            observed = (lower <= ppm) & (ppm <= upper) & np.isfinite(ppm) & np.isfinite(intensity)
            if observed.sum() < 2:
                raise ValueError('Insufficient observed spectral bins for reference integration')
            order = np.argsort(ppm[observed])
            total += trapezoid(intensity[observed][order], ppm[observed][order])
        areas.append(total)
    return np.asarray(areas)


def build_rows(repo: Path = REPO) -> list[dict]:
    """Read each PMG-284 spectrum and derive its signed integrals and area fractions."""
    import numpy as np

    calibration = json.loads(physical_path(CALIBRATION, repo).read_text(encoding='utf8'))
    experiment = 'PMG 284 monitoring'
    components = calibration['substances_names'][experiment]
    windows = [[window] for window in calibration['integration_intervals'][experiment]]
    path = physical_path(SPECTRA, repo)
    matrix = np.loadtxt(path, ndmin=2)
    if matrix.shape[1] != 1000:
        raise ValueError('PMG-284 requires a ppm column followed by 999 spectra')
    spectrum_hash = physical_sha(path)

    ppm = matrix[:, 0]
    rows = []
    for column, intensity in enumerate(matrix[:, 1:].T, start=1):
        signed = integrate_spectrum(ppm, intensity, windows)
        positive = np.maximum(signed, 0)
        if not np.isfinite(signed).all() or positive.sum() <= 0:
            raise ValueError(f'PMG-284 spectrum {column} has invalid component integrals')
        rows.append({
            'record_id': f'{TRAJECTORY}:spectrum_column_{column}',
            'trajectory_id': TRAJECTORY,
            'spectrum_path': SPECTRA.as_posix(),
            'spectrum_sha256': spectrum_hash,
            'spectrum_column': column,
            'native_spectrum_column': column,
            'elapsed_seconds': (column - 1) * 5.0,
            'component_names': components,
            'reference_integrals': signed.tolist(),
            'reference_area_fractions': (positive / positive.sum()).tolist(),
            'missing_signal_bins': int(np.isnan(intensity).sum()),
            'reference_kind': (
                'derived_trapezoid_integrals_from_measured_CCBY_spectra; '
                'clipped-and-normalised areas are analytical estimates, not absolute concentrations'
            ),
            'integration_windows_ppm': windows,
        })
    return rows


def build_table(repo: Path, destination: Path) -> None:
    """Write deterministic gzip JSONL in the released table format."""
    rows = build_rows(repo)
    with destination.open('wb') as raw, gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as compressed:
        for row in rows:
            line = json.dumps(row, sort_keys=True, separators=(',', ':'), allow_nan=False)
            compressed.write((line + '\n').encode('utf8'))


def main(argv=None):
    """Prepare, plan, verify, or rebuild the PMG-284 spectrum table."""
    return table_main(
        'analysis', 'analytical_interpretation', ('coverage', 'process/nmr_pmg_284_snapshots'), argv,
        build_table=build_table,
        source_pins={CALIBRATION.as_posix(): CALIBRATION_SHA256},
        acquire=acquire,
    )


if __name__ == '__main__':
    raise SystemExit(main())
