"""Read the single FlowER reproducibility dump without extracting raw text copies."""
from __future__ import annotations

from contextlib import contextmanager
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import zipfile

from synrxn.reconstruction.families.representation.atom_mapping._sources.aam_sources import require

SCHEMA = 'synrxn.flower-reproducibility-dump.v1'
OUTPUT_SCHEMA = 'synrxn.flower-center-selection.v2'
COLUMNS = ('r_id', 'ground_truth', 'rxn', 'original_id', 'steps', 'role')


class DumpText:
    """Reopen one compressed native source for either binary hashing or text streaming."""

    def __init__(self, archive, member):
        self.archive, self.member = Path(archive), member

    @contextmanager
    def open(self, mode='r'):
        with zipfile.ZipFile(self.archive) as archive:
            with archive.open(self.member) as member:
                with gzip.GzipFile(fileobj=member) as source:
                    if 'b' in mode:
                        yield source
                    else:
                        with io.TextIOWrapper(source, encoding='utf8') as text:
                            yield text


class FlowerDump:
    """Verify current recipe inputs and snapshots; history is retained but not used as input."""

    def __init__(self, path):
        self.path = Path(path).resolve()
        with zipfile.ZipFile(self.path) as archive:
            self.metadata = json.loads(archive.read('source.json'))
            require(self.metadata['schema'] == SCHEMA, 'Unsupported FlowER dump')
            names = [item.filename for item in archive.infolist()]
            require(len(names) == len(set(names)), 'Duplicate dump members')
            for member, expected in self.metadata['files'].items():
                if member.startswith('history/'):
                    continue
                with archive.open(member) as source:
                    actual = hashlib.file_digest(source, 'sha256').hexdigest()
                require(actual == expected, f'Dump checksum mismatch: {member}')
        self.sha256 = self.digest()

    def digest(self):
        with self.path.open('rb') as source:
            return hashlib.file_digest(source, 'sha256').hexdigest()

    def read(self, member):
        with zipfile.ZipFile(self.path) as archive:
            value = archive.read(member)
        require(hashlib.sha256(value).hexdigest() == self.metadata['files'][member],
                f'Dump checksum mismatch: {member}')
        return value

    def copy(self, member, target):
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(self.path) as archive:
            with archive.open(member) as source, target.open('wb') as output:
                shutil.copyfileobj(source, output, 1024 * 1024)
        with target.open('rb') as source:
            checksum = hashlib.file_digest(source, 'sha256').hexdigest()
        require(checksum == self.metadata['files'][member], f'Dump copy checksum mismatch: {member}')

    def elementary(self, entry):
        value = gzip.decompress(self.read(entry['elementary_member'])).decode()
        rows = list(csv.DictReader(io.StringIO(value)))
        require(all(row['n_steps'] == '1' for row in rows), 'Unexpected multistep elementary input')
        require(len(rows) == entry['expected_steps']['1'], 'Elementary selection count mismatch')
        return rows

    def text(self, entry):
        return DumpText(self.path, entry['member'])
