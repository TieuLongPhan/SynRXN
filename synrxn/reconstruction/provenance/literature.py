# Direct implementation; historical source identity is explicit below.
"""Validate the research artifacts and downloaded source integrity offline."""
from synrxn.reconstruction.paths import REPO, historical_file as __historical_file, source_path
from synrxn.reconstruction.provenance.compat import configure_runtime as __configure_runtime
__source_file__ = __historical_file('kdd/literature/validate.py')
import argparse
import csv
import hashlib
import json
import re
from synrxn.reconstruction.paths import HistoricalPath as Path
ROOT = Path(__source_file__).resolve().parent

def markdown_link_problems(root):
    """Check links in owned documents, leaving archived source bytes untouched."""
    problems = []
    for path in root.rglob('*.md'):
        relative = path.relative_to(root)
        if 'raw' in relative.parts or 'papers' in relative.parts:
            continue
        if relative.is_relative_to('coverage/rights_recheck'):
            continue
        for target in re.findall('\\]\\(([^)]+)\\)', path.read_text()):
            target = target.strip('<>')
            if target.startswith(('https://', 'http://', '#', 'mailto:')):
                continue
            file_target = target.split('#')[0]
            if file_target and not source_path(path.parent / file_target).exists():
                problems.append(str(relative) + ' has missing link ' + target)
    return problems

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        help='Write a validation report here; default prints without modifying inputs')
    args = parser.parse_args(argv)
    problems = []
    selected = json.loads((ROOT / 'selected_papers.json').read_text())['papers']
    ids = {record['id'] for record in selected}
    if len(ids) != len(selected):
        problems.append('Duplicate selected paper IDs')
    downloads = json.loads((ROOT / 'downloads.json').read_text())['results']
    if {record['paper_id'] for record in downloads} != ids:
        problems.append('Download ledger does not cover exactly the selected papers')
    pdf_count = 0
    for record in downloads:
        if record['status'] != 'downloaded':
            continue
        path = ROOT / record['pdf']
        data = path.read_bytes()
        if not data.lstrip().startswith(b'%PDF-'):
            problems.append(str(path) + ' is not a PDF')
        if hashlib.sha256(data).hexdigest() != record['sha256'] or len(data) != record['bytes']:
            problems.append(str(path) + ' differs from its pinned bytes')
        if not record.get('text') or not (ROOT / record['text']).exists():
            problems.append(str(path) + ' has no extracted text')
        pdf_count += 1
    for record in json.loads((ROOT / 'full_text_xml.json').read_text())['records']:
        if record.get('file'):
            path = ROOT / record['file']
            if hashlib.sha256(path.read_bytes()).hexdigest() != record['request']['sha256']:
                problems.append(str(path) + ' differs from its source hash')
    reviews = json.loads((ROOT / 'reviews.json').read_text())['results']
    for forum in reviews:
        if forum['paper_id'] not in ids:
            problems.append('Unmatched review paper ' + forum['paper_id'])
        if re.search('/(2023|2027)(?:/|$)', forum['review_venue_root']):
            problems.append('Review venue outside 2024-2026: ' + forum['review_venue_root'])
        note_ids = [note['id'] for note in forum['notes']]
        if len(note_ids) != len(set(note_ids)):
            problems.append('Duplicate notes in ' + forum['forum_id'])
        if sum(forum['counts'].values()) != len(forum['notes']):
            problems.append('Inconsistent review counts in ' + forum['forum_id'])
    screening = list(csv.DictReader((ROOT / 'track_screening.csv').open()))
    official = json.loads((ROOT / 'track_inventory.json').read_text())['records']
    if len(screening) != len(official):
        problems.append('Screening ledger does not cover every official KDD record')
    if {row['paper_id'] for row in screening if row['decision'] == 'include'} != {record['id'] for record in selected if record['cohort'] == 'kdd_track'}:
        problems.append('Selected KDD papers differ from screening decisions')
    bibliography = (ROOT / 'references.bib').read_text()
    if set(re.findall('@\\w+\\{([^,]+),', bibliography)) != ids:
        problems.append('Bibliography does not cover exactly the selected papers')
    # Source ledgers outside the literature directory retain historical links;
    # the current documentation tree is checked by its own Sphinx build.
    problems.extend(markdown_link_problems(ROOT))
    report = {'selected_papers': len(ids), 'pdf_hashes_verified': pdf_count, 'kdd_track_screening_records': len(screening), 'review_forums_checked': len(reviews), 'problems': problems, 'passed': not problems}
    if args.output is not None:
        output = args.output.resolve()
        if output.is_relative_to(REPO / 'archive'):
            parser.error('Historical evidence is read-only; choose a run output outside archive/')
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf8')
    print(json.dumps(report))
    raise SystemExit(1 if problems else 0)
__configure_runtime(globals(), 'kdd/literature/validate.py')
if __name__ == '__main__':
    main()
