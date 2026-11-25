"""Stream uncapped Flower paths and checkpoint MC assignments in bounded batches."""
from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
from itertools import islice
import json
import multiprocessing as mp
from pathlib import Path
import re
import time

import networkx as nx

from synrxn.reconstruction.families.representation.atom_mapping._sources.aam_sources import require, write_json
from synrxn.reconstruction.families.representation.atom_mapping._sources.flower_mc_export import pathway_endpoints
from synrxn.reconstruction.families.representation.atom_mapping._sources.flower_mc_store import DiskClusters, set_meta
from synrxn.reconstruction.families.representation.atom_mapping._sources.flower_mechanistic_center import growing_centers, mechanistic_center, pathway_mtg

MAP = re.compile(r':(\d+)\]')
GROUP = re.compile(r'\d+(?:::pd\d+)?\Z')


def initialize_worker():
    import logging
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.warning')
    RDLogger.DisableLog('rdApp.error')
    logging.getLogger('synkit').setLevel(logging.CRITICAL)


def source_index(path, expected):
    """Verify complete source bytes and locate the last line of interleaved groups."""
    endings, checksum = {}, hashlib.sha256()
    with path.open('rb') as handle:
        for number, line in enumerate(handle, 1):
            checksum.update(line)
            group = line.rstrip(b'\r\n').rsplit(b'|', 1)[-1].decode()
            if GROUP.fullmatch(group):
                endings[group] = number
    require(checksum.hexdigest() == expected, 'Original source checksum mismatch')
    return endings


def source_groups(path, endings, counters):
    """Release each source mechanism at its last recorded line, retaining witnesses."""
    active, terminals = defaultdict(list), defaultdict(set)
    with path.open() as handle:
        for number, line in enumerate(handle, 1):
            counters['raw_source_lines'] += 1
            try:
                reaction, group = line.rstrip('\r\n').rsplit('|', 1)
                left, right = reaction.split('>>')
            except ValueError:
                counters['malformed_lines'] += 1
                continue
            left, right = ('.'.join(sorted(side.split('.'))) for side in (left, right))
            first, second = MAP.findall(left), MAP.findall(right)
            valid = (bool(first) and len(first) == len(set(first))
                     and len(second) == len(set(second)) and set(first) == set(second))
            if left == right:
                counters['no_change_markers'] += 1
            elif valid:
                counters['elementary_source_rows'] += 1
            else:
                counters['nonbijective_rows'] += 1
            if not GROUP.fullmatch(group):
                continue
            if left == right:
                terminals[group].add(left)
            elif valid:
                active[group].append((left, right, number))
            if endings[group] == number:
                yield group, active.pop(group, []), terminals.pop(group, set())
                del endings[group]
    require(not active and not terminals and not endings, 'Unfinished source groups')


def choose_paths(edges, terminals, seed_cap):
    """Keep separate first-edge seeds and follow one continuation until natural termination."""
    adjacency, indegree, seen = defaultdict(list), Counter(), set()
    for edge in edges:
        left, right, _ = edge
        if (left, right) not in seen:
            adjacency[left].append(edge)
            indegree[right] += 1
            seen.add((left, right))
    roots = sorted((edge for choices in adjacency.values() for edge in choices if not indegree[edge[0]]),
                   key=lambda edge: edge[2])
    statistics = Counter(seeds_selected=min(len(roots), seed_cap), seeds_skipped=max(0, len(roots) - seed_cap))
    paths = []
    for seed in roots[:seed_cap]:
        path, visited = [seed], set(seed[:2])
        while True:
            choices = adjacency.get(path[-1][1], [])
            if not choices:
                statistics['stop_recorded_terminal' if path[-1][1] in terminals else 'stop_no_continuation'] += 1
                break
            statistics['downstream_alternatives_skipped'] += len(choices) - 1
            following = choices[0]
            if following[1] in visited:
                statistics['stop_revisited_state'] += 1
                break
            path.append(following)
            visited.add(following[1])
        paths.append(path)
        statistics[f'length:{len(path)}'] += 1
    return paths, dict(statistics)


def member_record(path, length, group, entry):
    lines = json.dumps([edge[2] for edge in path[:length]], separators=(',', ':'))
    seed = hashlib.sha256(f"{entry['sha256']}:{group}:{path[0][2]}".encode()).hexdigest()
    identity = ':'.join((entry['sha256'], group, str(length), seed, lines))
    raw = path[0][0] + '>>' + path[length - 1][1]
    return {'membership_id': hashlib.sha256(identity.encode()).hexdigest(), 'source_split': entry['split'],
            'source_group': group, 'parent_source_group': group.split('::', 1)[0], 'n_steps': str(length),
            'seed_id': seed, 'source_lines': lines, 'reference_id': hashlib.sha256(raw.encode()).hexdigest(),
            'rc_cluster_id': ''}


def fallback_centers(reactions):
    """Retain valid shorter prefixes even if a later step cannot form an MTG."""
    for length in range(2, len(reactions) + 1):
        try:
            graph, statistics = mechanistic_center(pathway_mtg(reactions[:length]))
            yield length, graph, statistics, None
        except Exception as error:
            yield length, None, None, f'{type(error).__name__}: {error}'


def project_group(task):
    group, edges, terminals, entry, seed_cap = task
    paths, statistics = choose_paths(edges, terminals, seed_cap)
    results = []
    for path in paths:
        if len(path) < 2:
            continue
        reactions = [left + '>>' + right for left, right, _ in path]
        try:
            projected = [(length, graph, stats, None) for length, graph, stats in growing_centers(reactions)]
        except Exception:
            projected = list(fallback_centers(reactions))
        for length, graph, stats, error in projected:
            record = member_record(path, length, group, entry)
            active = pathway_endpoints(reactions[:length], set(graph)) if graph else None
            if not graph and error is None:
                error = 'Empty mechanistic center'
            if graph:
                graph.graph['mc_bucket_hash'] = nx.weisfeiler_lehman_graph_hash(
                    graph, node_attr='label', edge_attr='label', iterations=3)
            results.append((record, graph, stats, active, error))
    return group, statistics, results


def save_groups(db, clusters, groups, seed, split):
    """Commit whole source groups, class templates and ranked witnesses atomically."""
    for group, statistics, results in groups:
        for record, graph, stats, active, error in results:
            length, identity = int(record['n_steps']), record['membership_id']
            center = None if error else clusters.assign(graph)
            packed = None if error else json.dumps({'nodes': [[node, {}] for node in sorted(graph)]})
            values = (identity, json.dumps(record, sort_keys=True), length, '',
                      'failed' if error else 'done', center, packed, json.dumps(stats), active, error)
            db.execute('INSERT INTO members(member_id,record,steps,legacy_rc,status,mc,graph,stats,active,error) '
                       'VALUES(?,?,?,?,?,?,?,?,?,?)', values)
            if not error:
                rank = f"{seed}:{split}:{length}:{record['reference_id']}:{identity}"
                priority = hashlib.sha256(rank.encode()).hexdigest()
                db.execute('INSERT INTO priorities VALUES(?,?,?,?)', (identity, length, center, priority))
        db.execute('INSERT INTO completed_groups VALUES(?,?)', (group, json.dumps(statistics)))
    db.commit()


def memory_usage():
    import os
    resident = 0
    for process in [os.getpid(), *(child.pid for child in mp.active_children())]:
        try:
            resident += int(Path(f'/proc/{process}/statm').read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE')
        except (OSError, ValueError, IndexError):
            pass
    available = next(int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines()
                     if line.startswith('MemAvailable:'))
    return resident, available


def process_sources(db, source, entry, args, progress_path):
    """Process source groups with bounded queues, worker recycling and durable resume."""
    endings = source_index(source, entry['sha256'])
    completed = {row[0] for row in db.execute('SELECT source_group FROM completed_groups')}
    counters = Counter()
    pending = ((group, edges, markers, entry, args.max_seeds)
               for group, edges, markers in source_groups(source, endings, counters) if group not in completed)
    clusters = DiskClusters(db)
    done = len(completed)
    classes = db.execute('SELECT COUNT(*) FROM centers').fetchone()[0]
    started, processed, peak = time.monotonic(), 0, 0
    batch = list(islice(pending, args.batch_size))
    while batch:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize_worker,
                                 mp_context=mp.get_context('spawn')) as pool:
            futures = pool.map(project_group, batch, chunksize=1)
            for cycle in range(args.recycle_batches):
                results = list(futures)
                following = list(islice(pending, args.batch_size)) if cycle + 1 < args.recycle_batches else None
                next_futures = pool.map(project_group, following, chunksize=1) if following else None
                try:
                    save_groups(db, clusters, results, args.seed, entry['split'])
                except BaseException:
                    db.rollback()
                    clusters.cache.clear()
                    raise
                done += len(batch)
                processed += sum(len(result[2]) for result in results)
                classes = db.execute('SELECT COUNT(*) FROM centers').fetchone()[0]
                resident, available = memory_usage()
                peak = max(peak, resident)
                status = {'stage': 'centers', 'split': entry['split'], 'completed_groups': done,
                          'new_prefixes': processed, 'mc_classes': classes, 'source_rows': counters['raw_source_lines'],
                          'elapsed_seconds': round(time.monotonic() - started, 1),
                          'rss_mib': round(resident / 2**20, 1),
                          'peak_rss_mib': round(peak / 2**20, 1),
                          'available_gib': round(available / 2**30, 1)}
                write_json(progress_path, status)
                print(json.dumps(status), flush=True)
                if args.max_new_groups and done - len(completed) >= args.max_new_groups:
                    return False
                pressured = resident > args.memory_budget_gib * 2**30 or available < 2 * 2**30
                if following is None:
                    batch = list(islice(pending, args.batch_size))
                    break
                batch, futures = following, next_futures
                if pressured:
                    args.workers = max(1, args.workers // 2)
                    break
                if not batch:
                    break
    set_meta(db, 'source_counts', dict(counters))
    set_meta(db, 'source_complete', True)
    db.commit()
    return True
