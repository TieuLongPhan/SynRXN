"""Durable pathway assignments and bounded exact MC classes for Flower splits."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from collections import OrderedDict

import networkx as nx

from synrxn.reconstruction.families.representation.atom_mapping._sources.aam_sources import require
from synrxn.reconstruction.families.representation.atom_mapping._sources.flower_mechanistic_center import ExactClusters

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS members(
 id INTEGER PRIMARY KEY,member_id TEXT UNIQUE NOT NULL,record TEXT NOT NULL,
 steps INTEGER NOT NULL,legacy_rc TEXT NOT NULL,status TEXT NOT NULL,
 mc INTEGER,graph TEXT,stats TEXT,active TEXT,error TEXT);
CREATE INDEX IF NOT EXISTS pending_members ON members(id) WHERE status='pending';
CREATE TABLE IF NOT EXISTS centers(id INTEGER PRIMARY KEY,graph TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS normalized(
 member_id TEXT PRIMARY KEY,ground_truth TEXT,rxn TEXT,details TEXT,error TEXT);
CREATE TABLE IF NOT EXISTS completed_groups(source_group TEXT PRIMARY KEY,statistics TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS center_buckets(
 id INTEGER PRIMARY KEY, nodes INTEGER,edges INTEGER,signature TEXT);
CREATE INDEX IF NOT EXISTS center_bucket ON center_buckets(nodes,edges,signature);
CREATE TABLE IF NOT EXISTS priorities(member_id TEXT PRIMARY KEY,steps INTEGER,mc INTEGER,priority TEXT);
CREATE INDEX IF NOT EXISTS priority_bucket ON priorities(steps,mc,priority);
"""


def pack_graph(graph: nx.Graph) -> str:
    data = {
        "nodes": [[node, attrs] for node, attrs in sorted(graph.nodes(data=True))],
        "edges": sorted([min(left, right), max(left, right), attrs] for left, right, attrs in graph.edges(data=True)),
    }
    return json.dumps(data, separators=(",", ":"), sort_keys=True)


def unpack_graph(text: str) -> nx.Graph:
    value = json.loads(text)
    graph = nx.Graph()
    graph.add_nodes_from(value["nodes"])
    graph.add_edges_from(value["edges"])
    return graph


def get_meta(db, key, default=None):
    row = db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return json.loads(row[0]) if row else default


def set_meta(db, key, value):
    db.execute("INSERT OR REPLACE INTO meta VALUES(?,?)", (key, json.dumps(value, sort_keys=True)))


def open_store(path: Path, configuration: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=FULL")
    db.execute("PRAGMA foreign_keys=ON")
    db.execute("PRAGMA cache_size=-32768")
    db.execute("PRAGMA temp_store=FILE")
    db.executescript(SCHEMA)
    previous = get_meta(db, "configuration")
    if previous is not None and previous != configuration:
        db.close()
        raise ValueError("Checkpoint source, recipe or implementation changed; select a fresh --checkpoint")
    set_meta(db, "configuration", configuration)
    db.commit()
    return db


def prepare_members(db, records: list[dict]) -> None:
    if get_meta(db, "prepared", False):
        require(
            db.execute("SELECT COUNT(*) FROM members").fetchone()[0] == len(records), "Checkpoint membership mismatch"
        )
        return
    for row in records:
        steps = int(row["n_steps"])
        db.execute(
            "INSERT INTO members(member_id,record,steps,legacy_rc,status) VALUES(?,?,?,?,?)",
            (
                row["membership_id"],
                json.dumps(row, sort_keys=True),
                steps,
                row["rc_cluster_id"],
                "elementary" if steps == 1 else "pending",
            ),
        )
    set_meta(db, "prepared", True)
    db.commit()


def load_clusters(db) -> ExactClusters:
    clusters = ExactClusters()
    for number, raw in db.execute("SELECT id,graph FROM centers ORDER BY id"):
        require(clusters.assign(unpack_graph(raw)) == number, "Stored MC class IDs are inconsistent")
    return clusters


def save_batch(db, clusters: ExactClusters, results: list[tuple]) -> None:
    """Commit all assignments and any new class templates in one transaction."""
    try:
        for member_id, graph, stats, active, error in results:
            if error:
                db.execute("UPDATE members SET status='failed',error=? WHERE member_id=?", (error, member_id))
                continue
            center = clusters.assign(graph)
            packed = pack_graph(graph)
            db.execute("INSERT OR IGNORE INTO centers VALUES(?,?)", (center, packed))
            db.execute(
                "UPDATE members SET status='done',mc=?,graph=?,stats=?,active=? WHERE member_id=?",
                (center, packed, json.dumps(stats, sort_keys=True), active, member_id),
            )
        db.commit()
    except BaseException:
        db.rollback()
        raise


def counts(db) -> dict:
    return dict(db.execute("SELECT status,COUNT(*) FROM members GROUP BY status"))


class DiskClusters:
    """Use exact labelled MC equality with a bounded graph cache and SQL buckets."""

    def __init__(self, db, capacity=128):
        self.db, self.capacity = db, capacity
        self.cache = OrderedDict()

    def assign(self, graph):
        signature = graph.graph.get("mc_bucket_hash") or nx.weisfeiler_lehman_graph_hash(
            graph, node_attr="label", edge_attr="label", iterations=3)
        bucket = (len(graph), graph.number_of_edges(), signature)
        match = nx.algorithms.isomorphism.categorical_node_match("label", None)
        edge_match = nx.algorithms.isomorphism.categorical_edge_match("label", None)
        query = "SELECT id FROM center_buckets WHERE nodes=? AND edges=? AND signature=?"
        for (number,) in self.db.execute(query, bucket):
            if number not in self.cache:
                raw = self.db.execute("SELECT graph FROM centers WHERE id=?", (number,)).fetchone()[0]
                self.cache[number] = unpack_graph(raw)
            self.cache.move_to_end(number)
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)
            if nx.is_isomorphic(graph, self.cache[number], node_match=match, edge_match=edge_match):
                return number
        number = self.db.execute("SELECT COALESCE(MAX(id),-1)+1 FROM centers").fetchone()[0]
        self.db.execute("INSERT INTO centers VALUES(?,?)", (number, pack_graph(graph)))
        self.db.execute("INSERT INTO center_buckets VALUES(?,?,?,?)", (number, *bucket))
        return number
