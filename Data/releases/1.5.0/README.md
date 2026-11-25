# SynRXN 1.5.0 prepared tables

This directory contains 102 admitted tables from the main SynRXN
1.5.0 snapshot, with 5,485,077 stored rows. A source table can support
several tasks. Row units vary across sources.

- [Table catalog](tables.json) lists the exact files, source table IDs, formats,
  row counts, checksums, and reuse terms.
- [ORDerly, OpenExp and USPTO Yield](tables-zstd/) use compressed CSV files
  with a DuckDB loading script. The other 99 tables retain their original
  CSV, gzip, Parquet or JSON Lines formats under `tables/`.
- [Source terms and attributions](notices/kdd/coverage/publication_licenses/CURRENT-DATA-TERMS.txt)
  apply by table. Paths under `tables/` are grouped by license partition.
  In particular, the phosphatase table is for noncommercial educational use.
- Files that index larger native assets, such as the Transition1x configuration
  data, refer to their upstream sources or the full SynRXN DuckDB snapshot.

Run `sha256sum -c SHA256SUMS` from this directory. In DuckDB, use
`read_csv` for CSV files, `read_parquet` for Parquet, and
`read_json_objects` with `format='newline_delimited'` for JSON Lines.
For CSV, set `header=true, all_varchar=true` to preserve identifiers
and source number spellings. DuckDB reads the compressed files directly.
