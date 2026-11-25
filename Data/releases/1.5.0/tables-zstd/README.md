# SynRXN 1.5.0 compressed table downloads

These five Zstandard-compressed CSV files contain three prepared SynRXN
tables. The export preserves all prepared CSV records. ORDerly is divided
at record boundaries, with the original header repeated in each part.

| Table | Rows | Files | Terms |
| --- | ---: | --- | --- |
| ORDerly condition recovery | 691,105 | `orderly_condition-01.csv.zst` through `orderly_condition-03.csv.zst` | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| OpenExp | 274,439 | `openexp.csv.zst` | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |
| USPTO Yield | 526,468 | `uspto_yield.csv.zst` | [MIT](https://opensource.org/license/mit) |

Run `sha256sum -c SHA256SUMS` from this directory to verify the download.
Open DuckDB in the same directory and run `.read load.sql` to create
`orderly_condition`, `openexp` and `uspto_yield` views. DuckDB decompresses
the files while scanning them.

## Attribution

- ORDerly benchmarks of chemical reactions by Daniel Wigh, Joe Arrowsmith,
  Kobi Felton, Alexander Pomberger and Alexei A. Lapkin.
  [Figshare, version 4](https://doi.org/10.6084/m9.figshare.23298467.v4)
- OpenExp by Zhiyuan Liu and colleagues.
  [ReactXT article](https://doi.org/10.18653/v1/2024.findings-acl.318) · [OpenExp archive](https://osf.io/3dv4k/)
- USPTO yield curation deposited by Alexander Minidis, with source data
  derived from Daniel Lowe's US patent dataset.
  [Yield dataset](https://doi.org/10.6084/m9.figshare.14414039.v1) · [Original USPTO dataset](https://doi.org/10.6084/m9.figshare.5104873.v1)

The [source terms](SOURCE-TERMS.txt) and [source locks](sources/) document
the applicable licenses and pinned upstream revisions.
