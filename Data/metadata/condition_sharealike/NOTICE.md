# SynRXN ShareAlike Partition

This directory is a **physically separate release partition**. Every artifact
here is an adaptation of a Creative Commons Attribution-ShareAlike 4.0
International source and must be redistributed under the same license.

Do not merge these artifacts into the main partition, the main
`manifest.json`, or any archive whose license notice does not carry the
ShareAlike obligation. The partition ships its own license text (`LICENSE`),
its own checksums, and its own manifest (`manifest.json`).

| Artifact | Rows | Upstream source | Upstream license |
| --- | ---: | --- | --- |
| `ord_hte_condition.csv.gz` | 123,720 | Sixteen non-overlapping Open Reaction Database HTE cohorts | CC BY-SA 4.0 |
| `openexp.csv.gz` | 274,439 | OpenExp reaction-procedure pairs (`OpenExp.tar` OSF snapshot) | CC BY-SA 4.0 |

Total: 2 artifacts, 398,159 records.

## Attribution

`ord_hte_condition.csv.gz` is a modified aggregate of sixteen datasets from the
Open Reaction Database. The ORD data repository licenses datasets under
`data/` and their metadata under CC BY-SA 4.0. SynRXN excludes the Pfizer 39k,
5,760-record Suzuki-Miyaura, Ahneman, and AstraZeneca 750 cohorts because those
source experiments already occur in the yield collection. Preserve ORD and
contributor attribution.

`openexp.csv.gz` is a modified table from the authors' exact 274,439-record
`OpenExp.tar` OSF snapshot. Appendix A.1 of the ReactXT paper states that
OpenExp is released under CC BY-SA and requires the same license for
distribution and sharing. SynRXN retains the original procedure and action
sequence and adds a deterministic transformation-group split because the
released code's random 8:1:1 split has no fixed seed.

## Provenance

Source locks, upstream checksums, licensing evidence, and the build report are
recorded under `../sources/condition/`. Rebuild with
`script/build_condition_dataset.py`.

License: <https://creativecommons.org/licenses/by-sa/4.0/>
Full text: `LICENSE`
