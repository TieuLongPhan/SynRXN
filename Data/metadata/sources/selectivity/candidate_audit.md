# Selectivity candidate audit

Reviewed 2026-08-22 under the policy requiring experimentally observed labels
and explicit redistribution terms.

| Candidate | Evidence | Decision |
| --- | --- | --- |
| HiTEA chiral salt resolutions | Wet-lab resolution screens; supplementary methods define chiral SFC-MS product ee; repository releases code and datasets under MIT | Include 50 rows with positive reported product yield; exclude 79 zero-reported-yield rows |
| Jensen `db-regioselectivity` | Database-only MIT repository explicitly identifies experimental EAS centers and row-level literature/patent references | Include all 535 substrates and 604 observed sites |
| ORD selectivity outcomes | CC BY-SA 4.0 permits redistribution, but selectivity semantics and independent overlap have not been audited | Defer to a separate ShareAlike candidate |
| Reisman regioselectivity data | Repository describes experimental dioxirane data, but exact file-level license scope and row provenance still require audit | Defer |
| Sigman-group tables | Experimental selectivity tables span repositories and supplementary files with different publication terms | Audit each source independently before ingestion |

Primary sources:

- https://doi.org/10.1038/s41557-023-01393-w
- https://github.com/emmaking-smith/HiTEA
- https://doi.org/10.1039/C7SC04156J
- https://github.com/jensengroup/db-regioselectivity
