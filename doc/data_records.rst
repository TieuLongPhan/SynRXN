:html_theme.sidebar_secondary.remove:

.. _data-records:

Data records
============

Look up source tables by name, column, or license. Open a record to see its
schema, citations, and loading code where available. Browse :doc:`families` for
the complete set of tables and their task assignments.

.. raw:: html
   :file: _templates/catalog-browser.html

.. raw:: html

   <script src="_static/data-records.js"></script>

Reuse terms
-----------

The terms below follow the current per-table release manifest, including
source and curation obligations. In a compound expression, ``AND`` retains all
listed obligations. Noncommercial, ShareAlike, database and GNU conditions are
preserved; a historical blanket curation label does not replace upstream terms.
The Diels--Alder table is held pending clarification of redistribution scope.

The manifest and supporting notices are under
``Data/metadata/reconstruction/coverage/``. Original-source citations are
collected on the :doc:`reference` page. Cite those sources together with the
SynRXN version used.

Reaction Rebalancing
--------------------

**Export group:** ``rbl``

Reaction rebalancing datasets evaluate whether a method can restore missing
species and recover a balanced reaction from an incomplete equation. The MNC,
MOS, and MBS subsets are derived from USPTO-50K-style reaction records, whereas
the Complex subset is based on manually validated complex-reaction mapping
collections. See :doc:`Reference <reference>` for complete bibliographic
records.

.. list-table::
   :header-rows: 1
   :widths: 18 10 25 25 12 10
   :class: synrxn-table

   * - Dataset
     - Records
     - Benchmark role
     - Principal columns
     - Source
     - Effective terms
   * - ``complex``
     - 1,748
     - Complex transformations and skeletal rearrangements.
     - ``r_id``, input ``rxn``, balanced ``ground_truth``
     - :cite:p:`jaworski2019automatic,lin2022atom`
     - ``CC-BY-4.0`` AND ``MIT`` AND ``LGPL-2.1-only``
   * - ``mbs``
     - 491
     - Missing species on both sides of the reaction.
     - ``r_id``, input ``rxn``, balanced ``ground_truth``
     - :cite:p:`liu2017retrosynthetic`
     - ``CC-BY-4.0`` AND ``CC0-1.0``
   * - ``mnc``
     - 33,147
     - Missing non-carbon species.
     - ``r_id``, input ``rxn``, balanced ``ground_truth``
     - :cite:p:`liu2017retrosynthetic`
     - ``CC-BY-4.0`` AND ``CC0-1.0``
   * - ``mos``
     - 12,781
     - Missing species on one side of the reaction.
     - ``r_id``, input ``rxn``, balanced ``ground_truth``
     - :cite:p:`liu2017retrosynthetic`
     - ``CC-BY-4.0`` AND ``CC0-1.0``


Atom-to-Atom Mapping
--------------------

**Export group:** ``aam``

The table below lists the ten mapping tables in the main snapshot. See
:ref:`atom-mapping-preparation` for a local preparation example.

Atom-to-atom mapping records compare predicted mappings with curated or
consensus reference mappings across synthetic and biochemical reaction domains.
See :doc:`Reference <reference>` for source publications and mapper references.

.. list-table::
   :header-rows: 1
   :widths: 18 10 25 25 12 10
   :class: synrxn-table

   * - Dataset
     - Records
     - Benchmark role
     - Principal columns
     - Source
     - Effective terms
   * - ``brenda``
     - 3,138
     - IUBMB-marked, exactly balanced biochemical reactions with SynKit-generated mappings and bijection QC.
     - ``r_id``, mapped ``ground_truth``, balanced ``rxn``, ``original_id``, ``pubmed_ids``
     - :cite:p:`hauenstein2026brenda`
     - ``CC-BY-4.0``
   * - ``ecoli``
     - 273
     - Biochemical mapping benchmark from E. coli reactions.
     - ``r_id``, ``ground_truth``, mapper outputs, ``rxn``, ``original_id``
     - :cite:p:`beier2026computing`
     - ``MIT``
   * - ``ehreact``
     - 939
     - Manually corrected mappings from nine EHreact manuscript systems.
     - ``r_id``, ``ground_truth``, ``rxn``, ``system``, ``mapping_curated``, ``validated``
     - :cite:p:`heid2021ehreact`
     - ``BSD-3-Clause``
   * - ``enzyme_map``
     - 47,974
     - Enzymatic atom-mapping benchmark from the EnzymeMap reaction collection.
     - ``r_id``, ``ground_truth``, ``rxn``, ``original_id``
     - :cite:p:`heid2023enzymemap`
     - ``CC-BY-4.0``
   * - ``flower_e``
     - 179,345
     - Unique standardized elementary steps from the corrected May-2026 FlowER test split.
     - ``r_id``, mapped ``ground_truth``, unmapped ``rxn``, ``original_id``
     - :cite:p:`joung2025electron,fong2026flower_data`
     - ``MIT``
   * - ``flower_c``
     - 38,745
     - Unique standardized multi-step root-to-terminal transformations reconstructed from FlowER elementary pathways.
     - ``r_id``, mapped ``ground_truth``, unmapped ``rxn``, ``original_id``, ``n_steps``
     - :cite:p:`joung2025electron,fong2026flower_data`
     - ``MIT``
   * - ``golden``
     - 1,758
     - Curated synthetic chemistry mapping benchmark.
     - ``r_id``, ``ground_truth``, mapper outputs, ``rxn``, ``original_id``
     - :cite:p:`chen2024precise,lin2022atom`
     - ``GPL-3.0-only`` AND ``LGPL-2.1-only`` AND ``MIT``
   * - ``natcomm``
     - 491
     - Reference mapping set from literature-derived reactions.
     - ``r_id``, ``ground_truth``, mapper outputs, ``rxn``, ``original_id``
     - :cite:p:`jaworski2019automatic`
     - ``CC-BY-4.0`` AND ``GPL-3.0-only`` AND ``MIT``
   * - ``recon3d``
     - 382
     - Biochemical mapping benchmark from Recon3D reactions.
     - ``r_id``, ``ground_truth``, mapper outputs, ``rxn``, ``original_id``
     - :cite:p:`litsa2018machine`
     - ``CC-BY-NC-4.0`` AND ``MIT``
   * - ``uspto_3k``
     - 3,000
     - USPTO-derived synthetic chemistry mapping benchmark.
     - ``r_id``, ``ground_truth``, mapper outputs, ``rxn``, ``original_id``
     - :cite:p:`chen2024precise`
     - ``GPL-3.0-only`` AND ``CC-BY-SA-4.0`` AND ``MIT``

The BRENDA subset requires the upstream ``isIubmbReaction`` marker, complete
participant structures, exact atom balance including hydrogen, and formal-charge
balance. Its atom mappings are generated computationally with SynKit and pass a
complete bijective explicit-atom map check. BRENDA does not expose a separate
row-level experimental-validation field for these records. These constant
selection and QC facts are enforced by the builder rather than repeated in
every row. A JSON source report can be requested as a downstream build
artifact.

The two FlowER tables use only ``flower_new_dataset/test.txt`` from the
corrected May-2026 public archive. In the source, the suffix after ``|`` groups
elementary state transitions belonging to an overall reaction. The elementary
table excludes exact no-change termination steps and exact duplicate mappings.
For the composite table, all records sharing a numeric suffix are reconstructed
as a directed mapped-state graph and every root is joined to each reachable
terminal indicated by a no-change marker (or a graph leaf when no marker is
present). Only composites spanning at least two elementary transitions are
retained; ``RC``, ``PC``, ``PM``, and ``RS`` groups remain elementary-only.
Both tables require complete bijective atom maps. The multi-gigabyte training
and validation splits and the request-only NameRxn-derived OOD sets are not
distributed. SynKit identifies the reaction center and converts non-reacting
explicit hydrogens in ``ground_truth`` to implicit hydrogen counts while
preserving reacting mapped hydrogens. The ``rxn`` field is regenerated as
canonical unmapped SMILES with removable hydrogens made implicit. Standalone
reacting hydrogen species such as hydride or molecular hydrogen remain explicit
because SMILES has no implicit representation for them. Standardization
collapses equivalent explicit-hydrogen encodings before final deduplication.


Reaction Classification
-----------------------

**Export group:** ``classification``

Classification datasets evaluate reaction class, template, or enzyme-label
prediction. Some corpora are provided in balanced/full and unbalanced/product
focused variants. See :doc:`Reference <reference>` for label-source and
benchmark citations.

.. list-table::
   :header-rows: 1
   :widths: 18 10 25 25 12 10
   :class: synrxn-table

   * - Dataset
     - Records
     - Benchmark role
     - Principal columns
     - Source
     - Effective terms
   * - ``ecreact``
     - 185,734
     - Enzyme Commission hierarchy classification.
     - ``r_id``, ``rxn``, ``ec1``, ``ec2``, ``ec3``, ``split``, ``orig_index``
     - :cite:p:`zeng2025claire`
     - ``CC-BY-4.0`` AND ``MIT`` AND ``ODbL-1.0``
   * - ``ehreact_activity``
     - 1,943
     - Binary reaction--assay activity prediction across nine systems.
     - ``r_id``, ``reaction_r_id``, ``ground_truth``, ``system``, ``assay``, ``activity``
     - :cite:p:`heid2021ehreact`
     - ``BSD-3-Clause``
   * - ``schneider_b``
     - 50,000
     - Balanced Schneider reaction-class benchmark.
     - ``r_id``, ``rxn``, ``label``, ``split``
     - :cite:p:`schneider2015development,vannguyen2025syncat`
     - ``CC-BY-4.0`` AND ``MIT`` AND ``CC-BY-NC-4.0``
   * - ``schneider_u``
     - 50,000
     - Unbalanced/product-focused Schneider variant.
     - ``r_id``, ``rxn``, ``label``, ``split``
     - :cite:p:`schneider2015development`
     - ``CC-BY-4.0`` AND ``MIT`` AND ``CC-BY-NC-4.0``
   * - ``syntemp``
     - 43,441
     - Hierarchical SynTemp template labels.
     - ``orig_index``, ``r_id``, ``rxn``, ``label_0``, ``label_1``, ``label_2``
     - :cite:p:`phan2025syntemp,liu2017retrosynthetic`
     - ``CC-BY-4.0`` AND ``CC0-1.0``
   * - ``tpl_b``
     - 445,115
     - Balanced template-label benchmark.
     - ``r_id``, ``rxn``, ``label``, ``split``
     - :cite:p:`vannguyen2025syncat,schwaller2021mapping`
     - ``CC-BY-4.0`` AND ``MIT``
   * - ``tpl_u``
     - 445,115
     - Unbalanced/product-focused template-label benchmark.
     - ``r_id``, ``rxn``, ``label``, ``split``
     - :cite:p:`schwaller2021mapping`
     - ``CC-BY-4.0`` AND ``MIT``
   * - ``uspto_50k_b``
     - 50,016
     - Balanced USPTO-50K class-label benchmark.
     - ``r_id``, ``rxn``, ``label``, ``split``
     - :cite:p:`vannguyen2025syncat,liu2017retrosynthetic`
     - ``CC-BY-4.0`` AND ``MIT`` AND ``CC-BY-NC-4.0``
   * - ``uspto_50k_u``
     - 50,016
     - Unbalanced/product-focused USPTO-50K variant.
     - ``r_id``, ``rxn``, ``label``, ``split``
     - :cite:p:`liu2017retrosynthetic`
     - ``CC-BY-4.0`` AND ``MIT`` AND ``CC-BY-NC-4.0``


Reaction Property Prediction
----------------------------

**Export group:** ``property``

Property datasets contain activation energies, enthalpies, rates, reaction free
energies and source-specific assay responses. Units and physical meaning are
defined by each source; the exported column name alone is insufficient. See
:doc:`Reference <reference>` for upstream corpus citations.

.. list-table::
   :header-rows: 1
   :widths: 18 10 25 25 12 10
   :class: synrxn-table

   * - Dataset
     - Records
     - Benchmark role
     - Principal columns
     - Source
     - Effective terms
   * - ``b97xd3``
     - 16,365
     - Quantum-chemistry property prediction.
     - ``r_id``, ``aam``, ``ea``, ``dh``
     - :cite:p:`grambow2020scientificdata,grambow2020zenodo`
     - ``CC-BY-4.0``
   * - ``cycloadd``
     - 5,269
     - Cycloaddition free-energy prediction.
     - ``r_id``, ``aam``, ``G_act``, ``G_r``, ``split``
     - :cite:p:`heid2023benchmark_chemprop`
     - ``CC-BY-4.0``
   * - ``e2``
     - 1,264
     - E2 activation-energy prediction.
     - ``r_id``, ``aam``, ``ea``, ``split``
     - :cite:p:`heid2023benchmark_chemprop`
     - ``CC-BY-4.0``
   * - ``e2sn2``
     - 3,625
     - E2/SN2 activation-energy prediction.
     - ``r_id``, ``aam``, ``ea``
     - :cite:p:`heid2022machine,vonrudorff2020thousands`
     - ``CC-BY-4.0``
   * - ``lograte``
     - 778
     - Log-rate prediction.
     - ``r_id``, ``aam``, ``lograte``
     - :cite:p:`heid2022machine,bhoorasingh2017automated`
     - ``CC-BY-NC-4.0``
   * - ``phosphatase``
     - 33,354
     - Source phosphatase assay-response prediction; not established percentage conversion.
     - ``r_id``, ``aam``, ``Conversion``, ``onehot``
     - :cite:p:`heid2022machine,huang2015panoramic`
     - ``CC-BY-4.0`` AND ``LicenseRef-PNAS-NC-Educational``
   * - ``rad6re``
     - 31,923
     - Reaction enthalpy prediction.
     - ``r_id``, ``aam``, ``dh``
     - :cite:p:`heid2022machine,stocker2020machine`
     - ``CC-BY-4.0``
   * - ``rdb7``
     - 23,852
     - Activation-energy prediction with predefined splits.
     - ``r_id``, ``aam``, ``ea``, ``split``
     - :cite:p:`heid2023benchmark_chemprop`
     - ``CC-BY-4.0``
   * - ``rgd1``
     - 353,984
     - Large-scale activation-energy prediction.
     - ``r_id``, ``aam``, ``ea``, ``split``
     - :cite:p:`heid2023benchmark_chemprop`
     - ``CC-BY-4.0``
   * - ``sn2``
     - 2,361
     - SN2 activation-energy prediction.
     - ``r_id``, ``aam``, ``ea``, ``split``
     - :cite:p:`heid2023benchmark_chemprop`
     - ``CC-BY-4.0``
   * - ``snar``
     - 503
     - SNAr activation-energy prediction.
     - ``r_id``, ``rxn``, ``ea``
     - :cite:p:`jorner2021machine`
     - ``CC-BY-3.0``


Synthesis Prediction
--------------------

**Export group:** ``synthesis``

Synthesis records support single-step product prediction, retrosynthesis,
reagent inference and reaction-center prediction. Multistep synthesis planning
is outside the research catalogue. The Diels--Alder entry provides metadata
only while redistribution rights remain unresolved. See
:doc:`Reference <reference>` for USPTO-derived benchmark citations.

.. list-table::
   :header-rows: 1
   :widths: 18 10 25 25 12 10
   :class: synrxn-table

   * - Dataset
     - Records
     - Benchmark role
     - Principal columns
     - Source
     - Effective terms
   * - ``da``
     - 11,011
     - Diels--Alder reaction benchmark.
     - ``r_id``, ``code``, ``reaction_original``, ``reaction``, ``rsmi``
     - :cite:p:`lam2024every`
     - Held: redistribution grant unresolved.
   * - ``uspto_500``
     - 143,535
     - Reagent and catalyst inference benchmark.
     - ``r_id``, ``rxn``, ``reagent``, ``split``
     - :cite:p:`lu2022unified`
     - ``CC-BY-4.0`` AND ``CC0-1.0``
   * - ``uspto_50k``
     - 50,016
     - USPTO-50K synthesis/retrosynthesis benchmark.
     - ``r_id``, ``aam``, ``split``, ``source``
     - :cite:p:`chen2024precise,liu2017retrosynthetic`
     - ``CC-BY-4.0`` AND ``MIT`` AND ``CC-BY-NC-4.0``
   * - ``uspto_mit``
     - 479,035
     - Large-scale USPTO-MIT synthesis prediction benchmark.
     - ``r_id``, ``aam``, ``split``, ``rc``
     - :cite:p:`jin2017predicting_outcomes`
     - ``CC-BY-4.0`` AND ``CC0-1.0``


Reaction Condition Prediction
-----------------------------

**Export group:** ``condition``

The development condition table normalizes ORDerly-condition v4 into a
37-column schema. It preserves the published train/test assignment and generic
agent slots while exposing solvents, temperature in degrees Celsius, reaction
time in hours, procedure text, optional reported yield, and leakage-control
groups. Missing values remain blank; numerical anomalies are retained with
explicit quality flags rather than clipped or imputed.

.. list-table::
   :header-rows: 1
   :widths: 18 10 25 25 12 10
   :class: synrxn-table

   * - Dataset
     - Records
     - Benchmark role
     - Principal columns
     - Source
     - Effective terms
   * - ``orderly_condition``
     - 691,105
     - Broad patent-derived reaction-condition prediction with published partitions.
     - ``agents``, ``solvents``, ``temperature_c``, ``time_hours``, grouping keys
     - :cite:p:`wigh2024orderly`
     - ``CC-BY-4.0``
   * - ``ord_hte_condition``
     - 123,720
     - Cross-family HTE condition-set prediction and condition-aware ranking.
     - Role-aware reagents/catalysts/ligands/bases/additives, solvents, operating conditions, grouping keys
     - :cite:p:`kearnes2021ord`
     - ``CC-BY-SA-4.0``
   * - ``openexp``
     - 274,439
     - Reaction-to-experimental-action and procedure prediction.
     - ``action_sequence``, ``procedure_details``, reactants/products, catalysts, solvents, grouping keys
     - :cite:p:`liu2024reactxt`
     - ``CC-BY-SA-4.0``
   * - ``uspto_500``
     - 143,535
     - Patent-derived reagent and catalyst inference with published partitions.
     - ``r_id``, ``rxn``, ``reagent``, ``split``
     - :cite:p:`lu2022unified`
     - ``CC-BY-4.0`` AND ``CC0-1.0``

The two pinned source partitions contain 691,142 rows. SynRXN quarantines 37
records whose required structures cannot be normalized. Among retained rows,
28 temperatures fall outside the documented pilot plausibility range of
-196--500 degrees Celsius, and 25,716 optional reported yields fall outside
0--100; these values remain available with flags.
The condition and synthesis IDs for ``uspto_500`` resolve to one table. Its SHA-256 is
``fad362c125f8fc6195d49748bbd9f381ca4c9d3f03935d82f457b350ab46bef1``.


Electron-Pushing Mechanisms
---------------------------

**Export group:** ``mechanism``

SynEPD polar records contain supplied electron-pushing mechanisms, replay
status and verification levels. Supplied arrow order does not establish physical
timing or a uniquely proven mechanism. Endpoint agreement and recovery of
mechanism events are evaluated separately.

.. list-table::
   :header-rows: 1
   :widths: 18 10 25 25 12 10
   :class: synrxn-table

   * - Dataset
     - Records
     - Benchmark role
     - Principal columns
     - Source
     - Effective terms
   * - ``synepd_polar``
     - 1,926
     - Polar mechanism prediction; 1,924 primary-tier records.
     - ``reaction_smiles``, ``electron_path_json``, ``reference_replay_status``, ``verification_level``
     - :cite:p:`synepd2026`
     - ``CC-BY-4.0``


Reaction Selectivity
--------------------

**Export group:** ``selectivity``

Only experimentally observed labels with explicit redistribution terms are
included. Site indices always refer to the uncanonicalized upstream
``substrate_smiles``; ``canonical_substrate_smiles`` is provided only for
identity grouping and must not be used to reinterpret those indices.

.. list-table::
   :header-rows: 1
   :widths: 22 10 24 24 12 8
   :class: synrxn-table

   * - Dataset
     - Records
     - Benchmark role
     - Principal columns
     - Source
     - Effective terms
   * - ``regiosqm_eas``
     - 535
     - Predict one or more experimentally observed EAS sites.
     - ``substrate_smiles``, ``candidate_site_indices``, ``observed_site_indices``
     - :cite:p:`kromann2018regiosqm`
     - ``MIT``
   * - ``hitea_chiral_resolution``
     - 50
     - Predict signed product enantiomeric excess for chiral salt resolutions.
     - ``canonical_substrate_smiles``, conditions, ``selectivity_value``
     - :cite:p:`kingsmith2024reactome`
     - ``MIT``

The RegioSQM table contains 604 observed aromatic C--H sites: 471 substrates
have one observed site, 60 have two, three have three, and one has four. The
HiTEA source contains 129 chiral-resolution rows, but 79 have zero in the
reported product-yield field and are excluded rather than treated as measured
zero ee. The retained 50 rows include three zero-ee results with positive
reported product yield. HiTEA's
signed convention does not identify the favored enantiomer, so values are not
converted to absolute ee and ``stereochemistry_complete`` remains false.
Exact inputs, checksums, method evidence, and license scope are recorded under
``Data/metadata/sources/selectivity/``.


Reaction Yield Prediction
-------------------------

**Export group:** ``yield``

These tables share a 33-column schema with ``yield_pct`` as the exported target.
The sources measure different endpoints: in HiTEA, the underlying field is
product UV-area percentage, rather than independently calibrated isolated
yield. Analyses must preserve those endpoint distinctions. Raw reaction strings,
conditions, source identifiers, reuse terms and grouping keys remain available;
blank optional fields indicate that no compatible source value was supplied.

.. list-table::
   :header-rows: 1
   :widths: 18 10 25 25 12 10
   :class: synrxn-table

   * - Dataset
     - Records
     - Benchmark role
     - Principal columns
     - Source
     - Effective terms
   * - ``buchwald_hartwig_ood``
     - 27,438
     - Multi-campaign Buchwald--Hartwig OOD yield prediction.
     - ``sample_id``, ``reaction_smiles``, ``yield_pct``, ``source``, grouping keys
     - :cite:p:`neves2026bh`
     - ``CC-BY-4.0``
   * - ``hitea``
     - 39,546
     - HTE product UV-area response prediction with reaction-class and analytical context.
     - ``sample_id``, ``reaction_smiles``, ``yield_pct``, conditions, ``reaction_class``
     - :cite:p:`kingsmith2024reactome`
     - ``MIT``
   * - ``uspto_yield``
     - 526,468
     - Large-scale patent reaction-yield prediction with published partitions.
     - ``sample_id``, ``reaction_smiles``, ``yield_pct``, ``upstream_split``
     - :cite:p:`minidis2021uspto_yield,lowe2017uspto`
     - ``MIT``

The BH input contains 27,497 source rows. SynRXN quarantines 59 measurements
above 100 percent and distributes the remaining 27,438. HiTEA non-numeric
product-selectivity labels remain in ``source_metadata_json`` and are separate
from the numeric product-response target. Source versions, checksums, and the build audit are
recorded under ``Data/metadata/sources/yield/``.


Choosing a dataset
------------------

Choose a scientific task in the research catalogue, then inspect the source
endpoint, eligible rows, measurement conditions and dependence groups. For example,
``schneider_b`` is a compact classification benchmark with a standard reaction
class label, ``b97xd3`` is a quantum-chemistry property dataset with activation
energy and enthalpy targets, and ``uspto_50k`` is a common synthesis or
retrosynthesis benchmark.

.. list-table:: Practical selection guide
   :header-rows: 1
   :widths: 24 40 36
   :class: synrxn-table

   * - Goal
     - Start with
     - Check before training
   * - Compare atom mappers
     - ``aam/golden``, ``aam/uspto_3k``, or biochemical AAM records.
     - Whether ``ground_truth`` and mapper-output columns match your metric.
   * - Train reaction classifiers
     - ``classification/schneider_b`` or ``classification/uspto_50k_b``.
     - Label granularity and whether balanced or unbalanced variants are needed.
   * - Predict reaction energies
     - ``property/b97xd3``, ``property/rgd1``, or mechanism-specific subsets.
     - Unit conventions, target column name, and split availability.
   * - Recover missing species
     - ``rbl/mnc``, ``rbl/mos``, ``rbl/mbs``, or ``rbl/complex``.
     - Whether the missing-species setting matches your method.
   * - Benchmark synthesis tasks
     - ``synthesis/uspto_50k``, ``synthesis/uspto_mit``, or ``synthesis/uspto_500``.
     - Whether the task is product prediction, retrosynthesis, reagent inference, or reaction-center prediction.
   * - Predict reaction yield
     - ``yield/buchwald_hartwig_ood`` for OOD evaluation, ``yield/hitea`` for HTE chemistry, or ``yield/uspto_yield`` for scale.
     - Yield measurement sources, duplicates, chemical-group leakage, zero-yield prevalence, and upstream partition policy.
   * - Predict reaction conditions
     - ``condition/orderly_condition`` for broad targets, ``condition_sharealike/ord_hte_condition`` for structured HTE contexts, ``condition_sharealike/openexp`` for procedures, or ``condition/uspto_500`` for reagent inference.
     - Target missingness, role semantics, ShareAlike boundary, numerical quality flags, chemical-group leakage, and upstream versus SynRXN split policy.
   * - Predict selectivity
     - ``selectivity/regiosqm_eas`` for regioselective sites or ``selectivity/hitea_chiral_resolution`` for signed product ee.
     - Raw atom-index basis, multiple observed sites, measurement method, sign convention, and the absence of a fixed split.

Record the selected table, source checksum, target definition and split membership
using the conventions in :doc:`data_concept` and :doc:`task_evaluation`.
