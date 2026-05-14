.. _data-records:

Data Records
============

This page summarizes the curated SynRXN records by benchmark family. It is
designed as a quick inventory for selecting datasets, checking target columns,
and finding the source citation and reuse license for each benchmark table.
Dataset-level source and literature citations are collected on the
:doc:`Reference <reference>` page.

.. note::

   The ``License`` column reports the license of the table as distributed in the
   SynRXN release. Upstream publications and external source repositories should
   still be cited through the ``Source`` column. Users who redistribute modified
   versions should also check the upstream source terms.

.. raw:: html

   <div class="synrxn-icon-grid">
     <div class="synrxn-icon-card">
       <span class="synrxn-icon"><i class="fa-solid fa-layer-group" aria-hidden="true"></i></span>
       <strong>Task families</strong>
       <p>Reaction rebalancing, atom mapping, classification, property prediction, and synthesis.</p>
     </div>
     <div class="synrxn-icon-card">
       <span class="synrxn-icon"><i class="fa-solid fa-table-list" aria-hidden="true"></i></span>
       <strong>Curated tables</strong>
       <p>Named benchmark records distributed as compressed CSV files.</p>
     </div>
     <div class="synrxn-icon-card">
       <span class="synrxn-icon"><i class="fa-solid fa-fingerprint" aria-hidden="true"></i></span>
       <strong>Citable release</strong>
       <p>Use the Scientific Data paper and archived data release for reproducible reporting.</p>
     </div>
   </div>


Reaction Rebalancing
--------------------

**Data folder:** ``rbl``

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
     - License
   * - ``complex``
     - 1,748
     - Complex transformations and skeletal rearrangements.
     - ``r_id``, input ``rxn``, balanced ``ground_truth``
     - :cite:p:`jaworski2019automatic,lin2022atom`
     - ``CC BY 4.0``
   * - ``mbs``
     - 491
     - Missing species on both sides of the reaction.
     - ``r_id``, input ``rxn``, balanced ``ground_truth``
     - :cite:p:`liu2017retrosynthetic`
     - ``CC BY 4.0``
   * - ``mnc``
     - 33,147
     - Missing non-carbon species.
     - ``r_id``, input ``rxn``, balanced ``ground_truth``
     - :cite:p:`liu2017retrosynthetic`
     - ``CC BY 4.0``
   * - ``mos``
     - 12,781
     - Missing species on one side of the reaction.
     - ``r_id``, input ``rxn``, balanced ``ground_truth``
     - :cite:p:`liu2017retrosynthetic`
     - ``CC BY 4.0``


Atom-to-Atom Mapping
--------------------

**Data folder:** ``aam``

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
     - License
   * - ``ecoli``
     - 273
     - Biochemical mapping benchmark from E. coli reactions.
     - ``r_id``, ``ground_truth``, mapper outputs, ``rxn``, ``original_id``
     - :cite:p:`beier2026computing`
     - ``CC BY 4.0``
   * - ``enzyme_map``
     - 47,974
     - Enzymatic atom-mapping benchmark from the EnzymeMap reaction collection.
     - ``r_id``, ``ground_truth``, ``rxn``, ``original_id``
     - :cite:p:`heid2023enzymemap`
     - ``CC BY 4.0``
   * - ``golden``
     - 1,758
     - Curated synthetic chemistry mapping benchmark.
     - ``r_id``, ``ground_truth``, mapper outputs, ``rxn``, ``original_id``
     - :cite:p:`chen2024precise,lin2022atom`
     - ``CC BY 4.0``
   * - ``natcomm``
     - 491
     - Reference mapping set from literature-derived reactions.
     - ``r_id``, ``ground_truth``, mapper outputs, ``rxn``, ``original_id``
     - :cite:p:`jaworski2019automatic`
     - ``CC BY 4.0``
   * - ``recon3d``
     - 382
     - Biochemical mapping benchmark from Recon3D reactions.
     - ``r_id``, ``ground_truth``, mapper outputs, ``rxn``, ``original_id``
     - :cite:p:`litsa2018machine`
     - ``CC BY 4.0``
   * - ``uspto_3k``
     - 3,000
     - USPTO-derived synthetic chemistry mapping benchmark.
     - ``r_id``, ``ground_truth``, mapper outputs, ``rxn``, ``original_id``
     - :cite:p:`chen2024precise`
     - ``CC BY 4.0``


Reaction Classification
-----------------------

**Data folder:** ``classification``

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
     - License
   * - ``ecreact``
     - 185,734
     - Enzyme Commission hierarchy classification.
     - ``r_id``, ``rxn``, ``ec1``, ``ec2``, ``ec3``, ``split``, ``orig_index``
     - :cite:p:`zeng2025claire`
     - ``CC BY 4.0``
   * - ``schneider_b``
     - 50,000
     - Balanced Schneider reaction-class benchmark.
     - ``r_id``, ``rxn``, ``label``, ``split``
     - :cite:p:`schneider2015development,vannguyen2025syncat`
     - ``CC BY 4.0``
   * - ``schneider_u``
     - 50,000
     - Unbalanced/product-focused Schneider variant.
     - ``r_id``, ``rxn``, ``label``, ``split``
     - :cite:p:`schneider2015development`
     - ``CC BY 4.0``
   * - ``syntemp``
     - 43,441
     - Hierarchical SynTemp template labels.
     - ``orig_index``, ``r_id``, ``rxn``, ``label_0``, ``label_1``, ``label_2``
     - :cite:p:`phan2025syntemp,liu2017retrosynthetic`
     - ``CC BY 4.0``
   * - ``tpl_b``
     - 445,115
     - Balanced template-label benchmark.
     - ``r_id``, ``rxn``, ``label``, ``split``
     - :cite:p:`vannguyen2025syncat,schwaller2021mapping`
     - ``CC BY 4.0``
   * - ``tpl_u``
     - 445,115
     - Unbalanced/product-focused template-label benchmark.
     - ``r_id``, ``rxn``, ``label``, ``split``
     - :cite:p:`schwaller2021mapping`
     - ``CC BY 4.0``
   * - ``uspto_50k_b``
     - 50,016
     - Balanced USPTO-50K class-label benchmark.
     - ``r_id``, ``rxn``, ``label``, ``split``
     - :cite:p:`vannguyen2025syncat,liu2017retrosynthetic`
     - ``CC BY 4.0``
   * - ``uspto_50k_u``
     - 50,016
     - Unbalanced/product-focused USPTO-50K variant.
     - ``r_id``, ``rxn``, ``label``, ``split``
     - :cite:p:`liu2017retrosynthetic`
     - ``CC BY 4.0``


Reaction Property Prediction
----------------------------

**Data folder:** ``property``

Property datasets evaluate continuous or probabilistic reaction attributes such
as activation energy, enthalpy, rate, conversion, or reaction free energies. See
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
     - License
   * - ``b97xd3``
     - 16,365
     - Quantum-chemistry property prediction.
     - ``r_id``, ``aam``, ``ea``, ``dh``
     - :cite:p:`grambow2020scientificdata,grambow2020zenodo`
     - ``CC BY 4.0``
   * - ``cycloadd``
     - 5,269
     - Cycloaddition free-energy prediction.
     - ``r_id``, ``aam``, ``G_act``, ``G_r``, ``split``
     - :cite:p:`heid2023benchmark_chemprop`
     - ``CC BY 4.0``
   * - ``e2``
     - 1,264
     - E2 activation-energy prediction.
     - ``r_id``, ``aam``, ``ea``, ``split``
     - :cite:p:`heid2023benchmark_chemprop`
     - ``CC BY 4.0``
   * - ``e2sn2``
     - 3,625
     - E2/SN2 activation-energy prediction.
     - ``r_id``, ``aam``, ``ea``
     - :cite:p:`heid2022machine,vonrudorff2020thousands`
     - ``CC BY 4.0``
   * - ``lograte``
     - 778
     - Log-rate prediction.
     - ``r_id``, ``aam``, ``lograte``
     - :cite:p:`heid2022machine,bhoorasingh2017automated`
     - ``CC BY 4.0``
   * - ``phosphatase``
     - 33,354
     - Phosphatase conversion prediction.
     - ``r_id``, ``aam``, ``Conversion``, ``onehot``
     - :cite:p:`heid2022machine,huang2015panoramic`
     - ``CC BY 4.0``
   * - ``rad6re``
     - 31,923
     - Reaction enthalpy prediction.
     - ``r_id``, ``aam``, ``dh``
     - :cite:p:`heid2022machine,stocker2020machine`
     - ``CC BY 4.0``
   * - ``rdb7``
     - 23,852
     - Activation-energy prediction with predefined splits.
     - ``r_id``, ``aam``, ``ea``, ``split``
     - :cite:p:`heid2023benchmark_chemprop`
     - ``CC BY 4.0``
   * - ``rgd1``
     - 353,984
     - Large-scale activation-energy prediction.
     - ``r_id``, ``aam``, ``ea``, ``split``
     - :cite:p:`heid2023benchmark_chemprop`
     - ``CC BY 4.0``
   * - ``sn2``
     - 2,361
     - SN2 activation-energy prediction.
     - ``r_id``, ``aam``, ``ea``, ``split``
     - :cite:p:`heid2023benchmark_chemprop`
     - ``CC BY 4.0``
   * - ``snar``
     - 503
     - SNAr activation-energy prediction.
     - ``r_id``, ``rxn``, ``ea``
     - :cite:p:`jorner2021machine`
     - ``CC BY 3.0``


Synthesis Prediction
--------------------

**Data folder:** ``synthesis``

Synthesis records support single-step forward synthesis, retrosynthesis,
reagent/catalyst inference, and reaction-center related workflows. The
Scientific Data synthesis table explicitly lists USPTO-50K, USPTO-MIT, and
USPTO-500 as the SynRXN synthesis-prediction corpora; the Diels--Alder record is
documented here as an additional class-specific synthesis dataset. See
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
     - License
   * - ``da``
     - 11,011
     - Diels--Alder reaction benchmark.
     - ``r_id``, ``code``, ``reaction_original``, ``reaction``, ``rsmi``
     - :cite:p:`lam2024every`
     - ``CC BY 3.0``
   * - ``uspto_500``
     - 143,535
     - Reagent and catalyst inference benchmark.
     - ``r_id``, ``rxn``, ``reagent``, ``split``
     - :cite:p:`lu2022unified`
     - ``CC BY 4.0``
   * - ``uspto_50k``
     - 50,016
     - USPTO-50K synthesis/retrosynthesis benchmark.
     - ``r_id``, ``aam``, ``split``, ``source``
     - :cite:p:`chen2024precise,liu2017retrosynthetic`
     - ``CC BY 4.0``
   * - ``uspto_mit``
     - 479,035
     - Large-scale USPTO-MIT synthesis prediction benchmark.
     - ``r_id``, ``aam``, ``split``, ``rc``
     - :cite:p:`jin2017predicting_outcomes`
     - ``CC BY 4.0``


Choosing a dataset
------------------

Use the task family first, then choose the dataset based on target type, scale,
and whether the benchmark already includes a published split. For example,
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

For a complete reproducibility record, pair the selected dataset with the source
mode and version described in :doc:`Data Concept <data_concept>`.