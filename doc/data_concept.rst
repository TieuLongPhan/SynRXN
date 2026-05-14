.. _data-concept:

Data Concept
============

SynRXN treats reaction data as versioned benchmark assets. Curated tables are
stored as compressed CSV files and loaded through :class:`synrxn.data.DataLoader`.
Each task folder represents a benchmark family; each dataset file is a named
record within that family.

Core model
----------

.. raw:: html

   <div class="synrxn-icon-grid">
     <div class="synrxn-icon-card">
       <span class="synrxn-icon"><i class="fa-solid fa-layer-group" aria-hidden="true"></i></span>
       <strong>Task family</strong>
       <p>A benchmark problem type such as atom mapping, classification, property prediction, rebalancing, or synthesis.</p>
     </div>
     <div class="synrxn-icon-card">
       <span class="synrxn-icon"><i class="fa-solid fa-table" aria-hidden="true"></i></span>
       <strong>Dataset record</strong>
       <p>A named compressed CSV table with stable columns, source metadata, and task-specific supervision.</p>
     </div>
     <div class="synrxn-icon-card">
       <span class="synrxn-icon"><i class="fa-solid fa-code-commit" aria-hidden="true"></i></span>
       <strong>Source snapshot</strong>
       <p>A Zenodo release, GitHub tag, exact commit, or development branch used to retrieve the data.</p>
     </div>
     <div class="synrxn-icon-card">
       <span class="synrxn-icon"><i class="fa-solid fa-shuffle" aria-hidden="true"></i></span>
       <strong>Split policy</strong>
       <p>Published split columns are preserved; new repeated splits can be generated with deterministic seeds.</p>
     </div>
   </div>

The package separates concerns clearly:

- ``Data/`` contains curated benchmark tables grouped by task.
- ``synrxn.data`` resolves, downloads, caches, verifies, lists, and loads those
  tables.
- ``synrxn.split`` provides deterministic split helpers when a dataset does not
  already include the split needed for an experiment.
- Builder scripts under ``script/`` rebuild curated artifacts from upstream raw
  sources for developer workflows.

Storage layout
--------------

A typical source checkout or data archive follows this structure:

.. code-block:: text

   Data/
   ├── aam/
   │   ├── golden.csv.gz
   │   └── uspto_3k.csv.gz
   ├── classification/
   │   ├── schneider_b.csv.gz
   │   └── uspto_50k_u.csv.gz
   ├── property/
   │   ├── b97xd3.csv.gz
   │   └── rgd1.csv.gz
   ├── rbl/
   │   └── mnc.csv.gz
   └── synthesis/
       └── uspto_50k.csv.gz

``Data/classification/schneider_b.csv.gz`` is loaded as dataset
``"schneider_b"`` with ``task="classification"``.

Task families
-------------

.. list-table::
   :header-rows: 1
   :widths: 18 28 29 25
   :class: synrxn-table

   * - Task
     - Goal
     - Typical targets
     - Example datasets
   * - ``aam``
     - Evaluate atom-to-atom mapping quality.
     - reference ``ground_truth`` mappings and mapper outputs
     - ``golden``, ``enzyme_map``, ``uspto_3k``
   * - ``classification``
     - Predict reaction class, template, or enzyme hierarchy.
     - ``label``, ``label_0``, ``label_1``, ``label_2``, ``ec1``--``ec3``
     - ``schneider_b``, ``tpl_b``, ``ecreact``
   * - ``property``
     - Predict continuous or probabilistic reaction properties.
     - ``ea``, ``dh``, ``G_act``, ``G_r``, ``lograte``, ``Conversion``
     - ``b97xd3``, ``rgd1``, ``cycloadd``
   * - ``rbl``
     - Restore missing species and recover balanced reactions.
     - balanced ``ground_truth`` reaction strings
     - ``mnc``, ``mos``, ``mbs``, ``complex``
   * - ``synthesis``
     - Support forward synthesis, retrosynthesis, reagent inference, or reaction-center prediction.
     - ``aam``, ``rxn``, ``reagent``, ``rc``, ``source``
     - ``da``, ``uspto_50k``, ``uspto_mit``, ``uspto_500``

Naming conventions
------------------

Dataset names are short, lowercase identifiers. Some names include suffixes that
clarify the reaction representation:

- ``*_b`` denotes balanced or fuller reaction records where auxiliary species can
  be included.
- ``*_u`` denotes unbalanced or product-focused records, often useful for
  template and reaction-class tasks.
- task-specific names such as ``e2``, ``sn2``, ``rgd1``, or ``cycloadd`` keep the
  upstream benchmark identity visible.

Common columns
--------------

Column availability depends on the task family and upstream source. The most
common conventions are:

- ``r_id``: stable row identifier used by most curated datasets.
- ``rxn``: reaction SMILES without atom mapping unless otherwise documented.
- ``aam``: atom-mapped reaction SMILES.
- ``ground_truth``: reference answer, such as the correct atom mapping or the
  balanced reaction string.
- ``label``, ``label_0``, ``label_1``, ``label_2``: flat or hierarchical class
  labels.
- ``split``: published split assignment when available.
- numeric target columns such as ``ea``, ``dh``, ``G_act``, ``G_r``,
  ``lograte``, or ``Conversion``.
- source-tracking columns such as ``original_id``, ``orig_index``, ``source``, or
  ``code``.

Source resolution
-----------------

:class:`synrxn.data.DataLoader` lets the same user code target different data
sources.

.. list-table::
   :header-rows: 1
   :widths: 20 28 28 24
   :class: synrxn-table

   * - Source
     - What it resolves
     - Recommended use
     - Stability
   * - ``zenodo``
     - Archived data release assets.
     - Published experiments and benchmark reports.
     - Citable and stable.
   * - ``github``
     - GitHub tags, releases, or branches.
     - Synchronizing data with repository tags.
     - Stable for tags, mutable for branches.
   * - ``commit``
     - Exact repository commit SHA.
     - Reproducible development snapshots.
     - Stable if the commit remains available.
   * - ``latest``
     - Current default branch tip.
     - Exploration and development.
     - Mutable; record the resolved commit.

Cache behavior
--------------

Use a persistent cache directory for performance and reproducibility.

.. code-block:: python

   from pathlib import Path
   from synrxn.data import DataLoader

   loader = DataLoader(
       task="classification",
       source="zenodo",
       version="1.0.0",
       cache_dir=Path("~/.cache/synrxn").expanduser(),
   )

   df = loader.load("schneider_b")

The cache avoids repeated downloads, keeps resolved assets local, and makes it
easier to archive the exact data used in an experiment.

Reproducibility checklist
-------------------------

Record these fields with every benchmark result:

.. raw:: html

   <div class="synrxn-callout">
     <p><strong>Minimum record:</strong> package version, task family, dataset name, source mode, data version or commit SHA, split column or split seed, preprocessing steps, and evaluation script commit.</p>
   </div>

For example:

.. code-block:: text

   package: synrxn==1.0.0
   task: classification
   dataset: schneider_b
   source: zenodo
   version: 1.0.0
   split: published split column
   cache: ~/.cache/synrxn

See also
--------

- :doc:`Data Records <data_records>` for the complete dataset inventory.
- :doc:`Tutorials and Examples <tutorials_and_examples>` for source-mode and
  split workflows.
- :doc:`API Reference <api>` for generated API documentation.
