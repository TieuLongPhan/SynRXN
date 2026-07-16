.. _home:

SynRXN
======

.. raw:: html

   <section class="synrxn-intro">
     <div class="synrxn-intro-copy">
       <p class="synrxn-kicker">Reaction benchmark infrastructure</p>
       <h1>Curated reaction benchmarks for reproducible reaction informatics.</h1>
       <p>SynRXN brings atom mapping, classification, property prediction, rebalancing, and synthesis data into one versioned, citable resource for fair model evaluation.</p>
       <p class="synrxn-intro-links"><a href="catalog.html">Browse the benchmark catalog <span aria-hidden="true">→</span></a><a href="getting_started.html">Read the quickstart</a><a href="api.html">API reference</a></p>
     </div>
     <dl class="synrxn-intro-facts">
       <div><dt>Coverage</dt><dd>Five reaction task families</dd></div>
       <div><dt>Release model</dt><dd>Versioned, manifest-verified assets</dd></div>
       <div><dt>Evaluation</dt><dd>Published or reproducible splits</dd></div>
     </dl>
   </section>

.. raw:: html

   <div class="synrxn-badge-row">
     <a class="synrxn-badge" href="https://pypi.org/project/synrxn/"><i class="fa-brands fa-python" aria-hidden="true"></i> PyPI</a>
     <a class="synrxn-badge" href="https://doi.org/10.1038/s41597-026-07260-w"><i class="fa-solid fa-file-lines" aria-hidden="true"></i> Scientific Data</a>
     <a class="synrxn-badge" href="https://doi.org/10.5281/zenodo.17297258"><i class="fa-solid fa-box-archive" aria-hidden="true"></i> Zenodo</a>
     <a class="synrxn-badge" href="https://github.com/TieuLongPhan/SynRXN"><i class="fa-brands fa-github" aria-hidden="true"></i> GitHub</a>
   </div>

Why SynRXN?
-----------

Benchmarking reaction-informatics methods is difficult when datasets, splits,
reaction representations, and provenance metadata are scattered across releases
or publications. SynRXN solves this by providing a consistent data layout,
version-aware access, documented schema conventions, and reproducible splitting
utilities.

.. raw:: html

   <div class="synrxn-icon-grid">
     <div class="synrxn-icon-card">
       <span class="synrxn-icon"><i class="fa-solid fa-database" aria-hidden="true"></i></span>
       <strong>Curated datasets</strong>
       <p>Compressed CSV records grouped by benchmark task, with stable columns and source citations.</p>
     </div>
     <div class="synrxn-icon-card">
       <span class="synrxn-icon"><i class="fa-solid fa-code-branch" aria-hidden="true"></i></span>
       <strong>Version-aware loading</strong>
       <p>Use archived Zenodo releases, GitHub tags, exact commits, or development snapshots.</p>
     </div>
     <div class="synrxn-icon-card">
       <span class="synrxn-icon"><i class="fa-solid fa-shuffle" aria-hidden="true"></i></span>
       <strong>Reproducible splits</strong>
       <p>Create repeated k-fold or train/validation/test partitions with controlled random seeds.</p>
     </div>
     <div class="synrxn-icon-card">
       <span class="synrxn-icon"><i class="fa-solid fa-plug" aria-hidden="true"></i></span>
       <strong>Accessible API</strong>
       <p>Load datasets as pandas DataFrames and integrate them directly into ML pipelines.</p>
     </div>
   </div>

Framework overview
------------------

.. figure:: figure/synrxn.png
   :alt: SynRXN framework overview
   :align: center
   :class: synrxn-hero-figure

   **Figure 1.** Curated reaction datasets are grouped by benchmark task,
   distributed through reproducible releases, loaded through a shared API, and
   evaluated with task-specific workflows.

The SynRXN pipeline separates the data lifecycle into four practical layers:

1. **Curated assets** under ``Data/<task>/<dataset>.csv.gz``.
2. **Versioned distribution** through Zenodo records, GitHub releases, or exact
   Git commits.
3. **Reusable utilities** for loading, caching, manifest handling, and splitting.
4. **Task-specific evaluation** for mapping, classification, property,
   rebalancing, and synthesis workflows.
   
Benchmark collections
---------------------

.. raw:: html

   <div class="synrxn-collection-grid">
     <a class="synrxn-collection-card" href="data_records.html#reaction-rebalancing">
       <span class="tagline"><i class="fa-solid fa-scale-balanced" aria-hidden="true"></i> RBL</span>
       <strong>Reaction rebalancing</strong>
       <p>Recover chemically balanced reactions when reactants, products, solvents, catalysts, or auxiliary species are missing.</p>
     </a>

     <a class="synrxn-collection-card" href="data_records.html#atom-to-atom-mapping">
       <span class="tagline"><i class="fa-solid fa-atom" aria-hidden="true"></i> AAM</span>
       <strong>Atom-to-atom mapping</strong>
       <p>Evaluate predicted atom correspondences against curated, rule-based, or consensus reference mappings.</p>
     </a>

     <a class="synrxn-collection-card" href="data_records.html#reaction-classification">
       <span class="tagline"><i class="fa-solid fa-tags" aria-hidden="true"></i> CLS</span>
       <strong>Reaction classification</strong>
       <p>Assign reaction classes, named-reaction labels, template identifiers, or hierarchical enzyme annotations.</p>
     </a>

     <a class="synrxn-collection-card" href="data_records.html#reaction-property-prediction">
       <span class="tagline"><i class="fa-solid fa-chart-line" aria-hidden="true"></i> PROP</span>
       <strong>Property prediction</strong>
       <p>Model kinetic, thermodynamic, and experimental reaction properties such as barriers, enthalpies, rates, yields, and free energies.</p>
     </a>

     <a class="synrxn-collection-card" href="data_records.html#synthesis-prediction">
       <span class="tagline"><i class="fa-solid fa-flask" aria-hidden="true"></i> SYN</span>
       <strong>Synthesis prediction</strong>
       <p>Support forward synthesis, retrosynthesis, reagent prediction, condition recommendation, and reaction-center identification.</p>
     </a>

     <a class="synrxn-collection-card todo" href="data_records.html#mechanism-prediction">
       <span class="tagline"><i class="fa-solid fa-gears" aria-hidden="true"></i> MECH</span>
       <strong>Mechanism prediction</strong>
       <p>TODO: add datasets for elementary steps, intermediates, and mechanistic pathways.</p>
     </a>
   </div>

Quick example
-------------

Install SynRXN, load a released classification benchmark, and inspect the first
records:

.. code-block:: bash

   pip install synrxn

.. code-block:: python

   from pathlib import Path
   from synrxn.data import DataLoader

   loader = DataLoader(
       task="classification",
       source="zenodo",
       version="1.0.0",
       cache_dir=Path("~/.cache/synrxn").expanduser(),
   )

   print(loader.available_names())
   df = loader.load("schneider_b")
   print(df.head())


Citation
--------

If you use SynRXN in published work, cite the primary data descriptor and the
exact Zenodo version used for your data archive. 

.. code-block:: bibtex

   @article{phan2026synrxn,
     title = {SynRXN: An Open Benchmark and Curated Dataset for Computational Reaction Modeling},
     author = {Phan, Tieu-Long and Nguyen Song, Nhu-Ngoc and Stadler, Peter F.},
     journal = {Scientific Data},
     volume = {13},
     pages = {625},
     year = {2026},
     doi = {10.1038/s41597-026-07260-w},
     url = {https://www.nature.com/articles/s41597-026-07260-w}
   }


.. toctree::
   :caption: Documentation
   :maxdepth: 2
   :hidden:

   Getting Started <getting_started>
   Data Concept <data_concept>
   Dataset Catalog <catalog>
   Parquet Queries and Service <query_and_service>
   AAM Validation <aam_validation>
   Data Records <data_records>
   Tutorials and Examples <tutorials_and_examples>

.. toctree::
   :caption: Section Navigation
   :maxdepth: 1
   :hidden:

   What's New <whats_new>
   API Reference <api>
   Paper <paper>
   Issues <issues>
   References <reference>
