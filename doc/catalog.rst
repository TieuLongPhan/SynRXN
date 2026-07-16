.. _dataset-catalog:

Dataset Catalog
===============

Choose a benchmark, inspect its reaction representation and split, then export
a compact comparison record for an experiment log. The catalog is built from
the same metadata and manifest used by the Python API and release checks.

.. raw:: html

   <div id="synrxn-catalog" class="catalog-app" aria-busy="true">
     <section class="catalog-benchmark-hero" aria-labelledby="catalog-title">
       <div>
         <p class="catalog-eyebrow">Benchmark workspace</p>
         <h2 id="catalog-title">Find the right reaction benchmark.</h2>
         <p>Every card exposes task, target, split coverage, provenance, and a reaction preview—so a dataset can be judged before it enters a model run.</p>
       </div>
       <div class="catalog-hero-actions">
         <a href="#catalog-results" class="catalog-primary-link">Explore benchmarks <span aria-hidden="true">↓</span></a>
         <button id="catalog-clear" type="button" class="catalog-quiet-button">Reset filters</button>
       </div>
     </section>
     <section class="catalog-scoreboard" aria-label="Release summary">
       <div class="catalog-score"><span id="catalog-datasets">—</span><small>benchmarks</small></div>
       <div class="catalog-score"><span id="catalog-records">—</span><small>reaction records</small></div>
       <div class="catalog-score"><span id="catalog-splits">—</span><small>with published splits</small></div>
       <div class="catalog-score catalog-score-integrity"><span><i class="fa-solid fa-shield-halved" aria-hidden="true"></i> Verified</span><small>manifest-backed release</small></div>
     </section>
     <section id="catalog-example" class="catalog-example" hidden aria-label="Example benchmark"></section>
     <section class="catalog-toolbar" aria-label="Dataset filters">
       <label>Search
         <input id="catalog-search" type="search" placeholder="Name, target, column, citation…" autocomplete="off">
       </label>
       <label>Task
         <select id="catalog-task"><option value="">All task families</option></select>
       </label>
       <label>License
         <select id="catalog-license"><option value="">All licenses</option></select>
       </label>
       <label>Target
         <select id="catalog-target"><option value="">All targets</option></select>
       </label>
       <label>Scale
         <select id="catalog-size"><option value="">Any scale</option><option value="small">Small (&lt; 10k)</option><option value="medium">Medium (10k–100k)</option><option value="large">Large (&gt; 100k)</option></select>
       </label>
       <label class="catalog-check"><input id="catalog-split" type="checkbox"> Published split</label>
     </section>
     <div class="catalog-status"><strong id="catalog-count">Loading catalog…</strong><span id="catalog-release"></span></div>
     <div id="catalog-error" class="catalog-error" hidden></div>
     <div id="catalog-results" class="catalog-results"></div>
     <section id="catalog-compare" class="catalog-compare" hidden aria-live="polite"></section>
     <section id="catalog-detail" class="catalog-detail" hidden aria-live="polite"></section>
     <noscript><p>JavaScript powers interactive filtering. The complete static
     inventory remains available in <a href="data_records.html">Data Records</a>.</p></noscript>
   </div>
   <script src="_static/catalog-data.js"></script>

The underlying catalog remains available without JavaScript through
:doc:`Data Records <data_records>` and from the command line:

.. code-block:: bash

   synrxn datasets list
   synrxn datasets describe property rgd1
