.. _tutorials-and-examples:

Tutorials and Examples
======================

This page collects practical workflows for loading SynRXN datasets, switching
between data sources, caching assets, creating reproducible splits, and rebuilding
curated records during development.

.. raw:: html

   <div class="synrxn-card-grid">
     <a class="synrxn-card" href="#load-a-released-dataset-from-zenodo">
       <strong><i class="fa-solid fa-box-archive" aria-hidden="true"></i> Released data</strong>
       <span>Use Zenodo when you need citable, stable benchmark snapshots.</span>
     </a>
     <a class="synrxn-card" href="#pin-a-github-release-or-commit">
       <strong><i class="fa-brands fa-github" aria-hidden="true"></i> Git snapshots</strong>
       <span>Use tags or exact commit SHAs to match source code and data state.</span>
     </a>
     <a class="synrxn-card" href="#make-reproducible-splits">
       <strong><i class="fa-solid fa-shuffle" aria-hidden="true"></i> Splits</strong>
       <span>Create repeated k-fold splits or train/validation/test partitions.</span>
     </a>
     <a class="synrxn-card" href="#rebuild-datasets-from-source">
       <strong><i class="fa-solid fa-screwdriver-wrench" aria-hidden="true"></i> Rebuilds</strong>
       <span>Regenerate curated artifacts and manifests from upstream raw sources.</span>
     </a>
   </div>

Canonical imports
-----------------

.. code-block:: python

   from pathlib import Path
   from synrxn import DatasetCatalog
   from synrxn.data import DataLoader
   from synrxn.split.repeated_kfold import RepeatedKFoldsSplitter

A reusable cache path keeps downloaded archives local:

.. code-block:: python

   CACHE = Path("~/.cache/synrxn").expanduser()

Task aliases
------------

``DataLoader`` normalizes common task aliases.

.. list-table::
   :header-rows: 1
   :widths: 30 30 40
   :class: synrxn-table

   * - Alias
     - Canonical task
     - Typical use
   * - ``class``
     - ``classification``
     - reaction class, template, and enzyme labels
   * - ``prop``
     - ``property``
     - reaction property prediction
   * - ``syn``
     - ``synthesis``
     - synthesis and retrosynthesis records

Browse the catalog and local data
---------------------------------

The packaged catalog can be inspected without a network request:

.. code-block:: python

   from synrxn import DataLoader, DatasetCatalog

   catalog = DatasetCatalog()
   for record in catalog.list(task="classification", has_split=True):
       print(record.name, record.targets, record.split_values)

   loader = DataLoader(
       task="classification",
       source="local",
       data_dir="Data",
   )
   test_sample = loader.load(
       "schneider_b",
       columns=["r_id", "label", "split"],
       filters={"split": "test"},
       nrows=10_000,
   )

For bounded-memory processing, iterate local records in batches:

.. code-block:: python

   for batch in loader.iter_batches("schneider_b", batch_size=25_000):
       print(len(batch))

Load a released dataset from Zenodo
-----------------------------------

Use Zenodo for a citable release snapshot. This is the recommended mode for
published experiments.

.. code-block:: python

   from pathlib import Path
   from synrxn.data import DataLoader

   loader = DataLoader(
       task="classification",
       source="zenodo",
       version="1.0.0",
       cache_dir=Path("~/.cache/synrxn").expanduser(),
   )

   print("Available datasets:", loader.available_names())

   df = loader.load("schneider_b")
   print("Rows:", len(df))
   print("Columns:", df.columns.tolist())
   print(df.head(3))

Expected workflow:

1. Resolve the versioned release.
2. Download the archive if it is not already cached.
3. Load the selected compressed CSV as a pandas DataFrame.
4. Use the documented columns for training or evaluation.

Pin a GitHub release or commit
------------------------------

GitHub release tag
~~~~~~~~~~~~~~~~~~

Use a release tag when you want data and code aligned with a repository tag.

.. code-block:: python

   loader = DataLoader(
       task="classification",
       source="github",
       version="v1.0.0",
       cache_dir=Path("~/.cache/synrxn").expanduser(),
       gh_enable=True,
   )

   print(loader.available_names())
   df = loader.load("schneider_b")
   print(df.shape)

Exact commit SHA
~~~~~~~~~~~~~~~~

Use a full commit SHA for the strongest development-snapshot reproducibility.

.. code-block:: python

   loader = DataLoader(
       task="property",
       source="commit",
       version="3e1612e2199e8b0e369fce3ed9aff3dda68e4c32",
       cache_dir=Path("~/.cache/synrxn").expanduser(),
       gh_enable=True,
   )

   df = loader.load("b97xd3")
   print(df[["r_id", "ea", "dh"]].head())

Development latest
~~~~~~~~~~~~~~~~~~

``latest`` is convenient during exploration but should not be the only version
record in a formal benchmark.

.. code-block:: python

   loader = DataLoader(
       task="classification",
       source="github",
       version="latest",
       cache_dir=Path("~/.cache/synrxn").expanduser(),
       gh_enable=True,
   )

   df = loader.load("schneider_b")
   print(df.shape)

.. raw:: html

   <div class="synrxn-callout">
     <p><strong>Best practice:</strong> If exploratory results from <code>latest</code> become important, rerun the experiment with the resolved commit SHA or an archived release before reporting the result.</p>
   </div>

Explore dataset schemas
-----------------------

List available datasets, load one table, and summarize column types.

.. code-block:: python

   loader = DataLoader(
       task="property",
       source="zenodo",
       version="1.0.0",
       cache_dir=CACHE,
   )

   for name in loader.available_names():
       print(name)

   df = loader.load("rgd1")
   print(df.dtypes)
   print(df.describe(include="all").T.head(10))

A compact helper for quick inspection:

.. code-block:: python

   def inspect_dataset(task: str, name: str, version: str = "1.0.0"):
       loader = DataLoader(task=task, source="zenodo", version=version, cache_dir=CACHE)
       df = loader.load(name)
       return {
           "task": task,
           "name": name,
           "shape": df.shape,
           "columns": df.columns.tolist(),
           "has_split": "split" in df.columns,
       }

   print(inspect_dataset("classification", "schneider_b"))
   print(inspect_dataset("property", "b97xd3"))

Make reproducible splits
------------------------

Use published splits when available
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some benchmark records contain a ``split`` column. Keep it for direct comparison
with upstream or previously published results.

.. code-block:: python

   df = loader.load("schneider_b")

   if "split" in df.columns:
       print(df["split"].value_counts(dropna=False))

Generate repeated k-fold splits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When a dataset does not include the split you need, use a deterministic splitter.

.. code-block:: python

   loader = DataLoader(
       task="property",
       source="zenodo",
       version="1.0.0",
       cache_dir=CACHE,
   )
   df = loader.load("b97xd3")

   splitter = RepeatedKFoldsSplitter(
       n_splits=5,
       n_repeats=3,
       random_state=2026,
       val_ratio=0.1,
   )

   split_indices = splitter.split(df)
   print(split_indices)

Export split indices
~~~~~~~~~~~~~~~~~~~~

Persist generated split indices next to model outputs.

.. code-block:: python

   import json
   from pathlib import Path

   out = Path("runs/b97xd3_splits.json")
   out.parent.mkdir(parents=True, exist_ok=True)

   with out.open("w") as fh:
       json.dump(split_indices.to_dict(), fh, indent=2)

   print("Wrote", out)

Train/evaluate loop sketch
~~~~~~~~~~~~~~~~~~~~~~~~~~

The exact training code depends on your model, but the pattern is stable:

.. code-block:: python

   for split_name, indices in split_indices.items():
       train_df = df.iloc[indices["train"]]
       valid_df = df.iloc[indices["valid"]]
       test_df = df.iloc[indices["test"]]

       # model = fit_model(train_df, valid_df)
       # metrics = evaluate_model(model, test_df)
       # save_metrics(split_name, metrics)

       print(split_name, len(train_df), len(valid_df), len(test_df))

Rebuild datasets from source
----------------------------

Dataset rebuilding is intended for maintainers and advanced users who need to
verify a release, add a dataset, or regenerate manifests after modifying build
scripts.

When to rebuild
~~~~~~~~~~~~~~~

- You want to verify that published archives are reproducible.
- You are developing new curated records.
- You modified preprocessing, schema normalization, or split-generation logic.
- You need to regenerate manifests and checksums.

CLI smoke test
~~~~~~~~~~~~~~

Inspect the available command-line interface:

.. code-block:: bash

   python -m synrxn --help
   python -m synrxn build --help
   python -m synrxn verify-manifest --help
   python -m synrxn validate --help

Verify the committed release before rebuilding it:

.. code-block:: bash

   python -m synrxn verify-manifest --manifest manifest.json --root Data
   python -m synrxn validate \
     --data-dir Data \
     --metadata Data/metadata.yaml \
     --manifest manifest.json

Typical rebuild command
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python -m synrxn build --classification -- --out-dir Data/classification

Regenerate the release manifest after the dataset builders finish:

.. code-block:: bash

   python -m synrxn.build_manifest \
     --data-dir Data \
     --manifest-output manifest.json \
     --citation-output /tmp/CITATION.generated.cff

A rebuild workflow should record:

- the raw upstream source and retrieval date,
- the SynRXN Git commit,
- the builder command and configuration,
- generated row counts and checksums,
- schema changes compared with the previous release.

Caching and reproducibility tips
--------------------------------

- Use a persistent cache such as ``~/.cache/synrxn``.
- Prefer ``source="zenodo"`` for published experiments.
- Prefer exact commit SHAs over ``latest`` for development snapshots.
- Keep generated split indices under version control or alongside run outputs.
- Record the package version and the data version in each experiment log.

Minimal experiment log
----------------------

.. code-block:: text

   package: synrxn==1.0.0
   task: property
   dataset: b97xd3
   source: zenodo
   version: 1.0.0
   cache_dir: ~/.cache/synrxn
   split: repeated k-fold, n_splits=5, n_repeats=3, random_state=2026, val_ratio=0.1
   model_commit: <your-method-commit>

See also
--------

- :doc:`Getting Started <getting_started>` for the first installation and load.
- :doc:`Data Concept <data_concept>` for task folders and schema conventions.
- :doc:`Data Records <data_records>` for dataset counts and citations.
- :doc:`API Reference <api>` for generated class and module documentation.
