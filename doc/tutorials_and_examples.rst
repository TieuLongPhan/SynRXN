.. _tutorials-and-examples:

Tutorials and Examples
======================

This page collects practical examples that show how to:

- load **SynRXN** datasets from different sources (Zenodo, GitHub releases, specific commits, latest),
- cache datasets locally for efficient reuse,
- build reproducible train/validation/test splits with
  :class:`synrxn.split.repeated_kfold.RepeatedKFoldsSplitter`, and
- optionally **rebuild** datasets from source using the CLI.

If you have not installed **SynRXN** yet, see :doc:`Getting Started <getting_started>` first.

Overview
--------

The examples below focus on two main components:

- :class:`synrxn.data.DataLoader` for accessing curated datasets and manifests.
- :class:`synrxn.split.repeated_kfold.RepeatedKFoldsSplitter` for building
  repeated k-fold splits in a reproducible way.

Loading datasets with ``DataLoader``
------------------------------------

The central entry point is :class:`synrxn.data.DataLoader`.  
It abstracts over:

- **task type** (e.g. ``"classification"`` vs ``"property"``),
- **source** (Zenodo, GitHub release, specific commit, latest),
- **version** (release number, tag, or commit SHA), and
- a local **cache directory**.

Canonical imports
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pathlib import Path
   from synrxn.data import DataLoader

1) Zenodo (stable release)
~~~~~~~~~~~~~~~~~~~~~~~~~~

Use this when you want a **stable, archived release** of the data, e.g. for a
paper or a long-term benchmark.

.. code-block:: python

   dl = DataLoader(
       task="classification",
       source="zenodo",
       version="0.0.6",  # SynRXN data release on Zenodo
       cache_dir=Path("~/.cache/synrxn").expanduser(),
   )

   print("Available datasets:", dl.available_names())
   df = dl.load("schneider_b")
   print("Rows:", len(df), "Columns:", df.columns.tolist())

This will download the archive from the Zenodo record linked to version
``0.0.6`` (if not already cached), verify checksums, and return a ``DataFrame``
for the requested dataset.

2) GitHub release tag
~~~~~~~~~~~~~~~~~~~~~

Use this when you want to align with a **GitHub release** (e.g. for synchronising
with code at the same tag).

.. code-block:: python

   dl = DataLoader(
       task="classification",
       source="github",
       version="v0.0.6",  # Git tag in the SynRXN repo
       cache_dir=Path("~/.cache/synrxn").expanduser(),
       gh_enable=True,
   )

   print(dl.available_names())
   df = dl.load("schneider_b")
   print("Rows:", len(df))

Here ``gh_enable=True`` allows :class:`DataLoader` to query and download assets
directly from the GitHub repository.

3) GitHub commit (pin to SHA)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For **maximum reproducibility**, pin to an exact commit SHA. This guarantees you
can re-create the exact state of the data and code at that revision.

.. code-block:: python

   dl = DataLoader(
       task="classification",
       source="commit",
       version="3e1612e2199e8b0e369fce3ed9aff3dda68e4c32",  # commit SHA
       cache_dir=Path("~/.cache/synrxn").expanduser(),
       gh_enable=True,
   )

   print(dl.available_names())
   df = dl.load("schneider_b")
   print(df.head(2))

In your experiment logs, record the commit SHA and the dataset name to make this
reproducible.

4) GitHub latest (development)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``version="latest"`` for **development and exploratory work**, when you want
to track the tip of the main branch.

.. code-block:: python

   dl = DataLoader(
       task="classification",
       source="github",
       version="latest",
       cache_dir=Path("~/.cache/synrxn").expanduser(),
       gh_enable=True,
   )

   print(dl.available_names())
   df = dl.load("schneider_b")
   print(df.shape)

Note that ``latest`` is **not stable** over time. If you obtain important results
using this configuration, record the manifest’s commit hash or export it
alongside your outputs.

Property datasets and repeated k-fold splitting
-----------------------------------------------

For property prediction tasks, **SynRXN** provides datasets such as ``"b97xd3"``.
You can combine :class:`DataLoader` with
:class:`synrxn.split.repeated_kfold.RepeatedKFoldsSplitter` to generate
reproducible splits.

Simple property-dataset example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pathlib import Path
   from synrxn.data import DataLoader
   from synrxn.split.repeated_kfold import RepeatedKFoldsSplitter

   dl = DataLoader(
       task="property",
       source="commit",
       version="latest",  # development; pin to a SHA for stable experiments
       cache_dir=Path("~/.cache/synrxn").expanduser(),
       gh_enable=True,
   )
   df = dl.load("b97xd3")

   splitter = RepeatedKFoldsSplitter(
       n_splits=5,
       n_repeats=2,
       ratio=(8, 1, 1),      # train:val:test ratio in "fold units"
       shuffle=True,
       random_state=1,       # ensures deterministic shuffling
   )

   # Pre-compute and store all splits (indices) internally
   splitter.prepare_splits(df, stratify=None)

   # Retrieve repetition 0, fold 0 as (train, val, test) DataFrames
   train_df, val_df, test_df = splitter.get_split(
       rep_index=0,
       fold_index=0,
       as_frame=True,
   )

   print("train/val/test:", len(train_df), len(val_df), len(test_df))

Typical output (exact numbers depend on dataset size) might look like:

.. code-block:: text

   train/val/test: 8000 1000 1000

Stratified splits (classification)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For classification tasks, you can preserve label distributions by passing a
column to ``stratify``:

.. code-block:: python

   df = dl.load("schneider_b")

   splitter = RepeatedKFoldsSplitter(
       n_splits=5,
       n_repeats=3,
       ratio=(8, 1, 1),
       shuffle=True,
       random_state=42,
   )

   splitter.prepare_splits(df, stratify=df["label"])   # column name may differ

   train_df, val_df, test_df = splitter.get_split(0, 0, as_frame=True)

Saving and reusing split indices
--------------------------------

Although :class:`RepeatedKFoldsSplitter` can reconstruct splits from its RNG
configuration, it is often convenient to save explicit indices.

If your version exposes a helper such as ``get_indices``, the pattern is:

.. code-block:: python

   import json

   # Example for repetition 0 (API sketch; adapt to your actual splitter)
   indices_rep0 = splitter.get_indices(rep_index=0)  # e.g. returns a dict with 'train', 'val', 'test'

   with open("splits_rep0.json", "w") as fh:
       json.dump(indices_rep0, fh, indent=2)

Later, in a different script:

.. code-block:: python

   import json

   with open("splits_rep0.json") as fh:
       saved = json.load(fh)

   # assuming saved["train"] etc. are index lists
   train_df = df.iloc[saved["train"]]
   val_df   = df.iloc[saved["val"]]
   test_df  = df.iloc[saved["test"]]

Rebuilding datasets via the CLI
-------------------------------

In addition to loading published datasets, **SynRXN** can rebuild dataset artifacts
from source using its command-line interface. This is useful when:

- you want to verify that the published Zenodo/GitHub archives are reproducible,
- you are developing new datasets or modifying existing build scripts, or
- you need to regenerate manifests and split indices after making changes.

Basic usage
~~~~~~~~~~~

From the root of the **SynRXN** repository (editable install recommended), run:

.. code-block:: bash

   cd SynRXN

   python -m synrxn --help

You should see a list of available subcommands and build targets, for example
(informally):

.. code-block:: text

   usage: python -m synrxn [-h] {build,...} ...

   positional arguments:
     {build,...}
       build        build datasets and manifests

   optional arguments:
     -h, --help     show this help message and exit

Property datasets
~~~~~~~~~~~~~~~~~

To rebuild the property-prediction datasets (e.g. ``b97xd3`` and related
benchmarks) and their manifests:

.. code-block:: bash

   python -m synrxn build --property

This will:

- download or locate the raw source files,
- process them into the canonical **SynRXN** format,
- generate or update manifest files (schema, counts, checksums, splits), and
- write the results to the configured output directory (typically under a
  dedicated data/build folder in the repository).

Dry-run mode
~~~~~~~~~~~~

If your version includes a dry-run flag, you can preview which commands would
run *without* writing files (useful for debugging and CI):

.. code-block:: bash

   python -m synrxn build --property --dry-run

Consult the CLI help in your installed version for the exact set of flags and
available build targets.

Practical tips and common patterns
----------------------------------

- Use a **persistent cache directory** (e.g. ``~/.cache/synrxn``) so you do not
  redownload datasets.
- For **published** work, prefer:

  - ``source="zenodo", version="0.0.6"`` (or later stable release), or
  - ``source="commit", version="<full SHA>"``

  and record this choice in your methods section.

- When debugging or exploring new datasets, ``version="latest"`` is convenient,
  but always log the associated manifest or commit for future reference.
- If you hit GitHub rate limits, either:

  - fall back to the Zenodo source, or
  - configure authentication (e.g. via ``GITHUB_TOKEN``) for the process.
- When you rebuild datasets with ``python -m synrxn build``, record:

  - the exact command (including flags),
  - the **SynRXN** version and repository commit, and
  - the resulting manifest(s).