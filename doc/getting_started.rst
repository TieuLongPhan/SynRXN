.. _getting_started:

Getting started
===============

Open a database snapshot
------------------------

With an existing ``synrxn.duckdb`` file and Python 3.11 or newer, install DuckDB::

    python -m pip install duckdb

Open the file read-only and list the tasks in the outcomes family.

.. code-block:: python

   from pathlib import Path

   import duckdb

   path = Path("synrxn.duckdb").resolve(strict=True)
   with duckdb.connect(str(path), read_only=True) as db:
       tasks = db.execute(
           "SELECT task_id, title FROM catalog.tasks WHERE family_id = ? ORDER BY task_id",
           ["outcomes"],
       ).fetchall()

   print(tasks)

The 1.5.0 snapshot has been built and verified locally. A public deposit is
pending. The :doc:`database guide <query_and_service>` lists the files and
shows how to read a source table, inspect its terms, and recover native assets.

Use the Python package
----------------------

Install SynRXN from the repository checkout::

    python -m venv .venv
    source .venv/bin/activate
    python -m pip install -e ".[query]"

Use this installation for ``DataLoader``, the task metric and split APIs, and
the reconstruction commands. From a checkout containing prepared data, load
a small sample with the dataset ID shown in :doc:`data_records`.

.. code-block:: python

   from synrxn import DataLoader

   loader = DataLoader(task="aam", source="local", data_dir="Data")
   examples = loader.load("ecoli", nrows=5)

``DataLoader`` resolves this ID through ``Data/families/``. A code-only checkout
contains the package and catalogue metadata. To obtain a prepared table, use
its pinned source or follow the :doc:`reconstruction guide <data_concept>`.

Before evaluating a model
-------------------------

Use the :doc:`family browser <families>` to select a task and inspect the
table's observation unit, targets and eligibility. Follow the
:doc:`evaluation guide <task_evaluation>` for dependencies between rows and
split construction. Record the source revision, table identity, and split
membership with the results.
