.. _api-reference:

API Reference
=============

This page provides an overview of the public Python API exposed by
:mod:`synrxn`. It focuses on:

- dataset access via :class:`synrxn.data.DataLoader`,
- splitting utilities via :class:`synrxn.split.repeated_kfold.RepeatedKFoldsSplitter`,
- the command-line interface entry point (:mod:`synrxn.main`), and

If you are looking for high-level usage examples, see
:doc:`Getting Started <getting_started>` and :ref:`tutorials-and-examples`.

-------------------------------
Data access: :mod:`synrxn.data`
-------------------------------

The :mod:`synrxn.data` module is the main entry point for accessing curated
datasets and their manifests.

.. automodule:: synrxn.data
   :members:
   :undoc-members:
   :show-inheritance:

Key classes
~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   synrxn.data.DataLoader

Typical responsibilities of :class:`synrxn.data.DataLoader` include:

- listing available datasets for a given task (classification, property, …),
- resolving data sources (Zenodo, GitHub release, specific commit, latest),
- managing local caching directories, and
- returning data in a convenient tabular form (e.g. Pandas ``DataFrame``).

For concrete examples of how to configure and use :class:`~synrxn.data.DataLoader`,
see :ref:`tutorials-and-examples`.

----------------------------------------------------------
Splitting utilities: :mod:`synrxn.split.repeated_kfold`
----------------------------------------------------------

The :mod:`synrxn.split.repeated_kfold` module provides tools for
reproducible repeated k-fold splitting of datasets.

.. automodule:: synrxn.split.repeated_kfold
   :members:
   :undoc-members:
   :show-inheritance:

Key classes
~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   synrxn.split.repeated_kfold.RepeatedKFoldsSplitter

Typical use cases include:

- generating repeated k-fold splits for property and classification tasks,
- deriving approximate train/validation/test splits via a user-specified ratio,
- preserving label distributions via stratified splitting, and
- exporting/importing split indices for exact reproducibility.

Examples of end-to-end usage are provided in
:ref:`tutorials-and-examples`.

------------------------------------------------
Command-line interface: :mod:`synrxn.main`
------------------------------------------------

The :mod:`synrxn.main` module exposes the command-line interface used when
invoking:

.. code-block:: bash

   python -m synrxn ...

In particular, it provides the ``build`` subcommand that can be used to
rebuild datasets and manifests from their original sources (intended for
advanced users and developers).

.. automodule:: synrxn.main
   :members:
   :undoc-members:
   :show-inheritance:

Depending on your installed version, you can inspect available commands via:

.. code-block:: bash

   python -m synrxn --help

and consult :ref:`tutorials-and-examples` for concrete ``python -m synrxn build``
usage patterns.
