Issues
======

Use GitHub issues to report documentation problems, dataset inconsistencies,
API bugs, feature requests, or questions about a specific benchmark release.

.. raw:: html

   <div class="synrxn-issue-actions">
     <a class="synrxn-button" href="https://github.com/TieuLongPhan/SynRXN/issues"><i class="fa-brands fa-github" aria-hidden="true"></i> Open GitHub Issues</a>
     <a class="synrxn-button secondary-light" href="https://github.com/TieuLongPhan/SynRXN"><i class="fa-solid fa-code-branch" aria-hidden="true"></i> Repository</a>
   </div>

What to include
---------------

.. list-table::
   :header-rows: 1
   :widths: 28 72
   :class: synrxn-table

   * - Issue type
     - Helpful details
   * - Dataset problem
     - Task, dataset name, row id if available, package version, Zenodo DOI, manifest checksum, and expected vs. observed behavior.
   * - API bug
     - Minimal code example, installed ``synrxn`` version, Python version, operating system, traceback, and cache/source settings.
   * - Documentation issue
     - Page URL or filename, section heading, incorrect text, and suggested correction.
   * - Feature request
     - Use case, task family, desired API or data fields, and whether the request affects reproducibility or backwards compatibility.

Before opening an issue
-----------------------

- Check :doc:`What's New <whats_new>` for recent release notes and migration
  guidance.
- Check :doc:`Data Records <data_records>` for current dataset names, row counts,
  and source citations.
- Include the exact Zenodo version DOI for data-related reports so maintainers
  can reproduce the issue.

Issue title examples
--------------------

.. code-block:: text

   [data] property/b97xd3 checksum mismatch in v1.0.0 archive
   [api] DataLoader fails when source="zenodo" and cache_dir is relative
   [docs] Citation page should update Zenodo DOI for v1.0.1
   [feature] Add manifest validation command to CLI
