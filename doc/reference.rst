Reference
=========

Use this page when citing SynRXN or tracing the upstream datasets and methods
used throughout the benchmark inventory.

Primary SynRXN citations
------------------------

Cite the paper for the benchmark description and cite the exact Zenodo version
for the data/software archive used in experiments.

.. list-table::
   :header-rows: 1
   :widths: 26 52 22
   :class: synrxn-table

   * - Item
     - When to cite
     - Reference key
   * - Scientific Data paper
     - Always cite this when SynRXN contributes to a publication.
     - ``phan2026synrxn``
   * - Zenodo version record
     - Cite the exact archived release used for data loading, manifests, and reproducible benchmarking.
     - ``phan_synrxn_zenodo_v100`` or the key exported by Zenodo

BibTeX
------

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

Zenodo version BibTeX template
------------------------------

Use Zenodo's exported citation for the exact release you used. The example below
shows the current documentation example and should be replaced when a newer data
archive is used.

.. code-block:: bibtex

   @misc{phan_synrxn_zenodo_v008,
     title = {synrxn: A Benchmarking Framework and Open Data Repository for Computer-Aided Synthesis Planning},
     author = {Phan, Tieu Long},
     publisher = {Zenodo},
     year = {2025},
     version = {v0.0.8},
     doi = {10.5281/zenodo.17672847},
     url = {https://doi.org/10.5281/zenodo.17672847}
   }

Recommended citation record
---------------------------

For reproducible benchmark reports, include both bibliographic and computational
metadata.

.. list-table::
   :header-rows: 1
   :widths: 28 72
   :class: synrxn-table

   * - Field
     - Example
   * - Paper citation
     - ``phan2026synrxn``
   * - Zenodo citation
     - Version-specific DOI exported by Zenodo, for example ``10.5281/zenodo.17672847``
   * - Package version
     - ``synrxn==1.0.0``
   * - Data source
     - ``zenodo`` / ``github`` / ``commit``
   * - Data version
     - release number, tag, full commit SHA, or Zenodo version DOI
   * - Dataset
     - ``classification/schneider_b`` or ``property/b97xd3``
   * - Split
     - published ``split`` column or generated split seed/configuration

Dataset and method references
-----------------------------

The bibliography below contains the primary SynRXN descriptor, the Zenodo release
entry, and the upstream dataset, benchmark, and method references cited
throughout the documentation, including the per-dataset sources listed in
:doc:`Data Records <data_records>`.

Bibliography
------------

.. bibliography:: refs.bib
   :style: unsrt
   :all:
