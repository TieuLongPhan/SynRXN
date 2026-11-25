.. _home:

SynRXN
======

.. container:: synrxn-home-hero

   .. container:: synrxn-home-intro

      .. rubric:: Reaction informatics database

      Explore reaction datasets by task, from atom mapping and reaction yield
      to kinetics, mechanisms, and enzyme function.

      .. raw:: html

         <nav class="synrxn-home-actions" aria-label="Data access">
           <a href="families.html">Browse the catalogue</a>
           <a href="query_and_service.html">Use DuckDB</a>
         </nav>

   .. figure:: _static/catalog-reactions/yield-hitea.cdk.svg
      :alt: Reaction diagram showing methylation of an indazole nitrogen

      N-methylation example from `HiTEA <data_records.html?record=yield%2Fhitea>`_.

Scientific families
-------------------

The main snapshot has **12 families, 40 tasks, and 102 data tables**. The public
deposit is pending.

.. rst-class:: synrxn-home-families

.. synrxn-family-summary

Shared tables contribute to each family that uses them. See the
:doc:`evaluation protocols <task_evaluation>` for metrics and splits, and the
:doc:`reconstruction guide <data_concept>` to build the snapshot from source.

Citation
--------

Phan, T.-L., Nguyen Song, N.-N., and Stadler, P. F.
*SynRXN: An Open Benchmark and Curated Dataset for Computational Reaction
Modeling.* Scientific Data **13**, 625 (2026).
`doi:10.1038/s41597-026-07260-w <https://doi.org/10.1038/s41597-026-07260-w>`_.

Cite the individual data sources and the exact data version used. The
:doc:`reference page <reference>` provides citations and bibliography.

.. toctree::
   :caption: Data
   :maxdepth: 1
   :hidden:

   Database Access <query_and_service>
   Families and Tasks <families>
   Task definitions <task_definitions>
   Data records <data_records>

.. toctree::
   :caption: Methods and software
   :maxdepth: 1
   :hidden:

   Getting Started <getting_started>
   Data Model and Reconstruction <data_concept>
   Metrics and Split Protocols <task_evaluation>
   Tutorials and Examples <tutorials_and_examples>
   API Reference <api>

.. toctree::
   :caption: Project
   :maxdepth: 1
   :hidden:

   References and citation <reference>
   Changes <whats_new>
   Issues <issues>
