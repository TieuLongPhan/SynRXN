.. _aam-validation:

AAM Validation
==============

SynRXN requires ``synkit>=1.5.0,<1.6.0`` and uses
``synkit.Chem.Reaction.Mapper.AAMValidator`` for the public
:func:`synrxn.metrics.template.acc_aam` metric. The retained SynRXN validator is
used as a compatibility reference while downstream code migrates.

Compatibility setting
---------------------

SynKit 1.5 must use ``strip_unbalanced_maps=True``—the instance default—to
reproduce the historical SynRXN benchmark. It removes atom-map labels present
on only one reaction side before comparing the reaction-center or ITS graph.
Strict mode (``False``) intentionally produces different results for reactions
with unmapped leaving or entering atoms.

.. code-block:: python

   from synkit.Chem.Reaction.Mapper import AAMValidator

   validator = AAMValidator(strip_unbalanced_maps=True)
   equivalent = validator.smiles_check(prediction, ground_truth, check_method="RC")

Full comparison
---------------

The row-level comparison covered 5,904 reactions in the five AAM artifacts
that contain predictions, four mapper columns, and both RC and ITS methods:
47,232 Boolean decisions in total. SynKit 1.5.0 and the SynRXN reference
validator produced zero mismatches and identical accuracies in all 40 result
vectors. ``enzyme_map`` is not included because it is a ground-truth-only
artifact without mapper prediction columns.

Reproduce the check with:

.. code-block:: bash

   python script/compare_aam_validators.py --methods RC ITS --n-jobs -1 \
       --output aam-validator-comparison.json

The checked summary is available as :download:`aam-validator-comparison.json
<_static/aam-validator-comparison.json>`.

RDKit compatibility
-------------------

The legacy SynRXN utilities now use ``rdMolStandardize.Normalizer``,
``TautomerEnumerator``, and ``Reionizer``. This restores compatibility with the
current RDKit API, which removed the older ``normalize``, ``tautomer``, and
``charge`` module imports.
