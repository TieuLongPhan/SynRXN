Task metrics, splits and dummy controls
=======================================

The 40 tasks in the main snapshot have metric, split and dummy-control profiles. These
profiles describe software interfaces. A source adapter must still establish
permitted inputs, reference completeness, units, candidate pools and dependence
before a dataset result is scientifically comparable.

Two different checks
--------------------

An **empirical dummy baseline** predicts held-out records without learning a
chemical model. It uses training statistics, declared candidate IDs or a causal
observed state. For example, a training median predicts one native response for
every held-out reaction, while a random condition selector shuffles available
candidate IDs without seeing their measured responses.

A **synthetic evaluator check** supplies tiny references with known answers:
perfect predictions, controlled wrong predictions and missing outputs. Reference
copies and oracle orderings belong only to this check. They are not models and
must never appear as empirical baseline results. Passing synthetic checks does
not validate source labels, experimental independence or a model's utility.

Responsibilities remain in the existing package:

* ``synrxn.metrics``: numerical and chemistry-aware scorers.
* ``synrxn.split``: membership algorithms and dependency constraints.
* ``synrxn.baseline.dummy``: train-only statistical and random controls.
* ``synrxn.benchmarks``: source-qualified contracts and synthetic metric checks.

Task-by-task evaluation
-----------------------

The following table gives the primary decision, useful weak controls and the
split unit to qualify for each task. Alternatives are separate experiments, not
interchangeable names for the same generalization claim. Geometry, mapping,
forward products, mechanisms, reactive potentials and dynamic control need
specialized baseline adapters; their discovery profiles report this explicitly.

Data acquisition and curation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 21 27 25 27

   * - Task ID
     - Primary scores
     - Dummy or scientific control
     - Primary split unit
   * - ``literature_extraction``
     - Entity and relation F1; document coverage
     - Empty annotations; sampled training annotations
     - Source document, including all annotation derivatives
   * - ``reaction_record_curation``
     - Edit F1; false-edit rate on clean controls
     - No edits; sampled training edits
     - Source document and uncorrupted parent record

Representation and search
~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 21 27 25 27

   * - Task ID
     - Primary scores
     - Dummy or scientific control
     - Primary split unit
   * - ``atom_mapping``
     - Symmetry-aware top-k correspondence recovery; valid coverage
     - Element-compatible correspondence adapter; missing-output check
     - Complete parent reaction and all mapping variants
   * - ``reaction_balancing``
     - Species recovery; atom/charge conservation checked separately
     - Empty completion; sampled training completion
     - Complete parent reaction and all deletion variants
   * - ``reaction_classification``
     - Macro-F1, MCC; per-class recall
     - Training majority/prior; uniform labels
     - Transformation groups; stratify only without breaking groups
   * - ``reaction_clustering``
     - ARI, NMI; stability reported separately
     - One cluster; random clusters with predeclared k
     - Transformation groups for inductive fitting; declare transductive variants
   * - ``reaction_retrieval``
     - Observed-positive recall@k and reciprocal rank
     - Uniform random order of available document IDs
     - Query variants; pin gallery and declare document exposure

Single-step transformations
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 21 27 25 27

   * - Task ID
     - Primary scores
     - Dummy or scientific control
     - Primary split unit
   * - ``forward_prediction``
     - Accepted-product top-k recovery; valid output coverage
     - Training-retrieval/product adapter; missing-output check
     - Parent reaction shared with precursor and center views
   * - ``retrosynthesis``
     - Recorded-precursor recall@k; reference coverage
     - Random order in a declared canonical candidate library
     - Parent reaction; alternatives and duplicate projections together
   * - ``reaction_center``
     - Site/bond precision and recall under qualified indices
     - Empty edits; sampled normalized training edits
     - Parent reaction with all site/candidate projections together

Outcomes and feasibility
~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 21 27 25 27

   * - Task ID
     - Primary scores
     - Dummy or scientific control
     - Primary split unit
   * - ``yield_conversion``
     - MAE and RMSE per native endpoint
     - Training median for MAE; mean for squared error
     - Replicate observation groups; separate chemistry/campaign transfer
   * - ``reaction_success``
     - Average precision, Brier score; AUROC and calibration
     - Training prevalence; constant 0.5; random probabilities
     - Same observations as yield; native labels retained

Selectivity and mixtures
~~~~~~~~~~~~~~~~~~~~~~~~

RegioSQM supplies atom-site labels, and Minot supplies recorded products.
Use the site-set metric for RegioSQM. Use observed formula recovery for the
solid-state impurity source and native yield MAE for the single known
benzaldehyde side product.

.. list-table::
   :header-rows: 1
   :widths: 21 27 25 27

   * - Task ID
     - Primary scores
     - Dummy or scientific control
     - Primary split unit
   * - ``regioselectivity``
     - Qualified site recovery or recorded-product recall
     - Empty site set; training-sampled sites or random candidate ranking
     - Substrate/reaction parents; all candidate sites together
   * - ``enantioselectivity``
     - Native ee/energy MAE; direction only with resolved sign
     - Training median/mean
     - Substrate/reaction groups; stereochemical variants retained
   * - ``product_distribution``
     - Named component MAEs; vector L1; coverage
     - Training component means/medians; whole-vector resampling
     - Replicate condition cells; declared solvent-pair or condition-region holdout
   * - ``impurity_prediction``
     - Observed-identity recall or known-impurity native MAE
     - Empty identity set; sampled training sets; scalar median
     - Mixture/replicate groups; declared composition/system or publication transfer

Conditions and procedures
~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 21 27 25 27

   * - Task ID
     - Primary scores
     - Dummy or scientific control
     - Primary split unit
   * - ``condition_recovery``
     - Role-aware set F1; native numeric errors
     - Empty recipe; sampled training recipe; numeric median
     - Transformation and parent-procedure groups
   * - ``condition_ranking``
     - Panel-macro observed regret at 1 and 3
     - Uniform random order within each complete measured panel
     - Whole panels plus audited cross-panel dependencies
   * - ``sequential_optimization``
     - Best measured response, regret and hit recovery versus budget
     - Random order with identical initialization/budget
     - Whole panels/campaigns; reveal only acquired outcomes
   * - ``procedure_generation``
     - Action-list exact match and edit distance
     - Empty procedure; sampled training action sequence
     - Parent procedure/document; paraphrases together

Physics and kinetics
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 21 27 25 27

   * - Task ID
     - Primary scores
     - Dummy or scientific control
     - Primary split unit
   * - ``activation_barrier``
     - Native-quantity MAE/RMSE, separated by evidence/method
     - Training median/mean per qualified endpoint
     - Parent reaction across sites, conformers, solvents and methods
   * - ``reaction_energetics``
     - Separate enthalpy/free-energy MAE/RMSE
     - Training median/mean per quantity
     - Same parent membership as overlapping barrier/rate endpoints
   * - ``reaction_rates``
     - Native or declared-log-rate MAE/RMSE
     - Training median/mean in the declared target scale
     - Reactant-pair/parent groups; distinguish from scaffold transfer
   * - ``reactive_potential``
     - Separate energy/force errors; complete-frame coverage
     - Constant-energy/zero-force adapter in native units
     - Complete reactive trajectories; optional reaction-family holdout
   * - ``kinetic_parameter_inference``
     - Native parameter error; held-out-trajectory error; identifiability separately
     - Training parameter template or source-specific fitted numerical control
     - Complete parameterized mechanism; two source mechanisms cannot form three mechanism-disjoint partitions

Mechanisms and networks
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 21 27 25 27

   * - Task ID
     - Primary scores
     - Dummy or scientific control
     - Primary split unit
   * - ``mechanistic_prediction``
     - Replay-qualified event and trajectory top-k recovery
     - Qualified mechanism-template adapter; missing/invalid-output checks
     - Complete mechanism and all steps/alternative proposals
   * - ``ts_geometry``
     - Symmetry-aware aligned RMSD; coverage
     - Endpoint/template geometry adapter with atom correspondence
     - Endpoint/TS triplet and conformer/reverse descendants
   * - ``reaction_network_modeling``
     - Directed stoichiometric-edge recovery; trajectory error
     - Empty network; source-specific training-network template
     - Complete network/mechanism and simulated descendants

Analysis and monitoring
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 21 27 25 27

   * - Task ID
     - Primary scores
     - Dummy or scientific control
     - Primary split unit
   * - ``analytical_interpretation``
     - Native component/area MAE; vector L1
     - Training component mean/median
     - Whole measured experiments and derived spectra
   * - ``reaction_progress_monitoring``
     - Causal next-step/trajectory errors versus persistence
     - Repeat last permitted observed value
     - Whole trajectories; causal windows and train-only templates

Components and new reactivity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 21 27 25 27

   * - Task ID
     - Primary scores
     - Dummy or scientific control
     - Primary split unit
   * - ``reaction_component_design``
     - Recorded-component recovery; finite measured-oracle utility
     - Random declared-library order; design constraints checked separately
     - Component-identity and campaign holdouts are distinct protocols
   * - ``new_reactivity_discovery``
     - Native hits and distinct hit-bearing substrate groups recovered versus budget
     - Uniform random measured-attempt order
     - Current proposed substrate-parent split; campaign/time transfer requires separate qualification

Process engineering
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 21 27 25 27

   * - Task ID
     - Primary scores
     - Dummy or scientific control
     - Primary split unit
   * - ``reactor_scale_modeling``
     - Native yield/STY/property MAE per source
     - Training median/mean per physical endpoint
     - Geometry/feedstock and original experiments; keep interpolation descendants together
   * - ``process_control``
     - Closed-loop tracking ISE, action effort and constraint violations
     - Fixed bounded action or random-policy adapter with actual simulator rollouts
     - Whole episodes; held-out initial states or systems
   * - ``process_impact_assessment``
     - Native coke mass-percentage-point MAE
     - Training median/mean
     - Physical parent experiment and feedstock; interpolation is not independent evidence

Biocatalysis and enzyme function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Reactzyme binding queries by enzyme sequence and ranks reactions in a
fixed gallery. All 1,573 native test reaction groups also occur in the native
training and validation pool. Report EnzEng mutation scores by source and
assay, with DMSDB reported separately from the curated enzyme-engineering and
literature-extracted collections.

.. list-table::
   :header-rows: 1
   :widths: 21 27 25 27

   * - Task ID
     - Primary scores
     - Dummy or scientific control
     - Primary split unit
   * - ``enzyme_activity``
     - Assay-macro discrimination and calibration
     - Within-training assay prevalence/majority
     - For EHreact: substrate/scaffold holdout within recorded assays; no unseen-protein claim
   * - ``enzyme_retrieval``
     - Observed-positive query recall@k and reciprocal rank
     - Uniform random order in the declared gallery
     - Separate sequence and reaction holdouts; pin gallery and query direction
   * - ``mutation_effects``
     - Within-assay delta MAE, direction and ranking
     - Zero change; training median/mean
     - Parent enzyme/assay components; separate within-parent mutation prediction
   * - ``active_sites``
     - Protein-macro observed-positive residue recall@k
     - Uniform random ranking of eligible residue IDs
     - Whole proteins and shared mechanism components
   * - ``enzyme_kinetic_parameters``
     - Separate native/log kcat, Km and Ki MAE/RMSE
     - Training median/mean per parameter and context
     - Enzyme/substrate measurement groups; audited homology clusters for transfer

Use a dummy predictor
---------------------

Profiles are discovered without loading datasets or importing chemical backends:

.. code:: python

   from synrxn.baseline.dummy import DummyPredictor, get_dummy_protocol
   from synrxn.metrics import evaluate_task

   print(get_dummy_protocol("product_distribution").as_dict())
   dummy = DummyPredictor("product_distribution", "train_mean", seed=7)
   dummy.fit(
       [[0.2, 0.6, 0.1], [0.4, 0.2, 0.2]],
       training_ids=["train-1", "train-2"],
   )
   predictions = dummy.predict(instance_ids=["held-out-1"])
   # [[0.3, 0.4, 0.15]], without normalizing the component sum.
   result = evaluate_task(
       "product_distribution", [[0.1, 0.7, 0.1]], predictions,
       unit="fraction", component_ids=["starting material", "product B", "product C"],
   )

Mean and median statistics, class frequencies and empirical resampling use the
training partition only. ``training_ids`` and prediction IDs must be disjoint.
A fixed seed and instance ID determine random predictions, so query order and
batching do not alter the result. This prevents accidental ID overlap; the caller
must still certify that the supplied training targets are actually training data.

Random ranking needs only available candidate IDs:

.. code:: python

   selector = DummyPredictor("condition_ranking", "random_order", seed=7)
   order = selector.predict(
       instance_ids=["held-out-panel"],
       candidate_ids=[["condition-A", "condition-B", "condition-C"]],
   )

Candidate-to-response dictionaries are rejected by the predictor. The measured
oracle is passed separately to the evaluator after selection. Available IDs and
pool membership must themselves follow the source contract. For sequential
comparisons, give every method the same initial observations and total budget.

``causal_persistence`` takes the last permitted observed value. It must not take
an unknown test target as that value. ``zero_change`` for mutation effects
predicts a native parent-to-variant delta of zero. Uniform probabilities, random
probabilities and sampled hard labels are distinct controls with different losses.

Verify the evaluator
--------------------

The following commands are provided for explicit execution. They do not train
models, generate fingerprints or benchmark source datasets::

    python -m synrxn.benchmarks.sanity --output .cache/synrxn/metric-sanity.json
    python -m synrxn.benchmarks.protocols --output .cache/synrxn/task-protocols.json

The sanity command checks each task's default variant on synthetic inputs:

* A perfect output reaches its specified ideal score.
* A controlled wrong output worsens that score.
* Missing outputs have zero coverage and the appropriate zero/undefined score.
* Chemistry checks require their supported backend; unavailable capabilities
  produce a nonzero, explicitly blocked report rather than a successful check.

Nondefault variants and source-specific validators need their own checks.
Mechanism checks require the supported hydrogen-preserving SynKit comparator.
In this local checkout, the validated backend is under
``.cache/synrxn/backends/synkit-1.6.3+synrxn.1``. From the repository root, use
it without changing the default environment::

    PYTHONPATH=.cache/synrxn/backends/synkit-1.6.3+synrxn.1 python -m synrxn.benchmarks.sanity --output .cache/synrxn/metric-sanity.json

The control fixture supplies state traces only to test arithmetic; it does not
simulate a controller. A real control baseline must produce a bounded policy's
closed-loop rollout. A transition-state RMSD check does not establish a saddle,
and a mechanism replay checks representation consistency rather than laboratory
mechanistic truth.

Useful analytical checks include constant probability 0.5 giving Brier score
0.25 on complete binary labels, tied-score AUROC 0.5 when both classes exist,
and perfect native scalar/component predictions giving zero error. Uniform
random top-one selection in a complete measured panel has expected response
equal to that panel's mean response. Finite random runs fluctuate around
expectations, so one draw should not be asserted to equal its expected score.

A random-ranking null can be repeated with fixed predeclared seeds to quantify
its Monte Carlo variation. Those repetitions are not additional experimental
datasets or independent chemical replications. Synthetic oracle copies and
reference permutations must remain labelled evaluator fixtures.

Metric interpretation and failures
----------------------------------

Inspect ``get_task_metrics(task_id).as_dict()`` for variants and context
requirements. Units are explicit: scalar/component/geometry targets require
``unit``; trajectories also need ``time_unit``; control adds ``action_unit`` and
bounds; energy and forces need their separate native units.

Every eligible reference retains a denominator. ``None`` represents a missing
prediction. Native regression and component errors are conditional on valid
finite outputs and must be compared together with coverage; dropping difficult
outputs cannot create a fair winning result. Classification and reference
recovery retain missing predictions as failures. Undefined quantities remain
``None``. Exact wrong probabilities can have infinite log loss; a strict JSON
consumer must choose an explicit representation for that quantity.

Composition errors preserve named component order and native scale. Finite
negative linear-model outputs still contribute error and are also reported as
physical-bound violations. The ``mean_total_mass_error`` field measures
predicted-versus-recorded component-sum disagreement. It does not identify the
physical mass omitted from the source measurement. No scorer renormalizes a
partially measured mixture.

Observed-positive retrieval is the default for the current literature, enzyme
association and active-site sources. Recall and reciprocal rank refer to the
supplied positive annotations. Unjudged candidates are not false positives or
confirmed negatives, and nDCG is suppressed in this mode. Complete-judgment
variants require an explicitly qualified candidate universe. Residue probability
AP additionally requires complete eligible binary labels; M-CSA's unannotated
residues cannot automatically supply true inactive examples.

Scope evaluation separates successful attempted experiments from distinct
successful substrate parents. Supply an audited candidate-to-parent mapping to
measure substrate coverage; repeated positive conditions for one substrate count
once. Fractions with no positive reference groups are undefined. Source hit
labels and any continuous conversion measurement remain separate endpoints.

Chemical normalization, atom/charge conservation, stereochemical sign,
identifiability, homology clustering, design constraints, saddle validation and
physical simulator fidelity require source-specific checks. Generic metric
coverage does not establish those properties.

Construct a split
-----------------

.. code:: python

   from synrxn.split import get_task_split_protocol, make_task_split

   protocol = get_task_split_protocol("condition_ranking")
   report = make_task_split(
       "condition_ranking", records,
       group_fields=("panel_id", "component_id"), seed=7,
       provenance={"source_revision": "pinned", "grouping_version": "audited-v1"},
   )
   train_rows = [records[i] for i in report["indices"]["train"]]

``group_fields`` means a transitive union of equal values in each field, not a
grouping by their combined tuple. A panel linked to another panel by a shared
component stays with it. Fields must be complete, source-audited and namespaced.
The splitter does not derive scaffolds or homology groups from arbitrary strings.

Whole documents, measured candidate panels, complete trajectories, mechanisms
and endpoint triplets are mandatory where required by the task. Native, random
diagnostic and transfer strategies retain those units. Audited equivalent
fields or complete hard-relationship components can be supplied explicitly.
Physical parent observations and accepted hard lineage remain together.
Campaign and broad parent identity are selectable transfer axes rather than
unconditional duplicate constraints: valid within-campaign substrate holdouts
must remain distinguishable from unseen-campaign evaluation.

Never pass ``split``, ``native_split``, ``proposed_split``, ``derived_split`` or
fold assignments as dependency groups. They specify memberships, not a reason
for records to stay together. The source registry currently mixes some of these
roles in ``group_fields``; the protocol audit separates them and leaves their
source qualification pending. A field named ``native_split`` does not prove its
labels came from a published native evaluation.

Available strategies include grouped and stratified-group assignment, audited
source/family/scaffold/protein/campaign/entity transfer, chronological grouping,
validated native memberships and random diagnostics. Fit preprocessing and
selection rules inside training only. Chronological splits require real source
timestamps and fail when interleaved hard dependencies prevent causal boundaries.
Native train/test-only releases keep their missing validation partition visible;
other requested partitions must have enough independent groups.

Report achieved sizes, IDs, groups, overlap, missing lineage, labels unseen in
training and the exact generalization axis. Duplicate-block constraints can
prevent exact row-ratio matching. Reuse one frozen membership artifact across
overlapping task projections. Retrieval galleries, finite-oracle candidate pools
and controller environments need contracts beyond a row split.

For current selected sources, HiTEA needs whole panels plus its prepared
cross-panel components; C06 supports solvent-pair/condition-region transfer;
QuantumPioneer supports proposed reactant-pair holdout. Minot's current scope
view has proposed substrate-parent memberships. Its column named
``native_split`` inherits an earlier generated expansion split and must not be
reported as a published native or chronological protocol. Campaign/date fields
need explicit extraction and qualification before stronger transfer claims.

Repeated-fold APIs reject empty or indivisible holdouts and discard stale
membership when the records change. Deterministic grouped outer folds repeat
the same partition. Use ``make_task_split`` with audited grouping fields for
source-specific protocols.

API discovery
-------------

.. code:: python

   from synrxn.baseline.dummy import list_dummy_protocols
   from synrxn.benchmarks.protocols import task_protocol_catalog
   from synrxn.metrics import list_task_metrics
   from synrxn.split import get_task_split_protocols

   metrics = list_task_metrics()
   splits = get_task_split_protocols()
   controls = list_dummy_protocols()
   audit = task_protocol_catalog()

The audit distinguishes full task-profile coverage, generic dummy availability,
specialized adapter requirements and unqualified source binding metadata. It
reads registry definitions and code profiles, not observations or model results.
