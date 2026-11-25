# SynRXN 2.0: Additional Reaction Datasets and Task Expansion

**Research date:** 20 October 2026  
**Scope:** 26 candidate dataset entries and 18 proposed task definitions  
**Purpose:** A source-linked survey and integration roadmap for expanding SynRXN beyond its current public catalogue.

## Executive recommendation

The strongest opportunities are **quantitative selectivity, measured condition ranking, product-distribution prediction, 3D transition states, enzyme-conditioned reactivity, and multistep synthesis planning**.

The recommended first integration pass is **BH-HTE-OOD, the Minot borylation data, Denmark CPA, Catechol, RDB19-Rad, Transition1x, and PaRoutes**, with **EnzEngDB and ReactZyme** next for the biochemical expansion.

These recommendations weigh task value, usable labels, integration effort, and source overlap. They are recommendations for curation and evaluation; the candidates have not already been merged into SynRXN.

The [current public SynRXN catalogue](https://synrxn.readthedocs.io/en/latest/data_records.html) already lists EnzymeMap, EHreact activity, phosphatase conversion, FlowER, several barrier datasets, log-rate prediction, and reagent/catalyst inference. These were treated as the baseline when judging what each addition contributes. The 18 task designs below include both new types of supervision and extensions of existing tasks.

Reported sizes use the source's actual unit—experiments, averaged measurements, unique reactions, reaction–solvent pairs, or geometry configurations. The number retained after SynRXN curation will differ. Counts from different datasets should not be added into a unique-reaction total.

## Contents

1. [Experimental reactions: conditions, outcomes, and selectivity](#1-experimental-reactions-conditions-outcomes-and-selectivity)
2. [Computed reactions: transition states, solvent effects, and kinetics](#2-computed-reactions-transition-states-solvent-effects-and-kinetics)
3. [Enzymes and multistep synthesis](#3-enzymes-and-multistep-synthesis)
4. [Additional candidates worth retaining](#4-additional-candidates-worth-retaining)
5. [Task definitions to prioritize](#5-task-definitions-to-prioritize)
6. [A manageable integration order](#6-a-manageable-integration-order)
7. Appendix: all 26 candidates, priorities, source/data links, license notes, and all 18 detailed task designs.

## 1. Experimental reactions: conditions, outcomes, and selectivity

**This would be the first area of expansion.** These datasets connect reaction structures to measured outcomes and decisions about which experiment to perform.

| Dataset and access | Reported size | Tasks and value for SynRXN 2.0 |
| --- | --- | --- |
| **[BH-HTE-OOD](https://github.com/schwallergroup/bh-hte-ood)** — curated Buchwald–Hartwig reactions, 2026 | Approximately **27,500 experiments**: about 11,300 new J&J records and 16,200 published HTE records | **Condition-dependent yield, reaction success, measured condition ranking, and generalization across sources.** The curated CSV contains aryl/amine and catalyst/base/solvent representations, outcome information, and source identifiers. A strong practical starting point. Existing HTE collections incorporated into it must be deduplicated. [Publication](https://www.nature.com/articles/s43588-026-01017-6); [archive](https://zenodo.org/records/19636649). |
| **[Cernak C–N coupling screen](https://open-reaction-database.org/dataset/ord_dataset-805ad863feef48579d95d86a728035f4)** — 2026 | **50,688 experiments**, covering **8 substrate pairs** | **Catalyst/ligand/condition ranking and sequential experiment selection.** Its strength is dense experimental condition coverage. The large experiment count should not be interpreted as broad substrate diversity: it uses two amines and four aryl halides. [Publication](https://doi.org/10.1021/jacs.6c05959). |
| **[Minot C–H borylation data](https://zenodo.org/records/20773622)** — 2026 | **6,865 HTE observations involving 568 substrates**, plus a separate regioselectivity panel | **Reaction outcome, atom-level site prediction, and experiment selection.** Particularly attractive because the release includes reaction outcomes, regioselectivity annotations, and successive active-learning rounds. Failed attempts remain specific to the tested conditions. [Publication](https://www.nature.com/articles/s43588-026-01063-0). |
| **[ORDerly condition benchmarks](https://figshare.com/articles/dataset/ORDerly_chemical_reactions_condition_benchmarks/23298467)** | Large, version-dependent processed datasets; freeze and recount the selected release | **Recovery of reported solvent and reagent sets.** This enriches SynRXN's existing reagent/catalyst inference with more structured condition labels. Much of the chemistry comes through ORD/USPTO, so its main contribution is improved supervision and task design, with substantial source overlap to audit. [Publication](https://doi.org/10.1021/acs.jcim.4c00292); [project](https://github.com/sustainable-processes/ORDerly). |
| **[Denmark CPA-catalysed thiol addition](https://pmc.ncbi.nlm.nih.gov/articles/PMC6417887/)** | **1,075 averaged catalyst–substrate labels**, from **2,150 duplicate assay runs** | **Quantitative enantioselectivity and chiral-catalyst ranking.** The design comprises 43 catalysts × 5 imines × 5 thiols. This gives a compact benchmark with controlled catalyst and substrate holdouts. Preserve the original stereochemical conventions and resolve data reuse terms before redistributing a curated version. [Publication](https://doi.org/10.1126/science.aau5631). |
| **[OrgAIcat / iSynth](https://github.com/DeepSynthesis/orgaicat)** | **More than 22,000 reaction data points** for aminocatalytic aldol and Michael chemistry | **Enantioselectivity prediction from catalyst, substrates, and conditions.** A larger literature-derived complement to the Denmark panel. Raw spreadsheets are available. Document, catalyst-family, and substrate-family splits would help distinguish interpolation from transfer to new chemistry. [Data directory](https://github.com/DeepSynthesis/orgaicat/tree/main/data). |
| **[Catechol solvent-selection benchmark](https://github.com/jpfolch/catechol_solvent_selection)** — 2025 | **1,227 condition observations** across 24 solvents and mixtures | **Condition-dependent product distributions and solvent optimization.** Inputs include solvent identities/ratio, temperature, and residence time; outputs include starting-material and two-product yields. It offers a useful multi-output task, although its chemical scope is narrow. [Publication](https://proceedings.neurips.cc/paper_files/paper/2025/file/ad61e68e2c4208d5d092b3ecd6064b46-Paper-Datasets_and_Benchmarks_Track.pdf). |

### Turning these datasets into distinct tasks

Three condition-related tasks deserve separate definitions:

1. **Recorded-condition recovery:** Given reactants and the intended product, predict the published solvent/catalyst/reagent set. ORDerly is suitable.
2. **Measured condition ranking:** Given a reaction and several experimentally tested conditions, rank those conditions by outcome. The HTE datasets are suitable.
3. **Sequential experiment selection:** Given previous experiments, select the next condition to test from a measured candidate pool.

The first evaluates recovery of a reported recipe. The second and third evaluate decisions against measured outcomes. An unreported condition cannot automatically be treated as unsuccessful.

Make **quantitative selectivity** explicit. Product stereochemistry, favoured-enantiomer identity, signed ee, absolute ee, and enantiomeric ratio are different targets. Where a barrier difference is derived from selectivity, preserve the temperature and conversion assumptions. This matters especially when combining the [Denmark](https://pmc.ncbi.nlm.nih.gov/articles/PMC6417887/) and [iSynth](https://github.com/DeepSynthesis/orgaicat) records.

For Catechol, use the task name **“residence-time- and condition-dependent product-distribution prediction.”** The measurements do not by themselves provide a complete kinetic mechanism or direct rate-constant labels. See the [published dataset description](https://proceedings.neurips.cc/paper_files/paper/2025/file/ad61e68e2c4208d5d092b3ecd6064b46-Paper-Datasets_and_Benchmarks_Track.pdf).

## 2. Computed reactions: transition states, solvent effects, and kinetics

These sources can broaden the physical information attached to a reaction. **Transition1x and OrgReact are especially valuable for adding structural outputs; RDB19-Rad is a practical addition for solvent-dependent radical chemistry.**

| Dataset and access | Reported size | Tasks and integration considerations |
| --- | --- | --- |
| **[Transition1x](https://figshare.com/articles/dataset/Transition1x/19614657)** | Approximately **9.6 million configurations**; **10,073 converged reaction pathways** | **Transition-state geometry, reactive energies/forces, and reaction-path modelling.** The configurations come from calculations on and around NEB reaction paths. Split by complete reaction/pathway, and check shared parent chemistry with existing small-molecule barrier datasets. NEB configurations are not physical time trajectories. [Publication](https://www.nature.com/articles/s41597-022-01870-w). |
| **[OrgReact](https://doi.org/10.6084/m9.figshare.31668682)** — 2026 | **9,649 validated computed reactions**, spanning **55 bond-transformation types** | **TS geometry, barrier prediction, and bond-edit-conditioned structure generation.** The release provides reactant, product, and TS coordinates plus barrier and bond-change tables. It is attractive for graph-edit/geometry links. Graph perception and mapping need validation, and seed overlap with Grambow/Transition1x must be tracked. [Publication](https://www.nature.com/articles/s41597-026-07771-6). |
| **[RDB19-Rad](https://zenodo.org/records/11493786)** | **5,600 elementary radical reactions**, with gas-phase and **40-solvent** calculations | **Solvent-dependent activation free energies and reaction free energies.** It contains mapped elementary reactions and validated transition states. Expanded solvent, direction, and resonance representations should remain linked to the underlying reaction identity. [Publication](https://doi.org/10.1021/acs.jpca.4c04121). |
| **[RxnSolvKSE / Chung–Green](https://zenodo.org/records/8423911)** | Final pretraining set: **26,448 unique reactions**, **7,796,583 reaction–solvent pairs**, **295 solvents** | **Solvent contributions to activation free energy and enthalpy.** The targets are solvent corrections, such as ΔΔG‡solv, rather than absolute solution barriers. A selected 500,000-pair subset offers a more manageable starting point. [Publication](https://pubs.rsc.org/en/content/articlehtml/2024/sc/d3sc05353a). |
| **[QuantumPioneer](https://zenodo.org/records/20060606)** — 2026 | **167,237 HAT transition states**; associated species and solvation-property collections | **Computed rate constants, Arrhenius parameters, and barriers.** The property tables include rates at 298 K and Arrhenius fits over specified temperature ranges. The chemistry is primarily hydrogen-atom transfer. Mapping is available in the QM representation, while property tables require validated joins. [Publication](https://doi.org/10.1021/jacs.6c10371); [data schema](https://github.com/QuantumPioneer/database). |
| **[LiPRED](https://doi.org/10.5281/zenodo.8396100)** — 2026 paper | **4,513 calculations for 28 reactions** | **Solution-phase barrier prediction, solvent ranking, and transfer between computational methods.** Useful as a controlled panel. Its size reflects repeated reaction/solvent/method calculations, so it contributes much less independent chemistry than the row count suggests. [Publication](https://pubs.rsc.org/en/content/articlehtml/2026/cp/d6cp00088f). |

### What these add to the task portfolio

Separate three kinds of supervision:

- **Reaction graph → scalar property:** an extension of existing SynRXN property prediction.
- **Reaction + solvent/temperature → conditional property:** a richer setting for testing whether a model responds correctly to conditions.
- **Reactant/product structures → transition-state structure or path:** a distinct structural prediction task.

For curation, keep **electronic barriers ΔE‡**, **free-energy barriers ΔG‡**, **solvent corrections ΔΔG‡solv**, and **computed rate constants** as separately defined targets. Their methods, temperature ranges, and standard-state conventions belong with the labels.

For a first release, the compact property tables from RDB19-Rad or QuantumPioneer would be easier to integrate than their complete calculation archives. Transition1x can initially support a selected geometry benchmark without requiring SynRXN to train a large reactive potential.

## 3. Enzymes and multistep synthesis

These datasets add information that ordinary reaction SMILES do not capture: the enzyme sequence, engineered variant, assay context, or relationships between steps in a synthesis route.

| Dataset and access | Reported size | Tasks and value |
| --- | --- | --- |
| **[EnzEngDB](https://zenodo.org/records/17310823)** — NAR 2026 | Manual collection: **6,234 sequence–function entries**, **635 reactions**, **1,846 variants**, from 40 papers | **Reaction-conditioned enzyme performance and mutation-effect prediction.** Entries connect enzyme sequences and reaction SMILES to outcomes such as yield, total turnover number, and selectivity. The published manual subset is a useful starting point; whole-site totals include additional extracted records and should be reported separately. [Publication](https://doi.org/10.1093/nar/gkaf1142); [project](https://github.com/fhalab/EnzymeEngineeringDB). |
| **[ReactZyme](https://github.com/WillHua127/ReactZyme)** | **178,463 positive enzyme–reaction pairs**, covering **7,726 distinct reactions** | **Reaction → enzyme retrieval**, **enzyme → reaction retrieval**, and compatibility ranking. This adds sequence-specific associations beyond EC classification. The published data lack atom mapping. Constructed negative pairs are unannotated distractors, not experimentally established inactivity. [Paper](https://arxiv.org/html/2408.13659v3). |
| **[CatPred-DB](https://zenodo.org/records/14775076)** | **23,197 kcat**, **41,174 Km**, and **11,929 Ki** entries | **Sequence-conditioned enzyme kinetics.** A valuable source, but the endpoints should remain separate. Complete product-side reaction equations are not guaranteed. Duplicate measurements were aggregated using different rules for different parameters, which must remain visible in the provenance. [Publication](https://www.nature.com/articles/s41467-025-57215-9); [project](https://github.com/maranasgroup/CatPred-DB). |
| **[PaRoutes 2.0](https://zenodo.org/records/7341155)** | The **n1 and n5 sets each contain 10,000 reference routes** | **Multistep retrosynthetic planning, route recovery, and route diversity.** It supplies targets, stock lists, routes, and evaluation resources. The two sets should not automatically be counted as 20,000 independent routes. Their USPTO origins also require checks against single-step training data. [Publication](https://doi.org/10.1039/D2DD00015F); [project](https://github.com/MolecularAI/PaRoutes). |

### Preference within this group

**EnzEngDB is the closest fit for extending a reaction-centred resource into enzyme engineering.** It supports questions such as: *For this reaction, which enzyme variant gives the best measured outcome?* Those comparisons should be made within compatible assays or campaigns. See the [EnzEngDB publication](https://doi.org/10.1093/nar/gkaf1142).

**ReactZyme provides a complementary retrieval task:** *Which enzyme sequences are associated with this reaction?* That is different from predicting an EC label, and it needs protein-cluster and reaction-level holdouts.

**CatPred should be a targeted kinetics module.** Start with kcat records that can be reliably connected to reaction identities. Km and especially Ki broaden the molecular-function scope, but they should not be presented as interchangeable reaction-rate labels.

**PaRoutes is the most direct route-level addition.** A reference route is one documented solution. Evaluation should allow other valid routes and report the starting-material stock and search budget alongside the solved fraction.

## 4. Additional candidates worth retaining

These are useful for particular directions, but should be placed after the main additions above or treated as extensions of existing SynRXN tasks.

| Candidate | Why retain it, and why it is a later priority |
| --- | --- |
| **[CHORISO](https://figshare.com/articles/dataset/CHORISO_-_chemical_reaction_SMILES_from_academic_journals/22598230)** | **2,224,239 unique reaction SMILES** from academic literature. Useful for forward/retro prediction and transfer beyond patent chemistry. Its stereo/regio challenge subsets concern product prediction; they do not provide quantitative ee or product-ratio measurements. [Paper](https://arxiv.org/abs/2312.09004); [project](https://github.com/schwallergroup/choriso). |
| **[CompleteRXN](https://huggingface.co/datasets/completeRXN-benchmark-26/completeRXN)** | **206,423 incomplete/complete reaction pairs** for completion and balancing. A relevant extension for SynRXN/SynRBL evaluation. The targets are FlowER-derived, so retain parent identifiers and label-generation provenance. The benchmark-code MIT license should not be assumed to cover all underlying data. [Paper](https://arxiv.org/html/2605.00222v2). |
| **[Parrot USPTO-Condition](https://github.com/wangxr0526/Parrot)** | An alternative source of catalyst, solvent, and reagent labels. Useful if its processing choices suit SynRXN better than ORDerly. Freeze and recount the chosen release, audit USPTO overlap, and keep the restricted Reaxys counterpart outside an assumed open-data release. [Publication](https://doi.org/10.34133/research.0231). |
| **[Perera Suzuki–Miyaura HTE](https://www.science.org/doi/10.1126/science.aap9112)** | **5,760 experiments** for yield and condition optimization. A useful additional experimental reaction class. Check overlap with aggregated HTE sources before adding it as another independent collection. |
| **[OC20NEB](https://facebookresearch.github.io/fairchem/oc20neb/)** | **932 DFT NEB trajectories** for surface reactions. Supports surface TS and barrier tasks. It requires periodic cells, slabs, and adsorbate context, making it a substantial representation extension beyond molecular reaction SMILES. [Publication](https://doi.org/10.1021/acscatal.4c04272). |
| **[RMG database](https://github.com/ReactionMechanismGenerator/RMG-database)** | Versioned kinetics libraries and rate-rule training data could support temperature-dependent kinetics and reaction-network tasks. The 2022 paper reports approximately 21,000 library reactions; a current frozen release needs recounting. Preserve whether parameters are measured, computed, or estimated. [Publication](https://doi.org/10.1021/acs.jcim.2c00965). |
| **[EasIFA data](https://github.com/wangxr0526/EasIFA)** | **102,944 training enzyme–reaction pairs**, with additional validation/test sets, for **reaction-conditioned active-site annotation**. Pairing uses EC identifiers, so retain that evidence level. It also overlaps ECREACT and, in a separate panel, M-CSA. [Publication](https://www.nature.com/articles/s41467-024-51511-6). |
| **[EnzChemRED](https://zenodo.org/records/11067998)** | **1,210 expert-curated PubMed abstracts** for extracting enzymes, chemical conversions, and linked identifiers. A useful literature-to-reaction task. These are annotated abstracts, not 1,210 complete atom-mapped reaction equations. [Publication](https://www.nature.com/articles/s41597-024-03835-7); [paper version](https://arxiv.org/abs/2404.14209). |
| **[PMechDB](https://deeprxn.ics.uci.edu/pmechdb/download) / [RMechDB](https://deeprxn.ics.uci.edu/rmechdb/howtouse)** | Potential sources of polar/radical elementary steps. Their official pages specify registration and CC BY-NC-ND terms. Retain them as candidates while resolving terms for modified redistribution and checking overlap with the mechanistic collections already planned. |

## 5. Task definitions to prioritize

The dataset list becomes more useful when each task has a precise input, output, and generalization claim. These are proposed benchmark designs; they are not claims that every source author used these exact protocols. The appendix reproduces all 18 detailed task designs from the companion catalogue.

| Proposed task | Input → target | Recommended evaluation |
| --- | --- | --- |
| **Measured condition ranking** | Reaction + candidate conditions → ranking by measured outcome | Hold out substrate pairs or campaigns. Report the yield lost by choosing the recommended condition instead of the best measured condition, plus ranking metrics. |
| **Quantitative stereoselectivity** | Substrates + chiral catalyst + conditions → ee or another explicitly defined selectivity target | Hold out catalyst and substrate families. Report error in native units; evaluate favoured configuration separately when the labels support it. |
| **Product-distribution prediction** | Reaction system + solvent + temperature + residence time → measured component yields | Hold out entire ramps or solvent systems. Report component-wise errors, including unreacted starting material. |
| **Sequential experiment selection** | Previous measured experiments → next experiment | Use a fixed measured candidate pool. Report best observed outcome versus experiment budget and variation across starting seeds. |
| **Transition-state geometry** | Reactant/product structures or reaction graph → TS coordinates | Split by parent reaction. Combine aligned geometry error with connectivity checks and a small, fixed set of saddle-point validations. |
| **Enzyme–reaction retrieval** | Reaction → ranked enzyme sequences, or sequence → ranked reactions | Use protein-cluster, reaction, and joint holdouts. Report Recall@k and MRR with a clearly defined candidate pool. |
| **Enzyme mutation effects** | Parent/variant sequence + reaction + assay context → change in performance | Hold out complete campaigns or lineages. Compare compatible assays and report within-campaign ranking or normalized changes. |
| **Multistep planning** | Target molecule + fixed stock → synthesis route | Report solved fraction, search budget, route diversity, and reference recovery separately. Accept alternative valid routes. |

This gives SynRXN 2.0 a clearer task expansion than counting every new label column as an independent task. For example, thresholding a measured yield creates a useful success-classification view, but the two views still share the same underlying experiments.

## 6. A manageable integration order

### First collection: experimental decisions and selectivity

Begin with **BH-HTE-OOD, the borylation data, Denmark CPA, and Catechol**.

Together, these support:

- Outcomes under specified conditions.
- Ranking experimentally tested conditions.
- Atom-level site prediction.
- Quantitative enantioselectivity.
- Multi-product response prediction.
- Selection of the next experiment.

**Cernak's screen** is a particularly useful addition if condition optimization becomes a central benchmark. **ORDerly** is useful for a broader recorded-condition recovery track.

### Second collection: geometry and solvent dependence

Add **Transition1x**, followed by **OrgReact**, and use **RDB19-Rad** as the first solvent-dependent radical panel.

RxnSolvKSE and QuantumPioneer can then extend this collection with relative solvent effects and temperature-dependent kinetic quantities. Their large numbers of related records make grouping and provenance especially important.

### Third collection: sequence and route context

Add **EnzEngDB and ReactZyme** for enzyme-specific tasks, and **PaRoutes** for multistep planning.

These require additional entities—protein sequences, assay records, and route graphs—but offer substantial task breadth. CatPred can follow once the linkage between kinetic measurements and reaction identities is sufficiently complete.

### Four curation decisions that would materially strengthen the release

**1. Separate reaction identity from observation identity.**

One transformation may have many solvent conditions, assay replicates, enzyme variants, or geometry frames. Keep a parent reaction identifier alongside observation-level identifiers. Report both counts.

**2. Preserve the type of evidence.**

Measured failure, thresholded low yield, missing annotation, a constructed retrieval distractor, and a computationally generated label have different meanings. They should remain distinguishable in the task tables.

**3. Audit overlap before assigning splits.**

Important overlap groups include USPTO-derived resources; BH collections that incorporate older HTE studies; Grambow/Transition1x/OrgReact; and biochemical resources sharing Rhea, BRENDA, or ECREACT provenance. A source-level test set can still contain chemistry already seen through another collection. Source provenance is documented in the linked dataset papers and repositories above.

**4. Keep task-specific representations available.**

A mapped reaction table can remain the common chemical index, with linked condition, protein, geometry, and route records. HTE observations often lack complete byproduct accounting; those records can still be useful for outcome prediction without being treated as fully balanced mechanistic ground truth.

**Recommended first integration pass:** BH-HTE-OOD, borylation, Denmark CPA, Catechol, RDB19-Rad, Transition1x, and PaRoutes; then EnzEngDB and ReactZyme for the biochemical expansion. This prioritizes distinct supervision and usable evaluation settings while keeping the initial benchmarking effort manageable.

---

## Appendix A. Candidate priorities, access, and license notes

The following 26 entries reproduce the final catalogue’s candidate identifiers, editorial priorities, license notes, and access links. A dash means that no separate link was recorded in that field. Priorities are recommendations, not measured scores.

| ID | Candidate | Priority | Data-license status | Paper / primary source | Data | Project | License evidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C01 | BH-HTE-OOD | 1 Integrate first | CC BY 4.0 on the Zenodo data record. Code Apache-2.0. Preserve original HTE provenance. | [Paper / primary source](<https://www.nature.com/articles/s43588-026-01017-6>) | [Data](<https://zenodo.org/records/19636649>) | [Project](<https://github.com/schwallergroup/bh-hte-ood>) | [License evidence](<https://zenodo.org/api/records/19636649>) |
| C02 | Borylation HTE and regioselectivity | 1 Integrate first | CC BY 4.0 on the Zenodo data record. | [Paper / primary source](<https://www.nature.com/articles/s43588-026-01063-0>) | [Data](<https://zenodo.org/records/20773622>) | — | [License evidence](<https://zenodo.org/api/records/20773622>) |
| C03 | Cernak C–N coupling screen | 1 Integrate first | ORD data repository specifies CC BY-SA 4.0. Preserve attribution and source obligations. | [Paper / primary source](<https://doi.org/10.1021/jacs.6c05959>) | [Data](<https://open-reaction-database.org/dataset/ord_dataset-805ad863feef48579d95d86a728035f4>) | [Project](<https://github.com/open-reaction-database/ord-data>) | — |
| C04 | ORDerly condition benchmarks | 1 Integrate first | Paper states dataset CC BY 4.0. Retain upstream ORD provenance and applicable terms. | [Paper / primary source](<https://doi.org/10.1021/acs.jcim.4c00292>) | [Data](<https://figshare.com/articles/dataset/ORDerly_chemical_reactions_condition_benchmarks/23298467>) | [Project](<https://github.com/sustainable-processes/ORDerly>) | — |
| C05 | Denmark CPA thiol addition | 1 Integrate first | Published SI is accessible. Separate data reuse terms require verification. | [Paper / primary source](<https://doi.org/10.1126/science.aau5631>) | [Data](<https://pmc.ncbi.nlm.nih.gov/articles/PMC6417887/>) | — | — |
| C06 | Catechol solvent-selection benchmark | 1 Integrate first | Repository MIT. Kaggle/deposit data terms require verification. | [Paper / primary source](<https://proceedings.neurips.cc/paper_files/paper/2025/file/ad61e68e2c4208d5d092b3ecd6064b46-Paper-Datasets_and_Benchmarks_Track.pdf>) | [Data](<https://github.com/jpfolch/catechol_solvent_selection>) | [Project](<https://github.com/jpfolch/catechol_solvent_selection>) | — |
| C07 | RDB19-Rad | 1 Integrate first | CC BY 4.0 on the Zenodo data record. | [Paper / primary source](<https://doi.org/10.1021/acs.jpca.4c04121>) | [Data](<https://zenodo.org/records/11493786>) | — | [License evidence](<https://zenodo.org/api/records/11493786>) |
| C08 | Transition1x | 1 Integrate first | MIT on the Figshare data record, including the HDF5 archive. | [Paper / primary source](<https://www.nature.com/articles/s41597-022-01870-w>) | [Data](<https://figshare.com/articles/dataset/Transition1x/19614657>) | — | [License evidence](<https://api.figshare.com/v2/articles/19614657>) |
| C09 | PaRoutes 2.0 | 1 Integrate first | CC BY 4.0 on the Zenodo data record. Software Apache-2.0. | [Paper / primary source](<https://doi.org/10.1039/D2DD00015F>) | [Data](<https://zenodo.org/records/7341155>) | [Project](<https://github.com/MolecularAI/PaRoutes>) | [License evidence](<https://zenodo.org/api/records/7341155>) |
| C10 | RxnSolvKSE (Chung–Green) | 2 Targeted expansion | CC BY 4.0 on the Zenodo data record. | [Paper / primary source](<https://pubs.rsc.org/en/content/articlehtml/2024/sc/d3sc05353a>) | [Data](<https://zenodo.org/records/8423911>) | — | [License evidence](<https://zenodo.org/api/records/8423911>) |
| C11 | QuantumPioneer HAT database | 2 Targeted expansion | CC BY 4.0 on the Zenodo data record. | [Paper / primary source](<https://doi.org/10.1021/jacs.6c10371>) | [Data](<https://zenodo.org/records/20060606>) | [Project](<https://github.com/QuantumPioneer/database>) | [License evidence](<https://zenodo.org/api/records/20060606>) |
| C12 | OrgReact | 2 Targeted expansion | CC BY 4.0 on the Figshare data record. | [Paper / primary source](<https://www.nature.com/articles/s41597-026-07771-6>) | [Data](<https://doi.org/10.6084/m9.figshare.31668682>) | — | [License evidence](<https://api.figshare.com/v2/articles/31668682>) |
| C13 | EnzEngDB | 2 Targeted expansion | CC BY 4.0 on the Zenodo data record. This verifies the data deposit; the repository code LICENSE is empty. | [Paper / primary source](<https://doi.org/10.1093/nar/gkaf1142>) | [Data](<https://zenodo.org/records/17310823>) | [Project](<https://github.com/fhalab/EnzymeEngineeringDB>) | [License evidence](<https://zenodo.org/api/records/17310823>) |
| C14 | ReactZyme | 2 Targeted expansion | Repository CC0-1.0. Preserve upstream Rhea/UniProt attribution and terms. | [Paper / primary source](<https://arxiv.org/html/2408.13659v3>) | [Data](<https://github.com/WillHua127/ReactZyme/tree/main/data>) | [Project](<https://github.com/WillHua127/ReactZyme>) | — |
| C15 | CatPred-DB | 2 Targeted expansion | MIT on the Zenodo dataset record. Preserve BRENDA/SABIO-RK provenance and verify applicable upstream terms. | [Paper / primary source](<https://www.nature.com/articles/s41467-025-57215-9>) | [Data](<https://zenodo.org/records/14775076>) | [Project](<https://github.com/maranasgroup/CatPred-DB>) | [License evidence](<https://zenodo.org/api/records/14775076>) |
| C16 | OrgAIcat / iSynth | 2 Targeted expansion | Official repository claims MIT. Confirm data provenance and source-specific terms. | [Paper / primary source](<https://github.com/DeepSynthesis/orgaicat>) | [Data](<https://github.com/DeepSynthesis/orgaicat/tree/main/data>) | [Project](<https://github.com/DeepSynthesis/orgaicat>) | — |
| C17 | CHORISO | 2 Targeted expansion | CC BY 4.0 on the Figshare data record. | [Paper / primary source](<https://arxiv.org/abs/2312.09004>) | [Data](<https://figshare.com/articles/dataset/CHORISO_-_chemical_reaction_SMILES_from_academic_journals/22598230>) | [Project](<https://github.com/schwallergroup/choriso>) | [License evidence](<https://api.figshare.com/v2/articles/22598230>) |
| C18 | Perera Suzuki–Miyaura HTE | 3 Optional | Original SI accessible. Separate data reuse license not verified. | [Paper / primary source](<https://www.science.org/doi/10.1126/science.aap9112>) | [Data](<https://www.science.org/doi/10.1126/science.aap9112>) | — | — |
| C19 | Parrot USPTO-Condition | 3 Optional | Code MIT. Data-specific terms need verification. | [Paper / primary source](<https://doi.org/10.34133/research.0231>) | [Data](<https://github.com/wangxr0526/Parrot>) | [Project](<https://github.com/wangxr0526/Parrot>) | — |
| C20 | OC20NEB | 3 Optional | Project dataset specifies CC BY 4.0. | [Paper / primary source](<https://doi.org/10.1021/acscatal.4c04272>) | [Data](<https://facebookresearch.github.io/fairchem/oc20neb/>) | — | — |
| C21 | LiPRED | 3 Optional | Paper states data CC BY. Verify exact deposited version and files. | [Paper / primary source](<https://pubs.rsc.org/en/content/articlehtml/2026/cp/d6cp00088f>) | [Data](<https://doi.org/10.5281/zenodo.8396100>) | — | — |
| C22 | RMG database | 3 Optional | Open repository. Blanket data terms and each source library require verification. | [Paper / primary source](<https://pubs.acs.org/jcisd8/article/62/20/4906/884055/RMG-Database-for-Chemical-Property-Prediction>) | [Data](<https://github.com/ReactionMechanismGenerator/RMG-database>) | [Project](<https://github.com/ReactionMechanismGenerator/RMG-database>) | — |
| C23 | EasIFA SwissProt E-RXN ASA | 3 Optional | Code MIT. Preserve source data licenses and evidence levels. | [Paper / primary source](<https://www.nature.com/articles/s41467-024-51511-6>) | [Data](<https://github.com/wangxr0526/EasIFA>) | [Project](<https://github.com/wangxr0526/EasIFA>) | — |
| C24 | EnzChemRED | 3 Optional | Official SIB resource specifies CC BY 4.0. | [Paper / primary source](<https://www.nature.com/articles/s41597-024-03835-7>) | [Data](<https://zenodo.org/records/11067998>) | [Project](<https://github.com/ncbi-nlp/EnzChemRED>) | — |
| C25 | CompleteRXN | 3 Optional | MIT applies to benchmark code. Preserve original source-data licenses; no blanket MIT data grant assumed. | [Paper / primary source](<https://arxiv.org/html/2605.00222v2>) | [Data](<https://huggingface.co/datasets/completeRXN-benchmark-26/completeRXN>) | — | — |
| C26 | PMechDB / RMechDB | 3 Hold for terms | CC BY-NC-ND terms reported by official pages. Hold modified redistribution pending permission/clarification. | [Paper / primary source](<https://deeprxn.ics.uci.edu/pmechdb/download>) | [Data](<https://deeprxn.ics.uci.edu/rmechdb/howtouse>) | — | — |

## Appendix B. Task design recommendations

All 18 task designs below reproduce the final workbook’s task definitions, inputs, targets, candidate links, proposed splits, metrics, and qualifications. They are design recommendations, not claims that the source authors used these exact splits or metrics. Candidate identifiers refer to Appendix A.

| Task | Inputs | Target | Candidate IDs | Recommended split | Suggested metrics | Evaluation qualification |
| --- | --- | --- | --- | --- | --- | --- |
| Yield / conversion | Reactants, catalyst and conditions | Continuous measured yield or conversion | C01, C03, C18 | Substrate-pair and source/campaign holdouts | MAE/RMSE in percentage points; Spearman rank | Keep analytical yield, isolated yield and conversion separate. |
| Reaction success | A fully specified attempted reaction | Observed success at a declared assay threshold | C01, C02 | Substrate and source holdouts; retain failed attempts | AUPRC, balanced accuracy, MCC; calibration | A thresholded measured yield is an operational label. Missing data and sampled decoys are not failed experiments. |
| Recorded-condition recovery | Reactants and desired product | Solvent/reagent/catalyst set, and conditions where available | C04, C19 | Patent/time and reaction-family holdouts | Top-k set recall; component F1; valid condition sets | Extends existing SynRXN reagent/catalyst inference. A reported recipe is one acceptable answer, not a complete success set. |
| Measured condition ranking | Fixed substrate pair and measured candidate conditions | Ranked catalyst/condition choices by observed response | C01, C03, C18 | Entire substrate-pair and campaign holdouts | NDCG; Spearman rank; top-k yield regret | Use measured comparable panels. Recorded-condition recovery alone cannot establish which condition is best. |
| Regioselectivity | Substrate, catalyst and conditions | Reactive atom/site or ranked product sites | C02, C17 | Substrate-scaffold and reaction-family holdouts | Top-k site accuracy; symmetry-aware atom equivalence | Product annotations do not establish complete product ratios or selectivity magnitudes. |
| Enantioselectivity | Reactants, chiral catalyst and conditions | ee, enantiomer ratio or ΔΔG‡ | C05, C16, C13 | Catalyst-family, substrate-family and paper holdouts | MAE in native units; sign/dominant-enantiomer accuracy | Preserve signed/absolute ee, temperature and target definition. ΔΔG‡ derived from ratios is not a directly measured barrier. |
| Sequential optimization | History of attempted conditions and responses | Next condition or catalyst experiment | C02, C03, C06, C18 | Hold out complete campaigns or substrate pairs | Best response versus experiment budget; regret; hit count | Use only observed candidate points for tabular replay. Label surrogate-based simulations separately. |
| Condition-dependent product distribution | Substrate, solvent composition, temperature and residence time | Starting-material and product yields under each condition | C06 | Group complete campaigns; hold out solvent mixtures or condition regions | Per-product MAE; composition error; selectivity ranking | These are condition-dependent outcomes, not physical trajectories or direct rate constants. |
| Solvent effects | Reaction and specified solvent | ΔΔG‡, ΔΔH‡ or solvent-dependent ΔG‡ | C07, C10, C21 | Group all solvents/methods for each reaction | MAE/RMSE by energy type; per-reaction solvent ranking | Relative solvent corrections and absolute free-energy barriers are different endpoints. |
| Rate constants / rate laws | Reaction, catalyst where relevant, temperature and pressure | log rate, Arrhenius parameters or enzyme kcat | C11, C15, C22 | Reaction family; enzyme homology; source laboratory | Log-space MAE/RMSE; rate-curve error over valid range | Preserve units, valid T/p ranges, assay aggregation and measured/computed/estimated provenance. |
| Transition-state geometry | Reactant/product structures or reaction graph | TS coordinates and verified saddle-point structure | C08, C12, C20 | Reaction/pathway grouping with parent-source deduplication | Aligned geometry error; connectivity; TS validation success | Geometry RMSD alone is not proof of a first-order saddle point connecting the specified endpoints. |
| Reactive energies and forces | Atomic configuration with composition and context | Potential energy and atomic forces | C08 | Group all configurations from each NEB reaction pathway | Energy MAE; force MAE; pathway/barrier consistency | NEB pathways are not physical time trajectories. Keep parent pathways and shared chemistry out of different splits. |
| Multistep planning | Target molecule and fixed starting-material stock | Complete synthesis route | C09 | Reference-route reaction/patent exclusion | Solved fraction; reference top-k; diversity; model-call budget | Reference recovery is one metric. Alternative valid routes should remain admissible. |
| Enzyme–reaction retrieval | Reaction or enzyme sequence | Ranked enzymes or ranked reactions | C14 | Enzyme-homology, reaction and joint holdouts | Recall@k, MRR, per-query enrichment | Unannotated pairs are decoys rather than experimentally established negatives. |
| Enzyme mutation effects | Parent/variant sequence, reaction and assay context | Change in yield, turnover or selectivity | C13 | Entire campaign, lineage and protein-scaffold holdouts | Within-campaign rank; MAE of normalized change | Only compare measurements with compatible assay conditions and normalization. |
| Reaction-conditioned active sites | Reaction and enzyme sequence/structure | Residue labels and site types | C23 | Protein clusters plus reaction-family grouping | Residue AUPRC, F1 and MCC | Retain EC-inferred pairing evidence and overlap with ECREACT/M-CSA. |
| Literature reaction extraction | Abstract or source text | Chemical conversions, enzymes and evidence links | C24 | Article-level splits; entity and source holdouts | Entity/relation F1; identifier-normalization accuracy | This task concerns information extraction, not a large balanced reaction collection. |
| Reaction completion / balancing | Incomplete reaction equation | Missing species and balanced completed equation | C25 | Parent reaction, patent and template-family grouping | Exact/set match; atom/charge balance; accepted alternatives | Preserve template-derived provenance. Several chemically plausible completions may exist. |
