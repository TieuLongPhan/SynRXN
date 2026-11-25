# Reading notes for the SynRXN KDD submission

Read the KDD scientific-integration and software precedents together with the direct reaction competitors. The closest positive examples demonstrate the utility of integration through task design and experiments. The direct competitors constrain what SynRXN can claim as new.

These notes are a focused synthesis of selected papers and public reviewing evidence. The [catalog](catalog.md) contains the wider collection and access status; catalog inclusion alone does not mean that every methods section has received a detailed critical review. Numerical descriptions refer to the consulted versions and should be rechecked against the final published paper before being used in the manuscript.

## KDD scientific and software precedents

| Paper | Contribution relevant to SynRXN | Preparation implication |
| --- | --- | --- |
| [BatteryLife, KDD 2025](https://arxiv.org/abs/2502.18807) | Integrates heterogeneous battery datasets and evaluates models across domains and aging conditions. Its training and label-preparation choices are part of the benchmark. | Demonstrate useful cross-source evaluation and explain target construction, rather than relying on pooled size. |
| [MetamatBench, KDD 2025](https://arxiv.org/abs/2505.20299) | Combines scientific data processing, model evaluation, application-specific validity measures, and an interface. Its analysis distinguishes constraints relevant to generation from those relevant to property prediction. | Tie each validity measure to the task being claimed. A uniform resource can support different scientific evaluation meanings. |
| [HtFLlib, KDD 2025](https://arxiv.org/abs/2506.03954) | Provides a modular library and controlled comparisons across datasets, model heterogeneity, and modalities, including computational and communication costs. | Software contribution can be substantial when it enables comparable experiments and useful conclusions. |
| [BeGIN, KDD 2025](https://arxiv.org/abs/2506.12468) | Studies how different noise-generation processes affect graph learning and noise-handling methods. Matching the corruption rate does not make different noise structures equivalent. | Generated labels need a documented mechanism and representativeness analysis. A single pooled error rate can hide important source effects. |
| [DCA-Bench, KDD 2025](https://arxiv.org/abs/2406.07275) | Evaluates agents on data-curation issues through an executable pipeline. Earlier ICLR feedback asks for greater domain-specific depth. | A domain-specific scientific audit should have a defined rubric, meaningful error cases, and evidence beyond a generic detection demo. |
| [EBES, KDD 2025](https://arxiv.org/abs/2410.03399) | Standardizes event-sequence evaluation and investigates modeling choices under a common interface. | An interface should support diagnostic empirical questions. Its earlier ICLR criticism of limited novelty remains useful positioning evidence. |
| [RL4CO, KDD 2025](https://arxiv.org/abs/2306.17100) | Integrates optimization problems and RL methods with modular components and evaluation support. Earlier review history questions differentiation and presentation. | Explain the technical value of the common interface and show it through comparable experiments. The preprint predates the requested publication years; the selected KDD publication is from 2025. |
| [NMRGym, KDD 2026](https://arxiv.org/abs/2601.15763) | A recent chemistry-domain benchmark for structure elucidation from NMR information. | Define what input information each setting supplies, and compare methods within those settings. |
| [CA-DEL, KDD 2026](https://arxiv.org/abs/2605.07439) | Separates noisy screening signals from external biophysical affinity data and evaluates transfer across these measurement settings. | Keep evidence tiers and target meaning explicit; proxy labels and independent measurements are different evaluation roles. |
| [FD-Bench, KDD 2026](https://arxiv.org/abs/2505.20349) | Standardizes scientific model components and fair evaluation. The earlier ICLR meta-review questions decomposition choices and requests direct physical-validity measures. | Justify the task decomposition and evaluate the scientific property claimed. Record which paper version a criticism concerns. |
| [MultiTab, KDD 2026](https://arxiv.org/abs/2505.14312) | A suite for evaluating multiple dimensions of tabular learning. | Coverage should map to concrete evaluation questions rather than become a single pooled score. |
| [CLUBench, KDD 2026](https://arxiv.org/abs/2605.29933) | Compares clustering methods and examines tuning, embeddings, data characteristics, and method selection. | Keep representation choice, label taxonomy, tuning, and clustering evaluation distinct. Strong simple methods are essential controls. |
| [Benchmarking on Tasks That Matter, KDD 2026](https://arxiv.org/abs/2606.27997) | Investigates whether smaller dataset selections preserve model rankings, with uncertainty and domain-dependent results. | A smaller main experiment set needs a defensible selection rationale; it cannot automatically stand for all catalog datasets. |
| [Concept drift evaluation framework, KDD 2026](https://arxiv.org/abs/2606.07789) | Couples controlled changes, comparable metrics, and a held-out-dataset model-selection procedure. | The validation and tuning policy is part of the benchmark protocol. |
| [TS-Arena, KDD 2026](https://arxiv.org/abs/2512.20761) | Registers predictions before future evaluation targets exist. | Prospective evidence has a stronger temporal interpretation than repartitioning an exposed archive. SynRXN should state the independence its own data actually support. |

## Adjacent scientific benchmarks with public decision rationales

### Reaction and enzyme tasks

[ReactZyme](https://proceedings.neurips.cc/paper_files/paper/2024/hash/2e68b2367d2e0bc8dd6f0ff86e07c2eb-Abstract-Datasets_and_Benchmarks_Track.html) formulates enzyme–reaction prediction as retrieval and studies time-, homology-, and reaction-based partitions. Its positive decision rationale values that formulation and split design even though absolute model performance remains limited. This is useful precedent for an informative benchmark with substantial remaining model headroom.

[CARE](https://proceedings.neurips.cc/paper_files/paper/2024/hash/05a7ad45d75a3082d7a3a70de8743140-Abstract-Datasets_and_Benchmarks_Track.html) separates enzyme classification from reaction-based retrieval and designs evaluation difficulty through different held-out entities and classes. Its methods and decision rationale make simple baselines, test-set construction, and potential pretraining exposure important to interpretation. A model that cannot meet the intended holdout assumptions should receive a separate, qualified evaluation role.

### Data quality and chemical targets

[WelQrate](https://proceedings.neurips.cc/paper_files/paper/2024/hash/5f2f8305cd1c5be7e8319aea306388ce-Abstract-Datasets_and_Benchmarks_Track.html) uses expert-designed curation involving confirmatory and counter screens, and evaluates ranking under the low hit rates relevant to screening. Its reviewing evidence also questions an overly strong gold-standard claim. Use the concrete curation and evaluation rationale without adopting that claim uncritically.

[MassSpecGym](https://proceedings.neurips.cc/paper_files/paper/2024/hash/c6c31413d5c53b7d1c343c1498734b0f-Abstract-Datasets_and_Benchmarks_Track.html) is a precedent for multiple clearly defined chemical annotation challenges backed by data cleaning and benchmark comparisons. Its decision rationale still asks about baseline breadth and split details, showing that a valuable dataset does not make those choices self-explanatory.

[CheMixHub](https://proceedings.neurips.cc/paper_files/paper/2025/hash/2f803abdcad9de35b45d5a656dade45c-Abstract-Datasets_and_Benchmarks_Track.html) is particularly close to SynRXN's integration problem: it curates public sources into mixture-property tasks and tests generalization to mixture sizes, molecules, and temperatures. Its decision rationale values the task gap, application-motivated protocols, diverse baselines, and reproduction support. This is a stronger model for the paper's argument than a catalog-growth narrative.

[ChemX](https://proceedings.neurips.cc/paper_files/paper/2025/hash/9e08a1db869a9646418e3371b24c6ae6-Abstract-Datasets_and_Benchmarks_Track.html) contributes manually curated, cross-validated chemical references and studies extraction methods. Its relevance is the connection between expert reference construction, reproducible schemas, and measured model limitations. SynRXN's source-level corrections need comparably explicit evidence for any correctness claim.

[BOOM](https://proceedings.neurips.cc/paper_files/paper/2025/hash/b94263f7f98c9766ea9a09761ddd88ee-Abstract-Datasets_and_Benchmarks_Track.html) investigates molecular property prediction outside the training distribution and analyzes pretraining, architecture, representation, and tuning. Its decision rationale accepts a bounded dataset scope while emphasizing statistical evidence and careful positioning. Breadth alone is not a useful proxy for benchmark quality.

### Curation and reusable software

[The State of Data Curation at NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/hash/605bbd006beee7e0589a51d6a50dcae1-Abstract-Datasets_and_Benchmarks_Track.html) supplies a rubric and assessment of dataset-development practices. Its decision rationale highlights differentiation from prior work as the central issue. It also shows that an empirically grounded curation study is a legitimate contribution without requiring a new predictive architecture.

[FlexMol](https://proceedings.neurips.cc/paper_files/paper/2024/hash/3e560698f840bceef29e9566b9bd7312-Abstract-Datasets_and_Benchmarks_Track.html) and [Torch-Uncertainty](https://proceedings.neurips.cc/paper_files/paper/2025/hash/9e27bc7e3946992376f2a0c401a2e6db-Abstract-Datasets_and_Benchmarks_Track.html) provide useful software precedents. Their decision rationales value modularity, usable interfaces, method/metric coverage, and empirical support. Source and metric adapters can matter scientifically when their value is made concrete.

## Direct reaction prior work that bounds novelty

| Work | What the next paper must distinguish |
| --- | --- |
| [SynRXN, Scientific Data 2026](https://www.nature.com/articles/s41597-026-07260-w) | The published resource already includes multiple task families and evaluation infrastructure. Re-describing those capabilities is not a new contribution. |
| [ORDerly, JCIM 2024](https://doi.org/10.1021/acs.jcim.4c00292) | Reaction cleaning and condition benchmarks already exist. Demonstrate additional source interpretation or evaluation capability. Full text is also saved from the author-deposited Europe PMC article. |
| [Syntheseus, 2024 publication](https://arxiv.org/abs/2310.19796) | Fair retrosynthesis evaluation and corrections to misleading comparisons are established themes. State the distinct reaction-task and evidence issues SynRXN addresses. |
| [Chemprop, JCIM 2024](https://doi.org/10.1021/acs.jcim.3c01250) | Existing reaction-property software provides a competitive baseline family and a comparison point for reproducibility. |
| [MechFinder dataset, Scientific Data 2024](https://www.nature.com/articles/s41597-024-03709-y) | Mechanistic annotation generation is upstream work. Preserve its generation assumptions and qualification rather than treating imported mechanisms as independently observed truth. |
| [FlowER, Nature 2025](https://www.nature.com/articles/s41586-025-09426-9) | Electron-based generative mechanism prediction and conservation-aware representation already have a substantial predictor precedent. |
| [FREA, 2026 preprint](https://arxiv.org/abs/2610.06614) | Expert feasibility judgments, several candidate sources, generated training supervision, and source-dependent verifier behavior are already provided. |
| [Robust OOD prediction of Buchwald–Hartwig reactions, NCS 2026](https://doi.org/10.1038/s43588-026-01017-6) | Direct prior work for the BH data already imported by SynRXN. It integrates cohorts and evaluates OOD prediction with experimental recommendation validation. Keep BH as supporting replication and differentiate the qualified panel/context evaluation. Author-deposited full-text XML is saved locally. |
| [oMeBench, 2025 preprint](https://arxiv.org/abs/2510.07731) | Mechanism elucidation and reasoning benchmarks already exist. State how a supplied electron-event reference and scoring policy differ. |
| [TSBench, 2026 preprint](https://arxiv.org/abs/2609.08503) | Physical grounding of mechanism evaluation is already an explicit benchmark objective. Computational replay needs its own narrower interpretation. |
| [FukuyamaBench study, 2026 preprint](https://arxiv.org/abs/2607.12771) | Hierarchical mechanism reasoning and pathway comparison are active evaluation topics. |
| [MechBench study, 2026 preprint](https://arxiv.org/abs/2609.33608) | Literature-derived mechanism references and transferable prediction create a direct comparison for coverage and transfer claims. |
| [MechFind, Nature Communications 2026](https://www.nature.com/articles/s41467-026-71957-0) | Enzyme mechanism prediction based on M-CSA-derived information is prior work. Source sharing needs to be part of independence claims. |
| [VenusRX-Bench, 2026 preprint](https://arxiv.org/abs/2610.11694) | Multi-source, multi-task enzymatic reaction evaluation makes a generic unified-benchmark claim insufficiently specific. |
| [Multiple-answer retrosynthesis evaluation, ICML 2026](https://openreview.net/forum?id=ABhGgV7pov) | Alternatives to single-answer exact match exist, but a judge can introduce new biases or circularity. Keep reference alternatives justified independently. |

The [contribution ledger](../contribution_ledger.csv) and [selected study plan](../preparation_plan.md) now fix the preparation scope. Existing sources can support the new study through qualified information, a useful evaluation capability, and controlled empirical evidence. Human qualification and predictive results remain execution work.
