window.SYNRXN_CATALOG = {
  "schema_version": "1.0",
  "release": {
    "version": "1.5.0",
    "doi": null,
    "partition": "sharealike",
    "release_status": "candidate",
    "generated_at": "2026-10-15T18:48:21Z",
    "dataset_count": 2,
    "integrity": "verified"
  },
  "datasets": [
    {
      "id": "condition_sharealike/openexp",
      "task": "condition_sharealike",
      "name": "openexp",
      "split_column": "synrxn_split",
      "split_status": "proposed",
      "title": "OpenExp experimental procedure benchmark",
      "description": "OpenExp reaction-procedure pairs normalized from the exact 274,439-record OSF archive with actions and original procedure text retained.",
      "benchmark_role": "Predict structured experimental action sequences from reactants, products, catalysts, and solvents.",
      "license": "CC BY-SA 4.0",
      "license_spdx": "CC-BY-SA-4.0",
      "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
      "redistribution_partition": "sharealike",
      "citations": [
        "liu2024reactxt"
      ],
      "targets": [
        "action_sequence"
      ],
      "split_values": [
        "train",
        "validation",
        "test"
      ],
      "row_identifier": "sample_id",
      "row_identifier_unique": true,
      "rows": 274439,
      "size": 128234586,
      "sha256": "f38c6bf858aa6d6d983c15199fc90f28e999ba6ea670229ab0307e70e919edc2",
      "columns": [
        "sample_id",
        "dataset",
        "source_record_id",
        "source_index",
        "raw_reaction_smiles",
        "reaction_smiles",
        "canonical_reaction_smiles",
        "reactants",
        "products",
        "agents",
        "reagents",
        "catalysts",
        "ligands",
        "bases",
        "additives",
        "agent_1",
        "agent_2",
        "agent_3",
        "solvents",
        "solvent_1",
        "solvent_2",
        "temperature_c",
        "temperature_plausible",
        "time_hours",
        "pressure_bar",
        "atmosphere",
        "stirring",
        "flow",
        "photochemistry",
        "electrochemistry",
        "reported_yield",
        "yield_valid_0_100",
        "condition_completeness",
        "condition_mask_json",
        "conditions_json",
        "action_sequence",
        "action_score",
        "procedure_details",
        "date_of_experiment",
        "grant_date",
        "source_dataset_id",
        "upstream_split",
        "synrxn_split",
        "source",
        "citation",
        "license_spdx",
        "license_url",
        "provenance",
        "duplicate_group",
        "chemical_split_group",
        "condition_group",
        "source_metadata_json"
      ],
      "column_metadata": {
        "sample_id": {
          "role": "identifier",
          "description": "Stable SynRXN identifier for a normalized record.",
          "logical_type": "string",
          "nullable": false,
          "unit": null,
          "quantity_kind": null
        },
        "dataset": {
          "role": "provenance",
          "description": "Normalized SynRXN dataset name.",
          "logical_type": "string",
          "nullable": false,
          "unit": null,
          "quantity_kind": null
        },
        "source_record_id": {
          "role": "source_identifier",
          "description": "Identifier retained from the upstream record.",
          "logical_type": "string",
          "nullable": false,
          "unit": null,
          "quantity_kind": null
        },
        "source_index": {
          "role": "source_identifier",
          "description": "Zero-based position of the record in the upstream table.",
          "logical_type": "string",
          "nullable": false,
          "unit": null,
          "quantity_kind": null
        },
        "raw_reaction_smiles": {
          "role": "provenance",
          "description": "Reaction SMILES exactly as supplied by the upstream source.",
          "logical_type": "string",
          "nullable": true,
          "unit": null,
          "quantity_kind": null
        },
        "reaction_smiles": {
          "role": "reaction",
          "description": "Recorded or normalized reaction SMILES with reactants, optional middle-field agents, and products.",
          "logical_type": "string",
          "nullable": false,
          "unit": null,
          "quantity_kind": null
        },
        "canonical_reaction_smiles": {
          "role": "reaction",
          "description": "Canonicalized reaction SMILES used for chemical grouping when canonicalization succeeds.",
          "logical_type": "string",
          "nullable": false,
          "unit": null,
          "quantity_kind": null
        },
        "reactants": {
          "role": "reaction",
          "description": "Reactant side of the normalized reaction SMILES.",
          "logical_type": "string",
          "nullable": false,
          "unit": null,
          "quantity_kind": null
        },
        "products": {
          "role": "reaction",
          "description": "Product side of the normalized reaction SMILES.",
          "logical_type": "string",
          "nullable": false,
          "unit": null,
          "quantity_kind": null
        },
        "agents": {
          "role": "target",
          "description": "Order-invariant dot-separated set of generic ORDerly agent molecules; no catalyst or reagent subclass is inferred.",
          "logical_type": "string",
          "nullable": true,
          "unit": null,
          "quantity_kind": null
        },
        "reagents": {
          "role": "target",
          "description": "Order-invariant dot-separated set of molecules explicitly annotated or conservatively inferred as reagents.",
          "logical_type": "string",
          "nullable": true,
          "unit": null,
          "quantity_kind": null
        },
        "catalysts": {
          "role": "target",
          "description": "Order-invariant dot-separated set of explicitly annotated catalyst molecules.",
          "logical_type": "string",
          "nullable": true,
          "unit": null,
          "quantity_kind": null
        },
        "ligands": {
          "role": "target",
          "description": "Order-invariant dot-separated set of explicitly annotated ligand molecules.",
          "logical_type": "string",
          "nullable": true,
          "unit": null,
          "quantity_kind": null
        },
        "bases": {
          "role": "target",
          "description": "Order-invariant dot-separated set of explicitly annotated base molecules.",
          "logical_type": "string",
          "nullable": true,
          "unit": null,
          "quantity_kind": null
        },
        "additives": {
          "role": "target",
          "description": "Order-invariant dot-separated set of explicitly annotated additive molecules.",
          "logical_type": "string",
          "nullable": true,
          "unit": null,
          "quantity_kind": null
        },
        "agent_1": {
          "role": "target",
          "description": "First generic agent slot retained from ORDerly.",
          "logical_type": "string",
          "nullable": true,
          "unit": null,
          "quantity_kind": null
        },
        "agent_2": {
          "role": "target",
          "description": "Second generic agent slot retained from ORDerly when present.",
          "logical_type": "string",
          "nullable": true,
          "unit": null,
          "quantity_kind": null
        },
        "agent_3": {
          "role": "target",
          "description": "Third generic agent slot retained from ORDerly when present.",
          "logical_type": "string",
          "nullable": true,
          "unit": null,
          "quantity_kind": null
        },
        "solvents": {
          "role": "target",
          "description": "Order-invariant dot-separated set of solvent molecules.",
          "logical_type": "string",
          "nullable": true,
          "unit": null,
          "quantity_kind": null
        },
        "solvent_1": {
          "role": "target",
          "description": "First solvent slot retained from ORDerly.",
          "logical_type": "string",
          "nullable": true,
          "unit": null,
          "quantity_kind": null
        },
        "solvent_2": {
          "role": "target",
          "description": "Second solvent slot retained from ORDerly when present.",
          "logical_type": "string",
          "nullable": true,
          "unit": null,
          "quantity_kind": null
        },
        "temperature_c": {
          "role": "feature",
          "description": "Reaction temperature in degrees Celsius when reported.",
          "logical_type": "string",
          "nullable": true,
          "unit": null,
          "quantity_kind": null
        },
        "temperature_plausible": {
          "role": "quality",
          "description": "Whether a reported temperature is within the documented pilot range of -196 to 500 degrees Celsius.",
          "logical_type": "string",
          "nullable": true,
          "unit": null,
          "quantity_kind": null
        },
        "time_hours": {
          "role": "feature",
          "description": "Reaction duration in hours when reported.",
          "logical_type": "string",
          "nullable": true,
          "unit": null,
          "quantity_kind": null
        },
        "pressure_bar": {
          "role": "target",
          "description": "Reported reaction pressure converted to bar when the source unit is supported.",
          "logical_type": "string",
          "nullable": true,
          "unit": null,
          "quantity_kind": null
        },
        "atmosphere": {
          "role": "target",
          "description": "Reported reaction atmosphere.",
          "logical_type": "string",
          "nullable": true,
          "unit": null,
          "quantity_kind": null
        },
        "stirring": {
          "role": "target",
          "description": "Reported stirring mode.",
          "logical_type": "string",
          "nullable": true,
          "unit": null,
          "quantity_kind": null
        },
        "flow": {
          "role": "target",
          "description": "Reported flow-reactor mode.",
          "logical_type": "string",
          "nullable": true,
          "unit": null,
          "quantity_kind": null
        },
        "photochemistry": {
          "role": "target",
          "description": "Reported illumination mode for photochemical reactions.",
          "logical_type": "string",
          "nullable": true,
          "unit": null,
          "quantity_kind": null
        },
        "electrochemistry": {
          "role": "target",
          "description": "Indicator that structured electrochemical conditions were reported.",
          "logical_type": "string",
          "nullable": true,
          "unit": null,
          "quantity_kind": null
        },
        "reported_yield": {
          "role": "context",
          "description": "Optional upstream yield retained as context without clipping or treating it as the condition benchmark target.",
          "logical_type": "string",
          "nullable": true,
          "unit": null,
          "quantity_kind": null
        },
        "yield_valid_0_100": {
          "role": "quality",
          "description": "Whether the optional upstream yield lies in the inclusive 0 to 100 range.",
          "logical_type": "string",
          "nullable": true,
          "unit": null,
          "quantity_kind": null
        },
        "condition_completeness": {
          "role": "quality",
          "description": "Fraction of declared reagent, solvent, temperature, and time slots that are populated.",
          "logical_type": "string",
          "nullable": false,
          "unit": null,
          "quantity_kind": null
        },
        "condition_mask_json": {
          "role": "quality",
          "description": "JSON presence mask distinguishing populated from missing condition slots.",
          "logical_type": "string",
          "nullable": false,
          "unit": null,
          "quantity_kind": null
        },
        "conditions_json": {
          "role": "context",
          "description": "Compact JSON preserving structured operating conditions beyond the normalized scalar columns.",
          "logical_type": "string",
          "nullable": false,
          "unit": null,
          "quantity_kind": null
        },
        "action_sequence": {
          "role": "target",
          "description": "Structured experimental action sequence for procedure prediction.",
          "logical_type": "string",
          "nullable": false,
          "unit": null,
          "quantity_kind": null
        },
        "action_score": {
          "role": "quality",
          "description": "Upstream confidence score for a machine-extracted experimental action sequence.",
          "logical_type": "string",
          "nullable": false,
          "unit": null,
          "quantity_kind": null
        },
        "procedure_details": {
          "role": "provenance",
          "description": "Upstream reaction procedure text.",
          "logical_type": "string",
          "nullable": false,
          "unit": null,
          "quantity_kind": null
        },
        "date_of_experiment": {
          "role": "provenance",
          "description": "Upstream experiment date when supplied.",
          "logical_type": "string",
          "nullable": true,
          "unit": null,
          "quantity_kind": null
        },
        "grant_date": {
          "role": "partition",
          "description": "Patent grant date retained for temporal splitting.",
          "logical_type": "string",
          "nullable": true,
          "unit": null,
          "quantity_kind": null
        },
        "source_dataset_id": {
          "role": "provenance",
          "description": "ORD dataset identifier from which the ORDerly row was extracted.",
          "logical_type": "string",
          "nullable": false,
          "unit": null,
          "quantity_kind": null
        },
        "upstream_split": {
          "role": "partition",
          "description": "Published upstream partition label retained without treating it as a universal SynRXN split.",
          "logical_type": "string",
          "nullable": true,
          "unit": null,
          "quantity_kind": null
        },
        "synrxn_split": {
          "role": "partition",
          "description": "Deterministic SynRXN split grouped by normalized chemical transformation; distinct from any upstream split.",
          "logical_type": "string",
          "nullable": false,
          "unit": null,
          "quantity_kind": null
        },
        "source": {
          "role": "provenance",
          "description": "Upstream source or subset marker.",
          "logical_type": "string",
          "nullable": false,
          "unit": null,
          "quantity_kind": null
        },
        "citation": {
          "role": "provenance",
          "description": "Upstream citation label embedded in the normalized record.",
          "logical_type": "string",
          "nullable": false,
          "unit": null,
          "quantity_kind": null
        },
        "license_spdx": {
          "role": "provenance",
          "description": "SPDX identifier for the upstream dataset license.",
          "logical_type": "string",
          "nullable": false,
          "unit": null,
          "quantity_kind": null
        },
        "license_url": {
          "role": "provenance",
          "description": "URL of the upstream dataset license.",
          "logical_type": "string",
          "nullable": false,
          "unit": null,
          "quantity_kind": null
        },
        "provenance": {
          "role": "provenance",
          "description": "Human-readable description of the pinned upstream source snapshot.",
          "logical_type": "string",
          "nullable": false,
          "unit": null,
          "quantity_kind": null
        },
        "duplicate_group": {
          "role": "partition",
          "description": "Stable hash grouping records with the same normalized experiment representation.",
          "logical_type": "string",
          "nullable": false,
          "unit": null,
          "quantity_kind": null
        },
        "chemical_split_group": {
          "role": "partition",
          "description": "Stable hash grouping records by normalized chemical transformation independently of yield.",
          "logical_type": "string",
          "nullable": false,
          "unit": null,
          "quantity_kind": null
        },
        "condition_group": {
          "role": "partition",
          "description": "Grouping key for reagent, solvent, temperature, and time contexts.",
          "logical_type": "string",
          "nullable": false,
          "unit": null,
          "quantity_kind": null
        },
        "source_metadata_json": {
          "role": "provenance",
          "description": "JSON object preserving additional upstream fields and unnormalized values.",
          "logical_type": "string",
          "nullable": false,
          "unit": null,
          "quantity_kind": null
        }
      },
      "sample": [
        {
          "sample_id": "openexp:0",
          "dataset": "openexp",
          "source_record_id": "0",
          "source_index": "0",
          "raw_reaction_smiles": "",
          "reaction_smiles": "CCCCCCCCOc1ccc(C(=O)O)cc1.O=S(Cl)Cl\u003e\u003eCCCCCCCCOc1ccc(C(=O)Cl)cc1",
          "canonical_reaction_smiles": "CCCCCCCCOc1ccc(C(=O)O)cc1.O=S(Cl)Cl\u003e\u003eCCCCCCCCOc1ccc(C(=O)Cl)cc1",
          "reactants": "CCCCCCCCOc1ccc(C(=O)O)cc1.O=S(Cl)Cl",
          "products": "CCCCCCCCOc1ccc(C(=O)Cl)cc1",
          "agents": "",
          "reagents": "",
          "catalysts": "",
          "ligands": "",
          "bases": "",
          "additives": "",
          "agent_1": "",
          "agent_2": "",
          "agent_3": "",
          "solvents": "",
          "solvent_1": "",
          "solvent_2": "",
          "temperature_c": "",
          "temperature_plausible": "",
          "time_hours": "5",
          "pressure_bar": "",
          "atmosphere": "",
          "stirring": "",
          "flow": "",
          "photochemistry": "",
          "electrochemistry": "",
          "reported_yield": "",
          "yield_valid_0_100": "",
          "condition_completeness": "0.250000",
          "condition_mask_json": "{\"catalysts\":false,\"solvents\":false,\"temperature_c\":false,\"time_hours\":true}",
          "conditions_json": "{\"extracted_duration\":{\"5 hours\":\"@3@\"},\"extracted_temperature\":{}}",
          "action_sequence": "ADD $1$ (10 g) ; ADD $2$ ; ADD $1$ (10 g) ; ADD $2$ ; REFLUX for @3@ ; CONCENTRATE ; YIELD $-1$.",
          "action_score": "0.91650390625",
          "procedure_details": "A large excess of thionyl chloride was added to 10 g of the p-octyloxybenzoic acid prepared in (1), and the mixture was refluxed for 5 hours. Excessive thionyl chloride was distilled off to give a crude end compound.",
          "date_of_experiment": "",
          "grant_date": "",
          "source_dataset_id": "OpenExp.tar",
          "upstream_split": "",
          "synrxn_split": "validation",
          "source": "OpenExp",
          "citation": "10.18653/v1/2024.findings-acl.318",
          "license_spdx": "CC-BY-SA-4.0",
          "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
          "provenance": "OSF file 3dv4k; OpenExp.tar v1 (2024-06-03)",
          "duplicate_group": "experiment:44d3b0bc09941d54bd24d476339ebb825350c38c292ee3992508743132676321",
          "chemical_split_group": "reaction:4870b6a1465ef7d52da4c11eece5d5885b4b73dc7c668eb4fbf5faf612ffdf6a",
          "condition_group": "condition:c1e39570904431cdb713f4b5d8799bf4614e5abb7f6d2aa83eb8d25779ed59d0",
          "source_metadata_json": "{\"extracted_molecules\":{\"CCCCCCCCOc1ccc(C(=O)Cl)cc1\":\"$-1$\",\"CCCCCCCCOc1ccc(C(=O)O)cc1\":\"$1$\",\"O=S(Cl)Cl\":\"$2$\"},\"molecules\":{\"p-octyloxybenzoic acid\":\"CCCCCCCCOc1ccc(C(=O)O)cc1\",\"thionyl chloride\":\"O=S(Cl)Cl\"}}"
        },
        {
          "sample_id": "openexp:1",
          "dataset": "openexp",
          "source_record_id": "1",
          "source_index": "1",
          "raw_reaction_smiles": "",
          "reaction_smiles": "BrCCC1OCCCO1.O=C([O-])[O-].O=Cc1ccc(O)cc1.[I-].[K+]\u003eCC#N.CCOC(C)=O\u003eO=Cc1ccc(OCCC2OCCCO2)cc1",
          "canonical_reaction_smiles": "BrCCC1OCCCO1.O=C([O-])[O-].O=Cc1ccc(O)cc1.[I-].[K+]\u003e\u003eO=Cc1ccc(OCCC2OCCCO2)cc1",
          "reactants": "BrCCC1OCCCO1.O=C([O-])[O-].O=Cc1ccc(O)cc1.[I-].[K+]",
          "products": "O=Cc1ccc(OCCC2OCCCO2)cc1",
          "agents": "",
          "reagents": "",
          "catalysts": "",
          "ligands": "",
          "bases": "",
          "additives": "",
          "agent_1": "",
          "agent_2": "",
          "agent_3": "",
          "solvents": "CC#N.CCOC(C)=O",
          "solvent_1": "CC#N.CCOC(C)=O",
          "solvent_2": "",
          "temperature_c": "",
          "temperature_plausible": "",
          "time_hours": "",
          "pressure_bar": "",
          "atmosphere": "",
          "stirring": "",
          "flow": "",
          "photochemistry": "",
          "electrochemistry": "",
          "reported_yield": "",
          "yield_valid_0_100": "",
          "condition_completeness": "0.250000",
          "condition_mask_json": "{\"catalysts\":false,\"solvents\":true,\"temperature_c\":false,\"time_hours\":false}",
          "conditions_json": "{\"extracted_duration\":{},\"extracted_temperature\":{}}",
          "action_sequence": "ADD $1$ (120 ml) ; ADD $2$ (13.8 g) ; ADD $3$ (21.1 g) ; ADD $4$ (1.8 g) ; ADD $5$ (35.8 g) ; ADD $6$ (200 ml) ; WASH with $5$ ; WASH with water ; WASH with sodium chloride ; COLLECTLAYER organic ; DRYSOLUTION over magnesium sulfate ; FILTER keep filtrate ; CONCENTRATE ; YIELD $-1$ (25.6 g).",
          "action_score": "0.487548828125",
          "procedure_details": "In a 500 ml flask, 120 ml of acetonitrile, 13.8 g of 4-hydroxybenzaldehyde, 21.1 g of 2-(2-bromoethyl)-1,3-dioxane, 1.8 g of potassium iodide, and 35.8 g of potassium carbonate were placed and then, reacted at 80° C. for 5 hrs. Thereafter, 200 ml of ethyl acetate was added to the reaction mixture and washed once with …",
          "date_of_experiment": "",
          "grant_date": "",
          "source_dataset_id": "OpenExp.tar",
          "upstream_split": "",
          "synrxn_split": "train",
          "source": "OpenExp",
          "citation": "10.18653/v1/2024.findings-acl.318",
          "license_spdx": "CC-BY-SA-4.0",
          "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
          "provenance": "OSF file 3dv4k; OpenExp.tar v1 (2024-06-03)",
          "duplicate_group": "experiment:b89754c7ad1fa9909597958cf770c52498ce29acbf84c4bd297e0af1e1efafd3",
          "chemical_split_group": "reaction:66fe1c07523cc051131e277cd5faa4e5713c20587427344846bb39fc3d8b2aa6",
          "condition_group": "condition:c25c0111657d0f53208e2f51fdd43f2212a1a4486eb4ce770137d2f9428b3384",
          "source_metadata_json": "{\"extracted_molecules\":{\"BrCCC1OCCCO1\":\"$3$\",\"CC#N\":\"$1$\",\"CCOC(C)=O\":\"$6$\",\"O=C([O-])[O-].[K+].[K+]\":\"$5$\",\"O=Cc1ccc(O)cc1\":\"$2$\",\"O=Cc1ccc(OCCC2OCCCO2)cc1\":\"$-1$\",\"[I-].[K+]\":\"$4$\"},\"molecules\":{\"2-(2-bromoethyl)-1,3-dioxane\":\"BrCCC1OCCCO1\",\"4-hydroxybenzaldehyde\":\"O=Cc1ccc(O)cc1\",\"K2CO3\":\"O=C([O-])[O-].[K+].[K+]\",\"…"
        },
        {
          "sample_id": "openexp:2",
          "dataset": "openexp",
          "source_record_id": "2",
          "source_index": "2",
          "raw_reaction_smiles": "",
          "reaction_smiles": "CC(C)(C)[O-].O=Cc1ccc(OCCC2OCCCO2)cc1.[K+]\u003eCCOCC.C[P+](c1ccccc1)(c1ccccc1)c1ccccc1.[Br-]\u003eC=Cc1ccc(OCCC2OCCCO2)cc1",
          "canonical_reaction_smiles": "CC(C)(C)[O-].O=Cc1ccc(OCCC2OCCCO2)cc1.[K+]\u003e\u003eC=Cc1ccc(OCCC2OCCCO2)cc1",
          "reactants": "CC(C)(C)[O-].O=Cc1ccc(OCCC2OCCCO2)cc1.[K+]",
          "products": "C=Cc1ccc(OCCC2OCCCO2)cc1",
          "agents": "C[P+](c1ccccc1)(c1ccccc1)c1ccccc1.[Br-]",
          "reagents": "",
          "catalysts": "C[P+](c1ccccc1)(c1ccccc1)c1ccccc1.[Br-]",
          "ligands": "",
          "bases": "",
          "additives": "",
          "agent_1": "C[P+](c1ccccc1)(c1ccccc1)c1ccccc1.[Br-]",
          "agent_2": "",
          "agent_3": "",
          "solvents": "CCOCC",
          "solvent_1": "CCOCC",
          "solvent_2": "",
          "temperature_c": "",
          "temperature_plausible": "",
          "time_hours": "",
          "pressure_bar": "",
          "atmosphere": "",
          "stirring": "",
          "flow": "",
          "photochemistry": "",
          "electrochemistry": "",
          "reported_yield": "",
          "yield_valid_0_100": "",
          "condition_completeness": "0.500000",
          "condition_mask_json": "{\"catalysts\":true,\"solvents\":true,\"temperature_c\":false,\"time_hours\":false}",
          "conditions_json": "{\"extracted_duration\":{},\"extracted_temperature\":{}}",
          "action_sequence": "ADD $1$ (2 liter, 57.2 g) ; ADD $2$ (500 ml) ; ADD $2$ ; STIR ; ADD $3$ (18.0 g) ; ADD $4$ (25.6 g) ; FILTER ; WASH with water ; WASH with sodium chloride ; WASH with water ; WASH with sodium chloride ; DRYSOLUTION over magnesium sulfate ; FILTER keep filtrate ; CONCENTRATE ; YIELD $-1$ (19.7 g).",
          "action_score": "0.54541015625",
          "procedure_details": "In another 2 liter flask, 57.2 g of methyl triphenylphosphonium bromide was placed. 500 ml of anhydrous ether was poured and then, stirred. To the resulting solution was slowly added 18.0 g of potassium tert-butoxide and reacted at room temperature for 3 hrs. 25.6 g of the above 4-(2-[1,3]-dioxane-2-yl-ethoxy)benzalde…",
          "date_of_experiment": "",
          "grant_date": "",
          "source_dataset_id": "OpenExp.tar",
          "upstream_split": "",
          "synrxn_split": "train",
          "source": "OpenExp",
          "citation": "10.18653/v1/2024.findings-acl.318",
          "license_spdx": "CC-BY-SA-4.0",
          "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
          "provenance": "OSF file 3dv4k; OpenExp.tar v1 (2024-06-03)",
          "duplicate_group": "experiment:a09f3a97510a5398de473dbbd0cd34099183ad62c1c819847b63c2c27cc70b80",
          "chemical_split_group": "reaction:51bbb99e0d58f26f65a36c4fd7e319b3263bf7cc6c5ea4a4712288a88749e673",
          "condition_group": "condition:b8b05b0bee6890c0bb7dfef9e10d85310425caf1c630fe178b68a6600be42726",
          "source_metadata_json": "{\"extracted_molecules\":{\"C=Cc1ccc(OCCC2OCCCO2)cc1\":\"$-1$\",\"CC(C)(C)[O-].[K+]\":\"$3$\",\"CCOCC\":\"$2$\",\"C[P+](c1ccccc1)(c1ccccc1)c1ccccc1.[Br-]\":\"$1$\",\"O=Cc1ccc(OCCC2OCCCO2)cc1\":\"$4$\"},\"molecules\":{\"4-(2-[1,3]-dioxane-2-yl-ethoxy)benzaldehyde\":\"O=Cc1ccc(OCCC2OCCCO2)cc1\",\"ether\":\"CCOCC\",\"methyl triphenylphosphonium bromide\"…"
        },
        {
          "sample_id": "openexp:3",
          "dataset": "openexp",
          "source_record_id": "3",
          "source_index": "3",
          "raw_reaction_smiles": "",
          "reaction_smiles": "CC(C)(C)[O-].O=Cc1ccc(OCCC2OCCO2)cc1.[K+]\u003eCCOCC.C[P+](c1ccccc1)(c1ccccc1)c1ccccc1.[Br-]\u003eC=Cc1ccc(OCCC2OCCO2)cc1",
          "canonical_reaction_smiles": "CC(C)(C)[O-].O=Cc1ccc(OCCC2OCCO2)cc1.[K+]\u003e\u003eC=Cc1ccc(OCCC2OCCO2)cc1",
          "reactants": "CC(C)(C)[O-].O=Cc1ccc(OCCC2OCCO2)cc1.[K+]",
          "products": "C=Cc1ccc(OCCC2OCCO2)cc1",
          "agents": "C[P+](c1ccccc1)(c1ccccc1)c1ccccc1.[Br-]",
          "reagents": "",
          "catalysts": "C[P+](c1ccccc1)(c1ccccc1)c1ccccc1.[Br-]",
          "ligands": "",
          "bases": "",
          "additives": "",
          "agent_1": "C[P+](c1ccccc1)(c1ccccc1)c1ccccc1.[Br-]",
          "agent_2": "",
          "agent_3": "",
          "solvents": "CCOCC",
          "solvent_1": "CCOCC",
          "solvent_2": "",
          "temperature_c": "",
          "temperature_plausible": "",
          "time_hours": "",
          "pressure_bar": "",
          "atmosphere": "",
          "stirring": "",
          "flow": "",
          "photochemistry": "",
          "electrochemistry": "",
          "reported_yield": "",
          "yield_valid_0_100": "",
          "condition_completeness": "0.500000",
          "condition_mask_json": "{\"catalysts\":true,\"solvents\":true,\"temperature_c\":false,\"time_hours\":false}",
          "conditions_json": "{\"extracted_duration\":{},\"extracted_temperature\":{}}",
          "action_sequence": "ADD $1$ (2 liter, 57.2 g) ; ADD $2$ (500 ml) ; ADD $2$ ; STIR ; ADD $3$ (18.0 g) ; ADD $4$ (23.7 g) ; FILTER ; WASH with water ; WASH with sodium chloride ; WASH with water ; WASH with sodium chloride ; DRYSOLUTION over magnesium sulfate ; FILTER keep filtrate ; CONCENTRATE ; YIELD $-1$ (20.8 g).",
          "action_score": "0.52734375",
          "procedure_details": "In another 2 liter flask, 57.2 g of methyl triphenylphosphonium bromide was placed. 500 ml of anhydrous ether was poured and then, stirred. To the resulting solution was slowly added 18.0 g of potassium tert-butoxide and reacted at room temperature for 3 hrs. 23.7 g of the above 4-(2-[1,3]-dioxolan-2-yl-ethoxy)benzald…",
          "date_of_experiment": "",
          "grant_date": "",
          "source_dataset_id": "OpenExp.tar",
          "upstream_split": "",
          "synrxn_split": "train",
          "source": "OpenExp",
          "citation": "10.18653/v1/2024.findings-acl.318",
          "license_spdx": "CC-BY-SA-4.0",
          "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
          "provenance": "OSF file 3dv4k; OpenExp.tar v1 (2024-06-03)",
          "duplicate_group": "experiment:8a91cb04e0093c4d15beb53d43900c5566e42bc20cbc3b1e88df09eb2196d7a2",
          "chemical_split_group": "reaction:503aeb8d4b6ecfe6d005da7122e323f4feecc3c84107013b9e34c4280a2e833b",
          "condition_group": "condition:b8b05b0bee6890c0bb7dfef9e10d85310425caf1c630fe178b68a6600be42726",
          "source_metadata_json": "{\"extracted_molecules\":{\"C=Cc1ccc(OCCC2OCCO2)cc1\":\"$-1$\",\"CC(C)(C)[O-].[K+]\":\"$3$\",\"CCOCC\":\"$2$\",\"C[P+](c1ccccc1)(c1ccccc1)c1ccccc1.[Br-]\":\"$1$\",\"O=Cc1ccc(OCCC2OCCO2)cc1\":\"$4$\"},\"molecules\":{\"4-(2-[1,3]-dioxolan-2-yl-ethoxy)benzaldehyde\":\"O=Cc1ccc(OCCC2OCCO2)cc1\",\"ether\":\"CCOCC\",\"methyl triphenylphosphonium bromide\":\"…"
        },
        {
          "sample_id": "openexp:4",
          "dataset": "openexp",
          "source_record_id": "4",
          "source_index": "4",
          "raw_reaction_smiles": "",
          "reaction_smiles": "COC(=O)Cl.Cn1cccc1S(=O)(=O)NC(C)(C)C.[Li][CH2]CCC\u003eC1CCOC1.CCOC(C)=O\u003eCOC(=O)c1ccn(C)c1S(=O)(=O)NC(C)(C)C",
          "canonical_reaction_smiles": "COC(=O)Cl.Cn1cccc1S(=O)(=O)NC(C)(C)C.[Li][CH2]CCC\u003e\u003eCOC(=O)c1ccn(C)c1S(=O)(=O)NC(C)(C)C",
          "reactants": "COC(=O)Cl.Cn1cccc1S(=O)(=O)NC(C)(C)C.[Li][CH2]CCC",
          "products": "COC(=O)c1ccn(C)c1S(=O)(=O)NC(C)(C)C",
          "agents": "",
          "reagents": "",
          "catalysts": "",
          "ligands": "",
          "bases": "",
          "additives": "",
          "agent_1": "",
          "agent_2": "",
          "agent_3": "",
          "solvents": "C1CCOC1.CCOC(C)=O",
          "solvent_1": "C1CCOC1.CCOC(C)=O",
          "solvent_2": "",
          "temperature_c": "-70",
          "temperature_plausible": "true",
          "time_hours": "",
          "pressure_bar": "",
          "atmosphere": "",
          "stirring": "",
          "flow": "",
          "photochemistry": "",
          "electrochemistry": "",
          "reported_yield": "",
          "yield_valid_0_100": "",
          "condition_completeness": "0.500000",
          "condition_mask_json": "{\"catalysts\":false,\"solvents\":true,\"temperature_c\":true,\"time_hours\":false}",
          "conditions_json": "{\"extracted_duration\":{\"1 h\":\"@9@\",\"overnight\":\"@10@\"},\"extracted_temperature\":{\"-10 \\u00b0C\":\"#7#\",\"-57 \\u00b0C\":\"#8#\",\"-70 \\u00b0C\":\"#6#\"}}",
          "action_sequence": "MAKESOLUTION with $1$ and $2$ (50 mL) ; SETTEMPERATURE #6# ; ADD $3$ (13 mL, 32.5 mmol, 2.5M) ; STIR for @9@ at #7# ; MAKESOLUTION with $4$ and $2$ (6 mL) ; STIR for @10@ at #8# ; ADD $5$ (200 mL) ; WASH with HCl (1 normal) ; WASH with brine ; DRYSOLUTION over Na2SO4 ; FILTER keep filtrate ; CONCENTRATE ; YIELD $-1$ (…",
          "action_score": "0.40673828125",
          "procedure_details": "To a solution of 2.5 g (11.5 mmol) of N-(1,1-dimethylethyl)-1-methyl-1H-pyrrole-2-sulfonamide in 50 mL THF under nitrogen atmosphere cooled to below -70° C. was added dropwise 13 mL (32.5 mmol) of 2.5M n-butyllithium at such a rate as to keep the temperature below -58° C. After stirring at -20° C. to -10° C. for 30 mi…",
          "date_of_experiment": "",
          "grant_date": "",
          "source_dataset_id": "OpenExp.tar",
          "upstream_split": "",
          "synrxn_split": "train",
          "source": "OpenExp",
          "citation": "10.18653/v1/2024.findings-acl.318",
          "license_spdx": "CC-BY-SA-4.0",
          "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
          "provenance": "OSF file 3dv4k; OpenExp.tar v1 (2024-06-03)",
          "duplicate_group": "experiment:0a8d728ca32d17118b0f1cb35d6fc0a14690ebb6ffaa04984559469264599f57",
          "chemical_split_group": "reaction:62a82f1fb5e8899ce64761d63ecffd7d2c9e0b287a553b0cb6d194ec6d4c8d21",
          "condition_group": "condition:701ae6c4868b12fb46267b22ef40b9a8f0512f55e1176eeb3046f65265713ef0",
          "source_metadata_json": "{\"extracted_molecules\":{\"C1CCOC1\":\"$2$\",\"CCOC(C)=O\":\"$5$\",\"COC(=O)Cl\":\"$4$\",\"COC(=O)c1ccn(C)c1S(=O)(=O)NC(C)(C)C\":\"$-1$\",\"Cn1cccc1S(=O)(=O)NC(C)(C)C\":\"$1$\",\"[Li]CCCC\":\"$3$\"},\"molecules\":{\"HCl\":\"Cl\",\"N-(1,1-dimethylethyl)-1-methyl-1H-pyrrole-2-sulfonamide\":\"Cn1cccc1S(=O)(=O)NC(C)(C)C\",\"THF\":\"C1CCOC1\",\"brine\":\"O.[Cl-].[…"
        }
      ],
      "split_counts": {
        "test": 27118,
        "train": 219665,
        "validation": 27656
      },
      "null_counts": {
        "sample_id": 0,
        "dataset": 0,
        "source_record_id": 0,
        "source_index": 0,
        "raw_reaction_smiles": 274439,
        "reaction_smiles": 0,
        "canonical_reaction_smiles": 0,
        "reactants": 0,
        "products": 0,
        "agents": 244779,
        "reagents": 274439,
        "catalysts": 244779,
        "ligands": 274439,
        "bases": 274439,
        "additives": 274439,
        "agent_1": 244779,
        "agent_2": 274439,
        "agent_3": 274439,
        "solvents": 22471,
        "solvent_1": 22471,
        "solvent_2": 274439,
        "temperature_c": 142018,
        "temperature_plausible": 142018,
        "time_hours": 116217,
        "pressure_bar": 274439,
        "atmosphere": 274439,
        "stirring": 274439,
        "flow": 274439,
        "photochemistry": 274439,
        "electrochemistry": 274439,
        "reported_yield": 274439,
        "yield_valid_0_100": 274439,
        "condition_completeness": 0,
        "condition_mask_json": 0,
        "conditions_json": 0,
        "action_sequence": 0,
        "action_score": 0,
        "procedure_details": 0,
        "date_of_experiment": 274439,
        "grant_date": 274439,
        "source_dataset_id": 0,
        "upstream_split": 274439,
        "synrxn_split": 0,
        "source": 0,
        "citation": 0,
        "license_spdx": 0,
        "license_url": 0,
        "provenance": 0,
        "duplicate_group": 0,
        "chemical_split_group": 0,
        "condition_group": 0,
        "source_metadata_json": 0
      },
      "target_summaries": {
        "action_sequence": {
          "kind": "categorical",
          "count": 274439,
          "unique": 270943,
          "top": [
            {
              "value": "ADD $1$ (0.16 mmol) ; ADD $2$ (2 eq, 0.3 mmol) ; ADD $3$ (3 mL) ; MAKESOLUTION with $4$ and $5$ (4M, 3 drops) ; MICROWAVE for @7@ at #6# ; CONCENTRATE ; ADD TFA (3 mL) ; MICROWAVE for @7@ at #6# ; CONCENTRATE ; DRYSOLUTION over unknown ; FILTER keep filtrate ; YIELD $-1$.",
              "count": 272
            },
            {
              "value": "ADD $1$ (0.16 mmol) ; ADD $2$ (0.3 mmol, 2 eq) ; ADD $3$ (3 mL) ; MAKESOLUTION with $4$ and $5$ (4M, 3 drops) ; MICROWAVE for @7@ at #6# ; CONCENTRATE ; ADD TFA (3 mL) ; MICROWAVE for @7@ at #6# ; CONCENTRATE ; DRYSOLUTION over unknown ; FILTER keep filtrate ; YIELD $-1$.",
              "count": 234
            },
            {
              "value": "ADD $1$ (0.16 mmol) ; ADD $2$ (2 eq, 0.3 mmol) ; ADD $3$ (3 mL) ; MAKESOLUTION with $4$ and $5$ (4M, 3 drops) ; ADD $3$ (3 mL) ; MAKESOLUTION with $4$ and $5$ (4M, 3 drops) ; MICROWAVE for @7@ at #6# ; CONCENTRATE ; ADD TFA (3 mL) ; MICROWAVE for @7@ at #6# ; CONCENTRATE ; DRYSOLUTION over unknown ; FILTER keep filtrate ; YIELD $-1$.",
              "count": 47
            },
            {
              "value": "ADD $1$ (0.16 mmol) ; ADD $2$ (0.3 mmol 2 eq) ; ADD $3$ (3 mL) ; MAKESOLUTION with $4$ and $5$ (4M, 3 drops) ; MICROWAVE for @7@ at #6# ; CONCENTRATE ; ADD TFA (3 mL) ; MICROWAVE for @7@ at #6# ; CONCENTRATE ; DRYSOLUTION over unknown ; FILTER keep filtrate ; YIELD $-1$.",
              "count": 46
            },
            {
              "value": "ADD $1$ ; ADD $2$ ; ADD $3$ ; ADD $4$ ; YIELD $-1$.",
              "count": 19
            },
            {
              "value": "ADD $1$ ; ADD $2$ ; ADD $3$ ; ADD $4$ ; STIR for @6@ at #5# ; YIELD $-1$.",
              "count": 17
            },
            {
              "value": "MAKESOLUTION with $1$ (40 mg, 0.09 mmol) and $2$ (1 mL) ; ADD $3$ (0.27 mmol) ; ADD $4$ (63 mg, 0.45 mmol) ; STIR for @8@ at #7# ; ADD $5$ at #7# ; ADD $6$ at #7# ; FILTER ; WASH with $5$ ; CONCENTRATE ; YIELD $-1$.",
              "count": 15
            },
            {
              "value": "ADD $1$ ; ADD $2$ ; ADD $3$ ; REFLUX ; YIELD $-1$.",
              "count": 15
            }
          ]
        }
      },
      "reaction_column": "reaction_smiles",
      "reaction_text": "CCCCCCCCOc1ccc(C(=O)O)cc1.O=S(Cl)Cl\u003e\u003eCCCCCCCCOc1ccc(C(=O)Cl)cc1",
      "depiction": "_static/catalog-sharealike-reactions/condition_sharealike-openexp.svg"
    },
    {
      "id": "condition_sharealike/ord_hte_condition",
      "task": "condition_sharealike",
      "name": "ord_hte_condition",
      "split_column": "synrxn_split",
      "split_status": "proposed",
      "title": "ORD high-throughput condition benchmark",
      "description": "Sixteen non-overlapping experimental ORD cohorts with role-aware inputs and structured operating conditions; 92 incomplete NiCOlit records are quarantined.",
      "benchmark_role": "Evaluate condition-set prediction and condition-aware ranking across reaction families, HTE platforms, and operating modes.",
      "license": "CC BY-SA 4.0",
      "license_spdx": "CC-BY-SA-4.0",
      "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
      "redistribution_partition": "sharealike",
      "citations": [
        "kearnes2021ord"
      ],
      "targets": [
        "reagents",
        "catalysts",
        "ligands",
        "bases",
        "additives",
        "solvents",
        "temperature_c",
        "time_hours",
        "pressure_bar"
      ],
      "split_values": [
        "train",
        "validation",
        "test"
      ],
      "row_identifier": "sample_id",
      "row_identifier_unique": true,
      "rows": 123720,
      "size": 24004331,
      "sha256": "9d77cb0ef9233dd7922e4a65ea89e046051581c9200dd3eb7c85d3ae06514c5c",
      "columns": [
        "sample_id",
        "dataset",
        "source_record_id",
        "source_index",
        "raw_reaction_smiles",
        "reaction_smiles",
        "canonical_reaction_smiles",
        "reactants",
        "products",
        "agents",
        "reagents",
        "catalysts",
        "ligands",
        "bases",
        "additives",
        "agent_1",
        "agent_2",
        "agent_3",
        "solvents",
        "solvent_1",
        "solvent_2",
        "temperature_c",
        "temperature_plausible",
        "time_hours",
        "pressure_bar",
        "atmosphere",
        "stirring",
        "flow",
        "photochemistry",
        "electrochemistry",
        "reported_yield",
        "yield_valid_0_100",
        "condition_completeness",
        "condition_mask_json",
        "conditions_json",
        "action_sequence",
        "action_score",
        "procedure_details",
        "date_of_experiment",
        "grant_date",
        "source_dataset_id",
        "upstream_split",
        "synrxn_split",
        "source",
        "citation",
        "license_spdx",
        "license_url",
        "provenance",
        "duplicate_group",
        "chemical_split_group",
        "condition_group",
        "source_metadata_json"
      ],
      "column_metadata": {
        "sample_id": {
          "role": "identifier",
          "description": "Stable SynRXN identifier for a normalized record.",
          "logical_type": "string",
          "nullable": false,
          "unit": null,
          "quantity_kind": null
        },
        "dataset": {
          "role": "provenance",
          "description": "Normalized SynRXN dataset name.",
          "logical_type": "string",
          "nullable": false,
          "unit": null,
          "quantity_kind": null
        },
        "source_record_id": {
          "role": "source_identifier",
          "description": "Identifier retained from the upstream record.",
          "logical_type": "string",
          "nullable": false,
          "unit": null,
          "quantity_kind": null
        },
        "source_index": {
          "role": "source_identifier",
          "description": "Zero-based position of the record in the upstream table.",
          "logical_type": "string",
          "nullable": false,
          "unit": null,
          "quantity_kind": null
        },
        "raw_reaction_smiles": {
          "role": "provenance",
          "description": "Reaction SMILES exactly as supplied by the upstream source.",
          "logical_type": "string",
          "nullable": true,
          "unit": null,
          "quantity_kind": null
        },
        "reaction_smiles": {
          "role": "reaction",
          "description": "Recorded or normalized reaction SMILES with reactants, optional middle-field agents, and products.",
          "logical_type": "string",
          "nullable": false,
          "unit": null,
          "quantity_kind": null
        },
        "canonical_reaction_smiles": {
          "role": "reaction",
          "description": "Canonicalized reaction SMILES used for chemical grouping when canonicalization succeeds.",
          "logical_type": "string",
          "nullable": false,
          "unit": null,
          "quantity_kind": null
        },
        "reactants": {
          "role": "reaction",
          "description": "Reactant side of the normalized reaction SMILES.",
          "logical_type": "string",
          "nullable": false,
          "unit": null,
          "quantity_kind": null
        },
        "products": {
          "role": "reaction",
          "description": "Product side of the normalized reaction SMILES.",
          "logical_type": "string",
          "nullable": false,
          "unit": null,
          "quantity_kind": null
        },
        "agents": {
          "role": "target",
          "description": "Order-invariant dot-separated set of generic ORDerly agent molecules; no catalyst or reagent subclass is inferred.",
          "logical_type": "string",
          "nullable": true,
          "unit": null,
          "quantity_kind": null
        },
        "reagents": {
          "role": "target",
          "description": "Order-invariant dot-separated set of molecules explicitly annotated or conservatively inferred as reagents.",
          "logical_type": "string",
          "nullable": true,
          "unit": null,
          "quantity_kind": null
        },
        "catalysts": {
          "role": "target",
          "description": "Order-invariant dot-separated set of explicitly annotated catalyst molecules.",
          "logical_type": "string",
          "nullable": true,
          "unit": null,
          "quantity_kind": null
        },
        "ligands": {
          "role": "target",
          "description": "Order-invariant dot-separated set of explicitly annotated ligand molecules.",
          "logical_type": "string",
          "nullable": true,
          "unit": null,
          "quantity_kind": null
        },
        "bases": {
          "role": "target",
          "description": "Order-invariant dot-separated set of explicitly annotated base molecules.",
          "logical_type": "string",
          "nullable": true,
          "unit": null,
          "quantity_kind": null
        },
        "additives": {
          "role": "target",
          "description": "Order-invariant dot-separated set of explicitly annotated additive molecules.",
          "logical_type": "string",
          "nullable": true,
          "unit": null,
          "quantity_kind": null
        },
        "agent_1": {
          "role": "target",
          "description": "First generic agent slot retained from ORDerly.",
          "logical_type": "string",
          "nullable": true,
          "unit": null,
          "quantity_kind": null
        },
        "agent_2": {
          "role": "target",
          "description": "Second generic agent slot retained from ORDerly when present.",
          "logical_type": "string",
          "nullable": true,
          "unit": null,
          "quantity_kind": null
        },
        "agent_3": {
          "role": "target",
          "description": "Third generic agent slot retained from ORDerly when present.",
          "logical_type": "string",
          "nullable": true,
          "unit": null,
          "quantity_kind": null
        },
        "solvents": {
          "role": "target",
          "description": "Order-invariant dot-separated set of solvent molecules.",
          "logical_type": "string",
          "nullable": true,
          "unit": null,
          "quantity_kind": null
        },
        "solvent_1": {
          "role": "target",
          "description": "First solvent slot retained from ORDerly.",
          "logical_type": "string",
          "nullable": true,
          "unit": null,
          "quantity_kind": null
        },
        "solvent_2": {
          "role": "target",
          "description": "Second solvent slot retained from ORDerly when present.",
          "logical_type": "string",
          "nullable": true,
          "unit": null,
          "quantity_kind": null
        },
        "temperature_c": {
          "role": "feature",
          "description": "Reaction temperature in degrees Celsius when reported.",
          "logical_type": "number",
          "nullable": true,
          "unit": null,
          "quantity_kind": null
        },
        "temperature_plausible": {
          "role": "quality",
          "description": "Whether a reported temperature is within the documented pilot range of -196 to 500 degrees Celsius.",
          "logical_type": "string",
          "nullable": true,
          "unit": null,
          "quantity_kind": null
        },
        "time_hours": {
          "role": "feature",
          "description": "Reaction duration in hours when reported.",
          "logical_type": "number",
          "nullable": false,
          "unit": null,
          "quantity_kind": null
        },
        "pressure_bar": {
          "role": "target",
          "description": "Reported reaction pressure converted to bar when the source unit is supported.",
          "logical_type": "number",
          "nullable": true,
          "unit": null,
          "quantity_kind": null
        },
        "atmosphere": {
          "role": "target",
          "description": "Reported reaction atmosphere.",
          "logical_type": "string",
          "nullable": true,
          "unit": null,
          "quantity_kind": null
        },
        "stirring": {
          "role": "target",
          "description": "Reported stirring mode.",
          "logical_type": "string",
          "nullable": true,
          "unit": null,
          "quantity_kind": null
        },
        "flow": {
          "role": "target",
          "description": "Reported flow-reactor mode.",
          "logical_type": "string",
          "nullable": true,
          "unit": null,
          "quantity_kind": null
        },
        "photochemistry": {
          "role": "target",
          "description": "Reported illumination mode for photochemical reactions.",
          "logical_type": "string",
          "nullable": true,
          "unit": null,
          "quantity_kind": null
        },
        "electrochemistry": {
          "role": "target",
          "description": "Indicator that structured electrochemical conditions were reported.",
          "logical_type": "string",
          "nullable": true,
          "unit": null,
          "quantity_kind": null
        },
        "reported_yield": {
          "role": "context",
          "description": "Optional upstream yield retained as context without clipping or treating it as the condition benchmark target.",
          "logical_type": "string",
          "nullable": true,
          "unit": null,
          "quantity_kind": null
        },
        "yield_valid_0_100": {
          "role": "quality",
          "description": "Whether the optional upstream yield lies in the inclusive 0 to 100 range.",
          "logical_type": "string",
          "nullable": true,
          "unit": null,
          "quantity_kind": null
        },
        "condition_completeness": {
          "role": "quality",
          "description": "Fraction of declared reagent, solvent, temperature, and time slots that are populated.",
          "logical_type": "string",
          "nullable": false,
          "unit": null,
          "quantity_kind": null
        },
        "condition_mask_json": {
          "role": "quality",
          "description": "JSON presence mask distinguishing populated from missing condition slots.",
          "logical_type": "string",
          "nullable": false,
          "unit": null,
          "quantity_kind": null
        },
        "conditions_json": {
          "role": "context",
          "description": "Compact JSON preserving structured operating conditions beyond the normalized scalar columns.",
          "logical_type": "string",
          "nullable": false,
          "unit": null,
          "quantity_kind": null
        },
        "action_sequence": {
          "role": "target",
          "description": "Structured experimental action sequence for procedure prediction.",
          "logical_type": "string",
          "nullable": true,
          "unit": null,
          "quantity_kind": null
        },
        "action_score": {
          "role": "quality",
          "description": "Upstream confidence score for a machine-extracted experimental action sequence.",
          "logical_type": "string",
          "nullable": true,
          "unit": null,
          "quantity_kind": null
        },
        "procedure_details": {
          "role": "provenance",
          "description": "Upstream reaction procedure text.",
          "logical_type": "string",
          "nullable": true,
          "unit": null,
          "quantity_kind": null
        },
        "date_of_experiment": {
          "role": "provenance",
          "description": "Upstream experiment date when supplied.",
          "logical_type": "string",
          "nullable": true,
          "unit": null,
          "quantity_kind": null
        },
        "grant_date": {
          "role": "partition",
          "description": "Patent grant date retained for temporal splitting.",
          "logical_type": "string",
          "nullable": true,
          "unit": null,
          "quantity_kind": null
        },
        "source_dataset_id": {
          "role": "provenance",
          "description": "ORD dataset identifier from which the ORDerly row was extracted.",
          "logical_type": "string",
          "nullable": false,
          "unit": null,
          "quantity_kind": null
        },
        "upstream_split": {
          "role": "partition",
          "description": "Published upstream partition label retained without treating it as a universal SynRXN split.",
          "logical_type": "string",
          "nullable": true,
          "unit": null,
          "quantity_kind": null
        },
        "synrxn_split": {
          "role": "partition",
          "description": "Deterministic SynRXN split grouped by normalized chemical transformation; distinct from any upstream split.",
          "logical_type": "string",
          "nullable": false,
          "unit": null,
          "quantity_kind": null
        },
        "source": {
          "role": "provenance",
          "description": "Upstream source or subset marker.",
          "logical_type": "string",
          "nullable": false,
          "unit": null,
          "quantity_kind": null
        },
        "citation": {
          "role": "provenance",
          "description": "Upstream citation label embedded in the normalized record.",
          "logical_type": "string",
          "nullable": false,
          "unit": null,
          "quantity_kind": null
        },
        "license_spdx": {
          "role": "provenance",
          "description": "SPDX identifier for the upstream dataset license.",
          "logical_type": "string",
          "nullable": false,
          "unit": null,
          "quantity_kind": null
        },
        "license_url": {
          "role": "provenance",
          "description": "URL of the upstream dataset license.",
          "logical_type": "string",
          "nullable": false,
          "unit": null,
          "quantity_kind": null
        },
        "provenance": {
          "role": "provenance",
          "description": "Human-readable description of the pinned upstream source snapshot.",
          "logical_type": "string",
          "nullable": false,
          "unit": null,
          "quantity_kind": null
        },
        "duplicate_group": {
          "role": "partition",
          "description": "Stable hash grouping records with the same normalized experiment representation.",
          "logical_type": "string",
          "nullable": false,
          "unit": null,
          "quantity_kind": null
        },
        "chemical_split_group": {
          "role": "partition",
          "description": "Stable hash grouping records by normalized chemical transformation independently of yield.",
          "logical_type": "string",
          "nullable": false,
          "unit": null,
          "quantity_kind": null
        },
        "condition_group": {
          "role": "partition",
          "description": "Grouping key for reagent, solvent, temperature, and time contexts.",
          "logical_type": "string",
          "nullable": false,
          "unit": null,
          "quantity_kind": null
        },
        "source_metadata_json": {
          "role": "provenance",
          "description": "JSON object preserving additional upstream fields and unnormalized values.",
          "logical_type": "string",
          "nullable": false,
          "unit": null,
          "quantity_kind": null
        }
      },
      "sample": [
        {
          "sample_id": "ord_hte_condition:ord-1179bffb972a424c9764eeebfd97ceb7",
          "dataset": "ord_hte_condition",
          "source_record_id": "ord-1179bffb972a424c9764eeebfd97ceb7",
          "source_index": "0",
          "raw_reaction_smiles": "",
          "reaction_smiles": "C1COCCN1.FC(F)(F)c1ccc(I)cc1\u003eCC(C)(C)[O-].CS(C)=O.[K+]\u003eCn1c(=O)c2c(ncn2C)n(C)c1=O.FC(F)(F)c1ccc(N2CCOCC2)cc1.FC(F)(F)c1cccc(N2CCOCC2)c1",
          "canonical_reaction_smiles": "C1COCCN1.FC(F)(F)c1ccc(I)cc1\u003e\u003eCn1c(=O)c2c(ncn2C)n(C)c1=O.FC(F)(F)c1ccc(N2CCOCC2)cc1.FC(F)(F)c1cccc(N2CCOCC2)c1",
          "reactants": "C1COCCN1.FC(F)(F)c1ccc(I)cc1",
          "products": "Cn1c(=O)c2c(ncn2C)n(C)c1=O.FC(F)(F)c1ccc(N2CCOCC2)cc1.FC(F)(F)c1cccc(N2CCOCC2)c1",
          "agents": "CC(C)(C)[O-].[K+]",
          "reagents": "",
          "catalysts": "",
          "ligands": "",
          "bases": "CC(C)(C)[O-].[K+]",
          "additives": "",
          "agent_1": "",
          "agent_2": "",
          "agent_3": "CC(C)(C)[O-].[K+]",
          "solvents": "CS(C)=O",
          "solvent_1": "CS(C)=O",
          "solvent_2": "",
          "temperature_c": "80",
          "temperature_plausible": "true",
          "time_hours": "18",
          "pressure_bar": "",
          "atmosphere": "NITROGEN",
          "stirring": "AGITATION",
          "flow": "",
          "photochemistry": "",
          "electrochemistry": "",
          "reported_yield": "0",
          "yield_valid_0_100": "true",
          "condition_completeness": "0.500000",
          "condition_mask_json": "{\"additives\":false,\"bases\":true,\"catalysts\":false,\"ligands\":false,\"reagents\":false,\"solvents\":true,\"temperature_c\":true,\"time_hours\":true}",
          "conditions_json": "{\"pressure\":{\"atmosphere\":{\"details\":\"dry nitrogen\",\"type\":\"NITROGEN\"},\"control\":{\"type\":\"SEALED\"}},\"stirring\":{\"details\":\"Eppendorf ThermoMixer\\u00ae C\",\"rate\":{\"rpm\":800},\"type\":\"AGITATION\"},\"temperature\":{\"control\":{\"details\":\"Eppendorf ThermoMixer\\u00ae C\",\"type\":\"DRY_ALUMINUM_PLATE\"},\"setpoint\":{\"units\":\"CELSIUS\"…",
          "action_sequence": "",
          "action_score": "",
          "procedure_details": "",
          "date_of_experiment": "",
          "grant_date": "",
          "source_dataset_id": "ord_dataset-805ad863feef48579d95d86a728035f4",
          "upstream_split": "",
          "synrxn_split": "train",
          "source": "Open Reaction Database",
          "citation": "10.1021/jacs.1c09820",
          "license_spdx": "CC-BY-SA-4.0",
          "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
          "provenance": "ord-data@05fe1a62cb36d6bc3c07592ac1bd4ff1a4d8e43c; Hugging Face parquet config ord_dataset-805ad863feef48579d95d86a728035f4",
          "duplicate_group": "experiment:92af00fbb675bbb38ff14fdd8fd425f214d8c53e2fcf1941bbd6a2840664edf0",
          "chemical_split_group": "reaction:6eb3c28325115dcd19c35a6ea405580e276703242c0caed46061fd542b3696fd",
          "condition_group": "condition:a4f96037d81f5bb6e24334c866eba4244a267c19cdc89f1622252f3aa80aa516",
          "source_metadata_json": "{\"identifiers\":[{\"type\":\"REACTION_TYPE\",\"value\":\"C\\u2013N coupling\"},{\"details\":\"Sample Name_x\",\"type\":\"CUSTOM\",\"value\":\"JD-03-11_80DEG_PLATE3_384\"},{\"details\":\"coded recipe\",\"type\":\"CUSTOM\",\"value\":\"L48.W.C8.W.B3.W.E1.W.N1.W\"},{\"details\":\"experiment\",\"type\":\"CUSTOM\",\"value\":\"JD-03-11\"}],\"provenance\":{\"city\":\"Ann Arbo…"
        },
        {
          "sample_id": "ord_hte_condition:ord-038c865303d64ba494c7d78db7ba5aff",
          "dataset": "ord_hte_condition",
          "source_record_id": "ord-038c865303d64ba494c7d78db7ba5aff",
          "source_index": "1",
          "raw_reaction_smiles": "",
          "reaction_smiles": "C1COCCN1.FC(F)(F)c1ccc(I)cc1\u003eCS(C)=O.C[Si](C)(C)[O-].[Na+]\u003eCn1c(=O)c2c(ncn2C)n(C)c1=O.FC(F)(F)c1ccc(N2CCOCC2)cc1.FC(F)(F)c1cccc(N2CCOCC2)c1",
          "canonical_reaction_smiles": "C1COCCN1.FC(F)(F)c1ccc(I)cc1\u003e\u003eCn1c(=O)c2c(ncn2C)n(C)c1=O.FC(F)(F)c1ccc(N2CCOCC2)cc1.FC(F)(F)c1cccc(N2CCOCC2)c1",
          "reactants": "C1COCCN1.FC(F)(F)c1ccc(I)cc1",
          "products": "Cn1c(=O)c2c(ncn2C)n(C)c1=O.FC(F)(F)c1ccc(N2CCOCC2)cc1.FC(F)(F)c1cccc(N2CCOCC2)c1",
          "agents": "C[Si](C)(C)[O-].[Na+]",
          "reagents": "",
          "catalysts": "",
          "ligands": "",
          "bases": "C[Si](C)(C)[O-].[Na+]",
          "additives": "",
          "agent_1": "",
          "agent_2": "",
          "agent_3": "C[Si](C)(C)[O-].[Na+]",
          "solvents": "CS(C)=O",
          "solvent_1": "CS(C)=O",
          "solvent_2": "",
          "temperature_c": "80",
          "temperature_plausible": "true",
          "time_hours": "18",
          "pressure_bar": "",
          "atmosphere": "NITROGEN",
          "stirring": "AGITATION",
          "flow": "",
          "photochemistry": "",
          "electrochemistry": "",
          "reported_yield": "0",
          "yield_valid_0_100": "true",
          "condition_completeness": "0.500000",
          "condition_mask_json": "{\"additives\":false,\"bases\":true,\"catalysts\":false,\"ligands\":false,\"reagents\":false,\"solvents\":true,\"temperature_c\":true,\"time_hours\":true}",
          "conditions_json": "{\"pressure\":{\"atmosphere\":{\"details\":\"dry nitrogen\",\"type\":\"NITROGEN\"},\"control\":{\"type\":\"SEALED\"}},\"stirring\":{\"details\":\"Eppendorf ThermoMixer\\u00ae C\",\"rate\":{\"rpm\":800},\"type\":\"AGITATION\"},\"temperature\":{\"control\":{\"details\":\"Eppendorf ThermoMixer\\u00ae C\",\"type\":\"DRY_ALUMINUM_PLATE\"},\"setpoint\":{\"units\":\"CELSIUS\"…",
          "action_sequence": "",
          "action_score": "",
          "procedure_details": "",
          "date_of_experiment": "",
          "grant_date": "",
          "source_dataset_id": "ord_dataset-805ad863feef48579d95d86a728035f4",
          "upstream_split": "",
          "synrxn_split": "train",
          "source": "Open Reaction Database",
          "citation": "10.1021/jacs.1c09820",
          "license_spdx": "CC-BY-SA-4.0",
          "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
          "provenance": "ord-data@05fe1a62cb36d6bc3c07592ac1bd4ff1a4d8e43c; Hugging Face parquet config ord_dataset-805ad863feef48579d95d86a728035f4",
          "duplicate_group": "experiment:bedc203e8a9474f0520cc32f59d0a0b8a2df85dbe09eee36d1bedef75b3ee9dc",
          "chemical_split_group": "reaction:6eb3c28325115dcd19c35a6ea405580e276703242c0caed46061fd542b3696fd",
          "condition_group": "condition:9e7f6d2c60d852795d7fb311bbeefd6f0b6e2cdebed4143a9c58bfcbb2f7aaeb",
          "source_metadata_json": "{\"identifiers\":[{\"type\":\"REACTION_TYPE\",\"value\":\"C\\u2013N coupling\"},{\"details\":\"Sample Name_x\",\"type\":\"CUSTOM\",\"value\":\"JD-03-11_80DEG_PLATE3_383\"},{\"details\":\"coded recipe\",\"type\":\"CUSTOM\",\"value\":\"L48.W.C8.W.B1.W.E1.W.N1.W\"},{\"details\":\"experiment\",\"type\":\"CUSTOM\",\"value\":\"JD-03-11\"}],\"provenance\":{\"city\":\"Ann Arbo…"
        },
        {
          "sample_id": "ord_hte_condition:ord-09e4698e8d14466fa617ce6752a39b7a",
          "dataset": "ord_hte_condition",
          "source_record_id": "ord-09e4698e8d14466fa617ce6752a39b7a",
          "source_index": "2",
          "raw_reaction_smiles": "",
          "reaction_smiles": "C1COCCN1.FC(F)(F)c1ccc(I)cc1\u003eCC(C)(C)[O-].CS(C)=O.[Cl][Cu].[K+]\u003eCn1c(=O)c2c(ncn2C)n(C)c1=O.FC(F)(F)c1ccc(N2CCOCC2)cc1.FC(F)(F)c1cccc(N2CCOCC2)c1",
          "canonical_reaction_smiles": "C1COCCN1.FC(F)(F)c1ccc(I)cc1\u003e\u003eCn1c(=O)c2c(ncn2C)n(C)c1=O.FC(F)(F)c1ccc(N2CCOCC2)cc1.FC(F)(F)c1cccc(N2CCOCC2)c1",
          "reactants": "C1COCCN1.FC(F)(F)c1ccc(I)cc1",
          "products": "Cn1c(=O)c2c(ncn2C)n(C)c1=O.FC(F)(F)c1ccc(N2CCOCC2)cc1.FC(F)(F)c1cccc(N2CCOCC2)c1",
          "agents": "CC(C)(C)[O-].[Cl][Cu].[K+]",
          "reagents": "",
          "catalysts": "[Cl][Cu]",
          "ligands": "",
          "bases": "CC(C)(C)[O-].[K+]",
          "additives": "",
          "agent_1": "[Cl][Cu]",
          "agent_2": "",
          "agent_3": "CC(C)(C)[O-].[K+]",
          "solvents": "CS(C)=O",
          "solvent_1": "CS(C)=O",
          "solvent_2": "",
          "temperature_c": "80",
          "temperature_plausible": "true",
          "time_hours": "18",
          "pressure_bar": "",
          "atmosphere": "NITROGEN",
          "stirring": "AGITATION",
          "flow": "",
          "photochemistry": "",
          "electrochemistry": "",
          "reported_yield": "4.12659",
          "yield_valid_0_100": "true",
          "condition_completeness": "0.625000",
          "condition_mask_json": "{\"additives\":false,\"bases\":true,\"catalysts\":true,\"ligands\":false,\"reagents\":false,\"solvents\":true,\"temperature_c\":true,\"time_hours\":true}",
          "conditions_json": "{\"pressure\":{\"atmosphere\":{\"details\":\"dry nitrogen\",\"type\":\"NITROGEN\"},\"control\":{\"type\":\"SEALED\"}},\"stirring\":{\"details\":\"Eppendorf ThermoMixer\\u00ae C\",\"rate\":{\"rpm\":800},\"type\":\"AGITATION\"},\"temperature\":{\"control\":{\"details\":\"Eppendorf ThermoMixer\\u00ae C\",\"type\":\"DRY_ALUMINUM_PLATE\"},\"setpoint\":{\"units\":\"CELSIUS\"…",
          "action_sequence": "",
          "action_score": "",
          "procedure_details": "",
          "date_of_experiment": "",
          "grant_date": "",
          "source_dataset_id": "ord_dataset-805ad863feef48579d95d86a728035f4",
          "upstream_split": "",
          "synrxn_split": "train",
          "source": "Open Reaction Database",
          "citation": "10.1021/jacs.1c09820",
          "license_spdx": "CC-BY-SA-4.0",
          "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
          "provenance": "ord-data@05fe1a62cb36d6bc3c07592ac1bd4ff1a4d8e43c; Hugging Face parquet config ord_dataset-805ad863feef48579d95d86a728035f4",
          "duplicate_group": "experiment:38c9bc7a660e14e55588eaab65cfa3be4d4e9b3be16de35237a70a0c3239d557",
          "chemical_split_group": "reaction:6eb3c28325115dcd19c35a6ea405580e276703242c0caed46061fd542b3696fd",
          "condition_group": "condition:9815480368b6af4b2fa56866d830c735df1de88548daa48b6d3cc95f27f21d1d",
          "source_metadata_json": "{\"identifiers\":[{\"type\":\"REACTION_TYPE\",\"value\":\"C\\u2013N coupling\"},{\"details\":\"Sample Name_x\",\"type\":\"CUSTOM\",\"value\":\"JD-03-11_80DEG_PLATE3_382\"},{\"details\":\"coded recipe\",\"type\":\"CUSTOM\",\"value\":\"L48.W.C7.W.B3.W.E1.W.N1.W\"},{\"details\":\"experiment\",\"type\":\"CUSTOM\",\"value\":\"JD-03-11\"}],\"provenance\":{\"city\":\"Ann Arbo…"
        },
        {
          "sample_id": "ord_hte_condition:ord-49995a8769a54dae91895c60357ab463",
          "dataset": "ord_hte_condition",
          "source_record_id": "ord-49995a8769a54dae91895c60357ab463",
          "source_index": "3",
          "raw_reaction_smiles": "",
          "reaction_smiles": "C1COCCN1.FC(F)(F)c1ccc(I)cc1\u003eCS(C)=O.C[Si](C)(C)[O-].[Cl][Cu].[Na+]\u003eCn1c(=O)c2c(ncn2C)n(C)c1=O.FC(F)(F)c1ccc(N2CCOCC2)cc1.FC(F)(F)c1cccc(N2CCOCC2)c1",
          "canonical_reaction_smiles": "C1COCCN1.FC(F)(F)c1ccc(I)cc1\u003e\u003eCn1c(=O)c2c(ncn2C)n(C)c1=O.FC(F)(F)c1ccc(N2CCOCC2)cc1.FC(F)(F)c1cccc(N2CCOCC2)c1",
          "reactants": "C1COCCN1.FC(F)(F)c1ccc(I)cc1",
          "products": "Cn1c(=O)c2c(ncn2C)n(C)c1=O.FC(F)(F)c1ccc(N2CCOCC2)cc1.FC(F)(F)c1cccc(N2CCOCC2)c1",
          "agents": "C[Si](C)(C)[O-].[Cl][Cu].[Na+]",
          "reagents": "",
          "catalysts": "[Cl][Cu]",
          "ligands": "",
          "bases": "C[Si](C)(C)[O-].[Na+]",
          "additives": "",
          "agent_1": "[Cl][Cu]",
          "agent_2": "",
          "agent_3": "C[Si](C)(C)[O-].[Na+]",
          "solvents": "CS(C)=O",
          "solvent_1": "CS(C)=O",
          "solvent_2": "",
          "temperature_c": "80",
          "temperature_plausible": "true",
          "time_hours": "18",
          "pressure_bar": "",
          "atmosphere": "NITROGEN",
          "stirring": "AGITATION",
          "flow": "",
          "photochemistry": "",
          "electrochemistry": "",
          "reported_yield": "0",
          "yield_valid_0_100": "true",
          "condition_completeness": "0.625000",
          "condition_mask_json": "{\"additives\":false,\"bases\":true,\"catalysts\":true,\"ligands\":false,\"reagents\":false,\"solvents\":true,\"temperature_c\":true,\"time_hours\":true}",
          "conditions_json": "{\"pressure\":{\"atmosphere\":{\"details\":\"dry nitrogen\",\"type\":\"NITROGEN\"},\"control\":{\"type\":\"SEALED\"}},\"stirring\":{\"details\":\"Eppendorf ThermoMixer\\u00ae C\",\"rate\":{\"rpm\":800},\"type\":\"AGITATION\"},\"temperature\":{\"control\":{\"details\":\"Eppendorf ThermoMixer\\u00ae C\",\"type\":\"DRY_ALUMINUM_PLATE\"},\"setpoint\":{\"units\":\"CELSIUS\"…",
          "action_sequence": "",
          "action_score": "",
          "procedure_details": "",
          "date_of_experiment": "",
          "grant_date": "",
          "source_dataset_id": "ord_dataset-805ad863feef48579d95d86a728035f4",
          "upstream_split": "",
          "synrxn_split": "train",
          "source": "Open Reaction Database",
          "citation": "10.1021/jacs.1c09820",
          "license_spdx": "CC-BY-SA-4.0",
          "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
          "provenance": "ord-data@05fe1a62cb36d6bc3c07592ac1bd4ff1a4d8e43c; Hugging Face parquet config ord_dataset-805ad863feef48579d95d86a728035f4",
          "duplicate_group": "experiment:8547649d89a0c1abfec3f7e13a126224361f213532c7a7a5ba9cb6ccf92cc80b",
          "chemical_split_group": "reaction:6eb3c28325115dcd19c35a6ea405580e276703242c0caed46061fd542b3696fd",
          "condition_group": "condition:6a681a822d0ec855e3f6950bd22acceb10db8039b7b1c4fb3b332a947fb02ca0",
          "source_metadata_json": "{\"identifiers\":[{\"type\":\"REACTION_TYPE\",\"value\":\"C\\u2013N coupling\"},{\"details\":\"Sample Name_x\",\"type\":\"CUSTOM\",\"value\":\"JD-03-11_80DEG_PLATE3_381\"},{\"details\":\"coded recipe\",\"type\":\"CUSTOM\",\"value\":\"L48.W.C7.W.B1.W.E1.W.N1.W\"},{\"details\":\"experiment\",\"type\":\"CUSTOM\",\"value\":\"JD-03-11\"}],\"provenance\":{\"city\":\"Ann Arbo…"
        },
        {
          "sample_id": "ord_hte_condition:ord-93d94d267e8b45d18269391f614ccd49",
          "dataset": "ord_hte_condition",
          "source_record_id": "ord-93d94d267e8b45d18269391f614ccd49",
          "source_index": "4",
          "raw_reaction_smiles": "",
          "reaction_smiles": "C1COCCN1.FC(F)(F)c1ccc(I)cc1\u003eCC(C)(C)[O-].CS(C)=O.Cc1cc(C)c(N2C=CN(c3c(C)cc(C)cc3C)[CH]2[Cu][Cl])c(C)c1.[K+]\u003eCn1c(=O)c2c(ncn2C)n(C)c1=O.FC(F)(F)c1ccc(N2CCOCC2)cc1.FC(F)(F)c1cccc(N2CCOCC2)c1",
          "canonical_reaction_smiles": "C1COCCN1.FC(F)(F)c1ccc(I)cc1\u003e\u003eCn1c(=O)c2c(ncn2C)n(C)c1=O.FC(F)(F)c1ccc(N2CCOCC2)cc1.FC(F)(F)c1cccc(N2CCOCC2)c1",
          "reactants": "C1COCCN1.FC(F)(F)c1ccc(I)cc1",
          "products": "Cn1c(=O)c2c(ncn2C)n(C)c1=O.FC(F)(F)c1ccc(N2CCOCC2)cc1.FC(F)(F)c1cccc(N2CCOCC2)c1",
          "agents": "CC(C)(C)[O-].Cc1cc(C)c(N2C=CN(c3c(C)cc(C)cc3C)[CH]2[Cu][Cl])c(C)c1.[K+]",
          "reagents": "",
          "catalysts": "Cc1cc(C)c(N2C=CN(c3c(C)cc(C)cc3C)[CH]2[Cu][Cl])c(C)c1",
          "ligands": "",
          "bases": "CC(C)(C)[O-].[K+]",
          "additives": "",
          "agent_1": "Cc1cc(C)c(N2C=CN(c3c(C)cc(C)cc3C)[CH]2[Cu][Cl])c(C)c1",
          "agent_2": "",
          "agent_3": "CC(C)(C)[O-].[K+]",
          "solvents": "CS(C)=O",
          "solvent_1": "CS(C)=O",
          "solvent_2": "",
          "temperature_c": "80",
          "temperature_plausible": "true",
          "time_hours": "18",
          "pressure_bar": "",
          "atmosphere": "NITROGEN",
          "stirring": "AGITATION",
          "flow": "",
          "photochemistry": "",
          "electrochemistry": "",
          "reported_yield": "5.089624",
          "yield_valid_0_100": "true",
          "condition_completeness": "0.625000",
          "condition_mask_json": "{\"additives\":false,\"bases\":true,\"catalysts\":true,\"ligands\":false,\"reagents\":false,\"solvents\":true,\"temperature_c\":true,\"time_hours\":true}",
          "conditions_json": "{\"pressure\":{\"atmosphere\":{\"details\":\"dry nitrogen\",\"type\":\"NITROGEN\"},\"control\":{\"type\":\"SEALED\"}},\"stirring\":{\"details\":\"Eppendorf ThermoMixer\\u00ae C\",\"rate\":{\"rpm\":800},\"type\":\"AGITATION\"},\"temperature\":{\"control\":{\"details\":\"Eppendorf ThermoMixer\\u00ae C\",\"type\":\"DRY_ALUMINUM_PLATE\"},\"setpoint\":{\"units\":\"CELSIUS\"…",
          "action_sequence": "",
          "action_score": "",
          "procedure_details": "",
          "date_of_experiment": "",
          "grant_date": "",
          "source_dataset_id": "ord_dataset-805ad863feef48579d95d86a728035f4",
          "upstream_split": "",
          "synrxn_split": "train",
          "source": "Open Reaction Database",
          "citation": "10.1021/jacs.1c09820",
          "license_spdx": "CC-BY-SA-4.0",
          "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
          "provenance": "ord-data@05fe1a62cb36d6bc3c07592ac1bd4ff1a4d8e43c; Hugging Face parquet config ord_dataset-805ad863feef48579d95d86a728035f4",
          "duplicate_group": "experiment:8a43e4c829293c46c3f81f98e927bfe3194e8a5e3c53efcf805d01b139a43d56",
          "chemical_split_group": "reaction:6eb3c28325115dcd19c35a6ea405580e276703242c0caed46061fd542b3696fd",
          "condition_group": "condition:99dd73f6bdf44748adabfb03a34b7125982575cd473ee797fd24fcda3b1ac4b5",
          "source_metadata_json": "{\"identifiers\":[{\"type\":\"REACTION_TYPE\",\"value\":\"C\\u2013N coupling\"},{\"details\":\"Sample Name_x\",\"type\":\"CUSTOM\",\"value\":\"JD-03-11_80DEG_PLATE3_380\"},{\"details\":\"coded recipe\",\"type\":\"CUSTOM\",\"value\":\"L48.W.C6.W.B3.W.E1.W.N1.W\"},{\"details\":\"experiment\",\"type\":\"CUSTOM\",\"value\":\"JD-03-11\"}],\"provenance\":{\"city\":\"Ann Arbo…"
        }
      ],
      "split_counts": {
        "test": 6555,
        "train": 109859,
        "validation": 7306
      },
      "null_counts": {
        "sample_id": 0,
        "dataset": 0,
        "source_record_id": 0,
        "source_index": 0,
        "raw_reaction_smiles": 75275,
        "reaction_smiles": 0,
        "canonical_reaction_smiles": 0,
        "reactants": 0,
        "products": 0,
        "agents": 2897,
        "reagents": 72991,
        "catalysts": 60538,
        "ligands": 70054,
        "bases": 20015,
        "additives": 99925,
        "agent_1": 60538,
        "agent_2": 72991,
        "agent_3": 12430,
        "solvents": 1670,
        "solvent_1": 1670,
        "solvent_2": 123720,
        "temperature_c": 4128,
        "temperature_plausible": 4128,
        "time_hours": 0,
        "pressure_bar": 122493,
        "atmosphere": 13409,
        "stirring": 10281,
        "flow": 122493,
        "photochemistry": 119410,
        "electrochemistry": 123720,
        "reported_yield": 8380,
        "yield_valid_0_100": 8380,
        "condition_completeness": 0,
        "condition_mask_json": 0,
        "conditions_json": 0,
        "action_sequence": 123720,
        "action_score": 123720,
        "procedure_details": 122493,
        "date_of_experiment": 123720,
        "grant_date": 123720,
        "source_dataset_id": 0,
        "upstream_split": 123720,
        "synrxn_split": 0,
        "source": 0,
        "citation": 0,
        "license_spdx": 0,
        "license_url": 0,
        "provenance": 0,
        "duplicate_group": 0,
        "chemical_split_group": 0,
        "condition_group": 0,
        "source_metadata_json": 0
      },
      "target_summaries": {
        "reagents": {
          "kind": "categorical",
          "count": 50729,
          "unique": 49,
          "top": [
            {
              "value": "CCN=C=NCCCN(C)C.Cl",
              "count": 9991
            },
            {
              "value": "C(=NC1CCCCC1)=NC1CCCCC1",
              "count": 8065
            },
            {
              "value": "CC(C)N=C=NC(C)C",
              "count": 2795
            },
            {
              "value": "CN(C)C(On1nnc2cccnc21)=[N+](C)C.F[P-](F)(F)(F)(F)F",
              "count": 1712
            },
            {
              "value": "F[P-](F)(F)(F)(F)F.c1cnc2c(c1)nnn2O[P+](N1CCCC1)(N1CCCC1)N1CCCC1",
              "count": 1700
            },
            {
              "value": "F[P-](F)(F)(F)(F)F.c1ccc2c(c1)nnn2O[P+](N1CCCC1)(N1CCCC1)N1CCCC1",
              "count": 1296
            },
            {
              "value": "CC(=O)O.CC(=O)O[BH-](OC(C)=O)OC(C)=O.CC(C)[O-].[Na+].[Ti+4]",
              "count": 1152
            },
            {
              "value": "CCN(C(C)C)C(C)C",
              "count": 1152
            }
          ]
        },
        "catalysts": {
          "kind": "categorical",
          "count": 63182,
          "unique": 1356,
          "top": [
            {
              "value": "[Cu][I]",
              "count": 4800
            },
            {
              "value": "CC(=O)[O][Pd][O]C(C)=O",
              "count": 4416
            },
            {
              "value": "[Cl][Cu]",
              "count": 4032
            },
            {
              "value": "[Cu][Br]",
              "count": 2880
            },
            {
              "value": "C[O]1CC[O](C)-\u003e[Ni]\u003c-1([Br])[Br]",
              "count": 2880
            },
            {
              "value": "CC(=O)[O-].[Cu+2]",
              "count": 2792
            },
            {
              "value": "CC(C)(C)c1ccc(C=[CH](c2ccc(C(C)(C)C)cc2)-\u003e[Ni](\u003c-[CH](=Cc2ccc(C(C)(C)C)cc2)c2ccc(C(C)(C)C)cc2)\u003c-[CH](=Cc2ccc(C(C)(C)C)cc2)c2ccc(C(C)(C)C)cc2)cc1",
              "count": 2688
            },
            {
              "value": "CN(C)CCN(C)C.[Cl-].[Cu+2].[OH-]",
              "count": 2280
            }
          ]
        },
        "ligands": {
          "kind": "categorical",
          "count": 53666,
          "unique": 279,
          "top": [
            {
              "value": "CN(C)c1ccccc1-c1ccccc1P(C1CCCCC1)C1CCCCC1",
              "count": 864
            },
            {
              "value": "c1ccc(-c2ccccc2P(C2CCCCC2)C2CCCCC2)cc1",
              "count": 800
            },
            {
              "value": "COc1cccc(OC)c1NC(=O)C(=O)O",
              "count": 800
            },
            {
              "value": "Cc1cccc(C)c1NC(=O)C(=O)O",
              "count": 800
            },
            {
              "value": "CC(C)c1cccc(C(C)C)c1-n1cc[n+](-c2c(C(C)C)cccc2C(C)C)c1.[Cl-]",
              "count": 800
            },
            {
              "value": "O=C(O)C(=O)Nc1cc(C(F)(F)F)cc(C(F)(F)F)c1",
              "count": 736
            },
            {
              "value": "COc1ccc(OC)c(P(C2CCCCC2)C2CCCCC2)c1-c1c(C(C)C)cc(C(C)C)cc1C(C)C",
              "count": 736
            },
            {
              "value": "CC1(C)c2cccc(P(c3ccccc3)c3ccccc3)c2Oc2c(P(c3ccccc3)c3ccccc3)cccc21",
              "count": 704
            }
          ]
        },
        "bases": {
          "kind": "categorical",
          "count": 103705,
          "unique": 38,
          "top": [
            {
              "value": "CCN(C(C)C)C(C)C",
              "count": 25938
            },
            {
              "value": "CC(C)(C)[O-].[Na+]",
              "count": 11168
            },
            {
              "value": "CN1CCOCC1",
              "count": 7797
            },
            {
              "value": "C[Si](C)(C)[O-].[Na+]",
              "count": 7712
            },
            {
              "value": "O=P([O-])([O-])[O-].[K+]",
              "count": 6943
            },
            {
              "value": "CC(C)(C)[O-].[K+]",
              "count": 6560
            },
            {
              "value": "CCN(CC)CC",
              "count": 4834
            },
            {
              "value": "CCN=P(N=P(N(C)C)(N(C)C)N(C)C)(N(C)C)N(C)C",
              "count": 3184
            }
          ]
        },
        "additives": {
          "kind": "categorical",
          "count": 23795,
          "unique": 10,
          "top": [
            {
              "value": "On1nnc2ccccc21",
              "count": 5988
            },
            {
              "value": "On1nnc2cccnc21",
              "count": 3195
            },
            {
              "value": "Oc1c(F)c(F)c(F)c(F)c1F",
              "count": 3157
            },
            {
              "value": "O=C1c2ccccc2C(=O)N1O",
              "count": 2649
            },
            {
              "value": "O=C1CCC(=O)N1O",
              "count": 2581
            },
            {
              "value": "CCOC(=O)C(C#N)=NO",
              "count": 1680
            },
            {
              "value": "On1nnc2ccc(Cl)cc21",
              "count": 1645
            },
            {
              "value": "Oc1cc(Cl)c(Cl)cc1Cl",
              "count": 1537
            }
          ]
        },
        "solvents": {
          "kind": "categorical",
          "count": 122050,
          "unique": 32,
          "top": [
            {
              "value": "CN(C)C=O",
              "count": 48853
            },
            {
              "value": "CS(C)=O",
              "count": 32448
            },
            {
              "value": "COCCOCCOC",
              "count": 15360
            },
            {
              "value": "CS(C)=O.Cc1cccc(C)c1",
              "count": 3072
            },
            {
              "value": "COCCOCCOC.CS(C)=O",
              "count": 3072
            },
            {
              "value": "ClCCCl",
              "count": 2792
            },
            {
              "value": "CC#N",
              "count": 2696
            },
            {
              "value": "CO",
              "count": 2376
            }
          ]
        },
        "temperature_c": {
          "kind": "numeric",
          "count": 119592,
          "min": 0.0,
          "max": 225.0,
          "mean": 42.72374448642049
        },
        "time_hours": {
          "kind": "numeric",
          "count": 123720,
          "min": 0.0333503166667,
          "max": 96.0,
          "mean": 11.66142671862081
        },
        "pressure_bar": {
          "kind": "numeric",
          "count": 1227,
          "min": 55.0,
          "max": 55.0,
          "mean": 55.0
        }
      },
      "reaction_column": "reaction_smiles",
      "reaction_text": "C1COCCN1.FC(F)(F)c1ccc(I)cc1\u003eCC(C)(C)[O-].CS(C)=O.[K+]\u003eCn1c(=O)c2c(ncn2C)n(C)c1=O.FC(F)(F)c1ccc(N2CCOCC2)cc1.FC(F)(F)c1cccc(N2CCOCC2)c1",
      "depiction": "_static/catalog-sharealike-reactions/condition_sharealike-ord_hte_condition.svg"
    }
  ]
}
