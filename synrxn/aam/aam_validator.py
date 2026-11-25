import networkx as nx
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional
from rdkit import Chem
from operator import eq
from joblib import Parallel, delayed
from networkx.algorithms.isomorphism import generic_node_match, generic_edge_match
from .its_construction import ITSConstruction
from .its_extraction import ITSExtraction
from .mol_to_graph import MolToGraph
from .rules_extraction import RuleExtraction
from .utils import enumerate_tautomers, mapping_success_rate
from itertools import combinations


class AAMValidator:
    def __init__(self):
        """Initializes the AAMValidator class."""
        pass

    @staticmethod
    def graph_from_smiles(smiles: str) -> nx.Graph:
        """Constructs a graph representation from a SMILES string.

        :param smiles: A SMILES string representing a molecule or a set of molecules.
        :type smiles: str

        :return: A graph representation of the molecule(s).
        :rtype: nx.Graph
        """
        mol = Chem.MolFromSmiles(smiles)
        graph = MolToGraph().mol_to_graph(mol, drop_non_aam=True)
        return graph

    @staticmethod
    def check_equivariant_graph(
        its_graphs: List[nx.Graph],
    ) -> Tuple[List[Tuple[int, int]], int]:
        """Checks for isomorphism among a list of ITS graphs and
        identifies all pairs of isomorphic graphs.

        :param its_graphs: A list of ITS graphs.
        :type its_graphs: List[nx.Graph]

        :return: Pairs of indices of isomorphic graphs and the number of pairs.
        :rtype: Tuple[List[Tuple[int, int]], int]
        """
        nodeLabelNames = ["typesGH"]
        nodeLabelDefault = ["*", False, 0, 0, ()]
        nodeLabelOperator = [eq, eq, eq, eq, eq]
        nodeMatch = generic_node_match(
            nodeLabelNames, nodeLabelDefault, nodeLabelOperator
        )
        edgeMatch = generic_edge_match("order", 1, eq)

        classified = []

        # Use combinations to check each unique pair of graphs without repetition
        for i, j in combinations(range(len(its_graphs)), 2):
            if nx.is_isomorphic(
                its_graphs[i], its_graphs[j], node_match=nodeMatch, edge_match=edgeMatch
            ):
                classified.append((i, j))

        return classified, len(classified)

    @staticmethod
    def smiles_check(
        mapped_smile: str,
        ground_truth: str,
        check_method: str = "RC",  # or 'ITS'
        ignore_aromaticity: bool = False,
    ) -> bool:
        """Checks the equivalence of mapped SMILES against ground truth
        using reaction center (RC) or ITS graph method.

        :param mapped_smile: The mapped SMILES string.
        :type mapped_smile: str
        :param ground_truth: The ground truth SMILES string.
        :type ground_truth: str
        :param check_method: The method used for validation
                             ('RC' or 'ITS').
        :type check_method: str
        :param ignore_aromaticity: Flag to ignore aromaticity
                                   in ITS graph construction.
        :type ignore_aromaticity: bool

        :return: True if the mapped SMILES is equivalent to the ground truth,
                  False otherwise.
        :rtype: bool
        """
        its_graphs = []
        rules_graphs = []
        try:
            for rsmi in [mapped_smile, ground_truth]:
                reactants_side, products_side = rsmi.split(">>")
                G = AAMValidator.graph_from_smiles(reactants_side)  # Reactants graph
                H = AAMValidator.graph_from_smiles(products_side)  # Products graph

                ITS = ITSConstruction.ITSGraph(G, H, ignore_aromaticity)
                its_graphs.append(ITS)

                rules = RuleExtraction.extract_reaction_rules(G, H, ITS, extend=False)
                rules_graphs.append(rules[2])

            _, equivariant = AAMValidator.check_equivariant_graph(
                rules_graphs if check_method == "RC" else its_graphs
            )

            return equivariant == 1

        except Exception as e:
            print("An error occurred:", str(e))
            return False

    def smiles_check_tautomer(
        mapped_smile: str,
        ground_truth: str,
        check_method: str = "RC",  # or 'ITS'
        ignore_aromaticity: bool = False,
    ) -> Optional[bool]:
        """Determines if a given mapped SMILE string is equivalent to any tautomer of
        a ground truth SMILES string using a specified comparison method.

        :param mapped_smile: The mapped SMILES string to check against the tautomers of
                             the ground truth.
        :type mapped_smile: str
        :param ground_truth: The reference SMILES string for generating possible
                             tautomers.
        :type ground_truth: str
        :param check_method: The method used for checking equivalence. Default is 'RC'.
                             Possible values are 'RC' for reaction center or 'ITS'.
        :type check_method: str
        :param ignore_aromaticity: Flag to ignore differences in aromaticity between
                                   the mapped SMILE and the tautomers.Default is False.
        :type ignore_aromaticity: bool

        :return: True if the mapped SMILE matches any of the enumerated tautomers
                  of the ground truth according to the specified check method.
                  Returns False if no match is found.
                  Returns None if an error occurs during processing.
        :rtype: Optional[bool]

        :raises Exception: If an error occurs during the tautomer enumeration
            or the comparison process.
        """
        try:
            ground_truth_tautomers = enumerate_tautomers(ground_truth)
            return any(
                AAMValidator.smiles_check(
                    mapped_smile, t, check_method, ignore_aromaticity
                )
                for t in ground_truth_tautomers
            )
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    @staticmethod
    def check_pair(
        mapping: Dict[str, str],
        mapped_col: str,
        ground_truth_col: str,
        check_method: str = "RC",
        ignore_aromaticity: bool = False,
        ignore_tautomers: bool = False,
    ) -> bool:
        """Checks the equivalence between the mapped and ground truth
        values within a given mapping dictionary, using a specified check method.
        The check can optionally ignore aromaticity.

        :param mapping: A dictionary containing the data entries to check.
        :type mapping: Dict[str, str]
        :param mapped_col: The key in the mapping dictionary corresponding
                           to the mapped value.
        :type mapped_col: str
        :param ground_truth_col: The key in the mapping dictionary corresponding
                                 to the ground truth value.
        :type ground_truth_col: str
        :param check_method: The method used for checking the equivalence.
                             Defaults to 'RC'.
        :type check_method: str, optional
        :param ignore_aromaticity: Flag to indicate whether aromaticity
                                   should be ignored during the check. Defaults to False.
        :type ignore_aromaticity: bool, optional
        :param ignore_tautomers: Flag to indicate whether tautomers
                                 should be ignored during the check. Defaults to False.
        :type ignore_tautomers: bool, optional

        :return: The result of the check, indicating whether the mapped value is
                  equivalent to the ground truth according to the specified method
                  and considerations regarding aromaticity.
        :rtype: bool
        """
        if ignore_tautomers:
            return AAMValidator.smiles_check(
                mapping[mapped_col],
                mapping[ground_truth_col],
                check_method,
                ignore_aromaticity,
            )
        else:
            return AAMValidator.smiles_check_tautomer(
                mapping[mapped_col],
                mapping[ground_truth_col],
                check_method,
                ignore_aromaticity,
            )

    @staticmethod
    def validate_smiles(
        data: Union[pd.DataFrame, List[Dict[str, str]]],
        id_col: str = "R-id",
        ground_truth_col: str = "ground_truth",
        mapped_cols: List[str] = ["rxn_mapper", "graphormer", "local_mapper"],
        check_method: str = "RC",
        ignore_aromaticity: bool = False,
        n_jobs: int = 1,
        verbose: int = 0,
        ensemble=False,
        strategies=[
            ["rxn_mapper", "graphormer", "local_mapper"],
            ["rxn_mapper", "graphormer", "local_mapper", "rdt"],
        ],
        ignore_tautomers=True,
    ) -> List[Dict[str, Union[str, float, List[bool]]]]:
        """Validates collections of mapped SMILES against their ground truths for
        multiple mappers and calculates the accuracy.

        :param data: The input data containing mapped and ground truth SMILES.
        :type data: Union[pd.DataFrame, List[Dict[str, str]]]
        :param id_col: The name of the column or key containing the reaction ID.
        :type id_col: str
        :param ground_truth_col: The name of the column or key containing
                                 the ground truth SMILES.
        :type ground_truth_col: str
        :param mapped_cols: The list of columns or keys containing
                            the mapped SMILES for different mappers.
        :type mapped_cols: List[str]
        :param check_method: The method used for validation ('RC' or 'ITS').
        :type check_method: str
        :param ignore_aromaticity: Flag to ignore aromaticity in ITS graph construction.
        :type ignore_aromaticity: bool
        :param n_jobs: The number of parallel jobs to run.
        :type n_jobs: int
        :param verbose: The verbosity level for joblib's parallel execution.
        :type verbose: int

        :return: A list of dictionaries, each
                  containing the mapper name, accuracy, and individual results for each SMILES pair.
        :rtype: List[Dict[str, Union[str, float, List[bool]]]]
        """

        validation_results = []

        for mapped_col in mapped_cols:

            if isinstance(data, pd.DataFrame):
                mappings = data.to_dict("records")
            elif isinstance(data, list):
                mappings = data
            else:
                raise ValueError(
                    "Data must be either a pandas DataFrame or a list of dictionaries."
                )

            results = Parallel(n_jobs=n_jobs, verbose=verbose)(
                delayed(AAMValidator.check_pair)(
                    mapping,
                    mapped_col,
                    ground_truth_col,
                    check_method,
                    ignore_aromaticity,
                    ignore_tautomers,
                )
                for mapping in mappings
            )
            accuracy = sum(results) / len(mappings) if mappings else 0
            mapped_data = [value[mapped_col] for value in mappings]

            validation_results.append(
                {
                    "mapper": mapped_col,
                    "accuracy": round(100 * accuracy, 2),
                    "results": results,
                    "success_rate": mapping_success_rate(mapped_data),
                }
            )
        if ensemble:
            for key, strategy in enumerate(strategies):
                mapped_cols = strategy
                its_graph, _ = ITSExtraction.parallel_process_smiles(
                    mapped_smiles_list=mappings,
                    mapper_names=mapped_cols,
                    n_jobs=n_jobs,
                    verbose=verbose,
                    export_full=False,
                    check_method=check_method,
                )
                id = [value["R-id"] for value in its_graph]
                data_ensemble = [value for value in mappings if value["R-id"] in id]
                data_ensemble = [
                    {
                        id_col: value[id_col],
                        f"ensemble_{key+1}": value[mapped_cols[-1]],
                        ground_truth_col: value[ground_truth_col],
                    }
                    for value in data_ensemble
                ]
                results = Parallel(n_jobs=n_jobs, verbose=verbose)(
                    delayed(AAMValidator.check_pair)(
                        mapping,
                        f"ensemble_{key+1}",
                        ground_truth_col,
                        check_method,
                        ignore_aromaticity,
                        ignore_tautomers,
                    )
                    for mapping in data_ensemble
                )
                accuracy = sum(results) / len(data_ensemble)
                validation_results.append(
                    {
                        "mapper": f"ensemble_{key+1}",
                        "accuracy": round(100 * accuracy, 2),
                        "results": results,
                        "success_rate": round(
                            100 * len(data_ensemble) / len(mappings), 2
                        ),
                    }
                )

        return validation_results, data_ensemble if ensemble else None
