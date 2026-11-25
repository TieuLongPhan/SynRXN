import networkx as nx
from typing import List, Set, Dict, Any
from joblib import Parallel, delayed
from itertools import chain


class RuleExtraction:

    @staticmethod
    def find_unequal_order_edges(G: nx.Graph) -> List[int]:
        """Identifies reaction center nodes in a graph based on unequal order edges.

        :param G: The graph in which to identify reaction center nodes.
        :type G: nx.Graph

        :return: Nodes that are part of edges with unequal orders, indicating reaction
                  centers.
        :rtype: List[int]
        """
        reaction_center_nodes: Set[int] = set()
        for u, v, data in G.edges(data=True):
            order = data.get("order", (1, 1))
            if isinstance(order, tuple) and order[0] != order[1]:
                if data.get("standard_order") != 0:
                    reaction_center_nodes.add(u)
                    reaction_center_nodes.add(v)
        return list(reaction_center_nodes)

    @staticmethod
    def find_nearest_neighbors(
        G: nx.Graph, center_nodes: List[int], n_knn: int = 1
    ) -> Set[int]:
        """Finds up to n_knn levels of nearest neighbors for specified center nodes within a
        graph.

        :param G: The graph to search within.
        :type G: nx.Graph
        :param center_nodes: Nodes for which to find nearest neighbors.
        :type center_nodes: List[int]
        :param n_knn: Levels of nearest neighbors to include.
        :type n_knn: int

        :return: Nodes including the center nodes and their up to n_knn-nearest
                  neighbors.
        :rtype: Set[int]
        """
        extended_nodes = set(center_nodes)
        for _ in range(n_knn):
            neighbors = set(
                chain.from_iterable(G.neighbors(node) for node in extended_nodes)
            )
            extended_nodes.update(neighbors)
        return extended_nodes

    @staticmethod
    def extract_subgraph(G: nx.Graph, node_indices: List[int]) -> nx.Graph:
        """Extracts a subgraph based on specified node indices from a given graph.

        :param G: The original graph.
        :type G: nx.Graph
        :param node_indices: Node indices to include in the subgraph.
        :type node_indices: List[int]

        :return: The extracted subgraph.
        :rtype: nx.Graph
        """

        return G.subgraph(node_indices).copy()

    @staticmethod
    def extract_reaction_rules(
        reactants_graph, products_graph, its_graph, extend: bool = False, n_knn: int = 1
    ) -> Dict[str, Any]:
        """Extracts transformation rules for a chemical reaction represented by graphs of
        reactants, products, and intermediate transition states (ITS).

        This method identifies the reaction center nodes, optionally extends them based on
        nearest neighbors, and constructs a tuple of subgraphs representing the rules.

        :param reactants_graph: Graph representing the reactants.
        :type reactants_graph: Graph
        :param products_graph: Graph representing the products.
        :type products_graph: Graph
        :param its_graph: Graph representing the intermediate transition states.
        :type its_graph: Graph
        :param extend: If True, extends the reaction center nodes by including
                       nearest neighbors. Defaults to True.
        :type extend: bool, optional
        :param n_knn: Specifies the number of nearest neighbors to consider
                      when extending the reaction center nodes. Defaults to 1.
        :type n_knn: int, optional

        :return: A tuple contains the subgraphs for the reactants, products, and ITS,
                  representing the extracted rules.
        :rtype: Tuple
        """
        reaction_center_nodes = RuleExtraction.find_unequal_order_edges(its_graph)
        if extend:
            reaction_center_nodes = RuleExtraction.find_nearest_neighbors(
                its_graph, reaction_center_nodes, n_knn
            )

        rules_network = tuple(
            RuleExtraction.extract_subgraph(graph, reaction_center_nodes)
            for graph in (reactants_graph, products_graph, its_graph)
        )
        rules_graph = rules_network[2]
        if n_knn == 0:
            rules_graph = RuleExtraction.remove_normal_egdes(
                rules_graph, "standard_order"
            )
        rules_network = (rules_network[0], rules_network[1], rules_graph)
        return rules_network

    @staticmethod
    def rules_extraction(
        reaction_dict: Dict[str, Any],
        mapper_type: str = "ITSGraph",
        extend: bool = True,
        n_knn: int = 1,
    ) -> Dict[str, Any]:
        """Extracts rules for a single reaction using a specified mapper, optionally
        extending reaction centers.

        :param reaction_dict: Single reaction representation.
        :type reaction_dict: Dict[str, Any]
        :param mapper_type: Mapper type for rule extraction.
        :type mapper_type: str
        :param extend: Flag to extend reaction center nodes.
        :type extend: bool
        :param n_knn: Nearest neighbor levels to include if extending.
        :type n_knn: int

        :return: Reaction dictionary updated with extracted rules.
                  'GraphRules': Tuple of reactants graph (L graph), products graph
                  (R graph), ITS graph (K graph)
        :rtype: Dict[str, Any]
        """
        reactants_graph, products_graph, its_graph = reaction_dict[mapper_type]
        reaction_center_nodes = RuleExtraction.find_unequal_order_edges(its_graph)

        if extend:
            reaction_center_nodes = RuleExtraction.find_nearest_neighbors(
                its_graph, reaction_center_nodes, n_knn
            )

        rules_network = tuple(
            RuleExtraction.extract_subgraph(graph, list(reaction_center_nodes))
            for graph in (reactants_graph, products_graph, its_graph)
        )

        reaction_dict["GraphRules"] = rules_network
        return reaction_dict

    @classmethod
    def process_rules_extraction(
        cls,
        reaction_dicts: List[Dict[str, Any]],
        mapper_type: str = "ITSGraph",
        n_jobs: int = 4,
        verbose: int = 0,
        extend: bool = True,
        n_knn: int = 1,
    ) -> List[Dict[str, Any]]:
        """Parallelizes rules extraction for multiple reactions, with options for extension
        and parallelism control.

        :param reaction_dicts: Reactions to process.
        :type reaction_dicts: List[Dict[str, Any]]
        :param mapper_type: Mapper type for rule extraction.
        :type mapper_type: str
        :param n_jobs: Parallel jobs count (-1 for all CPUs).
        :type n_jobs: int
        :param verbose: Verbosity level.
        :type verbose: int
        :param extend: Flag to extend reaction center nodes.
        :type extend: bool
        :param n_knn: Nearest neighbor levels to include if extending.
        :type n_knn: int

        :return: Updated reactions with extracted rules.
        :rtype: List[Dict[str, Any]]
        """
        results = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(cls.rules_extraction)(reaction_dict, mapper_type, extend, n_knn)
            for reaction_dict in reaction_dicts
        )
        return results

    @staticmethod
    def remove_normal_egdes(graph: nx.Graph, property_key: str) -> nx.Graph:
        """Create a copy of the input graph and remove edges based on a specified property.

        This function iterates over the edges of the graph and removes any edge for which
        the specified property's values are the same at both the start and end nodes.

        :param graph: The input graph from which edges will be filtered.
        :type graph: nx.Graph
        :param property_key: The key for the edge attribute that should be examined.
        :type property_key: str

        :return: A new graph with the specified edges removed. The original graph is not
                  modified.
        :rtype: nx.Graph
        """
        # Create a copy of the graph to avoid inplace modification
        filtered_graph = graph.copy()

        # Iterate over the edges and check the specified property
        for edge in list(filtered_graph.edges(data=True)):
            # Unpack edge data
            start_node, end_node, attributes = edge

            if property_key in attributes and attributes[property_key] == 0:
                # Remove the edge if the property values are equal
                filtered_graph.remove_edge(start_node, end_node)

        return filtered_graph
