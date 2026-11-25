import os
from typing import List, Dict, Any, Set, Tuple, Optional, Generator
import json
import pickle
import random
import subprocess
import logging
import pandas as pd
from sklearn.model_selection import train_test_split


def setup_logging(log_level: str = "INFO", log_filename: str = None) -> logging.Logger:
    """Configures the logging for an application. It sets up logging to either the console
    or a file based on whether a log filename is provided. The function adjusts the
    logging level dynamically according to the specified log level.

    :param log_level: Specifies the logging level. Accepted values are 'DEBUG', 'INFO', 'WARNING',
                      'ERROR', 'CRITICAL'. Default is 'INFO'.
    :type log_level: str, optional
    :param log_filename: Specifies the filename of the log file. If provided, logs will be written
                         to this file. If None, logs will be written to the console. Default is None.
    :type log_filename: str, optional

    :return: The configured logger object.
    :rtype: logging.Logger

    :raises ValueError: If the specified log_level is not recognized as a valid logging level.
    """
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    if log_filename:
        os.makedirs(os.path.dirname(log_filename), exist_ok=True)
        logging.basicConfig(
            level=numeric_level, format=log_format, filename=log_filename, filemode="w"
        )
    else:
        logging.basicConfig(level=numeric_level, format=log_format)

    return logger


def save_database(database: list[dict], pathname: str = "./Data/database.json") -> None:
    """Save a database (a list of dictionaries) to a JSON file.

    :param database: The database to be saved.
    :param pathname: The path where the database will be saved.
                     Defaults to './Data/database.json'.

    :raises TypeError: If the database is not a list of dictionaries.
    :raises ValueError: If there is an error writing the file.
    """
    if not all(isinstance(item, dict) for item in database):
        raise TypeError("Database should be a list of dictionaries.")

    try:
        with open(pathname, "w") as f:
            json.dump(database, f)
    except IOError as e:
        raise ValueError(f"Error writing to file {pathname}: {e}")


def load_database(pathname: str = "./Data/database.json") -> List[Dict]:
    """Load a database (a list of dictionaries) from a JSON file.

    :param pathname: The path from where the database will be loaded.
                     Defaults to './Data/database.json'.

    :return: The loaded database.

    :raises ValueError: If there is an error reading the file.
    """
    try:
        with open(pathname, "r") as f:
            database = json.load(f)  # Load the JSON data from the file
        return database
    except IOError as e:
        raise ValueError(f"Error reading to file {pathname}: {e}")


def save_to_pickle(data: List[Dict[str, Any]], filename: str) -> None:
    """Save a list of dictionaries to a pickle file.

    :param data: A list of dictionaries to be saved.
    :type data: List[Dict[str, Any]]
    :param filename: The name of the file where the data will be saved.
    :type filename: str
    """
    with open(filename, "wb") as file:
        pickle.dump(data, file)


def load_from_pickle(filename: str) -> List[Any]:
    """Load data from a pickle file.

    :param filename: The name of the pickle file to load data from.
    :type filename: str

    :return: The data loaded from the pickle file.
    :rtype: List[Any]
    """
    with open(filename, "rb") as file:
        return pickle.load(file)


def stratified_random_sample(
    data: List[Dict], property_key: str, samples_per_class: int, seed: int = None
) -> List[Dict]:
    """Stratifies and samples data from a list of dictionaries based on a specified property.

    :param data: The data to sample from, a list of dictionaries.
    :type data: List[Dict]
    :param property_key: The key in the dictionaries to stratify by.
    :type property_key: str
    :param samples_per_class: The number of samples to take from each class.
    :type samples_per_class: int
    :param seed: The seed for the random number generator for reproducibility.
    :type seed: int

    :return: A list of sampled dictionaries.
    :rtype: List[Dict]
    """

    if seed is not None:
        random.seed(seed)

    # Group data by the specified property
    stratified_data = {}
    for item in data:
        key = item.get(property_key)
        if key in stratified_data:
            stratified_data[key].append(item)
        else:
            stratified_data[key] = [item]

    # Sample data from each group
    sampled_data = []
    for key, items in stratified_data.items():
        if len(items) >= samples_per_class:
            sampled_data.extend(random.sample(items, samples_per_class))
        else:
            raise ValueError(
                f"Not enough data to sample {samples_per_class} items for class {key}"
            )

    return sampled_data


def ensure_directory_exists(directory_name):
    """Checks if a directory with the given name exists, and if not, creates it.

    :param directory_name: The name of the directory to check and potentially create.
    :type directory_name: str
    """
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
        print(f"Directory '{directory_name}' created.")
    else:
        print(f"Directory '{directory_name}' already exists.")


def run_shell_command(command="mod_post"):
    """Executes a shell command and prints its output.

    :param command: The shell command to execute.
    :type command: str
    """
    try:
        # Execute the command and capture the output
        result = subprocess.run(
            command,
            shell=True,
            text=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print("Command output:", result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error occurred while executing the command.")
        print("Error Code:", e.returncode)
        print("Error Message:", e.stderr)


def create_unique_value_dict(original_dict):
    """Creates a dictionary where each unique value from the original dictionary
    is mapped to the first key associated with that value.

    :param original_dict: The original dictionary with potentially duplicate values.
    :type original_dict: dict

    :return: A new dictionary with unique values and their first corresponding keys.
    :rtype: dict
    """
    new_dict = {}
    for key, value in original_dict.items():
        if value not in new_dict:
            new_dict[value] = key

    return new_dict


def train_val_test_split_df(
    df: pd.DataFrame,
    target: str,
    train_size: float = 0.8,
    val_size: float = 0.1,
    test_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits a DataFrame into train, validation, and test sets with stratification.

    :param df: The DataFrame to split.
    :type df: pd.DataFrame
    :param target: The name of the column to stratify by.
    :type target: str
    :param train_size: Proportion of the dataset to include in the train split.
    :type train_size: float
    :param val_size: Proportion of the dataset to include in the validation split.
    :type val_size: float
    :param test_size: Proportion of the dataset to include in the test split.
    :type test_size: float
    :param random_state: The seed used by the random number generator.
    :type random_state: int

    :return: A tuple containing the training
              set, validation set, and test set as DataFrames.
    :rtype: Tuple[pd.DataFrame, pd.UtilityModel, pd.DataFrame]

    :raises AssertionError: If the sum of train_size, val_size, and test_size does not equal 1.
    """
    # Ensure the sizes sum to 1
    assert (
        train_size + val_size + test_size == 1
    ), "The sum of train_size, val_size, and test_size must equal 1"

    # Split the data into training and a temporary set
    train_df, temp_df = train_test_split(
        df, train_size=train_size, stratify=df[target], random_state=random_state
    )

    # Calculate the relative size of the test set from the remaining data
    test_size_relative = test_size / (val_size + test_size)

    # Further split the temporary set into validation and test sets
    val_df, test_df = train_test_split(
        temp_df,
        train_size=1 - test_size_relative,
        stratify=temp_df[target],
        random_state=random_state,
    )

    return train_df, val_df, test_df


def modify_smiles(smiles):
    """Modify the input SMILES string by replacing '.[HH]' with '[H][H]'.

    :param smiles: The input SMILES string representing a chemical reaction.
    :type smiles: str

    :return: The modified SMILES string.
    :rtype: str
    """
    return smiles.replace("[HH]", "[H][H]")


def prune_branches(
    data: List[List[Dict[str, Optional[int]]]],
) -> List[List[Dict[str, Optional[int]]]]:
    """Prunes branches in a hierarchical structure where any parent does not exist in the
    immediately preceding layer. Each layer is represented as a list of nodes, and
    each node is a dictionary with keys including 'Cluster_id' (integer) for the node's
    identifier and 'Parent' (integer or None for root nodes) for the node's parent ID
    in the previous layer.

    :param data: Hierarchical data where
                 each sub-list represents a layer.
    :type data: List[List[Dict[str, Optional[int]]]]

    :return: Pruned hierarchical data containing
              only nodes with existing parents.

              Each node's existence in a given layer depends on the presence of its 'Parent' ID in
              the set of 'Cluster_id's from the immediately previous layer. Root nodes in the
              first layer should have their 'Parent' set to None.
    :rtype: List[List[Dict[str, Optional[int]]]]
    """
    # Dictionary to keep track of existing cluster IDs for each layer
    existing_ids: Dict[int, Set[int]] = {0: {item["Cluster_id"] for item in data[0]}}

    # Process each layer after the first
    for layer_index in range(1, len(data)):
        layer = data[layer_index]
        # Store current layer's cluster IDs temporarily
        current_ids: Set[int] = set()

        new_layer: List[Dict[str, Optional[int]]] = []
        for item in layer:
            # Check if the single parent exists in the previous layer's IDs
            if item["Parent"] in existing_ids[layer_index - 1]:
                new_layer.append(item)
                current_ids.add(item["Cluster_id"])

        # Update the data layer with pruned nodes
        data[layer_index] = new_layer
        # Update existing IDs for this layer
        existing_ids[layer_index] = current_ids

    return data


def reindex_data(
    data: List[List[Dict[str, Optional[int]]]], starting_indices: List[int]
) -> List[List[Dict[str, Optional[int]]]]:
    """Reindexes 'Cluster_id', 'Parent', and optionally 'Child' fields in hierarchical data.
    First, each layer is normalized to start indexing from 0. Then, a layer-specific
    starting index from 'starting_indices' is added to each 'Cluster_id' in the layer.

    :param data: Hierarchical data
                 where each sub-list represents a layer.
    :type data: List[List[Dict[str, Optional[int]]]]
    :param starting_indices: The list of starting indices for 'Cluster_id'
                             in each layer.
    :type starting_indices: List[int]

    :return: Hierarchical data with updated indices.
    :rtype: List[List[Dict[str, Optional[int]]]]
    """
    reindexed_data = []

    # Initialize id_map for reindexing all layers from 0, then apply starting_indices
    id_map = []
    for layer in data:
        current_map = {item["Cluster_id"]: idx for idx, item in enumerate(layer)}
        id_map.append(current_map)

    for layer_index, layer in enumerate(data):
        new_layer = []
        for item in layer:
            new_item = item.copy()
            # Apply the starting index for the current layer
            new_item["Cluster_id"] = (
                id_map[layer_index][item["Cluster_id"]] + starting_indices[layer_index]
            )

            # Update the 'Parent' field to the new index if it's not None
            if item["Parent"] is not None and layer_index > 0:
                new_item["Parent"] = (
                    id_map[layer_index - 1][item["Parent"]]
                    + starting_indices[layer_index - 1]
                )

            if "Child" in item and (layer_index + 1) < len(data):
                new_item["Child"] = [
                    id_map[layer_index + 1][child] + starting_indices[layer_index + 1]
                    for child in item["Child"]
                    if child in id_map[layer_index + 1]
                ]

            new_layer.append(new_item)
        reindexed_data.append(new_layer)

    return reindexed_data


def load_from_pickle_generator(file_path: str) -> Generator[Any, None, None]:
    """Yield individual items from lists stored sequentially in a pickle file.

    :param file_path: Path to the pickle file.
    :type file_path: str
    :yield: Each item in each serialized list, in stored order.
    :ytype: Any
    """
    with open(file_path, "rb") as file:
        while True:
            try:
                batch_items = pickle.load(file)
                for item in batch_items:
                    yield item
            except EOFError:
                break


def collect_data(num_batches: int, temp_dir: str, file_template: str) -> List[Any]:
    """Collects and aggregates data from multiple pickle files into a single list.

    :param num_batches: The number of batch files to process.
    :type num_batches: int
    :param temp_dir: The directory where the batch files are stored.
    :type temp_dir: str
    :param file_template: The template string for batch file names, expecting an integer
                          formatter.
    :type file_template: str

    :return: A list of aggregated data items from all batch files.
    :rtype: List[Any]
    """
    collected_data: List[Any] = []
    for i in range(num_batches):
        file_path = os.path.join(temp_dir, file_template.format(i))
        for item in load_from_pickle_generator(file_path):
            collected_data.append(item)
    return collected_data
