from pathlib import Path
from typing import Optional, Union, Dict, Any
import gzip
import pandas as pd


def save_df_gz(
    df: pd.DataFrame,
    path: Union[str, Path],
    *,
    index: bool = False,
    encoding: str = "utf-8",
    compresslevel: int = 9,
    to_csv_kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save a pandas DataFrame to a gzip-compressed CSV file.

    :param df: DataFrame to save.
    :type df: pandas.DataFrame
    :param path: Destination file path. Common convention is to use a `.csv.gz` suffix.
    :type path: str or pathlib.Path
    :param index: Whether to write row names (index). Defaults to False.
    :type index: bool
    :param encoding: Text encoding used when writing the CSV. Defaults to 'utf-8'.
    :type encoding: str
    :param compresslevel: Gzip compression level (1-9). Higher is more compression and slower. Defaults to 9.
    :type compresslevel: int
    :param to_csv_kwargs: Additional keyword arguments forwarded to `pandas.DataFrame.to_csv`
                         (for example: `sep`, `float_format`, `na_rep`, etc.). Defaults to None.
    :type to_csv_kwargs: dict or None
    :raises OSError: If the file cannot be written (e.g., permission/IO error).
    :returns: None
    :rtype: None
    """
    path = Path(path)
    to_csv_kwargs = dict(to_csv_kwargs or {})
    # write text-mode gzip file and let pandas write into it
    with gzip.open(
        path, mode="wt", compresslevel=compresslevel, encoding=encoding
    ) as fh:
        df.to_csv(fh, index=index, **to_csv_kwargs)


def load_df_gz(
    path: Union[str, Path],
    *,
    encoding: str = "utf-8",
    read_csv_kwargs: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Load a gzip-compressed CSV file into a pandas DataFrame.

    :param path: Path to the `.csv.gz` file to read.
    :type path: str or pathlib.Path
    :param encoding: Text encoding used when reading the CSV. Defaults to 'utf-8'.
    :type encoding: str
    :param read_csv_kwargs: Additional keyword arguments forwarded to `pandas.read_csv`
                           (for example: `sep`, `index_col`, `parse_dates`, `dtype`, etc.). Defaults to None.
    :type read_csv_kwargs: dict or None
    :raises FileNotFoundError: If the input path does not exist.
    :raises OSError: If the file cannot be opened/read.
    :returns: The loaded DataFrame.
    :rtype: pandas.DataFrame
    """
    read_csv_kwargs = dict(read_csv_kwargs or {})
    with gzip.open(path, mode="rt", encoding=encoding) as fh:
        return pd.read_csv(fh, **read_csv_kwargs)
