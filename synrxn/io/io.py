import io
import json
import gzip
import requests
import pandas as pd
from pathlib import Path
from typing import Optional, Union, Dict, Any


def load_json_from_raw_github(
    url: str,
    as_frame: bool = True,
    lines: Optional[bool] = None,
) -> Union[pd.DataFrame, list, dict]:
    """
    Load JSON data from a raw GitHub URL that may be plain JSON, NDJSON (one JSON
    object per line), or gzipped JSON. The function attempts gzip decompression
    first, and falls back to plain-text decoding if the content is not gzipped.

    :param url: URL pointing to the raw content (e.g.
                "https://raw.githubusercontent.com/.../file.json.gz").
    :type url: str

    :param as_frame: If True, attempts to return a `pandas.DataFrame` (using
                     `pd.read_json`). If DataFrame conversion fails, returns a
                     parsed Python object (list/dict). If False, returns a Python
                     object directly.
    :type as_frame: bool, optional (default: True)

    :param lines: Force NDJSON (newline-delimited JSON) parsing when True, force
                  standard JSON parsing when False. If None (default), the
                  function heuristically detects NDJSON by looking for multiple
                  lines that start with '{' (a common NDJSON pattern).
    :type lines: bool or None, optional

    :return: If `as_frame` is True and conversion succeeds, returns a
             `pandas.DataFrame`. Otherwise returns the parsed Python JSON object
             (list or dict), or a list of objects when NDJSON and `as_frame` is
             False.
    :rtype: pandas.DataFrame or list or dict

    :raises requests.exceptions.HTTPError: If the HTTP request returned an error status.
    :raises json.JSONDecodeError: If JSON decoding fails for the detected mode.
    :raises Exception: Unexpected exceptions may propagate (e.g., encoding issues).

    :examples:

    >>> url = "https://raw.githubusercontent.com/owner/repo/main/data.json.gz"
    >>> df = load_json_from_raw_github(url)  # try to get a DataFrame
    >>> obj = load_json_from_raw_github(url, as_frame=False)  # Python object
    >>> df_lines = load_json_from_raw_github(url, lines=True)  # force NDJSON
    """
    # Fetch remote content
    r = requests.get(url, stream=True)
    r.raise_for_status()

    buf = io.BytesIO(r.content)

    # Try decompressing as gzip; fall back to plain text if not gzipped
    try:
        with gzip.GzipFile(fileobj=buf) as fh:
            text = fh.read().decode("utf-8")
    except (OSError, gzip.BadGzipFile):
        buf.seek(0)
        text = buf.read().decode("utf-8")

    # Determine NDJSON (lines) mode if user didn't force it
    if lines is None:
        # Heuristic: multiple lines and many lines beginning with '{'
        stripped = text.lstrip()
        is_ndjson = ("\n" in text) and stripped.startswith("{") and ("\n{" in text)
        lines_mode = is_ndjson
    else:
        lines_mode = bool(lines)

    # Parse and return
    if as_frame:
        # Try to produce a DataFrame where possible
        try:
            if lines_mode:
                return pd.read_json(io.StringIO(text), lines=True)
            else:
                return pd.read_json(io.StringIO(text))
        except ValueError:
            # pd.read_json couldn't parse (maybe nested, or not table-like) -> fallback
            return json.loads(text)
    else:
        if lines_mode:
            return [json.loads(line) for line in text.splitlines() if line.strip()]
        else:
            return json.loads(text)


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
