"""Read and normalize native values for expansion sources."""

from synrxn.reconstruction.expansion.storage import clean


def meaningful(value):
    """Return whether a source value is present."""
    value = clean(value)
    return value is not None and str(value).strip().lower() not in {"", "nan", "blank_cell", "none"}


def selected(row, names):
    """Return the present values for the named fields."""
    return {k: clean(row[k]) for k in names if k in row and meaningful(row[k])}


def csv_rows(path_or_file, sep=","):
    """Yield cleaned records with original row numbers from a CSV."""
    import pandas as pd

    ordinal = 0
    for frame in pd.read_csv(path_or_file, sep=sep, chunksize=4096, low_memory=False):
        for row in frame.to_dict("records"):
            yield ordinal, clean(row)
            ordinal += 1
