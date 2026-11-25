"""Round trips preserve identifiers, missing values and array metadata."""
import numpy as np
import pandas as pd
import pytest

from synrxn.io.io import load_df_gz, load_results_json, save_df_gz, save_results_json


@pytest.mark.parametrize("shape", [(2, 3), (0, 3)])
def test_results_preserve_array_shape_dtype_and_plain_metadata(tmp_path, shape):
    expected = np.arange(np.prod(shape), dtype=np.int32).reshape(shape)
    path = tmp_path / "results.json"
    save_results_json(path, {"fold": {"values": expected, "metric": "MAE", "missing": None}})
    actual = load_results_json(path)["fold"]
    np.testing.assert_array_equal(actual["values"], expected)
    assert actual["values"].dtype == expected.dtype
    assert actual["values"].shape == shape
    assert actual["metric"] == "MAE"
    assert actual["missing"] is None


def test_compressed_csv_preserves_text_identifiers_and_missing_measurements(tmp_path):
    expected = pd.DataFrame({"id": ["001", "002"], "note": ['α, "quoted"\nline', "β"],
                             "yield": [12.5, float("nan")]})
    path = tmp_path / "observations.csv.gz"
    save_df_gz(expected, path)
    actual = load_df_gz(path, read_csv_kwargs={"dtype": {"id": str}})
    pd.testing.assert_frame_equal(actual, expected)
