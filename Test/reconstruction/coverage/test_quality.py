"""Small, independent source fixtures for the streaming content-quality audit."""
import csv
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from synrxn.reconstruction.coverage import quality


class ContentQualityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.repo = Path(self.temporary.name)

    def record(self, name, rows, format_=None):
        path = self.repo / name
        result = {"namespace": "fixture", "key": name, "path": name, "expected_rows": rows,
                  "expected_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        if format_:
            result["format"] = format_
        return result

    def test_quoted_multiline_csv_is_one_logical_record_and_zero_is_valid(self):
        with (self.repo / "data.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["record_id", "text", "target"])
            writer.writerow(["A", "first line\nsecond line", 0])
        result = quality.audit_table(self.record("data.csv", 1), self.repo)
        self.assertEqual(result["status"], "profiled")
        self.assertEqual(result["observed_rows"], 1)
        self.assertEqual(result["columns"]["target"]["missing_count"], 0)
        self.assertFalse(result["malformed_rows"])

    def test_repeated_group_or_reaction_is_not_a_duplicate_row_identifier(self):
        (self.repo / "data.csv").write_text("record_id,reaction_smiles,parent_group\nA,C>>O,G\nB,C>>O,G\n")
        result = quality.audit_table(self.record("data.csv", 2), self.repo)
        self.assertEqual(result["candidate_identifiers"]["record_id"]["duplicate_occurrences"], 0)
        self.assertEqual(set(result["candidate_identifiers"]), {"record_id"})

    def test_exact_duplicate_ids_are_review_findings_without_altering_rows(self):
        (self.repo / "data.csv").write_text("source_record_id,target\nA,0\nA,1\nA,2\n")
        result = quality.audit_table(self.record("data.csv", 3), self.repo)
        self.assertEqual(result["status"], "profiled")
        identifiers = result["candidate_identifiers"]["source_record_id"]
        self.assertEqual(identifiers["duplicate_occurrences"], 2)
        self.assertEqual(identifiers["duplicate_distinct_values"], 1)
        self.assertEqual(result["observed_rows"], 3)

    def test_potential_missing_strings_are_separate_from_empty_values(self):
        (self.repo / "data.csv").write_text("r_id,target\nA,NaN\nB,null\nC,\nD,0\n")
        result = quality.audit_table(self.record("data.csv", 4), self.repo)
        target = result["columns"]["target"]
        self.assertEqual(target["potential_missing_string_count"], 2)
        self.assertEqual(target["missing_empty_string"], 1)
        self.assertEqual(target["missing_count"], 1)

    def test_short_csv_row_records_absent_field_without_inventing_eligibility(self):
        (self.repo / "data.csv").write_text("record_id,target,group\nA,0,G\nB,1\n")
        result = quality.audit_table(self.record("data.csv", 2), self.repo)
        self.assertEqual(result["status"], "profiled")
        self.assertEqual(result["malformed_rows"], 1)
        self.assertEqual(result["columns"]["group"]["absent_rows"], 1)
        self.assertEqual(result["columns"]["group"]["missing_count"], 1)

    def test_actual_none_nonfinite_and_false_are_distinguished(self):
        with gzip.open(self.repo / "data.jsonl.gz", "wt") as handle:
            for identifier, value in [("A", None), ("B", float("nan")), ("C", False), ("D", 0)]:
                handle.write(json.dumps({"sample_id": identifier, "target": value}) + "\n")
        result = quality.audit_table(self.record("data.jsonl.gz", 4), self.repo)
        target = result["columns"]["target"]
        self.assertEqual(target["missing_none"], 1)
        self.assertEqual(target["missing_nonfinite_float"], 1)
        self.assertEqual(target["missing_count"], 2)
        self.assertEqual(target["types"]["bool"], 1)

    def test_json_payload_missing_keys_null_and_empty_reference_lists(self):
        rows = [{"sample_id": "A", "inputs_json": '{"text":"source"}', "targets_json": '{"yield":null,"entities":[]}'},
                {"sample_id": "B", "inputs_json": '{"text":"source"}', "targets_json": '{"entities":[]}'}]
        with gzip.open(self.repo / "data.jsonl.gz", "wt") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        result = quality.audit_table(self.record("data.jsonl.gz", 2), self.repo)
        logical = result["logical_json"]["targets_json"]
        self.assertEqual(logical["parsed_object_rows"], 2)
        self.assertEqual(logical["logical_keys"]["yield"]["missing_none"], 1)
        self.assertEqual(logical["logical_keys"]["yield"]["absent_rows"], 1)
        self.assertEqual(logical["logical_keys"]["yield"]["missing_count"], 2)
        self.assertEqual(logical["logical_keys"]["entities"]["empty_container_count"], 2)
        self.assertEqual(logical["logical_keys"]["entities"]["missing_count"], 0)
        self.assertTrue(any(x["kind"] == "logical_key_missing_count" for x in result["review_findings"]))

    def test_json_cell_parse_error_is_a_finding_not_a_container_failure(self):
        (self.repo / "data.csv").write_text('record_id,targets_json\nA,broken-json\nB,{}\n')
        result = quality.audit_table(self.record("data.csv", 2), self.repo)
        self.assertEqual(result["status"], "profiled")
        self.assertEqual(result["logical_json"]["targets_json"]["parse_errors"], 1)

    def test_malformed_jsonl_container_is_fatal_but_other_rows_are_profiled(self):
        with gzip.open(self.repo / "data.jsonl.gz", "wt") as handle:
            handle.write('{"record_id":"A","target":0}\n{broken\n{"record_id":"B","target":1}\n')
        result = quality.audit_table(self.record("data.jsonl.gz", 3), self.repo)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["observed_rows"], 3)
        self.assertEqual(result["profiled_object_rows"], 2)
        self.assertEqual(result["malformed_rows"], 1)

    def test_nonobject_jsonl_row_is_reported_as_malformed_row(self):
        (self.repo / "data.jsonl").write_text('null\n{"record_id":"B"}\n')
        result = quality.audit_table(self.record("data.jsonl", 2), self.repo)
        self.assertEqual(result["malformed_rows"], 1)
        self.assertEqual(result["profiled_object_rows"], 1)
        self.assertEqual(result["status"], "profiled")

    def test_checksum_and_rowcount_mismatches_are_fatal(self):
        (self.repo / "data.csv").write_text("record_id,target\nA,0\n")
        record = self.record("data.csv", 2)
        record["expected_sha256"] = "0" * 64
        result = quality.audit_table(record, self.repo)
        self.assertEqual(result["status"], "failed")
        self.assertTrue(any("Checksum mismatch" in value for value in result["fatal_errors"]))
        self.assertTrue(any("Row count mismatch" in value for value in result["fatal_errors"]))

    def test_compressed_csv_container_crc_failure_is_fatal(self):
        path = self.repo / "data.csv.gz"
        with gzip.open(path, "wt", newline="") as handle:
            handle.write("record_id,target\nA,0\n")
        path.write_bytes(path.read_bytes()[:-5])
        result = quality.audit_table(self.record("data.csv.gz", 1), self.repo)
        self.assertEqual(result["status"], "failed")
        self.assertTrue(result["container_errors"])

    def test_parquet_nullable_fields_are_profiled_without_string_coercion(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        pq.write_table(pa.table({"record_id": ["A", "B", "C"], "target": [0.0, None, float("inf")]}), self.repo / "data.parquet")
        result = quality.audit_table(self.record("data.parquet", 3), self.repo)
        self.assertEqual(result["status"], "profiled")
        self.assertEqual(result["columns"]["target"]["missing_count"], 2)
        self.assertEqual(result["columns"]["target"]["missing_nonfinite_float"], 1)

    def test_nonuniform_json_objects_report_absent_columns(self):
        (self.repo / "data.jsonl").write_text('{"record_id":"A","target":0}\n{"record_id":"B","other":1}\n')
        record = self.record("data.jsonl", 2)
        result = quality.build(self.repo, inventory=[record])
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["tables"][0]["columns"]["target"]["absent_rows"], 1)
        self.assertEqual(result["summary"]["observed_rows"], 2)


if __name__ == "__main__":
    unittest.main()
