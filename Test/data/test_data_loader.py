import unittest
from synrxn.data.data_loader import DataLoader
from types import SimpleNamespace
import io
import pandas as pd
import tempfile
import gzip


# Fake Zenodo client that DataLoader will accept by replacing its _zenodo attr
class FakeZenodoClient:
    def __init__(self):
        # simulate a record id
        self._record_id = 123
        # file index with a direct csv.gz and an archive
        self._file_index = {
            "Data/rbl/mis.csv.gz": {
                "key": "Data/rbl/mis.csv.gz",
                "links": {"self": "https://example/file/mis.csv.gz"},
                "checksum": "",
            },
            "TieuLongPhan/SynRXN-v0.0.5.zip": {
                "key": "TieuLongPhan/SynRXN-v0.0.5.zip",
                "links": {"self": "https://example/file/zip"},
                "checksum": "",
            },
        }

    def resolve_record_id(self, concept_doi, version):
        return self._record_id

    def build_file_index(self, record_id):
        return self._file_index

    def available_names(self, task, record_id, file_index, include_archives=True):
        # If include archives True, pretend archive has "Data/rbl/arch.csv"
        names = []
        for k in file_index:
            if k.startswith("Data/rbl/") and k.endswith(".csv.gz"):
                names.append("mis")
        if include_archives:
            names += ["arch"]
        return sorted(set(names))

    def get_download_response(self, meta, record_id):
        """
        Return a lightweight object imitating a requests.Response:
        - For Data/rbl/mis.csv.gz return gzipped CSV bytes.
        - For archives, return a zip bytes containing Data/rbl/arch.csv.
        """
        key = meta.get("key", "")
        if key.endswith("mis.csv.gz"):
            # create gzipped csv bytes
            csv_bytes = b"a,b\n1,2\n"
            buf = io.BytesIO()
            with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
                gz.write(csv_bytes)
            gz_bytes = buf.getvalue()
            return SimpleNamespace(
                status_code=200,
                headers={
                    "Content-Type": "application/octet-stream",
                    "Content-Length": str(len(gz_bytes)),
                },
                iter_content=lambda chunk_size=8192, _b=gz_bytes: (
                    _b[i : i + chunk_size] for i in range(0, len(_b), chunk_size)
                ),
                content=gz_bytes,
                url="https://fake/mis.csv.gz",
                close=lambda: None,
            )
        else:
            # for archives return a zip bytes containing Data/rbl/arch.csv
            b = io.BytesIO()
            import zipfile

            with zipfile.ZipFile(b, mode="w") as zf:
                zf.writestr(
                    "TieuLongPhan/SynRXN-v0.0.5/Data/rbl/arch.csv", "x,y\n9,8\n"
                )
            bb = b.getvalue()
            return SimpleNamespace(
                status_code=200,
                headers={
                    "Content-Type": "application/zip",
                    "Content-Length": str(len(bb)),
                },
                iter_content=lambda chunk_size=8192, _bb=bb: (
                    _bb[i : i + chunk_size] for i in range(0, len(_bb), chunk_size)
                ),
                content=bb,
                url="https://fake/archive.zip",
                close=lambda: None,
            )

    def stream_to_temp_and_verify(self, resp, meta, suffix):
        # write resp.content to temp file and return path
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(resp.content)
        tmp.flush()
        tmp.close()
        return tmp.name

    def list_archive_members_cached(self, record_id, archive_key, meta):
        # pretend archive contains this member
        return ["TieuLongPhan/SynRXN-v0.0.5/Data/rbl/arch.csv"]

    def extract_member_bytes(self, archive_path, member_name):
        # simulate reading member from the bytes we produced earlier
        if "arch.csv" in member_name or member_name.endswith("arch.csv"):
            return b"x,y\n9,8\n"
        return None

    # ---- ADDED: find_keys helper expected by DataLoader ----
    def find_keys(self, file_index, term):
        t = term.lower()
        return [k for k in file_index.keys() if t in k.lower()]


class FakeGitHubClient:
    def __init__(self):
        pass

    def list_names(self, task):
        return ["gh_one", "gh_two"]

    def raw_url(self, task, name, ext):
        # Return a plain CSV URL only for the ".csv" ext so loader will pick it
        if name == "gh_one" and ext == ".csv":
            return f"https://raw/{task}/{name}{ext}"
        return None


class TestDataLoader(unittest.TestCase):
    def setUp(self):
        # create a DataLoader but replace its clients with fakes to avoid network
        self.dl = DataLoader(
            task="rbl",
            version="0.0.5",
            source="zenodo",
            gh_enable=True,
            resolve_on_init=False,
        )
        self.dl._zenodo = FakeZenodoClient()
        self.dl._github = FakeGitHubClient()
        self.dl._record_id = 123
        self.dl._file_index = self.dl._zenodo.build_file_index(123)
        # ensure no cache dir needed
        self.dl.cache_dir = None

    def test_available_names_zenodo(self):
        names = self.dl.available_names(refresh=True)
        self.assertIn("mis", names)
        self.assertIn("arch", names)

    def test_load_direct_zenodo_file(self):
        df = self.dl.load("mis")
        # should parse into DataFrame
        self.assertIsInstance(df, pd.DataFrame)
        self.assertListEqual(list(df.columns), ["a", "b"])

    def test_load_from_archive(self):
        # Name 'arch' only exists inside archive per fake client
        df = self.dl.load("arch")
        self.assertIsInstance(df, pd.DataFrame)
        self.assertListEqual(list(df.columns), ["x", "y"])

    def test_github_fallback(self):
        # Use source='github' (valid for DataLoader) and fake GitHub client
        dl2 = DataLoader(
            task="rbl", source="github", gh_enable=True, resolve_on_init=False
        )
        dl2._zenodo = self.dl._zenodo
        dl2._github = FakeGitHubClient()
        dl2._record_id = 123
        dl2._file_index = dl2._zenodo.build_file_index(123)

        # Patch DataLoader._session.get to return CSV bytes
        original_session_get = dl2._session.get

        def fake_get(url, timeout=None, stream=False):
            class R:
                status_code = 200
                content = b"a,b\n5,6\n"

                def close(self):
                    pass

            return R()

        dl2._session.get = fake_get

        # ensure raw_url only returns plain .csv so loader picks the CSV branch
        dl2._github.raw_url = lambda task, name, ext: (
            "https://fake/gh_one.csv" if name == "gh_one" and ext == ".csv" else None
        )

        df = dl2.load("gh_one")
        self.assertIsInstance(df, pd.DataFrame)
        self.assertListEqual(list(df.columns), ["a", "b"])

        # restore
        dl2._session.get = original_session_get

    def test_find_zenodo_keys(self):
        keys = self.dl.find_zenodo_keys("mis")
        self.assertTrue(any("mis" in k for k in keys))


if __name__ == "__main__":
    unittest.main()
