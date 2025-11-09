import unittest
from synrxn.data.github_client import GitHubClient


class FakeResp:
    def __init__(self, status_code=200, json_data=None):
        self.status_code = status_code
        self._json = json_data or []
        self.headers = {}
        self.content = b""

    def json(self):
        return self._json

    def close(self):
        # mimic requests.Response.close()
        return None

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


class FakeSession:
    def __init__(self):
        self.calls = []

    def get(self, url, timeout=None, stream=False):
        self.calls.append((url, timeout, stream))
        # Simulate a GitHub API listing for a specific task path
        if "contents" in url:
            # return two files
            return FakeResp(json_data=[{"name": "a.csv"}, {"name": "b.csv.gz"}])
        # Simulate raw content availability; treat all raw requests as existing
        return FakeResp(status_code=200)


class TestGitHubClient(unittest.TestCase):
    def setUp(self):
        self.session = FakeSession()
        self.client = GitHubClient(
            session=self.session,
            timeout=5,
            owner="owner",
            repo="repo",
            ref_candidates=[("heads", "main")],
        )

    def test_list_names(self):
        names = self.client.list_names("rbl")
        self.assertIn("a", names)
        self.assertIn("b", names)  # b.csv.gz -> "b"
        self.assertEqual(sorted(names), sorted(names))

    def test_raw_url_success(self):
        # raw_url should attempt GET; our FakeSession returns 200 so we expect a URL back
        url = self.client.raw_url("rbl", "a", ".csv")
        self.assertIsInstance(url, str)
        # ensure the session was asked at least once
        self.assertTrue(len(self.session.calls) >= 1)


if __name__ == "__main__":
    unittest.main()
