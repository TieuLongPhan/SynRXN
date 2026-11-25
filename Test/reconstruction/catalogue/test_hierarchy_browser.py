"""Published catalogue views must preserve task contracts and release decisions."""
import copy
import json
from html.parser import HTMLParser

import pytest

from synrxn.reconstruction.catalogue.hierarchy import public_inventory, render_fragment
from synrxn.reconstruction.paths import REPO, physical_path


class Links(HTMLParser):
    def __init__(self, source):
        super().__init__()
        self.hrefs = []
        self.feed(source)

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.hrefs.extend(value for key, value in attrs if key == "href")


@pytest.fixture
def inventory():
    return json.loads(physical_path("kdd/hierarchy/hierarchy.json").read_text())


def test_held_records_have_no_payload_link_or_export_data_path(inventory, tmp_path):
    page = render_fragment(inventory, output=tmp_path, data_links=True)
    links = Links(page).hrefs
    assert any(url.endswith("/canonical/aam/brenda/data.csv.gz") for url in links)
    assert not any(url.endswith("/synthesis/da.csv.gz") for url in links)
    assert page.count('data-admission="held"') == 2  # One source used in two tasks.
    public = public_inventory(inventory)
    held = [table for family in public["families"] for task in family["tasks"]
            for table in task["tables"] if not table["release_eligible"]]
    assert {table["key"] for table in held} == {"synthesis/da"}
    assert all(table["data_path"] is None for table in held)
    assert public["summary"]["admitted_table_views"] == 104
    assert public["summary"]["metadata_only_table_views"] == 1


def test_published_browser_has_no_repository_relative_payload_links(inventory):
    page = render_fragment(inventory, exports={"Current inventory (JSON)": "_static/families.json"})
    links = Links(page).hrefs
    assert "_static/families.json" in links
    assert all(url.startswith(("#", "https://", "http://", "_static/")) for url in links)
    assert "working assumption" not in page
    assert 'class="catalogue-task"' in page
    assert "Unmapped reaction" in page
    assert "energies_path" in page
    assert "forces_path" in page
    assert "9,624,594 atomic configurations" in page
    assert "Corpus with generated negative examples" in page
    assert "Expert feasibility judgments" in page


def test_public_export_uses_current_license_decisions_without_mutating_source(inventory):
    original = copy.deepcopy(inventory)
    public = public_inventory(inventory, repo=REPO)
    tables = {table["key"]: table for family in public["families"]
              for task in family["tasks"] for table in task["tables"]}
    assert tables["aam/golden"]["release_license_expression"] == "GPL-3.0-only AND LGPL-2.1-only AND MIT"
    assert tables["aam/recon3d"]["release_license_expression"] == "CC-BY-NC-4.0 AND MIT"
    assert tables["synthesis/da"]["release_license_expression"] is None
    assert "https://doi.org/10.1038/s41467-024-46364-y" in tables["aam/golden"]["source_urls"]
    task = next(task for family in public["families"] for task in family["tasks"]
                if task["id"] == "reactive_potential")
    assert task["question"].startswith("Atomic configuration")
    assert task["tables"][0]["task_fields"]["targets"] == ["energies_path", "forces_path"]
    assert "working assumption" not in public["scope"]
    assert inventory == original
