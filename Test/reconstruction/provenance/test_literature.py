"""Keep archived upstream evidence separate from owned documentation checks."""

from synrxn.reconstruction.provenance import literature as validate


def test_source_readmes_are_archived_but_owned_document_links_still_checked(tmp_path):
    archived = tmp_path / "coverage/rights_recheck/classification/assets/upstream/README.md"
    archived.parent.mkdir(parents=True)
    archived.write_text("[Upstream example](images/example.png)\n")
    owned = tmp_path / "coverage/license_recheck.md"
    owned.write_text(
        "[Evidence](rights_recheck/classification/assets/upstream/README.md)\n"
        "[Missing project report](missing-report.md)\n"
    )
    # Exclusion applies to the archived evidence root, not any similarly named
    # directory elsewhere in the project's documentation.
    other = tmp_path / "preparation/rights_recheck/notes.md"
    other.parent.mkdir(parents=True)
    other.write_text("[Missing owned note](missing-note.md)\n")

    assert sorted(validate.markdown_link_problems(tmp_path)) == [
        "coverage/license_recheck.md has missing link missing-report.md",
        "preparation/rights_recheck/notes.md has missing link missing-note.md",
    ]
