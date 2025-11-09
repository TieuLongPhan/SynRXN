"""
Global constants used by synrxn.data.
"""

CONCEPT_DOI: str = "10.5281/zenodo.17297258"

GH_OWNER: str = "TieuLongPhan"
GH_REPO: str = "SynRXN"

# Zenodo REST API endpoints
ZENODO_RECORD_API: str = "https://zenodo.org/api/records/{record_id}"
ZENODO_SEARCH_API: str = "https://zenodo.org/api/records"

# GitHub URL templates
# NOTE: For raw content, GitHub accepts a branch, tag, or *commit SHA* directly in the URL.
# e.g., https://raw.githubusercontent.com/{owner}/{repo}/{ref}/Data/<task>/<file>
GH_RAW_TPL: str = "https://raw.githubusercontent.com/{owner}/{repo}/{ref}/Data"

# GitHub Contents API supports a `ref` query parameter which can be a branch, tag, or commit SHA.
GH_API_TPL: str = (
    "https://api.github.com/repos/{owner}/{repo}/contents/Data/{task}?ref={ref}"
)
