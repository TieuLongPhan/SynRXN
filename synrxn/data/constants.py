"""
Global constants used by synrxn.data.
"""

CONCEPT_DOI: str = "10.5281/zenodo.17297258"

GH_OWNER: str = "TieuLongPhan"
GH_REPO: str = "SynRXN"

ZENODO_RECORD_API: str = "https://zenodo.org/api/records/{record_id}"
ZENODO_SEARCH_API: str = "https://zenodo.org/api/records"

GH_RAW_TPL: str = (
    "https://raw.githubusercontent.com/{owner}/{repo}/refs/{ref_type}/{ref}/Data"
)
GH_API_TPL: str = (
    "https://api.github.com/repos/{owner}/{repo}/contents/Data/{task}?ref={ref}"
)
