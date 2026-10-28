"""Single source of truth for where external audit data and outputs live.

Downloaded external data (temporary, deleted by 99_cleanup.py) go under
EXTERNAL_ROOT/<paper>/. Results and figures live in the repo under
experiments/external_audits/<paper>/{results,figures}/.
"""
from pathlib import Path

EXTERNAL_ROOT = Path("/mnt/bigdrive/external_audits")
AUDIT_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = AUDIT_DIR.parent.parent
NOTES_DIR = REPO_ROOT / "research_notes" / "Missingness confound audit of transfer papers"


def paper_root(paper: str) -> Path:
    root = EXTERNAL_ROOT / paper
    root.mkdir(parents=True, exist_ok=True)
    return root


def results_dir(paper: str) -> Path:
    d = AUDIT_DIR / paper / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d


def figures_dir(paper: str) -> Path:
    d = AUDIT_DIR / paper / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def manifest_path(paper: str) -> Path:
    return paper_root(paper) / "manifest.json"
