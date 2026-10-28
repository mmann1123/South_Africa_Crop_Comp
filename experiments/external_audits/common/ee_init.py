"""Earth Engine initialisation using the project stored by `earthengine authenticate`."""
from __future__ import annotations

import json
from pathlib import Path

CREDENTIALS = Path.home() / ".config" / "earthengine" / "credentials"


def project_id() -> str:
    if not CREDENTIALS.exists():
        raise RuntimeError(
            f"{CREDENTIALS} not found; run `earthengine authenticate --auth_mode=notebook` in the ee env"
        )
    cred = json.loads(CREDENTIALS.read_text())
    proj = cred.get("project")
    if not proj:
        raise RuntimeError(f"{CREDENTIALS} has no 'project' key; re-authenticate choosing a Cloud project")
    return str(proj)


def init() -> str:
    import ee

    proj = project_id()
    ee.Initialize(project=proj)
    return proj
