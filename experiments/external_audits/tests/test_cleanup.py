import pytest

from experiments.external_audits.common import cleanup, paths


def _results(tmp_path, monkeypatch, paper, rows_ok=True):
    res = tmp_path / "results"
    res.mkdir()
    monkeypatch.setattr(paths, "results_dir", lambda p: res)
    for rel, min_rows in cleanup.expected_outputs(paper):
        n = min_rows if rows_ok else max(0, min_rows - 1)
        rel.parent.mkdir(parents=True, exist_ok=True) if False else None
        (res / rel.name).write_text("h\n" + "x\n" * n)
    return res


def test_complete_false_when_missing(tmp_path, monkeypatch):
    res = tmp_path / "results"
    res.mkdir()
    monkeypatch.setattr(paths, "results_dir", lambda p: res)
    ok, problems = cleanup.complete("timematch")
    assert not ok and problems


def test_complete_true_when_present(tmp_path, monkeypatch):
    _results(tmp_path, monkeypatch, "timematch")
    ok, problems = cleanup.complete("timematch")
    assert ok and problems == []


def test_purge_keeps_manifest(tmp_path, monkeypatch):
    _results(tmp_path, monkeypatch, "timematch")
    ext = tmp_path / "ext"
    monkeypatch.setattr(paths, "EXTERNAL_ROOT", ext)
    root = paths.paper_root("timematch")
    (root / "manifest.json").write_text("[]")
    (root / "central_directory.json").write_text("{}")
    (root / "a.bin").write_bytes(b"x" * 10)
    (root / "sub").mkdir()
    (root / "sub" / "b.bin").write_bytes(b"y" * 5)
    freed = cleanup.purge("timematch")
    assert freed == 15
    assert sorted(p.name for p in root.iterdir()) == ["central_directory.json", "manifest.json"]


def test_purge_refuses_when_incomplete(tmp_path, monkeypatch):
    _results(tmp_path, monkeypatch, "timematch", rows_ok=False)
    ext = tmp_path / "ext"
    monkeypatch.setattr(paths, "EXTERNAL_ROOT", ext)
    root = paths.paper_root("timematch")
    (root / "a.bin").write_bytes(b"x")
    with pytest.raises(RuntimeError):
        cleanup.purge("timematch")
    assert (root / "a.bin").exists()
    assert cleanup.purge("timematch", force=True) == 1
