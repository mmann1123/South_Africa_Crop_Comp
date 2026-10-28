import hashlib
import json

import pytest

from experiments.external_audits.common import fetch, paths, runlog


class _StubResponse:
    def __init__(self, body: bytes, status: int = 200):
        self._body = body
        self.status_code = status
        self.headers = {"Content-Length": str(len(body))}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size):
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i : i + chunk_size]

    @property
    def content(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def stub_http(monkeypatch):
    calls = []

    def fake_get(url, headers=None, stream=False, timeout=None, **kw):
        calls.append({"url": url, "headers": headers or {}})
        body = b"hello"
        rng = (headers or {}).get("Range")
        if rng:
            start, end = rng.replace("bytes=", "").split("-")
            body = b"0123456789abcdefghij"[int(start) : int(end) + 1]
            return _StubResponse(body, 206)
        return _StubResponse(body)

    monkeypatch.setattr(fetch.requests, "get", fake_get)
    return calls


def test_download_writes_manifest_and_sha(tmp_path, stub_http):
    manifest = tmp_path / "manifest.json"
    dest = fetch.download("http://x/hello", tmp_path / "hello.bin", manifest, key="k/hello")
    assert dest.read_bytes() == b"hello"
    entries = fetch.manifest_entries(manifest)
    assert entries[0]["key"] == "k/hello"
    assert entries[0]["sha256"] == hashlib.sha256(b"hello").hexdigest()
    assert entries[0]["bytes"] == 5
    assert entries[0]["url"] == "http://x/hello"
    assert "fetched_at" in entries[0]


def test_download_skips_when_present(tmp_path, stub_http):
    manifest = tmp_path / "manifest.json"
    fetch.download("http://x/hello", tmp_path / "hello.bin", manifest, key="k/hello")
    fetch.download("http://x/hello", tmp_path / "hello.bin", manifest, key="k/hello")
    assert len(stub_http) == 1
    assert len(fetch.manifest_entries(manifest)) == 1


def test_range_get_sends_range_header(stub_http):
    data = fetch.range_get("http://x/big", 10, 10)
    assert stub_http[0]["headers"]["Range"] == "bytes=10-19"
    assert data == b"abcdefghij"


def test_runlog_writes_json(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "results_dir", lambda paper: tmp_path)
    with runlog.RunLog("x", {"a": 1}, paper="demo") as log:
        log.count("rows", 3)
        log.add("note", "ok")
    out = json.loads((tmp_path / "run_log_x.json").read_text())
    assert out["script"] == "x"
    assert out["args"] == {"a": 1}
    assert out["seed"] == 42
    assert out["counts"] == {"rows": 3}
    assert out["note"] == "ok"
    assert "git_commit" in out and "packages" in out and "started" in out and "ended" in out


def test_paper_root_creates_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "EXTERNAL_ROOT", tmp_path)
    root = paths.paper_root("denethor")
    assert root == tmp_path / "denethor"
    assert root.is_dir()


def test_download_retries_on_server_error(tmp_path, monkeypatch):
    attempts = []

    def flaky_get(url, headers=None, stream=False, timeout=None, **kw):
        attempts.append(url)
        if len(attempts) < 3:
            return _StubResponse(b"", 520)
        return _StubResponse(b"hello")

    monkeypatch.setattr(fetch.requests, "get", flaky_get)
    monkeypatch.setattr(fetch.time, "sleep", lambda s: None)
    manifest = tmp_path / "manifest.json"
    dest = fetch.download("http://x/flaky", tmp_path / "f.bin", manifest, key="k/f")
    assert dest.read_bytes() == b"hello"
    assert len(attempts) == 3
