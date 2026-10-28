import io
import zipfile

from experiments.external_audits.timematch import zipindex


def _build(tmp_path, force_zip64=False):
    buf = io.BytesIO()
    stored = b"stored-bytes-" * 10
    deflated = b"deflate me please " * 50
    with zipfile.ZipFile(buf, "w") as zf:
        zi = zipfile.ZipInfo("a/stored.bin")
        zi.compress_type = zipfile.ZIP_STORED
        with zf.open(zi, "w", force_zip64=force_zip64) as fh:
            fh.write(stored)
        zi2 = zipfile.ZipInfo("meta/blocks/deflated.bin")
        zi2.compress_type = zipfile.ZIP_DEFLATED
        with zf.open(zi2, "w", force_zip64=force_zip64) as fh:
            fh.write(deflated)
    data = buf.getvalue()
    return data, {"a/stored.bin": stored, "meta/blocks/deflated.bin": deflated}


def test_read_central_directory_local_zip(tmp_path):
    data, originals = _build(tmp_path)
    read = lambda s, l: data[s : s + l]  # noqa: E731
    members = zipindex.read_central_directory(read, len(data))
    infos = {i.filename: i for i in zipfile.ZipFile(io.BytesIO(data)).infolist()}
    assert {m.name for m in members} == set(infos)
    for m in members:
        assert m.uncompressed_size == infos[m.name].file_size
        assert m.compressed_size == infos[m.name].compress_size
        assert m.header_offset == infos[m.name].header_offset
        assert m.method == infos[m.name].compress_type


def test_extract_member_stored_and_deflated(tmp_path):
    data, originals = _build(tmp_path)
    read = lambda s, l: data[s : s + l]  # noqa: E731
    members = zipindex.read_central_directory(read, len(data))
    for m in members:
        out = zipindex.extract_member(read, m, tmp_path / m.name)
        assert out.read_bytes() == originals[m.name]


def test_read_central_directory_zip64(tmp_path):
    data, originals = _build(tmp_path, force_zip64=True)
    read = lambda s, l: data[s : s + l]  # noqa: E731
    members = zipindex.read_central_directory(read, len(data))
    assert {m.name for m in members} == set(originals)
    for m in members:
        assert zipindex.extract_member(read, m, tmp_path / m.name).read_bytes() == originals[m.name]
