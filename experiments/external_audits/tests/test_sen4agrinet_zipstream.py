import struct
import zlib

from experiments.external_audits.sen4agrinet import zipstream


def local(name: bytes, data: bytes, descriptor: str | None = None) -> bytes:
    crc = zlib.crc32(data)
    flags = 0x08 if descriptor else 0
    if descriptor:
        hdr = struct.pack("<4sHHHHHIIIHH", b"PK\x03\x04", 20, flags, 0, 0, 0, 0, 0, 0, len(name), 0)
    else:
        hdr = struct.pack("<4sHHHHHIIIHH", b"PK\x03\x04", 20, flags, 0, 0, 0, crc, len(data), len(data), len(name), 0)
    out = hdr + name + data
    if descriptor == "32":
        out += struct.pack("<4sIII", b"PK\x07\x08", crc, len(data), len(data))
    elif descriptor == "64":
        out += struct.pack("<4sIQQ", b"PK\x07\x08", crc, len(data), len(data))
    return out


def chunked(b: bytes, n: int):
    return (b[i : i + n] for i in range(0, len(b), n))


def run(blob: bytes, wanted: set[str], tmp_path, chunk=7):
    seen = []
    walk = zipstream.walk(
        chunked(blob, chunk),
        lambda h: tmp_path / h.name if h.name in wanted else None,
        lambda h, d, sha: seen.append((h.name, h.uncompressed_size, d is not None)),
    )
    return seen, walk


def test_stored_members_and_skip(tmp_path):
    payload = bytes(range(256)) * 20
    blob = local(b"dir/", b"") + local(b"a.json", payload) + local(b"b.bin", b"x" * 1000) + b"PK\x01\x02" + b"\0" * 40
    seen, _ = run(blob, {"a.json"}, tmp_path)
    assert [s[0] for s in seen] == ["dir/", "a.json", "b.bin"]
    assert (tmp_path / "a.json").read_bytes() == payload
    assert not (tmp_path / "b.bin").exists()


def test_data_descriptor_32_and_64(tmp_path):
    p1 = (b"PK\x07\x08 fake signature inside the data " * 50) + bytes(range(256)) * 100
    p2 = b"second" * 5000
    blob = local(b"one.csv", p1, "32") + local(b"two.csv", p2, "64") + local(b"after.txt", b"tail")
    seen, _ = run(blob, {"one.csv", "two.csv", "after.txt"}, tmp_path, chunk=4099)
    assert (tmp_path / "one.csv").read_bytes() == p1
    assert (tmp_path / "two.csv").read_bytes() == p2
    assert (tmp_path / "after.txt").read_bytes() == b"tail"
    assert [s[1] for s in seen] == [len(p1), len(p2), 4]


def test_descriptor_member_skipped(tmp_path):
    p = b"z" * 3000
    blob = local(b"skip.csv", p, "64") + local(b"keep.json", b"{}")
    seen, _ = run(blob, {"keep.json"}, tmp_path)
    assert seen[0] == ("skip.csv", 3000, False) and (tmp_path / "keep.json").read_bytes() == b"{}"
