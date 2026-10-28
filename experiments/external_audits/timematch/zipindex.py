"""Read a (Zip64) zip central directory through a range-read callable and
extract single members, so small files can be pulled out of a 74 GB archive
without downloading it."""
from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

Reader = Callable[[int, int], bytes]  # read(start, length) -> bytes

_EOCD_SIG = b"PK\x05\x06"
_EOCD64_LOC_SIG = b"PK\x06\x07"
_EOCD64_SIG = b"PK\x06\x06"
_CD_SIG = b"PK\x01\x02"
_LOCAL_SIG = b"PK\x03\x04"


@dataclass(frozen=True)
class ZipMember:
    name: str
    header_offset: int
    compressed_size: int
    uncompressed_size: int
    method: int


def _find_eocd(read: Reader, total_size: int) -> tuple[bytes, int]:
    tail_len = min(total_size, 65536 + 22)
    tail_start = total_size - tail_len
    tail = read(tail_start, tail_len)
    pos = tail.rfind(_EOCD_SIG)
    if pos < 0:
        raise ValueError("end of central directory signature not found")
    return tail, tail_start + pos


def read_central_directory(read: Reader, total_size: int) -> list[ZipMember]:
    tail, eocd_abs = _find_eocd(read, total_size)
    eocd = tail[eocd_abs - (total_size - len(tail)) :][:22]
    (_, _, _, _, cd_entries, cd_size, cd_offset, _) = struct.unpack("<4sHHHHIIH", eocd)
    # Zip64: locator sits 20 bytes before the EOCD when any field is saturated.
    loc_abs = eocd_abs - 20
    if loc_abs >= 0:
        tail_start = total_size - len(tail)
        loc = tail[loc_abs - tail_start : loc_abs - tail_start + 20] if loc_abs >= tail_start else read(loc_abs, 20)
        if loc[:4] == _EOCD64_LOC_SIG:
            eocd64_off = struct.unpack("<Q", loc[8:16])[0]
            rec = read(eocd64_off, 56)
            if rec[:4] != _EOCD64_SIG:
                raise ValueError("bad Zip64 end of central directory record")
            cd_entries = struct.unpack("<Q", rec[32:40])[0]
            cd_size = struct.unpack("<Q", rec[40:48])[0]
            cd_offset = struct.unpack("<Q", rec[48:56])[0]
    cd = read(cd_offset, cd_size)
    members: list[ZipMember] = []
    p = 0
    for _ in range(cd_entries):
        if cd[p : p + 4] != _CD_SIG:
            raise ValueError(f"bad central directory entry at {p}")
        # fields: 10 method, 20 csize, 24 usize, 28 nlen, 30 xlen, 32 clen, 42 header offset
        method = struct.unpack("<H", cd[p + 10 : p + 12])[0]
        csize, usize = struct.unpack("<II", cd[p + 20 : p + 28])
        nlen, xlen, clen = struct.unpack("<HHH", cd[p + 28 : p + 34])
        hoff = struct.unpack("<I", cd[p + 42 : p + 46])[0]
        name = cd[p + 46 : p + 46 + nlen].decode("utf-8", "replace")
        extra = cd[p + 46 + nlen : p + 46 + nlen + xlen]
        if 0xFFFFFFFF in (csize, usize, hoff):
            usize, csize, hoff = _zip64_extra(extra, usize, csize, hoff)
        members.append(ZipMember(name, hoff, csize, usize, method))
        p += 46 + nlen + xlen + clen
    return members


def _zip64_extra(extra: bytes, usize: int, csize: int, hoff: int) -> tuple[int, int, int]:
    q = 0
    while q + 4 <= len(extra):
        tag, size = struct.unpack("<HH", extra[q : q + 4])
        body = extra[q + 4 : q + 4 + size]
        if tag == 0x0001:
            r = 0
            if usize == 0xFFFFFFFF:
                usize = struct.unpack("<Q", body[r : r + 8])[0]
                r += 8
            if csize == 0xFFFFFFFF:
                csize = struct.unpack("<Q", body[r : r + 8])[0]
                r += 8
            if hoff == 0xFFFFFFFF:
                hoff = struct.unpack("<Q", body[r : r + 8])[0]
                r += 8
            break
        q += 4 + size
    return usize, csize, hoff


def member_data_offset(read: Reader, member: ZipMember) -> int:
    hdr = read(member.header_offset, 30)
    if hdr[:4] != _LOCAL_SIG:
        raise ValueError(f"bad local header for {member.name}")
    nlen, xlen = struct.unpack("<HH", hdr[26:30])
    return member.header_offset + 30 + nlen + xlen


def extract_member(read: Reader, member: ZipMember, dest: Path) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    start = member_data_offset(read, member)
    raw = read(start, member.compressed_size) if member.compressed_size else b""
    if member.method == 0:
        data = raw
    elif member.method == 8:
        data = zlib.decompressobj(-15).decompress(raw)
    else:
        raise ValueError(f"unsupported compression method {member.method} for {member.name}")
    if len(data) != member.uncompressed_size:
        raise IOError(f"{member.name}: got {len(data)} bytes, expected {member.uncompressed_size}")
    dest.write_bytes(data)
    return dest
