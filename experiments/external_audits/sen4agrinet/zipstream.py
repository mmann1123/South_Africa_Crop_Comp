"""Forward-only reader for a zip that is streamed over HTTP.

Dropbox builds the S4A-Models.zip folder download on the fly: it ignores Range
requests and has no usable central directory, but the members are stored
(method 0) with their sizes in the local file headers. So the archive can be
walked once from the front, keeping only the members a caller selects and
discarding the rest as they pass. Nothing is written for skipped members.
"""
from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

_LOCAL = b"PK\x03\x04"
_CD = b"PK\x01\x02"
_EOCD = b"PK\x05\x06"
_ZIP64_EOCD = b"PK\x06\x06"
_DESC = b"PK\x07\x08"


@dataclass(frozen=True)
class Header:
    name: str
    method: int
    flags: int
    compressed_size: int
    uncompressed_size: int


class Stream:
    """Buffered forward reader over an iterator of byte chunks."""

    def __init__(self, chunks: Iterator[bytes]):
        self._it = iter(chunks)
        self._buf = b""
        self.pos = 0

    def read(self, n: int) -> bytes:
        while len(self._buf) < n:
            try:
                self._buf += next(self._it)
            except StopIteration:
                break
        out, self._buf = self._buf[:n], self._buf[n:]
        self.pos += len(out)
        return out

    def iter_until_descriptor(self, block: int = 1 << 22) -> Iterator[bytes]:
        """Yield the data of a stored member whose sizes follow it in a data
        descriptor (PK 07 08, crc32, compressed size, uncompressed size, with
        4-byte or 8-byte sizes). A signature is accepted only when the sizes
        that follow equal the number of data bytes already seen, which makes a
        chance match inside the data negligible. The descriptor is consumed."""
        seen = 0
        eof = False
        while True:
            while not eof and len(self._buf) < block + 64:
                try:
                    self._buf += next(self._it)
                except StopIteration:
                    eof = True
            buf = self._buf
            start = 0
            while True:
                i = buf.find(_DESC, start)
                if i < 0 or len(buf) - i < 24:
                    break
                here = seen + i
                tail = buf[i : i + 24]
                c64, u64 = struct.unpack("<QQ", tail[8:24])
                c32, u32 = struct.unpack("<II", tail[8:16])
                width = 0
                if c64 == here and u64 == here:
                    width = 24
                elif c32 == here and u32 == here:
                    width = 16
                if width:
                    if i:
                        yield buf[:i]
                    self._buf = buf[i + width :]
                    self.pos += i + width
                    return
                start = i + 1
            if eof:
                raise EOFError("stream ended inside a member whose size is in a data descriptor")
            emit = len(buf) - 27
            yield buf[:emit]
            seen += emit
            self.pos += emit
            self._buf = buf[emit:]

    def iter_bytes(self, n: int, block: int = 1 << 22) -> Iterator[bytes]:
        left = n
        while left > 0:
            b = self.read(min(block, left))
            if not b:
                raise EOFError(f"stream ended with {left} bytes of a member unread")
            left -= len(b)
            yield b


def _zip64_sizes(extra: bytes, usize: int, csize: int) -> tuple[int, int]:
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
            return usize, csize
        q += 4 + size
    raise ValueError("zip64 sizes expected but no zip64 extra field")


def read_header(stream: Stream) -> Header | None:
    """Next local file header, or None at the central directory / end of data."""
    sig = stream.read(4)
    if sig in (_CD, _EOCD, _ZIP64_EOCD, b""):
        return None
    if sig != _LOCAL:
        raise ValueError(f"unexpected signature {sig!r} at stream offset {stream.pos - 4}")
    fixed = stream.read(26)
    _ver, flags, method, _t, _d, _crc, csize, usize, nlen, xlen = struct.unpack("<HHHHHIIIHH", fixed)
    name = stream.read(nlen).decode("utf-8", "replace")
    extra = stream.read(xlen)
    if 0xFFFFFFFF in (csize, usize):
        usize, csize = _zip64_sizes(extra, usize, csize)
    if method != 0:
        raise ValueError(f"{name}: compression method {method}, only stored members are supported")
    return Header(name, method, flags, csize, usize)


def walk(
    chunks: Iterator[bytes],
    want: Callable[[Header], Path | None],
    on_member: Callable[[Header, Path | None, str | None], None] | None = None,
) -> int:
    """Walk the archive once. `want(header)` returns a destination path to keep
    the member (written atomically via .part, SHA-256 computed on the way) or
    None to skip it. Calls on_member(header, dest, sha256) per member. Returns
    the number of stream bytes consumed."""
    stream = Stream(chunks)
    while True:
        h = read_header(stream)
        if h is None:
            return stream.pos
        unknown = bool(h.flags & 0x08)
        isdir = h.name.endswith("/")
        dest = None if isdir else want(h)
        data = stream.iter_until_descriptor() if unknown else stream.iter_bytes(h.compressed_size)
        if dest is None:
            n = 0
            for b in data:
                n += len(b)
            if on_member:
                on_member(Header(h.name, h.method, h.flags, n, n), None, None)
            continue
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_suffix(dest.suffix + ".part")
        sha = hashlib.sha256()
        n = 0
        with open(part, "wb") as fh:
            for b in data:
                fh.write(b)
                sha.update(b)
                n += len(b)
        part.replace(dest)
        if on_member:
            on_member(Header(h.name, h.method, h.flags, n, n), dest, sha.hexdigest())
