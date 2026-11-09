"""ISO 9660 for picture CDs: an empty file system, or real files with a Joliet tree.

Burning tools reject an image without a recognisable file system. With no files, five
sectors are enough: primary volume descriptor (LBA 16), terminator (17), the two path tables
(18, 19) and an empty root directory (20); they sit inside the picture's inner margin. With
files, `FileSystem` lays out both directory trees and the file data. Layout per ECMA-119;
numbers are both-endian where the standard says so.
"""

from __future__ import annotations

import datetime
import os
import struct

import numpy as np

EMPTY_SECTORS = range(16, 21)
_PVD, _TERM, _LPATH, _MPATH, _ROOT = EMPTY_SECTORS
def _both32(v: int) -> bytes:
    return struct.pack("<I", v) + struct.pack(">I", v)


def _both16(v: int) -> bytes:
    return struct.pack("<H", v) + struct.pack(">H", v)


def _dir_record(name: bytes, when: datetime.datetime) -> bytes:
    rec = bytearray(33 + len(name) + (1 if len(name) % 2 == 0 else 0))
    rec[0] = len(rec)
    rec[2:10] = _both32(_ROOT)
    rec[10:18] = _both32(2048)
    rec[18:25] = bytes([when.year - 1900, when.month, when.day, when.hour, when.minute, when.second, 0])
    rec[25] = 0x02  # directory
    rec[28:32] = _both16(1)
    rec[32] = len(name)
    rec[33:33 + len(name)] = name
    return bytes(rec)


def _text(s: str, n: int) -> bytes:
    return s.encode("ascii")[:n].ljust(n, b" ")


def _date17(when: datetime.datetime | None) -> bytes:
    if when is None:
        return b"0" * 16 + b"\x00"
    return when.strftime("%Y%m%d%H%M%S").encode() + b"00\x00"


def empty_sectors(total_sectors: int, volume_id: str) -> dict[int, bytes]:
    when = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None, microsecond=0)
    pvd = bytearray(2048)
    pvd[0] = 1
    pvd[1:6] = b"CD001"
    pvd[6] = 1
    pvd[8:40] = _text("", 32)
    pvd[40:72] = _text(volume_id.upper(), 32)
    pvd[80:88] = _both32(total_sectors)
    pvd[120:124] = _both16(1)
    pvd[124:128] = _both16(1)
    pvd[128:132] = _both16(2048)
    pvd[132:140] = _both32(10)
    pvd[140:144] = struct.pack("<I", _LPATH)
    pvd[148:152] = struct.pack(">I", _MPATH)
    pvd[156:190] = _dir_record(b"\x00", when)
    pvd[190:813] = b" " * (813 - 190)
    pvd[813:830] = _date17(when)
    pvd[830:847] = _date17(when)
    pvd[847:864] = _date17(None)
    pvd[864:881] = _date17(None)
    pvd[881] = 1

    term = bytearray(2048)
    term[0] = 255
    term[1:6] = b"CD001"
    term[6] = 1

    lpath = bytearray(2048)
    lpath[0:10] = bytes([1, 0]) + struct.pack("<I", _ROOT) + struct.pack("<H", 1) + b"\x00\x00"
    mpath = bytearray(2048)
    mpath[0:10] = bytes([1, 0]) + struct.pack(">I", _ROOT) + struct.pack(">H", 1) + b"\x00\x00"

    root = bytearray(2048)
    dot, dotdot = _dir_record(b"\x00", when), _dir_record(b"\x01", when)
    root[0:len(dot)] = dot
    root[len(dot):len(dot) + len(dotdot)] = dotdot
    return {_PVD: bytes(pvd), _TERM: bytes(term), _LPATH: bytes(lpath), _MPATH: bytes(mpath), _ROOT: bytes(root)}


def write_empty(path: str, total_sectors: int, volume_id: str) -> None:
    with open(path, "r+b") as f:
        for lba, data in empty_sectors(total_sectors, volume_id).items():
            f.seek(lba * 2048)
            f.write(data)


# A file system with real files, for a picture that leaves room for data on the inner zone.
# Primary tree with 8.3 names (any reader) plus a Joliet tree with the real names (Windows,
# Linux, macOS). Both trees point at the same file extents. Layout from LBA 16: PVD, Joliet
# SVD, terminator, four path tables, primary directories, Joliet directories, file data.
# With `allowed` sectors (a picture/data mask), file data goes only into those sectors and a
# file that crosses a gap is written as several extents (ECMA-119 9.1.6 multi-extent bit).

JOLIET_BAD = set('*/:;?\\')
_D_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")


class _Node:
    def __init__(self, name: str, src: str | None, is_dir: bool, when: datetime.datetime):
        self.name, self.src, self.is_dir, self.when = name, src, is_dir, when
        self.size = 0 if is_dir else os.path.getsize(src)
        self.children: list[_Node] = []
        self.parent: _Node | None = None
        self.iso = b""        # primary identifier
        self.jol = b""        # Joliet identifier (UCS-2 big-endian)
        self.lba = 0          # file extent, or primary directory extent
        self.extents: list[tuple[int, int]] = []  # file data as (lba, bytes); several when masked
        self.jlba = 0         # Joliet directory extent
        self.number = 0       # path table directory number
        self.sectors = 0      # primary directory size in sectors
        self.jsectors = 0


def _d_name(s: str, n: int) -> str:
    return "".join(ch if ch in _D_CHARS else "_" for ch in s.upper())[:n]


def _assign_names(d: _Node) -> None:
    used = set()
    for ch in d.children:
        if ch.is_dir:
            stem, ext = _d_name(ch.name, 8) or "_", ""
        else:
            base, dot, e = ch.name.rpartition(".")
            stem, ext = (_d_name(base, 8), _d_name(e, 3)) if dot and base else (_d_name(ch.name, 8), "")
            stem = stem or "_"
        cand, k = stem, 1
        while (cand, ext) in used:
            tail = f"~{k}"
            cand, k = stem[:8 - len(tail)] + tail, k + 1
        used.add((cand, ext))
        ch.iso = cand.encode() if ch.is_dir else f"{cand}.{ext};1".encode()
        if any(c in JOLIET_BAD or ord(c) > 0xFFFF or ord(c) < 32 for c in ch.name) or len(ch.name) > 64:
            raise ValueError(f"file name not allowed on a Joliet disc (max 64 chars, none of * / : ; ? \\): {ch.name!r}")
        ch.jol = ch.name.encode("utf-16-be")


def _scan(path: str, name: str, depth: int) -> _Node:
    when = datetime.datetime.fromtimestamp(os.path.getmtime(path), datetime.timezone.utc).replace(tzinfo=None)
    if os.path.isfile(path):
        return _Node(name, path, False, when)
    if not os.path.isdir(path):
        raise ValueError(f"not a regular file or directory: {path}")
    if depth > 7:
        raise ValueError(f"directories nested deeper than ISO 9660 allows (8 levels): {path}")
    node = _Node(name, None, True, when)
    for entry in sorted(os.listdir(path)):
        child = _scan(os.path.join(path, entry), entry, depth + 1)
        child.parent = node
        node.children.append(child)
    _assign_names(node)
    node.children.sort(key=lambda c: c.iso)
    return node


def _record(node: _Node, ident: bytes, lba: int, size: int, more: bool = False) -> bytes:
    rec = bytearray(33 + len(ident) + (1 if len(ident) % 2 == 0 else 0))
    rec[0] = len(rec)
    rec[2:10] = _both32(lba)
    rec[10:18] = _both32(size)
    w = node.when
    rec[18:25] = bytes([w.year - 1900, w.month, w.day, w.hour, w.minute, w.second, 0])
    rec[25] = (0x02 if node.is_dir else 0x00) | (0x80 if more else 0x00)
    rec[28:32] = _both16(1)
    rec[32] = len(ident)
    rec[33:33 + len(ident)] = ident
    return bytes(rec)


def _dir_records(d: _Node, joliet: bool) -> list[bytes]:
    def ext(n):
        return (n.jlba, n.jsectors * 2048) if joliet else (n.lba, n.sectors * 2048)
    parent = d.parent or d
    recs = [_record(d, b"\x00", *ext(d)), _record(parent, b"\x01", *ext(parent))]
    kids = sorted(d.children, key=lambda c: c.jol) if joliet else d.children
    for c in kids:
        ident = c.jol if joliet else c.iso
        if c.is_dir:
            recs.append(_record(c, ident, *ext(c)))
            continue
        extents = c.extents or [(c.lba, c.size)]
        recs += [_record(c, ident, lba, size, more=i < len(extents) - 1) for i, (lba, size) in enumerate(extents)]
    return recs


def _dir_sectors(recs: list[bytes]) -> int:
    used, n = 0, 1
    for r in recs:
        if used + len(r) > 2048:
            n, used = n + 1, 0
        used += len(r)
    return n


def _pack_dir(recs: list[bytes], n: int) -> bytes:
    out, used = bytearray(), 0
    for r in recs:
        if used + len(r) > 2048:
            out += bytes(2048 - used)
            used = 0
        out += r
        used += len(r)
    return bytes(out).ljust(n * 2048, b"\x00")


def _path_table(dirs: list[_Node], joliet: bool, endian: str) -> bytes:
    out = bytearray()
    for d in dirs:
        name = b"\x00" if d.parent is None else (d.jol if joliet else d.iso)
        out += bytes([len(name), 0]) + struct.pack(endian + "I", d.jlba if joliet else d.lba)
        out += struct.pack(endian + "H", (d.parent or d).number) + name + (b"\x00" if len(name) % 2 else b"")
    return bytes(out)


class FileSystem:
    """ISO 9660 + Joliet image of a directory, laid out from LBA 16.

    `end_lba` is the first sector after the file system and file data; a picture may only
    start after it (the CD writer also checks the CIRC footprint of that last sector).
    With `allowed` (ascending LBAs free for data, e.g. from a picture/data mask) file data
    fills only those sectors, inner first, and `end_lba` is just past the last one used;
    the directories still sit contiguously from LBA 16 and end at `meta_end`.
    With `by_size` the data goes in smallest file first (ties in listing order), so the biggest
    files sit furthest out; the directory listing keeps its usual order.
    """

    def __init__(self, src_dir: str, volume_id: str = "CDPICTURE", allowed=None, by_size=False):
        if not volume_id or len(volume_id) > 16 or any(ch in JOLIET_BAD or ord(ch) > 0xFFFF for ch in volume_id):
            raise ValueError(f"volume label must be 1-16 characters, none of * / : ; ? \\ (Joliet): {volume_id!r}")
        if not os.path.isdir(src_dir):
            raise ValueError(f"data directory not found: {src_dir}")
        self.volume_id = volume_id
        self.root = _scan(src_dir, "", 0)
        dirs, queue = [], [self.root]
        while queue:  # breadth first: path table order (level, parent number, name)
            d = queue.pop(0)
            d.number = len(dirs) + 1
            dirs.append(d)
            queue += [c for c in d.children if c.is_dir]
        if len(dirs) > 65535:
            raise ValueError("too many directories for an ISO 9660 path table")
        self.dirs = dirs
        self.files = []
        stack = [self.root]
        while stack:
            d = stack.pop()
            self.files += [c for c in d.children if not c.is_dir]
            stack += reversed([c for c in d.children if c.is_dir])
        self.allowed = None if allowed is None else np.asarray(allowed, np.int64)
        self.by_size = by_size
        self.pt_bytes = len(_path_table(dirs, False, "<"))
        self.jpt_bytes = len(_path_table(dirs, True, "<"))
        pt, jpt = -(-self.pt_bytes // 2048), -(-self.jpt_bytes // 2048)
        self.l_pt, self.m_pt, self.l_jpt, self.m_jpt = 19, 19 + pt, 19 + 2 * pt, 19 + 2 * pt + jpt
        # Directory sizes depend on extent counts only, which depend on where the directories
        # end: size, place, and repeat until the sizes settle (once, unless masked). Sizes never
        # shrink between rounds, so a fragmented mask can't make them oscillate; a directory may
        # keep a spare zero-padded sector.
        sizes = None
        for _ in range(50):
            for d in dirs:
                d.sectors = max(d.sectors, _dir_sectors(_dir_records(d, False)))
                d.jsectors = max(d.jsectors, _dir_sectors(_dir_records(d, True)))
            if sizes == (now := [(d.sectors, d.jsectors) for d in dirs]):
                break
            sizes = now
            lba = 19 + 2 * pt + 2 * jpt
            for d in dirs:
                d.lba, lba = lba, lba + d.sectors
            for d in dirs:
                d.jlba, lba = lba, lba + d.jsectors
            self.meta_end = lba
            self._place(lba)
        else:
            raise ValueError("directory layout did not settle")

    def _place(self, start: int) -> None:
        order = sorted(self.files, key=lambda f: f.size) if self.by_size else self.files
        if self.allowed is None:
            lba = start
            for f in order:
                f.lba = lba if f.size else 0
                f.extents = [(f.lba, f.size)]
                lba += -(-f.size // 2048)
            self.end_lba = lba
            return
        pool = self.allowed[self.allowed >= start]
        need = sum(-(-f.size // 2048) for f in self.files)
        if need > len(pool):
            raise ValueError(f"files need {need} sectors ({need * 2048 / 1e6:.1f} MB), the data area has "
                             f"{len(pool)} ({len(pool) * 2048 / 1e6:.1f} MB)")
        i = 0
        for f in order:
            k = -(-f.size // 2048)
            run = pool[i:i + k]
            i += k
            if not k:
                f.lba, f.extents = 0, [(0, 0)]
                continue
            cuts = np.flatnonzero(np.diff(run) != 1) + 1
            starts, lengths = run[np.r_[0, cuts]], np.diff(np.r_[0, cuts, k])
            f.extents = [(int(a), int(n) * 2048) for a, n in zip(starts, lengths)]
            f.extents[-1] = (f.extents[-1][0], f.size - 2048 * (k - int(lengths[-1])))
            f.lba = f.extents[0][0]
        self.end_lba = int(pool[i - 1]) + 1 if i else start

    def data_sectors(self, total_sectors: int) -> np.ndarray:
        """True for every sector holding file-system metadata or file data."""
        used = np.zeros(total_sectors, bool)
        used[16:self.meta_end] = True
        for f in self.files:
            for lba, size in f.extents:
                used[lba:lba + -(-size // 2048)] = True
        return used

    def entries(self) -> list[dict]:
        def rel(n):
            parts = []
            while n.parent is not None:
                parts.append(n.name)
                n = n.parent
            return "/".join(reversed(parts))
        return [{"path": rel(f), "lba": f.lba, "bytes": f.size, "extents": len(f.extents)} for f in self.files]

    def _descriptor(self, joliet: bool, total_sectors: int, when: datetime.datetime) -> bytes:
        vd = bytearray(2048)
        vd[0] = 2 if joliet else 1
        vd[1:6] = b"CD001"
        vd[6] = 1
        if joliet:
            vd[8:40] = " ".encode("utf-16-be") * 16
            vd[40:72] = self.volume_id.ljust(16).encode("utf-16-be")
            vd[88:91] = b"%/E"
        else:
            vd[8:40] = _text("", 32)
            vd[40:72] = _text(_d_name(self.volume_id, 32), 32)
        vd[80:88] = _both32(total_sectors)
        vd[120:124] = _both16(1)
        vd[124:128] = _both16(1)
        vd[128:132] = _both16(2048)
        vd[132:140] = _both32(self.jpt_bytes if joliet else self.pt_bytes)
        vd[140:144] = struct.pack("<I", self.l_jpt if joliet else self.l_pt)
        vd[148:152] = struct.pack(">I", self.m_jpt if joliet else self.m_pt)
        vd[156:190] = _dir_records(self.root, joliet)[0]
        vd[190:702] = " ".encode("utf-16-be") * 256 if joliet else b" " * 512
        vd[702:813] = b" " * 111 if not joliet else bytes(111)
        vd[813:830] = _date17(when)
        vd[830:847] = _date17(when)
        vd[847:864] = _date17(None)
        vd[864:881] = _date17(None)
        vd[881] = 1
        return bytes(vd)

    def write(self, path: str, total_sectors: int) -> None:
        """Overwrite LBA 16 .. end_lba-1 of an existing image with the file system and files."""
        if self.end_lba > total_sectors:
            raise ValueError(f"files need {self.end_lba} sectors, the image has {total_sectors}")
        when = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None, microsecond=0)
        term = bytearray(2048)
        term[0:7] = b"\xffCD001\x01"
        meta = {16: self._descriptor(False, total_sectors, when), 17: self._descriptor(True, total_sectors, when),
                18: bytes(term)}
        for lba, joliet, endian in ((self.l_pt, False, "<"), (self.m_pt, False, ">"),
                                    (self.l_jpt, True, "<"), (self.m_jpt, True, ">")):
            table = _path_table(self.dirs, joliet, endian)
            meta[lba] = table.ljust(-(-len(table) // 2048) * 2048, b"\x00")
        for d in self.dirs:
            meta[d.lba] = _pack_dir(_dir_records(d, False), d.sectors)
            meta[d.jlba] = _pack_dir(_dir_records(d, True), d.jsectors)
        with open(path, "r+b") as out:
            for lba, data in meta.items():
                out.seek(lba * 2048)
                out.write(data)
            for f in self.files:
                if not f.size:
                    continue
                with open(f.src, "rb") as src:
                    for lba, size in f.extents:
                        out.seek(lba * 2048)
                        left = size
                        while left:
                            chunk = src.read(min(left, 1 << 20))
                            if not chunk:
                                raise ValueError(f"file shrank while writing: {f.src}")
                            out.write(chunk)
                            left -= len(chunk)
                    out.write(bytes(-f.size % 2048))  # only the last extent is partial

    def verify(self, path: str) -> list[str]:
        """Compare every file's bytes in the image with its source file."""
        problems = []
        with open(path, "rb") as img:
            for f in self.files:
                with open(f.src, "rb") as src:
                    for lba, size in f.extents:
                        img.seek(lba * 2048)
                        left = size
                        while left:
                            n = min(left, 1 << 20)
                            if img.read(n) != src.read(n):
                                problems.append(f"file content differs: {f.src}")
                                left = -1
                                break
                            left -= n
                        if left < 0:
                            break
        return problems
