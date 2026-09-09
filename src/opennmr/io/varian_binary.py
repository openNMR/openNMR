"""
Varian/Agilent ``fid`` + ``procpar`` reading, without ``nmrglue``.

Replaces ``nmrglue.varian.read``/``read_procpar`` for the two shapes
this project actually asks for -- a 1D FID and a 2D array read
``as_2d`` -- so the Varian importer carries no third-party dependency
for its raw read. Deliberately *not* a reimplementation of nmrglue's
full generality: its ``find_shape``/``find_torder``/``reorder_data``
machinery handles 3D/4D and imaging trace orderings that
:mod:`opennmr.io.varian` never requests (it passes ``as_2d=True`` for
2D and squeezes single-block data for 1D), and guessing at formats
there is no data here to check against is how subtly wrong readers get
written.

Every structural constant below is taken from the format itself, and
the whole module is pinned by a test asserting **byte-identical**
output against ``nmrglue`` on this project's own real Varian datasets.
"""

from __future__ import annotations

import os
import struct

import numpy as np

#: The 32-byte file header: six 4-byte ints, two 2-byte shorts, one
#: 4-byte int, big-endian -- ``nblocks``, ``ntraces``, ``np``,
#: ``ebytes``, ``tbytes``, ``bbytes``, ``vers_id``, ``status``,
#: ``nbheaders``. Varian files are always big-endian.
_FILE_HEADER = ">6lhhl"
_FILE_HEADER_SIZE = 32

#: Each block is preceded by ``nbheaders`` block headers of 28 bytes.
#: They carry per-block scaling/status that this project does not use,
#: so they are skipped rather than parsed -- matching nmrglue's own
#: ``skip_blockheader``.
_BLOCK_HEADER_SIZE = 28

#: Status bits of the file header's ``status`` word that decide the
#: on-disk sample type.
_S_32 = 0x4
_S_FLOAT = 0x8


def _sample_dtype(status: int) -> np.dtype:
    """The real (not complex) sample type the file stores.

    Float wins over the 32-bit flag when both are set, which is the
    order Varian's own status word implies and nmrglue follows."""
    if status & _S_FLOAT:
        return np.dtype(">f4")
    if status & _S_32:
        return np.dtype(">i4")
    return np.dtype(">i2")


def _complex_dtype(sample: np.dtype) -> str:
    """Complex width to pair with a given sample type.

    int32 gets complex128 while int16/float32 get complex64 --
    nmrglue's own mapping, kept because widening float32 to complex128
    would make byte-comparison against it fail on precision alone."""
    return "complex128" if sample == np.dtype(">i4") else "complex64"


def read_fid(path: str, as_2d: bool = False) -> tuple[dict, np.ndarray]:
    """Read a Varian ``fid`` binary into ``(header, complex_data)``.

    Returns the file header as a dict and the de-interleaved complex
    data, shaped ``(nblocks, np // 2)``. With ``as_2d=False`` a
    single-block file is squeezed to 1D, which is what a 1D
    acquisition is; a multi-block file is returned 2D either way
    rather than being reshaped on a guess.
    """
    with open(path, "rb") as handle:
        values = struct.unpack(_FILE_HEADER, handle.read(_FILE_HEADER_SIZE))
        header = dict(zip(
            ("nblocks", "ntraces", "np", "ebytes", "tbytes", "bbytes",
             "vers_id", "status", "nbheaders"),
            values,
        ))
        if header["ntraces"] != 1:
            raise NotImplementedError(
                f"{path!r} stores {header['ntraces']} traces per block; only "
                f"single-trace blocks are supported. nmrglue's own "
                f"read_fid_ntraces covers this case if it is ever needed."
            )
        sample = _sample_dtype(header["status"])
        points = header["np"]
        raw = np.empty((header["nblocks"], points), dtype=sample)
        for block in range(header["nblocks"]):
            handle.seek(_BLOCK_HEADER_SIZE * header["nbheaders"], os.SEEK_CUR)
            raw[block] = np.frombuffer(handle.read(points * sample.itemsize), sample)

    # Interleaved real/imaginary pairs. Built exactly as nmrglue does
    # -- real part first, imaginary multiplied into the target complex
    # type -- so the result is bit-identical rather than merely equal.
    complex_type = _complex_dtype(sample)
    data = raw[..., ::2] + np.array(raw[..., 1::2] * 1.0j, dtype=complex_type)

    if not as_2d and data.shape[0] == 1:
        return header, np.squeeze(data)
    return header, data


def read_procpar(path: str) -> dict:
    """Parse a Varian ``procpar`` into ``{name: {...}}``.

    Each parameter is three or more lines: a header line of eleven
    fields, a values line (or, for strings, one line per value), and an
    enumerables line. Values stay **strings**, exactly as nmrglue
    leaves them -- the importer's own ``_procpar_value`` does the
    casting, and converting here would change what every caller sees.
    """
    fields = ("name", "subtype", "basictype", "maxvalue", "minvalue",
              "stepsize", "Ggroup", "Dgroup", "protection", "active", "intptr")
    parameters: dict[str, dict] = {}
    with open(path, "rb") as handle:
        size = os.stat(path).st_size
        while handle.tell() != size:
            header = handle.readline().decode().split()
            if not header:
                continue
            entry = dict(zip(fields, header))

            line = handle.readline().decode()
            count = int(line.split()[0])
            if entry["basictype"] == "1":        # numbers: all on one line
                entry["values"] = line.split()[1:]
            elif entry["basictype"] == "2":      # strings: one per line
                values = [line.split('"')[1]]
                for _ in range(count - 1):
                    values.append(handle.readline().decode().split('"')[1])
                entry["values"] = values
            else:
                entry["values"] = []

            line = handle.readline().decode()
            entry["enumerable"] = line.split()[0]
            if entry["enumerable"] != "0":
                if entry["basictype"] == "1":
                    entry["enumerables"] = line.split()[1:]
                elif entry["basictype"] == "2":
                    entry["enumerables"] = line.split('"')[1::2]

            parameters[entry["name"]] = entry
    return parameters


def read(path: str, as_2d: bool = False) -> tuple[dict, np.ndarray]:
    """``nmrglue.varian.read``'s shape, for the calls this project
    makes: ``(dic, data)`` with the procpar under ``dic["procpar"]``."""
    header, data = read_fid(os.path.join(path, "fid"), as_2d=as_2d)
    header["procpar"] = read_procpar(os.path.join(path, "procpar"))
    return header, data
