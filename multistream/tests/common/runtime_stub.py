"""Small Runtime ABI stand-in for installed-package system scenarios.

Only the external rtGetSocSpec boundary is replaced. The installed package,
public graph API, profiling parser, planner, and stream allocator run as-is.
"""

import ctypes


class RuntimeStub:
    def __init__(self, ai_core_count=24, vector_core_count=48):
        self.counts = (ai_core_count, vector_core_count)
        self.calls = []
        self.argtypes = None
        self.restype = None

    def __call__(self, label, key, buffer, max_len):
        self.calls.append((label, key, max_len))
        if label != b"SoCInfo":
            return 1
        values = {b"ai_core_cnt": self.counts[0], b"vector_core_cnt": self.counts[1]}
        if key not in values:
            return 1
        buffer.value = str(values[key]).encode("ascii")
        return 0


class _Library:
    def __init__(self, query):
        self.rtGetSocSpec = query


def install(monkeypatch=None):
    """Install the stub at ctypes.CDLL, leaving other dynamic libraries alone."""
    query = RuntimeStub()
    real_cdll = ctypes.CDLL

    def load(name, *args, **kwargs):
        if name == "libruntime.so":
            return _Library(query)
        return real_cdll(name, *args, **kwargs)

    if monkeypatch is None:
        ctypes.CDLL = load
    else:
        monkeypatch.setattr(ctypes, "CDLL", load)
    return query
