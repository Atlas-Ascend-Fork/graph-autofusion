"""MS-ST-01: installed wheel provenance and lazy Runtime loading."""

from pathlib import Path
import subprocess
import sys

import pytest


pytestmark = [pytest.mark.st, pytest.mark.package]


@pytest.fixture
def run_installed(installed_wheel):
    _, target = installed_wheel
    helpers = Path(__file__).resolve().parents[1]

    def run(script):
        prelude = (
            f"import sys; sys.path[:0] = {[str(target), str(helpers)]!r}\n"
            "import multistream as m\n"
            "from pathlib import Path\n"
            f"Path(m.__file__).resolve().relative_to({str(target)!r})\n"
        )
        return subprocess.check_output(
            [sys.executable, "-I", "-c", prelude + script],
            cwd=target.parent,
            timeout=30,
            text=True,
        )

    return run


def test_ms_st_01_installed_wheel_and_lazy_import(run_installed):
    run_installed("""
assert set(m.__all__) == {
    'DAGGraph', 'DAGNode', 'NodeCost', 'CoreType', 'StreamPlan', 'MergeStrategy', 'plan'
}
assert 'multistream.common.runtime_platform' in sys.modules
from multistream.common import runtime_platform
assert runtime_platform._get_soc_spec is None
""")


def test_ms_st_01_without_runtime(run_installed):
    """A missing Runtime loader fails planning even if CANN exists on the host."""
    script = """
import ctypes
import multistream as m
graph = m.DAGGraph('missing_runtime')
graph.add_node('n', 'Op')
real_cdll = ctypes.CDLL
def unavailable(name, *args, **kwargs):
    if name == 'libruntime.so':
        raise OSError('libruntime.so unavailable')
    return real_cdll(name, *args, **kwargs)
ctypes.CDLL = unavailable
try:
    m.plan(graph)
except RuntimeError as exc:
    assert 'libruntime.so' in str(exc)
else:
    raise AssertionError('planning unexpectedly succeeded')
"""
    run_installed(script)


@pytest.mark.runtime
def test_ms_st_01_installed_wheel_can_plan(run_installed):
    run_installed("""
from common.runtime_stub import install
query = install()
graph = m.DAGGraph('installed')
node = graph.add_node('node', 'Op')
result = m.plan(graph)
assert result.get_stream_id(node) == 0
assert result.get_required_stream_count() == 1
assert len(query.calls) == 2
""")
