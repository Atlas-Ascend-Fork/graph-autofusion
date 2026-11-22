# NPU Test Cases

This is the English companion of [README.md](README.md).

The Chinese document is the current source of truth for the NPU test cases,
runtime requirements, and execution commands.

The following mirrors the latest additions to the Chinese document:

`run.py` defaults to the current Python interpreter and its standard-library
prefix while keeping overlay and build dynamic-library paths at the highest
priority. When selecting another interpreter through `AF_NPU_PYTHON`, also set
`AF_NPU_PYTHONHOME` to that interpreter's `sys.base_prefix`. Do not use `/usr`
as a common prefix for all interpreters.
Preflight verifies module origins and actually imports the required modules;
missing Python dependencies or incompatible dynamic libraries stop the run
immediately instead of mixing environment failures into case results.
Overlay files are first copied within the same directory and then replaced
atomically, avoiding truncated extensions already loaded by other processes.
Concurrent tasks should still use independent overlays and working directories.

Build artifacts must also match the runtime system ABI: extensions and dynamic
libraries built inside a container should run in the same system environment
and must not be moved directly onto a host machine with an older glibc. When
real hardware is required, mount local devices and drivers into a matching
runtime container, verify NPU eager mode first, and only then run the suite;
keep the preflight checks enabled and confirm that all modules come from the
current overlay.
The current runner installs artifacts with `cmake --install` into
`build/dev-install/` and stores logs under `runs/<commit>-<run-id>/`. Each run
rebuilds that install tree, then verifies Python module provenance, dynamic library
paths, the build hash and device availability before starting cases.

For an installed-tree run, use `run.py --install-dir <build/dev-install>
--build-dir <build/autofuse>`. This mode does not create an overlay and prefers the
installed Python package and libraries.

The reduce fusion matrix currently covers ten topology-tail cases plus the
`fork_mul` and `multi_two_sums` large cases, for twelve cells total. The runner's
`AF_MLIR_BROADCAST_REJECT` diagnostics report the original subgraph, plan, stage and
template IDs, the rejected edge, the reason and the relevant axis, repeat, stride
and comparison details. One diagnostic is emitted for the first failed branch of
each input edge; diagnostic count must not be interpreted as case count.
