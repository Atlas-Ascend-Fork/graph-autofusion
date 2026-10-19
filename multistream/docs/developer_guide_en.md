# Developer Guide

This guide describes the Multistream code structure, testing, dependency installation, and troubleshooting for module developers. See the [README](../README_en.md) for environment setup and public API usage.

## Directory Overview

```text
multistream/
  algorithm/
    planning.py           # Validate arguments, query Runtime, prepare durations, and return results
    stream_allocator.py   # Separate serial groups, cover paths, and assign physical streams
    path_cover.py         # Path covering algorithm
    merger.py             # Base stream merge strategies
    weighted_merger.py    # Weighted merging using profiling data
    profiling.py          # CSV parsing and name matching
  common/
    runtime_platform.py   # Runtime ABI wrapper using ctypes
    log.py                # Module logging
    errors.py             # Internal consistency checks
  minidag/                # Graphs, nodes, value types, and StreamPlan
  tests/
    common/               # Runtime ABI stub
    fixtures/             # Shared fixtures for test graphs and profiling data
    utils/                # Graph construction and profiling helpers
    ut/                   # Algorithms, data model, public API, and error cases
    st/                   # End-to-end planning, consistency across processes, packaging, and performance
    conftest.py           # Load fixtures, mark tests, and replace Runtime interfaces
  scripts/
    ms_ut_cfg.toml         # UT coverage configuration
    ms_st_cfg.toml         # ST coverage configuration
  docs/                   # Developer documentation
  pyproject.toml          # Python package and pytest configuration
```

## Main Flow and Considerations

`plan()` first validates arguments and checks the graph for cycles. It then queries Runtime core counts, prepares node duration and resource usage for the current plan, and assigns streams. After allocation, it validates node stream IDs and returns a `StreamPlan`. A failure raises an exception instead of returning a partial result.

The algorithm first handles nodes with serial labels and nodes whose near-zero duration excludes them from stream assignment. It then assigns streams to the remaining nodes while preserving the original dependencies. Finally, it appends one stream per serial group. Changes to this code must account for excluded nodes, serial labels, stream limits, and starting stream IDs, in addition to ordinary branching graphs.

The Runtime wrapper caches only dynamic library function bindings and queries core counts again for every plan. The application must keep the target SoC unchanged during a query. Do not replace Runtime with the test stub in production or supply default core counts when a query fails.

## Dependency Management

Package metadata is centralized in [pyproject.toml](../pyproject.toml). `project.dependencies` is empty, while `project.optional-dependencies.dev` defines development dependencies including setuptools, wheel, pytest, pytest-cov, pytest-xdist, and coverage. Install them in a dedicated virtual environment:

```bash
# Enter the module directory from the repository root. Run subsequent commands here.
cd multistream
python3 -m pip install -e ".[dev]"
```

Tests replace Runtime interfaces with a stub by default, so Python planning logic can be tested without an Ascend device. Wheel tests build offline through the CMake delivery target and install into temporary directories; CMake, Make, setuptools, and wheel must already be installed. System tests for installation scripts also require the relevant shell tools. Refer to the individual tests for specific requirements.

## Shared Test Fixtures

Fixtures prepare graphs, data, and execution environments. When a test declares a fixture name as a parameter, pytest invokes that fixture automatically. Fixtures with `autouse=True` run automatically without being named as parameters.

| Fixture | Location | Purpose |
|---------|----------|---------|
| `stub_runtime` | `tests/conftest.py` | Replaces Runtime library loading with a stub and clears cached function bindings before and after tests |
| `linear_graph`, `deep_chain_graph` | `tests/fixtures/graphs.py` | Return functions that create chain graphs with a specified node count |
| `diamond_graph` | `tests/fixtures/graphs.py` | Provides a graph with branching and joining paths |
| `multi_branch_graph`, `large_graph` | `tests/fixtures/graphs.py` | Create branching graphs of a specified size or DAGs with a fixed random seed |
| `serial_labeled_graph` | `tests/fixtures/graphs.py` | Provides a graph with multiple serial labels |
| `op_summary_csv` | `tests/fixtures/profiling.py` | Writes CSV files in `tmp_path`, with customizable headers and rows |
| `installed_wheel` | `tests/st/conftest.py` | Builds the wheel once per test session, installs it into a temporary directory, and returns the wheel path and installation directory |
| `record_environment` | `tests/st/conftest.py` | Automatically records the Python version, package path, and Runtime stub information |

Shared fixtures are loaded through `pytest_plugins` in `tests/conftest.py`. Interfaces replaced with monkeypatch in a parent process are not automatically replaced in child processes. Follow existing subprocess tests and call `install()` from `tests/common/runtime_stub.py` to replace Runtime interfaces in the child process.

## Test Conventions

1. Name test files `test_*.py`. Place algorithm and API error tests in `tests/ut/`, and tests spanning modules, installation, packaging, or performance in `tests/st/`.
2. `tests/conftest.py` automatically adds the `ut` or `st` marker based on the directory. Add `runtime`, `package`, or `perf` markers as needed; their definitions are in `pyproject.toml`.
3. Prefer public APIs for graph construction. Reuse fixtures for chain graphs, branching graphs, serial groups, and profiling data. Use `tmp_path` or `tmp_path_factory` for temporary files.
4. When changing algorithms, test both base strategies, strategy switching triggered by profiling, starting stream IDs, excluded nodes, and serial groups. When changing result objects, check node ownership and whether successive planning calls affect each other.
5. When changing traversal or merge order, check consistency across processes and different `PYTHONHASHSEED` values. When changing Runtime calls, check interface arguments, error handling, and whether updated target core counts are queried again.

Existing UT/ST use stubbed Runtime interfaces to check planning logic and interface calls. The caller must still test stream creation, synchronization, and performance gains on real devices.

## Running and Debugging Tests

### Basic Workflow

```bash
# Unit tests
python3 -m pytest tests/ut -m ut

# System tests
python3 -m pytest tests/st -m st

# UT and ST
python3 -m pytest tests/ut tests/st

# Run one module with logging enabled
python3 -m pytest tests/ut/test_multistream.py -v --log-cli-level=INFO
```

Without an explicit directory, pytest's `testpaths` collects only `tests/ut`. Adding `-m st` alone does not collect system tests. Add `-n 8` to run unit tests in parallel when needed.

### Coverage

UT and ST use separate configurations. Run these commands from `multistream/`:

```bash
python3 -m pytest tests/ut -m ut \
  --cov-config=scripts/ms_ut_cfg.toml --cov=. \
  --cov-report=term-missing --cov-report=html --cov-report=xml

python3 -m pytest tests/st -m st \
  --cov-config=scripts/ms_st_cfg.toml --cov=. \
  --cov-report=term-missing --cov-report=html --cov-report=xml
```

| Output | UT Path | ST Path |
|--------|---------|---------|
| Raw coverage data | `coverage/ut/.coverage` | `coverage/st/.coverage` |
| HTML report | `coverage/ut/html/index.html` | `coverage/st/html/index.html` |
| XML report | `coverage/ut/coverage.xml` | `coverage/st/coverage.xml` |

Paths are relative to the module directory. Both configurations enable branch coverage and exclude test, build, and virtual environment directories. UT requires at least 90% coverage; ST has no coverage threshold.

### Performance Tests

```bash
python3 -m pytest tests/st/test_perf.py -m perf -s
```

Performance tests measure planning time and result query time at different graph sizes, and check recursion depth when handling long chains. Reports include median and P95 timings to track the overhead of calling `plan()`.

| Environment Variable | Default | Description |
|----------------------|---------|-------------|
| `MULTISTREAM_PERF_WARMUP` | `3` | Number of warmup runs |
| `MULTISTREAM_PERF_RUNS` | `10` | Number of timed runs |
| `MULTISTREAM_PERF_JSON` | `multistream_perf.json` | Report path relative to the current working directory |
| `MULTISTREAM_PERF_STRICT` | `0` | Set to `1` to fail tests when duration exceeds the limits defined by the tests |

By default, exceeding a duration limit produces a warning without failing the test. On a machine with a stable load, set `MULTISTREAM_PERF_STRICT=1` to enforce timing limits. Running the full ST suite may also generate a performance report.

## File Management and Contributions

Use pytest temporary directories for temporary test inputs and `coverage/` for coverage output. Set the performance report path through its environment variable. Maintain ignore rules in the repository root `.gitignore`, and do not commit generated test files or reports.

Read the project's [Contribution Guide](../../CONTRIBUTING.md) before contributing. Update the [README](../README_en.md) and relevant tests when interfaces or behavior change. Update this guide when shared fixtures, execution commands, or output paths change.
