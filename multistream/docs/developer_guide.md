# 开发者指南

本文档面向 Multistream 模块的开发人员，介绍代码结构、测试方法、依赖安装和问题排查。环境准备和公开接口用法见 [README](../README.md)。

## 目录结构概览

```text
multistream/
  algorithm/
    planning.py           # 参数校验、Runtime 查询、准备节点耗时和返回结果
    stream_allocator.py   # 串行组拆分、路径覆盖与物理流分配
    path_cover.py         # 路径覆盖算法
    merger.py             # 基础流合并策略
    weighted_merger.py    # 根据 profiling 数据进行加权合并
    profiling.py          # CSV 解析及名称匹配
  common/
    runtime_platform.py   # ctypes Runtime ABI 封装
    log.py                # 模块日志
    errors.py             # 内部一致性检查
  minidag/                # 图、节点、值类型和 StreamPlan
  tests/
    common/               # Runtime ABI stub
    fixtures/             # 生成测试用图和 profiling 数据的公共函数
    utils/                # 图构造及 profiling 辅助函数
    ut/                   # 算法、数据模型、公开接口和异常情况测试
    st/                   # 完整规划流程、跨进程结果一致性、打包与性能测试
    conftest.py           # 加载公共测试函数、自动标记测试和替换 Runtime 接口
  scripts/
    ms_ut_cfg.toml         # UT 覆盖率配置
    ms_st_cfg.toml         # ST 覆盖率配置
  docs/                   # 开发资料
  pyproject.toml          # Python 包与 pytest 配置
```

## 主要流程和注意事项

`plan()` 先检查参数和图是否有环，再查询 Runtime 核数，准备本次规划使用的节点耗时和资源占用信息，然后分配流。分配完成后，检查节点的流 ID 是否符合要求，并返回 `StreamPlan`。如果中途失败，会抛出异常，不返回未完成的结果。

算法先单独处理带串行标签的节点，以及耗时接近零、不参与分流的节点，再为剩余节点分配流，同时保留原有依赖关系。最后，为每个串行组追加一条流。修改这部分代码时，除了普通分支图，还需要检查不参与分流的节点、串行标签、流数上限和起始流 ID 等情况。

Runtime 封装只缓存动态库函数绑定，每次规划重新查询核数。调用程序需要保证查询期间不切换目标 SoC。正式运行时不能用测试 stub 代替 Runtime，查询失败时也不能自行填入默认核数。

## 依赖管理

包信息集中在 [pyproject.toml](../pyproject.toml)：`project.dependencies` 为空，`project.optional-dependencies.dev` 定义 setuptools、wheel、pytest、pytest-cov、pytest-xdist 和 coverage 等开发依赖。请在独立虚拟环境中安装：

```bash
# 从仓库根目录进入模块目录，后续命令均在此目录执行。
cd multistream
python3 -m pip install -e ".[dev]"
```

测试默认用模拟实现（stub）代替 Runtime 接口，因此没有 Ascend 设备也能测试 Python 规划逻辑。wheel 测试通过 CMake 交付入口离线构建，并安装至临时目录，因此运行前必须已安装 CMake、Make、setuptools 和 wheel。安装脚本相关 ST 还需具备相应 shell 工具，具体约束以用例为准。

## 公共测试函数

这些函数负责准备测试所需的图、数据和运行环境。用例在参数中写出函数名，pytest 就会自动调用对应函数；设置了 `autouse=True` 的函数则会自动执行。

| 函数 | 位置 | 用途 |
|------|------|------|
| `stub_runtime` | `tests/conftest.py` | 用模拟实现替换 Runtime 动态库加载，并在测试前后清空函数绑定缓存 |
| `linear_graph`、`deep_chain_graph` | `tests/fixtures/graphs.py` | 返回一个函数，调用它可以创建指定节点数的链式图 |
| `diamond_graph` | `tests/fixtures/graphs.py` | 提供分叉、汇合图 |
| `multi_branch_graph`、`large_graph` | `tests/fixtures/graphs.py` | 按规模构造分支图或固定随机种子的 DAG |
| `serial_labeled_graph` | `tests/fixtures/graphs.py` | 提供包含多个串行标签的图 |
| `op_summary_csv` | `tests/fixtures/profiling.py` | 在 `tmp_path` 中写 CSV，支持自定义表头和数据行 |
| `installed_wheel` | `tests/st/conftest.py` | 每次测试运行只构建一次 wheel，安装到临时目录，返回 wheel 路径和安装目录 |
| `record_environment` | `tests/st/conftest.py` | 自动记录 Python 版本、包路径和 Runtime stub 信息 |

公共测试函数通过 `tests/conftest.py` 的 `pytest_plugins` 加载。父进程中用 monkeypatch 替换的接口不会自动传给子进程。子进程测试需要参考已有用例，调用 `tests/common/runtime_stub.py` 中的 `install()` 来替换 Runtime 接口。

## 测试编写约定

1. 用例文件命名为 `test_*.py`，算法和接口异常情况用例放在 `tests/ut/`，跨模块、安装打包和性能用例放在 `tests/st/`。
2. `tests/conftest.py` 按目录自动添加 `ut` 或 `st` 标记；按需补充 `runtime`、`package`、`perf` 标记，定义见 `pyproject.toml`。
3. 优先使用公开接口构图，复用公共测试函数来创建链式图、分支图、串行组和 profiling 数据；临时文件使用 `tmp_path` 或 `tmp_path_factory`。
4. 修改算法时，测试两种基础策略、profiling 引起的策略切换、起始流 ID、不参与分流的节点和串行组等情况。修改结果对象时，检查传入的节点是否属于原图，以及多次规划是否会互相影响。
5. 修改遍历或合并顺序时，检查不同进程、不同 `PYTHONHASHSEED` 下的结果是否一致。修改 Runtime 调用代码时，检查接口参数、错误码处理，以及目标核数变化后能否重新获取。

现有 UT/ST 使用模拟的 Runtime 接口，主要检查规划逻辑和接口调用是否正确。流能否在设备上正确创建、同步，以及是否带来加速效果，还需要调用方在真实环境中测试。

## 运行与调试

### 基础流程

```bash
# 单元测试
python3 -m pytest tests/ut -m ut

# 系统测试
python3 -m pytest tests/st -m st

# UT 与 ST
python3 -m pytest tests/ut tests/st

# 定位单个模块，并显示日志
python3 -m pytest tests/ut/test_multistream.py -v --log-cli-level=INFO
```

不指定目录时，pytest 的 `testpaths` 默认只收集 `tests/ut`；仅增加 `-m st` 不会自动收集 ST。需要并行执行 UT 时可增加 `-n 8`。

### 启用覆盖率

UT 与 ST 分别使用已有配置，命令在 `multistream/` 目录执行：

```bash
python3 -m pytest tests/ut -m ut \
  --cov-config=scripts/ms_ut_cfg.toml --cov=. \
  --cov-report=term-missing --cov-report=html --cov-report=xml

python3 -m pytest tests/st -m st \
  --cov-config=scripts/ms_st_cfg.toml --cov=. \
  --cov-report=term-missing --cov-report=html --cov-report=xml
```

| 输出文件 | UT 路径 | ST 路径 |
|------|---------|---------|
| 原始覆盖率数据 | `coverage/ut/.coverage` | `coverage/st/.coverage` |
| HTML 报告 | `coverage/ut/html/index.html` | `coverage/st/html/index.html` |
| XML 报告 | `coverage/ut/coverage.xml` | `coverage/st/coverage.xml` |

上述路径相对于模块目录。两份配置均启用分支覆盖率并排除测试、构建和虚拟环境目录；UT 配置要求覆盖率不低于 90%，ST 配置未设置覆盖率门槛。

### 性能测试

```bash
python3 -m pytest tests/st/test_perf.py -m perf -s
```

性能用例测试不同节点数下的规划耗时、读取结果的耗时，以及处理长链式图时的递归深度。报告记录耗时中位数、P95 等数据，用来观察调用 `plan()` 的时间开销。

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `MULTISTREAM_PERF_WARMUP` | `3` | 预热次数 |
| `MULTISTREAM_PERF_RUNS` | `10` | 计时次数 |
| `MULTISTREAM_PERF_JSON` | `multistream_perf.json` | 报告路径，相对于当前工作目录 |
| `MULTISTREAM_PERF_STRICT` | `0` | 设置为 `1` 时，耗时超过用例设定的上限时，测试失败 |

默认情况下，耗时超过上限只会告警，不会使测试失败。建议在机器负载稳定时设置 `MULTISTREAM_PERF_STRICT=1`，将耗时检查作为通过条件。完整 ST 运行也可能产生性能报告。

## 文件管理与贡献

临时测试输入使用 pytest 临时目录，覆盖率输出使用 `coverage/`，性能报告路径可通过环境变量指定。忽略规则统一维护在仓库根目录的 `.gitignore`，不要提交测试生成的临时文件和报告。

贡献前阅读项目[贡献指南](../../CONTRIBUTING.md)。接口或行为变化应同步更新 [README](../README.md) 与相关测试；公共测试函数、运行方式或输出文件路径变化应同步更新本指南。
