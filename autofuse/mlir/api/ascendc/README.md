# AscendC API 层（fork 自 PyAsc）

本目录是 AFIR 向 AscendC API 方向 lowering 所依赖的方言/转换/序列化实现，代码 fork 自
[PyAsc](https://gitcode.com/cann/pyasc.git)，在上游 `master` 的基础上做 autofuse 定制化改动。
当前 fork 基线是哪个上游 commit，看下面「0. 上游同步记录」表格的最后一行。

AFIR 未来会向多种目标 API/ISA lowering（AscendC API、
PTO-ISA、CCE intrinsic 等），`api/` 是这些 lowering 目标的统一容器，`ascendc` 是其中一个具体实现。
本目录之下未来的兄弟目录预期是 `api/pto_isa/`、`api/cce_intrinsic/` 之类。整体架构和本目录在
`conversion/AFIRToAscendC` lowering 路径中的位置见 [../../docs/README.md](../../docs/README.md)
和 [Autofuse MLIR 架构重构设计](../../docs/design/2026-08-06-autofuse-mlir-architecture-refactoring.md)。

不再以 git submodule 方式引入 PyAsc，而是直接拷贝源码分叉演进（不跟随上游自动更新），原因：

- Autofuse 需要对 AscendC dialect/转换做持续的定制开发（新增算子、修改 emit 逻辑），submodule + patch
  的工作流下这些改动需要维护成 patch 文件，patch 数量和体量会随时间不可控地增长，冲突成本越来越高。
- 直接拷贝后可以用常规的 code review + git 历史来管理这些改动，不需要专门的 patch 导出/应用脚本。

## 0. 上游同步记录

每次从上游 PyAsc 同步代码后，在下表追加一行（追加在表格末尾，最新记录在最下面），记录同步时间和
同步到的上游 commit，方便查看当前 fork 基线与上游的距离。

| 同步日期 | 上游 commit | 说明 |
|---|---|---|
| 2026-08-26 | `45f4198` | fork 初始基线：基于上游 `master` 最新 commit 进行 autofuse 定制化改动，之后不再以 submodule 方式跟随上游 |
| 2026-09-20 | `b44c97b` | 全量同步上游 `master`。落到 `include/ lib/ bin/` 的只有 `1677997`（infra 升级：pass 改名/带选项、Reg API、CVGroup 规范化）、`3d2e2ff`（Linear / fused infer attention 所需 API）、`86c84f9`（AscTile 顺带改的 Asc 主干），其余 8 个 commit 只动 python/test/docs/CI，按第 1 节的拷贝方案跳过 |

## 1. 代码拷贝方案

只拷贝编译进 target 所必需的部分，不拷贝 PyAsc 项目自身的工程/社区协作配套文件。

已拷贝（原始目录结构、文件内容保持上游原样，方便未来对照上游 diff；少数不得不就地改的
地方逐条列在第 4 节）：

```
api/ascendc/
├── CMakeLists.txt      # 精简自 PyAsc 顶层 CMakeLists.txt，去掉 python/test 相关部分
├── include/ascir/       # Dialect / API 头文件，原样拷贝
├── lib/
│   ├── Dialect/         # Asc、EmitAsc 方言实现
│   ├── Target/          # AscendC 代码序列化（Target/AscendC）
│   └── TableGen/        # GenAPITypedefs / GenAPITypes / GenOpEmitDecls 等生成器
└── bin/
    └── ascir-translate.cpp   # 唯一被 autofuse 实际使用的独立可执行工具
```

autofuse 自己加的 op / emitter / interface **不写进上述任何一个上游文件**，而是集中在三个
`Autofuse/` 子目录里，见下面第 4 节。

未拷贝（PyAsc 项目自身的工程配套，autofuse 不需要）：

- `python/`（`python/asc/`、`python/src/` pybind11 绑定）和 `include/ascir/Extension/`
  （`PyOpBuilder.h`、`PythonExtension.h`，直接 `#include <pybind11/pybind11.h>`）—— autofuse 当前没有
  任何代码 `import` PyAsc 的 Python 包，也不使用 `libpyasc` pybind 模块。若未来确实需要从 Python 侧
  直接操作 AscendC dialect，再评估是否按需引入。
- `experimental/`（AscTile 方言及其 lowering）—— 上游自己标为 experimental，默认构建开关关闭，
  且与 AFIR 的 lowering 路径没有交集。
- `test/`、`python/test/`、`python/tutorials/`、`docs/`、`scripts/`
- `bin/ascir-opt.cpp`、`bin/ascir-lsp.cpp`（全仓库未被任何构建/脚本/文档引用，只保留实际使用的
  `ascir-translate`）
- `.clang-format`、`.clang-tidy`、`.pre-commit-config.yaml`、`classify_rule.yaml`、
  `torch_npu_install.sh`、`CONTRIBUTING.md`、`SECURITY.md`、`OAT.xml`、
  `requirements-build.txt`、`requirements-runtime.txt`、`pyproject.toml`、`setup.py`、`LICENSE`
  （PyAsc 作为独立 Python 包/开源项目的打包、社区协作、合规扫描配套，随 fork 一起去掉，
  统一遵循 autofuse 仓库级别的对应约定）

## 2. 代码依赖原则

为了让这份 fork 代码尽量保持"干净、独立、易同步"，约定以下原则：

1. **对外只暴露 `include/ascir/` 下的公开头文件**。AFIR 侧代码（`conversion/AFIRToAscendC`、
   `tools/af-opt` 等）只允许 `#include "ascir/..."` 下的公开头，
   不允许直接 include `lib/` 内部实现头或 `.cpp` 文件。这条边界在拷贝前就已经是事实上的现状，
   拷贝后继续保持。
2. **通过 CMake target 而非源码路径耦合**。AFIR 侧只依赖 `MLIRAsc`、`MLIRAscTransforms`、
   `MLIRAscUtils`、`MLIREmitAsc`、`MLIRTargetAsc` 这几个 target，不假设 `api/ascendc/` 内部的
   目录/文件划分。内部可以自由重构，只要这些 target 名和它们的公开头文件不变，外部不受影响。
3. **`api/ascendc/` 内部不 include autofuse 其他目录的代码**（不反向依赖 `dialect/AFIR`、
   `conversion/`、`bridge/` 等）。保持单向依赖，既是为了避免循环依赖，也是为了保留未来把
   `api/ascendc/` 整体拆分成独立仓库的可能性。
4. **不引入超出编译 target 所需的构建依赖**。已去掉 Python3/pybind11 等 PyAsc 原本为构建 Python
   绑定引入的依赖；后续给 `api/ascendc/` 加构建选项时同样只为"编译出 MLIR C++ target"这一个目标
   服务，不为假设的将来用途预留开关。
5. **内部命名空间/include 路径保持 `ascir::` / `ascir/...` 不变**，不与 autofuse 自己的
   `autofuse/ascir/`（`ascir_ops.h` 等）混淆——两者是同名但完全不同的代码，`api/ascendc/`
   仅通过上面第2点提到的 target 名对外暴露，不存在头文件路径冲突。

## 3. 从上游 PyAsc 同步更新的方法与策略

不再使用 submodule + 自动 patch 应用的方式，改为**按需手动同步**：

1. 当需要参考上游 PyAsc 的某次改进（bug fix、新算子支持等）时，clone 一份上游仓库到本地临时目录，
   基于本 README「0. 上游同步记录」表格中最新一行记录的上游 commit，用 `git log`/`git diff` 找到
   相关的上游 commit。
2. 用 `git diff <base>..<upstream_commit> -- include/ lib/` 之类的命令生成针对本目录结构的 diff，
   人工审查后决定是否手动应用到 `api/ascendc/` 下对应路径（因为已经不拷贝 `python/`、`test/` 等目录，
   涉及这些路径的上游 diff 直接跳过）。
3. 应用后在对应改动的 commit message 里注明来源（例如 `sync from pyasc <commit-hash>: <upstream 摘要>`），
   便于以后追溯这段代码最初的出处，但不需要（也不应该）保留 patch 文件或自动化同步脚本——
   这是一次性的手动移植动作，移植完成后代码即视为 autofuse 自己维护的代码。
4. 如果本目录内已经产生了与上游实现方式不同的自主演进（例如自定义算子、自定义 emit 逻辑），
   同步上游改动时以不破坏这些自主演进为前提，冲突时优先保留 autofuse 自己的实现，仅挑选上游中
   与自主演进不冲突、且确实有价值的部分（如上游的 bug fix）。
5. 定期（没有固定周期，视上游活跃度和 autofuse 自身需求）review 一次上游仓库的变更日志，
   评估是否有值得同步的内容；不追求"保持与上游一致"，只追求"按需吸收有价值的改动"。
6. 每次完成一轮同步后，无论本次实际改动大小，都在「0. 上游同步记录」表格追加一行，记录同步日期和
   本次同步到的上游 commit（取本次参考到的最新上游 commit，不要求是上游 HEAD），方便后续同步时
   快速定位起点，无需翻查 commit 历史。

## 4. autofuse 定制内容：`Autofuse/` 子树与上游接缝

第 1 节说「文件内容保持不变」，这条要靠一个约定守住：**autofuse 新增的 op、emitter、
interface、属性一律放进 `Autofuse/` 子树，不写进上游文件**。否则上游文件会被逐步改成
autofuse 的文件——上一轮就出现过 `Basic/OpVecBinary.td` 386 行里有 345 行是 autofuse 加的，
一次全量同步手工解了 13 处冲突。

三个子目录：

```
include/ascir/Dialect/Asc/IR/Autofuse/    # ODS：op / interface / 属性
├── Ops.td            # 唯一入口，被上游 ../Ops.td 的最后一行 include
├── Base.td           # ApiExtensionHeader、SrcTensorsFromSrc
├── Interfaces.td     # Autofuse_OpInterface + ApiExtensionOpInterface（自己一个 tablegen 入口）
├── Attributes.td     # ReduceLayout 枚举，被上游 ../Core/Attributes.td 最后一行 include
└── Op*.td            # 各算子族的 def

include/ascir/Target/Asc/Autofuse/        # emitter 声明
├── Ops.h             # PrintableOpTypes + printModulePrelude，唯一被 Translation.cpp include 的头
└── *.h

lib/Target/AscendC/Autofuse/              # emitter 实现（CMake 是 GLOB_RECURSE，加文件不用改）
└── *.cpp
```

### 接缝

新增 op / emitter **不需要碰上游文件**：往 `Autofuse/Op*.td` 加 def、往
`Autofuse/Ops.td` 的 include 列表和 `Autofuse/Ops.h` 的 `PrintableOpTypes` 里各加一行即可。
下面这些是子树接进上游的全部位置，改它们的场合只有「子树多了一类东西」：

| 上游文件 | 内容 | 为什么必须在这里 |
|---|---|---|
| `include/ascir/Dialect/Asc/IR/Ops.td` | `include "Autofuse/Ops.td"` | op 要进同一个 `-gen-op-decls`/`-gen-opemit-*` 批次 |
| `include/ascir/Dialect/Asc/IR/Core/Attributes.td` | `include "Autofuse/Attributes.td"` | 枚举要进同一个 `-gen-enum-decls`（这个 backend 不筛主文件） |
| `include/ascir/Dialect/Asc/IR/CMakeLists.txt` | 3 行，给 `Autofuse/Interfaces.td` 开一个 tablegen 入口 | `-gen-op-interface-decls` **只发主文件里定义的 interface**（`OpInterfacesGen.cpp` 的 `getAllInterfaceDefinitions` 显式按 `getMainFileID()` 过滤），所以 interface 不能靠 include 搭车 |
| `include/ascir/Dialect/Asc/IR/Asc.h` / `lib/Dialect/Asc/IR/OpInterfaces.cpp` | 各 1 行，include 上一条生成的 `.inc` | 同上 |
| `lib/Target/AscendC/Translation.cpp` | include `Autofuse/Ops.h`；`printModulePrelude`；第二个 `addCases` | `addCases` 本来就能调两次，上游那个巨型 tuple 不用动 |
| `lib/Target/AscendC/External/Arith.cpp` | include + 1 行改用 `getResultBitWidth` | 上游 `MulUIExtendedOp` 的 emitter 不能被重载替换，只能就地改；规则本身在子树里，两边共用一份 |

### 不能外移、留在上游文件里的改动

- `IR/Base.td` 的 `VecScalarL2Op::isSetMask` 缺省值（`UnitAttr` → `DefaultValuedAttr<BoolAttr, "true">`）。
  TableGen 不能 reopen 一个 class，而这条改的是**上游 op**（`adds` / `muls` …）的 ODS。
- `Target/Asc/CodeEmitter.h` 的两个 `getOrCreateName` 重载。它们是成员函数，生成的 emitter
  按 `emitter.getOrCreateName(...)` 调用，外部函数接不上。
- `bin/ascir-translate.cpp` 注册 affine / bufferization 两个方言。
- 与上游 LLVM 版本适配相关的就地修改（`applyPatternsAndFoldGreedily` → `applyPatternsGreedily`、
  `builder.create<X>(...)` → `X::create(builder, ...)`、`TableGenMain` 多一个实参等）。这些本来
  就不是「定制」，同步上游时按上游新写法取舍即可。

### 新加的 td 必须接进 include 图

`Autofuse/Ops.td` 是唯一入口，新文件要往它的 include 列表里加一行，否则里面的 def 一个都不会
被 tablegen 展开——**写错了不报错，写对了也用不上**。这不是假设：最初集成带进来 7 个 `*Ext.td`
从来没被 include 过，41 个 def 一直是死文本，接 `afir.where` 和 `afir.clip_by_value` 时两次差点
被当成现成供给用，2026-09-24 已删除。`tools/af-mlir-tools/check_ascendc_td_reachable.py` 就是
守这条的门禁。

### 格式

本目录整个在仓库级 pre-commit `exclude` 里，`Autofuse/` 子树也一样：它写在上游的风格里
（4 空格、120 列、函数体 Allman），与相邻的上游文件保持一致比与 autofuse 其余代码一致更有用。
不要对本目录跑仓库的 `clang-format`。
