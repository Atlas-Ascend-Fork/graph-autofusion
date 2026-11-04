# Ascend MLIR AGENT 调测介绍

## 1. 本地环境调测（CPU仿真）

### 1.1 MacOS 操作系统

编译方案：Docker LLVM 镜像 + CANN Toolkit Volume

Docker镜像名称：`autofuse-mlir-dev:arm64`

CANN Volume名称：`cann-volume`

编译执行 MLIR UT/ST 命令：

```shell
AF_CANN_VOLUME=cann-volume \
bash autofuse/mlir/scripts/enter_dev_container.sh -- \
bash scripts/test/run_autofuse_test.sh -u -s -m mlir -j 10
```

其他编译命令可自行拓展

### 1.2 Ubuntu 操作系统

编译方案：本地编译 LLVM + 本地安装 CANN Toolkit
LLVM 构建路径通常为: `/opt/llvm`
CANN Toolkit安装路径通常为：`${HOME}/Ascend`

编译执行 MLIR UT/ST 命令：

```shell
source ${HOME}/Ascend/ascend-toolkit/set_env.sh
bash scripts/test/run_autofuse_test.sh -u -s -m mlir -j 10
```

其他编译命令可自行拓展

## 2. 远程 Ubuntu 环境调测（真实NPU）

### 2.1 连接方式

建议使用 `npussh` 工具远程连接 CANNLab 提供的 NPU 环境，使用方式与原生的 `ssh` 一致。

工具安装和使用方式:
> Step1：在个人开发环境上，安装命令行工具：
>
> `npm install -g npussh`
>
> Step2：首次使用时登录 CANNLab：
>
> `npussh login`
>
> Step3：查看当前账号下可用的设备：
>
> `npussh list`
>
> 输出演示`AF_MLIR_0814	运行中	8e939b1bbxxxxxxx	8e939b1bbxxxxxxx`
>
> Step4：使用上一步输出的`设备名称`或`ID`登录：
>
> `npussh <设备名称或ID>`
>
> 示例：`npussh AF_MLIR_0814` 或者 `npussh 8e939b1bbxxxxxxx`

若已经安装并登录 `npussh` ，后续可直接连接：

```shell
$ npussh list

npussh AF_MLIR_0814
```

本工具还同时支持 `npuscp`、`npursync`、`npusftp` 等命令，使用方式与对应的`scp`、`rsync`、`sftp` 一致。详见[使用文档](https://gitcode.com/niuyuhu/cannlab-ssh/blob/main/packages/npussh/README.md)

### 2.2 编译命令

编译方案：本地编译安装 LLVM + 本地安装 CANN Toolkit

LLVM 安装路径：`/opt/llvm`

CANN Toolkit 安装路径：`$HOME/Ascend`

CANN Driver 安装路径：`/usr/local/Ascend/driver`

### 2.3 执行用例和调测

#### 2.3.1 编译执行 MLIR 相关的 UT、ST、E2E CPU 用例命令

```shell
export AF_LLVM_ROOT=/opt/llvm
export LLVM_BUILD_DIR=/opt/llvm
export LLVM_DIR=/opt/llvm/lib/cmake/llvm
export MLIR_DIR=/opt/llvm/lib/cmake/mlir
source ${HOME}/Ascend/ascend-toolkit/set_env.sh

bash scripts/test/run_autofuse_test.sh -u -s -m mlir -j 40
```

#### 2.3.2 执行 MLIR E2E NPU 用例命令

```shell
python3 autofuse/mlir/test/e2e/npu/load_add_store.py
```

#### 2.3.3 NPU 硬件信息查看命令

```shell
npu-smi info
```

### 2.4 使用准则

#### 2.4.1 运行产生的临时文件、debug、profiling、dump、日志和 cache 只放在 `/mnt/workspace/agent/<case>-<sha>-<purpose>` 或 `/tmp`。不要把临时产物放在源码目录。

#### 2.4.2 如果不指定代码目录，则默认在 `/mnt/workspace/agent/<case>-<sha>-<purpose>` 中从 `git clone git@gitcode.com:niuyuhu/graph-autofusion.git` 上克隆最新的 `af-mlir` 分支代码进行调试；

#### 2.4.3 不允许改动 `/opt/llvm`、`${HOME}/Ascend`、`/usr/local/Ascend` 和其他 CANN/驱动安装目录。禁止把 build 中的 `pyautofuse.so` 或 Python 文件复制覆盖到 CANN 安装包。

#### 2.4.4 一个 shell 只 source 一套 CANN，并使用显式的 Python 解释器。运行前检查 `torch`、`torch_npu`、`autofuse.pyautofuse`、`autofuse.ascendc_compile` 和 `autofuse.compile_adapter` 的实际导入路径。

```bash
"$PYTHON_BIN" - <<'PY'
import importlib.util
import sys

print("sys.executable=", sys.executable)
for name in (
    "torch",
    "torch_npu",
    "autofuse.pyautofuse",
    "autofuse.ascendc_compile",
    "autofuse.compile_adapter",
):
    spec = importlib.util.find_spec(name)
    print(name, spec.origin if spec else None)
PY
```

#### 2.4.5 `pyautofuse.so`、`ascendc_compile.py`、`compile_adapter.py` 和 `libaihac_codegen.so` 必须来自同一份源码/build 版本。涉及 Python 修复时，确认对应函数存在；手工 `chmod` 不能替代版本验证。

```bash
"$PYTHON_BIN" - <<'PY'
from autofuse import ascendc_compile

print("ascendc_compile.__file__=", ascendc_compile.__file__)
print("has_remove_group_other_write=",
      hasattr(ascendc_compile, "remove_group_other_write"))
PY
```

#### 2.4.6 使用本地 build 时通过临时 overlay 同时覆盖匹配版本的 Python 文件和 `pyautofuse.so`，并用 `ldd` 检查 `libaihac_codegen.so` 的实际解析路径。不能只看 `RUNPATH`。

```bash
export PYTHONPATH="$OVERLAY_DIR:${PYTHONPATH-}"
export LD_LIBRARY_PATH="$BUILD_DIR/autofuse:${LD_LIBRARY_PATH-}"
```

运行扩展前检查动态库实际解析结果：

```bash
PY_MODULE="$("$PYTHON_BIN" - <<'PY'
import importlib.util
print(importlib.util.find_spec("autofuse.pyautofuse").origin)
PY
)"
ldd "$PY_MODULE" | grep -E 'libaihac_codegen|libaihac_ir|libgraph_af|libgraph_base'
```

#### 2.4.7 切换分支或更换关键依赖后，不复用未经核验的旧 `build/`、TorchInductor cache 或 NPU kernel cache。每个实验使用独立输出目录；不同 route 不共用 cache。

```bash
export RUN_DIR="/mnt/workspace/agent/<case>-<git-sha>-<purpose>"
mkdir -p "$RUN_DIR"
export TORCH_COMPILE_DEBUG_DIR="$RUN_DIR/torch_compile_debug"
export TORCHINDUCTOR_CACHE_DIR="$RUN_DIR/torchinductor"
export INDUCTOR_CACHE_DIR="$RUN_DIR/inductor_cache"
```

#### 2.4.8 真实 NPU 按 eager、legacy、MLIR 的顺序分阶段验证。只有源码、Python 扩展、codegen 动态库和 cache 已确认一致，才能把失败归因于代码变更。

#### 2.4.9 每次问题验证完成后，询问用户是否清理本次文件和数据：`/mnt/workspace/agent/<case>-<sha>-<purpose>`，禁止 `rm -rf /tmp/*`、按扩展名批量删除或删除来源不明文件。

#### 2.4.10 本环境编译可以使用 `-j 40` 并发编译验证。

#### 2.4.11 每次问题定位完成后，输出根因分析和经验总结，并咨询用户是否记录到本文档中第3节中。


## 3. 经验总结

### 3.1 NPU 0x507035 越界：ATT 的资源分配语义与下降取用点分离在两个目录，改任一侧都要对照另一侧

**症状**：3 个 add 级联的 e2e（`load_add3_store.py`），形状 (128,23,5) 与 (256,23,5) 过，(128,128,128) 挂——vector core 异常 0x507035「VEC 指令读写 UB 地址越界」。单 add 的用例全过。

**根因**：`b{id}_size`（ATT 侧按 `max(同 id 的每个张量) ⊔ Σ(同 reuse 组)` 聚合）的语义是 TBuf 只装**一个 tile**；而 MLIR 下降（`BinaryElementwise.cpp` / `UnaryElementwise.cpp` / `DataMovement.cpp` 的 `ScheduledContiguous` 支——这一档今天叫 `Scheduled`）给非队列（TBuf）成员加了 `localOffset`（= a0Tb 归纳变量 × a0t_size）——按多 tile 分配的偏移去算一块单 tile 的缓冲。(128,128,128) 的 `a0Tb_size=8`，写到 7×tile 处越界；小形状 `a0Tb_size=1` 偏移恒为 0，探不到。

**判别方法**：e2e 报 0x507035 时，先 dump tiling（ctypes 直接调 kernel.so 里的 `AutofuseTiling` 拿到 `a0t_size` / `a0Tb_size` / `block_dim` / `b0_size` 的确切值），再看生成代码（`output_code.py`）里本地张量的下标——是 `[0]`（legacy 语义）还是 `[归纳变量 × tile]`。块内循环多于一趟才触发的越界，优先怀疑「分配侧单 tile、取用侧多 tile」这类两侧语义漂移。

**修复方向**：调度循环每轮的 UB tile 起点为 0——队列按 slot 轮转，TBuf 复用当前 tile 的缓冲；不能把调度迭代号乘 tile 大小再加到 UB 地址。多维 DMA 在 tile 内生成的行循环仍可有非零 `localOffset`（`TransferPlan.cpp::emitScheduledTransfer`），它与跨 tile 的 GM 起点偏移不同。**不要**反向去改 ATT 的 `b{id}_size` 聚合语义（见 `autofuse/att/` 稳定性约束）。

**通用教训**：身份规划（ATT / TilingData 字段语义）与取用点（`AFIRToAscendC` conversion）分离在两个目录，两边没有共享的断言。改任一侧时必须对照另一侧的分配语义；用例要覆盖 `a0Tb_size > 1` 的形状（元素总数 ≳ 数十万），小形状用例对块内偏移类 bug 全盲。

### 3.2 远程复现环境的版本一致性陷阱

CANN 自带的 site-packages 里的 autofuse 与本地 clone 的提交可能差很多天（违反规则 2.4.5），症状是 `ascendc_compile.py` 缺符号、生成的 kernel 与源码对不上。可用 overlay 方案：clone 到 `/mnt/workspace/agent/` 后重编 `pyautofuse`，PYTHONPATH 指向 overlay 的 `python/site-packages`（目录层级要模仿 site-packages），并给 overlay 根补上指向真实 CANN 的 `tools` / `lib64` / `opp` / `aarch64-linux` / `include` / `pkg_inc` 符号链接（`ascendc_compile.py` 按 `PYF_PATH/../../..` 找 CANN 根，还依赖 `asc_op_compile_base` 模块与 `securec.h`）。cmake 要 `export LLVM_BUILD_DIR=/opt/llvm`（它读环境变量而不是 `-DLLVM_DIR`），目标用 `pyautofuse`；跑 lit 还要补编 `af-opt` / `af-afir-gen` / `ascir-translate` 并开 `-DENABLE_AUTOFUSE_MLIR_TESTS=ON`。

### 3.3 缺省极性反了的接口：`isSetMask` 一律打成 `0`，而全部用例的形状都探不到

**症状**：`Adds` / `Muls` / `Maxs` / `Mins` / `LeakyRelu` 的 `tensor_scalar` 形态发出去的
是 `AscendC::Adds<float, 0>(dst, src, scalar, count)`，legacy 发的是不带模板实参的
`Adds(dst, src, scalar, count)`。**全部 43 条 CPU E2E 用例都是绿的**，所以这条在清单里
挂了很久，定性是「两种取值都编得过，差别在 API 内部还设不设 mask，要上 NPU 比」。

**根因**：两个缺省的极性相反。AscendC 声明上是 `template <typename T, bool isSetMask = true>`；
MLIR 这边 ODS 写的是 `UnitAttr:$isSetMask`，而 `UnitAttr` 的缺省（属性不写）是 `false`。
**危险的那一侧正好是什么都不写时落到的那一侧。**

`isSetMask` 不是「要不要多设一次 mask」的微优化，是**由谁设 mask** 的分工（见
`compiler/tikcpp/tikcfw/impl/dav_c220/kernel_operator_vec_binary_scalar_impl.h` 的
Level 2 分支，m200 / c100 同形）：

- `true`：API 自己 `SetMaskCount()` + `SetMask(0, count)` + 收尾 `ResetMask()` / `SetMaskNorm()`；
- `false`：API 整段跳过，按 `repeatTimes = 1` 直发一条指令，**`count` 根本没被读**，
  靠调用方事先把 mask 置成 count 模式并设好 count。

今天没有任何 pass 建立那个前置条件，于是一趟算多于一个 repeat 就开始错。

**为什么 43 条用例全绿**：它们的形状**全是 `[128]`**。切完一趟的 count 不超过一个
repeat（f32 是 64 个），而 count ≤ 一个 repeat 时两种取值结果相同。换成 `[128, 23, 5]`
当场红：14720 个元素错 14528 个，第一个错的是下标 64。

**判别方法**：

1. **「要上 NPU 才能验」先核。** CPU 模拟器编的就是同一份 `dav_c220` 头（与
   `x86_64-linux/asc/impl/basic_api/dav_c220/*` 逐字节相同），`ASCENDC_CPU_DEBUG` 只影响
   `#if` 包着的检查段，算子主体两边同源。确认同源之后，「验不了」就不成立，问题在用例。
2. **最快的一次性实验是手写单文件程序链 tikicpulib**，不必走 E2E 框架：编译/链接参数抄
   `build/.../CMakeFiles/MlirE2eCpu_<case>.dir/flags.make` 和 `link.txt`，把 `-lpem_davinci`
   换成 `$CANN/x86_64-linux/simulator/dav_2201/lib`、`-lcpudebug` 换成
   `$CANN/tools/cpudebug/lib64/Ascend910B1`，再补 `$CANN/lib64/libc_sec.so`，运行时
   `LD_LIBRARY_PATH` 带上 `$CANN/lib64`。同一个文件里放 kernel 和 host `main`，
   host 侧 `#include "tikicpulib.h"` 用 `GmAlloc` / `SetKernelMode(AIV_MODE)` / `ICPU_RUN_KF`。
   **要对照着测三种组合**（缺省 / 显式 `0` / 显式 `0` 但调用方先设 mask），否则会把
   「有前置条件的取值」误判成「坏掉的取值」，进而选错修法。

**修复方向**：改缺省，不要改调用方，也不要在 emitter 里把这个模板实参屏蔽掉。
`Base.td` 的 `VecScalarL2Op`：`UnitAttr:$isSetMask` → `DefaultValuedAttr<BoolAttr, "true">:$isSetMask`。
一处覆盖 9 个 op，新增 conversion 点不会漏，旋钮保留（想用那个优化的 pass 显式写
`isSetMask = false`）。

- **不要屏蔽 emitter**：`false` 有正当用途（一串同 count 的 L2 调用把那四条提到循环外），
  屏蔽掉就永久封死了。`DuplicateL2Op` 的 ODS 里干脆没有这个参数，所以它的 emitter 不打
  是对的，**不能照抄到这里**。
- **不要在 conversion 里逐处 `setAttr`**：缺省仍停在危险侧，漏一处就静默回到 `0`。

**通用教训**：

1. **见到「这个属性今天没人设」，先别急着去设它——先看不设它对应的是哪一侧。** 跨语言
   镜像一个 C++ 模板参数时，`UnitAttr`（不写 = false）只在「false 是安全值」时才对；
   AscendC 里大量 `= true` 的模板参数都不满足这个条件。
2. **`[128]` 这个形状有一整类盲区。** 凡是「只在一趟算多于一个 repeat 时才错」的 bug，
   用例集全盲——这和 3.1 那条「`a0Tb_size = 1` 时块内偏移恒为 0」是同一个形状的问题：
   **小形状用例对「一趟不止一次」类 bug 全盲**。挑一条判据开工前，先问它在 `[128]` 上
   是不是根本走不到。

### 3.4 跨核归约全核写同一个地址：GM 偏移乘了局部侧的步长，而局部侧描述的不是落点

**症状**：RCore（多核两阶段归约）在 `nA < corenum` 的形状上数值全错——`(8, 131072)`、
`(16, 65536)`、`(20, 131072)` 三个形状 100% 对不上，而 `(40, 32768)`（`nA == corenum`）
和全部单核形状都是绿的。错的不是个别元素，是整条结果。

一开始最像的解释是核间同步：phase 1 每个核写自己那份 partial sum，phase 2 才能读。
但 `SyncAll` 查下来位置和数量都对——**同步不是真因**。

**根因**：`TransferPlan` 算 GM 偏移时，每一项乘的是**局部侧** descriptor 的步长，而局部
侧未必描述落点。Load 的两者同源（局部侧是结果，而结果那张 `strides` 表本来就是 GM 的读取
式样，UB 侧的几何在 `vectorized_strides` 里）；**Store 不是**：它的局部侧是 `operand_descs`，
描述的是**生产者**那块 UB 张量。归约把一根轴压掉之后，生产者在那根轴上步长为 0（每一轮写
的是同一个累加器），而落点在同一根轴上有真实步长（每个核把自己那段的 partial sum 写进
workspace 的不同格）。拿局部侧的 0 去算 GM 地址，这一项会被「步长为 0 就整项跳过」那条
优化剔掉，于是**所有核写同一个地址、互相覆盖**。

本次 `(40,32768)` 探针的 tiling 中核数由 A 轴单独喂满、R 方向不再分核，那根轴上的步长退化成 0 也是对的，
所以这一档探不到——**这个缺陷从来没有被任何用例覆盖过**。

**判别方法**：多核结果错而单核对时，先看生成代码里 GM 子视图的偏移表达式**有没有块号那一
项**（`v163 = v110 * c20_idx` 这种形状）。整项缺失说明它在某处被当成 0 剔掉了，而不是算错
了值。顺着往上查的是「这个 0 是从哪张 descriptor 取的」，不是「同步够不够」——同步缺失也可能表现为稳定错误，不能只凭是否偶发排除它；本例由实际地址表达式确认了写入重叠。

**修复方向（2026-10-22 核对）**：Store 的 GM 起点使用目的调度视图。当前
`getGlobalDescriptor` 在 destination 携带 `vectorized_axis` 时选择它，否则沿用局部视图
（图输出 destination 可能只有原始轴）。不再要求两张 axis 表逐字相同；计算 GM 起点时遍历
目的侧轴表，排除局部 vectorized_axis，必须保留 workspace 目的侧独有的块索引轴。
早期“轴表相等才取 destination”的补丁只覆盖同轴表场景，不是现行契约。

**通用教训**：

1. **「局部/全局」这对词描述的是存储空间，不是描述子的角色。** 一个 op 上挂着好几张
   descriptor，`operand_descs` / `result_descs` / `destination_descs` 各自描述的张量不同，
   而它们**在不同 op 上对应的空间不一样**。凡是要「乘落点的步长」的地方，都要显式回答
   「这一侧描述的是不是落点」，不能靠 Load 上碰巧成立的同源关系类推到 Store。
2. **「步长为 0 就整项跳过」这类优化会把取错的表和取对的表变得一样安静。** 取对时它省一次
   乘加，取错时它把错误从「乘了个奇怪的数」变成「整项不见了」——后者在生成代码里看不出缺
   了什么。这类剔零优化旁边要写清楚剔的依据来自哪张表。
3. **适用域的边界形状要专门造用例。** RCore 的可解条件是 `Ceiling(A/a0t) ≤ corenum`，而
   这些探针通过小 A、大 R 促使 R 方向分核；是否跨核应以实际 tiling 的 R 分片数及 phase 2 合并份数大于 1 为准；既有用例全落在 `nA == corenum` 那一档，于是
   整条跨核路径零覆盖。挑形状时要按**判据的分档**取点，不是按大小取点。

### 3.5 归约累加的 count 是「槽位跨度」不是「元素个数」：RA 下 A 组多轴时漏掉每个 tile 尾部几行

**症状**：三维 ARA（`[A_outer, R, A_inner]` 归约中间维）钉 RCommon 时，`[64, 32, 5]` 在
A_outer = 2, 5, 8, … 整行错，`[64, 8, 5]` 在 A_outer = 4, 9, 14, … 整行错。错的位置是
**每个 tile 的最后几行**，有严格周期。而 `[4, 32, 5]`、`[8, 256, 3]`、`[16, 16, 12]` 全绿。

**根因**：`emitAccumulation` / `emitMeanScale` 的 count 取 `align(Π A_actual)`，这等于假设
A 的结果槽位在 destination 里**连续**。AR 保证这一点；RA 是否连续取决于实际布局：

- **AR** 布局 `[A…, R…]`，对齐补白落在物理内维 R 上、与 A 无关，高阶 API 把 `Π A_actual`
  个结果连续写在 destination 开头。count 取元素个数正是对的。
- **RA** 布局 `[R…, A…]`，A 是物理内维，destination **镜像源的内维布局**：槽位
  `(a_outer, a_inner)` 落在 `a_outer × align(A_inner_nominal) + a_inner`。A 组多于一根轴
  且最内 A 的 nominal 不是 32B 整数倍时，有效槽位是**跨步**的。`[64, 8, 5]` 每 tile 装 5 根
  A_outer，count = `align8(5×5) = 32`，而真实跨度是 `5 + 4×8 = 37` —— 槽位 32..36 整行
  没被拷进 / 累加进 destination。

小形状压不出来：UB 预算会把每 tile 的 A_outer 压成 1 行，此时跨步与连续恰好等价。
**这是该缺陷从未被任何用例覆盖过的原因。**

**判别方法**：错的是**整行**且位置有周期（每 N 行错第 N 行）时，先算「一个 tile 装几根
A_outer」，再把 count 公式和 `1 + Σ (actual-1) × stride` 的真实跨度各算一遍对比。周期等于
每 tile 的 A_outer 根数，就是这一类。注意与 3.4 区分：3.4 是**全核写同一地址**导致的稳定
全错，这一条是**每个 tile 尾部缺一块**的周期性错。

**修复方向**：count 改成 destination 里 A 槽位的**跨度**，与
`materializeScheduledLocalSpan` 同一套公式。两处退化必须保留，否则既有产物不再逐字节相同：
AR 下直接复用 `outerActual` 那个 SSA 值（不要重算，重算会多出一串等价的 `arith.muli`）；
RA 的 A 组单轴时求和循环一次都不进，退化成 `actual` 本身。

**通用教训**：

1. **「个数」和「跨度」在有对齐补白的布局里不是一回事。** 凡是要喂给向量指令一个 count 的
   地方，都要显式回答「这段地址里夹不夹补白」。夹补白时补白会被一起算进去（累加成垃圾值、
   被一起乘 scale），这没问题——它们不 Store 回 GM；少算才是错。
2. **在册的「只记录不修改」项要当成未证实的猜测读。** 本条在 2026-09-21 的 RA 方案里被记成
   「这是 **AR** 今天已有的隐含前提」，按红线 7 挂起。后续于 2026-10-02 探测，结论是**反的**：
   AR 完全不受影响，缺陷只在 RA 下成立。既有的 `load_sum_store_arar` 注册就是 A 组两根轴、
   内层 A 为 4（`align(4) = 8`，不对齐）的 AR 语料，修前修后都过——这就是 AR 连续的实证。
   挂起一条缺陷时，连同它一起写下的**形态判断**也没有被验证过。
3. **探测用例要按「公式里有几个自由项」取档，不是按大小取一个点。** count 公式里有
   `Σ (actual-1) × stride` 这一项，单档只能钉住一个 A_outer 每-tile 数。只造一条
   `[4, 32, 5]` 会得到「探测通过、销号」的错误结论。

### 3.6 数值不对先分清「判据不合适」和「算错了」；在册结论会过期

本轮两次差点走偏，都不是产品缺陷，但都长得像缺陷。

**（一）长归约的 fp32 判据写死 `atol` 会假报缺陷。**
910B 上跑 RCore×RA `(131072, 20)`，`assert_close(rtol=1e-5, atol=1e-5)` 挂了。但：20 个元素
只错 1 个；绝对偏差 4.58e-5；由 `绝对偏差 / 相对偏差` 反推该元素真值 ≈ 0.77，而 131072 个
`randn` 求和的典型量级是 `sqrt(131072) ≈ 362` —— 这一格发生了约 470 倍的**相消**，相对判据
在这里没有意义。fp32 累加的舍入噪声约 `sqrt(n) × eps × E|x| ≈ 3.4e-5`，与实测吻合到 1.35 倍。

本组随机 FP32 用例采用经验容差：`atol = max(1e-5, margin × sqrt(n) × eps_fp32)`，`margin` 取 8。
这不是任意输入的严格误差上界；输入幅值、抵消和特殊值变化时需独立分析，不能自动放宽业务标准。

一眼区分噪声与真缺陷：

| | fp32 噪声 | 真缺陷（地址错位 / 丢项 / 漏块）|
|---|---|---|
| 错的元素数 | 个别（如 1/20） | 整行、整片，或 100% |
| 偏差量级 | 与 `sqrt(n) × eps × E|x|` 同阶 | **与真值本身同阶** |
| 规律性 | 换形状随机出现 | 有周期或稳定全错（见 3.4、3.5）|

**（二）「在册阻塞」的结论会被后续提交悄悄拆除，而文档不会自己更新。**
矩阵里 RCore 两格被记作「卡在 `Workspace` 节点与 tiling_key 命名分歧上，进不了 lowering」。
复测发现两道门都早已被别的提交修掉（命名由 TilingData 对齐 ATT 那次统一，`Workspace` 的
节点级处理由 workspace 链路那一串提交补齐）。**钉一下 plan 就直接编过了**，而文档还停在旧
结论上，下一个人会照着它继续绕路。

**通用教训**：

1. **接手一条在册阻塞，第一步是花五分钟复现它，不是读它的分析。** 分析可能是对的，但前提
   可能已经不成立；复现的成本远低于照着过期结论设计方案。
2. **按判据的分档取形状，不是按大小取形状。** RCore 是否跨核归约要检查实际 R 分片数与 phase 2 合并份数；`nA < corenum` 只是本组探针的选形状经验，
   本次 R 不分核的对照及单核 CPU E2E（`block_dim = 1`）不覆盖核间合并。注册用例时要把
   「这条覆盖不到什么」写进注释，否则它的绿会被当成覆盖到了。
3. **放宽判据之前必须先证明是噪声。** 先按上表三条核对，再改判据；顺序反了就是在掩盖缺陷。
