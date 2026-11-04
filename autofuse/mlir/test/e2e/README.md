# 1 使用范围

CPU 和 NPU 用例使用不同入口：CPU 手工构造 `AscGraph`，经过 Optimizer 和 MLIR codegen 后，
通过 tikicpulib 验证生成的 AscendC kernel；NPU 从 Inductor 构造输入进入编译链，在真实设备上验证结果。
目前只实现了 CPU 用例，NPU 的 Inductor 入口和真机执行尚未实现。

下文依次介绍目录、运行方法、新增用例和验证边界。

# 2 目录

- `cpu/support/`：CPU 生成器入口、用例函数声明、源码输出和 tiling stub。
  图构造工具、runtime stub 和 tikicpulib 都由 CPU CMake 引入。
- `cpu/cases/<name>/`：同一个 CPU 用例的手工构图、生成代码检查、内存分配、kernel 调用、数值断言和 CMake 注册。
- `npu/`：说明尚未实现的 Inductor 入口和真机测试边界，不复用 CPU 的手工构图和生成器。

每个用例都有自己的生成器、生成目录和测试程序。即使用例都生成名为 `AutofuseTilingData` 的类型，
它们也不会进入同一个程序。图构造工具只编译一次，供各生成器链接。

# 3 运行现有用例

前置条件：Linux CANN 开发环境，已加载 `set_env.sh`，且工程已经使用
`ENABLE_AUTOFUSE_MLIR=ON` 和 `ENABLE_AUTOFUSE_MLIR_TESTS=ON` 配置。
CPU E2E 默认开启，并要求 CANN 提供 `AF_MLIR_E2E_CPU_TIKICPULIB_TARGET` 指定的目标。

从仓库根目录运行当前已实现的命令，编译并验证数值结果：

```bash
cmake --build build --target check-e2e-cpu -j 8
cmake --build build --target check-e2e-cpu-load_add_store_1x67x209 -j 8
```

单个用例的目标名是**注册名**，不是目录名：注册名按约定是「目录名 + `_` + 形状后缀」
（`load_add_store` 目录注册了 `load_add_store_1x67x209` / `load_add_store_128x23x5` /
`load_add_store_128x128x128` 三条）。形状改了注册名跟着改，`ctest -N` 可以列出当前的全部名字。

编译完成后，可使用 CTest 只运行 CPU 用例：

```bash
ctest --test-dir build/autofuse/mlir/test/e2e -L '^mlir_e2e_cpu$' --output-on-failure --no-tests=error -j 8
```

`check-e2e` 汇总 E2E 后端，`check-mlir` 和 MLIR ST 脚本调用该汇总目标。
CPU E2E 是 MLIR 测试的必备部分，随 `ENABLE_AUTOFUSE_MLIR_TESTS=ON` 一起启用；
缺少 tikicpulib 时配置失败，而不是静默跳过 CPU 验证。

# 4 新增和拆分用例

以当前已实现的 `load_add_store` 为例：

1. `cpu/cases/load_add_store/Case.cpp` 实现 `BuildGraph()` 和 `CheckGeneratedKernel()`。
   `Case.h` 定义图构造和执行代码共用的元素数量。
2. CPU 生成器调用 `BuildGraph()`，执行优化和 codegen，再将源码交给 `CheckGeneratedKernel()` 检查。
3. CPU 的 `WriteSources()` 去掉 host launch，输出 kernel 和 tiling 头文件。
4. `cpu/cases/load_add_store/KernelTest.cpp` 运行生成的 kernel 并检查结果。

新用例按同样方式创建目录，在 `cpu/CMakeLists.txt` 中添加 `add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/<name>" cases/<name>)`。
用例的 CMake 使用当前已实现的注册函数：

```cmake
add_mlir_e2e_cpu_case(load_add_store_1x67x209
    OUTPUT_SHAPE "[1, 67, 209]"
    INPUT_SHAPES "[1, 67, 209]" "[1, 67, 209]"
    GENERATOR_SOURCES ${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_add_store/Case.cpp
    TEST_SOURCES ${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_add_store/KernelTest.cpp
)
```

形状由 CMake 给，用例源码从生成的 `CaseConfig.h` 读，所以换形状不用改 C++。**新用例不要用
一根轴、也不要用只含 2 的幂的形状**：那样切出来的每趟 count 恒是 32B 对齐，「尾块不对齐」
和「一趟多于一个 repeat」两类判据都探不到。现有用例是 2~4 轴、每轴取 [1, 512] 里的随机值。

`GENERATOR_SOURCES` 和 `TEST_SOURCES` 都接受多个文件。用例增长时可把图构造、源码检查和不同输入的
测试拆成多个 `.cpp`，只需补充对应列表；每个生成器只能有一个 `BuildGraph()` 和 `CheckGeneratedKernel()` 定义。
CPU 生成器与执行框架无需修改。

MLIR 内部头文件从 `autofuse/mlir/` 根目录写完整路径，例如
`#include "test/e2e/cpu/support/E2eCase.h"`。CMake 中的源码和脚本路径使用 `${MLIR_CODE_ROOT_DIR}/...`。
外部库头文件遵循所属库的 include 规则；`autofuse_tiling_data.h` 从当前用例的生成目录查找。

优化、codegen 或源码检查失败时，生成器返回失败，构建不会继续编译 kernel；源码检查失败不会写出新产物。
注册时漏写 `TEST_SOURCES` 或 `GENERATOR_SOURCES` 会在 CMake 配置阶段报错。

# 5 验证边界

CPU tiling stub 返回 `false`，当前执行用例手工设置 tiling 数据。这条链路覆盖优化、MLIR codegen、
生成源码编译和 tikicpulib 数值执行，不覆盖 ATT tiling 计算、host launch 或真实设备运行。
CPU 生成器复用已有 runtime stub，让优化器可以在没有 NPU 的主机上查询平台信息；stub 的平台数据
来自测试桩，修改 SoC 配置时须同时确认桩数据和手工 tiling 是否适用。

CPU 生成器调用的 `GenerateForInductor()` 是 codegen 接口；这个调用不表示测试运行了 Inductor。

未来 NPU 用例应在 `npu/` 下使用 Inductor 构造输入，覆盖 Inductor 接入、MLIR codegen 和真机执行，
并提供独立开关及标签。NPU 测试应走真实 tiling 和 runtime，不能调用 CPU 的手工 `BuildGraph()`、
链接 CPU runtime/tiling stub，或裁掉 host launch。目前这些 NPU 测试能力均未实现。
