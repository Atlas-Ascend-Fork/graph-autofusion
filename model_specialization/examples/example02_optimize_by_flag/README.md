# 指定 Scope 常量化编译样例

## 用例功能

本样例捕获两次自定义 Add 调用，通过 `aclmdlRISpecOptimizeByFlag` 仅对 Scope 内的第二次调用进行常量化编译。

## 目录结构

```text
examples/
├── common/
│   └── sample_resources.h  # 共用 Runtime、Tensor 和 executor 资源管理
└── example02_optimize_by_flag/
    ├── CMakeLists.txt      # Host 编译工程
    ├── main.cpp            # 数据准备、捕获、优化、执行和校验
    ├── README.md
    └── README_en.md
```

## 用例介绍

共用的自定义 Add 算子对两个形状为 `[1024]` 的 float32 Tensor 逐元素相加，即 `z = x + y`。输入设置为 `x[i] = i`、`y[i] = 1`。

本样例依次捕获 `t = x + y` 和 `z = t + y`，仅将第二次 Add 放入常量化 Scope。优化并执行模型后，Host 同时校验中间结果 `t[i] = i + 1` 和最终结果 `z[i] = i + 2`。

### 调用实现

1. 初始化 ACL、Context、Stream，准备输入数据并创建 float32 Tensor。
2. 调用 `aclnnAddCustomGetWorkspaceSize`，申请 workspace，并设置 executor 可重复执行。
3. 捕获算子调用，仅将第二次 Add 放入常量化 Scope。
4. 结束捕获，调用 `aclmdlRISpecOptimizeByFlag`。
5. 执行模型并同步 Stream，将输出拷回 Host，校验计算结果。
6. 先销毁模型，再释放 executor、Tensor、Device 内存及 Runtime 资源。

## 环境依赖

| 产品 | 是否支持 |
| --- | --- |
| Ascend 950PR/Ascend 950DT 系列产品 | √ |
| Atlas A3 系列产品 | √ |
| Atlas A2 系列产品 | √ |

请参考[源码构建指南](../../../docs/zh/build.md)准备环境。本样例需要 CMake 3.16 或更高版本、C++17 编译器和支持本样例的 CANN 环境。请安装或构建 graph-autofusion，确保可使用 `libmodel_specialization.so`。

CANN 环境须提供 ASC 算子包构建和 Kernel Spec 编译能力。两个样例共用 `../custom_op`，算子包只需构建和安装一次。

## 执行命令

在本样例目录下依次执行以下步骤。

1. 加载 CANN 环境：

   ```bash
   source /usr/local/Ascend/cann/set_env.sh
   ```

   如使用其他安装路径，请加载对应的 `set_env.sh`。

2. 构建共用自定义算子包：

   ```bash
   cmake -S ../custom_op -B ../build/operator -DASCEND_COMPUTE_UNIT=Ascendxxx
   cmake --build ../build/operator --target binary package -j8
   ```

   将 `Ascendxxx` 替换为实际运行的 AI 处理器系列。工程已通过 `KERNEL_SPEC_MODE Normal` 开启 Kernel Spec，算子包包含 Kernel 二进制、编译资源和源码。

3. 将算子包安装到样例构建目录：

   ```bash
   mkdir -p ../build/opp
   bash ../build/operator/custom_opp_*.run --quiet --install-path="$(cd ../build/opp && pwd)"
   source ../build/opp/vendors/customize/bin/set_env.bash
   export ASCEND_CUSTOM_OPP_PATH="$(cd ../build/opp/vendors/customize && pwd)"
   ```

   两个样例共用此算子包。已构建并安装时，跳过第 2 步及第 3 步的前两条命令；重新打开终端时，加载 CANN 环境后，仍需执行 `source` 和 `export ASCEND_CUSTOM_OPP_PATH`。

4. 编译 Host 程序：

   ```bash
   cmake -S . -B build/host
   cmake --build build/host -j8
   ```

   模型库不在默认查找目录时，在配置命令中追加 `-DMODEL_SPECIALIZATION_LIBRARY=/path/to/libmodel_specialization.so`；算子安装到其他位置时，追加 `-DOPP_VENDOR_PATH=/path/to/vendors/customize`。

5. 在 NPU 上运行：

   ```bash
   export ASCEND_GLOBAL_LOG_LEVEL=1
   export ASCEND_SLOG_PRINT_TO_STDOUT=1
   ./build/host/demo 0 > build/host/specialization.log 2>&1
   ```

   前两条命令开启 INFO 级别日志及控制台输出，日志和程序输出一并保存到当前样例目录的 `build/host/specialization.log`。运行后可按“预期执行结果”中的说明查看日志，无需再次运行样例。

   参数为设备 ID，省略时使用设备 0。

   > [!WARNING]
   > **注意**：本样例仅支持 NPU 运行模式，不支持 CPU 调试或 SIM 仿真模式。

## 预期执行结果

样例执行成功时会输出：

```text
Numerical check passed: 1024 float elements
Numerical check passed: 1024 float elements
Example passed
```

`Example passed` 表示模型执行和输出校验通过。模型使用原始 Kernel 时也可能得到正确结果，因此仅凭该输出无法确认常量化是否生效。

第 5 步已开启日志。可使用以下命令查看常量化编译、任务更新和程序校验结果：

```bash
grep -E "specialization completed|specialization failed|HandleTaskUpdate: update task finish|Mismatch|Numerical check passed|Example passed" build/host/specialization.log
```

```bash
grep -F "[IN_SCOPE]" build/host/specialization.log
```

先检查日志中 AddCustom 二进制的加载路径，确认来自 `../build/opp/vendors/customize`。再查看目标 Kernel 的 `specialization completed` 和对应任务的 `HandleTaskUpdate: update task finish`，分别确认常量化编译完成和任务更新完成。

通过 `[IN_SCOPE]` 日志确认只选中了第二个 AddCustom，并检查只有该 Kernel 任务被更新。两次调用的 Kernel 名称相同，请按任务 ID 区分。
