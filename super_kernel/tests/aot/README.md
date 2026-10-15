# SuperKernel AOT 主机侧测试

本目录包含 SuperKernel AOT 的单元测试（UT）和主机侧系统测试（ST）。
ST 从公开 API 进入完整 AOT 主机流程，以共享 Runtime/ACL stub 提供外部模型和任务状态。
ST 链接测试专用 `libascendsk_st.so`，该库使用与生产 `ascendsk`
相同的 AOT 源码列表（包含 `sk_dump_json.cpp`）及编译、可见性、ABI 和链接加固配置。
测试库不安装，不链接真实设备内核或 Runtime 库。

## 目录与边界

| 目录 | 职责 |
| --- | --- |
| `ut/` | 内部类和函数的单元测试，保留 gtest/mockcpp；包括共享模型夹具的七个 UT |
| `st/` | 从四个公开 API 验证跨模块行为，只包含公开 API 与测试依赖头文件 |
| `depends/` | `super_kernel_aot_stub` 共享库、Runtime/ACL stub 和外部 RI 模型夹具 |
| `cmake/` | 公共测试选项、运行与覆盖率脚本 |

ST 可执行文件和 `ascendsk_st` 链接同一个 `super_kernel_aot_stub` 共享库，因此故障注入、
任务状态、分配记录和销毁回调只有一份。`intf_llt_options` 提供 ABI、coverage 和 sanitizer
选项；UT 的 `intf_llt_ut` 另行引入 gtest/mockcpp，ST 与共享 stub 不依赖 mockcpp。

| 公开 API | 验证边界 |
| --- | --- |
| `aclskOptimize` | 外部 RI 模型读取、scope 处理、融合任务替换、跨流同步、模型更新和资源清理 |
| `aclskScopeBegin` | scope 起始 marker 的名称、流和派发行为，以及非法输入 |
| `aclskScopeEnd` | scope 结束 marker 的名称、流和派发顺序，以及非法输入 |
| `aclskScopeVerify` | 公开图描述的校验和拆分结果，包括返回节点指向原始输入 |

当前 ST 场景覆盖：无 marker 时默认全模型 scope、单 scope 融合、scope 外跨流 event 保持、
融合 record 后外部 wait/reset 重写与同步内存清理、未配对 scope、模型 update 失败、
begin/end marker 和 verify。扩展场景包括 cube/MIX 1:1/MIX 1:2 入口选择、逐算子调试、
跨流同名 scope 合并、调试 JSON 的选项和任务顺序、Runtime 查询/入口解析/同步内存初始化失败，
以及 Verify 的输出容量、动态核限制、跨流死锁和非法输入。进一步覆盖 SIMT 动态 UBUF、
多流 scope 与内存 wait/write、选项边界、编译器能力及入口绑定、混合流水线、DFX 异常回调，
以及 profiling 启停、退出导出和失败清理。夹具 UT 覆盖查询容量与状态保持、
多模型隔离和参数快照所有权、同名不同核类型的元数据隔离、参数深拷贝，
非 kernel 参数复制、编译器 metadata 字节协议，以及旧 UT 使用的不透明设备地址兼容行为。

profiling 和 SIMT 场景在独立子进程中运行，隔离生产单例；子进程正常退出以验证 recorder
清理并写回覆盖率，超时 15 秒则失败并回收进程。子进程不继承 GTest 分片配置，以免精确过滤的
用例被再次分片后漏跑。DFX 仅复制真实融合入口参数并提供外部异常寄存器和标准 ELF 符号，
不拼装或解析私有 SK 参数。profiling 当前验证空事件导出，不模拟设备生成的事件记录。

测试执行不需要 NPU。当前顶层 CMake 配置仍会查找 CANN 包并配置 ASC 编译器，因此仍需可用的
CANN Toolkit 和环境配置。这些测试验证主机流程与 stub 契约，不能证明真实 Runtime ABI
兼容性、设备执行结果或数值正确性；这些结论需要实际设备测试。

## 运行

在仓库根目录配置当前环境对应的 CANN Toolkit 后运行：

```bash
source /path/to/cann/set_env.sh
bash build.sh -s --module=superkernel --impl=cpp --no-autofuse -j 8
```

可选择第三方依赖缓存、用例过滤和覆盖率。以下命令展示三者组合；不需要的参数可以省略：

```bash
bash build.sh -s --module=superkernel --impl=cpp --no-autofuse -j 8 \
    --cann_3rd_lib_path=/path/to/third_party \
    --test_case='AotSystemTest.Optimize*' -c
```

ST 覆盖率输出到 `super_kernel/coverage/cpp_st/`，HTML 入口为 `html/index.html`。
ST 只扫描自身构建目录中的覆盖率数据，报告仅保留 `super_kernel/src/aot` 生产文件，
不计入 stub、用例或第三方头。UT 报告继续使用 `super_kernel/coverage/cpp_ut/`。
过滤运行得到的是所选场景的覆盖率，不能作为完整测试覆盖率。`-c` 路由会先清理构建目标再运行。

已有构建目录可使用独立 CMake 开关和目标：

```bash
cmake -S . -B build -DBUILD_AUTOFUSE=OFF \
    -DENABLE_CPP_UTEST=OFF -DENABLE_CPP_STEST=ON -DENABLE_GCOV=OFF \
    -DGTEST_FILTER=
cmake --build build --target super_kernel_aot_stest -j 8
cmake --build build --target run_super_kernel_aot_stest -j 8
```

首次配置还应按本地环境提供 `ASCEND_INSTALL_PATH` 和 `CANN_3RD_LIB_PATH`。
`ENABLE_CPP_UTEST` 与 `ENABLE_CPP_STEST` 可以分别开启，也可以同时开启；切换时应显式设置
两者，避免继承缓存中的旧开关。`build.sh` 会显式设置所选测试开关。
UT 对应目标为 `super_kernel_aot_utest` 和 `run_super_kernel_aot_utest`。
开启 `ENABLE_GCOV=ON` 后，ST 覆盖率目标为 `collect_coverage_data_cpp_st`，
UT 保留 `collect_coverage_data`。直接配置过滤器时使用
`-DGTEST_FILTER=--gtest_filter=AotSystemTest.Optimize*`，并对整个参数加引号以避免 shell 展开。

## 扩展夹具与用例

新增 ST 放入 `st/test_*.cpp`，CMake 会自动发现；复用 `st_fixture.h` 的 `AotSystemTest`。
通过 `sk::test::Model` 创建外部模型，使用 `AddStream`、`AddKernel`、`AddEvent`、`AddTask` 构造输入，
调用公开 API 后使用 `Tasks`、`Snapshot`、`Launches` 和更新计数观察结果。
不要包含生产私有头文件，也不要直接构造内部 `SkGraph` 或调用内部优化器替代公开入口。

新增 Runtime 行为应在 `depends/` 中表达外部 API 契约，并补充夹具 UT，再以 ST 验证生产流程。
夹具负责句柄有效性、参数所有权和可观察状态，不应复制融合算法来计算预期结果。
`SetParams` 保存 host 参数、配置属性与 opInfo 的副本；函数元数据在进程内保持稳定，
避免生产 binary cache 引用失效。`KernelSpec` 描述核类型、比例和编译器 capability，
`SetKernelBindings` 覆盖外部二进制绑定，必须在首次 Optimize 消费该 binary 之前设置。
`AddTask` 只接受非 kernel 任务；其中引用的外部地址由调用方保持有效。
当前模型夹具提供 AIC/AIV/MIX 元数据、record/wait/reset 事件与内存 wait/write，
用例串行运行，不提供通用设备模拟或并发 Runtime 状态。

资源生命周期必须先 destroy、后 reset：让 `Model` 析构或显式调用 `Model::Destroy()`，
执行已注册的生产清理回调后，再重置全局 stub 状态。需要断言资源释放时，先调用 `Destroy()`，
再检查分配数和回调数；提前 reset 会擦除状态并掩盖泄漏。用例中的局部模型在 fixture 的
`TearDown()` 前析构，可保持这一顺序。

## 失败诊断

ST fixture 会在失败时输出缓冲的生产日志，成功时清空缓冲。运行目标会把用例输出和错误输出
写到终端，并传播测试失败退出码。指定过滤器但未选中任何用例
会失败，出现 skipped 用例也会失败。运行器使用临时日志并在退出时清理；需要保留日志时可运行：

```bash
set -o pipefail
bash build.sh -s --module=superkernel --impl=cpp --no-autofuse -j 8 \
    --test_case='AotSystemTest.*' 2>&1 | tee /tmp/super-kernel-aot-st.log
```

先区分 CANN/第三方依赖配置失败、编译链接失败和用例断言失败。链接出现 `testing::*`
未定义符号时检查 gtest 核心库是否进入链接命令；资源断言失败时检查模型销毁与 stub reset
顺序。断言信息中的任务类型、disabled 状态、参数更新次数和同步地址可帮助定位改写差异。

当前保留两处生产问题的复现：混合流水线的末尾任务同时声明 wait/set 能力时，early-start
可能构造没有关联节点的同步并返回失败；单 kernel 核数超出设备上限时，scope 拆分可能不终止。
后者用例 `DISABLED_RuntimeCoreLimitKeepsOversizedKernelOutsideFusion` 默认禁用，不能计入通过数量，
修复生产代码前不要在常规测试中启用。
