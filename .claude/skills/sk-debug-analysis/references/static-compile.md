# 静态编译问题分析

## 1. 所需产物与初步检查

优先获取本次运行的 `sk_meta` 和 `static_kernel_compile_outputs`。分析静态编译产物未命中时，还需要至少 INFO 级别的 plog。

需要补采静态编译过程产物时，在运行前设置：

```bash
export ASCEND_GLOBAL_LOG_LEVEL=0
```

先检查 `sk_meta/<PID>/<modelId>/bin_files/*.o`，该目录保存 SK 获取到的真实运行时算子二进制。结合编译日志和安装记录，按以下四类问题排查：

1. 算子不支持静态编译；
2. 子算子静态编译失败；
3. 静态编译产物安装失败；
4. 静态编译产物未成功命中。

## 2. 算子不支持静态编译

**观察现象：** 目标算子对应的二进制没有 `static_kernel_` 前缀，且静态编译产物目录中没有对应的编译日志。

先排除以下情况：

- `sk_entry_*.o`：SK 框架二进制。
- 静态编译黑名单中的算子：检查对应版本的 [op_info_serialize.cpp](https://gitcode.com/cann/opbase/blob/master/src/nnopbase/common/op_info_record/op_info_serialize.cpp)。
- 不走静态编译的算子：这里是针对 ACLNN 算子，`<<<>>>` 算子没有静态编译过程，需要用户手动配置 `SK_BIND`，应检查 `SK_BIND` 配置；如果 `SK_BIND` 缺失或异常也可能导致该问题。

排除上述情况后，预期走静态编译的算子应具有 `static_kernel_` 前缀；缺少前缀时，继续分析编译、安装和运行命中情况。

## 3. 子算子静态编译失败

按以下顺序检查：

1. **查看打屏日志。** 静态算子编译会汇报编译情况，查找对应算子的编译失败信息。
2. **查看失败汇总。** 在 `static_kernel_compile_outputs` 中搜索 `summary.txt`，读取其中对应算子的编译结果和具体报错，不能仅凭文件存在判断失败。
3. **结合实际运行产物检查。** 查看 `bin_files` 中对应算子是否取得静态二进制；是否编译失败仍以编译报错为依据。

如果未使能 SK，可在 profiling 中查找与 `kernel_detail` 同级的 `task_time.csv`，筛选对应计算流，观察算子是否具有 `static_kernel_` 前缀，再结合编译日志排查。

## 4. 静态编译产物安装失败

1. 查看打屏中的安装提示和报错。
2. 检查对应安装目录中是否存在编译产物。
3. 若安装报错指向空间不足，检查当前机器的可用空间。

编译成功不代表安装成功。确认安装结果后，再分析运行时是否命中。

## 5. 静态编译产物未成功命中

### 5.1 从 plog 确认静态或动态执行

需要至少 INFO 级别的 plog。搜索以下日志，将 `<OpType>` 替换为目标算子类型：

| 执行类型 | ACLNN 自动框架日志 | ACLNN 手写框架日志 |
|---|---|---|
| 静态 kernel | `Launch static kernel <OpType>` | `Available static bin for op <OpType>` |
| 动态 kernel | `Launch dynamic kernel <OpType>` | `Available bin for op <OpType>` |

根据命中的日志识别该算子使用的框架和执行类型，再按以下场景分析。

### 5.2 全部未命中或运行产物不一致：优先排查缓存

针对 ACLNN 算子，以下情况优先考虑缓存影响：

- 预期静态编译的算子全部未命中，`bin_files` 中对应算子均无 `static_kernel_` 前缀。
- `static_kernel_compile_outputs` 中生成的产物与 `bin_files` 中实际运行产物不相符。

保存已有现场后，可通过清理缓存并重跑验证。以下命令会删除产物，应先确认实际路径、影响范围和操作授权；不要在只读分析时执行：

```bash
rm -rf ./static_kernel_compile_outputs
rm -rf "${ASCEND_HOME_PATH}/opp/static_kernel/"
```

重跑后再次比较新编译产物与实际运行产物，确认缓存影响是否消除。

### 5.3 部分未命中或同一算子静动态混用：核对输入配置

针对 ACLNN 算子，以下情况优先考虑编译产物与实际调用的输入类型、shape 不同，即 simplified key 不匹配：

- 部分算子走静态 kernel，部分算子走动态 kernel。
- 同一算子既走静态 kernel，也走动态 kernel。

**定位步骤：**

1. **确认静态 kernel run 包已经安装。** 以 run 包生成时间辅助缩小日志范围，同时确认所检查的调用发生在包安装之后。
2. **定位动态调用。** 从安装后的日志中找到目标算子的动态 kernel 执行记录，可优先查找最后一次动态执行。
3. **提取调用入参。** 从该调用前后的 plog 中提取算子输入类型、shape 等入参配置。
4. **遍历编译配置。** 将该调用入参与所有相关静态 kernel JSON 对比，检查是否存在完整匹配。存在多个编译产物目录时，需要全部检查，不能只看一个目录或首个 JSON。
5. **根据匹配结果选择后续路径。** 按下一节判断应继续排查 ACLNN 组件还是网络脚本的静态编译触发过程。

可检查的 JSON 位置如下，时间戳、PID、SoC 和算子类型按现场替换：

```text
# 静态编译入参 select JSON
./static_kernel_compile_outputs/ts<timestamp>_pid<pid>_outputs/<pid>_opcompile_selected/<OpType>_*.json

# 静态 kernel run 包中的 JSON
./static_kernel/ai_core/static_kernel_<timestamp>_<pid>/<soc_version>/<OpType>/static_kernel_<OpType>_*.json

# 静态编译中间产物中的 JSON
./static_kernel_compile_outputs/ts<timestamp>_pid<pid>_outputs/tmp_<timestamp>/static_kernel_<OpType>_*.json
```

### 5.4 匹配结果与后续定位

- **存在完整匹配的 JSON，但安装后的调用仍选择动态 kernel：** 将问题定位到 ACLNN 运行时选择路径，继续分析为何没有使用匹配的静态产物。
- **所有 JSON 均无法匹配：** 检查网络脚本，定位为什么没有触发该算子此入参配置的静态编译。

编译和安装成功、可见入参相符却仍动态执行时，继续比较编译侧与安装侧索引的属性类型及 key 编码，并确认本次进程实际读取的登记项。可按需参考 [案例：静态编译成功但运行未命中](cases/static-registration-key-mismatch.md)，了解如何从产物确认异常阶段、区分注册 key 与运行查询 key，并明确候选原因、证据缺口和下一步验证。

静态编译的触发条件是：model 生成 NpuGraph 图后，在非捕获状态下，首次执行该入参配置。检查网络脚本是否满足这一条件。

## 6. 深入分析入口

### 相关组件与源码

- **AscendC 框架：** 提供静态编译能力识别、产物组装、安装卸载，以及算子基础 API、C API 和高阶 API 等能力。
- **opc 算子编译工具：** 提供静态编译执行能力。
- **ACLNN 算子框架：** 分为手写框架和自动框架，可通过前述执行日志识别。

静态编译相关子包为 `cann-tbe-tik`。工具源码随 CANN 包发布，位置为：

```text
$ASCEND_HOME_PATH/python/site-packages/op_compile_tool/static_compile
```

### `<<<>>>` 算子的 AOT 编译优化

需要分析这类算子的静态 tiling 编译优化时，参考：

- [AOT 编译优化说明](https://gitcode.com/cann/asc-devkit/blob/master/docs/zh/guide/programming_guide/advanced_programming/aot_compilation_optimization.md)
- [Matmul AOT 编译优化示例](https://gitcode.com/cann/asc-devkit/tree/master/examples/01_simd_cpp_api/02_features/06_aot_compilation/matmul_aot_compilation)
