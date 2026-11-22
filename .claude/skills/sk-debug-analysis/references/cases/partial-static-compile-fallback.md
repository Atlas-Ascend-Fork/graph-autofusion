# 案例：DynamicQuant 静态编译失败

用户提供 `run.log`、`sk_meta`、`static_kernel_compile_outputs` 和 `log`。只分析已有产物，定位到失败算子及阶段。

下文已摘录关键证据，无需访问原现场文件。分析其他问题时，应从用户当前现场获取对应材料。

## 1. 从运行日志发现部分编译失败

运行日志中出现以下记录，包名已省略：

```text
[summary] 1/2 JSON files compiled successfully
generate run package ... success
[op_compiler] static kernel compile success
```

虽然打包和总体编译流程提示成功，但只有 1/2 个 JSON 编译成功，应继续查看失败汇总。

## 2. 查看 summary.txt，找到失败算子

在当前现场的 `static_kernel_compile_outputs` 中查找并读取 `summary.txt`，记录如下，长文件标识已省略：

```text
there are 1/2 JSON files that failed to static compile.
1. static_kernel_DynamicQuant_..._compile_error.log
```

## 3. 读取具体错误，确认定位

打开当前汇总列出的 DynamicQuant 编译日志，确认错误内容，而不只看文件名。日志记录 `op_compiler call opc failed`，并给出：

```text
[ERROR][FAULT_INJECTION] DynamicQuant static compilation deliberately failed
```

至此确认：**本次异常发生在 DynamicQuant 的静态编译阶段。** 具体错误日志标记为主动故障注入；本案例定位到此，不展开注入机制。

可复用步骤：**运行日志发现部分失败 → summary.txt 找到失败算子 → 对应编译日志确认具体错误。** 不仅凭总体成功提示或错误日志文件名下结论。
