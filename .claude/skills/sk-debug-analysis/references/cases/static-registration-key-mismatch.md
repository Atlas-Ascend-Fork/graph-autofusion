# 案例：静态编译成功但运行未命中

用户提供 `net01_sk_options` 的 `sk_meta`、`static_kernel_compile_outputs` 和 `log`，希望定位 IncreFlashAttention（IFA）未融入 SK 的原因。下面按实际检查顺序整理关键证据和判断。

## 1. 从 SK 日志发现 IFA 绑定失败

`sk_fusion_fail_reasons.log` 中，IFA 的记录包含：

```text
resolvedNum=0
isFusible=0
reason: OP_UNSUPPORT [BINDMAP_INIT_EMPTY]
```

这说明 SK 没有解析到该节点的绑定入口，但还不能解释上游原因。继续检查 `bin_files`，发现本次取得的 IFA 二进制与静态编译目录中的产物不同。

因此先追查：**IFA 是否编译成功、是否安装，以及运行时究竟选中了哪个二进制。**

## 2. 确认目标静态二进制已编译、已安装

编译打屏日志记录 run 包生成成功。进一步打开 IFA 对应的编译日志，看到以下成功记录，省略长文件标识：

```text
Op[IncreFlashAttention] of index[0] compile success, kernelName:[static_kernel_IncreFlashAttention_...].
Opc tool compile success.
```

`log/installation/install.log` 也记录安装成功。安装快照中的 IFA 静态 `.o` 与编译侧最终打包文件 SHA-256 相同，大小均为 67,000 字节。

至此确认：**IFA 静态编译成功，且对应二进制已安装。** 接下来检查为何运行没有使用它。

## 3. 从 plog 确认静态查找失败后选择了动态 kernel

在安装后的本次 IFA 调用中，plog 记录如下，省略时间和日志前缀：

```text
Start find binInfo, opType is IncreFlashAttention, hashKey is 645, verbLen is 515.
There is no available IncreFlashAttention binInfo for hashKey 645.
Cannot find static kernel bin with stride information, trying to find without stride again.
```

随后不带 stride 的查找也失败，执行日志明确记录：

```text
Launch dynamic kernel IncreFlashAttention task, tilingKey is 11000000000000001, ...
```

该记录与 SK 节点及 `bin_files` 中的动态 IFA 对应，运行二进制大小为 1,567,288 字节。

因此，问题已经定位到 **ACLNN 静态产物查找未命中**。SK 随后取得的是动态 IFA，其绑定解析失败是本次未融合的直接表现。stdout 中已有输出校验通过，程序也执行完成。

## 4. 对比调用配置，进一步发现登记 key 编码不同

为检查是否编译了不同的输入配置，将本次调用日志与静态编译 JSON 对比：Q/输出均为 `[3,1,1024]`，K/V 均为 `[3,256,1024]`，stride、offset 相符；`actual_seq_lengths=[99,199,180]`，属性为 `[16,1.0,"BSH",0,0,1]`。可见配置没有发现不匹配。

继续比较编译侧和安装侧的 `binary_info_config.json`，发现 IFA 的 `simplifiedKey`、`simplifiedKeyWithPlatform` 中存在同一处差异：

| 索引位置 | `scale_value=1.0` 对应的 key 片段 |
| --- | --- |
| 编译产物及原始编译包 | `.../10/3f800000/485342/0/0/1` |
| 安装侧全局索引及包内索引 | `.../10/3ff0000000000000/485342/0/0/1` |

`3f800000` 和 `3ff0000000000000` 分别是数值 `1.0` 的 float32、float64 编码。差异在于编码宽度。

再查 plog，`NnopbaseCollectorConvertStaticVerbKey` 确实读取了包含 `3ff0000000000000` 的 IFA 登记 key，随后以 hashKey 917 插入表。这证明该差异索引已被本次进程使用。

此处看到的是**登记 key**；第 3 步看到的是**运行查询的哈希及失败结果**。日志没有打印完整运行查询 key，尚不能逐字节比较两侧。因此，登记编码差异是重要的未命中线索，其因果作用还需验证。

## 5. 本次定位结论

**已经确认：** IFA 静态编译、安装成功，但本次静态查找未命中，运行选择了动态 IFA，SK 未能解析其绑定入口。

**进一步找到的异常点：** 安装侧登记 key 与原始编译产物的浮点属性编码不同，而且本次进程确实读取了该差异。它是当前最有证据支持的候选原因。

下一步建议保留现场，在隔离副本中仅将实际生效的 IFA 两个登记字段恢复为原始编译包值，保持二进制、输入和其他选项不变，用新进程验证。若静态命中及绑定解析恢复，可确认该登记差异的因果作用；若仍未命中，则补采完整查询 key 继续比较。
