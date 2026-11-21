# v35 NDDMA 模型增补

ATT 先核对 Codegen 描述符和 dtype size，再归一化物理轴。联合连续的 GM/UB 轴可折叠为物理 1D，按 1/2/4/8 字节分别取已有 1D 形式的标定参数。非连续的有效轴保留，估值使用对应布局的公式。动态条件生成原表达式与拟合式的选择；低核和无法证明适用的描述符保持各自原估值路径。

| 分支 | 估值与适用域 |
| --- | --- |
| Packed NG2 | 1/2/4 字节类型，高核，内侧 GM unit stride 且 UB pitch 等于 inner。小 inner 与大 inner 分开拟合，分段以代码中的 inner 字节阈值为准。符号 pitch 用相等条件选择。 |
| Padded FP32 NG2 | M=12–32、N=193–519、GM stride `[1,M]`、UB pitch 为 N 对齐到 8 元素且实际有 padding、16–56 核。估值为 `162.529402 + B/65536 * (6594.471272 + 565.933501 * max(C-32,0)/24)`。 |
| Small INT64 NG2 | M=24–64、N=2–8、GM stride `[1,M]`、UB pitch 为 N 或 N+1、16–56 核，启动主导约 506 cycle。 |
| Packed rank 3 | 4 字节类型、连续批间 pitch、每批在 GM/UB 间交换平面内两轴，外轴实际大于 1、3–56 核。估值为 `177.398972 + 76.277101 * B/1024`。 |
| Sparse rank 3 | 4 字节类型，`[W,32,2]`，W=192–256，GM pitch 65536 或 131072 元素、UB packed、40–56 核。保留带宽和启动项，将旧 stride 残差乘 0.017212331646550515。 |
| Sparse rank 4 | 4 字节类型，W=512/1024，GM pitch 65536 元素，UB pitch 按生产描述符对齐。两个 tile 轴受 `tile1<=3`、`tile2<=4 或 tile2=8`、`tile1*tile2<=12`、`W*tile1<=2048` 限制。packed/padded 旧残差分别乘 0.016809401000806315 和 0.030215324807753154。 |

B 是每核每调用 payload 字节数，C 是实际活跃核数。这里的核数界限来自各拟合域。通用的 NDDMA cacheline 补齐参数与 MTE2 的公共 head 分开计算。`NDDMA_RUNTIME_DISPATCH[match|else]` 描述生成期保留的模型分支。两侧可再包含 guard，求解时使用同一估值表达式。

[用例清单](cases.csv)与[实测归档](results/pr2921-d68/report.md)保存各组已归档训练、留出、已知候选和域外对照。归一化、物理 1D、动态 packed/padded 守卫、rank 3/4 和 INT64 等价及回退用例见 `NddmaModelV2.*`。同一核数可能有不同 descriptor，不能将残差比例从表中的布局外推到其他转置。


## 窄连续源与长转置轴

GM 连续宽度只有 1–8 个 FP32、转置轴为 512–16384 元素时，按 UB payload 的 packed NG2 式可能低估实际读请求。
新增分支沿用启动、线性搬运与核间竞争的结构，按 strided position 数计价，具体域与冻结系数见
[独立报告](results/narrow-source-v1/report.md)。
只应用于 FP32、源连续轴 step=1、GM pitch=30–512 元素、UB packed 的有效物理描述符。
单位轴的静态物理 1D 与符号 tile=1 使用相同的拟合成本。
域外沿用原式，型号与设备总核数不参与分派。活跃核数 C 仍由当前候选提供。

该修正针对 0120、0134、0166 默认窄读候选低估，采样核数为 24/32/40/48/56；
更高核数的真实设备验收仍需对应环境执行。
UT 覆盖物理单位轴、其他 dtype、域外描述符和符号边界。

## Packed 二维 NDDMA 的直通写回

[Direct Store 增补设计](results/direct-store-v1/design.md)在结构证据和物理描述符均命中时，
为原 packed NG2 估值增加独立标定的读写并发成本。它不修改孤立 NDDMA 系数，不更改 Codegen 同步。
训练、留出、适用域与当前默认验证状态见同目录 report.md。
