# Direct Store NDDMA：冻结结果

业务候选、完整 tiling、Host 计时和 17/48 例端到端回归记录由 [CANN-DevTools](https://gitcode.com/zhang_shengjie/CANN-DevTools/blob/6733c737b04f2e6289ebeb5784adaccb7c1e6d30/att_generalization/report/20261119/pr3065-model-regression/README.md) 按固定版本跟踪。本目录保留 API 建模依据、独立训练/留出数据及验证摘要。

产品基线是 PR #3065 的 caeee3d1e64cfd57a5765291a5ec9c046ef09d3a；实际设备为 Ascend950PR_9579，56 AIV。
该结果不代表用户 64 核整网已验证。

## 独立采集与拟合

最初采集 19 个描述符。最终适用域为 GM 连续源≤32B，即 FP32 M=2–8；域内 7 train、5 holdout、2 known，
其余 5 点仅作为域外对照。冻结新域及系数后另采 5 个未见过的独立 holdout，最终域内共 19 点。各点均通过输出及保护区校验。
使用仓内 `tools/modeling/v35/dma/src/steady_load.asc` 的 padded 5D NDDMA、TQueBind 双缓冲、NORMAL hint、
4 GiB GM 池与调用/核/launch 地址轮换。每点预热 3 次、有效 10 次；K=1/2/4/10。
每核 MTE2 周期为 `aiv_total_cycles*aiv_mte2_ratio/Block Num`，K 对中位周期的斜率为每 API 成本。

`isolated-samples.json` 使用 tail_store，`samples.json` 使用 span_store 完整写回。
完整写回的采集器还包含保护区，而生产 Store 使用实际数据和行间 pitch，因此该实验是受控干预而非逐条相同的生产 kernel。
known `[8,3952]` 的 MTE2 从 9637.1 增到 16107.2 cycle；真实生产主块为 `[8,3960]`，
因采集保护区预算用相邻、相同对齐的 3952 验证，二者不可混作同一描述符。
额外对照排除了 padded5D 与 native2D、核间地址推进顺序作为主要解释。

保留原 packed NG2 API 式，只对结构条件命中的完整读写流水增加 `k*max(B*C-U,0)`。
源宽超过 32B 时，简单并发项会使候选过度降核，域外控制 `[1024,1536,12]` 因此保留原模型。
该控制的完整 tiling 恢复为原 55 核，避免首版扩展到 48B 连续源时降至 38 核造成退化。
这次收窄按 DMA block 的物理粒度划分，不按用例或设备总核数。
系数仅从域内 7 个训练点重拟合；新增独立 5 点最大误差 26.06%，未参与拟合。
系数冻结时不使用 holdout 或 known；拟合工具可从 samples.json 重建：

| 集合 | 点数 | 平均绝对相对误差 | 最大绝对相对误差 |
| --- | ---: | ---: | ---: |
| Train | 7 | 10.27% | 23.73% |
| Holdout（原5+新增独立5） | 10 | 13.04% | 26.06% |
| Known | 2 | 15.88% | 22.54% |

k=0.0038940645164791633，U=4456448 bytes。适用域、外推限制和回退见 design.md。
采样中最高活跃核数为 56，64/72 仅有公式 UT，不计真机验证。

## 默认选解

PyTorch 的纯 permute 返回视图，不产生该搬运 kernel。本次用 permute(2,0,1).contiguous() 实体化相同输入输出转置，
输入 FP32 `[2048,2048,8]`、输出 `[8,2048,2048]`。它不等同于整网 BMM→独立转置→softmax 的编译边界。
每个进程独立缓存；warmup20、Event5组×40次；profiler warmup2、active5。

| 版本 | PGO | kernel 中位耗时 μs | Event 中位 μs | 主导流水 | 精度 |
| --- | --- | ---: | ---: | --- | --- |
| #3065 原默认 | 关 | 217.780 | 219.722 | MTE2 99.2% | Pass |
| 最终修正后默认 | 关 | 200.597 | 202.868 | VEC 98.3% | Pass |
| 同一最终包快解 | 请求 Top10 | 200.580 | 202.304 | VEC 98.3% | Pass |

原默认 codegen SHA256 `9046392aebea224846602973bfbb20565252f9eaf6495d488ad9688dc5cf2d1e`；
最终包 `0aa255b9ebd8db824168b4339a3c070e2a182220bcb64643c39a164ba345adae`。
最终默认与同包 PGO 都是 result0/group0/case0，tile=[8,1585]、每核循环48次、block_dim=56，
q0/q1/index buffer 均为50720B。原默认为 case1，tile=[8,3960]、每核循环19次、q0=126720B。
本入口由 tiling data 中的 result/local key 条件直接分派，没有另一个独立的 TILING_KEY 宏编号。
完整 tiling data 与设备源码 SHA 相同，实际调用为 `graph_hint_0_b0_x0_y1_s0_g0_c0`。
索引 Store 类型单独修正不足以改变默认模板；没有以修改 MTE3 或搜索规则迫使翻转。

[精简验证证据](https://gitcode.com/zhang_shengjie/CANN-DevTools/blob/6733c737b04f2e6289ebeb5784adaccb7c1e6d30/att_generalization/report/20261119/pr3065-model-regression/direct-store-v1/verification.json)包含源文件和实际库摘要、最终 tiling、分派及 profiler 摘要。
[拟合前后图](fit-comparison.svg)区分 train、holdout 与 known。

## Host 与控制集

30组反序交错的新进程测原生静态 Inductor `AutofuseTiling` 入口；每进程首调一次、重复调用1000次。
首调中位84.0865→85.2830μs（+1.42%），重复调用均值的中位0.029524→0.029539μs。
复用同进程静态缓存时不重新求解；这不是说首次 tiling 没有成本。原始标量记录见 [host-timing.json](https://gitcode.com/zhang_shengjie/CANN-DevTools/blob/6733c737b04f2e6289ebeb5784adaccb7c1e6d30/att_generalization/report/20261119/pr3065-model-regression/direct-store-v1/host-timing.json)。
没有新增设备运行时逻辑，最终 kernel 源码与同包 PGO 完全一致。

| 控制 | Base kernel μs | 修正后控制实测 μs | 完整选解身份 |
| --- | ---: | ---: | --- |
| FP32 transpose+softmax | 1135.920 | 1139.869 | 相同 |
| FP16 实体化转置 | 115.880 | 109.538 | 相同 |
| FP32 连续源48B | 107.623 | 94.985 | 相同，恢复55核 |
| FP32 连续源128B | 354.561 | 352.631 | 相同 |

控制集先在收窄包实测，再以最终包重新生成 Host/设备源码；两包代码生成结果逐字相同，复用已有控制实测。
各控制与 Base 的完整 tiling、实际 kernel 源码一致。控制样本的计时变化不归因于此次修正；没有把波动当成收益。

`att_accuracy_level=0` 仅用于前期隔离 AutoTuning；主验收保持 accuracy_level=1。

## 验证状态

297 项原生 NddmaModelV2 / UTestAscirPerfV2 UT 全部通过，涵盖静态/符号、dtype、padding、源宽边界与真实图消费者门禁。
完整 tiling、实际 result/group/case 调用及 kernel 源码已核对；同包默认/PGO 命中同一个执行身份。
控制集覆盖 softmax 向量消费者、FP16、48B 与128B 连续源，最终完整 tiling 和 kernel 源码与 Base 一致。
17/48 已完成本次改动的 Host 增量影响核验，逐例选解均未变化，依授权未重复上板；见 [核验报告](https://gitcode.com/zhang_shengjie/CANN-DevTools/blob/6733c737b04f2e6289ebeb5784adaccb7c1e6d30/att_generalization/report/20261119/pr3065-model-regression/direct-store-v1/regression-17-48.md)。
64/72 核完整 Host 复算均选择 case0；453 本轮尚未核验，64/72 核真机和动态 shape 设备执行仍未验证。
