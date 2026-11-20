# FP32 稠密窄行普通 Load 校准

业务候选、完整 tiling、Host 计时和 17/48 例端到端回归记录由 [CANN-DevTools](https://gitcode.com/zhang_shengjie/CANN-DevTools/blob/6733c737b04f2e6289ebeb5784adaccb7c1e6d30/att_generalization/report/20261119/pr3065-model-regression/README.md) 按固定版本跟踪。本目录保留 API 建模依据、独立训练/留出数据及验证摘要。

## 结论

沿用 `LoadStoreFunc`：`cycles = h + bytes / (a + b/active_cores)`。
新增适用域：FP32，删除静态单位轴后恰为二维，512–2048 行，行宽 8–32 元素且按 32B 对齐，GM/UB 零行间隙，16–64KiB，16–56 个活跃核。
硬件 LoopMode、单行、非连续布局、其他 dtype 和域外候选继续走既有模型。范围是采集覆盖域，不是设备总核数特判。
系数为 h=142.17870407217006、a=1.1266553015411542、b=1126.6553015411544。

| 数据角色 | 点数 | 最大相对误差 |
| --- | ---: | ---: |
| 训练 | 7 | 9.20% |
| 独立留出 | 5 | 11.24% |
| 已知目标候选，未参与拟合 | 1 | 12.39% |
| 系数冻结后新增留出 | 3 | 8.12% |

系数仅用 train 行求解；留出、已知点及后续冻结留出均不参与参数选择。未修改其他 dtype 的系数。
[拟合图](fit.svg)、[拟合结果](fit.json)、[用例](cases.csv)、[预测与实测](samples.csv)可离线核对。

## 为什么修正普通 Load

同一连续源的 `[2048,8]` FP32、56 活跃核，用同一采集器和同步协议得到每核每 API 的 MTE2 斜率：

| API | cycle |
| --- | ---: |
| 普通二维 Pad Load，禁用 L2 | 3683.19 |
| 硬件 LoopMode，禁用 L2 | 18839.87 |
| NDDMA，禁用 L2 | 4515.75 |
| NDDMA，NORMAL hint | 4812.14 |

旧普通 Load 估值约 5100 cycle，NDDMA 估值约 4918 cycle。普通 Load 的高估会把实测更快的候选排在后面。
本次校准其自身成本，不抬高 NDDMA 来补偿。LoopMode 是一次生产 API 加硬件循环，不是 2048 次 C++ API 调用；其域外误差没有被本分支覆盖。
完整控制点及各 K 中位值见 [api-control-samples.csv](api-control-samples.csv)。

## 采集和复现

使用仓内 `autofuse/tools/modeling/v35/dma`，同一 `att_dma_load_loop` 二进制批量采集所有描述符。
生产 API、队列、tail-store 消费、保护区及同步检查保持原有实现。4GiB 输入池；核、调用及 launch 间推进到不同 GM offset；
主拟合用 `C` 模式的禁用 L2 hint，NDDMA 的 NORMAL 对照单独记录。无额外 point flush。
K=1/2/4/10，每 K 16 次预热、16 次有效采样；使用 `aiv_total_cycles × aiv_mte2_ratio / Block Num` 的中位值对 K 做线性斜率。
全部描述符通过精度和保护区检查。二进制、包及环境摘要见 [versions.json](versions.json)。

从仓库根执行，`work` 和 `lock` 由调用者设置为私有输出目录与统一设备锁；构建方法见 [DMA README](../../../../../../../tools/modeling/v35/dma/README.md)。

```bash
archive=autofuse/att/modeling/v35/mte2/load/results/dense-bulk-v1
tool=autofuse/tools/modeling/v35/dma
# build/att_dma_load_loop 编译一次，后续批次复用。
ATT_HOST_THREADS=2 ATT_FIXED_POOL_GIB=4 bash "$tool/collect.sh" \
  loop "$work/build" "$archive/api-controls.csv" C "$lock" "$work/train-controls"
python3 "$tool/summarize_nddma.py" "$archive/api-controls.csv" \
  "$work/train-controls" "$work/train-controls.json" --samples-per-k 16
# extra-train.csv 与 frozen-holdout.csv 分别放到新目录采集，保持角色隔离。
python3 autofuse/tools/modeling/v35/mte2/calibrate.py "$archive/cases.csv" "$archive/samples.csv" \
  --fit LoadAlignedDenseBulkPayload --policy C_fresh_disable
```

`api-controls.csv` 保留三种 API 的原始描述符与采样配置；`extra-train.csv` 只补训练点；`frozen-holdout.csv` 在系数冻结后采集。
原始小型 op_summary、launch JSONL、精度回执及源码快照由调用者放在固定实验目录，归档清单与摘要用于关联，完整 profiler 目录和二进制不入库。

## 默认选解与 PGO 独立复测

本轮包加载 `libaihac_codegen.so` 摘要为 `41d11e66bc198619da8e2a92e7eb53b5ca26590884e26a458015b58d60200550`。
默认关闭 native PGO 且外层 `TORCHINDUCTOR_ASCENDC_AUTOTUNE_TOPN=1`；避免将外层候选实测误报为 ATT 默认。
PGO 对照独立重放此前 Top10 搜索的冻结快解，核对完整 tiling、设备分派、源码和依赖；本轮没有重新搜索 Top10。
所有结果为当前 56 AIV 真机、kernel_details 的 5 个有效 step 中位数，精度 `rtol=1e-4, atol=1e-5` 通过。

| 片段 | 修改前默认 μs | 本轮默认 μs | 本轮 PGO μs | 默认/PGO | 默认实际设备 case |
| --- | ---: | ---: | ---: | ---: | --- |
| FP32 `[2048,2048,8]` 实体化 permute(2,0,1) | 200.597 | 200.435 | 200.280 | 1.0008 | result0/g0/case0 |
| 相同输入 permute+amax/sub/exp/sum/div | 1133.721 | 444.757 | 439.568 | 1.0118 | result0，g0/g1/g2 为 case1/1/4 |

第一轮本轮新默认分别为 200.459 / 443.565μs，独立复测结果一致。
链的局部 tiling_key 是 1/1/2，实际设备函数 case 是 1/1/4；不能将 g2 的局部 key2 当作设备 case2。
新默认各 group 56 核；PGO 使用 54/54/55 核。两者模板组合和设备源码相同，完整 tiling 并不相同。
独立转置的默认与 PGO 完整 tiling、源码和二进制相同。

链另有 Reduce RA 符号内轴分支修正，见 [Reduce 设计](../../../../vector/reduce/design.md)。两项按先后独立验证：
仅 RA 修正先选中 g0 NDDMA、约 500μs；再校准普通 Load 才选中 PGO 的 g0/g1 快模板。未修改设备算法、调度模板或 NDDMA 参数。
按新公式复算完整候选，旧默认为 964056 cycle，新默认为 754998 cycle，PGO 冻结配置为 766161 cycle；
旧公式却将旧默认 1365455 cycle 排在 PGO 1404261 cycle 前。详见 [candidate-costs.json](https://gitcode.com/zhang_shengjie/CANN-DevTools/blob/6733c737b04f2e6289ebeb5784adaccb7c1e6d30/att_generalization/report/20261119/pr3065-model-regression/dense-bulk-v1/candidate-costs.json)。
[逐片段结果](https://gitcode.com/zhang_shengjie/CANN-DevTools/blob/6733c737b04f2e6289ebeb5784adaccb7c1e6d30/att_generalization/report/20261119/pr3065-model-regression/dense-bulk-v1/target-results.json)含完整 tiling、局部 key、实际设备函数、关键 profile 行及指纹。

## 影响筛选和工程性能

[regression-identities.json](https://gitcode.com/zhang_shengjie/CANN-DevTools/blob/6733c737b04f2e6289ebeb5784adaccb7c1e6d30/att_generalization/report/20261119/pr3065-model-regression/dense-bulk-v1/regression-identities.json)记录 17 例的 40 个输入和 48 例 transpose 去重后的 36 个输入。
与修改前私有包 e273 比较，完整 tiling、workspace、block dim 和设备源码全部一致；调用入口、配置及其他库身份沿用冻结输入。
设备性能结果按授权复用，0 个变化输入需要上板；这不是 65 例新跑性能通过的声明。本轮未追加 453 例。

64/72 核仅做 native Host 资源复算；独立转置仍选 case0，链仍选 case1/1/4。
64 核链各 group 为 56/64/64，72 核为 56/71/72；没有把总核数写进产品分支。

CPU 绑核、每配置独立进程的一次首次 tiling + 100 次同 shape 重复调用，单位 μs：

| 片段 / 可用 AIV | 首次，修改前→本轮 | 同进程重复调用中位数，修改前→本轮 |
| --- | ---: | ---: |
| 链 / 56 | 702.885→718.929 | 0.691→0.672 |
| 链 / 64 | 718.718→725.509 | 0.681→0.691 |
| 链 / 72 | 729.815→760.241 | 0.681→0.681 |
| 转置 / 56 | 150.315→149.224 | 0.661→0.631 |
| 转置 / 64 | 160.271→167.561 | 0.651→0.631 |
| 转置 / 72 | 164.667→162.664 | 0.641→0.661 |

这是少量 Host 样本，不作为精确统计上限。[host-checks.json](https://gitcode.com/zhang_shengjie/CANN-DevTools/blob/6733c737b04f2e6289ebeb5784adaccb7c1e6d30/att_generalization/report/20261119/pr3065-model-regression/dense-bulk-v1/host-checks.json)保存完整输入和结果。
模型每符号 RA 增加 3 个条件表达式，稠密二维 Load 增加 1 个守卫；无候选遍历、额外 tiling 求解、设备指令或日志。
17 例 31/40 个 Host 源文件随估值公式改变，设备源码全部不变。独立冷编译总耗时受编译缓存和进程启动影响，未据此宣称编译加速。
本轮相关 Reduce、Load、Transpose、NDDMA UT 共 195 项通过；源代码与适用域边界测试保持静态/符号路径一致。

### 端到端输入与运行口径

输入均为 seed=0 的 FP32 `[2048,2048,8]`，在当前框架中用 `torch.compile(fullgraph=True)` 编译以下纯函数。导入 `torch_npu` 与 Autofuse 后端后再 compile；仅 permute 的 view 不属于实体化转置。

```python
def materialized(x):
    return x.permute(2, 0, 1).contiguous()

def chain(x):
    t = x.permute(2, 0, 1)
    e = (t - t.amax(-1, keepdim=True)).exp()
    return e / e.sum(-1, keepdim=True)
```

新默认使用独立编译缓存，设置 `AUTOFUSE_FLAGS=--autofuse_enable_pgo=false`、`TORCHINDUCTOR_ASCENDC_AUTOTUNE_TOPN=1`、`AUTOFUSE_DFX_FLAGS=--att_accuracy_level=1`。PGO 搜索使用 native PGO=true 和外层 TopN=10，找到快解后保留完整配置，在独立进程重放。
预热 20 次；Event 计时 5 组、每组 40 次。精度单独比较 eager。最终 kernel 比较来自 Level2 PipeUtilization profiler，wait=1、warmup=2、active=5，每轮调用后 synchronize 与 profiler.step。仅保存关键 CSV、完整 tiling、设备源码摘要及版本。

Host 编译实测：链 Host 编译中位数 3.0073→3.0311s（+0.79%）；独立转置 Host 编译中位数 4.2158→4.2247s（+0.21%）。使用生产 host_compile、相同工具链/SDK、每次独立输出、CCACHE_DISABLE=1、单 worker，3 轮 A/B 反序配对；该指标只包含 Host 编译，不含前端/设备编译。少量样本不作为最坏耗时保证。见 [host-compile-timing.json](https://gitcode.com/zhang_shengjie/CANN-DevTools/blob/6733c737b04f2e6289ebeb5784adaccb7c1e6d30/att_generalization/report/20261119/pr3065-model-regression/dense-bulk-v1/host-compile-timing.json)。

链的 Host 生成源码为 3742331→4740103B（+26.66%），Host tiling 动态库为 2087408→2099168B（+0.56%）；独立转置分别为 350748→351778B、525816→526360B。符号 RA 同时保留三个成本分支会增大 Host 表达式源码，实际编译中位数增幅为 0.79%，设备源码保持相同。
