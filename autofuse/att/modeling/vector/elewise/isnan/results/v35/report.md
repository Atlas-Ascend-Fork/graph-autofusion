# IsNan 模型验证报告

> 本报告基于 [samples.csv](samples.csv) 的实测数据（5 条，float32→bool 主路径）。
> model_predict_cycles 按 [design.md](../../../design.md) 公式推导，5 条全部与闭式公式一致。
> 误差定义：`error = (predict - measured) / measured * 100%`，正值表示高估。

## 环境与版本

见 [manifest.json](manifest.json)。

## 预测口径

- 全部为 float32 → bool 分支（codegen 主路径，输出 `ReinterpretCast<bool>`），
  `outer_loop = 100`，`rt = Ceil(cal_count / 64)`。
- 预测公式：`(46+6·rt)·cc`（head 20 + 标定 latency 26 + Ne/Store/MaskPack×2 吞吐 6rt）。

## 误差分析

### 用例明细

| input→output | outer_loop | cal_count | rt | 预测 (cycles) | 实测 (cycles) | 误差 | 单块折算 (模型 6) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| float32→bool | 100 | 64 | 1 | 5200 | 5561.70 | -6.50% | 9.62 |
| float32→bool | 100 | 128 | 2 | 5800 | 5615.12 | +3.29% | 5.08 |
| float32→bool | 100 | 256 | 4 | 7000 | 6819.56 | +2.65% | 5.55 |
| float32→bool | 100 | 512 | 8 | 9400 | 9399.63 | +0.00% | 6.00 |
| float32→bool | 100 | 1024 | 16 | 14200 | 14620.55 | -2.88% | 6.26 |

整体：平均 -0.69%，MAE 3.06%，最大绝对误差 6.50%（rt=1）。

### 关键发现

1. **rt=8 完全吻合**（误差 +0.00%，单块折算 6.00 vs 模型 6），验证了标定常数 26 与
   6 cycles/块（Ne + Store + MaskPack×2）的计数。
2. **rt=1 低估 -6.5%**：单块折算 9.62 vs 模型 6，单块场景存在约 +3.6 cycles 未建模的
   固定开销（与 Compare/Where 小规模场景一致的行进入开销趋势）。
3. **rt=2/4 轻微高估**（+3.3%/+2.7%，单块 5.08~5.55），rt=16 轻微低估（-2.9%，单块 6.26），
   误差随 rt 非单调，量级均在 ±7% 内，疑似含测量波动。
4. 样本量小（单分支 5 条），上述规律需更多梯度用例确认。

## 结论

- 整体 MAE 3.06%（优于 Compare 12.6%、Where 4.1%），标定常数与指令
  计数基本正确，可用于性能评估。
- 改进方向：单块（rt=1）场景补约 +3~4 cycles 固定开销。
