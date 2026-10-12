# IsInf 模型验证报告

> 本报告基于 [samples.csv](samples.csv) 的实测数据（5 条，float32→bool 主路径，用例集沿用
> IsNan 同参数梯度）。model_predict_cycles 按 [design.md](../../../design.md) 公式推导，
> 5 条全部与闭式公式一致。误差定义：`error = (predict - measured) / measured * 100%`，
> 正值表示高估。

## 环境与版本

见 [manifest.json](manifest.json)。

## 预测口径

- 全部为 float32 → bool 分支（codegen 输出恒为 bool，MaskPack 恒执行），
  `outer_loop = 100`，`rt = Ceil(cal_count / 64)`。
- 预测公式：`(48+7·rt)·cc`（head 20 + 标定 latency 28 + CompareScalarEQ 2rt +
  MaskPack×2 4rt + Store rt）。

## 误差分析

### 用例明细

| input→output | outer_loop | cal_count | rt | 预测 (cycles) | 实测 (cycles) | 误差 | 单块折算 (模型 7) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| float32→bool | 100 | 64 | 1 | 5500 | 5906.92 | -6.89% | 11.07 |
| float32→bool | 100 | 128 | 2 | 6200 | 6045.45 | +2.56% | 6.23 |
| float32→bool | 100 | 256 | 4 | 7600 | 7216.82 | +5.31% | 6.04 |
| float32→bool | 100 | 512 | 8 | 10400 | 10080.17 | +3.17% | 6.60 |
| float32→bool | 100 | 1024 | 16 | 16000 | 16004.03 | -0.03% | 7.00 |

整体：平均 +0.82%，MAE 3.59%，最大绝对误差 6.89%（rt=1）。

### 关键发现

1. **rt=16 完全吻合**（误差 -0.03%，单块折算 7.00 vs 模型 7），验证了标定常数 28 与
   7 cycles/块（CompareScalarEQ×2 + MaskPack×2 + Store）的计数。
2. **rt=1 低估 -6.9%**：单块折算 11.07 vs 模型 7，单块场景存在约 +4 cycles 未建模固定
   开销，与 IsNan（+3.6）/ IsFinite（+4.4）同参数点趋势一致，三个同族 API 的单块
   开销量级相同（+3~4 cycles）。
3. **rt=2/4/8 轻微高估**（+2.6%/+5.3%/+3.2%，单块折算 6.0~6.6 vs 7），误差随 rt
   非单调，与 IsNan/IsFinite 的中段波动类似。
4. 样本量小（单分支 5 条），上述规律需更多梯度用例确认。

## 结论

- 整体 MAE 3.59%（与 IsNan 3.06%、IsFinite 4.61% 同水平），标定常数
  与指令计数基本正确，可用于性能评估。
- 改进方向：单块（rt=1）场景补约 +4 cycles 固定开销（同族三个 API 一致）。
