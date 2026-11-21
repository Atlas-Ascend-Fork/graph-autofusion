# IsFinite 模型验证报告

> 本报告基于 [samples.csv](samples.csv) 的实测数据（5 条，float32→bool 主路径，用例集沿用
> IsNan 同参数梯度）。model_predict_cycles 按 [design.md](../../../design.md) 公式推导，
> 5 条全部与闭式公式一致。误差定义：`error = (predict - measured) / measured * 100%`，
> 正值表示高估。

## 环境与版本

见 [manifest.json](manifest.json)。

## 预测口径

- 全部为 float32 → bool 分支（codegen 主路径，输出 `ReinterpretCast<bool>`），
  `outer_loop = 100`，`rt = Ceil(cal_count / 64)`。
- 预测公式：`(36+8·rt)·cc`（head 20 + 标定 latency 16 + Ne/Store 2rt + MaskOr 2rt +
  MaskPack×2 4rt）。

## 误差分析

### 用例明细

| input→output | outer_loop | cal_count | rt | 预测 (cycles) | 实测 (cycles) | 误差 | 单块折算 (模型 8) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| float32→bool | 100 | 64 | 1 | 4400 | 4841.14 | -9.11% | 12.41 |
| float32→bool | 100 | 128 | 2 | 5200 | 5238.04 | -0.73% | 8.19 |
| float32→bool | 100 | 256 | 4 | 6800 | 6170.92 | +10.19% | 6.43 |
| float32→bool | 100 | 512 | 8 | 10000 | 9764.63 | +2.41% | 7.71 |
| float32→bool | 100 | 1024 | 16 | 16400 | 16301.96 | +0.60% | 7.94 |

整体：平均 +0.67%，MAE 4.61%，最大绝对误差 10.19%（rt=4）。

### 关键发现

1. **rt=2/8/16 拟合良好**（误差 -0.7%/+2.4%/+0.6%，单块折算 7.7~8.2 vs 模型 8），验证了
   标定常数 16 与 8 cycles/块（Ne + Store + MaskOr×2 + MaskPack×2）的计数。
2. **rt=1 低估 -9.1%**：单块折算 12.41 vs 模型 8，单块场景存在约 +4.4 cycles 未建模固定
   开销，与 IsNan 同参数点（+3.6 cycles）、Compare/Where 小规模场景的「行进入开销」趋势
   一致。
3. **rt=4 高估 +10.2%**：单块折算 6.43，明显低于相邻 rt 档位（rt=2/8 为 8.19/7.71），
   误差随 rt 非单调，与 IsNan 的 rt=2/4 波动类似，疑似测量波动，建议复测。
4. 样本量小（单分支 5 条），上述规律需更多梯度用例确认。

## 结论

- 整体 MAE 4.61%（略差于 IsNan 3.06%，优于 Compare 12.6%），标定常数
  与指令计数基本正确，可用于性能评估。
- 改进方向：单块（rt=1）场景补约 +4 cycles 固定开销；rt=4 异常点复测确认。
