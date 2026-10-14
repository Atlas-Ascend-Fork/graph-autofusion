# Cast 模型验证报告

> 本报告基于 [samples.csv](samples.csv) 的实测数据（17 条，覆盖 `CastPerf` 的 8 类 dtype 分发分支）。
> model_predict_cycles 为 `CastPerf` 实现（`autofuse/v35/att/api_perf_register/ascendc_regbase_perf.cpp`）
> 的模型输出。误差定义：`error = (predict - measured) / measured * 100%`，正值表示高估。

## 环境与版本

见 [manifest.json](manifest.json)。

## 用例与分支覆盖

| 分发分支 | dtype 对 | 用例数 |
| --- | --- | --- |
| 同位宽整数（`AddSameBitIntegerCastPerf`） | int8→uint8、uint64→int64 | 2 |
| 通用浮点（默认 `kCast` 单次） | fp16→fp32、fp32→fp16 | 2 |
| 两段转换（int32→fp32→fp16） | int32→float16 | 1 |
| B64（`AddB64CastPerf`） | int64→fp32、fp32→int64、int32→int64 | 3 |
| B64 transfer（`AddB64TransferCastPerf`） | fp16→int64、int64→fp16 | 2 |
| B8 specific（`AddB8SpecificCastPerf`） | uint8→fp32/int32/int16、int16→uint8/int8 | 5 |
| Int64→UInt8（`AddInt64ToUInt8CastPerf`） | int64→uint8 | 1 |
| UInt8→Int64（`AddUInt8ToInt64CastPerf`） | uint8→int64 | 1 |

## 误差分析

### 分支汇总

| 分发分支 | 用例数 | 平均误差 | 误差区间 |
| --- | --- | --- | --- |
| 同位宽整数 | 2 | -28.25% | [-37.32%, -19.19%] |
| 通用浮点 | 2 | +40.86% | [+34.84%, +46.88%] |
| 两段转换 | 1 | +95.49% | [+95.49%, +95.49%] |
| B64 | 3 | +69.88% | [+63.17%, +73.87%] |
| B64 transfer | 2 | +41.76% | [+41.71%, +41.81%] |
| B8 specific | 5 | +128.01% | [+97.94%, +147.92%] |
| Int64→UInt8 | 1 | +64.30% | [+64.30%, +64.30%] |
| UInt8→Int64 | 1 | +27.58% | [+27.58%, +27.58%] |
| 全部 | 17 | +67.40% | [-37.32%, +147.92%] |

整体 MAE 74.05%，最大绝对误差 147.92%。

### 用例明细

| input | output | outer_loop | cal_count | 预测 (cycles) | 实测 (cycles) | 误差 | 分发分支 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| int8 | uint8 | 7 | 57264 | 2518 | 3116 | -19.19% | 同位宽整数 |
| float16 | int64 | 32 | 12528 | 64423 | 45462 | +41.71% | B64 transfer |
| int64 | float16 | 32 | 12528 | 64423 | 45428 | +41.81% | B64 transfer |
| uint8 | float32 | 16 | 25056 | 32583 | 16461 | +97.94% | B8 specific |
| uint8 | int32 | 16 | 25056 | 38855 | 16554 | +134.72% | B8 specific |
| uint8 | int16 | 10 | 40080 | 16725 | 6746 | +147.92% | B8 specific |
| int16 | uint8 | 10 | 40080 | 10445 | 4927 | +112.00% | B8 specific |
| int16 | int8 | 10 | 40080 | 16725 | 6758 | +147.48% | B8 specific |
| int64 | float32 | 38 | 10552 | 64539 | 39554 | +63.17% | B64 |
| float32 | int64 | 38 | 10552 | 39345 | 22629 | +73.87% | B64 |
| int64 | uint8 | 29 | 13824 | 64227 | 39092 | +64.30% | Int64→UInt8 |
| uint8 | int64 | 29 | 13824 | 89283 | 69981 | +27.58% | UInt8→Int64 |
| int32 | int64 | 38 | 10552 | 39345 | 22797 | +72.59% | B64 |
| uint64 | int64 | 51 | 7864 | 14341 | 22879 | -37.32% | 同位宽整数 |
| int32 | float16 | 19 | 21096 | 26330 | 13469 | +95.49% | 两段转换 |
| float16 | float32 | 19 | 21096 | 13771 | 10213 | +34.84% | 通用浮点 |
| float32 | float16 | 19 | 21096 | 13771 | 9376 | +46.88% | 通用浮点 |

### 关键发现

1. **系统性高估**：17 条中 15 条高估，整体平均 +67.40%。
2. **同位宽整数分支唯一低估**（-19% ~ -37%）：该分支仅计 UpdateMask×2 + Load + Store，
   实测开销显著高于建模，实际执行路径可能包含额外指令或 DataCopy 非对齐开销。
3. **B8 specific 分支误差最大**（+98% ~ +148%）：`AddB8SpecificCastPerf` 中两次 `kCast`
   默认项（5, 2）叠加 UnPack / Pack 计数高估明显；同为 B8 的 uint8→fp32（+97.94%）好于
   int16→int8 / uint8→int16（约 +148%）。
4. **UInt8→Int64 相对最好**（+27.58%）：该分支 store 使用 dtype-mapping 特例（uint8→int64
   吞吐 5），标定相对接近实测。
5. **B64 / B64 transfer / 两段转换**高估 +42% ~ +96%，介于上述两者之间。

## 结论

- 模型趋势方向基本一致（除同位宽分支外均为高估），但数值偏差大（MAE 74.05%）。
