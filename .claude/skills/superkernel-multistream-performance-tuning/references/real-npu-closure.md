# 真实 NPU 闭环契约

`multistream_npu_closure.py` 是跨阶段验收层，不替代已有 analyzer、runner 或 result contract。
它解决单个 artifact 各自合法、但整条实验链没有证明同一 request/trial 的问题。

## 必需链路

闭环固定绑定以下对象及其 SHA256：

1. 可通过 request contract 的 immutable incumbent 和目标 range；
2. 至少三个 occurrence、完整依赖证据、稳定 Cube/Vector overlap 和 dispatch inversion；
3. 由 analyzer route 2/3 物化的唯一源码 action；
4. 重排后至少三个 occurrence 的 dispatch-order pass evidence；
5. incumbent/candidate x SK-off/SK-on 四组 profile；
6. 三个独立 clean 进程及 semantic evidence；
7. result-v2 的 `accepted` 或 `no_gain` 终态。

`no_gain` 是闭环成功、性能候选失败：它必须含实际执行的 rejected trial，并保持 incumbent。
无候选仍是单网络的合法分析结论，但不能替代“修正逻辑已跑通”的真实闭环验收。

## 执行序语义

SK-off `kernel_details.csv` 的原始行序是算子执行序。`Start Time(us)` 和 `Duration(us)` 用于
验证跨流 overlap，不用于重新排列执行序。Cube/Vector 起点相差亚微秒时可能交替或相同，若
按起点排序会把稳定执行序误判成不稳定。

SK-on `sk_fused_nodes.log` 是父 SK 内下发序。候选必须同时证明：完整 child set 未变化、仍为
至少双流、下发序等于 materialized statement order。

## 真实验收样本

`evals/real-npu-reference/receipt.json` 记录一次 Ascend910_9392、npugraph_ex、静态核、
双流父 SK 的闭环。route 2 确实把父 SK 下发序从 Cube->Vector 改为 Vector->Cube；三对独立
clean3 未达到 2% 门槛，最终为 `no_gain`，incumbent 保持不变。
