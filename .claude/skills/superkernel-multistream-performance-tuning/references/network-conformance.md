# 多网络 Conformance

通用能力不能用“在一个模型上运行过”验收。`multistream_network_conformance.py` 把网络差异
限制在 adapter 和归一化 evidence receipt 内，对同一组 generic core 文件做多网络复核。

## Adapter 契约

`superkernel-multistream-conformance-adapter-v1` 声明：

- 稳定 adapter ID 与网络类别；
- 已实现的 conformance capability；
- 每份 evidence 的相对路径、schema 和 SHA256；
- evidence 推导出的预期决策。

当前 conformance evidence 分为：

- `real_npu_closure`：必须是完整真实 NPU 闭环 receipt，决策为 `accepted` 或 `no_gain`；
- `resource_screening`：必须使用三字段资源分类，且 `stable_parent_match_count=0` 才能返回
  `no_reorder_candidate`。

`no_reorder_candidate` 表示该网络当前没有合法业务算子重排，不表示 adapter 或通用核心失败。

## Suite 验收

plan 至少包含两个不同 `network_class`，且至少一个 adapter 具备真实 NPU closure。plan 同时
密封 capture producer、core-family、identity、dependency、operator-order 和 closure 等通用
文件。run 阶段会：

1. 重放每个 adapter 与 evidence fingerprint；
2. 验证 evidence 决策语义；
3. 计算唯一 generic-core fingerprint；
4. 扫描 core 文件不得出现任何 adapter ID；
5. 生成确定性 report fingerprint。

真实两网络报告位于 `evals/network-conformance/report.json`。模型专用 ID 只出现在
该 eval 的 adapter/evidence 中，不进入 analyzer、planner、runner 或 contract。
