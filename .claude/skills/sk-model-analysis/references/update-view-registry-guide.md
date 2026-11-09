# 历史日志适配参考

本文描述旧信息库，用于维护 Log Parser 的输入适配；新 Graph View 使用 `graph_report` 标准模型。旧 scope/task 页面及“log 主、JSON shadow”描述不再是公共运行合同。当前合同见 [artifact-contract.md](artifact-contract.md)。

# Update View 信息库构建指导

## 当前结论
这份指导已经按当前 `sk-model-analysis` 实现校正。

当前 `update view` 不是“3 个平铺 registry”的实现，而是一个**分层组合模型**：

- `scope_library.scopes[]`
  - scope 的稳定主骨架
- `scope_library.scopes[].update`
  - scope 内的 update 视图载荷
- `graph_library.node_update_registry.rows`
  - graph-backed node 的标准化 update 行
- `scope_library.device_task_library.sections`
  - task queue / synthesized custom 的下钻与绑定层

另外要注意两点：

- 上述对象都是**按单个 model 或 model-instance**导出的；跨 model 实例联动时，不能只靠 `node_id`
- `scope-graph.html` 是当前 update-aware 主视图，`task-queue-graph.html` 仍是下钻页，不反过来主导 update 叙事

## 1. 为什么旧版理解需要更新
旧版文档把目标信息库抽象成：

- `scope_node_registry`
- `scope_update_registry`
- `node_update_registry`

这个理解在“原始日志解析层”还勉强成立，但已经不等于当前稳定导出契约。

现在真正落地的实现，多了几层旧文档没有覆盖的事实：

- scope 级 update 边界已经被折叠进 `scope_library.scopes[].update`
- graph-backed update 和 synthesized custom node 已经分层处理
- device task section 已经先走 `bound_scope_id` 绑定，再在必要时 fallback 到 `scope_name`
- `scope-graph` 渲染时，不只吃结构化 row，也会用 `node_details` 里的显式 `addrValue` 日志做地址覆盖

如果继续按“3 个平铺 registry”设计，后面会漏掉：

- `scope_id`
- `graph_node_key`
- `custom_instance_key`
- `bound_scope_id`
- `synthesized_custom_nodes`
- `node_details -> explicit addr overlay`

## 2. 当前信息层次

### 2.1 原始解析层
`parse_update_execution()` 当前会先从 `super_kernel.log` 解析出这些中间对象：

- `scope_updates`
- `task_updates`
- `node_update_results`
- `event_addr_updates`
- `event_memory_resources`
- `graph_update`

这层更接近“原始日志抽取结果”，不是稳定对外 JSON 契约。

后续如果要加新 consumer：

- 优先从这层补采事实
- 但不要直接把这层原样暴露成新的最终页面契约，除非确实有跨页面共享需求

### 2.2 稳定导出层
当前真正稳定的对外对象主要是下面四类。

#### A. `scope_library.scopes[]`
这是 update view 的主骨架。

当前至少包含：

- `scope_id`
- `scope_export_ordinal`
- `scope_names`
- `node_ids`
- `streams`
- `details`
- `update`

职责是：

- 提供 scope 级组织关系
- 给 scope graph 提供稳定入口
- 承载 scope 局部的 update 结果

#### B. `scope_library.scopes[].update`
这是当前最接近“scope_update_registry + scope 局部 node update 结果”的对象。

当前字段重点是：

- `begin_detail`
- `finish_detail`
- `stream_count`
- `update_total_nodes`
- `streams`
- `graph_backed_updates`
- `synthesized_custom_nodes`
- `diagnostics`
- `node_details`

这里要特别说明：

- `streams` 就是 scope 级 update 边界与 stream 轮廓
- `graph_backed_updates` 是当前 scope 内、已按 `node_id` 过滤过的 graph-backed update 行
- `synthesized_custom_nodes` 是 task/update 侧存在、但不一定存在于 graph membership 的对象
- `node_details` 不是纯历史垃圾
  - 当前 `scope-graph` 仍会从这里抽取 `Updated notify|wait|reset node addrValue...` 这类显式地址事实
  - 所以在地址事实完全结构化之前，不能简单删掉或忽略

#### C. `graph_library.node_update_registry.rows`
这是 graph-backed node update 的标准化行集合。

当前 row 的有效字段是：

- `node_id`
- `type`
- `op_info_ptr`
- `op_info_size`
- `func_handle`
- `args`
- `args_size`
- `num_blocks`
- `addr`
- `value`
- `flag`
- `detail`
- `line`

当前要注意：

- 字段名是 `type`，不是旧文档里的 `update_target_type`
- 当前 row 里没有稳定的 `event_id` 字段
- `event/value-wait` 关系需要结合 node library、task/device 细节、以及 `node_details` 里的显式地址事实一起看

#### D. `scope_library.device_task_library.sections`
这是 update view 的下钻绑定层，不是主骨架，但现在已经是正式信息源。

当前关键字段有：

- `bound_scope_id`
- `bound_scope_binding_source`
- `scope_name`
- `queues`
- `task_identity_diagnostics`

这层承担的是：

- 把 device task section 尽量绑定回某个 scope
- 区分 graph-backed task 和 synthesized custom task
- 为 `task-queue-graph` 和 synthesized custom node 渲染提供直接来源

## 3. 当前真正应该使用的连接键

### 3.1 real scope
真实 scope 的首选主键是：

- `scope_id`

不是：

- `scopeName`

`scope_name` 现在主要用于：

- profiling / 文本上下文对齐
- device section fallback 绑定
- 人类可读展示

不能把它当成唯一、权威、无歧义的主键。

### 3.2 graph-backed node
graph-backed node 的首选连接键是：

- `node_id`

更稳妥的身份描述是：

- `graph_node_key`
  - 本质上是 `node_id + stream_id/stream_idx_in_graph/node_idx_in_stream` 的组合

这在以下场景更重要：

- 多实例分区
- 同一 `node_id` 行需要更强上下文时
- task identity / tracing / graph membership 联动

### 3.3 device section -> scope
当前绑定顺序是：

1. `bound_scope_id`
2. `scope_name` fallback

因此后续 agent 不要直接从 `scope_name` 反推“这一定就是这个 scope”。

### 3.4 synthesized custom node
这类对象不能按 graph node 的方式处理。

当前更合适的身份键是：

- `custom_instance_key`
- `scope_export_ordinal`

它们的特点是：

- 可能没有可信的 graph `node_id`
- 本质上是 task/update 侧合成出来的对象
- 渲染时需要与 queue、scope 共同考虑，而不是只做 `node_id join`

### 3.5 跨 model / model-instance
`node_id`、`scope_id` 都默认只在**单个 model-instance report 内**使用。

跨实例联动时，至少要把这些维度一起带上：

- `model_ri`
- `model_instance_id`

否则不同实例之间的 `node_id` / `scope_id` 可能被误拼。

## 4. 当前推荐的数据流

### 1. 选 scope
先从 `scope_library.scopes[]` 按 `scope_id` 选中 scope。

### 2. 建 scope graph 的 node 主体
按 `scope.node_ids` 去 `graph_library.node_library` 里找 graph-backed node。

### 3. 合并 graph-backed update
优先使用两层结果：

- 全局标准行：`graph_library.node_update_registry.rows`
- scope 局部过滤行：`scope.update.graph_backed_updates`

当前 `scope-graph` 的实际做法是：

- 先按 `node_id` 建全局 `node_update_by_id`
- 再叠 scope 局部 `graph_backed_updates`

所以后续 agent 若只做 scope 内渲染，通常直接看 `scope.update.graph_backed_updates` 更省事。

### 4. 叠加显式地址事实
对于 `VALUE_WRITE / VALUE_WAIT`：

- 不能只信 `node_update_registry.rows[].addr`
- 当前还会再从 `scope.update.node_details` 中抽取
  - `Updated notify|wait|reset node addrValue: ...`

这一步当前仍然是必要的，因为 `scope-graph` 会用它生成 `effective_addr`。

### 5. 渲染 synthesized custom node
从 `scope.update.synthesized_custom_nodes` 取 task/update 侧合成节点，单独建模：

- 它们不是 graph membership 节点
- 不能强行塞进 `graph_library.node_library`
- 应该作为 update/task 侧对象展示，并保留到 task queue 的跳转

### 6. 再进入 task view
当需要看：

- kernel 对应的 task queue
- synthesized custom 的 queue 落点
- device task identity 绑定

再进入：

- `scope_library.device_task_library.sections`
- `task-queue-graph.html`

不要反过来从 task queue 推导整张 update 主图。

## 5. 当前最重要的语义边界

### 5.1 scope 级 update 边界不再是独立 top-level registry
旧版文档的 `scope_update_registry` 思路，对应到当前实现，更接近：

- `scope_library.scopes[].update`
- 以及其中的 `streams`

如果只是做当前页面或局部分析，不需要重新造一个 top-level `scope_update_registry`。

### 5.2 node update row 只表达 graph-backed 事实
`graph_library.node_update_registry.rows` 当前只表达：

- graph-backed node 的 update 结果

它不自动覆盖：

- synthesized custom node
- device section 绑定
- task queue 自定义 event 的身份归属

### 5.3 task queue 不是 update view 的主入口
task queue 当前用于：

- scope 内 kernel/task 的下钻
- synthesized custom 的来源解释
- queue 侧 identity 与 args 事实查看

但不是 update view 的骨架。

## 6. 后续扩展建议

### 6.1 如果只是要增强当前 update view
优先改：

- `parse_update_execution()`
- `build_scope_library_export()`
- `build_node_update_registry()`
- `sk_scope_visualizer.py`

### 6.2 如果只是 scope 局部消费
优先往：

- `scope_library.scopes[].update`

里补，而不是先新增平铺 registry。

### 6.3 只有在下面场景才考虑新增 top-level registry
- 多页面共享同一批 update 事实
- 该事实不天然附属于单个 scope
- 当前 `scope.update` / `node_update_registry.rows` 明显不够表达

## 7. 不建议做的事

### 1. 不要再把 `scopeName` 当唯一主键
当前它只能作为 fallback 或展示名。

### 2. 不要假设所有 task 都能回映到 graph node
`synthesized_custom_nodes` 明确说明这件事并不成立。

### 3. 不要在地址事实完全结构化前丢掉 `node_details`
当前显式 `addrValue` 覆盖仍依赖它。

### 4. 不要为了“看起来整齐”重新造一套平铺 `scope_update_registry`
除非已经有明确 consumer 需要跨 scope 共享这层数据。

## 8. 给后续 agent 的直接执行建议

### 第一步
把 `scope_library.scopes[]` 当成 update view 的主骨架，而不是先扁平化成 3 张表。

### 第二步
对 graph-backed node，按 `node_id` 合并：

- `graph_library.node_library`
- `graph_library.node_update_registry.rows`
- `scope.update.graph_backed_updates`

### 第三步
对 scope 级 update 轮廓，直接消费：

- `scope.update.streams`
- `scope.update.diagnostics`

### 第四步
对 synthesized custom，单独走：

- `scope.update.synthesized_custom_nodes`
- `scope_library.device_task_library.sections`

### 第五步
对 `VALUE_WRITE / VALUE_WAIT` 地址事实，记得再叠：

- `scope.update.node_details`

中的显式 `addrValue` 行。

## 当前边界
这份指导当前只覆盖：

- update view 的实际导出模型
- 当前稳定连接键
- scope / graph / task 三层之间的职责边界

不覆盖：

- 前端视觉与交互布局
- profiling 页面组织
- task queue 深层图形布局
