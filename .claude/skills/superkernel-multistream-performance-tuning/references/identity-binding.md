# 算子身份绑定协议

`superkernel-multistream-identity-registry-v1` 是模型 adapter 与通用 order analyzer 之间的
身份边界。它把源码 statement、SK-off 执行项和 SK-on 下发项绑定到相同
`operator_id + statement_id`，但不通过算子名称或行序推断身份。

## 输入

adapter 生成 `superkernel-multistream-identity-observations-v1`。每个 observation 包含：

- `domain`: `source`、`sk_off` 或 `sk_on`；
- 非空结构化 `identity`，其 canonical JSON fingerprint 作为 raw identity key；
- 标准 `operator_id` 和 `statement_id`；
- SK 侧的 `alignment_id`，source 侧固定为 `null`；
- 受支持的精确 `binding_method`；
- 已绑定 SHA256 的 `evidence_path` 和文件内 `evidence_locator`。

允许的方法只有：

| domain | binding method |
|---|---|
| source | `exact_source_span`、`graph_debug_handle` |
| sk_off | `compiler_origin_uid`、`projected_trace_exact` |
| sk_on | `compiler_origin_uid`、`sk_meta_origin_exact` |

`operator_name`、CSV/log ordinal、Task ID 和绝对时间偏移不能作为 binding method。

## 构建与校验

```bash
python3 scripts/multistream_identity_binding.py build \
  --observations identity-observations.json --artifact-root EXPERIMENT \
  --out identity-registry.json

python3 scripts/multistream_identity_binding.py validate \
  --registry identity-registry.json --artifact-root EXPERIMENT --require-complete
```

registry 同时提供按 raw identity 查询的 `forward_index` 和按
`operator_id + statement_id` 汇总的 entries。以下情况产生 blocker：

- 同一 SK raw identity 在同一 occurrence 映射到多个 operator/statement；
- 一个 operator 映射到多个 statement；
- 缺少 source、SK-off 或 SK-on domain；
- 任一侧少于三个 occurrence；
- SK-off 与 SK-on 的 alignment 集合不同。

capture producer 插件只能使用通过 `--require-complete` 校验的 registry 生成用于重排授权的
operator-order capture。registry 不完整时保留 blocker 并停止该候选，不允许补猜映射。
