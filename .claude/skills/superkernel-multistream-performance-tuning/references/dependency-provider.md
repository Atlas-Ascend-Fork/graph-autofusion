# 依赖证据 Provider 协议

业务算子重排不得把“没有观察到依赖”解释为“无依赖”。网络接入层必须通过
`superkernel-multistream-dependency-provider-v1` 提交图或源码证据，再由通用 merger 生成
`superkernel-multistream-dependency-evidence-v1`。

## Provider fragment

每个 provider 声明：

- 稳定 `provider_id` 和 `provider_kind`；
- provider 实现文件及 SHA256；
- 所有输入图、源码审查或 runtime metadata 文件及 SHA256；
- 明确覆盖的 dependency kinds；
- hard edges 及其文件内 evidence locator；
- provider 自身发现的 blocker。

支持 `fx_graph`、`exported_graph`、`source_review` 和 `runtime_metadata`。所有 provider 的覆盖
并集必须包含：DATA、STREAM_ORDER、EVENT、WAIT、BARRIER、COMMUNICATION、
CACHE_MUTATION、SIDE_EFFECT、CONTROL_FLOW。

## 合并与使用

```bash
python3 scripts/multistream_dependency_evidence.py build \
  --fragments dependency-fragment-set.json --artifact-root EXPERIMENT \
  --out dependency-evidence.json

python3 scripts/multistream_dependency_evidence.py validate \
  --evidence dependency-evidence.json --artifact-root EXPERIMENT --require-complete
```

merger 对 edge 做保守并集，并保留每条 edge 的全部 provider locator。覆盖不全、provider
blocker、非法 statement 或 hard-edge cycle 都会使 evidence 不完整。

operator-order capture v3 的每个 target 必须绑定完整 dependency evidence，且 capture 中的
`hard_dependencies` 必须与 evidence 完全一致。v2 capture 是历史诊断格式，不能授权新的业务
算子重排。
