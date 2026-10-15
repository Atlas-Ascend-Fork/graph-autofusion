# Adapter 支持矩阵

`scripts/multistream_adapter_support.py` 用证据化 capability declaration 判断一个网络 adapter
能否执行 `option`、`scope_split`、`range_exclusion` 和
`dependency_safe_operator_reorder`。它回答“接入能力是否完备”，不代替候选收益判断。

每个 adapter 必须完整声明固定 capability catalog。`supported` 必须引用至少一个已封印
artifact；`unsupported` 必须给出 `{code, detail}` blocker。不能省略未知项，也不能只写
“已支持”。artifact 会校验 schema、文件 SHA256 和关键语义，例如 conformance 必须 pass、
NPU closure 必须有 executed outcome、lease inventory 必须无 blocker。

动作要求由通用代码中的 `ACTION_REQUIREMENTS` 唯一定义。矩阵逐项输出：

- `available`：所有 capability 与必需 artifact 已满足；
- `blocked`：列出缺失 capability、artifact 或当前证据 blocker；
- `NO_STABLE_SAME_PARENT_PAIR`：resource screening 已证明当前没有合法 reorder pair。这不表示
  capture/classifier 通用能力失败，也不能用来冒充真实 reorder 闭环。

`availability_scope=adapter_capability_not_candidate_recommendation` 表示 available 只授权进入候选
规划，不能跳过 dispatch、four-profile、correctness 和 clean gate。开始真实实验前还必须执行
[adapter-capability-preflight.md](adapter-capability-preflight.md)，把 capability matrix 与本轮
producer/validator/expected-output 计划绑定起来。

## 命令

```bash
python3 <skill-dir>/scripts/multistream_adapter_support.py freeze-adapter \
  --draft adapter-draft.json --root <repo-root> --out adapter.json
python3 <skill-dir>/scripts/multistream_adapter_support.py build \
  --matrix-id NETWORKS --adapter path/to/a.json --adapter path/to/b.json \
  --root <repo-root> --out support-matrix.json
python3 <skill-dir>/scripts/multistream_adapter_support.py validate \
  --matrix support-matrix.json --root <repo-root>
```

冻结本轮采集计划并执行强制 preflight：

```bash
python3 <skill-dir>/scripts/multistream_adapter_support.py freeze-collection-plan \
  --draft collection-plan.draft.json --root <repo-root> \
  --out collection-plan.json

python3 <skill-dir>/scripts/multistream_adapter_support.py preflight \
  --preflight-id S3-P1-PREFLIGHT \
  --matrix support-matrix.json --collection-plan collection-plan.json \
  --adapter-id deepseek-v4-a3-v1 --action range_exclusion \
  --root <repo-root> --out adapter-preflight.json
```

`preflight` 即使 blocked 也先写不可变 receipt，然后返回非零状态 2。调用方必须同时检查
进程返回值和 receipt 的 `status=ready`，不得只看文件存在。

新网络先生成 capability adapter，再读取矩阵 blockers 补齐接入项。adapter、artifact 或动作
需求变化后旧矩阵无法重放，必须生成新的不可覆盖输出。
