# 多流自动化流水线

本页描述五项自动化能力的调用顺序。所有输出均为不可变 JSON；已有输出不得覆盖。

## 1. Short-trace 归因

先从 sibling schema 1.2 profiling result 生成不可漏项的多流 target 清单：

```bash
python3 scripts/multistream_opportunity_discovery.py discover \
  --profiling-analysis incumbent/profiling-analysis-result.json \
  --artifact-root . \
  --out multistream-opportunity-discovery.json
```

清单必须包含所有 performance-exact 且双方至少三个 occurrence 的多流 SK，包括
`net_effect=beneficial`。request 的 screening targets 从该清单生成，初始
`parallelism_effect=unknown`、`optimization_status=blocked`，不能由调用方预判结果。

模型 profile 命令生成 `superkernel-multistream-short-trace-capture-v2`。新模型接入必须通过
`scripts/multistream_capture_producer.py` 的指纹化插件协议生产 capture；插件负责解析模型
命令的原始 CSV/JSON/sk_meta，producer host 负责绑定输入文件、封印 fingerprint，并调用本
节 analyzer 做 conformance replay。插件不得自行填写 `capture_fingerprint`。

插件使用 `superkernel-multistream-capture-plugin-v1`，声明稳定 `PLUGIN_ID`、支持的
`CAPABILITIES`，并实现 `produce_capture(kind, context)`。先检查插件：

```bash
python3 scripts/multistream_capture_producer.py inspect-plugin \
  --plugin-root adapters/MODEL --plugin capture_plugin.py \
  --plugin-fingerprint sha256:PLUGIN
```

再冻结 `superkernel-multistream-capture-job-v1`，执行 `produce --job JOB --plugin-root ROOT`。
job 必须绑定每个原始输入的相对路径和 SHA256，并为 short trace 指定 request、为 operator
order 指定 request fingerprint、为 post-dispatch 指定 action manifest。输出包括不可覆盖的
capture 和 `superkernel-multistream-capture-production-receipt-v1`。

capture 必须显式绑定
request/trial、父 SK 的 device/model/SK identity、graph occurrence fingerprint，以及
baseline/candidate 至少三个同名 `alignment_id`。每个 child event 必须包含 origin identity、
lane、stream、core family、start、duration 和 event kind，并列出原始 trace 文件 SHA256。

```bash
python3 scripts/multistream_trace_analysis.py analyze \
  --request multistream-request.json \
  --capture trials/MS-O1/short-trace-capture.json \
  --out trials/MS-O1/trace-analysis.json
```

工具逐 occurrence 输出 interval、duration sum、union、max active streams、stream pair/CV/
same-resource/mix overlap 和 wait/sync，并汇总 P50/P90/MAD。overflow、少于三个 occurrence、
alignment 不一致或 lane/stream/core identity 不完整都会阻断直接调度结论。

分析后必须审计覆盖率：

```bash
python3 scripts/multistream_opportunity_discovery.py audit-coverage \
  --discovery multistream-opportunity-discovery.json \
  --trace-analysis trials/MS-SCREEN/trace-analysis.json \
  --artifact-root . \
  --out multistream-opportunity-coverage.json
```

`complete=false` 时不得输出“没有 beneficial + degraded 机会”。trace analysis v2 额外输出
lost overlap、C/V retention、child-work inflation、wait/sync delta、latent/actionable opportunity。
任意 overlap 丢失可判 `degraded`，但只有纯 C/V 丢失可进入 action planner；Mix/同资源退化
保留为 `blocked` 诊断。

### 1.1 执行序、下发序与重排授权

先阅读 [identity-binding.md](identity-binding.md)。模型 adapter 将 SK-off
`kernel_details.csv` 聚合到业务算子实例，并将 SK-on `sk_meta`
子算子下发序绑定到相同 `operator_id + statement_id`，生成
`superkernel-multistream-operator-order-capture-v3`。读取
[dependency-provider.md](dependency-provider.md)，先合并完整 dependency evidence。capture
还必须包含同一 exact range 内连续 statement byte span、dependency evidence 路径以及与其
完全一致的 hard dependency：

adapter 必须先用 `multistream_identity_binding.py` 生成并以 `--require-complete` 校验 identity
registry。capture plugin 从 registry 读取稳定 identity；不得在插件内部按名称或 ordinal
临时关联 SK-off、SK-on 和源码。

两个 v2 capture 的事件都保存 SK-off `kernel_details.csv` 原始
`accelerator_core/block_num/mix_block_num`。分析器用
`multistream_core_family.py` 派生资源类型；producer 不再提交可伪造的 `core_family`。

```bash
python3 scripts/multistream_operator_order.py analyze \
  --capture operator-order-capture.json \
  --request-fingerprint sha256:REQUEST \
  --out operator-order-analysis.json
```

只有至少三个 occurrence 都证明跨可靠 stream 的真实 `Cube <-> Vector` overlap，且 SK-on
下发序对同一 stable execution preference 形成一致 inversion 时，才输出
`multistream_reorder_authorized=true`。单流或 identity 不完整直接 blocked。

## 2. 候选矩阵

`superkernel-multistream-action-catalog-v1` 只列 wrapper/environment 已接受的 option value，
以及 analyzer exact boundary 上已受审的 source insertion。Stage O history 使用
`superkernel-multistream-option-history-v1`。

```bash
python3 scripts/multistream_candidate_planner.py plan \
  --request multistream-request.json \
  --trace-analysis trials/MS-O1/trace-analysis.json \
  --action-catalog action-catalog.json \
  --operator-order-analysis operator-order-analysis.json \
  --out candidate-matrix.json
```

矩阵按证据、风险和 catalog priority 排序，最多选取 request `max_trials`。相同 global
option pointer/value 已在 Stage O 结算时只记录 dedup reason，不再次执行。排名不改变 clean
E2E acceptance gate。

当 request 包含 `dependency_safe_operator_reorder` 时，planner 先生成一个 route 2 最小
permutation；route 3 bounded topology 候选预先冻结，但只有 route 2 已改变下发序并结算为
稳定、无回归但未达到增益阈值后才能激活。物化 route 3 必须绑定 route 2 的 passed dispatch
evidence 和 rejected clean3 semantic evidence。两条路线都必须保留 stable execution
preference 与 hard dependency，且不能与 option、scope split 或 range exclusion 合并。

## 3. Adapter 与统一租约

使用 `superkernel-multistream-model-run-spec-v1` 生成 frozen adapter：

```bash
python3 scripts/multistream_adapter_generator.py \
  --spec model-run-spec.json --out model-adapter.json
```

generator 自动展开五阶段标准 validator、semantic evidence 路径、clean3/5 run count 和
reject code。父普通实验的 NPU 命令必须通过：

```bash
python3 ../superkernel-runtime-common/scripts/device_lease_runner.py \
  --lease-root /abs/shared-leases --device-id 0 \
  --lease-timeout-seconds 300 --timeout-seconds 1800 \
  --cwd /abs/model --manifest-out parent-command.json -- /abs/run.sh
```

该入口和多流 runner 共用 `npu-device-X.lock`。模型仓级 source identity 使用显式 file list：

```bash
python3 scripts/multistream_execution.py snapshot-source \
  --source-root isolated/source --artifact-root . \
  --source-revision REV --file-list source-files.json \
  --manifest-out source-manifest.json
```

重排动作使用：

```bash
python3 scripts/multistream_execution.py materialize-reorder \
  --input isolated/source/model.py --output isolated/source-trials/MS-R1/model.py \
  --operator-order-analysis operator-order-analysis.json \
  --range-id RANGE --after-order route2-order.json \
  --trial-id MS-R1 --artifact-root . --manifest-out trials/MS-R1/action-manifest.json
```

compile variables 必须为 reorder plan 冻结一个尚可不存在的
`dispatch_order_evidence` 相对路径。correctness 通过后，模型 capture producer 生成
post-reorder dispatch capture，再执行：

```bash
python3 scripts/multistream_operator_order.py verify-dispatch \
  --action-manifest trials/MS-R1/action-manifest.json \
  --dispatch-capture trials/MS-R1/post-dispatch-capture.json \
  --artifact-root . --out trials/MS-R1/dispatch-order-evidence.json
```

runner 在启动 profile phase 前确定性重放该 evidence；下发序未改变、child set 改变或任一
occurrence 退化为单流时立即 blocked。

## 4. Trace 与 result-v2

重排 trial 的 profile phase 使用 `superkernel-multistream-four-profile-plan-v1` 固定四个角色：
`incumbent_sk_off`、`incumbent_sk_on`、`candidate_sk_off`、`candidate_sk_on`。先执行：

```bash
python3 scripts/multistream_four_profile.py freeze-plan \
  --draft four-profile-plan-draft.json --out four-profile-plan.json
python3 scripts/multistream_four_profile.py run --plan four-profile-plan.json
```

该 executor 必须作为父 multistream runner 的 device profile phase 运行，并核对父进程注入的
device lease marker。每个角色分别密封命令、validator、必需产物和 manifest；四个角色全部
通过后才生成 `superkernel-multistream-four-profile-summary-v1`。summary 固定给出 incumbent
fusion、candidate fusion、SK-off change 和 SK-on change 四种比较关系，后续 analyzer 不得
混用 profile 角色。

profile semantic validator 可增加 `--trace-analysis`。若 trial 声明
`mechanism_status=validated`，result validator 要求 capture/request/trial/range identity
完全一致，全部源文件 SHA256 未变化，overflow=false、blockers 为空、至少三个 occurrence，
并且 `parallelism_effect=improved`、对应 finding 为 `optimization_status=validated`。

不满足时只能写 `unproven/not_observed/blocked`，不会影响 clean E2E accepted 的独立判断。

## 4.1 真实 NPU 闭环验收

真实闭环先准备 `superkernel-multistream-npu-closure-v1` draft，artifact 必须完整包含 request、
operator-order capture v3、materialized action manifest、passed dispatch-order evidence、
four-profile plan/summary、clean3 semantic evidence 和 result-v2。然后密封并重放：

```bash
python3 scripts/multistream_npu_closure.py seal \
  --draft npu-closure-draft.json --artifact-root EXPERIMENT_ROOT \
  --out npu-closure.json
python3 scripts/multistream_npu_closure.py validate \
  --manifest npu-closure.json --artifact-root EXPERIMENT_ROOT
```

runtime 必须证明 `npugraph_ex + static_kernel_compile + SuperKernel scope + >=2 streams`
在真实 NPU 上执行。审计器会重放所有现有 analyzer/contract，而非只核对文件存在；必须存在
一个 analyzer 授权并实际执行的 reorder trial。最终只允许 `accepted` 或 `no_gain`，后者必须
绑定同一 trial 的 `rejected` 决策和 clean `reject` evidence。

## 5. Derived family 生命周期

`bootstrap_derived_family.py` 创建 SEED 后，冻结并运行 fresh BASE：

```bash
python3 ../superkernel-runtime-common/scripts/derived_family_lifecycle.py freeze-plan \
  --draft fresh-base-plan-draft.json --out fresh-base-plan.json
python3 ../superkernel-runtime-common/scripts/derived_family_lifecycle.py run-base \
  --plan fresh-base-plan.json --registry derived-family-registry.json
```

模型命令必须生成 `superkernel-derived-fresh-base-receipt-v1`，证明 correctness、profiling、
analysis、exactly-five clean 全部通过并绑定所有 artifact SHA256。完成普通生命周期后：

```bash
python3 ../superkernel-runtime-common/scripts/derived_family_lifecycle.py complete-lifecycle \
  --registry derived-family-registry.json --family-id S3M \
  --result families/S3M/ordinary-result.json
python3 ../superkernel-runtime-common/scripts/derived_family_lifecycle.py merge-ledger \
  --registry derived-family-registry.json --family-id S3M \
  --ledger performance-ledger.json --output performance-ledger.next.json
```

状态严格为 `seed_registered -> fresh_base_completed -> ordinary_lifecycle_completed ->
ledger_merged`。任何前置失败均保持 `ledger_merge_allowed=false`。
