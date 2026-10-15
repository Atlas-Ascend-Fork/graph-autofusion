# 隔离 trial 自动回滚与清理

多流分支达到 `accepted`、`rejected`、`blocked` 或 `failed` 后，使用
`scripts/multistream_cleanup.py` 清理隔离分支的可变状态。执行器不做 reverse patch，
也不直接删除文件；它把计划中逐项绑定指纹的目标原子迁移到独立 quarantine，因此失败后
可以续跑，误操作时也可以人工恢复。

## 安全边界

- `isolation_root`、`incumbent_root`、`quarantine_root` 必须是三个互不包含的绝对路径；
- target 必须是 isolation 下的安全相对路径，不能包含符号链接、重叠父子目标或缺失对象；
- plan 冻结每个 target 和整个 incumbent 的内容指纹；执行前后都复核 incumbent；
- `preserved_paths` 声明保留的状态、profile、result 和日志，且不得与 target 重叠；
- 已迁移 target 与原位置不能同时存在；partial move 只在 quarantine 指纹吻合时续跑；
- receipt 已存在时只做全量重放校验，不重复迁移，因而同一计划可幂等调用。

`accepted` 终态只能清理已经完成交付后的 scratch/cache 等可变项；仍是交付物的 candidate
源码必须列入 `preserved_paths`。其他终态通常将 candidate source/config/cache 列入 targets，
但仍保留审计 artifact。

## 使用方式

先写 draft，其中 `targets` 是相对 isolation root 的路径列表：

```json
{
  "schema_version": "superkernel-multistream-cleanup-plan-v1",
  "cleanup_id": "cleanup-MS-R1",
  "trial_id": "MS-R1",
  "terminal_state": "rejected",
  "isolation_root": "/experiment/isolated/MS-R1",
  "incumbent_root": "/experiment/incumbent",
  "quarantine_root": "/experiment/quarantine",
  "targets": ["source", "config", "cache"],
  "preserved_paths": ["artifacts"]
}
```

然后冻结、执行和重放校验：

```bash
python3 <skill-dir>/scripts/multistream_cleanup.py freeze-plan \
  --draft cleanup-draft.json --out cleanup-plan.json
python3 <skill-dir>/scripts/multistream_cleanup.py run \
  --plan cleanup-plan.json --receipt cleanup-receipt.json
python3 <skill-dir>/scripts/multistream_cleanup.py validate \
  --plan cleanup-plan.json --receipt cleanup-receipt.json
```

任一步骤报告 incumbent 变化、路径越界、符号链接、双份 target 或指纹不一致时，清理必须
`blocked`，不能扩大 target 或清空共享目录。quarantine 的最终删除不属于自动流程。
