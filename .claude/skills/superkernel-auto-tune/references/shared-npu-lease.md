# 共享 NPU lease 协议

所有可能启动 NPU 工作负载的父流程必须使用同一绝对 `lease_root` 和升序 `device_ids`。
标准环境标记为：

```text
SUPERKERNEL_DEVICE_LEASE_HELD=1
SUPERKERNEL_DEVICE_IDS=0,1
SUPERKERNEL_DEVICE_LEASE_ROOT=/absolute/shared/lease/root
```

`multistream_runner.shared_device_leases` 是唯一的 acquire-or-inherit 入口。没有父标记时，
它按设备 ID 升序获取 `npu-device-X.lock`；三个父标记完整且精确匹配时复用父租约；标记缺失、
部分存在、device 或 root 冲突都 fail closed。父级 `device_lease_runner.py` 必须把规范标记注入
子进程，嵌套的多流 runner 因而不会再次获取同一把锁。
父命令 manifest v2 记录 `lease_mode=acquired|inherited`；争锁超时或 marker 冲突时为
`blocked`，且 `lease_mode=null`。

四 profile 编排不单独获取 lease。它在启动任何 role 前要求匹配的父标记，因此四份 profile
位于一个不可被其他父流程插入的连续租约中。派生 family fresh-BASE 始终委托给父 wrapper。
离线 analysis phase 不需要设备，也不因外层持有设备而重复加锁。

## 入口审计

每次修改进程启动逻辑后生成并验证清单：

```bash
python3 <skill-dir>/scripts/shared_npu_lease_inventory.py \
  --repository-root <repo-root> --out shared-npu-lease-inventory.json
python3 <skill-dir>/scripts/shared_npu_lease_inventory.py \
  --repository-root <repo-root> --validate shared-npu-lease-inventory.json
```

扫描覆盖生产脚本的 `subprocess`、底层 `_run_argv`、父 wrapper 委托和已知离线子进程。
出现未登记启动点、登记点消失、入口函数缺失或守卫 token 缺失时，报告为 `blocked`。
不能通过把新 launcher 描述为“模型专用脚本”来绕过清单。
