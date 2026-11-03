# MLIR Agent 协作与 Git 提交指南

---

## 1. 开发验证
基础 UT 验证：参考 autofuse/mlir/README.md 中的 `Docker编译方案`和`本地编译方案` 进行编译验证。

NPU case 验证：
```bash
bash autofuse/tests/st/codegen/ascir_tool/test_ascir.sh --mode=0 --case=<case_name>
```

---

## 2. 分支角色

```text
个人 feature 分支 ──PR──> niuyuhu:af-mlir ──PR──> cann:develop
                              ↑
                              └── 普通 Merge cann:develop
```

| 角色 | 仓库与分支 | 用途 |
|---|---|---|
| 官方目标分支 | `cann/graph-autofusion:develop` | MLIR 代码最终合入目标 |
| 团队集成分支 | `niuyuhu/graph-autofusion:af-mlir` | 汇总所有 MLIR feature PR，持续同步官方 develop |
| 个人开发分支 | `<developer>/graph-autofusion:feat/mlir-<topic>` | 单个需求/Bug/迁移任务的隔离分支 |

MLIR 协作以 `develop` 为主线；不把 `master` 合入 `af-mlir`，也不从 `master` 创建 MLIR feature 分支。

**不可违反的规则：**

1. `af-mlir` 是多人共享长期分支，禁止 rebase，禁止 force push。
2. `develop → af-mlir` 用普通 Merge，不用 Squash。
3. `af-mlir → develop` 用普通 Merge，合入前必须取消 Squash。
4. 个人独占 feature 分支可以 rebase 到最新 `af-mlir`；多人共用的 feature 分支用 Merge。
5. feature PR 合入 `af-mlir` 推荐 Squash Merge。
6. 不得直接向共享 `af-mlir` 推送功能代码，必须通过 PR。
7. 不使用 `git add .`/`git add -A`；显式暂存任务涉及的文件。
8. 不提交 build、coverage、日志、截图、dump、kernel_meta、临时文档等任务外文件。
9. 不使用 `git reset --hard`、`git checkout --` 等命令覆盖用户工作区。
10. 发现来源不明的文件或提交先停止，不用 `ours`/`theirs` 整侧覆盖来隐藏问题。

**Remote 约定：** 不得仅凭 remote 名称判断仓库，先执行 `git remote -v` 按 URL 识别角色（`cann/graph-autofusion` 建议命名 `upstream`；`niuyuhu/graph-autofusion` 建议命名 `origin` 或 `team`；自己的 fork 通常是 `origin`）。

---

## 3. Agent 自动执行协议

处理 MLIR Git/PR 任务时按序执行：

1. 完整读取本文和仓库根目录 `AGENTS.md`。
2. 检查当前分支、remote、工作区状态、目标分支最新 SHA。
3. 明确源仓库、源分支、目标仓库和目标分支，不硬编码 remote 名称。
4. 记录并保护所有任务外脏文件和未跟踪文件。
5. 只执行用户授权范围内的修改、提交、推送或 PR 操作。
6. 提交前展示变更文件、关键 diff、验证结果和未验证项。
7. 推送用正常 push；只有个人 feature rebase 后允许 `--force-with-lease`。
8. 推送后重新 fetch，核对本地 SHA 与远端 SHA 一致。
9. 最终报告源/目标 SHA、提交、文件数量、验证结果和残余风险。

用户明确说“提交”“推送”“提 PR”或“处理冲突”时，Agent 可在上述门禁通过后连续完成对应操作，不需要对每个安全、可逆的常规步骤重复询问。以下操作仍必须获得明确授权：

- force push 共享 `af-mlir`。
- 删除远端分支或 tag。
- 关闭、合入或修改不在当前任务范围内的 PR。
- 丢弃或覆盖用户已有修改。
- 改变 PR 的目标仓库或目标分支。

---

## 4. 提交前检查清单

- [ ] 已从最新 `team/af-mlir` 创建或同步 feature 分支。
- [ ] 已逐文件检查 diff，没有使用 `git add .` 或 `git add -A`。
- [ ] 新功能或 Bug 修复已有对应测试；未运行项已明确说明。
- [ ] C++、Python、文档和配置已通过对应格式/静态检查。
- [ ] Commit message 符合 `<type>: <简短描述>`。
- [ ] PR 描述符合仓库模板，测试结果和未覆盖项如实填写。
- [ ] `develop → af-mlir` 和 `af-mlir → develop` 均使用普通 Merge；`af-mlir → develop` 已明确关闭 Squash。
- [ ] 推送后已验证远端 SHA 和 PR 文件列表。

---

## 5. 安全红线

- 不硬编码密码、token、账号、私有服务器地址或私有安装路径；GitCode API token 只从 `GITCODE_API_TOKEN` 环境变量或本地 Git 配置读取，不写入脚本或提交，不出现在命令输出和日志中。
- 不把 CANN/NPU 服务器信息写入正式文档。
