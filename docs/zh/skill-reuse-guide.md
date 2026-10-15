# Graph-autofusion Skill 复用指南（面向无 cannbot 的外部开发者）

## 概述

本文档面向**没有 cannbot** 的外部开发者，说明如何复用 graph-autofusion 工程内置的 Skills：既可接入其他 AI 工具（Claude Code / Cursor / GitHub Copilot 等）自动加载，也可作为人工开发指南阅读。

工程内置 30 个本地 Skill：9 个通用开发 Skill、7 个 SuperKernel 算子流水线 Skill、14 个 SuperKernel 自动调优与分析 Skill；另有 4 个需单独安装的远程 GitCode 协作 Skill。

### 与其他文档的关系

| 文档 | 定位 | 受众 |
|------|------|------|
| [opencode-skill-management.md](./opencode-skill-management.md) | 三层 Skill 架构、管理机制、贡献流程 | Skill 维护者、贡献者 |
| **本文档** | **无 cannbot 时如何复用内置 Skill** | **外部开发者（接入其他 AI 工具或人工阅读）** |

## Skill 总览

### 通用开发 Skill（9 个，git 跟踪）

| Skill | 定位 | 脚本依赖 | SKILL.md |
|-------|------|----------|----------|
| `af-build-runner` | 编译构建辅助（`build.sh` 参数、CMake 错误、依赖匹配） | 无 | [链接](../../.claude/skills/af-build-runner/SKILL.md) |
| `af-test-developer` | UT/ST 测试开发辅助（gtest/mockcpp、pytest、覆盖率） | 无 | [链接](../../.claude/skills/af-test-developer/SKILL.md) |
| `af-code-reviewer` | 代码审查与贡献规范（红线、格式、PR 模板） | 无 | [链接](../../.claude/skills/af-code-reviewer/SKILL.md) |
| `af-reg-ascir` | ASCIR 注册辅助（新增/修改算子、dtype、tmp buffer、UT/ST 生成） | 无 | [链接](../../.claude/skills/af-reg-ascir/SKILL.md) |
| `cann-toolkit-installer` | CANN Toolkit 自动下载安装（参数解析、校验、静默安装） | 内嵌 bash 逻辑 | [链接](../../.claude/skills/cann-toolkit-installer/SKILL.md) |
| `default-skills` | 默认远程 Skill 安装入口 | `scripts/install-default-skills.sh` | [链接](../../.claude/skills/default-skills/SKILL.md) |
| `af-inductor-cv-validator` | Inductor CV 模型验证矩阵 | `scripts/run_inductor_cv_matrix.py` | [链接](../../.claude/skills/af-inductor-cv-validator/SKILL.md) |
| `af-perf-modeler` | ATT 性能公式与 codegen 参数建模 | 无 | [链接](../../.claude/skills/af-perf-modeler/SKILL.md) |
| `af-prune-graph-metadef` | Graph/MetaDef 裁剪与黑名单校验 | `scripts/check-blacklist.sh` | [链接](../../.claude/skills/af-prune-graph-metadef/SKILL.md) |

### SuperKernel 算子流水线 Skill（7 个，git 跟踪）

| Skill | 定位 | 脚本依赖 | SKILL.md |
|-------|------|----------|----------|
| `sk-operator-pipeline` | SK 算子交付流水线总入口（路由、索引） | `scripts/` | [链接](../../.claude/skills/sk-operator-pipeline/SKILL.md) |
| `sk-operator-asset-adapter` | 算子资产适配（用户目录 → JSON contract） | `scripts/*.py` | [链接](../../.claude/skills/sk-operator-asset-adapter/SKILL.md) |
| `sk-operator-validate` | 适配产物规范校验（contract、源码、兼容性） | `scripts/` | [链接](../../.claude/skills/sk-operator-validate/SKILL.md) |
| `sk-operator-codegen` | SK binding 代码生成（Args struct + `__sk__` + SK_BIND） | `scripts/` | [链接](../../.claude/skills/sk-operator-codegen/SKILL.md) |
| `sk-operator-sample-gen` | 样例生成与验证 contract（输入、oracle、runner） | `scripts/` | [链接](../../.claude/skills/sk-operator-sample-gen/SKILL.md) |
| `sk-operator-build-package` | SK/ACLGraph 编译打包（bisheng 编译 → wheel） | `scripts/*.py` | [链接](../../.claude/skills/sk-operator-build-package/SKILL.md) |
| `sk-model-analysis` | 整网诊断（hang/coredump 定位、性能分析、可视化） | `scripts/*.py` | [链接](../../.claude/skills/sk-model-analysis/SKILL.md) |

### SuperKernel 自动调优与分析 Skill（14 个，git 跟踪）

这些 Skill 从 `superkernel-skill` 迁入 `.claude/skills/`，与算子交付流水线分开，面向模型级融合调优和 profiling 分析。

| Skill | 定位 | SKILL.md |
|-------|------|----------|
| `superkernel-auto-tune` | 全生命周期调优总入口与阶段调度 | [链接](../../.claude/skills/superkernel-auto-tune/SKILL.md) |
| `superkernel-intake-preparation` | 冻结运行环境、工作负载、控制项与可选实验意图 | [链接](../../.claude/skills/superkernel-intake-preparation/SKILL.md) |
| `superkernel-s0-baseline` | 建立五个独立进程的 SK-off clean 性能基线 | [链接](../../.claude/skills/superkernel-s0-baseline/SKILL.md) |
| `superkernel-stage-a-scope-selection` | 筛选完整 scope 策略矩阵并选择候选 | [链接](../../.claude/skills/superkernel-stage-a-scope-selection/SKILL.md) |
| `superkernel-stage-o-option-tuning` | 针对冻结的 Stage A 胜者增量调优 wrapper 选项 | [链接](../../.claude/skills/superkernel-stage-o-option-tuning/SKILL.md) |
| `superkernel-base-profile-source-mapping` | 采集 BASE profiling、逐 SK 分析并确认精确源码映射 | [链接](../../.claude/skills/superkernel-base-profile-source-mapping/SKILL.md) |
| `superkernel-optional-experiments` | 执行用户授权的多流及 P/FINAL 源码范围实验 | [链接](../../.claude/skills/superkernel-optional-experiments/SKILL.md) |
| `superkernel-final-e2e-report` | 最终 clean E2E 验证与中文调优报告 | [链接](../../.claude/skills/superkernel-final-e2e-report/SKILL.md) |
| `superkernel-source-range-from-smap` | 从已完成会话的已验证 BASE/SMAP 派生独立 P/FINAL 实验 | [链接](../../.claude/skills/superkernel-source-range-from-smap/SKILL.md) |
| `superkernel-fusion-performance-analysis` | SK-off/SK-on 性能对比、回退归因与源码映射分析 | [链接](../../.claude/skills/superkernel-fusion-performance-analysis/SKILL.md) |
| `superkernel-multistream-performance-tuning` | 基于新鲜 profiling 调优融合后的多流重叠与调度 | [链接](../../.claude/skills/superkernel-multistream-performance-tuning/SKILL.md) |
| `superkernel-sk-failure-isolation` | 根据 plog/sk_meta 隔离执行、精度、超时或 hang 故障 | [链接](../../.claude/skills/superkernel-sk-failure-isolation/SKILL.md) |
| `superkernel-sk-prof-timeline` | 将逐核 sk_prof trace 折叠为 Cube/Vector 时间线 | [链接](../../.claude/skills/superkernel-sk-prof-timeline/SKILL.md) |
| `superkernel-runtime-common` | 内部共享运行时：分析、设备租约、生命周期、台账与报告 | [链接](../../.claude/skills/superkernel-runtime-common/SKILL.md) |

### 远程 GitCode 协作 Skill（4 个，自动安装）

> 这 4 个 Skill 不在 git 中，需先安装（见"复用注意事项 - 远程 Skill 安装"）。

| Skill | 定位 | 触发场景 |
|-------|------|----------|
| `gitcode-pr` | 创建 PR、获取评论、查看讨论 | 创建 PR、查看 PR 改动、获取 PR 评论 |
| `gitcode-issue` | 读取 Issue 详情和评论 | 查看 issue、读取 issue 评论 |
| `gitcode-pipeline` | 触发流水线并监控状态 | 触发流水线、盯 CI、查看流水线状态 |
| `api-doc-generator` | 生成 API 接口文档 | 生成接口文档、接口说明 |

## 复用方式一：接入其他 AI 工具

### 通用原理

每个 Skill 的核心是 `SKILL.md` 文件，由两部分组成：

1. **Frontmatter（YAML 元数据）**：声明 `name` 和 `description`，AI 工具据此判断何时加载该 Skill。
2. **正文（Markdown 指令）**：描述功能、使用步骤、约束，作为 AI 助手的执行指令。

```markdown
---
name: skill-name
description: |
  简短描述功能。
  **必须触发的场景**：列出关键词和触发场景。
---

## 功能说明
...（正文，AI 工具按此执行）
```

接入 AI 工具的核心思路：**将 `SKILL.md` 转换为目标工具的指令文件格式**，或直接作为系统提示词片段。

### Claude Code

Claude Code 原生支持 `.claude/skills/` 路径约定，**无需转换**。

```bash
git clone https://gitcode.com/cann/graph-autofusion.git
cd graph-autofusion
# 用 Claude Code 打开本目录，自动加载 .claude/skills/*/SKILL.md
claude
```

适用场景：使用 Claude Code 的开发者。注意远程 4 个 Skill 需先运行 `default-skills` 安装（见注意事项）。

### Cursor

Cursor 使用 `.cursor/rules/*.mdc` 文件作为指令。转换要点：

- SKILL.md 的 `description` → Cursor rule 的 frontmatter（`description` + `globs`）
- SKILL.md 正文 → Cursor rule 正文，直接复用
- 一个 SKILL.md 对应一个 `.mdc` 文件

```cursor-rule
---
description: graph-autofusion 编译构建辅助（build.sh、CMake、依赖匹配）
globs: ["build.sh", "CMakeLists.txt", "cmake/**"]
alwaysApply: false
---
（粘贴 af-build-runner/SKILL.md 正文）
```

适用场景：使用 Cursor 的开发者。建议 `globs` 按 Skill 关注的文件类型设置，实现按需触发。

### GitHub Copilot

GitHub Copilot 支持两种方式：

**方式一：单文件聚合**（`.github/copilot-instructions.md`）

将多个 SKILL.md 的关键内容聚合到一个文件，适合 Skill 数量少的场景。

**方式二：多文件拆分**（`.github/instructions/*.instructions.md`，需 VS Code 1.100+）

每个 SKILL.md 对应一个 `.instructions.md` 文件，frontmatter 声明 `applyTo`：

```github-instruction
---
applyTo: "build.sh,CMakeLists.txt,cmake/**"
---
（粘贴 af-build-runner/SKILL.md 正文）
```

适用场景：使用 GitHub Copilot 的开发者。推荐方式二，与 Skill 一一对应，便于维护。

### 通用方法（不支持指令文件加载的 AI 工具）

对不支持指令文件加载的 AI 工具（如网页版 ChatGPT、通义千问等），将所需 SKILL.md 内容**作为系统提示词片段粘贴**：

1. 按当前任务定位相关 Skill（参考"Skill 总览"清单）。
2. 读取对应 `.claude/skills/<name>/SKILL.md` 全文。
3. 在 AI 工具的系统提示词或对话开头粘贴："请按以下指令辅助我完成 graph-autofusion 开发：\n\n{SKILL.md 正文}"。
4. 按需选用，避免一次性粘贴全部 Skill（超出上下文窗口）。

## 复用方式二：作为人工开发指南阅读

不使用 AI 工具时，SKILL.md 同样包含人工可读的开发价值（命令速查、检查清单、流程说明）。

### 通用 Skill 关键速查

| Skill | 人工可读价值 | 关键要点 |
|-------|-------------|----------|
| `af-build-runner` | 编译命令速查、增量编译策略、常见错误表 | 所有编译命令必须加 `-j 8` 限制并行度，避免 OOM |
| `af-test-developer` | 测试模块表、运行命令、覆盖率生成 | `autofuse_e2e` 仅支持 ST（`-s`），不支持 UT（`-u`） |
| `af-code-reviewer` | PR 自查清单、commit message 格式、高风险问题速查 | C++ 4 空格缩进、行宽 120；commit 格式 `<类型>: <简短描述>` |
| `cann-toolkit-installer` | Toolkit 安装参数、流程、依赖 | 默认版本 9.1.0，架构自动检测，安装约 10 分钟 |
| `default-skills` | 远程 Skill 安装入口 | 触发 `install-default-skills.sh` 从 gitcode 拉取 |

详细内容请直接阅读对应 [SKILL.md](../../.claude/skills/af-build-runner/SKILL.md)。

### SK 算子流水线阶段速查

SK 算子交付流水线按以下阶段顺序执行，每阶段对应一个 Skill：

| 阶段 | Skill | 作用 |
|------|-------|------|
| 总入口 | `sk-operator-pipeline` | 运行可定制资产适配器、校验契约、调度各阶段 Skill |
| 1. 资产适配 | `sk-operator-asset-adapter` | 将用户算子仓/源码树/构建资产转换为稳定 JSON contract |
| 2. 规范校验 | `sk-operator-validate` | 校验 contract、源码结构、兼容性，输出 findings |
| 3. 代码生成 | `sk-operator-codegen` | 生成 SK binding（Args struct + `__sk__` template + SK_BIND） |
| 4. 样例生成 | `sk-operator-sample-gen` | 构造运行输入、oracle、runner、differential contract |
| 5. 编译打包 | `sk-operator-build-package` | 调用 bisheng 编译 SK/ACLGraph 扩展，打包为 wheel |
| 诊断 | `sk-model-analysis` | 整网诊断：hang/coredump 定位、性能分析、scope/task 可视化 |

人工阅读时，建议从 `sk-operator-pipeline` 的 SKILL.md入口了解整体流程。

### SuperKernel 自动调优流程速查

从 [superkernel-auto-tune](../../.claude/skills/superkernel-auto-tune/SKILL.md) 开始：

```text
Intake -> S0 -> Stage A -> Stage O -> BASE profiling / analysis / SMAP
       -> [multistream and/or P/FINAL] -> final clean E2E / report
```

Intake 冻结可选模式 `none`、`multistream`、`source-range` 或 `both`；可选实验仅在用户授权后执行，无收益或失败时保留原胜者。最终结论以 clean E2E 时延为准。已有已完成且验证通过的父会话时，可通过 `superkernel-source-range-from-smap` 派生独立 P/FINAL 实验。`superkernel-runtime-common` 是共享依赖，不作为用户入口。

### 远程 Skill 触发场景速查

| Skill | 触发场景 |
|-------|----------|
| `gitcode-pr` | 创建 PR、推送代码到远程、查看 PR 改动、获取 PR 评论 |
| `gitcode-issue` | 查看 issue 详情、读取 issue 评论 |
| `gitcode-pipeline` | 触发流水线、查看流水线状态、等待流水线结果 |
| `api-doc-generator` | 生成接口文档、增加接口说明 |

## 复用注意事项

### 脚本依赖

- **7 个 `sk-*` Skill** 含 Python 脚本（`scripts/*.py`），AI 工具触发执行时需 **Python 3.9+**。人工阅读无需执行脚本，只需理解 SKILL.md 描述的工作流。
- **`cann-toolkit-installer`** 内嵌 bash 逻辑（下载、校验、安装），执行时需 **bash >= 5.1.16**。
- **`default-skills`** 的 `scripts/install-default-skills.sh` 用于安装远程 Skill，需网络可访问 gitcode.com。

人工阅读场景下，脚本仅作参考，不强制执行。

### SuperKernel 调优技能的完整复用

- 跨工程复用时，将全部 14 个 `.claude/skills/superkernel-*` 目录放在同一个 skills 根目录下；保留 `scripts/`、`references/`、`schemas/`、`phase.json`、`agents/` 等现有资源。阶段脚本通过同级相对路径加载控制器和共享运行时，不能只复制 `SKILL.md`。
- 在其他工具中转换指令格式时，也必须保留这些资源，并将脚本路径定位到对应技能目录。`agents/openai.yaml` 仅为可选 UI 元数据，不注册自定义阶段 Agent。
- 完整调优需要宿主提供原生通用子 Agent 或命令适配器，并显式加载阶段 Skill；仅粘贴提示词可用于阅读分析，不能替代执行环境。
- 执行需要 Python 3；NPU 阶段还需要适配工作负载的 Ascend/CANN、模型依赖及 profiling 工具。具体依赖以各 Skill 和环境预检结果为准。阅读文档无需 NPU。

### 远程 Skill 安装

4 个远程 Skill（`gitcode-pr`、`gitcode-issue`、`gitcode-pipeline`、`api-doc-generator`）不在 git 中，需通过以下方式安装：

**方式一：通过 cannbot 触发 `default-skills`**（有 cannbot 环境）

在 opencode 中输入"安装默认 skills"，自动执行 `install-default-skills.sh`。

**方式二：手动安装**（无 cannbot 环境）

```bash
# 克隆远程 skill 仓库
git clone --depth 1 https://gitcode.com/cann-agent/skills.git /tmp/cann-skills

# 拷贝到工程的 _remote 目录
mkdir -p .claude/skills/_remote
cp -r /tmp/cann-skills/gitcode-pr .claude/skills/_remote/
cp -r /tmp/cann-skills/gitcode-issue .claude/skills/_remote/
cp -r /tmp/cann-skills/gitcode-pipeline .claude/skills/_remote/
cp -r /tmp/cann-skills/api-doc-generator .claude/skills/_remote/

# 创建符号链接到一级目录
ln -sf _remote/gitcode-pr .claude/skills/gitcode-pr
ln -sf _remote/gitcode-issue .claude/skills/gitcode-issue
ln -sf _remote/gitcode-pipeline .claude/skills/gitcode-pipeline
ln -sf _remote/api-doc-generator .claude/skills/api-doc-generator

# 清理临时目录
rm -rf /tmp/cann-skills
```

安装后，`.claude/skills/` 一级目录出现 4 个符号链接，AI 工具可自动发现。

### `.gitignore` 白名单规则

`.claude/skills/.gitignore` 采用"**默认忽略 + 白名单放行**"策略：

```gitignore
*
!af-build-runner/
!af-build-runner/**
...（其余本地 Skill 白名单条目）
!.gitignore
```

复用现有 Skill 不受影响。**如新增自建 Skill**，需在 `.gitignore` 中添加白名单条目：

```gitignore
!my-skill/
!my-skill/**
```

详见 [opencode-skill-management.md](./opencode-skill-management.md) 的"`.gitignore` 配置"章节。

### 第三方插件 Skill 不在范围

`superpowers` 等第三方插件 Skill 由 `.opencode/opencode.json` 配置，位于 `~/.cache/opencode/`，非工程内置，本文档不提供复用指引。如需了解，参考 [opencode-skill-management.md](./opencode-skill-management.md) 的"第三层：第三方插件 Skills"章节。

## FAQ

**Q：SKILL.md 是什么？**

A：工程内置的 AI 助手指令文件，位于 `.claude/skills/<name>/SKILL.md`，由 frontmatter（`name`/`description`）和正文（功能、步骤、约束）组成。cannbot 启动时自动加载并按关键词触发；无 cannbot 时可接入其他 AI 工具或人工阅读。

**Q：没有 cannbot 能用这些 Skill 吗？**

A：能。两种方式：① 将 SKILL.md 接入 Claude Code / Cursor / GitHub Copilot 等 AI 工具（见"复用方式一"）；② 作为人工开发指南阅读（见"复用方式二"）。

**Q：接入 Cursor 后 Skill 不触发怎么办？**

A：检查 `.cursor/rules/*.mdc` 的 frontmatter：`description` 是否包含相关关键词、`globs` 是否匹配当前编辑的文件、`alwaysApply` 是否需要设为 `true`。以 Cursor 官方文档为准。

**Q：`sk-*` Skill 的脚本需要手动跑吗？**

A：不需要。人工阅读场景下，脚本仅作参考，理解 SKILL.md 描述的工作流即可。AI 工具触发时按需执行，需 Python 3.9+。

**Q：远程 Skill 和本地 Skill 有什么区别？**

A：本地 Skill（30 个）在 git 中跟踪，`git clone` 即得；远程 Skill（4 个）不在 git 中，需通过 `default-skills` 或手动克隆安装到 `.claude/skills/_remote/`。详见 [opencode-skill-management.md](./opencode-skill-management.md)。

**Q：如何贡献新 Skill？**

A：参考 [opencode-skill-management.md](./opencode-skill-management.md) 的"新增本地 Skill"章节：创建模块化 Skill 包 → 编写 `SKILL.md` + `README.md` → 在 `.gitignore` 添加白名单 → 提交 PR。
