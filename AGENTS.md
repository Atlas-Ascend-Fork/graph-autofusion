# AGENTS.md

本文件为 agent 在 Graph-autofusion 仓库中工作时提供指导。所有回答默认使用中文，命令、路径和代码保持原文。

## 项目概述

Graph-autofusion 是面向昇腾（Ascend）芯片的融合加速组件集合，当前包含 SuperKernel 和 Autofuse 两个主要组件。项目通过 SuperKernel、自动融合、ASCIR/AscendC codegen 等能力减少任务调度和数据搬运开销。

### 关键目录

| 目录 | 用途 |
|------|------|
| `super_kernel/` | SuperKernel 组件源码、Python/C++ 测试、设备端验证用例 |
| `autofuse/` | Autofuse 自动融合组件源码、优化 pass、codegen、测试框架 |
| `cmake/` | CMake 公共脚本、依赖和打包逻辑 |
| `scripts/` | 构建、测试、OAT 检查等脚本 |
| `docs/` | 构建、设计、规范和组件说明文档 |
| `.claude/skills/` | 项目专用 agent skills |

## 构建与测试

### 编译构建

**使用技能**： `af-build-runner`

**触发场景**： 编译、构建、build、cmake、make、打包、build.sh、构建失败、编译选项、第三方依赖。

关键要求：所有构建命令必须限制并行度，优先使用 `-j 8`，避免 Autofuse 编译 OOM。

```bash
sh build.sh --pkg -j 8
sh build.sh --pkg --no-autofuse -j 8
cmake --build build --target <target> -j 8
```

### 测试开发与运行

**使用技能**： `af-test-developer`

**触发场景**： 写测试、UT、ST、gtest、pytest、mock、coverage、测试失败、运行测试。

常用命令：

```bash
sh build.sh -u --module=superkernel --impl=py
sh build.sh -u --module=superkernel --impl=cpp
sh build.sh -u --module=autofuse_framework -j 8
sh build.sh -s --module=autofuse_e2e -j 8
```

## 需求开发与设计文档

新增功能、需求、特性或修改关键流程前，先明确假设、影响范围和验证方式。

任何输出设计文档/spec 的场景（用户直接要求写设计文档、输出设计方案、需求设计等），**必须**先读取 `docs/guidelines/design_document_template.md`，然后按照模板格式输出。模板中的每个章节都必须覆盖。即使所用技能有自己的格式要求，也要以本模板为准。

设计文档必须额外完成：

- [ ] **跨特性交叉影响**：读取 `docs/guidelines/cross_feature_check.md`，逐项分析 SuperKernel、Autofuse、打包交付、Python/C++ 接口和测试场景影响。
- [ ] **编码红线**：读取 `docs/guidelines/编码红线.md`，确认方案不触碰安全、资源、ABI/API、图改写确定性和运行时生命周期红线。
- [ ] **测试设计**：根据变更类型说明 UT/ST/RDV/pytest/gtest 覆盖策略。
- [ ] **性能评估**：涉及编译期 pass、codegen、运行时调度、拷贝或日志时，明确性能影响。

示例输出：

```markdown
### 设计文档检查结果
- [x] 跨特性交叉影响：已按 cross_feature_check.md 检查 Autofuse optimize/codegen 与 SuperKernel 交付场景，确认无额外影响
- [x] 编码红线：已检查资源生命周期、ABI/API、图改写确定性和运行时接口约束，无违反
- [x] 测试设计：新增 optimize UT 和 codegen E2E ST，覆盖正向和异常输入
```

## 开发规范

### GitCode PR / Issue / CI

GitCode PR、Issue、流水线相关操作使用项目默认 skills：

- 创建 PR、查看 PR 评论、查看 PR diff：`gitcode-pr`
- 查看 Issue：`gitcode-issue`
- 触发或查看流水线：`gitcode-pipeline`
- skills 缺失或安装默认 skills：`default-skills`

### 代码检视

**使用技能**： `af-code-reviewer`

**触发场景**： 代码审查、review、检视 PR、代码规范、贡献规范、clang-format、commit message、PR 模板、pre-commit、代码检查。

代码检视前必须读取：

- `docs/guidelines/编码红线.md`
- `docs/guidelines/cross_feature_check.md`
- `CONTRIBUTING.md`
- `.clang-format`
- `.pre-commit-config.yaml`

### Autofuse 模板语义命名

新增模板、实现策略、API 分支或回退，以及影响名称的分组、图改写和 reuse 映射变更前，必须读取并遵循：

- [Codegen 语义命名规范](docs/zh/autofuse/design/features/codegen.md#template-semantic-naming)：统一词表、最终 API 采集、完整身份、不确定状态、测试及性能要求。
- [Schedule 语义信息传递](docs/zh/autofuse/design/features/schedule.md#template-semantic-propagation)：策略、组级角色、成功转换及复制/回退规则。

名称适配必须随模板交付；仅数值参数变化时说明沿用既有分类的依据。PR 中列明采集点、身份/reuse 处理及行为测试证据；不得以默认父名或 `Adaptive` 掩盖遗漏。复用公共 helper 和现有测试，不另建命名词表、模板清单或选择算法。自动接入 CI 检查仍待实施，开发者和 reviewer 必须按上述规范检查。

### 代码风格

- 遵循 Google 开源代码规范和项目 `.clang-format`。
- C++ 使用 4 空格缩进，行宽 120，不使用 Tab。
- `if` / `for` / `while` / `do-while` 语句应使用大括号。
- 不新增硬编码密钥、账号、公网地址、芯片类型或框架类型判断。
- 不做与需求无关的重构、格式化或清理。
- 修改 C++ 文件后，优先对相关文件执行 `clang-format -i <file>`。

## 编码前先思考

**不要想当然。不要掩饰困惑。把权衡摆在明面上。**

实施前：

- 明确陈述你的假设。如果不确定，就问。
- 如果存在多种理解，全部列出来，不要默默选一个。
- 如果有更简单的方案，说出来。必要时提出反对意见。
- 如果有不清楚的地方，停下来。指出困惑之处。提问。

## 最小完整变更

**最小修改指完成需求所需的最小完整解决方案，不等同于最少的代码行数或最少的文件数。**

- 只修改与需求直接相关的范围。
- 优先复用现有架构、抽象、接口和扩展点，不通过局部特判绕过既有设计。
- 当现有结构无法正确承载需求时，进行完成需求所必需的边界调整、抽象补充或接口扩展。
- 保持职责清晰、依赖方向稳定、行为确定，并与同类功能采用一致的实现方式。
- 不得为了减少当前改动而引入重复逻辑、隐式状态、临时分支、跨层调用或难以维护的特殊路径。
- 不实现超出需求的功能，不添加未经要求的“灵活性”或可配置性，不为不可能发生的场景做错误处理。
- 不为一次性代码做抽象；如果抽象是保持架构一致性或避免重复逻辑所必需的，应说明其必要性。
- 不得借机进行与需求无关的重构、格式化、清理或预留式设计。

区分以下三类改动：

- **必需改动**：没有它需求无法实现。
- **一致性改动**：没有它会破坏既有架构、接口或同类行为。
- **顺手改动**：虽然代码可能更“干净”，但不是本需求所必需。

只允许前两类改动，禁止第三类改动。每一处改动都必须能够说明其对实现需求、保持架构一致性、修复本次改动引入的影响或提供必要验证覆盖的必要性。

出现以下情况时，应停止继续叠加局部修改，重新审视设计，避免形成“补丁集中营”：

- 同一逻辑出现多个特殊分支。
- 多个层次重复判断同一个条件。
- 新增代码无法复用已有抽象。
- 为绕开接口限制而增加隐式状态或回调。
- 测试只能覆盖具体样例，无法表达通用规则。
- 修改后调用方、实现方和测试方的职责变得不清晰。

如果局部修补与架构设计冲突，应优先采用范围适度的结构化修改，并明确说明扩大范围的原因、替代方案和风险。结构化修改必须只围绕当前需求涉及的边界，不得顺带处理其他历史问题。

问问自己：“这是最小的完整解决方案，还是为了少改代码而留下了结构性补丁？”

## 精准修改

**只改必须改的，以及保持架构闭环所必需的改动；只清理自己弄乱的。**

编辑现有代码时：

- 不要“改进”相邻的代码、注释或格式。
- 不要重构没有问题的东西。
- 匹配既有风格，即使你会用不同的写法。
- 如果你注意到无关的死代码，提出来，不要删除它。

当你的修改产生了孤立代码时：

- 删除因你的修改而变得未使用的导入、变量、函数。
- 不要删除之前就存在的死代码，除非被要求。

检验标准：每一处改动都应该能追溯到用户的请求，或能明确证明是实现该请求所必需的架构一致性、影响修复或验证改动。

## 目标驱动执行

**定义成功标准。循环验证直到达标。**

将任务转化为可验证的目标：

- “添加校验” -> “为无效输入编写测试，然后让它们通过”。
- “修复 bug” -> “编写能复现该 bug 的测试，然后让它通过”。
- “重构 X” -> “确保重构前后测试都通过”。

对于多步骤任务，简要说明计划：

```text
1. [步骤] -> 验证: [检查项]
2. [步骤] -> 验证: [检查项]
3. [步骤] -> 验证: [检查项]
```

明确的成功标准让你能够独立循环迭代。模糊的标准（“让它能用”）则需要不断沟通确认。
