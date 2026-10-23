---
name: af-code-reviewer
description: |
  graph-autofusion 项目代码审查和贡献规范辅助。
  **必须触发的场景**：用户提到代码审查、review、检视PR、检查PR、代码规范、贡献规范、CONTRIBUTING、代码风格、clang-format、commit message、PR模板、pre-commit、代码检查、编码红线、设计规范等。
---

# graph-autofusion 代码审查与贡献规范

辅助进行 graph-autofusion 的代码检视、规范合规检查和 PR 提交前自查。审查目标是发现高信号问题：确定的 Bug、明确的规范违反、兼容性风险和测试缺口。

## 使用原则

- 先读取规范，再评价代码；不得凭记忆审查。
- 只标记能被代码或规范明确证明的问题；不确定的问题作为疑问提出，不作为缺陷结论。
- 只审查本次变更引入或暴露的问题；不要要求修改无关历史代码。
- 优先指出会导致编译失败、运行错误、资源泄露、ABI/API 不兼容、图行为不等价或交付失败的问题。
- 输出 findings 时按严重程度排序，并给出文件路径和行号。

## 代码检视强制检查清单

执行代码审查或 PR 自查前，必须逐项完成：

- [ ] **编码红线**：完整读取 `docs/guidelines/编码红线.md`，逐条核对变更是否违反安全、资源、ABI/API、图改写、runtime 生命周期和构建交付红线。
- [ ] **跨特性交叉影响**：读取 `docs/guidelines/cross_feature_check.md`，按 SuperKernel、Autofuse、AscendC/runtime、Python/C++ 绑定、构建打包、测试和性能场景逐项分析影响。
- [ ] **贡献规范**：读取 `CONTRIBUTING.md`，检查 Issue/PR、commit message、CLA 和贡献流程要求。
- [ ] **代码格式**：读取 `.clang-format`，确认 C++ 文件符合格式要求。
- [ ] **Pre-commit**：读取 `.pre-commit-config.yaml`，确认 clang-format 和 OAT 检查要求。
- [ ] **Autofuse 模板命名**：涉及新增模板、实现策略、API 分支/回退或影响名称的分组、图改写、reuse 映射时，读取 [Codegen 语义命名规范](../../../docs/zh/autofuse/design/features/codegen.md#template-semantic-naming) 和 [Schedule 传递规则](../../../docs/zh/autofuse/design/features/schedule.md#template-semantic-propagation)，按下述专项要求审查。
- [ ] **PR 模板**：若准备提交 PR，读取 `.gitcode/PULL_REQUEST_TEMPLATE.zh-CN.md` 或 `.gitcode/PULL_REQUEST_TEMPLATE.en-US.md`。
- [ ] **设计文档/spec 模板**：任何输出设计文档/spec 的场景（包括但不限于 superpowers brainstorming skill、用户直接要求写设计文档、输出设计方案），必须先读取 `docs/guidelines/design_document_template.md`，按模板格式输出并覆盖每个章节；即使 superpowers skill 有自己的格式要求，也要以本模板为准。

检查结果必须使用以下格式输出：

```markdown
### 代码检视检查结果
- [x] 编码红线：已检查 docs/guidelines/编码红线.md，未发现违反项
- [x] 跨特性交叉影响：已按 cross_feature_check.md 检查，变更涉及 {模块/目录}，结论为 {结论}
- [x] 贡献规范：已检查 CONTRIBUTING.md，commit/PR/Issue 要求为 {结论}
- [x] 代码格式：已按 .clang-format 检查相关 C++ 文件
- [x] Pre-commit：已确认 clang-format 和 OAT 检查要求
```

## 审查流程

### 步骤 1：确认审查对象

明确审查范围：

- 本地未提交变更：使用 `git diff`、`git diff --stat` 和相关文件内容。
- 指定 commit 范围：使用 `git diff <base>...<head>`。
- GitCode PR：使用 `gitcode-pr` skill 获取 PR 信息、文件列表、diff 和评论；不要使用网页抓取。

如果 PR 已关闭、是草稿、明显不需要审查，或用户只要求解释代码而不是 review，先说明判断并询问是否继续。

### 步骤 2：列出规范上下文

输出本次审查实际使用的规范文件路径列表，至少包含：

- `AGENTS.md`
- `CONTRIBUTING.md`
- `docs/guidelines/编码红线.md`
- `docs/guidelines/cross_feature_check.md`
- `.clang-format`
- `.pre-commit-config.yaml`
- 变更文件所在目录的 README、developer guide 或测试指南（如存在）

### 步骤 3：获取变更摘要

说明变更涉及的文件、模块和意图。每个子任务或子 agent 都必须知道 PR 标题、描述和变更摘要，避免脱离作者意图审查。

### 步骤 4：逐类审查

#### 规范合规性

检查变更是否违反项目规范，尤其是：

- 编码红线中的禁止项
- `.clang-format` 和 Google 代码风格
- 贡献流程、commit message 和 PR 模板
- 新增需求是否有 Issue 或设计说明

#### 功能正确性

只标记明确可证明的问题：

- 语法错误、类型错误、缺少 include/import、未解析符号
- 无论输入如何都会产生错误结果的逻辑错误
- 错误码或异常路径明显错误
- 图改写丢失数据边或控制边
- 异步 runtime 调用后的内存生命周期不满足要求

#### 兼容性和交付

重点检查：

- C/C++ ABI/API、Python 绑定、脚本参数、配置项是否兼容
- CMake、`build.sh`、打包内容、安装路径是否被破坏
- `--no-autofuse`、增量构建、离线依赖场景是否受影响

#### 测试覆盖

检查变更是否有对应测试：

- Bug 修复必须有复现用例。
- 新功能必须覆盖正向、异常和边界场景。
- SuperKernel Python 变更优先 pytest；C++/AOT 变更优先 gtest/RDV。
- Autofuse optimize/codegen 变更优先 UT + E2E ST。

#### Autofuse 模板语义命名

仅在上述模板命名变更范围内检查；具体格式、词表和边界以 Codegen/Schedule 规范为准：

- 分类或继承理由是否明确，是否在实际策略选择、成功转换和最终 API 选择点采集，复制/失败/回退后是否仍反映最终实现。
- 是否保持当前调用、候选及完整逻辑身份，组级 role 是否被父策略覆盖，reuse 是否串组，快照是否与最终 data 同步选择。
- 是否复用公共词表和规范化入口，是否准确区分合理父名/`Adaptive`、`ambiguous` 和 `unavailable`，而非掩盖接入遗漏。
- 测试是否验证实际分支、完整身份、名称和状态，以及本次受影响的边界；PR 是否分开记录名称适配、设备执行、精度、性能和未执行项。
- INFO 关闭时是否避免新增诊断构造，诊断失败是否影响真实求解，原编译、tiling 和 kernel 性能 gate 是否保持。自动接入 CI 尚未实施，不能将现有 CI 通过视为命名适配已验证。

#### 性能与日志

检查新增循环、图遍历、字符串处理、日志、内存拷贝和 runtime 调度是否可能带来明显退化。高频路径新增默认开启日志应视为风险。

### 步骤 5：验证每个问题

对每个发现项再次验证：

- 问题是否只由本次变更引入。
- 是否有明确规范或代码证据。
- 行号是否准确。
- 修复建议是否能完全解决问题。

不能验证的问题从 findings 中移除，最多作为“疑问/建议”单独列出。

### 步骤 6：输出审查结果

如果发现问题，按以下格式输出：

```markdown
### Findings
1. [严重程度] `path/to/file.cc:123` 问题标题
   说明：...
   依据：引用具体规范或代码事实。
   建议：...

### 代码检视检查结果
...

### 疑问
- ...

### 验证
- 已运行：...
- 未运行：...（原因）
```

如果未发现问题：

```markdown
未发现问题。已检查 Bug、规范合规性、跨特性交叉影响和测试覆盖。

### 代码检视检查结果
...

### 残余风险
- ...
```

## PR 提交前自查清单

### 1. 代码规范

- [ ] 代码符合 Google 开源代码规范和项目 `.clang-format`。
- [ ] C++ 代码已通过 `clang-format` 格式化。
- [ ] `if` / `for` / `while` / `do-while` 使用大括号。
- [ ] 无硬编码密钥、账号、公网地址、芯片类型或框架类型判断。
- [ ] 资源申请、释放和异常分支完整。
- [ ] 图改写保持数据边和控制边等价。

### 2. 代码格式化

项目使用 `.clang-format`，关键规则：

| 规则 | 值 |
|------|-----|
| 标准 | C++11 |
| 缩进宽度 | 4 空格 |
| 行宽限制 | 120 字符 |
| 大括号风格 | Custom（函数定义后换行） |
| 指针对齐 | Left |
| Tab | 不使用（UseTab: Never） |
| 命名空间缩进 | None |
| 排序 includes | 不自动排序 |

```bash
clang-format -i path/to/file.cpp
pre-commit run --all-files
```

### 3. Commit Message 规范

格式：`<类型>: <简短描述>`

| 类型 | 说明 | 示例 |
|------|------|------|
| feat | 新功能 | feat: 添加用户注册功能 |
| fix | 修复 bug | fix: 修复登录态过期问题 |
| docs | 文档更新 | docs: 更新 API 使用说明 |
| style | 代码格式调整 | style: 调整代码缩进 |
| refactor | 重构 | refactor: 优化服务类结构 |
| perf | 性能优化 | perf: 减少查询次数 |
| test | 测试相关 | test: 添加登录功能单元测试 |
| chore | 构建/工具链 | chore: 更新配置 |
| ci | CI 配置 | ci: 添加自动化测试流程 |

### 4. PR 模板

提交 PR 时按 `.gitcode/PULL_REQUEST_TEMPLATE.zh-CN.md` 填写：

- 描述：业务背景、目的和方案。
- 变更类型：Bug 修复 / 新功能 / 代码风格 / 重构 / 构建 / 文档。
- 关联 Issue：涉及新增特性、新接口、新配置或流程变更时必须先讨论。
- 如何测试：列出实际执行命令和结果。
- 核对清单：确认代码风格、自测、文档、commit 规范。

### 5. Pre-commit 检查

项目配置了 `.pre-commit-config.yaml`：

1. `clang-format` (v16.0.0)：自动格式化 C/C++ 代码。
2. `OAT Compliance Check`：开源审计检查。

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
pre-commit run oat-check --all-files
```

## 常见高风险问题

- 缺少错误处理或错误码被吞掉。
- 外部输入未校验即作为长度、索引或偏移。
- 资源异常分支未释放。
- 图改写只考虑数据边，忽略控制边。
- 使用无序容器遍历结果决定图结构。
- 新增 pass 依赖顺序但未说明。
- runtime 异步拷贝后释放 host/device 内存。
- 修改 CMake 或打包逻辑但未验证 `build.sh --pkg -j 8`。
- 新功能或 bug 修复无对应 UT/ST。

## 贡献场景

| 场景 | 流程 |
|------|------|
| Bug 修复 | 新建 `Bug-Report` Issue → `/assign` → 复现用例 → 修复 → 提交 PR |
| 新功能 | 新建 `Requirement` Issue → 方案讨论 → 设计文档 → 实现和测试 → 提交 PR |
| 文档纠错 | 新建 `Documentation` Issue → `/assign` → 修复 → 提交 PR |
| 帮助他人 | 在 Issue 中评论交流 → `/assign` → 协助解决 |
