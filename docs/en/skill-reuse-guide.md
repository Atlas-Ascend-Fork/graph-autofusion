# Graph-autofusion Skill Reuse Guide (for External Developers Without cannbot)

## Overview

This document is for **external developers without cannbot**, explaining how to reuse the Skills built into the graph-autofusion project: either integrate them into other AI tools (Claude Code / Cursor / GitHub Copilot, etc.) for automatic loading, or read them as a manual development guide.

The project ships 30 local Skills: 9 general development Skills, 7 SuperKernel operator pipeline Skills, and 14 SuperKernel auto-tuning and analysis Skills. Another 4 remote GitCode collaboration Skills require separate installation.

### Relationship with Other Documents

| Document | Purpose | Audience |
|----------|---------|----------|
| [opencode-skill-management.md](../zh/opencode-skill-management.md) | Three-layer Skill architecture, management, contribution | Skill maintainers, contributors |
| **This document** | **How to reuse built-in Skills without cannbot** | **External developers (integrating with other AI tools or manual reading)** |

## Skill Overview

### General Development Skills (9, git-tracked)

| Skill | Purpose | Script Dependency | SKILL.md |
|-------|---------|-------------------|----------|
| `af-build-runner` | Build assistant (`build.sh` args, CMake errors, dependency matching) | None | [Link](../../.claude/skills/af-build-runner/SKILL.md) |
| `af-test-developer` | UT/ST test development (gtest/mockcpp, pytest, coverage) | None | [Link](../../.claude/skills/af-test-developer/SKILL.md) |
| `af-code-reviewer` | Code review and contribution standards (redlines, format, PR template) | None | [Link](../../.claude/skills/af-code-reviewer/SKILL.md) |
| `af-reg-ascir` | ASCIR registration assistant (add/modify ops, dtype, tmp buffer, UT/ST gen) | None | [Link](../../.claude/skills/af-reg-ascir/SKILL.md) |
| `cann-toolkit-installer` | Auto-download and install CANN Toolkit (parsing, verify, silent install) | Embedded bash logic | [Link](../../.claude/skills/cann-toolkit-installer/SKILL.md) |
| `default-skills` | Default remote Skill installation entry | `scripts/install-default-skills.sh` | [Link](../../.claude/skills/default-skills/SKILL.md) |
| `af-inductor-cv-validator` | Inductor CV model validation matrix | `scripts/run_inductor_cv_matrix.py` | [Link](../../.claude/skills/af-inductor-cv-validator/SKILL.md) |
| `af-perf-modeler` | ATT performance formulas and codegen parameter modeling | None | [Link](../../.claude/skills/af-perf-modeler/SKILL.md) |
| `af-prune-graph-metadef` | Graph/MetaDef pruning and blacklist checks | `scripts/check-blacklist.sh` | [Link](../../.claude/skills/af-prune-graph-metadef/SKILL.md) |

### SuperKernel Operator Pipeline Skills (7, git-tracked)

| Skill | Purpose | Script Dependency | SKILL.md |
|-------|---------|-------------------|----------|
| `sk-operator-pipeline` | SK operator delivery pipeline entry (routing, index) | `scripts/` | [Link](../../.claude/skills/sk-operator-pipeline/SKILL.md) |
| `sk-operator-asset-adapter` | Operator asset adaptation (user dir → JSON contract) | `scripts/*.py` | [Link](../../.claude/skills/sk-operator-asset-adapter/SKILL.md) |
| `sk-operator-validate` | Asset contract validation (contract, source, compatibility) | `scripts/` | [Link](../../.claude/skills/sk-operator-validate/SKILL.md) |
| `sk-operator-codegen` | SK binding code generation (Args struct + `__sk__` + SK_BIND) | `scripts/` | [Link](../../.claude/skills/sk-operator-codegen/SKILL.md) |
| `sk-operator-sample-gen` | Sample generation and validation contract (input, oracle, runner) | `scripts/` | [Link](../../.claude/skills/sk-operator-sample-gen/SKILL.md) |
| `sk-operator-build-package` | SK/ACLGraph build and package (bisheng compile → wheel) | `scripts/*.py` | [Link](../../.claude/skills/sk-operator-build-package/SKILL.md) |
| `sk-model-analysis` | Full-network diagnosis (hang/coredump, perf analysis, visualization) | `scripts/*.py` | [Link](../../.claude/skills/sk-model-analysis/SKILL.md) |

### SuperKernel Auto-Tuning and Analysis Skills (14, git-tracked)

These Skills were moved from `superkernel-skill` into `.claude/skills/`. They cover model-level fusion tuning and profiling analysis, separately from the operator delivery pipeline.

| Skill | Purpose | SKILL.md |
|-------|---------|----------|
| `superkernel-auto-tune` | Full-lifecycle tuning controller and phase dispatch | [Link](../../.claude/skills/superkernel-auto-tune/SKILL.md) |
| `superkernel-intake-preparation` | Freeze runtime, workload, controls, and optional experiment intent | [Link](../../.claude/skills/superkernel-intake-preparation/SKILL.md) |
| `superkernel-s0-baseline` | Establish the five-process SK-off clean performance baseline | [Link](../../.claude/skills/superkernel-s0-baseline/SKILL.md) |
| `superkernel-stage-a-scope-selection` | Screen the full scope-strategy matrix and select a candidate | [Link](../../.claude/skills/superkernel-stage-a-scope-selection/SKILL.md) |
| `superkernel-stage-o-option-tuning` | Incrementally tune wrapper options for the frozen Stage A winner | [Link](../../.claude/skills/superkernel-stage-o-option-tuning/SKILL.md) |
| `superkernel-base-profile-source-mapping` | Collect BASE profiles, analyze each SK, and verify exact source mapping | [Link](../../.claude/skills/superkernel-base-profile-source-mapping/SKILL.md) |
| `superkernel-optional-experiments` | Run user-authorized multistream and P/FINAL source-range experiments | [Link](../../.claude/skills/superkernel-optional-experiments/SKILL.md) |
| `superkernel-final-e2e-report` | Final clean E2E validation and Chinese tuning report | [Link](../../.claude/skills/superkernel-final-e2e-report/SKILL.md) |
| `superkernel-source-range-from-smap` | Derive independent P/FINAL experiments from a completed session with verified BASE/SMAP | [Link](../../.claude/skills/superkernel-source-range-from-smap/SKILL.md) |
| `superkernel-fusion-performance-analysis` | Compare SK-off/SK-on performance, attribute regressions, and analyze source mapping | [Link](../../.claude/skills/superkernel-fusion-performance-analysis/SKILL.md) |
| `superkernel-multistream-performance-tuning` | Tune fused multistream overlap and scheduling using fresh profiles | [Link](../../.claude/skills/superkernel-multistream-performance-tuning/SKILL.md) |
| `superkernel-sk-failure-isolation` | Isolate execution, correctness, timeout, or hang failures using plog/sk_meta | [Link](../../.claude/skills/superkernel-sk-failure-isolation/SKILL.md) |
| `superkernel-sk-prof-timeline` | Compact per-core sk_prof traces into Cube/Vector timelines | [Link](../../.claude/skills/superkernel-sk-prof-timeline/SKILL.md) |
| `superkernel-runtime-common` | Internal shared runtime for analysis, device leases, lifecycle, ledger, and reporting | [Link](../../.claude/skills/superkernel-runtime-common/SKILL.md) |

### Remote GitCode Collaboration Skills (4, auto-installed)

> These 4 Skills are NOT in git; install first (see "Reuse Notes - Remote Skill Installation").

| Skill | Purpose | Trigger Scenarios |
|-------|---------|-------------------|
| `gitcode-pr` | Create PR, fetch comments, view discussions | Create PR, view PR changes, get PR comments |
| `gitcode-issue` | Read issue details and comments | View issue, read issue comments |
| `gitcode-pipeline` | Trigger pipeline and monitor status | Trigger pipeline, watch CI, view pipeline status |
| `api-doc-generator` | Generate API documentation | Generate API docs, add API descriptions |

## Reuse Method 1: Integrate with Other AI Tools

### General Principle

Each Skill centers on a `SKILL.md` file with two parts:

1. **Frontmatter (YAML metadata)**: declares `name` and `description`; AI tools use this to decide when to load the Skill.
2. **Body (Markdown instructions)**: describes functionality, steps, constraints; serves as execution instructions for the AI assistant.

```markdown
---
name: skill-name
description: |
  Brief description of functionality.
  **Required trigger scenarios**: list keywords and trigger scenarios.
---

## Functionality
... (body, AI tools execute accordingly)
```

The core idea for integration: **convert `SKILL.md` to the target tool's instruction file format**, or use it directly as a system prompt fragment.

### Claude Code

Claude Code natively supports the `.claude/skills/` path convention; **no conversion needed**.

```bash
git clone https://gitcode.com/cann/graph-autofusion.git
cd graph-autofusion
# Open this directory with Claude Code; it auto-loads .claude/skills/*/SKILL.md
claude
```

Applicable scenario: developers using Claude Code. Note that the 4 remote Skills require running `default-skills` installation first (see Notes).

### Cursor

Cursor uses `.cursor/rules/*.mdc` files as instructions. Conversion notes:

- SKILL.md `description` → Cursor rule frontmatter (`description` + `globs`)
- SKILL.md body → Cursor rule body, reused directly
- One SKILL.md maps to one `.mdc` file

```cursor-rule
---
description: graph-autofusion build assistant (build.sh, CMake, dependency matching)
globs: ["build.sh", "CMakeLists.txt", "cmake/**"]
alwaysApply: false
---
(paste af-build-runner/SKILL.md body here)
```

Applicable scenario: developers using Cursor. Set `globs` to the file types each Skill targets, enabling on-demand triggering.

### GitHub Copilot

GitHub Copilot supports two approaches:

**Approach 1: Single aggregated file** (`.github/copilot-instructions.md`)

Aggregate key content from multiple SKILL.md files into one file; suitable when few Skills are used.

**Approach 2: Multi-file split** (`.github/instructions/*.instructions.md`, requires VS Code 1.100+)

Each SKILL.md maps to one `.instructions.md` file with `applyTo` frontmatter:

```github-instruction
---
applyTo: "build.sh,CMakeLists.txt,cmake/**"
---
(paste af-build-runner/SKILL.md body here)
```

Applicable scenario: developers using GitHub Copilot. Approach 2 is recommended for one-to-one mapping with Skills, easing maintenance.

### General Method (AI tools without instruction file support)

For AI tools that do not support instruction file loading (e.g., web-based ChatGPT, Qwen), **paste the required SKILL.md content as a system prompt fragment**:

1. Identify the relevant Skill for your current task (see "Skill Overview").
2. Read the full `.claude/skills/<name>/SKILL.md`.
3. Paste at the start of your AI tool's system prompt or conversation: "Please assist me with graph-autofusion development per the following instructions:\n\n{SKILL.md body}".
4. Select on demand; avoid pasting all Skills at once (may exceed context window).

## Reuse Method 2: Read as a Manual Development Guide

Without AI tools, SKILL.md files still contain manually readable development value (command references, checklists, process descriptions).

### General Skill Quick Reference

| Skill | Manually Readable Value | Key Points |
|-------|------------------------|------------|
| `af-build-runner` | Build command reference, incremental build strategy, common errors | All build commands must use `-j 8` to limit parallelism, avoiding OOM |
| `af-test-developer` | Test module table, run commands, coverage generation | `autofuse_e2e` supports only ST (`-s`), not UT (`-u`) |
| `af-code-reviewer` | PR self-check checklist, commit message format, high-risk issues | C++ 4-space indent, line width 120; commit format `<type>: <short description>` |
| `cann-toolkit-installer` | Toolkit install args, process, dependencies | Default version 9.1.0, auto-detected architecture, ~10 min install |
| `default-skills` | Remote Skill installation entry | Triggers `install-default-skills.sh` to pull from gitcode |

For details, read the corresponding [SKILL.md](../../.claude/skills/af-build-runner/SKILL.md) directly.

### SK Operator Pipeline Stage Quick Reference

The SK operator delivery pipeline executes in the following stage order; each stage maps to a Skill:

| Stage | Skill | Purpose |
|-------|-------|---------|
| Entry | `sk-operator-pipeline` | Run customizable asset adapter, validate contracts, dispatch stage Skills |
| 1. Asset adaptation | `sk-operator-asset-adapter` | Convert user operator repo/source tree/build assets to stable JSON contract |
| 2. Validation | `sk-operator-validate` | Validate contract, source structure, compatibility; output findings |
| 3. Code generation | `sk-operator-codegen` | Generate SK binding (Args struct + `__sk__` template + SK_BIND) |
| 4. Sample generation | `sk-operator-sample-gen` | Build run inputs, oracle, runner, differential contract |
| 5. Build and package | `sk-operator-build-package` | Invoke bisheng to compile SK/ACLGraph extension, package as wheel |
| Diagnosis | `sk-model-analysis` | Full-network diagnosis: hang/coredump location, perf analysis, scope/task visualization |

When reading manually, start from `sk-operator-pipeline`'s SKILL.md for the overall flow.

### SuperKernel Auto-Tuning Workflow Quick Reference

Start with [superkernel-auto-tune](../../.claude/skills/superkernel-auto-tune/SKILL.md):

```text
Intake -> S0 -> Stage A -> Stage O -> BASE profiling / analysis / SMAP
       -> [multistream and/or P/FINAL] -> final clean E2E / report
```

Intake freezes the optional mode as `none`, `multistream`, `source-range`, or `both`. Optional experiments require user authorization; failed or non-improving branches preserve the incumbent. Final decisions use clean E2E latency. For a completed and verified parent session, `superkernel-source-range-from-smap` derives an independent P/FINAL experiment. `superkernel-runtime-common` is a shared dependency, not a user entry point.

### Remote Skill Trigger Scenario Quick Reference

| Skill | Trigger Scenarios |
|-------|-------------------|
| `gitcode-pr` | Create PR, push to remote, view PR changes, get PR comments |
| `gitcode-issue` | View issue details, read issue comments |
| `gitcode-pipeline` | Trigger pipeline, view pipeline status, wait for pipeline result |
| `api-doc-generator` | Generate API docs, add API descriptions |

## Reuse Notes

### Script Dependencies

- **7 `sk-*` Skills** contain Python scripts (`scripts/*.py`); AI tool-triggered execution requires **Python 3.9+**. Manual reading does not require executing scripts; only understanding the workflow described in SKILL.md.
- **`cann-toolkit-installer`** embeds bash logic (download, verify, install); execution requires **bash >= 5.1.16**.
- **`default-skills`**'s `scripts/install-default-skills.sh` installs remote Skills; requires network access to gitcode.com.

In manual reading scenarios, scripts are for reference only and are not required to run.

### Reusing the Complete SuperKernel Tuning Bundle

- When reusing across projects, keep all 14 `.claude/skills/superkernel-*` directories under one skills root, including their existing `scripts/`, `references/`, `schemas/`, `phase.json`, and `agents/` resources. Phase scripts load the controller and shared runtime through sibling-relative paths; copying only `SKILL.md` is insufficient.
- When converting instructions for another tool, retain these resources and resolve script paths against each Skill directory. `agents/openai.yaml` is optional UI metadata; it does not register custom phase Agents.
- Full tuning requires host-native generic subagents or a command adapter that explicitly loads each phase Skill. Pasted prompts support reading and analysis but do not replace the execution environment.
- Execution requires Python 3. NPU stages also require Ascend/CANN, model dependencies, and profiling tools appropriate to the workload. Follow each Skill and its environment preflight for exact dependencies. Reading the documentation does not require an NPU.

### Remote Skill Installation

The 4 remote Skills (`gitcode-pr`, `gitcode-issue`, `gitcode-pipeline`, `api-doc-generator`) are not in git; install via:

**Method 1: Trigger `default-skills` via cannbot** (with cannbot environment)

Enter "install default skills" in opencode; it auto-runs `install-default-skills.sh`.

**Method 2: Manual installation** (without cannbot environment)

```bash
# Clone the remote skill repo
git clone --depth 1 https://gitcode.com/cann-agent/skills.git /tmp/cann-skills

# Copy to the project's _remote directory
mkdir -p .claude/skills/_remote
cp -r /tmp/cann-skills/gitcode-pr .claude/skills/_remote/
cp -r /tmp/cann-skills/gitcode-issue .claude/skills/_remote/
cp -r /tmp/cann-skills/gitcode-pipeline .claude/skills/_remote/
cp -r /tmp/cann-skills/api-doc-generator .claude/skills/_remote/

# Create symlinks to the top-level directory
ln -sf _remote/gitcode-pr .claude/skills/gitcode-pr
ln -sf _remote/gitcode-issue .claude/skills/gitcode-issue
ln -sf _remote/gitcode-pipeline .claude/skills/gitcode-pipeline
ln -sf _remote/api-doc-generator .claude/skills/api-doc-generator

# Clean up temp directory
rm -rf /tmp/cann-skills
```

After installation, 4 symlinks appear in `.claude/skills/` top level, discoverable by AI tools.

### `.gitignore` Whitelist Rule

`.claude/skills/.gitignore` uses an "**ignore by default + whitelist**" strategy:

```gitignore
*
!af-build-runner/
!af-build-runner/**
... (remaining local Skill whitelist entries)
!.gitignore
```

Reusing existing Skills is unaffected. **To add a new custom Skill**, add a whitelist entry to `.gitignore`:

```gitignore
!my-skill/
!my-skill/**
```

See [opencode-skill-management.md](../zh/opencode-skill-management.md) "`.gitignore` Configuration" section for details.

### Third-Party Plugin Skills Out of Scope

Third-party plugin Skills like `superpowers` are configured by `.opencode/opencode.json`, located in `~/.cache/opencode/`, and are not project-built-in. This document does not provide reuse guidance for them. See [opencode-skill-management.md](../zh/opencode-skill-management.md) "Third Layer: Third-Party Plugin Skills" section.

## FAQ

**Q: What is a SKILL.md?**

A: A built-in AI assistant instruction file at `.claude/skills/<name>/SKILL.md`, consisting of frontmatter (`name`/`description`) and body (functionality, steps, constraints). cannbot auto-loads and triggers by keyword on startup; without cannbot, it can be integrated into other AI tools or read manually.

**Q: Can I use these Skills without cannbot?**

A: Yes. Two methods: 1) Integrate SKILL.md into Claude Code / Cursor / GitHub Copilot and other AI tools (see "Reuse Method 1"); 2) Read as a manual development guide (see "Reuse Method 2").

**Q: What if a Skill doesn't trigger after integrating with Cursor?**

A: Check the `.cursor/rules/*.mdc` frontmatter: whether `description` includes relevant keywords, whether `globs` matches the files being edited, whether `alwaysApply` should be `true`. Refer to Cursor's official documentation.

**Q: Do I need to manually run the `sk-*` Skill scripts?**

A: No. In manual reading scenarios, scripts are for reference only; understanding the workflow in SKILL.md is sufficient. AI tools execute them on demand, requiring Python 3.9+.

**Q: What's the difference between remote and local Skills?**

A: Local Skills (30) are git-tracked; `git clone` gets them. Remote Skills (4) are not in git; install via `default-skills` or manual clone to `.claude/skills/_remote/`. See [opencode-skill-management.md](../zh/opencode-skill-management.md).

**Q: How do I contribute a new Skill?**

A: See the "Add a Local Skill" section of [opencode-skill-management.md](../zh/opencode-skill-management.md): create a modular Skill package → write `SKILL.md` + `README.md` → add whitelist entry to `.gitignore` → submit a PR.
