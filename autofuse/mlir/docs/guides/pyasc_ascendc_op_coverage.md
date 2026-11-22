# PyAsc AscendC 算子覆盖矩阵

> 文档类型：能力参考

## 范围

本矩阵记录 PyAsc `ascendc` / `emitasc` 是否能表达 Phase 2 需要迁移的 Autofuse codegen 算子族。
当前结论基于 PyAsc TableGen `.td` 源文件提取出的 op inventory；具体语义、tiling、shape、
buffer 生命周期和代码生成稳定性仍需在后续纵向切片中验证。

## 扫描原理

`scan_pyasc_coverage.py` 递归扫描 PyAsc 的 `include/ascir/Dialect/Asc/IR/**/*.td` 和
`include/ascir/Dialect/EmitAsc/IR/**/*.td`：

- 对显式 `def ... : SomeOp<"..."> { let arguments/results = ... }`，直接提取 op 名、参数、结果和 TableGen 类型约束。
- 对 `def ... : SomeBaseOp<"...">;`，从对应 `class SomeBaseOp` 继承 `arguments/results`。
- 对 PyAsc 常见的 `defm` / `multiclass`，做轻量展开，覆盖 `baseMnemonic # "_l0"` 这类字符串拼接，生成 `ascendc.add_l0` 等 op。
- 输出中的 `constraint` / `type_hints` 是 MLIR TableGen 类型约束，不等同于运行时 dtype 白名单。

脚本不做完整 TableGen 语义解释；如果后续遇到复杂条件、复杂 DAG 或 generated `.inc` 才能确定的信息，需要再接入
`mlir-tblgen` 生成物做交叉校验。

## 初始矩阵

| Autofuse 算子族 | PyAsc 支持情况 | 决策 | 说明 |
|---|---|---|---|
| elewise | 当前扫描提取 170 个相关 op | 作为首条候选切片 | 覆盖首条纵向切片，仍需验证 dtype、broadcast 和 scalar 变体 |
| brc | 当前扫描提取 2 个相关 op | 作为首条候选切片 | 适合和 elewise 一起验证基础广播表达 |
| reduce | 当前扫描提取 28 个相关 op | tiling/state 支持不完整时后置 | 需要验证 reduce axis、workspace、tiling 和同步 |
| concat | **0 个**：PyAsc 方言里没有 data move / concat 供给 | 后置 | 原先记的 18 个全部来自 `OpVecDataMoveExt.td`，那个文件从未被 `Ops.td` include，2026-09-24 随 L-17 删除。要接 concat 得先在 ODS 里建模 |
| gather | 当前扫描提取 6 个相关 op | index/shape 语义确认后迁移 | 对 index 边界、mask 和动态 shape 敏感 |
| transpose | 当前扫描提取 6 个相关 op | layout 表达确认后迁移 | 对 layout、stride、连续性和 workspace 敏感 |
| datacopy | 当前扫描提取 22 个相关 op | 可作为内存搬运验证切片 | 对 GM/UB 地址空间、对齐和 pipeline 生命周期敏感 |
| cube | 当前扫描提取 58 个相关 op | 不放入首条切片 | 编译、tiling、workspace 和执行风险更高 |

当前脚本总计提取 613 个 PyAsc `ascendc` / `emitasc` op。其中 292 个映射到上表 7 个有供给的
Autofuse 候选 family（`concat` 一个都没有），另外 321 个归为 `other`，表示暂未映射到 Phase 2
首批候选 family 的 PyAsc op。

## 扫描命令

```bash
python3 autofuse/mlir/tools/af-mlir-tools/scan_pyasc_coverage.py
python3 autofuse/mlir/tools/af-mlir-tools/scan_pyasc_coverage.py --format json > /tmp/pyasc_ops.json
python3 autofuse/mlir/tools/af-mlir-tools/scan_pyasc_coverage.py --format csv --output autofuse/mlir/docs/guides/pyasc_ascendc_op_inventory.csv
```

`--check` 把生成结果和仓内快照比对，不一致即非零退出并打印 diff：

```bash
python3 autofuse/mlir/tools/af-mlir-tools/scan_pyasc_coverage.py \
    --format csv --check --output autofuse/mlir/docs/guides/pyasc_ascendc_op_inventory.csv
```

这条已经挂进 pre-commit（hook id `pyasc-op-inventory`）：**改完 `api/ascendc/` 下的 `.td` 必须重新生成
CSV 并一起提交**，否则提交被拦。该 hook 用 `always_run` 而不是 `files`——触发源 `api/ascendc/`
和产物 `.csv` 都在仓库级 `exclude` 里，挂 `files` 会永远 Skipped；整轮扫描 0.1s。

Markdown 输出包含 family summary 和完整 op inventory。CSV 快照保存在
`autofuse/mlir/docs/guides/pyasc_ascendc_op_inventory.csv`，便于表格工具筛选。JSON 输出面向后续自动化消费。
主要字段包括：

- `op_name`：如 `ascendc.data_copy_l0`、`ascendc.add_l0`、`emitasc.call_opaque`。
- `td_symbol`：TableGen def 或 defm 展开后的 symbol。
- `arguments` / `results`：每项包含 `name`、`constraint`、`type_hints`、`raw`。
- `param_type_lists`：PyAsc emitter 使用的参数模板映射信息。
- `status`：`explicit-def`、`inherited-class`、`expanded-defm` 或 `no-signature`。

该脚本输出的是候选证据和类型约束，不替代人工语义评审。

## 已知偏差

快照按 `.td` 文本提取，不跑 TableGen，因此有三处已知失真，读数时要扣掉：

- **`_reg` 后缀丢失**：`class UnaryRegOp` / `BinaryRegOp` / `VecScalarRegOp` 把 `baseMnemonic # "_reg"`
  传给 `RegOp`，脚本只取 `def` 那一层的实参，不跟到 class 基类去拼。于是 `def AbsRegOp : UnaryRegOp<"abs", "Abs">`
  被记成 `ascendc.abs`，真实 mnemonic 是 `ascendc.abs_reg`。`OpReg.td` 里 31 个 op 受影响，
  其中 `exp` / `log` / `xor` 还因此和 `Adv/Math.td` 的同名真 op 撞成重复行。直接写 `RegOp<"...">`
  的（如 `reduce_max_reg`）不受影响。
- ~~**扫了没进方言的文件**~~ **已消失（2026-09-24）**：7 个从未被 `Ops.td` include 的 `*Ext.td`
  随 L-17 删除，它们贡献的 `broadcast` / `concat` / `reduce_prod` 三组重复行一并消失，concat
  family 也随之归零。`check_ascendc_td_reachable.py` 的豁免表现在只剩 `Doc.td`。FAMILIES 表里
  仍引用着一个从来不存在的 `OpVecUnaryExt.td`，不影响结果。
- **`AnyType: $scalar` 取不到名字**：`Base.td` 的 `VecScalarRegOp` 冒号后多一个空格，脚本的
  `:\$name` 正则匹配不上，该参数名记成 `?`。TableGen 本身接受这种写法，不影响构建。

剩下两条都是提取侧的问题，不是 ODS 缺陷；修它们会大幅改写快照，留待单独处理。

曾经还有第四条，**已随 L-17 的子树拆分消失**：`Base.td` 里 autofuse 加的 `ApiExtensionHeader`
带一个 `code decl = [{ ... }]`，脚本的括号配平在那里断掉，`Base.td` 后面 35 个 multiclass 就
全都认不出来了——`defm Abs : UnaryL012Op<...>` 只记成一行 `ascendc.abs`（status
`template-instance`），而不是展开成 `abs_l0` / `abs_l1` / `abs_l2`。这一条没人知道，因为它的
表现是「少了几十行」而不是报错。定制内容移进 `Autofuse/` 之后 `Base.td` 恢复上游原样，快照
从 581 行涨到 654 行，涨的全是这批本来就该有的展开行（删掉 7 个死文件后回落到 654 → 613）。**`Autofuse/Base.td` 里那个 `code` 块
仍然会把该文件的解析打断，只是那个文件里没有 multiclass，没有东西可丢。**
