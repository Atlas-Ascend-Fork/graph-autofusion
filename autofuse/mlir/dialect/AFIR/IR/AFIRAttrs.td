// -----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------

//===- AFIRAttrs.td - AFIR attribute definitions -----------*- tablegen -*-===//
//
// Attributes for the AFIR dialect. These carry the schedule/tensor semantics
// that the schedule-result bridge needs to preserve (axis, position/mem,
// vectorization, graph-level tiling info).
//
//===----------------------------------------------------------------------===//

#ifndef AFIR_ATTRS
#define AFIR_ATTRS

include "AFIRAttrTypeBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/EnumAttr.td"
include "AFIREnums.td"

//===----------------------------------------------------------------------===//
// Function role attribute
//===----------------------------------------------------------------------===//

def FunctionRoleAttr : AFIR_Attr<"FunctionRole", "function_role"> {
  let summary = "Role of an AFIR-containing function";
  let parameters = (ins "::mlir::afir::FunctionRole":$value);
  let assemblyFormat = "`<` $value `>`";
}

//===----------------------------------------------------------------------===//
// Pipe: which asynchronous engine work runs on
//===----------------------------------------------------------------------===//
//
// `AFIR_PipeEnum` 的属性形态，供 `afir.pipe_barrier` 一类以 pipe 为参数的 Op 使用。
// 语义与取值的说明在枚举定义处（AFIREnums.td）。
def PipeAttr : EnumAttr<AFIR_Dialect, AFIR_PipeEnum, "pipe">;

//===----------------------------------------------------------------------===//
// Kernel argument classification
//===----------------------------------------------------------------------===//

def KernelArgAttr : AFIR_Attr<"KernelArg", "kernel_arg"> {
  let summary = "Role of one kernel argument in the AFIR kernel ABI";
  let description = [{
    挂在 `func.func` 的 arg attr 上，声明这个参数在 kernel ABI 里是什么。设计文档
    3.6 把 kernel_arg 分类列为「Kernel 程序容器」family 的缺口。

    它替代了此前三套各管一段的机制：

    | 此前 | 形态 | 失效方式 |
    |---|---|---|
    | `afir.input_index` / `afir.output_index` | 一对互斥的整数 arg attr | 「两个都挂」是合法 IR，只能在消费端事后发现；「都不挂」则退化成按类型猜 |
    | `afir.num_inputs` | 函数属性，记数据参数的个数 | 同一事实的第二份编码（D2）：角色已经逐参数写在 arg attr 上，个数还要另存一份，消费端再靠 `take_front(N)` 做位置算术（D3） |
    | workspace / TilingData | 无标记，靠**类型相等**与**结构体名字等于 "TilingData"** 反推 | 参数身份由消费端反推而不是自己声明，每个消费端都要把反推规则重写一遍，且规则不在任何对外文档里 |

    现在每个 kernel 参数自己说明身份，个数与位置都从这组标记数出来，不再另存。

    `index` 是该参数在其角色内部的序号（前端图的第几个输入 / 第几个输出），不是参数
    列表下标——两者在 buffer 形态下并不相等。`workspace` 全 kernel 至多一个，因而没有
    序号，verifier 拒绝给它写 `index`。`tiling` 的序号可选：Plan 的 `_main` 一 stage 收
    一个 tiling 形参，那里的序号是 stage；template 与聚合入口只有一个，不带号。

    **为什么不新造 `afir.kernel` Op**：设计约束 6 是「不重造标准方言」。kernel 需要的
    是符号名、参数表、参数属性、区域与返回——`func.func` / `func.return` 全都有，且
    `SymbolTable`、inliner、`FunctionOpInterface` 一族的现成设施都挂在上面。昇腾特有的
    只有「这个函数是 kernel body 还是 kernel entry」和「每个参数是什么」，它们分别由
    `#afir.function_role` 与本属性表达。
  }];

  let parameters = (ins
    "::mlir::afir::KernelArgRole":$role,
    DefaultValuedParameter<"int64_t", "-1">:$index
  );

  let assemblyFormat = "`<` `role` `=` $role (`,` `index` `=` $index^)? `>`";
  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// Symbolic expression attribute
//===----------------------------------------------------------------------===//
def ExprAttr : AFIR_Attr<"Expr", "expr"> {
  let summary = "Symbolic expression in AFIR canonical string form backed by SymEngine";

  let parameters = (ins
    "StringAttr":$value
  );

  let assemblyFormat = "`<` $value `>`";
  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// Canonical tensor and resource descriptors
//===----------------------------------------------------------------------===//

def RoleAttr : AFIR_Attr<"Role", "role"> {
  let summary = "Canonical role of an AFIR tensor value";
  let parameters = (ins "::mlir::afir::Role":$value);
  let assemblyFormat = "`<` $value `>`";
}

def SpaceAttr : AFIR_Attr<"Space", "space"> {
  let summary = "Canonical memory space of an AFIR buffer";
  let parameters = (ins "::mlir::afir::Space":$value);
  let assemblyFormat = "`<` $value `>`";
}

class AFIR_OptionalInt64Parameter<string description = ""> :
    DefaultValuedParameter<"int64_t", "std::numeric_limits<int64_t>::min()", description>;

def TensorDescAttr : AFIR_Attr<"TensorDesc", "tensor_desc"> {
  let summary = "Canonical tensor and schedule descriptor";
  let parameters = (ins
    OptionalArrayRefParameter<"int64_t">:$axis,
    OptionalArrayRefParameter<"ExprAttr", "array of repeat expressions">:$repeats,
    OptionalArrayRefParameter<"ExprAttr", "array of stride expressions">:$strides,
    OptionalArrayRefParameter<"int64_t">:$vectorized_axis,
    OptionalArrayRefParameter<"ExprAttr", "array of vectorized stride expressions">:$vectorized_strides,
    "RoleAttr":$role
  );
  let assemblyFormat = "`<`"
    "(`axis` `=` `[` $axis^ `]` `,`)?"
    "(`repeats` `=` `[` $repeats^ `]` `,`)?"
    "(`strides` `=` `[` $strides^ `]` `,`)?"
    "(`vectorized_axis` `=` `[` $vectorized_axis^ `]` `,`)?"
    "(`vectorized_strides` `=` `[` $vectorized_strides^ `]` `,`)?"
    "`role` `=` $role"
    "`>`";
  let genVerifyDecl = 1;
}

def BufferDescAttr : AFIR_Attr<"BufferDesc", "buffer_desc"> {
  let summary = "Canonical queue, buffer, temporary and workspace descriptor";
  let description = [{
    一块存储的规划结果。字节大小有**两个互斥**的来源：

    - `size_bytes_expr`：这块存储自己的大小表达式，符号在 TilingData 里解析。GM 与
      workspace 走这条，表达式由 bridge 从张量 repeats 算出。
    - `size_from_resource`：大小取自**这个资源自己**的 canonical 尺寸字段
      （`q<buffer_id>_size` / `b<buffer_id>_size`，由 host 侧 tiling 填）。片上的
      queue 与 UB buffer 走这条。

    片上资源不能用表达式，是因为那个数不是单个张量的字节数：ATT 侧按
    `max(同 id 的每个张量) ⊔ Σ(同 reuse 组)` 聚合，一个描述单个值的 descriptor 表达
    不了「还要跟共享这个 id 的其他张量比大小」。在 ATT 的 UB 预算迁进来之前，MLIR 只能
    引用那个数。
  }];
  let parameters = (ins
    "SpaceAttr":$space,
    DefaultValuedParameter<"int64_t", "-1">:$buffer_id,
    DefaultValuedParameter<"int64_t", "-1">:$reuse_id,
    DefaultValuedParameter<"int64_t", "-1">:$slot_count,
    DefaultValuedParameter<"int64_t", "-1">:$queue_depth,
    OptionalParameter<"ExprAttr">:$size_bytes_expr,
    DefaultValuedParameter<"bool", "false">:$size_from_resource,
    DefaultValuedParameter<"int64_t", "-1">:$alignment_bytes,
    AFIR_OptionalInt64Parameter<"optional lifetime axis id">:$life_time_axis_id
  );
  let assemblyFormat = "`<`"
    "`space` `=` $space `,`"
    "(`buffer_id` `=` $buffer_id^ `,`)?"
    "(`reuse_id` `=` $reuse_id^ `,`)?"
    "(`slot_count` `=` $slot_count^ `,`)?"
    "(`queue_depth` `=` $queue_depth^ `,`)?"
    "(`size_bytes_expr` `=` $size_bytes_expr^ `,`)?"
    "(`size_from_resource` `=` $size_from_resource^ `,`)?"
    "(`alignment_bytes` `=` $alignment_bytes^ `,`)?"
    "(`life_time_axis_id` `=` $life_time_axis_id^)?"
    "`>`";
  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// Memory layout (memref layout attribute)
//===----------------------------------------------------------------------===//

def LayoutAttr : AFIR_Attr<"Layout", "layout",
    [DeclareAttrInterfaceMethods<MemRefLayoutAttrInterface, ["getAffineMap", "verifyLayout"]>]> {
  let summary = "Memory layout of a memref: outer order x intra-tile order x tile extents";
  let description = [{
    `memref` 的 layout 扩展点此前完全没有被使用（D8），NZ / fractal 因此无从表达——这是
    Cube 路线的根因阻塞。本属性按 ADR-16 的**正交分解**填上这个扩展点：

    | 维度 | 取值 | 含义 |
    |---|---|---|
    | `outer`  | `row_major` / `col_major` | 未分块时元素的遍历序；分块时**基块网格**的遍历序 |
    | `tiling` | `none` / `row_major` / `col_major` | 基块**内部**的遍历序；`none` 表示不分块 |
    | `tile`   | 整数数组 | 基块在最后两维上的长度，**单位是元素** |

    布局只作用于**最后两维**，前导维原样透传——双缓冲的 stage 维（`memref<2x64x64xf16>`
    的第 0 维）不参与矩阵解释。

    四种 (outer, tiling) 组合与昇腾的两字母格式名一一对应：

    | outer | tiling | 昇腾格式 |
    |---|---|---|
    | `row_major` | `none` | ND（identity） |
    | `col_major` | `none` | DN / 转置 |
    | `col_major` | `row_major` | `FRACTAL_NZ` |
    | `row_major` | `col_major` | `FRACTAL_ZN` |

    **`rank` 为什么是参数**：`MemRefLayoutAttrInterface::getAffineMap()` 不接收 rank，也
    不允许返回 NULL，而 affine map 的定义域维数必须等于 memref 的 rank（否则 memref 构造
    期即报 "layout mismatch between rank and affine map"）。因此 rank 必须由属性自己携带。
    这份冗余由 `verifyLayout` 逐点核对，不是靠约定——与 `StridedLayoutAttr` 每维存一个
    stride 是同一形态。

    **`tile` 为什么是元素数而不是字节数**：ADR-16 原文把基块尺寸写成「基块字节数」。
    字节数换算成各维长度需要元素宽度，而 `getAffineMap()` 同样拿不到元素类型——换算做不了。
    元素数才是 affine map、stride 计算与搬运描述符共同需要的形式；字节数可由
    `product(tile) * 元素字节宽` 从 memref 反算，另存一份即是同一事实的第二份编码（D2）。

    **分块形态的代价：脱离 MLIR 的 strided 工具链**。分块布局给出的是多结果的
    semi-affine 映射（`floordiv` / `mod`），不是 strided 布局。因此
    `getStridesAndOffset` 在其上必然返回 failure，而所有以「layout 是 strided 或单结果
    映射」为前提的设施都会随之失效或退化——`memref.subview` / `memref.expand_shape` 一族
    的结果类型推导、memref 到 LLVM 的下降、以及依赖 stride 做别名分析的 pass。这不是
    实现不足：分块布局本来就不是 strided，线性 offset 需要矩阵一整条边的长度，而
    `getAffineMap()` 拿不到 shape。

    当前没有生产者产出分块布局，所以碰不到；步 6 的来源问题（3.0 待确认项）一旦落地，
    Cube 路径上的 L1 / L0 搬运与切分就会正面撞上这条边界，届时要么走专门的搬运 Op、
    要么在分块维之外保持 strided。
  }];

  let parameters = (ins
    "::mlir::afir::LayoutOrder":$outer,
    DefaultValuedParameter<"::mlir::afir::TileOrder", "::mlir::afir::TileOrder::None">:$tiling,
    OptionalArrayRefParameter<"int64_t", "base tile extents in elements">:$tile,
    "int64_t":$rank
  );

  let assemblyFormat = "`<`"
    "`outer` `=` $outer `,`"
    "(`tiling` `=` $tiling^ `,`)?"
    "(`tile` `=` `[` $tile^ `]` `,`)?"
    "`rank` `=` $rank"
    "`>`";
  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// Axis and Loop Attributes (maps from af::Axis / ge.proto.AxisDef)
//===----------------------------------------------------------------------===//

def AxisAttr : AFIR_Attr<"Axis", "axis"> {
  let summary = "Axis definition for computational loops";
  let description = [{
    Defines an axis for loop-based computation including axis type, symbolic
    size/alignment expressions, binding info, and split source.
  }];

  let parameters = (ins
    DefaultValuedParameter<"int64_t", "-1">:$id,
    "StringAttr":$name,
    "::mlir::afir::AxisType":$axis_type,
    DefaultValuedParameter<"bool", "false">:$bind_block,
    "ExprAttr":$size,
    OptionalParameter<"ExprAttr">:$align,
    OptionalArrayRefParameter<"int64_t">:$from,
    DefaultValuedParameter<"int64_t", "-1">:$split_pair_other_id
  );

  // 末尾三项各自可缺省，且都以同一个 `,` 开头——ODS 的可选组表达不了这个形态：
  // 它按组内**首个字面量**判断该组是否出现，三组的首字面量都是逗号，于是解析器一见到
  // 逗号就锁定 `align` 那一组，后面写 `from` 直接报 "expected 'align'"。打印方向没有
  // 这个问题，因此症状是**打印得出、读不回**：`from = [0]` 不带 `align` 时属性不再
  // round-trip。之所以一直没暴露，是因为 bridge 产出的轴始终带 `align`。
  //
  // 交给 custom 指令逐项识别关键字，文本形态与此前逐字一致。
  let assemblyFormat = "`<`"
    "`id` `=` $id `,`"
    "`name` `=` $name `,`"
    "`axis_type` `=` $axis_type `,`"
    "(`bind_block` `=` $bind_block^ `,`)?"
    "`size` `=` $size ``"
    "custom<AxisTail>($align, $from, $split_pair_other_id)``"
    "`>`";
}

//===----------------------------------------------------------------------===//
// Axis binding: ties one SSA value (or one loop) back to a schedule axis
//===----------------------------------------------------------------------===//

def AxisBindingAttr : AFIR_Attr<"AxisBinding", "axis_binding"> {
  let summary = "Binds an SSA value or a loop to a schedule axis and its role";
  let description = [{
    `LoopGenerate` 物化循环后，轴不再以 `AxisAttr` 的形式出现在函数体里，而是散成
    一组标量 SSA 值和 `scf.for`。本属性是把它们认回某个轴的唯一通路：`axis` 指向
    `#afir.asc_graph` 里的 `AxisAttr.id`，`role` 说明这个值是该轴的什么。

    定义 op 的 `afir.axis_binding` 可保存单个本属性或属性数组；多个轴共用一个 SSA 值时，
    数组保留全部 (axis, role) 绑定。

    它替代了 `afir.axis_id` + `afir.axis_role` 这对无类型属性——后者把角色写成自由
    字符串，拼错时既不报错也查不到值，失败方式是静默的 lowering 回退。
  }];

  let parameters = (ins
    "int64_t":$axis,
    "::mlir::afir::AxisValueRole":$role
  );

  let assemblyFormat = "`<` `axis` `=` $axis `,` `role` `=` $role `>`";
  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// Schedule placement of one operation
//===----------------------------------------------------------------------===//

def ScheduleAttr : AFIR_Attr<"Schedule", "sched"> {
  let summary = "Axes an operation is scheduled over, and its innermost loop axis";
  let description = [{
    调度阶段给每个算子确定的位置：`axis` 是它被调度在哪些轴上，`loop_axis` 是其中
    最内层的那个。两者原本是一对独立的无类型属性（`afir.sched_axis` 数组 +
    `afir.loop_axis` 整数），彼此的约束（`loop_axis` 必须出现在 `axis` 里）只能在
    消费端逐处重查；合为一个 Attr 后由 verifier 一次钉死。

    `exec_condition` 是同一次调度定下的第三件事：这个算子的结果在 `loop_axis` 上多久
    才变一次，也就是它的缓存复用周期（上游即 `node->attr.sched.exec_condition`，与
    `axis` / `loop_axis` 同在一个 `SchedInfo` 里）。缺省 `no_cache` 表示每个迭代都算。

    **它不是位置之外的另一类事实，所以不另立一个属性。** 三者一起才说得清「这个算子在
    哪根轴上、隔多久执行一次」；分成两个属性时，只挂了 `exec_condition` 却没有
    `loop_axis` 的组合在类型上是合法的，而那种组合无法生成出任何谓词。
  }];

  let parameters = (ins
    ArrayRefParameter<"int64_t", "axis ids this operation is scheduled over">:$axis,
    DefaultValuedParameter<"int64_t", "-1">:$loop_axis,
    DefaultValuedParameter<"::mlir::afir::ExecCondition",
                           "::mlir::afir::ExecCondition::NoCache">:$exec_condition
  );

  let assemblyFormat = "`<` `axis` `=` `[` $axis `]` (`,` `loop_axis` `=` $loop_axis^)?"
    "(`,` `exec_condition` `=` $exec_condition^)? `>`";
  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// Graph Attributes (tiling key + axes + graph type + size vars)
//===----------------------------------------------------------------------===//

def AscGraphAttrGroupsAttr : AFIR_Attr<"AscGraphAttrGroups", "asc_graph"> {
  let summary = "Ascend graph attribute groups";

  let parameters = (ins
    DefaultValuedParameter<"int64_t", "-1">:$tiling_key,
    ArrayRefParameter<"AxisAttr", "array of Axis">:$axes,
    "::mlir::afir::AscGraphType":$type,
    OptionalArrayRefParameter<"ExprAttr", "array of size variable expressions">:$size_var
  );

  let assemblyFormat = "`<`"
    "(`tiling_key` `=` $tiling_key^ `,`)?"
    "`axes` `=` `[` $axes `]` `,`"
    "`type` `=` $type"
    "(`,` `size_var` `=` `[` $size_var^ `]`)?"
    "`>`";
}

//===----------------------------------------------------------------------===//
// The target one plan was compiled against
//===----------------------------------------------------------------------===//

def TargetAttr : AFIR_Attr<"Target", "target"> {
  let summary = "SoC and the resource envelope the plan module was compiled against";
  let description = [{
    挂在 plan module 上，记录这份 plan 是为哪个目标、按哪一组资源上限切出来的。三个
    字段都由 `bridge/CodegenOrchestration` 在**下降开始之前**从 `ascend::TargetProfile`
    读出：`soc` 取 `TargetIdentity::socVersion`，`aiv_num` 取
    `CoreTopology::vectorCoreCount`，`ub_size` 取 UB 的 `capacityBytes`。

    **`soc` 取 `socVersion` 而不是 `shortSocVersion` 或 `getFingerprint()`，是因为它是**可
    反解的键**：`TargetSelection::builtinPlatform(soc)` 收的正是这个串，将来让
    `resolveSessionTargetProfile` 在没有 `--target-profile` 时回退到读它，dump 出来的
    module 就能不带命令行直接重跑。注意它只认到 SoC，不认到具体 profile 文件——
    `--target-profile` 指定的外部 profile 反解不回来，那需要另存 `getFingerprint()`。
  }];

  let parameters = (ins
    "StringAttr":$soc,
    "int64_t":$aiv_num,
    "int64_t":$ub_size
  );

  let assemblyFormat = "`<` `soc` `=` $soc `,` `aiv_num` `=` $aiv_num `,` `ub_size` `=` $ub_size `>`";
  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// The logical reduction domain one reduce operation's input stands for
//===----------------------------------------------------------------------===//

def ReductionDomainAttr : AFIR_Attr<"ReductionDomain", "reduction_domain"> {
  let summary = "The full logical reduction domain this operation's input stands for";
  let description = [{
    标记单个归约算子输入所代表的完整逻辑归约域，当前用于 `afir.mean` 的分母计算。
    bridge 在内存物化前追踪输入来源并写入；下降侧不能仅凭阶段内的 Original 轴
    推断该域，因为跨阶段生成的轴可能只表示 partial 数量。

    `axes` 为 Original 轴 ID，`sizes` 为一一对应的尺寸表达式，其乘积为分母 `N`。
    属性缺席表示归约域尚未证明，消费侧必须拒绝。
  }];

  let parameters = (ins
    ArrayRefParameter<"int64_t", "original axis ids the domain is attributed to">:$axes,
    ArrayRefParameter<"ExprAttr", "size expression of each axis; N is their product">:$sizes
  );

  let assemblyFormat = "`<` `axes` `=` `[` $axes `]` `,` `sizes` `=` `[` $sizes `]` `>`";
  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// Scalar value attribute (for Data/Output nodes)
//===----------------------------------------------------------------------===//

def ScalarValueAttr : AFIR_Attr<"ScalarValue", "value"> {
  let summary = "representing the value of scalar";

  let parameters = (ins
    "StringAttr":$value
  );

  let assemblyFormat = "`<` struct(params) `>`";
}


#endif // AFIR_ATTRS
