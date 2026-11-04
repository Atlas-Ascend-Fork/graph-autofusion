// -----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------

//===- AFIREnums.td - AFIR dialect enum definitions --------*- tablegen -*-===//
//
// Enumerations for the AFIR dialect. These mirror autofuse's schedule/mem
// concepts (Role, Space, AxisType, AscGraphType).
//
//===----------------------------------------------------------------------===//

#ifndef AFIR_ENUMS
#define AFIR_ENUMS

include "AFIRAttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// Function Role Enumeration
//===----------------------------------------------------------------------===//

def AFIR_FunctionRoleEnum : I32EnumAttr<"FunctionRole",
    "AFIR function role",
    [
      I32EnumAttrCase<"KernelBody", 0, "kernel_body">,
      I32EnumAttrCase<"KernelEntry", 1, "kernel_entry">
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::afir";
  let underlyingType = "uint8_t";
}

//===----------------------------------------------------------------------===//
// Tensor/value role and transfer direction
//===----------------------------------------------------------------------===//

def AFIR_RoleEnum : I32EnumAttr<"Role",
    "AFIR value role",
    [
      I32EnumAttrCase<"Input", 0, "input">,
      I32EnumAttrCase<"Output", 1, "output">,
      I32EnumAttrCase<"Compute", 2, "compute">,
      I32EnumAttrCase<"Constant", 3, "constant">
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::afir";
  let underlyingType = "uint8_t";
}

def AFIR_SpaceEnum : I32EnumAttr<"Space",
    "AFIR memory space",
    [
      I32EnumAttrCase<"GM", 0, "gm">,
      I32EnumAttrCase<"UB", 1, "ub">,
      I32EnumAttrCase<"L1", 2, "l1">,
      I32EnumAttrCase<"L2", 3, "l2">,
      I32EnumAttrCase<"L0A", 4, "l0a">,
      I32EnumAttrCase<"L0B", 5, "l0b">,
      I32EnumAttrCase<"L0C", 6, "l0c">
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::afir";
  let underlyingType = "uint8_t";
}

//===----------------------------------------------------------------------===//
// Axis Type Enumeration (matches af::Axis::Type)
//===----------------------------------------------------------------------===//

def AFIR_AxisTypeAttr : I32EnumAttr<"AxisType",
    "Axis type for computational loops",
    [
      I32EnumAttrCase<"Original", 0>,
      I32EnumAttrCase<"BlockOuter", 1>,
      I32EnumAttrCase<"BlockInner", 2>,
      I32EnumAttrCase<"TileOuter", 3>,
      I32EnumAttrCase<"TileInner", 4>,
      I32EnumAttrCase<"Merged", 5>,
      I32EnumAttrCase<"Invalid", 6>
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::afir";
  let underlyingType = "uint8_t";
}

//===----------------------------------------------------------------------===//
// Axis value role: what an SSA value derived from a schedule axis represents
//===----------------------------------------------------------------------===//
//
// `LoopGenerate` 物化循环时会为每个轴产生若干标量值（轴长、循环次数、尾块大小、
// 索引……），下游的 `TransferPlan` 需要按「哪个轴的哪个角色」把它们找回来。这条
// 反向引用在 IR 结构上不存在，此前用一对无类型属性
// （`afir.axis_id` 整数 + `afir.axis_role` 自由字符串）承载，角色拼错不会报错。
//
// 枚举化之后拼错在解析期即失败，取值集合也进入 `-gen-dialect-doc`。
def AFIR_AxisValueRoleEnum : I32EnumAttr<"AxisValueRole",
    "Role of an SSA value derived from a schedule axis",
    [
      // `scf.for` 本身：该循环迭代的是哪个轴。
      I32EnumAttrCase<"Loop", 0, "loop">,
      // 轴的当前索引值。
      I32EnumAttrCase<"AxisIndex", 1, "axis_index">,
      // 轴的名义长度。
      I32EnumAttrCase<"AxisSize", 2, "axis_size">,
      // 轴上的循环次数（按 tile 切分后的迭代数）。
      I32EnumAttrCase<"LoopSize", 3, "loop_size">,
      // 尾块长度；整除时为 0。
      I32EnumAttrCase<"TailSize", 4, "tail_size">,
      // 当前迭代的有效长度：整块取 axis_size，尾块取 tail_size。
      I32EnumAttrCase<"ActualSize", 5, "actual_size">
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::afir";
  let underlyingType = "uint8_t";
}

//===----------------------------------------------------------------------===//
// Execute condition: when a cacheable operation has to run again
//===----------------------------------------------------------------------===//
//
// 调度器给「可缓存」算子（广播链上的 Load / Broadcast 及其间的单入节点）定下的复用
// 周期，来源是 `af::ExecuteCondition`（`ascendc_ir_def.h:221`，由
// `optimize/autoschedule/node_cache_marker.cpp` 写入 `node->attr.sched.exec_condition`）。
//
// 它说的是「这块 UB 在什么时候才需要重算」：缓存值在一个复用周期内不变，周期起点算一次、
// 其余迭代直接读上一次的结果。取值不进 lowering 的判断，只决定循环体里那个 `if` 的谓词
// 怎么建（见 `materializeExecConditionGuard`）。
//
// 拼写与 legacy 的 `ExecuteConditionToString`（`ascir/meta/ascendc_graph_txt_dumper.cpp:941`）
// 逐字相同，AscGraph 的 txt dump 与 AFIR 的 dump 才对得上。
//
// **`kConditionInvalid` 没有对应取值**，是有意的：它在上游表示「还没定」，照搬进来会给
// 生成侧一个既不是「不缓存」也不是某个周期的第三种状态。bridge 碰到它报不支持。
def AFIR_ExecConditionEnum : I32EnumAttr<"ExecCondition",
    "Reuse cycle of a cacheable operation: when it has to be executed again",
    [
      // 不缓存：每个迭代都算。
      I32EnumAttrCase<"NoCache", 0, "no_cache">,
      // 广播轴与非广播轴合成一根轴后再切分：缓存值只在一个广播复用周期内不变，
      // 每个周期起点要重算。
      I32EnumAttrCase<"CacheBlockSplitFusedBroadcastAxis", 1, "cache_block_split_fused_brc_axis">,
      // 分到 TILE_OUT 及以外的原始轴都是广播轴：缓存值在整个分块循环内不变，
      // 只在第一次迭代算。
      I32EnumAttrCase<"CacheBlockSplitOriginBroadcastAxis", 2, "cache_block_split_origin_brc_axis">
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::afir";
  let underlyingType = "uint8_t";
}

//===----------------------------------------------------------------------===//
// Memory layout: outer order and intra-tile order
//===----------------------------------------------------------------------===//
//
// 布局按 ADR-16 做**正交分解**而不是扁平枚举：外层布局（基块网格的遍历序）× 分块布局
// （基块内部的遍历序）× 基块尺寸参数。扁平枚举（`nd|nz|dn|fractal`）随组合爆炸，且基块
// 大小跨代变化时还要新增枚举项。
//
// 这套分解与昇腾自己的两字母格式名恰好同构——`FRACTAL_NZ` 的 "N" 是外层按列、"Z" 是块内
// 按行，`FRACTAL_ZN` 反之。也就是说硬件侧的命名本来就是两维的，扁平枚举才是丢信息的那个。

def AFIR_LayoutOrderEnum : I32EnumAttr<"LayoutOrder",
    "Traversal order of the outer (untiled or tile-grid) dimensions",
    [
      I32EnumAttrCase<"RowMajor", 0, "row_major">,
      I32EnumAttrCase<"ColMajor", 1, "col_major">
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::afir";
  let underlyingType = "uint8_t";
}

def AFIR_TileOrderEnum : I32EnumAttr<"TileOrder",
    "Traversal order inside one base tile, or none when the layout is not tiled",
    [
      // 不分块：外层布局直接作用于元素。
      I32EnumAttrCase<"None", 0, "none">,
      I32EnumAttrCase<"RowMajor", 1, "row_major">,
      I32EnumAttrCase<"ColMajor", 2, "col_major">
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::afir";
  let underlyingType = "uint8_t";
}

//===----------------------------------------------------------------------===//
// Pipe: the asynchronous engine an operation executes on
//===----------------------------------------------------------------------===//
//
// 同一条 pipe 内程序序即执行序，跨 pipe 之间没有任何顺序保证——这是同步语义的全部
// 出发点（设计文档 3.6 / ADR-19）。
//
// 取值是 AFIR 自己的抽象引擎，不是 AscendC 的 MTE2 / MTE3 / V：搬入、搬出、计算彼此
// 异步是硬件无关的事实，映射到目标上的哪条物理 pipe 是 lowering 的事（ADR-20：与
// AscendC 的同构是多后端映射的结果，不是借用）。
//
// 这个词表此前只以 `RaceCheck.h` 里的一个内部 C++ enum（`ExecutionStream`）加一个
// 手写的 name 映射存在——竞争检查器和同步原语要谈的是同一件事，各留一份就是 D2。
//
// **一个取值对应一台引擎。** 此前 `compute` 一个值兼管矢量、标量与 Cube，而这三者在
// 硬件上是三台独立的引擎、彼此异步。写 Cube kernel 时 `L1 -> L0A/L0B` 的搬运、MAD
// 本身、以及 `L0C` 的搬出各在一台引擎上，塌成一个值就等于说它们之间不需要同步——而
// lowering 会把它照直映射成 `PIPE_V`，既不报错也不正确。
def AFIR_PipeEnum : I32EnumAttr<"Pipe",
    "Asynchronous engine an AFIR operation executes on",
    [
      // 片外到片上的搬运（`AFIRTransferIn` trait）。
      I32EnumAttrCase<"TransferIn", 0, "transfer_in">,
      // 片上到片外的搬运（`AFIRTransferOut` trait）。
      I32EnumAttrCase<"TransferOut", 1, "transfer_out">,
      // 矢量计算。
      I32EnumAttrCase<"Vector", 2, "vector">,
      // 片上到片上的搬运：Cube 路线上把数据从暂存搬进乘加单元的输入端。
      I32EnumAttrCase<"TransferLocal", 3, "transfer_local">,
      // 矩阵乘加。
      I32EnumAttrCase<"Cube", 4, "cube">,
      // 乘加结果的后处理与搬出。名字带硬件味，但它确实是一台独立引擎，抽象不出更中性
      // 的叫法——ADR-20 说的就是这件事：同构是多后端映射的结果。
      I32EnumAttrCase<"Fixpipe", 5, "fixpipe">,
      // 标量。
      I32EnumAttrCase<"Scalar", 6, "scalar">
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::afir";
  let underlyingType = "uint8_t";
}

//===----------------------------------------------------------------------===//
// Kernel argument role
//===----------------------------------------------------------------------===//
//
// kernel 参数在 ABI 里扮演什么角色（设计文档 3.6「Kernel 程序容器」）。
//
// 此前这个问题由三套互不相干的机制分别回答：数据参数靠一对**互斥**的整数 arg attr
// （`afir.input_index` / `afir.output_index`，「两个都挂」只能在消费端事后发现）；
// workspace 靠**类型相等**（参数类型恰好等于 `getAscendCWorkspaceType`）；TilingData
// 靠**结构体名字**等于 "TilingData"。三者都不是参数自己声明的身份，而是消费端反推的
// 结论，因此每个消费端都要把这套反推重写一遍，且反推规则不在任何对外文档里。
//
// 枚举化之后 kernel ABI 的参数词表就是这个枚举本身，进 `-gen-dialect-doc`。
def AFIR_KernelArgRoleEnum : I32EnumAttr<"KernelArgRole",
    "Role of a kernel argument in the AFIR kernel ABI",
    [
      // 主机侧交进来的输入 GM 张量，`index` 是它在前端图输入里的序号。
      I32EnumAttrCase<"Input", 0, "input">,
      // 写回主机的输出 GM 张量，`index` 是它在前端图输出里的序号。
      I32EnumAttrCase<"Output", 1, "output">,
      // 编译期分配、运行时由 launch 侧提供的临时 GM 空间。整个 kernel 至多一个。
      I32EnumAttrCase<"Workspace", 2, "workspace">,
      // host tiling 求解结果的描述符（当前形态是 TilingData 结构体）。至多一个。
      I32EnumAttrCase<"Tiling", 3, "tiling">,
      // 主机侧**按值**传进来的标量输入（AscGraph 的 `ScalarData`）。它不是一块 GM 存储，
      // 因此没有 `afir.buffer_desc`，在 `afir.arg_descs` 里只占一格退化的描述子。
      //
      // **`index` 与 `Input` 是同一个序号空间**，不是各排各的：legacy 的 `Data` 与
      // `ScalarData` 共用同一个 `index` 属性，kernel 形参就按它交错排列
      // （`codegen_kernel.cpp` 的 `ParseGraph`）。所以同一个 `index` 上同时出现
      // `input` 与 `scalar_input` 是冲突，由 `getKernelArgumentKey` 判掉。
      //
      // 为什么不复用 `Input` 再按类型（非 `ShapedType`）反推：那样一个**本该是 GM 张量**
      // 的参数被写成了标量类型时，会被静默当成合法的标量输入收下，而没有任何用例会红。
      // 角色写出来，两个方向才都拦得住。
      I32EnumAttrCase<"ScalarInput", 4, "scalar_input">,
      // workspace 张量相对 GM 基址的字节偏移，每个张量一个参数，以 buffer id（index）区分。
      // body 签名建立时 tiling 类型尚未生成，偏移由 entry 读取后传入，沿用 legacy ABI。
      I32EnumAttrCase<"WorkspaceOffset", 5, "workspace_offset">
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::afir";
  let underlyingType = "uint8_t";
}

//===----------------------------------------------------------------------===//
// AscGraphType Enumeration
//===----------------------------------------------------------------------===//

def AFIR_AscGraphTypeAttr : I32EnumAttr<"AscGraphType",
    "Ascend graph type",
    [
      I32EnumAttrCase<"Compute", 0>,
      I32EnumAttrCase<"Invalid", 1>
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::afir";
  let underlyingType = "uint8_t";
}

#endif // AFIR_ENUMS
