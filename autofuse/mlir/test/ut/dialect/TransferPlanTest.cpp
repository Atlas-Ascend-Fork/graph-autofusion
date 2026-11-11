/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// `buildTransferPlan` 的覆盖。此前这个文件不存在——搬运算子唯一会算错地址的分析，零
// 单元测试。
//
// 为什么不是 LIT：这里要钉的两条判据，失败形态都是「IR 完全合法、产出也合法，只是
// 搬的不是该搬的东西」：
//
//   - 向量化轴不止一根时取 `front()`：产出一个结构正确的一维搬运计划，把多维 tile 当成
//     一维搬。LIT 能断言产出的文本，但产出本身没有异常形状，没有可断言的锚点。
//   - 步长用了 `size_var` 之外的符号：物化时凭空造一个没人赋值的 tiling 参数，运行期读
//     未初始化内存。这在编译期的 IR 文本里同样看不出来。
//
// 两条都必须成对写：只有正例时，实现退化成「全拒绝」也能全绿。

#include "dialect/AFIR/Analysis/TransferPlan.h"

#include <algorithm>
#include <cstddef>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "dialect/AFIR/IR/AFIR.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::afir;

namespace {

// 轴形态取最常见的一种，与 `afir-to-ascendc-scheduled-transfer.mlir` 同形：
//
//   z (Original, s0)
//     ├─ tile  → zT (TileOuter, id 1)   zt (TileInner, id 2, zt_size)
//     └─ zT 再 block → zTB (BlockOuter, id 3)   zTb (BlockInner, id 4, zTb_size)
//
// `sched.loop_axis = 4`，所以 BlockOuter 反查得 3、TileOuter 反查得 1。
// `globalVectorizedAxis` 的哨兵：空串已经有含义（「整个字段不出现」），不能拿它兼职。
inline constexpr const char *kSameAsLocalVectorizedAxis = "<same-as-local>";
// 同样的哨兵，给 GM 侧的步长表用。空串在这里也已经有含义（一张空表），不能兼职。
inline constexpr const char *kSameAsLocalStrides = "<same-as-local>";

struct DescriptorShape {
  // 局部侧 descriptor 的 `vectorized_axis` / `vectorized_strides`。轴数是本期最重要的
  // 判据，所以它是第一个可调的旋钮。
  //
  // 两者置空表示整个字段不出现——`vectorized_axis` 是 optional 参数，写
  // `vectorized_axis = []` 反而解析不过（"expected integer value"）。
  std::string vectorizedAxis = "[2]";
  std::string vectorizedStrides = R"AF(#afir.expr<"1">)AF";
  // GM 侧 descriptor 的同名字段。默认取哨兵值，表示「与局部侧逐字相同」——这是常态，
  // 两侧描述的是同一次搬运。只有钉「`TransferPlanSide` 读的是哪一侧」的用例才把它调开：
  // 两侧都一样的话，读错一侧也看不出来。
  std::string globalVectorizedAxis = kSameAsLocalVectorizedAxis;
  std::string globalVectorizedStrides;
  // descriptor 的完整轴表（GM 视图）与对齐的步长表。轴数必须与张量秩一致，所以调整轴表
  // 的用例同时要调 `memref`。
  std::string axis = "3, 4, 2";
  std::string memref = "memref<2x8x16xf16>";
  std::string strides = R"AF(#afir.expr<"(zTb_size * zt_size)">, #afir.expr<"zt_size">, #afir.expr<"1">)AF";
  // `ParseStore` 专用：GM 侧（`destination_descs`）的轴表 / 范围表 / 步长表。默认都取
  // 哨兵值，表示「与局部侧逐字相同」——只有钉「GM 偏移读的是哪一侧」的用例才调开。
  //
  // Store 的两侧本来就可能不同：局部侧是生产者那块 UB 张量的布局，GM 侧是落点的
  // 布局。归约把一根轴压掉时，前者在那根轴上是 0，后者不是。轴表则在落点没被调度过时
  // （搬出到 kernel 输出）整个换成原始轴。
  std::string destinationAxis = kSameAsLocalStrides;
  // 落点的 memref。`afir.store` 的验证器要求 descriptor 的轴数等于落点的秩，所以换轴基
  // 的用例必须连它一起换（同时也要把 `indexingMaps` 的第二张图改成对应的投影）。
  std::string destinationMemref = kSameAsLocalStrides;
  std::string destinationRepeats = kSameAsLocalStrides;
  std::string destinationStrides = kSameAsLocalStrides;
  // 落点侧的 `vectorized_axis` / `vectorized_strides`。哨兵值表示「轴表与局部侧相同时才
  // 出现」——搬出到 kernel 输出（换轴基）与写已调度的 workspace（同轴基）这两种形态各占
  // 一头，默认覆盖得到。
  //
  // 但两者不是非此即彼：RCore 一阶段写 workspace 时落点既被调度过（有
  // `vectorized_axis`）**又**比局部侧多几根轴（归约方向的块号），那一档要把它单独打开。
  std::string destinationVectorizedAxis = kSameAsLocalStrides;
  std::string destinationVectorizedStrides;
  // 与 `axis` 对齐的范围表。有效视图归约的第一条规则（extent == 1 剔除）读的就是它。
  std::string repeats =
      R"AF(#afir.expr<"Ceiling((Ceiling((s0 / (zt_size))) / (zTb_size)))">, #afir.expr<"zTb_size">, #afir.expr<"zt_size">)AF";
  std::string sizeVar =
      R"AF(#afir.expr<"s0">, #afir.expr<"zt_size">, #afir.expr<"zTb_size">, #afir.expr<"s1">, #afir.expr<"y1t_size">)AF";
  // zTB（id 3）那一行 `size` 之后的尾部。默认是普通的切分 outer；RCore 的合并块轴把
  // `from` 写成两根源轴并且**没有** `split_pair_other_id`，反查就是在这种轴上落空的。
  std::string blockOuterTail = R"AF(align = #afir.expr<"1">, from = [1], split_pair_other_id = 4)AF";
  // 搬运两端的 `indexing_maps`。默认是 identity——正因为多维 tile 的 map 完全可能也是
  // identity，"调度路径拒绝后回落 identity" 才是一个真实的静默错误。
  std::string indexingMaps = "affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>";
  // 只对 `ParseLowered` 有意义：插在循环体里、`afir.load` 之前的额外 IR。多根向量化轴的
  // 用例要为每根会缩水的 tile 内轴补一份 `actual_size` 绑定——真实 IR 里这些由
  // `LoopGenerate::generatePairedInnerActualSize` 生成，这里手写。
  std::string loopBodyPrologue;
};

class TransferPlanTest : public testing::Test {
 protected:
  void SetUp() override {
    context.getOrLoadDialect<AFIRDialect>();
    context.getOrLoadDialect<arith::ArithDialect>();
    context.getOrLoadDialect<func::FuncDialect>();
    context.getOrLoadDialect<memref::MemRefDialect>();
    context.getOrLoadDialect<scf::SCFDialect>();
  }

  OwningOpRef<ModuleOp> Parse(const DescriptorShape &shape) {
    const std::string vectorizedClause = shape.vectorizedAxis.empty()
                                             ? std::string()
                                             : "  vectorized_axis = " + shape.vectorizedAxis +
                                                   ", vectorized_strides = [" + shape.vectorizedStrides + "],\n";
    // GM 侧默认与局部侧同形；只有钉「读的是哪一侧」的用例才把它调开。
    const std::string globalVectorizedClause =
        shape.globalVectorizedAxis == kSameAsLocalVectorizedAxis
            ? vectorizedClause
            : (shape.globalVectorizedAxis.empty()
                   ? std::string()
                   : "  vectorized_axis = " + shape.globalVectorizedAxis + ", vectorized_strides = [" +
                         shape.globalVectorizedStrides + "],\n");
    const std::string descriptors = R"mlir(
#gm_buffer = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"(2 * s0)">, life_time_axis_id = -1>
#ub_queue = #afir.buffer_desc<space = <ub>, buffer_id = 0, slot_count = 2, queue_depth = 2, size_bytes_expr = <"(2 * zt_size)">, alignment_bytes = 32, life_time_axis_id = -1>
#gm_desc = #afir.tensor_desc<
  axis = [)mlir" + shape.axis +
                                    R"mlir(],
  repeats = [)mlir" + shape.repeats +
                                    R"mlir(],
  strides = [)mlir" + shape.strides +
                                    R"mlir(],
)mlir" + globalVectorizedClause +
                                    R"mlir(  role = <input>>
#ub_desc = #afir.tensor_desc<
  axis = [)mlir" + shape.axis +
                                    R"mlir(],
  repeats = [)mlir" + shape.repeats +
                                    R"mlir(],
  strides = [)mlir" + shape.strides +
                                    R"mlir(],
)mlir" + vectorizedClause +
                                    R"mlir(  role = <compute>>
)mlir";

    const std::string source = descriptors + R"mlir(
module {
  func.func private @kernel(%input: )mlir" +
                               shape.memref + R"mlir( {afir.buffer_desc = #gm_buffer}) attributes {
    afir.arg_descs = [#gm_desc],
    afir.asc_graph = #afir.asc_graph<
      tiling_key = 0,
      axes = [
        <id = 0, name = "z", axis_type = Original, size = #afir.expr<"s0">>,
        <id = 1, name = "zT", axis_type = TileOuter, size = #afir.expr<"Ceiling((s0 / (zt_size)))">,
          align = #afir.expr<"1">, from = [0], split_pair_other_id = 2>,
        <id = 2, name = "zt", axis_type = TileInner, size = #afir.expr<"zt_size">,
          align = #afir.expr<"1">, from = [0], split_pair_other_id = 1>,
        <id = 3, name = "zTB", axis_type = BlockOuter, size = #afir.expr<"Ceiling((Ceiling((s0 / (zt_size))) / (zTb_size)))">,
          )mlir" + shape.blockOuterTail +
                               R"mlir(>,
        <id = 4, name = "zTb", axis_type = BlockInner, size = #afir.expr<"zTb_size">,
          align = #afir.expr<"1">, from = [1], split_pair_other_id = 3>,
        <id = 5, name = "y", axis_type = Original, size = #afir.expr<"s1">>,
        <id = 6, name = "yT", axis_type = TileOuter, size = #afir.expr<"Ceiling((s1 / (y1t_size)))">,
          align = #afir.expr<"1">, from = [5], split_pair_other_id = 7>,
        <id = 7, name = "yt", axis_type = TileInner, size = #afir.expr<"y1t_size">,
          align = #afir.expr<"1">, from = [5], split_pair_other_id = 6>,
        <id = 8, name = "yo", axis_type = Original, size = #afir.expr<"s1">>],
      type = Compute,
      size_var = [)mlir" + shape.sizeVar +
                               R"mlir(]>,
    afir.function_role = #afir.function_role<kernel_body>
  } {
    %tile = memref.alloc() {afir.buffer_desc = #ub_queue} : )mlir" +
                               shape.memref + R"mlir(
    %loaded = afir.load %input outs(%tile) {
      afir.destination_descs = [#ub_desc],
      afir.operand_descs = [#gm_desc],
      afir.result_descs = [#ub_desc],
      afir.sched = #afir.sched<axis = [3, 4], loop_axis = 4>,
      indexing_maps = [)mlir" + shape.indexingMaps +
                               R"mlir(]
    } : )mlir" + shape.memref + ", " +
                               shape.memref + " -> " + shape.memref + R"mlir(
    return
  }
}
)mlir";
    return parseSourceString<ModuleOp>(source, &context);
  }

  // 搬出方向。与 `Parse` 唯一的实质差别是 op 没有结果，于是局部侧落在 `operand_descs`
  // 上——而 Store 的 `operand_descs` 描述的是生产者那块 UB 张量，不是落点。
  //
  // 为什么要单独一个函数而不是给 `Parse` 加开关：`Parse` 的 `afir.load` 两侧 descriptor
  // 逐字相同，读错一侧看不出来（`PlanSideSelectsWhichDescriptorIsLocal` 只能靠调开
  // `vectorized_axis` 才钉得住）。搬出方向要钉的是**步长**读错侧，两张表必须能各写各的。
  OwningOpRef<ModuleOp> ParseStore(const DescriptorShape &shape) {
    const std::string destinationAxis =
        shape.destinationAxis == kSameAsLocalStrides ? shape.axis : shape.destinationAxis;
    const std::string destinationMemref =
        shape.destinationMemref == kSameAsLocalStrides ? shape.memref : shape.destinationMemref;
    const std::string destinationRepeats =
        shape.destinationRepeats == kSameAsLocalStrides ? shape.repeats : shape.destinationRepeats;
    const std::string destinationStrides =
        shape.destinationStrides == kSameAsLocalStrides ? shape.strides : shape.destinationStrides;
    // 落点换了轴基就没有 `vectorized_axis`——真实 IR 里搬出到 kernel 输出的那份
    // `destination_descs` 正是这样（图的出口没被切 tile，`vectorized_axis` 整个字段不出现）。
    const std::string destinationVectorizedAxis =
        shape.destinationVectorizedAxis == kSameAsLocalStrides
            ? (shape.destinationAxis == kSameAsLocalStrides ? shape.vectorizedAxis : std::string())
            : shape.destinationVectorizedAxis;
    const std::string destinationVectorizedStrides = shape.destinationVectorizedAxis == kSameAsLocalStrides
                                                         ? shape.vectorizedStrides
                                                         : shape.destinationVectorizedStrides;
    const std::string destinationVectorizedClause =
        destinationVectorizedAxis.empty() ? std::string()
                                          : "  vectorized_axis = " + destinationVectorizedAxis +
                                                ", vectorized_strides = [" + destinationVectorizedStrides + "],\n";
    const std::string source = R"mlir(
#gm_buffer = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"(2 * s0)">, life_time_axis_id = -1>
#ub_queue = #afir.buffer_desc<space = <ub>, buffer_id = 0, slot_count = 2, queue_depth = 2, size_bytes_expr = <"(2 * zt_size)">, alignment_bytes = 32, life_time_axis_id = -1>
#gm_desc = #afir.tensor_desc<
  axis = [)mlir" + destinationAxis +
                               R"mlir(],
  repeats = [)mlir" + destinationRepeats +
                               R"mlir(],
  strides = [)mlir" + destinationStrides +
                               R"mlir(],
)mlir" + destinationVectorizedClause +
                               R"mlir(  role = <output>>
#ub_desc = #afir.tensor_desc<
  axis = [)mlir" + shape.axis +
                               R"mlir(],
  repeats = [)mlir" + shape.repeats +
                               R"mlir(],
  strides = [)mlir" + shape.strides +
                               R"mlir(],
  vectorized_axis = )mlir" + shape.vectorizedAxis +
                               R"mlir(, vectorized_strides = [)mlir" + shape.vectorizedStrides +
                               R"mlir(],
  role = <compute>>

module {
  func.func private @kernel(%output: )mlir" +
                               destinationMemref + R"mlir( {afir.buffer_desc = #gm_buffer}) attributes {
    afir.arg_descs = [#gm_desc],
    afir.asc_graph = #afir.asc_graph<
      tiling_key = 0,
      axes = [
        <id = 0, name = "z", axis_type = Original, size = #afir.expr<"s0">>,
        <id = 1, name = "zT", axis_type = TileOuter, size = #afir.expr<"Ceiling((s0 / (zt_size)))">,
          align = #afir.expr<"1">, from = [0], split_pair_other_id = 2>,
        <id = 2, name = "zt", axis_type = TileInner, size = #afir.expr<"zt_size">,
          align = #afir.expr<"1">, from = [0], split_pair_other_id = 1>,
        <id = 3, name = "zTB", axis_type = BlockOuter, size = #afir.expr<"Ceiling((Ceiling((s0 / (zt_size))) / (zTb_size)))">,
          )mlir" + shape.blockOuterTail +
                               R"mlir(>,
        <id = 4, name = "zTb", axis_type = BlockInner, size = #afir.expr<"zTb_size">,
          align = #afir.expr<"1">, from = [1], split_pair_other_id = 3>,
        <id = 5, name = "y", axis_type = Original, size = #afir.expr<"s1">>,
        <id = 6, name = "yT", axis_type = TileOuter, size = #afir.expr<"Ceiling((s1 / (y1t_size)))">,
          align = #afir.expr<"1">, from = [5], split_pair_other_id = 7>,
        <id = 7, name = "yt", axis_type = TileInner, size = #afir.expr<"y1t_size">,
          align = #afir.expr<"1">, from = [5], split_pair_other_id = 6>,
        <id = 8, name = "yo", axis_type = Original, size = #afir.expr<"s1">>],
      type = Compute,
      size_var = [)mlir" + shape.sizeVar +
                               R"mlir(]>,
    afir.function_role = #afir.function_role<kernel_body>
  } {
    %tile = memref.alloc() {afir.buffer_desc = #ub_queue} : )mlir" +
                               shape.memref + R"mlir(
    afir.store %tile outs(%output) {
      afir.destination_descs = [#gm_desc],
      afir.operand_descs = [#ub_desc],
      afir.result_descs = [],
      afir.sched = #afir.sched<axis = [3, 4], loop_axis = 4>,
      indexing_maps = [)mlir" + shape.indexingMaps +
                               R"mlir(]
    } : )mlir" + shape.memref + ", " +
                               destinationMemref + R"mlir(
    return
  }
}
)mlir";
    return parseSourceString<ModuleOp>(source, &context);
  }

  // 循环生成之后的形态：轴已经散成 `scf.for` 与打了 `#afir.axis_binding` 的标量值。
  // 偏移计算只在这种 IR 上才有意义——它要把轴认回来。
  //
  // 与 `Parse` 共用 `DescriptorShape`：轴表与两张步长表是旋钮（多根向量化轴的生成用例要
  // 调它们），而绑定与调度循环的嵌套是手写的定值——这里钉的是「序号从哪儿取」。
  OwningOpRef<ModuleOp> ParseLowered(const DescriptorShape &shape) {
    const std::string source = R"mlir(
#gm_buffer = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"(2 * s0)">, life_time_axis_id = -1>
#ub_queue = #afir.buffer_desc<space = <ub>, buffer_id = 0, slot_count = 2, queue_depth = 2, size_bytes_expr = <"(2 * zt_size)">, alignment_bytes = 32, life_time_axis_id = -1>
#gm_desc = #afir.tensor_desc<
  axis = [)mlir" + shape.axis +
                               R"mlir(],
  repeats = [)mlir" + shape.repeats +
                               R"mlir(],
  strides = [)mlir" + shape.strides +
                               R"mlir(],
  vectorized_axis = )mlir" + shape.vectorizedAxis +
                               R"mlir(, vectorized_strides = [)mlir" + shape.vectorizedStrides +
                               R"mlir(],
  role = <input>>
#ub_desc = #afir.tensor_desc<
  axis = [)mlir" + shape.axis +
                               R"mlir(],
  repeats = [)mlir" + shape.repeats +
                               R"mlir(],
  strides = [)mlir" + shape.strides +
                               R"mlir(],
  vectorized_axis = )mlir" + shape.vectorizedAxis +
                               R"mlir(, vectorized_strides = [)mlir" + shape.vectorizedStrides +
                               R"mlir(],
  role = <compute>>

module {
  func.func private @kernel(%input: )mlir" +
                               shape.memref + R"mlir( {afir.buffer_desc = #gm_buffer}) attributes {
    afir.arg_descs = [#gm_desc],
    afir.asc_graph = #afir.asc_graph<
      tiling_key = 0,
      axes = [
        <id = 0, name = "z", axis_type = Original, size = #afir.expr<"s0">>,
        <id = 1, name = "zT", axis_type = TileOuter, size = #afir.expr<"Ceiling((s0 / (zt_size)))">,
          align = #afir.expr<"1">, from = [0], split_pair_other_id = 2>,
        <id = 2, name = "zt", axis_type = TileInner, size = #afir.expr<"zt_size">,
          align = #afir.expr<"1">, from = [0], split_pair_other_id = 1>,
        <id = 3, name = "zTB", axis_type = BlockOuter, size = #afir.expr<"Ceiling((Ceiling((s0 / (zt_size))) / (zTb_size)))">,
          align = #afir.expr<"1">, from = [1], split_pair_other_id = 4>,
        <id = 4, name = "zTb", axis_type = BlockInner, size = #afir.expr<"zTb_size">,
          align = #afir.expr<"1">, from = [1], split_pair_other_id = 3>,
        <id = 5, name = "y", axis_type = Original, size = #afir.expr<"s1">>,
        <id = 6, name = "yT", axis_type = TileOuter, size = #afir.expr<"Ceiling((s1 / (y1t_size)))">,
          align = #afir.expr<"1">, from = [5], split_pair_other_id = 7>,
        <id = 7, name = "yt", axis_type = TileInner, size = #afir.expr<"y1t_size">,
          align = #afir.expr<"1">, from = [5], split_pair_other_id = 6>,
        <id = 8, name = "yo", axis_type = Original, size = #afir.expr<"s1">>,
        <id = 9, name = "yo2", axis_type = Original, size = #afir.expr<"s0">>],
      type = Compute,
      size_var = [)mlir" + shape.sizeVar +
                               R"mlir(]>,
    afir.function_role = #afir.function_role<kernel_body>
  } {
    %zt = afir.tiling_param "zt_size" : index
    %ztb = afir.tiling_param "zTb_size" : index
    %y1t = afir.tiling_param "y1t_size" : index
    %s0 = afir.tiling_param "s0" : index
    %s1 = afir.tiling_param "s1" : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %tile = memref.alloc() {afir.buffer_desc = #ub_queue} : )mlir" +
                               shape.memref + R"mlir(
    %block = arith.remsi %c0, %ztb {afir.axis_binding = #afir.axis_binding<axis = 3, role = axis_index>} : index
    scf.for %round = %c0 to %ztb step %c1 {
      %actual = arith.addi %zt, %c0 {afir.axis_binding = #afir.axis_binding<axis = 2, role = actual_size>} : index
)mlir" + shape.loopBodyPrologue +
                               R"mlir(
      %loaded = afir.load %input outs(%tile) {
        afir.destination_descs = [#ub_desc],
        afir.operand_descs = [#gm_desc],
        afir.result_descs = [#ub_desc],
        afir.sched = #afir.sched<axis = [3, 4], loop_axis = 4>,
        indexing_maps = [)mlir" +
                               shape.indexingMaps +
                               R"mlir(]
      } : )mlir" + shape.memref +
                               ", " + shape.memref + " -> " + shape.memref + R"mlir(
    } {afir.axis_binding = #afir.axis_binding<axis = 4, role = loop>}
    return
  }
}
)mlir";
    return parseSourceString<ModuleOp>(source, &context);
  }

  // 模块里唯一的那个 `afir.load`。
  Operation *FindLoad(ModuleOp module) {
    Operation *found = nullptr;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "afir.load") {
        found = op;
      }
    });
    return found;
  }

  // 模块里唯一的那个 `afir.store`。
  Operation *FindStore(ModuleOp module) {
    Operation *found = nullptr;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "afir.store") {
        found = op;
      }
    });
    return found;
  }

  MLIRContext context;
};

TEST_F(TransferPlanTest, SharedValuePreservesAllAxisAndRoleBindings) {
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @shared() {
        %size = afir.tiling_param "n" : index
        return
      }
    }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto func = *module->getOps<func::FuncOp>().begin();
  Operation *definition = &func.front().front();
  Operation *user = func.front().getTerminator();
  const auto first = AxisBindingAttr::get(&context, 0, AxisValueRole::AxisSize);
  const auto second = AxisBindingAttr::get(&context, 2, AxisValueRole::AxisSize);
  const auto loopSize = AxisBindingAttr::get(&context, 0, AxisValueRole::LoopSize);
  addAxisBinding(definition, second);
  EXPECT_EQ(definition->getAttr(kAxisBindingAttrName), second);
  addAxisBinding(definition, first);
  addAxisBinding(definition, loopSize);
  addAxisBinding(definition, first);
  EXPECT_EQ(getAxisBindings(definition).size(), 3U);
  for (auto binding : {first, second, loopSize}) {
    EXPECT_EQ(findAxisValue(func, user, binding.getAxis(), binding.getRole()), definition->getResult(0));
  }
  EXPECT_FALSE(findAxisValue(func, user, 1, AxisValueRole::AxisSize));
  Attribute ordered = definition->getAttr(kAxisBindingAttrName);
  definition->removeAttr(kAxisBindingAttrName);
  for (auto binding : {loopSize, first, second}) {
    addAxisBinding(definition, binding);
  }
  EXPECT_EQ(definition->getAttr(kAxisBindingAttrName), ordered);
}

// 有效视图，写成可直接比对的形状：每项 `{extent, gmStride, ubStride}`。
using DimTriple = std::tuple<std::string, std::string, std::string>;

std::vector<DimTriple> DimsOf(const TransferPlan &plan) {
  auto text = [](ExprAttr attr) { return attr ? attr.getValue().getValue().str() : std::string(); };
  std::vector<DimTriple> result;
  for (const TransferDim &dim : plan.dims) {
    result.emplace_back(text(dim.extent), text(dim.globalStride), text(dim.localStride));
  }
  return result;
}

// 两根向量化轴的形态：外层是 A 侧的 tile 内轴（id 2），内层按用例挑。归约的四条规则都
// 要至少两根轴才看得出来，所以规则用例全部基于这张表。
DescriptorShape TwoAxisShape(int64_t innerAxis, StringRef innerExtent, StringRef outerGlobalStride,
                             StringRef outerLocalStride, StringRef innerGlobalStride = "1",
                             StringRef innerLocalStride = "1", StringRef outerExtent = "zt_size") {
  DescriptorShape shape;
  shape.axis = "3, 4, 2, " + std::to_string(innerAxis);
  shape.memref = "memref<2x8x16x4xf16>";
  shape.repeats = R"AF(#afir.expr<"Ceiling((Ceiling((s0 / (zt_size))) / (zTb_size)))">, #afir.expr<"zTb_size">, )AF" +
                  ("#afir.expr<\"" + outerExtent + "\">, #afir.expr<\"" + innerExtent + "\">").str();
  shape.strides = R"AF(#afir.expr<"(zTb_size * zt_size)">, #afir.expr<"zt_size">, )AF" +
                  ("#afir.expr<\"" + outerGlobalStride + "\">, #afir.expr<\"" + innerGlobalStride + "\">").str();
  shape.vectorizedAxis = "[2, " + std::to_string(innerAxis) + "]";
  shape.vectorizedStrides =
      ("#afir.expr<\"" + outerLocalStride + "\">, #afir.expr<\"" + innerLocalStride + "\">").str();
  shape.indexingMaps =
      "affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>";
  return shape;
}

// 一张 `{轴 id, 步长}` 表，写成可直接比对的形状。
std::vector<std::pair<int64_t, std::string>> OuterAxesOf(const TransferPlan &plan) {
  std::vector<std::pair<int64_t, std::string>> result;
  for (const TransferOuterAxis &outer : plan.outerAxes) {
    result.emplace_back(outer.axis,
                        outer.globalStride ? outer.globalStride.getValue().getValue().str() : std::string());
  }
  return result;
}

// `TransferPlanSide` 决定读哪一侧的 descriptor，这条规则今天没有别的东西钉着。
//
// 默认的 `Automatic` 按「有没有结果」选：Load 的局部侧是结果（`result_descs`），Store 的
// 局部侧是操作数（`operand_descs`）。计算算子两种都有结果，所以它们显式传 `Operand`——
// 要的是输入那块 tile 的几何，与 legacy 以 `x.actual_size` 为准同一条。
//
// 选错一侧的后果是拿到 GM 侧的向量化轴，而那不是同一件事：这里把 GM 侧调成「没切 tile」，
// 于是两侧给出的计划种类都不一样，读错立刻看得出来。三个既有消费者的调用形态就是这两种。
TEST_F(TransferPlanTest, PlanSideSelectsWhichDescriptorIsLocal) {
  DescriptorShape shape;
  shape.globalVectorizedAxis = "";
  shape.globalVectorizedStrides = "";
  OwningOpRef<ModuleOp> module = Parse(shape);
  ASSERT_TRUE(module);
  Operation *load = FindLoad(*module);
  ASSERT_NE(load, nullptr);

  // Load 有结果：`Automatic` 取 `result_descs`，那一侧切了 tile。
  FailureOr<TransferPlan> automatic = buildTransferPlan(load);
  ASSERT_TRUE(succeeded(automatic));
  EXPECT_EQ(automatic->kind, TransferPlanKind::Scheduled);
  ASSERT_EQ(automatic->dims.size(), 1U);
  EXPECT_EQ(automatic->dims.front().extent.getValue().getValue(), "zt_size");

  // 同一个 op 显式要 operand 侧：那是 GM 侧的 descriptor，没有 `vectorized_axis`，
  // 于是回落 identity。两种取法给出两种结果，正是「读的确实是不同的那一侧」。
  FailureOr<TransferPlan> operandSide = buildTransferPlan(load, TransferPlanSide::Operand);
  ASSERT_TRUE(succeeded(operandSide));
  EXPECT_EQ(operandSide->kind, TransferPlanKind::Identity);
}

// 基线：一根向量化轴，集合差是 `[3, 4]`，各自带 `strides` 表里对应那一格。
TEST_F(TransferPlanTest, SingleVectorizedAxisBuildsScheduledPlan) {
  OwningOpRef<ModuleOp> module = Parse(DescriptorShape{});
  ASSERT_TRUE(module);
  Operation *load = FindLoad(*module);
  ASSERT_NE(load, nullptr);

  FailureOr<TransferPlan> plan = buildTransferPlan(load);
  ASSERT_TRUE(succeeded(plan));
  EXPECT_EQ(plan->kind, TransferPlanKind::Scheduled);
  // 一维 tile：有效视图归约成一项，extent 就是那根向量化轴的范围。
  ASSERT_EQ(plan->dims.size(), 1U);
  EXPECT_EQ(plan->dims.front().extent.getValue().getValue(), "zt_size");
  // 顺序跟 `axis` 表（外 -> 内），不是跟轴 id 排序：偏移求和本身与顺序无关，但生成的
  // 加法链顺序要可预期，lit 才钉得住。
  EXPECT_EQ(OuterAxesOf(*plan),
            (std::vector<std::pair<int64_t, std::string>>{{3, "(zTb_size * zt_size)"}, {4, "zt_size"}}));
}

// 向量化轴是被排除的那一批，不是被漏掉的。两根向量化轴时集合差只剩一项——这条同时
// 说明集合差的结果随 `vectorized_axis` 变，而不是固定取前两根。
//
// （两根向量化轴当前被能力门禁拒在前面，所以这里直接查 `buildScheduledPlan` 的下游结果
// 不可行；改用「把 id 4 也标成向量化轴」的一根形态。）
TEST_F(TransferPlanTest, VectorizedAxisIsExcludedFromOuterAxes) {
  DescriptorShape shape;
  shape.vectorizedAxis = "[4]";
  shape.vectorizedStrides = R"AF(#afir.expr<"1">)AF";
  OwningOpRef<ModuleOp> module = Parse(shape);
  ASSERT_TRUE(module);
  Operation *load = FindLoad(*module);
  ASSERT_NE(load, nullptr);

  FailureOr<TransferPlan> plan = buildTransferPlan(load);
  ASSERT_TRUE(succeeded(plan));
  EXPECT_EQ(OuterAxesOf(*plan), (std::vector<std::pair<int64_t, std::string>>{{3, "(zTb_size * zt_size)"}, {2, "1"}}));
}

// 合并块轴。RCore 把 A 与 R 的两根 TileOuter 先合成一根再做 block 切分，合并出来的轴
// 没有配对 inner（`split_pair_other_id` 缺省为 -1）。
//
// 此前的 `splitPairOtherId == loop_axis` 反查在这种轴上直接落空，`buildScheduledPlan`
// 返回 failure，整条搬运退回 identity——搬整块。集合差不查配对：这根轴在 `axis` 里、不在
// `vectorized_axis` 里，照常贡献一项。
TEST_F(TransferPlanTest, MergedBlockAxisContributesAnOffsetTerm) {
  DescriptorShape shape;
  shape.blockOuterTail = R"AF(align = #afir.expr<"1">, from = [1, 5])AF";
  OwningOpRef<ModuleOp> module = Parse(shape);
  ASSERT_TRUE(module);
  Operation *load = FindLoad(*module);
  ASSERT_NE(load, nullptr);

  FailureOr<TransferPlan> plan = buildTransferPlan(load);
  ASSERT_TRUE(succeeded(plan)) << "合并块轴上反查落空，集合差不应该跟着落空";
  EXPECT_EQ(OuterAxesOf(*plan),
            (std::vector<std::pair<int64_t, std::string>>{{3, "(zTb_size * zt_size)"}, {4, "zt_size"}}));
}

// GM 偏移的步长必须取自 GM 侧（`destination_descs`）的步长表，而不是局部侧的。
//
// Store 的局部侧是 `operand_descs`，它描述的是生产者那块 UB 张量。归约把一根轴压掉
// 之后，生产者在那根轴上的步长是 0（每一轮写的是同一个累加器），而落点在那根轴上有真实
// 的步长（每一轮写落点的不同一格）。两者不是同一件事。
//
// 这正是 RCore 两段式 reduce 的形态：phase 1 每个核把自己那一段的 partial sum 写进
// workspace 的不同格，workspace 的大小也是按这个二维布局分配的。用局部侧的步长算 GM
// 地址，这一项会因为「步长为 0」被整项跳过，于是所有核写同一个地址、互相覆盖——IR 合法、
// 产出合法，只是算错。真机上 `nA == corenum` 时这一维恰好只有一格，所以看不出来。
TEST_F(TransferPlanTest, StoreOuterStrideComesFromTheGlobalSideDescriptor) {
  DescriptorShape shape;
  // 局部侧（生产者 UB 张量）在块轴 3 上被压平。
  shape.strides = R"AF(#afir.expr<"0">, #afir.expr<"zt_size">, #afir.expr<"1">)AF";
  // GM 侧（落点）在同一根轴上有真实步长。
  shape.destinationStrides = R"AF(#afir.expr<"(zTb_size * zt_size)">, #afir.expr<"zt_size">, #afir.expr<"1">)AF";
  OwningOpRef<ModuleOp> module = ParseStore(shape);
  ASSERT_TRUE(module);
  Operation *store = FindStore(*module);
  ASSERT_NE(store, nullptr);

  FailureOr<TransferPlan> plan = buildTransferPlan(store);
  ASSERT_TRUE(succeeded(plan));
  EXPECT_EQ(plan->kind, TransferPlanKind::Scheduled);
  EXPECT_EQ(OuterAxesOf(*plan),
            (std::vector<std::pair<int64_t, std::string>>{{3, "(zTb_size * zt_size)"}, {4, "zt_size"}}))
      << "块轴在落点上有步长，不能因为它在生产者那块 UB 张量上被压平就整项消失";
}

// 成对的反例：步长为 0 的轴仍然要剔掉，判据只是换成看 GM 侧那张表。
//
// 只写上一条的话，实现改成「无脑读 destination_descs 且不再剔零」也能全绿，而那会把
// `ZeroStrideTermIsDropped` 钉住的那条规则弄丢。两条合起来才说明：剔零还在，只是剔的
// 是 GM 侧的零。
TEST_F(TransferPlanTest, StoreZeroStrideIsJudgedOnTheGlobalSideDescriptor) {
  DescriptorShape shape;
  // 与上一条正好对调：局部侧有步长，GM 侧才是被压平的那一侧。
  shape.strides = R"AF(#afir.expr<"(zTb_size * zt_size)">, #afir.expr<"zt_size">, #afir.expr<"1">)AF";
  shape.destinationStrides = R"AF(#afir.expr<"0">, #afir.expr<"zt_size">, #afir.expr<"1">)AF";
  OwningOpRef<ModuleOp> module = ParseStore(shape);
  ASSERT_TRUE(module);
  Operation *store = FindStore(*module);
  ASSERT_NE(store, nullptr);

  FailureOr<TransferPlan> plan = buildTransferPlan(store);
  ASSERT_TRUE(succeeded(plan));
  EXPECT_EQ(OuterAxesOf(*plan), (std::vector<std::pair<int64_t, std::string>>{{3, "0"}, {4, "zt_size"}}))
      << "剔零发生在偏移发射时（computeGlobalOffset），不在集合差里——这里钉的是取到的那一格";
}

// 两侧轴基不同时，GM 侧那张表回答不了「这根调度轴动一格落点挪多远」，只能按局部侧算。
//
// `destination_descs` 是 Store 节点自己的 `.y`，而 `.y` 落在哪个轴基上取决于它是什么：
// 写 workspace 时它被调度过，轴表与局部侧一致（上面两条）；写 kernel 输出时它是图的
// 出口，轴表是原始轴。搬出到输出参数是最常见的形态，CPU E2E 的每个用例都走它——
// 不区分轴基就会在那里报 `'afir.store' op has no supported Ascend implementation`。
TEST_F(TransferPlanTest, StoreWithADifferentDestinationAxisBasisKeepsTheLocalStrides) {
  DescriptorShape shape;
  shape.strides = R"AF(#afir.expr<"(zTb_size * zt_size)">, #afir.expr<"zt_size">, #afir.expr<"1">)AF";
  // 图出口的形态：原始轴、未切 tile，轴表与调度轴 `3, 4, 2` 完全对不上。
  shape.destinationAxis = "0";
  shape.destinationRepeats = R"AF(#afir.expr<"s0">)AF";
  shape.destinationStrides = R"AF(#afir.expr<"1">)AF";
  shape.destinationMemref = "memref<256xf16>";
  shape.indexingMaps = "affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>";
  OwningOpRef<ModuleOp> module = ParseStore(shape);
  ASSERT_TRUE(module);
  Operation *store = FindStore(*module);
  ASSERT_NE(store, nullptr);

  FailureOr<TransferPlan> plan = buildTransferPlan(store);
  ASSERT_TRUE(succeeded(plan)) << "轴基不同不是缺信息，是落点没被调度过——不能因此拒掉整条搬运";
  EXPECT_EQ(plan->kind, TransferPlanKind::Scheduled);
  EXPECT_EQ(OuterAxesOf(*plan),
            (std::vector<std::pair<int64_t, std::string>>{{3, "(zTb_size * zt_size)"}, {4, "zt_size"}}));
}

// NDDMA can retain its input GM strides after broadcasting. A plain output
// descriptor has no vectorized axes, but still owns the Store address layout.
TEST_F(TransferPlanTest, StoreProjectsOriginalDestinationStridesInsteadOfProducerStrides) {
  DescriptorShape shape;
  shape.strides = R"AF(#afir.expr<"0">, #afir.expr<"0">, #afir.expr<"0">)AF";
  shape.destinationAxis = "0";
  shape.destinationRepeats = R"AF(#afir.expr<"s0">)AF";
  shape.destinationStrides = R"AF(#afir.expr<"1">)AF";
  shape.destinationMemref = "memref<256xf16>";
  shape.indexingMaps = "affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>";
  OwningOpRef<ModuleOp> module = ParseStore(shape);
  ASSERT_TRUE(module);
  FailureOr<TransferPlan> plan = buildTransferPlan(FindStore(*module));
  ASSERT_TRUE(succeeded(plan));
  EXPECT_EQ(plan->kind, TransferPlanKind::Scheduled);
  EXPECT_EQ(OuterAxesOf(*plan),
            (std::vector<std::pair<int64_t, std::string>>{{3, "(zTb_size * zt_size)"}, {4, "zt_size"}}));
  ASSERT_EQ(plan->dims.size(), 1U);
  EXPECT_EQ(plan->dims.front().globalStride.getValue().getValue(), "1");
  EXPECT_EQ(plan->dims.front().localStride.getValue().getValue(), "1");
}

// H.1（2026-11-10 剩余工作清单）：局部侧与落点共用同一根轴 id，落点没有
// `vectorized_axis`，但两侧在这根轴上的步长不一致——局部侧（生产者 UB 张量）的步长是
// 广播语义下的 0（这个值不随这根轴变化），落点（真实 GM 输出）在同一根轴上有真实步长
// （每一轮写的是不同一行）。
//
// 旧实现只核对局部侧**向量化轴**自己的两列是否互相一致，核对通过就整份沿用局部侧，
// 根本没有拿落点的步长表比对过非向量化轴。于是这根轴的真实步长被取成 0，在
// `materializeScheduledGlobalOffset` 里被「零 stride 对偏移无贡献」整项剔掉——多份广播
// 输出全部写到同一个 GM 地址，CPU 仿真数值大面积为 0。复现见
// `load_broadcast_abs_store_1x65536`（`[1, 65536] -> [33, 65536]`）。
TEST_F(TransferPlanTest, StoreSharedNonVectorizedAxisTakesTheDestinationStrideNotTheProducersBroadcastStride) {
  DescriptorShape shape;
  // 轴 3（局部侧的外层块轴）在生产者那块 UB 张量上步长为 0：计算结果不随它变化。
  shape.strides = R"AF(#afir.expr<"0">, #afir.expr<"zt_size">, #afir.expr<"1">)AF";
  // 落点在同一根轴上有真实步长，且与真实复现一样没有 `vectorized_axis`。
  shape.destinationStrides = R"AF(#afir.expr<"(zTb_size * zt_size)">, #afir.expr<"zt_size">, #afir.expr<"1">)AF";
  shape.destinationVectorizedAxis = "";
  OwningOpRef<ModuleOp> module = ParseStore(shape);
  ASSERT_TRUE(module);
  Operation *store = FindStore(*module);
  ASSERT_NE(store, nullptr);

  FailureOr<TransferPlan> plan = buildTransferPlan(store);
  ASSERT_TRUE(succeeded(plan));
  EXPECT_EQ(plan->kind, TransferPlanKind::Scheduled);
  EXPECT_EQ(OuterAxesOf(*plan),
            (std::vector<std::pair<int64_t, std::string>>{{3, "(zTb_size * zt_size)"}, {4, "zt_size"}}))
      << "共享轴在生产者那块 UB 张量上步长为 0 不代表落点也是 0，不能因此整项消失";
}

// 落点被调度过、但轴表比局部侧多几根轴：偏移要把多出来的那几根一并算进去。
//
// 这是 RCore 一阶段写 workspace 的形态。归约轴跨在保留轴两侧时（`[R, A, R, R]` 归约
// {0,2,3}），调度器把不相邻的两根归约轴融成一根，落点上于是多出「归约方向的块号」这根轴
// （workspace 的行号，步长 = 一行的长度），而局部侧那块 UB 上它根本不存在——UB 里只有这一
// 个核算出来的一段 partial sum。
//
// 与 `StoreWithADifferentDestinationAxisBasisKeepsTheLocalStrides` 是一对，两条合起来才
// 说明判据是「落点讲不讲调度轴」（看 `vectorized_axis`）而不是「两张轴表逐字相同」。只按
// 后者判会把这一档误判成「落点没被调度过」，退回局部侧算偏移，行号那一项整个消失：所有核
// 往第 0 行写、互相覆盖。真机上表现为结果恰好是真值的 1/行数，工作区只有一行时看不出来。
TEST_F(TransferPlanTest, StoreWithExtraScheduledDestinationAxesTakesTheGlobalAxisTable) {
  DescriptorShape shape;
  // 落点在局部侧的 `zTB, zTb, zt` 之间插进 `yT`（id 6），当行号用，步长是一整行的长度。
  shape.destinationAxis = "3, 6, 4, 2";
  // 行号那一维在张量类型上是 1：每个核只写一行，`foldBlockScatteredDestination` 把它的
  // 范围折成 1 而保留步长。搬运两端的静态元素数因此仍然相等。
  shape.destinationMemref = "memref<2x1x8x16xf16>";
  shape.destinationRepeats =
      R"AF(#afir.expr<"Ceiling((Ceiling((s0 / (zt_size))) / (zTb_size)))">, #afir.expr<"1">, #afir.expr<"zTb_size">, #afir.expr<"zt_size">)AF";
  shape.destinationStrides =
      R"AF(#afir.expr<"(zTb_size * zt_size)">, #afir.expr<"128">, #afir.expr<"zt_size">, #afir.expr<"1">)AF";
  // 落点是被调度过的张量，`vectorized_axis` 与局部侧同为最内那根 tile 轴。
  shape.destinationVectorizedAxis = "[2]";
  shape.destinationVectorizedStrides = R"AF(#afir.expr<"1">)AF";
  shape.indexingMaps = "affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>";
  OwningOpRef<ModuleOp> module = ParseStore(shape);
  ASSERT_TRUE(module);
  Operation *store = FindStore(*module);
  ASSERT_NE(store, nullptr);

  FailureOr<TransferPlan> plan = buildTransferPlan(store);
  ASSERT_TRUE(succeeded(plan));
  EXPECT_EQ(plan->kind, TransferPlanKind::Scheduled);
  EXPECT_EQ(OuterAxesOf(*plan),
            (std::vector<std::pair<int64_t, std::string>>{{3, "(zTb_size * zt_size)"}, {6, "128"}, {4, "zt_size"}}))
      << "只在落点上出现的轴（workspace 行号）也要贡献偏移项，否则所有核写同一行";
}

// 整张 `axis` 表都在 tile 内时回落 identity，不是建一个「偏移恒为 0」的调度计划。
//
// 两者的搬运长度来源不同（identity 取本地张量的整体大小，调度路径取轴尺寸），改道会在
// 未切 tile 的单轴图上换掉产物。这条边界与 `MissingVectorizedAxisFallsBackToIdentityPlan`
// 是一对：一个是"没切 tile"，一个是"切了但 tile 就是整体"。
TEST_F(TransferPlanTest, TileCoveringEveryAxisFallsBackToIdentityPlan) {
  DescriptorShape shape;
  shape.axis = "2";
  shape.memref = "memref<16xf16>";
  shape.repeats = R"AF(#afir.expr<"zt_size">)AF";
  shape.strides = R"AF(#afir.expr<"1">)AF";
  shape.indexingMaps = "affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>";
  OwningOpRef<ModuleOp> module = Parse(shape);
  ASSERT_TRUE(module);
  Operation *load = FindLoad(*module);
  ASSERT_NE(load, nullptr);

  FailureOr<TransferPlan> plan = buildTransferPlan(load);
  ASSERT_TRUE(succeeded(plan));
  EXPECT_EQ(plan->kind, TransferPlanKind::Identity);
}

// `axis` / `repeats` / `strides` 是按下标对齐的几张表，长度不等时「第 i 根轴的 stride」
// 没有意义。`reduceEffectiveView` 在取下标前有一道长度校验，但它从 IR 走不到：
// `TensorDescAttr` 的 verifier 已经把三条长度关系全挡在解析期（归属见
// `test/lit/smoke/afir_attrs_invalid.mlir`）。
//
// 这条用例断言的就是这个归属关系本身。计划构造器里那道校验留着是因为
// `reduceEffectiveView` 也能被 C++ 直接调用，绕过解析——但它不再是这一层的职责，UT 里
// 也造不出能触发它的模块。
TEST_F(TransferPlanTest, MismatchedDescriptorTablesAreRejectedBeforeThePlanBuilder) {
  DescriptorShape shape;
  shape.repeats = R"AF(#afir.expr<"zTb_size">, #afir.expr<"zt_size">)AF";
  shape.strides = R"AF(#afir.expr<"zt_size">, #afir.expr<"1">)AF";
  EXPECT_FALSE(Parse(shape)) << "长度不一致的描述符表应当在 verifier 就被拒";
}

// 两根不连续的向量化轴恰好归约成两项——最内两项直接进 `DataCopyPadExtend`，一层循环都不
// 生成。这是搬运能力的基线形态；多于两项由 `ThreeExtentsEmitOneOuterLoop` 那一组覆盖。
TEST_F(TransferPlanTest, TwoEffectiveExtentsAreSupported) {
  OwningOpRef<ModuleOp> module = Parse(TwoAxisShape(/*innerAxis=*/7, "y1t_size", "s1", "y1t_size"));
  ASSERT_TRUE(module);
  Operation *load = FindLoad(*module);
  ASSERT_NE(load, nullptr);

  FailureOr<TransferPlan> plan = buildTransferPlan(load);
  ASSERT_TRUE(succeeded(plan));
  EXPECT_EQ(DimsOf(*plan), (std::vector<DimTriple>{{"zt_size", "s1", "y1t_size"}, {"y1t_size", "1", "1"}}));
}

TEST_F(TransferPlanTest, MissingVectorizedAxisFallsBackToIdentityPlan) {
  DescriptorShape shape;
  shape.vectorizedAxis = "";
  shape.vectorizedStrides = "";
  OwningOpRef<ModuleOp> module = Parse(shape);
  ASSERT_TRUE(module);
  Operation *load = FindLoad(*module);
  ASSERT_NE(load, nullptr);

  FailureOr<TransferPlan> plan = buildTransferPlan(load);
  ASSERT_TRUE(succeeded(plan));
  EXPECT_EQ(plan->kind, TransferPlanKind::Identity);
}

// 符号闭合门禁的反例：步长用了 `size_var` 之外的符号。
//
// 放行的后果不是报错，是物化时凭空造一个没人赋值的 `afir.tiling_param`，运行期读未初始化
// 内存当地址用。
//
// 这条同时钉住不回落 identity。S4 把项数上限删掉之后，这道闭合门禁是调度路径上仅剩的
// 一处「明确拒绝」，而本夹具两张 `indexing_maps` 默认都是 identity——回落会把一次明确拒绝
// 换成一个搬整块的 identity 计划，正是这套改动要消除的那类静默错误。
TEST_F(TransferPlanTest, StrideSymbolOutsideSizeVarIsRejected) {
  DescriptorShape shape;
  shape.strides = R"AF(#afir.expr<"(rogue_size * zt_size)">, #afir.expr<"zt_size">, #afir.expr<"1">)AF";
  OwningOpRef<ModuleOp> module = Parse(shape);
  ASSERT_TRUE(module);
  Operation *load = FindLoad(*module);
  ASSERT_NE(load, nullptr);

  EXPECT_TRUE(failed(buildTransferPlan(load)));
}

// 正例侧。没有它，门禁写成「一律拒绝」也能让上一条通过。
//
// 同时钉住闭合的成立形式：`size_var` 只列每次切分的 inner 尺寸，outer 是派生的
// `Ceiling(parent / inner)`，所以步长里出现的符号恰好落在 `size_var` 内。
TEST_F(TransferPlanTest, StrideSymbolsWithinSizeVarAreAccepted) {
  DescriptorShape shape;
  shape.strides = R"AF(#afir.expr<"(zTb_size * zt_size)">, #afir.expr<"zt_size">, #afir.expr<"1">)AF";
  OwningOpRef<ModuleOp> module = Parse(shape);
  ASSERT_TRUE(module);
  Operation *load = FindLoad(*module);
  ASSERT_NE(load, nullptr);

  EXPECT_TRUE(succeeded(buildTransferPlan(load)));
}

// 常量步长没有自由符号，与 `size_var` 是否为空无关。缺了这条，门禁写成「`size_var`
// 为空就拒绝」会把全常量形状挡在外面。
TEST_F(TransferPlanTest, ConstantStridesNeedNoSizeVariables) {
  DescriptorShape shape;
  shape.strides = R"AF(#afir.expr<"128">, #afir.expr<"16">, #afir.expr<"1">)AF";
  shape.sizeVar = R"AF(#afir.expr<"s0">, #afir.expr<"zt_size">, #afir.expr<"zTb_size">)AF";
  OwningOpRef<ModuleOp> module = Parse(shape);
  ASSERT_TRUE(module);
  Operation *load = FindLoad(*module);
  ASSERT_NE(load, nullptr);

  EXPECT_TRUE(succeeded(buildTransferPlan(load)));
}

// `emitScheduledTransfer` 是生成器，没有返回值可看。这个夹具把回调收到的那一份
// `ScheduledTileIndices` 抓出来——只在「不会建外层循环」（有效视图 <= 2 项）的用例上
// 这样用才安全：真有循环时这些值落在循环体内，出了回调就不再支配任何使用点。
FailureOr<ScheduledTileIndices> CaptureTileIndices(Operation *op, OpBuilder &builder, const TransferPlan &plan) {
  ScheduledTileIndices captured;
  if (failed(emitScheduledTransfer(op, builder, plan, [&](OpBuilder &, const ScheduledTileIndices &indices) {
        captured = indices;
        return success();
      }))) {
    return failure();
  }
  return captured;
}

// 带某个 `#afir.axis_binding` 的那个值。
Value FindBoundValue(func::FuncOp func, int64_t axis, AxisValueRole role) {
  Value found;
  func.walk([&](Operation *op) {
    if (hasAxisBinding(op, axis, role) && op->getNumResults() == 1) {
      found = op->getResult(0);
    }
  });
  return found;
}

// 两种序号来源各走一遍：block-outer 轴没有 `scf.for`，序号取 `axis_index` 绑定；
// block-inner 轴有循环，序号取归纳变量。
//
// 取错的后果全是静默错值：把循环轴的序号也当成 `axis_index` 去查，在真实 IR 里会命中
// 源轴（被 block 切开的那根 TileOuter）的标记，两者差一个整块偏移。
TEST_F(TransferPlanTest, OffsetSumsBlockIndexAndLoopInduction) {
  OwningOpRef<ModuleOp> module = ParseLowered(DescriptorShape{});
  ASSERT_TRUE(module);
  Operation *load = FindLoad(*module);
  ASSERT_NE(load, nullptr);
  auto func = load->getParentOfType<func::FuncOp>();
  ASSERT_TRUE(func);

  FailureOr<TransferPlan> plan = buildTransferPlan(load);
  ASSERT_TRUE(succeeded(plan));
  OpBuilder builder(load);
  FailureOr<ScheduledTileIndices> indices = CaptureTileIndices(load, builder, *plan);
  ASSERT_TRUE(succeeded(indices));

  // 两项求和，外层在左：`axis` 表的顺序决定加法链的顺序。
  auto sum = indices->globalOffset.getDefiningOp<arith::AddIOp>();
  ASSERT_TRUE(sum);
  auto blockTerm = sum.getLhs().getDefiningOp<arith::MulIOp>();
  auto roundTerm = sum.getRhs().getDefiningOp<arith::MulIOp>();
  ASSERT_TRUE(blockTerm);
  ASSERT_TRUE(roundTerm);

  EXPECT_EQ(blockTerm.getLhs(), FindBoundValue(func, 3, AxisValueRole::AxisIndex));
  auto loop = load->getParentOfType<scf::ForOp>();
  ASSERT_TRUE(loop);
  EXPECT_EQ(roundTerm.getLhs(), loop.getInductionVar());

  // 步长直接来自 `strides` 表，不再由轴尺寸重建。`"zt_size"` 是一个裸符号，物化时直接
  // 绑回入口块那一份 `afir.tiling_param`，不多产一个 `afir.symbolic_eval`。
  auto tilingParam = roundTerm.getRhs().getDefiningOp<TilingParamOp>();
  ASSERT_TRUE(tilingParam);
  EXPECT_EQ(tilingParam.getSymbol(), "zt_size");
  EXPECT_TRUE(blockTerm.getRhs().getDefiningOp<SymbolicEvalOp>())
      << "复合步长 (zTb_size * zt_size) 应当物化成一次符号求值";

  // 长度取向量化轴的 `actual_size` 绑定本身，不再就地重算尾块 select。
  EXPECT_EQ(indices->blockLen, FindBoundValue(func, 2, AxisValueRole::ActualSize));
  // 一维退化：blockCount 恒 1、两侧缝隙恒 0。
  EXPECT_TRUE(matchPattern(indices->blockCount, m_One()));
  EXPECT_TRUE(matchPattern(indices->globalStride, m_Zero()));
  EXPECT_TRUE(matchPattern(indices->localStride, m_Zero()));
}

// 步长为 1 的项省掉乘法：偏移就是序号本身。消掉的是一维路径上全部的新增算术——
// 没有这一条，一维产物会平白多两次乘 1。
TEST_F(TransferPlanTest, UnitStrideTermSkipsTheMultiply) {
  DescriptorShape shape;
  shape.strides = R"AF(#afir.expr<"zt_size">, #afir.expr<"1">, #afir.expr<"1">)AF";
  OwningOpRef<ModuleOp> module = ParseLowered(shape);
  ASSERT_TRUE(module);
  Operation *load = FindLoad(*module);
  ASSERT_NE(load, nullptr);

  FailureOr<TransferPlan> plan = buildTransferPlan(load);
  ASSERT_TRUE(succeeded(plan));
  OpBuilder builder(load);
  FailureOr<ScheduledTileIndices> indices = CaptureTileIndices(load, builder, *plan);
  ASSERT_TRUE(succeeded(indices));

  auto sum = indices->globalOffset.getDefiningOp<arith::AddIOp>();
  ASSERT_TRUE(sum);
  auto loop = load->getParentOfType<scf::ForOp>();
  ASSERT_TRUE(loop);
  EXPECT_EQ(sum.getRhs(), loop.getInductionVar()) << "步长为 1 时这一项应当就是归纳变量本身";
}

// 步长恒为 0 的轴整项不出现。归约的 destination 在被归约轴上就是这个形态——那根轴在
// 输出张量里不动，乘出来恒为 0。
TEST_F(TransferPlanTest, ZeroStrideTermIsDropped) {
  DescriptorShape shape;
  shape.strides = R"AF(#afir.expr<"0">, #afir.expr<"zt_size">, #afir.expr<"1">)AF";
  OwningOpRef<ModuleOp> module = ParseLowered(shape);
  ASSERT_TRUE(module);
  Operation *load = FindLoad(*module);
  ASSERT_NE(load, nullptr);

  FailureOr<TransferPlan> plan = buildTransferPlan(load);
  ASSERT_TRUE(succeeded(plan));
  OpBuilder builder(load);
  FailureOr<ScheduledTileIndices> indices = CaptureTileIndices(load, builder, *plan);
  ASSERT_TRUE(succeeded(indices));

  // 只剩循环轴那一项，没有加法。
  EXPECT_FALSE(indices->globalOffset.getDefiningOp<arith::AddIOp>());
  auto roundTerm = indices->globalOffset.getDefiningOp<arith::MulIOp>();
  ASSERT_TRUE(roundTerm);
  auto loop = load->getParentOfType<scf::ForOp>();
  ASSERT_TRUE(loop);
  EXPECT_EQ(roundTerm.getLhs(), loop.getInductionVar());
}

// 向量化轴上查不到 `actual_size` 时失败，不回落成 `axis_size`。
//
// 回落的后果是尾块多搬一截：越界或读到别的 tile 的数据，编译期完全看不出来。这条用例
// 把它钉成一次失败。
TEST_F(TransferPlanTest, MissingActualSizeBindingFailsInsteadOfGuessing) {
  OwningOpRef<ModuleOp> module = ParseLowered(DescriptorShape{});
  ASSERT_TRUE(module);
  Operation *load = FindLoad(*module);
  ASSERT_NE(load, nullptr);
  auto func = load->getParentOfType<func::FuncOp>();
  ASSERT_TRUE(func);
  Value actual = FindBoundValue(func, 2, AxisValueRole::ActualSize);
  ASSERT_TRUE(actual);
  actual.getDefiningOp()->removeAttr(kAxisBindingAttrName);

  FailureOr<TransferPlan> plan = buildTransferPlan(load);
  ASSERT_TRUE(succeeded(plan));
  OpBuilder builder(load);
  EXPECT_TRUE(failed(CaptureTileIndices(load, builder, *plan)));
}

//===----------------------------------------------------------------------===//
// 有效视图的四条归约规则。每条都成对：正例说明规则会触发，反例说明它不会乱触发。
//
// 只有正例时，把规则写成「无条件执行」也能全绿，而那正是危险的方向——多剔一根轴就是少
// 搬一维，多合一次就是把不连续的两段当成连续的。
//===----------------------------------------------------------------------===//

// 规则 1 正例：范围为 1 的轴不占一维。
TEST_F(TransferPlanTest, UnitExtentAxisIsDropped) {
  // 外层 z0t 的范围写成 1，于是只剩内层 yo 一项。
  DescriptorShape shape = TwoAxisShape(/*innerAxis=*/8, "s1", "s1", "s1", "1", "1", /*outerExtent=*/"1");
  OwningOpRef<ModuleOp> module = Parse(shape);
  ASSERT_TRUE(module);
  Operation *load = FindLoad(*module);
  ASSERT_NE(load, nullptr);

  FailureOr<TransferPlan> plan = buildTransferPlan(load);
  ASSERT_TRUE(succeeded(plan));
  EXPECT_EQ(DimsOf(*plan), (std::vector<DimTriple>{{"s1", "1", "1"}}));
}

TEST_F(TransferPlanTest, InnerUnitPreservesPaddedLocalRows) {
  DescriptorShape shape = TwoAxisShape(/*innerAxis=*/8, "1", "1", "8", "0", "0");
  OwningOpRef<ModuleOp> module = Parse(shape);
  ASSERT_TRUE(module);
  auto plan = buildTransferPlan(FindLoad(*module));
  ASSERT_TRUE(succeeded(plan));
  EXPECT_EQ(DimsOf(*plan), (std::vector<DimTriple>{{"zt_size", "1", "8"}, {"1", "1", "1"}}));
}

TEST_F(TransferPlanTest, InnerUnitPreservesBothRowPitches) {
  DescriptorShape shape = TwoAxisShape(/*innerAxis=*/8, "1", "16", "8", "0", "0");
  OwningOpRef<ModuleOp> module = Parse(shape);
  ASSERT_TRUE(module);
  auto plan = buildTransferPlan(FindLoad(*module));
  ASSERT_TRUE(succeeded(plan));
  EXPECT_EQ(DimsOf(*plan), (std::vector<DimTriple>{{"zt_size", "16", "8"}, {"1", "1", "1"}}));
}

TEST_F(TransferPlanTest, InnerUnitDoesNotSplitContiguousRows) {
  DescriptorShape shape = TwoAxisShape(/*innerAxis=*/8, "1", "1", "1", "0", "0");
  OwningOpRef<ModuleOp> module = Parse(shape);
  ASSERT_TRUE(module);
  auto plan = buildTransferPlan(FindLoad(*module));
  ASSERT_TRUE(succeeded(plan));
  EXPECT_EQ(DimsOf(*plan), (std::vector<DimTriple>{{"zt_size", "1", "1"}}));
}

// 规则 1 反例：范围不是 1 的轴留着。与正例只差 `outerExtent` 一个字段。
TEST_F(TransferPlanTest, NonUnitExtentAxisIsKept) {
  DescriptorShape shape = TwoAxisShape(/*innerAxis=*/8, "s1", "(2 * s1)", "(3 * s1)");
  OwningOpRef<ModuleOp> module = Parse(shape);
  ASSERT_TRUE(module);
  Operation *load = FindLoad(*module);
  ASSERT_NE(load, nullptr);

  FailureOr<TransferPlan> plan = buildTransferPlan(load);
  ASSERT_TRUE(succeeded(plan));
  EXPECT_EQ(DimsOf(*plan), (std::vector<DimTriple>{{"zt_size", "(2 * s1)", "(3 * s1)"}, {"s1", "1", "1"}}));
}

// 规则 2 正例：两侧 stride 都为 0 且前序轴的 UB stride 为 1 时剔除。
//
// 归约算子的 destination 在被归约轴上两侧 stride 都是 0——那一维在输出张量里不存在。
// 这是单张量事实，调度器的 graph-wide `RemoveAllZeroStrideLoopAxis` 看不见（输入
// 张量在同一根轴上 stride 不是 0）。
TEST_F(TransferPlanTest, ZeroStrideAxisIsDropped) {
  // 内层 yo 两侧都是 0；外层 z0t 的 UB stride 是 1，满足 legacy 的剔除条件。
  DescriptorShape shape = TwoAxisShape(/*innerAxis=*/8, "s1", "s1", "1", "0", "0");
  OwningOpRef<ModuleOp> module = Parse(shape);
  ASSERT_TRUE(module);
  Operation *load = FindLoad(*module);
  ASSERT_NE(load, nullptr);

  FailureOr<TransferPlan> plan = buildTransferPlan(load);
  ASSERT_TRUE(succeeded(plan));
  EXPECT_EQ(DimsOf(*plan), (std::vector<DimTriple>{{"zt_size", "s1", "1"}}));
}

// 规则 2 反例：同样是两侧 stride 都为 0，但前序轴的 UB stride 既不是 1 也不是 0，这根轴
// 必须保留。
//
// 这正是 `ShouldIgnoreDataCopyZeroAxis` 存在的理由，legacy 的注释写着「如果前序轴的
// stride 不为 1，则保留当前轴」。写成无条件跳过会让那一维的重复次数整个丢掉——DMA 少搬
// 一维，不报错。
TEST_F(TransferPlanTest, ZeroStrideAxisIsKeptWhenPreviousLocalStrideIsNotUnit) {
  DescriptorShape shape = TwoAxisShape(/*innerAxis=*/8, "s1", "s1", "(3 * s1)", "0", "0");
  OwningOpRef<ModuleOp> module = Parse(shape);
  ASSERT_TRUE(module);
  Operation *load = FindLoad(*module);
  ASSERT_NE(load, nullptr);

  FailureOr<TransferPlan> plan = buildTransferPlan(load);
  ASSERT_TRUE(succeeded(plan));
  EXPECT_EQ(DimsOf(*plan), (std::vector<DimTriple>{{"zt_size", "s1", "(3 * s1)"}, {"s1", "0", "0"}}));
}

// 规则 3 正例：两侧都连续时合并成一项，extent 取乘积、stride 取最内那根的。
//
// 内层 yo：extent=s1，两侧 stride 都是 1。外层 z0t 的两侧 stride 都恰好是 `1 × s1`，
// 于是这两维在两个空间里都是一整块连续内存。
TEST_F(TransferPlanTest, ContiguousAxesAreMerged) {
  DescriptorShape shape = TwoAxisShape(/*innerAxis=*/8, "s1", "s1", "s1");
  OwningOpRef<ModuleOp> module = Parse(shape);
  ASSERT_TRUE(module);
  Operation *load = FindLoad(*module);
  ASSERT_NE(load, nullptr);

  FailureOr<TransferPlan> plan = buildTransferPlan(load);
  ASSERT_TRUE(succeeded(plan));
  EXPECT_EQ(DimsOf(*plan), (std::vector<DimTriple>{{"(s1 * zt_size)", "1", "1"}}));
}

// 规则 3 反例之一：GM 连续但 UB 不连续。两侧都要连续才能合并。
//
// 只看一侧的后果很具体：UB 里两行之间隔着对齐填充，合并之后 DMA 会把填充区也当成数据
// 连着写，下一行的开头被覆盖。
TEST_F(TransferPlanTest, AxesContiguousOnlyInGlobalSpaceAreNotMerged) {
  DescriptorShape shape = TwoAxisShape(/*innerAxis=*/8, "s1", "s1", "(8 + s1)");
  OwningOpRef<ModuleOp> module = Parse(shape);
  ASSERT_TRUE(module);
  Operation *load = FindLoad(*module);
  ASSERT_NE(load, nullptr);

  FailureOr<TransferPlan> plan = buildTransferPlan(load);
  ASSERT_TRUE(succeeded(plan));
  EXPECT_EQ(DimsOf(*plan), (std::vector<DimTriple>{{"zt_size", "s1", "(8 + s1)"}, {"s1", "1", "1"}}));
}

// 规则 3 反例之二：证不出相等就不合并。
//
// 判据是三值的：`s1` 与 `y1t_size` 是两个独立符号，运行期可能相等也可能不等，符号层
// 给出的是 `Unknown`。安全方向是不合并——合错是静默搬错地址，不合只是多一维。字符串比较
// 在这里会给出同样的答案，但在 `(s1 * zt_size)` 与 `(zt_size * s1)` 上会给错，所以判据
// 必须走符号层。
TEST_F(TransferPlanTest, UnprovableContiguityDoesNotMerge) {
  DescriptorShape shape = TwoAxisShape(/*innerAxis=*/8, "y1t_size", "s1", "s1");
  OwningOpRef<ModuleOp> module = Parse(shape);
  ASSERT_TRUE(module);
  Operation *load = FindLoad(*module);
  ASSERT_NE(load, nullptr);

  FailureOr<TransferPlan> plan = buildTransferPlan(load);
  ASSERT_TRUE(succeeded(plan));
  EXPECT_EQ(DimsOf(*plan), (std::vector<DimTriple>{{"zt_size", "s1", "s1"}, {"y1t_size", "1", "1"}}));
}

// 规则 4：内侧紧邻的是 TileInner 时不合并，哪怕两侧都连续。
//
// 与 `ContiguousAxesAreMerged` 只差内层轴的类型：那里是 `yo`（Original，id 8），这里是
// `yt`（TileInner，id 7），两侧 stride 与 extent 的形状完全对应。
//
// 理由是尾块：TileInner 的范围在最后一轮会缩水，而合并后一项只有一个 extent 表达式，
// 表达不了「内层缩了、外层没缩」。合并之后尾块那一轮会多搬一截。
TEST_F(TransferPlanTest, MergeDoesNotCrossTileInnerBoundary) {
  DescriptorShape shape = TwoAxisShape(/*innerAxis=*/7, "y1t_size", "y1t_size", "y1t_size");
  OwningOpRef<ModuleOp> module = Parse(shape);
  ASSERT_TRUE(module);
  Operation *load = FindLoad(*module);
  ASSERT_NE(load, nullptr);

  FailureOr<TransferPlan> plan = buildTransferPlan(load);
  ASSERT_TRUE(succeeded(plan));
  EXPECT_EQ(DimsOf(*plan), (std::vector<DimTriple>{{"zt_size", "y1t_size", "y1t_size"}, {"y1t_size", "1", "1"}}))
      << "内侧是 TileInner，两侧连续也不能合并";
}

//===----------------------------------------------------------------------===//
// 外层循环：有效视图多于两项时，超出的每一项在 DMA 外面生成一层 `scf.for`。
//
// 三条用例是一组：两项（不生成循环）是反例。没有它，实现写成「一律生成一层循环」也能让
// 三项、四项两条通过，而那会让一维与二维搬运平白多一层只跑一轮的循环。
//===----------------------------------------------------------------------===//

// N 根向量化轴、且四条归约规则一条都不触发的降级后形态。两侧 stride 一律取互不相同的
// 符号——这里要钉的是生成，归约有它自己的成对用例。
//
// 每根会缩水的 tile 内轴都要一份 `actual_size` 绑定（真实 IR 里由 `LoopGenerate` 生成），
// 所以 `yt`（id 7）那一份写在循环体序言里。
DescriptorShape OuterLoopShape(size_t vectorizedAxisCount) {
  DescriptorShape shape;
  shape.loopBodyPrologue =
      R"AF(      %yactual = arith.addi %y1t, %c0 {afir.axis_binding = #afir.axis_binding<axis = 7, role = actual_size>} : index)AF";
  // 差集恒为 `[3, 4]`：块轴与块内轴都在 `axis` 表里、都不在 `vectorized_axis` 里。
  const std::string outerAxes = "3, 4, ";
  const std::string outerRepeats =
      R"AF(#afir.expr<"Ceiling((Ceiling((s0 / (zt_size))) / (zTb_size)))">, #afir.expr<"zTb_size">, )AF";
  const std::string outerStrides = R"AF(#afir.expr<"(zTb_size * zt_size)">, #afir.expr<"zt_size">, )AF";
  switch (vectorizedAxisCount) {
    case 2:
      shape.axis = outerAxes + "2, 7";
      shape.memref = "memref<2x8x16x4xf16>";
      shape.repeats = outerRepeats + R"AF(#afir.expr<"zt_size">, #afir.expr<"y1t_size">)AF";
      shape.strides = outerStrides + R"AF(#afir.expr<"s0">, #afir.expr<"1">)AF";
      shape.vectorizedAxis = "[2, 7]";
      shape.vectorizedStrides = R"AF(#afir.expr<"zt_size">, #afir.expr<"1">)AF";
      shape.indexingMaps =
          "affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>";
      break;
    case 3:
      shape.axis = outerAxes + "2, 7, 8";
      shape.memref = "memref<2x8x16x4x16xf16>";
      shape.repeats = outerRepeats + R"AF(#afir.expr<"zt_size">, #afir.expr<"y1t_size">, #afir.expr<"s1">)AF";
      shape.strides = outerStrides + R"AF(#afir.expr<"zTb_size">, #afir.expr<"s0">, #afir.expr<"1">)AF";
      shape.vectorizedAxis = "[2, 7, 8]";
      shape.vectorizedStrides = R"AF(#afir.expr<"y1t_size">, #afir.expr<"zt_size">, #afir.expr<"1">)AF";
      shape.indexingMaps =
          "affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, "
          "affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>";
      break;
    default:
      shape.axis = outerAxes + "2, 7, 8, 9";
      shape.memref = "memref<2x8x16x4x16x8xf16>";
      shape.repeats =
          outerRepeats + R"AF(#afir.expr<"zt_size">, #afir.expr<"y1t_size">, #afir.expr<"s1">, #afir.expr<"s0">)AF";
      shape.strides =
          outerStrides + R"AF(#afir.expr<"s0">, #afir.expr<"s1">, #afir.expr<"zTb_size">, #afir.expr<"1">)AF";
      shape.vectorizedAxis = "[2, 7, 8, 9]";
      // 两侧的步长逐层都不一样，而且顺序对不上（GM 是 s0/s1/zTb_size，UB 是
      // y1t_size/zt_size/s1）。这正是 4.2.6 说的常态：UB 的轴序与 GM 的排布不一致。生成
      // 不受它影响是因为每一项自带两侧步长，循环拿归纳变量分别去乘——只要有一处串了空间，
      // 下面比对两侧步长的断言就会相等。
      shape.vectorizedStrides =
          R"AF(#afir.expr<"y1t_size">, #afir.expr<"zt_size">, #afir.expr<"s1">, #afir.expr<"1">)AF";
      shape.indexingMaps =
          "affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>, "
          "affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>";
      break;
  }
  return shape;
}

// 一次生成过程的观测结果：回调收到的四参与两侧偏移，加上回调落在哪一层循环体里。
struct EmittedTransfer {
  ScheduledTileIndices indices;
  // 最内那层由生成器建出来的 `scf.for`；没有外层循环时是空的。
  SmallVector<scf::ForOp, 2> loops;
};

// 从回调所在的块一路往外数，直到撞见夹具自己那层调度循环（绑定了 `role = loop`）。
// 数出来的就是生成器新建的层数——这正是「层数必须是 N − 2」要钉的量。
FailureOr<EmittedTransfer> EmitAndObserve(Operation *op, OpBuilder &builder, const TransferPlan &plan) {
  EmittedTransfer observed;
  if (failed(emitScheduledTransfer(op, builder, plan, [&](OpBuilder &body, const ScheduledTileIndices &indices) {
        observed.indices = indices;
        for (Operation *parent = body.getInsertionBlock()->getParentOp(); parent != nullptr;
             parent = parent->getParentOp()) {
          auto loop = dyn_cast<scf::ForOp>(parent);
          if (!loop || loop->hasAttr(kAxisBindingAttrName)) {
            break;
          }
          observed.loops.push_back(loop);
        }
        // 外 -> 内。
        std::reverse(observed.loops.begin(), observed.loops.end());
        return success();
      }))) {
    return failure();
  }
  return observed;
}

// 反例。两项进 `DataCopyPadExtend` 的 blockCount / blockLen，一层循环都不该生成。
//
// 没有这条，「一律生成一层」会让一维与二维搬运各多一层只跑一轮的循环——数值仍然对，产物
// 平白多一层嵌套，而上面两条正例发现不了。
TEST_F(TransferPlanTest, TwoExtentsEmitNoOuterLoop) {
  OwningOpRef<ModuleOp> module = ParseLowered(OuterLoopShape(2));
  ASSERT_TRUE(module);
  Operation *load = FindLoad(*module);
  ASSERT_NE(load, nullptr);

  FailureOr<TransferPlan> plan = buildTransferPlan(load);
  ASSERT_TRUE(succeeded(plan));
  ASSERT_EQ(plan->dims.size(), 2U);
  OpBuilder builder(load);
  FailureOr<EmittedTransfer> emitted = EmitAndObserve(load, builder, *plan);
  ASSERT_TRUE(succeeded(emitted));

  EXPECT_TRUE(emitted->loops.empty());
  // 本地侧偏移退化成常量 0：tile 在 buffer 里的起点就是基址。
  EXPECT_TRUE(matchPattern(emitted->indices.localOffset, m_Zero()));
}

// 三项：一层。归约后的项数不再是能力边界——此前超过两项时这里报形状不支持。
TEST_F(TransferPlanTest, ThreeExtentsEmitOneOuterLoop) {
  OwningOpRef<ModuleOp> module = ParseLowered(OuterLoopShape(3));
  ASSERT_TRUE(module);
  Operation *load = FindLoad(*module);
  ASSERT_NE(load, nullptr);
  auto func = load->getParentOfType<func::FuncOp>();
  ASSERT_TRUE(func);

  FailureOr<TransferPlan> plan = buildTransferPlan(load);
  ASSERT_TRUE(succeeded(plan));
  EXPECT_EQ(DimsOf(*plan), (std::vector<DimTriple>{
                               {"zt_size", "zTb_size", "y1t_size"},
                               {"y1t_size", "s0", "zt_size"},
                               {"s1", "1", "1"},
                           }));

  OpBuilder builder(load);
  FailureOr<EmittedTransfer> emitted = EmitAndObserve(load, builder, *plan);
  ASSERT_TRUE(succeeded(emitted));
  ASSERT_EQ(emitted->loops.size(), 1U);
  Value inductionVar = emitted->loops.front().getInductionVar();

  // 上界是最外那一项的实际范围，不是标称尺寸：尾块那一轮少几行。
  EXPECT_EQ(emitted->loops.front().getUpperBound(), FindBoundValue(func, 2, AxisValueRole::ActualSize));
  // GM 侧的行内偏移叠在 tile 起点之上，UB 侧从 0 起算。
  auto globalSum = emitted->indices.globalOffset.getDefiningOp<arith::AddIOp>();
  ASSERT_TRUE(globalSum);
  auto globalTerm = globalSum.getRhs().getDefiningOp<arith::MulIOp>();
  ASSERT_TRUE(globalTerm);
  EXPECT_EQ(globalTerm.getLhs(), inductionVar);
  auto localTerm = emitted->indices.localOffset.getDefiningOp<arith::MulIOp>();
  ASSERT_TRUE(localTerm);
  EXPECT_EQ(localTerm.getLhs(), inductionVar);
  EXPECT_FALSE(emitted->indices.localOffset.getDefiningOp<arith::AddIOp>())
      << "只有一层循环时本地侧偏移只有一项，不该出现加法";
  // 两侧步长各取各的：GM 侧是 `s0`，UB 侧是 `zt_size`，接反了这两个 EXPECT 会互换。
  EXPECT_NE(globalTerm.getRhs(), localTerm.getRhs());

  // 最内两项照旧进 DMA 的 blockCount / blockLen。
  EXPECT_EQ(emitted->indices.blockCount, FindBoundValue(func, 7, AxisValueRole::ActualSize));
}

// 四项：两层。层数是 N − 2，不是「有循环就行」。
TEST_F(TransferPlanTest, FourExtentsEmitTwoOuterLoops) {
  OwningOpRef<ModuleOp> module = ParseLowered(OuterLoopShape(4));
  ASSERT_TRUE(module);
  Operation *load = FindLoad(*module);
  ASSERT_NE(load, nullptr);
  auto func = load->getParentOfType<func::FuncOp>();
  ASSERT_TRUE(func);

  FailureOr<TransferPlan> plan = buildTransferPlan(load);
  ASSERT_TRUE(succeeded(plan));
  EXPECT_EQ(DimsOf(*plan), (std::vector<DimTriple>{
                               {"zt_size", "s0", "y1t_size"},
                               {"y1t_size", "s1", "zt_size"},
                               {"s1", "zTb_size", "s1"},
                               {"s0", "1", "1"},
                           }));

  OpBuilder builder(load);
  FailureOr<EmittedTransfer> emitted = EmitAndObserve(load, builder, *plan);
  ASSERT_TRUE(succeeded(emitted));
  ASSERT_EQ(emitted->loops.size(), 2U);
  EXPECT_EQ(emitted->loops[0].getUpperBound(), FindBoundValue(func, 2, AxisValueRole::ActualSize));
  EXPECT_EQ(emitted->loops[1].getUpperBound(), FindBoundValue(func, 7, AxisValueRole::ActualSize));

  // 两层各自贡献一项，两侧各自累加。漏掉任何一项都表现为每轮搬同一块——数值上整段重复，
  // 文本比对看不出来。
  auto localSum = emitted->indices.localOffset.getDefiningOp<arith::AddIOp>();
  ASSERT_TRUE(localSum);
  auto outerLocal = localSum.getLhs().getDefiningOp<arith::MulIOp>();
  auto innerLocal = localSum.getRhs().getDefiningOp<arith::MulIOp>();
  ASSERT_TRUE(outerLocal);
  ASSERT_TRUE(innerLocal);
  EXPECT_EQ(outerLocal.getLhs(), emitted->loops[0].getInductionVar());
  EXPECT_EQ(innerLocal.getLhs(), emitted->loops[1].getInductionVar());

  auto globalSum = emitted->indices.globalOffset.getDefiningOp<arith::AddIOp>();
  ASSERT_TRUE(globalSum);
  auto innerGlobal = globalSum.getRhs().getDefiningOp<arith::MulIOp>();
  ASSERT_TRUE(innerGlobal);
  EXPECT_EQ(innerGlobal.getLhs(), emitted->loops[1].getInductionVar());
  auto globalPrefix = globalSum.getLhs().getDefiningOp<arith::AddIOp>();
  ASSERT_TRUE(globalPrefix);
  auto outerGlobal = globalPrefix.getRhs().getDefiningOp<arith::MulIOp>();
  ASSERT_TRUE(outerGlobal);
  EXPECT_EQ(outerGlobal.getLhs(), emitted->loops[0].getInductionVar());

  // 同一层的两侧步长取自同一项的两个字段，逐层互不相同（见 `OuterLoopShape` 的注释）。
  // 任何一处把空间串了——比如两侧都用 GM 那一格——这两条就会相等。
  EXPECT_NE(outerGlobal.getRhs(), outerLocal.getRhs());
  EXPECT_NE(innerGlobal.getRhs(), innerLocal.getRhs());
}

// `materializeScheduledLocalSpan` 生成的是一条**左结合**的加法链：`1` 起头，每一项加一个
// `(actual - 1) * stride`。拆回去按 `plan.dims` 的顺序给出每项的两个因子，断言才能逐项
// 说「这一项的 extent 取的是 actual、stride 取的是 nominal」，而不是只数一数有几个乘法。
struct LocalSpanTerm {
  Value actual;
  Value stride;
};

FailureOr<SmallVector<LocalSpanTerm, 4>> DecomposeLocalSpan(Value span, size_t termCount) {
  SmallVector<LocalSpanTerm, 4> terms;
  Value cursor = span;
  for (size_t index = 0; index < termCount; ++index) {
    auto sum = cursor.getDefiningOp<arith::AddIOp>();
    if (!sum) {
      return failure();
    }
    auto distance = sum.getRhs().getDefiningOp<arith::MulIOp>();
    if (!distance) {
      return failure();
    }
    auto reduced = distance.getLhs().getDefiningOp<arith::SubIOp>();
    if (!reduced) {
      return failure();
    }
    terms.push_back(LocalSpanTerm{reduced.getLhs(), distance.getRhs()});
    cursor = sum.getLhs();
  }
  // 链底必须正好是字面量 1：`TensorActualSize` 数的是「首个到末个有效元素」，少了这个 1
  // 就是每一维都少算一个元素。
  if (!matchPattern(cursor, m_One())) {
    return failure();
  }
  std::reverse(terms.begin(), terms.end());
  return terms;
}

FailureOr<SmallVector<LocalSpanTerm, 4>> MaterializeAndDecompose(Operation *load, const TransferPlan &plan) {
  OpBuilder builder(load);
  FailureOr<Value> span = materializeScheduledLocalSpan(load, builder, plan);
  if (failed(span)) {
    return failure();
  }
  return DecomposeLocalSpan(*span, plan.dims.size());
}

// 一维退化：跨度就是最内那一项的 actual extent（`1 + (actual - 1) * 1`）。这条同时说明
// 公式没有为一维走另一条分支——一维与多维是同一条生成路径。
TEST_F(TransferPlanTest, LocalSpanOfOneExtentIsTheInnermostActualExtent) {
  OwningOpRef<ModuleOp> module = ParseLowered(DescriptorShape{});
  ASSERT_TRUE(module);
  Operation *load = FindLoad(*module);
  ASSERT_NE(load, nullptr);
  auto func = load->getParentOfType<func::FuncOp>();
  ASSERT_TRUE(func);

  FailureOr<TransferPlan> plan = buildTransferPlan(load);
  ASSERT_TRUE(succeeded(plan));
  ASSERT_EQ(plan->dims.size(), 1U);
  FailureOr<SmallVector<LocalSpanTerm, 4>> terms = MaterializeAndDecompose(load, *plan);
  ASSERT_TRUE(succeeded(terms));
  ASSERT_EQ(terms->size(), 1U);
  EXPECT_EQ((*terms)[0].actual, FindBoundValue(func, 2, AxisValueRole::ActualSize));
  EXPECT_TRUE(matchPattern((*terms)[0].stride, m_One()));
}

// 多维局部视图的物理跨度：每一项的 extent 取运行期 actual，stride 取标称 tiling
// 参数。尾块只缩行数，不缩 UB 的行距——两者取同一个公式的话，尾块那一轮算出来的跨度会比
// 真实布局短，最后几行算不到，且不报错。
TEST_F(TransferPlanTest, LocalSpanUsesActualExtentAndNominalStride) {
  OwningOpRef<ModuleOp> module = ParseLowered(OuterLoopShape(3));
  ASSERT_TRUE(module);
  Operation *load = FindLoad(*module);
  ASSERT_NE(load, nullptr);
  auto func = load->getParentOfType<func::FuncOp>();
  ASSERT_TRUE(func);

  FailureOr<TransferPlan> plan = buildTransferPlan(load);
  ASSERT_TRUE(succeeded(plan));
  ASSERT_EQ(plan->dims.size(), 3U);
  FailureOr<SmallVector<LocalSpanTerm, 4>> terms = MaterializeAndDecompose(load, *plan);
  ASSERT_TRUE(succeeded(terms));
  ASSERT_EQ(terms->size(), 3U);

  Value axis2Actual = FindBoundValue(func, 2, AxisValueRole::ActualSize);
  Value axis7Actual = FindBoundValue(func, 7, AxisValueRole::ActualSize);
  ASSERT_TRUE(axis2Actual && axis7Actual);
  EXPECT_EQ((*terms)[0].actual, axis2Actual);
  EXPECT_EQ((*terms)[1].actual, axis7Actual);
  // 最外一项的 UB 行距是 `y1t_size`，与轴 7 的 actual 同名不同值。取到 actual 就说明
  // stride 也被尾块替换过了。
  EXPECT_NE((*terms)[0].stride, axis7Actual);
  EXPECT_NE((*terms)[1].stride, axis2Actual);
  EXPECT_TRUE(matchPattern((*terms)[2].stride, m_One()));
}

// 合并之后一项就是一项：两侧都连续的相邻轴在有效视图里已经并成 `(s1 * zt_size)`，跨度
// 公式不需要知道它原来是两根轴。
TEST_F(TransferPlanTest, LocalSpanOfMergedAxesHasOneTerm) {
  OwningOpRef<ModuleOp> module = ParseLowered(TwoAxisShape(/*innerAxis=*/8, "s1", "s1", "s1"));
  ASSERT_TRUE(module);
  Operation *load = FindLoad(*module);
  ASSERT_NE(load, nullptr);

  FailureOr<TransferPlan> plan = buildTransferPlan(load);
  ASSERT_TRUE(succeeded(plan));
  ASSERT_EQ(plan->dims.size(), 1U);
  FailureOr<SmallVector<LocalSpanTerm, 4>> terms = MaterializeAndDecompose(load, *plan);
  ASSERT_TRUE(succeeded(terms));
  ASSERT_EQ(terms->size(), 1U);
  EXPECT_TRUE(matchPattern((*terms)[0].stride, m_One()));
}

// 保留下来的 zero-stride 项对跨度贡献 0：它在 UB 里原地重复，不往后走。项照常生成，因为
// 按项数生成与按项数分解是同一条规则，特判掉反而会让调用方数不准自己有几项。
TEST_F(TransferPlanTest, LocalSpanOfZeroStrideExtentContributesNothing) {
  OwningOpRef<ModuleOp> module = ParseLowered(TwoAxisShape(/*innerAxis=*/8, "s1", "s1", "(3 * s1)", "0", "0"));
  ASSERT_TRUE(module);
  Operation *load = FindLoad(*module);
  ASSERT_NE(load, nullptr);

  FailureOr<TransferPlan> plan = buildTransferPlan(load);
  ASSERT_TRUE(succeeded(plan));
  ASSERT_EQ(plan->dims.size(), 2U);
  FailureOr<SmallVector<LocalSpanTerm, 4>> terms = MaterializeAndDecompose(load, *plan);
  ASSERT_TRUE(succeeded(terms));
  ASSERT_EQ(terms->size(), 2U);
  EXPECT_TRUE(matchPattern((*terms)[1].stride, m_Zero()));
}

// identity 计划没有有效视图可折算，必须失败而不是给一个兜底长度：兜底出来的长度算多算少
// 都不报错。
TEST_F(TransferPlanTest, LocalSpanOfIdentityPlanFails) {
  // 用 `Parse`：`ParseLowered` 无条件生成 `vectorized_axis = `，空表解析不过；而这条判据在
  // 取任何绑定之前就返回，用不上循环生成之后的形态。
  DescriptorShape shape;
  shape.vectorizedAxis = "";
  shape.vectorizedStrides = "";
  OwningOpRef<ModuleOp> module = Parse(shape);
  ASSERT_TRUE(module);
  Operation *load = FindLoad(*module);
  ASSERT_NE(load, nullptr);

  FailureOr<TransferPlan> plan = buildTransferPlan(load);
  ASSERT_TRUE(succeeded(plan));
  ASSERT_EQ(plan->kind, TransferPlanKind::Identity);
  OpBuilder builder(load);
  EXPECT_TRUE(failed(materializeScheduledLocalSpan(load, builder, *plan)));
}

// 即使标称布局连续，内层切分轴的尾块也不能与外层合成连续段。
TEST_F(TransferPlanTest, VectorSegmentsKeepAnInnerTailBoundary) {
  DescriptorShape shape = OuterLoopShape(2);
  shape.vectorizedStrides = R"AF(#afir.expr<"y1t_size">, #afir.expr<"1">)AF";
  OwningOpRef<ModuleOp> module = ParseLowered(shape);
  ASSERT_TRUE(module);
  Operation *load = FindLoad(*module);
  ASSERT_NE(load, nullptr);
  auto segments = collapseVectorizedAxes(load);
  ASSERT_TRUE(succeeded(segments));
  ASSERT_EQ(segments->segments.size(), 2U);
  EXPECT_EQ(segments->segments[0].axes, SmallVector<int64_t>{2});
  EXPECT_EQ(segments->segments[1].axes, SmallVector<int64_t>{7});
  OpBuilder builder(load);
  auto actual = materializeActualSegmentExtent(load, builder, segments->segments[1]);
  ASSERT_TRUE(succeeded(actual));
  auto func = load->getParentOfType<func::FuncOp>();
  EXPECT_EQ(*actual, FindBoundValue(func, 7, AxisValueRole::ActualSize));
}

TEST_F(TransferPlanTest, VectorSegmentRejectsMissingInnerActualSize) {
  OwningOpRef<ModuleOp> module = ParseLowered(OuterLoopShape(2));
  ASSERT_TRUE(module);
  Operation *load = FindLoad(*module);
  ASSERT_NE(load, nullptr);
  auto segments = collapseVectorizedAxes(load);
  ASSERT_TRUE(succeeded(segments));
  ASSERT_EQ(segments->segments.size(), 2U);
  auto func = load->getParentOfType<func::FuncOp>();
  Value actual = FindBoundValue(func, 7, AxisValueRole::ActualSize);
  ASSERT_TRUE(actual);
  actual.getDefiningOp()->removeAttr(kAxisBindingAttrName);
  OpBuilder builder(load);
  EXPECT_TRUE(failed(materializeActualSegmentExtent(load, builder, segments->segments[1])));
}

}  // namespace
