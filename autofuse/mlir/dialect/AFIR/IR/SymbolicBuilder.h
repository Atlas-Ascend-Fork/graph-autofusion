/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_MLIR_DIALECT_AFIR_IR_SYMBOLICBUILDER_H
#define AUTOFUSE_MLIR_DIALECT_AFIR_IR_SYMBOLICBUILDER_H

#include "dialect/AFIR/IR/AFIR.h"
#include "llvm/ADT/MapVector.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::afir {

using SymbolBindingMap = llvm::MapVector<ExprAttr, Value>;

FailureOr<ExprAttr> makeExprAttr(MLIRContext *context, StringRef canonicalExpression);

FailureOr<Value> materializeSymbolicExpr(OpBuilder &builder, Location location, ExprAttr expression,
                                         const SymbolBindingMap &bindings, Type resultType = {});
// `lhs * rhs` 的规范形式。
//
// **不能拼字符串再 Parse**：`Expression::Parse` 要求输入已经是规范形式，`"((1) * (32))"`
// 会被它以「expression is not canonical; canonical form is '32'」拒掉，表现成判据不成立
// 而不是报错。所以走 `Substitute`：模板本身是规范的，替换值来自现成的 `ExprAttr`（也已规范），
// 代进去由 SymEngine 自己得到规范结果。
//
// 两个消费者：合轴的 stride 连续性判据（`VectorSegmentation.cpp`）与搬运偏移的强度削减
// （`TransferPlan.cpp`）——两处问的是同一件事「外侧 stride 是不是内侧 stride × 内侧长度」。
FailureOr<ExprAttr> multiplyExprs(MLIRContext *context, ExprAttr lhs, ExprAttr rhs);

// 将 extent 向上对齐到 alignNum 的整数倍，alignNum 必须为正。
// 通过规范表达式模板替换构造 alignNum * Ceiling(extent / alignNum)，用于物理跨度校验。
FailureOr<ExprAttr> alignUpExpr(MLIRContext *context, ExprAttr extent, int64_t alignNum);

// 相邻两根轴（外、内）上的步长**连续**吗：`外 == 内 × 内侧长度`，可证时为真。
//
// **这是三处共用的那一条判据**，不是一个顺手的小工具：
//
//   - 合轴（`VectorSegmentation.cpp`）：对所有参与张量都成立才把两轴并进一段；
//   - 搬运偏移的强度削减（`TransferPlan.cpp` 的 `axisUnitStride`）：对各支成立才把整棵
//     子树折成一条乘法；
//   - ND-DMA 的分段（`Implementations/Nddma.cpp`）：GM 侧不成立就在这里把一段切开。
//
// 三处问的是同一句话。各写一份的后果不是算错，是哪天放宽了其中一处而别处没动——于是同一个
// 形状在合轴上判连续、在段步长上判不连续，而两边都不会报错。
//
// 判等先比规范串，比不出来再上证明器。乘法走 `multiplyExprs`，**不能拼字符串造乘积**（见上）。
bool provablyContiguous(MLIRContext *context, ExprAttr outerStride, ExprAttr innerStride, ExprAttr innerExtent);

// 入口块顶部那一份 `afir.tiling_param`，没有就新建。
//
// 「顶部」不是风格选择：这些值是函数里所有运行期尺寸的根，任何消费者——循环生成物化
// 轴尺寸、资源物化算缓冲字节数、`memref.alloc(%n)` 的动态尺寸——都排在它们之后，放在
// 入口块顶部是唯一不需要每个消费者各自操心支配关系的位置。
//
// 一个名字一份：`Pure` 让 CSE 也能合并重复的，但依赖 CSE 意味着在它跑之前 IR 里就有
// 两份同名的值，读 IR 的人要先判断它们是不是同一个东西。这里直接不产生第二份。
//
// 取 `Block &` 而不是 `func::FuncOp`：本文件属于 `AFIRDialect`，它不链接
// `MLIRFuncDialect`，而调用方（bridge、conversion、AFIR 层 pass）各自都拿得到入口块。
TilingParamOp getOrCreateTilingParam(Block &entry, StringRef symbol);

// 同上，但**新建那一条时用调用方的 `builder`**（插入点临时挪到入口块顶部，返回前还原）。
//
// 给 conversion 用。那里的 `builder` 是 `ConversionPatternRewriter`，用它建出来的 Op 才
// 进得了 dialect conversion 的建档：要么被 legalize，要么随 pattern 回滚。自己开一个
// `OpBuilder` 建的那条谁都看不见——`applyFullConversion` 连它非法都不知道，会把它原样
// 留在下降完的 IR 里（实测：descriptor 里出现一个上游没物化过的符号时，下降完的函数里
// 留着一条 `afir.tiling_param`，一条诊断都没有）。
TilingParamOp getOrCreateTilingParam(OpBuilder &builder, Block &entry, StringRef symbol);

// 把一个运行期尺寸表达式物化成入口块顶部的一个 index 值。
//
// 自由符号逐个变成 `afir.tiling_param`，表达式本身变成一个 `afir.symbolic_eval`，两者
// 都落在入口块的「尺寸序言」里——即块首那一段连续的 `afir.tiling_param` /
// `afir.symbolic_eval`。放在那里，结果支配函数体里的任何消费者，调用方不必再自己挑
// 插入点；循环生成把这一段整体当作序言看待，不进循环树。
//
// 常量表达式（`"16"`）不归这里：它不需要任何运行期参数，调用方直接建常量即可，绕开
// 本函数也就绕开了一次没有操作数的 `afir.symbolic_eval`。
FailureOr<Value> materializeRuntimeSize(Block &entry, Location location, ExprAttr expression);

// 与 `materializeRuntimeSize` 一样把自由符号绑到入口块的 `afir.tiling_param`，但表达式
// **在 `builder` 的当前插入点**物化，不进尺寸序言。
//
// 给 conversion 用：那里的插入点由 `ConversionPatternRewriter` 掌管，把 Op 建到别处会绕开
// 它的建档，pattern 失败时回滚不到。符号绑定仍然走入口块——`afir.tiling_param` 本来就该在
// 那里，而且循环生成那一步已经把同一批符号物化过，`getOrCreateTilingParam` 一个名字一份，
// 这里取到的多半就是那一份。**「多半」不是「一定」**：`AFIRMaterializeTilingParamsPass`
// 收的是 `size_var` 与 `afir.buffer_desc` 的符号，descriptor 里的 stride / extent 不在其中，
// 所以真会有就地新建的时候——那一条也必须由 `builder` 建，理由同 `getOrCreateTilingParam`
// 的那个重载。
//
// **在 conversion 里用它，那个 pass 必须自带 `populateSymbolicLoweringPatterns`**：非字面量的
// 表达式会发一条 `afir.symbolic_eval`，而进 conversion 之前这个 op 已经被
// `AFIRMemoryAllocatePass` 清空了，那一层不会顺手替你收尾。少了它，新建的 op 没人 legalize，
// dialect conversion 会把整个 pattern 回滚、报在根算子上，看起来像判据写错了。
FailureOr<Value> materializeSizeAt(OpBuilder &builder, Block &entry, Location location, ExprAttr expression);

// descriptor 里的一个 **extent** 表达式落成 `index` 值，物化点是 `builder` 的当前插入点。
//
// 字面量走常量：绕开一次没有操作数的 `afir.symbolic_eval`，也与
// `MemoryAllocate/CanonicalResources.cpp` 的 `materializeExpression` 同一条判据。
// 字面量必须**为正**——extent 为 0 的 descriptor 是错的，放过去就是发一条 size 为 0 的调用。
FailureOr<Value> materializeExtentValue(Operation *op, OpBuilder &builder, ExprAttr expression);

// 同上，但收 0：**stride 可以是 0**（广播轴上每趟读同一份）。
//
// 与 `materializeExtentValue` 分开只为一件事：合用一个就得把「必须为正」那条判据放宽，
// 那样 extent 为 0 的错误 descriptor 也会被放过去。
FailureOr<Value> materializeStrideValue(Operation *op, OpBuilder &builder, ExprAttr expression);

}  // namespace mlir::afir

#endif  // AUTOFUSE_MLIR_DIALECT_AFIR_IR_SYMBOLICBUILDER_H
