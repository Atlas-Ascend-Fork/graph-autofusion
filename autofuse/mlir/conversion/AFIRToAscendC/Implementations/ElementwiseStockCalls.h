/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- ElementwiseStockCalls.h - which target ops are stock calls -*- C++ -*-===//
//
// 一句话：这张表是「我真的是标准调用」这句话的唯一说法，说错了整张注册表建不起来。
//
// 为什么非要说：`ApiExtensionHeaderOf` 对没声明 `ApiExtensionOpInterface` 的 op 走的是
// 主模板，返回空头名列表。而「这个 op 本来就是标准调用」与「ODS 忘了声明
// `ApiExtensionOpInterface`」在那里长得一模一样——两者都是空列表。于是家族判据照过，
// `AddKernelIncludes` 一个 include 都不生成，最后炸在设备编译器上。
//
// 注册表原有的「对不上就建不起来」只查了另一个方向（声明了、但写反了族），对这个方向是
// 聋的。补上的就是这一句：注册期要求 `headers.empty() == kIsStockCall<TargetOp>`。
//
// 两个方向都会红：
//
//   - 新 op 的 ODS 忘了声明 `ApiExtensionOpInterface`——头名取空，又不在这张表上，
//     注册失败；
//   - 表上的 op 哪天在 ODS 里加了扩展头——头名非空，注册同样失败。
//
// 表分成两段，判据对两段完全一样，分段只是为了读的人知道它们为什么是标准调用：
// `Any` 那段是两张支持表给同一个 `AscendC::` 调用，另一段是某一侧走包装、另一侧直接用
// 标准调用。
//
//===----------------------------------------------------------------------===//

#ifndef AUTOFUSE_MLIR_CONVERSION_AFIRTOASCENDC_IMPLEMENTATIONS_ELEMENTWISESTOCKCALLS_H_
#define AUTOFUSE_MLIR_CONVERSION_AFIRTOASCENDC_IMPLEMENTATIONS_ELEMENTWISESTOCKCALLS_H_

#include "conversion/AFIRToAscendC/Implementations/ElementwiseApiFamily.h"

#include "ascir/Dialect/Asc/IR/Asc.h"

namespace mlir::ascend::kernel::elementwise {

#define AFIR_ELEMENTWISE_STOCK_CALL(TargetOp) \
  template <>                                 \
  inline constexpr bool kIsStockCall<::mlir::ascendc::TargetOp> = true

// 家族恒为 `Any` 的一段：两张支持表给的是同一个 `AscendC::` 调用。
AFIR_ELEMENTWISE_STOCK_CALL(ExpL2Op);
AFIR_ELEMENTWISE_STOCK_CALL(LnL2Op);
AFIR_ELEMENTWISE_STOCK_CALL(ReluL2Op);
AFIR_ELEMENTWISE_STOCK_CALL(SqrtL2Op);
AFIR_ELEMENTWISE_STOCK_CALL(GeluOp);
AFIR_ELEMENTWISE_STOCK_CALL(LeakyReluL2Op);
AFIR_ELEMENTWISE_STOCK_CALL(AddL2Op);
AFIR_ELEMENTWISE_STOCK_CALL(MulL2Op);
AFIR_ELEMENTWISE_STOCK_CALL(MaxL2Op);
AFIR_ELEMENTWISE_STOCK_CALL(MinL2Op);
AFIR_ELEMENTWISE_STOCK_CALL(AddsL2Op);
AFIR_ELEMENTWISE_STOCK_CALL(MulsL2Op);
AFIR_ELEMENTWISE_STOCK_CALL(MaxsL2Op);
AFIR_ELEMENTWISE_STOCK_CALL(MinsL2Op);

// 只有一侧是标准调用的一段：另一侧走包装。reg 侧八个，mem 侧四个。
AFIR_ELEMENTWISE_STOCK_CALL(AbsL2Op);
AFIR_ELEMENTWISE_STOCK_CALL(ReciprocalL2Op);
AFIR_ELEMENTWISE_STOCK_CALL(SigmoidOp);
AFIR_ELEMENTWISE_STOCK_CALL(IsNanL2Op);
AFIR_ELEMENTWISE_STOCK_CALL(IsFiniteL2Op);
AFIR_ELEMENTWISE_STOCK_CALL(IsInfL2Op);
AFIR_ELEMENTWISE_STOCK_CALL(RsqrtL2Op);
AFIR_ELEMENTWISE_STOCK_CALL(BitwiseAndL2Op);
AFIR_ELEMENTWISE_STOCK_CALL(TanhOp);
AFIR_ELEMENTWISE_STOCK_CALL(ErfOp);
// log2 / floor / lgamma 的 reg 侧使用 CANN 标准 Log2 / Floor / Lgamma，
// 不是扩展包装，必须登记为 stock call。否则 checkFamilyMatchesHeader 拒绝注册表，
// 使 session 创建失败，影响所有依赖该表的 pipeline 测试。
AFIR_ELEMENTWISE_STOCK_CALL(Log2Op);
AFIR_ELEMENTWISE_STOCK_CALL(FloorOp);
AFIR_ELEMENTWISE_STOCK_CALL(LgammaOp);
// 第 7 批。`digamma` / `erfc` / `trunc` 与上面三个同一种情况：legacy 一个包装头都不给
// （`LoadApiHeaderFiles` 没有重载），走的是 `AscendC::Digamma` / `Erfc` / `Trunc`。
// 这一批另外 19 个在 `Adv/Math.td` 也有同名的 `UnaryMathOp`，但它们走 fork 的包装，
// 不在这张表里——照名字往这里加会让那 19 个的 family 判据答反。
AFIR_ELEMENTWISE_STOCK_CALL(DigammaOp);
AFIR_ELEMENTWISE_STOCK_CALL(ErfcOp);
AFIR_ELEMENTWISE_STOCK_CALL(TruncOp);
AFIR_ELEMENTWISE_STOCK_CALL(SubL2Op);
AFIR_ELEMENTWISE_STOCK_CALL(DivL2Op);
// `afir.broadcast` 的原生类型两侧都发它；mem 侧 ui8 单独发 DuplicateU8Op。
// legacy 的 `duplicate.h` 里确实有 `Duplicate`
// 的包装重载，但只覆盖 int64 / uint64 / int8 / uint8——那几种不在该 implementation 收的
// 集合里，生成的就是标准调用（见 Broadcast.cpp 的 `kDuplicateElementTypes`）。
AFIR_ELEMENTWISE_STOCK_CALL(DuplicateL2Op);
// 第 8 批。`log10` / `hypot` 与第 6、7 批那六个（log2 / floor / lgamma / digamma /
// erfc / trunc）同一种情况：legacy 一个包装头都不给，走的是 `AscendC::Log10` /
// `AscendC::Hypot`。两个 op 都是在 Autofuse 子树里新建的标准调用 def，所以这张表与
// 那边的 `ApiExtensionOpInterface`（没声明）是同一件事的两面。
AFIR_ELEMENTWISE_STOCK_CALL(Log10L2Op);
AFIR_ELEMENTWISE_STOCK_CALL(HypotL2Op);
// step 2a。`CastL2Op` 生成的是标准的 `AscendC::Cast`，不引扩展头——取整到整数那一类
// （`floor_to_int` 以及 2b 的 `round_to_int` / `trunc_to_int`）都以它为目标，只把 RoundMode
// 固定住。漏这一行的后果正是这张表上面那段说的：整张注册表建不起来，而报出来的是
// `failed to build the frozen AscendC implementation registry`，离这里隔着好几层。
//
// 注意同目录的 `CastExtendContiguousOp` / `CastExtendContiguousRegBaseOp` / 本批新增的
// `Ceil2IntExtendRegBaseOp` **都不在这张表上**：它们走 fork 的包装、有扩展头。同一个
// `AscendC::Cast` 既有标准调用那一支也有包装那一支，照名字往这里加会把包装那几个答反。
AFIR_ELEMENTWISE_STOCK_CALL(CastL2Op);
// step 3。`FmaL2Op` 是这张表上第一个三元的：走 `AscendC::Fma`（`adv_api/math/fma.h`），
// legacy 的 `FmaAscIrCodegenImplV2` 不覆写 `LoadApiHeaderFiles`，一个包装都没有。
// 与 `HypotL2Op` 逐项同形（标准调用 + 一块 tmp），只是多一个源。前置（store 两端 dtype
// 不同改走 capability 拒收）已接，`abs_fma_bf16_test` 实测干净回落、0 格硬失败。
AFIR_ELEMENTWISE_STOCK_CALL(FmaL2Op);
// `afir.pad` 的包装（`PadOp`）一个扩展头都不引：`AscendC::Pad` 是 CANN 官方 API，
// 头文件（`lib/pad/pad.h`）走 `ApiExtensionHeader` 的**第三段**官方头列表，
// 不占前两段，所以它在「表上的 op 加了扩展头会红」那条判据里长得和标准调用一样。
// legacy 两侧发的是同一个标准调用（tier D1），归第一段「两侧同一个 AscendC:: 调用」。
AFIR_ELEMENTWISE_STOCK_CALL(PadOp);

#undef AFIR_ELEMENTWISE_STOCK_CALL

}  // namespace mlir::ascend::kernel::elementwise

#endif  // AUTOFUSE_MLIR_CONVERSION_AFIRTOASCENDC_IMPLEMENTATIONS_ELEMENTWISESTOCKCALLS_H_
