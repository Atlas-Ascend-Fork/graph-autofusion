/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- AFIRElementwiseSupport.h - Shared elementwise op bodies --*- C++ -*-===//
//
// elementwise Op 的方法体由 ODS 里的 `AFIR_Bufferizable*ElementwiseOp` 生成
// （见 Ops/Elementwise.td），生成的代码落在 AFIRDialect.cpp，真正的实现在
// Ops/Elementwise.cpp。这个头文件就是两者之间的接缝。
//
// 为什么用生成而不是手写：一个 bufferizable elementwise Op 需要 verify、getRole、
// getTiedResultIndex、getEffects、getSpeculatability、inferReturnTypes 六个方法，同一
// arity 下它们逐字相同。手抄的话每加一个 op 就多六份拷贝，而且抄错（比如漏掉
// `isBufferizedForm()` 那一支）不会编译报错，只会静默放行。
//
//===----------------------------------------------------------------------===//

#ifndef AUTOFUSE_MLIR_DIALECT_AFIR_IR_AFIRELEMENTWISESUPPORT_H
#define AUTOFUSE_MLIR_DIALECT_AFIR_IR_AFIRELEMENTWISESUPPORT_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::afir::elementwise {

using MemoryEffect = SideEffects::EffectInstance<MemoryEffects::Effect>;

// tensor 形态：操作数与结果都必须是 tensor，且结果形状等于各输入形状的广播。
LogicalResult verifyLogicalForm(Operation *op);

// buffer 形态：不允许出现 tensor，结果与 destination 同型，destination 形状等于各输入
// 的广播。tmp buffer 操作数不参与其中任何一条——它是字节，不是计算里的一块 tile。
LogicalResult verifyBufferForm(Operation *op, ValueRange inputs, Value destination);

/// 调度形态下的单输入广播：source 可以只在非向量化外轴上比 destination 小。
///
/// 这是 buffer-form 的**调度语义**，不是 tensor-form 的结果类型推导：scheduler 可能删掉
/// redundant Broadcast，把「广播视图 + unary」折叠成「小 source tile + full destination」。
/// 判据由 canonical descriptor 钉住——差异轴必须 stride 为 0，且不能落在 source 或
/// destination 的向量化轴上；否则 AscendC API 会按 destination 的窗口读出小 tile。
bool sourceBroadcastsToDestination(Operation *op, Value source, Value destination, const char **reason = nullptr);

// buffer 形态的内存效果：输入读，destination 写，tmp buffer 也算写。tmp buffer 那条不能省
// ——没有它，两个共享同一块 tmp buffer 的 Op 之间没有任何东西挡住重排。
void getBufferFormEffects(Operation *op, MutableOperandRange destination, MutableOperandRange tmpBuffer,
                          SmallVectorImpl<MemoryEffect> &effects);

// tensor 形态的结果类型：形状取各输入的广播；元素类型用 `resultElementType`，传空则
// 取第一个输入的。无秩输入返回第一个输入的类型（元素类型按同一规则替换）。
//
// 元素类型要显式传，是因为这一族里有结果 dtype 与输入不同的 op（谓词族出 `ui8`）：
// 「取第一个输入的」对它们是错的，而错的表现是 tensor 形态静默造出 `f32` 结果，
// 再被 ODS 的类型约束在别处报成一个看不懂的错。
LogicalResult inferLogicalReturnType(ValueRange inputs, Type resultElementType,
                                     SmallVectorImpl<Type> &inferredReturnTypes);

/// 结果元素类型与输入是什么关系。**一条规则，两个读点**：
///
///   - ODS 的 tensor 形态结果推导（`AFIR_BufferizableElementwiseMethods` 的
///     `inferReturnTypes`）；
///   - lowering 的 capability 判据（三条 `*Elementwise.cpp` 的
///     `areElementTypesSupported`）。
///
/// 分开写就会分开错，而错法是静默的：一边推出 f32、另一边只认「目的地与输入同 dtype」，
/// op 建得出来但永远选不出 implementation。原先这条关系写在三处——ODS 的
/// `logicalResultElementType` 代码块、ODS 的 `sameElementType` 位、conversion 各自的
/// `ResultElementType` 枚举，`truediv` 是唯一共用了一个函数的。现在三处共用这一个枚举：
/// ODS 的那一位由它推导（见 `Ops/Elementwise.td`），conversion 直接用它。
enum class ResultElementRule {
  /// 结果与输入同 dtype。ODS 那一位因此是 1（`AllElementTypesMatch` 把结果也绑进去）。
  Same,
  /// 结果恒为 `ui8`，与输入 dtype 无关：谓词族、逻辑族、比较族。
  Uint8,
  /// 结果恒为 **signless i32**，与输入 dtype 无关：取整到整数那一族（`ceil2int` /
  /// `floor_to_int`）。与 `Uint8` 逐项同形，只换位宽与有无符号——AFIR 的 i32 是无符号性的
  /// （`isSignlessInteger(32)`），而 `Uint8` 那一位是 `IntegerType::Unsigned`，两者不是
  /// 同一种构造，抄错了 ODS 会拒收而且报的是「属性/类型不满足约束」那种看不出所以然的错。
  ///
  /// **`round_to_int` / `trunc_to_int` 不走这一档**：它们的合法组合是一张逐对对表
  /// （f32->i64 / f16->i8 / ... ），归 `CastPair`。
  Int32,
  /// i32 的结果是 f32，其余跟着输入。只有 `afir.truediv`，legacy 的 `kTrueDivTypePairs`
  /// 就是这三行。
  SameOrF32ForI32,
  /// 合法组合是一张逐行对表，**推不出来**：`afir.cast`。`inferResultElementType` 对这一
  /// 档返回空 `Type`，ODS 那边因此不给 cast 取 `inferReturnTypes`（理由见 Elementwise.td
  /// 的 `AFIR_BufferizableElementwiseTraits`），capability 那边走 `isSupportedCastPair`。
  CastPair,
};

/// 按 `rule` 从输入元素类型推结果元素类型。收的可以是元素类型，也可以是整个 shaped type。
/// `CastPair` 推不出来，返回空 `Type`。
Type inferResultElementType(MLIRContext *context, Type input, ResultElementRule rule);

/// `rule` 下 `input -> result` 这一对合法吗。`CastPair` 不走这里（它要知道是哪一侧的对表,
/// 见 `isSupportedCastPair`）。
bool matchesResultElementRule(Type input, Type result, ResultElementRule rule);

/// `afir.cast` 的 (输入, 输出) 对表出自哪一侧。两张表不是包含关系：mem 独有 i4 那一族
/// 3 对，reg 独有 i8 / i64 那两族 10 对。
enum class CastSupply { MemBased, RegBased };

/// `supply` 那一侧供得起 `input -> result` 这一对吗。收的可以是元素类型，也可以是整个
/// shaped type。
///
/// **两个读点**：lowering 的 capability 判据（`UnaryElementwise.cpp` 的
/// `ResultElementType::CastPair`）按侧查这一个，ODS 的 `verify()` 查并集
/// （`verifyCastPair`）。两处共用 `Ops/Cast.cpp` 里那两张表，理由与
/// `trueDivResultElementType` 逐字相同。
bool isSupportedCastPair(CastSupply supply, Type input, Type result);

/// ODS 的 `verify()` 用的那一条：并集里有没有这一对。有哪一侧供得起就放行，
/// 选哪一侧是 lowering 的事。
LogicalResult verifyCastPair(Operation *op, Type input, Type result);

/// `afir.logical_not` 的 (输入, 结果) 关系。**两侧给的不是同一条**，这是继 `truediv`
/// 之后第二个这样的 op，也是一元里第一个：
///
///   - mem 侧的包装 `LogicalNot(LocalTensor<T> dst, LocalTensor<T> src, ...)` 两端同
///     dtype，legacy 的 `kLogicalNotTypePairs`（v1）逐行都是 `T -> T`；
///   - reg 侧的包装 `LogicalNotExtend(LocalTensor<uint8_t> dst, LocalTensor<T> src, ...)`
///     结果恒为 `ui8`，v2 那张对表逐行都是 `T -> ui8`。
///
/// 所以结果元素类型既不是常量也不是输入的函数，`InferTypeOpInterface` 推不出来——与
/// `afir.cast` 同一种，处理办法也照它：不声明那条接口，自己拼 `verify()`。这里查的是
/// 并集，逐侧那一半由 lowering 的 `ResultElementType` 说
/// （`Same` / `Uint8Logical`）。
LogicalResult verifyLogicalNotPair(Operation *op, Type input, Type result);

}  // namespace mlir::afir::elementwise

#endif  // AUTOFUSE_MLIR_DIALECT_AFIR_IR_AFIRELEMENTWISESUPPORT_H
