/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- ElementwiseApiFamily.h - shared elementwise lowering criteria -*- C++ -*-===//
//
// 一元、二元、三元三条 elementwise lowering 路径共用的判据。合在一起不是为了少写几行，
// 而是因为**分开写就会分开错**，而且错得不会有人发现：
//
//   - 家族判据答错时，capability 照样回答 Supported，include 照样发得出、key 照样解析
//     得到，只是内联进 kernel 的是另一张供给表的头文本；
//   - tmp buffer 判据两条路松紧不一时，同一份调度会在一元上被拒，在二元上把那块 UB
//     悄悄丢掉。
//
// 三条路的差别只有输入个数，判据本身没有一条应该不一样。
//
//===----------------------------------------------------------------------===//

#ifndef AUTOFUSE_MLIR_CONVERSION_AFIRTOASCENDC_IMPLEMENTATIONS_ELEMENTWISEAPIFAMILY_H_
#define AUTOFUSE_MLIR_CONVERSION_AFIRTOASCENDC_IMPLEMENTATIONS_ELEMENTWISEAPIFAMILY_H_

#include "support/AfirElementTypes.h"
#include "target/Ascend/Capability/CapabilityProvider.h"

#include <optional>
#include <string>
#include <type_traits>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::ascend::kernel::elementwise {

/// 这个 implementation 的调用出自哪张扩展 API 供给表。
///
/// 两档「没有可选项、也就没有要问的 feature」，但它们**不是同一件事**：
///
///   - `Any`：两张表给的是同一个 `AscendC::` 标准调用，一个扩展头都不引；
///   - `Shared`：两张表共用**同一个扩展包装**（同 apiName、同头、同 tmp 函数）。
///     这是对照矩阵的 tier D1：`api_regbase/` 下根本没有对应文件，reg 侧不是「另有
///     一份实现」，而是就用 mem 那份。elementwise 里 5 个——Axpy 与 Add / Mul /
///     Maximum / Minimum 的 scalar⊕scalar 形态（后四个的那一档还没接）。
///
/// 分成两档而不是放宽 `Any`：`Any` 那条「扩展包装总是出自某一张表」仍然要全查，
/// 放宽的话 `logical.h` 那种 mem 专有的包装也能被注册成两侧通用。
enum class ApiFamily {
  Any,
  Shared,
  MemBased,
  RegBased,
};

/// 规范化的操作数 / 目的地 / 结果描述子都在。少一个就说明前面的 pass 没跑完，
/// 这时候答「支持」比答「不支持」更坏。
bool hasCanonicalDescriptors(Operation *op);

// 元素类型：**这里放的是拼写，不是集合。** 每个 implementation 收哪几种类型仍然写在各自
// 的注册处（`afir.neg` 不收 i16、`afir.mul` 收、`afir.gelu` 只收浮点——合成一个的代价是
// 把某一侧悄悄放宽）。共用的只有「怎么把一个 `Type` 对到一位上」这件事，两条路各写一份
// 的话，第一个只在一边加的类型就会让两张表说岔。
//
// **位本身不在这里定义**，在 `support/AfirElementTypes.h` 的 `AFIR_ELEMENT_TYPE_TABLE`
// ——那是「AFIR 的元素类型域有哪几位」的真源，bridge 的白名单、字节宽度、对账断言的拼写
// 表都从同一张表生成。这里只是给注册处一个短名字：注册处写满屏的 `kF16 | kF32`，让它们
// 变成 `kAfirBitF16 | kAfirBitF32` 是一次与需求无关的全仓改名。
//
// 举例说明哪一位几乎只出现在哪一侧：**bf16 在 legacy 里是 reg 侧独有的一位。**
// `ascir_op_lowering_matrix.csv` 的 `mem_dtypes` 里只有 Load / Store / Cast / Concat /
// RemovePad / MatMul 那一族带 bf16，计算族一个都没有；`reg_dtypes` 里有 94 个 op 带它。
// 所以它几乎只出现在 `ApiFamily::RegBased` 的注册处——开在 mem 侧那一份上就是一句没人读
// 的假话，而 `ElementTypeMatrixTest` 查的是「不得比 legacy 窄」，不会拦住写宽了的那一侧。
enum ElementTypes : uint32_t {
#define AFIR_ELEMENTWISE_ELEMENT_TYPE_ALIAS(NAME, BIT, CSV, BUILDER, BYTES, GE, KIND, WIDTH) \
  k##NAME = afir::kAfirBit##NAME,
  AFIR_ELEMENT_TYPE_TABLE(AFIR_ELEMENTWISE_ELEMENT_TYPE_ALIAS)
#undef AFIR_ELEMENTWISE_ELEMENT_TYPE_ALIAS
};

/// `type` 在这张表上的那一位；AFIR 这一层没给它编号就是 0。
uint32_t elementTypeBit(Type type);

/// `allowed` 这个集合收不收 `type`。
///
/// 「没编号」与「编了号但这个 implementation 不收」是两回事，但对调用方是同一个答案。
bool isSupportedElementType(uint32_t allowed, Type type);

/// 目标属于 `family` 这张供给表吗。返回 `nullopt` 表示属于、可以往下走；否则返回的
/// 就是该报出去的那个结论。
///
/// profile 没描述这个 feature 就停在 undetermined，不默认倒向任何一边：猜错的后果是
/// 一侧编译不过、另一侧悄悄少了 tmp buffer。
std::optional<instruction::CapabilitySupport> checkApiFamily(ApiFamily family,
                                                             const instruction::QueryContext &context);

/// 调度分配的 tmp buffer 块数与选中的 API 要的对得上吗。
///
/// 这一条与家族判据是 AND 关系，而且是**双向**的：多给了一块要报 unsupported（不能把
/// 那块 UB 悄悄丢掉），少给了一块也要报——AFIR 造不出这块 buffer，它的大小来自 ATT
/// 侧的聚合，不来自这里。
std::optional<instruction::CapabilitySupport> checkTmpBuffer(Operation *op, bool needsTmpBuffer);

/// 扩展 op 上写死的头文件名与前导头名。标准调用两者都没有，取空。
template <typename TargetOp, typename = void>
struct ApiExtensionHeaderOf {
  static llvm::ArrayRef<llvm::StringRef> get() {
    return {};
  }
  static llvm::ArrayRef<llvm::StringRef> getPreludes() {
    return {};
  }
};
template <typename TargetOp>
struct ApiExtensionHeaderOf<TargetOp, std::void_t<decltype(TargetOp::getExtensionHeaders())>> {
  static llvm::ArrayRef<llvm::StringRef> get() {
    return TargetOp::getExtensionHeaders();
  }
  static llvm::ArrayRef<llvm::StringRef> getPreludes() {
    return TargetOp::getExtensionPreludeHeaders();
  }
};

/// 注册期记下的一条：某个 AFIR op 的某个 implementation 属于哪张供给表、收哪几种元素类型。
struct RegisteredFamily {
  std::string afirOpName;
  std::string id;
  ApiFamily family;
  /// 这个 implementation 收的元素类型集合（`ElementTypes` 的位或）。
  ///
  /// `0` 表示这个 implementation 不按集合判元素类型——`cast` 与 `logical_not` 走的是
  /// `ResultElementRule::CastPair` 的逐行对表，集合在那一档根本不被读。
  uint32_t elementTypes = 0U;
};

/// 记一条。同一个 `(afirOpName, id)` 重复记只留第一条——注册表在一个进程里会被建很多次
/// （每个用例一张），不去重的话这张表按用例个数翻倍。
void recordRegisteredFamily(llvm::StringRef afirOpName, llvm::StringRef id, ApiFamily family, uint32_t elementTypes);

/// 注册期记下的全部条目。
///
/// **这是一张侧表，不是注册表的状态。** `ImplementationRegistry` 是 target 层的通用设施，
/// 它下面还挂着 data movement 这类根本没有家族概念的 implementation；把 `ApiFamily`
/// 塞进它的条目里是让通用层认识 elementwise 的概念。代价是这张表只有「被注册过」这一个
/// 事实，查不到「注册成功没有」——但注册失败时整张注册表都建不起来，两者不会分叉。
///
/// 唯一的读点是 `BridgeOperationRegistryTest` 的按族覆盖断言。
llvm::ArrayRef<RegisteredFamily> registeredFamilies();

/// 这个 AscendC op 建出来的是标准调用：ODS 上没有 `ApiExtensionOpInterface`，一个扩展头
/// 都不引。
///
/// 缺省是「不是」，所以新加的 op 要么声明扩展头，要么在 `ElementwiseStockCalls.h` 上留
/// 一行；两样都没有，注册期就过不去。缺省站这一边的理由见那个文件的头注释：忘了声明
/// 与本来就是标准调用在类型上分不开，只能让忘了的那一侧报错。
template <typename TargetOp>
inline constexpr bool kIsStockCall = false;

/// 这些头名里点到 reg-base 那张表的供给了吗。`xxx_reg_base.h` 这个后缀是它唯一的标记。
inline bool namesRegBasedSupply(llvm::ArrayRef<llvm::StringRef> headers) {
  return llvm::any_of(headers, [](llvm::StringRef header) { return header.ends_with("_reg_base.h"); });
}

inline bool hasAnyHeader(llvm::ArrayRef<llvm::StringRef> headers) {
  return llvm::any_of(headers, [](llvm::StringRef header) { return !header.empty(); });
}

/// 一个 op 声明的**全部**头名（扩展头 + 前导头）与 `family` 对不对得上。
///
/// **这条判据原先有两个前提，`sign` 把两个都推翻了。**
///
/// 前提一「每个头都属于某一族，所以逐项都要对」。不成立：`sign` 两侧分别是
/// `{cast.h, sign.h}` 与 `{cast.h, sign_reg_base.h}`，**`cast.h` 两边是同一个文件**。
/// 两张注册表（`ascendc_api_registry.cpp` 与 `ascendc_reg_base_api_register.cpp`）灌的是
/// 同一个 `AscendCApiRegistry` 单例，`cast.h` 在里面只有一条，reg 侧照样取得到。
/// `broadcast` 的 `duplicate.h` 也是这样。
///
/// 前提二「前导头有各自的族，所以两张列表各判各的」。也不成立：`utils_reg_base.h` 不是
/// `utils.h` 的 reg 版本，它是另一个文件（micro-API 的 `CastTrait` 常量表），不声明
/// `MAX_REPEAT_TIME` / `Rational`。而 reg 侧的 `sign_reg_base.h` 用了 `MAX_REPEAT_TIME`,
/// **它要的就是 mem 那份 `utils.h`**。前导头同样可以是中立的。
///
/// 于是能查的只剩一个方向，而它恰好是要紧的那个：
///
///   - mem 侧的 op **一个** `_reg_base.h` 都不许点名——那是「一份 reg-base 的头文本被
///     内联进 mem kernel」的入口，也是这条判据当初要拦的东西；
///   - reg 侧的 op **至少要点名一个**，否则它根本够不着自己那张表的供给；
///   - `Shared`（tier D1）与 mem 侧同一条：两侧共用的那份包装住在 `api/` 下，
///     `api_regbase/` 下没有对应文件，所以点名任何一个 `_reg_base.h` 都说明它其实
///     不共用；
///   - 其余名字随意：中立的头（`cast.h` / `duplicate.h` / `utils.h`）两侧都能用。
///
/// 反方向（reg 侧的 op 混进一个 mem 专有的包装）查不了：mem 那张表里装的既有 mem 专有的
/// 包装，也有这些中立的头，名字上分不开。legacy 自己就在这么干——reg 侧的 `sign` 点的是
/// `cast.h` 而不是存在的 `cast_reg_base.h`。
inline bool headersMatchFamily(llvm::ArrayRef<llvm::StringRef> headers, llvm::ArrayRef<llvm::StringRef> preludes,
                               ApiFamily family) {
  if (!hasAnyHeader(headers) && !hasAnyHeader(preludes)) {
    return true;  // 标准调用、无前导：没有第二份编码要对。
  }
  if (family == ApiFamily::Any) {
    return false;  // 标准调用不该带扩展头；真的两张表通用要写 `Shared`。
  }
  const bool regBased = namesRegBasedSupply(headers) || namesRegBasedSupply(preludes);
  return regBased == (family == ApiFamily::RegBased);
}

/// `family` 决定选不选中这个 implementation，扩展头名决定往 kernel 里内联哪份文本——
/// 同一件事的两份编码，写反了没有任何一层会报错：capability 照答 Supported，
/// `AddKernelIncludes` 照发那个 include，而那个 key 在供给表里也解析得到，于是一份
/// reg-base 的头文本被内联进一个 mem-based 的 kernel。今天两份 `neg.h` 除 include
/// guard 外逐字相同所以看不出来，`broadcast.h` 那一对不是。
///
/// 所以在注册时就对上——对不上就让整张注册表建不起来，凡是用到它的用例
/// （`BridgeOperationRegistryTest` 起）当场全红。
///
/// **前导头一起进这条判据**，不单独判：理由见上——它可以是中立的。
///
/// **还有一条判据在这里一起查：「我真的是标准调用」这句话必须有人说出来。**
/// 不说的话，「ODS 忘了声明 `ApiExtensionOpInterface`」与「本来就是标准调用」在这里长
/// 得一模一样，两者都是空头名列表，于是上面那条判据直接放行。名单与理由见
/// `ElementwiseStockCalls.h`。
template <typename TargetOp>
LogicalResult checkFamilyMatchesHeader(ApiFamily family) {
  using Headers = ApiExtensionHeaderOf<TargetOp>;
  const bool stockCall = !hasAnyHeader(Headers::get()) && !hasAnyHeader(Headers::getPreludes());
  if (stockCall != kIsStockCall<TargetOp>) {
    return failure();
  }
  return success(headersMatchFamily(Headers::get(), Headers::getPreludes(), family));
}

}  // namespace mlir::ascend::kernel::elementwise

#endif  // AUTOFUSE_MLIR_CONVERSION_AFIRTOASCENDC_IMPLEMENTATIONS_ELEMENTWISEAPIFAMILY_H_
