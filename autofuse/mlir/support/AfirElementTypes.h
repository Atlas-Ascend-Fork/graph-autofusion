/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- AfirElementTypes.h - AFIR 元素类型供给表的真源 ----------*- C++ -*-===//
//
// **「AFIR 的元素类型域有哪几位」只写在这里一遍。**
//
// 改造之前这件事写了四遍，而且四遍的拼写、顺序、口径各不相同：
//
//   1. `Implementations/ElementwiseApiFamily.h` 的 `ElementTypes` 位；
//   2. `bridge/ScheduleToAfir.cpp` 的 `IsAfirModeledDType`（一张 `ge::DataType` 的白名单）；
//   3. 同一个文件里的 `DTypeStr` 与 `ElementByteWidth`（两张各自独立的 switch）；
//   4. `test/ut/conversion/ElementTypeMatrixTest.cpp` 的 `kModelableDtypes`。
//
// **漏掉其中任何一处都不会红**，而且四种漏法的症状完全不同，所以每一次都要重新查：
//
//   - 漏 1：implementation 收不到那一位，图建得出来但选不出 implementation——干净回落，
//     只是白回落；
//   - 漏 2：bridge 当场拒掉，连图都不建——也是干净回落，但首因标签指向 dtype，把真正的
//     卡点盖住（i64 那 11 格就是这么被盖了一整期）；
//   - 漏 3（`ElementByteWidth`）：**算不出字节数**，workspace 尺寸表达式缺项；
//   - 漏 4：那条「不得比 legacy 窄」的断言对新那一位**恒真**——它不会红，只是不再查。
//
// 所以这里的两张表是**位置表**，不是「谁收哪几种」：后者是每个 op 自己的事实，写在各自的
// 注册处，由 `ElementTypeMatrixTest` 对着 `docs/guides/ascir_op_lowering_matrix.csv` 对账。
// 这里只回答「AFIR 这一层表达得出哪几种元素类型」。
//
// **加一位 = 把一行从 `AFIR_ELEMENT_TYPE_NAMED_ONLY` 搬到 `AFIR_ELEMENT_TYPE_TABLE`。**
// 两张表分开的理由见 `AFIR_ELEMENT_TYPE_NAMED_ONLY` 上面那段——「叫得出名字」与「建模得
// 出来」是两个问题，合成一张表会把拒收诊断里的那个类型名弄丢。
//
//===----------------------------------------------------------------------===//

#ifndef AUTOFUSE_MLIR_SUPPORT_AFIRELEMENTTYPES_H
#define AUTOFUSE_MLIR_SUPPORT_AFIRELEMENTTYPES_H

#include <cstdint>
#include <optional>
#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

namespace mlir::afir {

/// 一位元素类型在 MLIR 这一侧长什么样。
///
/// 只有五种，而不是「每个 dtype 一个」：整数那两类由 `width` 区分，所以加一位整数不需要
/// 在这里加枚举值，也不需要改下面任何一个 switch。浮点三种在 MLIR 里是三个独立的类，
/// 宽度推不出来（`f16` 与 `bf16` 同宽），所以各占一个。
enum class AfirElementTypeKind : uint8_t {
  SignlessInteger,
  UnsignedInteger,
  Float16,
  Float32,
  BFloat16,
};

// AFIR 建模得出来的元素类型。每一行的列依次是：
//
//   NAME     位的名字。`ElementTypes` 的 `k<NAME>` 与它同名。
//   BIT      位号。显式写出来而不靠行序，是为了让快照式断言（`kKnownNarrower`）在表里
//            插一行时不跟着变。**位号不得复用**：旧位号出现在测试快照里。
//   CSV      `ascir_op_lowering_matrix.csv` 两列 dtype 的拼写。对账靠它，不能自己另起。
//   BUILDER  `AfirBuilder::parseDtype` 收的长拼写，也是 bridge 发给它的那个字符串。
//   BYTES    一个元素占几字节。**这是真源，不是 `getIntOrFloatBitWidth() / 8`**——理由见
//            下面 `afirElementByteWidth` 那段。
//   GE       调度侧 `ge::DataType` 的枚举名。这里只是一个 token，不引 ge 的头；要用它的
//            地方自己拼 `ge::` 前缀（只有 bridge 拼得出来，也只有 bridge 需要）。
//   KIND     上面那个枚举的值。
//   WIDTH    MLIR 整数类型的位宽；浮点那三行不读这一列，填它的实际位宽以免误导。
#define AFIR_ELEMENT_TYPE_TABLE(M)                                \
  M(F16, 0, "f16", "float16", 2, DT_FLOAT16, Float16, 16)         \
  M(F32, 1, "f32", "float32", 4, DT_FLOAT, Float32, 32)           \
  M(I16, 2, "i16", "int16", 2, DT_INT16, SignlessInteger, 16)     \
  M(I32, 3, "i32", "int32", 4, DT_INT32, SignlessInteger, 32)     \
  M(UI8, 4, "ui8", "uint8", 1, DT_UINT8, UnsignedInteger, 8)      \
  M(UI16, 5, "ui16", "uint16", 2, DT_UINT16, UnsignedInteger, 16) \
  M(BF16, 6, "bf16", "bfloat16", 2, DT_BF16, BFloat16, 16)

// legacy 叫得出名字、但 AFIR 这一层还没建模的元素类型。列依次是 CSV / BUILDER / BYTES / GE，
// 含义与上面那张表同名的列一样。
//
// **字节数这一列这里也要有**，因为 `ElementByteWidth` 答的范围比供给表宽：workspace 的尺寸
// 表达式对一个 ui32 的张量也得算得出来，哪怕没有哪个 op 收 ui32。把这一列省掉的后果是那种
// 图报 `InvalidWorkspace: workspace peer tensor has unsupported dtype or size`——一条和 dtype
// 建模毫无关系的错。
//
// **这张表的唯一用途是让拒收诊断说得出那个类型名。** 合进上面那张表会同时丢掉两件事：
// 「叫得出名字」与「建模得出来」的区分（`DTypeStr` 与 `IsAfirModeledDType` 答的是两个
// 问题），以及
//
//     UnsupportedOperation: element type 'int64' is not in the AFIR supply table
//
// 这条诊断里的 `int64`——换成「不认识的 dtype 值 9」之后，接管表的首因列就再也指不出是
// 哪一位缺了，而那一列正是排期靠的东西。
//
// **这里缺一行的后果是报出 `scheduler dtype value N is not modeled by AFIR`**（bool 那 4
// 格今天就是这样），比缺名字更难查一层。
#define AFIR_ELEMENT_TYPE_NAMED_ONLY(M)                  \
  M("i8", "int8", 1, DT_INT8, SignlessInteger, 8)        \
  M("i64", "int64", 8, DT_INT64, SignlessInteger, 64)    \
  M("ui32", "uint32", 4, DT_UINT32, UnsignedInteger, 32) \
  M("ui64", "uint64", 8, DT_UINT64, UnsignedInteger, 64)

/// 两张表共用的那几列，展开成数据。`csvSpelling` / `builderSpelling` / `byteWidth` /
/// `kind` / `width` 的含义与表上同名的列一样；`bit` 只有建模得出来的那几位有，另一张表
/// 上的行取 0。
struct AfirNameableElementType {
  llvm::StringRef csvSpelling;
  llvm::StringRef builderSpelling;
  int64_t byteWidth;
  AfirElementTypeKind kind;
  unsigned width;
};

/// 一位的位掩码。`ElementTypes` 的 `k<NAME>` 由它生成。
enum AfirElementTypeBits : uint32_t {
#define AFIR_ELEMENT_TYPE_BIT(NAME, BIT, CSV, BUILDER, BYTES, GE, KIND, WIDTH) kAfirBit##NAME = 1U << (BIT),
  AFIR_ELEMENT_TYPE_TABLE(AFIR_ELEMENT_TYPE_BIT)
#undef AFIR_ELEMENT_TYPE_BIT
};

/// 表里的一行，展开成数据。遍历用。
struct AfirElementTypePosition {
  llvm::StringRef csvSpelling;
  llvm::StringRef builderSpelling;
  uint32_t bit;
  int64_t byteWidth;
  AfirElementTypeKind kind;
  unsigned width;
};

inline llvm::ArrayRef<AfirElementTypePosition> afirElementTypePositions() {
  static const AfirElementTypePosition positions[] = {
#define AFIR_ELEMENT_TYPE_ROW(NAME, BIT, CSV, BUILDER, BYTES, GE, KIND, WIDTH) \
  {CSV, BUILDER, kAfirBit##NAME, BYTES, AfirElementTypeKind::KIND, WIDTH},
      AFIR_ELEMENT_TYPE_TABLE(AFIR_ELEMENT_TYPE_ROW)
#undef AFIR_ELEMENT_TYPE_ROW
  };
  return positions;
}

/// 叫得出名字的全部元素类型：建模得出来的那几位 + 还没建模的那几位。
///
/// 解析拼写、算字节数这类「只要认得出来就行」的事用它；「收不收」用
/// `afirElementTypePositions()`。两者分开是因为拒收诊断要印那个类型名。
inline llvm::ArrayRef<AfirNameableElementType> afirNameableElementTypes() {
  static const AfirNameableElementType entries[] = {
#define AFIR_NAMEABLE_MODELED(NAME, BIT, CSV, BUILDER, BYTES, GE, KIND, WIDTH) \
  {CSV, BUILDER, BYTES, AfirElementTypeKind::KIND, WIDTH},
      AFIR_ELEMENT_TYPE_TABLE(AFIR_NAMEABLE_MODELED)
#undef AFIR_NAMEABLE_MODELED
#define AFIR_NAMEABLE_NAMED_ONLY(CSV, BUILDER, BYTES, GE, KIND, WIDTH) \
  {CSV, BUILDER, BYTES, AfirElementTypeKind::KIND, WIDTH},
          AFIR_ELEMENT_TYPE_NAMED_ONLY(AFIR_NAMEABLE_NAMED_ONLY)
#undef AFIR_NAMEABLE_NAMED_ONLY
  };
  return entries;
}

/// `type` 是 `kind` / `width` 这一位吗。
///
/// 整数那两类要同时比符号性与位宽：`isSignlessInteger(16)` 与 `isUnsignedInteger(16)` 在
/// MLIR 里是两个不同的类型，混掉的后果是 `ui16` 的图领到 `i16` 的 implementation。
inline bool afirElementTypeMatches(Type type, AfirElementTypeKind kind, unsigned width) {
  switch (kind) {
    case AfirElementTypeKind::SignlessInteger:
      return type.isSignlessInteger(width);
    case AfirElementTypeKind::UnsignedInteger:
      return type.isUnsignedInteger(width);
    case AfirElementTypeKind::Float16:
      return type.isF16();
    case AfirElementTypeKind::Float32:
      return type.isF32();
    case AfirElementTypeKind::BFloat16:
      return type.isBF16();
  }
  return false;
}

/// `type` 落在表上的哪一位；不在表上就是 0。
inline uint32_t afirElementTypeBit(Type type) {
  for (const AfirElementTypePosition &position : afirElementTypePositions()) {
    if (afirElementTypeMatches(type, position.kind, position.width)) {
      return position.bit;
    }
  }
  return 0U;
}

/// `type` 一个元素占几字节；不在表上就是 `nullopt`。
///
/// **不要换回 `type.getIntOrFloatBitWidth() / 8`。** 那个写法对今天表上的七位都对，所以
/// 看不出问题，但它对「位宽不是 8 的倍数」的类型**静默算出 0**，而 AFIR 的类型域里迟早
/// 会有这样一位（`bool` 在 MLIR 里的自然选择是 `i1`，在 AscendC 侧却占一整字节）。
/// 查过的三处除数写法：`Analysis/CanonicalBindings.cpp` 与
/// `MemoryAllocate/CanonicalMaterialization.cpp` 各有一道保护，第三处
/// `api/ascendc/.../Asc/Utils/Utils.cpp` 的 `getTypeSize` 一道都没有——而那个文件是 PyAsc
/// 上游原件、逐字未定制，不该由我们这一侧去改它的算法。所以字节数在 AFIR 这一层由表给，
/// 不由位宽推。
inline std::optional<int64_t> afirElementByteWidth(Type type) {
  for (const AfirElementTypePosition &position : afirElementTypePositions()) {
    if (afirElementTypeMatches(type, position.kind, position.width)) {
      return position.byteWidth;
    }
  }
  return std::nullopt;
}

/// 表上全部位的并集，也就是「AFIR 的元素类型域」。
inline uint32_t afirAllElementTypeBits() {
  uint32_t mask = 0U;
  for (const AfirElementTypePosition &position : afirElementTypePositions()) {
    mask |= position.bit;
  }
  return mask;
}

/// CSV 拼写 → 位；不认识就是 0。`ascir_op_lowering_matrix.csv` 的两列用它切。
inline uint32_t afirElementTypeBitFromCsvSpelling(llvm::StringRef spelling) {
  for (const AfirElementTypePosition &position : afirElementTypePositions()) {
    if (position.csvSpelling == spelling) {
      return position.bit;
    }
  }
  return 0U;
}

/// 位掩码 → `f16;f32` 这样的可读串，诊断与断言失败信息用。空集给 `-`。
inline std::string afirDescribeElementTypes(uint32_t mask) {
  std::string text;
  for (const AfirElementTypePosition &position : afirElementTypePositions()) {
    if ((mask & position.bit) != 0U) {
      if (!text.empty()) {
        text += ";";
      }
      text += position.csvSpelling.str();
    }
  }
  return text.empty() ? "-" : text;
}

}  // namespace mlir::afir

#endif  // AUTOFUSE_MLIR_SUPPORT_AFIRELEMENTTYPES_H
