/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- ElementTypeMatrixTest.cpp --------------------------------*- C++ -*-===//
//
// 每个 implementation 收哪几种元素类型，这一条此前没有权威来源。
//
// 那个集合是人照着 legacy 的 `REG_ASC_IR` 读出来、手抄进注册处的，抄错不会让任何东西
// 变红。已经咬过两次，两次都是事后才发现：`floor_div` 写的集合与 mem 侧完全不相交；
// 比较族六行 ODS 里四行是错的（三行写了 legacy 从来没有的 `F64`、四行写了 `I64`、
// `lt` 少了 `I32`）。
//
// 这条断言把注册处的位掩码与 `docs/guides/ascir_op_lowering_matrix.csv` 的
// `mem_dtypes` / `reg_dtypes` 两列比对。那两列由
// `tools/af-mlir-tools/scan_ascir_op_lowering_matrix.py` 从 `REG_ASC_IR` 抽出来，
// 表本身由 pre-commit 的门禁盯着不许与 legacy 漂移——两者合起来，注册处的集合才算
// 有权威来源，也能比对，而不是手抄结果。
//
// 判据是「不得比 legacy 窄」，不是「逐字相等」。 两个方向的后果不对称：
//
//   - 少写了是真损失：这条路在一个 legacy 收的类型上答 unsupported，整张图静默回落
//     legacy，没有任何用例会红。这是这条断言要抓的东西。
//   - 多写了不是：一个不在 legacy `REG_ASC_IR` 列表里的 dtype，它的 AscGraph 节点在
//     调度之前就被 legacy 拒了，根本到不了 bridge。所以「比 legacy 宽」既伤不到人，也用
//     不上；将来 AFIR 真能比 legacy 多支持一种时，这条判据不该拦着。
//
// 所以这里查的是 `注册处 ⊇ legacy 可建模部分`。多出来的那部分另有一道更紧的闸：ODS 的
// 元素类型列表。注册处写了而 ODS 没有的类型，op 本身就建不出来——那不是「更宽」，是一句
// 没人读的假话，见下面已知差异表里记的那一批。
//
// 比较的是能被 AFIR 建模的那个子集：哪几位算建模得出来由
// `support/AfirElementTypes.h` 的 `AFIR_ELEMENT_TYPE_TABLE` 说，legacy 收而那张表上没有
// 的（第一波后只剩 i4 / bool）在这一层表达不出来，不能拿来当差异。
//
//===----------------------------------------------------------------------===//

#include "bridge/AfirBuilder/AfirBuilder.h"

#include "conversion/AFIRToAscendC/Implementations/ElementwiseApiFamily.h"
#include "conversion/AFIRToAscendC/RegisterAscendCImplementations.h"
#include "dialect/AFIR/IR/AFIR.h"
#include "dialect/AFIR/IR/AFIRElementwiseSupport.h"
#include "support/AfirElementTypes.h"
#include "target/Ascend/Capability/ImplementationRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace mlir::ascend::kernel::elementwise {
namespace {

// 表里的拼写与位的对应、以及「AFIR 建模哪几位」这件事本身，权威定义都在
// `support/AfirElementTypes.h`。这里只是把它用上。
//
// 改造之前这是第三份拷贝，而它漏一位的后果最隐蔽：`kModelableDtypes` 里没有的拼写按
// 「表达不出来」跳过，于是下面那条「不得比 legacy 窄」的断言对新那一位恒真——它不会
// 红，只是不再查。bf16 刚加上的时候就差点这样过去。

std::string describe(uint32_t mask) {
  return afir::afirDescribeElementTypes(mask);
}

// `f16;f32;i32` → 位掩码，表里 AFIR 建模不出来的拼写直接跳过。
uint32_t maskOf(llvm::StringRef list) {
  uint32_t mask = 0U;
  llvm::SmallVector<llvm::StringRef> parts;
  list.split(parts, ';', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  for (llvm::StringRef part : parts) {
    mask |= afir::afirElementTypeBitFromCsvSpelling(part);
  }
  return mask;
}

// 最小的 RFC4180 切分：`reason` 一列里有逗号，生成表时是带引号的。
//
// 行尾的 `\r` 要先去掉：表是 python `csv.writer` 写的，它的缺省行尾是 `\r\n`，
// 不去掉的话最后一列的列名会带着回车，列查找当场落空、整张表读成空的。
std::vector<std::string> splitCsvLine(llvm::StringRef line) {
  line = line.rtrim("\r");
  std::vector<std::string> fields;
  std::string current;
  bool quoted = false;
  for (size_t index = 0; index < line.size(); ++index) {
    const char ch = line[index];
    if (quoted) {
      if (ch != '"') {
        current.push_back(ch);
      } else if (index + 1 < line.size() && line[index + 1] == '"') {
        current.push_back('"');
        ++index;
      } else {
        quoted = false;
      }
      continue;
    }
    if (ch == '"') {
      quoted = true;
    } else if (ch == ',') {
      fields.push_back(current);
      current.clear();
    } else {
      current.push_back(ch);
    }
  }
  fields.push_back(current);
  return fields;
}

struct MatrixRow {
  std::string memDtypes;
  std::string regDtypes;
};

// legacy op 名 → 两侧的 dtype 列。读不到表就返回空，由调用方 ASSERT——表没了的话这条
// 断言会静默退化成恒真，那比不写还坏。
llvm::StringMap<MatrixRow> readMatrix(const std::string &path) {
  llvm::StringMap<MatrixRow> rows;
  std::ifstream stream(path);
  if (!stream) {
    return rows;
  }
  std::string line;
  if (!std::getline(stream, line)) {
    return rows;
  }
  const std::vector<std::string> header = splitCsvLine(line);
  auto columnOf = [&header](llvm::StringRef name) -> size_t {
    for (size_t index = 0; index < header.size(); ++index) {
      if (header[index] == name) {
        return index;
      }
    }
    return header.size();
  };
  const size_t opColumn = columnOf("op");
  const size_t memColumn = columnOf("mem_dtypes");
  const size_t regColumn = columnOf("reg_dtypes");
  if (opColumn == header.size() || memColumn == header.size() || regColumn == header.size()) {
    return rows;
  }
  while (std::getline(stream, line)) {
    if (line.empty()) {
      continue;
    }
    const std::vector<std::string> fields = splitCsvLine(line);
    if (fields.size() <= regColumn) {
      continue;
    }
    rows[fields[opColumn]] = MatrixRow{fields[memColumn], fields[regColumn]};
  }
  return rows;
}

// 目前没有已知差异，所以这里没有差异表。
//
// 这条断言第一次跑时照出 28 处不一致，2026-09-23 一次清完，分三类：
//
//   - 17 处注册处比 ODS 和 legacy 都宽：`registerUnary` 原先有个叫
//     `kLegacyUnaryElementTypes`（= f16|f32|i32）的缺省实参，于是 `exp` / `ln` / `sqrt` /
//     `sigmoid` / `reciprocal` / 谓词族 / `clip_by_value` 这些 legacy 只收 f16 / f32 的
//     op 也一并领到了 i32。ODS 本来就拒 i32，那一位是句没人读的假话。缺省实参已去掉
//     ——集合是每个 op 自己的事实，不该有缺省。
//   - 4 处注册处比 ODS 和 legacy 都窄：选择族漏了 i16（ODS 的
//     `AFIR_SelectElementTypes` 有、legacy 两侧也有）。补上。
//   - 10 处 legacy 比 AFIR 宽：reg 侧的 `abs`（i16/ui8）、`neg`（i16）、`sub` /
//     `floor_div`（ui8/ui16）、逻辑族（ui16）、`where`（ui8/ui16）、`clip_by_value`
//     （i16/i32/ui8/ui16）。这一类要同时放宽 ODS 与注册处，已一并做掉——**ODS 取两侧并集,
//     每一侧收哪些由各自的注册处说**，所以 `where` 与 `select` 的 ODS 表拆成了两张。
//
// 2026-09-23 补：第一个真的比 legacy 窄的 implementation 来了，按上面那句话办——
// 下面这张快照表逐 id 钉死「注册的」和「按表算出来的」两个集合，循环本身一点没放宽。

// 比 legacy 窄、且窄得有理由的 implementation。这不是豁免，是快照：两个集合都钉死，
// 任何一边动了都会红，逼着动的人回来重读 `reason` 还成不成立。
//
// 收窄的理由必须是「legacy 在那个 dtype 上生成的根本不是这条路生成的调用」，不能是「还没
// 对过」——「还没对过」的正确做法是对完再注册。
struct KnownNarrowerRow {
  llvm::StringRef id;
  uint32_t registered;  // 注册处今天写的
  uint32_t expected;    // 按对照表算出来的
  llvm::StringRef reason;
};
constexpr KnownNarrowerRow kKnownNarrower[] = {
    {"add.broadcast_inline.reg_base", kF16 | kF32 | kI16 | kI32 | kUI16 | kUI32 | kBF16,
     kF16 | kF32 | kI16 | kI32 | kUI8 | kUI16 | kBF16 | kI64 | kI8 | kUI32 | kUI64,
     "BrcInlineTemplateV2 requires RemovePad-supported types; 8/64-bit broadcasts are excluded by legacy"},
    {"sub.broadcast_inline.reg_base", kF16 | kF32 | kI16 | kI32 | kUI16 | kUI32 | kBF16,
     kF16 | kF32 | kI16 | kI32 | kUI8 | kUI16 | kBF16 | kI64 | kI8 | kUI32 | kUI64,
     "BrcInlineTemplateV2 requires RemovePad-supported types; 8/64-bit broadcasts are excluded by legacy"},
    {"minimum.broadcast_inline.reg_base", kF16 | kF32 | kI16 | kI32 | kUI16 | kUI32 | kBF16,
     kF16 | kF32 | kI16 | kI32 | kUI8 | kUI16 | kBF16 | kI64 | kI8 | kUI32 | kUI64,
     "BrcInlineTemplateV2 requires RemovePad-supported types; 8/64-bit broadcasts are excluded by legacy"},
    {"maximum.broadcast_inline.reg_base", kF16 | kF32 | kI16 | kI32 | kUI16 | kUI32 | kBF16,
     kF16 | kF32 | kI16 | kI32 | kUI8 | kUI16 | kBF16 | kI64 | kI8 | kUI32 | kUI64,
     "BrcInlineTemplateV2 requires RemovePad-supported types; 8/64-bit broadcasts are excluded by legacy"},
    {"mul.broadcast_inline.reg_base", kF16 | kF32 | kI16 | kI32 | kUI32 | kBF16,
     kF16 | kF32 | kI16 | kI32 | kUI8 | kBF16 | kI64 | kI8 | kUI32 | kUI64,
     "BrcInlineTemplateV2 requires RemovePad-supported types; 8/64-bit broadcasts are excluded by legacy"},
    {"axpy.tensor_tensor.reg_base", kBF16 | kI64 | kUI64, kF16 | kF32 | kBF16 | kI64 | kUI64,
     "f16/f32 由两侧共用行收，A5 专属行只注册扩展类型"},
    {"relu.tensor.reg_base", kI64, kF16 | kF32 | kI32 | kI64 | kUI8,
     "共用行收 f16/f32/i32，A5 额外收 i64；连续 Relu 不接受 ui8"},
    // mem 侧的 Duplicate 档**有意**摘除 i64：legacy 对 int64 的 `BroadcastScalar` 发的是
    // 带 tmp 尾参的四参调用 `Duplicate(dst, v, n, tmp)`（`broadcast_api_call.cpp:246`，
    // 只在 int64 / uint64 时拼那个尾巴），而这一档发的是不带尾巴的标准
    // `AscendC::Duplicate`——把 i64 注册上就会发出一条实参个数对不上的调用。补齐 =
    // ascendc 方言补带 tmp 尾参的形态、pattern 按 dtype 分派，然后删本行。
    // ui8 不在此列：mem 侧已用 DuplicateU8Op 接入 duplicate.h 的包装。
    {"broadcast.tensor.duplicate", kF16 | kF32 | kI16 | kI32 | kUI8 | kUI16 | kBF16 | kUI32,
     kF16 | kF32 | kI16 | kI32 | kUI8 | kUI16 | kI64 | kI8 | kUI32 | kUI64,
     "i64 / ui64 在 legacy mem 侧走带 tmp 尾参的四参 Duplicate，不是这一档发的三参标准调用；"
     "i8 走 duplicate.h 的 int8 重载（ui8 已用 DuplicateU8Op 单独接入，int8 未接）"},
    // 标量源那一档与上面一行同一个理由、同一条收窄（legacy `BroadcastScalar` 的
    // `in.is_constant` 一支对 int64 同样拼 tmp 尾参）。两行一起删或一起留。
    {"broadcast.scalar.duplicate", kF16 | kF32 | kI16 | kI32 | kUI8 | kUI16 | kBF16 | kUI32,
     kF16 | kF32 | kI16 | kI32 | kUI8 | kUI16 | kI64 | kI8 | kUI32 | kUI64,
     "i64 / ui64 在 legacy mem 侧走带 tmp 尾参的四参 Duplicate，不是这一档发的三参标准调用；"
     "i8 走 duplicate.h 的 int8 重载（ui8 已用 DuplicateU8Op 单独接入，int8 未接）"},
    // reg 侧的 Duplicate 只窄 ui8 一位：legacy 的 `BroadcastScalar` 生成的是不带限定的
    // `Duplicate(dst, v, n)`，而 `duplicate.h` 在全局作用域给 int8 / uint8 各摆了一条
    // 非模板重载，`ui8` 绑的是包装（先 `ReinterpretCast<uint16_t>` 再发一条半宽的
    // `Duplicate`），不是这条路发的标准 `AscendC::Duplicate`。把 `ui8` 也注册上，发出的
    // 就是一条 legacy 从来没发过的调用——比答 unsupported 更坏。i64 不窄：reg 侧的
    // stock Duplicate 按 8 字节收（SupportBytes<1,2,4,8>），legacy reg 列里就有它。
    // bf16 是合法加宽——AscendC::Duplicate 按模板收 bfloat16_t（与 f16 同 16 位），
    // legacy 只是没在 REG_ASC_IR 声明，同 kOdsWiderThanLegacy 的 broadcast 行一个口径。
    {"broadcast.tensor.duplicate_reg_base", kF16 | kF32 | kI16 | kI32 | kUI16 | kBF16 | kI64 | kUI32 | kUI64,
     kF16 | kF32 | kI16 | kI32 | kUI8 | kUI16 | kI64 | kI8 | kUI32 | kUI64,
     "ui8 / i8 在 legacy 走 duplicate.h 的包装重载，不是标准 AscendC::Duplicate"},
    // 标量源那一档（legacy `BroadcastScalar` 的 `in.is_constant` 一支）与上面一行同一个
    // 理由、同一条收窄：reg 侧仍只发标准 `AscendC::Duplicate`，而 legacy 在 ui8 上发的是
    // `duplicate.h` 的包装重载。两行一起删或一起留。bf16 加宽同上一行口径。
    {"broadcast.scalar.duplicate_reg_base", kF16 | kF32 | kI16 | kI32 | kUI16 | kBF16 | kI64 | kUI32 | kUI64,
     kF16 | kF32 | kI16 | kI32 | kUI8 | kUI16 | kI64 | kI8 | kUI32 | kUI64,
     "ui8 / i8 在 legacy 走 duplicate.h 的包装重载，不是标准 AscendC::Duplicate"},
    // compare.h 的 i64 标量重载仅支持 EQ/NE，且现有调度未保证该重载的临时区容量。
    {"eq.tensor_scalar", kF16 | kF32 | kI32, kF16 | kF32 | kI32 | kI64,
     "i64 标量包装需要额外的临时区，现有调度无法保证容量"},
    {"ne.tensor_scalar", kF16 | kF32 | kI32, kF16 | kF32 | kI32 | kI64,
     "i64 标量包装需要额外的临时区，现有调度无法保证容量"},
    {"le.tensor_scalar", kF16 | kF32 | kI32, kF16 | kF32 | kI32 | kI64, "i64 标量包装仅支持 EQ/NE"},
    {"gt.tensor_scalar", kF16 | kF32 | kI32, kF16 | kF32 | kI32 | kI64, "i64 标量包装仅支持 EQ/NE"},
    {"ge.tensor_scalar", kF16 | kF32 | kI32, kF16 | kF32 | kI32 | kI64, "i64 标量包装仅支持 EQ/NE"},
    {"floor_div.tensor.extend_reg_base", kF16 | kF32 | kI16 | kI32 | kUI16 | kUI32 | kUI64,
     kF16 | kF32 | kI16 | kI32 | kUI8 | kUI16 | kI64 | kI8 | kUI32 | kUI64,
     "i64 SIMT 包装未处理 INT64_MIN / -1；i8 / ui8 在 FloorDivExtend 分派里没有对应分支"
     "（int16/int32、uint16/32/64、half/float 三支之外），实例化后静默空算。"
     "ui8 为存量问题，本波钉住"},
    {"add.tensor_tensor.reg_base", kUI8 | kUI16 | kBF16 | kI64 | kI8 | kUI32 | kUI64,
     kF16 | kF32 | kI16 | kI32 | kUI8 | kUI16 | kBF16 | kI64 | kI8 | kUI32 | kUI64,
     "拆行记账（非能力缺口）：交集位由 Any 主行收，reg 行 dtype 互斥只收交集外位；该形态无能力缺口"},
    {"add.tensor_scalar.reg_base", kUI8 | kBF16 | kI64 | kI8 | kUI64,
     kF16 | kF32 | kI16 | kI32 | kUI8 | kUI16 | kBF16 | kI64 | kI8 | kUI32 | kUI64,
     "能力缺口：Adds 无 ui16/ui32（kernel_operator_vec_binary_scalar_impl.h:252）；交集位由 Any 主行收（拆行记账）"},
    {"maximum.tensor_tensor.reg_base", kUI8 | kUI16 | kBF16 | kI64 | kI8 | kUI32 | kUI64,
     kF16 | kF32 | kI16 | kI32 | kUI8 | kUI16 | kBF16 | kI64 | kI8 | kUI32 | kUI64,
     "拆行记账（非能力缺口）：交集位由 Any 主行收，reg 行 dtype 互斥只收交集外位；该形态无能力缺口"},
    {"maximum.tensor_scalar.reg_base", kUI8 | kBF16 | kI64 | kI8 | kUI64,
     kF16 | kF32 | kI16 | kI32 | kUI8 | kUI16 | kBF16 | kI64 | kI8 | kUI32 | kUI64,
     "能力缺口：Maxs 无 ui16/ui32（kernel_operator_vec_binary_scalar_impl.h:348）；交集位由 Any 主行收（拆行记账）"},
    {"minimum.tensor_tensor.reg_base", kUI8 | kUI16 | kBF16 | kI64 | kI8 | kUI32 | kUI64,
     kF16 | kF32 | kI16 | kI32 | kUI8 | kUI16 | kBF16 | kI64 | kI8 | kUI32 | kUI64,
     "拆行记账（非能力缺口）：交集位由 Any 主行收，reg 行 dtype 互斥只收交集外位；该形态无能力缺口"},
    {"minimum.tensor_scalar.reg_base", kUI8 | kBF16 | kI64 | kI8 | kUI64,
     kF16 | kF32 | kI16 | kI32 | kUI8 | kUI16 | kBF16 | kI64 | kI8 | kUI32 | kUI64,
     "能力缺口：Mins 无 ui16/ui32（kernel_operator_vec_binary_scalar_impl.h:390）；交集位由 Any 主行收（拆行记账）"},
    {"mul.tensor_tensor.reg_base", kBF16 | kI64 | kUI32 | kUI64,
     kF16 | kF32 | kI16 | kI32 | kUI8 | kBF16 | kI64 | kI8 | kUI32 | kUI64,
     "能力缺口：连续 Mul 无 8 位分支（kernel_operator_vec_binary_continuous_impl.h）；f16/f32/i16/i32 为拆行记账位，由 "
     "Any 主行 mul.tensor_tensor 收，两行并集 = reg 列"},
    {"mul.tensor_scalar.reg_base", kBF16 | kI64 | kUI64,
     kF16 | kF32 | kI16 | kI32 | kUI8 | kBF16 | kI64 | kI8 | kUI32 | kUI64,
     "能力缺口：Muls 无 8 位与 ui32（kernel_operator_vec_binary_scalar_impl.h:306）；f16/f32/i16/i32 为拆行记账位，由 "
     "Any 主行 mul.tensor_scalar 收"},
    {"fma.tensor", kF16 | kF32, kF16 | kF32 | kI16 | kUI8 | kBF16 | kI8,
     "AscendC::Fma 的 static_assert 只收 half/float（adv_api/math/fma.h）；legacy 对四种整数/低精"
     "走 GetConversionDtype 先 cast 成 f16/f32 再调，AFIR 无输入转换机制。恢复需先建 cast 路径"},
    {"transpose.ub_ub_reg_base", kF16 | kF32 | kI16 | kI32 | kUI16 | kBF16 | kUI32,
     kF16 | kF32 | kI16 | kI32 | kUI16 | kBF16 | kI64 | kUI32, "RegBased Transpose 使用不支持 8 字节元素的 Gather"},
    // mem 档**有意**摘除 i32（连带无位的 ui32 同摘）：v1 布局对非 float 的 transpose 只给
    // 32B（4 字节整型折 8 元素），而 ConfusionTranspose 公式写死尾轴 16 元素对齐——布局
    // 与公式错位，数值必错（legacy 生产同错，其 ST 因手搓布局掩盖且无 int32 用例；实测
    // 1700/1702）。reg 档不受影响（regbase 显式传 strides）。根因链、实测证据与恢复条件
    // 见 `Implementations/Transpose.cpp` 的「口径 A 修订」注释与设计文档
    // `2026-10-19-afir-transpose-e2e-design.md` 收尾结论。补齐 = v1 布局/公式口径对齐后，
    // 把 `kTransposeMemElementTypes` 加回 kI32 并删本行。
    {"transpose.ub_ub_extend", kF16 | kF32 | kI16 | kUI16, kF16 | kF32 | kI16 | kI32 | kUI16 | kUI32,
     "i32/ui32 在 mem 档因 v1 布局(8元素)与 ConfusionTranspose 公式(写死16元素)错位而摘除，"
     "legacy 生产同错；reg 档保留（regbase 显式传 strides）"},
};

// ODS 比 legacy 的声明宽，且宽得有理由。这不是豁免，是快照：多出来的那几位逐位
// 钉死，任何一边动了都会红，逼着动的人回来重读理由。
//
// 判据二（下面那条 `OdsElementTypesAgreeWithRegistrationsAndLegacy`）原本是无条件的：
// 「ODS 声称一个 legacy 两侧都不供的 dtype，那一位就是假话——症状是图建得出来、选不出
// implementation、白回落」。那条推理有一个前提：**legacy 的 `REG_ASC_IR` 声明 = 两侧
// AscendC 头真正供得起的集合**。`afir.broadcast` 上这个前提不成立——
// `v35/ascendc/api_regbase/broadcast_extend_impl.h:260` 的
// `static_assert(SupportBytes<T, 1, 2, 4, 8>())` 收 bf16，而 `REG_ASC_IR(Broadcast)` 两侧
// 都没声明它。功能需求 3 的判定标准是以头文件为准，不以 legacy 注册表为准，所以这一位该开。
//
// 这张表不会把判据二变成一句空话，因为下面多查一条：钉在这里的每一位都必须**真有
// 一个 implementation 注册着它**。判据二要挡的「白回落」正是「ODS 收、没人供」，而
// 「ODS 收、有人供、只是 legacy 没声明」是另一回事。两者的区别就是那一条。
//
// 新增一行的格式：afir op 名 / 多出来的那几位 / 一句话理由，理由里要有供得起它的那个头
// 文件的 `file:line`。
struct KnownWiderRow {
  llvm::StringRef afirOpName;
  uint32_t extraBits;  // ODS 有、legacy 两列都没有的那几位
  llvm::StringRef reason;
};
constexpr KnownWiderRow kOdsWiderThanLegacy[] = {
    {"afir.select", afir::kAfirBitBF16, "v2 MaskedFill 在调度前改写成 Select，使用支持 bf16 的 WhereExtend"},
    {"afir.broadcast", afir::kAfirBitBF16,
     "broadcast_extend_impl.h:260 的 SupportBytes<T,1,2,4,8> 收 b16，legacy 的 "
     "REG_ASC_IR(Broadcast) 两侧只是没声明 bf16"},
};

const KnownWiderRow *findKnownWider(llvm::StringRef afirOpName) {
  for (const KnownWiderRow &row : kOdsWiderThanLegacy) {
    if (row.afirOpName == afirOpName) {
      return &row;
    }
  }
  return nullptr;
}

const KnownNarrowerRow *findKnownNarrower(llvm::StringRef id) {
  for (const KnownNarrowerRow &row : kKnownNarrower) {
    if (row.id == id) {
      return &row;
    }
  }
  return nullptr;
}

TEST(ElementTypeMatrixTest, RegisteredElementTypesMatchTheLoweringMatrix) {
  const llvm::StringMap<MatrixRow> matrix = readMatrix(AF_MLIR_OP_LOWERING_MATRIX_CSV);
  ASSERT_FALSE(matrix.empty()) << "读不到对照表 " << AF_MLIR_OP_LOWERING_MATRIX_CSV;

  MLIRContext context;
  context.getOrLoadDialect<afir::AFIRDialect>();
  ascend::instruction::ImplementationRegistry registry(context);
  ASSERT_TRUE(succeeded(registerAscendCImplementations(registry)));

  // afir op 名 → legacy 节点名。这张对应关系只有 bridge 那张表说了算，自己按命名规则
  // 推是第二份名单（`afir.truediv` ↔ `TrueDiv`、`afir.isnan` ↔ `Isnan`，规则本来就不齐）。
  llvm::StringMap<llvm::StringRef> legacyName;
  for (const afir::bridge::AFIRBridgeOpMapping &mapping : afir::bridge::getAFIRBridgeableOperations()) {
    legacyName[mapping.afir_operation_name] = mapping.scheduler_name;
  }

  size_t checked = 0;
  for (const RegisteredFamily &row : registeredFamilies()) {
    if (row.elementTypes == 0U) {
      continue;  // 逐行对表那一档，集合不被读；由下面那条断言盯住它不会悄悄变多。
    }
    auto legacy = legacyName.find(row.afirOpName);
    ASSERT_NE(legacy, legacyName.end()) << row.afirOpName << " 不在 bridge 表里";
    auto entry = matrix.find(legacy->second);
    ASSERT_NE(entry, matrix.end()) << legacy->second.str() << " 不在对照表里";

    // `Any` / `Shared` 只有**一个** implementation 供两侧用，所以它的集合要取两列的
    // 交集：取并集的话，在窄的那一侧会生成一个 legacy 根本没注册的调用。
    // `MemBased` / `RegBased` 各自只看自己那一列。
    const uint32_t memMask = maskOf(entry->second.memDtypes);
    const uint32_t regMask = maskOf(entry->second.regDtypes);
    uint32_t expected = 0U;
    llvm::StringRef side;
    switch (row.family) {
      case ApiFamily::MemBased:
        expected = memMask;
        side = "mem";
        break;
      case ApiFamily::RegBased:
        expected = regMask;
        side = "reg";
        break;
      case ApiFamily::Any:
      case ApiFamily::Shared:
        expected = memMask & regMask;
        side = "两侧共用，取交集";
        break;
    }
    ++checked;
    const std::string registered = describe(row.elementTypes);
    const std::string wanted = describe(expected);

    // 已知窄差那几行：不放过，改成对快照。两个集合都要逐字对上——注册处放宽了要来这里
    // 删行，对照表变了要来这里重读理由。
    if (const KnownNarrowerRow *known = findKnownNarrower(row.id)) {
      EXPECT_EQ(row.elementTypes, known->registered)
          << row.id << " 的注册集合变了（现在是 " << registered << "，快照记的是 " << describe(known->registered)
          << "）：回 kKnownNarrower 更新，或者已经补齐了就删掉这一行";
      EXPECT_EQ(expected, known->expected)
          << row.id << " 按对照表算出来的集合变了（现在是 " << wanted << "，快照记的是 " << describe(known->expected)
          << "）：legacy 动过了，回去重读 " << known->reason.str();
      continue;
    }

    EXPECT_EQ(expected & ~row.elementTypes, 0U)
        << row.id << "（" << side.str() << "）比 legacy 窄：注册的是 " << registered << "，legacy 的 "
        << legacy->second.str() << " 两侧收的是 mem=" << entry->second.memDtypes << " reg=" << entry->second.regDtypes
        << "，至少要收 " << wanted << "，缺的是 " << describe(expected & ~row.elementTypes);
  }
  EXPECT_GT(checked, 0u) << "一条都没对上说明侧表没填或者表读空了";

  // 反向断言：表里的 id 必须真的还在注册表里、也必须真的还窄着。补齐了却忘了删行的话，
  // 那一行就会一直替它挡着上面那条判据。
  llvm::StringSet<> registeredIds;
  for (const RegisteredFamily &row : registeredFamilies()) {
    registeredIds.insert(row.id);
  }
  for (const KnownNarrowerRow &known : kKnownNarrower) {
    EXPECT_TRUE(registeredIds.contains(known.id)) << known.id.str() << " 已经不在注册表里了，删掉这一行";
    EXPECT_NE(known.expected & ~known.registered, 0U)
        << known.id.str() << " 已经不比 legacy 窄了，删掉这一行让它回到上面那条判据";
  }
}

// ODS 的元素类型列表必须等于注册处那两份位掩码的并集（投影到 AFIR 的支持表上）。
//
// 这是功能需求 3 第一步的那条门禁。判据不是人为定义，而是「AFIR 能不能表达某个 dtype」
// 这件事的定义：AFIR 表达得出来，当且仅当至少有一侧的 implementation 供得起——所以 ODS
// 那一份是推导得出来的，不是独立事实。
//
// 之前没有任何机器查过这条关系，于是两个方向都出过事，而且两个方向都不会红：
//
//   - 注册处宽、ODS 窄：注册的那一位是句没人读的假话。op 根本建不出来，这个
//     implementation 永远选不到。文件头记的 2026-09-23 清掉的 17 处就是这一类，当时是人
//     逐个读出来的。
//   - ODS 宽、注册处窄：op 建得出来、选不出 implementation，整图干净回落。不报错，
//     只是白回落，而接管表上这一格的首因会指向「没有 Ascend 实现」而不是「这一位没注册」
//     ——排期的人按首因挑活，就会去改错的那一层。
//
// 投影到支持表上这一步是必须的，不是放宽：`load` / `store` / `cast` / `broadcast` 的
// ODS 列表里有 i4 这类在 AFIR 位表上没有位的类型（搬运与 cast 要能承载还没建模的
// dtype），任何位掩码都表示不出它们，拿它们来比只能永远失败。
// ODS 的元素类型列表与注册处的关系有两条判据，而且两条都有方向。
//
// 这是功能需求 3 第一步的那条门禁。写成「ODS == 注册处的并集」是错的，而且错得有后果，
// 所以把两个方向分开记清楚：
//
//   1. 注册处 ⊆ ODS（硬规则）。注册了一位而 ODS 拒它，那一位就是句没人读的假话：op
//      根本建不出来，这个 implementation 永远选不到。`ElementTypeMatrixTest` 文件头记的
//      2026-09-23 清掉的 17 处就是这一类，当时是人逐个读出来的。
//
//   2. ODS ⊆ legacy 两列的并集（投影到 AFIR 的支持表上）。ODS 声称一个 legacy 两侧都
//      不供的 dtype，那一位同样是假话——只是症状换成了「图建得出来、选不出 implementation、
//      白回落」，一条诊断都没有。这一条有一张例外快照（`kOdsWiderThanLegacy`）：
//      legacy 的声明不等于 AscendC 头供得起的集合，两者不一致时按头文件走。那张表上每一位
//      都要有 implementation 真注册着它，所以「白回落」仍然挡着，理由见那张表上面那段。
//
// 反过来「ODS 比注册处宽」不是缺陷，是刻意的，而且是硬要求。 理由见 `Ops/Data.td` 里
// `AFIR_DataMoveElementTypes` 上那段：一个 dtype 只要过了 bridge 的 `IsAfirModeledDType`，
// 那张图里每个 op 都会被建出来；ODS 不收它就失败在 verifier 上，那是硬失败，把本可以
// 干净回落的图变红。所以「ODS 收、但没有 implementation 能降」正是要的状态。
//
// 这个缝隙对 `ApiFamily::Any` / `Shared` 的 op 是必然存在的：一份 implementation 服务
// 两侧，收的集合只能是两列的交集（取并集会在窄的那一侧生成一个 legacy 根本没注册的
// 调用），而 ODS 要盖住两列的并集。所以 add / mul / maximum / minimum / axpy / exp /
// ln / sqrt 这 8 个 op 上 bf16 在 ODS 里而不在注册处里，是对的——要让它真能降下去，得先
// 把那个 implementation 拆成 mem / reg 两份，而不是去删 ODS 的那一位。
//
// 投影到支持表上这一步是必须的，不是放宽：`load` / `store` / `cast` / `broadcast` 的
// ODS 列表里有 i4 这类在 AFIR 位表上没有位的类型，任何位掩码都表示不出它们，
// 拿它们来比只能永远失败。
// `OdsElementTypesAgreeWithRegistrationsAndLegacy` 的 helper 们(拆分以满足 nbnc<=50;
// 语义与报错文案逐字保留,判据顺序不变)。

// afir op 名 → ODS 那一份元素类型列表。展开 bridge 那张 op 类名单,所以这里不会有第二
// 份名单;新加一个 bridge op 而忘了给它的 ODS 基类挂上 `AFIR_OdsElementTypes` 的话,
// 这个 TU 直接编不过,而不是静默少查一个 op。
void loadOdsSpellingsAndLegacyNames(llvm::StringMap<llvm::StringRef> &odsSpellings,
                                    llvm::StringMap<llvm::StringRef> &legacyName) {
#define AFIR_TEST_ODS_ELEMENT_TYPES(SCHEDULER_NAME, OP, FORM)              \
  odsSpellings[OP::getOperationName()] = OP::getOdsElementTypeSpellings(); \
  legacyName[OP::getOperationName()] = SCHEDULER_NAME;
  AFIR_BRIDGE_OPERATION_REGISTRY(AFIR_TEST_ODS_ELEMENT_TYPES)
#undef AFIR_TEST_ODS_ELEMENT_TYPES
}

// 每个 op 把它各个 implementation 的集合并起来。
llvm::StringMap<uint32_t> registeredUnionOf() {
  llvm::StringMap<uint32_t> registeredUnion;
  for (const RegisteredFamily &row : registeredFamilies()) {
    if (row.elementTypes == 0U) {
      continue;  // 逐行对表那一档(cast / logical_not),集合不被读。
    }
    registeredUnion[row.afirOpName] |= row.elementTypes;
  }
  return registeredUnion;
}

// ODS 列表里认得出的那几位。认不出的拼写要当场报出来——那说明 MLIR 的类型记录名字
// 与支持表的 CSV 拼写不再是「小写一下就相等」,而那个巧合是这条门禁的前提。
uint32_t odsBitsFromSpellings(const llvm::StringRef afirOpName, const llvm::StringRef spellings) {
  uint32_t odsBits = 0U;
  llvm::SmallVector<llvm::StringRef> parts;
  spellings.split(parts, ';', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  for (llvm::StringRef part : parts) {
    const uint32_t bit = afir::afirElementTypeBitFromCsvSpelling(part);
    if (bit != 0U) {
      odsBits |= bit;
      continue;
    }
    // 支持表上没有位的那些(现在只剩 i4)是故意留在 ODS 里的,跳过;
    // 其余一律是拼写对不上,必须红。
    EXPECT_TRUE(part == "i4") << afirOpName.str() << " 的 ODS 列表里有一个供给表认不出的拼写 `" << part.str()
                              << "`:要么它该进 AFIR_ELEMENT_TYPE_TABLE / AFIR_ELEMENT_TYPE_NAMED_ONLY,"
                              << "要么 MLIR 的类型记录名字变了,这条门禁的拼写前提已经不成立";
  }
  return odsBits;
}

// 快照上钉着的那几位:legacy 只是没声明,头文件供得起。放进 `legacySupplies` 之前先查
// 它真有 implementation 注册着——否则这张表就替判据二挡住了它本该抓的「白回落」。
uint32_t mergeKnownWiderIntoSupply(const llvm::StringRef afirOpName, const uint32_t registered, const uint32_t odsBits,
                                   uint32_t legacySupplies) {
  if (const KnownWiderRow *wider = findKnownWider(afirOpName)) {
    EXPECT_EQ(wider->extraBits & ~odsBits, 0U)
        << afirOpName.str() << " 的 ODS 已经不收 " << afir::afirDescribeElementTypes(wider->extraBits & ~odsBits)
        << " 了:回 kOdsWiderThanLegacy 删掉这一行";
    EXPECT_EQ(wider->extraBits & ~registered, 0U)
        << afirOpName.str() << " 在 kOdsWiderThanLegacy 上钉着 " << afir::afirDescribeElementTypes(wider->extraBits)
        << ",但没有任何 implementation 注册 " << afir::afirDescribeElementTypes(wider->extraBits & ~registered)
        << ":那就是判据二要抓的「ODS 收、没人供、白回落」,不是「legacy 没声明」。"
        << "要么去注册处开位,要么把这一行删掉";
    EXPECT_EQ(wider->extraBits & legacySupplies, 0U)
        << afirOpName.str() << " 的 legacy 声明已经收了 "
        << afir::afirDescribeElementTypes(wider->extraBits & legacySupplies) << ":这一位不再是例外,"
        << "回 kOdsWiderThanLegacy 删掉它、让它回到判据二,并重读当初的理由 " << wider->reason.str();
    legacySupplies |= wider->extraBits;
  }
  return legacySupplies;
}

TEST(ElementTypeMatrixTest, OdsElementTypesAgreeWithRegistrationsAndLegacy) {
  const llvm::StringMap<MatrixRow> matrix = readMatrix(AF_MLIR_OP_LOWERING_MATRIX_CSV);
  ASSERT_FALSE(matrix.empty()) << "读不到对照表 " << AF_MLIR_OP_LOWERING_MATRIX_CSV;

  MLIRContext context;
  context.getOrLoadDialect<afir::AFIRDialect>();
  ascend::instruction::ImplementationRegistry registry(context);
  ASSERT_TRUE(succeeded(registerAscendCImplementations(registry)));

  llvm::StringMap<llvm::StringRef> odsSpellings;
  llvm::StringMap<llvm::StringRef> legacyName;
  loadOdsSpellingsAndLegacyNames(odsSpellings, legacyName);
  ASSERT_FALSE(odsSpellings.empty());

  const llvm::StringMap<uint32_t> registeredUnion = registeredUnionOf();
  ASSERT_FALSE(registeredUnion.empty());

  size_t checked = 0;
  for (const auto &entry : registeredUnion) {
    const llvm::StringRef afirOpName = entry.first();
    const uint32_t registered = entry.second;
    auto ods = odsSpellings.find(afirOpName);
    ASSERT_NE(ods, odsSpellings.end()) << afirOpName.str() << " 不在 bridge 的 op 表里";
    auto legacy = legacyName.find(afirOpName);
    ASSERT_NE(legacy, legacyName.end()) << afirOpName.str() << " 不在 bridge 的 op 表里";
    auto matrixRow = matrix.find(legacy->second);
    ASSERT_NE(matrixRow, matrix.end()) << legacy->second.str() << " 不在对照表里";

    const uint32_t odsBits = odsBitsFromSpellings(afirOpName, ods->second);
    uint32_t legacySupplies = maskOf(matrixRow->second.memDtypes) | maskOf(matrixRow->second.regDtypes);
    legacySupplies = mergeKnownWiderIntoSupply(afirOpName, registered, odsBits, legacySupplies);
    ++checked;

    // 判据一:注册处 ⊆ ODS。
    EXPECT_EQ(registered & ~odsBits, 0U) << afirOpName.str() << " 注册了 ODS 不收的位:注册处是 "
                                         << afir::afirDescribeElementTypes(registered) << ",ODS 认得出的是 "
                                         << afir::afirDescribeElementTypes(odsBits) << ",多出来的 "
                                         << afir::afirDescribeElementTypes(registered & ~odsBits)
                                         << " 是句没人读的假话(op 建不出来,这个 implementation 永远选不到)";

    // 判据二:ODS ⊆ legacy 两列的并集。
    EXPECT_EQ(odsBits & ~legacySupplies, 0U)
        << afirOpName.str()
        << " 的 ODS 声称了 legacy 两侧都不供的位:" << afir::afirDescribeElementTypes(odsBits & ~legacySupplies)
        << "(ODS 认得出 " << afir::afirDescribeElementTypes(odsBits) << ",legacy 的 " << legacy->second.str()
        << " 两列并起来是 mem=" << matrixRow->second.memDtypes << " reg=" << matrixRow->second.regDtypes
        << ")。那一位该从 ODS 列表里删掉——留着的后果是图建得出来、选不出 implementation、白回落";
  }
  EXPECT_GT(checked, 0u) << "一条都没对上说明侧表没填或者表读空了";

  // 反向断言:快照上的 op 必须真的还被上面那个循环查着。它从 `registeredUnion` 里消失的
  // 话,那一行就会静默停止生效,而不是红。
  for (const KnownWiderRow &row : kOdsWiderThanLegacy) {
    EXPECT_TRUE(registeredUnion.contains(row.afirOpName))
        << row.afirOpName.str() << " 已经没有带元素类型集合的 implementation 了,删掉这一行";
  }
}

TEST(ElementTypeMatrixTest, CastPairsRespectPlatformAvailability) {
  using afir::elementwise::CastSupply;
  using afir::elementwise::isSupportedCastPair;
  MLIRContext context;
  Type ui8 = IntegerType::get(&context, 8, IntegerType::Unsigned);
  Type ui64 = IntegerType::get(&context, 64, IntegerType::Unsigned);
  Type i64 = IntegerType::get(&context, 64);
  Type f32 = Float32Type::get(&context);
  EXPECT_TRUE(isSupportedCastPair(CastSupply::MemBased, ui8, i64));
  EXPECT_TRUE(isSupportedCastPair(CastSupply::RegBased, ui8, i64));
  EXPECT_TRUE(isSupportedCastPair(CastSupply::RegBased, f32, ui64));
  EXPECT_FALSE(isSupportedCastPair(CastSupply::MemBased, f32, ui64));
}

// 不按集合判元素类型的 implementation：它们走 `ResultElementType::CastPair` 的逐行对表
// （表在 `Ops/Cast.cpp`），集合在那一档根本不被读，所以上面那条跳过它们。
// 名单钉死，否则「跳过」会悄悄长大，上面那条就越查越少。
TEST(ElementTypeMatrixTest, OnlyCastPairImplementationsOptOutOfElementTypeSets) {
  MLIRContext context;
  context.getOrLoadDialect<afir::AFIRDialect>();
  ascend::instruction::ImplementationRegistry registry(context);
  ASSERT_TRUE(succeeded(registerAscendCImplementations(registry)));

  std::vector<std::string> optedOut;
  for (const RegisteredFamily &row : registeredFamilies()) {
    if (row.elementTypes == 0U) {
      optedOut.push_back(row.id);
    }
  }
  std::sort(optedOut.begin(), optedOut.end());
  const std::vector<std::string> expected = {
      "cast.tensor.extend",
      "cast.tensor.extend_reg_base",
  };
  EXPECT_EQ(optedOut, expected);
}

}  // namespace
}  // namespace mlir::ascend::kernel::elementwise

namespace mlir::ascend::kernel::elementwise {

// =====================================================================================
// 第二波扩表补充的两条门禁(独立用例:现有两条测试的函数体是 CI 棘轮的存量基线,
// 不能再往里加行)。
// =====================================================================================

// 判据一的补充(拆行盲区):同一个 (op, 侧, 形态) 的全部注册行并起来,必须盖住那一侧的
// legacy 列(投影到建模位)。按 id 逐行比时,「交集位由 Any 主行收」的拆行记账会让 reg 行
// 整行豁免覆盖判据;这一条把比较单位换成并集。缺口只能落在 kFormCoverageGaps 里,且必须
// 是能力缺口——「还没对过」不是合法理由,与 kKnownNarrower 同一条纪律。
struct FormCoverageGap {
  llvm::StringRef afirOpName;
  llvm::StringRef side;  // "mem" | "reg"
  llvm::StringRef form;  // "tensor" | "scalar"
  uint32_t missing;
  llvm::StringRef reason;
};
constexpr FormCoverageGap kFormCoverageGaps[] = {
    // ---- CANN 标量/连续调用的 SupportType 能力缺口(kernel_operator_vec_binary_*)----
    {"afir.mul", "reg", "tensor", kUI8 | kI8,
     "连续 Mul 无 8 位分支（kernel_operator_vec_binary_continuous_impl.h 的 MulImpl）"},
    {"afir.mul", "reg", "scalar", kUI8 | kI8 | kUI32,
     "Muls 无 8 位与 ui32（kernel_operator_vec_binary_scalar_impl.h:306）"},
    {"afir.add", "reg", "scalar", kUI16 | kUI32, "Adds 无 ui16/ui32（scalar_impl:252）"},
    {"afir.maximum", "reg", "scalar", kUI16 | kUI32, "Maxs 无 ui16/ui32（scalar_impl:348）"},
    {"afir.minimum", "reg", "scalar", kUI16 | kUI32, "Mins 无 ui16/ui32（scalar_impl:390）"},
    // ---- 包装 / 调用形态缺口 ----
    {"afir.fma", "reg", "tensor", kI16 | kUI8 | kBF16 | kI8,
     "AscendC::Fma 仅 half/float（adv_api/math/fma.h static_assert）；legacy 走 GetConversionDtype "
     "先 cast 再调，AFIR 无输入转换机制"},
    {"afir.floor_div", "reg", "tensor", kUI8 | kI8 | kI64,
     "FloorDivExtend 分派无 uint8/int8 分支；i64 SIMT 未处理 INT64_MIN/-1"},
    {"afir.broadcast", "mem", "scalar", kI64 | kI8 | kUI64,
     "i64/ui64 是四参带 tmp 尾参调用、i8 走 duplicate.h 重载，三参 duplicate 档发不出；"
     "segmented 档只收 tile 源"},
    {"afir.broadcast", "reg", "scalar", kUI8 | kI8, "duplicate.h 的 1 字节重载未接进这档"},
    {"afir.transpose", "mem", "tensor", kI32 | kUI32,
     "v1 布局（非 float 折 8 元素）与 ConfusionTranspose 公式（写死 16 元素）错位，数值必错；"
     "legacy 生产同错"},
    {"afir.transpose", "reg", "tensor", kI64, "RegBased Transpose 的 Gather 不支持 8 字节元素"},
    {"afir.eq", "mem", "scalar", kI64, "compare.h i64 标量重载仅 EQ/NE 且需额外临时区"},
    {"afir.ne", "mem", "scalar", kI64, "compare.h i64 标量重载仅 EQ/NE"},
    {"afir.le", "mem", "scalar", kI64, "compare.h i64 标量重载仅 EQ/NE"},
    {"afir.gt", "mem", "scalar", kI64, "compare.h i64 标量重载仅 EQ/NE"},
    {"afir.ge", "mem", "scalar", kI64, "compare.h i64 标量重载仅 EQ/NE"},
    // ---- Any/Shared 单行只能收两列交集的拆行缺口（存量，见本文件判据二注释）----
    {"afir.exp", "reg", "tensor", kBF16, "Any 单行只收两列交集；reg 侧 bf16 需拆 mem/reg 行"},
    {"afir.ln", "reg", "tensor", kBF16, "同 exp：Any 单行只收交集，reg 侧 bf16 需拆行"},
    {"afir.sqrt", "reg", "tensor", kBF16, "同 exp：Any 单行只收交集，reg 侧 bf16 需拆行"},
    {"afir.relu", "reg", "tensor", kUI8, "连续 Relu 调用不接受 ui8"},
    {"afir.ub2ub", "reg", "tensor", kI64, "Shared 单行只收交集；reg 侧 i64 需拆行"},
};

// 聚合 (op|side|form) → 注册位并集。Any/Shared 一行供两侧,计入两侧;id 含 scalar 归
// 标量档,其余(含 row_broadcast_inline / reduce 的 ar_ra)都算 tensor 档。
llvm::StringMap<uint32_t> buildFormUnion() {
  llvm::StringMap<uint32_t> unionOf;
  for (const RegisteredFamily &row : registeredFamilies()) {
    if (row.elementTypes == 0U) {
      continue;
    }
    const llvm::StringRef form = row.id.find("scalar") != std::string::npos ? "scalar" : "tensor";
    switch (row.family) {
      case ApiFamily::MemBased:
      case ApiFamily::RegBased: {
        const llvm::StringRef side = row.family == ApiFamily::MemBased ? "mem" : "reg";
        unionOf[(row.afirOpName + "|" + side + "|" + form).str()] |= row.elementTypes;
        break;
      }
      case ApiFamily::Any:
      case ApiFamily::Shared: {
        unionOf[(row.afirOpName + "|mem|" + form).str()] |= row.elementTypes;
        unionOf[(row.afirOpName + "|reg|" + form).str()] |= row.elementTypes;
        break;
      }
    }
  }
  return unionOf;
}

// 单个 "op|side|form" 键的覆盖核对:并集与那一侧 legacy 列比,缺口要么为零、要么与
// kFormCoverageGaps 的登记逐位一致。缺口非零而未登记 = 新能力缺口漏报;登记与实际不一致 =
// 登记过期,两条都会红。
void checkFormCoverageEntry(const llvm::StringRef key, const uint32_t registered,
                            const llvm::StringMap<MatrixRow> &matrix,
                            const llvm::StringMap<llvm::StringRef> &legacyName) {
  const auto sep1 = key.find('|');
  const auto sep2 = key.find('|', sep1 + 1);
  const llvm::StringRef opName = key.substr(0, sep1);
  const llvm::StringRef side = key.substr(sep1 + 1, sep2 - sep1 - 1);
  const llvm::StringRef form = key.substr(sep2 + 1);
  auto legacy = legacyName.find(opName);
  ASSERT_NE(legacy, legacyName.end()) << opName.str() << " 不在 bridge 表里";
  auto matrixRow = matrix.find(legacy->second);
  ASSERT_NE(matrixRow, matrix.end()) << legacy->second.str() << " 不在对照表里";
  const uint32_t column = side == "mem" ? maskOf(matrixRow->second.memDtypes) : maskOf(matrixRow->second.regDtypes);
  const uint32_t missing = column & ~registered;
  const FormCoverageGap *declared = nullptr;
  for (const FormCoverageGap &gap : kFormCoverageGaps) {
    if (gap.afirOpName == opName && gap.side == side && gap.form == form) {
      declared = &gap;
      break;
    }
  }
  if (declared == nullptr) {
    EXPECT_EQ(missing, 0U) << opName.str() << " (" << side.str() << "," << form.str() << ") 并集缺 "
                           << describe(missing) << " —— 是能力缺口就登记进 kFormCoverageGaps 并写明理由";
  } else {
    EXPECT_EQ(missing, declared->missing)
        << opName.str() << " (" << side.str() << "," << form.str() << ") 实际缺 " << describe(missing) << " 与登记的 "
        << describe(declared->missing) << " 不一致,回表更新";
  }
}

TEST(ElementTypeMatrixTest, FormUnionCoversLegacySideColumns) {
  const llvm::StringMap<MatrixRow> matrix = readMatrix(AF_MLIR_OP_LOWERING_MATRIX_CSV);
  ASSERT_FALSE(matrix.empty());

  MLIRContext context;
  context.getOrLoadDialect<afir::AFIRDialect>();
  ascend::instruction::ImplementationRegistry registry(context);
  ASSERT_TRUE(succeeded(registerAscendCImplementations(registry)));

  llvm::StringMap<llvm::StringRef> legacyName;
  for (const afir::bridge::AFIRBridgeOpMapping &mapping : afir::bridge::getAFIRBridgeableOperations()) {
    legacyName[mapping.afir_operation_name] = mapping.scheduler_name;
  }

  // (op, 侧, 形态) → 注册位并集。Any/Shared 一行供两侧,计入两侧;形态按 id 里是否带
  // scalar 归为标量档,其余(含 row_broadcast_inline/reduce 的 ar_ra 等)都算 tensor 档。
  const llvm::StringMap<uint32_t> unionOf = buildFormUnion();
  ASSERT_FALSE(unionOf.empty());

  for (const auto &entry : unionOf) {
    checkFormCoverageEntry(entry.getKey(), entry.second, matrix, legacyName);
  }
  llvm::SmallVector<std::string> surplusGapRows;
  for (const FormCoverageGap &gap : kFormCoverageGaps) {
    const std::string key = (gap.afirOpName + "|" + gap.side + "|" + gap.form).str();
    if (unionOf.find(key) == unionOf.end()) {
      surplusGapRows.push_back((gap.afirOpName + " (" + gap.side + "," + gap.form + ")").str());
    }
  }
  EXPECT_TRUE(surplusGapRows.empty()) << "登记的缺口里有找不到对应 (op,侧,形态) 组的";
  for (const std::string &row : surplusGapRows) {
    ADD_FAILURE() << row;
  }
}

// 反向 ODS 判据:bridge op 的 legacy 两列并集(投影到建模位)必须 ⊆ ODS。升表之后
// IsAfirModeledDType 对新位放行,含该 dtype 的图会整张建出来——ODS 不收就是 verifier
// 硬失败,而不是「没有 implementation」的干净回落。缺口登记进 kOdsHardFailures 并写明
// 按硬失败处理的理由;能补齐 ODS 的不要留在这里。
constexpr FormCoverageGap kOdsHardFailures[] = {
    // 探针跑完后按事实填写
};

TEST(ElementTypeMatrixTest, OdsCoversLegacyModeledSupply) {
  const llvm::StringMap<MatrixRow> matrix = readMatrix(AF_MLIR_OP_LOWERING_MATRIX_CSV);
  ASSERT_FALSE(matrix.empty());

  llvm::StringMap<llvm::StringRef> odsSpellings;
  llvm::StringMap<llvm::StringRef> legacyName;
#define AFIR_TEST_ODS_ELEMENT_TYPES(SCHEDULER_NAME, OP, FORM)              \
  odsSpellings[OP::getOperationName()] = OP::getOdsElementTypeSpellings(); \
  legacyName[OP::getOperationName()] = SCHEDULER_NAME;
  AFIR_BRIDGE_OPERATION_REGISTRY(AFIR_TEST_ODS_ELEMENT_TYPES)
#undef AFIR_TEST_ODS_ELEMENT_TYPES
  ASSERT_FALSE(odsSpellings.empty());

  for (const auto &entry : odsSpellings) {
    const llvm::StringRef opName = entry.getKey();
    auto legacy = legacyName.find(opName);
    ASSERT_NE(legacy, legacyName.end());
    auto matrixRow = matrix.find(legacy->second);
    ASSERT_NE(matrixRow, matrix.end()) << legacy->second.str() << " 不在对照表里";
    const uint32_t supply = maskOf(matrixRow->second.memDtypes) | maskOf(matrixRow->second.regDtypes);
    uint32_t odsBits = 0U;
    llvm::SmallVector<llvm::StringRef> parts;
    entry.getValue().split(parts, ';');
    for (const llvm::StringRef part : parts) {
      const uint32_t bit = afir::afirElementTypeBitFromCsvSpelling(part.trim());
      if (bit != 0U) {
        odsBits |= bit;
      }
    }
    uint32_t missing = supply & ~odsBits;
    for (const FormCoverageGap &gap : kOdsHardFailures) {
      if (gap.afirOpName == opName) {
        missing &= ~gap.missing;
      }
    }
    EXPECT_EQ(missing, 0U) << opName.str() << " 的 ODS 缺 legacy 供给的 " << describe(missing)
                           << ":升表后这是 verifier 硬失败,要么补 ODS,要么登记 "
                              "kOdsHardFailures 写明按硬失败处理";
  }
}

}  // namespace mlir::ascend::kernel::elementwise
