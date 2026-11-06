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
// **每个 implementation 收哪几种元素类型，这一条此前没有真源。**
//
// 那个集合是人照着 legacy 的 `REG_ASC_IR` 读出来、手抄进注册处的，抄错不会让任何东西
// 变红。已经咬过两次，两次都是事后才发现：`floor_div` 写的集合与 mem 侧完全不相交；
// 比较族六行 ODS 里四行是错的（三行写了 legacy 从来没有的 `F64`、四行写了 `I64`、
// `lt` 少了 `I32`）。
//
// 这条断言把注册处的位掩码与 `docs/guides/ascir_op_lowering_matrix.csv` 的
// `mem_dtypes` / `reg_dtypes` 两列对账。那两列由
// `tools/af-mlir-tools/scan_ascir_op_lowering_matrix.py` 从 `REG_ASC_IR` 抽出来，
// 表本身由 pre-commit 的门禁盯着不许与 legacy 漂移——两者合起来，注册处的集合才算
// **有真源且可对账**，而不是抄的。
//
// **判据是「不得比 legacy 窄」，不是「逐字相等」。** 两个方向的后果不对称：
//
//   - **少写了是真损失**：这条路在一个 legacy 收的类型上答 unsupported，整张图静默回落
//     legacy，没有任何用例会红。这是这条断言要抓的东西。
//   - **多写了不是**：一个不在 legacy `REG_ASC_IR` 列表里的 dtype，它的 AscGraph 节点在
//     调度之前就被 legacy 拒了，根本到不了 bridge。所以「比 legacy 宽」既伤不到人，也用
//     不上；将来 AFIR 真能比 legacy 多支持一种时，这条判据不该拦着。
//
// 所以这里查的是 `注册处 ⊇ legacy 可建模部分`。多出来的那部分另有一道更紧的闸：ODS 的
// 元素类型列表。注册处写了而 ODS 没有的类型，op 本身就建不出来——那不是「更宽」，是一句
// 没人读的假话，见下面已知差异表里记的那一批。
//
// 比较的是**能被 AFIR 建模的那个子集**：哪几位算建模得出来由
// `support/AfirElementTypes.h` 的 `AFIR_ELEMENT_TYPE_TABLE` 说，legacy 收而那张表上没有
// 的（i8 / i64 / ui32 / ui64 / bool）在这一层表达不出来，不能拿来当差异。
//
//===----------------------------------------------------------------------===//

#include "bridge/AfirBuilder/AfirBuilder.h"

#include "conversion/AFIRToAscendC/Implementations/ElementwiseApiFamily.h"
#include "conversion/AFIRToAscendC/RegisterAscendCImplementations.h"
#include "dialect/AFIR/IR/AFIR.h"
#include "support/AfirElementTypes.h"
#include "target/Ascend/Capability/ImplementationRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringRef.h"

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace mlir::ascend::kernel::elementwise {
namespace {

// 表里的拼写与位的对应、以及「AFIR 建模哪几位」这件事本身，真源都在
// `support/AfirElementTypes.h`。这里只是把它用上。
//
// **改造之前这是第三份拷贝**，而它漏一位的后果最隐蔽：`kModelableDtypes` 里没有的拼写按
// 「表达不出来」跳过，于是下面那条「不得比 legacy 窄」的断言对新那一位**恒真**——它不会
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

// **今天没有已知差异，所以这里没有差异表。**
//
// 这条断言第一次跑时照出 28 处不一致，2026-09-23 一次清完，分三类：
//
//   - **17 处注册处比 ODS 和 legacy 都宽**：`registerUnary` 原先有个叫
//     `kLegacyUnaryElementTypes`（= f16|f32|i32）的缺省实参，于是 `exp` / `ln` / `sqrt` /
//     `sigmoid` / `reciprocal` / 谓词族 / `clip_by_value` 这些 legacy 只收 f16 / f32 的
//     op 也一并领到了 i32。ODS 本来就拒 i32，那一位是句没人读的假话。缺省实参已去掉
//     ——集合是每个 op 自己的事实，不该有缺省。
//   - **4 处注册处比 ODS 和 legacy 都窄**：选择族漏了 i16（ODS 的
//     `AFIR_SelectElementTypes` 有、legacy 两侧也有）。补上。
//   - **10 处 legacy 比 AFIR 宽**：reg 侧的 `abs`（i16/ui8）、`neg`（i16）、`sub` /
//     `floor_div`（ui8/ui16）、逻辑族（ui16）、`where`（ui8/ui16）、`clip_by_value`
//     （i16/i32/ui8/ui16）。这一类要同时放宽 ODS 与注册处，已一并做掉——**ODS 取两侧并集,
//     每一侧收哪些由各自的注册处说**，所以 `where` 与 `select` 的 ODS 表拆成了两张。
//
// **2026-09-23 补：第一个真的比 legacy 窄的 implementation 来了**，按上面那句话办——
// 下面这张快照表逐 id 钉死「注册的」和「按表算出来的」两个集合，循环本身一点没放宽。

// 比 legacy 窄、且窄得有理由的 implementation。**这不是豁免，是快照**：两个集合都钉死，
// 任何一边动了都会红，逼着动的人回来重读 `reason` 还成不成立。
//
// 收窄的理由必须是「legacy 在那个 dtype 上发的根本不是这条路发的调用」，不能是「还没
// 对过」——「还没对过」的正确做法是对完再注册。
struct KnownNarrowerRow {
  llvm::StringRef id;
  uint32_t registered;  // 注册处今天写的
  uint32_t expected;    // 按对照表算出来的
  llvm::StringRef reason;
};
constexpr KnownNarrowerRow kKnownNarrower[] = {
    // legacy 的 `BroadcastScalar` 发的是**不带限定**的 `Duplicate(dst, v, n)`，而
    // `duplicate.h` 在全局作用域给 int8 / uint8 / int64 / uint64 各摆了一条非模板重载。
    // 这六位里落在那四条上的只有 `ui8`：它绑的是包装（先 `ReinterpretCast<uint16_t>`
    // 再发一条半宽的 `Duplicate`），不是这条路发的 `AscendC::Duplicate`。把 `ui8` 也
    // 注册上，发出去的就是一条 legacy 从来没发过的调用——比答 unsupported 更坏。
    // 要收 `ui8`，得另起一个带 `duplicate.h` 扩展头、发包装调用的 implementation。
    {"broadcast.tensor.duplicate", kF16 | kF32 | kI16 | kI32 | kUI16, kF16 | kF32 | kI16 | kI32 | kUI8 | kUI16,
     "ui8 在 legacy 走 duplicate.h 的包装重载，不是标准 AscendC::Duplicate"},
    {"broadcast.tensor.duplicate_reg_base", kF16 | kF32 | kI16 | kI32 | kUI16, kF16 | kF32 | kI16 | kI32 | kUI8 | kUI16,
     "ui8 在 legacy 走 duplicate.h 的包装重载，不是标准 AscendC::Duplicate"},
    // mem 档**有意**摘除 i32（连带无位的 ui32 同摘）：v1 布局对非 float 的 transpose 只给
    // 32B（4 字节整型折 8 元素），而 ConfusionTranspose 公式写死尾轴 16 元素对齐——布局
    // 与公式错位，数值必错（legacy 生产同错，其 ST 因手搓布局掩盖且无 int32 用例；实测
    // 1700/1702）。reg 档不受影响（regbase 显式传 strides）。根因链、实测证据与恢复条件
    // 见 `Implementations/Transpose.cpp` 的「口径 A 修订」注释与设计文档
    // `2026-10-19-afir-transpose-e2e-design.md` 收尾结论。补齐 = v1 布局/公式口径对齐后，
    // 把 `kTransposeMemElementTypes` 加回 kI32 并删本行。
    {"transpose.ub_ub_extend", kF16 | kF32 | kI16 | kUI16, kF16 | kF32 | kI16 | kI32 | kUI16,
     "i32/ui32 在 mem 档因 v1 布局(8元素)与 ConfusionTranspose 公式(写死16元素)错位而摘除，"
     "legacy 生产同错；reg 档保留（regbase 显式传 strides）"},
};

// ODS 比 legacy 的**声明**宽，且宽得有理由。**这不是豁免，是快照**：多出来的那几位逐位
// 钉死，任何一边动了都会红，逼着动的人回来重读理由。
//
// 判据二（下面那条 `OdsElementTypesAgreeWithRegistrationsAndLegacy`）原本是无条件的：
// 「ODS 声称一个 legacy 两侧都不供的 dtype，那一位就是假话——症状是图建得出来、选不出
// implementation、白回落」。那条推理有一个前提：**legacy 的 `REG_ASC_IR` 声明 = 两侧
// AscendC 头真正供得起的集合**。`afir.broadcast` 上这个前提不成立——
// `v35/ascendc/api_regbase/broadcast_extend_impl.h:260` 的
// `static_assert(SupportBytes<T, 1, 2, 4, 8>())` 收 bf16，而 `REG_ASC_IR(Broadcast)` 两侧
// 都没声明它。功能需求 3 的口径是**以头文件为准，不以 legacy 注册表为准**，所以这一位该开。
//
// **这张表不会把判据二变成一句空话**，因为下面多查一条：钉在这里的每一位都必须**真有
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
    // **交集**：取并集的话，在窄的那一侧会发出一个 legacy 根本没注册的调用。
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

// **ODS 的元素类型列表必须等于注册处那两份位掩码的并集**（投影到 AFIR 的供给表上）。
//
// 这是功能需求 3 第一步的那条门禁。判据不是人定的口径，而是「AFIR 能不能表达某个 dtype」
// 这件事的定义：AFIR 表达得出来，当且仅当至少有一侧的 implementation 供得起——所以 ODS
// 那一份是**推导得出来的**，不是独立事实。
//
// 之前没有任何机器查过这条关系，于是两个方向都出过事，而且**两个方向都不会红**：
//
//   - **注册处宽、ODS 窄**：注册的那一位是句没人读的假话。op 根本建不出来，这个
//     implementation 永远选不到。文件头记的 2026-09-23 清掉的 17 处就是这一类，当时是人
//     逐个读出来的。
//   - **ODS 宽、注册处窄**：op 建得出来、选不出 implementation，整图干净回落。不报错，
//     只是白回落，而接管表上这一格的首因会指向「没有 Ascend 实现」而不是「这一位没注册」
//     ——排期的人按首因挑活，就会去改错的那一层。
//
// **投影到供给表上**这一步是必须的，不是放宽：`load` / `store` / `cast` / `broadcast` 的
// ODS 列表里有 i8 / i64 / ui32 / ui64 这些**在 AFIR 的位表上根本没有位**的类型。它们在那里
// 是故意的（搬运与 cast 要能承载还没建模的 dtype），但任何位掩码都表示不出它们，所以拿它们
// 来比就只能永远失败。
// **ODS 的元素类型列表与注册处的关系有两条判据，而且两条都有方向。**
//
// 这是功能需求 3 第一步的那条门禁。写成「ODS == 注册处的并集」是错的，而且错得有后果，
// 所以把两个方向分开记清楚：
//
//   1. **注册处 ⊆ ODS**（硬规则）。注册了一位而 ODS 拒它，那一位就是句没人读的假话：op
//      根本建不出来，这个 implementation 永远选不到。`ElementTypeMatrixTest` 文件头记的
//      2026-09-23 清掉的 17 处就是这一类，当时是人逐个读出来的。
//
//   2. **ODS ⊆ legacy 两列的并集**（投影到 AFIR 的供给表上）。ODS 声称一个 legacy 两侧都
//      不供的 dtype，那一位同样是假话——只是症状换成了「图建得出来、选不出 implementation、
//      白回落」，一条诊断都没有。**这一条有一张例外快照**（`kOdsWiderThanLegacy`）：
//      legacy 的声明不等于 AscendC 头供得起的集合，两者不一致时按头文件走。那张表上每一位
//      都要有 implementation 真注册着它，所以「白回落」仍然挡着，理由见那张表上面那段。
//
// **反过来「ODS 比注册处宽」不是缺陷，是刻意的，而且是硬要求。** 理由见 `Ops/Data.td` 里
// `AFIR_DataMoveElementTypes` 上那段：一个 dtype 只要过了 bridge 的 `IsAfirModeledDType`，
// 那张图里每个 op 都会被建出来；ODS 不收它就**失败在 verifier 上，那是硬失败**，把本可以
// 干净回落的图变红。所以「ODS 收、但没有 implementation 能降」正是要的状态。
//
// 这个缝隙对 `ApiFamily::Any` / `Shared` 的 op 是**必然存在**的：一份 implementation 服务
// 两侧，收的集合只能是两列的**交集**（取并集会在窄的那一侧发出一个 legacy 根本没注册的
// 调用），而 ODS 要盖住两列的**并集**。所以 add / mul / maximum / minimum / axpy / exp /
// ln / sqrt 这 8 个 op 上 bf16 在 ODS 里而不在注册处里，是对的——要让它真能降下去，得先
// 把那个 implementation 拆成 mem / reg 两份，而不是去删 ODS 的那一位。
//
// **投影到供给表上**这一步是必须的，不是放宽：`load` / `store` / `cast` / `broadcast` 的
// ODS 列表里有 i8 / i64 / ui32 / ui64 这些在 AFIR 的位表上根本没有位的类型，任何位掩码都
// 表示不出它们，拿它们来比只能永远失败。
TEST(ElementTypeMatrixTest, OdsElementTypesAgreeWithRegistrationsAndLegacy) {
  const llvm::StringMap<MatrixRow> matrix = readMatrix(AF_MLIR_OP_LOWERING_MATRIX_CSV);
  ASSERT_FALSE(matrix.empty()) << "读不到对照表 " << AF_MLIR_OP_LOWERING_MATRIX_CSV;

  MLIRContext context;
  context.getOrLoadDialect<afir::AFIRDialect>();
  ascend::instruction::ImplementationRegistry registry(context);
  ASSERT_TRUE(succeeded(registerAscendCImplementations(registry)));

  // afir op 名 → ODS 那一份元素类型列表。展开 bridge 那张 op 类名单，所以这里不会有第二
  // 份名单；新加一个 bridge op 而忘了给它的 ODS 基类挂上 `AFIR_OdsElementTypes` 的话，
  // 这个 TU 直接编不过，而不是静默少查一个 op。
  llvm::StringMap<llvm::StringRef> odsSpellings;
  llvm::StringMap<llvm::StringRef> legacyName;
#define AFIR_TEST_ODS_ELEMENT_TYPES(SCHEDULER_NAME, OP, FORM)              \
  odsSpellings[OP::getOperationName()] = OP::getOdsElementTypeSpellings(); \
  legacyName[OP::getOperationName()] = SCHEDULER_NAME;
  AFIR_BRIDGE_OPERATION_REGISTRY(AFIR_TEST_ODS_ELEMENT_TYPES)
#undef AFIR_TEST_ODS_ELEMENT_TYPES
  ASSERT_FALSE(odsSpellings.empty());

  // 每个 op 把它各个 implementation 的集合并起来。
  llvm::StringMap<uint32_t> registeredUnion;
  for (const RegisteredFamily &row : registeredFamilies()) {
    if (row.elementTypes == 0U) {
      continue;  // 逐行对表那一档（cast / logical_not），集合不被读。
    }
    registeredUnion[row.afirOpName] |= row.elementTypes;
  }
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

    // ODS 列表里认得出的那几位。认不出的拼写要当场报出来——那说明 MLIR 的类型记录名字
    // 与供给表的 CSV 拼写不再是「小写一下就相等」，而那个巧合是这条门禁的前提。
    uint32_t odsBits = 0U;
    llvm::SmallVector<llvm::StringRef> parts;
    ods->second.split(parts, ';', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
    for (llvm::StringRef part : parts) {
      const uint32_t bit = afir::afirElementTypeBitFromCsvSpelling(part);
      if (bit != 0U) {
        odsBits |= bit;
        continue;
      }
      // 供给表上没有位的那些（i8 / i64 / ui32 / ui64 / i4）是故意留在 ODS 里的，跳过；
      // 其余一律是拼写对不上，必须红。
      EXPECT_TRUE(part == "i8" || part == "i64" || part == "ui32" || part == "ui64" || part == "i4")
          << afirOpName.str() << " 的 ODS 列表里有一个供给表认不出的拼写 `" << part.str()
          << "`：要么它该进 AFIR_ELEMENT_TYPE_TABLE / AFIR_ELEMENT_TYPE_NAMED_ONLY，"
          << "要么 MLIR 的类型记录名字变了，这条门禁的拼写前提已经不成立";
    }

    uint32_t legacySupplies = maskOf(matrixRow->second.memDtypes) | maskOf(matrixRow->second.regDtypes);
    // 快照上钉着的那几位：legacy 只是没声明，头文件供得起。放进 `legacySupplies` 之前先查
    // 它**真有 implementation 注册着**——否则这张表就替判据二挡住了它本该抓的「白回落」。
    if (const KnownWiderRow *wider = findKnownWider(afirOpName)) {
      EXPECT_EQ(wider->extraBits & ~odsBits, 0U)
          << afirOpName.str() << " 的 ODS 已经不收 " << afir::afirDescribeElementTypes(wider->extraBits & ~odsBits)
          << " 了：回 kOdsWiderThanLegacy 删掉这一行";
      EXPECT_EQ(wider->extraBits & ~registered, 0U)
          << afirOpName.str() << " 在 kOdsWiderThanLegacy 上钉着 " << afir::afirDescribeElementTypes(wider->extraBits)
          << "，但没有任何 implementation 注册 " << afir::afirDescribeElementTypes(wider->extraBits & ~registered)
          << "：那就是判据二要抓的「ODS 收、没人供、白回落」，不是「legacy 没声明」。"
          << "要么去注册处开位，要么把这一行删掉";
      EXPECT_EQ(wider->extraBits & legacySupplies, 0U)
          << afirOpName.str() << " 的 legacy 声明已经收了 "
          << afir::afirDescribeElementTypes(wider->extraBits & legacySupplies) << "：这一位不再是例外，"
          << "回 kOdsWiderThanLegacy 删掉它、让它回到判据二，并重读当初的理由 " << wider->reason.str();
      legacySupplies |= wider->extraBits;
    }
    ++checked;

    // 判据一：注册处 ⊆ ODS。
    EXPECT_EQ(registered & ~odsBits, 0U) << afirOpName.str() << " 注册了 ODS 不收的位：注册处是 "
                                         << afir::afirDescribeElementTypes(registered) << "，ODS 认得出的是 "
                                         << afir::afirDescribeElementTypes(odsBits) << "，多出来的 "
                                         << afir::afirDescribeElementTypes(registered & ~odsBits)
                                         << " 是句没人读的假话（op 建不出来，这个 implementation 永远选不到）";

    // 判据二：ODS ⊆ legacy 两列的并集。
    EXPECT_EQ(odsBits & ~legacySupplies, 0U)
        << afirOpName.str() << " 的 ODS 声称了 legacy 两侧都不供的位："
        << afir::afirDescribeElementTypes(odsBits & ~legacySupplies) << "（ODS 认得出 "
        << afir::afirDescribeElementTypes(odsBits) << "，legacy 的 " << legacy->second.str()
        << " 两列并起来是 mem=" << matrixRow->second.memDtypes << " reg=" << matrixRow->second.regDtypes
        << "）。那一位该从 ODS 列表里删掉——留着的后果是图建得出来、选不出 implementation、白回落";
  }
  EXPECT_GT(checked, 0u) << "一条都没对上说明侧表没填或者表读空了";

  // 反向断言：快照上的 op 必须真的还被上面那个循环查着。它从 `registeredUnion` 里消失的
  // 话，那一行就会静默停止生效，而不是红。
  for (const KnownWiderRow &row : kOdsWiderThanLegacy) {
    EXPECT_TRUE(registeredUnion.contains(row.afirOpName))
        << row.afirOpName.str() << " 已经没有带元素类型集合的 implementation 了，删掉这一行";
  }
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
