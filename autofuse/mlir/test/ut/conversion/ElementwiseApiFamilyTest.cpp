/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- ElementwiseApiFamilyTest.cpp - 家族 <-> 头名 的注册期判据 -*- C++ -*-===//
//
// 这条判据此前没有任何用例：它是「写反了就把整张注册表拆掉」，而没人写反过，所以把
// 它整个删成 `return success()` 不会有一处变红——正是那种看着有、实际没被验过的判据。
//
// 判据本身很小，但它挡住的失败是静默的：家族与头名对不上时，capability 照答
// Supported、`AddKernelIncludes` 照生成 include、那个 key 在支持表里也解析得到，于是一份
// reg-base 的头文本被内联进一个 mem-based 的 kernel。今天两份 `neg.h` 除 include guard
// 外逐字相同所以看不出来，`broadcast.h` 那一对不是。
//
//===----------------------------------------------------------------------===//

#include "conversion/AFIRToAscendC/Implementations/ElementwiseApiFamily.h"

#include "gtest/gtest.h"

namespace mlir::ascend::kernel::elementwise {
namespace {

TEST(ElementwiseApiFamily, AcceptsAListFromTheFamilyThatSuppliesIt) {
  const llvm::StringRef mem[] = {"abs.h"};
  const llvm::StringRef reg[] = {"neg_reg_base.h"};
  EXPECT_TRUE(headersMatchFamily(mem, {}, ApiFamily::MemBased));
  EXPECT_TRUE(headersMatchFamily(reg, {}, ApiFamily::RegBased));
}

TEST(ElementwiseApiFamily, RejectsAListFromTheOtherFamily) {
  const llvm::StringRef mem[] = {"abs.h"};
  const llvm::StringRef reg[] = {"neg_reg_base.h"};
  EXPECT_FALSE(headersMatchFamily(mem, {}, ApiFamily::RegBased));
  EXPECT_FALSE(headersMatchFamily(reg, {}, ApiFamily::MemBased));
}

// 扩展包装总是出自某一张支持表。`Any` 是「两侧同一个标准调用」的意思，它和「引用了
// 一个扩展头」不可能同时成立——放过去的后果是这个 implementation 在两个平台上都被选中，
// 其中一个平台内联进来的是另一张表的文本。
TEST(ElementwiseApiFamily, RejectsAnyForAnOperationThatNamesAnExtensionHeader) {
  const llvm::StringRef mem[] = {"abs.h"};
  const llvm::StringRef reg[] = {"neg_reg_base.h"};
  EXPECT_FALSE(headersMatchFamily(mem, {}, ApiFamily::Any));
  EXPECT_FALSE(headersMatchFamily(reg, {}, ApiFamily::Any));
}

// 前导头与扩展头一起判，不各判各的。
//
// 原先它们分开判，理由是「前导头有各自的族」——写的是 `utils.h` / 将来的
// `utils_reg_base.h`。这个前提是错的：`utils_reg_base.h` 不是 `utils.h` 的 reg 版本，
// 它是另一个文件（micro-API 的 `CastTrait` 常量表），不声明 `MAX_REPEAT_TIME` /
// `Rational`。而 reg 侧的 `sign_reg_base.h` 用了 `MAX_REPEAT_TIME`，**它要的就是 mem
// 那份 `utils.h`**。分开判的话这条注册直接被拒。
TEST(ElementwiseApiFamily, AcceptsTheSharedPreludeOnEitherFamily) {
  const llvm::StringRef memHeaders[] = {"cast.h", "sign.h"};
  const llvm::StringRef regHeaders[] = {"cast.h", "sign_reg_base.h"};
  const llvm::StringRef sharedPrelude[] = {"utils.h"};

  EXPECT_TRUE(headersMatchFamily(memHeaders, sharedPrelude, ApiFamily::MemBased));
  EXPECT_TRUE(headersMatchFamily(regHeaders, sharedPrelude, ApiFamily::RegBased));

  // 家族仍然由扩展头那边定死，前导中立不影响它。
  EXPECT_FALSE(headersMatchFamily(memHeaders, sharedPrelude, ApiFamily::RegBased));
  EXPECT_FALSE(headersMatchFamily(regHeaders, sharedPrelude, ApiFamily::MemBased));
}

// 前导头里出现 `_reg_base.h` 照样把整个 op 判成 reg 侧的——这一条保证前导不是判据的
// 盲区：一个 mem 侧的 op 写 `utils_reg_base.h` 当场拒。
TEST(ElementwiseApiFamily, CountsThePreludeWhenItNamesRegBasedSupply) {
  const llvm::StringRef regPrelude[] = {"utils_reg_base.h"};
  EXPECT_FALSE(headersMatchFamily({}, regPrelude, ApiFamily::MemBased));
  EXPECT_TRUE(headersMatchFamily({}, regPrelude, ApiFamily::RegBased));
}

// 这条判据原先是「逐项都要对」，`sign` 把它推翻了。
//
// 当时的理由是「一张列表里混进一个另一类的名字，跟整条写反一样坏」，前提是每个头都
// 属于某一类。`sign` 不是：它两侧分别是 `{cast.h, sign.h}` 与 `{cast.h, sign_reg_base.h}`,
// `cast.h` 两边是同一个文件——两张注册表灌的是同一个 `AscendCApiRegistry` 单例，
// 它在里面只有一条，reg 侧照样取得到。`broadcast` 的 `duplicate.h` 也是这样。
//
// 所以判的是「这份列表有没有点名 reg-base 可用实现」。
TEST(ElementwiseApiFamily, AcceptsAFamilyNeutralHeaderAlongsideTheFamilysOwn) {
  // `sign` 的两侧，逐字照抄 legacy 的 `LoadApiHeaderFiles`。
  const llvm::StringRef signMem[] = {"cast.h", "sign.h"};
  const llvm::StringRef signReg[] = {"cast.h", "sign_reg_base.h"};
  EXPECT_TRUE(headersMatchFamily(signMem, {}, ApiFamily::MemBased));
  EXPECT_TRUE(headersMatchFamily(signReg, {}, ApiFamily::RegBased));

  // 两侧仍然分得开：mem 那份没点名 reg-base 可用实现，reg 那份点了。
  EXPECT_FALSE(headersMatchFamily(signMem, {}, ApiFamily::RegBased));
  EXPECT_FALSE(headersMatchFamily(signReg, {}, ApiFamily::MemBased));
}

// 松掉的只有「列表里不许出现中立的头」。要挡的那个方向一点没松：mem 侧的列表里出现
// 任何一个 `_reg_base.h` 仍然当场拒——那正是「一份 reg-base 的头文本被内联进 mem kernel」
// 的入口。
TEST(ElementwiseApiFamily, StillRejectsRegBasedSupplyNamedByAMemBasedOperation) {
  const llvm::StringRef mixedTail[] = {"duplicate.h", "where_reg_base.h"};
  EXPECT_FALSE(headersMatchFamily(mixedTail, {}, ApiFamily::MemBased));

  const llvm::StringRef mixedHead[] = {"where_reg_base.h", "duplicate.h"};
  EXPECT_FALSE(headersMatchFamily(mixedHead, {}, ApiFamily::MemBased));

  // 反过来，一个 reg-base 的 op 一个 `_reg_base.h` 都没点名，同样拒——它拿不到自己那张
  // 表的任何可用实现。
  const llvm::StringRef neutralOnly[] = {"cast.h"};
  EXPECT_FALSE(headersMatchFamily(neutralOnly, {}, ApiFamily::RegBased));
}

// 标准调用两张列表都是空的，对任何家族都成立——空列表不是「查不到」，是「没什么要查」。
TEST(ElementwiseApiFamily, AcceptsAnEmptyHeaderListForEveryFamily) {
  EXPECT_TRUE(headersMatchFamily({}, {}, ApiFamily::Any));
  EXPECT_TRUE(headersMatchFamily({}, {}, ApiFamily::MemBased));
  EXPECT_TRUE(headersMatchFamily({}, {}, ApiFamily::RegBased));
}

}  // namespace
}  // namespace mlir::ascend::kernel::elementwise
