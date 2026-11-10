// -----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------

//===- AFPlanDialect.td - afplan dialect definition --------*- tablegen -*-===//
//
// `afplan` 是四层命名空间里的候选身份层（设计文档 2.2 / 2.3 / ADR-5）。
//
// 判别：当候选集中只有一个候选，这条信息还影响生成的 kernel 代码吗？
//   是 → `afir`（公开面）。例：`tiling_key` 仍决定 TilingData 结构与 entry 分派。
//   否，只影响候选之间的区分与分派 → `afplan`（内部）。
//
// 它是仅属性方言：无 Op、无 Type。注册成本接近零，但把「这是哪个候选、凭什么选
// 它」从 `afir` 的公开面里物理隔离出来——对外公开时用户不应看到 `score_func` 这类
// autofuse 内部字段。
//
//===----------------------------------------------------------------------===//

#ifndef AFPLAN_DIALECT
#define AFPLAN_DIALECT

include "mlir/IR/OpBase.td"

def AFPlan_Dialect : Dialect {
  let name = "afplan";
  let summary = "Candidate identity and selection metadata for AFIR plans";
  let description = [{
    Attribute-only dialect carrying which candidate a module or function is, and
    what the runtime needs in order to dispatch among candidates. It has no
    operations and no types: candidate identity is metadata about AFIR payload,
    never payload itself.

    Nothing in the `afir` dialect depends on `afplan`; the dependency is strictly
    one-directional (`afplan` attaches to `afir` containers, not the reverse).
  }];

  let cppNamespace = "::mlir::afplan";

  let useDefaultAttributePrinterParser = 1;
  let usePropertiesForAttributes = 0;

  let extraClassDeclaration = [{
    // `afplan.*` 是 discardable 属性：拼错名字不会报错，只会被无声忽略，失败形态是
    // 候选分派静默走错分支。与 `afir` 的同名门禁一致，这里把未登记的 `afplan.` 前缀
    // 属性直接判错。
    LogicalResult verifyOperationAttribute(Operation *op, NamedAttribute attribute) override;
  }];
}

class AFPlan_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<AFPlan_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

#endif // AFPLAN_DIALECT
