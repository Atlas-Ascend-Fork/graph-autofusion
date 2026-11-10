// -----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------

//===- AFPlanAttrs.td - afplan attribute definitions -------*- tablegen -*-===//
//
// 方言名已包含 plan，属性名不重复该词，使用 afplan.candidate 而非 plan_candidate（设计文档 2.3）。
//
//===----------------------------------------------------------------------===//

#ifndef AFPLAN_ATTRS
#define AFPLAN_ATTRS

include "AFPlanDialect.td"
include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// Candidate identity
//===----------------------------------------------------------------------===//

def CandidateAttr : AFPlan_Attr<"Candidate", "candidate"> {
  let summary = "Identity of one plan candidate: which subgraph, which plan";
  let description = [{
    一个候选由 (subgraph, plan) 唯一确定。两者原本是 module 上两个独立的整数属性
    （`afir.subgraph_id` / `afir.plan_id`），每个消费端各读一遍、各校验一遍是否非负；
    聚合入口里还有第三份编码——一个带 `subgraph_id` / `plan_id` 两个键的匿名字典。

    合为一个属性后，「候选身份」只有一种写法，校验也只有一处。
  }];

  let parameters = (ins
    "int64_t":$subgraph,
    "int64_t":$plan
  );

  let assemblyFormat = "`<` `subgraph` `=` $subgraph `,` `plan` `=` $plan `>`";
  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// Variant identity inside one candidate
//===----------------------------------------------------------------------===//

def VariantAttr : AFPlan_Attr<"Variant", "variant"> {
  let summary = "Identity of one template variant inside a plan: stage and template";
  let description = [{
    一个候选内部可以有多个 stage、每个 stage 多个 template；kernel entry 按
    (stage, template) 分派到具体的 kernel body。原本是 func 上两个独立的整数属性
    （`afir.stage_id` / `afir.template_id`），成对读取但可以各自缺失。
  }];

  // 参数名是 `template_id` 而不是 `template`：后者是 C++ 关键字，生成的访问器无法编译。
  // 汇编形式仍写 `template =`，IR 文本不受影响。
  let parameters = (ins
    "int64_t":$stage,
    "int64_t":$template_id
  );

  let assemblyFormat = "`<` `stage` `=` $stage `,` `template` `=` $template_id `>`";
  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// Shape of the candidate's dispatch table
//===----------------------------------------------------------------------===//

def StagesAttr : AFPlan_Attr<"Stages", "stages"> {
  let summary = "Number of template variants per stage in one plan";
  let description = [{
    entry 生成器用它铺开 (stage, template) 分派表。

    原本还有一个并行的 `afir.stage_count`，其值恒等于本数组的长度，且**全仓无读点**
    ——与步 2 清掉的那批影子属性同一形态，随本步一并删除。
  }];

  let parameters = (ins
    ArrayRefParameter<"int64_t", "template count of each stage">:$template_counts
  );

  let assemblyFormat = "`<` `template_counts` `=` `[` $template_counts `]` `>`";
  let genVerifyDecl = 1;
}

#endif // AFPLAN_ATTRS
