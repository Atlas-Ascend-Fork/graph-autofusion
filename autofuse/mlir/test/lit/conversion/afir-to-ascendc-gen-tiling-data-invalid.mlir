// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// RUN: af-opt --afir-gen-tiling-data --verify-diagnostics --split-input-file %s

// expected-error @+1 {{GenTilingData: expected afir.module_kind=plan}}
module @not_a_plan {
}

// -----

// expected-error @+1 {{GenTilingData: missing Plan module symbol}}
module attributes {
  afir.module_kind = "plan"
} {
}

// -----

// expected-error @+1 {{GenTilingData: Cube/CV Plan is not supported, afir.cube_type=1}}
module @cube_plan attributes {
  afir.module_kind = "plan",
  afir.cube_type = 1 : i64
} {
}

// -----

// stage/template 坐标合并为 afplan.variant 后，不存在只提供半个坐标的形态；
// 类型错误由方言校验，见 smoke/afplan_attrs_invalid.mlir。
// 本用例检查没有任何 template 的 plan：未声明 variant 的函数不计为 template。
// expected-error @+1 {{GenTilingData: Plan has no templates}}
module @no_templates attributes {
  afir.module_kind = "plan"
} {
  func.func private @not_a_template(%input: tensor<1xf16>) {
    return
  }
}

// -----

module @missing_graph_facts attributes {
  afir.module_kind = "plan",
  afplan.candidate = #afplan.candidate<subgraph = 0, plan = 0>
} {
  // expected-error @+1 {{GenTilingData: template is missing afir.asc_graph facts}}
  func.func private @stage0_template0(%input: tensor<1xf16>)
      attributes {afplan.variant = #afplan.variant<stage = 0, template = 0>} {
    return
  }
}

// -----

module @conflicting_api_tiling_fields attributes {
  afir.module_kind = "plan",
  afplan.candidate = #afplan.candidate<subgraph = 0, plan = 0>
} {
  // expected-error @+1 {{GenTilingData: conflicting API tiling schema for field api}}
  func.func private @stage0_template0(%input: tensor<1xf16>) attributes {
      afir.asc_graph = #afir.asc_graph<tiling_key = 0, axes = [
        <id = 0, name = "x", axis_type = Original, size = #afir.expr<"1">>], type = Compute>,
      afir.api_tiling_fields = [{field = "api", type = "Api", fields = ["a"]}],
      afplan.variant = #afplan.variant<stage = 0, template = 0>
  } {
    return
  }
  func.func private @stage0_template1(%input: tensor<1xf16>) attributes {
      afir.asc_graph = #afir.asc_graph<tiling_key = 1, axes = [
        <id = 0, name = "x", axis_type = Original, size = #afir.expr<"1">>], type = Compute>,
      afir.api_tiling_fields = [{field = "api", type = "Api", fields = ["b"]}],
      afplan.variant = #afplan.variant<stage = 0, template = 1>
  } {
    return
  }
}

// -----

// 缺少候选身份时无法生成与 ATT 一致的 stage 类型名和成员名，必须提前报错。
// 不能使用默认名称，否则会延迟到 host 编译时出现 incomplete type 错误。
// expected-error @+1 {{GenTilingData: missing afplan.candidate}}
module @missing_candidate attributes {
  afir.module_kind = "plan"
} {
  func.func private @stage0_template0(%input: tensor<1xf16>) attributes {
      afir.asc_graph = #afir.asc_graph<tiling_key = 0, axes = [
        <id = 0, name = "x", axis_type = Original, size = #afir.expr<"1">>], type = Compute>,
      afplan.variant = #afplan.variant<stage = 0, template = 0>
  } {
    return
  }
}

// -----

// expected-error @+1 {{GenTilingData: missing Plan module symbol}}
module attributes {afir.module_kind = "scheduled_result"} {
}
