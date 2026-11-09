// RUN: af-opt --afir-gen-kernel-entry %s | FileCheck %s
// Static selection keeps template 7 rather than renumbering it to template 0.
module @plan2 attributes {
  afir.module_kind = "plan",
  afplan.static_selection,
  afplan.candidate = #afplan.candidate<subgraph = 0, plan = 2>,
  afplan.stages = #afplan.stages<template_counts = [1]>,
  ascendc.tiling_schema = !emitasc.py_struct<"plan2TilingData", [
    !emitasc.py_struct<"AscGraph0ScheduleResult2G0TilingData", [ui32], ["tiling_key"]>],
    ["graph0_result2_g0_tiling_data"]>
} {
  func.func private @stage0_template7(
    %tiling: !emitasc.py_struct<"AscGraph0ScheduleResult2G0TilingData", [ui32], ["tiling_key"]>)
    attributes {
      afir.asc_graph = #afir.asc_graph<tiling_key = 19, axes = [<id = 0, name = "x", axis_type = Original, size = #afir.expr<"1">>], type = Compute>,
      ascendc.num_inputs = 0 : i32,
      afplan.variant = #afplan.variant<stage = 0, template = 7>
    } { return }
}
// CHECK: func.func @plan2_main
// CHECK: func.call @stage0_template7
// CHECK-NOT: func.call @stage0_template0
