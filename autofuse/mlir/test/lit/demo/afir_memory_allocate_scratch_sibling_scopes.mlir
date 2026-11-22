// RUN: af-opt --afir-kernel-body-signature --afir-memory-allocate %s | FileCheck %s
// Distinct scratch views share a physical ID across sibling loops. The raw
// TBuf view must dominate both loops; scoping it at the first use is invalid SSA.
#input = #afir.tensor_desc<axis = [0, 1], repeats = [<"2">, <"8">], strides = [<"8">, <"1">], vectorized_axis = [0, 1], vectorized_strides = [<"8">, <"1">], role = <compute>>
#output = #afir.tensor_desc<axis = [0, 1], repeats = [<"2">, <"1">], strides = [<"1">, <"0">], vectorized_axis = [0, 1], vectorized_strides = [<"1">, <"0">], role = <compute>>
module {
  func.func private @sibling_scopes() attributes {
    afir.arg_descs = [], afir.function_role = #afir.function_role<kernel_body>,
    afir.asc_graph = #afir.asc_graph<axes = [<id = 0, name = "A", axis_type = Original, size = <"2">>, <id = 1, name = "R", axis_type = Original, size = <"8">>, <id = 2, name = "first", axis_type = Original, size = <"2">>, <id = 3, name = "second", axis_type = Original, size = <"2">>], type = Compute>
  } {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %input = memref.alloc() {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 0, size_bytes_expr = <"64">, life_time_axis_id = -1>} : memref<2x8xf32>
    %dst = memref.alloc() {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 4, size_bytes_expr = <"32">, life_time_axis_id = -1>} : memref<2x1xf32>
    scf.for %i2 = %c0 to %c2 step %c1 {
      %api2 = memref.alloc() {afir.scratch_scope = 2 : i64, afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 1, size_bytes_expr = <"256">, life_time_axis_id = -1>} : memref<256xui8>
      %round2 = memref.alloc() {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 2, size_bytes_expr = <"32">, life_time_axis_id = -1>} : memref<32xui8>
      %sum2 = afir.sum %input outs(%dst) scratch(%api2, %round2) {afir.scratch_roles = ["api_scratch", "round_result"], afir.operand_descs = [#input], afir.destination_descs = [#output], afir.result_descs = [#output]} : memref<2x8xf32>, memref<2x1xf32>, scratch(memref<256xui8>, memref<32xui8>) -> memref<2x1xf32>
    } {afir.axis_binding = #afir.axis_binding<axis = 2, role = loop>}
    scf.for %i3 = %c0 to %c2 step %c1 {
      %api3 = memref.alloc() {afir.scratch_scope = 3 : i64, afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 1, size_bytes_expr = <"256">, life_time_axis_id = -1>} : memref<256xui8>
      %round3 = memref.alloc() {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 3, size_bytes_expr = <"32">, life_time_axis_id = -1>} : memref<32xui8>
      %sum3 = afir.sum %input outs(%dst) scratch(%api3, %round3) {afir.scratch_roles = ["api_scratch", "round_result"], afir.operand_descs = [#input], afir.destination_descs = [#output], afir.result_descs = [#output]} : memref<2x8xf32>, memref<2x1xf32>, scratch(memref<256xui8>, memref<32xui8>) -> memref<2x1xf32>
    } {afir.axis_binding = #afir.axis_binding<axis = 3, role = loop>}
    return
  }
}
// CHECK-LABEL: func.func private @sibling_scopes
// CHECK: ascendc.tbuf.get_tensor
// CHECK: scf.for
// CHECK: ascendc.reinterpret_cast %[[RAW:[a-zA-Z0-9_]+]]
// CHECK-SAME: buffer_id = 1
// CHECK-SAME: afir.scratch_scope = 2 : i64
// CHECK: afir.sum
// CHECK: scf.for
// CHECK: ascendc.reinterpret_cast %[[RAW]]
// CHECK-SAME: buffer_id = 1
// CHECK-SAME: afir.scratch_scope = 3 : i64
// CHECK: afir.sum
