// 失败注入：memory-allocate 中途失败后 rollback 不崩（FR6 回归）。
//
// 取 `afir-transpose-reg-base.mlir` 的合法输入，把 #dst_tile 的 vectorized_axis 改出
// 一个不在 axis 列表里的 id（9）。descriptor 解析与绑定校验都放行，失败发生在第 4 相
// materializeCanonicalValues 的队列成员尺寸物化（CanonicalMaterialization.cpp 的
// materializeVectorizedAxisSpan）——此时 createCanonicalResources 已经记录了
// TQue/TBuf/pipe/断言链等回滚条目，rollback() 带着非空记录表执行。钉住三件事：进程
// 不崩、`error:` 打印、退出码非零（`not` 断言）。
//
// RUN: not af-opt --target=Ascend950DT_950x --afir-kernel-body-signature --afir-materialize-tiling-params \
// RUN:   --afir-loop-generate --afir-memory-allocate %s 2>&1 | FileCheck %s

// CHECK: error: InvalidDescriptor: vectorized axis 9 is not present in the queue member axis list

#axes = #afir.asc_graph<
  tiling_key = 0,
  axes = [<id = 0, name = "b", axis_type = Original, size = #afir.expr<"1024">, split_pair_other_id = 0>,
          <id = 1, name = "z", axis_type = Original, size = #afir.expr<"16">, split_pair_other_id = 1>,
          <id = 2, name = "w", axis_type = Original, size = #afir.expr<"8">, split_pair_other_id = 2>,
          <id = 3, name = "bt", axis_type = TileInner, size = #afir.expr<"bt_size">, from = [0],
           split_pair_other_id = 6>,
          <id = 4, name = "bTB", axis_type = BlockOuter,
           size = #afir.expr<"Ceiling((Ceiling((1024 / (bt_size))) / (bTb_size)))">, from = [6],
           split_pair_other_id = 5>,
          <id = 5, name = "bTb", axis_type = BlockInner, size = #afir.expr<"bTb_size">, from = [6],
           split_pair_other_id = 4>,
          <id = 6, name = "bT", axis_type = TileOuter, size = #afir.expr<"Ceiling((1024 / (bt_size)))">,
           from = [0], split_pair_other_id = 3>],
  type = Compute,
  size_var = [#afir.expr<"bt_size">, #afir.expr<"bTb_size">]>

// 输入 tile：`(bt, z, w)` 布局，行距 8（w 连续、z 步 8、bt 步 128 = 16×8）。
#src_tile = #afir.tensor_desc<axis = [4, 5, 3, 1, 2],
    repeats = [#afir.expr<"(1024 / (bTb_size * bt_size))">, #afir.expr<"bTb_size">, #afir.expr<"bt_size">,
               #afir.expr<"16">, #afir.expr<"8">],
    strides = [#afir.expr<"(128 * bTb_size * bt_size)">, #afir.expr<"(128 * bt_size)">, #afir.expr<"128">,
               #afir.expr<"8">, #afir.expr<"1">],
    vectorized_axis = [3, 1, 2],
    vectorized_strides = [#afir.expr<"128">, #afir.expr<"8">, #afir.expr<"1">],
    role = <compute>>

// 输出 tile：`(bt, w, z)` 布局——同一批元素换成 z 连续（z 步 1、w 步 16、bt 步 128 = 8×16）。
// `vectorized_axis.front()` 仍是 `bt`，切分轴不动，perm 全在内层。
#dst_tile = #afir.tensor_desc<axis = [4, 5, 3, 2, 1],
    repeats = [#afir.expr<"(1024 / (bTb_size * bt_size))">, #afir.expr<"bTb_size">, #afir.expr<"bt_size">,
               #afir.expr<"8">, #afir.expr<"16">],
    strides = [#afir.expr<"(128 * bTb_size * bt_size)">, #afir.expr<"(128 * bt_size)">, #afir.expr<"128">,
               #afir.expr<"16">, #afir.expr<"1">],
    vectorized_axis = [3, 2, 9],
    vectorized_strides = [#afir.expr<"128">, #afir.expr<"16">, #afir.expr<"1">],
    role = <compute>>

// GM 侧入口（与 `afir-transpose-mem-extend.mlir` 同一张图的 f32 版）：`afir.load` 把 tile
// 搬进 UB，maps 按 axis 并集规则机械生成（domain [0,1,2 | 4,5,3]）。
#gm_in = #afir.tensor_desc<axis = [0, 1, 2],
    repeats = [#afir.expr<"1024">, #afir.expr<"16">, #afir.expr<"8">],
    strides = [#afir.expr<"128">, #afir.expr<"8">, #afir.expr<"1">], role = <input>>

func.func private @kept_transpose_tile(
    %input: memref<1024x16x8xf32> {afir.kernel_arg = #afir.kernel_arg<role = input, index = 0>,
                                   afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"524288">,
                                                                        life_time_axis_id = -1>}) attributes {
  afir.arg_descs = [#gm_in],
  afir.asc_graph = #axes,
  afir.function_role = #afir.function_role<kernel_body>
} {
  %tile = afir.tiling_param "bt_size" : index
  %block = afir.tiling_param "bTb_size" : index
  %outer = afir.symbolic_eval %block, %tile {expr = #afir.expr<"(1024 / (__arg0 * __arg1))">}
      : (index, index) -> index
  %src = memref.alloc(%outer, %block, %tile) {
      afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 0, reuse_id = 0,
                                           size_from_resource = true, life_time_axis_id = -1>}
      : memref<?x?x?x16x8xf32>
  %dst = memref.alloc(%outer, %block, %tile) {
      afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 0, reuse_id = 0, slot_count = 1,
                                           queue_depth = 2, size_from_resource = true,
                                           life_time_axis_id = -1>}
      : memref<?x?x?x8x16xf32>
  // tmp 是全生命周期 TBuf（`life_time_axis_id = -1`、无队列字段），独占整块资源；元素类型
  // ui8——两个包装的形参都是 `LocalTensor<uint8_t>`（字节scratch，不是同 T 的数据缓冲）。
  %tmp = memref.alloc(%tile) {
      afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 1, reuse_id = 0,
                                           size_from_resource = true, life_time_axis_id = -1>}
      : memref<?xui8>
  %loaded = afir.load %input outs(%src) {
    afir.destination_descs = [#src_tile],
    afir.operand_descs = [#gm_in],
    afir.result_descs = [#src_tile],
    afir.sched = #afir.sched<axis = [4, 5, 3, 1, 2], loop_axis = 5>,
    indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4, d5, d1, d2)>]
  } : memref<1024x16x8xf32>, memref<?x?x?x16x8xf32> -> memref<?x?x?x16x8xf32>
  %result = afir.transpose %loaded outs(%dst) tmp(%tmp) {
    afir.operand_descs = [#src_tile],
    afir.destination_descs = [#dst_tile],
    afir.result_descs = [#dst_tile],
    afir.sched = #afir.sched<axis = [4, 5, 3, 1, 2], loop_axis = 5>,
    indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4, d3)>]
  } : memref<?x?x?x16x8xf32>, memref<?x?x?x8x16xf32>, memref<?xui8> -> memref<?x?x?x8x16xf32>
  return
}
