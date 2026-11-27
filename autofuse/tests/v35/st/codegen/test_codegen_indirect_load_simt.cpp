/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * You can obtain a copy of the License in the root of the software repository.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// [IndirectLoad SIMT ST 覆盖] codegen_st 直驱 IndirectLoadRegApiCall（与既有
// test_codegen_broadcast_reg_api_call.cpp 同法）：现有 ST e2e 组图全部走 Gather op
// （GatherRegApiCall），IndirectLoad 的 SIMT lowering/双发/guard 路径在 ST 模块内
// 从未被执行（ST_report 增量覆盖率门禁 15.6% < 80% 的根因）。本文件按真实调度后
// 中间态手写 fixture，覆盖动态 embedding 的三条改动路径：
//   1) 单 gather 双发（分类/双发判定/guard/调用点，含定义与调用两侧）；
//   2) fused 组双发（GenerateSimt 的 fused 发射分支与 fused guard）；
//   3) post-Reduce 放开双发（定义侧双套 body）。
// fixture 与 UT（test_codegen_indirect_load_api_call.cpp）同构、经实跑验证。

#include <gtest/gtest.h>

#include <sstream>
#include <string>

#include "ascendc_ir.h"
#include "ascir_ops.h"
#include "ascir_ops_utils.h"
#include "codegen_kernel.h"
#include "graph/ascendc_ir/utils/asc_tensor_utils.h"
#include "indirect_load_utils.h"
#include "platform_context.h"
#include "runtime_stub.h"
#include "v35/codegen/reg_api_call/reg_indirect_load_api_call.h"
#include "v35/codegen/vec_func_call/vec_func_call.h"
#include "graph/utils/node_utils_ex.h"

namespace {
using namespace af::ops;
using namespace af::ascir_op;
using namespace codegen;
using namespace ascgen_utils::indirect_load;

// SIMT lowering 查询平台、TPipe 队列分配经 RuntimeStub——与 UT 侧 SimtCodegenContext
// 同构的桩上下文（ST 二进制无框架级初始化，按用例自持）。
class IndirectLoadSimtStContext {
 public:
  IndirectLoadSimtStContext() {
    ge::PlatformContext::GetInstance().Reset();
    ge::PlatformContext::GetInstance().SetPlatform("3510");
    ge::RuntimeStub::SetInstance(std::make_shared<ge::RuntimeStubV2Common>());
  }

  ~IndirectLoadSimtStContext() {
    ge::RuntimeStub::Reset();
    ge::PlatformContext::GetInstance().Reset();
  }
};

struct IndirectLoadSimtStGraph {
  af::AscGraph graph;
  af::Expression s0;
  af::Expression s1;
  af::Expression s2;
  af::Expression s3;
  af::Axis z0;
  af::Axis z1;
  af::Axis z2;
  af::Axis z3;

  explicit IndirectLoadSimtStGraph(const char *name)
      : graph(name),
        s0(graph.CreateSizeVar("s0")),
        s1(graph.CreateSizeVar("s1")),
        s2(graph.CreateSizeVar("s2")),
        s3(graph.CreateSizeVar("s3")),
        z0(graph.CreateAxis("z0", s0)),
        z1(graph.CreateAxis("z1", s1)),
        z2(graph.CreateAxis("z2", s2)),
        z3(graph.CreateAxis("z3", s3)) {}

  IndirectLoadSimtStGraph(const char *name, int64_t size0, int64_t size1, int64_t size2, int64_t size3)
      : graph(name),
        s0(af::Symbol(size0)),
        s1(af::Symbol(size1)),
        s2(af::Symbol(size2)),
        s3(af::Symbol(size3)),
        z0(graph.CreateAxis("z0", s0)),
        z1(graph.CreateAxis("z1", s1)),
        z2(graph.CreateAxis("z2", s2)),
        z3(graph.CreateAxis("z3", s3)) {}
};

void SetupIndirectLoadStTiler(const af::AscGraph &graph, Tiler &tiler) {
  for (const auto &axis : graph.GetAllAxis()) {
    if (axis != nullptr) {
      tiler.AddAxis(*axis);
      tiler.AddSizeVar(af::SizeVar(axis->size));
    }
  }
}

ascir::AxisId FindStOuterTbAxis(const af::AscGraph &graph) {
  for (const auto &axis : graph.GetAllAxis()) {
    if (axis != nullptr && axis->name == "z2z3Tb") {
      return axis->id;
    }
  }
  return af::kIdNone;
}

// ==================== 单 gather 动态 embedding（rank2/axis0，符号步长） ====================

void BuildStSimtEmbeddingGraph(IndirectLoadSimtStGraph &g) {
  const af::Expression One = af::sym::kSymbolOne;

  Data x_data("x", g.graph);
  x_data.ir_attr.SetIndex(0);
  x_data.y.dtype = ge::DT_FLOAT16;
  x_data.attr.sched.axis = {g.z0.id, g.z1.id};
  *x_data.y.axis = {g.z0.id, g.z1.id};
  *x_data.y.repeats = {g.s0, g.s1};
  *x_data.y.strides = {g.s1, One};

  Data idx_data("idx", g.graph);
  idx_data.ir_attr.SetIndex(1);
  idx_data.y.dtype = ge::DT_INT32;
  idx_data.attr.sched.axis = {g.z2.id, g.z3.id};
  *idx_data.y.axis = {g.z2.id, g.z3.id};
  *idx_data.y.repeats = {g.s2, One};
  *idx_data.y.strides = {One, af::ops::Zero};

  Load x_load("x_load");
  g.graph.AddNode(x_load);
  x_load.x = x_data.y;
  x_load.y.dtype = ge::DT_FLOAT16;
  x_load.attr.sched.axis = {g.z0.id, g.z1.id};
  *x_load.y.axis = {g.z0.id, g.z1.id};
  *x_load.y.repeats = {g.s0, g.s1};
  *x_load.y.strides = {g.s1, One};

  Load idx_load("idx_load");
  g.graph.AddNode(idx_load);
  idx_load.x = idx_data.y;
  idx_load.y.dtype = ge::DT_INT32;
  idx_load.attr.sched.axis = {g.z2.id, g.z3.id};
  *idx_load.y.axis = {g.z2.id, g.z3.id};
  *idx_load.y.repeats = {g.s2, One};
  *idx_load.y.strides = {One, af::ops::Zero};

  IndirectLoad il("indirect_load");
  g.graph.AddNode(il);
  il.x1 = x_load.y;
  il.x2 = idx_load.y;
  il.ir_attr.SetAxis(0);
  il.attr.sched.axis = {g.z2.id, g.z3.id};
  *il.y.axis = {g.z2.id, g.z3.id};
  *il.y.repeats = {g.s2, g.s3};
  *il.y.strides = {g.s3, One};

  Store store("store");
  g.graph.AddNode(store);
  store.x = il.y;
  store.y.dtype = ge::DT_FLOAT16;
  store.attr.sched.axis = {g.z2.id, g.z3.id};
  *store.y.axis = {g.z2.id, g.z3.id};
  *store.y.repeats = {g.s2, g.s3};
  *store.y.strides = {g.s3, One};

  Output y_out("y");
  g.graph.AddNode(y_out);
  y_out.ir_attr.SetIndex(0);
  y_out.x = store.y;
  y_out.y.dtype = ge::DT_FLOAT16;

  for (const char *name : {"x", "idx"}) {
    auto n = g.graph.FindNode(name);
    n->attr.api.compute_type = af::ComputeType::kComputeInvalid;
    n->attr.api.type = af::ApiType::kAPITypeBuffer;
    n->attr.api.unit = af::ComputeUnit::kUnitNone;
  }
  for (const char *name : {"x_load", "idx_load"}) {
    auto n = g.graph.FindNode(name);
    n->attr.api.compute_type = af::ComputeType::kComputeLoad;
    n->attr.api.type = af::ApiType::kAPITypeCompute;
    n->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  }
  auto il_node = g.graph.FindNode("indirect_load");
  il_node->attr.api.compute_type = af::ComputeType::kComputeLoad;
  il_node->attr.api.type = af::ApiType::kAPITypeCompute;
  il_node->attr.api.unit = af::ComputeUnit::kUnitVector;
  auto store_node = g.graph.FindNode("store");
  store_node->attr.api.compute_type = af::ComputeType::kComputeStore;
  store_node->attr.api.type = af::ApiType::kAPITypeCompute;
  store_node->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  auto y_node = g.graph.FindNode("y");
  y_node->attr.api.compute_type = af::ComputeType::kComputeInvalid;
  y_node->attr.api.type = af::ApiType::kAPITypeBuffer;
  y_node->attr.api.unit = af::ComputeUnit::kUnitNone;

  // 输出轴 merge/split（z2z3 命名为模板 outer 轴依赖）；idx_load/x_load 豁免：
  // embedding 分类的 payload 不变性按源视图原始轴号（z2/z3）证明，轴改写折叠后
  // find 失败会使分类坍塌为 kGeneric。
  auto z2_id = g.z2.id;
  auto z3_id = g.z3.id;
  auto z2z3 = g.graph.MergeAxis({z2_id, z3_id});
  auto [z2z3T, z2z3t] = g.graph.TileSplit(z2z3->id);
  auto [z2z3TB, z2z3Tb] = g.graph.BlockSplit(z2z3T->id);

  for (auto n : g.graph.GetAllNodes()) {
    if (IsOps<Data>(n) || IsOps<Output>(n) || n->GetName() == "x_load" || n->GetName() == "idx_load") continue;
    g.graph.ApplyMerge(n, z2z3->id);
    g.graph.ApplySplit(n, z2z3T->id, z2z3t->id);
    g.graph.ApplySplit(n, z2z3TB->id, z2z3Tb->id);
    g.graph.ApplyReorder(n, {z2z3TB->id, z2z3Tb->id, z2z3t->id});
  }

  il_node->attr.sched.loop_axis = z2z3Tb->id;
  il_node->outputs[0].attr.vectorized_axis = {z2z3t->id};
  il_node->outputs[0].attr.vectorized_strides = {One};
  store_node->attr.sched.loop_axis = z2z3Tb->id;
  store_node->outputs[0].attr.vectorized_axis = {z2z3t->id};
  store_node->outputs[0].attr.vectorized_strides = {One};

  auto x = g.graph.FindNode("x");
  x->outputs[0].attr.mem.tensor_id = 0;
  x->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  x->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareGM;
  x->outputs[0].attr.mem.position = af::Position::kPositionGM;
  x->outputs[0].attr.que.id = af::kIdNone;

  auto idx = g.graph.FindNode("idx");
  idx->outputs[0].attr.mem.tensor_id = 3;
  idx->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  idx->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareGM;
  idx->outputs[0].attr.mem.position = af::Position::kPositionGM;
  idx->outputs[0].attr.que.id = af::kIdNone;

  auto x_load_node = g.graph.FindNode("x_load");
  x_load_node->outputs[0].attr.mem.tensor_id = 4;
  x_load_node->outputs[0].attr.mem.reuse_id = 4;
  x_load_node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  x_load_node->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareUB;
  x_load_node->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  x_load_node->outputs[0].attr.que.id = 1;
  x_load_node->outputs[0].attr.que.depth = 2;
  x_load_node->outputs[0].attr.que.buf_num = 2;

  auto idx_load_node = g.graph.FindNode("idx_load");
  idx_load_node->outputs[0].attr.mem.tensor_id = 5;
  idx_load_node->outputs[0].attr.mem.reuse_id = 5;
  idx_load_node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  idx_load_node->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareUB;
  idx_load_node->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  idx_load_node->outputs[0].attr.que.id = 2;
  idx_load_node->outputs[0].attr.que.depth = 2;
  idx_load_node->outputs[0].attr.que.buf_num = 2;

  il_node->outputs[0].attr.dtype = ge::DT_FLOAT16;
  il_node->outputs[0].attr.mem.tensor_id = 1;
  il_node->outputs[0].attr.mem.reuse_id = 1;
  il_node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  il_node->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareUB;
  il_node->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  il_node->outputs[0].attr.que.id = 0;
  il_node->outputs[0].attr.que.depth = 2;
  il_node->outputs[0].attr.que.buf_num = 2;

  store_node->outputs[0].attr.mem.tensor_id = 2;
  store_node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  store_node->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareGM;
  store_node->outputs[0].attr.mem.position = af::Position::kPositionGM;
  store_node->outputs[0].attr.que.id = af::kIdNone;
}

// 按 embedding 形态 stamp 模板与逻辑视图（index 尾轴零步长由 idx_load 源视图经
// payload 不变性证明，逻辑 index 视图取 dense 全步长与布局校验兼容）。
void AnnotateStSimtEmbeddingTemplate(af::AscGraph &graph) {
  auto il = graph.FindNode("indirect_load");
  ASSERT_NE(il, nullptr);
  ASSERT_EQ(::ascir::SetTemplateId(il, ::ascir::TemplateId::kIndirectLoadSimt), af::SUCCESS);
  ASSERT_EQ(SetImplementation(il, Implementation::kDefault), af::SUCCESS);
  const auto input_load = graph.FindNode("x_load");
  const auto index_load = graph.FindNode("idx_load");
  const auto store = graph.FindNode("store");
  ASSERT_NE(input_load, nullptr);
  ASSERT_NE(index_load, nullptr);
  ASSERT_NE(store, nullptr);
  ASSERT_EQ(SetTemplateRole(input_load, TemplateRole::kSimtInputBoundary), af::SUCCESS);
  ASSERT_EQ(SetTemplateRole(index_load, TemplateRole::kSimtDirectGmBoundary), af::SUCCESS);
  ASSERT_EQ(SetTemplateRole(store, TemplateRole::kSimtDirectGmBoundary), af::SUCCESS);

  auto axes = graph.GetAllAxis();
  af::AxisId outer_id = af::kIdNone;
  for (const auto &axis : axes) {
    if (axis == nullptr) continue;
    if (axis->name == "z2z3") outer_id = axis->id;
  }
  ASSERT_NE(outer_id, af::kIdNone);
  TemplateAxes t{outer_id, af::kIdNone, af::kIdNone};
  ASSERT_EQ(SetTemplateAxes(il, t), af::SUCCESS);

  const af::Expression One = af::sym::kSymbolOne;
  TemplateLogicalView view;
  view.input.axis_ids = {axes[0]->id, axes[1]->id};
  view.input.sizes = {graph.FindAxis(axes[0]->id)->size, graph.FindAxis(axes[1]->id)->size};
  view.input.strides = {graph.FindAxis(axes[1]->id)->size, One};
  view.index.axis_ids = {axes[2]->id, axes[3]->id};
  view.index.sizes = {graph.FindAxis(axes[2]->id)->size, graph.FindAxis(axes[3]->id)->size};
  view.index.strides = {graph.FindAxis(axes[3]->id)->size, One};
  view.output.axis_ids = {axes[2]->id, axes[3]->id};
  view.output.sizes = {graph.FindAxis(axes[2]->id)->size, graph.FindAxis(axes[3]->id)->size};
  view.output.strides = {graph.FindAxis(axes[3]->id)->size, One};
  ASSERT_EQ(ClassifyIndirectLoadLayout(view.input, view.input), af::SUCCESS);
  ASSERT_EQ(ClassifyIndirectLoadLayout(view.index, view.index), af::SUCCESS);
  ASSERT_EQ(SetTemplateLogicalView(il, view), af::SUCCESS);
}

// ==================== kStrided 基线（axis=1 尾轴 gather，动态符号步长） ====================

void BuildStSimtStridedGraph(IndirectLoadSimtStGraph &g) {
  const af::Expression One = af::sym::kSymbolOne;

  Data x_data("x", g.graph);
  x_data.ir_attr.SetIndex(0);
  x_data.y.dtype = ge::DT_FLOAT16;
  x_data.attr.sched.axis = {g.z0.id, g.z1.id};
  *x_data.y.axis = {g.z0.id, g.z1.id};
  *x_data.y.repeats = {g.s0, g.s1};
  *x_data.y.strides = {g.s1, One};

  Data idx_data("idx", g.graph);
  idx_data.ir_attr.SetIndex(1);
  idx_data.y.dtype = ge::DT_INT32;
  idx_data.attr.sched.axis = {g.z2.id, g.z3.id};
  *idx_data.y.axis = {g.z2.id, g.z3.id};
  *idx_data.y.repeats = {g.s2, g.s3};
  *idx_data.y.strides = {g.s3, One};

  Load x_load("x_load");
  g.graph.AddNode(x_load);
  x_load.x = x_data.y;
  x_load.y.dtype = ge::DT_FLOAT16;
  x_load.attr.sched.axis = {g.z0.id, g.z1.id};
  *x_load.y.axis = {g.z0.id, g.z1.id};
  *x_load.y.repeats = {g.s0, g.s1};
  *x_load.y.strides = {g.s1, One};

  Load idx_load("idx_load");
  g.graph.AddNode(idx_load);
  idx_load.x = idx_data.y;
  idx_load.y.dtype = ge::DT_INT32;
  idx_load.attr.sched.axis = {g.z2.id, g.z3.id};
  *idx_load.y.axis = {g.z2.id, g.z3.id};
  *idx_load.y.repeats = {g.s2, g.s3};
  *idx_load.y.strides = {g.s3, One};

  IndirectLoad il("indirect_load");
  g.graph.AddNode(il);
  il.x1 = x_load.y;
  il.x2 = idx_load.y;
  il.ir_attr.SetAxis(1);
  il.attr.sched.axis = {g.z2.id, g.z3.id};
  *il.y.axis = {g.z2.id, g.z3.id};
  *il.y.repeats = {g.s2, g.s3};
  *il.y.strides = {g.s3, One};

  Store store("store");
  g.graph.AddNode(store);
  store.x = il.y;
  store.y.dtype = ge::DT_FLOAT16;
  store.attr.sched.axis = {g.z2.id, g.z3.id};
  *store.y.axis = {g.z2.id, g.z3.id};
  *store.y.repeats = {g.s2, g.s3};
  *store.y.strides = {g.s3, One};

  Output y_out("y");
  g.graph.AddNode(y_out);
  y_out.ir_attr.SetIndex(0);
  y_out.x = store.y;
  y_out.y.dtype = ge::DT_FLOAT16;

  for (const char *name : {"x", "idx"}) {
    auto n = g.graph.FindNode(name);
    n->attr.api.compute_type = af::ComputeType::kComputeInvalid;
    n->attr.api.type = af::ApiType::kAPITypeBuffer;
    n->attr.api.unit = af::ComputeUnit::kUnitNone;
  }
  for (const char *name : {"x_load", "idx_load"}) {
    auto n = g.graph.FindNode(name);
    n->attr.api.compute_type = af::ComputeType::kComputeLoad;
    n->attr.api.type = af::ApiType::kAPITypeCompute;
    n->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  }
  auto il_node = g.graph.FindNode("indirect_load");
  il_node->attr.api.compute_type = af::ComputeType::kComputeLoad;
  il_node->attr.api.type = af::ApiType::kAPITypeCompute;
  il_node->attr.api.unit = af::ComputeUnit::kUnitVector;
  auto store_node = g.graph.FindNode("store");
  store_node->attr.api.compute_type = af::ComputeType::kComputeStore;
  store_node->attr.api.type = af::ApiType::kAPITypeCompute;
  store_node->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  auto y_node = g.graph.FindNode("y");
  y_node->attr.api.compute_type = af::ComputeType::kComputeInvalid;
  y_node->attr.api.type = af::ApiType::kAPITypeBuffer;
  y_node->attr.api.unit = af::ComputeUnit::kUnitNone;

  auto z2_id = g.z2.id;
  auto z3_id = g.z3.id;
  auto z2z3 = g.graph.MergeAxis({z2_id, z3_id});
  auto [z2z3T, z2z3t] = g.graph.TileSplit(z2z3->id);
  auto [z2z3TB, z2z3Tb] = g.graph.BlockSplit(z2z3T->id);

  for (auto n : g.graph.GetAllNodes()) {
    if (IsOps<Data>(n) || IsOps<Output>(n) || n->GetName() == "x_load") continue;
    g.graph.ApplyMerge(n, z2z3->id);
    g.graph.ApplySplit(n, z2z3T->id, z2z3t->id);
    g.graph.ApplySplit(n, z2z3TB->id, z2z3Tb->id);
    g.graph.ApplyReorder(n, {z2z3TB->id, z2z3Tb->id, z2z3t->id});
  }

  il_node->attr.sched.loop_axis = z2z3Tb->id;
  il_node->outputs[0].attr.vectorized_axis = {z2z3t->id};
  il_node->outputs[0].attr.vectorized_strides = {One};
  store_node->attr.sched.loop_axis = z2z3Tb->id;
  store_node->outputs[0].attr.vectorized_axis = {z2z3t->id};
  store_node->outputs[0].attr.vectorized_strides = {One};

  auto x = g.graph.FindNode("x");
  x->outputs[0].attr.mem.tensor_id = 0;
  x->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  x->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareGM;
  x->outputs[0].attr.mem.position = af::Position::kPositionGM;
  x->outputs[0].attr.que.id = af::kIdNone;

  auto idx = g.graph.FindNode("idx");
  idx->outputs[0].attr.mem.tensor_id = 3;
  idx->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  idx->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareGM;
  idx->outputs[0].attr.mem.position = af::Position::kPositionGM;
  idx->outputs[0].attr.que.id = af::kIdNone;

  auto x_load_node = g.graph.FindNode("x_load");
  x_load_node->outputs[0].attr.mem.tensor_id = 4;
  x_load_node->outputs[0].attr.mem.reuse_id = 4;
  x_load_node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  x_load_node->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareUB;
  x_load_node->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  x_load_node->outputs[0].attr.que.id = 1;
  x_load_node->outputs[0].attr.que.depth = 2;
  x_load_node->outputs[0].attr.que.buf_num = 2;

  auto idx_load_node = g.graph.FindNode("idx_load");
  idx_load_node->outputs[0].attr.mem.tensor_id = 5;
  idx_load_node->outputs[0].attr.mem.reuse_id = 5;
  idx_load_node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  idx_load_node->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareUB;
  idx_load_node->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  idx_load_node->outputs[0].attr.que.id = 2;
  idx_load_node->outputs[0].attr.que.depth = 2;
  idx_load_node->outputs[0].attr.que.buf_num = 2;

  il_node->outputs[0].attr.dtype = ge::DT_FLOAT16;
  il_node->outputs[0].attr.mem.tensor_id = 1;
  il_node->outputs[0].attr.mem.reuse_id = 1;
  il_node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  il_node->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareUB;
  il_node->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  il_node->outputs[0].attr.que.id = 0;
  il_node->outputs[0].attr.que.depth = 2;
  il_node->outputs[0].attr.que.buf_num = 2;

  store_node->outputs[0].attr.mem.tensor_id = 2;
  store_node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  store_node->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareGM;
  store_node->outputs[0].attr.mem.position = af::Position::kPositionGM;
  store_node->outputs[0].attr.que.id = af::kIdNone;
}

// 经典 kStrided 标注（dense 全步长逻辑视图）。
void AnnotateStSimtStridedTemplate(af::AscGraph &graph) {
  auto il = graph.FindNode("indirect_load");
  ASSERT_NE(il, nullptr);
  ASSERT_EQ(::ascir::SetTemplateId(il, ::ascir::TemplateId::kIndirectLoadSimt), af::SUCCESS);
  ASSERT_EQ(SetImplementation(il, Implementation::kDefault), af::SUCCESS);
  const auto input_load = graph.FindNode("x_load");
  const auto index_load = graph.FindNode("idx_load");
  const auto store = graph.FindNode("store");
  ASSERT_NE(input_load, nullptr);
  ASSERT_NE(index_load, nullptr);
  ASSERT_NE(store, nullptr);
  ASSERT_EQ(SetTemplateRole(input_load, TemplateRole::kSimtInputBoundary), af::SUCCESS);
  ASSERT_EQ(SetTemplateRole(index_load, TemplateRole::kSimtDirectGmBoundary), af::SUCCESS);
  ASSERT_EQ(SetTemplateRole(store, TemplateRole::kSimtDirectGmBoundary), af::SUCCESS);

  auto axes = graph.GetAllAxis();
  af::AxisId outer_id = af::kIdNone;
  for (const auto &axis : axes) {
    if (axis == nullptr) continue;
    if (axis->name == "z2z3") outer_id = axis->id;
  }
  ASSERT_NE(outer_id, af::kIdNone);
  TemplateAxes t{outer_id, af::kIdNone, af::kIdNone};
  ASSERT_EQ(SetTemplateAxes(il, t), af::SUCCESS);

  const af::Expression One = af::sym::kSymbolOne;
  TemplateLogicalView view;
  view.input.axis_ids = {axes[0]->id, axes[1]->id};
  view.input.sizes = {graph.FindAxis(axes[0]->id)->size, graph.FindAxis(axes[1]->id)->size};
  view.input.strides = {graph.FindAxis(axes[1]->id)->size, One};
  view.index.axis_ids = {axes[2]->id, axes[3]->id};
  view.index.sizes = {graph.FindAxis(axes[2]->id)->size, graph.FindAxis(axes[3]->id)->size};
  view.index.strides = {graph.FindAxis(axes[3]->id)->size, One};
  view.output.axis_ids = {axes[2]->id, axes[3]->id};
  view.output.sizes = {graph.FindAxis(axes[2]->id)->size, graph.FindAxis(axes[3]->id)->size};
  view.output.strides = {graph.FindAxis(axes[3]->id)->size, One};
  ASSERT_EQ(ClassifyIndirectLoadLayout(view.input, view.input), af::SUCCESS);
  ASSERT_EQ(ClassifyIndirectLoadLayout(view.index, view.index), af::SUCCESS);
  ASSERT_EQ(SetTemplateLogicalView(il, view), af::SUCCESS);
}

// ==================== fused 组：两成员 embedding + 共享归并/Store ====================

void BuildStSimtFusedEmbeddingGraph(IndirectLoadSimtStGraph &g) {
  const af::Expression One = af::sym::kSymbolOne;

  for (size_t m = 0UL; m < 2UL; ++m) {
    const std::string suffix = std::to_string(m);
    Data x_data(("x" + suffix).c_str(), g.graph);
    x_data.ir_attr.SetIndex(static_cast<int32_t>(m));
    x_data.y.dtype = ge::DT_FLOAT16;
    x_data.attr.sched.axis = {g.z0.id, g.z1.id};
    *x_data.y.axis = {g.z0.id, g.z1.id};
    *x_data.y.repeats = {g.s0, g.s1};
    *x_data.y.strides = {g.s1, One};

    Data idx_data(("idx" + suffix).c_str(), g.graph);
    idx_data.ir_attr.SetIndex(static_cast<int32_t>(2 + m));
    idx_data.y.dtype = ge::DT_INT32;
    idx_data.attr.sched.axis = {g.z2.id, g.z3.id};
    *idx_data.y.axis = {g.z2.id, g.z3.id};
    *idx_data.y.repeats = {g.s2, One};
    *idx_data.y.strides = {One, af::ops::Zero};

    Load x_load(("x_load" + suffix).c_str());
    g.graph.AddNode(x_load);
    x_load.x = x_data.y;
    x_load.y.dtype = ge::DT_FLOAT16;
    x_load.attr.sched.axis = {g.z0.id, g.z1.id};
    *x_load.y.axis = {g.z0.id, g.z1.id};
    *x_load.y.repeats = {g.s0, g.s1};
    *x_load.y.strides = {g.s1, One};

    Load idx_load(("idx_load" + suffix).c_str());
    g.graph.AddNode(idx_load);
    idx_load.x = idx_data.y;
    idx_load.y.dtype = ge::DT_INT32;
    idx_load.attr.sched.axis = {g.z2.id, g.z3.id};
    *idx_load.y.axis = {g.z2.id, g.z3.id};
    *idx_load.y.repeats = {g.s2, One};
    *idx_load.y.strides = {One, af::ops::Zero};

    IndirectLoad il(("indirect_load" + suffix).c_str());
    g.graph.AddNode(il);
    il.x1 = x_load.y;
    il.x2 = idx_load.y;
    il.ir_attr.SetAxis(0);
    il.attr.sched.axis = {g.z2.id, g.z3.id};
    *il.y.axis = {g.z2.id, g.z3.id};
    *il.y.repeats = {g.s2, g.s3};
    *il.y.strides = {g.s3, One};
  }

  Add merge_add("merge_add");
  g.graph.AddNode(merge_add);
  // 跨节点接线用 OpDescUtils/CreateOperatorFromNode + AscOpOutput（fused 分支既有模式）。
  {
    af::Operator il0_op = af::OpDescUtils::CreateOperatorFromNode(g.graph.FindNode("indirect_load0"));
    merge_add.x1 = af::AscOpOutput(&il0_op, 0U);
  }
  {
    af::Operator il1_op = af::OpDescUtils::CreateOperatorFromNode(g.graph.FindNode("indirect_load1"));
    merge_add.x2 = af::AscOpOutput(&il1_op, 0U);
  }
  merge_add.y.dtype = ge::DT_FLOAT16;
  merge_add.attr.sched.axis = {g.z2.id, g.z3.id};
  *merge_add.y.axis = {g.z2.id, g.z3.id};
  *merge_add.y.repeats = {g.s2, g.s3};
  *merge_add.y.strides = {g.s3, One};

  Store store("store");
  g.graph.AddNode(store);
  store.x = merge_add.y;
  store.y.dtype = ge::DT_FLOAT16;
  store.attr.sched.axis = {g.z2.id, g.z3.id};
  *store.y.axis = {g.z2.id, g.z3.id};
  *store.y.repeats = {g.s2, g.s3};
  *store.y.strides = {g.s3, One};

  Output y_out("y");
  g.graph.AddNode(y_out);
  y_out.ir_attr.SetIndex(0);
  y_out.x = store.y;
  y_out.y.dtype = ge::DT_FLOAT16;

  for (const char *name : {"x0", "x1", "idx0", "idx1", "y"}) {
    auto n = g.graph.FindNode(name);
    n->attr.api.compute_type = af::ComputeType::kComputeInvalid;
    n->attr.api.type = af::ApiType::kAPITypeBuffer;
    n->attr.api.unit = af::ComputeUnit::kUnitNone;
  }
  for (const char *name : {"x_load0", "x_load1", "idx_load0", "idx_load1"}) {
    auto n = g.graph.FindNode(name);
    n->attr.api.compute_type = af::ComputeType::kComputeLoad;
    n->attr.api.type = af::ApiType::kAPITypeCompute;
    n->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  }
  for (const char *name : {"indirect_load0", "indirect_load1"}) {
    auto n = g.graph.FindNode(name);
    n->attr.api.compute_type = af::ComputeType::kComputeLoad;
    n->attr.api.type = af::ApiType::kAPITypeCompute;
    n->attr.api.unit = af::ComputeUnit::kUnitVector;
  }
  auto merge_add_node = g.graph.FindNode("merge_add");
  merge_add_node->attr.api.compute_type = af::ComputeType::kComputeElewise;
  merge_add_node->attr.api.type = af::ApiType::kAPITypeCompute;
  merge_add_node->attr.api.unit = af::ComputeUnit::kUnitVector;
  auto store_node = g.graph.FindNode("store");
  store_node->attr.api.compute_type = af::ComputeType::kComputeStore;
  store_node->attr.api.type = af::ApiType::kAPITypeCompute;
  store_node->attr.api.unit = af::ComputeUnit::kUnitMTE2;

  auto z2_id = g.z2.id;
  auto z3_id = g.z3.id;
  auto z2z3 = g.graph.MergeAxis({z2_id, z3_id});
  auto [z2z3T, z2z3t] = g.graph.TileSplit(z2z3->id);
  auto [z2z3TB, z2z3Tb] = g.graph.BlockSplit(z2z3T->id);

  for (auto n : g.graph.GetAllNodes()) {
    if (IsOps<Data>(n) || IsOps<Output>(n) || n->GetName() == "x_load0" || n->GetName() == "x_load1" ||
        n->GetName() == "idx_load0" || n->GetName() == "idx_load1") {
      continue;
    }
    g.graph.ApplyMerge(n, z2z3->id);
    g.graph.ApplySplit(n, z2z3T->id, z2z3t->id);
    g.graph.ApplySplit(n, z2z3TB->id, z2z3Tb->id);
    g.graph.ApplyReorder(n, {z2z3TB->id, z2z3Tb->id, z2z3t->id});
  }

  for (const char *name : {"indirect_load0", "indirect_load1", "merge_add", "store"}) {
    auto n = g.graph.FindNode(name);
    n->attr.sched.loop_axis = z2z3Tb->id;
    n->outputs[0].attr.vectorized_axis = {z2z3t->id};
    n->outputs[0].attr.vectorized_strides = {One};
  }

  auto set_gm = [](const af::AscNodePtr &node, ascir::TensorId tensor_id) {
    node->outputs[0].attr.mem.tensor_id = tensor_id;
    node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
    node->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareGM;
    node->outputs[0].attr.mem.position = af::Position::kPositionGM;
    node->outputs[0].attr.que.id = af::kIdNone;
  };
  auto set_ub_que = [](const af::AscNodePtr &node, ascir::TensorId tensor_id, ascir::TensorId reuse_id,
                       af::Position position, int64_t que_id) {
    node->outputs[0].attr.mem.tensor_id = tensor_id;
    node->outputs[0].attr.mem.reuse_id = reuse_id;
    node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
    node->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareUB;
    node->outputs[0].attr.mem.position = position;
    node->outputs[0].attr.que.id = que_id;
    node->outputs[0].attr.que.depth = 2;
    node->outputs[0].attr.que.buf_num = 2;
  };
  set_gm(g.graph.FindNode("x0"), 0);
  set_gm(g.graph.FindNode("x1"), 6);
  set_gm(g.graph.FindNode("idx0"), 3);
  set_gm(g.graph.FindNode("idx1"), 7);
  set_gm(g.graph.FindNode("store"), 2);
  set_ub_que(g.graph.FindNode("x_load0"), 4, 4, af::Position::kPositionVecIn, 1);
  set_ub_que(g.graph.FindNode("idx_load0"), 5, 5, af::Position::kPositionVecIn, 2);
  set_ub_que(g.graph.FindNode("x_load1"), 9, 9, af::Position::kPositionVecIn, 4);
  set_ub_que(g.graph.FindNode("idx_load1"), 10, 10, af::Position::kPositionVecIn, 5);
  set_ub_que(g.graph.FindNode("indirect_load0"), 1, 1, af::Position::kPositionVecOut, 0);
  set_ub_que(g.graph.FindNode("indirect_load1"), 8, 8, af::Position::kPositionVecOut, 3);
  set_ub_que(g.graph.FindNode("merge_add"), 11, 11, af::Position::kPositionVecOut, 6);
  g.graph.FindNode("indirect_load0")->outputs[0].attr.dtype = ge::DT_FLOAT16;
  g.graph.FindNode("indirect_load1")->outputs[0].attr.dtype = ge::DT_FLOAT16;
}

// embedding 形态的 fused 组标注：per-member 逻辑视图、L2 角色与 merge_info
// （register_merge + 单 store），镜像调度侧 PlanSimtFusedEmission 的产物。
void AnnotateStSimtFusedEmbeddingTemplate(IndirectLoadSimtStGraph &g) {
  af::AxisId outer_id = af::kIdNone;
  for (const auto &axis : g.graph.GetAllAxis()) {
    if (axis != nullptr && axis->name == "z2z3") outer_id = axis->id;
  }
  ASSERT_NE(outer_id, af::kIdNone);
  const auto axes = g.graph.GetAllAxis();
  const af::Expression One = af::sym::kSymbolOne;
  TemplateLogicalView view;
  view.input.axis_ids = {axes[0]->id, axes[1]->id};
  view.input.sizes = {g.graph.FindAxis(axes[0]->id)->size, g.graph.FindAxis(axes[1]->id)->size};
  view.input.strides = {g.graph.FindAxis(axes[1]->id)->size, One};
  view.index.axis_ids = {axes[2]->id, axes[3]->id};
  view.index.sizes = {g.graph.FindAxis(axes[2]->id)->size, g.graph.FindAxis(axes[3]->id)->size};
  view.index.strides = {g.graph.FindAxis(axes[3]->id)->size, One};
  view.output.axis_ids = {axes[2]->id, axes[3]->id};
  view.output.sizes = {g.graph.FindAxis(axes[2]->id)->size, g.graph.FindAxis(axes[3]->id)->size};
  view.output.strides = {g.graph.FindAxis(axes[3]->id)->size, One};

  for (const char *name : {"indirect_load0", "indirect_load1"}) {
    auto il = g.graph.FindNode(name);
    ASSERT_NE(il, nullptr);
    ASSERT_EQ(::ascir::SetTemplateId(il, ::ascir::TemplateId::kIndirectLoadSimt), af::SUCCESS);
    ASSERT_EQ(SetImplementation(il, Implementation::kDefault), af::SUCCESS);
    TemplateAxes t{outer_id, af::kIdNone, af::kIdNone};
    ASSERT_EQ(SetTemplateAxes(il, t), af::SUCCESS);
    ASSERT_EQ(ClassifyIndirectLoadLayout(view.input, view.input), af::SUCCESS);
    ASSERT_EQ(ClassifyIndirectLoadLayout(view.index, view.index), af::SUCCESS);
    ASSERT_EQ(SetTemplateLogicalView(il, view), af::SUCCESS);
    ASSERT_EQ(SetTemplateRole(g.graph.FindNode(std::string(name).replace(0, 13, "x_load").c_str()),
                              TemplateRole::kSimtInputBoundary),
              af::SUCCESS);
    ASSERT_EQ(SetTemplateRole(g.graph.FindNode(std::string(name).replace(0, 13, "idx_load").c_str()),
                              TemplateRole::kSimtDirectGmBoundary),
              af::SUCCESS);
    ASSERT_EQ(SetSimtGroupMember(il), af::SUCCESS);
    bool metadata_supported = false;
    ASSERT_EQ(FinalizeLoweringMetadata(il, metadata_supported), af::SUCCESS);
    ASSERT_TRUE(metadata_supported);
  }
  ASSERT_EQ(SetTemplateRole(g.graph.FindNode("merge_add"), TemplateRole::kSimtInlineTransform), af::SUCCESS);
  ASSERT_EQ(SetTemplateRole(g.graph.FindNode("store"), TemplateRole::kSimtDirectGmBoundary), af::SUCCESS);
  auto primary = g.graph.FindNode("indirect_load0");
  ASSERT_EQ(SetSimtFusedPrimary(primary, {"indirect_load0", "indirect_load1"}), af::SUCCESS);
  SimtFusedMergeInfo merge_info;
  merge_info.register_merge = true;
  merge_info.merge_node_names = {"merge_add"};
  merge_info.merge_store_names = {"store"};
  ASSERT_EQ(SetSimtFusedMergeInfo(primary, merge_info), af::SUCCESS);
}

// ==================== post-Reduce：IL(axis=0) → Add(广播 addend) → Sum → store ====================

void BuildStSimtEmbeddingPostReduceGraph(IndirectLoadSimtStGraph &g) {
  const af::Expression One = af::sym::kSymbolOne;
  const af::Expression Zero = af::ops::Zero;

  Data x_data("x", g.graph);
  x_data.ir_attr.SetIndex(0);
  x_data.y.dtype = ge::DT_FLOAT16;
  x_data.attr.sched.axis = {g.z0.id, g.z1.id};
  *x_data.y.axis = {g.z0.id, g.z1.id};
  *x_data.y.repeats = {g.s0, g.s1};
  *x_data.y.strides = {g.s1, One};

  Data idx_data("idx", g.graph);
  idx_data.ir_attr.SetIndex(1);
  idx_data.y.dtype = ge::DT_INT32;
  idx_data.attr.sched.axis = {g.z2.id, g.z3.id};
  *idx_data.y.axis = {g.z2.id, g.z3.id};
  *idx_data.y.repeats = {g.s2, One};
  *idx_data.y.strides = {One, Zero};

  Load x_load("x_load");
  g.graph.AddNode(x_load);
  x_load.x = x_data.y;
  x_load.y.dtype = ge::DT_FLOAT16;
  x_load.attr.sched.axis = {g.z0.id, g.z1.id};
  *x_load.y.axis = {g.z0.id, g.z1.id};
  *x_load.y.repeats = {g.s0, g.s1};
  *x_load.y.strides = {g.s1, One};

  Load idx_load("idx_load");
  g.graph.AddNode(idx_load);
  idx_load.x = idx_data.y;
  idx_load.y.dtype = ge::DT_INT32;
  idx_load.attr.sched.axis = {g.z2.id, g.z3.id};
  *idx_load.y.axis = {g.z2.id, g.z3.id};
  *idx_load.y.repeats = {g.s2, One};
  *idx_load.y.strides = {One, Zero};

  Data addend_data("addend", g.graph);
  addend_data.ir_attr.SetIndex(2);
  addend_data.y.dtype = ge::DT_FLOAT16;
  addend_data.attr.sched.axis = {g.z2.id, g.z3.id};
  *addend_data.y.axis = {g.z2.id, g.z3.id};
  *addend_data.y.repeats = {g.s2, g.s3};
  *addend_data.y.strides = {g.s3, One};

  Load addend_load("addend_load");
  g.graph.AddNode(addend_load);
  addend_load.x = addend_data.y;
  addend_load.y.dtype = ge::DT_FLOAT16;
  addend_load.attr.sched.axis = {g.z2.id, g.z3.id};
  *addend_load.y.axis = {g.z2.id, g.z3.id};
  *addend_load.y.repeats = {g.s2, g.s3};
  *addend_load.y.strides = {g.s3, One};

  IndirectLoad il("indirect_load");
  g.graph.AddNode(il);
  il.x1 = x_load.y;
  il.x2 = idx_load.y;
  il.ir_attr.SetAxis(0);
  il.attr.sched.axis = {g.z2.id, g.z3.id};
  *il.y.axis = {g.z2.id, g.z3.id};
  *il.y.repeats = {g.s2, g.s3};
  *il.y.strides = {g.s3, One};

  // Add 双输入同算术 dtype（广播 addend fp16）：Add 的 SIMT 标量可发判据要求
  // HasSameSimtBinaryArithmeticDtype，接 INT32 索引张量会在求值器空跑期失败。
  Add output_transform("output_transform");
  g.graph.AddNode(output_transform);
  output_transform.x1 = il.y;
  output_transform.x2 = addend_load.y;
  output_transform.y.dtype = ge::DT_FLOAT16;
  output_transform.attr.sched.axis = {g.z2.id, g.z3.id};
  *output_transform.y.axis = {g.z2.id, g.z3.id};
  *output_transform.y.repeats = {g.s2, g.s3};
  *output_transform.y.strides = {g.s3, One};

  Sum reduce("reduce");
  g.graph.AddNode(reduce);
  reduce.x = output_transform.y;
  reduce.attr.api.compute_type = af::ComputeType::kComputeReduce;
  reduce.attr.sched.axis = {g.z2.id, g.z3.id};
  *reduce.y.axis = {g.z2.id, g.z3.id};
  *reduce.y.repeats = {One, g.s3};
  *reduce.y.strides = {Zero, One};

  Store store("store");
  g.graph.AddNode(store);
  store.x = reduce.y;
  store.y.dtype = ge::DT_FLOAT16;
  store.attr.sched.axis = {g.z2.id, g.z3.id};
  *store.y.axis = {g.z2.id, g.z3.id};
  *store.y.repeats = {One, g.s3};
  *store.y.strides = {Zero, One};

  Output y_out("y");
  g.graph.AddNode(y_out);
  y_out.ir_attr.SetIndex(0);
  y_out.x = store.y;
  y_out.y.dtype = ge::DT_FLOAT16;

  for (const char *name : {"x", "idx", "addend"}) {
    auto n = g.graph.FindNode(name);
    n->attr.api.compute_type = af::ComputeType::kComputeInvalid;
    n->attr.api.type = af::ApiType::kAPITypeBuffer;
    n->attr.api.unit = af::ComputeUnit::kUnitNone;
  }
  for (const char *name : {"x_load", "idx_load", "addend_load"}) {
    auto n = g.graph.FindNode(name);
    n->attr.api.compute_type = af::ComputeType::kComputeLoad;
    n->attr.api.type = af::ApiType::kAPITypeCompute;
    n->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  }
  auto il_node = g.graph.FindNode("indirect_load");
  il_node->attr.api.compute_type = af::ComputeType::kComputeLoad;
  il_node->attr.api.type = af::ApiType::kAPITypeCompute;
  il_node->attr.api.unit = af::ComputeUnit::kUnitVector;
  auto transform_node = g.graph.FindNode("output_transform");
  transform_node->attr.api.compute_type = af::ComputeType::kComputeElewise;
  transform_node->attr.api.type = af::ApiType::kAPITypeCompute;
  transform_node->attr.api.unit = af::ComputeUnit::kUnitVector;
  auto reduce_node = g.graph.FindNode("reduce");
  reduce_node->attr.api.compute_type = af::ComputeType::kComputeReduce;
  reduce_node->attr.api.type = af::ApiType::kAPITypeCompute;
  reduce_node->attr.api.unit = af::ComputeUnit::kUnitVector;
  auto store_node = g.graph.FindNode("store");
  store_node->attr.api.compute_type = af::ComputeType::kComputeStore;
  store_node->attr.api.type = af::ApiType::kAPITypeCompute;
  store_node->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  auto y_node = g.graph.FindNode("y");
  y_node->attr.api.compute_type = af::ComputeType::kComputeInvalid;
  y_node->attr.api.type = af::ApiType::kAPITypeBuffer;
  y_node->attr.api.unit = af::ComputeUnit::kUnitNone;

  auto z2_id = g.z2.id;
  auto z3_id = g.z3.id;
  auto z2z3 = g.graph.MergeAxis({z2_id, z3_id});
  auto [z2z3T, z2z3t] = g.graph.TileSplit(z2z3->id);
  auto [z2z3TB, z2z3Tb] = g.graph.BlockSplit(z2z3T->id);

  for (auto n : g.graph.GetAllNodes()) {
    if (IsOps<Data>(n) || IsOps<Output>(n) || n->GetName() == "x_load" || n->GetName() == "idx_load") continue;
    g.graph.ApplyMerge(n, z2z3->id);
    g.graph.ApplySplit(n, z2z3T->id, z2z3t->id);
    g.graph.ApplySplit(n, z2z3TB->id, z2z3Tb->id);
    g.graph.ApplyReorder(n, {z2z3TB->id, z2z3Tb->id, z2z3t->id});
  }
  // GM 侧载（addend_load）在 metadata 采集后恢复逻辑 (z2, z3) 视图（与既有
  // BuildSimtPostReduceGraph 同法）。
  auto addend_node = g.graph.FindNode("addend_load");
  addend_node->outputs[0].attr.axis = {z2_id, z3_id};
  addend_node->outputs[0].attr.repeats = {g.s2, g.s3};
  addend_node->outputs[0].attr.strides = {g.s3, One};

  for (const char *name : {"indirect_load", "output_transform", "reduce", "store"}) {
    auto n = g.graph.FindNode(name);
    n->attr.sched.loop_axis = z2z3Tb->id;
    n->outputs[0].attr.vectorized_axis = {z2z3t->id};
    n->outputs[0].attr.vectorized_strides = {One};
  }

  auto set_gm = [](const af::AscNodePtr &node, ascir::TensorId tensor_id) {
    node->outputs[0].attr.mem.tensor_id = tensor_id;
    node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
    node->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareGM;
    node->outputs[0].attr.mem.position = af::Position::kPositionGM;
    node->outputs[0].attr.que.id = af::kIdNone;
  };
  auto set_ub_que = [](const af::AscNodePtr &node, ascir::TensorId tensor_id, ascir::TensorId reuse_id,
                       af::Position position, int64_t que_id) {
    node->outputs[0].attr.mem.tensor_id = tensor_id;
    node->outputs[0].attr.mem.reuse_id = reuse_id;
    node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
    node->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareUB;
    node->outputs[0].attr.mem.position = position;
    node->outputs[0].attr.que.id = que_id;
    node->outputs[0].attr.que.depth = 2;
    node->outputs[0].attr.que.buf_num = 2;
  };
  set_gm(g.graph.FindNode("x"), 0);
  set_gm(g.graph.FindNode("idx"), 3);
  set_gm(g.graph.FindNode("addend"), 8);
  set_gm(g.graph.FindNode("store"), 2);
  set_ub_que(g.graph.FindNode("x_load"), 4, 4, af::Position::kPositionVecIn, 1);
  set_ub_que(g.graph.FindNode("idx_load"), 5, 5, af::Position::kPositionVecIn, 2);
  set_ub_que(g.graph.FindNode("addend_load"), 9, 9, af::Position::kPositionVecIn, 5);
  il_node->outputs[0].attr.dtype = ge::DT_FLOAT16;
  il_node->outputs[0].attr.mem.tensor_id = 1;
  il_node->outputs[0].attr.mem.reuse_id = 1;
  il_node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  il_node->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareUB;
  il_node->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  il_node->outputs[0].attr.que.id = 0;
  il_node->outputs[0].attr.que.depth = 2;
  il_node->outputs[0].attr.que.buf_num = 2;
  // post Reduce 候选以 output_transform（reduce 输入生产者）输出 tensor_id 为
  // output_result_tensor_id，漏设即 Init 期 "tensor IDs remain unassigned"。
  transform_node->outputs[0].attr.dtype = ge::DT_FLOAT16;
  set_ub_que(transform_node, 6, 6, af::Position::kPositionVecOut, 3);
  reduce_node->outputs[0].attr.dtype = ge::DT_FLOAT16;
  set_ub_que(reduce_node, 7, 7, af::Position::kPositionVecOut, 4);
}

void AnnotateStSimtEmbeddingPostReduceTemplate(af::AscGraph &graph) {
  auto il = graph.FindNode("indirect_load");
  ASSERT_NE(il, nullptr);
  ASSERT_EQ(::ascir::SetTemplateId(il, ::ascir::TemplateId::kIndirectLoadSimt), af::SUCCESS);
  ASSERT_EQ(SetImplementation(il, Implementation::kDefault), af::SUCCESS);
  ASSERT_EQ(SetTemplateRole(graph.FindNode("x_load"), TemplateRole::kSimtInputBoundary), af::SUCCESS);
  ASSERT_EQ(SetTemplateRole(graph.FindNode("idx_load"), TemplateRole::kSimtDirectGmBoundary), af::SUCCESS);
  ASSERT_EQ(SetTemplateRole(graph.FindNode("addend_load"), TemplateRole::kSimtDirectGmBoundary), af::SUCCESS);
  ASSERT_EQ(SetTemplateRole(graph.FindNode("output_transform"), TemplateRole::kSimtInlineTransform), af::SUCCESS);
  ASSERT_EQ(SetTemplateRole(graph.FindNode("store"), TemplateRole::kSimtDirectGmBoundary), af::SUCCESS);

  auto axes = graph.GetAllAxis();
  af::AxisId outer_id = af::kIdNone;
  for (const auto &axis : axes) {
    if (axis == nullptr) continue;
    if (axis->name == "z2z3") outer_id = axis->id;
  }
  ASSERT_NE(outer_id, af::kIdNone);
  TemplateAxes t{outer_id, af::kIdNone, af::kIdNone};
  ASSERT_EQ(SetTemplateAxes(il, t), af::SUCCESS);

  const af::Expression One = af::sym::kSymbolOne;
  TemplateLogicalView view;
  view.input.axis_ids = {axes[0]->id, axes[1]->id};
  view.input.sizes = {graph.FindAxis(axes[0]->id)->size, graph.FindAxis(axes[1]->id)->size};
  view.input.strides = {graph.FindAxis(axes[1]->id)->size, One};
  view.index.axis_ids = {axes[2]->id, axes[3]->id};
  view.index.sizes = {graph.FindAxis(axes[2]->id)->size, graph.FindAxis(axes[3]->id)->size};
  view.index.strides = {graph.FindAxis(axes[3]->id)->size, One};
  view.output.axis_ids = {axes[2]->id, axes[3]->id};
  view.output.sizes = {graph.FindAxis(axes[2]->id)->size, graph.FindAxis(axes[3]->id)->size};
  view.output.strides = {graph.FindAxis(axes[3]->id)->size, One};
  ASSERT_EQ(ClassifyIndirectLoadLayout(view.input, view.input), af::SUCCESS);
  ASSERT_EQ(ClassifyIndirectLoadLayout(view.index, view.index), af::SUCCESS);
  ASSERT_EQ(SetTemplateLogicalView(il, view), af::SUCCESS);
}

}  // namespace

// [ST 覆盖·单 gather 动态 embedding] 定义与调用两侧双发：定义侧两套 body 结构体、
// 调用侧 use_u32 运行期 guard 与 u32 实例——钉住双发门控不被回退（覆盖分类/
// 双发判定/ResolveSimtMetadata/guard collect-join/invocation 全链路）。
TEST(CodegenIndirectLoadSimtSt, EmbeddingDynamicDualEmitsDefinitionAndInvocation) {
  IndirectLoadSimtStContext context;
  IndirectLoadSimtStGraph g("st_simt_embedding_dynamic");
  BuildStSimtEmbeddingGraph(g);
  AnnotateStSimtEmbeddingTemplate(g.graph);

  Tiler tiler;
  SetupIndirectLoadStTiler(g.graph, tiler);
  TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(g.graph);
  auto il_node = g.graph.FindNode("indirect_load");
  ASSERT_NE(il_node, nullptr);
  IndirectLoadRegApiCall call("IndirectLoadRegApiCall");
  ASSERT_EQ(call.Init(il_node), af::SUCCESS);
  ApiTensor x1, x2;
  x1.id = il_node->inputs[0].attr.mem.tensor_id;
  x2.id = il_node->inputs[1].attr.mem.tensor_id;
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);
  tpipe.AddTensor(il_node->inputs[0]);
  tpipe.AddTensor(il_node->inputs[1]);

  std::stringstream ss;
  ASSERT_EQ(call.GenerateFuncDefinition(tpipe, tiler, ss), af::SUCCESS);
  const std::string definition = ss.str();
  EXPECT_NE(definition.find("struct IndirectLoadSimtBody_indirect_load_u64"), std::string::npos)
      << "the definition must carry the uint64 body variant.";
  EXPECT_NE(definition.find("struct IndirectLoadSimtBody_indirect_load_u32"), std::string::npos)
      << "dynamic embedding must dual emit the uint32 body variant.";

  const auto outer_tb = FindStOuterTbAxis(g.graph);
  ASSERT_NE(outer_tb, af::kIdNone);
  std::string result;
  ASSERT_EQ(call.Generate(tpipe, {outer_tb}, result), af::SUCCESS);
  EXPECT_NE(result.find("indirect_load_simt_use_u32"), std::string::npos)
      << "the invocation must pick the offset width at runtime.";
  EXPECT_NE(result.find("IndirectLoadSimtBody_indirect_load_u32"), std::string::npos)
      << "the uint32 instance must be instantiated at the call site.";
  EXPECT_NE(result.find("<= 0x7FFFFFFFULL"), std::string::npos)
      << "the runtime guard must bound every view by INT32_MAX.";
}

// [ST 覆盖·fused 组] 两成员 embedding + register_merge/单 store：覆盖 GenerateSimt
// 的 fused 发射分支（FusedCase/Ctx/Body 成套双发、fused guard collect-join、
// 双套调用实例），钉住 fused 双发与逐成员视图上界 guard。
TEST(CodegenIndirectLoadSimtSt, FusedEmbeddingGroupDualEmits) {
  IndirectLoadSimtStContext context;
  IndirectLoadSimtStGraph g("st_simt_fused_embedding_dynamic");
  BuildStSimtFusedEmbeddingGraph(g);
  AnnotateStSimtFusedEmbeddingTemplate(g);

  auto il0 = g.graph.FindNode("indirect_load0");
  ASSERT_NE(il0, nullptr);
  Tiler tiler;
  SetupIndirectLoadStTiler(g.graph, tiler);
  TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(g.graph);

  IndirectLoadRegApiCall call("IndirectLoadRegApiCall");
  ASSERT_EQ(call.Init(il0), af::SUCCESS);
  ApiTensor x1, x2;
  x1.id = il0->inputs[0].attr.mem.tensor_id;
  x2.id = il0->inputs[1].attr.mem.tensor_id;
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);
  for (const char *name : {"x_load0", "idx_load0", "x_load1", "idx_load1", "indirect_load0", "indirect_load1"}) {
    const auto node = g.graph.FindNode(name);
    ASSERT_NE(node, nullptr);
    tpipe.AddTensor(*node->outputs()[0]);
  }

  const auto outer_tb = FindStOuterTbAxis(g.graph);
  ASSERT_NE(outer_tb, af::kIdNone);
  std::string generated;
  ASSERT_EQ(call.Generate(tpipe, {outer_tb}, generated), af::SUCCESS);

  const std::string decl = "indirect_load_simt_fused_use_u32 = ";
  size_t guard_count = 0UL;
  for (size_t pos = generated.find(decl); pos != std::string::npos; pos = generated.find(decl, pos + decl.size())) {
    ++guard_count;
  }
  EXPECT_EQ(guard_count, 1UL) << "the fused guard declaration must be emitted exactly once.";
  EXPECT_NE(generated.find("struct FusedBody_u64 {"), std::string::npos)
      << "the fused emission must define the uint64 body.";
  EXPECT_NE(generated.find("struct FusedBody_u32 {"), std::string::npos)
      << "the fused emission must dual emit the uint32 body.";
  EXPECT_NE(generated.find("AscendC::IndirectLoadSimtFused<FusedBody_u32>("), std::string::npos)
      << "the uint32 fused instance must be instantiated.";
  EXPECT_NE(generated.find("AscendC::IndirectLoadSimtFused<FusedBody_u64>("), std::string::npos)
      << "the uint64 fused instance must be instantiated.";
  EXPECT_NE(generated.find("<= 0x7FFFFFFFULL"), std::string::npos)
      << "the fused guard must bound every member view by INT32_MAX.";
}

// [ST 覆盖·静态不双发] 静态 shape（常量求值直接判宽度）不产生双发：覆盖双发门控的
// 拒绝路径（SimtSupportsDualOffsetBody 返回 false），静态路径保持单 body、无运行期
// 宽度开关（双发对静态是冗余同类型副本 + 恒真判定）。
TEST(CodegenIndirectLoadSimtSt, EmbeddingStaticKeepsSingleBody) {
  IndirectLoadSimtStContext context;
  IndirectLoadSimtStGraph g("st_simt_embedding_static", 4, 8, 2, 8);
  BuildStSimtEmbeddingGraph(g);
  AnnotateStSimtEmbeddingTemplate(g.graph);

  Tiler tiler;
  SetupIndirectLoadStTiler(g.graph, tiler);
  TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(g.graph);
  auto il_node = g.graph.FindNode("indirect_load");
  ASSERT_NE(il_node, nullptr);
  IndirectLoadRegApiCall call("IndirectLoadRegApiCall");
  ASSERT_EQ(call.Init(il_node), af::SUCCESS);
  ApiTensor x1, x2;
  x1.id = il_node->inputs[0].attr.mem.tensor_id;
  x2.id = il_node->inputs[1].attr.mem.tensor_id;
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);
  tpipe.AddTensor(il_node->inputs[0]);
  tpipe.AddTensor(il_node->inputs[1]);

  std::stringstream ss;
  ASSERT_EQ(call.GenerateFuncDefinition(tpipe, tiler, ss), af::SUCCESS);
  const std::string definition = ss.str();
  EXPECT_NE(definition.find("struct IndirectLoadSimtBody_indirect_load"), std::string::npos)
      << "the static path must keep a single body.";
  EXPECT_EQ(definition.find("IndirectLoadSimtBody_indirect_load_u32"), std::string::npos)
      << "static shapes must not emit a dual body.";
  EXPECT_EQ(definition.find("use_u32"), std::string::npos) << "static shapes must not emit a runtime width switch.";
}

// [ST 覆盖·kStrided 基线] 动态 kStrided（rank2/axis1 尾轴 gather）非 embedding 门控：
// 覆盖双发判定的 kStrided 尾段拒绝路径（rank==3/axis==1 才 eligible），保持单 body。
// [ST 覆盖·转置输入发射] 输入逻辑视图为非单调（转置）strides 时，SIMT 发射端生成
// 单 body 的 strided 寻址（坐标按维独立乘加，不依赖单调性）。图复用 kStrided 基线
// （rank2），仅输入视图改标转置 strides；转置+零步长广播轴的分类语义由 UT 分类
// 用例锁定，零步长轴 repeats=1 的发射消费由 E2E 端到端覆盖。
void AnnotateStSimtTransposedStridedTemplate(af::AscGraph &graph) {
  auto il = graph.FindNode("indirect_load");
  ASSERT_NE(il, nullptr);
  ASSERT_EQ(::ascir::SetTemplateId(il, ::ascir::TemplateId::kIndirectLoadSimt), af::SUCCESS);
  ASSERT_EQ(SetImplementation(il, Implementation::kDefault), af::SUCCESS);
  const auto input_load = graph.FindNode("x_load");
  const auto index_load = graph.FindNode("idx_load");
  const auto store = graph.FindNode("store");
  ASSERT_NE(input_load, nullptr);
  ASSERT_NE(index_load, nullptr);
  ASSERT_NE(store, nullptr);
  ASSERT_EQ(SetTemplateRole(input_load, TemplateRole::kSimtInputBoundary), af::SUCCESS);
  ASSERT_EQ(SetTemplateRole(index_load, TemplateRole::kSimtDirectGmBoundary), af::SUCCESS);
  ASSERT_EQ(SetTemplateRole(store, TemplateRole::kSimtDirectGmBoundary), af::SUCCESS);

  auto axes = graph.GetAllAxis();
  af::AxisId outer_id = af::kIdNone;
  for (const auto &axis : axes) {
    if (axis == nullptr) continue;
    if (axis->name == "z2z3") outer_id = axis->id;
  }
  ASSERT_NE(outer_id, af::kIdNone);
  TemplateAxes t{outer_id, af::kIdNone, af::kIdNone};
  ASSERT_EQ(SetTemplateAxes(il, t), af::SUCCESS);

  const af::Expression One = af::sym::kSymbolOne;
  TemplateLogicalView view;
  // 转置视图：dim0 stride=1、dim1 stride=s0（行主序单调性在 dim0 违例，倍数链 1→s0 证明）。
  view.input.axis_ids = {axes[0]->id, axes[1]->id};
  view.input.sizes = {graph.FindAxis(axes[0]->id)->size, graph.FindAxis(axes[1]->id)->size};
  view.input.strides = {One, graph.FindAxis(axes[0]->id)->size};
  view.index.axis_ids = {axes[2]->id, axes[3]->id};
  view.index.sizes = {graph.FindAxis(axes[2]->id)->size, graph.FindAxis(axes[3]->id)->size};
  view.index.strides = {graph.FindAxis(axes[3]->id)->size, One};
  view.output.axis_ids = {axes[2]->id, axes[3]->id};
  view.output.sizes = {graph.FindAxis(axes[2]->id)->size, graph.FindAxis(axes[3]->id)->size};
  view.output.strides = {graph.FindAxis(axes[3]->id)->size, One};
  // 纯转置（无广播轴）的倍数链证明是既有能力：默认参数即判 kStrided（回归见 UT
  // ClassifiesTransposedDenseLayoutAsStrided）。此处按 AnalyzeInputPath 的 SIMT
  // 调用参数分类并断言 kStrided，使发射端走 strided 路径；转置+零步长广播轴的
  // 分类语义由 UT 分类用例锁定。
  ASSERT_EQ(ClassifyIndirectLoadLayout(view.input, view.input, false, true), af::SUCCESS);
  ASSERT_EQ(view.input.kind, IndirectLoadLayoutKind::kStrided);
  ASSERT_EQ(ClassifyIndirectLoadLayout(view.index, view.index), af::SUCCESS);
  ASSERT_EQ(SetTemplateLogicalView(il, view), af::SUCCESS);
}

TEST(CodegenIndirectLoadSimtSt, StridedDynamicKeepsSingleBody) {
  IndirectLoadSimtStContext context;
  IndirectLoadSimtStGraph g("st_simt_strided_dynamic");
  BuildStSimtStridedGraph(g);
  AnnotateStSimtStridedTemplate(g.graph);

  Tiler tiler;
  SetupIndirectLoadStTiler(g.graph, tiler);
  TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(g.graph);
  auto il_node = g.graph.FindNode("indirect_load");
  ASSERT_NE(il_node, nullptr);
  IndirectLoadRegApiCall call("IndirectLoadRegApiCall");
  ASSERT_EQ(call.Init(il_node), af::SUCCESS);
  ApiTensor x1, x2;
  x1.id = il_node->inputs[0].attr.mem.tensor_id;
  x2.id = il_node->inputs[1].attr.mem.tensor_id;
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);
  tpipe.AddTensor(il_node->inputs[0]);
  tpipe.AddTensor(il_node->inputs[1]);

  std::stringstream ss;
  ASSERT_EQ(call.GenerateFuncDefinition(tpipe, tiler, ss), af::SUCCESS);
  const std::string definition = ss.str();
  EXPECT_NE(definition.find("struct IndirectLoadSimtBody_indirect_load"), std::string::npos)
      << "the strided path must keep a single body.";
  EXPECT_EQ(definition.find("IndirectLoadSimtBody_indirect_load_u32"), std::string::npos)
      << "rank2/axis1 strided must not dual emit.";
  EXPECT_EQ(definition.find("use_u32"), std::string::npos) << "strided must not emit a runtime width switch.";
}

// 转置（非单调 strides）输入视图：分类为 kStrided 后，SIMT 发射端按维独立乘加寻址
// 生成单 body，不依赖 strides 单调性。
TEST(CodegenIndirectLoadSimtSt, TransposedStridedInputEmitsSingleBody) {
  IndirectLoadSimtStContext context;
  IndirectLoadSimtStGraph g("st_simt_transposed_strided_dynamic");
  BuildStSimtStridedGraph(g);
  AnnotateStSimtTransposedStridedTemplate(g.graph);

  Tiler tiler;
  SetupIndirectLoadStTiler(g.graph, tiler);
  TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(g.graph);
  auto il_node = g.graph.FindNode("indirect_load");
  ASSERT_NE(il_node, nullptr);
  IndirectLoadRegApiCall call("IndirectLoadRegApiCall");
  ASSERT_EQ(call.Init(il_node), af::SUCCESS);
  ApiTensor x1, x2;
  x1.id = il_node->inputs[0].attr.mem.tensor_id;
  x2.id = il_node->inputs[1].attr.mem.tensor_id;
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);
  tpipe.AddTensor(il_node->inputs[0]);
  tpipe.AddTensor(il_node->inputs[1]);

  std::stringstream ss;
  ASSERT_EQ(call.GenerateFuncDefinition(tpipe, tiler, ss), af::SUCCESS);
  const std::string definition = ss.str();
  EXPECT_NE(definition.find("struct IndirectLoadSimtBody_indirect_load"), std::string::npos)
      << "the transposed strided path must keep a single body.";
  EXPECT_EQ(definition.find("use_u32"), std::string::npos)
      << "transposed strided must not emit a runtime width switch.";
}

// [ST 覆盖·post-Reduce] 定义路径双发（定义/调用共用 DecideDualOffsetBody 唯一入口，
// 定义侧双套 body 即证明 post-reduce 未被整体排除——排除条件已收窄为 prepass-active）。
TEST(CodegenIndirectLoadSimtSt, EmbeddingPostReduceStillDualEmits) {
  IndirectLoadSimtStContext context;
  IndirectLoadSimtStGraph g("st_simt_embedding_post_reduce_dynamic");
  BuildStSimtEmbeddingPostReduceGraph(g);
  AnnotateStSimtEmbeddingPostReduceTemplate(g.graph);

  Tiler tiler;
  SetupIndirectLoadStTiler(g.graph, tiler);
  TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(g.graph);
  auto il_node = g.graph.FindNode("indirect_load");
  ASSERT_NE(il_node, nullptr);
  IndirectLoadRegApiCall call("IndirectLoadRegApiCall");
  ASSERT_EQ(call.Init(il_node), af::SUCCESS);
  ApiTensor x1, x2;
  x1.id = il_node->inputs[0].attr.mem.tensor_id;
  x2.id = il_node->inputs[1].attr.mem.tensor_id;
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);
  tpipe.AddTensor(il_node->inputs[0]);
  tpipe.AddTensor(il_node->inputs[1]);

  std::stringstream ss;
  ASSERT_EQ(call.GenerateFuncDefinition(tpipe, tiler, ss), af::SUCCESS);
  const std::string definition = ss.str();

  // 语义锚：post-reduce 未使 kEmbedding 分类回退，且图确实按 post-reduce 候选降级。
  ascgen_utils::indirect_load::IndirectLoadLoweringMetadata semantic_meta;
  ASSERT_EQ(ascgen_utils::indirect_load::GetLoweringMetadata(il_node, semantic_meta), af::SUCCESS);
  ASSERT_EQ(semantic_meta.simt.policy.policy, ascgen_utils::indirect_load::SimtAddressPolicy::kEmbedding)
      << "post-reduce must not affect the kEmbedding classification.";
  ASSERT_TRUE(semantic_meta.simt.has_post_reduce) << "the graph must lower as a post-reduce candidate.";

  EXPECT_NE(definition.find("struct IndirectLoadSimtBody_indirect_load_u64"), std::string::npos)
      << "post-reduce candidates must keep the uint64 body.";
  EXPECT_NE(definition.find("struct IndirectLoadSimtBody_indirect_load_u32"), std::string::npos)
      << "post-reduce candidates must dual emit (exclusion narrowed to prepass-active only).";
  // [P3-b 漂移分支] 本图 has_post_reduce=true 且无 prepass——定义已按首答（双发）落缓存；
  // 此处翻转 prepass_active 重求值走排除臂并触发漂移可见日志（GELOGW 属预期日志噪音），
  // 返回值仍须为缓存首答（定义/调用物理一致的关键）。
  EXPECT_TRUE(call.DecideDualOffsetBody(true)) << "the cached first answer must survive prepass flip.";
}

// ==================== post-Reduce 输出重定向 × VF boundary Load ====================

// VF 子图链 sub_x -> sub_load -> abs -> sub_store -> sub_output：单轴 z2，
// boundary Load 形态与分区器产物同构（sub_x 经 vf 根输入映射到根图生产者边）。
void BuildStVfBoundarySubGraph(af::AscGraph &sub_graph, const af::AxisId z2, const af::Expression &s2) {
  Data sub_x_op("sub_x", sub_graph);
  sub_x_op.ir_attr.SetIndex(0);
  sub_x_op.attr.api.unit = af::ComputeUnit::kUnitNone;
  Load sub_load_op("sub_load");
  Abs abs_op("abs");
  Store sub_store_op("sub_store");
  Output sub_output_op("sub_output");
  sub_output_op.ir_attr.SetIndex(0);
  sub_output_op.attr.api.unit = af::ComputeUnit::kUnitNone;

  sub_load_op.x = sub_x_op.y;
  sub_load_op.attr.sched.axis = {z2};
  sub_load_op.attr.sched.loop_axis = -1;
  *sub_load_op.y.axis = {z2};
  *sub_load_op.y.repeats = {s2};
  *sub_load_op.y.strides = {af::sym::kSymbolOne};

  abs_op.x = sub_load_op.y;
  abs_op.attr.sched.axis = {z2};
  abs_op.attr.sched.loop_axis = -1;
  *abs_op.y.axis = {z2};
  *abs_op.y.repeats = {s2};
  *abs_op.y.strides = {af::sym::kSymbolOne};

  sub_store_op.x = abs_op.y;
  sub_store_op.attr.sched.axis = {z2};
  sub_store_op.attr.sched.loop_axis = -1;
  *sub_store_op.y.axis = {z2};
  *sub_store_op.y.repeats = {s2};
  *sub_store_op.y.strides = {af::sym::kSymbolOne};

  sub_output_op.x = sub_store_op.y;
}

// 根图挂 VectorFunc 消费链 vf -> store2 -> y2；il 的输出边作 vf 输入（图边 tensor
// 保持分区快照语义）。il 句柄由图节点重建（Build 函数内的 op 句柄已析构）。
void AttachStVfBoundaryRoot(IndirectLoadSimtStGraph &g, VectorFunc &vf_op, Store &store_op, Output &y_op) {
  IndirectLoad il_handle("indirect_load");
  const auto il_node = g.graph.FindNode("indirect_load");
  ASSERT_NE(il_node, nullptr);
  ASSERT_EQ(af::NodeUtilsEx::SetNodeToOperator(il_handle, il_node), ge::GRAPH_SUCCESS);
  g.graph.AddNode(store_op);
  g.graph.AddNode(y_op);
  af::AscGraph vf_sub_graph("vf_boundary_sub");
  BuildStVfBoundarySubGraph(vf_sub_graph, g.z2.id, g.s2);
  g.graph.AddSubGraph(vf_sub_graph);
  vf_op.SetAttr("sub_graph_name", "vf_boundary_sub");
  vf_op.InstanceOutputy(1);
  vf_op.x = {il_handle.y};
  *vf_op.y[0].axis = {g.z2.id};
  *vf_op.y[0].repeats = {g.s2};
  *vf_op.y[0].strides = {af::sym::kSymbolOne};

  store_op.x = vf_op.y[0];
  store_op.ir_attr.SetOffset(af::ops::Zero);
  *store_op.y.axis = {g.z2.id};
  *store_op.y.repeats = {g.s2};
  *store_op.y.strides = {af::sym::kSymbolOne};

  y_op.ir_attr.SetIndex(1);
  y_op.x = store_op.y;
  y_op.y.dtype = ge::DT_FLOAT16;
}

// vf/store2 节点输出 attr：vf 输出注册为新 UB 队列 tensor，子图计算节点输出按
// 分区中间态补齐 position/tensor id。
void SetStVfBoundaryTensorAttrs(IndirectLoadSimtStGraph &g) {
  auto set_vec_in = [](const af::AscNodePtr &node, const af::AxisId z2, const int64_t tensor_id) {
    node->outputs[0].attr.vectorized_axis = {z2};
    node->outputs[0].attr.vectorized_strides = {af::sym::kSymbolOne};
    node->outputs[0].attr.dtype = ge::DT_FLOAT16;
    node->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
    node->outputs[0].attr.mem.tensor_id = tensor_id;
  };
  af::AscGraph vf_sub_graph("vf_boundary_sub");
  ASSERT_EQ(g.graph.FindSubGraph("vf_boundary_sub", vf_sub_graph), af::SUCCESS);
  auto sub_load = vf_sub_graph.FindNode("sub_load");
  ASSERT_NE(sub_load, nullptr);
  sub_load->attr.sched.loop_axis = -1;
  set_vec_in(sub_load, g.z2.id, 12);
  auto abs = vf_sub_graph.FindNode("abs");
  ASSERT_NE(abs, nullptr);
  abs->attr.sched.loop_axis = -1;
  set_vec_in(abs, g.z2.id, 13);
  auto sub_store = vf_sub_graph.FindNode("sub_store");
  ASSERT_NE(sub_store, nullptr);
  sub_store->attr.sched.loop_axis = -1;
  set_vec_in(sub_store, g.z2.id, 14);

  auto vf = g.graph.FindNode("vf");
  ASSERT_NE(vf, nullptr);
  vf->attr.sched.loop_axis = -1;
  vf->outputs[0].attr.vectorized_axis = {g.z2.id};
  vf->outputs[0].attr.vectorized_strides = {af::sym::kSymbolOne};
  vf->outputs[0].attr.dtype = ge::DT_FLOAT16;
  vf->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  vf->outputs[0].attr.mem.tensor_id = 10;
  vf->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  vf->outputs[0].attr.que.id = 6;
  vf->outputs[0].attr.opt.merge_scope = af::kIdNone;
}

// post-Reduce 输出重定向后，IL ApiCall 输出 id 与图边快照 id 不同源；VF 消费该边时
// 函数定义必须把 boundary Load 校正到重定向后的 id（与签名/实参同源），否则函数体
// 引用无写入也无声明的旧 buffer。覆盖 VfCall boundary 改写映射构建、递归改写与
// UB 输入 id 校正的完整路径（ST_report 增量行覆盖）。
TEST(CodegenIndirectLoadSimtSt, EmbeddingPostReduceRedirectsVfBoundaryLoad) {
  IndirectLoadSimtStContext context;
  IndirectLoadSimtStGraph g("st_simt_post_reduce_vf_boundary");
  BuildStSimtEmbeddingPostReduceGraph(g);
  AnnotateStSimtEmbeddingPostReduceTemplate(g.graph);

  Tiler tiler;
  SetupIndirectLoadStTiler(g.graph, tiler);
  TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(g.graph);
  auto il_node = g.graph.FindNode("indirect_load");
  ASSERT_NE(il_node, nullptr);
  IndirectLoadRegApiCall il_call("IndirectLoadRegApiCall");
  ASSERT_EQ(il_call.Init(il_node), af::SUCCESS);
  // post-Reduce 候选在 Init 内改写输出 id：ApiCall 输出指向合成链 result tensor。
  ASSERT_EQ(il_call.outputs[0].id, 6) << "post-reduce must redirect the output id.";

  VectorFunc vf_op("vf");
  Store store_op("store2");
  Output y_op("y2");
  AttachStVfBoundaryRoot(g, vf_op, store_op, y_op);
  SetStVfBoundaryTensorAttrs(g);
  tpipe.AddTensor(g.graph.FindNode("output_transform")->outputs[0]);
  tpipe.AddTensor(g.graph.FindNode("vf")->outputs[0]);

  VfCall vf_call;
  ASSERT_EQ(vf_call.Init(g.graph.FindNode("vf")), af::SUCCESS);
  vf_call.inputs.push_back(&il_call.outputs[0]);
  std::stringstream ss;
  ASSERT_EQ(vf_call.GenerateFuncDefinition(tpipe, tiler, ss), af::SUCCESS);
  const std::string definition = ss.str();

  const auto load_align_offset = definition.find("LoadAlign");
  ASSERT_NE(load_align_offset, std::string::npos) << definition;
  const auto line_begin = definition.rfind('\n', load_align_offset);
  const auto line_end = definition.find('\n', load_align_offset);
  const auto load_align_line =
      definition.substr(line_begin == std::string::npos ? 0UL : line_begin + 1UL,
                        line_end == std::string::npos ? std::string::npos : line_end - line_begin - 1UL);
  EXPECT_NE(load_align_line.find("local_6"), std::string::npos)
      << "the boundary Load must read the redirected tensor: " << load_align_line;
  EXPECT_EQ(load_align_line.find("local_1"), std::string::npos)
      << "the boundary Load must not read the stale edge tensor: " << load_align_line;
  EXPECT_NE(definition.find("local_6_addr"), std::string::npos) << "the signature must use the redirected tensor.";
}
