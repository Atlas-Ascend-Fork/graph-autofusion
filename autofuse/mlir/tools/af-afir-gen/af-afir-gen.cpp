/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- af-afir-gen.cpp - Emit AFIR from a built-in schedule example ----*- C++ -*-===//
//
// Standalone driver that exercises the AfirBuilder core with a hand-built
// elewise+broadcast PlanInfo and prints the resulting AFIR module. Pipe into
// `af-opt --convert-afir-to-ascendc` to run the AFIR-to-AscendC lowering.
//
//===----------------------------------------------------------------------===//

#include "bridge/AfirBuilder/AfirBuilder.h"

#include <cstdlib>

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir::afir::bridge;

// 示例张量的静态维、符号尺寸表达式和轴编号。
struct TensorShapeSpec {
  std::vector<int64_t> shape;
  std::vector<std::string> shapeExprs;
  std::vector<int64_t> axes;
};

// 示例只区分 GM（position=0）与片上存储（非零），片上张量统一使用队列。
// 这里构造最小合法输入，不覆盖调度器的全部存储形态。
static TensorValueDTO makeTensor(TensorShapeSpec shapeSpec, int64_t tensorId, int64_t position,
                                 mlir::afir::Role role = mlir::afir::Role::Compute) {
  TensorValueDTO value;
  value.tensor.shape = std::move(shapeSpec.shape);
  value.tensor.shape_exprs = std::move(shapeSpec.shapeExprs);
  value.tensor.dtype = "float16";
  value.tensor.axis = std::move(shapeSpec.axes);
  value.tensor.role = role;
  value.schedule.tensorId = tensorId;
  value.schedule.domain = position == 0 ? BindingDomain::Buffer : BindingDomain::Queue;
  value.schedule.queueId = position == 0 ? -1 : tensorId;
  value.schedule.bufferId = tensorId;
  value.schedule.queueDepth = position == 0 ? -1 : 2;
  value.schedule.queueBufferCount = position == 0 ? -1 : 2;
  value.buffer.space = position == 0 ? mlir::afir::Space::GM : mlir::afir::Space::UB;
  value.buffer.bufferId = tensorId;
  value.buffer.slotCount = position == 0 ? -1 : 2;
  value.buffer.queueDepth = position == 0 ? -1 : 2;
  value.buffer.lifeTimeAxisId = -1;
  value.buffer.sizeBytesExpr =
      value.tensor.shape_exprs.size() == 2 && value.tensor.shape_exprs[0] == "s0" ? "(16 * s0)" : "16";
  return value;
}

// 这个示例 template 的三个 GM 侧张量：两个输入与一个输出。
struct ElewiseBrcBoundary {
  const TensorValueDTO &x;
  const TensorValueDTO &b;
  const TensorValueDTO &y;
};

// x/b 各自 Load 上 UB，b 再 Broadcast 开，相加后 Store 回 y。
static void appendElewiseBrcNodes(TemplateInfo &g, ElewiseBrcBoundary boundary, int64_t dyn) {
  auto ubIn = [](std::vector<int64_t> shape, std::vector<std::string> exprs, int64_t tid) {
    return makeTensor({std::move(shape), std::move(exprs), {0, 1}}, tid, /*vector_in=*/1, mlir::afir::Role::Compute);
  };
  auto ubCalc = [dyn](int64_t tid) { return makeTensor({{dyn, 8}, {"s0", "8"}, {0, 1}}, tid, /*vector_calc=*/3); };

  auto node = [](std::string name, std::string type, std::vector<std::string> inputNames, TensorValueDTO output,
                 std::vector<TensorValueDTO> inputs, std::vector<int64_t> schedAxis = {}) {
    NodeInfo info;
    info.name = std::move(name);
    info.type = std::move(type);
    info.input_names = std::move(inputNames);
    info.inputs = std::move(inputs);
    info.output = std::move(output);
    info.sched_axis = std::move(schedAxis);
    return info;
  };
  g.nodes.push_back(node("x", "Data", {}, boundary.x, {}));
  g.nodes.push_back(node("b", "Data", {}, boundary.b, {}));
  g.nodes.push_back(node("load_x", "Load", {"x"}, ubIn({dyn, 8}, {"s0", "8"}, 2), {boundary.x}, {0, 1}));
  g.nodes.push_back(node("load_b", "Load", {"b"}, ubIn({1, 8}, {"1", "8"}, 3), {boundary.b}, {0, 1}));
  TensorValueDTO loadB = g.nodes.back().output;
  g.nodes.push_back(node("bc_b", "Broadcast", {"load_b"}, ubCalc(4), {loadB}, {0, 1}));
  TensorValueDTO loadX = g.nodes[2].output;
  TensorValueDTO bcB = g.nodes.back().output;
  g.nodes.push_back(node("add", "Add", {"load_x", "bc_b"}, ubCalc(5), {loadX, bcB}, {0, 1}));
  TensorValueDTO add = g.nodes.back().output;
  TensorValueDTO store = makeTensor({{dyn, 8}, {"s0", "8"}, {0, 1}}, 6, /*vector_out=*/2, mlir::afir::Role::Output);
  g.nodes.push_back(node("store_y", "Store", {"add"}, store, {add}, {0, 1}));
  g.nodes.push_back(node("y", "Output", {"store_y"}, boundary.y, {store}));
}

static TemplateInfo makeTemplate(int64_t stageId, int64_t templateId) {
  const int64_t dyn = std::getenv("AF_AFIR_GEN_BUFFER") ? 32 : mlir::ShapedType::kDynamic;
  TensorValueDTO xTy = makeTensor({{dyn, 8}, {"s0", "8"}, {0, 1}}, 0, /*gm=*/0, mlir::afir::Role::Input);
  TensorValueDTO bTy = makeTensor(
      {{1, 8}, {"1", "8"}, std::getenv("AF_AFIR_GEN_BUFFER") ? std::vector<int64_t>{0, 1} : std::vector<int64_t>{1}}, 1,
      /*gm=*/0, mlir::afir::Role::Input);
  TensorValueDTO yTy = makeTensor({{dyn, 8}, {"s0", "8"}, {0, 1}}, 6, /*gm=*/0, mlir::afir::Role::Output);

  TemplateInfo g;
  g.name = "stage" + std::to_string(stageId) + "_template" + std::to_string(templateId);
  g.original_name = "elewise_brc";
  g.stage_id = stageId;
  g.template_id = templateId;
  g.tiling_key = templateId;
  g.size_vars = {"s0"};
  g.inputs = {{"x", xTy, 0}, {"b", bTy, 1}};
  g.outputs = {{"y", yTy, 0}};
  g.buffer_form = std::getenv("AF_AFIR_GEN_BUFFER") != nullptr;
  g.axes = {
      {/*id=*/0, "m", "original", "s0"},
      {/*id=*/1, "n", "tile_inner", "8"},
  };

  appendElewiseBrcNodes(g, {xTy, bTy, yTy}, dyn);
  return g;
}

static PlanInfo makeElewiseBrcExample() {
  PlanInfo result;
  result.name = "elewise_brc_plan0";
  result.subgraph_id = 0;
  result.plan_id = 0;
  result.template_counts_per_stage = {2, 1};
  result.origin_vars = {"s0"};
  result.templates = {makeTemplate(0, 0), makeTemplate(0, 1), makeTemplate(1, 0)};
  return result;
}

int main() {
  mlir::MLIRContext context;
  PlanInfo plan = makeElewiseBrcExample();
  auto module = BuildAfirModule(context, plan);
  if (!module) {
    llvm::errs() << "AfirBuilder failed\n";
    return 1;
  }
  module->print(llvm::outs());
  return 0;
}
