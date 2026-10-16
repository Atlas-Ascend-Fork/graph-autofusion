/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <ascendc_ir.h>
#include <ascir_ops.h>
#include <ascir_utils.h>
#include <algorithm>
#include <iostream>

#include "gtest/gtest.h"

#include "ascendc_ir.h"
#include "ascir_ops.h"
#include "ascir_utils.h"

#include "graph/utils/graph_utils.h"
#include "graph_utils_ex.h"

#define private public
#include "autoschedule/autoschedule.h"
#include "autoschedule/alignment_handler.h"
#undef private
#include "ascir_ops_utils.h"
#include "autoschedule/tiling_group.h"
#include "schedule_utils.h"
#include "ascir_utils.h"
#include "platform_context.h"
#include "platform/v1/platformv1.h"
#include "ascgraph_info_complete.h"
#include "indirect_load_utils.h"
#include "schedule_result.h"
#include "task_generator/indirect_load_schedule_case_generator.h"

using namespace std;
using namespace ascir;
using namespace ge;
using namespace af::ops;
using namespace af::ascir_op;
using namespace optimize::autoschedule;

static void Construct_LoadAbsStore(af::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  af::ascir_op::Data x("x", graph);
  x.attr.sched.axis = {z0.id, z1.id, z2.id};
  x.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  x.attr.api.type = af::ApiType::kAPITypeBuffer;
  x.y.dtype = ge::DT_FLOAT16;
  *x.y.axis = {z0.id, z1.id, z2.id};
  *x.y.repeats = {s0, s1, s2};
  *x.y.strides = {s1 * s2, s2, One};

  af::ascir_op::Load load("load");
  graph.AddNode(load);
  load.x = x.y;
  load.attr.sched.axis = {z0.id, z1.id, z2.id};
  load.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load.y.dtype = ge::DT_FLOAT16;
  *load.y.axis = {z0.id, z1.id, z2.id};
  *load.y.repeats = {s0, s1, s2};
  *load.y.strides = {s1 * s2, s2, One};

  af::ascir_op::Abs abs("abs");
  graph.AddNode(abs);
  abs.x = load.y;
  abs.attr.sched.axis = {z0.id, z1.id, z2.id};
  abs.attr.api.compute_type = af::ComputeType::kComputeElewise;
  abs.y.dtype = ge::DT_FLOAT16;
  *abs.y.axis = {z0.id, z1.id, z2.id};
  *abs.y.repeats = {s0, s1, s2};
  *abs.y.strides = {s1 * s2, s2, One};

  af::ascir_op::Store store("store");
  graph.AddNode(store);
  store.x = abs.y;
  store.attr.sched.axis = {z0.id, z1.id, z2.id};
  store.attr.api.compute_type = af::ComputeType::kComputeStore;
  store.y.dtype = ge::DT_FLOAT16;
  *store.y.axis = {z0.id, z1.id, z2.id};
  *store.y.repeats = {s0, s1, s2};
  *store.y.strides = {s1 * s2, s2, One};

  af::ascir_op::Output y("y");
  y.x = store.y;
  y.attr.sched.axis = {z0.id, z1.id, z2.id};
  y.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  y.attr.api.type = af::ApiType::kAPITypeBuffer;
  y.y.dtype = ge::DT_FLOAT16;
  *y.y.axis = {z0.id, z1.id, z2.id};
  *y.y.repeats = {s0, s1, s2};
  *y.y.strides = {s1 * s2, s2, One};
}

static void Construct_LoadGatherStore(af::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar("s3");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);
  auto z4 = graph.CreateAxis("z4", s2 + s3);

  af::ascir_op::Data x1("x1", graph);
  x1.attr.sched.axis = {z0.id, z1.id};
  x1.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  x1.attr.api.type = af::ApiType::kAPITypeBuffer;
  x1.y.dtype = ge::DT_FLOAT;
  *x1.y.axis = {z0.id, z1.id};
  *x1.y.repeats = {s0, s1};
  *x1.y.strides = {s1, One};

  af::ascir_op::Data x2("x2", graph);
  x2.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  x2.attr.api.type = af::ApiType::kAPITypeBuffer;
  x2.y.dtype = ge::DT_INT32;
  x2.attr.sched.axis = {z4.id};
  *x2.y.axis = {z4.id};
  *x2.y.repeats = {s2 + s3};
  *x2.y.strides = {One};

  af::ascir_op::Gather gather("gather");
  graph.AddNode(gather);
  gather.x1 = x1.y;
  gather.x2 = x2.y;
  gather.attr.api.compute_type = af::ComputeType::kComputeGather;
  gather.y.dtype = ge::DT_FLOAT;
  gather.attr.sched.axis = {z0.id, z4.id};
  *gather.y.axis = {z0.id, z4.id};
  *gather.y.repeats = {s0, s2 + s3};
  *gather.y.strides = {s2 + s3, One};
  gather.ir_attr.SetAxis(1);

  af::ascir_op::Store store("store");
  graph.AddNode(store);
  store.x = gather.y;
  store.attr.api.compute_type = af::ComputeType::kComputeStore;
  store.y.dtype = ge::DT_FLOAT;
  store.attr.sched.axis = {z0.id, z4.id};
  *store.y.axis = {z0.id, z4.id};
  *store.y.repeats = {s0, s2 + s3};
  *store.y.strides = {s2 + s3, One};

  af::ascir_op::Output y("y");
  y.x = store.y;
  y.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  y.attr.api.type = af::ApiType::kAPITypeBuffer;
  y.y.dtype = ge::DT_FLOAT;
  y.attr.sched.axis = {z0.id, z4.id};
  *y.y.axis = {z0.id, z4.id};
  *y.y.repeats = {s0, s2 + s3};
  *y.y.strides = {s2 + s3, One};
}

static void Construct_ElementwiseAbs(af::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar("s0");

  auto z0 = graph.CreateAxis("z0", s0);

  af::ascir_op::Data x("x", graph);
  x.attr.sched.axis = {z0.id};
  x.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  x.attr.api.type = af::ApiType::kAPITypeBuffer;
  x.y.dtype = ge::DT_FLOAT16;
  *x.y.axis = {z0.id};
  *x.y.repeats = {s0};
  *x.y.strides = {One};

  af::ascir_op::Load load("load");
  graph.AddNode(load);
  load.x = x.y;
  load.attr.sched.axis = {z0.id};
  load.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load.y.dtype = ge::DT_FLOAT16;
  *load.y.axis = {z0.id};
  *load.y.repeats = {s0};
  *load.y.strides = {One};

  af::ascir_op::Abs abs("abs");
  graph.AddNode(abs);
  abs.x = load.y;
  abs.attr.sched.axis = {z0.id};
  abs.attr.api.compute_type = af::ComputeType::kComputeElewise;
  abs.y.dtype = ge::DT_FLOAT16;
  *abs.y.axis = {z0.id};
  *abs.y.repeats = {s0};
  *abs.y.strides = {One};

  af::ascir_op::Store store("store");
  graph.AddNode(store);
  store.x = abs.y;
  store.attr.sched.axis = {z0.id};
  store.attr.api.compute_type = af::ComputeType::kComputeStore;
  store.y.dtype = ge::DT_FLOAT16;
  *store.y.axis = {z0.id};
  *store.y.repeats = {s0};
  *store.y.strides = {One};

  af::ascir_op::Output y("y");
  y.x = store.y;
  y.attr.sched.axis = {z0.id};
  y.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  y.attr.api.type = af::ApiType::kAPITypeBuffer;
  y.y.dtype = ge::DT_FLOAT16;
  *y.y.axis = {z0.id};
  *y.y.repeats = {s0};
  *y.y.strides = {One};
}

static void Construct_ElementwiseFusion(af::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar("s0");

  auto z0 = graph.CreateAxis("z0", s0);

  af::ascir_op::Data x("x", graph);
  x.attr.sched.axis = {z0.id};
  x.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  x.attr.api.type = af::ApiType::kAPITypeBuffer;
  x.y.dtype = ge::DT_FLOAT16;
  *x.y.axis = {z0.id};
  *x.y.repeats = {s0};
  *x.y.strides = {One};

  af::ascir_op::Load load("load");
  graph.AddNode(load);
  load.x = x.y;
  load.attr.sched.axis = {z0.id};
  load.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load.y.dtype = ge::DT_FLOAT16;
  *load.y.axis = {z0.id};
  *load.y.repeats = {s0};
  *load.y.strides = {One};

  af::ascir_op::Abs abs0("abs0");
  graph.AddNode(abs0);
  abs0.x = load.y;
  abs0.attr.sched.axis = {z0.id};
  abs0.attr.api.compute_type = af::ComputeType::kComputeElewise;
  abs0.y.dtype = ge::DT_FLOAT16;
  *abs0.y.axis = {z0.id};
  *abs0.y.repeats = {s0};
  *abs0.y.strides = {One};

  af::ascir_op::Abs abs1("abs1");
  graph.AddNode(abs1);
  abs1.x = abs0.y;
  abs1.attr.sched.axis = {z0.id};
  abs1.attr.api.compute_type = af::ComputeType::kComputeElewise;
  abs1.y.dtype = ge::DT_FLOAT16;
  *abs1.y.axis = {z0.id};
  *abs1.y.repeats = {s0};
  *abs1.y.strides = {One};

  af::ascir_op::Store store("store");
  graph.AddNode(store);
  store.x = abs1.y;
  store.attr.sched.axis = {z0.id};
  store.attr.api.compute_type = af::ComputeType::kComputeStore;
  store.y.dtype = ge::DT_FLOAT16;
  *store.y.axis = {z0.id};
  *store.y.repeats = {s0};
  *store.y.strides = {One};

  af::ascir_op::Output y("y");
  y.x = store.y;
  y.attr.sched.axis = {z0.id};
  y.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  y.attr.api.type = af::ApiType::kAPITypeBuffer;
  y.y.dtype = ge::DT_FLOAT16;
  *y.y.axis = {z0.id};
  *y.y.repeats = {s0};
  *y.y.strides = {One};
}

/**
 *                   store
 *                     |
 *                   mul0
 *                  /   \
 *               add0  exp1
 *              /    \ /
 *    (remove)brc1    \
 *             |      |
 *            exp0   brc0(remove)
 *              \   /
 *              abs0
 *               |
 *             load0
 *              |
 *            data0
 */
static void Construct_RedundantBroadcastFusion(af::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  af::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.attr.sched.axis = {z0.id, z1.id, z2.id};
  data0.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  data0.attr.api.type = af::ApiType::kAPITypeBuffer;
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id, z2.id};
  *data0.y.repeats = {One, s1, s2};
  *data0.y.strides = {Zero, s2, One};

  af::ascir_op::Load load0("load0");
  graph.AddNode(load0);
  load0.x = data0.y;
  load0.attr.sched.axis = {z0.id, z1.id, z2.id};
  load0.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load0.y.dtype = ge::DT_FLOAT16;
  *load0.y.axis = {z0.id, z1.id, z2.id};
  *load0.y.repeats = {One, s1, s2};
  *load0.y.strides = {Zero, s2, One};

  af::ascir_op::Abs abs0("abs0");
  graph.AddNode(abs0);
  abs0.x = load0.y;
  abs0.attr.sched.axis = {z0.id, z1.id, z2.id};
  abs0.attr.api.compute_type = af::ComputeType::kComputeElewise;
  abs0.y.dtype = ge::DT_FLOAT16;
  *abs0.y.axis = {z0.id, z1.id, z2.id};
  *abs0.y.repeats = {One, s1, s2};
  *abs0.y.strides = {Zero, s2, One};

  af::ascir_op::Exp exp0("exp0");
  graph.AddNode(exp0);
  exp0.x = abs0.y;
  exp0.attr.sched.axis = {z0.id, z1.id, z2.id};
  exp0.attr.api.compute_type = af::ComputeType::kComputeElewise;
  exp0.y.dtype = ge::DT_FLOAT16;
  *exp0.y.axis = {z0.id, z1.id, z2.id};
  *exp0.y.repeats = {One, s1, s2};
  *exp0.y.strides = {Zero, s2, One};

  af::ascir_op::Broadcast brc0("brc0");
  graph.AddNode(brc0);
  brc0.x = abs0.y;
  brc0.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc0.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc0.y.dtype = ge::DT_FLOAT16;
  *brc0.y.axis = {z0.id, z1.id, z2.id};
  *brc0.y.repeats = {s0, s1, s2};
  *brc0.y.strides = {s1 * s2, s2, One};

  af::ascir_op::Broadcast brc1("brc1");
  graph.AddNode(brc1);
  brc1.x = exp0.y;
  brc1.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc1.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc1.y.dtype = ge::DT_FLOAT16;
  *brc1.y.axis = {z0.id, z1.id, z2.id};
  *brc1.y.repeats = {s0, s1, s2};
  *brc1.y.strides = {s1 * s2, s2, One};

  af::ascir_op::Add add0("add0");
  graph.AddNode(add0);
  add0.x1 = brc0.y;
  add0.x2 = brc1.y;
  add0.attr.sched.axis = {z0.id, z1.id, z2.id};
  add0.attr.api.compute_type = af::ComputeType::kComputeElewise;
  add0.y.dtype = ge::DT_FLOAT16;
  *add0.y.axis = {z0.id, z1.id, z2.id};
  *add0.y.repeats = {s0, s1, s2};
  *add0.y.strides = {s1 * s2, s2, One};

  af::ascir_op::Abs exp1("exp1");
  graph.AddNode(exp1);
  exp1.x = brc0.y;
  exp1.attr.sched.axis = {z0.id, z1.id, z2.id};
  exp1.attr.api.compute_type = af::ComputeType::kComputeElewise;
  exp1.y.dtype = ge::DT_FLOAT16;
  *exp1.y.axis = {z0.id, z1.id, z2.id};
  *exp1.y.repeats = {s0, s1, s2};
  *exp1.y.strides = {s1 * s2, s2, One};

  af::ascir_op::Mul mul0("mul0");
  graph.AddNode(mul0);
  mul0.x1 = add0.y;
  mul0.x2 = exp1.y;
  mul0.attr.sched.axis = {z0.id, z1.id, z2.id};
  mul0.attr.api.compute_type = af::ComputeType::kComputeElewise;
  mul0.y.dtype = ge::DT_FLOAT16;
  *mul0.y.axis = {z0.id, z1.id, z2.id};
  *mul0.y.repeats = {s0, s1, s2};
  *mul0.y.strides = {s1 * s2, s2, One};

  af::ascir_op::Store store("store");
  graph.AddNode(store);
  store.x = mul0.y;
  store.attr.sched.axis = {z0.id, z1.id, z2.id};
  store.attr.api.compute_type = af::ComputeType::kComputeStore;
  store.y.dtype = ge::DT_FLOAT16;
  *store.y.axis = {z0.id, z1.id, z2.id};
  *store.y.repeats = {s0, s1, s2};
  *store.y.strides = {s1 * s2, s2, One};

  af::ascir_op::Output y("y");
  y.ir_attr.SetIndex(0);
  y.x = store.y;
  y.attr.sched.axis = {z0.id, z1.id, z2.id};
  y.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  y.attr.api.type = af::ApiType::kAPITypeBuffer;
  y.y.dtype = ge::DT_FLOAT16;
  *y.y.axis = {z0.id, z1.id, z2.id};
  *y.y.repeats = {s0, s1, s2};
  *y.y.strides = {s1 * s2, s2, One};
}

static void Construct_BroadcastExpandNonVectorizedAxis(af::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  af::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  data0.attr.api.type = af::ApiType::kAPITypeBuffer;
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id};
  *data0.y.repeats = {One, s1};
  *data0.y.strides = {Zero, One};

  af::ascir_op::Load load0("load0");
  graph.AddNode(load0);
  load0.x = data0.y;
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load0.y.dtype = ge::DT_FLOAT16;
  *load0.y.axis = {z0.id, z1.id};
  *load0.y.repeats = {One, s1};
  *load0.y.strides = {Zero, One};
  *load0.y.vectorized_axis = {z1.id};

  af::ascir_op::Abs abs0("abs0");
  graph.AddNode(abs0);
  abs0.x = load0.y;
  abs0.attr.sched.axis = {z0.id, z1.id};
  abs0.attr.api.compute_type = af::ComputeType::kComputeElewise;
  abs0.y.dtype = ge::DT_FLOAT16;
  *abs0.y.axis = {z0.id, z1.id};
  *abs0.y.repeats = {One, s1};
  *abs0.y.strides = {Zero, One};
  *abs0.y.vectorized_axis = {z1.id};

  af::ascir_op::Broadcast brc0("brc0");
  graph.AddNode(brc0);
  brc0.x = abs0.y;
  brc0.attr.sched.axis = {z0.id, z1.id};
  brc0.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc0.y.dtype = ge::DT_FLOAT16;
  *brc0.y.axis = {z0.id, z1.id};
  *brc0.y.repeats = {s0, s1};
  *brc0.y.strides = {s1, One};
  *brc0.y.vectorized_axis = {z1.id};

  af::ascir_op::Sum reduce_sum("reduce_sum");
  graph.AddNode(reduce_sum);
  reduce_sum.x = brc0.y;
  reduce_sum.attr.sched.axis = {z0.id, z1.id};
  reduce_sum.attr.api.compute_type = af::ComputeType::kComputeReduce;
  reduce_sum.y.dtype = ge::DT_FLOAT16;
  *reduce_sum.y.axis = {z0.id, z1.id};
  *reduce_sum.y.repeats = {One, s1};
  *reduce_sum.y.strides = {Zero, One};
  *reduce_sum.y.vectorized_axis = {z1.id};

  af::ascir_op::Store store0("store0");
  graph.AddNode(store0);
  store0.x = reduce_sum.y;
  store0.attr.sched.axis = {z0.id, z1.id};
  store0.attr.api.compute_type = af::ComputeType::kComputeStore;
  store0.y.dtype = ge::DT_FLOAT16;
  *store0.y.axis = {z0.id, z1.id};
  *store0.y.repeats = {One, s1};
  *store0.y.strides = {Zero, One};

  af::ascir_op::Output y("y");
  y.ir_attr.SetIndex(0);
  y.x = store0.y;
  y.attr.sched.axis = {z0.id, z1.id};
  y.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  y.attr.api.type = af::ApiType::kAPITypeBuffer;
  y.y.dtype = ge::DT_FLOAT16;
  *y.y.axis = {z0.id, z1.id};
  *y.y.repeats = {One, s1};
  *y.y.strides = {Zero, One};
}

static void Construct_BroadcastExpandNonVectorizedAxisReduceDifferentAxis(af::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  af::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  data0.attr.api.type = af::ApiType::kAPITypeBuffer;
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id};
  *data0.y.repeats = {One, s1};
  *data0.y.strides = {Zero, One};

  af::ascir_op::Load load0("load0");
  graph.AddNode(load0);
  load0.x = data0.y;
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load0.y.dtype = ge::DT_FLOAT16;
  *load0.y.axis = {z0.id, z1.id};
  *load0.y.repeats = {One, s1};
  *load0.y.strides = {Zero, One};
  *load0.y.vectorized_axis = {z1.id};

  af::ascir_op::Abs abs0("abs0");
  graph.AddNode(abs0);
  abs0.x = load0.y;
  abs0.attr.sched.axis = {z0.id, z1.id};
  abs0.attr.api.compute_type = af::ComputeType::kComputeElewise;
  abs0.y.dtype = ge::DT_FLOAT16;
  *abs0.y.axis = {z0.id, z1.id};
  *abs0.y.repeats = {One, s1};
  *abs0.y.strides = {Zero, One};
  *abs0.y.vectorized_axis = {z1.id};

  af::ascir_op::Broadcast brc0("brc0");
  graph.AddNode(brc0);
  brc0.x = abs0.y;
  brc0.attr.sched.axis = {z0.id, z1.id};
  brc0.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc0.y.dtype = ge::DT_FLOAT16;
  *brc0.y.axis = {z0.id, z1.id};
  *brc0.y.repeats = {s0, s1};
  *brc0.y.strides = {s1, One};
  *brc0.y.vectorized_axis = {z1.id};

  af::ascir_op::Sum reduce_sum("reduce_sum");
  graph.AddNode(reduce_sum);
  reduce_sum.x = brc0.y;
  reduce_sum.attr.sched.axis = {z0.id, z1.id};
  reduce_sum.attr.api.compute_type = af::ComputeType::kComputeReduce;
  reduce_sum.y.dtype = ge::DT_FLOAT16;
  *reduce_sum.y.axis = {z0.id, z1.id};
  *reduce_sum.y.repeats = {s0, One};
  *reduce_sum.y.strides = {One, Zero};
  *reduce_sum.y.vectorized_axis = {z1.id};

  af::ascir_op::Store store0("store0");
  graph.AddNode(store0);
  store0.x = reduce_sum.y;
  store0.attr.sched.axis = {z0.id, z1.id};
  store0.attr.api.compute_type = af::ComputeType::kComputeStore;
  store0.y.dtype = ge::DT_FLOAT16;
  *store0.y.axis = {z0.id, z1.id};
  *store0.y.repeats = {s0, One};
  *store0.y.strides = {One, Zero};

  af::ascir_op::Output y("y");
  y.ir_attr.SetIndex(0);
  y.x = store0.y;
  y.attr.sched.axis = {z0.id, z1.id};
  y.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  y.attr.api.type = af::ApiType::kAPITypeBuffer;
  y.y.dtype = ge::DT_FLOAT16;
  *y.y.axis = {z0.id, z1.id};
  *y.y.repeats = {s0, One};
  *y.y.strides = {One, Zero};
}

static void Construct_BroadcastAddNonVectorizedAxis(af::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  af::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.attr.sched.axis = {z1.id};
  data0.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  data0.attr.api.type = af::ApiType::kAPITypeBuffer;
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z1.id};
  *data0.y.repeats = {s1};
  *data0.y.strides = {One};

  af::ascir_op::Load load0("load0");
  graph.AddNode(load0);
  load0.x = data0.y;
  load0.attr.sched.axis = {z1.id};
  load0.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load0.y.dtype = ge::DT_FLOAT16;
  *load0.y.axis = {z1.id};
  *load0.y.repeats = {s1};
  *load0.y.strides = {One};
  *load0.y.vectorized_axis = {z1.id};

  af::ascir_op::Broadcast brc_missing_axis("brc_missing_axis");
  graph.AddNode(brc_missing_axis);
  brc_missing_axis.x = load0.y;
  brc_missing_axis.attr.sched.axis = {z0.id, z1.id};
  brc_missing_axis.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc_missing_axis.y.dtype = ge::DT_FLOAT16;
  *brc_missing_axis.y.axis = {z0.id, z1.id};
  *brc_missing_axis.y.repeats = {s0, s1};
  *brc_missing_axis.y.strides = {s1, One};
  *brc_missing_axis.y.vectorized_axis = {z1.id};

  af::ascir_op::Sum reduce_missing_axis("reduce_missing_axis");
  graph.AddNode(reduce_missing_axis);
  reduce_missing_axis.x = brc_missing_axis.y;
  reduce_missing_axis.attr.sched.axis = {z0.id, z1.id};
  reduce_missing_axis.attr.api.compute_type = af::ComputeType::kComputeReduce;
  reduce_missing_axis.y.dtype = ge::DT_FLOAT16;
  *reduce_missing_axis.y.axis = {z0.id, z1.id};
  *reduce_missing_axis.y.repeats = {One, s1};
  *reduce_missing_axis.y.strides = {Zero, One};
  *reduce_missing_axis.y.vectorized_axis = {z1.id};

  af::ascir_op::Store store0("store0");
  graph.AddNode(store0);
  store0.x = reduce_missing_axis.y;
  store0.attr.sched.axis = {z0.id, z1.id};
  store0.attr.api.compute_type = af::ComputeType::kComputeStore;
  store0.y.dtype = ge::DT_FLOAT16;
  *store0.y.axis = {z0.id, z1.id};
  *store0.y.repeats = {One, s1};
  *store0.y.strides = {Zero, One};

  af::ascir_op::Output y("y");
  y.ir_attr.SetIndex(0);
  y.x = store0.y;
  y.attr.sched.axis = {z0.id, z1.id};
  y.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  y.attr.api.type = af::ApiType::kAPITypeBuffer;
  y.y.dtype = ge::DT_FLOAT16;
  *y.y.axis = {z0.id, z1.id};
  *y.y.repeats = {One, s1};
  *y.y.strides = {Zero, One};
}

/**
 *                    store (s0,s1,s2,s3,s4)
 *                      |
 *                    add0 (s0,s1,s2,s3,s4)
 *                   /    \
 *                  |     exp0 (s0,s1,s2,s3,s4)
 *                   \    /
 *                    \  /
 *                   brc4 (s0,s1,s2,s3,s4)
 *                    |
 *                  brc3 (1,s1,s2,s3,s4)
 *                   |
 *                 brc2 (1,1,s2,s3,s4)
 *                  |
 *                brc1 (1,1,1,s3,s4)
 *                 |
 *               brc0 (1,1,1,1,s4)
 *                |
 *              abs0 (1,1,1,1,1)
 *               |
 *             load0 (1,1,1,1,1)
 *              |
 *            data0 (1,1,1,1,1)
 */
static void Construct_ContinuesBroadcastScalarFusion(af::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar("s3");
  auto s4 = graph.CreateSizeVar("s4");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);
  auto z4 = graph.CreateAxis("z4", s4);

  af::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  data0.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  data0.attr.api.type = af::ApiType::kAPITypeBuffer;
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *data0.y.repeats = {One, One, One, One, One};
  *data0.y.strides = {Zero, Zero, Zero, Zero, Zero};

  af::ascir_op::Load load0("load0");
  graph.AddNode(load0);
  load0.x = data0.y;
  load0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  load0.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load0.y.dtype = ge::DT_FLOAT16;
  *load0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *load0.y.repeats = {One, One, One, One, One};
  *load0.y.strides = {Zero, Zero, Zero, Zero, Zero};

  af::ascir_op::Abs abs0("abs0");
  graph.AddNode(abs0);
  abs0.x = load0.y;
  abs0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  abs0.attr.api.compute_type = af::ComputeType::kComputeElewise;
  abs0.y.dtype = ge::DT_FLOAT16;
  *abs0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *abs0.y.repeats = {One, One, One, One, One};
  *abs0.y.strides = {Zero, Zero, Zero, Zero, Zero};

  af::ascir_op::Broadcast brc0("brc0");
  graph.AddNode(brc0);
  brc0.x = abs0.y;
  brc0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  brc0.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc0.y.dtype = ge::DT_FLOAT16;
  *brc0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *brc0.y.repeats = {One, One, One, One, s4};
  *brc0.y.strides = {Zero, Zero, Zero, Zero, One};

  af::ascir_op::Broadcast brc1("brc1");
  graph.AddNode(brc1);
  brc1.x = brc0.y;
  brc1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  brc1.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc1.y.dtype = ge::DT_FLOAT16;
  *brc1.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *brc1.y.repeats = {One, One, One, s3, s4};
  *brc1.y.strides = {Zero, Zero, Zero, s4, One};

  af::ascir_op::Broadcast brc2("brc2");
  graph.AddNode(brc2);
  brc2.x = brc1.y;
  brc2.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  brc2.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc2.y.dtype = ge::DT_FLOAT16;
  *brc2.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *brc2.y.repeats = {One, One, s2, s3, s4};
  *brc2.y.strides = {Zero, Zero, s3 * s4, s4, One};

  af::ascir_op::Broadcast brc3("brc3");
  graph.AddNode(brc3);
  brc3.x = brc2.y;
  brc3.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  brc3.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc3.y.dtype = ge::DT_FLOAT16;
  *brc3.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *brc3.y.repeats = {One, s1, s2, s3, s4};
  *brc3.y.strides = {Zero, s2 * s3 * s4, s3 * s4, s4, One};

  af::ascir_op::Broadcast brc4("brc4");
  graph.AddNode(brc4);
  brc4.x = brc3.y;
  brc4.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  brc4.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc4.y.dtype = ge::DT_FLOAT16;
  *brc4.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *brc4.y.repeats = {s0, s1, s2, s3, s4};
  *brc4.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};

  af::ascir_op::Exp exp0("exp0");
  graph.AddNode(exp0);
  exp0.x = brc4.y;
  exp0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  exp0.attr.api.compute_type = af::ComputeType::kComputeElewise;
  exp0.y.dtype = ge::DT_FLOAT16;
  *exp0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *exp0.y.repeats = {s0, s1, s2, s3, s4};
  *exp0.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};

  af::ascir_op::Add add0("add0");
  graph.AddNode(add0);
  add0.x1 = exp0.y;
  add0.x2 = brc4.y;
  add0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  add0.attr.api.compute_type = af::ComputeType::kComputeElewise;
  add0.y.dtype = ge::DT_FLOAT16;
  *add0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *add0.y.repeats = {s0, s1, s2, s3, s4};
  *add0.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};

  af::ascir_op::Store store("store");
  graph.AddNode(store);
  store.x = add0.y;
  store.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  store.attr.api.compute_type = af::ComputeType::kComputeStore;
  store.y.dtype = ge::DT_FLOAT16;
  *store.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *store.y.repeats = {s0, s1, s2, s3, s4};
  *store.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};

  af::ascir_op::Output y("y");
  y.ir_attr.SetIndex(0);
  y.x = store.y;
  y.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  y.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  y.attr.api.type = af::ApiType::kAPITypeBuffer;
  y.y.dtype = ge::DT_FLOAT16;
  *y.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *y.y.repeats = {s0, s1, s2, s3, s4};
  *y.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};
}

/**
 *                    store0
 *                      |
 *                    add0
 *                   /    \
 *                  |     exp0 (s0,s1,s2,s3,s4)
 *                   \    /
 *                    \  /
 *                   brc4 (s0,s1,s2,s3,s4)
 *                    |
 *                  brc3 (1,s1,s2,s3,s4)
 *                   |
 *         store1  brc1 (1,1,s2,s3,s4)
 *           \      |
 *          exp1  brc0 (1,1,s2,1,s4)
 *            \   /
 *            abs0 (1,1,s2,1,1)
 *             |
 *           load0 (1,1,s2,1,1)
 *            |
 *          data0 (1,1,s2,1,1)
 */
static void Construct_ContinuesBroadcastPartBrcFusion(af::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar("s3");
  auto s4 = graph.CreateSizeVar("s4");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);
  auto z4 = graph.CreateAxis("z4", s4);

  af::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  data0.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  data0.attr.api.type = af::ApiType::kAPITypeBuffer;
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *data0.y.repeats = {One, One, s2, One, One};
  *data0.y.strides = {Zero, Zero, One, Zero, Zero};

  af::ascir_op::Load load0("load0");
  graph.AddNode(load0);
  load0.x = data0.y;
  load0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  load0.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load0.y.dtype = ge::DT_FLOAT16;
  *load0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *load0.y.repeats = {One, One, s2, One, One};
  *load0.y.strides = {Zero, Zero, One, Zero, Zero};

  af::ascir_op::Abs abs0("abs0");
  graph.AddNode(abs0);
  abs0.x = load0.y;
  abs0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  abs0.attr.api.compute_type = af::ComputeType::kComputeElewise;
  abs0.y.dtype = ge::DT_FLOAT16;
  *abs0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *abs0.y.repeats = {One, One, s2, One, One};
  *abs0.y.strides = {Zero, Zero, One, Zero, Zero};

  af::ascir_op::Broadcast brc0("brc0");
  graph.AddNode(brc0);
  brc0.x = abs0.y;
  brc0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  brc0.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc0.y.dtype = ge::DT_FLOAT16;
  *brc0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *brc0.y.repeats = {One, One, s2, One, s4};
  *brc0.y.strides = {Zero, Zero, s4, Zero, One};

  af::ascir_op::Broadcast brc1("brc1");
  graph.AddNode(brc1);
  brc1.x = brc0.y;
  brc1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  brc1.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc1.y.dtype = ge::DT_FLOAT16;
  *brc1.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *brc1.y.repeats = {One, One, s2, s3, s4};
  *brc1.y.strides = {Zero, Zero, s3 * s4, s4, One};

  af::ascir_op::Broadcast brc3("brc3");
  graph.AddNode(brc3);
  brc3.x = brc1.y;
  brc3.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  brc3.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc3.y.dtype = ge::DT_FLOAT16;
  *brc3.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *brc3.y.repeats = {One, s1, s2, s3, s4};
  *brc3.y.strides = {Zero, s2 * s3 * s4, s3 * s4, s4, One};

  af::ascir_op::Broadcast brc4("brc4");
  graph.AddNode(brc4);
  brc4.x = brc3.y;
  brc4.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  brc4.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc4.y.dtype = ge::DT_FLOAT16;
  *brc4.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *brc4.y.repeats = {s0, s1, s2, s3, s4};
  *brc4.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};

  af::ascir_op::Exp exp0("exp0");
  graph.AddNode(exp0);
  exp0.x = brc4.y;
  exp0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  exp0.attr.api.compute_type = af::ComputeType::kComputeElewise;
  exp0.y.dtype = ge::DT_FLOAT16;
  *exp0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *exp0.y.repeats = {s0, s1, s2, s3, s4};
  *exp0.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};

  af::ascir_op::Add add0("add0");
  graph.AddNode(add0);
  add0.x1 = exp0.y;
  add0.x2 = brc4.y;
  add0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  add0.attr.api.compute_type = af::ComputeType::kComputeElewise;
  add0.y.dtype = ge::DT_FLOAT16;
  *add0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *add0.y.repeats = {s0, s1, s2, s3, s4};
  *add0.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};

  af::ascir_op::Store store0("store0");
  graph.AddNode(store0);
  store0.x = add0.y;
  store0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  store0.attr.api.compute_type = af::ComputeType::kComputeStore;
  store0.y.dtype = ge::DT_FLOAT16;
  *store0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *store0.y.repeats = {s0, s1, s2, s3, s4};
  *store0.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};

  af::ascir_op::Output y("y");
  y.ir_attr.SetIndex(0);
  y.x = store0.y;
  y.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  y.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  y.attr.api.type = af::ApiType::kAPITypeBuffer;
  y.y.dtype = ge::DT_FLOAT16;
  *y.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *y.y.repeats = {s0, s1, s2, s3, s4};
  *y.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};

  af::ascir_op::Exp exp1("exp1");
  graph.AddNode(exp1);
  exp1.x = abs0.y;
  exp1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  exp1.attr.api.compute_type = af::ComputeType::kComputeElewise;
  exp1.y.dtype = ge::DT_FLOAT16;
  *exp1.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *exp1.y.repeats = {One, One, s2, One, One};
  *exp1.y.strides = {Zero, Zero, One, Zero, Zero};

  af::ascir_op::Store store1("store1");
  graph.AddNode(store1);
  store1.x = exp1.y;
  store1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  store1.attr.api.compute_type = af::ComputeType::kComputeStore;
  store1.y.dtype = ge::DT_FLOAT16;
  *store1.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *store1.y.repeats = {One, One, s2, One, One};
  *store1.y.strides = {Zero, Zero, One, Zero, Zero};

  af::ascir_op::Output y1("y1");
  y1.ir_attr.SetIndex(1);
  y1.x = store1.y;
  y1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  y1.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  y1.attr.api.type = af::ApiType::kAPITypeBuffer;
  y1.y.dtype = ge::DT_FLOAT16;
  *y1.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *y1.y.repeats = {One, One, s2, One, One};
  *y1.y.strides = {Zero, Zero, One, Zero, Zero};
}

/**
 *              store0
 *                |
 *              brc1 (s1,s2,s3,s4)
 *               |
 *             brc0 (1,s2,s3,s4)
 *              |
 *            abs0 (1,s2,1,s4)
 *             |
 *           load0 (1,s2,1,s4)
 *            |
 *          data0 (1,s2,1,s4)
 */
static void Construct_ContinuesBroadcastIntervalBrcBABAFusion(af::AscGraph &graph) {
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar("s3");
  auto s4 = graph.CreateSizeVar("s4");

  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);
  auto z4 = graph.CreateAxis("z4", s4);

  af::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.attr.sched.axis = {z1.id, z2.id, z3.id, z4.id};
  data0.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  data0.attr.api.type = af::ApiType::kAPITypeBuffer;
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z1.id, z2.id, z3.id, z4.id};
  *data0.y.repeats = {One, s2, One, s4};
  *data0.y.strides = {Zero, s4, Zero, One};

  af::ascir_op::Load load0("load0");
  graph.AddNode(load0);
  load0.x = data0.y;
  load0.attr.sched.axis = {z1.id, z2.id, z3.id, z4.id};
  load0.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load0.y.dtype = ge::DT_FLOAT16;
  *load0.y.axis = {z1.id, z2.id, z3.id, z4.id};
  *load0.y.repeats = {One, s2, One, s4};
  *load0.y.strides = {Zero, s4, Zero, One};

  af::ascir_op::Abs abs0("abs0");
  graph.AddNode(abs0);
  abs0.x = load0.y;
  abs0.attr.sched.axis = {z1.id, z2.id, z3.id, z4.id};
  abs0.attr.api.compute_type = af::ComputeType::kComputeElewise;
  abs0.y.dtype = ge::DT_FLOAT16;
  *abs0.y.axis = {z1.id, z2.id, z3.id, z4.id};
  *abs0.y.repeats = {One, s2, One, s4};
  *abs0.y.strides = {Zero, s4, Zero, One};

  af::ascir_op::Broadcast brc0("brc0");
  graph.AddNode(brc0);
  brc0.x = abs0.y;
  brc0.attr.sched.axis = {z1.id, z2.id, z3.id, z4.id};
  brc0.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc0.y.dtype = ge::DT_FLOAT16;
  *brc0.y.axis = {z1.id, z2.id, z3.id, z4.id};
  *brc0.y.repeats = {One, s2, s3, s4};
  *brc0.y.strides = {Zero, s3 * s4, One, One};

  af::ascir_op::Broadcast brc1("brc1");
  graph.AddNode(brc1);
  brc1.x = brc0.y;
  brc1.attr.sched.axis = {z1.id, z2.id, z3.id, z4.id};
  brc1.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc1.y.dtype = ge::DT_FLOAT16;
  *brc1.y.axis = {z1.id, z2.id, z3.id, z4.id};
  *brc1.y.repeats = {s1, s2, s3, s4};
  *brc1.y.strides = {s2 * s3 * s4, s3 * s4, s4, One};

  af::ascir_op::Store store0("store0");
  graph.AddNode(store0);
  store0.x = brc1.y;
  store0.attr.sched.axis = {z1.id, z2.id, z3.id, z4.id};
  store0.attr.api.compute_type = af::ComputeType::kComputeStore;
  store0.y.dtype = ge::DT_FLOAT16;
  *store0.y.axis = {z1.id, z2.id, z3.id, z4.id};
  *store0.y.repeats = {s1, s2, s3, s4};
  *store0.y.strides = {s2 * s3 * s4, s3 * s4, s4, One};

  af::ascir_op::Output y("y");
  y.ir_attr.SetIndex(0);
  y.x = store0.y;
  y.attr.sched.axis = {z1.id, z2.id, z3.id, z4.id};
  y.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  y.attr.api.type = af::ApiType::kAPITypeBuffer;
  y.y.dtype = ge::DT_FLOAT16;
  *y.y.axis = {z1.id, z2.id, z3.id, z4.id};
  *y.y.repeats = {s1, s2, s3, s4};
  *y.y.strides = {s2 * s3 * s4, s3 * s4, s4, One};
}

/**
 *              store0
 *                |
 *              brc1 (s1,s2,s3,s4)
 *               |
 *             brc0 (s1,1,s3,s4)
 *              |
 *            abs0 (s1,1,s3,1)
 *             |
 *           load0 (s1,1,s3,1)
 *            |
 *          data0 (s1,1,s3,1)
 */
static void Construct_ContinuesBroadcastIntervalBrcABABFusion(af::AscGraph &graph) {
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar("s3");
  auto s4 = graph.CreateSizeVar("s4");

  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);
  auto z4 = graph.CreateAxis("z4", s4);

  af::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.attr.sched.axis = {z1.id, z2.id, z3.id, z4.id};
  data0.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  data0.attr.api.type = af::ApiType::kAPITypeBuffer;
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z1.id, z2.id, z3.id, z4.id};
  *data0.y.repeats = {s1, One, s2, One};
  *data0.y.strides = {s2, Zero, One, Zero};

  af::ascir_op::Load load0("load0");
  load0.x = data0.y;
  load0.attr.sched.axis = {z1.id, z2.id, z3.id, z4.id};
  load0.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load0.y.dtype = ge::DT_FLOAT16;
  *load0.y.axis = {z1.id, z2.id, z3.id, z4.id};
  *load0.y.repeats = {s1, One, s2, One};
  *load0.y.strides = {s2, Zero, One, Zero};

  af::ascir_op::Abs abs0("abs0");
  abs0.x = load0.y;
  abs0.attr.sched.axis = {z1.id, z2.id, z3.id, z4.id};
  abs0.attr.api.compute_type = af::ComputeType::kComputeElewise;
  abs0.y.dtype = ge::DT_FLOAT16;
  *abs0.y.axis = {z1.id, z2.id, z3.id, z4.id};
  *abs0.y.repeats = {s1, One, s2, One};
  *abs0.y.strides = {s2, Zero, One, Zero};

  af::ascir_op::Broadcast brc0("brc0");
  brc0.x = abs0.y;
  brc0.attr.sched.axis = {z1.id, z2.id, z3.id, z4.id};
  brc0.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc0.y.dtype = ge::DT_FLOAT16;
  *brc0.y.axis = {z1.id, z2.id, z3.id, z4.id};
  *brc0.y.repeats = {s1, One, s2, s4};
  *brc0.y.strides = {s2 * s4, Zero, s4, One};

  af::ascir_op::Broadcast brc1("brc1");
  brc1.x = brc0.y;
  brc1.attr.sched.axis = {z1.id, z2.id, z3.id, z4.id};
  brc1.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc1.y.dtype = ge::DT_FLOAT16;
  *brc1.y.axis = {z1.id, z2.id, z3.id, z4.id};
  *brc1.y.repeats = {s1, s2, s3, s4};
  *brc1.y.strides = {s2 * s3 * s4, s3 * s4, s4, One};

  af::ascir_op::Store store0("store0");
  store0.x = brc1.y;
  store0.attr.sched.axis = {z1.id, z2.id, z3.id, z4.id};
  store0.attr.api.compute_type = af::ComputeType::kComputeStore;
  store0.y.dtype = ge::DT_FLOAT16;
  *store0.y.axis = {z1.id, z2.id, z3.id, z4.id};
  *store0.y.repeats = {s1, s2, s3, s4};
  *store0.y.strides = {s2 * s3 * s4, s3 * s4, s4, One};

  af::ascir_op::Output y("y");
  y.ir_attr.SetIndex(0);
  y.x = store0.y;
  y.attr.sched.axis = {z1.id, z2.id, z3.id, z4.id};
  y.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  y.attr.api.type = af::ApiType::kAPITypeBuffer;
  y.y.dtype = ge::DT_FLOAT16;
  *y.y.axis = {z1.id, z2.id, z3.id, z4.id};
  *y.y.repeats = {s1, s2, s3, s4};
  *y.y.strides = {s2 * s3 * s4, s3 * s4, s4, One};
}

/**
 *               store0
 *                 |
 *             reduce_sum (s0,s1,1,s3,s4)
 *                |
 *              brc1 (s0,s1,s2,s3,s4)
 *               |
 *             brc0 (s0,1,s2,s3,s4)
 *             |
 *           load0 (s0,1,s2,s3,1)
 *            |
 *          data0 (s0,1,s2,s3,1)
 */
static void Construct_ContinuesBroadcastIntervalBrcBAABFusion(af::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar("s3");
  auto s4 = graph.CreateSizeVar("s4");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);
  auto z4 = graph.CreateAxis("z4", s4);

  af::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  data0.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  data0.attr.api.type = af::ApiType::kAPITypeBuffer;
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *data0.y.repeats = {s0, One, s2, s3, One};
  *data0.y.strides = {s2 * s3, Zero, s3, One, Zero};

  af::ascir_op::Load load0("load0");
  load0.x = data0.y;
  load0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  load0.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load0.y.dtype = ge::DT_FLOAT16;
  *load0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *load0.y.repeats = {s0, One, s2, s3, One};
  *load0.y.strides = {s2 * s3, Zero, s3, One, Zero};

  af::ascir_op::Broadcast brc0("brc0");
  brc0.x = load0.y;
  brc0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  brc0.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc0.y.dtype = ge::DT_FLOAT16;
  *brc0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *brc0.y.repeats = {s0, One, s2, s3, s4};
  *brc0.y.strides = {s2 * s3 * s4, Zero, s3 * s4, s4, One};

  af::ascir_op::Broadcast brc1("brc1");
  brc1.x = brc0.y;
  brc1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  brc1.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc1.y.dtype = ge::DT_FLOAT16;
  *brc1.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *brc1.y.repeats = {s0, s1, s2, s3, s4};
  *brc1.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};

  af::ascir_op::Sum sum("reduce_sum");
  sum.x = brc1.y;
  sum.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  sum.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  sum.y.dtype = ge::DT_FLOAT16;
  *sum.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *sum.y.repeats = {s0, s1, One, s3, s4};
  *sum.y.strides = {s1 * s3 * s4, s3 * s4, Zero, s4, One};

  af::ascir_op::Store store0("store0");
  store0.x = brc1.y;
  store0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  store0.attr.api.compute_type = af::ComputeType::kComputeStore;
  store0.y.dtype = ge::DT_FLOAT16;
  *store0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *store0.y.repeats = {s0, s1, One, s3, s4};
  *store0.y.strides = {s1 * s3 * s4, s3 * s4, Zero, s4, One};

  af::ascir_op::Output y("y");
  y.ir_attr.SetIndex(0);
  y.x = store0.y;
  y.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  y.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  y.attr.api.type = af::ApiType::kAPITypeBuffer;
  y.y.dtype = ge::DT_FLOAT16;
  *y.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *y.y.repeats = {s0, s1, One, s3, s4};
  *y.y.strides = {s1 * s3 * s4, s3 * s4, Zero, s4, One};
}

/**
 *                    store0
 *                      |
 *                    add0
 *                   /    \
 *                  |     exp0 (s0,s1,s2,s3,s4)
 *                  \    /
 *                  \  /
 *                  brc2 (s0,s1,s2,s3,s4)
 *                   |
 *         store1  brc1 (s0,1,s2,s3,s4)
 *           \      |
 *          exp1  brc0 (s0,1,1,s3,s4)
 *            \   /
 *            abs0 (s0,1,1,1,s4)
 *             |
 *           load0 (s0,1,1,1,s4)
 *            |
 *          data0 (s0,1,1,1,s4)
 */
static void Construct_ContinuesBroadcastPartBrcInMiddleFusion(af::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar("s3");
  auto s4 = graph.CreateSizeVar("s4");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);
  auto z4 = graph.CreateAxis("z4", s4);

  af::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  data0.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  data0.attr.api.type = af::ApiType::kAPITypeBuffer;
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *data0.y.repeats = {s0, One, One, One, s4};
  *data0.y.strides = {s4, Zero, Zero, Zero, One};

  af::ascir_op::Load load0("load0");
  graph.AddNode(load0);
  load0.x = data0.y;
  load0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  load0.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load0.y.dtype = ge::DT_FLOAT16;
  *load0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *load0.y.repeats = {s0, One, One, One, s4};
  *load0.y.strides = {s4, Zero, Zero, Zero, One};

  af::ascir_op::Abs abs0("abs0");
  graph.AddNode(abs0);
  abs0.x = load0.y;
  abs0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  abs0.attr.api.compute_type = af::ComputeType::kComputeElewise;
  abs0.y.dtype = ge::DT_FLOAT16;
  *abs0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *abs0.y.repeats = {s0, One, One, One, s4};
  *abs0.y.strides = {s4, Zero, Zero, Zero, One};

  af::ascir_op::Broadcast brc0("brc0");
  graph.AddNode(brc0);
  brc0.x = abs0.y;
  brc0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  brc0.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc0.y.dtype = ge::DT_FLOAT16;
  *brc0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *brc0.y.repeats = {s0, One, One, s3, s4};
  *brc0.y.strides = {s3 * s4, Zero, Zero, s4, One};

  af::ascir_op::Broadcast brc1("brc1");
  graph.AddNode(brc1);
  brc1.x = brc0.y;
  brc1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  brc1.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc1.y.dtype = ge::DT_FLOAT16;
  *brc1.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *brc1.y.repeats = {s0, One, s2, s3, s4};
  *brc1.y.strides = {s2 * s3 * s4, Zero, s3 * s4, s4, One};

  af::ascir_op::Broadcast brc2("brc2");
  graph.AddNode(brc2);
  brc2.x = brc1.y;
  brc2.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  brc2.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc2.y.dtype = ge::DT_FLOAT16;
  *brc2.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *brc2.y.repeats = {s0, s1, s2, s3, s4};
  *brc2.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};

  af::ascir_op::Exp exp0("exp0");
  graph.AddNode(exp0);
  exp0.x = brc2.y;
  exp0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  exp0.attr.api.compute_type = af::ComputeType::kComputeElewise;
  exp0.y.dtype = ge::DT_FLOAT16;
  *exp0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *exp0.y.repeats = {s0, s1, s2, s3, s4};
  *exp0.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};

  af::ascir_op::Add add0("add0");
  graph.AddNode(add0);
  add0.x1 = exp0.y;
  add0.x2 = brc2.y;
  add0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  add0.attr.api.compute_type = af::ComputeType::kComputeElewise;
  add0.y.dtype = ge::DT_FLOAT16;
  *add0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *add0.y.repeats = {s0, s1, s2, s3, s4};
  *add0.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};

  af::ascir_op::Store store0("store0");
  graph.AddNode(store0);
  store0.x = add0.y;
  store0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  store0.attr.api.compute_type = af::ComputeType::kComputeStore;
  store0.y.dtype = ge::DT_FLOAT16;
  *store0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *store0.y.repeats = {s0, s1, s2, s3, s4};
  *store0.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};

  af::ascir_op::Output y("y");
  y.ir_attr.SetIndex(0);
  y.x = store0.y;
  y.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  y.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  y.attr.api.type = af::ApiType::kAPITypeBuffer;
  y.y.dtype = ge::DT_FLOAT16;
  *y.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *y.y.repeats = {s0, s1, s2, s3, s4};
  *y.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};

  af::ascir_op::Exp exp1("exp1");
  graph.AddNode(exp1);
  exp1.x = abs0.y;
  exp1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  exp1.attr.api.compute_type = af::ComputeType::kComputeElewise;
  exp1.y.dtype = ge::DT_FLOAT16;
  *exp1.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *exp1.y.repeats = {One, One, s2, One, One};
  *exp1.y.strides = {Zero, Zero, One, Zero, Zero};

  af::ascir_op::Store store1("store1");
  graph.AddNode(store1);
  store1.x = exp1.y;
  store1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  store1.attr.api.compute_type = af::ComputeType::kComputeStore;
  store1.y.dtype = ge::DT_FLOAT16;
  *store1.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *store1.y.repeats = {One, One, s2, One, One};
  *store1.y.strides = {Zero, Zero, One, Zero, Zero};

  af::ascir_op::Output y1("y1");
  y1.ir_attr.SetIndex(1);
  y1.x = store1.y;
  y1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  y1.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  y1.attr.api.type = af::ApiType::kAPITypeBuffer;
  y1.y.dtype = ge::DT_FLOAT16;
  *y1.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *y1.y.repeats = {One, One, s2, One, One};
  *y1.y.strides = {Zero, Zero, One, Zero, Zero};
}

/**
 *         store1  store0
 *           \      /
 *         abs0  brc1 (1,1,s2,s3,s4)
 *            \  /
 *            brc0 (1,1,s2,1,s4)
 *             |
 *           load0 (1,1,s2,1,1)
 *            |
 *          data0 (1,1,s2,1,1)
 */
static void Construct_ContinuesBroadcastMultiOutFusion(af::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar("s3");
  auto s4 = graph.CreateSizeVar("s4");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);
  auto z4 = graph.CreateAxis("z4", s4);

  af::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  data0.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  data0.attr.api.type = af::ApiType::kAPITypeBuffer;
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *data0.y.repeats = {One, One, s2, One, One};
  *data0.y.strides = {Zero, Zero, One, Zero, Zero};

  af::ascir_op::Load load0("load0");
  graph.AddNode(load0);
  load0.x = data0.y;
  load0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  load0.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load0.y.dtype = ge::DT_FLOAT16;
  *load0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *load0.y.repeats = {One, One, s2, One, One};
  *load0.y.strides = {Zero, Zero, One, Zero, Zero};

  af::ascir_op::Broadcast brc0("brc0");
  graph.AddNode(brc0);
  brc0.x = load0.y;
  brc0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  brc0.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc0.y.dtype = ge::DT_FLOAT16;
  *brc0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *brc0.y.repeats = {One, One, s2, One, s4};
  *brc0.y.strides = {Zero, Zero, s4, Zero, One};

  af::ascir_op::Broadcast brc1("brc1");
  graph.AddNode(brc1);
  brc1.x = brc0.y;
  brc1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  brc1.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc1.y.dtype = ge::DT_FLOAT16;
  *brc1.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *brc1.y.repeats = {One, One, s2, s3, s4};
  *brc1.y.strides = {Zero, Zero, s3 * s4, s4, One};

  af::ascir_op::Store store0("store0");
  graph.AddNode(store0);
  store0.x = brc1.y;
  store0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  store0.attr.api.compute_type = af::ComputeType::kComputeStore;
  store0.y.dtype = ge::DT_FLOAT16;
  *store0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *store0.y.repeats = {One, One, s2, s3, s4};
  *store0.y.strides = {Zero, Zero, s3 * s4, s4, One};

  af::ascir_op::Output y("y");
  y.ir_attr.SetIndex(0);
  y.x = store0.y;
  y.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  y.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  y.attr.api.type = af::ApiType::kAPITypeBuffer;
  y.y.dtype = ge::DT_FLOAT16;
  *y.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *y.y.repeats = {One, One, s2, s3, s4};
  *y.y.strides = {Zero, Zero, s3 * s4, s4, One};

  af::ascir_op::Abs abs0("abs0");
  graph.AddNode(abs0);
  abs0.x = brc0.y;
  abs0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  abs0.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  abs0.y.dtype = ge::DT_FLOAT16;
  *abs0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *abs0.y.repeats = {One, One, s2, One, s4};
  *abs0.y.strides = {Zero, Zero, s4, Zero, One};

  af::ascir_op::Store store1("store1");
  graph.AddNode(store1);
  store1.x = abs0.y;
  store1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  store1.attr.api.compute_type = af::ComputeType::kComputeStore;
  store1.y.dtype = ge::DT_FLOAT16;
  *store1.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *store1.y.repeats = {One, One, s2, One, s4};
  *store1.y.strides = {Zero, Zero, s4, Zero, One};

  af::ascir_op::Output y1("y1");
  y1.ir_attr.SetIndex(1);
  y1.x = store1.y;
  y1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  y1.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  y1.attr.api.type = af::ApiType::kAPITypeBuffer;
  y1.y.dtype = ge::DT_FLOAT16;
  *y1.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *y1.y.repeats = {One, One, s2, One, s4};
  *y1.y.strides = {Zero, Zero, s4, Zero, One};
}

/**
 *                    store0
 *                      |
 *                   reduce0 (s0,s1,s2,s3,1)
 *                     |
 *                   brc4 (s0,s1,s2,s3,s4)
 *                    |
 *                  brc3 (1,s1,s2,s3,s4)
 *                   |
 *         store1  brc1 (1,1,s2,s3,s4)
 *           \      |
 *          exp1  brc0 (1,1,s2,1,s4)
 *            \   /
 *            abs0 (1,1,s2,1,1)
 *             |
 *           load0 (1,1,s2,1,1)
 *            |
 *          data0 (1,1,s2,1,1)
 */
static void Construct_ContinuesBroadcastWithReduceFusion(af::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar("s3");
  auto s4 = graph.CreateSizeVar("s4");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);
  auto z4 = graph.CreateAxis("z4", s4);

  af::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  data0.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  data0.attr.api.type = af::ApiType::kAPITypeBuffer;
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *data0.y.repeats = {One, One, s2, One, One};
  *data0.y.strides = {Zero, Zero, One, Zero, Zero};

  af::ascir_op::Load load0("load0");
  graph.AddNode(load0);
  load0.x = data0.y;
  load0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  load0.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load0.y.dtype = ge::DT_FLOAT16;
  *load0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *load0.y.repeats = {One, One, s2, One, One};
  *load0.y.strides = {Zero, Zero, One, Zero, Zero};

  af::ascir_op::Abs abs0("abs0");
  graph.AddNode(abs0);
  abs0.x = load0.y;
  abs0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  abs0.attr.api.compute_type = af::ComputeType::kComputeElewise;
  abs0.y.dtype = ge::DT_FLOAT16;
  *abs0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *abs0.y.repeats = {One, One, s2, One, One};
  *abs0.y.strides = {Zero, Zero, One, Zero, Zero};

  af::ascir_op::Broadcast brc0("brc0");
  graph.AddNode(brc0);
  brc0.x = abs0.y;
  brc0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  brc0.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc0.y.dtype = ge::DT_FLOAT16;
  *brc0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *brc0.y.repeats = {One, One, s2, One, s4};
  *brc0.y.strides = {Zero, Zero, s4, Zero, One};

  af::ascir_op::Broadcast brc1("brc1");
  graph.AddNode(brc1);
  brc1.x = brc0.y;
  brc1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  brc1.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc1.y.dtype = ge::DT_FLOAT16;
  *brc1.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *brc1.y.repeats = {One, One, s2, s3, s4};
  *brc1.y.strides = {Zero, Zero, s3 * s4, s4, One};

  af::ascir_op::Broadcast brc3("brc3");
  graph.AddNode(brc3);
  brc3.x = brc1.y;
  brc3.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  brc3.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc3.y.dtype = ge::DT_FLOAT16;
  *brc3.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *brc3.y.repeats = {One, s1, s2, s3, s4};
  *brc3.y.strides = {Zero, s2 * s3 * s4, s3 * s4, s4, One};

  af::ascir_op::Broadcast brc4("brc4");
  graph.AddNode(brc4);
  brc4.x = brc3.y;
  brc4.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  brc4.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc4.y.dtype = ge::DT_FLOAT16;
  *brc4.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *brc4.y.repeats = {s0, s1, s2, s3, s4};
  *brc4.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};

  af::ascir_op::Sum reduce_sum("reduce_sum");
  graph.AddNode(reduce_sum);
  reduce_sum.x = brc4.y;
  reduce_sum.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  reduce_sum.attr.api.compute_type = af::ComputeType::kComputeReduce;
  reduce_sum.y.dtype = ge::DT_FLOAT16;
  *reduce_sum.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *reduce_sum.y.repeats = {s0, s1, s2, s3, One};
  *reduce_sum.y.strides = {s1 * s2 * s3, s2 * s3, s3, One, Zero};

  af::ascir_op::Store store0("store0");
  graph.AddNode(store0);
  store0.x = reduce_sum.y;
  store0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  store0.attr.api.compute_type = af::ComputeType::kComputeStore;
  store0.y.dtype = ge::DT_FLOAT16;
  *store0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *store0.y.repeats = {s0, s1, s2, s3, One};
  *store0.y.strides = {s1 * s2 * s3, s2 * s3, s3, One, Zero};

  af::ascir_op::Output y("y");
  y.ir_attr.SetIndex(0);
  y.x = store0.y;
  y.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  y.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  y.attr.api.type = af::ApiType::kAPITypeBuffer;
  y.y.dtype = ge::DT_FLOAT16;
  *y.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *y.y.repeats = {s0, s1, s2, s3, One};
  *y.y.strides = {s1 * s2 * s3, s2 * s3, s3, One, Zero};

  af::ascir_op::Exp exp1("exp1");
  graph.AddNode(exp1);
  exp1.x = abs0.y;
  exp1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  exp1.attr.api.compute_type = af::ComputeType::kComputeElewise;
  exp1.y.dtype = ge::DT_FLOAT16;
  *exp1.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *exp1.y.repeats = {One, One, s2, One, One};
  *exp1.y.strides = {Zero, Zero, One, Zero, Zero};

  af::ascir_op::Store store1("store1");
  graph.AddNode(store1);
  store1.x = exp1.y;
  store1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  store1.attr.api.compute_type = af::ComputeType::kComputeStore;
  store1.y.dtype = ge::DT_FLOAT16;
  *store1.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *store1.y.repeats = {One, One, s2, One, One};
  *store1.y.strides = {Zero, Zero, One, Zero, Zero};

  af::ascir_op::Output y1("y1");
  y1.ir_attr.SetIndex(1);
  y1.x = store1.y;
  y1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  y1.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  y1.attr.api.type = af::ApiType::kAPITypeBuffer;
  y1.y.dtype = ge::DT_FLOAT16;
  *y1.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *y1.y.repeats = {One, One, s2, One, One};
  *y1.y.strides = {Zero, Zero, One, Zero, Zero};
}

void Construct_Softmax(af::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar("s3");

  auto z0 = graph.CreateAxis("z0", s0 * s1 * s2);
  auto z1 = graph.CreateAxis("z1", s3);

  auto axis = {z0.id, z1.id};

  Data arg4_1("arg4_1", graph);
  //   graph.AddNode(arg4_1);
  arg4_1.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  arg4_1.attr.api.type = af::ApiType::kAPITypeBuffer;
  arg4_1.y.dtype = ge::DT_FLOAT16;

  Load b0_load("b0_load");
  //   graph.AddNode(b0_load);
  b0_load.x = arg4_1.y;
  b0_load.attr.sched.axis = axis;
  b0_load.attr.api.compute_type = af::ComputeType::kComputeLoad;
  b0_load.y.dtype = ge::DT_FLOAT16;
  *b0_load.y.axis = axis;
  *b0_load.y.repeats = {s0 * s1 * s2, s3};
  *b0_load.y.strides = {s3, One};

  af::ascir_op::Max b0_max("b0_max");
  //   graph.AddNode(b0_max);
  b0_max.x = b0_load.y;
  b0_max.attr.sched.axis = axis;
  b0_max.attr.api.compute_type = af::ComputeType::kComputeReduce;
  b0_max.y.dtype = ge::DT_FLOAT16;
  *b0_max.y.axis = axis;
  *b0_max.y.repeats = {s0 * s1 * s2, s3};
  *b0_max.y.strides = {One, Zero};

  Output buf1("buf1");
  //   graph.AddNode(buf1);
  Load b1_load("b1_load");
  //   graph.AddNode(b1_load);
  b1_load.x = arg4_1.y;
  b1_load.attr.sched.axis = axis;
  b1_load.attr.api.compute_type = af::ComputeType::kComputeLoad;
  b1_load.y.dtype = ge::DT_FLOAT16;
  *b1_load.y.axis = axis;
  *b1_load.y.repeats = {s0 * s1 * s2, s3};
  *b1_load.y.strides = {s3, One};

  Broadcast b1_broadcast("b1_broadcast");
  //   graph.AddNode(b1_broadcast);
  b1_broadcast.x = b0_max.y;
  b1_broadcast.attr.sched.axis = axis;
  b1_broadcast.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  b1_broadcast.y.dtype = ge::DT_FLOAT16;
  *b1_broadcast.y.axis = axis;
  *b1_broadcast.y.repeats = {s0 * s1 * s2, s3};
  *b1_broadcast.y.strides = {s3, One};

  af::ascir_op::Sub b1_sub("b1_sub");
  //   graph.AddNode(b1_sub);
  b1_sub.x1 = b1_load.y;
  b1_sub.x2 = b1_broadcast.y;
  b1_sub.attr.sched.axis = axis;
  b1_sub.attr.api.compute_type = af::ComputeType::kComputeElewise;
  b1_sub.y.dtype = ge::DT_FLOAT16;
  *b1_sub.y.axis = axis;
  *b1_sub.y.repeats = {s0 * s1 * s2, s3};
  *b1_sub.y.strides = {s3, One};

  Exp b1_exp("b1_exp");
  //   graph.AddNode(b1_exp);
  b1_exp.x = b1_sub.y;
  b1_exp.attr.sched.axis = axis;
  b1_exp.attr.api.compute_type = af::ComputeType::kComputeElewise;
  b1_exp.y.dtype = ge::DT_FLOAT16;
  *b1_exp.y.axis = axis;
  *b1_exp.y.repeats = {s0 * s1 * s2, s3};
  *b1_exp.y.strides = {s3, One};

  Store b1_store("b1_store");
  //   graph.AddNode(b1_store);
  b1_store.x = b1_exp.y;
  b1_store.attr.sched.axis = axis;
  b1_store.attr.api.compute_type = af::ComputeType::kComputeStore;
  b1_store.y.dtype = ge::DT_FLOAT16;
  *b1_store.y.axis = axis;
  *b1_store.y.repeats = {s0 * s1 * s2, s3};
  *b1_store.y.strides = {s3, One};

  buf1.x = b1_store.y;
  buf1.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  buf1.attr.api.type = af::ApiType::kAPITypeBuffer;
  buf1.y.dtype = ge::DT_FLOAT16;

  Load b2_load("b2_load");
  //   graph.AddNode(b2_load);
  b2_load.x = buf1.y;
  b2_load.attr.sched.axis = axis;
  b2_load.attr.api.compute_type = af::ComputeType::kComputeLoad;
  b2_load.y.dtype = ge::DT_FLOAT16;
  *b2_load.y.axis = axis;
  *b2_load.y.repeats = {s0 * s1 * s2, s3};
  *b2_load.y.strides = {s3, One};

  Sum b2_sum("b2_sum");
  //   graph.AddNode(b2_sum);
  b2_sum.x = b2_load.y;
  b2_sum.attr.sched.axis = axis;
  b2_sum.attr.api.compute_type = af::ComputeType::kComputeReduce;
  b2_sum.y.dtype = ge::DT_FLOAT16;
  *b2_sum.y.axis = axis;
  *b2_sum.y.repeats = {s0 * s1 * s2, s3};
  *b2_sum.y.strides = {One, Zero};

  Output buf3("buf3");
  //   graph.AddNode(buf3);
  Load b3_load("b3_load");
  //   graph.AddNode(b3_load);
  b3_load.x = buf1.y;
  b3_load.attr.sched.axis = axis;
  b3_load.attr.api.compute_type = af::ComputeType::kComputeLoad;
  b3_load.y.dtype = ge::DT_FLOAT16;
  *b3_load.y.axis = axis;
  *b3_load.y.repeats = {s0 * s1 * s2, s3};
  *b3_load.y.strides = {s3, One};

  Broadcast b3_broadcast("b3_broadcast");
  //   graph.AddNode(b3_broadcast);
  b3_broadcast.x = b2_sum.y;
  b3_broadcast.attr.sched.axis = axis;
  b3_broadcast.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  b3_broadcast.y.dtype = ge::DT_FLOAT16;
  *b3_broadcast.y.axis = axis;
  *b3_broadcast.y.repeats = {s0 * s1 * s2, s3};
  *b3_broadcast.y.strides = {s3, One};

  af::ascir_op::Div b3_div("b3_div");
  //   graph.AddNode(b3_div);
  b3_div.x1 = b3_load.y;
  b3_div.x2 = b3_broadcast.y;
  b3_div.attr.sched.axis = axis;
  b3_div.attr.api.compute_type = af::ComputeType::kComputeElewise;
  b3_div.y.dtype = ge::DT_FLOAT16;
  *b3_div.y.axis = axis;
  *b3_div.y.repeats = {s0 * s1 * s2, s3};
  *b3_div.y.strides = {s3, One};

  Store b3_store("b3_store");
  //   graph.AddNode(b3_store);
  b3_store.x = b3_div.y;
  b3_store.attr.sched.axis = axis;
  b3_store.attr.api.compute_type = af::ComputeType::kComputeStore;
  b3_store.y.dtype = ge::DT_FLOAT16;
  *b3_store.y.axis = axis;
  *b3_store.y.repeats = {s0 * s1 * s2, s3};
  *b3_store.y.strides = {s3, One};

  buf3.x = b3_store.y;
  buf3.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  buf3.attr.api.type = af::ApiType::kAPITypeBuffer;
  buf3.y.dtype = ge::DT_FLOAT16;
}

void Construct_Normal_Struct(af::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar("s3");

  auto z0 = graph.CreateAxis("z0", s0 * s1 * s2);
  auto z1 = graph.CreateAxis("z1", s3);

  auto axis = {z0.id, z1.id};

  Data arg4_1("arg4_1", graph);
  arg4_1.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  arg4_1.attr.api.type = af::ApiType::kAPITypeBuffer;
  arg4_1.y.dtype = ge::DT_FLOAT16;

  Load b0_load("b0_load");
  b0_load.x = arg4_1.y;
  b0_load.attr.sched.axis = axis;
  b0_load.attr.api.compute_type = af::ComputeType::kComputeLoad;
  b0_load.y.dtype = ge::DT_FLOAT16;
  *b0_load.y.axis = axis;
  *b0_load.y.repeats = {s0 * s1 * s2, s3};
  *b0_load.y.strides = {s3, One};

  Exp b1_exp("b1_exp");
  b1_exp.x = b0_load.y;
  b1_exp.attr.sched.axis = axis;
  b1_exp.attr.api.compute_type = af::ComputeType::kComputeElewise;
  b1_exp.attr.api.type = af::ApiType::kAPITypeCompute;
  b1_exp.y.dtype = ge::DT_FLOAT16;
  *b1_exp.y.axis = axis;
  *b1_exp.y.repeats = {s0 * s1 * s2, s3};
  *b1_exp.y.strides = {s3, One};

  Abs b0_abs("b0_abs");
  b0_abs.x = b1_exp.y;
  b0_abs.attr.sched.axis = axis;
  b0_abs.attr.api.compute_type = af::ComputeType::kComputeElewise;
  b0_abs.y.dtype = ge::DT_FLOAT16;
  *b0_abs.y.axis = axis;
  *b0_abs.y.repeats = {s0 * s1 * s2, s3};
  *b0_abs.y.strides = {s3, One};

  af::ascir_op::Max b0_max("b0_max");
  b0_max.x = b0_abs.y;
  b0_max.attr.sched.axis = axis;
  b0_max.attr.api.compute_type = af::ComputeType::kComputeReduce;
  b0_max.y.dtype = ge::DT_FLOAT16;
  *b0_max.y.axis = axis;
  *b0_max.y.repeats = {s0 * s1 * s2, s3};
  *b0_max.y.strides = {One, Zero};

  Broadcast b1_broadcast("b1_broadcast");
  b1_broadcast.x = b0_max.y;
  b1_broadcast.attr.sched.axis = axis;
  b1_broadcast.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  b1_broadcast.y.dtype = ge::DT_FLOAT16;
  *b1_broadcast.y.axis = axis;
  *b1_broadcast.y.repeats = {s0 * s1 * s2, s3};
  *b1_broadcast.y.strides = {s3, One};

  af::ascir_op::Sub b1_sub("b1_sub");
  b1_sub.x1 = b1_exp.y;
  b1_sub.x2 = b1_broadcast.y;
  b1_sub.attr.sched.axis = axis;
  b1_sub.attr.api.compute_type = af::ComputeType::kComputeElewise;
  b1_sub.y.dtype = ge::DT_FLOAT16;
  *b1_sub.y.axis = axis;
  *b1_sub.y.repeats = {s0 * s1 * s2, s3};
  *b1_sub.y.strides = {s3, One};

  Store b1_store("b1_store");
  b1_store.x = b1_sub.y;
  b1_store.attr.sched.axis = axis;
  b1_store.attr.api.compute_type = af::ComputeType::kComputeStore;
  b1_store.y.dtype = ge::DT_FLOAT16;
  *b1_store.y.axis = axis;
  *b1_store.y.repeats = {s0 * s1 * s2, s3};
  *b1_store.y.strides = {s3, One};
}

template <typename Op>
static void SetIndirectLoadView(Op &op, af::DataType dtype, const std::vector<af::AxisId> &axes,
                                const std::vector<af::Expression> &repeats,
                                const std::vector<af::Expression> &strides) {
  op.y.dtype = dtype;
  op.attr.sched.axis = axes;
  *op.y.axis = axes;
  *op.y.repeats = repeats;
  *op.y.strides = strides;
}

static af::AscGraph BuildIndirectLoadAutoScheduleGraph(int64_t axis, bool with_input_pre, bool with_post_reduce,
                                                       bool with_output_transform = true) {
  af::AscGraph graph("indirect_load_autoschedule_ut");
  const auto x0_size = graph.CreateSizeVar(2);
  const auto x1_size = graph.CreateSizeVar(8);
  const auto x2_size = graph.CreateSizeVar(4);
  const auto x3_size = graph.CreateSizeVar(3);
  const auto y0_size = graph.CreateSizeVar(2);
  const auto y1_size = graph.CreateSizeVar(with_input_pre ? 9 : 8);
  const auto y2_size = graph.CreateSizeVar(4);
  const auto y3_size = graph.CreateSizeVar(3);
  const auto x0 = graph.CreateAxis("x0", x0_size);
  const auto x1 = graph.CreateAxis("x1", x1_size);
  const auto x2 = graph.CreateAxis("x2", x2_size);
  const auto x3 = graph.CreateAxis("x3", x3_size);
  const auto y0 = graph.CreateAxis("y0", y0_size);
  const auto y1 = graph.CreateAxis("y1", y1_size);
  const auto y2 = graph.CreateAxis("y2", y2_size);
  const auto y3 = graph.CreateAxis("y3", y3_size);
  const std::vector<af::AxisId> input_axes = {x0.id, x1.id, x2.id, x3.id};
  const std::vector<af::AxisId> output_axes = {y0.id, y1.id, y2.id, y3.id};
  const std::vector<af::Expression> input_repeats = {x0_size, x1_size, x2_size, x3_size};
  const std::vector<af::Expression> output_repeats = {y0_size, y1_size, y2_size, y3_size};
  const std::vector<af::Expression> input_strides = {x1_size * x2_size * x3_size, x2_size * x3_size, x3_size, One};
  const std::vector<af::Expression> output_strides = {y1_size * y2_size * y3_size, y2_size * y3_size, y3_size, One};

  Data x("x", graph);
  x.ir_attr.SetIndex(0);
  x.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  x.attr.api.type = af::ApiType::kAPITypeBuffer;
  SetIndirectLoadView(x, ge::DT_FLOAT16, input_axes, input_repeats, input_strides);

  Load input_load("input_load");
  input_load.x = x.y;
  SetIndirectLoadView(input_load, ge::DT_FLOAT16, input_axes, input_repeats, input_strides);

  Data index("index", graph);
  index.ir_attr.SetIndex(1);
  index.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  index.attr.api.type = af::ApiType::kAPITypeBuffer;
  SetIndirectLoadView(index, ge::DT_INT32, output_axes, output_repeats, output_strides);
  Load index_load("index_load");
  index_load.x = index.y;
  SetIndirectLoadView(index_load, ge::DT_INT32, output_axes, output_repeats, output_strides);

  IndirectLoad indirect_load("indirect_load");
  if (with_input_pre) {
    Abs input_abs("input_abs");
    input_abs.x = input_load.y;
    SetIndirectLoadView(input_abs, ge::DT_FLOAT16, input_axes, input_repeats, input_strides);
    indirect_load.x1 = input_abs.y;
  } else {
    indirect_load.x1 = input_load.y;
  }
  indirect_load.x2 = index_load.y;
  indirect_load.ir_attr.SetAxis(axis);
  SetIndirectLoadView(indirect_load, ge::DT_FLOAT16, output_axes, output_repeats, output_strides);

  Store store("store");
  std::vector<af::Expression> final_repeats = output_repeats;
  std::vector<af::Expression> final_strides = output_strides;
  if (with_post_reduce) {
    Sum reduce("reduce");
    if (with_output_transform) {
      Abs output_abs("output_abs");
      output_abs.x = indirect_load.y;
      SetIndirectLoadView(output_abs, ge::DT_FLOAT16, output_axes, output_repeats, output_strides);
      reduce.x = output_abs.y;
    } else {
      reduce.x = indirect_load.y;
    }
    reduce.attr.api.compute_type = af::ComputeType::kComputeReduce;
    final_repeats = {y0_size, One, One, One};
    final_strides = {One, Zero, Zero, Zero};
    SetIndirectLoadView(reduce, ge::DT_FLOAT16, output_axes, final_repeats, final_strides);
    store.x = reduce.y;
    SetIndirectLoadView(store, ge::DT_FLOAT16, output_axes, final_repeats, final_strides);
  } else {
    store.x = indirect_load.y;
    SetIndirectLoadView(store, ge::DT_FLOAT16, output_axes, output_repeats, output_strides);
  }

  Output output("output");
  output.ir_attr.SetIndex(0);
  output.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  output.attr.api.type = af::ApiType::kAPITypeBuffer;
  output.x = store.y;
  SetIndirectLoadView(output, ge::DT_FLOAT16, output_axes, final_repeats, final_strides);
  return graph;
}

static bool InsertStridedIndexElement(af::AscGraph &graph) {
  const auto index = graph.FindNode("index");
  const auto index_load = graph.FindNode("index_load");
  const auto indirect_load = graph.FindNode("indirect_load");
  if (index == nullptr || index_load == nullptr || indirect_load == nullptr || index_load->outputs().empty()) {
    return false;
  }
  const auto &index_attr = index_load->outputs()[0]->attr;
  if (index_attr.axis.size() != 4UL || index_attr.repeats.size() != 4UL) {
    return false;
  }
  const auto &repeats = index_attr.repeats;
  const auto gap = One + One;
  const std::vector<af::Expression> strides = {repeats[1] * repeats[2] * repeats[3] * gap,
                                               repeats[2] * repeats[3] * gap, repeats[3], One};
  index->outputs()[0]->attr.strides = strides;
  index_load->outputs()[0]->attr.strides = strides;

  Abs index_abs("index_abs");
  SetIndirectLoadView(index_abs, ge::DT_INT32, index_attr.axis, index_attr.repeats, strides);
  graph.AddNode(index_abs);
  const auto index_abs_node = graph.FindNode("index_abs");
  if (index_abs_node == nullptr) {
    return false;
  }
  const auto load_out = index_load->GetOutDataAnchor(0);
  const auto indirect_in = indirect_load->GetInDataAnchor(1);
  return af::GraphUtils::RemoveEdge(load_out, indirect_in) == ge::GRAPH_SUCCESS &&
         af::GraphUtils::AddEdge(load_out, index_abs_node->GetInDataAnchor(0)) == ge::GRAPH_SUCCESS &&
         af::GraphUtils::AddEdge(index_abs_node->GetOutDataAnchor(0), indirect_in) == ge::GRAPH_SUCCESS;
}

static bool SetStridedInputPrePath(af::AscGraph &graph) {
  const std::vector<af::AscNodePtr> input_path = {graph.FindNode("x"), graph.FindNode("input_load"),
                                                  graph.FindNode("input_abs")};
  if (std::any_of(input_path.begin(), input_path.end(),
                  [](const af::AscNodePtr &node) { return node == nullptr || node->outputs().empty(); })) {
    return false;
  }
  const auto &attr = input_path.front()->outputs()[0]->attr;
  if (attr.repeats.size() != 4UL) {
    return false;
  }
  const auto &repeats = attr.repeats;
  const auto gap = One + One;
  const std::vector<af::Expression> strides = {repeats[1] * repeats[2] * repeats[3] * gap,
                                               repeats[2] * repeats[3] * gap, repeats[3], One};
  for (const af::AscNodePtr &node : input_path) {
    node->outputs()[0]->attr.strides = strides;
  }
  return true;
}

// 索引视图外层带 padding（GM 行步长为紧凑跨度的 2 倍）：索引为 kStrided 布局，可求解
// 多行 tile。输入与索引均为 kDense 时 lowering 选稠密 facade（无行基址项），保持固定
// 单行 tile，故求解类用例需索引侧带 padding 触发 kStrided 路径。
static bool SetPaddedIndexOuterStride(af::AscGraph &graph) {
  const std::vector<af::AscNodePtr> index_path = {graph.FindNode("index"), graph.FindNode("index_load")};
  if (std::any_of(index_path.begin(), index_path.end(),
                  [](const af::AscNodePtr &node) { return node == nullptr || node->outputs().empty(); })) {
    return false;
  }
  const auto &attr = index_path.front()->outputs()[0]->attr;
  if (attr.repeats.size() != 4UL) {
    return false;
  }
  const auto &repeats = attr.repeats;
  const auto gap = One + One;
  const std::vector<af::Expression> strides = {repeats[1] * repeats[2] * repeats[3] * gap, repeats[2] * repeats[3],
                                               repeats[3], One};
  for (const af::AscNodePtr &node : index_path) {
    node->outputs()[0]->attr.strides = strides;
  }
  return true;
}

static std::vector<af::AscGraph>::iterator FindIndirectLoadCandidate(std::vector<af::AscGraph> &candidates,
                                                                     ascir::TemplateId template_id) {
  return std::find_if(candidates.begin(), candidates.end(), [template_id](const af::AscGraph &candidate) {
    const auto indirect_load = candidate.FindNode("indirect_load");
    return indirect_load != nullptr && ascir::GetTemplateIdOrDefault(*indirect_load) == template_id;
  });
}

static const af::Axis *FindDerivedAxis(af::AscGraph &graph, af::Axis::Type type, af::AxisId source) {
  const auto axes = graph.GetAllAxis();
  const auto iter = std::find_if(axes.begin(), axes.end(), [type, source](const auto &axis) {
    return axis != nullptr && axis->type == type && axis->from == std::vector<af::AxisId>{source};
  });
  return iter == axes.end() ? nullptr : iter->get();
}

static bool AxisDescendsFrom(af::AscGraph &graph, af::AxisId axis_id, af::AxisId origin_id) {
  if (axis_id == origin_id) {
    return true;
  }
  const auto *axis = graph.FindAxis(axis_id);
  if (axis == nullptr) {
    return false;
  }
  return std::any_of(axis->from.begin(), axis->from.end(),
                     [&graph, origin_id](af::AxisId source) { return AxisDescendsFrom(graph, source, origin_id); });
}

static std::vector<af::AxisId> GetAxisOrigins(af::AscGraph &graph, af::AxisId axis_id) {
  const auto *axis = graph.FindAxis(axis_id);
  if (axis == nullptr) {
    return {};
  }
  if (axis->from.empty()) {
    return {axis_id};
  }
  std::vector<af::AxisId> origins;
  for (af::AxisId source : axis->from) {
    auto source_origins = GetAxisOrigins(graph, source);
    origins.insert(origins.end(), source_origins.begin(), source_origins.end());
  }
  std::sort(origins.begin(), origins.end());
  origins.erase(std::unique(origins.begin(), origins.end()), origins.end());
  return origins;
}

namespace optimize {
class AutoSchedulerUT : public ::testing::Test {
 protected:
  void SetUp() override {}
};

TEST_F(AutoSchedulerUT, IndirectLoadStridedIndexPathIsAlignedDuringSchedule) {
  auto graph = BuildIndirectLoadAutoScheduleGraph(1L, false, false);
  ASSERT_TRUE(InsertStridedIndexElement(graph));
  IndirectLoadScheduleCaseGenerator generator;
  std::vector<af::AscGraph> candidates;
  std::vector<std::string> score_functions;
  ASSERT_EQ(generator.Generate(graph, candidates, score_functions), af::SUCCESS);
  auto simd = FindIndirectLoadCandidate(candidates, ascir::TemplateId::kIndirectLoadSimd);
  ASSERT_NE(simd, candidates.end());
  const auto index_abs = simd->FindNode("index_abs");
  ASSERT_NE(index_abs, nullptr);
  ASSERT_EQ(ascgen_utils::indirect_load::GetTemplateRole(index_abs),
            ascgen_utils::indirect_load::TemplateRole::kStridedUbPath);

  std::vector<autoschedule::AutoScheduleOutput> results;
  AutoSchedule autoschedule(*simd, results);
  ASSERT_EQ(autoschedule.DoAutoSchedule(), af::SUCCESS);
  ASSERT_EQ(results.size(), 1UL);
  const auto scheduled_index_abs = results[0].scheduled_graph.FindNode("index_abs");
  ASSERT_NE(scheduled_index_abs, nullptr);
  ASSERT_EQ(scheduled_index_abs->outputs().size(), 1UL);
  EXPECT_FALSE(scheduled_index_abs->outputs()[0]->attr.vectorized_strides.empty());
}

TEST_F(AutoSchedulerUT, IndirectLoadSolvedTileTilingGeneratesOnlyTemplateOuterCase) {
  auto graph = BuildIndirectLoadAutoScheduleGraph(1L, false, false);
  ASSERT_TRUE(SetPaddedIndexOuterStride(graph));
  IndirectLoadScheduleCaseGenerator generator;
  std::vector<af::AscGraph> candidates;
  std::vector<std::string> score_functions;
  ASSERT_EQ(generator.Generate(graph, candidates, score_functions), af::SUCCESS);
  auto simd = FindIndirectLoadCandidate(candidates, ascir::TemplateId::kIndirectLoadSimd);
  ASSERT_NE(simd, candidates.end());
  auto candidate_indirect_load = simd->FindNode("indirect_load");
  ASSERT_NE(candidate_indirect_load, nullptr);
  ascgen_utils::indirect_load::TemplateAxes template_axes;
  ASSERT_EQ(ascgen_utils::indirect_load::GetTemplateAxes(candidate_indirect_load, template_axes), af::SUCCESS);

  std::vector<autoschedule::AutoScheduleOutput> results;
  AutoSchedule autoschedule(*simd, results);
  ASSERT_EQ(autoschedule.DoAutoSchedule(), af::SUCCESS);
  ASSERT_EQ(results.size(), 1UL);
  auto &scheduled_graph = results[0].scheduled_graph;
  auto scheduled_indirect_load = scheduled_graph.FindNode("indirect_load");
  ASSERT_NE(scheduled_indirect_load, nullptr);
  ascgen_utils::indirect_load::TemplateAxes scheduled_axes;
  ASSERT_EQ(ascgen_utils::indirect_load::GetTemplateAxes(scheduled_indirect_load, scheduled_axes), af::SUCCESS);
  EXPECT_EQ(scheduled_axes.outer_axis, template_axes.outer_axis);

  const auto *outer = scheduled_graph.FindAxis(scheduled_axes.outer_axis);
  const auto *tile_outer =
      FindDerivedAxis(scheduled_graph, af::Axis::Type::kAxisTypeTileOuter, scheduled_axes.outer_axis);
  const auto *tile_inner =
      FindDerivedAxis(scheduled_graph, af::Axis::Type::kAxisTypeTileInner, scheduled_axes.outer_axis);
  ASSERT_NE(outer, nullptr);
  ASSERT_NE(tile_outer, nullptr);
  ASSERT_NE(tile_inner, nullptr);
  // 无后置 Reduce 的裸 gather：TileInner 行数为调度期按 UB 容量求解的变量（非固定 1），
  // 多行 tile 使 staging/模板调用按 tile 批量执行。
  EXPECT_NE(af::SymbolicUtils::StaticCheckEq(tile_inner->size, One), af::TriBool::kTrue);
  EXPECT_EQ(tile_outer->split_pair_other_id, tile_inner->id);
  EXPECT_EQ(tile_inner->split_pair_other_id, tile_outer->id);
  ASSERT_EQ(scheduled_indirect_load->outputs().size(), 1UL);
  const auto *inner = scheduled_graph.FindAxis(scheduled_axes.inner_axis);
  ASSERT_NE(inner, nullptr);
  // TileInner 前插进向量化视图，成为模板一次处理的窗口行维度。
  std::vector<af::AxisId> expected_vectorized{tile_inner->id};
  expected_vectorized.insert(expected_vectorized.end(), inner->from.begin(), inner->from.end());
  EXPECT_EQ(scheduled_indirect_load->outputs()[0]->attr.vectorized_axis, expected_vectorized);
  for (af::AxisId axis : expected_vectorized) {
    EXPECT_NE(std::find(scheduled_indirect_load->outputs()[0]->attr.axis.begin(),
                        scheduled_indirect_load->outputs()[0]->attr.axis.end(), axis),
              scheduled_indirect_load->outputs()[0]->attr.axis.end());
  }
}

TEST_F(AutoSchedulerUT, IndirectLoadSimtKeepsDirectGmBoundariesOutsideMainTiling) {
  auto graph = BuildIndirectLoadAutoScheduleGraph(1L, false, false);
  IndirectLoadScheduleCaseGenerator generator;
  std::vector<af::AscGraph> candidates;
  std::vector<std::string> score_functions;
  ASSERT_EQ(generator.Generate(graph, candidates, score_functions), af::SUCCESS);
  auto simt = FindIndirectLoadCandidate(candidates, ascir::TemplateId::kIndirectLoadSimt);
  ASSERT_NE(simt, candidates.end());

  std::map<std::string, std::pair<std::vector<af::AxisId>, af::AscTensorAttr>> boundary_views;
  const std::map<std::string, ascgen_utils::indirect_load::TemplateRole> expected_roles = {
      {"input_load", ascgen_utils::indirect_load::TemplateRole::kSimtInputBoundary},
      {"store", ascgen_utils::indirect_load::TemplateRole::kSimtDirectGmBoundary}};
  for (const auto &[name, expected_role] : expected_roles) {
    const auto node = simt->FindNode(name.c_str());
    ASSERT_NE(node, nullptr);
    ASSERT_EQ(node->outputs().size(), 1UL);
    ASSERT_EQ(ascgen_utils::indirect_load::GetTemplateRole(node), expected_role);
    boundary_views.emplace(name, std::make_pair(node->attr.sched.axis, node->outputs()[0]->attr));
  }

  std::vector<autoschedule::AutoScheduleOutput> results;
  AutoSchedule autoschedule(*simt, results);
  ASSERT_EQ(autoschedule.DoAutoSchedule(), af::SUCCESS);
  ASSERT_EQ(results.size(), 1UL);
  auto &scheduled_graph = results[0].scheduled_graph;
  for (const auto &[name, original] : boundary_views) {
    const auto node = scheduled_graph.FindNode(name.c_str());
    ASSERT_NE(node, nullptr);
    ASSERT_EQ(node->outputs().size(), 1UL);
    // SIMT 直访 GM 边界节点不参与主 tiling 循环（核心语义保持）。其 tensor view 自
    // 293b93ae 起由 SyncSimtBoundaryViews 统一 merge/split 到模板轴空间，不再保持
    // 候选期的原始轴形态，但覆盖的原始轴集合与原始视图一致（语义等价）。
    EXPECT_EQ(node->attr.sched.loop_axis, af::kIdNone) << name;
    std::vector<af::AxisId> original_origins;
    for (af::AxisId axis_id : original.second.axis) {
      const auto origins = GetAxisOrigins(scheduled_graph, axis_id);
      original_origins.insert(original_origins.end(), origins.begin(), origins.end());
    }
    std::vector<af::AxisId> synced_origins;
    for (af::AxisId axis_id : node->outputs()[0]->attr.axis) {
      const auto origins = GetAxisOrigins(scheduled_graph, axis_id);
      synced_origins.insert(synced_origins.end(), origins.begin(), origins.end());
    }
    std::sort(original_origins.begin(), original_origins.end());
    original_origins.erase(std::unique(original_origins.begin(), original_origins.end()), original_origins.end());
    std::sort(synced_origins.begin(), synced_origins.end());
    synced_origins.erase(std::unique(synced_origins.begin(), synced_origins.end()), synced_origins.end());
    EXPECT_EQ(synced_origins, original_origins) << name;
    EXPECT_EQ(node->outputs()[0]->attr.vectorized_axis.size(), original.second.vectorized_axis.size()) << name;
  }

  const auto indirect_load = scheduled_graph.FindNode("indirect_load");
  ASSERT_NE(indirect_load, nullptr);
  bool has_block_inner = false;
  bool has_tile_inner = false;
  for (af::AxisId axis_id : indirect_load->attr.sched.axis) {
    const auto *axis = scheduled_graph.FindAxis(axis_id);
    ASSERT_NE(axis, nullptr);
    has_block_inner = has_block_inner || axis->type == af::Axis::Type::kAxisTypeBlockInner;
    has_tile_inner = has_tile_inner || axis->type == af::Axis::Type::kAxisTypeTileInner;
  }
  EXPECT_TRUE(has_block_inner);
  EXPECT_TRUE(has_tile_inner);
}

TEST_F(AutoSchedulerUT, IndirectLoadSchedulerUsesOnlyTemplateOuterForMainBlockTiling) {
  auto graph = BuildIndirectLoadAutoScheduleGraph(1L, false, false);
  IndirectLoadScheduleCaseGenerator generator;
  std::vector<af::AscGraph> candidates;
  std::vector<std::string> score_functions;
  ASSERT_EQ(generator.Generate(graph, candidates, score_functions), af::SUCCESS);
  auto simd = FindIndirectLoadCandidate(candidates, ascir::TemplateId::kIndirectLoadSimd);
  ASSERT_NE(simd, candidates.end());
  auto indirect_load = simd->FindNode("indirect_load");
  ASSERT_NE(indirect_load, nullptr);
  ascgen_utils::indirect_load::TemplateAxes template_axes;
  ASSERT_EQ(ascgen_utils::indirect_load::GetTemplateAxes(indirect_load, template_axes), af::SUCCESS);

  std::vector<autoschedule::AutoScheduleOutput> results;
  AutoSchedule autoschedule(*simd, results);
  ASSERT_EQ(autoschedule.DoAutoSchedule(), af::SUCCESS);
  ASSERT_EQ(results.size(), 1UL);
  auto &scheduled_graph = results[0].scheduled_graph;
  const auto template_outer_origins = GetAxisOrigins(scheduled_graph, template_axes.outer_axis);
  ASSERT_FALSE(template_outer_origins.empty());
  std::vector<const af::Axis *> block_outers;
  std::vector<const af::Axis *> block_inners;
  for (const auto &axis : scheduled_graph.GetAllAxis()) {
    if (axis == nullptr) {
      continue;
    }
    if (axis->type == af::Axis::Type::kAxisTypeBlockOuter) {
      block_outers.push_back(axis.get());
    } else if (axis->type == af::Axis::Type::kAxisTypeBlockInner) {
      block_inners.push_back(axis.get());
    }
  }
  ASSERT_EQ(block_outers.size(), 1UL);
  ASSERT_EQ(block_inners.size(), 1UL);
  for (const auto *axis : {block_outers[0], block_inners[0]}) {
    EXPECT_TRUE(AxisDescendsFrom(scheduled_graph, axis->id, template_axes.outer_axis));
    EXPECT_FALSE(AxisDescendsFrom(scheduled_graph, axis->id, template_axes.inner_axis));
    EXPECT_EQ(GetAxisOrigins(scheduled_graph, axis->id), template_outer_origins);
  }
  EXPECT_EQ(block_outers[0]->split_pair_other_id, block_inners[0]->id);
  EXPECT_EQ(block_inners[0]->split_pair_other_id, block_outers[0]->id);
}

TEST_F(AutoSchedulerUT, IndirectLoadAxisZeroSyntheticOuterParticipatesWithConsistentTensorRanks) {
  auto graph = BuildIndirectLoadAutoScheduleGraph(0L, false, false);
  IndirectLoadScheduleCaseGenerator generator;
  std::vector<af::AscGraph> candidates;
  std::vector<std::string> score_functions;
  ASSERT_EQ(generator.Generate(graph, candidates, score_functions), af::SUCCESS);
  auto simd = FindIndirectLoadCandidate(candidates, ascir::TemplateId::kIndirectLoadSimd);
  ASSERT_NE(simd, candidates.end());
  auto indirect_load = simd->FindNode("indirect_load");
  ASSERT_NE(indirect_load, nullptr);
  ascgen_utils::indirect_load::TemplateAxes template_axes;
  ASSERT_EQ(ascgen_utils::indirect_load::GetTemplateAxes(indirect_load, template_axes), af::SUCCESS);
  const auto *synthetic_outer = simd->FindAxis(template_axes.outer_axis);
  ASSERT_NE(synthetic_outer, nullptr);
  EXPECT_TRUE(synthetic_outer->from.empty());
  EXPECT_EQ(af::SymbolicUtils::StaticCheckEq(synthetic_outer->size, One), af::TriBool::kTrue);

  std::vector<autoschedule::AutoScheduleOutput> results;
  AutoSchedule autoschedule(*simd, results);
  ASSERT_EQ(autoschedule.DoAutoSchedule(), af::SUCCESS);
  ASSERT_EQ(results.size(), 1UL);
  auto &scheduled_graph = results[0].scheduled_graph;
  for (const auto &node : scheduled_graph.GetAllNodes()) {
    if (ScheduleUtils::IsBuffer(node)) {
      continue;
    }
    EXPECT_TRUE(std::any_of(node->attr.sched.axis.begin(), node->attr.sched.axis.end(),
                            [&scheduled_graph, &template_axes](af::AxisId axis_id) {
                              return AxisDescendsFrom(scheduled_graph, axis_id, template_axes.outer_axis);
                            }))
        << node->GetName();
    for (const auto &output : node->outputs()) {
      ASSERT_NE(output, nullptr);
      ASSERT_EQ(output->attr.axis.size(), output->attr.repeats.size()) << node->GetName();
      ASSERT_EQ(output->attr.axis.size(), output->attr.strides.size()) << node->GetName();
      bool found_synthetic_axis = false;
      for (size_t i = 0UL; i < output->attr.axis.size(); ++i) {
        if (AxisDescendsFrom(scheduled_graph, output->attr.axis[i], template_axes.outer_axis)) {
          found_synthetic_axis = true;
          EXPECT_EQ(af::SymbolicUtils::StaticCheckEq(output->attr.repeats[i], One), af::TriBool::kTrue)
              << node->GetName();
        }
      }
      EXPECT_TRUE(found_synthetic_axis) << node->GetName();
    }
  }
}

TEST_F(AutoSchedulerUT, IndirectLoadSimdInputPreUsesMergedInputInnerInFinalTensorView) {
  auto graph = BuildIndirectLoadAutoScheduleGraph(1L, true, false);
  ASSERT_TRUE(SetPaddedIndexOuterStride(graph));
  IndirectLoadScheduleCaseGenerator generator;
  std::vector<af::AscGraph> candidates;
  std::vector<std::string> score_functions;
  ASSERT_EQ(generator.Generate(graph, candidates, score_functions), af::SUCCESS);
  auto simd = FindIndirectLoadCandidate(candidates, ascir::TemplateId::kIndirectLoadSimd);
  ASSERT_NE(simd, candidates.end());
  auto indirect_load = simd->FindNode("indirect_load");
  ASSERT_NE(indirect_load, nullptr);
  ascgen_utils::indirect_load::TemplateAxes template_axes;
  ASSERT_EQ(ascgen_utils::indirect_load::GetTemplateAxes(indirect_load, template_axes), af::SUCCESS);
  const auto *input_inner = simd->FindAxis(template_axes.input_inner_axis);
  ASSERT_NE(input_inner, nullptr);
  ASSERT_EQ(input_inner->type, af::Axis::Type::kAxisTypeMerged);
  ASSERT_EQ(input_inner->from.size(), 3UL);

  std::vector<autoschedule::AutoScheduleOutput> results;
  AutoSchedule autoschedule(*simd, results);
  ASSERT_EQ(autoschedule.DoAutoSchedule(), af::SUCCESS);
  ASSERT_EQ(results.size(), 1UL);
  auto input_abs = results[0].scheduled_graph.FindNode("input_abs");
  ASSERT_NE(input_abs, nullptr);
  ASSERT_EQ(ascgen_utils::indirect_load::GetTemplateRole(input_abs),
            ascgen_utils::indirect_load::TemplateRole::kSimdInputPre);
  ASSERT_EQ(input_abs->outputs().size(), 1UL);
  const auto &output = input_abs->outputs()[0]->attr;
  EXPECT_NE(
      std::find(input_abs->attr.sched.axis.begin(), input_abs->attr.sched.axis.end(), template_axes.input_inner_axis),
      input_abs->attr.sched.axis.end());
  EXPECT_NE(std::find(output.axis.begin(), output.axis.end(), template_axes.input_inner_axis), output.axis.end());
  EXPECT_EQ(output.axis.size(), output.repeats.size());
  EXPECT_EQ(output.axis.size(), output.strides.size());
  // 可求解多行 tile：TileInner 前插进输入 pre 节点的向量化窗口（多行 staging），内层
  // 步长仍为 1；TileInner 步长为输入行步长。
  const auto *solved_tile_inner =
      FindDerivedAxis(results[0].scheduled_graph, af::Axis::Type::kAxisTypeTileInner, template_axes.outer_axis);
  ASSERT_NE(solved_tile_inner, nullptr);
  EXPECT_EQ(output.vectorized_axis, (std::vector<af::AxisId>{solved_tile_inner->id, template_axes.input_inner_axis}));
  ASSERT_EQ(output.vectorized_strides.size(), 2UL);
  EXPECT_EQ(af::SymbolicUtils::StaticCheckEq(output.vectorized_strides[1], One), af::TriBool::kTrue);
}

TEST_F(AutoSchedulerUT, IndirectLoadSimdInputPreKeepsOriginalVectorizedAxesWhenTensorMergeIsSkipped) {
  auto graph = BuildIndirectLoadAutoScheduleGraph(1L, true, false);
  ASSERT_TRUE(SetStridedInputPrePath(graph));
  IndirectLoadScheduleCaseGenerator generator;
  std::vector<af::AscGraph> candidates;
  std::vector<std::string> score_functions;
  ASSERT_EQ(generator.Generate(graph, candidates, score_functions), af::SUCCESS);
  auto simd = FindIndirectLoadCandidate(candidates, ascir::TemplateId::kIndirectLoadSimd);
  ASSERT_NE(simd, candidates.end());
  const auto indirect_load = simd->FindNode("indirect_load");
  ASSERT_NE(indirect_load, nullptr);
  ascgen_utils::indirect_load::TemplateAxes template_axes;
  ASSERT_EQ(ascgen_utils::indirect_load::GetTemplateAxes(indirect_load, template_axes), af::SUCCESS);
  const auto *input_inner = simd->FindAxis(template_axes.input_inner_axis);
  ASSERT_NE(input_inner, nullptr);
  ASSERT_EQ(input_inner->type, af::Axis::Type::kAxisTypeMerged);
  const auto input_inner_from = input_inner->from;
  ASSERT_EQ(input_inner_from.size(), 3UL);

  std::vector<autoschedule::AutoScheduleOutput> results;
  AutoSchedule autoschedule(*simd, results);
  ASSERT_EQ(autoschedule.DoAutoSchedule(), af::SUCCESS);
  ASSERT_EQ(results.size(), 1UL);
  const auto input_abs = results[0].scheduled_graph.FindNode("input_abs");
  ASSERT_NE(input_abs, nullptr);
  EXPECT_EQ(ascgen_utils::indirect_load::GetTemplateRole(input_abs),
            ascgen_utils::indirect_load::TemplateRole::kSimdInputPreStridedUbPath);
  ASSERT_EQ(input_abs->outputs().size(), 1UL);
  const auto &output = input_abs->outputs()[0]->attr;
  EXPECT_NE(
      std::find(input_abs->attr.sched.axis.begin(), input_abs->attr.sched.axis.end(), template_axes.input_inner_axis),
      input_abs->attr.sched.axis.end());
  EXPECT_EQ(std::find(output.axis.begin(), output.axis.end(), template_axes.input_inner_axis), output.axis.end());
  EXPECT_EQ(output.vectorized_axis, input_inner_from);
  EXPECT_EQ(output.vectorized_strides,
            (std::vector<af::Expression>{af::Symbol(64), af::Symbol(16), af::sym::kSymbolOne}));
  // 对齐（padding）输入视图不走可求解多行 tile：staging 剥离 padding 与模板按逻辑步长
  // 寻址在多行窗口下不一致，保持固定单行 tile。
  const auto *strided_tile_inner =
      FindDerivedAxis(results[0].scheduled_graph, af::Axis::Type::kAxisTypeTileInner, template_axes.outer_axis);
  ASSERT_NE(strided_tile_inner, nullptr);
  EXPECT_EQ(af::SymbolicUtils::StaticCheckEq(strided_tile_inner->size, One), af::TriBool::kTrue);
  for (const af::AxisId vectorized_axis : output.vectorized_axis) {
    EXPECT_NE(std::find(output.axis.begin(), output.axis.end(), vectorized_axis), output.axis.end());
  }
}

TEST_F(AutoSchedulerUT, IndirectLoadPostReduceSetsConnectedInputVectorizedViewAndOuterRepeats) {
  auto graph = BuildIndirectLoadAutoScheduleGraph(1L, false, true);
  IndirectLoadScheduleCaseGenerator generator;
  std::vector<af::AscGraph> candidates;
  std::vector<std::string> score_functions;
  ASSERT_EQ(generator.Generate(graph, candidates, score_functions), af::SUCCESS);
  auto simd = FindIndirectLoadCandidate(candidates, ascir::TemplateId::kIndirectLoadSimd);
  ASSERT_NE(simd, candidates.end());
  auto indirect_load = simd->FindNode("indirect_load");
  ASSERT_NE(indirect_load, nullptr);
  ascgen_utils::indirect_load::TemplateAxes template_axes;
  ASSERT_EQ(ascgen_utils::indirect_load::GetTemplateAxes(indirect_load, template_axes), af::SUCCESS);
  const auto *inner = simd->FindAxis(template_axes.inner_axis);
  ASSERT_NE(inner, nullptr);
  ASSERT_FALSE(inner->from.empty());
  const af::AxisId first_inner_axis = inner->from.front();
  std::vector<autoschedule::AutoScheduleOutput> results;
  AutoSchedule autoschedule(*simd, results);
  ASSERT_EQ(autoschedule.DoAutoSchedule(), af::SUCCESS);
  ASSERT_EQ(results.size(), 1UL);
  auto &scheduled_graph = results[0].scheduled_graph;
  auto reduce = scheduled_graph.FindNode("reduce");
  auto output_abs = scheduled_graph.FindNode("output_abs");
  ASSERT_NE(reduce, nullptr);
  ASSERT_NE(output_abs, nullptr);
  ASSERT_EQ(ascgen_utils::indirect_load::GetInputProducer(reduce, 0UL), output_abs);
  ASSERT_EQ(reduce->inputs.Size(), 1UL);
  const auto &reduce_input = reduce->inputs[0].attr;
  const auto first_inner = std::find(reduce_input.axis.begin(), reduce_input.axis.end(), first_inner_axis);
  ASSERT_NE(first_inner, reduce_input.axis.end());
  const size_t inner_offset = static_cast<size_t>(std::distance(reduce_input.axis.begin(), first_inner));
  EXPECT_EQ(
      reduce_input.vectorized_axis,
      std::vector<af::AxisId>(reduce_input.axis.begin() + static_cast<int64_t>(inner_offset), reduce_input.axis.end()));
  ASSERT_EQ(output_abs->outputs().size(), 1UL);
  const auto &producer_output = output_abs->outputs()[0]->attr;
  EXPECT_FALSE(producer_output.vectorized_axis.empty());
  EXPECT_EQ(reduce_input.vectorized_axis, producer_output.vectorized_axis);
  EXPECT_EQ(reduce_input.vectorized_strides, producer_output.vectorized_strides);

  ASSERT_EQ(reduce->outputs().size(), 1UL);
  const auto &reduce_output = reduce->outputs()[0]->attr;
  const auto output_first_inner = std::find(reduce_output.axis.begin(), reduce_output.axis.end(), first_inner_axis);
  ASSERT_NE(output_first_inner, reduce_output.axis.end());
  const size_t output_inner_offset = static_cast<size_t>(std::distance(reduce_output.axis.begin(), output_first_inner));
  for (size_t i = 0UL; i < output_inner_offset; ++i) {
    EXPECT_EQ(af::SymbolicUtils::StaticCheckEq(reduce_output.repeats[i], One), af::TriBool::kTrue);
  }
}

TEST_F(AutoSchedulerUT, IndirectLoadSimtPostReducePreservesOuterRepeatsAndKeepsVectorRegionsConsistent) {
  auto graph = BuildIndirectLoadAutoScheduleGraph(1L, false, true, false);
  IndirectLoadScheduleCaseGenerator generator;
  std::vector<af::AscGraph> candidates;
  std::vector<std::string> score_functions;
  ASSERT_EQ(generator.Generate(graph, candidates, score_functions), af::SUCCESS);
  auto simt = FindIndirectLoadCandidate(candidates, ascir::TemplateId::kIndirectLoadSimt);
  ASSERT_NE(simt, candidates.end());
  const auto candidate_il = simt->FindNode("indirect_load");
  ASSERT_NE(candidate_il, nullptr);
  ascgen_utils::indirect_load::TemplateAxes template_axes;
  ASSERT_EQ(ascgen_utils::indirect_load::GetTemplateAxes(candidate_il, template_axes), af::SUCCESS);
  const auto *inner = simt->FindAxis(template_axes.inner_axis);
  ASSERT_NE(inner, nullptr);
  ASSERT_FALSE(inner->from.empty());
  const af::AxisId first_inner_axis = inner->from.front();

  std::vector<autoschedule::AutoScheduleOutput> results;
  AutoSchedule autoschedule(*simt, results);
  ASSERT_EQ(autoschedule.DoAutoSchedule(), af::SUCCESS);
  ASSERT_EQ(results.size(), 1UL);
  auto &scheduled_graph = results[0].scheduled_graph;
  const auto indirect_load = scheduled_graph.FindNode("indirect_load");
  const auto reduce = scheduled_graph.FindNode("reduce");
  ASSERT_NE(indirect_load, nullptr);
  ASSERT_NE(reduce, nullptr);
  const auto scheduled_axes = scheduled_graph.GetAllAxis();
  const auto feature_tile_inner =
      std::find_if(scheduled_axes.begin(), scheduled_axes.end(), [&template_axes](const auto &axis) {
        return axis != nullptr && axis->type == ascir::Axis::Type::kAxisTypeTileInner && axis->from.size() == 1UL &&
               axis->from.front() == template_axes.outer_axis;
      });
  ASSERT_NE(feature_tile_inner, scheduled_axes.end());
  ASSERT_EQ(indirect_load->outputs().size(), 1UL);
  const auto &il_output = indirect_load->outputs()[0]->attr;
  const auto first_inner = std::find(il_output.axis.begin(), il_output.axis.end(), first_inner_axis);
  ASSERT_NE(first_inner, il_output.axis.end());
  const size_t inner_offset = static_cast<size_t>(std::distance(il_output.axis.begin(), first_inner));
  bool has_non_unit_outer_repeat = false;
  for (size_t i = 0UL; i < inner_offset; ++i) {
    has_non_unit_outer_repeat |= af::SymbolicUtils::StaticCheckEq(il_output.repeats[i], One) != af::TriBool::kTrue;
  }
  EXPECT_TRUE(has_non_unit_outer_repeat);
  ASSERT_EQ(ascgen_utils::indirect_load::GetInputProducer(reduce, 0UL), indirect_load);
  ASSERT_EQ(reduce->inputs.Size(), 1UL);
  EXPECT_EQ(reduce->inputs[0].attr.vectorized_axis, il_output.vectorized_axis);
  EXPECT_EQ(reduce->inputs[0].attr.vectorized_strides, il_output.vectorized_strides);
  EXPECT_FALSE(reduce->inputs[0].attr.vectorized_axis.empty());
  EXPECT_EQ(reduce->inputs[0].attr.vectorized_axis.front(), (*feature_tile_inner)->id);
}

TEST_F(AutoSchedulerUT, TilingGroup_gen_elementwise_tilingGroup) {
  af::AscGraph graph("LoadAbsStore");
  Construct_LoadAbsStore(graph);

  auto abs = graph.FindNode("abs");

  AxisGroup group;
  TilingGroup::GenElewiseTilingGroup(*abs, group);

  AxisGroup expect_group;
  size_t base_order = 0UL;
  for (auto axis : abs->attr.sched.axis) {
    expect_group.y_group.push_back(axis);
    expect_group.axes_order.push_back(base_order++);
  }

  EXPECT_EQ(group, expect_group);
}

TEST_F(AutoSchedulerUT, AutoSchedule_eletwise_gen_tilingGroup) {
  af::AscGraph graph("LoadAbsStore");
  Construct_LoadAbsStore(graph);

  auto abs = graph.FindNode("abs");

  std::vector<autoschedule::AutoScheduleOutput> results;
  AutoSchedule autoSchedule(graph, results);
  EXPECT_EQ(TilingGroup::GenTilingGroup(graph, autoSchedule.axes_group_), 0);
  EXPECT_EQ(autoSchedule.axes_group_.x_group.size(), 0UL);
  EXPECT_EQ(autoSchedule.axes_group_.y_group.size(), 3UL);
  EXPECT_EQ(autoSchedule.axes_group_.r_group.size(), 0UL);
  std::vector<af::AxisId> y_group(abs->attr.sched.axis);
  EXPECT_EQ(autoSchedule.axes_group_.y_group, y_group);
}

TEST_F(AutoSchedulerUT, AutoSchedule_eletwise_fusion_gen_tilingGroup) {
  af::AscGraph graph("LoadAbsStore");
  Construct_ElementwiseFusion(graph);

  auto abs = graph.FindNode("abs1");

  std::vector<autoschedule::AutoScheduleOutput> impl_graphs;
  AutoSchedule autoSchedule(graph, impl_graphs);
  EXPECT_EQ(TilingGroup::GenTilingGroup(graph, autoSchedule.axes_group_), 0);
  EXPECT_EQ(autoSchedule.axes_group_.x_group.size(), 0);
  EXPECT_EQ(autoSchedule.axes_group_.y_group.size(), 1);
  EXPECT_EQ(autoSchedule.axes_group_.r_group.size(), 0);

  std::vector<af::AxisId> y_group(abs->attr.sched.axis);
  EXPECT_EQ(autoSchedule.axes_group_.y_group, y_group);
}

TEST_F(AutoSchedulerUT, AutoSchedule_Tiling) {
  af::AscGraph graph("LoadAbsStore");
  Construct_LoadAbsStore(graph);

  auto abs = graph.FindNode("abs");
  std::vector<af::AxisId> y_group = abs->attr.sched.axis;
  AxisGroup axes_group;
  axes_group.y_group = y_group;
  axes_group.axes_order = {0, 1, 2};
  TilingCase tiling_case;
  tiling_case.ub_tiling_id_y = y_group[1];
  tiling_case.block_tiling_id = 0;
  Scheduler scheduler(graph, axes_group, tiling_case);
  std::vector<af::AxisId> new_axis_order;
  scheduler.DoScheduler();

  EXPECT_EQ(scheduler.tiling_case_.ub_tiling_id_x, -1);
  EXPECT_EQ(scheduler.tiling_case_.ub_tiling_id_y, 1);
  EXPECT_EQ(scheduler.tiling_case_.ub_tiling_id_r, -1);
  EXPECT_EQ(scheduler.tiling_case_.block_tiling_id, 5);
  EXPECT_EQ(graph.GetAllAxis().size(), 8);
  EXPECT_EQ(std::get<0>(scheduler.tiling_case_.ub_tiling_y)->id, 3);
  EXPECT_EQ(std::get<1>(scheduler.tiling_case_.ub_tiling_y)->id, 4);
  EXPECT_EQ(std::get<0>(scheduler.tiling_case_.block_tiling)->id, 6);
  EXPECT_EQ(std::get<1>(scheduler.tiling_case_.block_tiling)->id, 7);
}

TEST_F(AutoSchedulerUT, Autoschedule_scheduler_elementwise_3axis) {
  af::AscGraph graph("LoadAbsStore");
  Construct_LoadAbsStore(graph);

  af::AscGraph except_graph("LoadAbsStore");
  except_graph.CopyFrom(graph);

  auto abs = graph.FindNode("abs");
  std::vector<af::AxisId> y_group = abs->attr.sched.axis;

  AxisGroup axes_group;
  axes_group.y_group = y_group;
  axes_group.axes_order = {0, 1, 2};
  TilingCase tiling_case;
  tiling_case.ub_tiling_id_y = 1;
  tiling_case.block_tiling_id = 0;
  Scheduler scheduler(graph, axes_group, tiling_case);
  scheduler.DoScheduler();
  auto dump_graph = utils::DebugStr(graph);

  int ub_axis_id = 1;

  auto output = except_graph.FindNode("store");
  // split ub
  auto z_ub_id = output->attr.sched.axis[ub_axis_id];
  auto [z_ub_out, z_ub_in] = except_graph.TileSplit(z_ub_id);

  // fuse outer axes
  std::vector<af::AxisId> axes{output->attr.sched.axis[0], z_ub_out->id};
  auto block_axis = except_graph.MergeAxis(axes);

  // split block
  auto [z_block_out, z_block_in] = except_graph.BlockSplit(block_axis->id);

  vector<af::AxisId> vectorize_axis = {z_ub_in->id, output->attr.sched.axis[2]};
  vector<af::Expression> vectorized_strides{graph.FindAxis(vectorize_axis[1])->size, One};

  for (auto n : except_graph.GetAllNodes()) {
    if (n->attr.api.type == af::ApiType::kAPITypeBuffer) {
      continue;
    }
    except_graph.ApplySplit(n, z_ub_out->id, z_ub_in->id);
    except_graph.ApplySchedAxisMerge(n, block_axis->id);
    except_graph.ApplySplit(n, z_block_out->id, z_block_in->id);
    n->outputs[0].attr.vectorized_axis = vectorize_axis;
    n->outputs[0].attr.vectorized_strides = vectorized_strides;
  }

  auto dump_except_graph = utils::DebugStr(except_graph);

  EXPECT_EQ(dump_graph, dump_except_graph);
}

TEST_F(AutoSchedulerUT, Autoschedule_scheduler_elementwise_1axis) {
  af::AscGraph graph("LoadAbsStore");
  Construct_ElementwiseAbs(graph);

  af::AscGraph except_graph("LoadAbsStore");
  except_graph.CopyFrom(graph);

  auto abs = graph.FindNode("abs");
  std::vector<af::AxisId> y_group = abs->attr.sched.axis;

  AxisGroup axes_group;
  axes_group.y_group = y_group;
  axes_group.axes_order = {0};
  TilingCase tiling_case;
  tiling_case.ub_tiling_id_y = 0;
  tiling_case.block_tiling_id = 0;
  Scheduler scheduler(graph, axes_group, tiling_case);
  scheduler.DoScheduler();
  auto dump_graph = utils::DebugStr(graph);

  int ub_axis_id = 0;

  auto output = except_graph.FindNode("store");
  // split ub
  auto z_ub_id = output->attr.sched.axis[ub_axis_id];
  auto [z_ub_out, z_ub_in] = except_graph.TileSplit(z_ub_id);

  // split block
  auto [z_block_out, z_block_in] = except_graph.BlockSplit(z_ub_out->id);

  vector<af::AxisId> vectorize_axis = {z_ub_in->id};
  vector<af::Expression> vectorized_strides{One};

  for (auto n : except_graph.GetAllNodes()) {
    if (optimize::ScheduleUtils::IsBuffer(n)) {
      continue;
    }
    except_graph.ApplySplit(n, z_ub_out->id, z_ub_in->id);
    except_graph.ApplySplit(n, z_block_out->id, z_block_in->id);
    n->outputs[0].attr.vectorized_axis = vectorize_axis;
    n->outputs[0].attr.vectorized_strides = vectorized_strides;
  }

  auto dump_except_graph = utils::DebugStr(except_graph);

  EXPECT_EQ(dump_graph, dump_except_graph);
}

TEST_F(AutoSchedulerUT, Autoschedule_autoschedule_elementwise_1axis) {
  af::AscGraph graph("LoadAbsStore");
  Construct_ElementwiseAbs(graph);

  af::AscGraph except_graph("LoadAbsStore_B0Y0");
  except_graph.CopyFrom(graph);

  std::vector<autoschedule::AutoScheduleOutput> results;
  AutoSchedule autoschedule(graph, results);
  autoschedule.DoAutoSchedule();
  ASSERT_EQ(results.size(), 1);
  auto dump_graph = utils::DebugStr(results[0].scheduled_graph);

  int ub_axis_id = 0;

  auto output = except_graph.FindNode("store");
  // split ub
  auto z_ub_id = output->attr.sched.axis[ub_axis_id];
  auto [z_ub_out, z_ub_in] = except_graph.TileSplit(z_ub_id);

  // split block
  auto [z_block_out, z_block_in] = except_graph.BlockSplit(z_ub_out->id);

  vector<af::AxisId> vectorize_axis = {z_ub_in->id};
  vector<af::Expression> vectorized_strides{One};

  for (auto n : except_graph.GetAllNodes()) {
    if (optimize::ScheduleUtils::IsBuffer(n)) {
      continue;
    }
    except_graph.ApplySplit(n, z_ub_out->id, z_ub_in->id);
    except_graph.ApplySplit(n, z_block_out->id, z_block_in->id);
    n->outputs[0].attr.vectorized_axis = vectorize_axis;
    n->outputs[0].attr.vectorized_strides = vectorized_strides;
  }

  auto dump_except_graph = utils::DebugStr(except_graph);

  EXPECT_EQ(dump_graph, dump_except_graph);
}

TEST_F(AutoSchedulerUT, Autoschedule_autoschedule_elementwise_fusion) {
  af::AscGraph graph("AbsFusion");
  Construct_ElementwiseFusion(graph);

  af::AscGraph except_graph("AbsFusion_B0Y0");
  except_graph.CopyFrom(graph);

  std::vector<autoschedule::AutoScheduleOutput> impl_graphs;
  AutoSchedule autoschedule(graph, impl_graphs);
  autoschedule.DoAutoSchedule();
  EXPECT_EQ(impl_graphs.size(), 1);
  auto dump_graph = utils::DebugStr(impl_graphs[0].scheduled_graph);

  int ub_axis_id = 0;

  auto output = except_graph.FindNode("store");
  // split ub
  auto z_ub_id = output->attr.sched.axis[ub_axis_id];
  auto [z_ub_out, z_ub_in] = except_graph.TileSplit(z_ub_id);

  // split block
  auto [z_block_out, z_block_in] = except_graph.BlockSplit(z_ub_out->id);

  vector<af::AxisId> vectorize_axis = {z_ub_in->id};
  vector<af::Expression> vectorized_strides{One};

  for (auto n : except_graph.GetAllNodes()) {
    if (optimize::ScheduleUtils::IsBuffer(n)) {
      continue;
    }
    except_graph.ApplySplit(n, z_ub_out->id, z_ub_in->id);
    except_graph.ApplySplit(n, z_block_out->id, z_block_in->id);
    n->outputs[0].attr.vectorized_axis = vectorize_axis;
    n->outputs[0].attr.vectorized_strides = vectorized_strides;
  }

  auto dump_except_graph = utils::DebugStr(except_graph);

  EXPECT_EQ(dump_graph, dump_except_graph);
}

TEST_F(AutoSchedulerUT, ConcatGraph_scheduler) {
  af::AscGraph graph("test_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto si = graph.CreateSizeVar("si");
  auto sj = graph.CreateSizeVar("sj");
  auto sk = graph.CreateSizeVar("sk");
  auto s2 = graph.CreateSizeVar("s2");

  auto so = graph.CreateSizeVar("so");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  auto zo = graph.CreateAxis("zo", so);

  af::ascir_op::Data data_i("data_i", graph);
  data_i.attr.sched.axis = {z0.id, z1.id, zo.id, z2.id};
  data_i.y.dtype = ge::DT_FLOAT16;
  *data_i.y.axis = {z0.id, z1.id, zo.id, z2.id};
  data_i.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  data_i.attr.api.type = af::ApiType::kAPITypeBuffer;

  af::ascir_op::Data data_j("data_j", graph);
  data_j.attr.sched.axis = {z0.id, z1.id, zo.id, z2.id};
  data_j.y.dtype = ge::DT_FLOAT16;
  *data_j.y.axis = {z0.id, z1.id, zo.id, z2.id};
  data_j.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  data_j.attr.api.type = af::ApiType::kAPITypeBuffer;

  af::ascir_op::Data data_k("data_k", graph);
  data_k.attr.sched.axis = {z0.id, z1.id, zo.id, z2.id};
  data_k.y.dtype = ge::DT_FLOAT16;
  *data_k.y.axis = {z0.id, z1.id, zo.id, z2.id};
  data_k.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  data_k.attr.api.type = af::ApiType::kAPITypeBuffer;

  af::ascir_op::Load load_i("load_i");
  load_i.x = data_i.y;
  load_i.attr.sched.axis = {z0.id, z1.id, zo.id, z2.id};
  *load_i.y.axis = {z0.id, z1.id, zo.id, z2.id};
  *load_i.y.repeats = {s0, s1, si, s2};
  *load_i.y.strides = {s1 * si * s2, si * s2, s2, af::ops::One};
  load_i.attr.api.compute_type = af::ComputeType::kComputeLoad;

  af::ascir_op::Load load_j("load_j");
  load_j.x = data_j.y;
  load_j.attr.sched.axis = {z0.id, z1.id, zo.id, z2.id};
  *load_j.y.axis = {z0.id, z1.id, zo.id, z2.id};
  *load_j.y.repeats = {s0, s1, sj, s2};
  *load_j.y.strides = {s1 * sj * s2, sj * s2, s2, af::ops::One};
  load_j.attr.api.compute_type = af::ComputeType::kComputeLoad;

  af::ascir_op::Load load_k("load_k");
  load_k.x = data_k.y;
  load_k.attr.sched.axis = {z0.id, z1.id, zo.id, z2.id};
  *load_k.y.axis = {z0.id, z1.id, zo.id, z2.id};
  *load_k.y.repeats = {s0, s1, sk, s2};
  *load_k.y.strides = {s1 * sk * s2, sk * s2, s2, af::ops::One};
  load_k.attr.api.compute_type = af::ComputeType::kComputeLoad;

  af::ascir_op::Concat concat("concat");

  concat.x = {load_i.y, load_j.y, load_k.y};

  concat.attr.sched.axis = {z0.id, z1.id, zo.id, z2.id};
  concat.y.dtype = ge::DT_FLOAT16;
  *concat.y.axis = {z0.id, z1.id, zo.id, z2.id};
  *concat.y.repeats = {s0, s1, so, s2};
  *concat.y.strides = {s1 * so * s2, so * s2, s2, af::ops::One};
  concat.attr.api.compute_type = af::ComputeType::kComputeConcat;

  af::ascir_op::Store store("store");
  store.x = concat.y;
  store.attr.sched.axis = {z0.id, z1.id, zo.id, z2.id};
  store.y.dtype = ge::DT_FLOAT16;
  *store.y.axis = {z0.id, z1.id, zo.id, z2.id};
  *store.y.repeats = {s0, s1, so, s2};
  *store.y.strides = {s1 * so * s2, so * s2, s2, af::ops::One};
  store.attr.api.compute_type = af::ComputeType::kComputeStore;

  af::ascir_op::Output y("y");
  y.x = store.y;
  y.attr.sched.axis = {z0.id, z1.id, zo.id, z2.id};
  y.y.dtype = ge::DT_FLOAT16;
  *y.y.axis = {z0.id, z1.id, zo.id, z2.id};
  y.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  y.attr.api.type = af::ApiType::kAPITypeBuffer;

  std::string res = utils::DebugStr(graph);
  ASSERT_NE(res, "");

  std::vector<autoschedule::AutoScheduleOutput> results;
  AutoSchedule schedule(graph, results);
  schedule.DoAutoSchedule();
  ASSERT_EQ(results.size(), 2UL);
}

TEST_F(AutoSchedulerUT, align_vectorized_strides) {
  af::AscGraph graph("LoadAbsStore");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = af::ops::One;

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  af::ascir_op::Data x("x", graph);
  x.attr.sched.axis = {z0.id, z1.id, z2.id};
  x.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  x.attr.api.type = af::ApiType::kAPITypeBuffer;
  x.y.dtype = ge::DT_FLOAT16;
  *x.y.axis = {z0.id, z1.id, z2.id};
  *x.y.repeats = {s0, s1, s2};
  *x.y.strides = {s1, One, Zero};

  af::ascir_op::Load load("load");
  graph.AddNode(load);
  load.x = x.y;
  load.attr.sched.axis = {z0.id, z1.id, z2.id};
  load.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load.y.dtype = ge::DT_FLOAT16;
  *load.y.axis = {z0.id, z1.id, z2.id};
  *load.y.repeats = {s0, s1, s2};
  *load.y.strides = {s1, One, Zero};

  af::ascir_op::Abs abs("abs");
  graph.AddNode(abs);
  abs.x = load.y;
  abs.attr.sched.axis = {z0.id, z1.id, z2.id};
  abs.attr.api.compute_type = af::ComputeType::kComputeElewise;
  abs.y.dtype = ge::DT_FLOAT16;
  *abs.y.axis = {z0.id, z1.id, z2.id};
  *abs.y.repeats = {s0, s1, s2};
  *abs.y.strides = {s1, One, Zero};

  af::ascir_op::Store store("store");
  graph.AddNode(store);
  store.x = abs.y;
  store.attr.sched.axis = {z0.id, z1.id, z2.id};
  store.attr.api.compute_type = af::ComputeType::kComputeStore;
  store.y.dtype = ge::DT_FLOAT16;
  *store.y.axis = {z0.id, z1.id, z2.id};
  *store.y.repeats = {s0, s1, s2};
  *store.y.strides = {s1, One, Zero};

  af::ascir_op::Output y("y");
  y.x = store.y;
  y.attr.sched.axis = {z0.id, z1.id, z2.id};
  y.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  y.attr.api.type = af::ApiType::kAPITypeBuffer;
  y.y.dtype = ge::DT_FLOAT16;
  *y.y.axis = {z0.id, z1.id, z2.id};
  *y.y.repeats = {s0, s1, s2};
  *y.y.strides = {s1, One, Zero};

  for (auto n : graph.GetAllNodes()) {
    if (optimize::ScheduleUtils::IsBuffer(n)) {
      continue;
    }
    n->outputs[0].attr.vectorized_axis = {z0.id, z1.id, z2.id};
  }

  AlignmentHandler handler;
  EXPECT_EQ(handler.AlignVectorizedStrides(graph), af::SUCCESS);
  auto abs_node = graph.FindNode("abs");

  EXPECT_EQ(std::string(abs_node->outputs[0].attr.vectorized_strides[0].Str().get()), "s1");
  EXPECT_EQ(std::string(abs_node->outputs[0].attr.vectorized_strides[1].Str().get()), "1");
  EXPECT_EQ(std::string(abs_node->outputs[0].attr.vectorized_strides[2].Str().get()), "0");
}

TEST_F(AutoSchedulerUT, align_vectorized_strides_last_zero) {
  af::AscGraph graph("align_vectorize");
  auto s0 = af::Symbol(10);
  auto s1 = af::Symbol(10);
  auto s2 = af::Symbol(20);
  auto s3 = af::Symbol(1);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);

  af::ascir_op::Data data("data", graph);
  data.y.dtype = ge::DT_FLOAT16;
  data.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  data.attr.api.type = af::ApiType::kAPITypeBuffer;

  af::ascir_op::Load load("load_i");
  load.x = data.y;
  load.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *load.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *load.y.repeats = {s0, s1, s2, s3};
  *load.y.strides = {s1 * s2, s2, One, Zero};
  load.attr.api.compute_type = af::ComputeType::kComputeLoad;
  *load.y.vectorized_axis = {z1.id, z2.id, z3.id};

  AlignmentHandler handler;
  EXPECT_EQ(handler.AlignVectorizedStrides(graph), af::SUCCESS);

  auto load_node = graph.FindNode("load_i");
  std::vector<af::Expression> golden_stride = {af::Symbol(20), One, Zero};
  EXPECT_EQ(load_node->outputs[0].attr.vectorized_strides, golden_stride);
}

TEST_F(AutoSchedulerUT, align_vectorized_strides_store_same_with_input) {
  af::AscGraph graph("align_vectorize");
  auto s0 = af::Symbol(7);
  auto s1 = af::Symbol(9);
  auto s2 = af::Symbol(8);
  auto s3 = af::Symbol(1);

  auto s10 = af::Symbol(10);

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);

  af::ascir_op::Data data("data", graph);
  data.y.dtype = ge::DT_FLOAT16;
  data.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  data.attr.api.type = af::ApiType::kAPITypeBuffer;

  af::ascir_op::Load load("load");
  load.y.dtype = ge::DT_FLOAT16;
  load.x = data.y;
  load.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *load.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *load.y.repeats = {s0, s1, s2, s10};
  *load.y.strides = {s1 * s2 * s10, s2 * s10, s10, One};
  load.attr.api.compute_type = af::ComputeType::kComputeLoad;
  *load.y.vectorized_axis = {z1.id, z2.id, z3.id};

  af::ascir_op::Store store("store");
  store.y.dtype = ge::DT_FLOAT16;
  store.x = load.y;
  store.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *store.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *store.y.repeats = {s0, s1, s2, s3};
  *store.y.strides = {s2 * s3, s3, One, Zero};
  store.attr.api.compute_type = af::ComputeType::kComputeStore;
  *store.y.vectorized_axis = {z1.id, z2.id, z3.id};

  AlignmentHandler handler;
  EXPECT_EQ(handler.AlignVectorizedStrides(graph), af::SUCCESS);

  auto s16 = af::Symbol(16);
  std::vector<af::Expression> golden_stride = {s2 * s16, s16, One};

  auto load_node = graph.FindNode("load");
  EXPECT_EQ(load_node->outputs[0].attr.vectorized_strides, golden_stride);

  auto store_node = graph.FindNode("store");
  EXPECT_EQ(store_node->outputs[0].attr.vectorized_strides, golden_stride);
}

TEST_F(AutoSchedulerUT, align_vectorized_strides_by_repeat) {
  af::AscGraph graph("LoadAbsStore");
  auto s0 = af::Symbol(10);
  auto s1 = af::Symbol(9);

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  af::ascir_op::Data x("x", graph);
  x.attr.sched.axis = {z0.id, z1.id};
  x.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  x.attr.api.type = af::ApiType::kAPITypeBuffer;
  x.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Load load("load");
  load.x = x.y;
  load.attr.sched.axis = {z0.id, z1.id};
  load.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load.y.dtype = ge::DT_FLOAT;
  *load.y.axis = {z0.id, z1.id};
  *load.y.repeats = {s0, s1};
  *load.y.strides = {s1, One};

  af::ascir_op::Load load1("load1");
  load1.x = x.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {s0, Zero};
  *load1.y.strides = {One, Zero};

  af::ascir_op::Concat concat("concat");
  concat.x = {load.y, load1.y};
  concat.attr.sched.axis = {z0.id, z1.id};
  concat.attr.api.compute_type = af::ComputeType::kComputeConcat;
  concat.y.dtype = ge::DT_FLOAT;
  *concat.y.axis = {z0.id, z1.id};
  *concat.y.repeats = {s0, s1 + One};
  *concat.y.strides = {s1 + One, One};

  af::ascir_op::Store store("store");
  graph.AddNode(store);
  store.x = concat.y;
  store.attr.sched.axis = {z0.id, z1.id};
  store.attr.api.compute_type = af::ComputeType::kComputeStore;
  store.y.dtype = ge::DT_FLOAT16;
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1 + One};
  *store.y.strides = {s1 + One, One};

  af::ascir_op::Output y("y");
  y.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  y.attr.api.type = af::ApiType::kAPITypeBuffer;
  y.x = store.y;
  y.attr.sched.axis = {z0.id, z1.id};

  for (auto n : graph.GetAllNodes()) {
    if (optimize::ScheduleUtils::IsBuffer(n)) {
      continue;
    }
    n->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  }

  AlignmentHandler handler;
  EXPECT_EQ(handler.AlignVectorizedStrides(graph), af::SUCCESS);

  auto load1_node = graph.FindNode("load1");
  EXPECT_EQ(std::string(load1_node->outputs[0].attr.vectorized_strides[0].Str().get()), "1");
  EXPECT_EQ(std::string(load1_node->outputs[0].attr.vectorized_strides[1].Str().get()), "0");

  auto load_node = graph.FindNode("load");
  EXPECT_EQ(std::string(load_node->outputs[0].attr.vectorized_strides[0].Str().get()), "9");
  EXPECT_EQ(std::string(load_node->outputs[0].attr.vectorized_strides[1].Str().get()), "1");
}

TEST_F(AutoSchedulerUT, merge_axis_load_broadcast) {
  af::AscGraph graph("graph_merge_axis");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar("s3");
  auto s4 = graph.CreateSizeVar("s4");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);
  auto z4 = graph.CreateAxis("z4", s4);

  af::ascir_op::Data data("data", graph);
  data.y.dtype = ge::DT_FLOAT16;
  data.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  data.attr.api.type = af::ApiType::kAPITypeBuffer;

  af::ascir_op::Load load("load_i");
  load.x = data.y;
  load.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *load.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *load.y.repeats = {s0, af::ops::One, af::ops::One, s3, s4};
  *load.y.strides = {s3 * s4, af::ops::Zero, af::ops::Zero, s4, af::ops::One};
  load.attr.api.compute_type = af::ComputeType::kComputeLoad;

  af::ascir_op::Broadcast broadcast("broadcast");
  broadcast.x = load.y;
  broadcast.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *broadcast.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *broadcast.y.repeats = {s0, s1, s2, s3, s4};
  *broadcast.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, af::ops::One};
  broadcast.attr.api.compute_type = af::ComputeType::kComputeBroadcast;

  af::ascir_op::Store store("store");
  store.x = broadcast.y;
  store.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  store.attr.api.compute_type = af::ComputeType::kComputeStore;
  store.y.dtype = ge::DT_FLOAT16;
  *store.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *store.y.repeats = {s0, s1, s2, s3, s4};
  *store.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, af::ops::One};

  af::ascir_op::Output y("y");
  y.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  y.attr.api.type = af::ApiType::kAPITypeBuffer;
  y.x = store.y;
  y.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};

  EXPECT_EQ(Optimizer::MergeContinuousAxis(graph), af::SUCCESS);

  std::vector<autoschedule::AutoScheduleOutput> results;
  AutoSchedule schedule(graph, results);
  schedule.DoAutoSchedule();
  EXPECT_EQ(schedule.schd_outputs_.size(), 5);

  EXPECT_EQ(load.attr.sched.axis.size(), 3);
  EXPECT_EQ(load.y.axis->size(), 3);
  std::vector<af::Expression> golden_repeats{s0, af::ops::One, s3 * s4};
  EXPECT_EQ(*load.y.repeats, golden_repeats);
  std::vector<af::Expression> golden_strides{s3 * s4, af::ops::Zero, af::ops::One};
  EXPECT_EQ(*load.y.strides, golden_strides);

  EXPECT_EQ(broadcast.attr.sched.axis.size(), 3);
  EXPECT_EQ(broadcast.y.axis->size(), 3);
  std::vector<af::Expression> golden_repeats_broadcast{s0, s1 * s2, s3 * s4};
  EXPECT_EQ(*broadcast.y.repeats, golden_repeats_broadcast);
  std::vector<af::Expression> golden_strides_broadcast{s1 * s2 * s3 * s4, s3 * s4, af::ops::One};
  EXPECT_EQ(*broadcast.y.strides, golden_strides_broadcast);
}

TEST_F(AutoSchedulerUT, prune_zero_axis_skip_last_one) {
  af::AscGraph graph("graph_merge_axis");
  auto s0 = af::sym::kSymbolOne;
  auto s1 = af::sym::kSymbolOne;
  auto s2 = af::sym::kSymbolOne;
  auto s3 = af::sym::kSymbolOne;
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);

  af::ascir_op::Data data("data", graph);
  data.y.dtype = ge::DT_FLOAT16;
  data.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  data.attr.api.type = af::ApiType::kAPITypeBuffer;

  af::ascir_op::Load load("load_i");
  load.x = data.y;
  load.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *load.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *load.y.repeats = {s0, s1, s2, s3};
  *load.y.strides = {af::sym::kSymbolZero, af::sym::kSymbolZero, af::sym::kSymbolZero, af::sym::kSymbolZero};
  load.attr.api.compute_type = af::ComputeType::kComputeLoad;

  std::vector<autoschedule::AutoScheduleOutput> results;
  AutoSchedule schedule(graph, results);
  schedule.axes_group_.x_group = {-1};
  schedule.axes_group_.y_group = {0, 1, 2, 3};
  schedule.axes_group_.r_group = {-1};
  schedule.axes_group_.axes_order = {0, 1, 2, 3};

  std::vector<TilingCase> tiling_cases;
  schedule.GenTilingCase(tiling_cases);
  EXPECT_EQ(tiling_cases.size(), 4UL);
  schedule.PruneTilingCase(tiling_cases);
  ASSERT_EQ(tiling_cases.size(), 1UL);
  auto all_axis = graph.GetAllAxis();
  EXPECT_EQ(tiling_cases[0].ub_tiling_id_y, 3);
}

TEST_F(AutoSchedulerUT, reorder_vectorized_axes_ok) {
  af::AscGraph graph("reorder_vectorized_axes");
  auto size2 = af::Symbol(2);
  auto z0 = graph.CreateAxis("z0", size2);
  auto z1 = graph.CreateAxis("z1", size2);
  auto z2 = graph.CreateAxis("z2", size2);
  auto z3 = graph.CreateAxis("z3", size2);
  auto z4 = graph.CreateAxis("z4", size2);
  auto z5 = graph.CreateAxis("z5", size2);
  auto z6 = graph.CreateAxis("z6", size2);
  auto z7 = graph.CreateAxis("z7", size2);
  auto z8 = graph.CreateAxis("z8", size2);
  auto z9 = graph.CreateAxis("z9", size2);

  af::ascir_op::Data data("data", graph);
  data.y.dtype = ge::DT_FLOAT16;
  data.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  data.attr.api.type = af::ApiType::kAPITypeBuffer;

  af::ascir_op::Load load("load_i");
  load.x = data.y;
  load.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id, z6.id, z7.id, z8.id, z9.id};
  *load.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id, z6.id, z7.id, z8.id, z9.id};
  *load.y.repeats = {size2, size2, size2, size2, size2, size2, size2, size2, size2, size2};
  *load.y.strides = {af::Symbol(512), af::Symbol(256), af::Symbol(128), af::Symbol(64), af::Symbol(32),
                     af::Symbol(16),  af::Symbol(8),   af::Symbol(4),   af::Symbol(2),  af::ops::One};
  load.attr.api.compute_type = af::ComputeType::kComputeLoad;

  af::ascir_op::Max max("max");
  max.x = load.y;
  max.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id, z6.id, z7.id, z8.id, z9.id};
  *max.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id, z6.id, z7.id, z8.id, z9.id};
  *max.y.repeats = {size2, One, size2, One, size2, One, size2, One, size2, One};
  *max.y.strides = {af::Symbol(16), af::ops::Zero, af::Symbol(8), af::ops::Zero, af::Symbol(4),
                    af::ops::Zero,  af::Symbol(2), af::ops::Zero, af::ops::One,  af::ops::Zero};
  max.attr.api.compute_type = af::ComputeType::kComputeReduce;

  AxisGroup axes_group;
  axes_group.y_group = {z0.id, z2.id, z4.id, z6.id, z8.id};
  axes_group.r_group = {z1.id, z3.id, z5.id, z7.id, z9.id};
  axes_group.axes_order = {0, 2, 4, 6, 8, 1, 3, 5, 7, 9};

  TilingCase tiling_case;
  tiling_case.ub_tiling_id_y = z0.id;
  tiling_case.ub_tiling_id_r = z9.id;

  Scheduler schedule(graph, axes_group, tiling_case, true, optimize::ReduceTemplateType::kRCore);
  EXPECT_EQ(schedule.DoScheduler(), 0);

  std::vector<int64_t> golden_axis{15, 16, 14, 11, 2, 4, 6, 8, 13};
  std::vector<int64_t> golden_output{15, 16, 11, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13};
  std::vector<int64_t> golden_vectorized_axis{11, 2, 4, 6, 8, 13};

  auto load_node = graph.FindNode("load_i");
  EXPECT_EQ(load_node->attr.sched.axis, golden_axis);
  EXPECT_EQ(load_node->outputs[0].attr.axis, golden_output);
  EXPECT_EQ(load_node->outputs[0].attr.vectorized_axis, golden_vectorized_axis);

  auto max_node = graph.FindNode("max");
  EXPECT_EQ(max_node->attr.sched.axis, golden_axis);
  EXPECT_EQ(max_node->outputs[0].attr.axis, golden_output);
  EXPECT_EQ(max_node->outputs[0].attr.vectorized_axis, golden_vectorized_axis);
}

TEST_F(AutoSchedulerUT, reorder_vectorized_reduce_full_load) {
  af::AscGraph graph("reorder_vectorized_axes");
  auto size2 = af::Symbol(2);
  auto z0 = graph.CreateAxis("z0", size2);
  auto z1 = graph.CreateAxis("z1", size2);
  auto z2 = graph.CreateAxis("z2", size2);
  auto z3 = graph.CreateAxis("z3", size2);
  auto z4 = graph.CreateAxis("z4", size2);
  auto z5 = graph.CreateAxis("z5", size2);
  auto z6 = graph.CreateAxis("z6", size2);
  auto z7 = graph.CreateAxis("z7", size2);
  auto z8 = graph.CreateAxis("z8", size2);
  auto z9 = graph.CreateAxis("z9", size2);

  af::ascir_op::Data data("data", graph);
  data.y.dtype = ge::DT_FLOAT16;
  data.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  data.attr.api.type = af::ApiType::kAPITypeBuffer;

  af::ascir_op::Load load("load_i");
  load.x = data.y;
  load.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id, z6.id, z7.id, z8.id, z9.id};
  *load.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id, z6.id, z7.id, z8.id, z9.id};
  *load.y.repeats = {size2, size2, size2, size2, size2, size2, size2, size2, size2, size2};
  *load.y.strides = {af::Symbol(512), af::Symbol(256), af::Symbol(128), af::Symbol(64), af::Symbol(32),
                     af::Symbol(16),  af::Symbol(8),   af::Symbol(4),   af::Symbol(2),  af::ops::One};
  load.attr.api.compute_type = af::ComputeType::kComputeLoad;

  af::ascir_op::Max max("max");
  max.x = load.y;
  max.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id, z6.id, z7.id, z8.id, z9.id};
  *max.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id, z6.id, z7.id, z8.id, z9.id};
  *max.y.repeats = {size2, One, size2, One, size2, One, size2, One, size2, One};
  *max.y.strides = {af::ops::Zero, af::Symbol(16), af::ops::Zero, af::Symbol(8), af::ops::Zero,
                    af::Symbol(4), af::ops::Zero,  af::Symbol(2), af::ops::Zero, af::ops::One};
  max.attr.api.compute_type = af::ComputeType::kComputeReduce;

  AxisGroup axes_group;
  axes_group.y_group = {z1.id, z3.id, z5.id, z7.id, z9.id};
  axes_group.n_group = {z0.id, z2.id, z4.id, z6.id, z8.id};
  axes_group.axes_order = {1, 3, 5, 7, 9};

  TilingCase tiling_case;
  tiling_case.ub_tiling_id_y = z5.id;

  Scheduler schedule(graph, axes_group, tiling_case, false, optimize::ReduceTemplateType::kAllLoad);
  EXPECT_EQ(schedule.DoScheduler(), 0);

  std::vector<int64_t> golden_axis{13, 14, 0, 2, 4, 6, 8, 11, 7, 9};
  std::vector<int64_t> golden_output{0, 1, 2, 3, 4, 10, 11, 6, 7, 8, 9};
  std::vector<int64_t> golden_vectorized_axis{0, 2, 4, 6, 8, 11, 7, 9};

  auto load_node = graph.FindNode("load_i");
  EXPECT_EQ(load_node->attr.sched.axis, golden_axis);
  EXPECT_EQ(load_node->outputs[0].attr.axis, golden_output);
  EXPECT_EQ(load_node->outputs[0].attr.vectorized_axis, golden_vectorized_axis);

  auto max_node = graph.FindNode("max");
  EXPECT_EQ(max_node->attr.sched.axis, golden_axis);
  EXPECT_EQ(max_node->outputs[0].attr.axis, golden_output);
  EXPECT_EQ(max_node->outputs[0].attr.vectorized_axis, golden_vectorized_axis);
}

TEST_F(AutoSchedulerUT, RemoveDuplicatedAxisFromGroup) {
  af::AscGraph graph("reorder_vectorized_axes");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar("s3");
  auto s4 = graph.CreateSizeVar("s4");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);
  auto z4 = graph.CreateAxis("z4", s4);

  AxisGroup axes_group;
  axes_group.x_group = {z0.id, z2.id, z4.id};
  axes_group.y_group = {z1.id, z3.id, z4.id};
  axes_group.axes_order = {0, 2, 4, 1, 3, 4};

  // remove x
  TilingCase case1;
  case1.ub_tiling_id_x = z0.id;
  case1.ub_tiling_id_y = z1.id;
  Scheduler sch1(graph, axes_group, case1);
  sch1.RemoveDuplicatedAxisFromGroup();
  std::vector<int64_t> golden_x1 = {z0.id, z2.id};
  std::vector<int64_t> golden_y1 = {z1.id, z3.id, z4.id};
  std::vector<size_t> golden_order1 = {0, 2, 1, 3, 4};
  EXPECT_EQ(sch1.axes_group_.x_group, golden_x1);
  EXPECT_EQ(sch1.axes_group_.y_group, golden_y1);
  EXPECT_EQ(sch1.axes_group_.axes_order, golden_order1);

  TilingCase case2;
  case2.ub_tiling_id_x = z0.id;
  case2.ub_tiling_id_y = z4.id;
  Scheduler sch2(graph, axes_group, case2);
  sch2.RemoveDuplicatedAxisFromGroup();
  std::vector<int64_t> golden_x2 = {z0.id, z2.id};
  std::vector<int64_t> golden_y2 = {z1.id, z3.id, z4.id};
  std::vector<size_t> golden_order2 = {0, 2, 1, 3, 4};
  EXPECT_EQ(sch2.axes_group_.x_group, golden_x2);
  EXPECT_EQ(sch2.axes_group_.y_group, golden_y2);
  EXPECT_EQ(sch2.axes_group_.axes_order, golden_order2);

  TilingCase case3;
  case3.ub_tiling_id_x = z4.id;
  case3.ub_tiling_id_y = z3.id;
  Scheduler sch3(graph, axes_group, case3);
  sch3.RemoveDuplicatedAxisFromGroup();
  std::vector<int64_t> golden_x3 = {z0.id, z2.id, z4.id};
  std::vector<int64_t> golden_y3 = {z1.id, z3.id};
  std::vector<size_t> golden_order3 = {0, 2, 4, 1, 3};
  EXPECT_EQ(sch3.axes_group_.x_group, golden_x3);
  EXPECT_EQ(sch3.axes_group_.y_group, golden_y3);
  EXPECT_EQ(sch3.axes_group_.axes_order, golden_order3);
}

TEST_F(AutoSchedulerUT, AutoSchedulerUT_reorder_vectorized_axes_ok) {
  af::AscGraph graph("reorder_vectorized_axes");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  af::ascir_op::Data data("data", graph);
  data.y.dtype = ge::DT_FLOAT16;
  data.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  data.attr.api.type = af::ApiType::kAPITypeBuffer;

  af::ascir_op::Load load("load");
  load.x = data.y;
  load.attr.sched.axis = {z0.id, z1.id};
  *load.y.axis = {z0.id, z1.id};
  *load.y.repeats = {s0, s1};
  *load.y.strides = {s1, af::sym::kSymbolOne};
  load.attr.api.compute_type = af::ComputeType::kComputeLoad;

  af::ascir_op::Max transpose("transpose");
  transpose.x = load.y;
  transpose.attr.sched.axis = {z0.id, z1.id};
  *transpose.y.axis = {z1.id, z0.id};
  *transpose.y.repeats = {s1, s0};
  *transpose.y.strides = {s0, af::sym::kSymbolOne};
  transpose.attr.api.compute_type = af::ComputeType::kComputeTranspose;

  af::ascir_op::Store store("store");
  store.x = transpose.y;
  store.attr.sched.axis = {z0.id, z1.id};
  *store.y.axis = {z1.id, z0.id};
  *store.y.repeats = {s1, s0};
  *store.y.strides = {s0, af::sym::kSymbolOne};
  store.attr.api.compute_type = af::ComputeType::kComputeStore;

  af::ascir_op::Output y("y");
  y.x = store.y;
  y.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  y.attr.api.type = af::ApiType::kAPITypeBuffer;

  std::vector<autoschedule::AutoScheduleOutput> impl_graphs;
  AutoSchedule auto_schedule(graph, impl_graphs);

  auto_schedule.axes_group_.x_group = {z0.id};
  auto_schedule.axes_group_.y_group = {z1.id};
  auto_schedule.axes_group_.r_group = {-1};
  auto_schedule.axes_group_.axes_order = {0, 1};

  std::vector<TilingCase> tiling_cases;
  auto_schedule.GenTilingCase(tiling_cases);

  ASSERT_EQ(tiling_cases.size(), 1UL);
  Scheduler scheduler(graph, auto_schedule.axes_group_, tiling_cases[0UL]);
  EXPECT_EQ(scheduler.DoScheduler(), af::SUCCESS);

  auto all_axis = graph.GetAllAxis();
  int64_t z0t_id = std::numeric_limits<int64_t>::max();
  int64_t z1t_id = std::numeric_limits<int64_t>::max();
  for (auto &axis : all_axis) {
    if (axis->name == "z0t") {
      z0t_id = axis->id;
    }
    if (axis->name == "z1t") {
      z1t_id = axis->id;
    }
  }

  auto load_node = graph.FindNode("load");
  ASSERT_NE(load_node, nullptr);
  EXPECT_EQ(load_node->attr.sched.axis[2], z0t_id);
  EXPECT_EQ(load_node->attr.sched.axis[3], z1t_id);
  EXPECT_EQ(load_node->outputs[0].attr.vectorized_axis[0], z0t_id);
  EXPECT_EQ(load_node->outputs[0].attr.vectorized_axis[1], z1t_id);

  auto transpose_node = graph.FindNode("transpose");
  ASSERT_NE(transpose_node, nullptr);
  EXPECT_EQ(transpose_node->attr.sched.axis[2], z1t_id);
  EXPECT_EQ(transpose_node->attr.sched.axis[3], z0t_id);
  EXPECT_EQ(transpose_node->outputs[0].attr.vectorized_axis[0], z1t_id);
  EXPECT_EQ(transpose_node->outputs[0].attr.vectorized_axis[1], z0t_id);

  auto store_node = graph.FindNode("store");
  ASSERT_NE(store_node, nullptr);
  EXPECT_EQ(store_node->attr.sched.axis[2], z1t_id);
  EXPECT_EQ(store_node->attr.sched.axis[3], z0t_id);
  EXPECT_EQ(store_node->outputs[0].attr.vectorized_axis[0], z1t_id);
  EXPECT_EQ(store_node->outputs[0].attr.vectorized_axis[1], z0t_id);
}

TEST_F(AutoSchedulerUT, DoTilingOk) {
  af::AscGraph graph("apply_tiling_pk");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar("s3");
  auto s4 = graph.CreateSizeVar("s4");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);
  auto z4 = graph.CreateAxis("z4", s4);
  af::ascir_op::Data data("data", graph);
  data.y.dtype = ge::DT_FLOAT16;
  data.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  data.attr.api.type = af::ApiType::kAPITypeBuffer;

  af::ascir_op::Load load("load_i");
  load.x = data.y;
  load.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *load.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *load.y.repeats = {s0, s1, s2, s3, s4};
  *load.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, af::ops::One};
  load.attr.api.compute_type = af::ComputeType::kComputeLoad;

  af::ascir_op::Max max("max");
  max.x = load.y;
  max.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *max.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *max.y.repeats = {s0, af::ops::One, s2, One, s4};
  *max.y.strides = {s2 * s4, af::ops::Zero, s4, Zero, af::ops::One};
  max.attr.api.compute_type = af::ComputeType::kComputeReduce;

  af::ascir_op::Store store("store");
  store.x = max.y;
  store.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  store.attr.api.compute_type = af::ComputeType::kComputeStore;
  store.y.dtype = ge::DT_FLOAT16;
  *store.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *store.y.repeats = {s0, af::ops::One, s2, One, s4};
  *store.y.strides = {s2 * s4, af::ops::Zero, s4, Zero, af::ops::One};

  af::ascir_op::Output y("y");
  y.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  y.attr.api.type = af::ApiType::kAPITypeBuffer;
  y.x = store.y;
  y.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};

  std::vector<autoschedule::AutoScheduleOutput> impl_graphs;
  AutoSchedule autoschedule(graph, impl_graphs);
  autoschedule.axes_group_.x_group = {z0.id};
  autoschedule.axes_group_.y_group = {z4.id};
  autoschedule.axes_group_.r_group = {z1.id, z3.id};
  autoschedule.axes_group_.n_group = {z2.id};
  autoschedule.axes_group_.axes_order = {0, 4, 1, 3};

  std::vector<TilingCase> tiling_cases;
  autoschedule.GenTilingCase(tiling_cases);
  EXPECT_EQ(tiling_cases.size(), 2UL);

  Scheduler scheduler(graph, autoschedule.axes_group_, tiling_cases[0UL]);
  EXPECT_EQ(scheduler.DoScheduler(), 0);
}

TEST_F(AutoSchedulerUT, Autoschedule_autoschedule_softmax_fusion_axesgroup) {
  af::AscGraph graph("SoftmaxFusion");
  Construct_Softmax(graph);

  af::AscGraph except_graph("SoftmaxFusion_B0Y0");
  except_graph.CopyFrom(graph);

  auto store = graph.FindNode("b3_store");
  std::vector<af::AxisId> axes = store->attr.sched.axis;
  std::vector<af::AxisId> y_group = {axes[0]};
  std::vector<af::AxisId> r_group = {axes[1]};

  AxisGroup axis_group;
  ASSERT_EQ(TilingGroup::GenTilingGroup(graph, axis_group), 0);
  EXPECT_EQ(axis_group.x_group.size(), 0);
  EXPECT_EQ(axis_group.y_group.size(), 1);
  EXPECT_EQ(axis_group.y_group, y_group);
  EXPECT_EQ(axis_group.r_group.size(), 1);
}

TEST_F(AutoSchedulerUT, Autoschedule_autoschedule_softmax_fusion_tilingcase) {
  af::AscGraph graph("SoftmaxFusion");
  Construct_Softmax(graph);

  af::AscGraph except_graph("SoftmaxFusion_B0Y0");
  except_graph.CopyFrom(graph);

  auto store = graph.FindNode("b3_store");
  std::vector<af::AxisId> axes = store->attr.sched.axis;

  std::vector<autoschedule::AutoScheduleOutput> impl_graphs;
  AutoSchedule autoschedule(graph, impl_graphs);
  EXPECT_EQ(TilingGroup::GenTilingGroup(graph, autoschedule.axes_group_), 0);
  TilingGroup::NormGroup(autoschedule.axes_group_);
  std::vector<TilingCase> tiling_cases;
  autoschedule.GenTilingCase(tiling_cases);
  ASSERT_EQ(tiling_cases.size(), 1);
  EXPECT_EQ(tiling_cases[0].ub_tiling_id_x, -1);
  EXPECT_EQ(tiling_cases[0].ub_tiling_id_y, axes[0]);
  EXPECT_EQ(tiling_cases[0].ub_tiling_id_r, axes[1]);
  EXPECT_EQ(tiling_cases[0].block_tiling_id, 0);
  EXPECT_EQ(tiling_cases[0].reduce_is_block, false);
}

TEST_F(AutoSchedulerUT, Autoschedule_autoschedule_softmax_fusion) {
  af::AscGraph graph("SoftmaxFusion");
  Construct_Softmax(graph);

  af::AscGraph except_graph("SoftmaxFusion_B0Y0");
  except_graph.CopyFrom(graph);

  std::vector<autoschedule::AutoScheduleOutput> impl_graphs;
  AutoSchedule autoschedule(graph, impl_graphs);
  autoschedule.DoAutoSchedule();
  EXPECT_EQ(impl_graphs.size(), 1);
  auto dump_graph = utils::DebugStr(impl_graphs[0].scheduled_graph);

  EXPECT_TRUE(ScheduleUtils::IsReduceArFullLoad(impl_graphs[0].scheduled_graph));
  EXPECT_FALSE(ScheduleUtils::IsNormStruct(impl_graphs[0].scheduled_graph));
}

TEST_F(AutoSchedulerUT, Autoschedule_autoschedule_normal_struct) {
  af::AscGraph graph("NormalStructFussion");
  Construct_Normal_Struct(graph);

  std::vector<autoschedule::AutoScheduleOutput> impl_graphs;
  AutoSchedule autoschedule(graph, impl_graphs);
  autoschedule.DoAutoSchedule();
  EXPECT_EQ(impl_graphs.size(), 1);
  auto dump_graph = utils::DebugStr(impl_graphs[0].scheduled_graph);
  EXPECT_TRUE(ScheduleUtils::IsNormStruct(impl_graphs[0].scheduled_graph));
}

TEST_F(AutoSchedulerUT, Autoschedule_autoschedule_remove_continues_broadcast_scalar) {
  af::AscGraph graph("Autoschedule_autoschedule_remove_continues_broadcast_scalar");
  Construct_ContinuesBroadcastScalarFusion(graph);

  std::vector<autoschedule::AutoScheduleOutput> impl_graphs;
  AutoSchedule autoschedule(graph, impl_graphs);
  autoschedule.DoAutoSchedule();
  EXPECT_EQ(impl_graphs.size(), 9);

  EXPECT_EQ(impl_graphs[0].scheduled_graph.FindNode("brc0"), nullptr);
  EXPECT_EQ(impl_graphs[0].scheduled_graph.FindNode("brc1"), nullptr);
  EXPECT_EQ(impl_graphs[0].scheduled_graph.FindNode("brc2"), nullptr);
  EXPECT_EQ(impl_graphs[0].scheduled_graph.FindNode("brc3"), nullptr);
  EXPECT_NE(impl_graphs[0].scheduled_graph.FindNode("brc4"), nullptr);
  const auto &impl0_brc4 = impl_graphs[0].scheduled_graph.FindNode("brc4");
  EXPECT_EQ(std::string(impl0_brc4->outputs[0].attr.repeats[0].Str().get()), "(s0 / (z0Tb_size * z0t_size))");
  EXPECT_EQ(std::string(impl0_brc4->outputs[0].attr.repeats[1].Str().get()), "z0Tb_size");
  EXPECT_EQ(std::string(impl0_brc4->outputs[0].attr.repeats[2].Str().get()), "z0t_size");
  EXPECT_EQ(std::string(impl0_brc4->outputs[0].attr.repeats[3].Str().get()), "s1");
  EXPECT_EQ(std::string(impl0_brc4->outputs[0].attr.repeats[4].Str().get()), "s2");
  EXPECT_EQ(std::string(impl0_brc4->outputs[0].attr.repeats[5].Str().get()), "s3");
  EXPECT_EQ(std::string(impl0_brc4->outputs[0].attr.repeats[6].Str().get()), "s4");

  EXPECT_EQ(impl_graphs[1].scheduled_graph.FindNode("brc0"), nullptr);
  EXPECT_EQ(impl_graphs[1].scheduled_graph.FindNode("brc1"), nullptr);
  EXPECT_EQ(impl_graphs[1].scheduled_graph.FindNode("brc2"), nullptr);
  EXPECT_NE(impl_graphs[1].scheduled_graph.FindNode("brc3"), nullptr);
  EXPECT_EQ(impl_graphs[1].scheduled_graph.FindNode("brc4"), nullptr);
  const auto &impl1_brc3 = impl_graphs[1].scheduled_graph.FindNode("brc3");
  // EXPECT_EQ(std::string(impl1_brc3->outputs[0].attr.repeats[0].Str().get()), "s0");
  EXPECT_EQ(std::string(impl1_brc3->outputs[0].attr.repeats[1].Str().get()), "(s1 / (z1t_size))");
  EXPECT_EQ(std::string(impl1_brc3->outputs[0].attr.repeats[2].Str().get()), "z1t_size");
  EXPECT_EQ(std::string(impl1_brc3->outputs[0].attr.repeats[3].Str().get()), "s2");
  EXPECT_EQ(std::string(impl1_brc3->outputs[0].attr.repeats[4].Str().get()), "s3");
  EXPECT_EQ(std::string(impl1_brc3->outputs[0].attr.repeats[5].Str().get()), "s4");

  EXPECT_EQ(impl_graphs[2].scheduled_graph.FindNode("brc0"), nullptr);
  EXPECT_EQ(impl_graphs[2].scheduled_graph.FindNode("brc1"), nullptr);
  EXPECT_NE(impl_graphs[2].scheduled_graph.FindNode("brc2"), nullptr);
  EXPECT_EQ(impl_graphs[2].scheduled_graph.FindNode("brc3"), nullptr);
  EXPECT_EQ(impl_graphs[2].scheduled_graph.FindNode("brc4"), nullptr);
  const auto &impl2_brc2 = impl_graphs[2].scheduled_graph.FindNode("brc2");
  // EXPECT_EQ(std::string(impl2_brc2->outputs[0].attr.repeats[0].Str().get()), "s0");
  // EXPECT_EQ(std::string(impl2_brc2->outputs[0].attr.repeats[1].Str().get()), "s1");
  EXPECT_EQ(std::string(impl2_brc2->outputs[0].attr.repeats[2].Str().get()), "(s2 / (z2t_size))");
  EXPECT_EQ(std::string(impl2_brc2->outputs[0].attr.repeats[3].Str().get()), "z2t_size");
  EXPECT_EQ(std::string(impl2_brc2->outputs[0].attr.repeats[4].Str().get()), "s3");
  EXPECT_EQ(std::string(impl1_brc3->outputs[0].attr.repeats[5].Str().get()), "s4");

  EXPECT_EQ(impl_graphs[3].scheduled_graph.FindNode("brc0"), nullptr);
  EXPECT_NE(impl_graphs[3].scheduled_graph.FindNode("brc1"), nullptr);
  EXPECT_EQ(impl_graphs[3].scheduled_graph.FindNode("brc2"), nullptr);
  EXPECT_EQ(impl_graphs[3].scheduled_graph.FindNode("brc3"), nullptr);
  EXPECT_EQ(impl_graphs[3].scheduled_graph.FindNode("brc4"), nullptr);
  const auto &impl3_brc1 = impl_graphs[3].scheduled_graph.FindNode("brc1");
  // EXPECT_EQ(std::string(impl3_brc1->outputs[0].attr.repeats[0].Str().get()), "s0");
  // EXPECT_EQ(std::string(impl3_brc1->outputs[0].attr.repeats[1].Str().get()), "s1");
  // EXPECT_EQ(std::string(impl3_brc1->outputs[0].attr.repeats[2].Str().get()), "s2");
  EXPECT_EQ(std::string(impl3_brc1->outputs[0].attr.repeats[3].Str().get()), "(s3 / (z3t_size))");
  EXPECT_EQ(std::string(impl3_brc1->outputs[0].attr.repeats[4].Str().get()), "z3t_size");
  EXPECT_EQ(std::string(impl3_brc1->outputs[0].attr.repeats[5].Str().get()), "s4");

  EXPECT_NE(impl_graphs[4].scheduled_graph.FindNode("brc0"), nullptr);
  EXPECT_EQ(impl_graphs[4].scheduled_graph.FindNode("brc1"), nullptr);
  EXPECT_EQ(impl_graphs[4].scheduled_graph.FindNode("brc2"), nullptr);
  EXPECT_EQ(impl_graphs[4].scheduled_graph.FindNode("brc3"), nullptr);
  EXPECT_EQ(impl_graphs[4].scheduled_graph.FindNode("brc4"), nullptr);
  const auto &impl4_brc0 = impl_graphs[4].scheduled_graph.FindNode("brc0");
  // EXPECT_EQ(std::string(impl4_brc0->outputs[0].attr.repeats[0].Str().get()), "s0");
  // EXPECT_EQ(std::string(impl4_brc0->outputs[0].attr.repeats[1].Str().get()), "s1");
  // EXPECT_EQ(std::string(impl4_brc0->outputs[0].attr.repeats[2].Str().get()), "s2");
  // EXPECT_EQ(std::string(impl4_brc0->outputs[0].attr.repeats[3].Str().get()), "s3");
  EXPECT_EQ(std::string(impl4_brc0->outputs[0].attr.repeats[4].Str().get()), "(s4 / (z4t_size))");
  EXPECT_EQ(std::string(impl4_brc0->outputs[0].attr.repeats[5].Str().get()), "z4t_size");
}

TEST_F(AutoSchedulerUT, Autoschedule_autoschedule_remove_continues_broadcast_part_one) {
  af::AscGraph graph("Autoschedule_autoschedule_remove_continues_broadcast_part_one");
  Construct_ContinuesBroadcastPartBrcFusion(graph);

  std::vector<autoschedule::AutoScheduleOutput> impl_graphs;
  AutoSchedule autoschedule(graph, impl_graphs);
  autoschedule.DoAutoSchedule();
  EXPECT_EQ(impl_graphs.size(), 9);

  EXPECT_EQ(impl_graphs[0].scheduled_graph.FindNode("brc0"), nullptr);
  EXPECT_EQ(impl_graphs[0].scheduled_graph.FindNode("brc3"), nullptr);
  auto impl_grp_0_brc1 = impl_graphs[0].scheduled_graph.FindNode("brc1");
  EXPECT_NE(impl_grp_0_brc1, nullptr);
  EXPECT_EQ(impl_grp_0_brc1->GetAllInDataAnchorsSize(), 1);
  EXPECT_EQ(impl_grp_0_brc1->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "abs0");
  auto impl_grp_0_brc4 = impl_graphs[0].scheduled_graph.FindNode("brc4");
  EXPECT_NE(impl_grp_0_brc4, nullptr);
  EXPECT_EQ(impl_grp_0_brc4->GetAllInDataAnchorsSize(), 1);
  EXPECT_EQ(impl_grp_0_brc4->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "brc1");

  EXPECT_EQ(impl_graphs[1].scheduled_graph.FindNode("brc0"), nullptr);
  EXPECT_EQ(impl_graphs[1].scheduled_graph.FindNode("brc2"), nullptr);
  EXPECT_EQ(impl_graphs[1].scheduled_graph.FindNode("brc4"), nullptr);
  auto impl_grp_1_brc1 = impl_graphs[1].scheduled_graph.FindNode("brc1");
  EXPECT_NE(impl_grp_1_brc1, nullptr);
  EXPECT_EQ(impl_grp_1_brc1->GetAllInDataAnchorsSize(), 1);
  EXPECT_EQ(impl_grp_1_brc1->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "abs0");
  auto impl_grp_1_brc3 = impl_graphs[1].scheduled_graph.FindNode("brc3");
  EXPECT_NE(impl_grp_1_brc3, nullptr);
  EXPECT_EQ(impl_grp_1_brc3->GetAllInDataAnchorsSize(), 1);
  EXPECT_EQ(impl_grp_1_brc3->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "brc1");

  EXPECT_EQ(impl_graphs[2].scheduled_graph.FindNode("brc0"), nullptr);
  EXPECT_EQ(impl_graphs[2].scheduled_graph.FindNode("brc2"), nullptr);
  EXPECT_EQ(impl_graphs[2].scheduled_graph.FindNode("brc3"), nullptr);
  EXPECT_EQ(impl_graphs[2].scheduled_graph.FindNode("brc4"), nullptr);
  auto impl_grp_2_brc1 = impl_graphs[2].scheduled_graph.FindNode("brc1");
  EXPECT_NE(impl_grp_2_brc1, nullptr);
  EXPECT_EQ(impl_grp_2_brc1->GetAllInDataAnchorsSize(), 1);
  EXPECT_EQ(impl_grp_2_brc1->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "abs0");
  auto impl_grp_2_exp0 = impl_graphs[2].scheduled_graph.FindNode("exp0");
  EXPECT_NE(impl_grp_2_exp0, nullptr);
  EXPECT_EQ(impl_grp_2_exp0->GetAllInDataAnchorsSize(), 1);
  EXPECT_EQ(impl_grp_2_exp0->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "brc1");

  EXPECT_EQ(impl_graphs[3].scheduled_graph.FindNode("brc0"), nullptr);
  EXPECT_EQ(impl_graphs[3].scheduled_graph.FindNode("brc2"), nullptr);
  EXPECT_EQ(impl_graphs[3].scheduled_graph.FindNode("brc3"), nullptr);
  EXPECT_EQ(impl_graphs[3].scheduled_graph.FindNode("brc4"), nullptr);
  auto impl_grp_3_brc1 = impl_graphs[3].scheduled_graph.FindNode("brc1");
  EXPECT_NE(impl_grp_3_brc1, nullptr);
  EXPECT_EQ(impl_grp_3_brc1->GetAllInDataAnchorsSize(), 1);
  EXPECT_EQ(impl_grp_3_brc1->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "abs0");

  EXPECT_EQ(impl_graphs[4].scheduled_graph.FindNode("brc1"), nullptr);
  EXPECT_EQ(impl_graphs[4].scheduled_graph.FindNode("brc2"), nullptr);
  EXPECT_EQ(impl_graphs[4].scheduled_graph.FindNode("brc3"), nullptr);
  EXPECT_EQ(impl_graphs[4].scheduled_graph.FindNode("brc4"), nullptr);
  auto impl_grp_4_exp0 = impl_graphs[4].scheduled_graph.FindNode("exp0");
  EXPECT_NE(impl_grp_4_exp0, nullptr);
  EXPECT_EQ(impl_grp_4_exp0->GetAllInDataAnchorsSize(), 1);
  EXPECT_EQ(impl_grp_4_exp0->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "brc0");
}

TEST_F(AutoSchedulerUT, Autoschedule_autoschedule_remove_continues_broadcast_interval_baba) {
  af::AscGraph graph("Autoschedule_autoschedule_remove_continues_broadcast_interval_baba");
  Construct_ContinuesBroadcastIntervalBrcBABAFusion(graph);

  std::vector<autoschedule::AutoScheduleOutput> results;
  AutoSchedule autoschedule(graph, results);
  autoschedule.DoAutoSchedule();
  EXPECT_EQ(results.size(), 7);

  EXPECT_EQ(results[0].scheduled_graph.FindNode("brc0"), nullptr);
  EXPECT_NE(results[0].scheduled_graph.FindNode("brc1"), nullptr);
  auto impl_grp_0_brc1 = results[0].scheduled_graph.FindNode("brc1");
  EXPECT_EQ(impl_grp_0_brc1->GetAllInDataAnchorsSize(), 1);
  EXPECT_EQ(impl_grp_0_brc1->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "abs0");
}

TEST_F(AutoSchedulerUT, Autoschedule_autoschedule_remove_continues_broadcast_interval_abab) {
  af::AscGraph graph("Autoschedule_autoschedule_remove_continues_broadcast_interval_abab");
  Construct_ContinuesBroadcastIntervalBrcABABFusion(graph);

  std::vector<autoschedule::AutoScheduleOutput> impl_graphs;
  AutoSchedule autoschedule(graph, impl_graphs);
  autoschedule.DoAutoSchedule();
  EXPECT_EQ(impl_graphs.size(), 7);

  EXPECT_EQ(impl_graphs[0].scheduled_graph.FindNode("brc0"), nullptr);
  EXPECT_NE(impl_graphs[0].scheduled_graph.FindNode("brc1"), nullptr);
  auto impl_grp_0_brc1 = impl_graphs[0].scheduled_graph.FindNode("brc1");
  EXPECT_EQ(impl_grp_0_brc1->GetAllInDataAnchorsSize(), 1);
  EXPECT_EQ(impl_grp_0_brc1->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "abs0");
}

TEST_F(AutoSchedulerUT, Autoschedule_autoschedule_remove_continues_broadcast_interval_baab) {
  af::AscGraph graph("Autoschedule_autoschedule_remove_continues_broadcast_interval_baab");
  Construct_ContinuesBroadcastIntervalBrcBAABFusion(graph);

  std::vector<autoschedule::AutoScheduleOutput> impl_graphs;
  AutoSchedule autoschedule(graph, impl_graphs);
  autoschedule.DoAutoSchedule();
  EXPECT_EQ(impl_graphs.size(), 9);

  EXPECT_NE(impl_graphs[0].scheduled_graph.FindNode("brc0"), nullptr);
  EXPECT_NE(impl_graphs[0].scheduled_graph.FindNode("brc1"), nullptr);
  auto impl_grp_0_brc0 = impl_graphs[0].scheduled_graph.FindNode("brc0");
  EXPECT_EQ(impl_grp_0_brc0->GetAllInDataAnchorsSize(), 1);
  EXPECT_EQ(impl_grp_0_brc0->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "load0");

  auto impl_grp_0_brc1 = impl_graphs[0].scheduled_graph.FindNode("brc1");
  EXPECT_EQ(impl_grp_0_brc1->GetAllInDataAnchorsSize(), 1);
  EXPECT_EQ(impl_grp_0_brc1->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "brc0");
}

TEST_F(AutoSchedulerUT, Autoschedule_autoschedule_remove_continues_broadcast_part_one_in_middle) {
  af::AscGraph graph("Autoschedule_autoschedule_remove_continues_broadcast_part_one_in_middle");
  Construct_ContinuesBroadcastPartBrcInMiddleFusion(graph);

  std::vector<autoschedule::AutoScheduleOutput> impl_graphs;
  AutoSchedule autoschedule(graph, impl_graphs);
  autoschedule.DoAutoSchedule();
  EXPECT_EQ(impl_graphs.size(), 9);

  EXPECT_EQ(impl_graphs[0].scheduled_graph.FindNode("brc0"), nullptr);
  EXPECT_EQ(impl_graphs[0].scheduled_graph.FindNode("brc1"), nullptr);
  EXPECT_NE(impl_graphs[0].scheduled_graph.FindNode("brc2"), nullptr);
  auto impl_grp_0_brc2 = impl_graphs[0].scheduled_graph.FindNode("brc2");
  EXPECT_NE(impl_grp_0_brc2, nullptr);
  EXPECT_EQ(impl_grp_0_brc2->GetAllInDataAnchorsSize(), 1);
  EXPECT_EQ(impl_grp_0_brc2->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "abs0");

  EXPECT_EQ(impl_graphs[1].scheduled_graph.FindNode("brc0"), nullptr);
  EXPECT_EQ(impl_graphs[1].scheduled_graph.FindNode("brc1"), nullptr);
  EXPECT_NE(impl_graphs[1].scheduled_graph.FindNode("brc2"), nullptr);
  auto impl_grp_1_brc2 = impl_graphs[1].scheduled_graph.FindNode("brc2");
  EXPECT_NE(impl_grp_1_brc2, nullptr);
  EXPECT_EQ(impl_grp_1_brc2->GetAllInDataAnchorsSize(), 1);
  EXPECT_EQ(impl_grp_1_brc2->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "abs0");

  EXPECT_EQ(impl_graphs[2].scheduled_graph.FindNode("brc0"), nullptr);
  EXPECT_NE(impl_graphs[2].scheduled_graph.FindNode("brc1"), nullptr);
  EXPECT_EQ(impl_graphs[2].scheduled_graph.FindNode("brc2"), nullptr);
  auto impl_grp_2_brc1 = impl_graphs[2].scheduled_graph.FindNode("brc1");
  EXPECT_NE(impl_grp_2_brc1, nullptr);
  EXPECT_EQ(impl_grp_2_brc1->GetAllInDataAnchorsSize(), 1);
  EXPECT_EQ(impl_grp_2_brc1->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "abs0");
  auto impl_grp_2_exp0 = impl_graphs[2].scheduled_graph.FindNode("exp0");
  EXPECT_NE(impl_grp_2_exp0, nullptr);
  EXPECT_EQ(impl_grp_2_exp0->GetAllInDataAnchorsSize(), 1);
  EXPECT_EQ(impl_grp_2_exp0->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "brc1");

  EXPECT_NE(impl_graphs[3].scheduled_graph.FindNode("brc0"), nullptr);
  EXPECT_EQ(impl_graphs[3].scheduled_graph.FindNode("brc1"), nullptr);
  EXPECT_EQ(impl_graphs[3].scheduled_graph.FindNode("brc2"), nullptr);
  auto impl_grp_3_brc0 = impl_graphs[3].scheduled_graph.FindNode("brc0");
  EXPECT_NE(impl_grp_3_brc0, nullptr);
  EXPECT_EQ(impl_grp_3_brc0->GetAllInDataAnchorsSize(), 1);
  EXPECT_EQ(impl_grp_3_brc0->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "abs0");
  auto impl_grp_3_exp0 = impl_graphs[3].scheduled_graph.FindNode("exp0");
  EXPECT_NE(impl_grp_3_exp0, nullptr);
  EXPECT_EQ(impl_grp_3_exp0->GetAllInDataAnchorsSize(), 1);
  EXPECT_EQ(impl_grp_3_exp0->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "brc0");
  auto impl_grp_3_add0 = impl_graphs[3].scheduled_graph.FindNode("add0");
  EXPECT_NE(impl_grp_3_add0, nullptr);
  EXPECT_EQ(impl_grp_3_add0->GetAllInDataAnchorsSize(), 2);
  EXPECT_EQ(impl_grp_3_add0->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "brc0");

  EXPECT_EQ(impl_graphs[4].scheduled_graph.FindNode("brc0"), nullptr);
  EXPECT_EQ(impl_graphs[4].scheduled_graph.FindNode("brc1"), nullptr);
  EXPECT_EQ(impl_graphs[4].scheduled_graph.FindNode("brc2"), nullptr);
  auto impl_grp_4_exp0 = impl_graphs[4].scheduled_graph.FindNode("exp0");
  EXPECT_NE(impl_grp_4_exp0, nullptr);
  EXPECT_EQ(impl_grp_4_exp0->GetAllInDataAnchorsSize(), 1);
  EXPECT_EQ(impl_grp_4_exp0->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "abs0");
  auto impl_grp_4_add0 = impl_graphs[4].scheduled_graph.FindNode("add0");
  EXPECT_NE(impl_grp_4_add0, nullptr);
  EXPECT_EQ(impl_grp_4_add0->GetAllInDataAnchorsSize(), 2);
  EXPECT_EQ(impl_grp_4_add0->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "abs0");
  auto impl_grp_4_exp1 = impl_graphs[4].scheduled_graph.FindNode("exp1");
  EXPECT_NE(impl_grp_4_exp1, nullptr);
  EXPECT_EQ(impl_grp_4_exp1->GetAllInDataAnchorsSize(), 1);
  EXPECT_EQ(impl_grp_4_exp1->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "abs0");
}

TEST_F(AutoSchedulerUT, Autoschedule_autoschedule_remove_continues_broadcast_multi_out) {
  af::AscGraph graph("Autoschedule_autoschedule_remove_continues_broadcast_multi_out");
  Construct_ContinuesBroadcastMultiOutFusion(graph);
  EXPECT_EQ(Optimizer::MergeContinuousAxis(graph), af::SUCCESS);

  std::vector<autoschedule::AutoScheduleOutput> impl_graphs;
  AutoSchedule autoschedule(graph, impl_graphs);
  autoschedule.DoAutoSchedule();
  EXPECT_EQ(impl_graphs.size(), 7);

  EXPECT_NE(impl_graphs[0].scheduled_graph.FindNode("brc0"), nullptr);
  EXPECT_NE(impl_graphs[0].scheduled_graph.FindNode("brc1"), nullptr);

  EXPECT_NE(impl_graphs[1].scheduled_graph.FindNode("brc0"), nullptr);
  EXPECT_NE(impl_graphs[1].scheduled_graph.FindNode("brc1"), nullptr);

  EXPECT_NE(impl_graphs[2].scheduled_graph.FindNode("brc0"), nullptr);
  EXPECT_NE(impl_graphs[2].scheduled_graph.FindNode("brc1"), nullptr);

  EXPECT_NE(impl_graphs[3].scheduled_graph.FindNode("brc0"), nullptr);
  EXPECT_EQ(impl_graphs[3].scheduled_graph.FindNode("brc1"), nullptr);
}

TEST_F(AutoSchedulerUT, Autoschedule_autoschedule_remove_continues_broadcast_with_reduce) {
  af::AscGraph graph("Autoschedule_autoschedule_remove_continues_broadcast_with_reduce");
  Construct_ContinuesBroadcastWithReduceFusion(graph);

  std::vector<autoschedule::AutoScheduleOutput> results;
  AutoSchedule autoschedule(graph, results);
  autoschedule.DoAutoSchedule();
  EXPECT_EQ(results.size(), 4);

  EXPECT_EQ(results[0].scheduled_graph.FindNode("brc0"), nullptr);
  EXPECT_NE(results[0].scheduled_graph.FindNode("brc1"), nullptr);
  EXPECT_EQ(results[0].scheduled_graph.FindNode("brc3"), nullptr);
  EXPECT_NE(results[0].scheduled_graph.FindNode("brc4"), nullptr);

  EXPECT_EQ(results[1].scheduled_graph.FindNode("brc0"), nullptr);
  EXPECT_NE(results[1].scheduled_graph.FindNode("brc1"), nullptr);
  EXPECT_NE(results[1].scheduled_graph.FindNode("brc3"), nullptr);

  EXPECT_EQ(results[2].scheduled_graph.FindNode("brc0"), nullptr);
  EXPECT_NE(results[2].scheduled_graph.FindNode("brc1"), nullptr);

  EXPECT_EQ(results[3].scheduled_graph.FindNode("brc0"), nullptr);
  EXPECT_NE(results[3].scheduled_graph.FindNode("brc1"), nullptr);
}

TEST_F(AutoSchedulerUT, Autoschedule_autoschedule_remove_redundant_broadcast) {
  af::AscGraph graph("RemoveRedundantBroadcast");
  Construct_RedundantBroadcastFusion(graph);
  EXPECT_EQ(Optimizer::MergeContinuousAxis(graph), af::SUCCESS);

  std::vector<autoschedule::AutoScheduleOutput> impl_graphs;
  AutoSchedule autoschedule(graph, impl_graphs);
  autoschedule.DoAutoSchedule();
  EXPECT_EQ(impl_graphs.size(), 3);
  // check don't remove brc
  auto impl_grp_0_exp1 = impl_graphs[0].scheduled_graph.FindNode("exp1");
  EXPECT_NE(impl_grp_0_exp1, nullptr);
  EXPECT_EQ(impl_grp_0_exp1->GetAllInDataAnchorsSize(), 1);
  EXPECT_EQ(impl_grp_0_exp1->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "brc0");

  auto impl_grp_0_add0 = impl_graphs[0].scheduled_graph.FindNode("add0");
  EXPECT_NE(impl_grp_0_add0, nullptr);
  EXPECT_EQ(impl_grp_0_add0->GetAllInDataAnchorsSize(), 2);
  EXPECT_EQ(impl_grp_0_add0->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "brc0");
  EXPECT_EQ(impl_grp_0_add0->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "brc1");

  EXPECT_NE(impl_graphs[0].scheduled_graph.FindNode("brc0"), nullptr);
  EXPECT_NE(impl_graphs[0].scheduled_graph.FindNode("brc1"), nullptr);

  // check remove brc
  auto impl_grp_1_exp1 = impl_graphs[1].scheduled_graph.FindNode("exp1");
  EXPECT_NE(impl_grp_1_exp1, nullptr);
  EXPECT_EQ(impl_grp_1_exp1->GetAllInDataAnchorsSize(), 1);
  EXPECT_EQ(impl_grp_1_exp1->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "abs0");

  auto impl_grp_1_add0 = impl_graphs[1].scheduled_graph.FindNode("add0");
  EXPECT_NE(impl_grp_1_add0, nullptr);
  EXPECT_EQ(impl_grp_1_add0->GetAllInDataAnchorsSize(), 2);
  EXPECT_EQ(impl_grp_1_add0->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "abs0");
  EXPECT_EQ(impl_grp_1_add0->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "exp0");

  EXPECT_EQ(impl_graphs[1].scheduled_graph.FindNode("brc0"), nullptr);
  EXPECT_EQ(impl_graphs[1].scheduled_graph.FindNode("brc1"), nullptr);
}

TEST_F(AutoSchedulerUT, Autoschedule_keep_broadcast_expand_non_vectorized_axis) {
  af::AscGraph graph("KeepBroadcastExpandNonVectorizedAxis");
  Construct_BroadcastExpandNonVectorizedAxis(graph);

  EXPECT_EQ(Scheduler::RemoveRedundantBroadcastNode(graph), af::SUCCESS);

  auto brc0 = graph.FindNode("brc0");
  ASSERT_NE(brc0, nullptr);
  auto reduce_sum = graph.FindNode("reduce_sum");
  ASSERT_NE(reduce_sum, nullptr);
  EXPECT_EQ(reduce_sum->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "brc0");
}

TEST_F(AutoSchedulerUT, Autoschedule_remove_broadcast_expand_non_vectorized_axis_reduce_different_axis) {
  af::AscGraph graph("RemoveBroadcastExpandNonVectorizedAxisReduceDifferentAxis");
  Construct_BroadcastExpandNonVectorizedAxisReduceDifferentAxis(graph);

  EXPECT_EQ(Scheduler::RemoveRedundantBroadcastNode(graph), af::SUCCESS);

  EXPECT_EQ(graph.FindNode("brc0"), nullptr);
  auto reduce_sum = graph.FindNode("reduce_sum");
  ASSERT_NE(reduce_sum, nullptr);
  EXPECT_EQ(reduce_sum->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "abs0");
}

TEST_F(AutoSchedulerUT, Autoschedule_remove_broadcast_add_non_vectorized_axis) {
  af::AscGraph graph("RemoveBroadcastAddNonVectorizedAxis");
  Construct_BroadcastAddNonVectorizedAxis(graph);

  EXPECT_EQ(Scheduler::RemoveRedundantBroadcastNode(graph), af::SUCCESS);

  EXPECT_EQ(graph.FindNode("brc_missing_axis"), nullptr);
  auto reduce_missing_axis = graph.FindNode("reduce_missing_axis");
  ASSERT_NE(reduce_missing_axis, nullptr);
  EXPECT_EQ(reduce_missing_axis->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "load0");
}

TEST_F(AutoSchedulerUT, Autoschedule_scheduler_gather_3axis) {
  af::AscGraph graph("LoadGatherStore");
  Construct_LoadGatherStore(graph);

  af::AscGraph except_graph("LoadGatherStore");
  except_graph.CopyFrom(graph);

  auto gather = graph.FindNode("gather");
  std::vector<af::AxisId> y_group = gather->attr.sched.axis;

  std::vector<autoschedule::AutoScheduleOutput> results;
  AutoSchedule schedule(graph, results);
  schedule.DoAutoSchedule();
  ASSERT_EQ(results.size(), 1UL);
}

TEST_F(AutoSchedulerUT, Autoschedule_scheduler_gather_param_is_1_axis) {
  af::AscGraph graph("LoadGatherStore");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  af::ascir_op::Data x1("x1", graph);
  x1.attr.sched.axis = {z0.id};
  x1.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  x1.attr.api.type = af::ApiType::kAPITypeBuffer;
  x1.y.dtype = ge::DT_FLOAT;
  *x1.y.axis = {z0.id};
  *x1.y.repeats = {s0};
  *x1.y.strides = {One};

  af::ascir_op::Data x2("x2", graph);
  x2.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  x2.attr.api.type = af::ApiType::kAPITypeBuffer;
  x2.y.dtype = ge::DT_INT32;
  x2.attr.sched.axis = {z1.id};
  *x2.y.axis = {z1.id};
  *x2.y.repeats = {s1};
  *x2.y.strides = {One};

  af::ascir_op::Gather gather("gather");
  graph.AddNode(gather);
  gather.x1 = x1.y;
  gather.x2 = x2.y;
  gather.attr.api.compute_type = af::ComputeType::kComputeGather;
  gather.y.dtype = ge::DT_FLOAT;
  gather.attr.sched.axis = {z1.id};
  *gather.y.axis = {z1.id};
  *gather.y.repeats = {s1};
  *gather.y.strides = {One};
  gather.ir_attr.SetAxis(0);

  af::ascir_op::Store store("store");
  graph.AddNode(store);
  store.x = gather.y;
  store.attr.api.compute_type = af::ComputeType::kComputeStore;
  store.y.dtype = ge::DT_FLOAT;
  store.attr.sched.axis = {z1.id};
  *store.y.axis = {z1.id};
  *store.y.repeats = {s1};
  *store.y.strides = {One};

  af::ascir_op::Output y("y");
  y.x = store.y;
  y.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  y.attr.api.type = af::ApiType::kAPITypeBuffer;
  y.y.dtype = ge::DT_FLOAT;
  y.attr.sched.axis = {z1.id};
  *y.y.axis = {z1.id};
  *y.y.repeats = {s1};
  *y.y.strides = {One};

  af::AscGraph except_graph("LoadGatherStore");
  except_graph.CopyFrom(graph);

  auto gather_node = graph.FindNode("gather");

  std::vector<autoschedule::AutoScheduleOutput> results;
  AutoSchedule schedule(graph, results);
  auto status = schedule.DoAutoSchedule();
  ASSERT_EQ(status, af::SUCCESS);
  ASSERT_EQ(results.size(), 1UL);
}

TEST_F(AutoSchedulerUT, Autoschedule_autoschedule_removepad_support_broadcast) {
  af::AscGraph graph("Autoschedule_autoschedule_removepad_support_broadcast");
  auto s0 = graph.CreateSizeVar(2);
  auto s1 = graph.CreateSizeVar(3);
  auto s2 = graph.CreateSizeVar(3);

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  af::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.attr.sched.axis = {z0.id, z1.id, z2.id};
  data0.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  data0.attr.api.type = af::ApiType::kAPITypeBuffer;
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id, z2.id};
  *data0.y.repeats = {One, s1, s2};
  *data0.y.strides = {Zero, s2, One};

  af::ascir_op::Load load0("load0");
  load0.x = data0.y;
  load0.attr.sched.axis = {z0.id, z1.id, z2.id};
  load0.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load0.y.dtype = ge::DT_FLOAT16;
  *load0.y.axis = {z0.id, z1.id, z2.id};
  *load0.y.repeats = {One, s1, s2};
  *load0.y.strides = {Zero, s2, One};

  af::ascir_op::Broadcast brc0("brc0");
  brc0.x = load0.y;
  brc0.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc0.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc0.y.dtype = ge::DT_FLOAT16;
  *brc0.y.axis = {z0.id, z1.id, z2.id};
  *brc0.y.repeats = {s0, s1, s2};
  *brc0.y.strides = {s1 * s2, s2, One};

  af::ascir_op::Data data1("data1", graph);
  data1.ir_attr.SetIndex(1);
  data1.attr.sched.axis = {z0.id, z1.id, z2.id};
  data1.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  data1.attr.api.type = af::ApiType::kAPITypeBuffer;
  data1.y.dtype = ge::DT_FLOAT16;
  *data1.y.axis = {z0.id, z1.id, z2.id};
  *data1.y.repeats = {s0, s1, s2};
  *data1.y.strides = {s1 * s2, s2, One};

  af::ascir_op::Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id, z2.id};
  load1.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load1.y.dtype = ge::DT_FLOAT16;
  *load1.y.axis = {z0.id, z1.id, z2.id};
  *load1.y.repeats = {s0, s1, s2};
  *load1.y.strides = {s1 * s2, s2, One};

  af::ascir_op::Add add0("add0");
  add0.x1 = brc0.y;
  add0.x2 = load1.y;
  add0.attr.sched.axis = {z0.id, z1.id, z2.id};
  add0.attr.api.compute_type = af::ComputeType::kComputeElewise;
  add0.y.dtype = ge::DT_FLOAT16;
  *add0.y.axis = {z0.id, z1.id, z2.id};
  *add0.y.repeats = {s0, s1, s2};
  *add0.y.strides = {s1 * s2, s2, One};

  af::ascir_op::Store store0("store0");
  store0.x = add0.y;
  store0.attr.sched.axis = {z0.id, z1.id, z2.id};
  store0.attr.api.compute_type = af::ComputeType::kComputeStore;
  store0.y.dtype = ge::DT_FLOAT16;
  *store0.y.axis = {z0.id, z1.id, z2.id};
  *store0.y.repeats = {s0, s1, s2};
  *store0.y.strides = {s1 * s2, s2, One};

  af::ascir_op::Output y0("y0");
  y0.ir_attr.SetIndex(0);
  y0.x = store0.y;
  y0.attr.sched.axis = {z0.id, z1.id, z2.id};
  y0.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  y0.attr.api.type = af::ApiType::kAPITypeBuffer;
  y0.y.dtype = ge::DT_FLOAT16;
  *y0.y.axis = {z0.id, z1.id, z2.id};
  *y0.y.repeats = {s0, s1, s2};
  *y0.y.strides = {s1 * s2, s2, One};

  af::ascir_op::Data data2("data2", graph);
  data2.ir_attr.SetIndex(2);
  data2.attr.sched.axis = {z0.id, z1.id, z2.id};
  data2.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  data2.attr.api.type = af::ApiType::kAPITypeBuffer;
  data2.y.dtype = ge::DT_FLOAT16;
  *data2.y.axis = {z0.id, z1.id, z2.id};
  *data2.y.repeats = {One, s1, s2};
  *data2.y.strides = {Zero, s2, One};

  af::ascir_op::Load load2("load2");
  load2.x = data2.y;
  load2.attr.sched.axis = {z0.id, z1.id, z2.id};
  load2.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load2.y.dtype = ge::DT_FLOAT16;
  *load2.y.axis = {z0.id, z1.id, z2.id};
  *load2.y.repeats = {One, s1, s2};
  *load2.y.strides = {Zero, s2, One};

  af::ascir_op::Broadcast brc2("brc2");
  brc2.x = load2.y;
  brc2.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc2.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc2.y.dtype = ge::DT_FLOAT16;
  *brc2.y.axis = {z0.id, z1.id, z2.id};
  *brc2.y.repeats = {s0, s1, s2};
  *brc2.y.strides = {s1 * s2, s2, One};

  af::ascir_op::Mul mul0("mul0");
  mul0.x1 = load1.y;
  mul0.x2 = brc2.y;
  mul0.attr.sched.axis = {z0.id, z1.id, z2.id};
  mul0.attr.api.compute_type = af::ComputeType::kComputeElewise;
  mul0.y.dtype = ge::DT_FLOAT16;
  *mul0.y.axis = {z0.id, z1.id, z2.id};
  *mul0.y.repeats = {s0, s1, s2};
  *mul0.y.strides = {s1 * s2, s2, One};

  af::ascir_op::Store store1("store1");
  store1.x = mul0.y;
  store1.attr.sched.axis = {z0.id, z1.id, z2.id};
  store1.attr.api.compute_type = af::ComputeType::kComputeStore;
  store1.y.dtype = ge::DT_FLOAT16;
  *store1.y.axis = {z0.id, z1.id, z2.id};
  *store1.y.repeats = {s0, s1, s2};
  *store1.y.strides = {s1 * s2, s2, One};

  af::ascir_op::Output y1("y1");
  y1.ir_attr.SetIndex(1);
  y1.x = store1.y;
  y1.attr.sched.axis = {z0.id, z1.id, z2.id};
  y1.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  y1.attr.api.type = af::ApiType::kAPITypeBuffer;
  y1.y.dtype = ge::DT_FLOAT16;
  *y1.y.axis = {z0.id, z1.id, z2.id};
  *y1.y.repeats = {s0, s1, s2};
  *y1.y.strides = {s1 * s2, s2, One};

  EXPECT_EQ(Optimizer::MergeContinuousAxis(graph), af::SUCCESS);
  std::vector<autoschedule::AutoScheduleOutput> impl_graphs;
  AutoSchedule autoschedule(graph, impl_graphs);
  autoschedule.DoAutoSchedule();
  EXPECT_EQ(impl_graphs.size(), 3);
  EXPECT_EQ(impl_graphs[0].scheduled_graph.GetName(), "Autoschedule_autoschedule_removepad_support_broadcast_B0Y3");
  EXPECT_EQ(impl_graphs[1].scheduled_graph.GetName(),
            "Autoschedule_autoschedule_removepad_support_broadcast_B0Y0_unaligned");
  EXPECT_EQ(impl_graphs[2].scheduled_graph.GetName(),
            "Autoschedule_autoschedule_removepad_support_broadcast_B0Y0_inline");

  EXPECT_NE(impl_graphs[1].scheduled_graph.FindNode("brc0"), nullptr);
  EXPECT_NE(impl_graphs[1].scheduled_graph.FindNode("brc2"), nullptr);
  EXPECT_NE(impl_graphs[1].scheduled_graph.FindNode("brc0_remove_pad_0"), nullptr);
  EXPECT_NE(impl_graphs[1].scheduled_graph.FindNode("brc2_remove_pad_0"), nullptr);
  EXPECT_EQ(af::AscGraphUtils::GetComputeGraph(impl_graphs[1].scheduled_graph)->GetAllNodesSize(), 16);
  const auto &impl1_remove_pad0 = impl_graphs[1].scheduled_graph.FindNode("brc0_remove_pad_0");
  EXPECT_EQ(impl1_remove_pad0->GetAllInDataAnchorsSize(), 1);
  EXPECT_EQ(impl1_remove_pad0->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "brc0");
  EXPECT_EQ(impl1_remove_pad0->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetOwnerNode()->GetName(), "add0");
  const auto &impl1_remove_pad2 = impl_graphs[1].scheduled_graph.FindNode("brc2_remove_pad_0");
  EXPECT_EQ(impl1_remove_pad2->GetAllInDataAnchorsSize(), 1);
  EXPECT_EQ(impl1_remove_pad2->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "brc2");
  EXPECT_EQ(impl1_remove_pad2->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetOwnerNode()->GetName(), "mul0");

  EXPECT_EQ(impl_graphs[0].scheduled_graph.FindNode("brc0"), nullptr);
  EXPECT_EQ(impl_graphs[0].scheduled_graph.FindNode("brc2"), nullptr);
  EXPECT_EQ(impl_graphs[0].scheduled_graph.FindNode("brc0_remove_pad_0"), nullptr);
  EXPECT_EQ(impl_graphs[0].scheduled_graph.FindNode("brc2_remove_pad_0"), nullptr);
  EXPECT_EQ(af::AscGraphUtils::GetComputeGraph(impl_graphs[0].scheduled_graph)->GetAllNodesSize(), 12);
}

TEST_F(AutoSchedulerUT, Autoschedule_autoschedule_removepad_not_support_reduce) {
  af::AscGraph graph("Autoschedule_autoschedule_remove_pad_not_support_reduce");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  af::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.attr.sched.axis = {z0.id, z1.id, z2.id};
  data0.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  data0.attr.api.type = af::ApiType::kAPITypeBuffer;
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id, z2.id};
  *data0.y.repeats = {One, s1, s2};
  *data0.y.strides = {Zero, s2, One};

  af::ascir_op::Load load0("load0");
  load0.x = data0.y;
  load0.attr.sched.axis = {z0.id, z1.id, z2.id};
  load0.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load0.y.dtype = ge::DT_FLOAT16;
  *load0.y.axis = {z0.id, z1.id, z2.id};
  *load0.y.repeats = {One, s1, s2};
  *load0.y.strides = {Zero, s2, One};

  af::ascir_op::Sum sum0("reduce0");
  sum0.x = load0.y;
  sum0.attr.sched.axis = {z0.id, z1.id, z2.id};
  sum0.attr.api.compute_type = af::ComputeType::kComputeReduce;
  sum0.y.dtype = ge::DT_FLOAT16;
  *sum0.y.axis = {z0.id, z1.id, z2.id};
  *sum0.y.repeats = {One, s1, One};
  *sum0.y.strides = {Zero, One, Zero};

  af::ascir_op::Store store0("store0");
  store0.x = sum0.y;
  store0.attr.sched.axis = {z0.id, z1.id, z2.id};
  store0.attr.api.compute_type = af::ComputeType::kComputeStore;
  store0.y.dtype = ge::DT_FLOAT16;
  *store0.y.axis = {z0.id, z1.id, z2.id};
  *store0.y.repeats = {One, s1, One};
  *store0.y.strides = {Zero, One, Zero};

  af::ascir_op::Output y0("y0");
  y0.ir_attr.SetIndex(0);
  y0.x = store0.y;
  y0.attr.sched.axis = {z0.id, z1.id, z2.id};
  y0.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  y0.attr.api.type = af::ApiType::kAPITypeBuffer;
  y0.y.dtype = ge::DT_FLOAT16;
  *y0.y.axis = {z0.id, z1.id, z2.id};
  *y0.y.repeats = {One, s1, One};
  *y0.y.strides = {Zero, One, Zero};

  std::vector<autoschedule::AutoScheduleOutput> impl_graphs;
  AutoSchedule autoschedule(graph, impl_graphs);
  autoschedule.DoAutoSchedule();
  EXPECT_EQ(impl_graphs.size(), 2);
  for (const auto &impl_graph : impl_graphs) {
    EXPECT_EQ(af::AscGraphUtils::GetComputeGraph(impl_graph.scheduled_graph)->GetAllNodesSize(), 5);
  }
}

TEST_F(AutoSchedulerUT, Autoschedule_autoschedule_removepad_not_support_dtype) {
  af::AscGraph graph("Autoschedule_autoschedule_removepad_not_support_dtype");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  af::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.attr.sched.axis = {z0.id, z1.id, z2.id};
  data0.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  data0.attr.api.type = af::ApiType::kAPITypeBuffer;
  data0.y.dtype = ge::DT_INT64;
  *data0.y.axis = {z0.id, z1.id, z2.id};
  *data0.y.repeats = {One, s1, s2};
  *data0.y.strides = {Zero, s2, One};

  af::ascir_op::Load load0("load0");
  load0.x = data0.y;
  load0.attr.sched.axis = {z0.id, z1.id, z2.id};
  load0.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load0.y.dtype = ge::DT_INT64;
  *load0.y.axis = {z0.id, z1.id, z2.id};
  *load0.y.repeats = {One, s1, s2};
  *load0.y.strides = {Zero, s2, One};

  af::ascir_op::Broadcast brc0("brc0");
  brc0.x = load0.y;
  brc0.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc0.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc0.y.dtype = ge::DT_INT64;
  *brc0.y.axis = {z0.id, z1.id, z2.id};
  *brc0.y.repeats = {s0, s1, s2};
  *brc0.y.strides = {s1 * s2, s2, One};

  af::ascir_op::Store store0("store0");
  store0.x = brc0.y;
  store0.attr.sched.axis = {z0.id, z1.id, z2.id};
  store0.attr.api.compute_type = af::ComputeType::kComputeStore;
  store0.y.dtype = ge::DT_INT64;
  *store0.y.axis = {z0.id, z1.id, z2.id};
  *store0.y.repeats = {s0, s1, s2};
  *store0.y.strides = {s1 * s2, s2, One};

  af::ascir_op::Output y0("y0");
  y0.ir_attr.SetIndex(0);
  y0.x = store0.y;
  y0.attr.sched.axis = {z0.id, z1.id, z2.id};
  y0.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  y0.attr.api.type = af::ApiType::kAPITypeBuffer;
  y0.y.dtype = ge::DT_INT64;
  *y0.y.axis = {z0.id, z1.id, z2.id};
  *y0.y.repeats = {s0, s1, s2};
  *y0.y.strides = {s1 * s2, s2, One};
  EXPECT_EQ(Optimizer::MergeContinuousAxis(graph), af::SUCCESS);
  std::vector<autoschedule::AutoScheduleOutput> impl_graphs;
  AutoSchedule autoschedule(graph, impl_graphs);
  autoschedule.DoAutoSchedule();
  EXPECT_EQ(impl_graphs.size(), 2);
  for (const auto &impl_graph : impl_graphs) {
    EXPECT_EQ(impl_graph.scheduled_graph.FindNode("brc0_remove_pad"), nullptr);
  }
}

TEST_F(AutoSchedulerUT, Autoschedule_autoschedule_removepad_support_load_slice) {
  af::AscGraph graph("removepad_support_load_slice");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto offset = af::Symbol(2);

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  af::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.attr.sched.axis = {z0.id, z1.id, z2.id};
  data0.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  data0.attr.api.type = af::ApiType::kAPITypeBuffer;
  data0.y.dtype = ge::DT_FLOAT;
  *data0.y.axis = {z0.id, z1.id, z2.id};

  af::ascir_op::Load load0("load0");
  load0.x = data0.y;
  load0.attr.sched.axis = {z0.id, z1.id, z2.id};
  load0.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.axis = {z0.id, z1.id, z2.id};
  *load0.y.repeats = {s0, s1, s2};
  *load0.y.strides = {s1 * s2 * offset, s2 * offset, One};

  af::ascir_op::Abs abs0("abs0");
  abs0.x = load0.y;
  abs0.attr.sched.axis = {z0.id, z1.id, z2.id};
  abs0.attr.api.compute_type = af::ComputeType::kComputeElewise;
  abs0.y.dtype = ge::DT_FLOAT;
  *abs0.y.axis = {z0.id, z1.id, z2.id};
  *abs0.y.repeats = {s0, s1, s2};
  *abs0.y.strides = {s1 * s2, s2, One};

  af::ascir_op::Store store0("store0");
  store0.x = abs0.y;
  store0.attr.sched.axis = {z0.id, z1.id, z2.id};
  store0.attr.api.compute_type = af::ComputeType::kComputeStore;
  store0.y.dtype = ge::DT_FLOAT;
  *store0.y.axis = {z0.id, z1.id, z2.id};
  *store0.y.repeats = {s0, s1, s2};
  *store0.y.strides = {s1 * s2, s2, One};

  af::ascir_op::Output y0("y0");
  y0.ir_attr.SetIndex(0);
  y0.x = store0.y;
  y0.attr.sched.axis = {z0.id, z1.id, z2.id};
  y0.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  y0.attr.api.type = af::ApiType::kAPITypeBuffer;
  y0.y.dtype = ge::DT_FLOAT;
  *y0.y.axis = {z0.id, z1.id, z2.id};
  *y0.y.repeats = {s0, s1, s2};
  *y0.y.strides = {s1 * s2, s2, One};
  EXPECT_EQ(Optimizer::MergeContinuousAxis(graph), af::SUCCESS);
  std::vector<autoschedule::AutoScheduleOutput> results;
  AutoSchedule autoschedule(graph, results);
  autoschedule.DoAutoSchedule();
  EXPECT_EQ(results.size(), 3);

  EXPECT_EQ(af::AscGraphUtils::GetComputeGraph(results[2].scheduled_graph)->GetAllNodesSize(), 6);
  const auto &remove_pad = results[2].scheduled_graph.FindNode("load0_remove_pad_0");
  EXPECT_NE(remove_pad, nullptr);
  EXPECT_EQ(remove_pad->GetAllInDataAnchorsSize(), 1);
  EXPECT_EQ(remove_pad->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "load0");
  EXPECT_EQ(remove_pad->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetOwnerNode()->GetName(), "abs0");
}

TEST_F(AutoSchedulerUT, Autoschedule_autoschedule_removepad_support_load_slice_inductor) {
  af::AscGraph graph("removepad_support_load_slice_inductor)");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto offset = af::Symbol(2);

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  af::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.attr.sched.axis = {z0.id, z1.id, z2.id};
  data0.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  data0.attr.api.type = af::ApiType::kAPITypeBuffer;
  data0.y.dtype = ge::DT_FLOAT;
  *data0.y.axis = {z0.id, z1.id, z2.id};

  af::ascir_op::Load load0("load0");
  load0.x = data0.y;
  load0.attr.sched.axis = {z0.id, z1.id, z2.id};
  load0.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.axis = {z0.id, z1.id, z2.id};
  *load0.y.repeats = {s0, s1, s2};
  *load0.y.strides = {s1 * s2 * s0, s2 * s0, s0};

  af::ascir_op::Abs abs0("abs0");
  abs0.x = load0.y;
  abs0.attr.sched.axis = {z0.id, z1.id, z2.id};
  abs0.attr.api.compute_type = af::ComputeType::kComputeElewise;
  abs0.y.dtype = ge::DT_FLOAT;
  *abs0.y.axis = {z0.id, z1.id, z2.id};
  *abs0.y.repeats = {s0, s1, s2};
  *abs0.y.strides = {s1 * s2, s2, One};

  af::ascir_op::Store store0("store0");
  store0.x = abs0.y;
  store0.attr.sched.axis = {z0.id, z1.id, z2.id};
  store0.attr.api.compute_type = af::ComputeType::kComputeStore;
  store0.y.dtype = ge::DT_FLOAT;
  *store0.y.axis = {z0.id, z1.id, z2.id};
  *store0.y.repeats = {s0, s1, s2};
  *store0.y.strides = {s1 * s2, s2, One};

  af::ascir_op::Output y0("y0");
  y0.ir_attr.SetIndex(0);
  y0.x = store0.y;
  y0.attr.sched.axis = {z0.id, z1.id, z2.id};
  y0.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  y0.attr.api.type = af::ApiType::kAPITypeBuffer;
  y0.y.dtype = ge::DT_FLOAT;
  *y0.y.axis = {z0.id, z1.id, z2.id};
  *y0.y.repeats = {s0, s1, s2};
  *y0.y.strides = {s1 * s2, s2, One};
  EXPECT_EQ(Optimizer::MergeContinuousAxis(graph), af::SUCCESS);

  std::vector<autoschedule::AutoScheduleOutput> results;
  AutoSchedule autoschedule(graph, results);
  autoschedule.DoAutoSchedule();
  EXPECT_EQ(results.size(), 3);

  EXPECT_EQ(af::AscGraphUtils::GetComputeGraph(results[2].scheduled_graph)->GetAllNodesSize(), 6);
  const auto &remove_pad = results[2].scheduled_graph.FindNode("load0_remove_pad_0");
  EXPECT_NE(remove_pad, nullptr);
  EXPECT_EQ(remove_pad->GetAllInDataAnchorsSize(), 1);
  EXPECT_EQ(remove_pad->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "load0");
  EXPECT_EQ(remove_pad->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetOwnerNode()->GetName(), "abs0");
}

TEST_F(AutoSchedulerUT, Autoschedule_autoschedule_removepad_support_concat) {
  af::AscGraph graph("Autoschedule_autoschedule_removepad_support_concat");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = af::Symbol(2);

  auto tmp = graph.CreateAxis("tmp", s0);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  af::ascir_op::Data x("x", graph);
  x.attr.sched.axis = {z0.id, z1.id};
  x.y.dtype = ge::DT_FLOAT16;
  x.ir_attr.SetIndex(0);
  x.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  x.attr.api.type = af::ApiType::kAPITypeBuffer;

  af::ascir_op::Load load("load");
  load.x = x.y;
  load.attr.sched.axis = {z0.id, z1.id};
  load.y.dtype = ge::DT_FLOAT16;
  *load.y.axis = {z0.id, z1.id};
  *load.y.repeats = {s0, One};
  *load.y.strides = {One, One};
  load.attr.api.compute_type = af::ComputeType::kComputeLoad;

  af::ascir_op::Data x1("x1", graph);
  x1.attr.sched.axis = {z0.id, z1.id};
  x1.y.dtype = ge::DT_FLOAT16;
  x1.ir_attr.SetIndex(1);
  x1.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  x1.attr.api.type = af::ApiType::kAPITypeBuffer;

  af::ascir_op::Load load1("load1");
  load1.x = x1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT16;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {s0, One};
  *load1.y.strides = {One, One};
  load1.attr.api.compute_type = af::ComputeType::kComputeLoad;

  af::ascir_op::Concat concat("concat");
  concat.x = {load.y, load1.y};
  concat.attr.sched.axis = {z0.id, z1.id};
  concat.y.dtype = ge::DT_FLOAT16;
  *concat.y.axis = {z0.id, z1.id};
  *concat.y.repeats = {s0, s1};
  *concat.y.strides = {s1, One};
  concat.attr.api.compute_type = af::ComputeType::kComputeConcat;

  af::ascir_op::Store store("store");
  store.x = concat.y;
  store.attr.sched.axis = {z0.id, z1.id};
  store.y.dtype = ge::DT_FLOAT16;
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1};
  *store.y.strides = {s1, One};
  store.attr.api.compute_type = af::ComputeType::kComputeStore;

  af::ascir_op::Output output0("output0");
  output0.x = store.y;
  output0.attr.sched.axis = {z0.id, z1.id};
  output0.attr.api.type = af::ApiType::kAPITypeBuffer;
  output0.y.dtype = ge::DT_FLOAT16;
  output0.ir_attr.SetIndex(0);

  af::ascir_op::Load load3("load3");
  load3.x = output0.y;
  load3.attr.sched.axis = {z0.id, z1.id};
  load3.y.dtype = ge::DT_FLOAT16;
  *load3.y.axis = {z0.id, z1.id};
  *load3.y.repeats = {s0, s1};
  *load3.y.strides = {s1, One};
  load3.attr.api.compute_type = af::ComputeType::kComputeLoad;

  af::ascir_op::Store store1("store1");
  store1.x = load3.y;
  store1.attr.sched.axis = {z0.id, z1.id};
  store1.y.dtype = ge::DT_FLOAT16;
  *store1.y.axis = {z0.id, z1.id};
  *store1.y.repeats = {s0, s1};
  *store1.y.strides = {s1, One};
  store1.attr.api.compute_type = af::ComputeType::kComputeStore;

  af::ascir_op::Output y("y");
  y.x = store1.y;
  y.attr.sched.axis = {z0.id, z1.id};
  y.attr.api.type = af::ApiType::kAPITypeBuffer;
  y.y.dtype = ge::DT_FLOAT16;
  y.ir_attr.SetIndex(0);

  std::vector<autoschedule::AutoScheduleOutput> impl_graphs;
  AutoSchedule autoschedule(graph, impl_graphs);
  autoschedule.DoAutoSchedule();
  EXPECT_EQ(impl_graphs.size(), 1);
}

TEST_F(AutoSchedulerUT, Autoschedule_autoschedule_removepad_support_concat_all_input_cols_aligned) {
  af::AscGraph graph("Autoschedule_autoschedule_removepad_support_concat");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = af::Symbol(128);
  auto s2 = af::Symbol(2);

  auto tmp = graph.CreateAxis("tmp", s0);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  af::ascir_op::Data x("x", graph);
  x.attr.sched.axis = {z0.id, z1.id, z2.id};
  x.y.dtype = ge::DT_FLOAT16;
  x.ir_attr.SetIndex(0);
  x.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  x.attr.api.type = af::ApiType::kAPITypeBuffer;

  af::ascir_op::Load load("load");
  load.x = x.y;
  load.attr.sched.axis = {z0.id, z1.id, z2.id};
  load.y.dtype = ge::DT_FLOAT16;
  *load.y.axis = {z0.id, z1.id, z2.id};
  *load.y.repeats = {s0, s1, s2};
  *load.y.strides = {s1 * s2, s2, One};
  load.attr.api.compute_type = af::ComputeType::kComputeLoad;

  af::ascir_op::Concat concat("concat");
  concat.x = {load.y, load.y};
  concat.attr.sched.axis = {z0.id, z1.id, z2.id};
  concat.y.dtype = ge::DT_FLOAT16;
  *concat.y.axis = {z0.id, z1.id, z2.id};
  *concat.y.repeats = {s0, s1 + s1, s2};
  *concat.y.strides = {(s1 + s1) * s2, s2, One};
  concat.attr.api.compute_type = af::ComputeType::kComputeConcat;

  af::ascir_op::Store store("store");
  store.x = concat.y;
  store.attr.sched.axis = {z0.id, z1.id, z2.id};
  store.y.dtype = ge::DT_FLOAT16;
  *store.y.axis = {z0.id, z1.id, z2.id};
  *store.y.repeats = {s0, s1 + s1, s2};
  *store.y.strides = {(s1 + s1) * s2, s2, One};
  store.attr.api.compute_type = af::ComputeType::kComputeStore;

  af::ascir_op::Output output0("output0");
  output0.x = store.y;
  output0.attr.sched.axis = {z0.id, z1.id, z2.id};
  output0.attr.api.type = af::ApiType::kAPITypeBuffer;
  output0.y.dtype = ge::DT_FLOAT16;
  output0.ir_attr.SetIndex(0);

  std::vector<autoschedule::AutoScheduleOutput> impl_graphs;
  AutoSchedule autoschedule(graph, impl_graphs);
  autoschedule.DoAutoSchedule();
  EXPECT_EQ(impl_graphs.size(), 1);
  EXPECT_NE(impl_graphs[0].scheduled_graph.GetName().find("unaligned"), std::string::npos);
}

/**
 *                 add0
 *              /      \
 *            /         \
 *          /            \
 *        /             brc1
 *       |(s0,s1,s2)     |(1,s1,s2)
 *     load0           load1
 *       |               |
 *     data0           data1
 */
TEST_F(AutoSchedulerUT, Autoschedule_autoschedule_brc_inline_static_support) {
  af::AscGraph graph("Autoschedule_autoschedule_brc_inline_static_support");
  auto s0 = af::Symbol(20);
  auto s1 = af::Symbol(32);

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  af::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  data0.attr.api.type = af::ApiType::kAPITypeBuffer;
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id};

  af::ascir_op::Load load0("load0");
  load0.x = data0.y;
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load0.y.dtype = ge::DT_FLOAT16;
  *load0.y.axis = {z0.id, z1.id};
  *load0.y.repeats = {s0, s1};
  *load0.y.strides = {s1, One};

  af::ascir_op::Data data1("data1", graph);
  data1.ir_attr.SetIndex(1);
  data1.attr.sched.axis = {z0.id, z1.id};
  data1.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  data1.attr.api.type = af::ApiType::kAPITypeBuffer;
  data1.y.dtype = ge::DT_FLOAT16;
  *data1.y.axis = {z0.id, z1.id};

  af::ascir_op::Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load1.y.dtype = ge::DT_FLOAT16;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {One, s1};
  *load1.y.strides = {Zero, One};

  af::ascir_op::Broadcast brc1("brc1");
  brc1.x = load1.y;
  brc1.attr.sched.axis = {z0.id, z1.id};
  brc1.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc1.y.dtype = ge::DT_FLOAT16;
  *brc1.y.axis = {z0.id, z1.id};
  *brc1.y.repeats = {s0, s1};
  *brc1.y.strides = {s1, One};

  af::ascir_op::Add add0("add0");
  add0.x1 = load0.y;
  add0.x2 = brc1.y;
  add0.attr.sched.axis = {z0.id, z1.id};
  add0.attr.api.compute_type = af::ComputeType::kComputeElewise;
  add0.y.dtype = ge::DT_FLOAT16;
  *add0.y.axis = {z0.id, z1.id};
  *add0.y.repeats = {s0, s1};
  *add0.y.strides = {s1, One};

  af::ascir_op::Store store0("store0");
  store0.x = add0.y;
  store0.attr.sched.axis = {z0.id, z1.id};
  store0.attr.api.compute_type = af::ComputeType::kComputeStore;
  store0.y.dtype = ge::DT_FLOAT16;
  *store0.y.axis = {z0.id, z1.id};
  *store0.y.repeats = {s0, s1};
  *store0.y.strides = {s1, One};

  af::ascir_op::Output y0("y0");
  y0.ir_attr.SetIndex(0);
  y0.x = store0.y;
  y0.attr.sched.axis = {z0.id, z1.id};
  y0.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  y0.attr.api.type = af::ApiType::kAPITypeBuffer;
  y0.y.dtype = ge::DT_FLOAT16;
  *y0.y.axis = {z0.id, z1.id};
  *y0.y.repeats = {s0, s1};
  *y0.y.strides = {s1, One};

  std::vector<autoschedule::AutoScheduleOutput> impl_graphs;
  AutoSchedule autoschedule(graph, impl_graphs);
  autoschedule.DoAutoSchedule();
  EXPECT_EQ(impl_graphs.size(), 2);

  EXPECT_EQ(impl_graphs[1].scheduled_graph.FindNode("brc1"), nullptr);
  EXPECT_EQ(af::AscGraphUtils::GetComputeGraph(impl_graphs[1].scheduled_graph)->GetAllNodesSize(), 7);
  const auto &impl1_add0 = impl_graphs[1].scheduled_graph.FindNode("add0");
  EXPECT_EQ(impl1_add0->GetAllInDataAnchorsSize(), 2);
  EXPECT_EQ(impl1_add0->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "load0");
  EXPECT_EQ(impl1_add0->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "load1");

  EXPECT_EQ("Autoschedule_autoschedule_brc_inline_static_support_B0Y0_inline",
            impl_graphs[1].scheduled_graph.GetName());
}

/**
 *                 add0
 *              /      \
 *            /         \
 *          /            \
 *        /             brc1
 *       |(s0,s1,s2)     |(1,s1,s2)
 *     load0           load1
 *       |               |
 *     data0           data1
 */
TEST_F(AutoSchedulerUT, Autoschedule_autoschedule_brc_inline_dynamic_support) {
  af::AscGraph graph("Autoschedule_autoschedule_brc_inline_dynamic_support");
  auto s0 = af::Symbol("20");
  auto s1 = af::Symbol("32");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  af::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  data0.attr.api.type = af::ApiType::kAPITypeBuffer;
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id};

  af::ascir_op::Load load0("load0");
  load0.x = data0.y;
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load0.y.dtype = ge::DT_FLOAT16;
  *load0.y.axis = {z0.id, z1.id};
  *load0.y.repeats = {s0, s1};
  *load0.y.strides = {s1, One};

  af::ascir_op::Data data1("data1", graph);
  data1.ir_attr.SetIndex(1);
  data1.attr.sched.axis = {z0.id, z1.id};
  data1.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  data1.attr.api.type = af::ApiType::kAPITypeBuffer;
  data1.y.dtype = ge::DT_FLOAT16;
  *data1.y.axis = {z0.id, z1.id};

  af::ascir_op::Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load1.y.dtype = ge::DT_FLOAT16;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {One, s1};
  *load1.y.strides = {Zero, One};

  af::ascir_op::Broadcast brc1("brc1");
  brc1.x = load1.y;
  brc1.attr.sched.axis = {z0.id, z1.id};
  brc1.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc1.y.dtype = ge::DT_FLOAT16;
  *brc1.y.axis = {z0.id, z1.id};
  *brc1.y.repeats = {s0, s1};
  *brc1.y.strides = {s1, One};

  af::ascir_op::Add add0("add0");
  add0.x1 = load0.y;
  add0.x2 = brc1.y;
  add0.attr.sched.axis = {z0.id, z1.id};
  add0.attr.api.compute_type = af::ComputeType::kComputeElewise;
  add0.y.dtype = ge::DT_FLOAT16;
  *add0.y.axis = {z0.id, z1.id};
  *add0.y.repeats = {s0, s1};
  *add0.y.strides = {s1, One};

  af::ascir_op::Store store0("store0");
  store0.x = add0.y;
  store0.attr.sched.axis = {z0.id, z1.id};
  store0.attr.api.compute_type = af::ComputeType::kComputeStore;
  store0.y.dtype = ge::DT_FLOAT16;
  *store0.y.axis = {z0.id, z1.id};
  *store0.y.repeats = {s0, s1};
  *store0.y.strides = {s1, One};

  af::ascir_op::Output y0("y0");
  y0.ir_attr.SetIndex(0);
  y0.x = store0.y;
  y0.attr.sched.axis = {z0.id, z1.id};
  y0.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  y0.attr.api.type = af::ApiType::kAPITypeBuffer;
  y0.y.dtype = ge::DT_FLOAT16;
  *y0.y.axis = {z0.id, z1.id};
  *y0.y.repeats = {s0, s1};
  *y0.y.strides = {s1, One};

  std::vector<autoschedule::AutoScheduleOutput> impl_graphs;
  AutoSchedule autoschedule(graph, impl_graphs);
  autoschedule.DoAutoSchedule();
  EXPECT_EQ(impl_graphs.size(), 4);

  EXPECT_EQ(impl_graphs[3].scheduled_graph.FindNode("brc1"), nullptr);
  EXPECT_EQ(af::AscGraphUtils::GetComputeGraph(impl_graphs[3].scheduled_graph)->GetAllNodesSize(), 7);
  const auto &impl1_add0 = impl_graphs[3].scheduled_graph.FindNode("add0");
  EXPECT_EQ(impl1_add0->GetAllInDataAnchorsSize(), 2);
  EXPECT_EQ(impl1_add0->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "load0");
  EXPECT_EQ(impl1_add0->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "load1");
}

/**
 *                 eq
 *              /      \
 *            /         \
 *          /            \
 *        /             brc1
 *       |(s0,s1,s2)     |(1,s1,s2)
 *     load0           load1
 *       |               |
 *     data0           data1
 */
TEST_F(AutoSchedulerUT, Autoschedule_autoschedule_brc_inline_static_not_support) {
  af::AscGraph graph("Autoschedule_autoschedule_brc_inline_static_not_support");
  auto s0 = af::Symbol(20);
  auto s1 = af::Symbol(32);

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  af::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  data0.attr.api.type = af::ApiType::kAPITypeBuffer;
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id};

  af::ascir_op::Load load0("load0");
  load0.x = data0.y;
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load0.y.dtype = ge::DT_FLOAT16;
  *load0.y.axis = {z0.id, z1.id};
  *load0.y.repeats = {s0, s1};
  *load0.y.strides = {s1, One};

  af::ascir_op::Data data1("data1", graph);
  data1.ir_attr.SetIndex(1);
  data1.attr.sched.axis = {z0.id, z1.id};
  data1.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  data1.attr.api.type = af::ApiType::kAPITypeBuffer;
  data1.y.dtype = ge::DT_FLOAT16;
  *data1.y.axis = {z0.id, z1.id};

  af::ascir_op::Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load1.y.dtype = ge::DT_FLOAT16;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {One, s1};
  *load1.y.strides = {Zero, One};

  af::ascir_op::Broadcast brc1("brc1");
  brc1.x = load1.y;
  brc1.attr.sched.axis = {z0.id, z1.id};
  brc1.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc1.y.dtype = ge::DT_FLOAT16;
  *brc1.y.axis = {z0.id, z1.id};
  *brc1.y.repeats = {s0, s1};
  *brc1.y.strides = {s1, One};

  af::ascir_op::Eq eq0("eq0");
  eq0.x1 = load0.y;
  eq0.x2 = brc1.y;
  eq0.attr.sched.axis = {z0.id, z1.id};
  eq0.attr.api.compute_type = af::ComputeType::kComputeElewise;
  eq0.y.dtype = ge::DT_FLOAT16;
  *eq0.y.axis = {z0.id, z1.id};
  *eq0.y.repeats = {s0, s1};
  *eq0.y.strides = {s1, One};

  af::ascir_op::Store store0("store0");
  store0.x = eq0.y;
  store0.attr.sched.axis = {z0.id, z1.id};
  store0.attr.api.compute_type = af::ComputeType::kComputeStore;
  store0.y.dtype = ge::DT_FLOAT16;
  *store0.y.axis = {z0.id, z1.id};
  *store0.y.repeats = {s0, s1};
  *store0.y.strides = {s1, One};

  af::ascir_op::Output y0("y0");
  y0.ir_attr.SetIndex(0);
  y0.x = store0.y;
  y0.attr.sched.axis = {z0.id, z1.id};
  y0.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  y0.attr.api.type = af::ApiType::kAPITypeBuffer;
  y0.y.dtype = ge::DT_FLOAT16;
  *y0.y.axis = {z0.id, z1.id};
  *y0.y.repeats = {s0, s1};
  *y0.y.strides = {s1, One};

  std::vector<autoschedule::AutoScheduleOutput> impl_graphs;
  AutoSchedule autoschedule(graph, impl_graphs);
  autoschedule.DoAutoSchedule();
  EXPECT_EQ(impl_graphs.size(), 2);
  EXPECT_EQ("Autoschedule_autoschedule_brc_inline_static_not_support_B0Y1", impl_graphs[1].scheduled_graph.GetName());
}

TEST_F(AutoSchedulerUT, test_vectorized_reduce_ra) {
  af::AscGraph graph("test");
  auto s0 = af::Symbol(20);
  auto s1 = af::Symbol(32);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data arg4_1("arg4_1", graph);
  arg4_1.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  arg4_1.attr.api.type = af::ApiType::kAPITypeBuffer;

  af::ascir_op::Max max("max");
  max.x = arg4_1.y;
  max.attr.api.compute_type = af::ComputeType::kComputeReduce;
  max.attr.api.type = af::ApiType::kAPITypeCompute;
  max.y.dtype = ge::DT_FLOAT;

  AxisGroup axis_group;
  axis_group.y_group = {z1.id};
  axis_group.n_group = {z0.id};
  axis_group.axes_order = {1};
  TilingCase tiling_case;
  tiling_case.ub_tiling_id_y = z1.id;
  tiling_case.ub_tiling_y = graph.TileSplit(z1.id, "z1T", "z1t");
  Scheduler scheduler(graph, axis_group, tiling_case, false);
  std::vector<int64_t> vectorized_axes;
  std::vector<size_t> vectorized_axes_order;
  scheduler.FindVectorizedAxes(vectorized_axes, vectorized_axes_order);

  std::vector<int64_t> golden_axes{3, 0};
  std::vector<size_t> golden_axes_order{4, 1};
  EXPECT_EQ(vectorized_axes, golden_axes);
  EXPECT_EQ(vectorized_axes_order, golden_axes_order);
}

TEST_F(AutoSchedulerUT, test_vectorized_reduce_ara) {
  af::AscGraph graph("test");
  auto s0 = af::Symbol(20);
  auto s1 = af::Symbol(32);
  auto s2 = af::Symbol(20);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data arg4_1("arg4_1", graph);
  arg4_1.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  arg4_1.attr.api.type = af::ApiType::kAPITypeBuffer;

  af::ascir_op::Max max("max");
  max.x = arg4_1.y;
  max.attr.api.compute_type = af::ComputeType::kComputeReduce;
  max.attr.api.type = af::ApiType::kAPITypeCompute;
  max.y.dtype = ge::DT_FLOAT;

  AxisGroup axis_group;
  axis_group.y_group = {z0.id, z2.id};
  axis_group.n_group = {z1.id};
  axis_group.axes_order = {0, 2};
  TilingCase tiling_case;
  tiling_case.ub_tiling_id_y = z0.id;
  tiling_case.ub_tiling_y = graph.TileSplit(z0.id, "z0T", "z0t");
  Scheduler scheduler(graph, axis_group, tiling_case, false);
  std::vector<int64_t> vectorized_axes;
  std::vector<size_t> vectorized_axes_order;
  scheduler.FindVectorizedAxes(vectorized_axes, vectorized_axes_order);

  std::vector<int64_t> golden_axes{4, 2, 1};
  std::vector<size_t> golden_axes_order{5, 7, 2};
  EXPECT_EQ(vectorized_axes, golden_axes);
  EXPECT_EQ(vectorized_axes_order, golden_axes_order);
}

/**
 *                 add
 *              /      \
 *            /         \
 *          /            \
 *        /             brc1
 *       |(s0,s1)        |(s0,1)
 *      brc0           load1
 *       |               |
 *     scalar          data1
 */
TEST_F(AutoSchedulerUT, Autoschedule_node_cache_marker_broadcast) {
  af::AscGraph graph("Autoschedule_node_cache_marker_broadcast");
  auto s0 = af::Symbol(20);
  auto s1 = af::Symbol(32);

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  af::ascir_op::Scalar data0("data0", graph);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  data0.attr.api.type = af::ApiType::kAPITypeBuffer;
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id};

  af::ascir_op::Broadcast brc0("brc0");
  brc0.x = data0.y;
  brc0.attr.sched.axis = {z0.id, z1.id};
  brc0.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc0.y.dtype = ge::DT_FLOAT16;
  *brc0.y.axis = {z0.id, z1.id};
  *brc0.y.repeats = {s0, s1};
  *brc0.y.strides = {s1, One};

  af::ascir_op::Data data1("data1", graph);
  data1.ir_attr.SetIndex(0);
  data1.attr.sched.axis = {z0.id, z1.id};
  data1.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  data1.attr.api.type = af::ApiType::kAPITypeBuffer;
  data1.y.dtype = ge::DT_FLOAT16;
  *data1.y.axis = {z0.id, z1.id};

  af::ascir_op::Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load1.y.dtype = ge::DT_FLOAT16;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {s0, One};
  *load1.y.strides = {One, Zero};

  af::ascir_op::Broadcast brc1("brc1");
  brc1.x = load1.y;
  brc1.attr.sched.axis = {z0.id, z1.id};
  brc1.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc1.y.dtype = ge::DT_FLOAT16;
  *brc1.y.axis = {z0.id, z1.id};
  *brc1.y.repeats = {s0, s1};
  *brc1.y.strides = {s1, One};

  af::ascir_op::Add add0("add0");
  add0.x1 = brc0.y;
  add0.x2 = brc1.y;
  add0.attr.sched.axis = {z0.id, z1.id};
  add0.attr.api.compute_type = af::ComputeType::kComputeElewise;
  add0.y.dtype = ge::DT_FLOAT16;
  *add0.y.axis = {z0.id, z1.id};
  *add0.y.repeats = {s0, s1};
  *add0.y.strides = {s1, One};

  af::ascir_op::Store store0("store0");
  store0.x = add0.y;
  store0.attr.sched.axis = {z0.id, z1.id};
  store0.attr.api.compute_type = af::ComputeType::kComputeStore;
  store0.y.dtype = ge::DT_FLOAT16;
  *store0.y.axis = {z0.id, z1.id};
  *store0.y.repeats = {One, s1};
  *store0.y.strides = {Zero, One};

  af::ascir_op::Output y0("y0");
  y0.ir_attr.SetIndex(0);
  y0.x = store0.y;
  y0.attr.sched.axis = {z0.id, z1.id};
  y0.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  y0.attr.api.type = af::ApiType::kAPITypeBuffer;
  y0.y.dtype = ge::DT_FLOAT16;
  *y0.y.axis = {z0.id, z1.id};

  // 验证防止重复判断
  af::ascir_op::Add add1("add1");
  add1.x1 = brc0.y;
  add1.x2 = brc1.y;
  add1.attr.sched.axis = {z0.id, z1.id};
  add1.attr.api.compute_type = af::ComputeType::kComputeElewise;
  add1.y.dtype = ge::DT_FLOAT16;
  *add1.y.axis = {z0.id, z1.id};
  *add1.y.repeats = {s0, s1};
  *add1.y.strides = {s1, One};

  af::ascir_op::Store store1("store1");
  store1.x = add0.y;
  store1.attr.sched.axis = {z0.id, z1.id};
  store1.attr.api.compute_type = af::ComputeType::kComputeStore;
  store1.y.dtype = ge::DT_FLOAT16;
  *store1.y.axis = {z0.id, z1.id};
  *store1.y.repeats = {One, s1};
  *store1.y.strides = {Zero, One};

  af::ascir_op::Output y1("y1");
  y1.ir_attr.SetIndex(0);
  y1.x = store0.y;
  y1.attr.sched.axis = {z0.id, z1.id};
  y1.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  y1.attr.api.type = af::ApiType::kAPITypeBuffer;
  y1.y.dtype = ge::DT_FLOAT16;
  *y1.y.axis = {z0.id, z1.id};

  std::vector<autoschedule::AutoScheduleOutput> impl_graphs;
  AutoSchedule autoschedule(graph, impl_graphs);
  autoschedule.DoAutoSchedule();
  EXPECT_EQ(impl_graphs.size(), 2);
  const auto &impl0 = impl_graphs[0];
  const auto &impl0_scalar_node = impl0.scheduled_graph.FindNode("data0");
  EXPECT_NE(impl0_scalar_node, nullptr);
  EXPECT_EQ(impl0_scalar_node->attr.sched.exec_condition, af::ExecuteCondition::kNoCache);
  const auto &impl0_brc0_node = impl0.scheduled_graph.FindNode("brc0");
  EXPECT_NE(impl0_brc0_node, nullptr);
  EXPECT_EQ(impl0_brc0_node->attr.sched.exec_condition, af::ExecuteCondition::kCacheBlockSplitFusedBroadcastAxis);
  const auto &impl0_brc1_node = impl0.scheduled_graph.FindNode("brc1");
  EXPECT_NE(impl0_brc1_node, nullptr);
  EXPECT_EQ(impl0_brc1_node->attr.sched.exec_condition, af::ExecuteCondition::kNoCache);
  const auto &impl0_add0_node = impl0.scheduled_graph.FindNode("add0");
  EXPECT_NE(impl0_add0_node, nullptr);
  EXPECT_EQ(impl0_add0_node->attr.sched.exec_condition, af::ExecuteCondition::kNoCache);

  const auto &impl1 = impl_graphs[1];
  const auto &impl1_scalar_node = impl1.scheduled_graph.FindNode("data0");
  EXPECT_NE(impl1_scalar_node, nullptr);
  EXPECT_EQ(impl1_scalar_node->attr.sched.exec_condition, af::ExecuteCondition::kNoCache);
  const auto &impl1_brc0_node = impl1.scheduled_graph.FindNode("brc0");
  EXPECT_NE(impl1_brc0_node, nullptr);
  EXPECT_EQ(impl1_brc0_node->attr.sched.exec_condition, af::ExecuteCondition::kCacheBlockSplitFusedBroadcastAxis);
  const auto &impl1_brc1_node = impl1.scheduled_graph.FindNode("brc1");
  EXPECT_NE(impl1_brc1_node, nullptr);
  EXPECT_EQ(impl1_brc1_node->attr.sched.exec_condition, af::ExecuteCondition::kCacheBlockSplitFusedBroadcastAxis);
  const auto &impl1_load1_node = impl1.scheduled_graph.FindNode("load1");
  EXPECT_NE(impl1_load1_node, nullptr);
  EXPECT_EQ(impl1_load1_node->attr.sched.exec_condition, af::ExecuteCondition::kCacheBlockSplitFusedBroadcastAxis);
  const auto &impl1_add0_node = impl1.scheduled_graph.FindNode("add0");
  EXPECT_NE(impl1_add0_node, nullptr);
  EXPECT_EQ(impl1_add0_node->attr.sched.exec_condition, af::ExecuteCondition::kNoCache);
}

/**
 *                 add
 *              /      \
 *            /      removepad
 *          /            \
 *        /             brc1
 *       |(s0,s1)        |(1,s1)
 *      load0          load1
 *       |              |
 *     data0          data1
 */
TEST_F(AutoSchedulerUT, Autoschedule_node_cache_marker_removepad) {
  af::AscGraph graph("Autoschedule_node_cache_marker_removepad");
  auto s0 = af::Symbol(20);
  auto s1 = af::Symbol(32);

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  af::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  data0.attr.api.type = af::ApiType::kAPITypeBuffer;
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id};

  af::ascir_op::Load load0("load0");
  load0.x = data0.y;
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load0.y.dtype = ge::DT_FLOAT16;
  *load0.y.axis = {z0.id, z1.id};
  *load0.y.repeats = {s0, s1};
  *load0.y.strides = {s1, One};

  af::ascir_op::Data data1("data1", graph);
  data1.ir_attr.SetIndex(1);
  data1.attr.sched.axis = {z0.id, z1.id};
  data1.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  data1.attr.api.type = af::ApiType::kAPITypeBuffer;
  data1.y.dtype = ge::DT_FLOAT16;
  *data1.y.axis = {z0.id, z1.id};

  af::ascir_op::Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load1.y.dtype = ge::DT_FLOAT16;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {One, s1};
  *load1.y.strides = {Zero, One};

  af::ascir_op::Broadcast brc1("brc1");
  brc1.x = load1.y;
  brc1.attr.sched.axis = {z0.id, z1.id};
  brc1.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc1.y.dtype = ge::DT_FLOAT16;
  *brc1.y.axis = {z0.id, z1.id};
  *brc1.y.repeats = {s0, s1};
  *brc1.y.strides = {s1, One};

  af::ascir_op::RemovePad remove_pad("remove_pad");
  remove_pad.x = brc1.y;
  remove_pad.attr.sched.axis = {z0.id, z1.id};
  remove_pad.attr.api.compute_type = af::ComputeType::kComputeElewise;
  remove_pad.y.dtype = ge::DT_FLOAT16;
  *remove_pad.y.axis = {z0.id, z1.id};
  *remove_pad.y.repeats = {s0, s1};
  *remove_pad.y.strides = {s1, One};

  af::ascir_op::Add add0("add0");
  add0.x1 = load0.y;
  add0.x2 = remove_pad.y;
  add0.attr.sched.axis = {z0.id, z1.id};
  add0.attr.api.compute_type = af::ComputeType::kComputeElewise;
  add0.y.dtype = ge::DT_FLOAT16;
  *add0.y.axis = {z0.id, z1.id};
  *add0.y.repeats = {s0, s1};
  *add0.y.strides = {s1, One};

  af::ascir_op::Store store0("store0");
  store0.x = add0.y;
  store0.attr.sched.axis = {z0.id, z1.id};
  store0.attr.api.compute_type = af::ComputeType::kComputeStore;
  store0.y.dtype = ge::DT_FLOAT16;
  *store0.y.axis = {z0.id, z1.id};
  *store0.y.repeats = {s0, s1};
  *store0.y.strides = {s1, One};

  af::ascir_op::Output y0("y0");
  y0.ir_attr.SetIndex(0);
  y0.x = store0.y;
  y0.attr.sched.axis = {z0.id, z1.id};
  y0.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  y0.attr.api.type = af::ApiType::kAPITypeBuffer;
  y0.y.dtype = ge::DT_FLOAT16;
  *y0.y.axis = {z0.id, z1.id};
  *y0.y.repeats = {s0, s1};
  *y0.y.strides = {s1, One};

  std::vector<autoschedule::AutoScheduleOutput> impl_graphs;
  AutoSchedule autoschedule(graph, impl_graphs);
  autoschedule.DoAutoSchedule();
  EXPECT_EQ(impl_graphs.size(), 2);
  const auto &impl0 = impl_graphs[0];
  const auto &impl0_scalar_node = impl0.scheduled_graph.FindNode("data0");
  EXPECT_NE(impl0_scalar_node, nullptr);
  EXPECT_EQ(impl0_scalar_node->attr.sched.exec_condition, af::ExecuteCondition::kNoCache);
  const auto &impl0_load1_node = impl0.scheduled_graph.FindNode("load1");
  EXPECT_NE(impl0_load1_node, nullptr);
  EXPECT_EQ(impl0_load1_node->attr.sched.exec_condition, af::ExecuteCondition::kCacheBlockSplitFusedBroadcastAxis);
  const auto &impl0_brc1_node = impl0.scheduled_graph.FindNode("brc1");
  EXPECT_NE(impl0_brc1_node, nullptr);
  EXPECT_EQ(impl0_brc1_node->attr.sched.exec_condition, af::ExecuteCondition::kCacheBlockSplitFusedBroadcastAxis);
  const auto &impl0_remove_pad_node = impl0.scheduled_graph.FindNode("remove_pad");
  EXPECT_NE(impl0_remove_pad_node, nullptr);
  EXPECT_EQ(impl0_remove_pad_node->attr.sched.exec_condition, af::ExecuteCondition::kCacheBlockSplitFusedBroadcastAxis);
  const auto &impl0_add0_node = impl0.scheduled_graph.FindNode("add0");
  EXPECT_NE(impl0_add0_node, nullptr);
  EXPECT_EQ(impl0_add0_node->attr.sched.exec_condition, af::ExecuteCondition::kNoCache);

  const auto &impl1 = impl_graphs[1];
  for (const auto &node : impl1.scheduled_graph.GetAllNodes()) {
    EXPECT_NE(node, nullptr);
    EXPECT_EQ(node->attr.sched.exec_condition, af::ExecuteCondition::kNoCache);
  }
}

TEST_F(AutoSchedulerUT, AutoSchedule_cube_gen_tilingGroup) {
  af::AscGraph graph("mutmul");

  auto s0 = graph.CreateSizeVar(31);
  auto s1 = graph.CreateSizeVar(1);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data data0("data0", graph);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  *data0.y.strides = {s1, af::ops::One};
  *data0.y.repeats = {s0, s1};
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT16;
  *load0.y.strides = {s1, af::ops::One};
  *load0.y.repeats = {s0, s1};

  Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT16;
  data1.attr.sched.axis = {z0.id, z1.id};
  *data1.y.axis = {z0.id, z1.id};
  data1.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  *data1.y.repeats = {One, One};
  *data1.y.strides = {Zero, Zero};
  data1.ir_attr.SetIndex(1);

  Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT16;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.strides = {Zero, Zero};
  *load1.y.repeats = {One, One};

  Data data2("data2", graph);
  data2.attr.sched.axis = {z0.id, z1.id};
  data2.y.dtype = ge::DT_FLOAT;
  *data2.y.axis = {z0.id, z1.id};
  data2.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  *data2.y.strides = {af::ops::Zero, af::ops::One};
  *data2.y.repeats = {af::ops::One, s1};
  data2.ir_attr.SetIndex(2);

  Load load2("load2");
  load2.attr.sched.axis = {z0.id, z1.id};
  load2.x = data2.y;
  *load2.y.axis = {z0.id, z1.id};
  load2.y.dtype = ge::DT_FLOAT;
  *load2.y.strides = {af::ops::Zero, af::ops::One};
  *load2.y.repeats = {af::ops::One, s1};

  MatMulOffset matmul("matmul");
  matmul.attr.sched.axis = {z0.id, z1.id};
  matmul.x1 = load0.y;
  matmul.x2 = load1.y;
  matmul.offset_w = load2.y;
  matmul.y.dtype = ge::DT_FLOAT;
  *matmul.y.axis = {z0.id, z1.id};
  *matmul.y.repeats = {s0, s1};
  *matmul.y.strides = {s1, af::ops::One};
  matmul.ir_attr.SetTranspose_x1(1);
  matmul.ir_attr.SetTranspose_x2(0);
  matmul.ir_attr.SetHas_relu(0);
  matmul.ir_attr.SetEnable_hf32(0);
  matmul.ir_attr.SetOffset_x(0);

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id};
  store_op.x = matmul.y;
  *store_op.y.axis = {z0.id, z1.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.strides = {s1, af::ops::One};
  *store_op.y.repeats = {s0, s1};
  store_op.ir_attr.SetOffset(af::ops::One);

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  std::vector<autoschedule::AutoScheduleOutput> impl_graphs;
  AutoSchedule autoSchedule(graph, impl_graphs);
  EXPECT_EQ(TilingGroup::GenTilingGroup(graph, autoSchedule.axes_group_), 0);
  EXPECT_EQ(autoSchedule.axes_group_.x_group.size(), 0UL);
  EXPECT_EQ(autoSchedule.axes_group_.y_group.size(), 2UL);
  EXPECT_EQ(autoSchedule.axes_group_.r_group.size(), 0UL);
}
}  // namespace optimize
