/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "ascir_ops.h"
#include "codegen_kernel.h"
#include "elewise/remove_pad_api_call.h"

using namespace af::ops;
using namespace codegen;
using namespace af::ascir_op;
using namespace testing;

namespace {
std::vector<af::Axis> MakeAxes(const std::vector<af::SizeVar> &sizes) {
  std::vector<af::Axis> axes;
  for (size_t i = 0UL; i < sizes.size(); ++i) {
    const auto type = i == 0UL ? af::Axis::Type::kAxisTypeTileOuter
                               : (i == 1UL ? af::Axis::Type::kAxisTypeTileInner : af::Axis::Type::kAxisTypeOriginal);
    axes.push_back(af::Axis{
        .id = static_cast<ascir::AxisId>(i), .name = "z" + std::to_string(i), .type = type, .size = sizes[i].expr});
  }
  return axes;
}

void InitTiler(codegen::Tiler &tiler, const std::vector<af::SizeVar> &sizes, const std::vector<af::Axis> &axes) {
  for (const auto &size : sizes) {
    tiler.AddSizeVar(size);
  }
  for (const auto &axis : axes) {
    tiler.AddAxis(axis);
  }
}

void PrepareAttr(af::AscTensor &tensor, const std::vector<af::Axis> &axes,
                 const std::vector<ascir::AxisId> &vectorized_axis,
                 const std::vector<af::Expression> &vectorized_strides) {
  af::Expression dense = One;
  tensor.attr.strides.assign(axes.size(), One);
  for (size_t i = axes.size(); i > 0UL; --i) {
    tensor.attr.strides[i - 1UL] = dense;
    dense = dense * axes[i - 1UL].size;
  }
  tensor.attr.axis.clear();
  tensor.attr.repeats.clear();
  for (const auto &axis : axes) {
    tensor.attr.axis.push_back(axis.id);
    tensor.attr.repeats.push_back(axis.size);
  }
  tensor.attr.vectorized_axis = vectorized_axis;
  tensor.attr.vectorized_strides = vectorized_strides;
  tensor.attr.mem.tensor_id = 1;
  tensor.attr.mem.alloc_type = af::AllocType::kAllocTypeBuffer;
  tensor.attr.mem.position = af::Position::kPositionVecIn;
  tensor.attr.opt.merge_scope = af::kIdNone;
  tensor.attr.buf.id = 2;
}

std::vector<Tensor> MakeTensors(const af::AscTensor &attr, std::string &dtype_name,
                                const std::vector<uint32_t> &vectorized_axis_pos,
                                const std::vector<ascir::SizeExpr> &axis_size) {
  std::vector<Tensor> tensors;
  tensors.emplace_back(attr, dtype_name);
  tensors.emplace_back(attr, dtype_name);
  for (auto &tensor : tensors) {
    tensor.is_constant = true;
    tensor.vectorized_axis_pos = vectorized_axis_pos;
    tensor.axis_size = axis_size;
  }
  return tensors;
}

std::string GenerateRemovePad(const std::vector<af::SizeVar> &sizes, const std::vector<ascir::AxisId> &vectorized_axis,
                              const std::vector<af::Expression> &vectorized_strides) {
  const auto axes = MakeAxes(sizes);
  codegen::Tiler tiler;
  InitTiler(tiler, sizes, axes);
  codegen::TPipe tpipe("tpipe", tiler);
  af::AscGraph graph("test");
  af::ascir_op::Data x("x", graph);
  af::AscTensor tensor = graph.FindNode("x")->outputs[0];
  PrepareAttr(tensor, axes, vectorized_axis, vectorized_strides);

  std::string dtype_name;
  Tensor::DtypeName(tensor.attr.dtype, dtype_name);
  std::vector<uint32_t> vectorized_axis_pos;
  std::vector<ascir::SizeExpr> axis_size;
  for (const auto &axis : axes) {
    axis_size.push_back(axis.size);
  }
  for (const auto axis_id : vectorized_axis) {
    vectorized_axis_pos.push_back(static_cast<uint32_t>(axis_id));
  }
  auto tensors = MakeTensors(tensor, dtype_name, vectorized_axis_pos, axis_size);

  std::vector<std::reference_wrapper<const Tensor>> inputs{std::cref(tensors[0])};
  std::vector<std::reference_wrapper<const Tensor>> outputs{std::cref(tensors[1])};
  const std::vector<ascir::AxisId> current_axis;
  codegen::RemovePadApiCall call("removepad");
  std::string result;
  EXPECT_EQ(call.Generate(tpipe, current_axis, inputs, outputs, result), af::SUCCESS);
  return result;
}
}  // namespace

class RemovePadApiCallTest : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(RemovePadApiCallTest, RemovePadApiCall_01) {
  const std::vector<af::SizeVar> sizes{af::SizeVar(af::Symbol("s0")), af::SizeVar(af::Symbol("s1")),
                                       af::SizeVar(af::Symbol("s2"))};
  const auto result = GenerateRemovePad(sizes, {1, 2}, {sizes[2].expr, One});
  EXPECT_NE(result.find("RemovePadExtend("), std::string::npos);
  EXPECT_EQ(result.find("KernelUtils::SizeAlign("), std::string::npos);
}

TEST_F(RemovePadApiCallTest, RemovePadApiCall_02) {
  const std::vector<af::SizeVar> sizes{af::SizeVar(af::Symbol("s0")), af::SizeVar(af::Symbol("s1"))};
  const auto result = GenerateRemovePad(sizes, {1}, {sizes[1].expr, One});
  EXPECT_NE(result.find("RemovePadExtend("), std::string::npos);
  EXPECT_NE(result.find("KernelUtils::SizeAlign("), std::string::npos);
}

TEST_F(RemovePadApiCallTest, RemovePadApiCall_03) {
  const std::vector<af::SizeVar> sizes{af::SizeVar(af::Symbol("s0")), af::SizeVar(af::Symbol("s1")),
                                       af::SizeVar(af::Symbol("s2"))};
  const auto result = GenerateRemovePad(sizes, {1, 2}, {sizes[2].expr + sizes[2].expr, sizes[2].expr + sizes[2].expr});
  EXPECT_NE(result.find("RemovePadExtend("), std::string::npos);
  EXPECT_EQ(result.find("Adds("), std::string::npos);
}

TEST_F(RemovePadApiCallTest, RemovePadApiCall_04) {
  const std::vector<af::SizeVar> sizes{af::SizeVar(af::Symbol("s0")), af::SizeVar(af::Symbol("s1")),
                                       af::SizeVar(af::Symbol("s2"))};
  const auto result = GenerateRemovePad(sizes, {1, 2}, {sizes[2].expr + sizes[2].expr, One});
  EXPECT_NE(result.find("RemovePadExtend("), std::string::npos);
  EXPECT_EQ(result.find("KernelUtils::SizeAlign("), std::string::npos);
}

TEST_F(RemovePadApiCallTest, RemovePadApiCall_05) {
  const std::vector<af::SizeVar> sizes{af::SizeVar(af::Symbol("s0")), af::SizeVar(af::Symbol("s1")),
                                       af::SizeVar(af::Symbol("s2"))};
  const auto result = GenerateRemovePad(sizes, {1, 2}, {sizes[2].expr, Zero});
  EXPECT_NE(result.find("RemovePadExtend("), std::string::npos);
  EXPECT_EQ(result.find("KernelUtils::SizeAlign("), std::string::npos);
}
