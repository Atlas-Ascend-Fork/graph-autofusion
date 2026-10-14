/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <fstream>
#include <gtest/gtest.h>
#include <exception>
#include <filesystem>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include "codegen_kernel.h"
#include "codegen.h"
#include "optimize.h"
#include "share_graph.h"
#include "../backend_codegen_common.h"
#include "ascir_ops.h"
#include "ascir_ops_utils.h"
#include "ascgraph_info_complete.h"
#include "ascgen_log.h"
#include "common/platform_context.h"
#include "runtime_stub.h"
#include "common_utils.h"
#include "../../../../../v35/codegen/reg_api_call/reg_api_call_utils.h"

namespace {
codegen::Tensor MakeCvFusionTensor(const ascir::TensorAttr &tensor_attr) {
  std::string dtype_name;
  EXPECT_EQ(codegen::Tensor::DtypeName(tensor_attr.attr.dtype, dtype_name), af::SUCCESS);
  codegen::Tensor tensor(tensor_attr, dtype_name);
  EXPECT_EQ(tensor.Init(), af::SUCCESS);
  return tensor;
}

struct CvFusionDataCopyTensors {
  codegen::Tensor gm;
  codegen::Tensor ub;
};

CvFusionDataCopyTensors MakeCvFusionDataCopyTensors() {
  af::AscGraph graph("cv_fusion_data_copy_params");
  af::ascir_op::Data data("data", graph);
  af::ascir_op::Load load("load");
  graph.AddNode(load);
  load.x = data.y;

  auto gm_node = graph.FindNode("data");
  gm_node->outputs[0].attr.dtype = ge::DT_FLOAT16;
  gm_node->outputs[0].attr.mem.tensor_id = 0;
  gm_node->outputs[0].attr.mem.reuse_id = 0;
  gm_node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  gm_node->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareGM;
  gm_node->outputs[0].attr.mem.position = af::Position::kPositionGM;
  gm_node->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto ub_node = graph.FindNode("load");
  ub_node->outputs[0].attr.dtype = ge::DT_FLOAT16;
  ub_node->outputs[0].attr.mem.tensor_id = 1;
  ub_node->outputs[0].attr.mem.reuse_id = 0;
  ub_node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeBuffer;
  ub_node->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareUB;
  ub_node->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  ub_node->outputs[0].attr.opt.merge_scope = af::kIdNone;
  return {MakeCvFusionTensor(gm_node->outputs[0]), MakeCvFusionTensor(ub_node->outputs[0])};
}
}  // namespace

class TestBackendMatmulEleBrc : public testing::Test {
 protected:
  void SetUp() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
    ge::PlatformContext::GetInstance().Reset();
    af::InstallV35SocPerfRuntimeStub();
  }
  void TearDown() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
    ge::RuntimeStub::Reset();
  }
};

TEST_F(TestBackendMatmulEleBrc, MatmulEleBrcCodegen) {
  bool gen_success = true;
  const std::map<std::string, std::string> shape_info;
  auto graph = ascir::ShareGraph::LoadMatmulElewiseBrcFusedGraph();

  std::cout << "KERNEL_SRC_LIST=" << KERNEL_SRC_LIST << std::endl;
  std::vector<std::string> parts = splitString(KERNEL_SRC_LIST, ':');
  std::string kernel_src_file_name =
      "matmul_elemwise_brc_test_kernel_ub.cpp";  // matmul_elemwise_brc_test_kernel_ub.cpp
  std::string tiling_src_file_name =
      "matmul_elemwise_brc_test_tiling_ub.cpp";          // matmul_elemwise_brc_test_tiling_ub.cpp
  std::string tiling_data_src_file_name = parts.back();  // autofuse_tiling_data.h

  try {
    optimize::Optimizer optimizer(optimize::OptimizerOptions{});
    codegen::Codegen codegen(codegen::CodegenOptions{});

    std::fstream kernel_file(kernel_src_file_name, std::ios::out);
    std::fstream tiling_file(tiling_src_file_name, std::ios::out);
    std::fstream tiling_data_file(tiling_data_src_file_name, std::ios::out);

    std::vector<::ascir::ScheduledResult> schedule_results;
    ascir::FusedScheduledResult fused_schedule_result;
    fused_schedule_result.node_idx_to_scheduled_results.push_back(schedule_results);
    EXPECT_EQ(optimizer.Optimize(graph, fused_schedule_result), 0);
    codegen::CodegenResult result;
    ::ascir::FusedScheduledResult ub_schedule_result = fused_schedule_result;
    ::ascir::FusedScheduledResult common_schedule_result = fused_schedule_result;
    if (ascgen_utils::IsCubeFusedScheduled(fused_schedule_result)) {
      // 过滤CVFusion的UBResult ub模板结果
      ascgen_utils::FilterCVFusionUBResult(ub_schedule_result);
      // 过滤CVFusion的CommonResult 兜底模板结果
      ascgen_utils::FilterCVFusionCommonResult(common_schedule_result);
    }

    // 分别生成ub和common模板的kernel和tiling
    EXPECT_EQ(codegen.Generate(shape_info, ub_schedule_result, result), 0);
    const std::string ub_kernel = RemoveSubDirInclude(result.kernel);
    EXPECT_FALSE(ub_kernel.empty());
    EXPECT_FALSE(result.tiling.empty());
    EXPECT_FALSE(result.tiling_data.empty());
    kernel_file << ub_kernel;
    tiling_file << result.tiling;
    tiling_data_file << result.tiling_data;

    kernel_src_file_name = "matmul_elemwise_brc_test_kernel_common.cpp";  // matmul_elemwise_brc_test_kernel_common.cpp
    tiling_src_file_name = "matmul_elemwise_brc_test_tiling_common.cpp";  // matmul_elemwise_brc_test_tiling_common.cpp
    tiling_data_src_file_name = parts.back();                             // autofuse_tiling_data.h
    std::fstream kernel_file_common(kernel_src_file_name, std::ios::out);
    std::fstream tiling_file_common(tiling_src_file_name, std::ios::out);
    std::fstream tiling_data_file_common(tiling_data_src_file_name, std::ios::out);
    codegen::CodegenResult result_common;
    EXPECT_EQ(codegen.Generate(shape_info, common_schedule_result, result_common), 0);
    const std::string common_kernel = RemoveSubDirInclude(result_common.kernel);
    EXPECT_FALSE(common_kernel.empty());
    EXPECT_FALSE(result_common.tiling.empty());
    EXPECT_FALSE(result_common.tiling_data.empty());
    kernel_file_common << common_kernel;
    tiling_file_common << result_common.tiling;
    tiling_data_file_common << result_common.tiling_data;
  } catch (...) {
    gen_success = false;
  }

  EXPECT_EQ(gen_success, true);
}

TEST_F(TestBackendMatmulEleBrc, MatmulToIntCastCvCodegen) {
  auto graph = ascir::ShareGraph::LoadMatmulToIntCastFusedGraph();
  GenerateCvBackendUbKernelWithCheck(
      graph, "matmul_to_int_cast_test_kernel_ub.cpp", "matmul_to_int_cast_test_tiling_ub.cpp", "autofuse_tiling_data.h",
      [](const std::string &kernel) {
        EXPECT_NE(kernel.find("AscendC::Cast(local_5[0], local_4[0], AscendC::RoundMode::CAST_RINT, "
                              "{ConvertToUint32(curAivM), ConvertToUint32(curAivN)}, "
                              "{ConvertToUint32(((curAivN + 8 - 1) / 8 * 8)), ConvertToUint32(1)}, "
                              "{ConvertToUint32(curAlignN), ConvertToUint32(1)});"),
                  std::string::npos);
        EXPECT_NE(kernel.find("CastExtend(local_6[0], local_5[0], "
                              "{ConvertToUint32(curAivM), ConvertToUint32(curAivN)}, "
                              "{ConvertToUint32(((curAivN + 8 - 1) / 8 * 8)), ConvertToUint32(1)}, "
                              "{ConvertToUint32(((curAivN + 8 - 1) / 8 * 8)), ConvertToUint32(1)});"),
                  std::string::npos);
        EXPECT_NE(kernel.find("AscendC::Cast(local_7[0], local_6[0], AscendC::RoundMode::CAST_TRUNC, "
                              "{ConvertToUint32(curAivM), ConvertToUint32(curAivN)}, "
                              "{ConvertToUint32(((curAivN + 8 - 1) / 8 * 8)), ConvertToUint32(1)}, "
                              "{ConvertToUint32(((curAivN + 8 - 1) / 8 * 8)), ConvertToUint32(1)});"),
                  std::string::npos);
        EXPECT_NE(kernel.find("AscendC::Cast(local_9[0], local_8[0], AscendC::RoundMode::CAST_FLOOR, "
                              "{ConvertToUint32(curAivM), ConvertToUint32(curAivN)}, "
                              "{ConvertToUint32(((curAivN + 8 - 1) / 8 * 8)), ConvertToUint32(1)}, "
                              "{ConvertToUint32(((curAivN + 8 - 1) / 8 * 8)), ConvertToUint32(1)});"),
                  std::string::npos);
        EXPECT_NE(kernel.find("KernelUtils::BlkAlign<int32_t>(curAivN)"), std::string::npos);
      });
}

TEST_F(TestBackendMatmulEleBrc, CvFusionDataCopyParamsUseDtypeAlignedFallback) {
  codegen::Tiler tiler;
  const CvFusionDataCopyTensors tensors = MakeCvFusionDataCopyTensors();
  std::string dtype_name = "half";

  codegen::ApiCallContext normal_context;
  EXPECT_EQ(codegen::GetCvInputAlignedSize(normal_context, tensors.gm, "curAivN"), "curAivN");
  codegen::ApiCallContext cv_context;
  cv_context.stage = codegen::ComputeStage::kCVFuseStage1;
  EXPECT_EQ(codegen::GetCvInputAlignedSize(cv_context, tensors.gm, "curAivN"), "((curAivN + 16 - 1) / 16 * 16)");

  codegen::TPipe ub_fuse_tpipe("tpipe", tiler);
  ub_fuse_tpipe.cv_fusion_type = ascir::CubeTemplateType::kUBFuse;
  codegen::CodegenApiParam ub_fuse_api_param;
  codegen::DmaSpecificParams ub_fuse_dma_params;
  codegen::BuildDataCopyApiParamInCVFusion(ub_fuse_tpipe, ub_fuse_api_param, ub_fuse_dma_params, tensors.gm, tensors.ub,
                                           dtype_name, true);
  EXPECT_EQ(ub_fuse_api_param.template_params[0], "AscendC::PaddingMode::Normal");
  EXPECT_EQ(ub_fuse_dma_params.data_copy_params.block_count.DebugStr(), "curAivM");
  EXPECT_EQ(ub_fuse_dma_params.data_copy_params.block_len.DebugStr(), "curAivN");
  EXPECT_EQ(ub_fuse_dma_params.data_copy_params.src_stride.DebugStr(), "(shapeN - curAivN)");
  EXPECT_EQ(ub_fuse_dma_params.data_copy_params.dst_stride.DebugStr(),
            "(KernelUtils::BlkAlign<half>(curAivN) - curAivN)");

  codegen::TPipe common_tpipe("tpipe", tiler);
  codegen::CodegenApiParam common_api_param;
  codegen::DmaSpecificParams common_dma_params;
  codegen::BuildDataCopyApiParamInCVFusion(common_tpipe, common_api_param, common_dma_params, tensors.gm, tensors.ub,
                                           dtype_name, true);
  EXPECT_EQ(common_dma_params.data_copy_params.block_len.DebugStr(), "load_block_len");
  EXPECT_EQ(common_dma_params.data_copy_params.src_stride.DebugStr(), "load_src_stride");
  EXPECT_EQ(common_dma_params.data_copy_params.dst_stride.DebugStr(), "load_dst_stride");
  ASSERT_EQ(common_api_param.api_post_process.size(), 1U);
  EXPECT_NE(common_api_param.api_post_process[0].find("AscendC::GatherMask"), std::string::npos);
  EXPECT_NE(common_api_param.api_post_process[0].find("KernelUtils::BlkAlign<half>(curAlignN)"), std::string::npos);

  codegen::CvApi2DParams cv_params;
  cv_params.first_dim = "curAivM";
  cv_params.last_dim = "curAivN";
  EXPECT_EQ(codegen::GenCvUint16Dims(cv_params), "{static_cast<uint16_t>(curAivM), static_cast<uint16_t>(curAivN)}");
  EXPECT_EQ(codegen::GenCvUint16Stride("curAlignN"), "{static_cast<uint16_t>(curAlignN), static_cast<uint16_t>(1)}");
}
