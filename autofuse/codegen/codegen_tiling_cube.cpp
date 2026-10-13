/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "codegen_tiling.h"
#include "codegen_tiling_data.h"
#include "codegen_tiling_utils.h"
#include "codegen_tiling_cube_wrapper.h"

#include <algorithm>
#include <cstring>

#include "acl/acl_rt.h"

#include "ascir_ops.h"
#include "ascir_ops_utils.h"
#include "common_utils.h"
#include "gen_tiling_impl.h"
#include "common/ge_common/debug/log.h"
#include "graph/ge_context.h"
#include "common/platform_context.h"
#include "graph/utils/type_utils.h"

namespace codegen {
using namespace af::ascir_op;
using namespace ascir;
using namespace af::ops;
using namespace ascgen_utils;

namespace {
constexpr int64_t kAscendcOpParaSize = 2 * 1024 * 1024;

std::string GetFullSocVersionForCubeTiling(const std::string &fallback_soc_version) {
  const char *soc_version = aclrtGetSocName();
  if (soc_version != nullptr && soc_version[0] != '\0') {
    return std::string(soc_version);
  }
  GELOGW("Failed to get full SoC_version for cube tiling, fallback to %s", fallback_soc_version.c_str());
  return fallback_soc_version;
}

std::string GetDeviceIdForCubeTiling() {
  return std::to_string(af::GetContext().DeviceId());
}

void AppendCvUbFusionStageSizeName(const ascir::FusedScheduledResult &fused_schedule_result, std::stringstream &ss) {
  if (!ascgen_utils::IsCubeUBFusedScheduled(fused_schedule_result) ||
      fused_schedule_result.node_idx_to_scheduled_results.empty() ||
      fused_schedule_result.node_idx_to_scheduled_results[0].empty() ||
      fused_schedule_result.node_idx_to_scheduled_results[0][0].schedule_groups.empty() ||
      fused_schedule_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.empty()) {
    return;
  }
  auto graph = fused_schedule_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0];
  ss << std::endl;
  ss << "extern \"C\" const char* GetCVUBFusionStageSizeName() {" << std::endl;
  for (auto axis : graph.GetAllAxis()) {
    if (axis->type == ascir::Axis::Type::kAxisTypeTileInner) {
      ss << "  return \"" << axis->name << "_size\";" << std::endl;
      GELOGD("gen GetCVUBFusionStageSizeName axis name:%s", axis->name.c_str());
    }
  }
  ss << "}" << std::endl;
}

}  // namespace
std::string DtypeToStr(ge::DataType dtype) {
  const std::map<ge::DataType, const ge::char_t *> kTypeName = {
      {ge::DT_FLOAT, "float32"}, {ge::DT_FLOAT16, "float16"}, {ge::DT_BF16, "bfloat16"}, {ge::DT_INT8, "int8"},
      {ge::DT_UINT8, "uint8"},   {ge::DT_INT16, "int16"},     {ge::DT_UINT16, "uint16"}, {ge::DT_INT32, "int32"},
      {ge::DT_UINT32, "uint32"}, {ge::DT_INT64, "int64"},     {ge::DT_UINT64, "uint64"}, {ge::DT_DOUBLE, "double"}};
  const auto &type_name_iter = kTypeName.find(dtype);
  if (type_name_iter == kTypeName.end()) {
    return "unknown";
  }
  return type_name_iter->second;
}

std::string TilingLib::GenCVTilingFunc() const {
  std::string find_cv_tiling_func = R"(
static int32_t g_basen_basem_align = 0;

int32_t get_g_basen_basem_align() {
  return g_basen_basem_align;
}

void set_g_basen_basem_align(int32_t value) {
  g_basen_basem_align = value;
}

extern "C" int64_t GenCVFusionTilingKey(char* config_file, int aiv_num, int ub_size) {
  uint64_t workspace_size;
  uint32_t block_dim;
  ResLimit limit;
  limit.aiv_num = aiv_num;
  limit.ub_size = ub_size - 256;
  set_g_basen_basem_align(basen_basem_align);
  OP_LOGI(OP_NAME, "basen_basem_align=%d, basen_align=%d, set_g_basen_basem_align=%d",
          basen_basem_align, basen_align, get_g_basen_basem_align());
  auto ret = AutofuseTilingWithConfig(config_file, &TilingDataValue, &workspace_size, &block_dim, &limit, 0);
  if (ret == -1) {
    uint32_t basen_basem_align_tmp = (uint32_t)basen_basem_align;
    // ub_size必大于 basen_basem_align_tmp
    limit.ub_size = limit.ub_size - basen_basem_align_tmp * cube_output_type_size; // 元素个数 * type_size
    set_g_basen_basem_align(basen_align);
    OP_LOGI(OP_NAME, "set_g_basen_basem_align=%d, ub_size=%u", get_g_basen_basem_align(), ub_size);
    ret = AutofuseTilingWithConfig(config_file, &TilingDataValue, &workspace_size, &block_dim, &limit, 1);
    if (ret == -1) {
      return -1;
    } else {
      return 1; // ub非全载模板返回1
    }
  }
  // need compute tile inner / basen * basem
  return 0; // ub全载模板返回0
}
)";
  return find_cv_tiling_func;
}

std::string TilingLib::GenTilingDataBlockDimAndWss() const {
  std::string get_block_dim_and_wss = R"(
extern "C" int GenTilingDataValueBlockDimAndWss(char* config_file, uint32_t aiv_num, uint32_t ub_size, uint64_t* workspace_size, uint32_t* block_dim) {
    ResLimit limit;
    limit.aiv_num = aiv_num;
    limit.ub_size = ub_size - 256;
    auto ret = AutofuseTilingWithConfig(config_file, &TilingDataValue, workspace_size, block_dim, &limit);
    if (ret == -1) {
        OP_LOGI(OP_NAME, "get_block_dim_and_wss return -1");
        return -1;
    } else {
        return 0;
    }
}
)";
  return get_block_dim_and_wss;
}

Status TilingLib::ExtractMatMulCubeInfoFromImplGraph(const af::AscGraph &impl_graph, MatMulCubeInfo &cube_info) const {
  for (const auto &node : impl_graph.GetAllNodes()) {
    if (node->attr.api.compute_type != af::ComputeType::kComputeCube) {
      continue;
    }
    ascgen_utils::MatMulAttr mm_attr_data;
    GE_CHK_STATUS_RET(ascgen_utils::ParseMatmulAttr(node, mm_attr_data), "ParseMatmulAttr failed for node[%s]",
                      node->GetName().c_str());

    cube_info.transpose_x1 = (mm_attr_data.transpose_x1 != 0) || (mm_attr_data.adj_x1 != 0);
    cube_info.transpose_x2 = (mm_attr_data.transpose_x2 != 0) || (mm_attr_data.adj_x2 != 0);
    cube_info.offset_x = mm_attr_data.offset_x;
    cube_info.is_batch = mm_attr_data.is_batch;
    cube_info.has_bias = mm_attr_data.is_bias;
    cube_info.has_offset_w = mm_attr_data.is_offset_w;
    cube_info.has_relu = (mm_attr_data.has_relu != 0);
    cube_info.enable_hf32 = mm_attr_data.enable_hf32;
    cube_info.matmul_node = node;

    GE_CHK_STATUS_RET(ascgen_utils::GetCubeOutputTypeSize(node, cube_info.type_size),
                      "GetCubeOutputTypeSize failed for node[%s]", node->GetName().c_str());

    GE_CHK_STATUS_RET(ascgen_utils::GetCubeInputNum(node, cube_info.input_num), "GetCubeInputNum failed for node[%s]",
                      node->GetName().c_str());

    return af::SUCCESS;
  }

  return af::FAILED;
}

Status TilingLib::ExtractMatMulCubeInfoFromFusedResult(const ascir::FusedScheduledResult &fused_schedule_result,
                                                       MatMulCubeInfo &cube_info) const {
  auto extract_from_impl_graphs = [this, &cube_info](const auto &schedule_groups) {
    for (const auto &schedule_group : schedule_groups) {
      for (const auto &impl_graph : schedule_group.impl_graphs) {
        if (ExtractMatMulCubeInfoFromImplGraph(impl_graph, cube_info) == af::SUCCESS) {
          return true;
        }
      }
    }
    return false;
  };

  auto process_scheduled_results = [&extract_from_impl_graphs](const auto &scheduled_results) {
    for (const auto &scheduled_result : scheduled_results) {
      if (scheduled_result.cube_type != ascir::CubeTemplateType::kDefault) {
        if (extract_from_impl_graphs(scheduled_result.schedule_groups)) {
          return true;
        }
      }
    }
    return false;
  };

  for (const auto &scheduled_results : fused_schedule_result.node_idx_to_scheduled_results) {
    if (process_scheduled_results(scheduled_results)) {
      return af::SUCCESS;
    }
  }

  return af::FAILED;
}

Status TilingLib::GetInputTensorInfoFromLoadNode(const ge::NodePtr &load_node, TensorInfo &tensor_info) const {
  GE_ASSERT_NOTNULL(load_node);
  const auto load_node_desc = load_node->GetOpDesc();
  GE_ASSERT_NOTNULL(load_node_desc);
  const auto load_tensor_desc = load_node_desc->MutableOutputDesc(0);
  GE_ASSERT_NOTNULL(load_tensor_desc);

  tensor_info.name = load_node->GetName();
  tensor_info.dtype = DtypeToStr(load_tensor_desc->GetDataType());
  tensor_info.format = ge::TypeUtils::FormatToSerialString(load_tensor_desc->GetFormat());

  auto tensor_attr = load_tensor_desc->GetAttrsGroup<ge::AscTensorAttr>();
  GE_ASSERT_NOTNULL(tensor_attr);

  for (const auto &repeat : tensor_attr->repeats) {
    tensor_info.shape.push_back(repeat);
  }

  tensor_info.ori_shape = tensor_info.shape;
  tensor_info.param_name = tensor_info.name;

  return af::SUCCESS;
}

Status TilingLib::ExtractInputsFromMatMulNode(const ge::AscNodePtr &matmul_node,
                                              std::vector<TensorInfo> &inputs) const {
  GE_ASSERT_NOTNULL(matmul_node);
  GE_ASSERT_NOTNULL(matmul_node->GetOpDesc());
  uint32_t input_num = matmul_node->GetOpDesc()->GetInputsSize();

  for (uint32_t i = 0U; i < input_num; ++i) {
    auto in_input_anchor = matmul_node->GetInDataAnchor(i);
    GE_ASSERT_NOTNULL(in_input_anchor);

    auto peer_out_anchor = in_input_anchor->GetPeerOutAnchor();
    GE_ASSERT_NOTNULL(peer_out_anchor);
    auto load_node = peer_out_anchor->GetOwnerNode();
    GE_ASSERT_NOTNULL(load_node);

    TensorInfo tensor_info;
    GE_CHK_STATUS(GetInputTensorInfoFromLoadNode(load_node, tensor_info), "Get matmul input info failed.");
    inputs.push_back(tensor_info);
  }

  return inputs.empty() ? af::FAILED : af::SUCCESS;
}

Status TilingLib::ExtractOutputsFromMatMulNode(const ge::AscNodePtr &matmul_node,
                                               std::vector<TensorInfo> &outputs) const {
  const auto mm_node_desc = matmul_node->GetOpDesc();
  GE_ASSERT_NOTNULL(mm_node_desc);
  const auto mm_tensor_desc = mm_node_desc->MutableOutputDesc(0);
  GE_ASSERT_NOTNULL(mm_tensor_desc);

  TensorInfo output_info;
  output_info.name = mm_tensor_desc->GetName() + "_output";
  output_info.dtype = DtypeToStr(mm_tensor_desc->GetDataType());
  output_info.format = ge::TypeUtils::FormatToSerialString(mm_tensor_desc->GetFormat());

  auto tensor_attr = mm_tensor_desc->GetAttrsGroup<ge::AscTensorAttr>();
  GE_ASSERT_NOTNULL(tensor_attr);
  for (const auto &repeat : tensor_attr->repeats) {
    output_info.shape.push_back(repeat);
  }

  output_info.ori_shape = output_info.shape;
  output_info.param_name = output_info.name;
  outputs.push_back(output_info);

  return af::SUCCESS;
}

std::string TilingLib::GenerateTensorInfoCode(const TensorInfo &tensor, const std::string &var_name) const {
  std::stringstream ss;
  ss << "TensorInfo " << var_name << ";\n";
  ss << "  " << var_name << ".name = \"" << tensor.name << "\";\n";
  ss << "  " << var_name << ".dtype = \"" << tensor.dtype << "\";\n";
  ss << "  " << var_name << ".format = \"" << tensor.format << "\";\n";
  ss << "  " << var_name << ".shape = " << VectorToStr(tensor.shape, '{', '}') << ";\n";
  ss << "  " << var_name << ".ori_shape = " << VectorToStr(tensor.ori_shape, '{', '}') << ";\n";
  ss << "  " << var_name << ".param_name = \"" << tensor.param_name << "\";\n";
  return ss.str();
}

std::string TilingLib::GenerateAttrInfoCode(const AttrInfo &attr, const std::string &var_name) const {
  std::stringstream ss;
  ss << "AttrInfo " << var_name << ";\n";
  ss << "  " << var_name << ".name = \"" << attr.name << "\";\n";
  ss << "  " << var_name << ".dtype = \"" << attr.dtype << "\";\n";

  if (attr.dtype == "bool") {
    ss << "  " << var_name << ".value_bool = " << (attr.value_bool ? "true" : "false") << ";\n";
  } else if (attr.dtype == "int") {
    ss << "  " << var_name << ".value_int = " << attr.value_int << ";\n";
  } else if (attr.dtype == "string") {
    ss << "  " << var_name << ".value_str = \"" << attr.value_str << "\";\n";
  } else if (attr.dtype == "float") {
    ss << "  " << var_name << ".value_float = " << attr.value_float << ";\n";
  }

  return ss.str();
}

void TilingLib::PrepareMatMulAttrs(const MatMulCubeInfo &cube_info, std::vector<AttrInfo> &attrs) const {
  AttrInfo attr1;
  attr1.name = cube_info.is_batch ? "adj_x1" : "transpose_x1";
  attr1.dtype = "bool";
  attr1.value_bool = cube_info.transpose_x1;
  attrs.push_back(attr1);

  AttrInfo attr2;
  attr2.name = cube_info.is_batch ? "adj_x2" : "transpose_x2";
  attr2.dtype = "bool";
  attr2.value_bool = cube_info.transpose_x2;
  attrs.push_back(attr2);

  AttrInfo attr3;
  attr3.name = "offset_x";
  attr3.dtype = "int";
  attr3.value_int = 0x80;
  attrs.push_back(attr3);

  AttrInfo attr4;
  if (cube_info.is_batch) {
    attr4.name = "enable_hf32";
    attr4.dtype = "bool";
    attr4.value_bool = cube_info.enable_hf32 ? 1 : 0;
  } else {
    attr4.name = "opImplMode";
    attr4.dtype = "int";
    attr4.value_int = cube_info.enable_hf32;
  }
  attrs.push_back(attr4);

  AttrInfo attr5;
  attr5.name = "ascendc_op_para_size";
  attr5.dtype = "int";
  attr5.value_int = kAscendcOpParaSize;
  attrs.push_back(attr5);

  AttrInfo attr6;
  attr6.name = "autofuse_has_bias";
  attr6.dtype = "bool";
  attr6.value_bool = cube_info.has_bias;
  attrs.push_back(attr6);

  AttrInfo attr7;
  attr7.name = "autofuse_has_offset_w";
  attr7.dtype = "bool";
  attr7.value_bool = cube_info.has_offset_w;
  attrs.push_back(attr7);
}

void TilingLib::GenerateTensorListCode(std::stringstream &code_ss, const std::vector<TensorInfo> &inputs,
                                       const std::vector<TensorInfo> &outputs) const {
  code_ss << "// Inputs\n";
  code_ss << "std::vector<TensorInfo> inputs;\n";
  for (size_t i = 0U; i < inputs.size(); ++i) {
    std::string var_name = "input_" + std::to_string(i);
    code_ss << GenerateTensorInfoCode(inputs[i], var_name);
    code_ss << "  inputs.push_back(" << var_name << ");\n";
  }
  code_ss << "\n";

  code_ss << "// Outputs\n";
  code_ss << "std::vector<TensorInfo> outputs;\n";
  for (size_t i = 0U; i < outputs.size(); ++i) {
    std::string var_name = "output_" + std::to_string(i);
    code_ss << GenerateTensorInfoCode(outputs[i], var_name);
    code_ss << "  outputs.push_back(" << var_name << ");\n";
  }
  code_ss << "\n";
}

void TilingLib::GenerateTilingCallCode(std::stringstream &code_ss, bool is_batch) const {
  code_ss << "// Call DoMatMulTiling\n";
  code_ss << "CubeKernelTilingWrapper wrapper;\n";
  code_ss << "TilingResult result = wrapper.DoMatMulTiling(compile_info, inputs, outputs, attrs, "
          << (is_batch ? "true" : "false") << ");\n";
  code_ss << "ws_size = result.workspace_size;\n";

  // 保存字节流到全局变量（用于静态shape常量生成）
  code_ss << "// Save tiling bytes for const generation in static shape\n";
  code_ss << "g_matmul_tiling_bytes = result.tiling_data;\n";
  code_ss << "std::memset(tiling_data->matmul_tiling_data, 0, sizeof(tiling_data->matmul_tiling_data));\n";
  code_ss << "size_t copy_size = std::min(result.tiling_data.size(), sizeof(tiling_data->matmul_tiling_data));\n";
  code_ss << "std::memcpy(tiling_data->matmul_tiling_data, result.tiling_data.data(), copy_size);\n";

  code_ss << "cube_block_dim = result.cube_used_core_num;\n";
  code_ss << "basem = result.cube_base_m;\n";
  code_ss << "basen = result.cube_base_n;\n";
  code_ss << "tiling_key = result.tiling_key;\n";
  code_ss << "OP_LOGI(OP_NAME, \"tiling_key=%ld, ws_size=%ld, cube_block_dim=%d, basem=%d, basen=%d\", tiling_key, "
             "ws_size, cube_block_dim, basem, basen);\n";
}

std::string TilingLib::GenerateMatMulTilingCode(const CompileInfo &compile_info, const std::vector<TensorInfo> &inputs,
                                                const std::vector<TensorInfo> &outputs,
                                                const std::vector<AttrInfo> &attrs, bool is_batch) const {
  std::stringstream code_ss;

  // 注意：全局变量g_matmul_tiling_bytes已在CallCubeTiling函数外部单独定义
  // 此处不重复定义

  code_ss << "// CompileInfo\n";
  code_ss << "CompileInfo compile_info;\n";
  code_ss << "compile_info.soc_version = \"" << compile_info.soc_version << "\";\n";
  code_ss << "compile_info.core_type = \"" << compile_info.core_type << "\";\n";
  auto device_id_iter = compile_info.extra_info.find("device_id");
  if (device_id_iter != compile_info.extra_info.end()) {
    code_ss << "compile_info.device_id = \"" << device_id_iter->second << "\";\n";
  }
  code_ss << "compile_info.aicore_num = " << compile_info.aicore_num << ";\n";
  code_ss << "compile_info.aiv_num = " << compile_info.aiv_num << ";\n";
  code_ss << "compile_info.op_kernel_lib = \"" << compile_info.op_kernel_lib << "\";\n";
  code_ss << "compile_info.op_impl_mode = \"" << compile_info.op_impl_mode << "\";\n\n";

  GenerateTensorListCode(code_ss, inputs, outputs);

  code_ss << "// Attributes\n";
  code_ss << "std::vector<AttrInfo> attrs;\n";
  for (size_t i = 0U; i < attrs.size(); ++i) {
    std::string var_name = "attr_" + std::to_string(i);
    code_ss << GenerateAttrInfoCode(attrs[i], var_name);
    code_ss << "  attrs.push_back(" << var_name << ");\n";
  }
  code_ss << "\n";

  GenerateTilingCallCode(code_ss, is_batch);
  return code_ss.str();
}

std::string TilingLib::ProcessCubeKernelTilingFromFusedResult(
    const ascir::FusedScheduledResult &fused_schedule_result) const {
  MatMulCubeInfo cube_info;
  GE_ASSERT_SUCCESS(ExtractMatMulCubeInfoFromFusedResult(fused_schedule_result, cube_info),
                    "[Extract][MatMulCubeInfo]Failed to extract MatMul cube info from FusedScheduledResult");

  std::vector<TensorInfo> inputs;
  GE_ASSERT_SUCCESS(ExtractInputsFromMatMulNode(cube_info.matmul_node, inputs),
                    "[Extract][Inputs]Failed to extract inputs from MatMul node[%s]",
                    cube_info.matmul_node->GetName().c_str());

  std::vector<TensorInfo> outputs;
  GE_ASSERT_SUCCESS(ExtractOutputsFromMatMulNode(cube_info.matmul_node, outputs),
                    "[Extract][Outputs]Failed to extract outputs from MatMul node[%s]",
                    cube_info.matmul_node->GetName().c_str());

  CompileInfo compile_info;
  ge::PlatformInfo platform_info;
  GE_ASSERT_SUCCESS(ge::PlatformContext::GetInstance().GetPlatformInfo(platform_info),
                    "Failed to get platform info for cube tiling.");
  compile_info.soc_version = GetFullSocVersionForCubeTiling(platform_info.soc_ver);
  compile_info.core_type = "AiCore";
  compile_info.extra_info["device_id"] = GetDeviceIdForCubeTiling();
  compile_info.aicore_num = 0;
  compile_info.aiv_num = 0;
  compile_info.op_kernel_lib = "";
  compile_info.op_impl_mode = "";

  std::vector<AttrInfo> attrs;
  PrepareMatMulAttrs(cube_info, attrs);

  return GenerateMatMulTilingCode(compile_info, inputs, outputs, attrs, cube_info.is_batch);
}

std::map<std::string, std::string> TilingLib::GenerateCVFusionStatic(
    const ::ascir::FusedScheduledResult &fused_schedule_result,
    const ascir::FusedScheduledResult &elemwise_schedule_result, const std::map<std::string, std::string> &shape_info,
    const std::string &pgo_dir, const std::string &core_num) const {
  std::stringstream ss;
  ss << TilingFuncDef(fused_schedule_result, elemwise_schedule_result, shape_info, pgo_dir, core_num) << std::endl;
  ss << TilingData("Autofuse").GenerateConst(elemwise_schedule_result, false) << std::endl;

  ss << GenCVTilingFunc();
  AppendCvUbFusionStageSizeName(elemwise_schedule_result, ss);
  ss << GenTilingDataBlockDimAndWss();
  return {{kTilingDefAndConstIdentify, ss.str()}};
}

std::map<std::string, std::string> TilingLib::GenerateCVFusionDynamic(
    const ascir::FusedScheduledResult &fused_schedule_result,
    const ascir::FusedScheduledResult &elemwise_schedule_result, const std::map<std::string, std::string> &shape_info,
    const std::string &pgo_dir, const std::string &core_num) const {
  std::stringstream ss;
  std::stringstream call_cube_tiling;
  std::stringstream shape_symbol;
  for (auto vars : GetFrontendShapeVars(fused_schedule_result)) {
    if (!(vars.IsConstExpr())) {
      std::string var_define = std::string(vars.Str().get());
      auto it = shape_info.find(var_define);
      if (it != shape_info.end()) {
        shape_symbol << "uint32_t " << var_define << ", ";
      }
    }
  }
  shape_symbol << "int64_t &ws_size, uint32_t &cube_block_dim, int64_t &tiling_key, uint32_t &basem, uint32_t &basen, "
                  "CVAutofuseTilingData *tiling_data";
  call_cube_tiling << "using namespace ge::autofuse;" << std::endl;
  AppendCvBaseAlignHelperDefs(call_cube_tiling);
  MatMulCubeInfo cube_info;
  GE_ASSERT_SUCCESS(ExtractMatMulCubeInfoFromFusedResult(fused_schedule_result, cube_info),
                    "[Extract][MatMulCubeInfo]Failed to extract MatMul cube info from FusedScheduledResult");
  AppendCvSafetyMixModeHelperDefs(call_cube_tiling, cube_info.is_batch);

  // 在CallCubeTiling函数之前定义全局变量（用于静态shape常量生成）
  call_cube_tiling << "// Global variable to store tiling bytes for const generation in static shape\n";
  call_cube_tiling << "std::vector<uint8_t> g_matmul_tiling_bytes;\n\n";

  call_cube_tiling << "extern \"C\" void CallCubeTiling(" << shape_symbol.str() << ") {" << std::endl;
  call_cube_tiling << ProcessCubeKernelTilingFromFusedResult(fused_schedule_result) << std::endl;
  call_cube_tiling << "}" << std::endl;
  ss << call_cube_tiling.str();

  AppendCvUbFusionStageSizeName(elemwise_schedule_result, ss);
  ss << TilingFuncDef(fused_schedule_result, elemwise_schedule_result, shape_info, pgo_dir, core_num) << std::endl;

  std::map<std::string, std::string> result;
  result[kTilingDefAndConstIdentify] = ss.str();
  result[kCubeKernelTilingWrapperHpp] = kCubeKernelTilingWrapperHppValue;
  result[kCubeKernelTilingWrapperCpp] = kCubeKernelTilingWrapperInclude;
  result[kCubeKernelTilingWrapperCpp] += kCubeKernelTilingWrapperCppValue;
  return result;
}

}  // namespace codegen
