/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "micro_api_call_factory.h"
#include "ascir_ops.h"

#include "micro_load_api_call.h"
#include "micro_dtype_utils.h"

namespace {
using codegen::DTYPE_SIZE_1BYTE;
using codegen::DTYPE_SIZE_2BYTE;
using codegen::DTYPE_SIZE_4BYTE;
using codegen::DTYPE_SIZE_8BYTE;
using codegen::GetDtypeSizeByName;
using codegen::GetUintDtypeNameBySize;

// 根据dtype转换比例确定LoadDist模式，并计算剩余需要的UnPack序列
// 支持的LoadDist UNPACK模式：DIST_UNPACK_B8(1→2), DIST_UNPACK_B16(2→4), DIST_UNPACK_B32(4→8), DIST_UNPACK4_B8(1→4)
// UnPack只支持三种路径：uint8→uint16、uint16→uint32、uint32→uint64
void DetermineLoadDistAndUnPackSequence(uint32_t input_dtype_size, uint32_t max_dtype_size, std::string &dist,
                                        int &remaining_unpack_times, std::vector<std::string> &remaining_dtype_sequence,
                                        std::string &load_result_dtype) {
  if (max_dtype_size <= input_dtype_size) {
    load_result_dtype = GetUintDtypeNameBySize(input_dtype_size);
    return;
  }

  // dist 已有值（如广播模式）时，不使用 LoadDist UNPACK，全部通过 UnPack 实现
  uint32_t load_result_size = input_dtype_size;  // LoadDist 处理后的 dtype 大小
  if (dist.empty()) {
    // 根据输入字节大小选择最优 LoadDist 模式
    if (input_dtype_size == DTYPE_SIZE_1BYTE) {
      // 1字节：优先用 DIST_UNPACK4_B8 (→4字节)，其次 DIST_UNPACK_B8 (→2字节)
      dist = (max_dtype_size >= DTYPE_SIZE_4BYTE) ? "DIST_UNPACK4_B8" : "DIST_UNPACK_B8";
      load_result_size = (max_dtype_size >= DTYPE_SIZE_4BYTE) ? DTYPE_SIZE_4BYTE : DTYPE_SIZE_2BYTE;
    } else if (input_dtype_size == DTYPE_SIZE_2BYTE) {
      dist = "DIST_UNPACK_B16";  // →4字节
      load_result_size = DTYPE_SIZE_4BYTE;
    } else if (input_dtype_size == DTYPE_SIZE_4BYTE) {
      dist = "DIST_UNPACK_B32";  // →8字节
      load_result_size = DTYPE_SIZE_8BYTE;
    }
  }

  // LoadDist 处理后的 dtype 名称
  load_result_dtype = GetUintDtypeNameBySize(load_result_size);

  // 计算从 load_result_size 到 max_dtype_size 需要的 UnPack 序列
  if (load_result_size < max_dtype_size) {
    remaining_unpack_times = 0;
    uint32_t current_size = load_result_size;
    while (current_size < max_dtype_size) {
      current_size *= 2;
      remaining_unpack_times++;
      if (current_size == DTYPE_SIZE_2BYTE) {
        remaining_dtype_sequence.push_back("uint16_t");
      } else if (current_size == DTYPE_SIZE_4BYTE) {
        remaining_dtype_sequence.push_back("uint32_t");
      } else if (current_size == DTYPE_SIZE_8BYTE) {
        remaining_dtype_sequence.push_back("uint64_t");
      }
    }
  }
}
}  // namespace

namespace codegen {
Status MicroLoadApiCall::Generate(const TensorManager &tensor_mng, const TPipe &tpipe, CallParam &param,
                                  string &result) {
  std::stringstream ss;
  // A CV broadcast view needs a scalar splat before any dtype unpacking.
  // Recompute the distribution for each generation, including reused calls.
  this->dist_.clear();
  if (tpipe.cv_fusion_type == ascir::CubeTemplateType::kUBFuse) {
    GE_ASSERT_SUCCESS(UpdateDistModeByStrideInfo(tpipe));
  }
  auto tensor_id = GetOutputTensorIdByIndex(0);
  GE_ASSERT_NOTNULL(tensor_mng.GetTensor(tensor_id));

  auto tensor_ptr = tensor_mng.GetTensor(tensor_id);
  auto input_dtype = tensor_ptr->dtype_;
  std::string input_dtype_name;
  Tensor::DtypeName(input_dtype, input_dtype_name);
  uint32_t input_dtype_size = ge::GetSizeByDataType(input_dtype);

  int remaining_unpack_times = 0;
  std::vector<std::string> remaining_dtype_sequence;
  std::string load_result_dtype;

  if (!param.max_dtype_size.empty()) {
    uint32_t max_dtype_size = GetDtypeSizeByName(param.max_dtype_size);
    DetermineLoadDistAndUnPackSequence(input_dtype_size, max_dtype_size, this->dist_, remaining_unpack_times,
                                       remaining_dtype_sequence, load_result_dtype);
  } else {
    load_result_dtype = GetUintDtypeNameBySize(input_dtype_size);
  }

  // Load
  std::string load_template_params = "";
  if (!this->dist_.empty()) {
    load_template_params = "<" + input_dtype_name + ", AscendC::MicroAPI::LoadDist::" + this->dist_ + ">";
  }
  ss << "AscendC::MicroAPI::LoadAlign" << load_template_params << "(" << tensor_ptr->name << ", "
     << *(tpipe.GetTensor(this->GetInputTensorIdByIndex(0))) << " + " << param.offset << ");" << std::endl;

  // UnPack
  for (int i = 0; i < remaining_unpack_times && i < static_cast<int>(remaining_dtype_sequence.size()); ++i) {
    if (remaining_dtype_sequence[i] == load_result_dtype) {
      continue;
    }
    ss << "AscendC::Reg::UnPack<" << remaining_dtype_sequence[i] << ", " << load_result_dtype
       << ">((AscendC::Reg::RegTensor<" << remaining_dtype_sequence[i] << ">&)" << tensor_ptr->name
       << ", (AscendC::Reg::RegTensor<" << load_result_dtype << ">&)" << tensor_ptr->name << ");" << std::endl;
    load_result_dtype = remaining_dtype_sequence[i];
  }

  result = ss.str();
  return af::SUCCESS;
}

Status MicroLoadApiCall::Init(const ascir::NodeView &node) {
  (void)node;
  return af::SUCCESS;
}

// 对于DIST_BRC_XXX属性，由于Init函数入参不包含TPipe字段，导致无法拿到输入tensor的stride信息，去判断是否需要使用DIST_BRC_XXX模式
// 因此需要在Generate函数中额外判断一次
// 典型场景: src0 (A0, B1) + src(A0, A1)
// src0在Load时，随路完成尾轴brc, 根据尾轴stride是否为0，判断是否采用尾轴brc
Status MicroLoadApiCall::UpdateDistModeByStrideInfo(const TPipe &tpipe) {
  auto tensor_id = GetInputTensorIdByIndex(0);
  const Tensor *tensor_ptr = tpipe.GetTensor(tensor_id);
  GE_ASSERT_NOTNULL(tensor_ptr);
  if (!tpipe.IsCvNarrowTensor(*tensor_ptr)) {
    if (tensor_ptr->vectorized_strides.empty()) {
      return af::SUCCESS;
    }
    const auto last_dim_stride = tensor_ptr->vectorized_strides.back();
    if (af::SymbolicUtils::StaticCheckEq(last_dim_stride.Simplify(), af::sym::kSymbolZero) != af::TriBool::kTrue) {
      return af::SUCCESS;
    }
  }

  const std::map<int, string> load_brc_dist_mode = {
      {DTYPE_SIZE_1BYTE, "DIST_BRC_B8"}, {DTYPE_SIZE_2BYTE, "DIST_BRC_B16"}, {DTYPE_SIZE_4BYTE, "DIST_BRC_B32"}};
  const auto dtype_size = ge::GetSizeByDataType(tensor_ptr->dtype);
  const auto mode = load_brc_dist_mode.find(dtype_size);
  GE_ASSERT_TRUE(mode != load_brc_dist_mode.end(), "Unsupported CV broadcast load dtype size: %d", dtype_size);
  this->dist_ = mode->second;
  return af::SUCCESS;
}

static MicroApiCallRegister<MicroLoadApiCall> register_micro_load_api_call("MicroLoadApiCall");
}  // namespace codegen
