/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __AUTOFUSE_MICRO_API_CALL_H__
#define __AUTOFUSE_MICRO_API_CALL_H__

#include <string>
#include <utility>
#include "ascir.h"
#include "ascgen_log.h"
#include "codegen_kernel.h"
namespace codegen {

struct ArangeParams {
  bool valid = false;
  std::string base;
  std::string step;
};

struct CallParam {
  std::string p_reg;
  std::string offset;
  std::string max_dtype_size;
  ArangeParams arange;
};

enum class TensorType : int8_t {
  UB_TENSOR = 0,
  REG_TENSOR = 1,
};

class MicroApiTensor : public Variable {
 public:
  virtual ~MicroApiTensor() = default;
  explicit MicroApiTensor(const ascir::TensorAttr &tensor, std::string &dtype_name, bool init_as_mask_reg = false);

  static const Type UBTensorTypes(std::string &dtype_name);
  static const Type RegTensorTypes(std::string &dtype_name);
  static const Type MaskRegTypes();

 public:
  uint32_t id_;
  ge::DataType dtype_;
  ascir::Position position_;
  vector<ascir::AxisId> axis_;
  vector<ascir::SizeExpr> axis_size_;
  vector<ascir::SizeExpr> axis_strides_;
  vector<ascir::AxisId> vectorized_axis_;
  vector<ascir::SizeExpr> vectorized_strides_;
  Uint32 size_;
  Uint32 actual_size_;
  bool init_as_mask_reg_ = false;
};

class TensorManager {
 public:
  Status AddTensor(const MicroApiTensor &tensor);
  const MicroApiTensor *GetTensor(ascir::TensorId id) const;
  Status GenerateVreg(std::string &result) const;

 private:
  map<ascir::TensorId, MicroApiTensor> tensors_;
};

class MicroApiCall {
 public:
  explicit MicroApiCall(std::string api_name) : api_name_(std::move(api_name)) {}
  virtual ~MicroApiCall() = default;

  // 生成micro api的调用
  virtual Status Generate(const TensorManager &tensor_mng, const TPipe &tpipe, CallParam &param, std::string &result);
  virtual bool HasArangeParam() const {
    return false;
  }
  virtual void GetArangeParams(const TPipe &, std::string &, std::string &) const {}

  // 生成outputs;
  virtual Status Init([[maybe_unused]] const ascir::NodeView &node) {
    this->unit = node->attr.api.unit;
    return af::SUCCESS;
  }

  void AddInput(ascir::TensorId id, TensorType type = TensorType::REG_TENSOR) {
    inputs_.emplace_back(type, id);
  }

  void AddOutput(ascir::TensorId id, TensorType type = TensorType::REG_TENSOR) {
    outputs_.emplace_back(type, id);
  }

  size_t GetInputSize() const {
    return inputs_.size();
  }

  size_t GetOutputSize() const {
    return outputs_.size();
  }

  // 调用者保证index合法性
  ascir::TensorId GetInputTensorIdByIndex(uint32_t index) const {
    return inputs_[index].second;
  }

  // 校正 UB 输入的 tensor id：生产者输出 id 被重定向（如 IndirectLoad post-Reduce）时，
  // boundary Load 须改用根图侧解析到的最终 id。
  void RewriteUbInputTensorId(uint32_t index, ascir::TensorId id) {
    if (index < inputs_.size() && inputs_[index].first == TensorType::UB_TENSOR) {
      inputs_[index].second = id;
    }
  }

  // 调用者保证index合法性
  ascir::TensorId GetOutputTensorIdByIndex(uint32_t index) const {
    return outputs_[index].second;
  }

  std::string GetMicroApiName() const {
    return api_name_;
  }

 protected:
  std::string api_name_;
  std::vector<std::pair<TensorType, ascir::TensorId>> inputs_;
  std::vector<std::pair<TensorType, ascir::TensorId>> outputs_;

 public:
  ascir::ComputeUnit unit{ascir::ComputeUnit::kUnitInvalid};
};
}  // namespace codegen
#endif  // __AUTOFUSE_MICRO_API_CALL_H__
