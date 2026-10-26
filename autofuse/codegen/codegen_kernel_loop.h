/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __CODEGEN_KERNEL_LOOP_H__
#define __CODEGEN_KERNEL_LOOP_H__

#include <set>
#include <utility>
#include <sstream>
#include "ascir.h"
#include "ascgen_log.h"
#include "ascir_ops_utils.h"
#include "codegen_api_param/codegen_api_param.h"

namespace codegen {
// 前向声明，避免循环依赖
class Tiler;
class TPipe;
class Tensor;
class ApiCall;
struct Loop;
class Axis;

enum class LoopType : int8_t { CALL = 0, LOOP };
enum class BoolType : int8_t { FALSE = 0, TRUE = 1, FAILED = 2 };

struct LoopBody {
  LoopType type;
  union {
    ApiCall *call;
    Loop *loop;
  };
};

struct ApiTensor {
  ascir::TensorId id;
  ascir::ReuseId reuse_id;
  struct ApiTensor *reuse_from;
  struct ApiTensor *reuse_next;
  struct ApiTensor *share_prev;
  struct ApiTensor *share_next;
  mutable int32_t share_order;
  const ApiCall *write;
  std::vector<const ApiCall *> reads;

  ApiTensor();
};

enum class ApiScene : int8_t {
  kDefault = 0,   // 非CV融合场景
  kCVFuseUBLoad,  // CV融合场景, load节点的输入tensor在UB上(Cube的输出)
};

enum class ComputeStage : int8_t {
  kDefault = 0,   // 非CV融合场景
  kCVFuseStage1,  // CV融合场景阶段1, Cube输出Tensor的生命周期之内
  kCVFuseStage2,  // CV融合场景阶段2, Cube输出Tensor的生命周期之外
};

struct ApiCallContext {
  ApiScene scene = ApiScene::kDefault;
  ComputeStage stage = ComputeStage::kDefault;

  bool isCVFusion() const {
    return scene != ApiScene::kDefault;
  }
};

class ApiCall {
 public:
  // Constructor and Destructor
  virtual ~ApiCall() = default;
  explicit ApiCall(const std::string &api_name) noexcept : api_name_(api_name) {}

  // Public Member Function
  virtual Status Init(const ascir::NodeView &node);
  virtual Status ParseAttr(const ascir::NodeView &node) {
    (void)node;
    return af::SUCCESS;
  }
  virtual Status BuildApiParam(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                               const std::vector<std::reference_wrapper<const Tensor>> &input,
                               const std::vector<std::reference_wrapper<const Tensor>> &output) const;
  virtual Status GenerateApiCallString(const TPipe &tpipe, std::string &result) const;
  virtual Status GenDimensionParam(const CodegenApiParam &api_param, const Tiler &tiler, std::stringstream &ss) const;
  virtual Status PreProcess(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                            const std::vector<std::reference_wrapper<const Tensor>> &outputs,
                            std::string &result) const;
  virtual Status GenerateFuncDefinition(const TPipe &tpipe, const Tiler &tiler, std::stringstream &ss) const;
  virtual Status Generate(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                          const std::vector<std::reference_wrapper<const Tensor>> &input,
                          const std::vector<std::reference_wrapper<const Tensor>> &output, std::string &result) const;
  virtual Status PostProcess(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                             const std::vector<std::reference_wrapper<const Tensor>> &outputs,
                             std::string &result) const;
  virtual Status Generate(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                          std::string &result) const;
  virtual Status GenerateMacro(std::string &result) const {
    (void)result;
    return af::SUCCESS;
  }
  virtual bool AreContiguousBufsPreferred() const {
    return false;
  }
  bool FreeInputs(const TPipe &tpipe, std::stringstream &ss) const;
  bool FreeUnusedOutputs(const TPipe &tpipe, std::stringstream &ss) const;
  bool SyncOutputs(const TPipe &tpipe, std::stringstream &ss) const;
  bool WaitInputs(const TPipe &tpipe, std::stringstream &ss) const;
  bool IsReadOutersideWrite(ascir::AxisId &target_id) const;
  Status AllocOutputs(const TPipe &tpipe, std::stringstream &ss) const;
  bool IsUnitLastRead(const ApiTensor &tensor) const;

  // 快速注册基本 dump 信息（api_name, inputs, outputs, tmp_buf, cal_count）
  // 供旧式 Generate 调用，不需要迁移到 BuildApiParam 即可输出初版 dump
  af::Status RegisterBasicDumpParam(const std::string &api_name,
                                    const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                                    const std::vector<std::reference_wrapper<const Tensor>> &outputs,
                                    const CombinedExpression &cal_count = CombinedExpression{},
                                    const std::string &tmp_buf_name = "") const;
  af::Status SetSemanticFeature(const std::string &semantic_feature) const;

  // Public Member Variables
  std::string api_name_;
  ascir::AxisId axis;
  std::string type;  // ascir type
  int64_t depth;
  ascir::ComputeUnit unit;
  ascir::ComputeType compute_type;
  std::vector<ApiTensor> outputs;
  std::vector<const ApiTensor *> inputs;
  bool enable_cache{false};
  bool is_input_tbuf_contiguous = false;
  bool skip_api_emit{false};
  std::string enable_cache_with_condition;
  // 用于标记Call节点执行状态
  // broadcast cache场景：在Call节点外生成控制条件
  af::ExecuteCondition exec_condition;
  ApiCallContext api_call_context = {ApiScene::kDefault, ComputeStage::kDefault};
  std::unordered_map<int64_t, int64_t> tmp_buf_id;
  af::AscNodePtr node;
  std::string graph_name;
  std::string node_name;

 private:
  bool WaitInputVector(const TPipe &tpipe, const ApiTensor *in, const Tensor &t, std::stringstream &ss) const;
  bool WaitInputMte(const TPipe &tpipe, const ApiTensor *in, const Tensor &t, std::stringstream &ss) const;
  BoolType WaitShareInputs(const TPipe &tpipe, const ApiTensor *in, const Tensor t, std::stringstream &ss) const;
  BoolType AllocShareOutputs(const TPipe &tpipe, const ApiTensor &out, const Tensor t, std::stringstream &ss) const;
  Status HandleVecOutAlloc(const TPipe &tpipe, const ApiTensor &out, const Tensor &t, std::stringstream &ss,
                           bool with_define) const;
};

struct Loop {
  ascir::AxisId axis_id;
  struct Loop *parent;
  std::vector<LoopBody> bodys;
  std::set<const ApiCall *> used_calls = {};
  bool is_graph_has_reduce_node = false;  // 当前图上是否有reduce节点
  bool is_ar = false;                     // 如果图上有reduce节点  是否为AR
  explicit Loop(const ascir::AxisId axis);
  ComputeStage compute_stage = ComputeStage::kDefault;

  void AddLoop(Loop *loop);
  void AddCall(ApiCall *call);

  /* 将会通过new 申请内存，需要通过Destruct释放 */
  Status ConstructFromNodes(ascir::NodeViewVisitorConst nodes, const Tiler &tiler, TPipe &tpipe);
  void Destruct();

  Status Generate(const Tiler &tiler, const TPipe &tpipe, std::string &result,
                  ComputeStage stage = ComputeStage::kDefault);
  const Tensor *GetReduceOutputTensor(const TPipe &tpipe) const;
  const Tensor *GetReduceInputTensor(const TPipe &tpipe) const;
  void CollectTensorCrossLoop(std::map<ascir::AxisId, std::map<Loop *, std::vector<ApiCall *>>> &api_calls);
  Status ActualSizeDefine(const Tiler &tiler, const TPipe &tpipe, std::string dtype_name, std::string &result);

 private:
  Status GenerateLoop(const Tiler &tiler, const TPipe &tpipe, std::vector<ascir::AxisId> &current_axis,
                      std::stringstream &ss);
  Status GenerateBody(const Tiler &tiler, const TPipe &tpipe, std::vector<ascir::AxisId> &current_axis,
                      std::stringstream &ss);
  void GenerateEnCacheCondition(const Tiler &tiler, const TPipe &tpipe, const Axis &axis, std::stringstream &ss) const;
  bool IsFindInUsedCalls(const ApiCall *call) const;
  std::string GetReduceType() const;
  bool IsHaveReduceType(const std::string &type) const;
  bool IsBodyContainLoop() const;
};
}  // namespace codegen
#endif
