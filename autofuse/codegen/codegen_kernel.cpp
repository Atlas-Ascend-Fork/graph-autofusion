/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "codegen_kernel.h"

#include <sstream>
#include <string>
#include <functional>
#include <stack>
#include "attr_utils.h"
#include "ascir_ops.h"
#include "common_utils.h"
#include "common/ge_common/debug/log.h"
#include "graph/ascendc_ir/utils/asc_tensor_utils.h"
#include "common/checker.h"
#include "api_call/utils/api_call_factory.h"
#include "ascir_utils.h"
#include "backend/backend_spec.h"
#include "graph/symbolizer/symbolic_utils.h"
#include "ascendc_api_registry.h"
#include "indirect_load_utils.h"
#include "optimize/platform/platform_factory.h"
#include "optimize/schedule_utils.h"
#include "common/platform_context.h"
#include "common/l2_cache_hint_manager.h"
#include "codegen_graph_check.h"

using namespace std;
using namespace af::ops;
using namespace codegen;
using namespace af::ascir_op;
using namespace ascgen_utils;

namespace {
constexpr uint32_t kFuncIdBegin = 20000000U;
constexpr const char kInputTensorDescName[] = "input_tensor_desc";
constexpr const char kOutputTensorDescName[] = "output_tensor_desc";

struct QueCollection {
  std::unordered_map<ascir::QueId, af::Position> que_id_to_src_position;
  std::set<ascir::QueId> need_bind_que_id;
  std::map<ascir::QueId, bool> direct_gm_ques;
};

Status AllocateQues(TPipe &tpipe, const QueCollection &collection) {
  for (const auto &iter : collection.que_id_to_src_position) {
    if (tpipe.ques.count(iter.first) > 0UL) {
      continue;
    }
    std::string position;
    GE_CHK_STATUS_RET(PositionValue(iter.second, position), "Codegen get position value failed");
    const bool need_que_bind = collection.need_bind_que_id.count(iter.first) > 0UL;
    if (need_que_bind) {
      std::string dst_position;
      GE_CHK_STATUS_RET(PositionValue(af::Position::kPositionVecOut, dst_position),
                        "Codegen get position value failed");
      const auto new_que = tpipe.ques.emplace(iter.first, TQue{iter.first, iter.second, position, dst_position});
      GE_CHK_BOOL_RET_STATUS(new_que.second, af::FAILED, "Codegen emplace que [%ld] failed", iter.first);
    } else {
      const auto new_que = tpipe.ques.emplace(iter.first, TQue{iter.first, iter.second, position});
      GE_CHK_BOOL_RET_STATUS(new_que.second, af::FAILED, "Codegen emplace que [%ld] failed", iter.first);
    }
    tpipe.ques.at(iter.first).skip_init_for_simt_direct_gm = collection.direct_gm_ques.at(iter.first);
  }
  for (auto &[id, que] : tpipe.ques) {
    if (id != tpipe.cube_output_que_id) {
      que.is_cv_ub_fusion = (tpipe.cv_fusion_type == ascir::CubeTemplateType::kUBFuse);
    }
  }
  return af::SUCCESS;
}

std::string GetTensorName(const ascir::TensorAttr &tensor) {
  const auto node = tensor.anchor.GetOwnerNodeBarePtr();
  if (node != nullptr && node->GetType() == ScalarData::Type) {
    return GenValidName(node->GetName());
  }
  if (af::ascir::AscTensorUtils::IsConstTensor(tensor)) {
    return "scalar_" + std::to_string(tensor.attr.mem.tensor_id);
  }
  if (tensor.attr.mem.alloc_type == af::AllocType::kAllocTypeGlobal) {
    return "global_" + std::to_string(tensor.attr.mem.tensor_id);
  }
  return "local_" + std::to_string(tensor.attr.mem.tensor_id);
}

Type GetTensorType(const ascir::TensorAttr &tensor, std::string &dtype_name) {
  if (af::ascir::AscTensorUtils::IsConstTensor(tensor)) {
    return Type(dtype_name);
  }
  if (tensor.attr.mem.alloc_type == af::AllocType::kAllocTypeGlobal) {
    return Tensor::GlobalTensorTypes(dtype_name);
  }
  return Tensor::LocalTensorTypes(dtype_name);
}

ascir::SizeExpr GetTensorElementCount(const ascir::TensorAttr &tensor) {
  ascir::SizeExpr size_expr = af::ops::One;
  for (const auto &axis_size : tensor.attr.repeats) {
    size_expr = af::sym::Mul(size_expr, axis_size).Simplify();
  }
  return size_expr;
}

bool IsSameTensorMemory(const ascir::TensorAttr &lhs, const ascir::TensorAttr &rhs) {
  if (lhs.attr.dtype != rhs.attr.dtype || lhs.attr.repeats.empty() || rhs.attr.repeats.empty()) {
    return false;
  }
  return GetTensorElementCount(lhs).Simplify() == GetTensorElementCount(rhs).Simplify();
}

std::string GetSingleWorkspaceName(const ascir::ImplGraph &graph) {
  std::string workspace_name;
  for (const auto &node : graph.GetAllNodes()) {
    if (!IsOps<Workspace>(node)) {
      continue;
    }
    if (!workspace_name.empty()) {
      return "";
    }
    workspace_name = node->GetName();
  }
  return workspace_name;
}

bool GetSingleConsumerByTensorId(const ascir::ImplGraph &graph, ascir::TensorId tensor_id, af::AscNodePtr &consumer) {
  size_t consumer_count = 0UL;
  for (const auto &node : graph.GetAllNodes()) {
    if (IsOps<Workspace>(node)) {
      continue;
    }
    for (const auto &input : node->inputs()) {
      if (input->attr.mem.tensor_id != tensor_id) {
        continue;
      }
      consumer = node;
      ++consumer_count;
    }
  }
  return consumer_count == 1UL && consumer != nullptr;
}

std::string GetWorkspacePathOutputName(const ascir::ImplGraph &graph, const std::string &workspace_name) {
  const auto workspace_node = graph.FindNode(workspace_name.c_str());
  if (workspace_node == nullptr || !IsOps<Workspace>(workspace_node) || workspace_node->outputs().size() != 1UL) {
    return "";
  }

  const ascir::TensorAttr *workspace_tensor = nullptr;
  auto tensor_id = workspace_node->outputs[0].attr.mem.tensor_id;
  size_t max_depth = 0UL;
  for (const auto &node : graph.GetAllNodes()) {
    (void)node;
    ++max_depth;
  }
  for (size_t depth = 0UL; depth < max_depth; ++depth) {
    af::AscNodePtr consumer;
    if (!GetSingleConsumerByTensorId(graph, tensor_id, consumer)) {
      return "";
    }
    if (IsOps<Output>(consumer)) {
      if (consumer->inputs().size() != 1UL) {
        return "";
      }
      if (workspace_tensor == nullptr || !IsSameTensorMemory(*workspace_tensor, *consumer->inputs()[0])) {
        return "";
      }
      return GenValidName(consumer->GetName());
    }
    if (consumer->outputs().size() != 1UL) {
      return "";
    }
    if (workspace_tensor == nullptr) {
      workspace_tensor = consumer->outputs()[0];
    }
    tensor_id = consumer->outputs()[0]->attr.mem.tensor_id;
  }
  return "";
}

std::string GetVectorGroupWorkspaceOutputName(const ascir::ScheduleGroup &schedule_group,
                                              const std::string &workspace_name) {
  std::string output_name;
  for (const auto &impl_graph : schedule_group.impl_graphs) {
    if (ascgen_utils::IsCubeType(impl_graph)) {
      return "";
    }
    if (GetSingleWorkspaceName(impl_graph) != workspace_name) {
      return "";
    }

    // vector读取外部输入时, 输出buffer可能与该输入复用, 复用会覆盖输入, 故禁用
    for (const auto &node : impl_graph.GetAllNodes()) {
      if (IsOps<Data>(node)) {
        return "";
      }
    }

    const auto current_output_name = GetWorkspacePathOutputName(impl_graph, workspace_name);
    if (current_output_name.empty() || (!output_name.empty() && output_name != current_output_name)) {
      return "";
    }
    output_name = current_output_name;
  }
  return output_name;
}

bool UpdateCommonCvReuseOutputName(const std::string &current_output_name, std::string &output_name) {
  if (current_output_name.empty()) {
    return true;
  }
  if (!output_name.empty() && output_name != current_output_name) {
    return false;
  }
  output_name = current_output_name;
  return true;
}

std::string GetCommonCvCubeReuseOutputName(const std::vector<ascir::ScheduleGroup> &schedule_groups,
                                           size_t group_index) {
  if (group_index >= schedule_groups.size() || schedule_groups[group_index].impl_graphs.size() != 1UL ||
      !ascgen_utils::IsCubeType(schedule_groups[group_index].impl_graphs[0])) {
    return "";
  }
  const auto workspace_name = GetSingleWorkspaceName(schedule_groups[group_index].impl_graphs[0]);
  if (workspace_name.empty()) {
    return "";
  }
  std::string output_name;
  for (size_t i = 0UL; i < schedule_groups.size(); ++i) {
    if (i == group_index) {
      continue;
    }
    const auto current_output_name = GetVectorGroupWorkspaceOutputName(schedule_groups[i], workspace_name);
    if (!UpdateCommonCvReuseOutputName(current_output_name, output_name)) {
      return "";
    }
  }
  return output_name;
}

std::string GetCommonCvVectorReuseOutputName(const std::vector<ascir::ScheduleGroup> &schedule_groups,
                                             size_t group_index) {
  if (group_index >= schedule_groups.size()) {
    return "";
  }
  std::string output_name;
  for (size_t i = 0UL; i < schedule_groups.size(); ++i) {
    if (i == group_index) {
      continue;
    }
    if (schedule_groups[i].impl_graphs.size() != 1UL || !ascgen_utils::IsCubeType(schedule_groups[i].impl_graphs[0])) {
      continue;
    }
    const auto workspace_name = GetSingleWorkspaceName(schedule_groups[i].impl_graphs[0]);
    if (workspace_name.empty()) {
      continue;
    }
    const auto current_output_name = GetVectorGroupWorkspaceOutputName(schedule_groups[group_index], workspace_name);
    if (!UpdateCommonCvReuseOutputName(current_output_name, output_name)) {
      return "";
    }
  }
  return output_name;
}

void AppendCommonCvVectorTilingFuncCall(const Kernel &kernel, const ascir::FusedScheduledResult &fused_schedule_result,
                                        const CodegenConfig &config, const std::string &graph_name, size_t index,
                                        bool enable_group_parallel, std::stringstream &res_ss) {
  if (config.is_inductor) {
    if (IsStaticSchedResult(fused_schedule_result)) {
      res_ss << "#ifdef INDUCTOR_CONST_TILING_DATA" << std::endl;
      res_ss << kernel.GenTilingFuncCall(graph_name, "kConstTilingData.tiling_data", index, enable_group_parallel,
                                         false, "kConstTilingData.tiling_data")
             << std::endl;
      res_ss << "#else" << std::endl;
      res_ss << kernel.GenTilingFuncCall(graph_name, "t.tiling_data", index, enable_group_parallel, false,
                                         "t.tiling_data")
             << std::endl;
      res_ss << "#endif" << std::endl;
    } else {
      res_ss << kernel.GenTilingFuncCall(graph_name, "t.tiling_data", index, enable_group_parallel, false,
                                         "t.tiling_data")
             << std::endl;
    }
  } else {
    res_ss << kernel.GenTilingFuncCall(graph_name, "t", index, enable_group_parallel, false) << std::endl;
  }
}

Status AppendKernelFuncPrologueByTilingKey(const ascir::FusedScheduledResult &fused_schedule_result,
                                           std::stringstream &ss, std::stringstream &ss1, bool use_list_tensor,
                                           const CodegenConfig &config, const std::string &kernel_task_type,
                                           bool &finished) {
  finished = false;
  std::string graph_name = GenValidName(fused_schedule_result.fused_graph_name.GetString());
  ss1 << Kernel::KernelFuncDeclare(graph_name, fused_schedule_result, use_list_tensor, config.is_inductor,
                                   IsConv2DFusedScheduled(fused_schedule_result))
      << " {" << std::endl;
  if (config.is_inductor) {
    if (ascgen_utils::IsCubeFusedScheduled(fused_schedule_result) && !IsConv2DFusedScheduled(fused_schedule_result)) {
      ss1 << "    if constexpr (MIX_MODE == 2) {" << std::endl;
      ss1 << "      KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);" << std::endl;
      ss1 << "    } else {" << std::endl;
      ss1 << "      KERNEL_TASK_TYPE_DEFAULT(" << kernel_task_type << ");" << std::endl;
      ss1 << "    }" << std::endl;
    } else {
      ss1 << "    KERNEL_TASK_TYPE_DEFAULT(" << kernel_task_type << ");" << std::endl;
    }
    return af::SUCCESS;
  }
  if (!ascgen_utils::IsCubeFusedScheduled(fused_schedule_result)) {
    ss1 << "  REGISTER_TILING_DEFAULT(" << "AutofuseTilingData);" << std::endl;
    if (IsEmptyTensorSence(fused_schedule_result)) {
      ss1 << std::endl << "}" << std::endl;
      ss << ss1.str();
      finished = true;
    } else {
      ss1 << "  GET_TILING_DATA(t, gm_tiling_data);" << std::endl;
    }
  } else if (ascgen_utils::IsCubeCommonFusedScheduled(fused_schedule_result)) {
    // cv 兜底模板需要在具体cube实现前先引入含有模板宏定义的头文件防止核类型启动错误
    ss << "#ifndef INDUCTOR_CV_FUSION" << std::endl;
    ss << "#include \"autofuse_cube_tiling_data.h\"" << std::endl;
    ss << "#endif" << std::endl;
  }
  return af::SUCCESS;
}
}  // namespace

std::ostream &operator<<(std::ostream &os, const Code &obj) {
  return os << obj.Str();
}

Type::Type(const string &type_name) : name(type_name) {}

std::string Type::Str() const {
  return name;
}

Variable::Variable(const Type &var_type, const string &var_name) : type(var_type), name(var_name) {}

std::string Variable::Str() const {
  return name;
}

std::string Variable::AsArg() const {
  stringstream ss;
  ss << this->type << " " << this->name;
  return ss.str();
}

std::string Variable::Define(std::string &&init, bool define_const) const {
  std::stringstream ss;
  if (define_const) {
    ss << "const ";
  }

  if (init.empty()) {
    ss << type << " " << name << ";";
  } else {
    ss << type << " " << name << " = " << std::move(init) << ";";
  }
  return ss.str();
}

std::string Variable::Assign(std::string &value) const {
  std::stringstream ss;
  ss << name << " = " << value << ";";
  return ss.str();
}

Axis::Axis(const ascir::Axis &axis)
    : ascir::Axis(axis),
      Variable(kIntT, axis.name),
      loop_size(Variable(kInt64T, axis.name + "_loop_size")),
      elem_size(axis.name + "_elem_size"),
      actual_size(axis.name + "_actual_size"),
      axis_size(axis.name + "_axis_size"),
      tail_size(axis.name + "_tail_size"),
      size_expr(af::Symbol(axis.size.Str().get())) {}

Status Tensor::DtypeName(ge::DataType dtype, std::string &dtype_name) {
  static const std::string kTypeNames[] = {
      [ge::DT_FLOAT] = "float",     [ge::DT_FLOAT16] = "half",    [ge::DT_INT8] = "int8_t",
      [ge::DT_INT32] = "int32_t",   [ge::DT_UINT8] = "uint8_t",   "",
      [ge::DT_INT16] = "int16_t",   [ge::DT_UINT16] = "uint16_t", [ge::DT_UINT32] = "uint32_t",
      [ge::DT_INT64] = "int64_t",   [ge::DT_UINT64] = "uint64_t", [ge::DT_DOUBLE] = "",
      [ge::DT_BOOL] = "bool",       [ge::DT_STRING] = "",         [ge::DT_DUAL_SUB_INT8] = "",
      [ge::DT_DUAL_SUB_UINT8] = "", [ge::DT_COMPLEX64] = "",      [ge::DT_COMPLEX128] = "",
      [ge::DT_QINT8] = "",          [ge::DT_QINT16] = "",         [ge::DT_QINT32] = "",
      [ge::DT_QUINT8] = "",         [ge::DT_QUINT16] = "",        [ge::DT_RESOURCE] = "",
      [ge::DT_STRING_REF] = "",     [ge::DT_DUAL] = "",           [ge::DT_VARIANT] = "",
      [ge::DT_BF16] = "bfloat16_t", [ge::DT_UNDEFINED] = "",      [ge::DT_INT4] = "int4x2_t",
      [ge::DT_UINT1] = "",          [ge::DT_INT2] = "",           [ge::DT_UINT2] = "",
      [ge::DT_COMPLEX32] = "",
  };
  GE_CHK_BOOL_RET_STATUS((dtype < (sizeof(kTypeNames) / sizeof(kTypeNames[0])) && kTypeNames[dtype] != ""), af::FAILED,
                         "Codegen unsupported data type:%d", static_cast<int32_t>(dtype));
  dtype_name = kTypeNames[dtype];
  return af::SUCCESS;
}

const Type Tensor::GlobalTensorTypes(std::string &dtype_name) {
  return Type("GlobalTensor<" + dtype_name + ">");
}

const Type Tensor::LocalTensorTypes(std::string &dtype_name) {
  return Type("LocalTensor<" + dtype_name + ">");
}

Tensor::Tensor(const ascir::TensorAttr &tensor, std::string &dtype_name, const std::string &tensor_name)
    : Variable(GetTensorType(tensor, dtype_name), GetTensorName(tensor)),
      id(tensor.attr.mem.tensor_id),
      reuse_id(tensor.attr.mem.reuse_id),
      dtype(tensor.attr.dtype),
      alloc_type(tensor.attr.mem.alloc_type),
      position(tensor.attr.mem.position),
      axis(tensor.attr.axis),
      axis_size(tensor.attr.repeats),
      axis_strides(tensor.attr.strides),
      vectorized_axis(tensor.attr.vectorized_axis),
      vectorized_strides(tensor.attr.vectorized_strides),
      que_id(tensor.attr.que.id),
      buf_id(tensor.attr.buf.id),
      size(this->name + "_size"),
      actual_size(this->name + "_actual_size"),
      que_depth(this->name + "_que_depth"),
      que_buf_num(this->name + "_que_buf_num"),
      que_share_offset("q" + std::to_string(tensor.attr.que.id) + "_reuse" + std::to_string(tensor.attr.mem.reuse_id) +
                       "_offset"),
      const_value(""),
      const_value_expr(af::Symbol(0)),
      que_depth_value(tensor.attr.que.depth),
      que_buf_num_value(tensor.attr.que.buf_num),
      merge_scope(tensor.attr.opt.merge_scope),
      is_constant(af::ascir::AscTensorUtils::IsConstTensor(tensor)),
      ub_scalar_name(this->name + "_ub_scalar") {
  (void)tensor_name;
}

Tensor::Tensor(const ascir::TensorAttr &tensor, std::string &dtype_name, const ascir::SizeExpr &value,
               const std::string &tensor_name)
    : Tensor(tensor, dtype_name, tensor_name) {
  this->const_value_expr = value;
  this->is_constant = true;
}

Tensor::Tensor(const std::string &value, const ascir::TensorAttr &tensor, std::string &dtype_name,
               const std::string &tensor_name)
    : Tensor(tensor, dtype_name, tensor_name) {
  this->const_value = value;
  this->is_constant = true;
}

Status Tensor::Init() {
  for (auto vec_axis : this->vectorized_axis) {
    auto pos = std::find(this->axis.begin(), this->axis.end(), vec_axis);
    GE_ASSERT_TRUE((pos != this->axis.end()), "Codegen vectorized axis[%ld] not found", vec_axis);
    this->vectorized_axis_pos.push_back(pos - this->axis.begin());
  }

  // repeats 大于0的时候, 先初始化is_ub_scalar为true, 如有有其一repeate不是One, 则赋值为false, 认为不是ub_scalar场景
  is_ub_scalar = (this->axis_size.size() > 0U);
  for (auto &repeate : this->axis_size) {
    if (repeate != One) {
      is_ub_scalar = false;
      break;
    }
  }

  GELOGD("t_name:%s, axis_id:%s, size:%s, strides:%s, v_axis_id:%s, v_axis_pos:%s, v_strides:%s, is_ub_scalar:%d",
         name.c_str(), VectorToStr(this->axis).c_str(), VectorToStr(this->axis_size).c_str(),
         VectorToStr(this->axis_strides).c_str(), VectorToStr(this->vectorized_axis).c_str(),
         VectorToStr(this->vectorized_axis_pos).c_str(), VectorToStr(this->vectorized_strides).c_str(),
         static_cast<int32_t>(is_ub_scalar));

  return af::SUCCESS;
}

Status Tensor::InitUbScalar(std::string &result) const {
  std::stringstream ss;
  std::string dtype_name;
  GE_CHK_STATUS_RET(Tensor::DtypeName(this->dtype, dtype_name), "data type:%d failed",
                    static_cast<int32_t>(this->dtype));
  ss << ub_scalar_name << " = ";
  ss << name << ".GetValue(0);" << std::endl;
  result = ss.str();
  return af::SUCCESS;
}

Status Tensor::GenDuplicateValueOfUbScalar(std::string &result) const {
  std::stringstream ss;
  std::string dtype_name;
  Tensor::DtypeName(this->dtype, dtype_name);

  std::string event_id = this->name + "_event_id";
  ss << "AscendC::PipeBarrier<PIPE_ALL>();" << std::endl;
  ss << "Duplicate(" << this->name << "[0], " << ub_scalar_name << ", " << "32/sizeof(" << dtype_name << "));"
     << std::endl;
  ss << "AscendC::PipeBarrier<PIPE_V>();" << std::endl;
  result = ss.str();
  return af::SUCCESS;
}

Status Tensor::DefineUbScalar(std::string &result) const {
  std::stringstream ss;
  std::string dtype_name;
  GE_CHK_STATUS_RET(Tensor::DtypeName(this->dtype, dtype_name), "data type:%d failed",
                    static_cast<int32_t>(this->dtype));
  ss << dtype_name << " " << ub_scalar_name << ";" << std::endl;
  result = ss.str();
  return af::SUCCESS;
}

Status Tensor::SetGlobalBuffer(Variable global, const std::string &offset, std::string &result) const {
  std::stringstream ss;
  std::string dtype_name;
  GE_CHK_STATUS_RET(Tensor::DtypeName(this->dtype, dtype_name), "Codegen get data type:%d failed",
                    static_cast<int32_t>(this->dtype));

  if (!offset.empty() && offset != "0") {
    ss << name << ".SetGlobalBuffer("
       << "(__gm__ " << dtype_name << "*)((__gm__ uint8_t*)(" << global << ") + (" << offset << ")));";
  } else {
    ss << name << ".SetGlobalBuffer("
       << "(__gm__ " << dtype_name << "*)" << global << ");";
  }
  result = ss.str();
  return af::SUCCESS;
}

Status codegen::PositionValue(ascir::Position position, std::string &result) {
  static std::unordered_map<size_t, std::string> position_values = {
      {static_cast<size_t>(af::Position::kPositionGM), "TPosition::GM"},
      {static_cast<size_t>(af::Position::kPositionVecIn), "TPosition::VECIN"},
      {static_cast<size_t>(af::Position::kPositionVecCalc), "TPosition::VECCALC"},
      {static_cast<size_t>(af::Position::kPositionVecOut), "TPosition::VECOUT"}};

  auto it = position_values.find(static_cast<size_t>(position));
  if (it == position_values.end()) {
    GELOGE(af::FAILED, "Codegen position value[%d] invalid", static_cast<int32_t>(position));
    return af::FAILED;
  }

  result = it->second;
  return af::SUCCESS;
}

MergeScope::MergeScope(ascir::MergeScopeId merge_scope_id, ascir::Position pos)
    : id(merge_scope_id),
      position(pos),
      size("m" + to_string(merge_scope_id) + "_size"),
      depth("m" + to_string(merge_scope_id) + "_que_depth"),
      buf_num("m" + to_string(merge_scope_id) + "_que_buf_num") {}

TQue::TQue(ascir::QueId que_id, ascir::Position pos, std::string &position_name)
    : Variable(Type("TQue<" + position_name + ", " + "1>"), "q" + to_string(que_id)),
      id(que_id),
      position(pos),
      size(this->name + "_size"),
      depth(this->name + "_depth"),
      buf_num(this->name + "_buf_num"),
      buf(Type("LocalTensor<uint8_t>"), name + "_buf") {}

TQue::TQue(ascir::QueId que_id, ascir::Position src_position, const std::string &src_position_name,
           const std::string &dst_position_name)
    : Variable(Type("TQueBind<" + src_position_name + ", " + dst_position_name + ", 1>"), "q" + to_string(que_id)),
      id(que_id),
      position(src_position),
      size(this->name + "_size"),
      depth(this->name + "_depth"),
      buf_num(this->name + "_buf_num"),
      buf(Type("LocalTensor<uint8_t>"), name + "_buf"),
      is_bind(true) {}

std::string TQue::AllocBuf(const bool with_define) const {
  stringstream ss;
  if (with_define && !is_cv_ub_fusion) {
    ss << this->buf.AsArg();
  } else {
    ss << this->buf.Str();
  }
  ss << " = " << this->name << ".AllocTensor<uint8_t>();" << std::endl;
  return ss.str();
}

std::string TQue::FreeBuf() const {
  stringstream ss;
  ss << this->name << ".FreeTensor(" << this->buf << ");" << std::endl;
  return ss.str();
}

std::string TQue::EnqueBuf() const {
  stringstream ss;
  ss << this->name << ".EnQue(" << this->buf << ");" << std::endl;
  return ss.str();
}

std::string TQue::DequeBuf(const bool is_unit_first) const {
  stringstream ss;
  if (is_unit_first) {
    ss << this->buf.AsArg() << " = " << this->name << ".DeQue<uint8_t>();" << std::endl;
  } else {
    ss << this->buf.name << " = " << this->name << ".DeQue<uint8_t>();" << std::endl;
  }
  return ss.str();
}

TBuf::TBuf(ascir::BufId buf_id, const ascir::Position pos, std::string &position_name)
    : Variable(Type("TBuf<" + position_name + ">"), "b" + to_string(buf_id)),
      id(buf_id),
      position(pos),
      size(this->name + "_size"),
      buf(Type("LocalTensor<uint8_t>"), name + "_buf") {}

std::string TBuf::AllocBuf(const bool with_define) const {
  stringstream ss;
  if (with_define) {
    ss << this->buf.AsArg();
  } else {
    ss << this->buf.Str();
  }
  ss << " = " << this->name << ".Get<uint8_t>();";
  return ss.str();
}

std::string TBuf::AllocBuf(std::string buf_name, std::string dtype_name, const bool with_define) const {
  stringstream ss;
  if (with_define) {
    ss << "LocalTensor<" << dtype_name << "> " << buf_name << " = " << this->name << ".Get<" << dtype_name << ">();";
  } else {
    ss << buf_name << " = " << this->name << ".Get<" << dtype_name << ">();";
  }
  return ss.str();
}

Tiler::Tiler(const std::string &tiling_data_type, const std::string &tiling_data_name)
    : tiling_data(Type{tiling_data_type}, tiling_data_name), gm_tiling(kGmAddrT, "gm_tiling"), block_dim("block_dim") {}

std::string Tiler::Offset(const std::vector<ascir::AxisId> &current_axis, const std::vector<ascir::AxisId> &axis,
                          const std::vector<ascir::SizeExpr> &strides) const {
  std::stringstream ss;
  bool is_first = true;

  for (auto iter = axis.begin(); iter != axis.end(); ++iter) {
    bool is_from = false;
    for (auto ca : current_axis) {
      if (this->IsFrom(ca, *iter)) {
        is_from = true;
        break;
      }
    }

    if (!is_from) {
      continue;
    }

    if (is_first) {
      is_first = false;
    } else {
      ss << " + ";
    }

    auto stride = strides[iter - axis.begin()];
    if (stride == 0) {
      ss << "0";
    } else if (stride == 1) {
      ss << "(int64_t)" << this->GetAxis(*iter);
    } else {
      ss << "(int64_t)" << this->GetAxis(*iter) << " * " << "(int64_t)" << this->Size(stride);
    }
  }

  if (is_first) {
    // Not axis in current_axis
    ss << "0";
  }
  return ss.str();
}

std::string Tiler::TensorVectorizedOffset(const std::vector<ascir::AxisId> &current_axis, const Tensor &tensor) const {
  std::vector<ascir::AxisId> current_vectorized_axis;
  for (auto a : current_axis) {
    if (find(tensor.vectorized_axis.begin(), tensor.vectorized_axis.end(), a) != tensor.vectorized_axis.end()) {
      current_vectorized_axis.emplace_back(a);
    }
  }
  return this->Offset(current_vectorized_axis, tensor.vectorized_axis, tensor.vectorized_strides);
}

std::string Tiler::Str() const {
  return tiling_data.Str();
}

void codegen::Tiler::AddSizeVar(ascir::SizeVar size) {
  std::string var_define;
  if (!(size.expr.IsConstExpr())) {
    var_define = std::string(size.expr.Str().get());
    std::string tiling_var = this->tiling_data.Str() + "->" + var_define;
    af::Expression tiling_sizevar = af::Symbol(tiling_var.c_str());
    this->sizes.emplace_back(std::make_pair(size.expr, tiling_sizevar));
  }
}

uint32_t codegen::Tiler::GetTilingCaseId() const {
  return this->tiling_case_id;
}

void codegen::Tiler::SetTilingCaseId(uint32_t tilingCaseId) {
  this->tiling_case_id = tilingCaseId;
}

Status codegen::Tiler::AddAxis(const ascir::Axis &axis) {
  auto [new_axis, insert_success] = this->axis_map.emplace(axis.id, codegen::Axis(axis));
  (void)new_axis;
  if (!insert_success) {
    GELOGE(af::FAILED, "Codegen insert axis[%ld] fail", axis.id);
    return af::FAILED;
  }
  return af::SUCCESS;
}

static bool GetSplitBAttr(const Tiler *tiler, const Axis &axis) {
  if (axis.type == ascir::Axis::Type::kAxisTypeTileInner || axis.type == ascir::Axis::Type::kAxisTypeTileOuter) {
    return true;
  }
  if (axis.type == ascir::Axis::Type::kAxisTypeInvalid) {
    return false;
  }
  for (const auto from : axis.from) {
    auto from_axis = tiler->GetAxis(from);
    if (GetSplitBAttr(tiler, from_axis)) {
      return true;
    }
  }
  return false;
}

void codegen::Tiler::AddAxisSplitBAttr() {
  for (auto &[id, cur_axis] : axis_map) {
    (void)id;
    cur_axis.is_split_b = GetSplitBAttr(this, cur_axis);
  }
}

bool Axis::IsOuter() const {
  return (type == ascir::Axis::Type::kAxisTypeBlockOuter || type == ascir::Axis::Type::kAxisTypeTileOuter);
}

bool Axis::IsInner() const {
  return (type == ascir::Axis::Type::kAxisTypeBlockInner || type == ascir::Axis::Type::kAxisTypeTileInner);
}

static bool IsMergeFromInner(const Tiler &tiler, const codegen::Axis &axis) {
  if (axis.from.size() == 0) {
    return axis.IsInner();
  }

  bool contain_inner = false;
  std::function<void(int32_t)> func = [&tiler, &contain_inner, &func](int32_t current_axis_id) {
    const auto &current_axis = tiler.GetAxis(current_axis_id);
    for (const auto &from : current_axis.from) {
      const auto &from_axis = tiler.GetAxis(from);
      if (from_axis.IsInner()) {
        contain_inner = true;
        break;
      } else if (from_axis.type == ascir::Axis::Type::kAxisTypeMerged) {
        func(from);
      }
    }
  };
  func(axis.id);
  return contain_inner;
}

bool Tiler::IsFrom(ascir::AxisId src, ascir::AxisId dst) const {
  if (src == dst) {
    return true;
  }

  const auto &axis = this->GetAxis(src);
  for (const auto &from : axis.from) {
    if (from == dst || IsFrom(from, dst)) {
      return true;
    }
  }

  return false;
}

bool Tiler::HasSameOriginAxis(ascir::AxisId src, ascir::AxisId dst) const {
  std::set<ascir::AxisId> src_origins;
  std::set<ascir::AxisId> dst_origins;
  std::function<void(int32_t, std::set<ascir::AxisId> &)> func = [this, &func](int32_t current_axis_id,
                                                                               std::set<ascir::AxisId> &origin_ids) {
    const auto &axis = this->GetAxis(current_axis_id);
    for (const auto &from : axis.from) {
      if (this->GetAxis(from).type == Axis::Type::kAxisTypeOriginal) {
        origin_ids.insert(from);
      } else {
        func(from, origin_ids);
      }
    }
  };

  func(src, src_origins);
  func(dst, dst_origins);

  for (auto id : src_origins) {
    if (dst_origins.count(id) != 0) {
      return true;
    }
  }
  return false;
}

std::string codegen::Tiler::Size(const ascir::SizeExpr &size, bool using_int_tiling_data) const {
  if (!size.IsValid()) {
    GELOGE(af::FAILED, "Tiler::Size input size is invalid");
    return "";
  }
  auto raw = size.Str();
  std::string const_expr_str = (raw != nullptr) ? std::string(raw.get()) : std::string();
  if (size.IsConstExpr()) {
    return (const_expr_str.find("Rational") != std::string::npos) ? af::SymbolicUtils::AsNumerDenomToString(size)
                                                                  : const_expr_str;
  }
  auto replaced = size.Replace(this->sizes);
  if (!replaced.IsValid()) {
    GELOGE(af::FAILED, "Tiler::Size Replace(sizes) failed, size=%s", const_expr_str.c_str());
    return "";
  }
  auto replaced_str = replaced.Str();
  if (replaced_str == nullptr) {
    GELOGE(af::FAILED, "Tiler::Size Str() returned null after replace, size=%s", const_expr_str.c_str());
    return "";
  }
  std::string str_ret(replaced_str.get());
  return (using_int_tiling_data || str_ret.find("Rational") != std::string::npos)
             ? af::SymbolicUtils::AsNumerDenomToString(replaced)
             : str_ret;
}

std::string codegen::Tiler::ActualSize(const ascir::SizeExpr &size, bool using_int_tiling_data) const {
  if (!size.IsValid()) {
    GELOGE(af::FAILED, "Tiler::ActualSize input size is invalid");
    return "";
  }
  auto replace_actual = size.Replace(this->actual_sizes);
  if (!replace_actual.IsValid()) {
    GELOGE(af::FAILED, "Tiler::ActualSize Replace(actual_sizes) failed");
    return "";
  }
  auto replaced = replace_actual.Replace(this->sizes);
  if (!replaced.IsValid()) {
    GELOGE(af::FAILED, "Tiler::ActualSize Replace(sizes) failed");
    return "";
  }
  auto cstr = replaced.Str();
  if (cstr == nullptr) {
    GELOGE(af::FAILED, "Tiler::ActualSize Str() returned null after replace");
    return "";
  }
  std::string str_ret(cstr.get());
  return (using_int_tiling_data || str_ret.find("Rational") != std::string::npos)
             ? af::SymbolicUtils::AsNumerDenomToString(replaced)
             : str_ret;
}

std::string Tiler::TensorActualSize(const Tensor &tensor) const {
  if (tensor.vectorized_axis.size() == 0) {
    return "1";
  }

  stringstream ss;
  int64_t count = 0;
  for (size_t i = 0; i < tensor.vectorized_axis.size(); i++) {
    auto &stride = tensor.vectorized_strides[i];
    if (stride == 0) {
      continue;
    }

    if (count >= 1) {
      ss << " + ";
    }

    auto axis = GetAxis(tensor.vectorized_axis[i]);
    auto axis_pos = tensor.vectorized_axis_pos[i];
    auto axis_size = tensor.axis_size[axis_pos];
    bool size_equal = af::SymbolicUtils::StaticCheckEq(axis_size, axis.size_expr) == af::TriBool::kTrue;
    if (axis.type == Axis::Type::kAxisTypeTileInner || size_equal) {
      ss << "(" + axis.actual_size.Str() + " - 1)";
    } else {
      ss << "(" + this->Size(axis_size) + " - 1)";
    }

    if (stride != 1) {
      ss << " * " << this->Size(stride);
    }
    count++;
  }

  ss << ((ss.str().size() == 0u) ? "1" : " + 1");
  return ss.str();
}

std::string Tiler::TensorVectorizedSize(const Tensor &tensor) const {
  if (tensor.vectorized_axis.size() == 0) {
    return "1";
  }
  stringstream ss;
  std::string blk_align;
  (void)KernelUtils::BlkAlign(tensor.dtype, blk_align);
  ss << blk_align << "(";
  int64_t count = 0;
  for (size_t i = 0; i < tensor.vectorized_axis.size(); i++) {
    auto axis_pos = tensor.vectorized_axis_pos[i];
    auto axis_size = tensor.axis_size[axis_pos];
    auto &stride = tensor.vectorized_strides[i];
    if (stride == 0) {
      continue;
    }
    if (count >= 1) {
      ss << " + ";
    }
    ss << "(" + this->Size(axis_size) + " - 1)";
    if (stride != 1) {
      ss << " * " << this->Size(stride);
    }
    count++;
  }

  ss << ((ss.str().size() == 0u) ? "1" : " + 1)");  // stride全为0的ub_scalar场景, size返回1
  return ss.str();
}

const Axis &Tiler::GetAxis(const ascir::AxisId id) const {
  auto iter = this->axis_map.find(id);
  if (iter == this->axis_map.end()) {
    GELOGE(af::FAILED, "Codegen axis[%ld] not found", id);
    throw std::runtime_error("Axis not found " + to_string(id));
  }

  return iter->second;
}

std::string codegen::Tiler::AxisSize(const ascir::AxisId id) const {
  return this->Size(this->GetAxis(id).size);
}

std::string codegen::Tiler::AxisSize(const Axis &axis) const {
  return this->Size(axis.size);
}

std::string codegen::Tiler::GenAxisSizeNew(const ascir::AxisId id) const {
  stringstream ss;

  const auto &axis = this->GetAxis(id);
  bool is_reduce_block = axis.type == ascir::Axis::Type::kAxisTypeBlockOuter && axis.from.size() > 1;
  if (axis.type == ascir::Axis::Type::kAxisTypeOriginal) {
    ss << "const " << axis.axis_size.AsArg() << " = " << this->AxisSize(axis) << ";" << endl;
    ss << "const " << axis.loop_size.AsArg() << " = " << axis.axis_size.Str() << ";" << endl;
    ss << "const " << axis.actual_size.AsArg() << " = " << axis.axis_size.Str() << ";" << endl;
  } else if (axis.IsOuter() && !is_reduce_block) {
    const auto &from = this->GetAxis(axis.from[0]);
    const auto &inner = this->GetAxis(axis.split_pair_other_id);
    ss << "const " << axis.axis_size.AsArg() << " = " << from.loop_size.Str() << " / " << inner.axis_size.Str() << ";"
       << endl;
    ss << "const " << axis.loop_size.AsArg() << " = " << axis.axis_size.Str() << " + (" << inner.tail_size << " > 0);"
       << endl;
  } else if (axis.IsInner()) {
    const auto &from = this->GetAxis(axis.from[0]);
    ss << "const " << axis.axis_size.AsArg() << " = " << this->AxisSize(axis) << ";" << endl;
    ss << "const " << axis.tail_size.AsArg() << " = " << from.loop_size << " % " << axis.axis_size << ";" << endl;
  } else if (axis.type == ascir::Axis::Type::kAxisTypeMerged || is_reduce_block) {
    ss << "const " << axis.axis_size.AsArg() << " = ";
    for (const auto &f : axis.from) {
      ss << this->GetAxis(f).loop_size.Str() << " * ";
    }
    ss << "1;" << endl;

    ss << "const " << axis.loop_size.AsArg() << " = " << axis.axis_size.Str() << ";" << endl;
    ss << "const " << axis.actual_size.AsArg() << " = " << axis.axis_size.Str() << ";" << endl;
  }

  return ss.str();
}

std::string codegen::Tiler::GenInnerLoopSizeAndActualSize(const ascir::AxisId id, const ascir::AxisId loop_axis) const {
  stringstream ss;

  auto axis = this->GetAxis(id);
  if (!axis.IsInner()) {
    return "";
  }

  if (IsFrom(loop_axis, axis.split_pair_other_id)) {
    auto outter = this->GetAxis(axis.split_pair_other_id);
    ss << axis.actual_size.AsArg() << " = " << outter.Str() << " < " << outter.axis_size << " ? "
       << axis.axis_size.Str() << " : " << axis.tail_size.Str() << ";" << endl;
    ss << axis.loop_size.AsArg() << " = " << axis.actual_size.Str() << ";" << endl;

    af::Expression actual_size = af::Symbol(axis.actual_size.name.c_str());
    this->actual_sizes.emplace_back(std::make_pair(axis.size_expr, actual_size));
  }

  return ss.str();
}

std::string codegen::Tiler::CalcFromAxis(const ascir::AxisId id, bool is_define) const {
  stringstream ss;
  auto axis = this->GetAxis(id);
  if (axis.IsInner()) {
    auto from = this->GetAxis(axis.from[0]);
    ss << (is_define ? from.AsArg() : from.Str()) << " = " << "block_dim_offset" << " + " << axis.Str() << ";"
       << std::endl;
    ss << this->CalcFromAxis(from.id, is_define);
  } else if (axis.type == Axis::Type::kAxisTypeMerged) {
    for (size_t i = 0; i < axis.from.size(); i++) {
      auto &from = this->GetAxis(axis.from[i]);
      ss << (is_define ? from.AsArg() : from.Str()) << " = " << axis.Str();
      ss << " / ";
      ss << "(";
      for (size_t j = i + 1; j < axis.from.size(); j++) {
        ss << this->GetAxis(axis.from[j]).loop_size << " * ";
      }
      ss << "1)";
      ss << " % " << this->GetAxis(axis.from[i]).loop_size << ";" << std::endl;
    }
    for (auto from : axis.from) {
      ss << this->CalcFromAxis(from, is_define);
    }
  }

  return ss.str();
}

void codegen::Tiler::BlockOutterAxisDefine(const ascir::AxisId id, std::stringstream &ss) {
  auto axis = this->GetAxis(id);
  if (axis.IsInner()) {
    return;
  }
  for (size_t i = 0; i < axis.from.size(); i++) {
    auto &from = this->GetAxis(axis.from[i]);
    ss << from.AsArg() << " = " << axis.Str();
    ss << " / (";
    for (size_t j = i + 1; j < axis.from.size(); j++) {
      ss << this->GetAxis(axis.from[j]).loop_size << " * ";
    }
    ss << "1)";
    ss << " % " << this->GetAxis(axis.from[i]).loop_size << ";" << std::endl;
    if (from.type == Axis::Type::kAxisTypeMerged) {
      BlockOutterAxisDefine(axis.from[i], ss);
    }
  }
}

std::string codegen::Tiler::BlockOutterAxisDefine() {
  stringstream code;
  code << this->block_dim.Define("GetBlockIdx()") << std::endl;
  if (enable_group_parallel_) {
    code << "const uint32_t block_offset = " << tiling_data.name << "->ub_size;  // reuse as block_offset" << std::endl;
    code << this->block_dim.name << " = " << this->block_dim.name << " >= block_offset ? " << this->block_dim.name
         << " - block_offset : " << this->block_dim.name << " + GetBlockNum() - block_offset;" << std::endl;
    // block_dim范围在调用前校验了，此处不需要重复校验
  } else {
    code << "if (" << this->block_dim.name << " >= " << tiling_data.name << "->block_dim) {" << std::endl
         << "  return;" << std::endl
         << "}" << std::endl;
  }
  for (auto &[id, axis] : this->axis_map) {
    (void)id;
    if (axis.type != ascir::Axis::Type::kAxisTypeBlockOuter) {
      continue;
    }

    stringstream axis_value;
    axis_value << this->block_dim.name << " % " << axis.loop_size;
    code << axis.Define(axis_value.str(), true) << std::endl;
    if (axis.from.size() > 1) {
      BlockOutterAxisDefine(id, code);
    }
  }

  return code.str();
}

void Tiler::EnableGroupParallel(bool enable_group_parallel) {
  enable_group_parallel_ = enable_group_parallel;
}

std::string KernelUtils::Max() {
  return "KernelUtils::Max";
}

std::string KernelUtils::Sum() {
  return "KernelUtils::Sum";
}

Status KernelUtils::BlkNum(ge::DataType dtype, std::string &result) {
  std::stringstream ss;
  std::string dtype_name;
  GE_CHK_STATUS_RET(Tensor::DtypeName(dtype, dtype_name), "Codegen get data type:%d failed",
                    static_cast<int32_t>(dtype));
  ss << "KernelUtils::BlkNum<" << dtype_name << ">";
  result = ss.str();
  return af::SUCCESS;
}

Status KernelUtils::BlkAlign(ge::DataType dtype, std::string &result) {
  std::string dtype_name;
  GE_CHK_STATUS_RET(Tensor::DtypeName(dtype, dtype_name), "Codegen get data type:%d failed",
                    static_cast<int32_t>(dtype));
  result = "KernelUtils::BlkAlign<" + dtype_name + ">";
  return af::SUCCESS;
}

std::string KernelUtils::SizeAlign() {
  return "KernelUtils::SizeAlign";
}

std::string KernelUtils::FindNearestPower2() {
  return "KernelUtils::FindNearestPower2";
}

TPipe::TPipe(const std::string &tpipe_name, const Tiler &tpipe_tiler)
    : Variable(Type{"TPipe"}, tpipe_name), tiler(tpipe_tiler), tmp_buf(Type{"LocalTensor<uint8_t>"}, "tmp_buf") {}

Status TPipe::AddTensor(const Tensor &tensor) {
  auto [ret, is_insert1] = this->tensors.emplace(tensor.id, tensor);
  GE_CHK_BOOL_RET_STATUS(is_insert1, af::FAILED, "Codegen tensor[%ld,%s] is already added", tensor.id,
                         tensor.name.c_str());

  auto &t = ret->second;
  if (t.merge_scope != af::kIdNone &&
      (t.alloc_type == af::AllocType::kAllocTypeQueue || t.alloc_type == af::AllocType::kAllocTypeBuffer)) {
    auto merge_scope = this->merge_scopes.find(t.merge_scope);
    if (merge_scope == this->merge_scopes.end()) {
      auto [new_scope, is_insert2] = this->merge_scopes.emplace(t.merge_scope, MergeScope{t.merge_scope, t.position});
      GE_CHK_BOOL_RET_STATUS(is_insert2, af::FAILED, "Codegen emplace merge_scope [%ld] failed", t.merge_scope);
      new_scope->second.tensors.push_back(t.id);
    } else {
      GE_CHK_BOOL_RET_STATUS(merge_scope->second.position == t.position, af::FAILED,
                             "Merge scope for tensor[%s] position mismatch between %d and %d", t.name.c_str(),
                             static_cast<int32_t>(t.position), static_cast<int32_t>(merge_scope->second.position));
      merge_scope->second.tensors.push_back(t.id);
    }
  }

  if (t.alloc_type == af::AllocType::kAllocTypeQueue) {
    GE_CHK_BOOL_RET_STATUS(t.que_id != af::kIdNone, af::FAILED, "Codegen tensor[%ld,%s] queue is none", t.id,
                           t.name.c_str());
    TQue *que = nullptr;
    auto iter = this->ques.find(t.que_id);
    GE_ASSERT_TRUE(iter != this->ques.end(), "Cannot find que with id [%ld], it may not be initialized correctly",
                   t.que_id);
    que = &iter->second;
    if (t.merge_scope != af::kIdNone) {
      que->merge_scopes.insert(t.merge_scope);
    } else {
      que->not_merge_tensors.insert(t.id);
    }
    que->share_group[t.reuse_id].push_back(t.id);
  } else if (t.alloc_type == af::AllocType::kAllocTypeBuffer) {
    GE_CHK_BOOL_RET_STATUS(t.buf_id != af::kIdNone, af::FAILED, "Codegen tensor[%ld,%s] buffer is none", t.id,
                           t.name.c_str());

    TBuf *buf = nullptr;
    auto iter = this->bufs.find(t.buf_id);
    if (iter == this->bufs.end()) {
      std::string position;
      GE_CHK_STATUS_RET(PositionValue(t.position, position), "Codegen get position value failed");
      auto [new_buf, is_insert5] = this->bufs.emplace(t.buf_id, TBuf{t.buf_id, t.position, position});
      GE_CHK_BOOL_RET_STATUS(is_insert5, af::FAILED, "Codegen emplace tbuf [%ld] failed", t.buf_id);
      buf = &new_buf->second;
    } else {
      buf = &iter->second;
    }

    GE_CHK_BOOL_RET_STATUS(buf->position == t.position, af::FAILED,
                           "Codegen buf position mismatch for tensor[%s] between %d and %d", t.name.c_str(),
                           static_cast<int32_t>(t.position), static_cast<int32_t>(buf->position));

    if (t.merge_scope != af::kIdNone) {
      buf->merge_scopes.insert(t.merge_scope);
    } else {
      buf->not_merge_tensors.insert(t.id);
    }
  }

  return af::SUCCESS;
}

Status TPipe::AddTensor(const ascir::TensorAttr &tensor_attr, const std::string &tensor_name) {
  auto tensor_val_name = GenValidName(tensor_name);
  std::string dtype_name;
  GE_CHK_STATUS_RET(Tensor::DtypeName(tensor_attr.attr.dtype, dtype_name), "Codegen get data type:%d failed",
                    static_cast<int32_t>(tensor_attr.attr.dtype));
  Tensor tensor(tensor_attr, dtype_name, tensor_val_name);
  GE_CHK_STATUS_RET(tensor.Init(), "Codegen tensor init failed");
  GE_CHK_STATUS_RET(this->AddTensor(tensor), "Codegen add tensor failed");
  return af::SUCCESS;
}

Status TPipe::AddTensor(const std::string &const_value, const ascir::TensorAttr &tensor_attr,
                        const std::string &tensor_name) {
  auto tensor_val_name = GenValidName(tensor_name);
  std::string dtype_name;
  GE_CHK_STATUS_RET(Tensor::DtypeName(tensor_attr.attr.dtype, dtype_name), "Codegen get data type:%d failed",
                    static_cast<int32_t>(tensor_attr.attr.dtype));

  // const_value预处理
  std::string pre_process_value;
  GE_CHK_STATUS_RET(ascgen_utils::ScalarValuePreProcess(const_value, dtype_name, pre_process_value),
                    "Scalar value pre process failed, ori_value:%s, dtype:%s", const_value.c_str(), dtype_name.c_str());
  GELOGD("ori_value:%s, dtype:%s, pre_process_value:%s", const_value.c_str(), dtype_name.c_str(),
         pre_process_value.c_str());
  Tensor tensor(pre_process_value, tensor_attr, dtype_name, tensor_val_name);
  GE_CHK_STATUS_RET(tensor.Init(), "Codegen tensor init failed");
  GE_CHK_STATUS_RET(this->AddTensor(tensor), "Codegen add tensor failed");

  return af::SUCCESS;
}

Status TPipe::AddTensor(const ascir::TensorAttr &tensor_attr, const ascir::SizeExpr &const_value,
                        const std::string &tensor_name) {
  auto tensor_val_name = GenValidName(tensor_name);
  std::string dtype_name;
  GE_CHK_STATUS_RET(Tensor::DtypeName(tensor_attr.attr.dtype, dtype_name), "Codegen get data type:%d failed",
                    static_cast<int32_t>(tensor_attr.attr.dtype));
  Tensor tensor(tensor_attr, dtype_name, const_value, tensor_val_name);
  GE_CHK_STATUS_RET(tensor.Init(), "Codegen tensor init failed");
  GE_CHK_STATUS_RET(this->AddTensor(tensor), "Codegen add tensor failed");

  return af::SUCCESS;
}

std::string TPipe::AllocTmpBuf(const TBuf &buf, const bool with_define) const {
  stringstream ss;
  if (with_define) {
    ss << this->tmp_buf.AsArg();
  } else {
    ss << this->tmp_buf.Str();
  }
  ss << "_" << to_string(buf.id) << " = " << buf.name << ".Get<uint8_t>();" << std::endl;
  return ss.str();
}

static bool IsNextNodeSupportScalar(const ascir::NodeView &node) {
  std::set<std::string> support_ub_scalar_nodes = {
      Load::Type,    Store::Type,   Div::Type,       TrueDiv::Type,    Mul::Type,       Add::Type,         Sub::Type,
      Minimum::Type, Maximum::Type, LogicalOr::Type, LogicalAnd::Type, Broadcast::Type, ClipByValue::Type, Eq::Type,
      Ne::Type,      Gt::Type,      Lt::Type,        Ge::Type,         Le::Type,        Pow::Type,         Where::Type};
  return support_ub_scalar_nodes.count(node->GetType()) > 0U;
}

Status Kernel::OutputTensorIsUbScalar(const ascir::NodeView &node, bool &is_ub_scalar) const {
  is_ub_scalar = true;
  auto desc = node->GetOpDesc();
  for (auto output : node->outputs()) {
    auto output_index = af::ascir::AscTensorUtils::Index(*output);
    auto tensor_name = node->GetName() + "_" + desc->GetOutputNameByIndex(output_index);

    auto tensor_val_name = GenValidName(tensor_name);
    std::string dtype_name;
    GE_CHK_STATUS_RET(Tensor::DtypeName(output->attr.dtype, dtype_name), "Codegen get data type:%d failed",
                      static_cast<int32_t>(output->attr.dtype));
    Tensor tensor(*output, dtype_name, tensor_val_name);
    GE_CHK_STATUS_RET(tensor.Init(), "Codegen tensor init failed");

    if (!tensor.is_ub_scalar) {
      is_ub_scalar = false;
      break;
    }
  }
  GELOGI("node:%s, output tensor is_ub_scalar:%d", node->GetNamePtr(), static_cast<int32_t>(is_ub_scalar));
  return af::SUCCESS;
}

static bool IsOutputOnlyLink2VFNode(const ascir::TensorView &tensor) {
  for (const auto &peer_input : tensor.anchor.GetPeerInDataAnchors()) {
    auto output_node = std::dynamic_pointer_cast<af::AscNode>(peer_input->GetOwnerNode());
    if (output_node == nullptr || output_node->GetType() != VectorFunc::Type) {
      return false;
    }
  }
  return true;
}

Status Kernel::ParseUbScalarOptimizationInfo(const ascir::NodeView &node, Tensor &t, ascir::TensorId id,
                                             bool is_all_link_vf) {
  if (t.is_ub_scalar && !af::ops::IsOps<af::ascir_op::Scalar>(node) && !is_all_link_vf) {
    // 下游节点有其中之一输出tensor不是ub_scalar, 且下游节点支持scalar输入, 则本tensor需要生成get value
    bool a_tenor_of_next_node_is_not_ub_scalar = false;
    bool is_next_node_support_ub_scalar = false;
    for (auto &out : node->outputs()) {
      if (out == nullptr) {
        continue;
      }
      for (auto &peer_input : out->anchor.GetPeerInDataAnchors()) {
        auto next_node = std::dynamic_pointer_cast<af::AscNode>(peer_input->GetOwnerNode());
        t.need_duplicate_value_of_ub_scalar =
            IsSupportBlkTensorInput(next_node) ? true : t.need_duplicate_value_of_ub_scalar;
        bool is_ub_scalar;
        GE_CHK_STATUS_RET(OutputTensorIsUbScalar(next_node, is_ub_scalar));
        if (!is_ub_scalar) {
          a_tenor_of_next_node_is_not_ub_scalar = true;
          is_next_node_support_ub_scalar = IsNextNodeSupportScalar(next_node);
          break;
        }
      }
      if (a_tenor_of_next_node_is_not_ub_scalar) {
        this->ub_scalar_tensors.emplace_back(id);
        break;
      }
    }
    t.need_gen_get_value_of_ub_scalar = a_tenor_of_next_node_is_not_ub_scalar && is_next_node_support_ub_scalar;
    GELOGD("node:%s, tensor_id:%d, is_ub_scalar:%d, need_gen_get_value_of_ub_scalar:%d", node->GetNamePtr(),
           static_cast<int32_t>(id), static_cast<int32_t>(t.is_ub_scalar),
           static_cast<int32_t>(t.need_gen_get_value_of_ub_scalar));
  }
  return af::SUCCESS;
}

Status Kernel::JudgeIsLoadLinkStoreAndVec(const ascir::NodeView &node, Tensor &t, ascir::TensorId id) const {
  // todo: 解决load多引用场景，被store, vec 同时引用的缺少mte3到mte2的同步的问题,
  // 临时方案，从这里解析下是否该场景
  if ((node->attr.api.compute_type == ascir::ComputeType::kComputeLoad) &&
      (!optimize::ScheduleUtils::IsGatherLikeLoad(std::dynamic_pointer_cast<af::AscNode>(node)))) {
    bool link_to_store = false;
    bool link_to_vec = false;
    for (auto &out : node->outputs()) {
      GE_CHK_BOOL_RET_STATUS_NOLOG(out != nullptr, af::SUCCESS);  // 节点悬空认为合法
      for (auto &peer_input : out->anchor.GetPeerInDataAnchors()) {
        auto next_node = std::dynamic_pointer_cast<af::AscNode>(peer_input->GetOwnerNode());
        link_to_store = IsOps<Store>(next_node) ? true : link_to_store;
        link_to_vec = IsOps<Store>(next_node) ? link_to_vec : true;
      }
    }
    t.is_load_link_store_and_vec = link_to_store && link_to_vec;
    GELOGD("node:%s, tensor_id:%d, is_load_link_store_and_vec:%d", node->GetNamePtr(), static_cast<int32_t>(id),
           static_cast<int32_t>(t.is_load_link_store_and_vec));
  }
  return af::SUCCESS;
}

Status Kernel::ParseOptimizeInfo(const ascir::NodeView &node, const ascir::TensorView &tensor) {
  // 如果是reduce节点，强制设置是非ub_scalar场景
  std::set<std::string> force_non_ub_scalar = {Max::Type,  Sum::Type, Min::Type, Mean::Type,
                                               Prod::Type, Any::Type, All::Type};
  ascir::TensorId id = tensor.attr.mem.tensor_id;
  auto tensor_ptr = this->tpipe.GetTensor(id);
  GE_CHK_BOOL_RET_STATUS(tensor_ptr != nullptr, af::FAILED, "Check[Param] tensor_ptr is nullptr");
  auto &t = *tensor_ptr;
  t.is_ub_scalar = (force_non_ub_scalar.count(node->GetType()) > 0U) ? false : t.is_ub_scalar;
  // 遍历下游节点不是白名单, 则返回, 下游节点是白名单
  GELOGD("node:%s, tensor_id:%d, is_ub_scalar:%d", node->GetNamePtr(), static_cast<int32_t>(id),
         static_cast<int32_t>(t.is_ub_scalar));
  bool is_all_link_vf = IsOutputOnlyLink2VFNode(tensor);
  GE_CHK_STATUS_RET(ParseUbScalarOptimizationInfo(node, t, id, is_all_link_vf));
  GE_CHK_STATUS_RET(JudgeIsLoadLinkStoreAndVec(node, t, id));
  ParseScalarNeedGenBlkTensors(node, id);

  return af::SUCCESS;
}

Status Kernel::ParseScalarNeedGenBlkTensors(const ascir::NodeView &node, ascir::TensorId id) {
  // 是scalar的节点，判断下是否支持 blk tensor 输入的 Ascir
  if (!IsOps<Scalar>(node) && !IsOps<ScalarData>(node) && !IsOps<IndexExpr>(node)) {
    return af::SUCCESS;
  }
  for (auto &out : node->outputs()) {
    GE_CHK_BOOL_EXEC(out != nullptr, continue);
    for (auto &peer_input : out->anchor.GetPeerInDataAnchors()) {
      auto next_node = std::dynamic_pointer_cast<af::AscNode>(peer_input->GetOwnerNode());
      if (IsSupportBlkTensorInput(next_node)) {
        this->tpipe.need_gen_blk_tensors.emplace_back(id);
        break;
      }
    }
  }
  return af::SUCCESS;
}

const TQue *TPipe::GetQue(const ascir::QueId id) const {
  auto iter = this->ques.find(id);
  GE_CHK_BOOL_EXEC(iter != this->ques.end(), return nullptr, "Codegen que[%d] not found", id);
  return &iter->second;
}

const TBuf &TPipe::GetBuf(const ascir::BufId id) const {
  auto iter = this->bufs.find(id);
  if (iter == this->bufs.end()) {
    GELOGE(af::FAILED, "Codegen buf[%d] not found", id);
    throw std::runtime_error("Buf not found " + to_string(id));
  }

  return iter->second;
}

const Tensor *TPipe::GetTensor(ascir::TensorId id) const {
  auto iter = tensors.find(id);
  GE_CHK_BOOL_EXEC(iter != tensors.end(), return nullptr, "Codegen tensor[%ld] not found", id);
  return &iter->second;
}

Tensor *TPipe::GetTensor(ascir::TensorId id) {
  auto iter = tensors.find(id);
  GE_CHK_BOOL_EXEC(iter != tensors.end(), return nullptr, "Codegen tensor[%ld] not found", id);
  return &iter->second;
}

Status TPipe::TensorAlloc(const Tensor &tensor, std::string &result) const {
  if (tensor.is_constant) {
    result = "";
    return af::SUCCESS;
  }

  std::stringstream ss;
  if (this->cv_fusion_type != ascir::CubeTemplateType::kUBFuse) {
    ss << tensor.Define() << std::endl;
  }

  const Variable *buf;
  if (tensor.alloc_type == af::AllocType::kAllocTypeBuffer) {
    buf = &GetBuf(tensor.buf_id).buf;
  } else if (tensor.alloc_type == af::AllocType::kAllocTypeQueue) {
    auto t_que = GetQue(tensor.que_id);
    GE_CHK_BOOL_RET_STATUS(t_que != nullptr, af::FAILED, "Codegen que[%ld] not found", tensor.que_id);
    buf = &t_que->buf;
  } else if (tensor.alloc_type == af::AllocType::kAllocTypeGlobal) {
    buf = &tensor;
  } else {
    GELOGE(af::FAILED, "Codegen tensor[%ld, %s] alloc type[%d] not supported", tensor.id, tensor.name.c_str(),
           static_cast<int32_t>(tensor.alloc_type));
    return af::FAILED;
  }

  GE_CHK_BOOL_RET_STATUS(tensor.merge_scope == af::kIdNone, af::FAILED,
                         "Codegen tensor[%ld, %s] merge scope[%ld] not supported", tensor.id, tensor.name.c_str(),
                         tensor.merge_scope);

  std::string dtype_name;
  GE_CHK_STATUS_RET(Tensor::DtypeName(tensor.dtype, dtype_name), "Codegen get data type:%d failed",
                    static_cast<int32_t>(tensor.dtype));
  if (tensor.alloc_type == af::AllocType::kAllocTypeQueue) {
    ss << tensor << " = " << *buf << "[" << tensor.que_share_offset << "]"
       << ".template ReinterpretCast<" << dtype_name << ">();" << std::endl;
  } else {
    ss << tensor << " = " << *buf << ".template ReinterpretCast<" << dtype_name << ">();" << std::endl;
  }
  result = ss.str();
  return af::SUCCESS;
}

Status TPipe::InitTQueBuffers(const TQue &que, std::string &result) const {
  if (que.skip_init_for_simt_direct_gm) {
    result.clear();
    return af::SUCCESS;
  }
  stringstream ss;
  std::string blk_align;
  GE_CHK_STATUS_RET(KernelUtils::BlkAlign(ge::DT_UINT8, blk_align), "Codegen blk align failed");
  if (this->cv_fusion_type == ascir::CubeTemplateType::kUBFuse || !using_att_calc_qbt_size_) {
    if (using_global_tpipe_) {
      ss << "GetTPipePtr()->InitBuffer(" << que << ", " << que.buf_num << ", " << blk_align << "(" << que.size << "));";
    } else {
      ss << this->name << "."
         << "InitBuffer(" << que << ", " << que.buf_num << ", " << blk_align << "(" << que.size << "));";
    }
  } else {
    ss << "// ";
    ss << this->name << "."
       << "InitBuffer(" << que << ", " << que.buf_num << ", " << blk_align << "(" << que.size << "));" << std::endl;
    ss << this->name << "."
       << "InitBuffer(" << que << ", " << que.buf_num << ", t->q" << std::to_string(que.id) << "_size);";
  }
  result = ss.str();
  return af::SUCCESS;
}

Status TPipe::InitTBufBuffer(const TBuf &buf, std::string &result) const {
  stringstream ss;
  std::string blk_align;
  GE_CHK_STATUS_RET(KernelUtils::BlkAlign(ge::DT_UINT8, blk_align), "Codegen blk align failed");
  if (this->cv_fusion_type == ascir::CubeTemplateType::kUBFuse || !using_att_calc_qbt_size_) {
    if (using_global_tpipe_) {
      ss << "GetTPipePtr()->InitBuffer(" << buf << ", " << blk_align << "(" << buf.size << "));";
    } else {
      ss << this->name << "." << "InitBuffer(" << buf << ", " << blk_align << "(" << buf.size << "));";
    }
  } else {
    ss << "// ";
    ss << this->name << "."
       << "InitBuffer(" << buf << ", " << blk_align << "(" << buf.size << "));" << std::endl;
    ss << this->name << "."
       << "InitBuffer(" << buf << ", t->b" << std::to_string(buf.id) << "_size);";
  }

  result = ss.str();
  return af::SUCCESS;
}

std::string TPipe::TensorSizeCalc() const {
  stringstream ss;

  for (const auto &pair : this->tensors) {
    const auto &t = pair.second;
    if (t.alloc_type == af::AllocType::kAllocTypeQueue) {
      const TQue *que = GetQue(t.que_id);
      if (que != nullptr && que->skip_init_for_simt_direct_gm) {
        continue;
      }
      ss << t.size.DefineConst(this->tiler.TensorVectorizedSize(t)) << std::endl;
      ss << t.que_buf_num.DefineConst(to_string(t.que_buf_num_value)) << std::endl;
    } else if (t.alloc_type == af::AllocType::kAllocTypeBuffer) {
      ss << t.size.DefineConst(this->tiler.TensorVectorizedSize(t)) << std::endl;
    }
  }
  return ss.str();
}

std::string TPipe::TensorActualSizeCalc(const ascir::TensorId id) const {
  stringstream ss;
  auto t_ptr = GetTensor(id);
  GE_CHK_BOOL_EXEC(t_ptr != nullptr, return "", "t_ptr nullptr");
  auto &t = *t_ptr;
  if (this->cv_fusion_type != ascir::CubeTemplateType::kUBFuse) {
    ss << t.actual_size.DefineConst(this->tiler.TensorActualSize(t));
  } else {
    ss << t.actual_size.Str() << " = " << this->tiler.TensorActualSize(t) << ";";
  }
  ss << std::endl;

  return ss.str();
}

Status TPipe::MergeScopeSizeCalc(std::string &result) const {
  stringstream ss;

  for (const auto &pair : this->merge_scopes) {
    const auto &merge_scope = pair.second;
    stringstream tensor_size_sum;
    stringstream tensor_bufnum_max;

    tensor_size_sum << KernelUtils::Sum() << "(";
    tensor_bufnum_max << KernelUtils::Max() << "(";

    bool first = true;
    for (auto tid : merge_scope.tensors) {
      auto tensor = this->tensors.find(tid);
      if (tensor == this->tensors.end()) {
        GELOGE(af::FAILED, "Codegen tensor[%ld] not found", tid);
        return af::FAILED;
      }

      if (tensor->second.alloc_type != af::AllocType::kAllocTypeQueue &&
          tensor->second.alloc_type != af::AllocType::kAllocTypeBuffer) {
        GELOGE(af::FAILED, "Codegen tensor[%ld] is not alloc from que/buf", tid);
        return af::FAILED;
      }

      if (first) {
        first = false;
      } else {
        tensor_size_sum << ", ";
        if (tensor->second.alloc_type == af::AllocType::kAllocTypeQueue) {
          tensor_bufnum_max << ", ";
        }
      }

      std::string dtype_name;
      GE_CHK_STATUS_RET(Tensor::DtypeName(tensor->second.dtype, dtype_name), "Codegen get data type:%d failed",
                        static_cast<int32_t>(tensor->second.dtype));
      tensor_size_sum << tensor->second.size << " * " << "sizeof(" << dtype_name << ")";
      if (tensor->second.alloc_type == af::AllocType::kAllocTypeQueue) {
        tensor_bufnum_max << tensor->second.que_buf_num;
      }
    }

    tensor_size_sum << ")";
    tensor_bufnum_max << ")";

    ss << merge_scope.size.DefineConst(tensor_size_sum.str()) << std::endl;
    ss << merge_scope.buf_num.DefineConst(tensor_bufnum_max.str()) << std::endl;
  }

  result = ss.str();
  return af::SUCCESS;
}

Status TPipe::LocalTQueAlloc(std::string &result) const {
  stringstream ss;

  for (auto &[id, que] : this->ques) {
    if (id == this->cube_output_que_id || que.skip_init_for_simt_direct_gm) {
      continue;
    }
    stringstream tensor_size_max;
    stringstream tensor_bufnum_max;

    tensor_size_max << KernelUtils::Max() << "(";
    tensor_bufnum_max << KernelUtils::Max() << "(";

    bool is_first = true;

    for (auto mid : que.merge_scopes) {
      auto merge_scope = this->merge_scopes.find(mid);
      GE_ASSERT_TRUE(merge_scope != this->merge_scopes.end(), "Codegen merge scope not found:%ld", mid);

      if (is_first) {
        is_first = false;
      } else {
        tensor_size_max << ", ";
        tensor_bufnum_max << ", ";
      }

      tensor_size_max << merge_scope->second.size;
      tensor_bufnum_max << merge_scope->second.buf_num;
    }

    uint32_t tensor_buf_num_max_val = 0;
    for (auto tid : que.not_merge_tensors) {
      auto tensor = this->tensors.find(tid);
      GE_ASSERT_TRUE(tensor != this->tensors.end(), "Codegen tensor not found:%ld", tid);

      if (is_first) {
        is_first = false;
      } else {
        tensor_size_max << ", ";
      }

      std::string dtype_name;
      GE_CHK_STATUS_RET(Tensor::DtypeName(tensor->second.dtype, dtype_name), "Codegen get data type:%d failed",
                        static_cast<int32_t>(tensor->second.dtype));
      tensor_size_max << tensor->second.size << " * sizeof(" << dtype_name << ")";
      tensor_buf_num_max_val = std::max(tensor_buf_num_max_val, tensor->second.que_buf_num_value);
    }
    tensor_bufnum_max << (que.merge_scopes.empty() ? "" : ", ") << tensor_buf_num_max_val;
    for (auto share_tensors : que.share_group) {
      if (share_tensors.second.size() <= 1) {  // 仅有1个tensor 不需要合并计算
        continue;
      }
      // 此处肯定不是第一个size
      tensor_size_max << ", ";
      // 多个tensor在一个que中且reuse id相同，需要计算reuse id相同这一组所有tensor size的综合
      bool is_first_share = true;
      for (auto tid : share_tensors.second) {
        auto tensor = this->tensors.find(tid);
        GE_ASSERT_TRUE(tensor != this->tensors.end(), "Codegen share tensor not found:%ld", tid);
        if (is_first_share) {
          is_first_share = false;
        } else {
          tensor_size_max << " + ";
        }
        std::string dtype_name;
        GE_CHK_STATUS_RET(Tensor::DtypeName(tensor->second.dtype, dtype_name), "Codegen get data type:%d failed",
                          static_cast<int32_t>(tensor->second.dtype));
        std::string tensor_byte_size = tensor->second.size.Str() + " * sizeof(" + dtype_name + ")";
        if (this->cv_fusion_type == ascir::CubeTemplateType::kUBFuse) {
          tensor_byte_size = "KernelUtils::BlkAlign<uint8_t>(" + tensor_byte_size + ")";
        }
        tensor_size_max << tensor_byte_size;
      }
    }

    tensor_size_max << ")";
    tensor_bufnum_max << ")";

    if (this->cv_fusion_type == ascir::CubeTemplateType::kUBFuse) {
      ss << que.size.DefineConst(tensor_size_max.str()) << std::endl;
      ss << que.buf_num.DefineConst(tensor_bufnum_max.str()) << std::endl;
    } else if (!using_att_calc_qbt_size_) {
      ss << que.size.DefineConst(tensor_size_max.str()) << std::endl;
      ss << que.buf_num.DefineConst(tensor_bufnum_max.str()) << std::endl;
      ss << que.Define() << std::endl;
    } else {
      ss << "// " << que.size.DefineConst(tensor_size_max.str()) << std::endl;
      ss << que.buf_num.DefineConst(tensor_bufnum_max.str()) << std::endl;
      ss << que.Define() << std::endl;
    }
    std::string init;
    GE_CHK_STATUS_RET(this->InitTQueBuffers(que, init), "Codegen init tque buffers failed");
    ss << init << std::endl;
  }

  result = ss.str();
  return af::SUCCESS;
}

Status TPipe::BlkTensorAllocAndInit(std::string &result) const {
  // 遍历所有需要生成blk tensor 的 tensor id
  stringstream ss;
  for (auto &id : this->need_gen_blk_tensors) {
    auto tensor_ptr = this->GetTensor(id);
    GE_CHK_BOOL_RET_STATUS(tensor_ptr != nullptr, af::FAILED, "BlkTensorAllocAndInit need_gen_blk_tensors failed");
    std::string scalar_t_buf_name = tensor_ptr->name + "_tbuf";
    std::string scalar_local_blk_tensor_name = "local_blk_tensor_of_" + tensor_ptr->name;
    ss << "TBuf<TPosition::VECCALC> " << scalar_t_buf_name << ";" << std::endl;
    if (using_global_tpipe_) {
      ss << "GetTPipePtr()->InitBuffer(" << scalar_t_buf_name << ", 32);" << std::endl;
    } else {
      ss << "tpipe.InitBuffer(" << scalar_t_buf_name << ", 32);" << std::endl;
    }
    ss << "LocalTensor<" << tensor_ptr->type << "> " << scalar_local_blk_tensor_name << " = " << scalar_t_buf_name
       << ".Get<" << tensor_ptr->type << ">();" << std::endl;

    const auto const_value =
        tensor_ptr->const_value.empty() ? tiler.Size(tensor_ptr->const_value_expr, true) : tensor_ptr->const_value;
    ss << "Duplicate(" << scalar_local_blk_tensor_name << "[0], static_cast<" << tensor_ptr->type << ">(" << const_value
       << "), static_cast<uint64_t>(32/"
       << "sizeof(" << tensor_ptr->type << ")));" << std::endl;
    ss << "AscendC::PipeBarrier<PIPE_V>();" << std::endl;
  }
  result = ss.str();
  return af::SUCCESS;
}

std::string TPipe::GenDuplicateBufAlloc(
    const std::set<std::pair<std::string, std::string>> &pre_api_extract_dup) const {
  std::stringstream ss;
  int32_t i = 1;
  for (auto [const_val, const_dtype] : pre_api_extract_dup) {
    const std::string index_str = std::to_string(i);
    ss << "TBuf<TPosition::VECCALC> builtin_tmp_buffer_" << index_str << ";" << std::endl;
    if (using_global_tpipe_) {
      ss << "GetTPipePtr()->InitBuffer(builtin_tmp_buffer_" << index_str << ", ONE_BLK_SIZE);" << std::endl;
    } else {
      ss << "tpipe.InitBuffer(builtin_tmp_buffer_" << index_str << ", ONE_BLK_SIZE);" << std::endl;
    }
    ss << "LocalTensor<uint8_t> builtin_tmp_buf_" << index_str << " = builtin_tmp_buffer_" << index_str
       << ".Get<uint8_t>();" << std::endl;
    std::string local_tensor_name = "local_blk_tensor_of_" + const_dtype + "_" + const_val;
    ss << "LocalTensor<" << const_dtype << "> " << local_tensor_name << " = builtin_tmp_buf_" << index_str
       << ".template ReinterpretCast<" << const_dtype << ">();" << std::endl;
    if (const_dtype == "half" || const_dtype == "float" || const_dtype == "double") {
      const_val += ".0";
    }
    ss << "Duplicate(" << local_tensor_name << "[0], (" << const_dtype << ")" << const_val << ", ONE_BLK_SIZE / sizeof("
       << const_dtype << "));" << std::endl;
    i++;
  }
  return ss.str();
}

Status TPipe::LocalTBufAlloc(const TBuf &buf, std::string &result, const bool with_define) const {
  stringstream ss;
  std::string reuse_dtype_name = "";
  std::vector<const Tensor *> reuse_buf_tensors;
  bool is_buf_reuse = true;
  stringstream tensor_size_max;

  GE_CHK_STATUS_RET(ParseTBufReuse(buf, reuse_dtype_name, is_buf_reuse, reuse_buf_tensors, tensor_size_max),
                    "Codegen parse tbuf reuse failed");

  if (this->cv_fusion_type == ascir::CubeTemplateType::kUBFuse) {
    ss << buf.size.DefineConst(tensor_size_max.str()) << std::endl;
  } else if (!using_att_calc_qbt_size_) {
    ss << buf.size.DefineConst(tensor_size_max.str()) << std::endl;
    ss << buf.Define() << std::endl;
  } else {
    ss << "// " << buf.size.DefineConst(tensor_size_max.str()) << std::endl;
    ss << buf.Define() << std::endl;
  }
  std::string init;
  GE_CHK_STATUS_RET(this->InitTBufBuffer(buf, init), "Codegen init tbuf buffer failed");
  ss << init << std::endl;

  if (!is_buf_reuse) {
    ss << buf.AllocBuf(with_define) << std::endl;
  } else {
    ss << buf.AllocBuf(reuse_buf_tensors[0]->name, reuse_dtype_name, with_define) << std::endl;
    reuse_buf_tensors[0]->no_need_realloc = true;
    for (size_t i = 1UL; i < reuse_buf_tensors.size(); i++) {
      reuse_buf_tensors[i]->no_need_realloc = true;
      if (with_define) {
        ss << "LocalTensor<" << reuse_dtype_name << "> ";
      }
      ss << reuse_buf_tensors[i]->name << " = " << reuse_buf_tensors[0]->name << ";" << std::endl;
    }
  }
  result = ss.str();
  return af::SUCCESS;
}

Status TPipe::LocalTBufAllocLoopTwice(std::string &result, const bool with_define) const {
  stringstream ss;
  std::string tmp;
  // 优先分配需要连续的TBuf, 不会与tmp buf进行复用
  for (const auto buf_id : this->contiguous_buf_ids) {
    const auto it = this->bufs.find(buf_id);
    GE_ASSERT_TRUE(it != this->bufs.cend(), "buf not found, buf_id = %ld", buf_id);
    const auto &buf = it->second;
    GE_CHK_STATUS_RET(this->LocalTBufAlloc(buf, tmp, with_define), "Codegen TBuf alloc failed(no tmp buf).");
    ss << tmp;
  }
  std::set<ascir::BufId> allocated{this->contiguous_buf_ids.cbegin(), this->contiguous_buf_ids.cend()};
  for (auto &pair : this->bufs) {
    if (allocated.find(pair.first) != allocated.end()) {
      continue;
    }
    auto &buf = pair.second;
    if (!buf.tmp_buf_reuse) {
      GE_CHK_STATUS_RET(this->LocalTBufAlloc(buf, tmp, with_define), "Codegen TBuf alloc failed(no tmp buf).");
      ss << tmp;
    }
  }
  for (auto &pair : this->bufs) {
    auto &buf = pair.second;
    if (buf.tmp_buf_reuse) {
      GE_CHK_STATUS_RET(this->LocalTBufAlloc(buf, tmp, with_define), "Codegen TBuf alloc failed(tmp buf).");
      ss << tmp;
      ss << this->AllocTmpBuf(buf, with_define);
    }
  }
  ss << std::endl;
  result = ss.str();
  return af::SUCCESS;
}

Status TPipe::LocalTensorQueBufAlloc(std::string &result) const {
  stringstream ss;
  std::string tmp;
  ss << this->TensorSizeCalc();
  GE_CHK_STATUS_RET(this->MergeScopeSizeCalc(tmp), "Codegen merge scope size failed");
  ss << tmp;
  ss << std::endl;
  GE_CHK_STATUS_RET(this->LocalTQueAlloc(tmp), "Codegen alloc local tque failed");
  ss << tmp;
  GE_CHK_STATUS_RET(this->LocalTBufAllocLoopTwice(tmp), "Codegen alloc local tbuf failed");
  ss << tmp << std::endl;

  result = ss.str();
  return af::SUCCESS;
}

std::string TPipe::SyncMte3ToMte2(const Tensor in_tensor) const {
  stringstream ss;
  std::string event_name = in_tensor.Str() + "_e";
  std::string sync_name = in_tensor.Str() + "_s";

  ss << "auto " << event_name << " = tpipe.AllocEventID<HardEvent::MTE3_MTE2>();" << std::endl
     << "TQueSync<PIPE_MTE3, PIPE_MTE2> " << sync_name << ";" << std::endl
     << sync_name << ".SetFlag(" << event_name << ");" << std::endl
     << sync_name << ".WaitFlag(" << event_name << ");" << std::endl
     << "tpipe.ReleaseEventID<HardEvent::MTE3_MTE2>(" << event_name << ");" << std::endl;
  return ss.str();
}

std::string TPipe::SyncMte2ToMte3(const Tensor in_tensor) const {
  stringstream ss;
  std::string event_name = in_tensor.Str() + "_e";
  std::string sync_name = in_tensor.Str() + "_s";

  ss << "auto " << event_name << " = tpipe.AllocEventID<HardEvent::MTE2_MTE3>();" << std::endl
     << "TQueSync<PIPE_MTE2, PIPE_MTE3> " << sync_name << ";" << std::endl
     << sync_name << ".SetFlag(" << event_name << ");" << std::endl
     << sync_name << ".WaitFlag(" << event_name << ");" << std::endl
     << "tpipe.ReleaseEventID<HardEvent::MTE2_MTE3>(" << event_name << ");" << std::endl;
  return ss.str();
}

Status TPipe::CollectQues(const ascir::ImplGraph &graph) {
  QueCollection collection;
  for (auto node : graph.GetAllNodes()) {
    if (node->attr.api.type == ge::ApiType::kAPITypeBuffer) {
      continue;
    }
    for (const auto &out_tensor : node->outputs()) {
      if (out_tensor->attr.mem.alloc_type != af::AllocType::kAllocTypeQueue) {
        continue;
      }
      const int64_t tensor_que_id = out_tensor->attr.que.id;
      const bool uses_direct_gm = ascgen_utils::indirect_load::GetTemplateBehavior(node).uses_direct_gm_pipeline;
      auto [direct_gm_iter, inserted] = collection.direct_gm_ques.emplace(tensor_que_id, uses_direct_gm);
      if (!inserted) {
        direct_gm_iter->second = direct_gm_iter->second && uses_direct_gm;
      }
      if (out_tensor->attr.mem.position == af::Position::kPositionVecIn) {
        collection.que_id_to_src_position.emplace(tensor_que_id, out_tensor->attr.mem.position);

        std::set<std::string> peer_node_types;
        for (const auto &peer_in_anchor : out_tensor->anchor.GetPeerInDataAnchorsPtr()) {
          if (peer_in_anchor != nullptr && peer_in_anchor->GetOwnerNodeBarePtr() != nullptr) {
            peer_node_types.emplace(peer_in_anchor->GetOwnerNodeBarePtr()->GetType());
          }
        }
        if ((peer_node_types.size() == 1U) && *peer_node_types.begin() == Store::Type) {
          collection.need_bind_que_id.emplace(tensor_que_id);
        }
      } else if (out_tensor->attr.mem.position == af::Position::kPositionVecOut) {
        collection.que_id_to_src_position.emplace(tensor_que_id, out_tensor->attr.mem.position);
      }
    }
  }
  return AllocateQues(*this, collection);
}

void TPipe::SetUsingAttCalcQBTSizeConfig(bool using_att_calc_qbt_size) {
  using_att_calc_qbt_size_ = using_att_calc_qbt_size;
}

void TPipe::SetUsingGlobalTpipe(bool using_global_tpipe) {
  using_global_tpipe_ = using_global_tpipe;
}

Kernel::Kernel(const std::string &kernel_name)
    : workspace_arg("workspace"),
      name(kernel_name),
      tiler(kernel_name + "TilingData", "t"),
      tpipe("tpipe", this->tiler),
      root_loop(af::kIdNone) {}

Kernel::~Kernel() {
  root_loop.Destruct();
}

std::string Kernel::TilingKeyFuncDeclare(const std::string &impl_graph_name, const std::string &tiling_data) const {
  const char *flags[] = {"inline", "__aicore__"};
  const char *return_type = "void";

  std::stringstream ss;
  for (auto flag : flags) {
    ss << flag << " ";
  }
  ss << return_type << " ";
  ss << CamelToLowerSneak(impl_graph_name) << "(";
  if (use_list_tensor_) {
    ss << "ListTensorDesc &input_tensor_desc, ListTensorDesc &output_tensor_desc, ";
  } else {
    for (auto &input : this->inputs) {
      ss << input.AsArg() << ", ";
    }
    for (auto &output : this->outputs) {
      ss << output.AsArg() << ", ";
    }
  }
  ss << this->workspace_arg.AsArg() << ", ";
  for (auto &workspace : this->workspaces) {
    ss << workspace.AsArg() << ", ";
  }
  ss << "const " << tiling_data << " *t";
  ss << ")";
  return ss.str();
}

Status Kernel::GlobalTensorInit(std::string &result, const std::string &workspace_buffer_arg_override) const {
  std::stringstream ss;
  GE_CHK_STATUS_RET(AppendInputGlobalTensorInit(ss));
  GE_CHK_STATUS_RET(AppendOutputGlobalTensorInit(ss));
  GE_CHK_STATUS_RET(AppendConstTensorInit(ss));
  GE_CHK_STATUS_RET(AppendUbScalarTensorInit(ss));
  GE_CHK_STATUS_RET(AppendWorkspaceTensorInit(ss, workspace_buffer_arg_override));
  GE_CHK_STATUS_RET(GenL2CacheHintCode(ss));
  result = ss.str();
  return af::SUCCESS;
}

Status Kernel::AppendInputGlobalTensorInit(std::stringstream &ss) const {
  for (std::size_t i = 0; i < this->inputs.size(); i++) {
    const auto &tensor = this->tpipe.tensors.find(this->input_tensors[i]);
    if (tensor == this->tpipe.tensors.end()) {
      GELOGE(af::FAILED, "Codegen input tensor id[%ld] not found", this->input_tensors[i]);
      return af::FAILED;
    }
    if (tensor->second.is_constant) {
      continue;
    }

    ss << tensor->second.Define() << std::endl;
    std::string local_result;
    if (use_list_tensor_) {
      auto input_index = input_name_to_index_.at(this->inputs[i].Str());
      auto input = GM_ADDR("input_tensor_desc.GetDataPtr<__gm__ uint8_t>(" + std::to_string(input_index) + ")");
      GE_CHK_STATUS_RET(tensor->second.SetGlobalBuffer(input, "", local_result), "Codegen set global buffer failed");
    } else {
      GE_CHK_STATUS_RET(tensor->second.SetGlobalBuffer(this->inputs[i], "", local_result),
                        "Codegen set global buffer failed");
    }
    ss << local_result << std::endl;
  }
  return af::SUCCESS;
}

Status Kernel::AppendOutputGlobalTensorInit(std::stringstream &ss) const {
  for (std::size_t i = 0; i < this->outputs.size(); i++) {
    const auto &tensor = this->tpipe.tensors.find(this->output_tensors[i]);
    if (tensor == this->tpipe.tensors.end()) {
      GELOGE(af::FAILED, "Codegen output tensor id[%ld] not found", this->output_tensors[i]);
      return af::FAILED;
    }

    ss << tensor->second.Define() << std::endl;
    std::string local_result;
    if (use_list_tensor_) {
      auto output_index = output_name_to_index_.at(this->outputs[i].Str());
      auto output = GM_ADDR("output_tensor_desc.GetDataPtr<__gm__ uint8_t>(" + std::to_string(output_index) + ")");
      GE_CHK_STATUS_RET(tensor->second.SetGlobalBuffer(output, "", local_result), "Codegen set global buffer failed");
    } else {
      GE_CHK_STATUS_RET(tensor->second.SetGlobalBuffer(this->outputs[i], "", local_result),
                        "Codegen set global buffer failed");
    }
    ss << local_result << std::endl;
  }
  return af::SUCCESS;
}

Status Kernel::GenL2CacheHintCode(std::stringstream &ss) const {
  GE_CHK_BOOL_RET_SPECIAL_STATUS(gm_tensor_sizes_ == nullptr, af::SUCCESS, "no need to  gen code for L2 cache hint");
  ss << "const int64_t kL2CacheSize = " << l2_size_ << ";" << std::endl;
  if (gm_tensor_sizes_->min_total_size > l2_size_) {
    ss << "constexpr bool enable_l2_cache_hint = true;" << std::endl;
  } else {
    ss << "const int64_t total_gm_size = " << this->tiler.Size(gm_tensor_sizes_->total_size) << ";" << std::endl;
    ss << "const bool enable_l2_cache_hint = total_gm_size >= kL2CacheSize;" << std::endl;
  }
  std::vector<std::string> input_names;
  for (std::size_t i = 0; i < this->inputs.size(); i++) {
    const auto &tensor = this->tpipe.tensors.at(this->input_tensors[i]);
    if (tensor.is_constant) {
      continue;
    }
    const size_t input_index = input_name_to_index_.at(this->inputs[i].Str());
    if (skip_l2_cache_hint_input_indices_.find(input_index) == skip_l2_cache_hint_input_indices_.end()) {
      input_names.push_back(tensor.name);
    }
  }
  if (!input_names.empty()) {
    ss << "if (enable_l2_cache_hint) {" << std::endl;
    for (const auto &input_name : input_names) {
      ss << "  " << input_name << ".SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);" << std::endl;
    }
    ss << "}" << std::endl;
  }
  for (size_t i = 0; i < this->outputs.size(); i++) {
    const auto &tensor = this->tpipe.tensors.at(this->output_tensors[i]);
    size_t output_index = output_name_to_index_.at(this->outputs[i].Str());
    GE_ASSERT_TRUE(output_index < gm_tensor_sizes_->output_sizes.size(), "output_index[%zu] out of range[%zu]",
                   output_index, gm_tensor_sizes_->output_sizes.size());
    const auto &output_size = gm_tensor_sizes_->output_sizes[output_index];
    if (output_size.IsConstExpr()) {
      int64_t output_size_value = 0;
      GE_ASSERT_TRUE(output_size.GetConstValue(output_size_value));
      constexpr int64_t kL2SizeScaleThreshold = 2;
      if (output_size_value > (l2_size_ * kL2SizeScaleThreshold)) {
        ss << tensor.name << ".SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);" << std::endl;
      } else {
        GELOGD("outputs[%zu] is static-shaped, size = %ld, skip gen hint code", output_index, output_size_value);
      }
    } else {
      ss << "if (" << this->tiler.Size(output_size) << " > kL2CacheSize * 2) {" << std::endl;
      ss << "  " << tensor.name << ".SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);" << std::endl;
      ss << "}" << std::endl;
    }
  }
  return af::SUCCESS;
}

Status Kernel::AppendConstTensorInit(std::stringstream &ss) const {
  for (std::size_t i = 0; i < this->constant_tensors.size(); i++) {
    auto tensor = this->tpipe.tensors.find(this->constant_tensors[i]);
    if (tensor == this->tpipe.tensors.end()) {
      GELOGE(af::FAILED, "Codegen concat tensor id[%ld] not found", this->constant_tensors[i]);
      return af::FAILED;
    }
    GELOGI("const_value_expr: %s", tensor->second.const_value_expr.Str().get());

    string const_value = tensor->second.const_value.empty() ? tiler.Size(tensor->second.const_value_expr, true)
                                                            : tensor->second.const_value;
    ss << tensor->second.DefineConst(const_value.c_str()) << std::endl;
    GELOGI("Define ss value: %s", ss.str().c_str());
  }
  return af::SUCCESS;
}

Status Kernel::AppendUbScalarTensorInit(std::stringstream &ss) const {
  for (std::size_t i = 0; i < this->ub_scalar_tensors.size(); i++) {
    auto tensor = this->tpipe.tensors.find(this->ub_scalar_tensors[i]);
    GE_ASSERT_TRUE((tensor != this->tpipe.tensors.end()), "Codegen ub_scalar tensor id[%ld] not found",
                   this->ub_scalar_tensors[i]);

    std::string def_ub_scalar;
    GE_CHK_STATUS_RET(tensor->second.DefineUbScalar(def_ub_scalar));
    ss << def_ub_scalar;
    GELOGI("Define ub_scalar var: %s", def_ub_scalar.c_str());
  }
  return af::SUCCESS;
}

Status Kernel::AppendWorkspaceTensorInit(std::stringstream &ss,
                                         const std::string &workspace_buffer_arg_override) const {
  const auto workspace_buffer_arg =
      workspace_buffer_arg_override.empty() ? this->workspace_arg : GM_ADDR("workspace_reuse");
  if (!workspace_buffer_arg_override.empty()) {
    ss << workspace_buffer_arg.AsArg() << " = " << workspace_buffer_arg_override.c_str() << ";" << std::endl;
  }

  auto it_ws_tensors = this->workspace_tensors.begin();
  for (size_t i = 0UL; i < this->workspaces.size(); i++) {
    GELOGI("Define workspace tensor id: %ld", it_ws_tensors->first);
    auto tensor = this->tpipe.tensors.find(it_ws_tensors->first);
    if (tensor == this->tpipe.tensors.end()) {
      GELOGE(af::FAILED, "Codegen workspace tensor id[%ld] not found", it_ws_tensors->first);
      return af::FAILED;
    }

    ss << tensor->second.Define() << std::endl;
    std::stringstream offset_ss;
    offset_ss << this->workspaces[i];
    std::string local_result;
    GE_CHK_STATUS_RET(tensor->second.SetGlobalBuffer(workspace_buffer_arg, offset_ss.str(), local_result),
                      "Codegen set global buffer failed");
    ss << local_result << std::endl;
    it_ws_tensors++;
  }
  return af::SUCCESS;
}

Status Kernel::LocalTensorQueBufAlloc(std::string &result, const ascir::ImplGraph &graph) const {
  (void)graph;
  stringstream ss;
  std::string tmp;

  ss << this->tpipe.Define() << std::endl;

  if (!this->pre_api_extract_dup.empty()) {
    ss << this->tpipe.GenDuplicateBufAlloc(this->pre_api_extract_dup) << std::endl;
  }

  GE_CHK_STATUS_RET(this->tpipe.BlkTensorAllocAndInit(tmp), "Codegen BlkTensorAllocAndInit failed");
  ss << tmp << std::endl;

  GE_CHK_STATUS_RET(this->tpipe.LocalTensorQueBufAlloc(tmp), "Codegen alloc local tensor que buf failed");
  ss << tmp << std::endl;

  result = ss.str();
  return af::SUCCESS;
}

Status Kernel::ParseWorkspaceTensor(const ascir::TensorAttr *tensor,
                                    const ascir::FusedScheduledResult &fused_schedule_result,
                                    std::set<int64_t> &output_indices,
                                    const std::unordered_map<ascir::TensorId, size_t> &output_tensorid_to_index,
                                    const std::map<size_t, std::string> output_index_to_name) {
  if (this->tpipe.tensors.find(tensor->attr.mem.tensor_id) == this->tpipe.tensors.cend()) {
    GE_CHK_STATUS_RET(this->tpipe.AddTensor(*tensor, "workspace"), "Codegen add tensor failed");
  }
  // workspace 复用 output节点场景需要将该节点添加到该kernel中 用于声明kernel函数入参及变量
  if (output_tensorid_to_index.find(tensor->attr.mem.tensor_id) != output_tensorid_to_index.cend()) {
    int64_t index;
    const auto &out_node = fused_schedule_result.output_nodes[output_tensorid_to_index.at(tensor->attr.mem.tensor_id)];
    GE_CHK_GRAPH_STATUS_RET(out_node->attr.ir_attr->GetAttrValue("index", index),
                            "Failed to get Workspace reuse Output index, node = %s", out_node->GetNamePtr());
    GE_ASSERT_TRUE(index >= 0, "invalid Workspace reuse Output index, node = %s, index = %ld", out_node->GetNamePtr(),
                   index);
    GE_ASSERT_TRUE(output_index_to_name.find(index) != output_index_to_name.cend(),
                   "Get workspace reuse output name failed.");
    const auto &output_name = output_index_to_name.at(index);
    GE_ASSERT_TRUE(!output_name.empty(), "Failed to get workspace reuse arg name, output_node = %s, index = %ld",
                   out_node->GetNamePtr(), index);
    if (output_indices.emplace(index).second) {
      this->outputs.emplace_back(GM_ADDR(GenValidName(output_name)));
      this->output_tensors.emplace_back(out_node->inputs[0].attr.mem.tensor_id);
    }
  }
  return af::SUCCESS;
}

namespace {
// GetTmpBuffer 会为 tmp 公式套上 v1 repeat cap 壳: Min(size, 65312)。CV UBFuse 的 tmp
// 内联公式不走 tiling 求解, stage 尺寸由 cube tiling 决定, 需求无法缩到 cap 以内, 因此
// 在 CV 收集点剥掉该壳取原始需求。数值须与 GetTmpBuffer 的 cap(255 * 256 + 32)保持一致。
// 非 cap 形态(无壳或常量非该值)原样返回; VV 路径不经此处理, 继续使用 capped size 参与
// host tiling 求解。
constexpr int64_t kMaxV1RepeatTmpBytes = 255LL * 256LL + 32LL;

// GetArgs 为非 const 接口, 此处按值传递以便调用。
af::Expression StripV1RepeatCap(af::Expression size) {
  if (!size.IsValid()) {
    return size;
  }
  auto args = size.GetArgs();
  if (args.size() != 2UL) {
    return size;
  }
  for (size_t i = 0UL; i < args.size(); i++) {
    int64_t cap_value = 0;
    if (args[i].IsConstExpr() && args[i].GetConstValue(cap_value) && cap_value == kMaxV1RepeatTmpBytes) {
      return args[1UL - i];
    }
  }
  return size;
}
}  // namespace

Status Kernel::ParseGraph(const ascir::ImplGraph &graph, const ascir::FusedScheduledResult &fused_schedule_result,
                          Kernel &kernel) {
  // Parse kernel input output
  GE_CHK_STATUS_RET(CheckGraphValidity(graph), "Graph: %s is invalid", graph.GetName().c_str());
  std::map<size_t, std::string> input_index_to_name;
  std::map<size_t, std::string> output_index_to_name;
  std::unordered_map<ascir::TensorId, size_t> output_tensorid_to_index;

  for (size_t i = 0U; i < fused_schedule_result.input_nodes.size(); ++i) {
    const auto &input = fused_schedule_result.input_nodes[i];
    GE_ASSERT_TRUE(IsOps<Data>(input) || IsOps<ScalarData>(input), "Codegen unsupported input[%s] type[%s]",
                   input->GetName().c_str(), input->GetType().c_str());
    const auto &normalized_name = GenValidName(input->GetName());
    kernel.input_name_to_index_[normalized_name] = i;
    input_index_to_name[i] = normalized_name;
    GELOGD("input_index = %zu, input_name = %s", i, normalized_name.c_str());
  }
  for (size_t i = 0U; i < fused_schedule_result.output_nodes.size(); ++i) {
    const auto &output = fused_schedule_result.output_nodes[i];
    const auto &normalized_name = GenValidName(output->GetName());
    kernel.output_name_to_index_[normalized_name] = i;
    output_index_to_name[i] = output->GetName();
    output_tensorid_to_index[output->inputs[0].attr.mem.tensor_id] = i;
    GELOGD("output_index = %zu, output_name = %s", i, normalized_name.c_str());
  }
  std::set<int64_t> input_indices;
  std::set<int64_t> output_indices;
  std::map<int64_t, std::pair<std::string, ascir::TensorId>> kernel_outputs;
  bool has_gather = false;
  for (const auto &node : graph.GetAllNodes()) {
    if (IsOps<Data>(node) || IsOps<ScalarData>(node)) {
      int64_t index;
      GE_CHK_GRAPH_STATUS_RET(node->attr.ir_attr->GetAttrValue("index", index), "Failed to get Data index, node = %s",
                              node->GetNamePtr());
      GE_ASSERT_TRUE(index >= 0, "invalid Data index, node = %s, index = %ld", node->GetNamePtr(), index);
      const auto &input_name = input_index_to_name[index];
      GE_ASSERT_TRUE(!input_name.empty(), "Failed to get arg name, input_node = %s, index = %ld", node->GetNamePtr(),
                     index);
      if (input_indices.emplace(index).second) {
        if (IsOps<Data>(node)) {
          kernel.inputs.emplace_back(GM_ADDR(GenValidName(input_name)));
        } else if (IsOps<ScalarData>(node)) {
          std::string dtype_name;
          GE_CHK_STATUS_RET(Tensor::DtypeName(node->outputs[0].attr.dtype, dtype_name), "data type:%d failed",
                            static_cast<int32_t>(node->outputs[0].attr.dtype));
          kernel.inputs.emplace_back(Variable(Type(dtype_name), input_name));
        }
        kernel.input_tensors.emplace_back(node->outputs[0].attr.mem.tensor_id);
      }
      continue;
    }
    if (IsOps<Output>(node)) {
      int64_t index;
      GE_CHK_GRAPH_STATUS_RET(node->attr.ir_attr->GetAttrValue("index", index), "Failed to get Data index, node = %s",
                              node->GetNamePtr());
      GE_ASSERT_TRUE(index >= 0, "invalid Data index, node = %s, index = %ld", node->GetNamePtr(), index);
      const auto &output_name = output_index_to_name[index];
      GE_ASSERT_TRUE(!output_name.empty(), "Failed to get arg name, output_node = %s, index = %ld", node->GetNamePtr(),
                     index);
      if (output_indices.emplace(index).second) {
        kernel_outputs[index] = std::make_pair(output_name, node->inputs[0].attr.mem.tensor_id);
      }
      continue;
    }
    has_gather =
        (has_gather || optimize::ScheduleUtils::IsGatherLikeLoad(std::dynamic_pointer_cast<af::AscNode>(node)));
  }
  for (const auto &pair : kernel_outputs) {
    kernel.outputs.emplace_back(GM_ADDR(GenValidName(pair.second.first)));
    kernel.output_tensors.emplace_back(pair.second.second);
  }

  std::vector<ascir::TensorId> workspace_tensor_id = GetWorkspaceTensorIdListInOneGraph(fused_schedule_result, graph);
  for (auto tId : workspace_tensor_id) {
    std::string workspaceStr = "workspace";
    workspaceStr = workspaceStr + std::to_string(tId);
    kernel.workspaces.emplace_back(Uint64(workspaceStr));
    kernel.workspace_tensors[tId] = "0";
  }

  for (auto node : graph.GetAllNodes()) {
    if (IsOps<Scalar>(node) || IsOps<IndexExpr>(node)) {
      kernel.constant_tensors.emplace_back(node->outputs[0].attr.mem.tensor_id);
    }
  }

  // Parse for tiler
  for (auto axis : graph.GetAllAxis()) {
    GE_CHK_STATUS_RET(kernel.tiler.AddAxis(*axis), "Codegen add axis failed");
  }
  kernel.tiler.AddAxisSplitBAttr();

  for (auto size : graph.GetAllSizeVar()) {
    kernel.tiler.AddSizeVar(*size);
  }

  GE_ASSERT_SUCCESS(kernel.tpipe.CollectQues(graph));
  // Parse for tpipe
  for (auto node : graph.GetAllNodes()) {
    if (IsOps<Output>(node) || IsOps<Data>(node) || IsOps<ScalarData>(node)) {
      continue;
    }
    if (ascgen_utils::indirect_load::ShouldSkipTpipeTensorCollection(node) && !IsOps<Store>(node)) {
      continue;
    }

    auto desc = node->GetOpDesc();
    for (auto output : node->outputs()) {
      auto output_index = af::ascir::AscTensorUtils::Index(*output);
      auto tensor_name = node->GetName() + "_" + desc->GetOutputNameByIndex(output_index);
      if (IsOps<Scalar>(node)) {
        std::string const_value;
        auto ir_attr = node->attr.ir_attr.get();
        if (ir_attr->GetAttrValue("value", const_value) != af::GRAPH_SUCCESS) {
          GELOGE(af::FAILED, "GetAttrValue const value failed");
          return af::FAILED;
        }
        GELOGI("Scalar node const value %s", const_value.c_str());
        // 不要将const_value放在参数第二个位置，会导致overload出现歧义
        GE_CHK_STATUS_RET(kernel.tpipe.AddTensor(const_value, *output, tensor_name), "Codegen add tensor failed");
        GE_CHK_STATUS_RET(kernel.ParseOptimizeInfo(node, *output));
      } else if (IsOps<IndexExpr>(node)) {
        af::Expression expr;
        auto ir_attr = node->attr.ir_attr.get();
        if (ir_attr->GetAttrValue("expr", expr) != af::GRAPH_SUCCESS) {
          GELOGE(af::FAILED, "GetAttrValue index expr failed");
          return af::FAILED;
        }
        GE_CHK_STATUS_RET(kernel.tpipe.AddTensor(*output, expr, tensor_name), "Codegen add tensor failed");
        GE_CHK_STATUS_RET(kernel.ParseOptimizeInfo(node, *output));
      } else if (IsOps<Workspace>(node)) {
        GE_CHK_STATUS_RET(kernel.ParseWorkspaceTensor(output, fused_schedule_result, output_indices,
                                                      output_tensorid_to_index, output_index_to_name),
                          "Codegen parse workspace tensor failed");
        kernel.has_workspace_node = true;
      } else if (IsOps<Store>(node)) {
        if (ascgen_utils::indirect_load::ShouldSkipTpipeTensorCollection(node)) {
          // Skipped Store still needs a GM tensor entry for the downstream Output binding.
          std::string dtype_name;
          GE_CHK_STATUS_RET(Tensor::DtypeName(output->attr.dtype, dtype_name), "Codegen get output dtype failed");
          Tensor gm_output(*output, dtype_name, tensor_name);
          gm_output.alloc_type = af::AllocType::kAllocTypeGlobal;
          gm_output.position = af::Position::kPositionGM;
          gm_output.que_id = af::kIdNone;
          GE_CHK_STATUS_RET(gm_output.Init(), "Codegen init direct-GM output tensor failed");
          GE_CHK_STATUS_RET(kernel.tpipe.AddTensor(gm_output), "Codegen add direct-GM output tensor failed");
          continue;
        }
        // 1. 多个Store节点写同一个Output的不同offset场景
        // 2. 多个schedule group之间通过workspace承接输出
        if (kernel.tpipe.tensors.find(output->attr.mem.tensor_id) == kernel.tpipe.tensors.cend()) {
          GE_CHK_STATUS_RET(kernel.tpipe.AddTensor(*output, tensor_name), "Codegen add tensor failed");
        }
      } else {
        GE_CHK_STATUS_RET(kernel.tpipe.AddTensor(*output, tensor_name), "Codegen add tensor failed");
        GE_CHK_STATUS_RET(kernel.ParseOptimizeInfo(node, *output));
      }
    }
  }

  for (auto node : fused_schedule_result.input_nodes) {
    auto desc = node->GetOpDesc();
    for (auto output : node->outputs()) {
      auto tensor_name = node->GetName() + "_" + desc->GetOutputNameByIndex(af::ascir::AscTensorUtils::Index(*output));
      if (IsOps<ScalarData>(node)) {
        GELOGI("ScalarData node const value %s", node->GetName().c_str());
        GE_CHK_STATUS_RET(kernel.tpipe.AddTensor(GenValidName(node->GetName()), *output, tensor_name),
                          "Codegen add tensor failed");
        GE_CHK_STATUS_RET(kernel.ParseOptimizeInfo(node, *output));
      } else {
        GE_CHK_STATUS_RET(kernel.tpipe.AddTensor(*output, tensor_name), "Codegen add tensor failed");
      }
    }
  }

  for (auto node : fused_schedule_result.workspace_nodes) {
    GE_ASSERT_TRUE(IsOps<Workspace>(node), "fused_schedule_result node[%s] is not workspace", node->GetName().c_str());
    auto &output = node->outputs[0U];
    // workspace作为schedule group之间的输入时需要在此处AddTensor
    // 作为schedule group之间的输出时映射为store的输出，在Store节点被AddTensor
    if (kernel.tpipe.tensors.find(output.attr.mem.tensor_id) == kernel.tpipe.tensors.cend()) {
      GELOGD("Workspace node[%s] is input in a schedule group, tensor id[%ld]", node->GetName().c_str(),
             output.attr.mem.tensor_id);
      GE_CHK_STATUS_RET(kernel.tpipe.AddTensor(output, "workspace"), "Codegen add tensor failed");
    }
  }

  for (auto node : graph.GetAllNodes()) {
    for (auto tmp_buffer : node->attr.tmp_buffers) {
      if (tmp_buffer.id == -1) {
        continue;
      }
      auto it = kernel.tpipe.bufs.find(tmp_buffer.id);
      GELOGD("reuse tmp buffer id is %ld.", tmp_buffer.id);
      // CV UBFuse 使用无 v1 repeat cap 的原始 tmp 需求; 其余路径(含 VV 的 host tiling
      // 求解)保持 capped size 原样, 不引入额外拷贝。
      af::Expression stripped;
      const af::Expression *tmp_size = &tmp_buffer.buf_desc.size;
      if (kernel.tpipe.cv_fusion_type == ascir::CubeTemplateType::kUBFuse) {
        stripped = StripV1RepeatCap(tmp_buffer.buf_desc.size);
        tmp_size = &stripped;
      }
      if (it == kernel.tpipe.bufs.end()) {
        std::string position = "TPosition::VECCALC";
        ascir::Position tensor_position = af::Position::kPositionVecCalc;
        auto [new_buf, is_insert5] =
            kernel.tpipe.bufs.emplace(tmp_buffer.id, TBuf{tmp_buffer.id, tensor_position, position});
        GE_CHK_BOOL_RET_STATUS(is_insert5, af::FAILED, "Codegen emplace tbuf [%ld] failed", tmp_buffer.id);
        new_buf->second.tmp_buf_size_list.emplace_back(*tmp_size);
        new_buf->second.tmp_buf_reuse = true;
      } else {
        it->second.tmp_buf_size_list.emplace_back(*tmp_size);
        it->second.tmp_buf_reuse = true;
      }
    }
  }
  uint32_t total_blk_num = 0U;
  GetApiExtractDupSet(graph, kernel.pre_api_extract_dup, total_blk_num);
  kernel.SetEnableParallelCompile((!has_gather));
  GE_CHK_STATUS_RET(kernel.InitL2CacheHintInfo(fused_schedule_result, graph));
  if (IsCVFusionUBGraph(graph, kernel.tpipe.cv_fusion_type)) {
    GE_CHK_STATUS_RET(kernel.tpipe.GetCVFusionCubeOutputUBTensorIdAndQueId(graph), "get cube output tensor id failed");
  }
  // Parse for loop
  return kernel.root_loop.ConstructFromNodes(graph.GetAllNodes(), kernel.tiler, kernel.tpipe);
}

Status Kernel::GenerateSubGraphFuncDef(const Loop *loop, std::stringstream &ss) const {
  // 用栈存储待处理的Loop节点，模拟递归调用栈
  GE_ASSERT_NOTNULL(loop);
  std::stack<const Loop *> loop_stack;
  loop_stack.push(loop);

  // 循环处理栈中所有节点，直到栈为空
  while (!loop_stack.empty()) {
    // 弹出栈顶节点（当前要处理的节点）
    const Loop *current_loop = loop_stack.top();
    loop_stack.pop();

    // 遍历当前节点的所有body
    for (auto &body : current_loop->bodys) {
      if (body.type == LoopType::LOOP) {
        GE_ASSERT_NOTNULL(body.loop);
        loop_stack.push(body.loop);
      } else if (body.type == LoopType::CALL) {
        GE_ASSERT_NOTNULL(body.call);
        GE_ASSERT_SUCCESS(body.call->GenerateFuncDefinition(tpipe, tiler, ss),
                          "gen func definition failed, api_name:%s", body.call->api_name_.c_str());
      }
    }
  }

  return af::SUCCESS;
}

Status Kernel::Generate(const std::string &impl_graph_name, const std::string &tiling_data, std::string &result,
                        const ascir::ImplGraph &graph, const std::string &workspace_buffer_arg_override) {
  if (ascgen_utils::IsCubeType(graph)) {
    return af::SUCCESS;
  }
  stringstream ss;

  GE_ASSERT_SUCCESS(GenerateSubGraphFuncDef(&(this->root_loop), ss));

  ss << this->TilingKeyFuncDeclare(impl_graph_name, tiling_data) << " {" << std::endl;

  std::string tmp;
  for (auto &[id, axis] : tiler.axis_map) {
    if (axis.IsInner() || IsMergeFromInner(tiler, axis)) {
      continue;
    }

    if (axis.IsOuter() && !(axis.type == ascir::Axis::Type::kAxisTypeBlockOuter && axis.from.size() > 1)) {
      ss << tiler.GenAxisSizeNew(axis.split_pair_other_id);
    }
    ss << tiler.GenAxisSizeNew(id);
  }

  ss << this->tiler.BlockOutterAxisDefine();
  ss << std::endl;
  GE_CHK_STATUS_RET(this->GlobalTensorInit(tmp, workspace_buffer_arg_override), "Codegen global tensor init failed");
  ss << tmp;
  ss << std::endl;
  GE_CHK_STATUS_RET(this->LocalTensorQueBufAlloc(tmp, graph), "Codegen alloc local tensor que buf failed");
  ss << tmp;

  GE_CHK_STATUS_RET(this->root_loop.Generate(this->tiler, this->tpipe, tmp), "Codegen root loop Generate failed");
  ss << tmp;

  ss << "}" << std::endl;

  if (ascir::utils::IsCodegenCompileEnabled()) {
    std::string prefix = ascir::utils::GetDumpFilePrefix();
    if (!prefix.empty()) {
      (void)CodegenApiParam::DumpGraphApiParams(graph, this->tiler, prefix);
    }
  }

  result = ss.str();
  return af::SUCCESS;
}

std::string Kernel::GetIncludeApiHeaderFiles(const ascir::FusedScheduledResult &fused_schedule_result) {
  std::set<std::string> api_header_list = {
      "basic_api/kernel_tpipe.h",
      "basic_api/kernel_tensor.h",
      "basic_api/kernel_type.h",
      "basic_api/kernel_operator_block_sync_intf.h",
      "basic_api/kernel_operator_data_copy_intf.h",
      "basic_api/kernel_common.h",
      "basic_api/kernel_operator_common_intf.h",
      "basic_api/kernel_operator_sys_var_intf.h",
      "basic_api/kernel_struct_binary.h",
      "basic_api/kernel_operator_limits_intf.h",
  };
  std::stringstream ss;
  for (const auto &header : api_header_list) {
    ss << "#include \"" << header << "\"" << std::endl;
  }
  for (size_t graph_id = 0; graph_id < fused_schedule_result.node_idx_to_scheduled_results.size(); graph_id++) {
    auto scheduled_results = fused_schedule_result.node_idx_to_scheduled_results[graph_id];
    for (size_t i = 0; i < scheduled_results.size(); i++) {
      auto schedule_groups = scheduled_results[i].schedule_groups;
      for (size_t j = 0; j < schedule_groups.size(); j++) {
        auto schedule_graphs = schedule_groups[j].impl_graphs;
        for (size_t k = 0; k < schedule_graphs.size(); k++) {
          for (const auto &node : schedule_graphs[k].GetAllNodes()) {
            auto impl = ascgen_utils::GetAscIrCodegenImpl(node->GetType());
            GE_ASSERT_NOTNULL(impl, "GetAscIrCodegenImpl of node %s[%s] is null", node->GetTypePtr(),
                              node->GetNamePtr());
            for (const auto &header_str : impl->IncludeApiHeaderFiles()) {
              if (api_header_list.count(header_str) == 0) {
                api_header_list.insert(header_str);
                ss << "#include \"" << header_str << "\"" << std::endl;
              }
            }
          }
        }
      }
    }
  }
  return ss.str();
}

std::string Kernel::IncludeAndDefines(const ascir::FusedScheduledResult &fused_schedule_result,
                                      const std::string &kernel_task_type, bool use_tensor_desc, bool is_inductor) {
  std::stringstream ss;

  ss << Kernel::GetIncludeApiHeaderFiles(fused_schedule_result);
  if (use_tensor_desc) {
    ss << "#include \"kernel_operator_list_tensor_intf.h\"" << std::endl;
  }
  ss << "#include \"autofuse_tiling_data.h\"" << std::endl;
  ss << std::endl;
  ss << "using namespace AscendC;" << std::endl;
  ss << std::endl;
  if (!is_inductor) {
    ss << "KERNEL_TASK_TYPE_DEFAULT(" << kernel_task_type << ");" << std::endl;
  }
  ss << std::endl;

  const static string kAscendcUtilsExtend = {
#include "utils_str.h"
  };

  const static string kAscendcBrcInline = {
#include "brc_inline_api_str.h"
  };

  ss << kAscendcUtilsExtend << kAscendcBrcInline << std::endl;

  return ss.str();
}

std::string Kernel::KernelFuncDeclare(const std::string &graph_name,
                                      const ascir::FusedScheduledResult &fused_schedule_result, bool use_list_tensor,
                                      bool is_inductor, bool is_conv2d) {
  std::stringstream ss;
  if (ascgen_utils::IsCubeFusedScheduled(fused_schedule_result)) {
    if (is_conv2d) {
      ss << "template<int8_t FmapTiling, int8_t WeightTiling, int8_t L1PingPong, int8_t L0PingPong, int8_t "
            "OutputOrder, int8_t IterOrder, int8_t GroupType, int8_t EnableSmallChannel, int8_t WeightUbTrans, int8_t "
            "FmapCopyMode, int8_t InnerBatch, int8_t DisContinuous, int8_t BatchOne, int8_t NoPad, int8_t SmallWeight, "
            "int8_t SmallKernel> "
         << std::endl;
    } else {
      if (is_inductor) {
        ss << "template <typename TILING_DATA_T, int8_t API_LEVEL, int8_t A_TRANS, int8_t B_TRANS, int8_t BATCH_MODEL, "
              "int8_t MODEL, int8_t "
              "FULL_LOAD, int8_t L0C2OUT_MODEL, int8_t FUSION_MODE, int8_t UB_MODE, int8_t MIX_MODE>"
           << std::endl;
      } else {
        ss << "template <int8_t API_LEVEL, int8_t A_TRANS, int8_t B_TRANS, int8_t BATCH_MODEL, int8_t MODEL, int8_t "
              "FULL_LOAD, int8_t L0C2OUT_MODEL>"
           << std::endl;
      }
    }
    const char *flags[] = {"__global__", "__aicore__"};
    for (auto flag : flags) {
      ss << flag << " ";
    }
  } else {
    const char *flags[] = {"extern \"C\"", "__global__", "__aicore__"};
    for (auto flag : flags) {
      ss << flag << " ";
    }
  }
  ss << "void " << CamelToLowerSneak(graph_name) << "(";

  if (use_list_tensor) {
    ss << GM_ADDR("inputs").AsArg() << ", " << GM_ADDR("outputs").AsArg() << ", ";
  } else {
    for (auto &input : fused_schedule_result.input_nodes) {
      if (IsOps<ScalarData>(input)) {
        std::string dtype_name;
        GE_ASSERT_SUCCESS(Tensor::DtypeName(input->outputs[0].attr.dtype, dtype_name), "data type:%d failed",
                          static_cast<int32_t>(input->outputs[0].attr.dtype));
        ss << dtype_name << " " << GenValidName(input->GetName()) << ", ";
      } else {
        ss << GM_ADDR(GenValidName(input->GetName())).AsArg() << ", ";
      }
    }
    for (auto &output : fused_schedule_result.output_nodes) {
      ss << GM_ADDR(GenValidName(output->GetName())).AsArg() << ", ";
    }
  }
  ss << GM_ADDR("workspace").AsArg();
  if (is_inductor) {
    if (ascgen_utils::IsCubeFusedScheduled(fused_schedule_result)) {
      if (IsStaticSchedResult(fused_schedule_result)) {
        ss << "\n#ifndef INDUCTOR_CONST_TILING_DATA\n, ";
      } else {
        ss << ", ";
      }
      ss << "TILING_DATA_T t";
      if (IsStaticSchedResult(fused_schedule_result)) {
        ss << "\n#endif\n";
      }
    } else {
      ss << ", ";
      ss << "AutofuseTilingData t";
    }
  } else {
    ss << ", ";
    ss << GM_ADDR("gm_tiling_data").AsArg();
  }
  ss << ")";
  return ss.str();
}

std::string Kernel::GenTilingFuncCall(const std::string &impl_graph_name, const std::string &tiling_data,
                                      uint32_t index, bool enable_group_parallel, bool need_sync_all,
                                      const std::string &workspace_tiling_data) const {
  std::stringstream ss;
  ss << (index == 0 ? "    if (" : " else if (");
  const auto tiling_data_expr =
      tiling_data.find('.') == std::string::npos ? CamelToLowerSneak(tiling_data) : tiling_data;
  if (enable_group_parallel) {
    ss << "MatchTilingKeyAndBlockDim(" << tiling_data_expr << ", " << index << ")";
  } else {
    ss << tiling_data_expr << ".tiling_key == " << std::to_string(index);
  }
  ss << ") {" << std::endl;
  ss << "      " << CamelToLowerSneak(impl_graph_name) << "(";
  if (use_list_tensor_) {
    ss << kInputTensorDescName << ", " << kOutputTensorDescName << ", ";
  } else {
    for (auto &input : this->inputs) {
      ss << input.Str() << ", ";
    }
    for (auto &output : this->outputs) {
      ss << output.Str() << ", ";
    }
  }
  ss << this->workspace_arg.Str() << ", ";
  for (auto &workspace : this->workspaces) {
    ss << workspace_tiling_data << "." << workspace.Str() << ", ";
  }
  ss << "&" << tiling_data;
  ss << ");" << std::endl;
  if (need_sync_all) {
    ss << "      SyncAll();" << std::endl;
  }
  ss << "    }";
  return ss.str();
}

std::string Kernel::GenTilingFuncCall(const std::string &impl_graph_name, const std::string &tiling_data) const {
  std::stringstream string_stream;
  string_stream << CamelToLowerSneak(impl_graph_name) << "(";
  if (use_list_tensor_) {
    string_stream << kInputTensorDescName << ", " << kOutputTensorDescName << ", ";
  } else {
    for (auto &input : this->inputs) {
      string_stream << input.Str() << ", ";
    }
    for (auto &output : this->outputs) {
      string_stream << output.Str() << ", ";
    }
  }
  string_stream << this->workspace_arg.Str() << ", ";
  for (auto &workspace : this->workspaces) {
    string_stream << "t." << workspace.Str() << ", ";
  }
  string_stream << "&" << tiling_data;
  string_stream << ");";
  return string_stream.str();
}

Status Kernel::GenSingleGroupKernelWithRegTilingKey(const ascir::FusedScheduledResult &fused_schedule_result,
                                                    const CodegenConfig &config, std::stringstream &ss,
                                                    std::stringstream &ss1, bool use_list_tensor) {
  std::string tiling_data_type = "AutofuseTilingData";
  std::unordered_set<const std::string *> kernel_file_ptr;
  auto schedule_graphs = fused_schedule_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs;
  std::vector<TilingFuncCall> func_calls;
  uint32_t tiling_key = 0U;
  for (size_t i = 0; i < schedule_graphs.size(); i++) {
    Kernel kernel(schedule_graphs[i].GetName());
    kernel.tiler.SetTilingCaseId(i);
    kernel.SetUsingAttCalcQBTSizeConfig(config.using_att_calc_qbt_size);
    kernel.SetUseListTensor(use_list_tensor);
    kernel.tpipe.is_inductor = config.is_inductor;
    GE_CHK_STATUS_RET(Kernel::ParseGraph(schedule_graphs[i], fused_schedule_result, kernel),
                      "Codegen parse graph failed");
    GE_CHK_STATUS_RET(kernel.GenerateKernelByNode(schedule_graphs[i], ss, kernel_file_ptr, false));
    std::string tmp;
    GE_CHK_STATUS_RET(kernel.Generate(schedule_graphs[i].GetName(), tiling_data_type, tmp, schedule_graphs[i]),
                      "Codegen generate kernel for %s failed", schedule_graphs[i].GetName().c_str());
    ss << tmp;
    std::string func_call;
    if (ascgen_utils::IsCubeType(schedule_graphs[i])) {
      func_call = kernel.GenCubeTilingFuncCall(schedule_graphs[i]);
    } else {
      func_call = kernel.GenTilingFuncCall(schedule_graphs[i].GetName(), "t");
    }
    func_calls.emplace_back(func_call, kernel.has_workspace_node, false);
  }
  std::vector<TilingFuncCall> current;
  std::vector<std::vector<TilingFuncCall>> per_group_func_calls;
  per_group_func_calls.emplace_back(std::move(func_calls));
  AppendFuncCall(ss1, per_group_func_calls, current, 0, tiling_key,
                 ascgen_utils::IsCubeFusedScheduled(fused_schedule_result));
  return af::SUCCESS;
}

Status Kernel::GenMulGroupKernelWithRegTilingKey(const ascir::FusedScheduledResult &fused_schedule_result,
                                                 const CodegenConfig &config, std::stringstream &ss,
                                                 std::stringstream &ss1, bool use_list_tensor) {
  std::string tiling_data_type = "AutofuseTilingData";
  std::unordered_set<const std::string *> kernel_file_ptr;
  uint32_t tiling_key = 0U;
  for (size_t graph_id = 0; graph_id < fused_schedule_result.node_idx_to_scheduled_results.size(); graph_id++) {
    auto scheduled_results = fused_schedule_result.node_idx_to_scheduled_results[graph_id];
    for (size_t i = 0; i < scheduled_results.size(); i++) {
      auto schedule_groups = scheduled_results[i].schedule_groups;
      auto enable_group_parallel = scheduled_results[i].enable_group_parallel;
      std::vector<std::vector<TilingFuncCall>> per_group_func_calls;
      ascir::CubeTemplateType cv_fusion_type = scheduled_results[i].cube_type;
      if (cv_fusion_type == ascir::CubeTemplateType::kCommon) {  // 暂时不处理兜底模板
        continue;
      }
      for (size_t j = 0; j < schedule_groups.size(); j++) {
        auto schedule_graphs = schedule_groups[j].impl_graphs;
        std::vector<TilingFuncCall> func_calls;
        bool vector_no_db_flag = true;
        for (size_t k = 0; k < schedule_graphs.size(); k++) {
          std::string tiling_data = "AscGraph" + std::to_string(graph_id) + "ScheduleResult" + std::to_string(i) + "G" +
                                    std::to_string(j) + "TilingData";
          Kernel kernel(schedule_graphs[k].GetName());
          kernel.SetUsingAttCalcQBTSizeConfig(config.using_att_calc_qbt_size);
          kernel.SetUseListTensor(use_list_tensor);
          kernel.tiler.SetTilingCaseId(k);
          kernel.tiler.EnableGroupParallel(enable_group_parallel);
          kernel.tpipe.cv_fusion_type = cv_fusion_type;
          kernel.tpipe.is_inductor = config.is_inductor;
          GE_CHK_STATUS_RET(Kernel::ParseGraph(schedule_graphs[k], fused_schedule_result, kernel),
                            "Codegen parse graph failed");
          GE_CHK_STATUS_RET(kernel.GenerateKernelByNode(schedule_graphs[k], ss, kernel_file_ptr, false));
          if (IsCVFusionUBGraph(schedule_graphs[k], cv_fusion_type)) {
            GE_CHK_STATUS_RET(
                kernel.GenerateVecFuncOfCVFusion(ss, vector_no_db_flag, IsConv2DFusedScheduled(fused_schedule_result)),
                "Gen CV fusion Func failed");
          } else {
            std::string tmp;
            GE_CHK_STATUS_RET(kernel.Generate(schedule_graphs[k].GetName(), tiling_data, tmp, schedule_graphs[k]),
                              "Codegen generate kernel for %s failed", schedule_graphs[k].GetName().c_str());
            ss << tmp;
          }
          std::string filed_name = "t.graph" + std::to_string(graph_id) + "_result" + std::to_string(i) + "_g" +
                                   std::to_string(j) + "_tiling_data";
          bool need_sync_all = kernel.has_workspace_node && j != schedule_groups.size() - 1;
          std::string func_call;
          if (ascgen_utils::IsCubeType(schedule_graphs[k])) {
            func_calls.emplace_back(kernel.GenCubeTilingFuncCall(schedule_graphs[k]), kernel.has_workspace_node, true);
          } else if (cv_fusion_type == ascir::CubeTemplateType::kUBFuse) {
            GE_CHK_STATUS_RET(kernel.InitCVFusionAddr(ss1, vector_no_db_flag), "Init CV Fusion Addr failed");
            vector_no_db_flag = false;  // 和schedule约定，不开db的vector在前面
            continue;
          } else {
            func_calls.emplace_back(kernel.GenTilingFuncCall(schedule_graphs[k].GetName(), filed_name),
                                    kernel.has_workspace_node, need_sync_all);
          }
        }
        if (!func_calls.empty()) {
          per_group_func_calls.emplace_back(std::move(func_calls));
        }
      }
      std::vector<TilingFuncCall> current;
      AppendFuncCall(ss1, per_group_func_calls, current, 0, tiling_key,
                     ascgen_utils::IsCubeFusedScheduled(fused_schedule_result));
    }
  }
  return af::SUCCESS;
}

Status Kernel::GenKernelFuncWithRegTilingKey(const ascir::FusedScheduledResult &fused_schedule_result,
                                             const CodegenConfig &config, std::stringstream &ss, std::stringstream &ss1,
                                             bool use_list_tensor) {
  if (ascgen_utils::IsSingleGroup(fused_schedule_result)) {
    GE_ASSERT_SUCCESS(GenSingleGroupKernelWithRegTilingKey(fused_schedule_result, config, ss, ss1, use_list_tensor));
  } else {
    GE_ASSERT_SUCCESS(GenMulGroupKernelWithRegTilingKey(fused_schedule_result, config, ss, ss1, use_list_tensor));
  }
  return af::SUCCESS;
}

Status Kernel::GenSingleGroupKernelWithParseTilingData(const ascir::FusedScheduledResult &fused_schedule_result,
                                                       const std::vector<af::AscGraph> &schedule_graphs,
                                                       const CodegenConfig &config, std::stringstream &ss,
                                                       std::stringstream &ss1, bool use_list_tensor,
                                                       std::unordered_set<const std::string *> &kernel_file_ptr) {
  for (size_t i = 0; i < schedule_graphs.size(); i++) {
    Kernel kernel(schedule_graphs[i].GetName());
    kernel.SetUsingAttCalcQBTSizeConfig(config.using_att_calc_qbt_size);
    kernel.SetUseListTensor(use_list_tensor);
    kernel.tiler.SetTilingCaseId(i);
    kernel.tpipe.is_inductor = config.is_inductor;
    GE_CHK_STATUS_RET(Kernel::ParseGraph(schedule_graphs[i], fused_schedule_result, kernel),
                      "Codegen parse graph failed");
    GE_CHK_STATUS_RET(kernel.GenerateKernelByNode(schedule_graphs[i], ss, kernel_file_ptr, false));
    std::string tmp;
    GE_CHK_STATUS_RET(kernel.Generate(schedule_graphs[i].GetName(), "AutofuseTilingData", tmp, schedule_graphs[i]),
                      "Codegen generate kernel for %s failed", schedule_graphs[i].GetName().c_str());
    ss << tmp;
    if (ascgen_utils::IsCubeType(schedule_graphs[i])) {
      ss1 << kernel.GenCubeTilingFuncCall(schedule_graphs[i]);
    } else {
      ss1 << kernel.GenTilingFuncCall(schedule_graphs[i].GetName(), "t", i);
    }
  }
  return af::SUCCESS;
}

Status Kernel::GenCubeCommonFuncForScheduleGroup(const ascir::FusedScheduledResult &fused_schedule_result,
                                                 const size_t graph_id, const size_t common_index,
                                                 const size_t group_index, const CodegenConfig &config,
                                                 std::stringstream &ss, std::stringstream &res_ss,
                                                 const bool use_list_tensor,
                                                 std::unordered_set<const std::string *> &kernel_file_ptr) {
  const auto &scheduled_results = fused_schedule_result.node_idx_to_scheduled_results[graph_id];
  const auto &schedule_groups = scheduled_results[common_index].schedule_groups;
  auto enable_group_parallel = scheduled_results[common_index].enable_group_parallel;
  ascir::CubeTemplateType cv_fusion_type = scheduled_results[common_index].cube_type;
  const auto &schedule_graphs = schedule_groups[group_index].impl_graphs;
  const auto cube_output_override = GetCommonCvCubeReuseOutputName(schedule_groups, group_index);
  const auto vector_workspace_override = GetCommonCvVectorReuseOutputName(schedule_groups, group_index);
  GE_ASSERT_TRUE(!schedule_graphs.empty(), "schedule_graphs is empty");
  for (size_t k = 0U; k < schedule_graphs.size(); k++) {
    Kernel kernel(schedule_graphs[k].GetName());
    kernel.SetUsingAttCalcQBTSizeConfig(config.using_att_calc_qbt_size);
    kernel.SetUseListTensor(use_list_tensor);
    kernel.tiler.SetTilingCaseId(k);
    kernel.tiler.EnableGroupParallel(enable_group_parallel);
    kernel.tpipe.cv_fusion_type = cv_fusion_type;
    kernel.tpipe.is_inductor = config.is_inductor;
    GE_CHK_STATUS_RET(Kernel::ParseGraph(schedule_graphs[k], fused_schedule_result, kernel),
                      "Codegen parse graph failed");
    auto is_dynamic = !IsStaticSchedResult(fused_schedule_result) || config.is_inductor;
    GE_CHK_STATUS_RET(kernel.GenerateKernelByNode(schedule_graphs[k], ss, kernel_file_ptr, is_dynamic),
                      "Gen api headers failed");

    if (ascgen_utils::IsCubeType(schedule_graphs[k])) {
      res_ss << kernel.GenCubeCommonTilingSingleFuncCall(schedule_graphs[k], cube_output_override);
      return af::SUCCESS;
    } else {
      std::string tmp;
      GE_CHK_STATUS_RET(kernel.Generate(schedule_graphs[k].GetName(), "AutofuseTilingData", tmp, schedule_graphs[k],
                                        vector_workspace_override),
                        "Codegen generate cv kernel for %s failed", schedule_graphs[k].GetName().c_str());
      ss << tmp;
      AppendCommonCvVectorTilingFuncCall(kernel, fused_schedule_result, config, schedule_graphs[k].GetName(), k,
                                         enable_group_parallel, res_ss);
    }
  }
  return af::SUCCESS;
}

Status Kernel::GenCubeCommonFuncForAIV(const ascir::FusedScheduledResult &fused_schedule_result, const size_t graph_id,
                                       const size_t common_index, const size_t group_index, const CodegenConfig &config,
                                       std::stringstream &ss, std::stringstream &vec_ss, const bool use_list_tensor,
                                       std::unordered_set<const std::string *> &kernel_file_ptr) {
  vec_ss << "if ASCEND_IS_AIV {" << std::endl;
  vec_ss << "    SyncAll<false>();" << std::endl;
  vec_ss << "    #ifdef CV_AIV_NUM" << std::endl;
  vec_ss << "        if (GetBlockIdx() >= CV_AIV_NUM) {" << std::endl;
  vec_ss << "            return;" << std::endl;
  vec_ss << "        }" << std::endl;
  vec_ss << "    #endif" << std::endl;
  if (!IsEmptyTensorSence(fused_schedule_result)) {
    vec_ss << "    GET_TILING_DATA(t, gm_tiling_data);" << std::endl;
    GE_ASSERT_SUCCESS(GenCubeCommonFuncForScheduleGroup(fused_schedule_result, graph_id, common_index, group_index,
                                                        config, ss, vec_ss, use_list_tensor, kernel_file_ptr));
  }
  vec_ss << "}" << std::endl;
  return af::SUCCESS;
}

Status Kernel::GenCubeCommonFuncForAICMix(const ascir::FusedScheduledResult &fused_schedule_result,
                                          const size_t graph_id, const size_t common_index, const size_t group_index,
                                          const CodegenConfig &config, std::stringstream &ss,
                                          std::stringstream &cube_ss, const bool use_list_tensor,
                                          std::unordered_set<const std::string *> &kernel_file_ptr) {
  cube_ss << "    #ifdef CV_AIC_NUM" << std::endl;
  cube_ss << "      if ASCEND_IS_AIC {" << std::endl;
  cube_ss << "        if (GetBlockIdx() >= CV_AIC_NUM) {" << std::endl;
  cube_ss << "            SyncAll<false>();" << std::endl;
  cube_ss << "            return;" << std::endl;
  cube_ss << "        }" << std::endl;
  cube_ss << "      }" << std::endl;
  cube_ss << "    #endif" << std::endl;
  cube_ss << "    uint64_t vec_wss =  0U;" << std::endl;
  cube_ss << "    #ifdef CV_VEC_WSS" << std::endl;
  cube_ss << "        vec_wss =  CV_VEC_WSS;" << std::endl;
  cube_ss << "    #endif" << std::endl;
  GE_ASSERT_SUCCESS(GenCubeCommonFuncForScheduleGroup(fused_schedule_result, graph_id, common_index, group_index,
                                                      config, ss, cube_ss, use_list_tensor, kernel_file_ptr));
  cube_ss << "    if ASCEND_IS_AIC {" << std::endl;
  cube_ss << "      SyncAll<false>();" << std::endl;
  cube_ss << "    }" << std::endl;
  return af::SUCCESS;
}

Status Kernel::GenCubeCommonFuncForAIC(const ascir::FusedScheduledResult &fused_schedule_result, const size_t graph_id,
                                       const size_t common_index, const size_t group_index, const CodegenConfig &config,
                                       std::stringstream &ss, std::stringstream &cube_ss, const bool use_list_tensor,
                                       std::unordered_set<const std::string *> &kernel_file_ptr) {
  cube_ss << "if ASCEND_IS_AIC {" << std::endl;
  cube_ss << "    #ifdef CV_AIC_NUM" << std::endl;
  cube_ss << "        if (GetBlockIdx() >= CV_AIC_NUM) {" << std::endl;
  cube_ss << "            SyncAll<false>();" << std::endl;
  cube_ss << "            return;" << std::endl;
  cube_ss << "        }" << std::endl;
  cube_ss << "    #endif" << std::endl;
  cube_ss << "    uint64_t vec_wss =  0U;" << std::endl;
  cube_ss << "    #ifdef CV_VEC_WSS" << std::endl;
  cube_ss << "        vec_wss =  CV_VEC_WSS;" << std::endl;
  cube_ss << "    #endif" << std::endl;
  GE_ASSERT_SUCCESS(GenCubeCommonFuncForScheduleGroup(fused_schedule_result, graph_id, common_index, group_index,
                                                      config, ss, cube_ss, use_list_tensor, kernel_file_ptr));
  cube_ss << "    SyncAll<false>();" << std::endl;
  cube_ss << "}" << std::endl;
  return af::SUCCESS;
}

Status Kernel::GenCubeCommonFuncOfCVFusion(const ascir::FusedScheduledResult &fused_schedule_result,
                                           const size_t graph_id, const size_t common_index,
                                           const CodegenConfig &config, std::stringstream &ss, std::stringstream &ss1,
                                           const bool use_list_tensor,
                                           std::unordered_set<const std::string *> &kernel_file_ptr, bool is_dynamic) {
  const auto &scheduled_results = fused_schedule_result.node_idx_to_scheduled_results[graph_id];
  const auto &schedule_groups = scheduled_results[common_index].schedule_groups;
  std::vector<std::vector<std::string>> per_group_func_calls;
  for (size_t j = 0U; j < schedule_groups.size(); j++) {
    std::vector<std::string> func_calls;
    auto schedule_graphs = schedule_groups[j].impl_graphs;
    GE_ASSERT_TRUE(!schedule_graphs.empty(), "schedule_graphs is empty");
    bool is_cube_group = ascgen_utils::IsCubeType(schedule_graphs[0]);
    if (is_cube_group) {
      // 区别cube group和vector group，cube group只有一张schedule_graphs
      std::stringstream cube_ss;
      if (is_dynamic) {
        cube_ss << "if (cv_tiling_data.mix_mode == 1) {" << std::endl;
        GE_ASSERT_SUCCESS(GenCubeCommonFuncForAICMixDynamic(fused_schedule_result, graph_id, common_index, j, config,
                                                            ss, cube_ss, use_list_tensor, kernel_file_ptr));
        cube_ss << "} else if (cv_tiling_data.mix_mode == 2) {" << std::endl;
        GE_ASSERT_SUCCESS(GenCubeCommonFuncForAIVMixDynamic(fused_schedule_result, graph_id, common_index, j, config,
                                                            ss, cube_ss, use_list_tensor, kernel_file_ptr));
      } else {
        cube_ss << "#ifdef CV_SAFETY_FUSION_MIX_MODE" << std::endl;
        GE_ASSERT_SUCCESS(GenCubeCommonFuncForAICMix(fused_schedule_result, graph_id, common_index, j, config, ss,
                                                     cube_ss, use_list_tensor, kernel_file_ptr));
      }
      if (is_dynamic) {
        cube_ss << "} else {" << std::endl;
        GE_ASSERT_SUCCESS(GenCubeCommonFuncForAICDynamic(fused_schedule_result, graph_id, common_index, j, config, ss,
                                                         cube_ss, use_list_tensor, kernel_file_ptr));
      } else {
        cube_ss << "#else" << std::endl;
        GE_ASSERT_SUCCESS(GenCubeCommonFuncForAIC(fused_schedule_result, graph_id, common_index, j, config, ss, cube_ss,
                                                  use_list_tensor, kernel_file_ptr));
      }
      if (is_dynamic) {
        cube_ss << "}" << std::endl;
      } else {
        cube_ss << "#endif" << std::endl;
      }
      func_calls.emplace_back(cube_ss.str());
    } else {
      std::stringstream vec_ss;
      if (is_dynamic) {
        GE_ASSERT_SUCCESS(GenCubeCommonFuncForAIVDynamic(fused_schedule_result, graph_id, common_index, j, config, ss,
                                                         vec_ss, use_list_tensor, kernel_file_ptr));
      } else {
        GE_ASSERT_SUCCESS(GenCubeCommonFuncForAIV(fused_schedule_result, graph_id, common_index, j, config, ss, vec_ss,
                                                  use_list_tensor, kernel_file_ptr));
      }
      func_calls.emplace_back(vec_ss.str());
    }
    if (!func_calls.empty()) {
      if (is_cube_group) {
        // 保证cube处理代码在vector之前
        per_group_func_calls.insert(per_group_func_calls.cbegin(), std::move(func_calls));
      } else {
        per_group_func_calls.emplace_back(std::move(func_calls));
      }
    }
  }
  AppendFuncCall(ss1, per_group_func_calls.cbegin(), per_group_func_calls.cend());
  return af::SUCCESS;
}

Status Kernel::GenMulGroupKernelWithParseTilingData(const ascir::FusedScheduledResult &fused_schedule_result,
                                                    const size_t graph_id, const CodegenConfig &config,
                                                    std::stringstream &ss, std::stringstream &ss1, bool use_list_tensor,
                                                    std::unordered_set<const std::string *> &kernel_file_ptr) {
  auto scheduled_results = fused_schedule_result.node_idx_to_scheduled_results[graph_id];
  uint32_t function_id = kFuncIdBegin;
  const bool support_parallel_compile = !config.is_inductor;
  for (size_t i = 0; i < scheduled_results.size(); i++) {
    auto schedule_groups = scheduled_results[i].schedule_groups;
    auto enable_group_parallel = scheduled_results[i].enable_group_parallel;
    ascir::CubeTemplateType cv_fusion_type = scheduled_results[i].cube_type;
    if (cv_fusion_type == ascir::CubeTemplateType::kCommon) {
      GE_ASSERT_SUCCESS(GenCubeCommonFuncOfCVFusion(fused_schedule_result, graph_id, i, config, ss, ss1,
                                                    use_list_tensor, kernel_file_ptr));
      continue;
    } else if (cv_fusion_type == ascir::CubeTemplateType::kDefault) {
      ss1 << (i == 0 ? "  if" : " else if ") << "(t." << "graph" << std::to_string(graph_id)
          << "_tiling_key == " << std::to_string(i) << ") {" << std::endl;
    }
    std::vector<std::vector<std::string>> per_group_func_calls;
    bool enable_parallel_compile = true;
    for (size_t j = 0; j < schedule_groups.size(); j++) {
      std::vector<std::string> func_calls;
      auto schedule_graphs = schedule_groups[j].impl_graphs;
      bool vector_no_db_flag = true;
      for (size_t k = 0; k < schedule_graphs.size(); k++) {
        std::string tiling_data = "AscGraph" + std::to_string(graph_id) + "ScheduleResult" + std::to_string(i) + "G" +
                                  std::to_string(j) + "TilingData";
        Kernel kernel(schedule_graphs[k].GetName());
        kernel.SetUsingAttCalcQBTSizeConfig(config.using_att_calc_qbt_size);
        kernel.SetUseListTensor(use_list_tensor);
        kernel.tiler.SetTilingCaseId(k);
        kernel.tiler.EnableGroupParallel(enable_group_parallel);
        kernel.tpipe.cv_fusion_type = cv_fusion_type;
        kernel.tpipe.is_inductor = config.is_inductor;
        GE_CHK_STATUS_RET(Kernel::ParseGraph(schedule_graphs[k], fused_schedule_result, kernel),
                          "Codegen parse graph failed");
        GE_CHK_STATUS_RET(kernel.GenerateKernelByNode(schedule_graphs[k], ss, kernel_file_ptr, false),
                          "Gen api headers failed");

        std::string filed_name = "t.graph" + std::to_string(graph_id) + "_result" + std::to_string(i) + "_g" +
                                 std::to_string(j) + "_tiling_data";
        bool need_sync_all = kernel.has_workspace_node && j != schedule_groups.size() - 1;
        if (ascgen_utils::IsCubeType(schedule_graphs[k])) {
          func_calls.emplace_back(kernel.GenCubeTilingFuncCall(schedule_graphs[k]));
        } else if (cv_fusion_type == ascir::CubeTemplateType::kUBFuse) {
          GE_CHK_STATUS_RET(
              kernel.GenerateVecFuncOfCVFusion(ss, vector_no_db_flag, IsConv2DFusedScheduled(fused_schedule_result)),
              "Gen CV fusion Func failed");
          GE_CHK_STATUS_RET(kernel.InitCVFusionAddr(ss1, vector_no_db_flag), "Init CV Fusion Addr failed");
          vector_no_db_flag = false;  // 和schedule约定，不开db的vector在前面
          continue;
        } else {
          std::string tmp;
          GE_CHK_STATUS_RET(kernel.Generate(schedule_graphs[k].GetName(), tiling_data, tmp, schedule_graphs[k]),
                            "Codegen generate kernel for %s failed", schedule_graphs[k].GetName().c_str());
          ss << tmp;
          func_calls.emplace_back(kernel.GenTilingFuncCall(schedule_graphs[k].GetName(), filed_name, k,
                                                           enable_group_parallel, need_sync_all));
          enable_parallel_compile = (enable_parallel_compile && kernel.GetEnableParallelCompile());
        }
      }
      if (!func_calls.empty()) {
        per_group_func_calls.emplace_back(std::move(func_calls));
      }
    }
    auto max_group_per_compile_unit = GetMaxGroupPerCompileUnit(enable_parallel_compile);
    if (per_group_func_calls.size() <= static_cast<size_t>(max_group_per_compile_unit)) {
      AppendFuncCall(ss1, per_group_func_calls.cbegin(), per_group_func_calls.cend());
    } else {
      const auto kernel_args = PackingFuncArgs("const AutofuseTilingData", fused_schedule_result, use_list_tensor);
      const auto packing_func_names = Kernel::GenPackingFunctions(
          ss, kernel_args, per_group_func_calls, max_group_per_compile_unit, function_id, support_parallel_compile);
      GenPackingFunctionCalls(ss1, kernel_args, packing_func_names);
    }
    if (cv_fusion_type == ascir::CubeTemplateType::kDefault) {
      ss1 << "  }";
    }
  }
  if (support_parallel_compile) {
    FakeTilingIds(ss, function_id);
  }
  return af::SUCCESS;
}

Status Kernel::GenKernelFuncWithParseTilingData(const ascir::FusedScheduledResult &fused_schedule_result,
                                                const CodegenConfig &config, std::stringstream &ss,
                                                std::stringstream &ss1, bool use_list_tensor) {
  std::unordered_set<const std::string *> kernel_file_ptr;
  for (size_t graph_id = 0; graph_id < fused_schedule_result.node_idx_to_scheduled_results.size(); graph_id++) {
    auto scheduled_results = fused_schedule_result.node_idx_to_scheduled_results[graph_id];
    if ((fused_schedule_result.node_idx_to_scheduled_results.size() == 1) && (scheduled_results.size() == 1) &&
        (scheduled_results[0].schedule_groups.size() == 1)) {
      auto schedule_graphs = scheduled_results[0].schedule_groups[0].impl_graphs;
      GE_ASSERT_SUCCESS(GenSingleGroupKernelWithParseTilingData(fused_schedule_result, schedule_graphs, config, ss, ss1,
                                                                use_list_tensor, kernel_file_ptr));
    } else {
      GE_ASSERT_SUCCESS(GenMulGroupKernelWithParseTilingData(fused_schedule_result, graph_id, config, ss, ss1,
                                                             use_list_tensor, kernel_file_ptr));
      ss1 << std::endl;
      if ((graph_id + 1) < fused_schedule_result.node_idx_to_scheduled_results.size()) {
        ss1 << "  SyncAll();" << std::endl;
      }
    }
  }
  return af::SUCCESS;
}

Status Kernel::GenCubeCommonFuncForAIVDynamic(const ascir::FusedScheduledResult &fused_schedule_result,
                                              const size_t graph_id, const size_t common_index,
                                              const size_t group_index, const CodegenConfig &config,
                                              std::stringstream &ss, std::stringstream &vec_ss,
                                              const bool use_list_tensor,
                                              std::unordered_set<const std::string *> &kernel_file_ptr) {
  vec_ss << "if ASCEND_IS_AIV {" << std::endl;
  vec_ss << "    if (cv_tiling_data.mix_mode == 2) {" << std::endl;
  vec_ss << "      SyncAll<true>();" << std::endl;
  vec_ss << "    } else {" << std::endl;
  vec_ss << "      SyncAll<false>();" << std::endl;
  vec_ss << "    }" << std::endl;
  vec_ss << "    if (cv_tiling_data.cv_aiv_num != 0) {" << std::endl;
  vec_ss << "        if (GetBlockIdx() >= cv_tiling_data.cv_aiv_num) {" << std::endl;
  vec_ss << "            return;" << std::endl;
  vec_ss << "        }" << std::endl;
  vec_ss << "    }" << std::endl;
  if (!IsEmptyTensorSence(fused_schedule_result)) {
    GE_ASSERT_SUCCESS(GenCubeCommonFuncForScheduleGroup(fused_schedule_result, graph_id, common_index, group_index,
                                                        config, ss, vec_ss, use_list_tensor, kernel_file_ptr));
  }
  vec_ss << "}" << std::endl;
  return af::SUCCESS;
}

Status Kernel::GenCubeCommonFuncForAICDynamic(const ascir::FusedScheduledResult &fused_schedule_result,
                                              const size_t graph_id, const size_t common_index,
                                              const size_t group_index, const CodegenConfig &config,
                                              std::stringstream &ss, std::stringstream &cube_ss,
                                              const bool use_list_tensor,
                                              std::unordered_set<const std::string *> &kernel_file_ptr) {
  cube_ss << "if ASCEND_IS_AIC {" << std::endl;
  cube_ss << "    if (cv_tiling_data.cv_aic_num != 0) {" << std::endl;
  cube_ss << "        if (GetBlockIdx() >= cv_tiling_data.cv_aic_num) {" << std::endl;
  cube_ss << "            SyncAll<false>();" << std::endl;
  cube_ss << "            return;" << std::endl;
  cube_ss << "        }" << std::endl;
  cube_ss << "    }" << std::endl;
  cube_ss << "    uint64_t vec_wss =  0U;" << std::endl;
  cube_ss << "    if (cv_tiling_data.cv_vec_wss != 0) {" << std::endl;
  cube_ss << "        vec_wss = cv_tiling_data.cv_vec_wss;" << std::endl;
  cube_ss << "    }" << std::endl;
  GE_ASSERT_SUCCESS(GenCubeCommonFuncForScheduleGroup(fused_schedule_result, graph_id, common_index, group_index,
                                                      config, ss, cube_ss, use_list_tensor, kernel_file_ptr));
  cube_ss << "    SyncAll<false>();" << std::endl;
  cube_ss << "}" << std::endl;
  return af::SUCCESS;
}

Status Kernel::GenCubeCommonFuncForAICMixDynamic(const ascir::FusedScheduledResult &fused_schedule_result,
                                                 const size_t graph_id, const size_t common_index,
                                                 const size_t group_index, const CodegenConfig &config,
                                                 std::stringstream &ss, std::stringstream &cube_ss,
                                                 const bool use_list_tensor,
                                                 std::unordered_set<const std::string *> &kernel_file_ptr) {
  cube_ss << "    if (cv_tiling_data.cv_aic_num != 0) {" << std::endl;
  cube_ss << "      if ASCEND_IS_AIC {" << std::endl;
  cube_ss << "        if (GetBlockIdx() >= cv_tiling_data.cv_aic_num) {" << std::endl;
  cube_ss << "            SyncAll<false>();" << std::endl;
  cube_ss << "            return;" << std::endl;
  cube_ss << "        }" << std::endl;
  cube_ss << "      }" << std::endl;
  cube_ss << "    }" << std::endl;
  cube_ss << "    uint64_t vec_wss =  0U;" << std::endl;
  cube_ss << "    if (cv_tiling_data.cv_vec_wss != 0) {" << std::endl;
  cube_ss << "        vec_wss =  cv_tiling_data.cv_vec_wss;" << std::endl;
  cube_ss << "    }" << std::endl;
  GE_ASSERT_SUCCESS(GenCubeCommonFuncForScheduleGroup(fused_schedule_result, graph_id, common_index, group_index,
                                                      config, ss, cube_ss, use_list_tensor, kernel_file_ptr));
  cube_ss << "    if ASCEND_IS_AIC {" << std::endl;
  cube_ss << "      SyncAll<false>();" << std::endl;
  cube_ss << "    }" << std::endl;
  return af::SUCCESS;
}

Status Kernel::GenCubeCommonFuncForAIVMixDynamic(const ascir::FusedScheduledResult &fused_schedule_result,
                                                 const size_t graph_id, const size_t common_index,
                                                 const size_t group_index, const CodegenConfig &config,
                                                 std::stringstream &ss, std::stringstream &cube_ss,
                                                 const bool use_list_tensor,
                                                 std::unordered_set<const std::string *> &kernel_file_ptr) {
  cube_ss << "    uint64_t vec_wss =  0U;" << std::endl;
  cube_ss << "    if (cv_tiling_data.cv_vec_wss != 0) {" << std::endl;
  cube_ss << "        vec_wss =  cv_tiling_data.cv_vec_wss;" << std::endl;
  cube_ss << "    }" << std::endl;
  GE_ASSERT_SUCCESS(GenCubeCommonFuncForScheduleGroup(fused_schedule_result, graph_id, common_index, group_index,
                                                      config, ss, cube_ss, use_list_tensor, kernel_file_ptr));
  cube_ss << "    SyncAll<true>();" << std::endl;
  return af::SUCCESS;
}

Status Kernel::GenCVKernelFuncWithMulGroup(const ascir::FusedScheduledResult &fused_schedule_result,
                                           const CodegenConfig &config, std::stringstream &ss, std::stringstream &ss1,
                                           bool use_list_tensor) {
  std::unordered_set<const std::string *> kernel_file_ptr;
  const bool support_parallel_compile = !config.is_inductor;
  if (config.is_inductor) {
    ss1 << "#ifdef INDUCTOR_CONST_TILING_DATA" << std::endl;
    ss1 << "const CVAutofuseTilingData& gm_tiling_data = kConstTilingData;" << std::endl;
    ss1 << "#else" << std::endl;
    ss1 << "const TILING_DATA_T& gm_tiling_data = t;" << std::endl;
    ss1 << "#endif" << std::endl;
  } else {
    ss1 << "GET_TILING_DATA_WITH_STRUCT(CVTilingData, cv_tiling_data, gm_tiling_data + "
           "sizeof(AutofuseTilingData));"
        << std::endl;
  }
  ss1 << "if constexpr (FUSION_MODE == 0) {" << std::endl;
  for (size_t graph_id = 0; graph_id < fused_schedule_result.node_idx_to_scheduled_results.size(); graph_id++) {
    auto scheduled_results = fused_schedule_result.node_idx_to_scheduled_results[graph_id];
    uint32_t function_id = kFuncIdBegin;
    for (size_t i = 0; i < scheduled_results.size(); i++) {
      auto schedule_groups = scheduled_results[i].schedule_groups;
      auto enable_group_parallel = scheduled_results[i].enable_group_parallel;
      ascir::CubeTemplateType cv_fusion_type = scheduled_results[i].cube_type;
      if (cv_fusion_type == ascir::CubeTemplateType::kCommon) {
        ss1 << "} else {" << std::endl;
        if (config.is_inductor) {
          ss1 << "#ifdef INDUCTOR_CONST_TILING_DATA" << std::endl;
          ss1 << "const CVTilingData& cv_tiling_data = kConstTilingData.cv_tiling_data;" << std::endl;
          ss1 << "#else" << std::endl;
          ss1 << "const CVTilingData &cv_tiling_data = t.cv_tiling_data;" << std::endl;
          ss1 << "#endif" << std::endl;
        }
        GE_ASSERT_SUCCESS(GenCubeCommonFuncOfCVFusion(fused_schedule_result, graph_id, i, config, ss, ss1,
                                                      use_list_tensor, kernel_file_ptr, true));
        ss1 << "}" << std::endl;
        continue;
      } else if (cv_fusion_type == ascir::CubeTemplateType::kDefault) {
        ss1 << (i == 0 ? "  if" : " else if ") << "(t." << "graph" << std::to_string(graph_id)
            << "_tiling_key == " << std::to_string(i) << ") {" << std::endl;
      }
      std::vector<std::vector<std::string>> per_group_func_calls;
      bool enable_parallel_compile = true;
      for (size_t j = 0; j < schedule_groups.size(); j++) {
        std::vector<std::string> func_calls;
        auto schedule_graphs = schedule_groups[j].impl_graphs;
        bool vector_no_db_flag = true;
        for (size_t k = 0; k < schedule_graphs.size(); k++) {
          std::string tiling_data = "AscGraph" + std::to_string(graph_id) + "ScheduleResult" + std::to_string(i) + "G" +
                                    std::to_string(j) + "TilingData";
          Kernel kernel(schedule_graphs[k].GetName());
          kernel.SetUsingAttCalcQBTSizeConfig(config.using_att_calc_qbt_size);
          kernel.SetUseListTensor(use_list_tensor);
          kernel.tiler.SetTilingCaseId(k);
          kernel.tiler.EnableGroupParallel(enable_group_parallel);
          kernel.tpipe.cv_fusion_type = cv_fusion_type;
          kernel.tpipe.is_inductor = config.is_inductor;
          GE_CHK_STATUS_RET(Kernel::ParseGraph(schedule_graphs[k], fused_schedule_result, kernel),
                            "Codegen parse graph failed");
          GE_CHK_STATUS_RET(kernel.GenerateKernelByNode(schedule_graphs[k], ss, kernel_file_ptr, true),
                            "Gen api headers failed");

          std::string filed_name = "t.graph" + std::to_string(graph_id) + "_result" + std::to_string(i) + "_g" +
                                   std::to_string(j) + "_tiling_data";
          bool need_sync_all = kernel.has_workspace_node && j != schedule_groups.size() - 1;
          if (ascgen_utils::IsCubeType(schedule_graphs[k])) {
            func_calls.emplace_back(kernel.GenCubeTilingFuncCall(schedule_graphs[k], true));
          } else if (cv_fusion_type == ascir::CubeTemplateType::kUBFuse) {
            GE_CHK_STATUS_RET(
                kernel.GenerateVecFuncOfCVFusion(ss, vector_no_db_flag, IsConv2DFusedScheduled(fused_schedule_result),
                                                 true, config.is_inductor),
                "Gen CV fusion Func failed");
            GE_CHK_STATUS_RET(kernel.InitCVFusionAddr(ss1, vector_no_db_flag, true, config.is_inductor),
                              "Init CV Fusion Addr failed");
            vector_no_db_flag = false;  // 和schedule约定，不开db的vector在前面
            continue;
          } else {
            std::string tmp;
            GE_CHK_STATUS_RET(kernel.Generate(schedule_graphs[k].GetName(), tiling_data, tmp, schedule_graphs[k]),
                              "Codegen generate kernel for %s failed", schedule_graphs[k].GetName().c_str());
            ss << tmp;
            func_calls.emplace_back(kernel.GenTilingFuncCall(schedule_graphs[k].GetName(), filed_name, k,
                                                             enable_group_parallel, need_sync_all));
            enable_parallel_compile = (enable_parallel_compile && kernel.GetEnableParallelCompile());
          }
        }
        if (!func_calls.empty()) {
          per_group_func_calls.emplace_back(std::move(func_calls));
        }
      }
      auto max_group_per_compile_unit = GetMaxGroupPerCompileUnit(enable_parallel_compile);
      if (per_group_func_calls.size() <= static_cast<size_t>(max_group_per_compile_unit)) {
        AppendFuncCall(ss1, per_group_func_calls.cbegin(), per_group_func_calls.cend());
      } else {
        const auto kernel_args = PackingFuncArgs("const AutofuseTilingData", fused_schedule_result, use_list_tensor);
        const auto packing_func_names = Kernel::GenPackingFunctions(
            ss, kernel_args, per_group_func_calls, max_group_per_compile_unit, function_id, support_parallel_compile);
        GenPackingFunctionCalls(ss1, kernel_args, packing_func_names);
      }
      if (cv_fusion_type == ascir::CubeTemplateType::kDefault) {
        ss1 << "  }";
      }
    }
    if (support_parallel_compile) {
      FakeTilingIds(ss, function_id);
    }
    return af::SUCCESS;
  }
  return af::SUCCESS;
}

int64_t Kernel::GetMaxGroupPerCompileUnit(bool enable_parallel_compile) {
  uint32_t max_group_per_compile_unit = std::numeric_limits<uint32_t>::max();
  if (enable_parallel_compile) {
    auto backend_spec = optimize::BackendSpec::GetInstance();
    if (backend_spec != nullptr) {
      max_group_per_compile_unit = backend_spec->max_group_num_per_compile_unit;
    }
  }
  return max_group_per_compile_unit;
}

af::Status Kernel::GenCubeCommonTiling(std::stringstream &ss, const bool is_batch, bool is_conv2d, bool is_dynamic,
                                       bool is_db) const {
  if (is_conv2d) {
    // conv2d
    ss << "AscendC::TPipe pipe;" << std::endl;
    ss << "  conv2d_v2<";
    ss << "FmapTiling, WeightTiling, L1PingPong, L0PingPong, OutputOrder, IterOrder, GroupType, EnableSmallChannel, "
          "WeightUbTrans, FmapCopyMode, InnerBatch, DisContinuous, BatchOne, NoPad, SmallWeight, SmallKernel>(";
  } else {
    if (is_dynamic) {
      if (is_db) {
        if (is_batch) {
          ss << "  batch_mat_mul_v3_fusion_db<TILING_DATA_T, ";
        } else {
          ss << "  mat_mul_v3_fusion_db<TILING_DATA_T, ";
        }
      } else {
        if (is_batch) {
          ss << "  batch_mat_mul_v3_fusion<TILING_DATA_T, ";
        } else {
          ss << "  mat_mul_v3_fusion<TILING_DATA_T, ";
        }
      }
    } else {
      if (is_batch) {
        ss << "#ifdef INDUCTOR_TILING_DATA" << std::endl;
        ss << "  batch_mat_mul_v3<TILING_DATA_T, " << std::endl;
        ss << "#else" << std::endl;
        ss << "  batch_mat_mul_v3<" << std::endl;
        ss << "#endif" << std::endl;
      } else {
        ss << "#ifdef INDUCTOR_TILING_DATA" << std::endl;
        ss << "  mat_mul_v3<TILING_DATA_T, " << std::endl;
        ss << "#else" << std::endl;
        ss << "  mat_mul_v3<" << std::endl;
        ss << "#endif" << std::endl;
      }
    }
    ss << "API_LEVEL, A_TRANS, B_TRANS, BATCH_MODEL, MODEL, FULL_LOAD, L0C2OUT_MODEL>(";
  }
  return af::SUCCESS;
}

namespace {
Status AppendConv2DInputArgs(std::stringstream &ss, const std::vector<Variable> &inputs,
                             const CubeTilingOptions &options) {
  ss << inputs[0].Str() << ", " << inputs[1].Str() << ", ";
  size_t next_input_idx = 2U;
  for (const bool is_present : {options.has_bias, options.has_offset_w, options.has_scale0}) {
    ss << (is_present ? inputs[next_input_idx++].Str() : "nullptr") << ", ";
  }
  return af::SUCCESS;
}

Status AppendMatMulInputArgs(std::stringstream &ss, const std::vector<Variable> &inputs,
                             const CubeTilingOptions &options) {
  if (inputs.size() < (2U + (options.has_bias ? 1U : 0U) + (options.has_offset_w ? 1U : 0U))) {
    // a、b 矩阵同输入时 AscGraph 有两个输入，但 AscBackend 只有一个，需补齐第二个实参。
    ss << inputs[0].Str() << ", ";
  }
  for (const auto &input : inputs) {
    ss << input.Str() << ", ";
  }
  if (!options.has_bias) {
    ss << "nullptr, ";
  }
  if (!options.has_offset_w) {
    ss << "nullptr, ";
  }
  return af::SUCCESS;
}

Status AppendCubeInputArgs(std::stringstream &ss, const std::vector<Variable> &inputs,
                           const CubeTilingOptions &options) {
  return options.is_conv2d ? AppendConv2DInputArgs(ss, inputs, options) : AppendMatMulInputArgs(ss, inputs, options);
}

void AppendCubeOutputArgs(std::stringstream &ss, const std::vector<GM_ADDR> &outputs,
                          const std::string &empty_output_arg) {
  for (const auto &output : outputs) {
    ss << output.Str() << ", ";
  }
  if (outputs.empty()) {
    ss << empty_output_arg << ", ";
  }
}

CubeTilingOptions GetCubeTilingOptions(const ascir::ImplGraph &impl_graph) {
  CubeTilingOptions options;
  options.is_conv2d = IsConv2DGraphType(impl_graph);
  if (options.is_conv2d) {
    options.has_bias = ascgen_utils::IsConv2DTypeWithBias(impl_graph);
    options.has_offset_w = ascgen_utils::IsConv2DTypeWithOffsetW(impl_graph);
    options.has_scale0 = ascgen_utils::IsConv2DTypeWithScale0(impl_graph);
  } else {
    options.is_batch = ascgen_utils::IsMatMulTypeWithBatch(impl_graph);
    options.has_bias = ascgen_utils::IsMatMulTypeWithBias(impl_graph);
    options.has_offset_w = ascgen_utils::IsMatMulTypeWithOffsetW(impl_graph);
  }
  return options;
}
}  // namespace

std::string Kernel::GenCubeTilingSingleFuncCall(const CubeTilingOptions &options) const {
  std::stringstream ss;
  GE_CHK_STATUS(GenCubeCommonTiling(ss, options.is_batch, options.is_conv2d, options.is_dynamic, options.is_db),
                "GenCubeCommonTilingHead failed");
  if (this->use_list_tensor_) {
    ss << kInputTensorDescName << ", " << kOutputTensorDescName << ", ";
  } else {
    if (options.is_conv2d) {
      GE_ASSERT_TRUE(this->inputs.size() >= 2U, "conv2d inputs num [%u] < 2", this->inputs.size());
    }
    GE_CHK_STATUS(AppendCubeInputArgs(ss, this->inputs, options), "Append cube input arguments failed");
    AppendCubeOutputArgs(ss, this->outputs, options.is_cv_fuse ? "nullptr" : "output_0");
  }
  ss << this->workspace_arg.Str() << ", ";
  ss << "gm_tiling_data";
  if (options.is_cv_fuse) {
    ss << (options.is_dynamic && options.is_db ? ", &CV_FUSION_ADDR_DB" : ", &CV_FUSION_ADDR");
  }
  ss << ");" << std::endl;
  return ss.str();
}

std::string Kernel::GenCubeCommonTilingSingleFuncCall(const ascir::ImplGraph &impl_graph,
                                                      const std::string &output_arg_override) const {
  const auto options = GetCubeTilingOptions(impl_graph);
  std::stringstream ss;
  GE_CHK_STATUS(GenCubeCommonTiling(ss, options.is_batch, options.is_conv2d), "GenCubeCommonTilingHead failed");
  if (this->use_list_tensor_) {
    ss << kInputTensorDescName << ", " << kOutputTensorDescName << ", ";
  } else {
    if (options.is_conv2d) {
      auto min_inputs_num =
          2U + (options.has_bias ? 1U : 0U) + (options.has_offset_w ? 1U : 0U) + (options.has_scale0 ? 1U : 0U);
      GE_ASSERT_TRUE(this->inputs.size() >= min_inputs_num, "conv2d inputs num [%u] < min_inputs_num [%u]",
                     this->inputs.size(), min_inputs_num);
    } else {
      auto min_inputs_num = 1U + (options.has_bias ? 1U : 0U) + (options.has_offset_w ? 1U : 0U);
      GE_ASSERT_TRUE(this->inputs.size() >= min_inputs_num, "cube inputs num [%u] < min_inputs_num [%u]",
                     this->inputs.size(), min_inputs_num);
    }
    GE_CHK_STATUS(AppendCubeInputArgs(ss, this->inputs, options), "Append cube input arguments failed");
    AppendCubeOutputArgs(ss, this->outputs,
                         output_arg_override.empty() ? this->workspace_arg.Str() : output_arg_override);
  }
  ss << this->workspace_arg.Str();
  // cube workspace的位置需要计算 workspace + vector的偏移
  ss << " + vec_wss, gm_tiling_data);" << std::endl;
  return ss.str();
}

std::string Kernel::GenCubeTilingFuncCall(const ascir::ImplGraph &impl_graph, bool is_dynamic) const {
  auto options = GetCubeTilingOptions(impl_graph);
  options.is_dynamic = is_dynamic;
  std::stringstream ss;
  if (is_dynamic) {
    options.is_cv_fuse = true;
    ss << "if constexpr (UB_MODE == 0) {" << std::endl;
    ss << GenCubeTilingSingleFuncCall(options);
    ss << "} else {" << std::endl;
    options.is_db = true;
    ss << GenCubeTilingSingleFuncCall(options);
    ss << "}" << std::endl;
  } else {
    ss << "#ifdef CV_UB_FUSION" << std::endl;
    options.is_cv_fuse = true;
    ss << GenCubeTilingSingleFuncCall(options);
    ss << "#else" << std::endl;
    options.is_cv_fuse = false;
    ss << GenCubeTilingSingleFuncCall(options);
    ss << "#endif" << std::endl;
  }
  return ss.str();
}

Status Kernel::GenKernelFuncByTilingKey(const ascir::FusedScheduledResult &fused_schedule_result, std::stringstream &ss,
                                        bool use_list_tensor, const CodegenConfig &config,
                                        const std::string &kernel_task_type) {
  std::stringstream ss1;
  bool finished = false;
  GE_ASSERT_SUCCESS(AppendKernelFuncPrologueByTilingKey(fused_schedule_result, ss, ss1, use_list_tensor, config,
                                                        kernel_task_type, finished));
  if (finished) {
    return af::SUCCESS;
  }
  if (use_list_tensor) {
    ss1 << "  ListTensorDesc " << kInputTensorDescName << "((__gm__ void *)inputs);" << std::endl;
    ss1 << "  ListTensorDesc " << kOutputTensorDescName << "((__gm__ void *)outputs);" << std::endl;
  }

  auto is_dynamic = !IsStaticSchedResult(fused_schedule_result);
  // cv 兜底模板当前没有区分c、v的tiling_key，默认都走tiling_data解析
  if (ascgen_utils::IsCubeFusedScheduled(fused_schedule_result) && (config.is_inductor || is_dynamic)) {
    GE_ASSERT_SUCCESS(GenCVKernelFuncWithMulGroup(fused_schedule_result, config, ss, ss1, use_list_tensor));
  } else if (ascgen_utils::CanUseTilingKey(fused_schedule_result) && !config.is_inductor &&
             !ascgen_utils::IsCubeCommonFusedScheduled(fused_schedule_result)) {
    GE_ASSERT_SUCCESS(GenKernelFuncWithRegTilingKey(fused_schedule_result, config, ss, ss1, use_list_tensor));
  } else {
    GE_ASSERT_SUCCESS(GenKernelFuncWithParseTilingData(fused_schedule_result, config, ss, ss1, use_list_tensor));
  }
  ss1 << std::endl << "}" << std::endl;
  ss << ss1.str();
  return af::SUCCESS;
}

void Kernel::SetUseListTensor(bool use_list_tensor) {
  use_list_tensor_ = use_list_tensor;
}

void Kernel::SetUsingAttCalcQBTSizeConfig(bool using_att_calc_qbt_size) {
  this->tpipe.SetUsingAttCalcQBTSizeConfig(using_att_calc_qbt_size);
}

void Kernel::SetEnableParallelCompile(bool enable_parallel_compile) {
  enable_parallel_compile_ = enable_parallel_compile;
}

bool Kernel::GetEnableParallelCompile() const {
  return enable_parallel_compile_;
}

Status Kernel::InitL2CacheHintInfo(const ascir::FusedScheduledResult &fused_scheduled_result,
                                   const ascir::ImplGraph &graph) {
  GE_CHK_BOOL_RET_SPECIAL_STATUS(!tpipe.is_inductor, af::SUCCESS, "not inductor");
  const auto &sizes = fused_scheduled_result.gm_tensor_sizes;
  if (sizes.input_sizes.empty() || sizes.output_sizes.empty()) {
    return af::SUCCESS;
  }
  GE_ASSERT_SUCCESS(optimize::L2CacheHintManager::GetL2Size(l2_size_), "Get l2_size failed");
  if (sizes.total_size.IsConstExpr()) {
    int64_t total_size_value = 0;
    GE_ASSERT_TRUE(sizes.total_size.GetConstValue(total_size_value));
    GE_CHK_BOOL_RET_SPECIAL_STATUS(total_size_value <= l2_size_, af::SUCCESS,
                                   "InitL2CacheHintInfo skip: total_size[%lld] <= l2_size[%lld].", total_size_value,
                                   l2_size_);
  }
  const auto symbol_absent = !optimize::L2CacheHintManager::AllExprSymbolsInGraph(sizes, graph);
  GE_CHK_BOOL_RET_SPECIAL_STATUS(symbol_absent, af::SUCCESS,
                                 "InitL2CacheHintInfo skip: expressions contain symbols not in current graph: %s.",
                                 graph.GetName().c_str());
  gm_tensor_sizes_ = &fused_scheduled_result.gm_tensor_sizes;
  skip_l2_cache_hint_input_indices_ = optimize::L2CacheHintManager::CollectSkipL2CacheHintIndices(graph);
  GELOGD("InitL2CacheHintInfo success");
  return af::SUCCESS;
}

void Kernel::AppendFuncCall(std::stringstream &ss, std::vector<std::vector<std::string>>::const_iterator begin,
                            std::vector<std::vector<std::string>>::const_iterator end) {
  for (auto it = begin; it != end; ++it) {
    for (const auto &call_statement : *it) {
      ss << call_statement;
    }
    ss << std::endl;
  }
}

void Kernel::AppendFuncCall(std::stringstream &ss, std::vector<std::vector<TilingFuncCall>> &per_group_func_calls,
                            std::vector<TilingFuncCall> &current, size_t depth, uint32_t &tiling_key, bool is_cube) {
  if (depth == per_group_func_calls.size()) {
    if (!is_cube) {
      ss << (tiling_key == 0U ? "  if " : " else if ") << "(TILING_KEY_IS(" << std::to_string(tiling_key) << ")) {"
         << std::endl;
    }
    for (const auto &tiling_func_call : current) {
      ss << "    " << tiling_func_call.func_call_ << std::endl;
      if (tiling_func_call.need_sync_all_) {
        ss << "    SyncAll();" << std::endl;
      }
    }
    if (!is_cube) {
      ss << "  }";
    }
    tiling_key++;
    return;
  }
  for (const auto &func_call : per_group_func_calls[depth]) {
    current.push_back(func_call);
    AppendFuncCall(ss, per_group_func_calls, current, depth + 1, tiling_key, is_cube);
    current.pop_back();
  }
}

std::vector<Variable> Kernel::PackingFuncArgs(const std::string &tiling_data_type,
                                              const ascir::FusedScheduledResult &fused_schedule_result,
                                              bool use_list_tensor) {
  std::vector<Variable> args;
  if (use_list_tensor) {
    args.emplace_back(Type("ListTensorDesc&"), kInputTensorDescName);
    args.emplace_back(Type("ListTensorDesc&"), kOutputTensorDescName);
  } else {
    for (auto &input : fused_schedule_result.input_nodes) {
      args.emplace_back(GM_ADDR(GenValidName(input->GetName())));
    }
    for (auto &output : fused_schedule_result.output_nodes) {
      args.emplace_back(GM_ADDR(GenValidName(output->GetName())));
    }
  }
  args.emplace_back(GM_ADDR("workspace"));
  args.emplace_back(Type(tiling_data_type + "&"), "t");
  return args;
}

std::vector<std::string> Kernel::GenPackingFunctions(std::stringstream &ss_define,
                                                     const std::vector<Variable> &kernel_args,
                                                     const std::vector<std::vector<std::string>> &per_group_func_calls,
                                                     int64_t max_group_per_compile_unit, uint32_t &function_id,
                                                     bool support_parallel_compile) {
  std::vector<std::string> func_names;
  auto remaining_groups = static_cast<int64_t>(per_group_func_calls.size());
  auto begin = per_group_func_calls.cbegin();
  while (remaining_groups > 0) {
    const auto num = std::min(remaining_groups, max_group_per_compile_unit);
    const auto end = begin + num;
    // 函数名需要以数字结尾, 但不能与tiling_key相同, 否则rts加载kernel会失败
    const auto &func_name = "packed_functions_8" + std::to_string(function_id);
    if (support_parallel_compile) {
      // codegen packing func definition
      ss_define << PackingFuncDeclare(func_name, kernel_args) << ";" << std::endl;
      ss_define << "#if TILING_KEY_VAR == " << function_id << std::endl;
    }
    ss_define << PackingFuncDeclare(func_name, kernel_args) << "{" << std::endl;
    AppendFuncCall(ss_define, begin, end);
    ss_define << "}" << std::endl;
    if (support_parallel_compile) {
      ss_define << "#endif" << std::endl;
    }
    remaining_groups -= num;
    begin += num;
    function_id += 1;
    func_names.emplace_back(func_name);
  }
  return func_names;
}

std::string Kernel::PackingFuncDeclare(const std::string &func_name, const std::vector<Variable> &kernel_args) {
  std::stringstream ss;
  ss << "extern \"C\" __aicore__ void ";
  ss << func_name << "(";
  std::vector<std::string> args;
  args.reserve(kernel_args.size());
  for (const auto &arg : kernel_args) {
    args.emplace_back(arg.AsArg());
  }
  ss << af::StringUtils::Join(args.cbegin(), args.cend(), ", ");
  ss << ")";
  return ss.str();
}

void Kernel::GenPackingFunctionCalls(stringstream &ss, const vector<Variable> &kernel_args,
                                     const vector<std::string> &func_names) {
  std::vector<std::string> args;
  args.reserve(kernel_args.size());
  for (const auto &arg : kernel_args) {
    args.emplace_back(arg.Str());
  }
  bool need_sync = false;
  const auto &func_args = af::StringUtils::Join(args.cbegin(), args.cend(), ", ");
  for (const auto &func_name : func_names) {
    if (need_sync) {
      ss << "    AscendC::PipeBarrier<PIPE_ALL>();" << std::endl;
    }
    std::string function_id = func_name.substr(func_name.rfind("_") + 1);
    ss << "    " << func_name << "(" << func_args << ");" << std::endl;
    need_sync = true;
  }
}

void Kernel::FakeTilingIds(stringstream &ss, uint32_t function_id_end) {
  if (function_id_end != kFuncIdBegin) {
    ss << "inline void fake_tiling_ids() {" << std::endl;
    ss << "  int32_t g_tilingKey = -1;" << std::endl;
    ss << "  if (TILING_KEY_IS(0)) {}" << std::endl;
    for (uint32_t function_id = kFuncIdBegin; function_id < function_id_end; ++function_id) {
      ss << "  if (TILING_KEY_IS(" << function_id << ")) {}" << std::endl;
    }
    ss << "}" << std::endl;
  }
}

Status Kernel::GenerateMacro(stringstream &ss) {
  std::stack<const Loop *> loop_stack;
  loop_stack.push(&(this->root_loop));
  while (!loop_stack.empty()) {
    const Loop *current_loop = loop_stack.top();
    loop_stack.pop();

    for (auto &body : current_loop->bodys) {
      if (body.type == LoopType::LOOP) {
        GE_ASSERT_NOTNULL(body.loop);
        loop_stack.push(body.loop);
      } else if (body.call->unit == af::ComputeUnit::kUnitCube) {
        string macro_result;
        body.call->GenerateMacro(macro_result);
        ss << macro_result << std::endl;
        break;
      }
    }
  }
  return af::SUCCESS;
}

Status Kernel::GenerateKernelByNode(const ascir::ImplGraph &graph, stringstream &ss,
                                    std::unordered_set<const std::string *> &kernel_file_ptr, bool is_dynamic) {
  GE_CHK_STATUS_RET(GenerateMacro(ss), "Generate Macro failed");
  std::string npu_arch;
  GE_ASSERT_SUCCESS(ge::PlatformContext::GetInstance().GetCurrentPlatformString(npu_arch));
  const bool need_marco = (npu_arch == "3510" || npu_arch == "9201" || npu_arch == "9202");
  if (need_marco) {
    ss << "#if defined(__DAV_C310__) || "
          "(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102 || __NPU_ARCH__ == 3510 || __NPU_ARCH__ == 9201 || "
          "__NPU_ARCH__ == 9202))"
       << std::endl;
  }
  const std::string kMatMulTilingKeyDynamic = "mat_mul_tiling_key_dynamic.h";
  const std::string kBatchMatMulV3TilingKeyDynamic = "batch_mat_mul_v3_tiling_key_dynamic.h";
  for (const auto &node : graph.GetAllNodes()) {
    auto impl = ascgen_utils::GetAscIrCodegenImpl(node->GetType());
    GE_ASSERT_NOTNULL(impl, "GetAscIrCodegenImpl of node %s[%s] is null", node->GetTypePtr(), node->GetNamePtr());
    for (const auto &header_str : impl->LoadApiHeaderFiles(is_dynamic)) {
      const auto &file = AscendCApiRegistry::GetInstance().GetFileContent(header_str);
      if (file.empty() || kernel_file_ptr.find(&(file)) != kernel_file_ptr.end()) {
        continue;
      }
      kernel_file_ptr.insert(&(file));
      const bool is_matmul_tiling_key =
          header_str == kMatMulTilingKeyDynamic || header_str == kBatchMatMulV3TilingKeyDynamic;
      if (is_matmul_tiling_key) {
        ss << "#ifndef ASCENDC_TPL_KERNEL" << std::endl;
        ss << "#define ASCENDC_TPL_KERNEL" << std::endl;
        ss << "#define AF_UNDEF_ASCENDC_TPL_KERNEL" << std::endl;
        ss << "#endif" << std::endl;
      }
      ss << file;
      if (is_matmul_tiling_key) {
        ss << "#ifdef AF_UNDEF_ASCENDC_TPL_KERNEL" << std::endl;
        ss << "#undef ASCENDC_TPL_KERNEL" << std::endl;
        ss << "#undef AF_UNDEF_ASCENDC_TPL_KERNEL" << std::endl;
        ss << "#endif" << std::endl;
      }
    }
  }
  if (need_marco) {
    ss << "#endif" << std::endl;
  }
  return af::SUCCESS;
}

Status Kernel::GlobalTensorDefine(std::string &result) const {
  std::stringstream ss;
  for (std::size_t i = 0; i < this->inputs.size(); i++) {
    const auto &tensor = this->tpipe.tensors.find(this->input_tensors[i]);
    GE_ASSERT_TRUE((tensor != this->tpipe.tensors.end()), "Codegen input tensor id[%ld] not found",
                   this->input_tensors[i]);
    if (tensor->second.is_constant) {
      continue;
    }
    ss << "    " << tensor->second.Define() << std::endl;
  }

  for (std::size_t i = 0; i < this->outputs.size(); i++) {
    const auto &tensor = this->tpipe.tensors.find(this->output_tensors[i]);
    GE_ASSERT_TRUE((tensor != this->tpipe.tensors.end()), "Codegen output tensor id[%ld] not found",
                   this->output_tensors[i]);
    ss << "    " << tensor->second.Define() << std::endl;
  }

  for (std::size_t i = 0; i < this->constant_tensors.size(); i++) {
    auto tensor = this->tpipe.tensors.find(this->constant_tensors[i]);
    GE_ASSERT_TRUE((tensor != this->tpipe.tensors.end()), "Codegen concat tensor id[%ld] not found",
                   this->constant_tensors[i]);
    GELOGI("const_value_expr: %s", tensor->second.const_value_expr.Str().get());

    string const_value = tensor->second.const_value.empty() ? tiler.Size(tensor->second.const_value_expr, true)
                                                            : tensor->second.const_value;
    ss << "    " << tensor->second.DefineConst(const_value.c_str()) << std::endl;
    GELOGI("Define ss value: %s", ss.str().c_str());
  }

  for (std::size_t i = 0; i < this->ub_scalar_tensors.size(); i++) {
    auto tensor = this->tpipe.tensors.find(this->ub_scalar_tensors[i]);
    GE_ASSERT_TRUE((tensor != this->tpipe.tensors.end()), "Codegen ub_scalar tensor id[%ld] not found",
                   this->ub_scalar_tensors[i]);

    std::string def_ub_scalar;
    GE_CHK_STATUS_RET(tensor->second.DefineUbScalar(def_ub_scalar));
    ss << "    " << def_ub_scalar;
    GELOGI("Define ub_scalar var: %s", def_ub_scalar.c_str());
  }
  ss << std::endl;
  result = ss.str();
  return af::SUCCESS;
}

Status Kernel::GlobalTensorAssign(std::string &result) const {
  std::stringstream ss;
  for (std::size_t i = 0; i < this->inputs.size(); i++) {
    const auto &tensor = this->tpipe.tensors.find(this->input_tensors[i]);
    GE_ASSERT_TRUE((tensor != this->tpipe.tensors.end()), "Codegen input tensor id[%ld] not found",
                   this->input_tensors[i]);
    if (tensor->second.is_constant) {
      continue;
    }
    std::string local_result;
    GE_CHK_STATUS_RET(tensor->second.SetGlobalBuffer(this->inputs[i], "", local_result),
                      "Codegen set global buffer failed");
    ss << local_result << std::endl;
  }

  for (std::size_t i = 0; i < this->outputs.size(); i++) {
    const auto &tensor = this->tpipe.tensors.find(this->output_tensors[i]);
    GE_ASSERT_TRUE((tensor != this->tpipe.tensors.end()), "Codegen output tensor id[%ld] not found",
                   this->output_tensors[i]);

    std::string local_result;
    GE_CHK_STATUS_RET(tensor->second.SetGlobalBuffer(this->outputs[i], "", local_result),
                      "Codegen set global buffer failed");
    ss << local_result << std::endl;
  }
  ss << std::endl;
  result = ss.str();
  return af::SUCCESS;
}

Status TPipe::GetCVFusionCubeOutputUBTensorIdAndQueId(const ascir::ImplGraph &graph) {
  for (auto node : graph.GetAllNodes()) {
    if (IsOps<Workspace>(node)) {
      for (auto &peer_input : node->outputs[0].anchor.GetPeerInDataAnchors()) {
        auto next_node = std::dynamic_pointer_cast<af::AscNode>(peer_input->GetOwnerNode());
        GE_ASSERT_NOTNULL(next_node, "Codegen CV Fusion get next node after workspace node failed");
        if (IsOps<Load>(next_node)) {
          this->cube_output_tensor_id = next_node->outputs[0].attr.mem.tensor_id;
          this->cube_output_que_id = next_node->outputs[0].attr.que.id;
          return af::SUCCESS;
        }
      }
      GELOGE(af::FAILED, "Codegen CV Fusion Load node next to Workspace not found");
      return af::FAILED;
    }
  }
  GELOGE(af::FAILED, "Codegen CV Fusion get Workspace node failed");
  return af::FAILED;
}

static void AddCommaIfNeeded(bool &is_first, std::stringstream &tensor_size_max) {
  if (is_first) {
    is_first = false;
  } else {
    tensor_size_max << ", ";
  }
}

Status TPipe::ParseTBufReuse(TBuf buf, std::string &reuse_dtype_name, bool &is_buf_reuse,
                             std::vector<const Tensor *> &reuse_buf_tensors, std::stringstream &tensor_size_max) const {
  tensor_size_max << KernelUtils::Max() << "(";

  bool is_first = true;
  for (auto mid : buf.merge_scopes) {
    auto merge_scope = this->merge_scopes.find(mid);
    if (merge_scope == this->merge_scopes.end()) {
      GELOGE(af::FAILED, "Codegen merge scope not found:%ld", mid);
      return af::FAILED;
    }
    AddCommaIfNeeded(is_first, tensor_size_max);
    tensor_size_max << merge_scope->second.size;
  }

  for (auto tid : buf.not_merge_tensors) {
    auto tensor = this->tensors.find(tid);
    if (tensor == this->tensors.end()) {
      GELOGE(af::FAILED, "Codegen tensor not found:%ld", tid);
      return af::FAILED;
    }
    AddCommaIfNeeded(is_first, tensor_size_max);
    std::string dtype_name;
    GE_CHK_STATUS_RET(Tensor::DtypeName(tensor->second.dtype, dtype_name), "Codegen get data type:%d failed",
                      static_cast<int32_t>(tensor->second.dtype));
    if (is_buf_reuse) {
      if (reuse_dtype_name == "") {
        reuse_dtype_name = dtype_name;
      } else {
        if (reuse_dtype_name != dtype_name) {
          is_buf_reuse = false;
        }
      }
    }
    tensor_size_max << tensor->second.size << " * sizeof(" << dtype_name << ")";
    reuse_buf_tensors.push_back(&tensor->second);
  }

  for (auto tmp_buf_size : buf.tmp_buf_size_list) {
    AddCommaIfNeeded(is_first, tensor_size_max);
    if (this->cv_fusion_type == ascir::CubeTemplateType::kUBFuse) {
      tensor_size_max << this->tiler.ActualSize(tmp_buf_size);
    } else {
      tensor_size_max << this->tiler.Size(tmp_buf_size);
    }
  }

  if (reuse_buf_tensors.size() == 0) {
    is_buf_reuse = false;
  }
  tensor_size_max << ")";
  return af::SUCCESS;
}

Status TPipe::LocalTensorDefine(std::string &result) const {
  stringstream ss;
  for (auto &pair : this->tensors) {
    auto &t = pair.second;
    if (t.alloc_type != af::AllocType::kAllocTypeGlobal) {
      ss << "    " << t.AsArg() << ";" << std::endl;
    }
  }
  ss << std::endl;
  result = ss.str();
  return af::SUCCESS;
}

std::string TPipe::TensorSizeDefine() const {
  stringstream ss;

  for (auto &pair : this->tensors) {
    auto &t = pair.second;
    if ((t.alloc_type == af::AllocType::kAllocTypeQueue) || (t.alloc_type == af::AllocType::kAllocTypeBuffer)) {
      ss << "    " << t.size.Define() << std::endl;
      ss << "    " << t.actual_size.Define() << std::endl;
    }
  }
  return ss.str();
}

Status TPipe::TensorSizeAssign(std::string dtype_name, std::string &result) const {
  stringstream ss;
  (void)dtype_name;

  for (auto &pair : this->tensors) {
    auto &t = pair.second;
    if ((t.alloc_type == af::AllocType::kAllocTypeQueue) || (t.alloc_type == af::AllocType::kAllocTypeBuffer)) {
      if (t.is_ub_scalar) {
        std::string tensor_dtype_name;
        GE_CHK_STATUS_RET(Tensor::DtypeName(t.dtype, tensor_dtype_name), "Codegen get data type:%d failed",
                          static_cast<int32_t>(t.dtype));
        ss << t.size.Str() << " = KernelUtils::BlkAlign<" << tensor_dtype_name << ">(1);" << std::endl;
      } else {
        // stage_size(元素个数)已归一: 主路径 = 求解器预算a0a1t(=行数*basen_align, 任意dtype的32B行对齐
        // 行宽均<=basen_align, 按此元素数分配即覆盖compute size且不超求解器预算);
        // 小a0a1t兜底时 = cube_ub元素数(容量, 更保守). CV与非CV统一使用stage_size.
        ss << t.size.Str() << " = stage_size;" << std::endl;
      }
    }
  }
  ss << std::endl;
  result = ss.str();
  return af::SUCCESS;
}

std::string TPipe::GenDuplicateBufDefine(
    const std::set<std::pair<std::string, std::string>> &pre_api_extract_dup) const {
  std::stringstream ss;
  int32_t i = 1;
  for (auto [const_val, const_dtype] : pre_api_extract_dup) {
    const std::string index_str = std::to_string(i);
    ss << "    TBuf<TPosition::VECCALC> builtin_tmp_buffer_" << index_str << ";" << std::endl;
    std::string local_tensor_name = "local_blk_tensor_of_" + const_dtype + "_" + const_val;
    ss << "    LocalTensor<" << const_dtype << "> " << local_tensor_name << ";" << std::endl;
    i++;
  }
  return ss.str();
}

std::string TPipe::GenDuplicateBufAssign(
    const std::set<std::pair<std::string, std::string>> &pre_api_extract_dup) const {
  std::stringstream ss;
  int32_t i = 1;
  for (auto [const_val, const_dtype] : pre_api_extract_dup) {
    const std::string index_str = std::to_string(i);
    if (using_global_tpipe_) {
      ss << "GetTPipePtr()->InitBuffer(builtin_tmp_buffer_" << index_str << ", ONE_BLK_SIZE);" << std::endl;
    } else {
      ss << "tpipe.InitBuffer(builtin_tmp_buffer_" << index_str << ", ONE_BLK_SIZE);" << std::endl;
    }
    ss << "LocalTensor<uint8_t> builtin_tmp_buf_" << index_str << " = builtin_tmp_buffer_" << index_str
       << ".Get<uint8_t>();" << std::endl;
    std::string local_tensor_name = "local_blk_tensor_of_" + const_dtype + "_" + const_val;
    ss << local_tensor_name << " = builtin_tmp_buf_" << index_str << ".template ReinterpretCast<" << const_dtype
       << ">();" << std::endl;
    if (const_dtype == "half" || const_dtype == "float" || const_dtype == "double") {
      const_val += ".0";
    }
    ss << "Duplicate(" << local_tensor_name << "[0], (" << const_dtype << ")" << const_val << ", ONE_BLK_SIZE / sizeof("
       << const_dtype << "));" << std::endl;
    i++;
  }
  return ss.str();
}

Status TPipe::BlkTensorDefine(std::string &result) const {
  // 遍历所有需要生成blk tensor 的 tensor id
  stringstream ss;
  for (auto &id : this->need_gen_blk_tensors) {
    auto tensor_ptr = this->GetTensor(id);
    GE_CHK_BOOL_RET_STATUS(tensor_ptr != nullptr, af::FAILED, "BlkTensorAllocAndInit need_gen_blk_tensors failed");
    std::string scalar_t_buf_name = tensor_ptr->name + "_tbuf";
    std::string scalar_local_blk_tensor_name = "local_blk_tensor_of_" + tensor_ptr->name;
    ss << "    TBuf<TPosition::VECCALC> " << scalar_t_buf_name << ";" << std::endl;
    ss << "    LocalTensor<" << tensor_ptr->type << "> " << scalar_local_blk_tensor_name << ";" << std::endl;
  }
  ss << std::endl;
  result = ss.str();
  return af::SUCCESS;
}

Status TPipe::BlkTensorAssign(std::string &result) const {
  // 遍历所有需要生成blk tensor 的 tensor id
  stringstream ss;
  for (auto &id : this->need_gen_blk_tensors) {
    auto tensor_ptr = this->GetTensor(id);
    GE_CHK_BOOL_RET_STATUS(tensor_ptr != nullptr, af::FAILED, "BlkTensorAllocAndInit need_gen_blk_tensors failed");
    std::string scalar_t_buf_name = tensor_ptr->name + "_tbuf";
    std::string scalar_local_blk_tensor_name = "local_blk_tensor_of_" + tensor_ptr->name;
    if (using_global_tpipe_) {
      ss << "GetTPipePtr()->InitBuffer(" << scalar_t_buf_name << ", 32);" << std::endl;
    } else {
      ss << "tpipe.InitBuffer(" << scalar_t_buf_name << ", 32);" << std::endl;
    }
    ss << scalar_local_blk_tensor_name << " = " << scalar_t_buf_name << ".Get<" << tensor_ptr->type << ">();"
       << std::endl;

    ss << "Duplicate(" << scalar_local_blk_tensor_name << "[0], static_cast<" << tensor_ptr->type << ">("
       << tensor_ptr->const_value << "), static_cast<uint64_t>(32/"
       << "sizeof(" << tensor_ptr->type << ")));" << std::endl;
    ss << "AscendC::PipeBarrier<PIPE_V>();" << std::endl;
  }
  ss << std::endl;
  result = ss.str();
  return af::SUCCESS;
}

Status Kernel::GenerateVecFuncOfCVFusion(std::stringstream &result, bool vector_no_db_flag, bool is_conv2d,
                                         bool is_dynamic, bool is_inductor) {
  this->tpipe.is_inductor = is_inductor;
  std::string tiling_data_type = "AutofuseTilingData";
  if (vector_no_db_flag) {
    if (is_conv2d) {
      result << R"(
// conv2d
#ifndef INDUCTOR_CV_FUSION
#include "autofuse_cube_tiling_data.h"
#endif
)" << std::endl;
    } else {
      result << R"(
#include "cmct/block/block_scheduler_policy.h"
#include "cmct/block/block_scheduler_utils.h"
#include "cmct/utils/status_utils.h"
#ifndef INDUCTOR_CV_FUSION
#include "autofuse_cube_tiling_data.h"
#endif
)" << std::endl;
    }
    if (!is_dynamic) {
      result << "#ifdef CV_UB_NO_DB" << std::endl;
    }
  } else {
    if (!is_dynamic) {
      result << "#ifdef CV_UB_DB" << std::endl;
    }
  }
  if (is_dynamic && !vector_no_db_flag) {
    result << R"(
class AutoFusionVectorDB {
  public:
    __aicore__ inline AutoFusionVectorDB() {};
)" << std::endl;
  } else {
    result << R"(
class AutoFusionVector {
  public:
    __aicore__ inline AutoFusionVector() {};
)" << std::endl;
  }

  result << "    struct Arguments {" << std::endl;
  for (auto &input : this->inputs) {
    const bool is_gm_input = input.type.name == kGmAddrT.name;
    result << "     " << input.AsArg() << (is_gm_input ? "{nullptr};" : "{0};") << std::endl;
  }
  for (auto &output : this->outputs) {
    result << "     " << output.AsArg() << "{nullptr};" << std::endl;
  }
  if (is_dynamic) {
    if (is_inductor) {
      result << "     uint32_t stage_size_name{0};" << std::endl;
      result << "     uint32_t cube_ub_stage_size{0};" << std::endl;
      result << "     uint32_t cube_m_stage_size{0};" << std::endl;
    } else {
      result << "     const CVAutofuseTilingData *cv_tiling_data{nullptr};" << std::endl;
    }
  }
  result << "    };" << std::endl << std::endl;

  result << "    struct Params {" << std::endl;
  for (auto &input : this->inputs) {
    const bool is_gm_input = input.type.name == kGmAddrT.name;
    result << "     " << input.AsArg() << (is_gm_input ? "{nullptr};" : "{0};") << std::endl;
  }
  for (auto &output : this->outputs) {
    result << "     " << output.AsArg() << "{nullptr};" << std::endl;
  }
  if (is_dynamic) {
    if (is_inductor) {
      result << "     uint32_t stage_size_name{0};" << std::endl;
      result << "     uint32_t cube_ub_stage_size{0};" << std::endl;
      result << "     uint32_t cube_m_stage_size{0};" << std::endl;
    } else {
      result << "     const CVAutofuseTilingData *cv_tiling_data{nullptr};" << std::endl;
    }
  }
  result << "    };" << std::endl;
  for (auto &input : this->inputs) {
    const bool is_gm_input = input.type.name == kGmAddrT.name;
    result << "    " << input.AsArg() << (is_gm_input ? "{nullptr};" : "{0};") << std::endl;
  }
  for (auto &output : this->outputs) {
    result << "    " << output.AsArg() << "{nullptr};" << std::endl;
  }
  if (is_dynamic) {
    if (is_inductor) {
      result << "     uint32_t stage_size_name{0};" << std::endl;
      result << "     uint32_t cube_ub_stage_size{0};" << std::endl;
      result << "     uint32_t cube_m_stage_size{0};" << std::endl;
    } else {
      result << "     const CVAutofuseTilingData *cv_tiling_data{nullptr};" << std::endl;
    }
  }
  result << std::endl;
  auto ub_tensor = this->tpipe.GetTensor(this->tpipe.cube_output_tensor_id);
  GE_CHK_BOOL_RET_STATUS(ub_tensor != nullptr, af::FAILED,
                         "Codegen CV Fusion MatmulOutput UB tensor id[%ld] "
                         "not found",
                         this->tpipe.cube_output_tensor_id);
  std::string dtype_name;
  GE_CHK_STATUS_RET(Tensor::DtypeName(ub_tensor->dtype, dtype_name), "data type:%d failed",
                    static_cast<int32_t>(ub_tensor->dtype));
  std::string tmp;
  GE_CHK_STATUS_RET(this->tpipe.LocalTensorDefine(tmp), "Local tbuf define failed");
  result << tmp;
  if (!is_conv2d) {
    result << "    TPipe tpipe;" << std::endl << std::endl;
  } else {
    this->tpipe.SetUsingGlobalTpipe(true);
  }
  for (auto &[id, que] : this->tpipe.ques) {
    if (id == this->tpipe.cube_output_que_id) {
      continue;
    }
    result << "    " << que.Define() << std::endl;
    result << "    " << que.buf.AsArg() << ";" << std::endl;
  }
  result << std::endl;
  for (auto &pair : this->tpipe.bufs) {
    auto &buf = pair.second;
    result << "    " << buf.Define() << std::endl;
    result << "    " << buf.buf.AsArg() << ";" << std::endl;
    if (buf.tmp_buf_reuse) {
      result << "    " << this->tpipe.tmp_buf.AsArg() << "_" << to_string(buf.id) << ";" << std::endl;
    }
  }
  result << "    TBuf<TPosition::VECCALC> buf_cube;" << std::endl;

  if (!this->pre_api_extract_dup.empty()) {
    result << this->tpipe.GenDuplicateBufDefine(this->pre_api_extract_dup) << std::endl;
  }

  GE_CHK_STATUS_RET(this->tpipe.BlkTensorDefine(tmp), "Block tensor define failed");
  result << tmp;

  GE_CHK_STATUS_RET(this->GlobalTensorDefine(tmp), "Global tensor define in cv-ub-fuse case failed");
  result << tmp;

  result << this->tpipe.TensorSizeDefine() << std::endl;
  result << "    LocalTensor<" << dtype_name << "> cLocal_;" << std::endl;
  if (is_conv2d) {
    // conv2d
    result << "__aicore__ inline void Init(Params const& params, AscendC::LocalTensor<" << dtype_name
           << ">& cLocal, int64_t &stage_size1, int64_t &stage_size2) {";
    result << std::endl;
    result << "GET_TILING_DATA_WITH_STRUCT(Conv2DTilingData, tmpTilingData, tmpTilingGM);" << std::endl;
    result << "const int32_t ub_align_value = 32 / sizeof(" << dtype_name << ");" << std::endl;
    result << "const int32_t basen_align = (tmpTilingData.hoL0 + ub_align_value - 1) / "
              "ub_align_value * ub_align_value;"
           << std::endl;
    std::string npu_arch;
    int32_t vec_num = 2;
    GE_ASSERT_SUCCESS(ge::PlatformContext::GetInstance().GetCurrentPlatformString(npu_arch));
    if (npu_arch == "5102") {
      vec_num = 1;
    }
    result << "stage_size1 = KernelUtils::Max(tmpTilingData.nL0 / " << vec_num << ", 16) * basen_align;" << std::endl;
    if (is_dynamic && is_inductor) {
      result << "#ifdef INDUCTOR_CONST_TILING_DATA" << std::endl;
      result << "const uint32_t stage_size_name = kConstTilingData.tiling_data.STAGE_SIZE_NAME;" << std::endl;
      result << "#else" << std::endl;
      result << "const uint32_t stage_size_name = params.stage_size_name;" << std::endl;
      result << "#endif" << std::endl;
    } else {
      if (is_dynamic) {
        result << "GET_TILING_DATA_WITH_STRUCT(AutofuseTilingData, autofuse_tiling_size, params.cv_tiling_data);"
               << std::endl;
      } else {
        result << "AutofuseTilingData autofuse_tiling_size;" << std::endl;
      }
      result << "const uint32_t stage_size_name = autofuse_tiling_size.STAGE_SIZE_NAME;" << std::endl;
    }
    result << "const int32_t compute_size = stage_size_name > 144 ? stage_size_name : 144;" << std::endl;
    result << "int32_t stage_size = compute_size;" << std::endl;
    result << "stage_size2 = compute_size;" << std::endl;
  } else {
    result << "__aicore__ inline void Init(Params const& params, AscendC::LocalTensor<" << dtype_name
           << ">& cLocal, int64_t l1M, int64_t l1NAlign, int64_t ubOffset, int64_t &stage_size_type) {";
    result << std::endl;
    if (is_dynamic && is_inductor) {
      result << "#ifdef INDUCTOR_CONST_TILING_DATA" << std::endl;
      result << "const AutofuseTilingData& autofuse_tiling_size = kConstTilingData.tiling_data;" << std::endl;
      result << "const uint32_t stage_size_name = autofuse_tiling_size.STAGE_SIZE_NAME;" << std::endl;
      result << "const int32_t basen_basem_align = kConstTilingData.cube_ub_stage_size * sizeof(" << dtype_name << ");"
             << std::endl;
      result << "stage_size_type = static_cast<int64_t>(kConstTilingData.cube_m_stage_size);" << std::endl;
      result << "#else" << std::endl;
      result << "const uint32_t stage_size_name = params.stage_size_name;" << std::endl;
      result << "const int32_t basen_basem_align = params.cube_ub_stage_size * sizeof(" << dtype_name << ");"
             << std::endl;
      result << "stage_size_type = static_cast<int64_t>(params.cube_m_stage_size);" << std::endl;
      result << "#endif" << std::endl;
    } else {
      result << "GET_TILING_DATA_WITH_STRUCT(MatMulV3BasicTilingData, tmpTilingData, tmpTilingGM);" << std::endl;
      result << "const int32_t ub_align_value = 32;" << std::endl;
      result
          << "const int32_t basen_align = (tmpTilingData.baseN + ub_align_value - 1) / ub_align_value * ub_align_value;"
          << std::endl;
      result << "const int32_t basen_basem_align = (tmpTilingData.baseM * basen_align * sizeof(" << dtype_name
             << ")) / 2 + basen_align * sizeof(" << dtype_name << ");" << std::endl;
      // 非inductor: ascbc静态流程在device头烧录kConstTilingData(求解器完整结果);
      // 动态shape时params携带cv_tiling_data指针读GM tiling; TF等前端在线GE编译的
      // 静态图两者皆无, 回退到autofuse_tiling_data.h中烧录的求解器结果
      // (z0z1t预算), 行数按 预算/上确界行宽 就地推导, 与conv2d分支同一模式
      result << "#ifdef INDUCTOR_CONST_TILING_DATA" << std::endl;
      result << "stage_size_type = static_cast<int64_t>(kConstTilingData.cube_m_stage_size);" << std::endl;
      result << "const uint32_t stage_size_name = kConstTilingData.stage_size_name;" << std::endl;
      result << "#else" << std::endl;
      if (is_dynamic) {
        result << "stage_size_type = static_cast<int64_t>(params.cv_tiling_data->cube_m_stage_size);" << std::endl;
        result << "const uint32_t stage_size_name = params.cv_tiling_data->stage_size_name;" << std::endl;
      } else {
        result << "AutofuseTilingData autofuse_tiling_size;" << std::endl;
        result << "const uint32_t stage_size_name = autofuse_tiling_size.STAGE_SIZE_NAME;" << std::endl;
        result << "stage_size_type = static_cast<int64_t>(stage_size_name / basen_align);" << std::endl;
      }
      result << "#endif" << std::endl;
    }
    // stage_size 统一为元素个数: 求解器预算 a0a1t, 过小时兜底到 C fragment 容量(cube_ub元素数)
    result << "int32_t stage_size = stage_size_name > 144 ? stage_size_name : " << std::endl;
    result << "static_cast<int32_t>(basen_basem_align / sizeof(" << dtype_name << "));" << std::endl;
  }
  GE_CHK_STATUS_RET(this->root_loop.ActualSizeDefine(this->tiler, this->tpipe, dtype_name, tmp),
                    "actual size define failed");
  result << tmp;
  if (is_conv2d) {
    result << ub_tensor->Str() << "_actual_size =  stage_size2;" << std::endl << std::endl;
  } else {
    result << ub_tensor->Str() << "_actual_size = stage_size;" << std::endl << std::endl;
  }
  GE_CHK_STATUS_RET(this->tpipe.TensorSizeAssign(dtype_name, tmp), "Tensor size assign failed");
  result << tmp;

  for (auto &input : this->inputs) {
    result << input << " = params." << input << ";" << std::endl;
  }
  for (auto &output : this->outputs) {
    result << output << " = params." << output << ";" << std::endl;
  }
  GE_CHK_STATUS_RET(this->GlobalTensorAssign(tmp), "Global tensor assign in cv-ub-fuse case failed");
  result << tmp;
  if (is_conv2d) {
    result << "GetTPipePtr()->InitBuffer(buf_cube, stage_size1 *  sizeof(" << dtype_name << "));" << std::endl;
  } else {
    // 行语义下 epilogue 读取 cLocal_ 上界 stageOffset+stageSize <= inputSize <= cube_ub,
    // buf_cube 只需 C fragment 容量 basen_basem_align, 不再按 stage_size 撑大
    result << "tpipe.InitBuffer(buf_cube, static_cast<uint32_t>(basen_basem_align));" << std::endl;
  }
  result << ub_tensor->name << " = buf_cube.Get<" << dtype_name << ">();" << std::endl;
  result << "cLocal = " << ub_tensor->name << ";" << std::endl << std::endl;
  result << "cLocal_ = " << ub_tensor->name << ";" << std::endl << std::endl;

  GE_CHK_STATUS_RET(this->tpipe.LocalTBufAllocLoopTwice(tmp, false), "Local tbuf define failed");
  result << tmp;

  if (!this->pre_api_extract_dup.empty()) {
    result << this->tpipe.GenDuplicateBufAssign(this->pre_api_extract_dup) << std::endl;
  }

  GE_CHK_STATUS_RET(this->tpipe.BlkTensorAssign(tmp), "Block tensor assign failed");
  result << tmp;

  GE_CHK_STATUS_RET(this->tpipe.LocalTQueAlloc(tmp), "Codegen alloc local tque failed");
  result << tmp;
  result << "}" << std::endl << std::endl;

  stringstream ss;
  GE_ASSERT_SUCCESS(this->GenerateSubGraphFuncDef(&(this->root_loop), ss));
  result << ss.str() << std::endl;
  if (is_conv2d) {
    result << "inline __aicore__ void auto_fusion_vector_stage1(int64_t offset, int64_t offsetH, int64_t offsetW, "
              "int64_t offsetCout, int64_t curAivN, int64_t curAlignN, int64_t shapeN, "
              "int64_t curAivM, int64_t shapeM, int64_t stageSize, int64_t stageOffset) {";
    result << "}" << std::endl;
    result << "inline __aicore__ void auto_fusion_vector_stage2(int64_t offset, int64_t offsetH, int64_t offsetW, "
              "int64_t offsetCout, int64_t curAivN, int64_t curAlignN, int64_t shapeN, "
              "int64_t curAivM, int64_t shapeM, int64_t stageSize, int64_t stageOffset) {";
  } else {
    result << "inline __aicore__ void auto_fusion_vector_stage1(int64_t offset, int64_t curAivM, int64_t curAivN, "
              "int64_t shapeN, int64_t shapeM, int64_t curAlignN, int64_t stageSize) {";
  }
  result << std::endl;
  result << "int64_t batch_num = offset / shapeN / shapeM;" << std::endl;
  if (is_conv2d) {
    result << "int64_t load_block_len = curAivN;" << std::endl;
    result << "int64_t load_src_stride = shapeN - curAivN;" << std::endl;
    result << "int64_t load_dst_stride = curAlignN - curAivN;" << std::endl;
  } else {
    result << "int64_t load_block_len = curAlignN;" << std::endl;
    result << "int64_t load_src_stride = shapeN - curAlignN;" << std::endl;
    result << "int64_t load_dst_stride = 0;" << std::endl;
    result << "if (shapeN < curAlignN) {" << std::endl;
    result << "load_block_len = curAivN;" << std::endl;
    result << "load_src_stride = 0;" << std::endl;
    result << "load_dst_stride = curAlignN - shapeN;" << std::endl;
    result << "}" << std::endl;
  }

  GE_CHK_STATUS_RET(this->root_loop.Generate(this->tiler, this->tpipe, tmp, ComputeStage::kCVFuseStage1),
                    "Codegen root loop Generate failed");
  result << tmp;
  result << "}" << std::endl;

  if (is_conv2d) {
    result << "inline __aicore__ void auto_fusion_vector_stage3(int64_t offset, int64_t offsetH, int64_t offsetW, "
              "int64_t offsetCout, int64_t curAivN, int64_t curAlignN, int64_t shapeN, "
              "int64_t curAivM, int64_t shapeM, int64_t stageSize, int64_t stageOffset) {";
  } else {
    result << "inline __aicore__ void auto_fusion_vector_stage2(int64_t offset, int64_t curAivM, int64_t curAivN, "
              "int64_t shapeN, int64_t shapeM, int64_t curAlignN, int64_t stageSize) {";
  }
  result << std::endl;
  result << "int64_t batch_num = offset / shapeN / shapeM;" << std::endl;
  result << "int64_t load_block_len = curAlignN;" << std::endl;
  result << "int64_t load_src_stride = shapeN - curAlignN;" << std::endl;
  result << "int64_t load_dst_stride = 0;" << std::endl;
  result << "if (shapeN < curAlignN) {" << std::endl;
  result << "load_block_len = curAivN;" << std::endl;
  result << "load_src_stride = 0;" << std::endl;
  result << "load_dst_stride = curAlignN - shapeN;" << std::endl;
  result << "}" << std::endl;
  GE_CHK_STATUS_RET(this->root_loop.Generate(this->tiler, this->tpipe, tmp, ComputeStage::kCVFuseStage2),
                    "Codegen root loop Generate failed");
  result << tmp;
  result << "}" << std::endl;

  if (is_conv2d) {
    result << "inline __aicore__ void operator()(int64_t offset, int64_t offsetH, int64_t offsetW, int64_t "
              "offsetCout, int64_t curAivN, int64_t curAlignN, int64_t shapeN, "
              "int64_t curAivM, int64_t shapeM, int64_t stageSize, int64_t stageOffset, uint8_t stage = 0) {"
           << std::endl
           << ub_tensor->name << " = cLocal_[stageOffset].template ReinterpretCast<" << dtype_name << ">();"
           << std::endl
           << "if (stage == 1) {" << std::endl
           << "  auto_fusion_vector_stage1(offset, offsetH, offsetW, offsetCout, curAivN, curAlignN, shapeN, curAivM, "
              "shapeM, stageSize, stageOffset);"
           << std::endl
           << "} else if (stage == 2) {" << std::endl
           << "  auto_fusion_vector_stage2(offset, offsetH, offsetW, offsetCout, curAivN, curAlignN, shapeN, curAivM, "
              "shapeM, stageSize, stageOffset);"
           << std::endl
           << "} else if (stage == 3) {" << std::endl
           << "  auto_fusion_vector_stage3(offset, offsetH, offsetW, offsetCout, curAivN, curAlignN, shapeN, curAivM, "
              "shapeM, stageSize, stageOffset);"
           << std::endl
           << "} else {" << std::endl
           << "  auto_fusion_vector_stage1(offset, offsetH, offsetW, offsetCout, curAivN, curAlignN, shapeN, curAivM, "
              "shapeM, stageSize, stageOffset);"
           << std::endl
           << "  auto_fusion_vector_stage2(offset, offsetH, offsetW, offsetCout, curAivN, curAlignN, shapeN, curAivM, "
              "shapeM, stageSize, stageOffset);"
           << std::endl
           << "  auto_fusion_vector_stage3(offset, offsetH, offsetW, offsetCout, curAivN, curAlignN, shapeN, curAivM, "
              "shapeM, stageSize, stageOffset);"
           << std::endl
           << "}" << std::endl
           << "}" << std::endl;
  } else {
    result << "inline __aicore__ void operator()(int64_t offset, int64_t curAivM, int64_t curAivN, int64_t shapeN, "
              "int64_t shapeM, int64_t curAlignN, int64_t stageSize, int64_t stageOffset, uint8_t stage = 0) {"
           << std::endl
           << ub_tensor->name << " = cLocal_[stageOffset].template ReinterpretCast<" << dtype_name << ">();"
           << std::endl
           << "if (stage == 1) {" << std::endl
           << "  auto_fusion_vector_stage1(offset, curAivM, curAivN, shapeN, shapeM, curAlignN, stageSize);"
           << std::endl
           << "} else if (stage == 2) {" << std::endl
           << "  auto_fusion_vector_stage2(offset, curAivM, curAivN, shapeN, shapeM, curAlignN, stageSize);"
           << std::endl
           << "} else {" << std::endl
           << "  auto_fusion_vector_stage1(offset, curAivM, curAivN, shapeN, shapeM, curAlignN, stageSize);"
           << std::endl
           << "  auto_fusion_vector_stage2(offset, curAivM, curAivN, shapeN, shapeM, curAlignN, stageSize);"
           << std::endl
           << "}" << std::endl
           << "}" << std::endl;
  }
  result << "};" << std::endl;
  if (!is_dynamic) {
    result << "#endif" << std::endl;  // CV_UB_NO_DB/CV_UB_DB
  }
  return af::SUCCESS;
}

Status Kernel::InitCVFusionAddr(std::stringstream &result, bool vector_no_db_flag, bool is_dynamic, bool is_inductor) {
  if (vector_no_db_flag) {
    result << "  AutoFusionVector::Params CV_FUSION_ADDR;\n";
    if (is_dynamic && is_inductor) {
      result << "  if constexpr (UB_MODE == 0) {" << std::endl;
    }
    for (auto input : this->inputs) {
      result << "  " << (is_dynamic && is_inductor ? "  " : "") << "CV_FUSION_ADDR." << input.Str() << " = "
             << input.Str() << ";" << std::endl;
    }
    size_t output_idx = 0;
    for (auto output : this->outputs) {
      result << "  " << (is_dynamic && is_inductor ? "  " : "") << "CV_FUSION_ADDR." << output.Str() << " = "
             << output.Str() << ";" << std::endl;
      result << "  " << (is_dynamic && is_inductor ? "  " : "") << "GM_ADDR output_" << output_idx++ << " = "
             << output.Str() << ";" << std::endl;
    }
    if (is_dynamic) {
      if (is_inductor) {
        result << "#ifdef INDUCTOR_CONST_TILING_DATA" << std::endl;
        result << "    CV_FUSION_ADDR.stage_size_name = kConstTilingData.stage_size_name;" << std::endl;
        result << "    CV_FUSION_ADDR.cube_ub_stage_size = kConstTilingData.cube_ub_stage_size;" << std::endl;
        result << "    CV_FUSION_ADDR.cube_m_stage_size = kConstTilingData.cube_m_stage_size;" << std::endl;
        result << "#else" << std::endl;
        result << "    CV_FUSION_ADDR.stage_size_name = t.stage_size_name;" << std::endl;
        result << "    CV_FUSION_ADDR.cube_ub_stage_size = t.cube_ub_stage_size;" << std::endl;
        result << "    CV_FUSION_ADDR.cube_m_stage_size = t.cube_m_stage_size;" << std::endl;
        result << "#endif" << std::endl;
        result << "  }" << std::endl;
      } else {
        result << "  CV_FUSION_ADDR.cv_tiling_data = gm_tiling_data;" << std::endl;
      }
    }
  } else {
    if (is_dynamic) {
      result << "  AutoFusionVectorDB::Params CV_FUSION_ADDR_DB;\n";
      if (is_inductor) {
        result << "  if constexpr (UB_MODE != 0) {" << std::endl;
      }
      for (auto input : this->inputs) {
        result << "  " << (is_inductor ? "  " : "") << "CV_FUSION_ADDR_DB." << input.Str() << " = " << input.Str()
               << ";" << std::endl;
      }
      for (auto output : this->outputs) {
        result << "  " << (is_inductor ? "  " : "") << "CV_FUSION_ADDR_DB." << output.Str() << " = " << output.Str()
               << ";" << std::endl;
      }
      if (is_inductor) {
        result << "#ifdef INDUCTOR_CONST_TILING_DATA" << std::endl;
        result << "    CV_FUSION_ADDR_DB.stage_size_name = kConstTilingData.stage_size_name;" << std::endl;
        result << "    CV_FUSION_ADDR_DB.cube_ub_stage_size = kConstTilingData.cube_ub_stage_size;" << std::endl;
        result << "    CV_FUSION_ADDR_DB.cube_m_stage_size = kConstTilingData.cube_m_stage_size;" << std::endl;
        result << "#else" << std::endl;
        result << "    CV_FUSION_ADDR_DB.stage_size_name = t.stage_size_name;" << std::endl;
        result << "    CV_FUSION_ADDR_DB.cube_ub_stage_size = t.cube_ub_stage_size;" << std::endl;
        result << "    CV_FUSION_ADDR_DB.cube_m_stage_size = t.cube_m_stage_size;" << std::endl;
        result << "#endif" << std::endl;
        result << "  }" << std::endl;
      } else {
        result << "  CV_FUSION_ADDR_DB.cv_tiling_data = gm_tiling_data;" << std::endl;
      }
    }
  }
  return af::SUCCESS;
}

static std::string GetScheduledResultInputOutput(const ascir::FusedScheduledResult &fused_schedule_result,
                                                 bool is_kernel_func_call) {
  std::stringstream ss;
  for (size_t i = 0U; i < fused_schedule_result.input_nodes.size(); i++) {
    auto &input = fused_schedule_result.input_nodes[i];
    if (IsOps<Data>(input)) {
      ss << (is_kernel_func_call ? "(uint8_t*)" : "void* ") << "input" << i << ", ";
    } else if (IsOps<ScalarData>(input)) {
      std::string dtype_name;
      GE_ASSERT_SUCCESS(Tensor::DtypeName(input->outputs[0].attr.dtype, dtype_name), "data type:%d failed",
                        static_cast<int32_t>(input->outputs[0].attr.dtype));
      ss << (is_kernel_func_call ? "" : (dtype_name + " ")) << "input" << i << ", ";
    }
  }
  int32_t index = 0;
  for (const auto &node : fused_schedule_result.output_nodes) {
    if (IsOps<Output>(node)) {
      ss << (is_kernel_func_call ? "(uint8_t*)" : "void* ") << "output" << index++ << ", ";
    }
  }
  return ss.str();
}

static std::string GetDynamicUbSize(const ascir::FusedScheduledResult &fused_schedule_result) {
  ge::PlatformInfo platform_info;
  GE_ASSERT_SUCCESS(ge::PlatformContext::GetInstance().GetPlatformInfo(platform_info),
                    "Failed to get platform info for cube tiling.");
  int64_t ub_size = platform_info.ub_size;
  for (size_t graph_id = 0; graph_id < fused_schedule_result.node_idx_to_scheduled_results.size(); graph_id++) {
    auto scheduled_results = fused_schedule_result.node_idx_to_scheduled_results[graph_id];
    for (size_t i = 0; i < scheduled_results.size(); i++) {
      auto schedule_groups = scheduled_results[i].schedule_groups;
      for (size_t j = 0; j < schedule_groups.size(); j++) {
        auto schedule_graphs = schedule_groups[j].impl_graphs;
        for (size_t k = 0; k < schedule_graphs.size(); k++) {
          for (const auto &node : schedule_graphs[k].GetAllNodes()) {
            const int64_t simt_dcache_size = ::ascir::GetDcacheSize(*node);
            if (simt_dcache_size > 0) {
              int64_t dynamic_ub_size = ub_size - simt_dcache_size;
              return std::to_string(dynamic_ub_size);
            }
          }
        }
      }
    }
  }
  return "nullptr";
}

std::string Kernel::GenKernelFuncCallForInductor(const ascir::FusedScheduledResult &fused_schedule_result) {
  std::string tiling_data_name;
  bool is_cv = ascgen_utils::IsCubeFusedScheduled(fused_schedule_result);
  if (is_cv) {
    tiling_data_name = "CVAutofuseTilingData";
  } else {
    tiling_data_name = "AutofuseTilingData";
  }
  std::string graph_name = CamelToLowerSneak(GenValidName(fused_schedule_result.fused_graph_name.GetString()));
  std::string extern_c = "extern \"C\"";
  std::stringstream ss;
  bool is_static = IsStaticSchedResult(fused_schedule_result);

  std::string cube_template_args;
  std::string cube_template_call;
  bool is_batch = false;
  if (is_cv) {
    bool is_conv = false;
    std::string input_type;
    std::string output_type;
    GE_ASSERT_SUCCESS(ascgen_utils::GetCubeInfo(fused_schedule_result, is_batch, is_conv, input_type, output_type),
                      "Failed to get cube info from FusedScheduledResult");
    if (is_batch) {
      cube_template_args = R"(
#ifdef INDUCTOR_CONST_TILING_DATA
  constexpr int64_t tiling_key = kConstTilingData.extern_tiling_data.cube_tiling_key;
  constexpr int8_t BATCH_API_LEVEL = static_cast<int8_t>((tiling_key >> 0) & 0xF);
  constexpr int8_t BATCH_A_TRANS = static_cast<int8_t>((tiling_key >> 4) & 0x3);
  constexpr int8_t BATCH_B_TRANS = static_cast<int8_t>((tiling_key >> 6) & 0x3);
  constexpr int8_t BATCH_ITER_MODEL = static_cast<int8_t>((tiling_key >> 8) & 0xF);
  constexpr int8_t BMODEL = static_cast<int8_t>((tiling_key >> 12) & 0xF);
  constexpr int8_t BATCH_FULL_LOAD = static_cast<int8_t>((tiling_key >> 16) & 0xF);
  constexpr int8_t BATCH_L0C2OUT_MODEL = static_cast<int8_t>((tiling_key >> 20) & 0xF);
  constexpr int8_t FUSION_MODE = static_cast<int8_t>(kConstTilingData.extern_tiling_data.cv_tiling_data.fusion_mode);
  constexpr int8_t UB_MODE = static_cast<int8_t>(kConstTilingData.extern_tiling_data.cv_tiling_data.ub_mode);
  constexpr int8_t MIX_MODE = static_cast<int8_t>(kConstTilingData.extern_tiling_data.cv_tiling_data.mix_mode);
  (void)tiling_data;
#else
  int64_t tiling_key = tiling_data->cube_tiling_key;
  int8_t BATCH_API_LEVEL = (tiling_key >> 0) & 0xF;
  int8_t BATCH_A_TRANS = (tiling_key >> 4) & 0x3;
  int8_t BATCH_B_TRANS = (tiling_key >> 6) & 0x3;
  int8_t BATCH_ITER_MODEL = (tiling_key >> 8) & 0xF;
  int8_t BMODEL = (tiling_key >> 12) & 0xF;
  int8_t BATCH_FULL_LOAD = (tiling_key >> 16) & 0xF;
  int8_t BATCH_L0C2OUT_MODEL = (tiling_key >> 20) & 0xF;
  int8_t FUSION_MODE = tiling_data->cv_tiling_data.fusion_mode;
  int8_t UB_MODE = tiling_data->cv_tiling_data.ub_mode;
  int8_t MIX_MODE = tiling_data->cv_tiling_data.mix_mode;
#endif
)";
    } else {
      cube_template_args = R"(
#ifdef INDUCTOR_CONST_TILING_DATA
  constexpr int64_t tiling_key = kConstTilingData.extern_tiling_data.cube_tiling_key;
  constexpr int8_t API_LEVEL = static_cast<int8_t>((tiling_key >> 0) & 0xF);
  constexpr int8_t A_TRANS = static_cast<int8_t>((tiling_key >> 4) & 0x3);
  constexpr int8_t B_TRANS = static_cast<int8_t>((tiling_key >> 6) & 0x3);
  constexpr int8_t BATCH_MODEL = static_cast<int8_t>((tiling_key >> 8) & 0xF);
  constexpr int8_t MODEL = static_cast<int8_t>((tiling_key >> 12) & 0xF);
  constexpr int8_t FULL_LOAD = static_cast<int8_t>((tiling_key >> 16) & 0xF);
  constexpr int8_t L0C2OUT_MODEL = static_cast<int8_t>((tiling_key >> 20) & 0xF);
  constexpr int8_t FUSION_MODE = static_cast<int8_t>(kConstTilingData.extern_tiling_data.cv_tiling_data.fusion_mode);
  constexpr int8_t UB_MODE = static_cast<int8_t>(kConstTilingData.extern_tiling_data.cv_tiling_data.ub_mode);
  constexpr int8_t MIX_MODE = static_cast<int8_t>(kConstTilingData.extern_tiling_data.cv_tiling_data.mix_mode);
  (void)tiling_data;
#else
  int64_t tiling_key = tiling_data->cube_tiling_key;
  int8_t API_LEVEL = (tiling_key >> 0) & 0xF;
  int8_t A_TRANS = (tiling_key >> 4) & 0x3;
  int8_t B_TRANS = (tiling_key >> 6) & 0x3;
  int8_t BATCH_MODEL = (tiling_key >> 8) & 0xF;
  int8_t MODEL = (tiling_key >> 12) & 0xF;
  int8_t FULL_LOAD = (tiling_key >> 16) & 0xF;
  int8_t L0C2OUT_MODEL = (tiling_key >> 20) & 0xF;
  int8_t FUSION_MODE = tiling_data->cv_tiling_data.fusion_mode;
  int8_t UB_MODE = tiling_data->cv_tiling_data.ub_mode;
  int8_t MIX_MODE = tiling_data->cv_tiling_data.mix_mode;
#endif
)";
    }
  }
  // 适配AscendC新的单算子编译工程
  ss << "void init_" << graph_name << "(void) {}" << std::endl;
  ss << extern_c << " int64_t AutofuseLaunch(uint32_t blockDim, void* stream, ";
  ss << GetScheduledResultInputOutput(fused_schedule_result, false);
  ss << "void* workspace, " << tiling_data_name << "* tiling_data";
  ss << ")" << std::endl;
  ss << "{" << std::endl;

  // CV 场景下根据 tiling_key 分发到不同模板
  if (is_cv) {
    ss << cube_template_args;
    std::string dynamic_kernel_args =
        GetScheduledResultInputOutput(fused_schedule_result, true) + "(uint8_t*)workspace, full_tiling";
    std::string static_kernel_args = GetScheduledResultInputOutput(fused_schedule_result, true) + "(uint8_t*)workspace";
    std::string kernel_args = dynamic_kernel_args + ")";
    std::string ub_dynamic_kernel_args =
        GetScheduledResultInputOutput(fused_schedule_result, true) + "(uint8_t*)workspace, ub_tiling";
    std::string ub_kernel_args = ub_dynamic_kernel_args + ")";
    if (is_static) {
      ss << "#ifdef INDUCTOR_CONST_TILING_DATA" << std::endl;
      ss << "  #define _KERNEL_ARGS " << static_kernel_args << std::endl;
      ss << "  #define _UB_KERNEL_ARGS " << static_kernel_args << std::endl;
      ss << "#else" << std::endl;
      ss << "  #define _KERNEL_ARGS " << dynamic_kernel_args << std::endl;
      ss << "  #define _UB_KERNEL_ARGS " << ub_dynamic_kernel_args << std::endl;
      ss << "#endif" << std::endl;
      kernel_args = "_KERNEL_ARGS)";
      ub_kernel_args = "_UB_KERNEL_ARGS)";
    }

    // 定义宏简化分发逻辑：
    // - API, BMODEL, FL, LO, ITER 是具体常量（由 if-else 分支决定）
    // - A_TRANS, B_TRANS 是运行时值（需要嵌套 if 判断）
    // - FUSION_MODE, UB_MODE 是运行时值（需要嵌套 if 判断）
    ss << std::endl;
    ss << "  // Helper macro: A_TRANS/B_TRANS are runtime values, must use compile-time constants" << std::endl;
    if (is_static) {
      ss << "#ifdef INDUCTOR_CONST_TILING_DATA" << std::endl;
      ss << "  #define _LAUNCH_MATMUL_UB(MATMUL_TILING, API, AT, BT, MD, FL, LO, UB) \\" << std::endl;
      ss << "    do { \\" << std::endl;
      ss << "      " << graph_name << "<CVAutofuseTilingData, API, AT, BT, 0, MD, FL, LO, 0, UB, 0><<<blockDim, "
         << GetDynamicUbSize(fused_schedule_result) << ", stream>>>(" << ub_kernel_args << "; \\" << std::endl;
      ss << "    } while(0)" << std::endl;
      ss << "  #define _LAUNCH_MATMUL_SAFETY(MATMUL_TILING, API, AT, BT, MD, FL, LO, MIX) \\" << std::endl;
      ss << "    do { \\" << std::endl;
      ss << "      " << graph_name << "<CVAutofuseTilingData, API, AT, BT, 0, MD, FL, LO, 1, 0, MIX><<<blockDim, "
         << GetDynamicUbSize(fused_schedule_result) << ", stream>>>(" << kernel_args << "; \\" << std::endl;
      ss << "    } while(0)" << std::endl;
      ss << "#else" << std::endl;
    }
    ss << "  #define _LAUNCH_MATMUL_UB(MATMUL_TILING, API, AT, BT, MD, FL, LO, UB) \\" << std::endl;
    ss << "    do { \\" << std::endl;
    ss << "      using CvUbTilingT = CVAutofuseUbTilingDataT<MATMUL_TILING>; \\" << std::endl;
    ss << "      const auto &ub_tiling = *reinterpret_cast<const CvUbTilingT *>(&tiling_data->stage_size_name); \\"
       << std::endl;
    ss << "      " << graph_name << "<CvUbTilingT, API, AT, BT, 0, MD, FL, LO, 0, UB, 0><<<blockDim, "
       << GetDynamicUbSize(fused_schedule_result) << ", stream>>>(" << ub_kernel_args << "; \\" << std::endl;
    ss << "    } while(0)" << std::endl;
    ss << "  #define _LAUNCH_MATMUL_SAFETY(MATMUL_TILING, API, AT, BT, MD, FL, LO, MIX) \\" << std::endl;
    ss << "    do { \\" << std::endl;
    ss << "      using CvFullTilingT = CVAutofuseTilingDataT<MATMUL_TILING>; \\" << std::endl;
    ss << "      const auto &full_tiling = *reinterpret_cast<const CvFullTilingT *>(&tiling_data->extern_tiling_data); "
          "\\"
       << std::endl;
    ss << "      " << graph_name << "<CvFullTilingT, API, AT, BT, 0, MD, FL, LO, 1, 0, MIX><<<blockDim, "
       << GetDynamicUbSize(fused_schedule_result) << ", stream>>>(" << kernel_args << "; \\" << std::endl;
    ss << "    } while(0)" << std::endl;
    if (is_static) {
      ss << "#endif" << std::endl;
    }
    ss << "  #define _DISPATCH_MATMUL(API, MD, FL, LO, MATMUL_TILING) \\" << std::endl;
    ss << "    do { \\" << std::endl;
    ss << "      if (A_TRANS == 0 && B_TRANS == 0) { \\" << std::endl;
    ss << "        if (FUSION_MODE == 0 && UB_MODE == 0) { _LAUNCH_MATMUL_UB(MATMUL_TILING, API, 0, 0, MD, FL, LO, 0); "
          "} \\"
       << std::endl;
    ss << "        else if (FUSION_MODE == 0 && UB_MODE == 1) { _LAUNCH_MATMUL_UB(MATMUL_TILING, API, 0, 0, MD, FL, "
          "LO, 1); } \\"
       << std::endl;
    ss << "        else if (FUSION_MODE == 1 && MIX_MODE == 0) { _LAUNCH_MATMUL_SAFETY(MATMUL_TILING, API, 0, 0, MD, "
          "FL, LO, 0); } \\"
       << std::endl;
    ss << "        else if (FUSION_MODE == 1 && MIX_MODE == 1) { _LAUNCH_MATMUL_SAFETY(MATMUL_TILING, API, 0, 0, MD, "
          "FL, LO, 1); } \\"
       << std::endl;
    ss << "        else if (FUSION_MODE == 1 && MIX_MODE == 2) { _LAUNCH_MATMUL_SAFETY(MATMUL_TILING, API, 0, 0, MD, "
          "FL, LO, 2); } \\"
       << std::endl;
    ss << "      } else if (A_TRANS == 0 && B_TRANS == 1) { \\" << std::endl;
    ss << "        if (FUSION_MODE == 0 && UB_MODE == 0) { _LAUNCH_MATMUL_UB(MATMUL_TILING, API, 0, 1, MD, FL, LO, 0); "
          "} \\"
       << std::endl;
    ss << "        else if (FUSION_MODE == 0 && UB_MODE == 1) { _LAUNCH_MATMUL_UB(MATMUL_TILING, API, 0, 1, MD, FL, "
          "LO, 1); } \\"
       << std::endl;
    ss << "        else if (FUSION_MODE == 1 && MIX_MODE == 0) { _LAUNCH_MATMUL_SAFETY(MATMUL_TILING, API, 0, 1, MD, "
          "FL, LO, 0); } \\"
       << std::endl;
    ss << "        else if (FUSION_MODE == 1 && MIX_MODE == 1) { _LAUNCH_MATMUL_SAFETY(MATMUL_TILING, API, 0, 1, MD, "
          "FL, LO, 1); } \\"
       << std::endl;
    ss << "        else if (FUSION_MODE == 1 && MIX_MODE == 2) { _LAUNCH_MATMUL_SAFETY(MATMUL_TILING, API, 0, 1, MD, "
          "FL, LO, 2); } \\"
       << std::endl;
    ss << "      } else if (A_TRANS == 1 && B_TRANS == 0) { \\" << std::endl;
    ss << "        if (FUSION_MODE == 0 && UB_MODE == 0) { _LAUNCH_MATMUL_UB(MATMUL_TILING, API, 1, 0, MD, FL, LO, 0); "
          "} \\"
       << std::endl;
    ss << "        else if (FUSION_MODE == 0 && UB_MODE == 1) { _LAUNCH_MATMUL_UB(MATMUL_TILING, API, 1, 0, MD, FL, "
          "LO, 1); } \\"
       << std::endl;
    ss << "        else if (FUSION_MODE == 1 && MIX_MODE == 0) { _LAUNCH_MATMUL_SAFETY(MATMUL_TILING, API, 1, 0, MD, "
          "FL, LO, 0); } \\"
       << std::endl;
    ss << "        else if (FUSION_MODE == 1 && MIX_MODE == 1) { _LAUNCH_MATMUL_SAFETY(MATMUL_TILING, API, 1, 0, MD, "
          "FL, LO, 1); } \\"
       << std::endl;
    ss << "        else if (FUSION_MODE == 1 && MIX_MODE == 2) { _LAUNCH_MATMUL_SAFETY(MATMUL_TILING, API, 1, 0, MD, "
          "FL, LO, 2); } \\"
       << std::endl;
    ss << "      } else if (A_TRANS == 1 && B_TRANS == 1) { \\" << std::endl;
    ss << "        if (FUSION_MODE == 0 && UB_MODE == 0) { _LAUNCH_MATMUL_UB(MATMUL_TILING, API, 1, 1, MD, FL, LO, 0); "
          "} \\"
       << std::endl;
    ss << "        else if (FUSION_MODE == 0 && UB_MODE == 1) { _LAUNCH_MATMUL_UB(MATMUL_TILING, API, 1, 1, MD, FL, "
          "LO, 1); } \\"
       << std::endl;
    ss << "        else if (FUSION_MODE == 1 && MIX_MODE == 0) { _LAUNCH_MATMUL_SAFETY(MATMUL_TILING, API, 1, 1, MD, "
          "FL, LO, 0); } \\"
       << std::endl;
    ss << "        else if (FUSION_MODE == 1 && MIX_MODE == 1) { _LAUNCH_MATMUL_SAFETY(MATMUL_TILING, API, 1, 1, MD, "
          "FL, LO, 1); } \\"
       << std::endl;
    ss << "        else if (FUSION_MODE == 1 && MIX_MODE == 2) { _LAUNCH_MATMUL_SAFETY(MATMUL_TILING, API, 1, 1, MD, "
          "FL, LO, 2); } \\"
       << std::endl;
    ss << "      } \\" << std::endl;
    ss << "    } while(0)" << std::endl;
    ss << std::endl;

    // Batch matmul 宏：需要额外处理 BATCH_ITER_MODEL 和 BMODEL
    ss << "  // Batch matmul: additional parameters ITER and BMODEL" << std::endl;
    if (is_static) {
      ss << "#ifdef INDUCTOR_CONST_TILING_DATA" << std::endl;
      ss << "  #define _LAUNCH_BATCH_MATMUL_UB(MATMUL_TILING, API, AT, BT, ITER, BMODEL, FL, LO, UB) \\" << std::endl;
      ss << "    do { \\" << std::endl;
      ss << "      " << graph_name << "<CVAutofuseTilingData, API, AT, BT, ITER, BMODEL, FL, LO, 0, UB, 0><<<blockDim, "
         << GetDynamicUbSize(fused_schedule_result) << ", stream>>>(" << ub_kernel_args << "; \\" << std::endl;
      ss << "    } while(0)" << std::endl;
      ss << "  #define _LAUNCH_BATCH_MATMUL_SAFETY(MATMUL_TILING, API, AT, BT, ITER, BMODEL, FL, LO, MIX) \\"
         << std::endl;
      ss << "    do { \\" << std::endl;
      ss << "      " << graph_name
         << "<CVAutofuseTilingData, API, AT, BT, ITER, BMODEL, FL, LO, 1, 0, MIX><<<blockDim, "
         << GetDynamicUbSize(fused_schedule_result) << ", stream>>>(" << kernel_args << "; \\" << std::endl;
      ss << "    } while(0)" << std::endl;
      ss << "#else" << std::endl;
    }
    ss << "  #define _LAUNCH_BATCH_MATMUL_UB(MATMUL_TILING, API, AT, BT, ITER, BMODEL, FL, LO, UB) \\" << std::endl;
    ss << "    do { \\" << std::endl;
    ss << "      using CvUbTilingT = CVAutofuseUbTilingDataT<MATMUL_TILING>; \\" << std::endl;
    ss << "      const auto &ub_tiling = *reinterpret_cast<const CvUbTilingT *>(&tiling_data->stage_size_name); \\"
       << std::endl;
    ss << "      " << graph_name << "<CvUbTilingT, API, AT, BT, ITER, BMODEL, FL, LO, 0, UB, 0><<<blockDim, "
       << GetDynamicUbSize(fused_schedule_result) << ", stream>>>(" << ub_kernel_args << "; \\" << std::endl;
    ss << "    } while(0)" << std::endl;
    ss << "  #define _LAUNCH_BATCH_MATMUL_SAFETY(MATMUL_TILING, API, AT, BT, ITER, BMODEL, FL, LO, MIX) \\"
       << std::endl;
    ss << "    do { \\" << std::endl;
    ss << "      using CvFullTilingT = CVAutofuseTilingDataT<MATMUL_TILING>; \\" << std::endl;
    ss << "      const auto &full_tiling = *reinterpret_cast<const CvFullTilingT *>(&tiling_data->extern_tiling_data); "
          "\\"
       << std::endl;
    ss << "      " << graph_name << "<CvFullTilingT, API, AT, BT, ITER, BMODEL, FL, LO, 1, 0, MIX><<<blockDim, "
       << GetDynamicUbSize(fused_schedule_result) << ", stream>>>(" << kernel_args << "; \\" << std::endl;
    ss << "    } while(0)" << std::endl;
    if (is_static) {
      ss << "#endif" << std::endl;
    }
    ss << "  #define _DISPATCH_BATCH_MATMUL(API, BMODEL, FL, LO, ITER, MATMUL_TILING) \\" << std::endl;
    ss << "    do { \\" << std::endl;
    ss << "      if (BATCH_A_TRANS == 0 && BATCH_B_TRANS == 0) { \\" << std::endl;
    ss << "        if (FUSION_MODE == 0 && UB_MODE == 0) { _LAUNCH_BATCH_MATMUL_UB(MATMUL_TILING, API, 0, 0, ITER, "
          "BMODEL, FL, LO, 0); } \\"
       << std::endl;
    ss << "        else if (FUSION_MODE == 0 && UB_MODE == 1) { _LAUNCH_BATCH_MATMUL_UB(MATMUL_TILING, API, 0, 0, "
          "ITER, BMODEL, FL, LO, 1); } \\"
       << std::endl;
    ss << "        else if (FUSION_MODE == 1 && MIX_MODE == 0) { _LAUNCH_BATCH_MATMUL_SAFETY(MATMUL_TILING, API, 0, 0, "
          "ITER, BMODEL, FL, LO, 0); } \\"
       << std::endl;
    ss << "        else if (FUSION_MODE == 1 && MIX_MODE == 1) { _LAUNCH_BATCH_MATMUL_SAFETY(MATMUL_TILING, API, 0, 0, "
          "ITER, BMODEL, FL, LO, 1); } \\"
       << std::endl;
    ss << "        else if (FUSION_MODE == 1 && MIX_MODE == 2) { _LAUNCH_BATCH_MATMUL_SAFETY(MATMUL_TILING, API, 0, 0, "
          "ITER, BMODEL, FL, LO, 2); } \\"
       << std::endl;
    ss << "      } else if (BATCH_A_TRANS == 0 && BATCH_B_TRANS == 1) { \\" << std::endl;
    ss << "        if (FUSION_MODE == 0 && UB_MODE == 0) { _LAUNCH_BATCH_MATMUL_UB(MATMUL_TILING, API, 0, 1, ITER, "
          "BMODEL, FL, LO, 0); } \\"
       << std::endl;
    ss << "        else if (FUSION_MODE == 0 && UB_MODE == 1) { _LAUNCH_BATCH_MATMUL_UB(MATMUL_TILING, API, 0, 1, "
          "ITER, BMODEL, FL, LO, 1); } \\"
       << std::endl;
    ss << "        else if (FUSION_MODE == 1 && MIX_MODE == 0) { _LAUNCH_BATCH_MATMUL_SAFETY(MATMUL_TILING, API, 0, 1, "
          "ITER, BMODEL, FL, LO, 0); } \\"
       << std::endl;
    ss << "        else if (FUSION_MODE == 1 && MIX_MODE == 1) { _LAUNCH_BATCH_MATMUL_SAFETY(MATMUL_TILING, API, 0, 1, "
          "ITER, BMODEL, FL, LO, 1); } \\"
       << std::endl;
    ss << "        else if (FUSION_MODE == 1 && MIX_MODE == 2) { _LAUNCH_BATCH_MATMUL_SAFETY(MATMUL_TILING, API, 0, 1, "
          "ITER, BMODEL, FL, LO, 2); } \\"
       << std::endl;
    ss << "      } else if (BATCH_A_TRANS == 1 && BATCH_B_TRANS == 0) { \\" << std::endl;
    ss << "        if (FUSION_MODE == 0 && UB_MODE == 0) { _LAUNCH_BATCH_MATMUL_UB(MATMUL_TILING, API, 1, 0, ITER, "
          "BMODEL, FL, LO, 0); } \\"
       << std::endl;
    ss << "        else if (FUSION_MODE == 0 && UB_MODE == 1) { _LAUNCH_BATCH_MATMUL_UB(MATMUL_TILING, API, 1, 0, "
          "ITER, BMODEL, FL, LO, 1); } \\"
       << std::endl;
    ss << "        else if (FUSION_MODE == 1 && MIX_MODE == 0) { _LAUNCH_BATCH_MATMUL_SAFETY(MATMUL_TILING, API, 1, 0, "
          "ITER, BMODEL, FL, LO, 0); } \\"
       << std::endl;
    ss << "        else if (FUSION_MODE == 1 && MIX_MODE == 1) { _LAUNCH_BATCH_MATMUL_SAFETY(MATMUL_TILING, API, 1, 0, "
          "ITER, BMODEL, FL, LO, 1); } \\"
       << std::endl;
    ss << "        else if (FUSION_MODE == 1 && MIX_MODE == 2) { _LAUNCH_BATCH_MATMUL_SAFETY(MATMUL_TILING, API, 1, 0, "
          "ITER, BMODEL, FL, LO, 2); } \\"
       << std::endl;
    ss << "      } else if (BATCH_A_TRANS == 1 && BATCH_B_TRANS == 1) { \\" << std::endl;
    ss << "        if (FUSION_MODE == 0 && UB_MODE == 0) { _LAUNCH_BATCH_MATMUL_UB(MATMUL_TILING, API, 1, 1, ITER, "
          "BMODEL, FL, LO, 0); } \\"
       << std::endl;
    ss << "        else if (FUSION_MODE == 0 && UB_MODE == 1) { _LAUNCH_BATCH_MATMUL_UB(MATMUL_TILING, API, 1, 1, "
          "ITER, BMODEL, FL, LO, 1); } \\"
       << std::endl;
    ss << "        else if (FUSION_MODE == 1 && MIX_MODE == 0) { _LAUNCH_BATCH_MATMUL_SAFETY(MATMUL_TILING, API, 1, 1, "
          "ITER, BMODEL, FL, LO, 0); } \\"
       << std::endl;
    ss << "        else if (FUSION_MODE == 1 && MIX_MODE == 1) { _LAUNCH_BATCH_MATMUL_SAFETY(MATMUL_TILING, API, 1, 1, "
          "ITER, BMODEL, FL, LO, 1); } \\"
       << std::endl;
    ss << "        else if (FUSION_MODE == 1 && MIX_MODE == 2) { _LAUNCH_BATCH_MATMUL_SAFETY(MATMUL_TILING, API, 1, 1, "
          "ITER, BMODEL, FL, LO, 2); } \\"
       << std::endl;
    ss << "      } \\" << std::endl;
    ss << "    } while(0)" << std::endl;
    ss << std::endl;

    if (is_batch) {
      // Batch matmul: 参考 batch_matmul_dynamic.h 的 if constexpr 分支结构
      ss << "  // Batch matmul template dispatch (matching batch_matmul_dynamic.h if constexpr branches)" << std::endl;
      ss << std::endl;

      // 分支1: HIGH_LEVEL, BASIC, NO_FULL_LOAD, ON_THE_FLY, FOR_BATCH
      ss << "  if (BATCH_API_LEVEL == 0 && BMODEL == 0 && BATCH_FULL_LOAD == 0 && BATCH_L0C2OUT_MODEL == 0 && "
            "BATCH_ITER_MODEL == 0) {"
         << std::endl;
      ss << "    _DISPATCH_BATCH_MATMUL(0, 0, 0, 0, 0, BatchMatMulV3TilingData);" << std::endl;
      // 分支2: BASIC_LEVEL, BASIC, NO_FULL_LOAD, ON_THE_FLY, FOR_BATCH
      ss << "  } else if (BATCH_API_LEVEL == 1 && BMODEL == 0 && BATCH_FULL_LOAD == 0 && BATCH_L0C2OUT_MODEL == 0 && "
            "BATCH_ITER_MODEL == 0) {"
         << std::endl;
      ss << "    _DISPATCH_BATCH_MATMUL(1, 0, 0, 0, 0, BatchMatMulV3BasicTilingData);" << std::endl;
      // 分支3: BASIC_LEVEL, BASIC, A_FULL_LOAD, ON_THE_FLY, FOR_BATCH
      ss << "  } else if (BATCH_API_LEVEL == 1 && BMODEL == 0 && BATCH_FULL_LOAD == 1 && BATCH_L0C2OUT_MODEL == 0 && "
            "BATCH_ITER_MODEL == 0) {"
         << std::endl;
      ss << "    _DISPATCH_BATCH_MATMUL(1, 0, 1, 0, 0, BatchMatMulV3BasicTilingData);" << std::endl;
      // 分支4: BASIC_LEVEL, BASIC, B_FULL_LOAD, ON_THE_FLY, FOR_BATCH
      ss << "  } else if (BATCH_API_LEVEL == 1 && BMODEL == 0 && BATCH_FULL_LOAD == 2 && BATCH_L0C2OUT_MODEL == 0 && "
            "BATCH_ITER_MODEL == 0) {"
         << std::endl;
      ss << "    _DISPATCH_BATCH_MATMUL(1, 0, 2, 0, 0, BatchMatMulV3BasicTilingData);" << std::endl;
      // 分支5: BASIC_LEVEL, BASIC, NO_FULL_LOAD, ON_THE_FLY, ITER_BATCH_SINGLE_BIAS
      ss << "  } else if (BATCH_API_LEVEL == 1 && BMODEL == 0 && BATCH_FULL_LOAD == 0 && BATCH_L0C2OUT_MODEL == 0 && "
            "BATCH_ITER_MODEL == 1) {"
         << std::endl;
      ss << "    _DISPATCH_BATCH_MATMUL(1, 0, 0, 0, 1, BatchMatMulV3IterBatchBasicTilingData);" << std::endl;
      // 分支6: BASIC_LEVEL, STREAM_K, NO_FULL_LOAD, ON_THE_FLY, FOR_BATCH
      ss << "  } else if (BATCH_API_LEVEL == 1 && BMODEL == 1 && BATCH_FULL_LOAD == 0 && BATCH_L0C2OUT_MODEL == 0 && "
            "BATCH_ITER_MODEL == 0) {"
         << std::endl;
      ss << "    _DISPATCH_BATCH_MATMUL(1, 1, 0, 0, 0, BatchMatMulV3BasicTilingData);" << std::endl;
      // 分支7: BASIC_LEVEL, STREAM_K, NO_FULL_LOAD, 1V2_ND_ALIG_FIXPIPE, FOR_BATCH
      ss << "  } else if (BATCH_API_LEVEL == 1 && BMODEL == 1 && BATCH_FULL_LOAD == 0 && BATCH_L0C2OUT_MODEL == 2 && "
            "BATCH_ITER_MODEL == 0) {"
         << std::endl;
      ss << "    _DISPATCH_BATCH_MATMUL(1, 1, 0, 2, 0, BatchMatMulV3BasicTilingData);" << std::endl;
      // 分支8: HIGH_LEVEL, BASIC, NO_FULL_LOAD, ON_THE_FLY, ITER_BATCH_SINGLE_BIAS
      ss << "  } else if (BATCH_API_LEVEL == 0 && BMODEL == 0 && BATCH_FULL_LOAD == 0 && BATCH_L0C2OUT_MODEL == 0 && "
            "BATCH_ITER_MODEL == 1) {"
         << std::endl;
      ss << "    _DISPATCH_BATCH_MATMUL(0, 0, 0, 0, 1, BatchMatMulV3TilingData);" << std::endl;
      // 分支9: BASIC_LEVEL, BASIC, NO_FULL_LOAD, 1V2_ND_ALIG_FIXPIPE, ITER_BATCH_SINGLE_BIAS
      ss << "  } else if (BATCH_API_LEVEL == 1 && BMODEL == 0 && BATCH_FULL_LOAD == 0 && BATCH_L0C2OUT_MODEL == 2 && "
            "BATCH_ITER_MODEL == 1) {"
         << std::endl;
      ss << "    _DISPATCH_BATCH_MATMUL(1, 0, 0, 2, 1, BatchMatMulV3IterBatchBasicTilingData);" << std::endl;
      // 分支10: BASIC_LEVEL, BASIC, NO_FULL_LOAD, ON_THE_FLY, MERGE_BATCH
      ss << "  } else if (BATCH_API_LEVEL == 1 && BMODEL == 0 && BATCH_FULL_LOAD == 0 && BATCH_L0C2OUT_MODEL == 0 && "
            "BATCH_ITER_MODEL == 3) {"
         << std::endl;
      ss << "    _DISPATCH_BATCH_MATMUL(1, 0, 0, 0, 3, BatchMatMulV3MergeBatchBasicTilingData);" << std::endl;
      // 分支11: BASIC_LEVEL, BASIC, NO_FULL_LOAD, ON_THE_FLY, BATCH_MATMUL_TO_MUL
      ss << "  } else if (BATCH_API_LEVEL == 1 && BMODEL == 0 && BATCH_FULL_LOAD == 0 && BATCH_L0C2OUT_MODEL == 0 && "
            "BATCH_ITER_MODEL == 2) {"
         << std::endl;
      ss << "    _DISPATCH_BATCH_MATMUL(1, 0, 0, 0, 2, BatchMatMulToMulBasicTilingData);" << std::endl;
      // 分支12: BASIC_LEVEL, K_EQUAL_ZERO, NO_FULL_LOAD, ON_THE_FLY, FOR_BATCH
      ss << "  } else if (BATCH_API_LEVEL == 1 && BMODEL == 2 && BATCH_FULL_LOAD == 0 && BATCH_L0C2OUT_MODEL == 0 && "
            "BATCH_ITER_MODEL == 0) {"
         << std::endl;
      ss << "    _DISPATCH_BATCH_MATMUL(1, 2, 0, 0, 0, MatMulV3KEqZeroBasicTilingData);" << std::endl;
      ss << "  } else {" << std::endl;
      ss << "    return -1;" << std::endl;
      ss << "  }" << std::endl;
    } else {
      // Matmul: 参考 mat_mul_v3.h 的 if constexpr 分支结构
      // 只检查 API_LEVEL, FULL_LOAD, MODEL, L0C2OUT_MODEL
      // BATCH_MODEL 固定为 0（matmul 不使用此参数）
      ss << "  // Matmul template dispatch (matching mat_mul_v3.h if constexpr branches)" << std::endl;
      ss << "  // BATCH_MODEL is fixed to 0 (not used in matmul)" << std::endl;
      ss << std::endl;

      // 分支1: BASIC_LEVEL, NO_FULL_LOAD, BASIC, ON_THE_FLY
      ss << "  if (API_LEVEL == 1 && FULL_LOAD == 0 && MODEL == 0 && L0C2OUT_MODEL == 0) {" << std::endl;
      ss << "    _DISPATCH_MATMUL(1, 0, 0, 0, MatMulV3BasicTilingData);" << std::endl;
      // 分支2: BASIC_LEVEL, NO_FULL_LOAD, SLICE, ON_THE_FLY
      ss << "  } else if (API_LEVEL == 1 && FULL_LOAD == 0 && MODEL == 5 && L0C2OUT_MODEL == 0) {" << std::endl;
      ss << "    _DISPATCH_MATMUL(1, 5, 0, 0, MatMulV3BasicTilingData);" << std::endl;
      // 分支3: BASIC_LEVEL, NO_FULL_LOAD, BASIC_SPLIT_K, ON_THE_FLY
      ss << "  } else if (API_LEVEL == 1 && FULL_LOAD == 0 && MODEL == 6 && L0C2OUT_MODEL == 0) {" << std::endl;
      ss << "    _DISPATCH_MATMUL(1, 6, 0, 0, MatMulV3BasicTilingData);" << std::endl;
      // 分支4: BASIC_LEVEL, B_FULL_LOAD, BASIC, ON_THE_FLY
      ss << "  } else if (API_LEVEL == 1 && FULL_LOAD == 2 && MODEL == 0 && L0C2OUT_MODEL == 0) {" << std::endl;
      ss << "    _DISPATCH_MATMUL(1, 0, 2, 0, MatMulV3BasicTilingData);" << std::endl;
      // 分支5: BASIC_LEVEL, A_FULL_LOAD, BASIC, ON_THE_FLY
      ss << "  } else if (API_LEVEL == 1 && FULL_LOAD == 1 && MODEL == 0 && L0C2OUT_MODEL == 0) {" << std::endl;
      ss << "    _DISPATCH_MATMUL(1, 0, 1, 0, MatMulV3BasicTilingData);" << std::endl;
      // 分支6: BASIC_LEVEL, NO_FULL_LOAD, STREAM_K, ON_THE_FLY
      ss << "  } else if (API_LEVEL == 1 && FULL_LOAD == 0 && MODEL == 1 && L0C2OUT_MODEL == 0) {" << std::endl;
      ss << "    _DISPATCH_MATMUL(1, 1, 0, 0, MatMulV3BasicTilingData);" << std::endl;
      // 分支7: BASIC_LEVEL, NO_FULL_LOAD, STREAM_K, 1V2_ND_ALIG_FIXPIPE
      ss << "  } else if (API_LEVEL == 1 && FULL_LOAD == 0 && MODEL == 1 && L0C2OUT_MODEL == 2) {" << std::endl;
      ss << "    _DISPATCH_MATMUL(1, 1, 0, 2, MatMulV3BasicTilingData);" << std::endl;
      // 分支8: BASIC_LEVEL, NO_FULL_LOAD, SK_SPLIT_K, ON_THE_FLY
      ss << "  } else if (API_LEVEL == 1 && FULL_LOAD == 0 && MODEL == 7 && L0C2OUT_MODEL == 0) {" << std::endl;
      ss << "    _DISPATCH_MATMUL(1, 7, 0, 0, MatMulV3BasicTilingData);" << std::endl;
      // 分支9: BASIC_LEVEL, NO_FULL_LOAD, SK_SPLIT_K, 1V2_ND_ALIG_FIXPIPE
      ss << "  } else if (API_LEVEL == 1 && FULL_LOAD == 0 && MODEL == 7 && L0C2OUT_MODEL == 2) {" << std::endl;
      ss << "    _DISPATCH_MATMUL(1, 7, 0, 2, MatMulV3BasicTilingData);" << std::endl;
      // 分支10: BASIC_LEVEL, NO_FULL_LOAD, K_EQUAL_ZERO, ON_THE_FLY
      ss << "  } else if (API_LEVEL == 1 && FULL_LOAD == 0 && MODEL == 2 && L0C2OUT_MODEL == 0) {" << std::endl;
      ss << "    _DISPATCH_MATMUL(1, 2, 0, 0, MatMulV3KEqZeroBasicTilingData);" << std::endl;
      // 分支11: BASIC_LEVEL, B_FULL_LOAD, BASIC, 1V1_ND_ALIG_FIXPIPE
      ss << "  } else if (API_LEVEL == 1 && FULL_LOAD == 2 && MODEL == 0 && L0C2OUT_MODEL == 1) {" << std::endl;
      ss << "    _DISPATCH_MATMUL(1, 0, 2, 1, MatMulV3BasicTilingData);" << std::endl;
      // 分支12: BASIC_LEVEL, NO_FULL_LOAD, BASIC, 1V1_ND_ALIG_FIXPIPE
      ss << "  } else if (API_LEVEL == 1 && FULL_LOAD == 0 && MODEL == 0 && L0C2OUT_MODEL == 1) {" << std::endl;
      ss << "    _DISPATCH_MATMUL(1, 0, 0, 1, MatMulV3BasicTilingData);" << std::endl;
      // 分支13: BASIC_LEVEL, NO_FULL_LOAD, BASIC, 1V2_ND_ALIG_FIXPIPE
      ss << "  } else if (API_LEVEL == 1 && FULL_LOAD == 0 && MODEL == 0 && L0C2OUT_MODEL == 2) {" << std::endl;
      ss << "    _DISPATCH_MATMUL(1, 0, 0, 2, MatMulV3BasicTilingData);" << std::endl;
      // 分支14: BASIC_LEVEL, B_FULL_LOAD, BASIC, 1V2_ND_ALIG_FIXPIPE
      ss << "  } else if (API_LEVEL == 1 && FULL_LOAD == 2 && MODEL == 0 && L0C2OUT_MODEL == 2) {" << std::endl;
      ss << "    _DISPATCH_MATMUL(1, 0, 2, 2, MatMulV3BasicTilingData);" << std::endl;
      // 分支15: BASIC_LEVEL, NO_FULL_LOAD, TO_MUL, ON_THE_FLY
      ss << "  } else if (API_LEVEL == 1 && FULL_LOAD == 0 && MODEL == 3 && L0C2OUT_MODEL == 0) {" << std::endl;
      ss << "    _DISPATCH_MATMUL(1, 3, 0, 0, MatMulToMulBasicTilingData);" << std::endl;
      // 分支16: BASIC_LEVEL, NO_FULL_LOAD, TO_MULTI_MUL, ON_THE_FLY
      ss << "  } else if (API_LEVEL == 1 && FULL_LOAD == 0 && MODEL == 4 && L0C2OUT_MODEL == 0) {" << std::endl;
      ss << "    _DISPATCH_MATMUL(1, 4, 0, 0, MatMulToVectorBasicTilingData);" << std::endl;
      ss << "  } else {" << std::endl;
      ss << "    return -1;" << std::endl;
      ss << "  }" << std::endl;
    }
    ss << "  #undef _LAUNCH_MATMUL_UB" << std::endl;
    ss << "  #undef _LAUNCH_MATMUL_SAFETY" << std::endl;
    ss << "  #undef _LAUNCH_BATCH_MATMUL_UB" << std::endl;
    ss << "  #undef _LAUNCH_BATCH_MATMUL_SAFETY" << std::endl;
    ss << "  #undef _DISPATCH_MATMUL" << std::endl;
    ss << "  #undef _DISPATCH_BATCH_MATMUL" << std::endl;
    if (is_static) {
      ss << "  #undef _KERNEL_ARGS" << std::endl;
      ss << "  #undef _UB_KERNEL_ARGS" << std::endl;
    }
  } else {
    ss << "  " << graph_name;
    ss << "<<<blockDim, " << GetDynamicUbSize(fused_schedule_result) << ", stream>>>(";
    ss << GetScheduledResultInputOutput(fused_schedule_result, true);
    ss << "(uint8_t*)workspace, *tiling_data);" << std::endl;
  }

  ss << "  return 0;" << std::endl;
  ss << "}" << std::endl;
  return ss.str();
}

static ApiCallRegister<ApiCall> register_api_call("ApiCall");
