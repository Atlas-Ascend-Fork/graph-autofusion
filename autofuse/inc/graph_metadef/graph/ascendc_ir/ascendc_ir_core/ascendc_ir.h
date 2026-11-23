/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GRAPH_ASCENDC_IR_H
#define GRAPH_ASCENDC_IR_H

#include <string>
#include "graph/symbolizer/symbolic.h"
#include "graph/node.h"
#include "graph/anchor.h"
#include "graph/compute_graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/operator_af.h"
#include "graph/utils/type_utils.h"
#include "ascendc_ir/ascendc_ir_check.h"
#include "graph/expression/const_values.h"
#include "ascendc_ir/ascendc_ir_core/ascendc_ir_def.h"

namespace af {
#ifdef AUTOFUSE_USE_GE_METADEF
using ge::AscendString;
using ge::ComputeGraph;
using ge::ComputeGraphPtr;
using ge::ConstNodePtr;
using ge::InDataAnchor;
using ge::InDataAnchorPtr;
using ge::Node;
using ge::NodePtr;
using ge::OpDesc;
using ge::OpDescPtr;
using ge::OpDescUtils;
using ge::Operator;
using ge::OutDataAnchor;
using ge::OutDataAnchorPtr;
using ge::PARAM_INVALID;
using ge::Status;
#endif
struct DiffAxesInfo {
  std::vector<AxisId> add_axes;
  std::vector<AxisId> del_axes;
};
struct View {
  std::vector<int64_t> axis_ids;
  std::vector<Expression> repeats;
  std::vector<Expression> strides;
};
using TransInfoRoadOfGraph = std::vector<OneTransInfo>;

// 默认实现
template <typename T>
std::string ViewMemberToString(const std::vector<T> &vec) {
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < vec.size(); ++i) {
    oss << vec[i];
    if (i < vec.size() - 1) {
      oss << ", ";
    }
  }
  oss << "]";
  return oss.str();
}

// 特化实现，针对 Expression 类型
template <>
inline std::string ViewMemberToString(const std::vector<Expression> &vec) {
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < vec.size(); ++i) {
    const Expression &expr = vec[i];
    oss << (expr.Str() != nullptr ? expr.Str().get() : std::string("null"));
    if (i < vec.size() - 1) {
      oss << ", ";
    }
  }
  oss << "]";
  return oss.str();
}

inline std::string ViewToString(const View &view) {
  std::string result = "{ axis: " + ViewMemberToString(view.axis_ids) +
                       ", repeats: " + ViewMemberToString(view.repeats) +
                       ", strides: " + ViewMemberToString(view.strides) + " }";
  return result;
}

class AscOutputAttrDataType {
 public:
  AscOutputAttrDataType(Operator *op, uint32_t output_index) : op_(op), output_index_(output_index) {}

  AscOutputAttrDataType &operator=(const ge::DataType &value) {
    if (op_ == nullptr) {
      GELOGE(PARAM_INVALID, "op_ is null");
      return *this;
    }
    const auto desc = OpDescUtils::GetOpDescFromOperator(*op_);
    if (desc == nullptr) {
      GELOGE(PARAM_INVALID, "desc is null");
      return *this;
    }
    if (desc->MutableOutputDesc(output_index_) == nullptr) {
      GELOGE(PARAM_INVALID, "output_index_ %u is invalid for %s %s", output_index_, desc->GetNamePtr(),
             desc->GetTypePtr());
      return *this;
    }
    desc->MutableOutputDesc(output_index_)->SetDataType(value);
    return *this;
  }

  operator ge::DataType() const {
    if (op_ == nullptr) {
      GELOGE(PARAM_INVALID, "op_ is null");
      return ge::DT_UNDEFINED;
    }
    const auto desc = OpDescUtils::GetOpDescFromOperator(*op_);
    if (desc == nullptr) {
      GELOGE(PARAM_INVALID, "desc is null");
      return ge::DT_UNDEFINED;
    }
    if (desc->MutableOutputDesc(output_index_) == nullptr) {
      GELOGE(PARAM_INVALID, "output_index_ %u is invalid for %s %s", output_index_, desc->GetNamePtr(),
             desc->GetTypePtr());
      return ge::DT_UNDEFINED;
    }
    return desc->MutableOutputDesc(output_index_)->GetDataType();
  };

 private:
  Operator *op_{nullptr};
  uint32_t output_index_{UINT32_MAX};
};

class AscOutputAttrFormat {
 public:
  AscOutputAttrFormat(Operator *op, uint32_t output_index) : op_(op), output_index_(output_index) {}

  AscOutputAttrFormat &operator=(const ge::Format &value) {
    if (op_ == nullptr) {
      GELOGE(PARAM_INVALID, "op_ is null");
      return *this;
    }
    const auto desc = OpDescUtils::GetOpDescFromOperator(*op_);
    if (desc == nullptr) {
      GELOGE(PARAM_INVALID, "desc is null");
      return *this;
    }
    if (desc->MutableOutputDesc(output_index_) == nullptr) {
      GELOGE(PARAM_INVALID, "output_index_ %u is invalid for %s %s", output_index_, desc->GetNamePtr(),
             desc->GetTypePtr());
      return *this;
    }
    desc->MutableOutputDesc(output_index_)->SetFormat(value);
    return *this;
  }

  operator ge::Format() const {
    if (op_ == nullptr) {
      GELOGE(PARAM_INVALID, "op_ is null");
      return ge::FORMAT_RESERVED;
    }
    const auto desc = OpDescUtils::GetOpDescFromOperator(*op_);
    if (desc == nullptr) {
      GELOGE(PARAM_INVALID, "desc is null");
      return ge::FORMAT_RESERVED;
    }
    if (desc->MutableOutputDesc(output_index_) == nullptr) {
      GELOGE(PARAM_INVALID, "output_index_ %u is invalid for %s %s", output_index_, desc->GetNamePtr(),
             desc->GetTypePtr());
      return ge::FORMAT_RESERVED;
    }
    return desc->MutableOutputDesc(output_index_)->GetFormat();
  };

 private:
  Operator *op_{nullptr};
  uint32_t output_index_{UINT32_MAX};
};

class AscOpOutput {
 public:
  class AscOpOutputOffsetHelper {
    friend class AscOpOutput;

   public:
    ~AscOpOutputOffsetHelper() = default;
    void operator=(const AscOpOutput &asc_op_output) {
      // asc_op_output.op_保证非空
      output_.op_ = asc_op_output.op_;
      output_.output_index = asc_op_output.output_index;
      output_.dtype = asc_op_output.dtype;
      output_.format = asc_op_output.format;
      output_.axis = asc_op_output.axis;
      output_.repeats = asc_op_output.repeats;
      output_.strides = asc_op_output.strides;
      *asc_op_output.vectorized_axis = output_.load_vectorized_axes_;
      output_.vectorized_axis = asc_op_output.vectorized_axis;
      output_.vectorized_strides = asc_op_output.vectorized_strides;
      output_.mem = asc_op_output.mem;
      output_.que = asc_op_output.que;
      output_.buf = asc_op_output.buf;
      output_.opt = asc_op_output.opt;
    }

   private:
    explicit AscOpOutputOffsetHelper(AscOpOutput &output) : output_(output) {}
    AscOpOutput &output_;
  };
  template <uint32_t INPUT_INDEX>
  friend class AscOpInput;
  template <uint32_t INPUT_INDEX>
  friend class AscOpDynamicInput;
  friend class VectorizedOutTensor;
  AscOpOutput()
      : op_(nullptr),
        load_vectorized_axes_({}),
        output_index(UINT32_MAX),
        dtype(nullptr, output_index),
        format(nullptr, output_index),
        axis(nullptr),
        repeats(nullptr),
        strides(nullptr),
        vectorized_axis(nullptr),
        vectorized_strides(nullptr),
        mem(nullptr),
        que(nullptr),
        buf(nullptr),
        opt(nullptr) {}
  AscOpOutput(Operator *op, uint32_t output_idx)
      : op_(op), output_index(output_idx), dtype(op, output_index), format(op, output_index) {
    TryInitTensorAttr();
  }
  explicit AscOpOutput(std::vector<int64_t> axis_ids)
      : op_(nullptr),
        load_vectorized_axes_(std::move(axis_ids)),
        output_index(0),
        dtype(nullptr, output_index),
        format(nullptr, output_index),
        axis(nullptr),
        repeats(nullptr),
        strides(nullptr),
        vectorized_axis(nullptr),
        vectorized_strides(nullptr),
        mem(nullptr),
        que(nullptr),
        buf(nullptr),
        opt(nullptr) {}
  AscOpOutput(const AscOpOutput &output) : AscOpOutput(output.op_, output.output_index) {}
  AscOpOutput(AscOpOutput &&output) noexcept : AscOpOutput(output.op_, output.output_index) {}
  AscOpOutput &operator=(AscOpOutput &&) = delete;
  AscOpOutput &operator=(const AscOpOutput &) = delete;

  bool SetContiguousView(const std::vector<Axis> &axes) {
    GE_ASSERT_NOTNULL(axis, "output tensor should bind to API by API function or by AutoOffset");
    GE_ASSERT_NOTNULL(repeats, "output tensor should bind to API by API function or by AutoOffset");
    GE_ASSERT_NOTNULL(strides, "output tensor should bind to API by API function or by AutoOffset");
    std::vector<AxisId> axes_ids;
    std::vector<Expression> tmp_repeats;
    std::vector<Expression> tmp_strides;
    axes_ids.reserve(axes.size());
    tmp_repeats.reserve(axes.size());
    tmp_strides.reserve(axes.size());

    std::for_each(axes.rbegin(), axes.rend(), [&axes_ids, &tmp_repeats, &tmp_strides](const Axis &tmp_axis) {
      if (tmp_strides.empty()) {
        tmp_strides.emplace_back(sym::kSymbolOne);
      } else {
        tmp_strides.emplace_back(*tmp_repeats.rbegin() * *tmp_strides.rbegin());
      }
      tmp_repeats.emplace_back(tmp_axis.size);
      axes_ids.emplace_back(tmp_axis.id);
    });
    std::reverse(axes_ids.begin(), axes_ids.end());
    std::reverse(tmp_repeats.begin(), tmp_repeats.end());
    std::reverse(tmp_strides.begin(), tmp_strides.end());

    *axis = axes_ids;
    *repeats = tmp_repeats;
    *strides = tmp_strides;
    return true;
  }

  const Operator &GetOwnerOp() const {
    return *op_;
  }

  Operator &MutableOwnerOp() const {
    return *op_;
  }

  void TryInitTensorAttr() {
    auto tensor_attr_ptr = AscTensorAttr::GetTensorAttrPtr(op_, output_index);
    if (tensor_attr_ptr == nullptr) {
      return;
    }
    auto &tensor_attr = *tensor_attr_ptr;
    axis = &tensor_attr.axis;
    repeats = &tensor_attr.repeats;
    strides = &tensor_attr.strides;
    vectorized_axis = &tensor_attr.vectorized_axis;
    vectorized_strides = &tensor_attr.vectorized_strides;
    mem = &tensor_attr.mem;
    que = &tensor_attr.que;
    buf = &tensor_attr.buf;
    opt = &tensor_attr.opt;
  }
  AscOpOutput &Use(const AscOpOutput &used_out) {
    if (op_ == nullptr) {
      GELOGE(PARAM_INVALID, "output tensor should bind to API by API function or by AutoOffset");
      return *this;
    }
    if (HasBindToContainer()) {
      GELOGE(PARAM_INVALID, "this tensor has been bound to a que or buf, can not be bound repeatedly.");
      return *this;
    }
    if (!used_out.HasBindToContainer()) {
      GELOGE(PARAM_INVALID, " tensor to be used has not been bound to any que or buf.");
      return *this;
    }
    if (used_out.que->id != kIdNone) {
      (void)UseTQue(used_out.mem->position, used_out.que->depth, used_out.que->buf_num, used_out.que->id);
    }
    if (used_out.buf->id != kIdNone) {
      (void)UseTBuf(used_out.mem->position, used_out.buf->id);
    }
    mem->reuse_id = used_out.mem->reuse_id;
    return *this;
  }

  AscOpOutput &TQue(const Position pos, const int64_t depth, const int64_t buf_num) {
    if (op_ == nullptr) {
      GELOGE(PARAM_INVALID, "output tensor should bind to API by API function or by AutoOffset");
      return *this;
    }
    mem->reuse_id = GenNextReuseId();
    (void)UseTQue(pos, depth, buf_num);
    return *this;
  }

  AscOpOutput &TBuf(const Position pos) {
    if (op_ == nullptr) {
      GELOGE(PARAM_INVALID, "output tensor should bind to API by API function or by AutoOffset");
      return *this;
    }
    mem->reuse_id = GenNextReuseId();
    UseTBuf(pos);
    return *this;
  }
  AscOpOutputOffsetHelper AutoOffset() {
    return AscOpOutputOffsetHelper(*this);
  }

 private:
  int64_t GenContainerId();
  int64_t GenNextReuseId();
  bool UseTQue(const Position pos, const int64_t depth, const int64_t buf_num, const int64_t id = kIdNone);
  bool UseTBuf(const Position pos, const int64_t id = kIdNone);
  bool HasBindToContainer() const;
  Operator *op_;
  std::vector<int64_t> load_vectorized_axes_;

 public:
  uint32_t output_index{UINT32_MAX};
  AscOutputAttrDataType dtype;
  AscOutputAttrFormat format;
  std::vector<int64_t> *axis{};
  std::vector<Expression> *repeats{};
  std::vector<Expression> *strides{};
  std::vector<int64_t> *vectorized_axis{};
  std::vector<Expression> *vectorized_strides{};
  MemAttr *mem{};
  MemQueAttr *que{};
  MemBufAttr *buf{};
  MemOptAttr *opt{};
};

class VectorizedOutTensor {
 public:
  explicit VectorizedOutTensor(std::vector<int64_t> vectorized_axis) : vectorized_axis_(std::move(vectorized_axis)) {}
  VectorizedOutTensor &operator=(const VectorizedOutTensor &) = delete;
  VectorizedOutTensor(const VectorizedOutTensor &) = delete;
  explicit operator AscOpOutput() const {
    GE_ASSERT_NOTNULL(op_);
    return AscOpOutput(op_, output_index_);
  }
  void operator=(AscOpOutput &&asc_op_output) {
    // 不支持更改归属
    if (op_ != nullptr) {
      AscendString name;
      (void)op_->GetName(name);
      GELOGE(FAILED, "Tensor has been bound to %s", name.GetString());
      return;
    }
    op_ = asc_op_output.op_;
    output_index_ = asc_op_output.output_index;
    // 修改归属op的向量化轴信息
    AscTensorAttr::GetTensorAttr(op_, output_index_).vectorized_axis = vectorized_axis_;
  }

 private:
  std::vector<int64_t> vectorized_axis_;
  Operator *op_{nullptr};
  uint32_t output_index_{UINT32_MAX};
};

graphStatus AddEdgeForNode(const Operator &src_op, int32_t src_index, Operator &dst_op, int32_t dst_index);
graphStatus LinkByIrIndex(const Operator &src_op, uint32_t src_ir_index, Operator &dst_op, uint32_t dst_ir_index,
                          uint32_t dynamic_index = 0U);
graphStatus SetDynamicInputNumByIrIndex(Operator &op, uint32_t ir_index, uint32_t dynamic_num);

template <uint32_t INPUT_INDEX>
class AscOpInput {
 public:
  explicit AscOpInput(Operator *op) : op_(op) {}

  AscOpInput<INPUT_INDEX> &operator=(const AscOpOutput &output) {
    if (op_ == nullptr || output.op_ == nullptr) {
      GELOGE(FAILED, "Invalid op, make sure construct func is called.");
      return *this;
    }
    LinkByIrIndex(*output.op_, output.output_index, *this->op_, INPUT_INDEX);
    return *this;
  }

 private:
  Operator *op_;
};

struct AscTensor {
  explicit AscTensor(const OutDataAnchor &an) : attr(AscTensorAttr::GetTensorAttr(an)), anchor(an) {}
  ~AscTensor() = default;
  AscTensorAttr &attr;          // not owner
  const OutDataAnchor &anchor;  // not owner
};

struct AscNodeOutputs {
  friend class AscNode;
  AscTensor &operator[](uint32_t index);
  std::vector<AscTensor *> operator()();

 private:
  explicit AscNodeOutputs(Node *node) : node_(node) {
    Init();
  }
  void Init();
  std::vector<AscTensor> tensors_;
  Node *node_;
};

struct AscNodeInputs {
  friend class AscNode;
  AscTensor &operator[](uint32_t index);
  std::vector<AscTensor *> operator()();
  uint32_t Size();

 private:
  explicit AscNodeInputs(Node *node) : node_(node) {
    Init();
  }
  void Init();
  std::vector<AscTensor> tensors_;
  Node *node_;
};

class AscNode : public Node {
 public:
  AscNode(const OpDescPtr &op_desc, const ComputeGraphPtr &compute_graph);
  AscNodeInputs inputs;
  AscNodeOutputs outputs;
  AscNodeAttr &attr;
};
using AscNodePtr = std::shared_ptr<AscNode>;

class AscNodeIter {
 public:
  explicit AscNodeIter(ComputeGraph::Vistor<NodePtr>::Iterator &&iter);
  AscNodeIter &operator++();
  AscNodePtr operator*();
  bool operator!=(const AscNodeIter &other) const;

 private:
  ComputeGraph::Vistor<NodePtr>::Iterator impl_;
};

class AscNodeVisitor {
 public:
  using Iterator = ComputeGraph::Vistor<NodePtr>::Iterator;
  AscNodeIter begin();
  AscNodeIter end();
  explicit AscNodeVisitor(ComputeGraph::Vistor<NodePtr> &&visitor);

 private:
  ComputeGraph::Vistor<NodePtr> impl_;
};

template <uint32_t INPUT_INDEX>
class AscOpDynamicInput {
 public:
  explicit AscOpDynamicInput(Operator *op) : op_(op) {}
  AscOpDynamicInput<INPUT_INDEX> &operator=(const std::initializer_list<AscOpOutput> &outputs) {
    return AssignImpl(outputs);
  }
  AscOpDynamicInput<INPUT_INDEX> &operator=(const std::vector<AscOpOutput> &outputs) {
    return AssignImpl(outputs);
  }

 private:
  template <typename Container>
  AscOpDynamicInput<INPUT_INDEX> &AssignImpl(const Container &outputs) {
    if (op_ == nullptr) {
      GELOGE(FAILED, "op_ is null");
      return *this;
    }
    if (inited_) {
      AscendString op_name;
      (void)op_->GetName(op_name);
      GELOGE(FAILED, "It is not allowed to set the dynamic input repeatedly, node:[%s].", op_name.GetString());
      return *this;
    }
    const size_t input_nums = outputs.size();
    SetDynamicInputNumByIrIndex(*this->op_, INPUT_INDEX, input_nums);
    size_t idx = 0UL;
    for (const auto &output : outputs) {
      if (output.op_ == nullptr) {
        GELOGE(FAILED, "Src tensor is null");
        return *this;
      }
      LinkByIrIndex(*output.op_, output.output_index, *this->op_, INPUT_INDEX, idx++);
    }
    inited_ = true;
    return *this;
  }
  Operator *op_{nullptr};
  bool inited_{false};
};

class AscGraphImpl;
namespace ascir {
namespace cg {
class CodeGenUtils;
}
}  // namespace ascir
class AscGraph {
  friend class ascir::cg::CodeGenUtils;
  friend class AscGraphUtils;

 public:
  explicit AscGraph(const char *name);
  ~AscGraph();
  void SetTilingKey(const uint32_t tiling_key);
  int64_t GetTilingKey() const;
  void SetGraphType(const AscGraphType type);
  AscGraphType GetGraphType() const;
  graphStatus CreateSizeVar(const Expression &expression);
  Expression CreateSizeVar(const int64_t value);
  Expression CreateSizeVar(const std::string &name);
  Axis &CreateAxis(const std::string &name, const Expression &size);
  Axis &CreateAxis(const std::string &name, Axis::Type type, const Expression &size, const std::vector<AxisId> &from,
                   AxisId split_peer);
  Axis *FindAxis(const int64_t axis_id);
  AscNodePtr AddNode(Operator &op);
  AscNodePtr FindNode(const char *name) const;
  std::pair<AxisPtr, AxisPtr> BlockSplit(const int64_t axis_id, const std::string &outer_axis_name = "",
                                         const std::string &inner_axis_name = "");
  std::pair<AxisPtr, AxisPtr> TileSplit(const int64_t axis_id, const std::string &outer_axis_name = "",
                                        const std::string &inner_axis_name = "");
  AxisPtr MergeAxis(const std::vector<int64_t> &axis_ids, const std::string &merge_axis_name = "");
  bool BindBlock(const int64_t outter_id, const int64_t inner_id);
  bool ApplySplit(const AscNodePtr &node, const int64_t outter_id, const int64_t inner_id);
  bool ApplyMerge(const AscNodePtr &node, const int64_t merged_axis_id);
  bool ApplyReorder(const AscNodePtr &node, const std::vector<int64_t> &reordered_axis);
  bool ApplySchedAxisMerge(const AscNodePtr &node, const int64_t merged_axis_id, const std::vector<int64_t> &original);
  bool ApplySchedAxisMerge(const AscNodePtr &node, const int64_t merged_axis_id);
  bool ApplyTensorAxisMerge(const AscNodePtr &node, const int64_t merged_axis_id, const std::vector<int64_t> &original);
  bool ApplyTensorAxisMerge(const AscNodePtr &node, const int64_t merged_axis_id);
  bool ApplySchedAxisReorder(const AscNodePtr &node, const std::vector<int64_t> &reordered_axis);
  bool ApplyTensorAxisReorder(const AscNodePtr &node, const std::vector<int64_t> &reordered_axis);
  bool TryApplyAxisReplace(const AscNodePtr &node, const Axis &src, const Axis &dst);
  AscNodeVisitor GetAllNodes() const;
  AscNodeVisitor GetInputNodes() const;
  std::vector<SizeVarPtr> GetAllSizeVar() const;
  std::vector<AxisPtr> GetAllAxis() const;
  TransInfoRoadOfGraph GetAllAxisTransInfo() const;
  std::string GetName() const;
  bool CheckValid() const;

  AscOpOutput CreateContiguousData(const char *name, const ge::DataType &dt, const std::vector<Axis> &axes,
                                   const size_t index, const ge::Format &format = ge::FORMAT_ND);

  AscOpOutput CreateContiguousOut(const char *name, const ge::DataType &dt, const std::vector<Axis> &axes,
                                  const ge::Format &format = ge::FORMAT_ND);
  bool CopyFrom(const AscGraph &graph);
  bool CopyAttrFrom(const AscGraph &src_graph);
  static bool CopyAscNodeTensorAttr(const AscNodePtr &src_node, AscNodePtr &dst_node);
  Status AddSubGraph(const AscGraph &graph) const;
  Status FindSubGraph(const std::string &name, AscGraph &graph) const;
  Status GetAllSubGraphs(std::vector<AscGraph> &graphs) const;

 private:
  bool CheckExprValid() const;
  bool CheckAxisValid() const;
  bool CheckTensorValid() const;
  bool CheckNodeConnectionValid() const;
  std::shared_ptr<AscGraphImpl> impl_;
};
}  // namespace af

namespace ge {
using af::AddEdgeForNode;
using af::AscGraph;
using af::AscNode;
using af::AscNodeInputs;
using af::AscNodeIter;
using af::AscNodeOutputs;
using af::AscNodePtr;
using af::AscNodeVisitor;
using af::AscOpDynamicInput;
using af::AscOpInput;
using af::AscOpOutput;
using af::AscOutputAttrDataType;
using af::AscOutputAttrFormat;
using af::AscTensor;
using af::DiffAxesInfo;
using af::LinkByIrIndex;
using af::SetDynamicInputNumByIrIndex;
using af::VectorizedOutTensor;
using af::View;
}  // namespace ge

#endif  // GRAPH_ASCENDC_IR_H
