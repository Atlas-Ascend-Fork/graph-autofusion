/* Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef TESTS_AUTOFUSE_FRAMEWORK_ASC_GRAPH_BUILDER_H_
#define TESTS_AUTOFUSE_FRAMEWORK_ASC_GRAPH_BUILDER_H_

#include "ascir_ops.h"
#include "graph/ascendc_ir/ascendc_ir_core/ascendc_ir.h"
#include "graph/ascendc_ir/ascendc_ir_core/ascendc_ir_def.h"
#include "graph/symbolizer/symbolic.h"
#include "graph/utils/graph_utils.h"
#include <string>
#include <vector>
#include <list>
#include <memory>
#include <cassert>

namespace af::testing {
inline Expression Sym(int64_t v) {
  return Symbol(v);
}
inline Expression Sym(const char *name) {
  return Symbol(name);
}

class AscGraphBuilder {
 public:
  explicit AscGraphBuilder(const std::string &name);

  ~AscGraphBuilder();

  AscGraphBuilder(const AscGraphBuilder &) = delete;

  AscGraphBuilder &operator=(const AscGraphBuilder &) = delete;

  AscGraphBuilder(AscGraphBuilder &&) = default;

  AscGraphBuilder &operator=(AscGraphBuilder &&) = delete;

  // 轴定义
  AscGraphBuilder &Loops(std::initializer_list<int64_t> sizes);

  AscGraphBuilder &Loops(std::initializer_list<Expression> sizes);

  AscGraphBuilder &Loops(const std::vector<Expression> &sizes);

  // 创建额外轴（用于混合 axis 组场景，如 Gather 的数据轴和索引轴）
  AxisId ExtraAxis(const std::string &name, const Expression &size);

  // buf 类节点
  AscGraphBuilder &Data(const std::string &name, int64_t index = 0, DataType dtype = af::DT_FLOAT);

  AscGraphBuilder &Data(const std::string &name, int64_t index, const std::vector<Expression> &shape,
                        const std::vector<Expression> &strides, DataType dtype = af::DT_FLOAT);

  // Data 支持自定义 axis（用于混合 axis 组场景）
  AscGraphBuilder &Data(const std::string &name, int64_t index, const std::vector<AxisId> &axes,
                        const std::vector<Expression> &shape, const std::vector<Expression> &strides,
                        DataType dtype = af::DT_FLOAT);

  AscGraphBuilder &Scalar(const std::string &name, const std::string &value, DataType dtype = af::DT_FLOAT);

  AscGraphBuilder &ScalarData(const std::string &name, int64_t index = 0, DataType dtype = af::DT_FLOAT);

  // IndexExpr 保持真实前端形态：scalar-like 节点输出视图为空（不补 axis/repeats/strides）。
  AscGraphBuilder &IndexExpr(const std::string &name, const std::string &expr, DataType dtype = af::DT_INT64);

  // Arange 输出视图由调用方显式给定（支持退化轴视图，如 {1, s1}/{0, 1} 的前缀扩维形态）。
  AscGraphBuilder &Arange(const std::string &name, const std::vector<Expression> &shape,
                          const std::vector<Expression> &strides, int64_t base = 0, int64_t step = 1,
                          DataType dtype = af::DT_INT64);
  AscGraphBuilder &Output(const std::string &name, const std::string &input, int64_t index = 0,
                          DataType dtype = af::DT_FLOAT);

  AscGraphBuilder &Workspace(const std::string &name, const std::string &input = "", DataType dtype = af::DT_FLOAT);

  // 搬运类 TODO 待补充nddma
  AscGraphBuilder &Load(const std::string &name, const std::string &input);

  AscGraphBuilder &Load(const std::string &name, const std::string &input, const std::vector<Expression> &shape,
                        const std::vector<Expression> &strides = {});

  AscGraphBuilder &Load(const std::string &name, const std::string &input, const std::vector<Expression> &shape,
                        const std::vector<Expression> &strides, const Expression &offset);

  AscGraphBuilder &Store(const std::string &name, const std::string &input);

  AscGraphBuilder &Store(const std::string &name, const std::string &input, const std::vector<Expression> &shape,
                         const std::vector<Expression> &strides);

  // Store with offset
  AscGraphBuilder &Store(const std::string &name, const std::string &input, const std::vector<Expression> &shape,
                         const std::vector<Expression> &strides, const Expression &offset);

  // broadcast
  AscGraphBuilder &Broadcast(const std::string &name, const std::string &input, const std::vector<int64_t> &brc_axes);

  AscGraphBuilder &Broadcast(const std::string &name, const std::string &input,
                             std::initializer_list<int64_t> brc_axes);

  AscGraphBuilder &Broadcast(const std::string &name, const std::string &input, const std::vector<Expression> &shape);

  // reduce - 通用接口
  template <typename ReduceOp>
  AscGraphBuilder &Reduce(const std::string &name, const std::string &input, const std::vector<size_t> &reduce_axes);

  AscGraphBuilder &Max(const std::string &name, const std::string &input, const std::vector<size_t> &reduce_axes) {
    return Reduce<ascir_op::Max>(name, input, reduce_axes);
  }

  AscGraphBuilder &Sum(const std::string &name, const std::string &input, const std::vector<size_t> &reduce_axes) {
    return Reduce<ascir_op::Sum>(name, input, reduce_axes);
  }

  AscGraphBuilder &Transpose(const std::string &name, const std::string &input, const std::vector<int64_t> &axes);

  AscGraphBuilder &Concat(const std::string &name, const std::vector<std::string> &inputs);

  // Concat with concat_dim: 自动计算输出 shape（沿 concat_dim 维求和）和 strides
  // 类似 tf.concat(values, axis)
  AscGraphBuilder &Concat(const std::string &name, const std::vector<std::string> &inputs, size_t concat_dim);

  // Concat with custom output shape/strides
  AscGraphBuilder &Concat(const std::string &name, const std::vector<std::string> &inputs,
                          const std::vector<Expression> &output_shape, const std::vector<Expression> &output_strides);

  // gather: data_input 和 index_input 使用各自的 axis，output_axis 合并两组轴
  AscGraphBuilder &Gather(const std::string &name, const std::string &data_input, const std::string &index_input,
                          int64_t gather_axis, const std::vector<AxisId> &output_axes,
                          const std::vector<Expression> &output_shape, const std::vector<Expression> &output_strides);

  // split: 动态输出算子，通过 SplitOutput 描述每个输出
  // Split 后可通过 "name:0", "name:1", ... 寻址各输出端口，连接下游节点
  struct SplitOutput {
    DataType dtype;
    std::vector<AxisId> axes;
    std::vector<Expression> repeats;
    std::vector<Expression> strides;
  };
  AscGraphBuilder &Split(const std::string &name, const std::string &input, const std::vector<SplitOutput> &outputs);

  // Frexp: 双输出异构 dtype 算子（mantissa: 浮点, exponent: int32）
  // 后续可通过 "name:0"(mantissa), "name:1"(exponent) 寻址各输出端口
  AscGraphBuilder &Frexp(const std::string &name, const std::string &input, DataType mantissa_dtype = af::DT_FLOAT,
                         DataType exponent_dtype = af::DT_INT32);

  // 常用的 elementwise 节点
  AscGraphBuilder &Abs(const std::string &name, const std::string &input) {
    return Op<ascir_op::Abs>(name, {input});
  }

  AscGraphBuilder &Sqrt(const std::string &name, const std::string &input) {
    return Op<ascir_op::Sqrt>(name, {input});
  }

  AscGraphBuilder &Exp(const std::string &name, const std::string &input) {
    return Op<ascir_op::Exp>(name, {input});
  }

  AscGraphBuilder &Relu(const std::string &name, const std::string &input) {
    return Op<ascir_op::Relu>(name, {input});
  }

  AscGraphBuilder &Neg(const std::string &name, const std::string &input) {
    return Op<ascir_op::Neg>(name, {input});
  }

  AscGraphBuilder &Cast(const std::string &name, const std::string &input, DataType dtype);

  AscGraphBuilder &Add(const std::string &name, const std::string &in1, const std::string &in2) {
    return Op<ascir_op::Add>(name, {in1, in2});
  }

  AscGraphBuilder &Sub(const std::string &name, const std::string &in1, const std::string &in2) {
    return Op<ascir_op::Sub>(name, {in1, in2});
  }

  AscGraphBuilder &Mul(const std::string &name, const std::string &in1, const std::string &in2) {
    return Op<ascir_op::Mul>(name, {in1, in2});
  }

  AscGraphBuilder &Div(const std::string &name, const std::string &in1, const std::string &in2) {
    return Op<ascir_op::Div>(name, {in1, in2});
  }

  AscGraphBuilder &Minimum(const std::string &name, const std::string &in1, const std::string &in2) {
    return Op<ascir_op::Minimum>(name, {in1, in2});
  }

  AscGraphBuilder &Maximum(const std::string &name, const std::string &in1, const std::string &in2) {
    return Op<ascir_op::Maximum>(name, {in1, in2});
  }

  AscGraphBuilder &Select(const std::string &name, const std::string &cond, const std::string &x,
                          const std::string &y) {
    return Op<ascir_op::Select>(name, {cond, x, y});
  }

  // 通用算子添加：自动从输入继承 tensor 属性
  template <typename OpType>
  AscGraphBuilder &Op(const std::string &name, const std::vector<std::string> &inputs, size_t follow_index = 0) {
    return AddOp<OpType>(name, inputs, follow_index);
  }

  // 通用算子添加：显式指定 tensor 属性
  template <typename OpType>
  AscGraphBuilder &Op(const std::string &name, const std::vector<std::string> &inputs,
                      const std::vector<Expression> &shape, const std::vector<Expression> &strides,
                      DataType dtype = af::DT_FLOAT, size_t follow_index = 0) {
    return AddOp<OpType>(name, inputs, shape, strides, dtype, follow_index);
  }

  AscGraph Build();

 private:
  struct Impl {
    std::string name_;
    AscGraph graph_;
    std::vector<AxisId> axis_ids_;
    std::vector<Expression> loop_repeats_;
    std::map<std::string, AscNodePtr> nodes_;
    // 存储 Operator 对象以支持动态输入算子（如 Concat）
    // 这些 Operator 对象需要在图的整个生命周期内保持有效
    std::list<std::vector<af::Operator>> dynamic_input_ops_;
    // 多输出节点的输出端口映射，如 Split 的 "split:0" → ("split", 0)
    std::map<std::string, std::pair<std::string, size_t>> output_ports_;

    explicit Impl(const std::string &name) : name_(name), graph_(name.c_str()) {}
  };

  std::unique_ptr<Impl> impl_;

  // 解析节点名和输出端口索引，支持 "name:port" 格式
  std::pair<AscNodePtr, size_t> ResolveOutput(const std::string &name) const {
    auto port_it = impl_->output_ports_.find(name);
    if (port_it != impl_->output_ports_.end()) {
      auto node_it = impl_->nodes_.find(port_it->second.first);
      assert(node_it != impl_->nodes_.end());
      return {node_it->second, port_it->second.second};
    }
    auto node_it = impl_->nodes_.find(name);
    assert(node_it != impl_->nodes_.end());
    return {node_it->second, 0};
  }

  const AscTensor &GetInputOutputTensor(const std::string &input) {
    auto [node, port] = ResolveOutput(input);
    assert(!node->outputs().empty());
    assert(port < node->outputs().size());
    return *node->outputs()[port];
  }

  template <typename OpType>
  AscNodePtr CreateNode(const std::string &name, OpType &op) {
    auto node = impl_->graph_.AddNode(op);
    impl_->nodes_[name] = node;
    return node;
  }

  template <typename OpType>
  AscNodePtr CreateNodeAndConnect(const std::string &name, OpType &op, const std::string &input,
                                  size_t input_index = 0) {
    auto node = impl_->graph_.AddNode(op);
    impl_->nodes_[name] = node;
    ConnectEdge(input, node, input_index);
    return node;
  }

  void ConnectEdge(const std::string &src_name, AscNodePtr dst_node, size_t dst_index = 0);

  AscGraphBuilder &BroadcastImpl(const std::string &name, const std::string &input,
                                 const std::vector<Expression> &output_shape);

  AscGraphBuilder &LoadImpl(const std::string &name, const std::string &input, const std::vector<Expression> *shape,
                            const std::vector<Expression> *strides, const Expression *offset = nullptr);

  AscGraphBuilder &StoreImpl(const std::string &name, const std::string &input, const std::vector<Expression> *shape,
                             const std::vector<Expression> *strides, const Expression *offset = nullptr);

  AscGraphBuilder &DataImpl(const std::string &name, int64_t index, const std::vector<AxisId> *axes,
                            const std::vector<Expression> *shape, const std::vector<Expression> *strides,
                            DataType dtype);

  // 通用添加算子的实现，默认 follow input[follow_index] 的 tensor 属性
  template <typename OpType>
  AscGraphBuilder &AddOp(const std::string &name, const std::vector<std::string> &inputs, size_t follow_index = 0) {
    auto op = OpType(name.c_str());

    op.attr.sched.axis = impl_->axis_ids_;
    if (!inputs.empty() && follow_index < inputs.size()) {
      auto [follow_node, follow_port] = ResolveOutput(inputs[follow_index]);
      assert(!follow_node->outputs().empty());
      assert(follow_port < follow_node->outputs().size());
      auto &follow_output = *follow_node->outputs()[follow_port];
      *op.y.axis = follow_output.attr.axis;
      *op.y.repeats = follow_output.attr.repeats;
      op.y.dtype = follow_output.attr.dtype;
    }

    auto node = impl_->graph_.AddNode(op);
    impl_->nodes_[name] = node;

    for (size_t i = 0; i < inputs.size(); ++i) {
      auto [src_node, src_port] = ResolveOutput(inputs[i]);
      assert(src_port < src_node->GetAllOutDataAnchors().size());
      GraphUtils::AddEdge(src_node->GetOutDataAnchor(src_port), node->GetInDataAnchor(i));
    }

    return *this;
  }

  template <typename OpType>
  AscGraphBuilder &AddOp(const std::string &name, const std::vector<std::string> &inputs,
                         const std::vector<Expression> &shape, const std::vector<Expression> &strides, DataType dtype,
                         size_t follow_index) {
    auto op = OpType(name.c_str());

    op.attr.sched.axis = impl_->axis_ids_;
    *op.y.repeats = shape;
    *op.y.strides = strides;
    op.y.dtype = dtype;
    if (!inputs.empty() && follow_index < inputs.size()) {
      auto [follow_node, follow_port] = ResolveOutput(inputs[follow_index]);
      assert(!follow_node->outputs().empty());
      assert(follow_port < follow_node->outputs().size());
      *op.y.axis = follow_node->outputs()[follow_port]->attr.axis;
    }

    auto node = impl_->graph_.AddNode(op);
    impl_->nodes_[name] = node;

    for (size_t i = 0; i < inputs.size(); ++i) {
      auto [src_node, src_port] = ResolveOutput(inputs[i]);
      assert(src_port < src_node->GetAllOutDataAnchors().size());
      GraphUtils::AddEdge(src_node->GetOutDataAnchor(src_port), node->GetInDataAnchor(i));
    }

    return *this;
  }
};
}  // namespace af::testing

#endif  // TESTS_AUTOFUSE_FRAMEWORK_ASC_GRAPH_BUILDER_H_
