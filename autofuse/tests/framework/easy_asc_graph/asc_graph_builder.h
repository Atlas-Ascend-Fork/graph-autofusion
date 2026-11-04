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

  // 运行期尺寸符号。`Loops({Sym(128)})` 建的是**常量**（`Symbol(128)`），图上没有 size
  // var；要让某一维是运行期值，得先在这里登记一个名字，再把它当轴尺寸传给 `Loops`。
  // 它同时会成为 host 侧 TilingData 的一个字段——调用方在求解 tiling 之前必须填好它。
  Expression SizeVar(const std::string &name);

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
  AscGraphBuilder &IndexExpr(const std::string &name, const Expression &expr, DataType dtype = af::DT_INT64);

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

  AscGraphBuilder &Ln(const std::string &name, const std::string &input) {
    return Op<ascir_op::Ln>(name, {input});
  }

  AscGraphBuilder &Relu(const std::string &name, const std::string &input) {
    return Op<ascir_op::Relu>(name, {input});
  }

  AscGraphBuilder &Neg(const std::string &name, const std::string &input) {
    return Op<ascir_op::Neg>(name, {input});
  }

  AscGraphBuilder &Reciprocal(const std::string &name, const std::string &input) {
    return Op<ascir_op::Reciprocal>(name, {input});
  }

  AscGraphBuilder &Sigmoid(const std::string &name, const std::string &input) {
    return Op<ascir_op::Sigmoid>(name, {input});
  }

  AscGraphBuilder &Gelu(const std::string &name, const std::string &input) {
    return Op<ascir_op::Gelu>(name, {input});
  }

  AscGraphBuilder &Sign(const std::string &name, const std::string &input) {
    return Op<ascir_op::Sign>(name, {input});
  }

  AscGraphBuilder &Tanh(const std::string &name, const std::string &input) {
    return Op<ascir_op::Tanh>(name, {input});
  }

  AscGraphBuilder &Erf(const std::string &name, const std::string &input) {
    return Op<ascir_op::Erf>(name, {input});
  }

  AscGraphBuilder &Rsqrt(const std::string &name, const std::string &input) {
    return Op<ascir_op::Rsqrt>(name, {input});
  }

  // `LogicalNot` 的输出 dtype 在两侧不一样（mem 侧跟着输入，reg 侧恒为 uint8），而
  // `Op<>` 那条通道是「跟着 `follow_index` 那个输入」，说不出 reg 侧那一档。CPU harness
  // 钉的是 Ascend910B1，走的正是 mem 侧，所以这里用通用通道；reg 侧由
  // afir-logical-not-reg-base-extension-api.mlir 钉。
  AscGraphBuilder &LogicalNot(const std::string &name, const std::string &input) {
    return Op<ascir_op::LogicalNot>(name, {input});
  }

  AscGraphBuilder &Cast(const std::string &name, const std::string &input, DataType dtype);

  // `negative_slope` 是 LeakyRelu 自己的 `ir_attr`，通用 `Op<>` 通道碰不到它。
  AscGraphBuilder &LeakyRelu(const std::string &name, const std::string &input, float negative_slope) {
    return AddConfiguredOp<ascir_op::LeakyRelu>(
        name, {input}, /*follow_index=*/0U,
        [negative_slope](ascir_op::LeakyRelu &op) { op.ir_attr.SetNegative_slope(negative_slope); });
  }

  // `alpha` 是 Axpy 自己的 `ir_attr`，与 `LeakyRelu` 的 `negative_slope` 同一条路：
  // 通用 `Op<>` 通道碰不到它，得在 `AddNode` 之前设好。
  AscGraphBuilder &Axpy(const std::string &name, const std::string &in1, const std::string &in2, float alpha) {
    return AddConfiguredOp<ascir_op::Axpy>(name, {in1, in2}, /*follow_index=*/0U,
                                           [alpha](ascir_op::Axpy &op) { op.ir_attr.SetAlpha(alpha); });
  }

  // 浮点谓词族：输入浮点、输出一个字节的真值，所以输出 dtype 不能跟着输入走。
  AscGraphBuilder &Isnan(const std::string &name, const std::string &input);
  AscGraphBuilder &IsFinite(const std::string &name, const std::string &input);
  AscGraphBuilder &IsInf(const std::string &name, const std::string &input);

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

  // 真除。输出 dtype 是一张 (输入, 输出) 对表（`kTrueDivTypePairs`）：i32 的商是 f32，
  // f16 / f32 跟着输入。通用 `Op<>` 通道只会把输入的 dtype 继承过来，i32 那一行说不
  // 出来，所以走 `AddConfiguredOp` —— 回调看到的 `op.y.dtype` 已经是继承好的那个。
  AscGraphBuilder &TrueDiv(const std::string &name, const std::string &in1, const std::string &in2) {
    return AddConfiguredOp<ascir_op::TrueDiv>(name, {in1, in2}, /*follow_index=*/0U, [](ascir_op::TrueDiv &op) {
      if (op.y.dtype == af::DT_INT32) {
        op.y.dtype = af::DT_FLOAT;
      }
    });
  }

  AscGraphBuilder &Minimum(const std::string &name, const std::string &in1, const std::string &in2) {
    return Op<ascir_op::Minimum>(name, {in1, in2});
  }

  AscGraphBuilder &Maximum(const std::string &name, const std::string &in1, const std::string &in2) {
    return Op<ascir_op::Maximum>(name, {in1, in2});
  }

  // 位模式一族。dtype 跟着输入走（DT_INT16 / DT_UINT16 / DT_INT32 / DT_UINT8），与算术
  // 那几个的通道完全一样——列在这里只是为了不让调用方去记 `ascir_op::` 的名字。
  AscGraphBuilder &BitwiseAnd(const std::string &name, const std::string &in1, const std::string &in2) {
    return Op<ascir_op::BitwiseAnd>(name, {in1, in2});
  }

  AscGraphBuilder &FloorDiv(const std::string &name, const std::string &in1, const std::string &in2) {
    return Op<ascir_op::FloorDiv>(name, {in1, in2});
  }

  AscGraphBuilder &Remainder(const std::string &name, const std::string &in1, const std::string &in2) {
    return Op<ascir_op::Remainder>(name, {in1, in2});
  }

  AscGraphBuilder &Pow(const std::string &name, const std::string &in1, const std::string &in2) {
    return Op<ascir_op::Pow>(name, {in1, in2});
  }

  // 逻辑族：两个输入同 dtype，输出恒为 `ui8`——谓词族那个形状的二元版本。
  //
  // 走 `AddConfiguredOp` 而不是 `Op<>`：通用通道把 `y.dtype` 从 `follow_index` 那个输入
  // 继承过来，对这一族是错的。回调在 `AddNode` 之前覆写它，axis / repeats 仍然照常继承
  // （两个输入同形，跟哪个都一样）。
  AscGraphBuilder &LogicalAnd(const std::string &name, const std::string &in1, const std::string &in2) {
    return AddConfiguredOp<ascir_op::LogicalAnd>(name, {in1, in2}, /*follow_index=*/0U,
                                                 [](ascir_op::LogicalAnd &op) { op.y.dtype = af::DT_UINT8; });
  }

  AscGraphBuilder &LogicalOr(const std::string &name, const std::string &in1, const std::string &in2) {
    return AddConfiguredOp<ascir_op::LogicalOr>(name, {in1, in2}, /*follow_index=*/0U,
                                                [](ascir_op::LogicalOr &op) { op.y.dtype = af::DT_UINT8; });
  }

  AscGraphBuilder &LogicalXor(const std::string &name, const std::string &in1, const std::string &in2) {
    return AddConfiguredOp<ascir_op::LogicalXor>(name, {in1, in2}, /*follow_index=*/0U,
                                                 [](ascir_op::LogicalXor &op) { op.y.dtype = af::DT_UINT8; });
  }

  // 比较族：与逻辑族逐字同一个形状——两个输入同 dtype，输出恒为 `ui8`，所以同样不能走
  // 通用 `Op<>` 通道（它会把 `y.dtype` 从输入继承过来）。六个只差 `ascir_op::` 的名字，
  // 所以这里是一个模板加六行；`CMPMODE` 不在 AscGraph 上，它由节点类型定死。
  template <typename OpType>
  AscGraphBuilder &CompareOp(const std::string &name, const std::string &in1, const std::string &in2) {
    return AddConfiguredOp<OpType>(name, {in1, in2}, /*follow_index=*/0U,
                                   [](OpType &op) { op.y.dtype = af::DT_UINT8; });
  }

  AscGraphBuilder &Eq(const std::string &name, const std::string &in1, const std::string &in2) {
    return CompareOp<ascir_op::Eq>(name, in1, in2);
  }

  AscGraphBuilder &Ne(const std::string &name, const std::string &in1, const std::string &in2) {
    return CompareOp<ascir_op::Ne>(name, in1, in2);
  }

  AscGraphBuilder &Lt(const std::string &name, const std::string &in1, const std::string &in2) {
    return CompareOp<ascir_op::Lt>(name, in1, in2);
  }

  AscGraphBuilder &Le(const std::string &name, const std::string &in1, const std::string &in2) {
    return CompareOp<ascir_op::Le>(name, in1, in2);
  }

  AscGraphBuilder &Gt(const std::string &name, const std::string &in1, const std::string &in2) {
    return CompareOp<ascir_op::Gt>(name, in1, in2);
  }

  AscGraphBuilder &Ge(const std::string &name, const std::string &in1, const std::string &in2) {
    return CompareOp<ascir_op::Ge>(name, in1, in2);
  }

  // 输出属性跟**第二个**输入（x），不是第一个。第一个是 ui8 真值掩码，跟它继承出来的
  // 输出 dtype 就是 ui8，而 Select 的输出与 x / y 同 dtype。
  AscGraphBuilder &Select(const std::string &name, const std::string &cond, const std::string &x,
                          const std::string &y) {
    return Op<ascir_op::Select>(name, {cond, x, y}, /*follow_index=*/1U);
  }

  // `Where` 与 `Select` 的形状、操作数顺序与 `follow_index` 逐字相同（两侧供的也是同一个
  // 包装：mem 侧 `Where`、reg 侧 `WhereExtend`），差别只在节点类型——而节点类型是
  // bridge 与 VF 下降查表的那一位，所以两个要分开建得出来。
  AscGraphBuilder &Where(const std::string &name, const std::string &cond, const std::string &x, const std::string &y) {
    return Op<ascir_op::Where>(name, {cond, x, y}, /*follow_index=*/1U);
  }

  // 通用算子添加：自动从输入继承 tensor 属性
  template <typename OpType>
  AscGraphBuilder &Op(const std::string &name, const std::vector<std::string> &inputs, size_t follow_index = 0) {
    return AddOp<OpType>(name, inputs, follow_index);
  }

  // 轴从 `follow_index` 继承，但输出 dtype 另给。
  //
  // `follow_index` 把「轴从哪来」和「dtype 从哪来」绑在了一个参数上，而有的图要把两者分开：
  // `afir.where` 的掩码是唯一带轴的输入（两个分支都是 UB 上的 Scalar），所以轴只能跟掩码；
  // 而它的 verifier 又要求两个分支与结果同 dtype，也就是不能跟掩码。指向 Scalar 会让输出拿到
  // 空轴集（候选枚举按 elementwise-broadcast 判据拒收），把 Scalar 改成掩码的 dtype 又会撞上
  // 「ui8 不能作标量操作数」。这个通道是那一类图的出口。
  template <typename OpType>
  AscGraphBuilder &OpWithDtype(const std::string &name, const std::vector<std::string> &inputs, size_t follow_index,
                               DataType dtype) {
    return AddConfiguredOp<OpType>(name, inputs, follow_index, [dtype](OpType &op) { op.y.dtype = dtype; });
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
    return AddConfiguredOp<OpType>(name, inputs, follow_index, [](OpType &) {});
  }

  // 同上，外加一个在 `AddNode` **之前**作用在 op 上的回调。带自有 `ir_attr` 的算子
  // （`LeakyRelu` 的 `negative_slope`）要用它：属性得在节点建进图之前设好。
  template <typename OpType, typename Configure>
  AscGraphBuilder &AddConfiguredOp(const std::string &name, const std::vector<std::string> &inputs, size_t follow_index,
                                   Configure &&configure) {
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
    configure(op);

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
