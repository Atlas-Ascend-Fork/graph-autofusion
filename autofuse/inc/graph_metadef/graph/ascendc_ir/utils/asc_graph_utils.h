/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_CXX_ASC_GRAPH_UTILS_H
#define METADEF_CXX_ASC_GRAPH_UTILS_H

#include "ascendc_ir/ascendc_ir_core/ascendc_ir.h"
#include "serialization/attr_serializer.h"
#include "proto/ascendc_ir.pb.h"

namespace af {
class AscGraphUtils {
 public:
  static ComputeGraphPtr GetComputeGraph(const AscGraph &asc_graph);
  /**
   * 在源数据锚点和目标数据锚点之间插入节点，仅维护数据边；从源节点继承调度，并从 src 对应输出继承
   * tensor 的 axis/repeats/strides 到插入节点指定输出。
   * @note [Autofuse 完备适配]
   */
  static graphStatus InsertNodeAfter(const OutDataAnchorPtr &src, const std::vector<InDataAnchorPtr> &dsts,
                                     const NodePtr &insert_node, const uint32_t input_index = 0U,
                                     const uint32_t output_index = 0U);
  /**
   * 在源数据锚点和其全部目标数据锚点之间插入节点，属性继承语义同指定 dsts 的重载。
   * @note [Autofuse 完备适配]
   */
  static graphStatus InsertNodeAfter(const OutDataAnchorPtr &src, const NodePtr &insert_node,
                                     const uint32_t input_index = 0U, const uint32_t output_index = 0U);
  /**
   * 通过 insert_op 创建节点后插入源数据锚点与目标数据锚点之间，属性继承语义同 NodePtr 重载。
   * @note [Autofuse 完备适配]
   */
  static NodePtr InsertNodeAfter(const OutDataAnchorPtr &src, const std::vector<InDataAnchorPtr> &dsts,
                                 const OpDescPtr &insert_op, const uint32_t input_index = 0U,
                                 const uint32_t output_index = 0U);
  /**
   * 在目标数据锚点前插入节点，仅维护数据边；从目标节点继承调度，并从原 src 对应输出继承 tensor 的
   * axis/repeats/strides 到插入节点指定输出。
   * @note [Autofuse 完备适配]
   */
  static graphStatus InsertNodeBefore(const InDataAnchorPtr &dst, const NodePtr &insert_node,
                                      const uint32_t input_index = 0U, const uint32_t output_index = 0U);
  /**
   * 通过 insert_op 创建节点后插入目标数据锚点前，属性继承语义同 NodePtr 重载。
   * @note [Autofuse 完备适配]
   */
  static NodePtr InsertNodeBefore(const InDataAnchorPtr &dst, const OpDescPtr &insert_op,
                                  const uint32_t input_index = 0U, const uint32_t output_index = 0U);
  static Status FromComputeGraph(const ComputeGraphPtr &compute_graph, AscGraph &graph);
  /**
   * @param compute_graph的node对象是Node类型时候，接口内部转换为AscNode
   * @return
   */
  static graphStatus ConvertComputeGraphToAscGraph(const ComputeGraphPtr &compute_graph, AscGraph &asc_graph);
  static graphStatus SerializeToBinary(const AscGraph &asc_graph, std::string &output);
  static graphStatus SerializeToReadable(const AscGraph &asc_graph, std::string &output);
  static graphStatus SerializeToProto(const AscGraph &asc_graph, ascendc_ir::proto::AscGraphDef &asc_graph_def);
  static graphStatus DeserializeFromBinary(const std::string &to_be_deserialized, AscGraph &out_asc_graph);
  static graphStatus DeserializeFromReadable(const std::string &to_be_deserialized, AscGraph &out_asc_graph);
  static graphStatus DeserializeFromProto(const ascendc_ir::proto::AscGraphDef &asc_graph_def, AscGraph &asc_graph);

 private:
  static graphStatus DeserializeGraphAttr(const AscGraph &asc_graph,
                                          const ascendc_ir::proto::AscGraphDef &asc_graph_def,
                                          const std::string &graph_name);
  static graphStatus DeserializeNodeOutputAttrs(const OpDescPtr &op_desc,
                                                const ascendc_ir::proto::AscNodeDef &asc_node_def);
  static graphStatus DeserializeNodeInputAttrs(const AscNodePtr &asc_node, const OpDescPtr &op_desc,
                                               const ascendc_ir::proto::AscNodeDef &asc_node_def,
                                               const AscGraph &asc_graph);
  static graphStatus DeserializeNodeDefinition(const AscNodePtr &asc_node,
                                               const ascendc_ir::proto::AscGraphDef &asc_graph_def,
                                               const int64_t node_index, const AscGraph &asc_graph);
};
class AscNodeSerializeUtils {
 public:
  static graphStatus SerializeIrDef(const AscNode &node, ascendc_ir::proto::IrDef &ir_def);
  static graphStatus SerializeAttrGroupsDef(const AscNode &node,
                                            ascendc_ir::proto::AscNodeAttrGroupsDef &asc_node_attr_groups_def);
};

class AscNodeDeserializeUtils {
 public:
  static graphStatus DeserializeIrDef(const ascendc_ir::proto::IrDef &ir_def, AscNode &node);
  static graphStatus DeserializeAttrGroupsDef(const ascendc_ir::proto::AscNodeAttrGroupsDef &asc_node_attr_groups_def,
                                              AscNode &node);
};
class ExpressionSerializer : public GeIrAttrSerializer {
 public:
  ExpressionSerializer() = default;
  graphStatus Serialize(const AnyValue &av, GeIrAttrDef &def) override;
  graphStatus Deserialize(const GeIrAttrDef &def, AnyValue &av) override;
};
}  // namespace af

#endif  // METADEF_CXX_ASC_GRAPH_UTILS_H
