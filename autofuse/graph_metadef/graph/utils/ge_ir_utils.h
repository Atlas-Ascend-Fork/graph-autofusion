/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMMON_GRAPH_UTILS_GE_IR_UTILS_H_
#define COMMON_GRAPH_UTILS_GE_IR_UTILS_H_

#include <google/protobuf/map.h>
#include <google/protobuf/repeated_field.h>
#include <google/protobuf/stubs/port.h>

#include <graph/anchor.h>
#include <graph/debug/ge_util.h>
#include <graph/detail/attributes_holder.h>
#include <graph/ge_tensor.h>
#include <graph/graph.h>
#include <graph/model.h>
#include <graph/node.h>
#include <graph/utils/graph_utils.h>
#include <graph/utils/type_utils.h>
#include <graph/types_af.h>
#include "normal_graph/ge_tensor_impl.h"

#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include "nlohmann/json.hpp"
#include "framework/common/debug/ge_log.h"
#include "proto/af_ir.pb.h"
#include "proto/onnx/af_onnx.pb.h"

namespace af {

///
///  @ingroup ge_ir_utils
///  @brief check, if not equal, log with tag
///  @param [in] const left_value, right_value reference, log_info_tag
///  @return bool
///
template <typename T>
bool IsEqual(const T &l_value, const T &r_value, const std::string &log_info_tag) {
  if ((l_value == r_value)) {
    return true;
  } else {
    GELOGD("Check not equal to %s", log_info_tag.c_str());
    return false;
  }
}

class OnnxUtils {
 public:
  static bool ConvertGeModelToModelProto(const af::Model &model, af::onnx::ModelProto &model_proto);

  static bool ConvertGeModelToModelProto(const af::Model &model, af::onnx::ModelProto &model_proto,
                                         DumpLevel dump_level);

 private:
  // Part 1: from IR convert to ONNX Protobuf
  static void AddAttrProto(af::onnx::NodeProto *const node_proto, const af::onnx::AttributeProto_AttributeType type,
                           const std::string &name, const void *const data);

  static void AddAttrProto(af::onnx::NodeProto *const node_proto, const af::onnx::AttributeProto_AttributeType type,
                           const std::string &name,
                           const ::google::protobuf::RepeatedField<::google::protobuf::int64> data);

  static void AddAttrProto(af::onnx::NodeProto *const node_proto, const af::onnx::AttributeProto_AttributeType type,
                           const std::string &name, const ::google::protobuf::RepeatedField<bool> data);

  static void AddAttrProto(af::onnx::NodeProto *const node_proto, const af::onnx::AttributeProto_AttributeType type,
                           const std::string &name, const ::google::protobuf::RepeatedField<float> data);

  static void AddAttrProto(af::onnx::NodeProto *const node_proto, const af::onnx::AttributeProto_AttributeType type,
                           const std::string &name, const ::google::protobuf::RepeatedPtrField<::std::string> data);

  static void AddListAttrProto(const std::string &attr_name, const ::af::proto::AttrDef &attr_def,
                               const std::string &prefix, const std::string &suffix, onnx::NodeProto *node_proto);

  static void AddAttrProtoFromNodeMembers(const NodePtr &node, af::onnx::NodeProto *const node_proto);

  static void AddAttrProtoFromAttribute(const std::pair<const std::string, ge::GeAttrValue> &string_attr_value,
                                        af::onnx::NodeProto *const node_proto);

  static void AddAttrProtoForOpInDesc(onnx::NodeProto *const node_proto, const OpDescPtr &op_desc);

  static void AddAttrProtoForOpOutDesc(onnx::NodeProto *const node_proto, const OpDescPtr &op_desc);

  static void AddAttrProtoForOpInAndOutDesc(af::onnx::NodeProto *const node_proto, const OpDescPtr &op_desc);

  static void AddAttrProtoForAttrsFromAttrMap(const ::google::protobuf::Map<std::string, af::proto::AttrDef> &attr_map,
                                              af::onnx::NodeProto *const node_proto, const std::string &prefix = "",
                                              const std::string &suffix = "");

  static af::onnx::TensorProto_DataType EncodeDataType(const ge::DataType data_type);

  static void EncodeNodeLinkForNetronVisual(const NodePtr &node, af::onnx::NodeProto *const node_proto);

  static bool EncodeNodeLink(const NodePtr &node, af::onnx::NodeProto *const node_proto);

  static bool EncodeNodeDesc(const NodePtr &node, af::onnx::NodeProto *const node_proto);

  static bool EncodeNode(const NodePtr &node, af::onnx::NodeProto *const node_proto);

  static void EncodeTypeProtoTensorType(const NodePtr &node, af::onnx::TypeProto_Tensor *const tensor_type);

  static void EncodeValueInfo(const NodePtr &node, af::onnx::ValueInfoProto *const value_info_proto);

  static bool EncodeGraph(const ConstComputeGraphPtr &graph, af::onnx::GraphProto *const graph_proto);

  /// Part 2: from ONNX Protobuf convert to IR
  /// Describes node's link relationships
  class NodeLinkInfo {
   public:
    NodeLinkInfo() = default;
    ~NodeLinkInfo() = default;
    NodeLinkInfo(std::string src_name, int32_t src_out_index, NodePtr dst_node, int32_t dst_in_index,
                 std::string dst_name)
        : src_node_name_(std::move(src_name)),
          src_out_index_(src_out_index),
          dst_node_(std::move(dst_node)),
          dst_in_index_(dst_in_index),
          dst_node_name_(std::move(dst_name)) {}

    std::string GetSrcNodeName() const {
      return src_node_name_;
    };
    int32_t GetSrcOutIndex() const {
      return src_out_index_;
    };
    NodePtr GetDstNode() const {
      return dst_node_;
    };
    int32_t GetDstInIndex() const {
      return dst_in_index_;
    };
    std::string GetDstNodeName() const {
      return dst_node_name_;
    };

   private:
    std::string src_node_name_;
    int32_t src_out_index_;
    NodePtr dst_node_;
    int32_t dst_in_index_;
    std::string dst_node_name_;
  };
  struct TensorDescToOnnxAttrHandler {
    std::string name;
    onnx::AttributeProto_AttributeType attr_type;
    using FuncCase0 = int64_t (*)(const GeTensorDescImpl::ExtMeta &);
    using FuncCase1 = std::string (*)(const GeTensorDescImpl::ExtMeta &);
    using FuncCase2 = std::vector<int64_t> (*)(const ConstGeTensorDescPtr &);
    using FuncCase3 = std::string (*)(const ConstGeTensorDescPtr &);
    union {
      FuncCase0 ext_meta_int_getter{nullptr};
      FuncCase1 ext_meta_str_getter;
      FuncCase2 member_ints_getter;
      FuncCase3 member_str_getter;
    };
    TensorDescToOnnxAttrHandler(std::string s, onnx::AttributeProto_AttributeType t, FuncCase3 func)
        : name(std::move(s)), attr_type(t), member_str_getter(func) {};
    TensorDescToOnnxAttrHandler(std::string s, onnx::AttributeProto_AttributeType t, FuncCase2 func)
        : name(std::move(s)), attr_type(t), member_ints_getter(func) {};
    TensorDescToOnnxAttrHandler(std::string s, onnx::AttributeProto_AttributeType t, FuncCase1 func)
        : name(std::move(s)), attr_type(t), ext_meta_str_getter(func) {};
    TensorDescToOnnxAttrHandler(std::string s, onnx::AttributeProto_AttributeType t, FuncCase0 func)
        : name(std::move(s)), attr_type(t), ext_meta_int_getter(func) {};
  };
  using TensordescAttrHandlers = std::vector<TensorDescToOnnxAttrHandler>;
  // Parse node name and index
  static bool ParseNameAndIndex(const std::string &node_name_index, std::string &node_name, int32_t &idx);

  static void DecodeAttribute(const af::onnx::AttributeProto &attr_proto, std::vector<std::string> &strings);

  static void DecodeAttribute(const af::onnx::AttributeProto &attr_proto, std::vector<int64_t> &ints);

  static void DecodeAttribute(const af::onnx::AttributeProto &attr_proto, int64_t &value);

  static void DecodeAttribute(const af::onnx::AttributeProto &attr_proto, std::string &value);

  static void DecodeNodeAttributeForOpOutDesc(const af::onnx::AttributeProto &attr_proto,
                                              const std::string &attr_name_for_output_desc, const int32_t index,
                                              const OpDescPtr &op_desc);

  static void DecodeNodeAttributeForOpInDesc(const af::onnx::AttributeProto &attr_proto,
                                             const std::string &attr_name_for_input_desc, const int32_t idx,
                                             const OpDescPtr &op_desc);

  static void DecodeNodeAttributeForOpInAndOutDesc(const af::onnx::AttributeProto &attr_proto,
                                                   const std::string &attr_name_for_input_output_desc,
                                                   const int32_t idx, const OpDescPtr &op_desc);

  static void DecodeNodeAttributeForOpDesc(const af::onnx::AttributeProto &attr_proto, OpDescPtr &op_desc);

  static bool DecodeNodeLinkImp(const NodeLinkInfo &item, const NodePtr &node_ptr);

  static bool DecodeNodeLink(const std::vector<af::onnx::NodeProto> &node_proto_vector,
                             const std::map<std::string, NodePtr> &node_map);

  static bool DecodeNodeDesc(const af::onnx::NodeProto *const node_proto, OpDescPtr &op_desc);

  static bool DecodeGraph(const int32_t recursion_depth, const af::onnx::GraphProto &graph_proto,
                          ComputeGraphPtr &graph);

  static void AddShapeFormatAndDtypeToJson(const af::ConstGeTensorDescPtr &desc, nlohmann::json &tensor_json);

  static void AddShapeFormatAndDtypeToProto(const af::ConstGeTensorDescPtr &desc, const std::string &prefix,
                                            const uint32_t idx, onnx::NodeProto *const node_proto);

  static void AddAllAttrToJson(const ConstGeTensorDescPtr &tensor_desc, nlohmann::json &tensor_json);

  static void AddAllAttrToProto(onnx::NodeProto *const node_proto, const ConstGeTensorDescPtr &tensor_desc,
                                const char_t *const prefix, const uint32_t idx);

  static void AddAllAttrGroupToJson(const ConstGeTensorDescPtr &tensor_desc, nlohmann::json &tensor_json);

  static void AddAllAttrGroupToProto(onnx::NodeProto *const node_proto, const ConstGeTensorDescPtr &tensor_desc,
                                     const char_t *const prefix, const uint32_t idx);

  static void AddCommonAttrIntoProto(onnx::NodeProto *const node_proto, const OpDescPtr &op_desc);
  static void AddCommonAttrGroupIntoProto(const OpDescPtr &op_desc, onnx::NodeProto *const node_proto);

  static bool AddInputAndOutputNodesForGraph(const onnx::GraphProto &graph_proto, ComputeGraphPtr &graph,
                                             const std::map<std::string, NodePtr> &node_map);
  template <typename DescGetter>
  static void ProcessTensorDescImpl(const OpDescPtr &op_desc, const string &desc_type, DescGetter desc_getter,
                                    onnx::NodeProto *node_proto);
  static void AddExtMetaToJson(const GeTensorDescImpl::ExtMeta &tensor_descriptor, nlohmann::json &tensor_json);
  static void AddExtMetaToProto(const GeTensorDescImpl::ExtMeta &tensor_descriptor, const std::string &prefix,
                                uint32_t index, onnx::NodeProto *node_proto);
  template <class T>
  static void AddJson(const std::string &name, nlohmann::json &json_holder, const T &json_obj) {
    try {
      json_holder[name] = json_obj;
    } catch (const std::exception &e) {
      GELOGW("Failed to init json object, err = %s, name = %s", e.what(), name.c_str());
      return;
    }
  }
  static DumpLevel dump_level_;
  static const TensordescAttrHandlers ext_meta_attr_handlers_;
  static const TensordescAttrHandlers normal_member_attr_handlers_;
};
}  // namespace af

#endif  // COMMON_GRAPH_UTILS_GE_IR_UTILS_H_
