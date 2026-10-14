/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "pyascir.h"

#include <limits>
#include <Python.h>
#include <structmember.h>
#include <google/protobuf/text_format.h>

#include "graph/operator_af.h"
#include "graph/ascendc_ir/utils/asc_graph_utils.h"
#include "graph/detail/model_serialize_imp.h"
#include "graph/utils/graph_utils.h"
#include "fusion/autofuse_attrs.h"

#include "ascendc_ir.h"
#include "ascir_ops.h"
#include "ascir_utils.h"
#include "graph/symbolizer/symbolic.h"
#include "graph/def_types.h"
#include "graph_utils_ex.h"
#include "pyascir_common_utils.h"
#include "pyascir_types.h"
#include "ascgen_log.h"
#include "ascgraph_info_complete.h"

namespace {
inline constexpr char kAscBackendType[] = "AscBackend";
inline constexpr char kIndexAttr[] = "index";
inline constexpr char kValueAttr[] = "value";
inline constexpr char kExprAttr[] = "expr";
inline constexpr char kBaseAttr[] = "base";
inline constexpr char kStepAttr[] = "step";
inline constexpr char kOffsetAttr[] = "offset";
inline constexpr char kAxisAttr[] = "axis";
inline constexpr char kHasRelu[] = "has_relu";
inline constexpr char kOffsetX[] = "offset_x";
inline constexpr char kTransposeX1[] = "transpose_x1";
inline constexpr char kTransposeX2[] = "transpose_x2";
inline constexpr char kAdjX1[] = "adj_x1";
inline constexpr char kAdjX2[] = "adj_x2";
inline constexpr char kEnableHf32[] = "enable_hf32";
inline constexpr char kOutputOpType[] = "Output";
inline constexpr char kDataOpType[] = "Data";
inline constexpr char kScalarDataOpType[] = "ScalarData";
inline constexpr char kAscGraphAttr[] = "ascgraph";
inline constexpr char kNegativeSlopeAttr[] = "negative_slope";
inline constexpr char kNegativeIndexSupportAttr[] = "negative_index_support";
inline constexpr char kNeedCheckBoundAttr[] = "need_check_bound";
inline constexpr char kMaxAttr[] = "max";
inline constexpr char kAlphaAttr[] = "alpha";
inline constexpr char kNAttr[] = "n";
inline constexpr char kErrorMsgAttr[] = "error_msg";
struct DTypeEntry {
  const char *py_name{nullptr};
  int64_t dtype_value{-1};
};
static const DTypeEntry dtype_entries[] = {
    {"float32", ge::DT_FLOAT},
    {"float16", ge::DT_FLOAT16},
    {"int8", ge::DT_INT8},
    {"int32", ge::DT_INT32},
    {"uint8", ge::DT_UINT8},
    {"int16", ge::DT_INT16},
    {"uint16", ge::DT_UINT16},
    {"uint32", ge::DT_UINT32},
    {"int64", ge::DT_INT64},
    {"uint64", ge::DT_UINT64},
    {"double", ge::DT_DOUBLE},
    {"bool", ge::DT_BOOL},
    {"string", ge::DT_STRING},
    {"dual_sub_int8", ge::DT_DUAL_SUB_INT8},
    {"dual_sub_uint8", ge::DT_DUAL_SUB_UINT8},
    {"complex64", ge::DT_COMPLEX64},
    {"complex128", ge::DT_COMPLEX128},
    {"qint8", ge::DT_QINT8},
    {"qint16", ge::DT_QINT16},
    {"qint32", ge::DT_QINT32},
    {"quint8", ge::DT_QUINT8},
    {"quint16", ge::DT_QUINT16},
    {"resource", ge::DT_RESOURCE},
    {"string_ref", ge::DT_STRING_REF},
    {"dual", ge::DT_DUAL},
    {"variant", ge::DT_VARIANT},
    {"bfloat16", ge::DT_BF16},
    {"bf16", ge::DT_BF16},
    {"undefined", ge::DT_UNDEFINED},
    {"int4", ge::DT_INT4},
    {"uint1", ge::DT_UINT1},
    {"int2", ge::DT_INT2},
    {"uint2", ge::DT_UINT2},
    {"complex32", ge::DT_COMPLEX32},
    {"hifloat8", ge::DT_HIFLOAT8},
    {"float8_e5m2", ge::DT_FLOAT8_E5M2},
    {"float8_e4m3fn", ge::DT_FLOAT8_E4M3FN},
    {"float8_e8m0", ge::DT_FLOAT8_E8M0},
    {"float6_e3m2", ge::DT_FLOAT6_E3M2},
    {"float6_e2m3", ge::DT_FLOAT6_E2M3},
    {"float4_e2m1", ge::DT_FLOAT4_E2M1},
    {"float4_e1m2", ge::DT_FLOAT4_E1M2},
};
// 这里定义一些非ascir的geir op但是要在ascir.ops的模块中暴露（对外不体现geir.ops）
namespace geir_op {
struct AscBackend : public af::Operator {
  inline explicit AscBackend(const char *name) : af::Operator(name, kAscBackendType) {
    this->DynamicInputRegister("x", 0U, true);
    this->DynamicOutputRegister("y", 0U, true);
  }
};
struct AscGraph : public af::Operator {
  static constexpr const char *Type = "AscGraph";
  inline explicit AscGraph(const char *name) : af::Operator(name, Type) {
    this->DynamicInputRegister("x", 0U, true);
    this->DynamicOutputRegister("y", 0U, true);
  }
};
}  // namespace geir_op
template <typename GraphObj>
bool CountInputsOutputs(GraphObj *graph, size_t &input_size, std::set<int64_t> &outputs) {
  for (const auto &node : graph->GetAllNodes()) {
    PY_ASSERT_NOTNULL(node);
    if (node->GetType() == kDataOpType || node->GetType() == kScalarDataOpType) {
      ++input_size;
    }
    if (node->GetType() == kOutputOpType) {
      int64_t index{-1};
      PY_ASSERT_NOTNULL(node->attr.ir_attr, "Op %s %s must set ir_attr for index.", node->GetNamePtr(),
                        node->GetTypePtr());
      auto ir_attr_of_output = node->attr.ir_attr->template DownCastTo<af::ascir_op::Output::AscOutputIrAttrDef>();
      PY_ASSERT_NOTNULL(ir_attr_of_output, "Op %s %s must set ir_attr for index.", node->GetNamePtr(),
                        node->GetTypePtr());
      PY_ASSERT_GRAPH_SUCCESS(ir_attr_of_output->GetIndex(index), "Op %s %s get ir_attr for index %ld failed.",
                              node->GetNamePtr(), node->GetTypePtr(), index);
      outputs.insert(index);
    }
  }
  return true;
}

template <typename OpT>
struct GraphHandler {};

uint64_t InferFuseType(const af::AscGraph &graph) {
  static const std::map<af::ComputeType, ge::loop::FuseType> kComputeTypeToFuseType = {
      {af::ComputeType::kComputeElewise, ge::loop::FuseType::kPointwise},
      {af::ComputeType::kComputeReduce, ge::loop::FuseType::kReduction},
      {af::ComputeType::kComputeConcat, ge::loop::FuseType::kConcat},
      {af::ComputeType::kComputeTranspose, ge::loop::FuseType::kTranspose},
      {af::ComputeType::kComputeGather, ge::loop::FuseType::kGather},
      {af::ComputeType::kComputeSplit, ge::loop::FuseType::kSplit},
      {af::ComputeType::kComputeCube, ge::loop::FuseType::kCube}};

  uint64_t fuse_type = (1UL << static_cast<uint64_t>(ge::loop::FuseType::kExtern));
  if (optimize::AscGraphInfoComplete::CompleteApiInfo(graph) == af::SUCCESS) {
    for (const auto &node : graph.GetAllNodes()) {
      auto it = kComputeTypeToFuseType.find(node->attr.api.compute_type);
      if (it != kComputeTypeToFuseType.end()) {
        fuse_type |= (1UL << static_cast<uint64_t>(it->second));
      }
    }
  }
  return fuse_type;
}

template <>
struct GraphHandler<geir_op::AscBackend> {
  static bool Handle(geir_op::AscBackend *op, af::AscGraph *graph) {
    auto op_desc = af::OpDescUtils::GetOpDescFromOperator(*op);
    PY_ASSERT_NOTNULL(op_desc);
    auto auto_fuse_attr = op_desc->GetOrCreateAttrsGroup<af::AutoFuseAttrs>();
    PY_ASSERT_NOTNULL(auto_fuse_attr);
    std::shared_ptr<af::AscGraph> asc_graph_ptr(graph, [](af::AscGraph *) {});
    auto_fuse_attr->SetAscGraph(asc_graph_ptr, auto_fuse_attr->GetFuseType());
    GetInterAttrs(auto_fuse_attr).fuse_type = InferFuseType(*graph);
    return true;
  }
};

template <>
struct GraphHandler<geir_op::AscGraph> {
  static bool Handle(geir_op::AscGraph *op, af::AscGraph *graph) {
    std::string asc_graph_str;
    PY_ASSERT_GRAPH_SUCCESS(af::AscGraphUtils::SerializeToReadable(*graph, asc_graph_str));
    auto op_desc = af::OpDescUtils::GetOpDescFromOperator(*op);
    PY_ASSERT_NOTNULL(op_desc);
    PY_ASSERT(af::AttrUtils::SetStr(op_desc, kAscGraphAttr, asc_graph_str));
    return true;
  }
};
}  // namespace
/** OpsOperatorAttr include sched api */
namespace pyascir {
class ApiInfo {
 public:
  struct Object {
    PyObject_HEAD af::ApiInfo *api_info;
  };
  static PyTypeObject type;
  static PyGetSetDef GetSetters[];
  static void Dealloc(PyObject *self);
  static PyObject *FromAscNode(af::AscNodeAttr &node_attr);
  static PyObject *get(PyObject *self, void *closure);
  static int set(PyObject *self, PyObject *value, void *closure);
};

void ApiInfo::Dealloc(PyObject *self) {
  Py_TYPE(self)->tp_free(self);
}

int ApiInfo::set(PyObject *self, PyObject *value, void *closure) {
  (void)closure;
  if (value == nullptr) {
    PyErr_SetString(PyExc_ValueError, "value is null");
    return -1;
  }
  const char *val = PyUnicode_AsUTF8(value);
  if (val == nullptr) {
    PyErr_SetString(PyExc_TypeError, "expected a string");
    return -1;
  }
  static const map<string, af::ComputeType> name_to_type = {
      {"load", af::ComputeType::kComputeLoad},           {"gather", af::ComputeType::kComputeGather},
      {"store", af::ComputeType::kComputeStore},         {"elemwise", af::ComputeType::kComputeElewise},
      {"broadcast", af::ComputeType::kComputeBroadcast}, {"reduce", af::ComputeType::kComputeReduce},
      {"transpose", af::ComputeType::kComputeTranspose}, {"concat", af::ComputeType::kComputeConcat},
      {"cube", af::ComputeType::kComputeCube},           {"split", af::ComputeType::kComputeSplit},
  };

  auto type_iter = name_to_type.find(string(val));
  if (type_iter == name_to_type.end()) {
    PyErr_SetString(PyExc_ValueError, "unknown compute type");
    return -1;
  }

  auto hint_type = ge::PtrToPtr<PyObject, ApiInfo::Object>(self);
  hint_type->api_info->compute_type = type_iter->second;
  return 0;
}

PyObject *ApiInfo::get(PyObject *self, void *closure) {
  (void)closure;
  auto hint_type = ge::PtrToPtr<PyObject, ApiInfo::Object>(self);
  if (hint_type->api_info == nullptr) {
    return PyUnicode_FromString("unknown");
  }
  static const map<af::ComputeType, string> type_to_name = {
      {af::ComputeType::kComputeLoad, "load"},
      {af::ComputeType::kComputeStore, "store"},
      {af::ComputeType::kComputeReduceStore, "reduce_store"},
      {af::ComputeType::kComputeElewise, "elewise"},
      {af::ComputeType::kComputeBroadcast, "broadcast"},
      {af::ComputeType::kComputeReduce, "reduce"},
      {af::ComputeType::kComputeTranspose, "transpose"},
      {af::ComputeType::kComputeConcat, "concat"},
      {af::ComputeType::kComputeGather, "gather"},
      {af::ComputeType::kComputeSplit, "split"},
      {af::ComputeType::kComputeCube, "cube"},
  };

  auto type_iter = type_to_name.find(hint_type->api_info->compute_type);
  if (type_iter == type_to_name.end()) {
    return PyUnicode_FromString("unknown");
  }
  return PyUnicode_FromString(type_iter->second.c_str());
}

PyObject *ApiInfo::FromAscNode(af::AscNodeAttr &node_attr) {
  auto hint_type = ge::PtrToPtr<PyObject, ApiInfo::Object>(ApiInfo::type.tp_alloc(&ApiInfo::type, 0));
  PY_ASSERT_NOTNULL(hint_type);
  hint_type->api_info = &node_attr.api;
  auto hint_py_obj = ge::PtrToPtr<ApiInfo::Object, PyObject>(hint_type);
  Py_IncRef(hint_py_obj);
  return hint_py_obj;
}

PyGetSetDef ApiInfo::GetSetters[] = {
    {"compute_type", ApiInfo::get, ApiInfo::set, "Compute type.", nullptr}, {nullptr} /* Sentinel */
};

PyTypeObject ApiInfo::type = {PyVarObject_HEAD_INIT(nullptr, 0)};

class SchedInfo {
 public:
  struct Object {
    PyObject_HEAD std::vector<int64_t> *axis{nullptr};
    int64_t *loop_axis{nullptr};
  };
  static PyTypeObject type;
  static PyGetSetDef GetSetters[];
  static void Dealloc(PyObject *self);
  static PyObject *get_axis(PyObject *self, void *closure);
  static int set_axis(PyObject *self, PyObject *value, void *closure);

  static PyObject *FromAscNode(af::AscNodeAttr &node_attr);
  static PyObject *FromGraph(af::AscGraphAttr &graph_attr);
};

void SchedInfo::Dealloc(PyObject *self) {
  Py_TYPE(self)->tp_free(self);
}

PyObject *SchedInfo::get_axis(PyObject *self, void *closure) {
  (void)closure;
  auto sched_info = ge::PtrToPtr<PyObject, SchedInfo::Object>(self);
  PY_ASSERT_NOTNULL(sched_info);
  PY_ASSERT_NOTNULL(sched_info->axis, "sched attr has not been inited.");
  const auto &axis = *sched_info->axis;
  auto list = PyList_New(axis.size());
  for (size_t i = 0UL; i < axis.size(); ++i) {
    PyList_SetItem(list, i, PyLong_FromLong(axis[i]));
  }

  return list;
}

int SchedInfo::set_axis(PyObject *self, PyObject *value, void *closure) {
  (void)closure;
  auto sched_info = ge::PtrToPtr<PyObject, SchedInfo::Object>(self);
  if (sched_info->axis == nullptr) {
    PyErr_SetString(PyExc_ValueError, "sched attr has not been inited.");
    return -1;
  }
  if (PyList_Check(value) == kPythonFail) {
    PyErr_SetString(PyExc_TypeError, "value must be a list");
    return -1;
  }
  sched_info->axis->resize(PyList_Size(value));
  for (int i = 0; i < PyList_Size(value); ++i) {
    auto axis = ge::PtrToPtr<PyObject, Axis::Object>(PyList_GetItem(value, i));
    if (PyObject_IsInstance(ge::PtrToPtr<Axis::Object, PyObject>(axis),
                            ge::PtrToPtr<PyTypeObject, PyObject>(&Axis::type)) == 0) {
      PyErr_Format(PyExc_ValueError, "axis on %d is not Axis type", i);
      return -1;
    }
    (*sched_info->axis)[i] = axis->id;
  }
  return 0;
}

PyObject *SchedInfo::FromAscNode(af::AscNodeAttr &node_attr) {
  auto sched_info = ge::PtrToPtr<PyObject, SchedInfo::Object>(SchedInfo::type.tp_alloc(&SchedInfo::type, 0));
  PY_ASSERT_NOTNULL(sched_info);
  sched_info->axis = &node_attr.sched.axis;
  sched_info->loop_axis = &node_attr.sched.loop_axis;
  auto sched_py_obj = ge::PtrToPtr<SchedInfo::Object, PyObject>(sched_info);
  Py_IncRef(sched_py_obj);
  return sched_py_obj;
}

PyObject *SchedInfo::FromGraph(af::AscGraphAttr &graph_attr) {
  auto sched_info = ge::PtrToPtr<PyObject, SchedInfo::Object>(SchedInfo::type.tp_alloc(&SchedInfo::type, 0));
  PY_ASSERT_NOTNULL(sched_info);
  sched_info->axis = &graph_attr.sched.axis;
  sched_info->loop_axis = &graph_attr.sched.loop_axis;
  auto sched_py_obj = ge::PtrToPtr<SchedInfo::Object, PyObject>(sched_info);
  Py_IncRef(sched_py_obj);
  return sched_py_obj;
}

PyGetSetDef SchedInfo::GetSetters[] = {
    {"axis", SchedInfo::get_axis, SchedInfo::set_axis, "Axis of scheduler.", nullptr}, {nullptr} /* Sentinel */
};

PyTypeObject SchedInfo::type = {PyVarObject_HEAD_INIT(nullptr, 0)};

PyObject *HintGraph::GetSched(PyObject *self_pyobject, void *closure) {
  (void)closure;
  auto self = reinterpret_cast<HintGraph::Object *>(self_pyobject);
  PY_ASSERT_NOTNULL(self);
  PY_ASSERT_NOTNULL(self->graph, "graph is not ready");
  const auto compute_graph = af::AscGraphUtils::GetComputeGraph(*self->graph);
  PY_ASSERT_NOTNULL(compute_graph);
  auto graph_attr = compute_graph->GetOrCreateAttrsGroup<af::AscGraphAttr>();
  PY_ASSERT_NOTNULL(graph_attr);
  return SchedInfo::FromGraph(*graph_attr);
}

template <typename OpType>
class IrAttr {
 public:
  struct Object {
    PyObject_HEAD af::AscIrAttrDefBase *ir_attr;
  };

  // 每个 OpType 拥有独立的类型对象和属性列表
  struct TypeData {
    PyTypeObject type;
    std::vector<PyGetSetDef> getsetters;
  };

  static TypeData &GetTypeData();
  static void Dealloc(PyObject *self);
  static PyObject *FromAscNode(af::AscNodeAttr &node_attr, const char *op_type);
  using handler = std::function<void(std::vector<PyGetSetDef> &)>;
  static const std::map<std::string, handler> attr_handlers;
};

// 初始化 TypeData
template <typename OpType>
auto IrAttr<OpType>::GetTypeData() -> typename IrAttr<OpType>::TypeData & {
  static TypeData data = []() {
    TypeData d = {};
    PyGetSetDef sentinel = {nullptr, nullptr, nullptr, nullptr, nullptr};
    d.getsetters.push_back(sentinel);  // 初始哨兵
    return d;
  }();
  return data;
}

template <typename OpType>
PyObject *IrAttr<OpType>::FromAscNode(af::AscNodeAttr &node_attr, const char *op_type) {
  auto &typedata = GetTypeData();
  auto &type = typedata.type;
  auto &getsetters = typedata.getsetters;

  getsetters.clear();
  type.tp_name = "IrAttr";
  type.tp_basicsize = sizeof(IrAttr::Object);
  type.tp_itemsize = 0;
  type.tp_dealloc = IrAttr::Dealloc;
  type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
  type.tp_doc = "ir attrs.";
  auto handle_iter = attr_handlers.find(op_type);
  if (handle_iter != attr_handlers.end()) {
    handle_iter->second(getsetters);  // 填充当前 OpType 的属性
  }
  PyGetSetDef sentinel = {nullptr, nullptr, nullptr, nullptr, nullptr};
  getsetters.push_back(sentinel);  // 重置哨兵
  type.tp_getset = getsetters.data();
  if (PyType_Ready(&type) < 0) {
    return nullptr;
  }
  auto ir_attr = ge::PtrToPtr<PyObject, IrAttr::Object>(type.tp_alloc(&type, 0));
  PY_ASSERT_NOTNULL(ir_attr);

  if (node_attr.ir_attr == nullptr) {
    Py_DECREF(ir_attr);
    return nullptr;
  }

  ir_attr->ir_attr = node_attr.ir_attr.get();
  return ge::PtrToPtr<IrAttr::Object, PyObject>(ir_attr);
}

template <typename OpType>
void IrAttr<OpType>::Dealloc(PyObject *self) {
  Py_TYPE(self)->tp_free(self);
}
class AscNodeAttr {
 public:
  struct Object {
    PyObject_HEAD PyObject *api;  // ApiInfo
    PyObject *sched;              // OpsOperatorAttrSched
    PyObject *ir_attr;
  };

  static PyMemberDef members[];
  static PyTypeObject type;

  static void Dealloc(PyObject *self);
  template <typename OpType>
  static PyObject *FromAscNode(af::AscNodeAttr &node_attr, const char *op_type);
};

PyMemberDef AscNodeAttr::members[] = {{"api", T_OBJECT_EX, offsetof(Object, api), 0, nullptr},
                                      {"sched", T_OBJECT_EX, offsetof(Object, sched), 0, nullptr},
                                      {"ir_attr", T_OBJECT_EX, offsetof(Object, ir_attr), 0, nullptr},
                                      {nullptr}};

PyTypeObject AscNodeAttr::type = {PyVarObject_HEAD_INIT(nullptr, 0)};

void AscNodeAttr::Dealloc(PyObject *self) {
  auto self_obj = ge::PtrToPtr<PyObject, Object>(self);
  Py_XDECREF(self_obj->sched);
  Py_XDECREF(self_obj->api);
  Py_XDECREF(self_obj->ir_attr);
  Py_TYPE(self)->tp_free(self);
}
template <typename OpType>
PyObject *AscNodeAttr::FromAscNode(af::AscNodeAttr &node_attr, const char *op_type) {
  auto attr = ge::PtrToPtr<PyObject, AscNodeAttr::Object>(AscNodeAttr::type.tp_alloc(&AscNodeAttr::type, 0));
  PY_ASSERT_NOTNULL(attr);

  attr->api = ApiInfo::FromAscNode(node_attr);
  attr->sched = SchedInfo::FromAscNode(node_attr);
  attr->ir_attr = IrAttr<OpType>::FromAscNode(node_attr, op_type);
  auto attr_py_obj = ge::PtrToPtr<AscNodeAttr::Object, PyObject>(attr);
  return attr_py_obj;
}
}  // namespace pyascir

/** OpsOperator and input/output */
namespace pyascir {
template <typename OpType, const char *attr_name = nullptr>
class OpsOperatorIrAttr {
 public:
  static int _setter(PyObject *self, PyObject *value, void *closure);
  static PyObject *_getter(PyObject *self, void *closure);
};

class OpsOperatorMethod {
 public:
  static PyObject *InferDtype(PyObject *self_pyobject, PyObject *args);
};
class OpsOperatorInput {
 public:
  static int _setter(PyObject *self, PyObject *value, void *closure);
  static int _setter_list(PyObject *self, PyObject *value, void *closure);
  static int _setter_or_setter_list(PyObject *self, PyObject *value, void *closure);
};

class OpsOperatorOutput {
 public:
  struct Object {
    PyObject_HEAD af::AscTensorAttr *attr_holder;
    int index;
    bool is_dynamic_ouptut;
    af::Operator *op;
  };

  static PyGetSetDef GetSetters[];
  static PyTypeObject type;
  static void Dealloc(PyObject *self);
  static PyObject *FromOp(int index, af::Operator *op, af::AscTensorAttr &tensor_attr, bool is_dynamic_ouptut = false);
  static PyMemberDef CreateMember(const char *name, Py_ssize_t offset);

  static PyObject *GetDtype(PyObject *self, void *closure);
  static int SetDtype(PyObject *self, PyObject *value, void *closure);
  static PyObject *GetAxis(PyObject *self, void *closure);
  static int SetAxis(PyObject *self, PyObject *value, void *closure);
  static PyObject *GetStrides(PyObject *self, void *closure);
  static int SetStrides(PyObject *self, PyObject *value, void *closure);
  static PyObject *GetRepeats(PyObject *self, void *closure);
  static int SetRepeats(PyObject *self, PyObject *value, void *closure);
};

PyGetSetDef OpsOperatorOutput::GetSetters[] = {
    {"axis", OpsOperatorOutput::GetAxis, OpsOperatorOutput::SetAxis, "Axis"},
    {"size", OpsOperatorOutput::GetRepeats, OpsOperatorOutput::SetRepeats, "Size along each axis."},
    {"strides", OpsOperatorOutput::GetStrides, OpsOperatorOutput::SetStrides, "Stride along each axis."},
    {"dtype", OpsOperatorOutput::GetDtype, OpsOperatorOutput::SetDtype, "Data type"},
    {nullptr} /* Sentinel */
};

PyTypeObject OpsOperatorOutput::type = {PyVarObject_HEAD_INIT(nullptr, 0)};

void OpsOperatorOutput::Dealloc(PyObject *self) {
  Py_TYPE(self)->tp_free(self);
}

int OpsOperatorOutput::SetDtype(PyObject *self, PyObject *value, void *closure) {
  (void)closure;
  auto dtype = PyLong_AsLong(value);
  auto self_ = ge::PtrToPtr<PyObject, OpsOperatorOutput::Object>(self);
  if (self_->attr_holder == nullptr) {
    PyErr_SetString(PyExc_ValueError, "tensor attr has not been inited.");
    return -1;
  }
  self_->attr_holder->dtype = static_cast<ge::DataType>(dtype);
  return 0;
}

PyObject *OpsOperatorOutput::GetDtype(PyObject *self, void *closure) {
  (void)closure;
  auto self_ = ge::PtrToPtr<PyObject, OpsOperatorOutput::Object>(self);
  PY_ASSERT_NOTNULL(self_->attr_holder, "tensor attr has not been inited.");
  auto dtype = self_->attr_holder->dtype;

  return PyLong_FromLong(static_cast<ge::DataType>(dtype));
}

PyObject *OpsOperatorOutput::GetAxis(PyObject *self, void *closure) {
  (void)closure;
  auto operator_output = ge::PtrToPtr<PyObject, OpsOperatorOutput::Object>(self);
  PY_ASSERT_NOTNULL(operator_output->attr_holder, "tensor attr has not been inited.");
  auto axis = operator_output->attr_holder->axis;
  auto list = PyList_New(axis.size());
  for (size_t i = 0UL; i < axis.size(); ++i) {
    PyList_SetItem(list, i, PyLong_FromLong(axis[i]));
  }

  return list;
}

int OpsOperatorOutput::SetAxis(PyObject *self, PyObject *value, void *closure) {
  (void)closure;
  auto operator_output = ge::PtrToPtr<PyObject, OpsOperatorOutput::Object>(self);
  if (operator_output->attr_holder == nullptr) {
    PyErr_SetString(PyExc_ValueError, "tensor attr has not been inited.");
    return -1;
  }
  if (PyList_Check(value) == kPythonFail) {
    PyErr_SetString(PyExc_TypeError, "value must be a list");
    return -1;
  }
  operator_output->attr_holder->axis.resize(PyList_Size(value));
  for (int i = 0; i < PyList_Size(value); ++i) {
    auto axis = ge::PtrToPtr<PyObject, Axis::Object>(PyList_GetItem(value, i));
    if (PyObject_IsInstance(ge::PtrToPtr<Axis::Object, PyObject>(axis),
                            ge::PtrToPtr<PyTypeObject, PyObject>(&Axis::type)) == 0) {
      PyErr_Format(PyExc_ValueError, "axis on %d is not Axis type", i);
      return -1;
    }
    operator_output->attr_holder->axis[i] = axis->id;
  }
  return 0;
}

PyObject *OpsOperatorOutput::GetStrides(PyObject *self, void *closure) {
  (void)closure;
  auto operator_output = ge::PtrToPtr<PyObject, OpsOperatorOutput::Object>(self);
  PY_ASSERT_NOTNULL(operator_output->attr_holder, "tensor attr has not been inited.");
  auto strides = operator_output->attr_holder->strides;
  auto list = PyList_New(strides.size());
  for (size_t i = 0UL; i < strides.size(); ++i) {
    PyList_SetItem(list, i, SizeExpr::FromSizeExpr(strides[i]));
  }
  return list;
}

int OpsOperatorOutput::SetStrides(PyObject *self, PyObject *value, void *closure) {
  (void)closure;
  auto operator_output = ge::PtrToPtr<PyObject, OpsOperatorOutput::Object>(self);
  PY_ASSERT_NOTNULL(operator_output->attr_holder, "tensor attr has not been inited.");
  if (PyList_Check(value) == 0) {
    PyErr_SetString(PyExc_TypeError, "value must be a list");
    return -1;
  }
  operator_output->attr_holder->strides.clear();
  for (int i = 0; i < PyList_Size(value); ++i) {
    auto strides = PyList_GetItem(value, i);
    if (PyObject_IsInstance(strides, ge::PtrToPtr<PyTypeObject, PyObject>(&SizeExpr::type)) == 0 &&
        PyLong_Check(strides) == 0) {
      PyErr_Format(PyExc_ValueError, "strides on %d is not SizeExpr or long type.", i);
      return -1;
    }
    operator_output->attr_holder->strides.push_back(SizeExpr::AsSizeExpr(strides));
  }
  return 0;
}

PyObject *OpsOperatorOutput::GetRepeats(PyObject *self, void *closure) {
  (void)closure;
  auto operator_output = ge::PtrToPtr<PyObject, OpsOperatorOutput::Object>(self);
  PY_ASSERT_NOTNULL(operator_output->attr_holder, "tensor attr has not been inited.");
  auto repeats = operator_output->attr_holder->repeats;
  auto list = PyList_New(repeats.size());
  for (size_t i = 0UL; i < repeats.size(); ++i) {
    PyList_SetItem(list, i, SizeExpr::FromSizeExpr(repeats[i]));
  }
  return list;
}

int OpsOperatorOutput::SetRepeats(PyObject *self, PyObject *value, void *closure) {
  (void)closure;
  auto operator_output = ge::PtrToPtr<PyObject, OpsOperatorOutput::Object>(self);
  if (operator_output->attr_holder == nullptr) {
    PyErr_SetString(PyExc_ValueError, "tensor attr has not been inited.");
    return -1;
  }
  if (PyList_Check(value) == 0) {
    PyErr_SetString(PyExc_TypeError, "value must be a list");
    return -1;
  }
  operator_output->attr_holder->repeats.clear();
  for (int i = 0; i < PyList_Size(value); ++i) {
    auto repeats = PyList_GetItem(value, i);
    if (PyObject_IsInstance(repeats, ge::PtrToPtr<PyTypeObject, PyObject>(&SizeExpr::type)) == 0 &&
        PyLong_Check(repeats) == 0) {
      PyErr_Format(PyExc_ValueError, "repeats on %d is not SizeExpr or long type.", i);
      return -1;
    }
    operator_output->attr_holder->repeats.push_back(SizeExpr::AsSizeExpr(repeats));
  }
  return 0;
}

PyObject *OpsOperatorOutput::FromOp(int index, af::Operator *op, af::AscTensorAttr &tensor_attr,
                                    bool is_dynamic_ouptut) {
  auto self =
      ge::PtrToPtr<PyObject, OpsOperatorOutput::Object>(OpsOperatorOutput::type.tp_alloc(&OpsOperatorOutput::type, 0));
  PY_ASSERT_NOTNULL(self);

  auto op_desc = af::OpDescUtils::GetOpDescFromOperator(*op);
  auto output_desc = op_desc->MutableOutputDesc(index);
  if (op_desc->GetType() != kDataOpType) {
    // 非Data类算子输出设置默认为无效值，需要构图显示指定或者按照注册信息推导
    // Data类兼容之前没有指定的场景，默认值为float32
    output_desc->SetDataType(ge::DT_UNDEFINED);
  }
  self->attr_holder = &tensor_attr;

  self->op = op;
  self->index = index;
  self->is_dynamic_ouptut = is_dynamic_ouptut;
  auto out_py_obj = ge::PtrToPtr<OpsOperatorOutput::Object, PyObject>(self);
  Py_IncRef(out_py_obj);
  return out_py_obj;
}

PyMemberDef OpsOperatorOutput::CreateMember(const char *name, Py_ssize_t offset) {
  return {name, T_OBJECT_EX, offset, 0, nullptr};
}

struct OpsOperatorTypeObject {
 private:
  PyTypeObject pytype;

 public:
  std::string op_type;
  std::vector<std::pair<std::string, af::IrInputType>> input_defs;
  std::vector<std::pair<std::string, af::IrOutputType>> output_defs;

  newfunc tp_new;
  initproc tp_init;
  destructor tp_dealloc;

  Py_ssize_t object_size;
  Py_ssize_t member_attr_offset;
  Py_ssize_t member_input_output_offset;

 public:
  void InitPyType();

  /**
   * Get PyType
   * @warning Init before get
   */
  PyTypeObject &GetPyType();
  static OpsOperatorTypeObject &GetOpsType(PyTypeObject *type);

 private:
  std::vector<PyMemberDef> members;
  std::vector<PyMethodDef> methods;
  std::vector<PyGetSetDef> getsetters;
};

template <typename OpType>
bool InitDynamicOutputs(OpType *op, const OpsOperatorTypeObject &ops_type, uint32_t dynamic_output_num) {
  if (ops_type.output_defs.size() != 1U) {
    std::stringstream ss;
    ss << "Dynamic output op should have exactly one ir output, actual " << ops_type.output_defs.size();
    PyErr_Format(PyExc_TypeError, "%s", ss.str().c_str());
    GELOGE(af::FAILED, "%s", ss.str().c_str());
    return false;
  }
  op->DynamicOutputRegister(ops_type.output_defs[0U].first.c_str(), dynamic_output_num);
  return true;
}

template <>
bool InitDynamicOutputs(af::ascir_op::Split *op, const OpsOperatorTypeObject &ops_type, uint32_t dynamic_output_num) {
  (void)ops_type;
  op->InstanceOutputy(dynamic_output_num);
  return true;
}

bool ParseOpsOperatorInitArgs(PyObject *args, bool support_dynamic_output, const char *&name,
                              uint32_t &dynamic_output_num, PyObject *&graph_object) {
  const auto arg_count = PyTuple_Size(args);
  const auto max_arg_count = support_dynamic_output ? 3 : 2;
  if (arg_count < 1 || arg_count > max_arg_count) {
    PyErr_Format(PyExc_TypeError, support_dynamic_output ? "Operator expects (name[, output_num][, graph])"
                                                         : "Operator expects (name[, graph])");
    return false;
  }

  auto name_obj = PyTuple_GetItem(args, 0);
  if (name_obj == nullptr || PyUnicode_Check(name_obj) == kPythonFail) {
    PyErr_SetString(PyExc_TypeError, "Operator name must be str.");
    return false;
  }
  name = PyUnicode_AsUTF8(name_obj);
  if (name == nullptr) {
    return false;
  }

  bool has_dynamic_output_num = false;
  for (Py_ssize_t i = 1; i < arg_count; ++i) {
    auto item = PyTuple_GetItem(args, i);
    PY_ASSERT_NOTNULL(item);
    if (support_dynamic_output && PyLong_Check(item) == kPythonSuccess) {
      if (has_dynamic_output_num) {
        PyErr_SetString(PyExc_TypeError, "Dynamic output num is duplicated.");
        return false;
      }
      const auto output_num = PyLong_AsUnsignedLong(item);
      if (PyErr_Occurred() != nullptr) {
        return false;
      }
      if (output_num > std::numeric_limits<uint32_t>::max()) {
        PyErr_Format(PyExc_OverflowError, "Dynamic output num %lu is out of range [0, %u].", output_num,
                     std::numeric_limits<uint32_t>::max());
        return false;
      }
      dynamic_output_num = static_cast<uint32_t>(output_num);
      has_dynamic_output_num = true;
      continue;
    }

    const auto is_hint_graph =
        PyObject_IsInstance(item, ge::PtrToPtr<PyTypeObject, PyObject>(&pyascir::HintGraph::type));
    if (is_hint_graph == kPythonError) {
      return false;
    }
    const auto is_fused_graph =
        PyObject_IsInstance(item, ge::PtrToPtr<PyTypeObject, PyObject>(&pyascir::FusedGraph::type));
    if (is_fused_graph == kPythonError) {
      return false;
    }
    if (is_hint_graph != 0 || is_fused_graph != 0) {
      if (graph_object != nullptr) {
        PyErr_SetString(PyExc_TypeError, "Graph arg is duplicated.");
        return false;
      }
      graph_object = item;
      continue;
    }

    PyErr_Format(PyExc_TypeError, support_dynamic_output ? "Operator expects (name[, output_num][, graph])"
                                                         : "Operator expects (name[, graph])");
    return false;
  }
  return true;
}

template <typename OpType>
bool SetupOutputs(PyObject *self_pyobject, OpType *op);

template <typename OpType>
class OpsOperator {
 public:
  struct Object {
    Operator::Object op_base;
    OpType *op;
    int input_output_num;

    PyObject *attr;  // AscNodeAttr
    PyObject *input_outputs[1];
  };

  static void OpsOperator_dealloc(PyObject *self_pyobject) {
    auto self = ge::PtrToPtr<PyObject, Object>(self_pyobject);
    delete self->op;
    self->op = nullptr;
    self->op_base.op = nullptr;
    Operator::Dealloc(ge::PtrToPtr<Object, PyObject>(self));
  }

  static PyObject *OpsOperator_new(PyTypeObject *type, PyObject *args, PyObject *kwargs) {
    auto self = ge::PtrToPtr<PyObject, Object>(Operator::New(type, args, kwargs));
    PY_ASSERT_NOTNULL(self);

    self->attr = Py_None;
    self->input_output_num = 0;
    self->op_base.op = nullptr;

    return ge::PtrToPtr<Object, PyObject>(self);
  }

  static int OpsOperator_init(PyObject *self_pyobject, PyObject *args, PyObject *kwargs) {
    (void)kwargs;
    const char *name_ptr = "";
    uint32_t dynamic_output_num = 0U;
    PyObject *graph_object{nullptr};
    auto ops_type = OpsOperatorTypeObject::GetOpsType(Py_TYPE(self_pyobject));
    const auto has_dynamic_output =
        ops_type.output_defs.size() == 1U && ops_type.output_defs[0U].second == af::kIrOutputDynamic;
    if (!ParseOpsOperatorInitArgs(args, has_dynamic_output, name_ptr, dynamic_output_num, graph_object)) {
      return -1;
    }

    auto op = new OpType(name_ptr);
    PY_ASSERT_NOTNULL(op);
    if (has_dynamic_output && dynamic_output_num > 0U) {
      if (!InitDynamicOutputs(op, ops_type, dynamic_output_num)) {
        delete op;
        return -1;
      }
    }
    // Ascir only need start node to pass graph as in_param
    if (ops_type.input_defs.empty()) {
      PY_ASSERT_NOTNULL(graph_object);
      if (PyObject_IsInstance(graph_object, ge::PtrToPtr<PyTypeObject, PyObject>(&pyascir::HintGraph::type)) != 0) {
        auto graph = ge::PtrToPtr<PyObject, pyascir::HintGraph::Object>(graph_object);
        PY_ASSERT_NOTNULL(graph->graph, "The input param graph is nullptr.");
        graph->graph->AddNode(*op);
      } else if (PyObject_IsInstance(graph_object, ge::PtrToPtr<PyTypeObject, PyObject>(&pyascir::FusedGraph::type)) !=
                 0) {
        auto graph = ge::PtrToPtr<PyObject, pyascir::FusedGraph::Object>(graph_object);
        PY_ASSERT_NOTNULL(graph->graph, "The input param graph is nullptr.");
        auto node = graph->graph->AddNode(af::OpDescUtils::GetOpDescFromOperator(*op));
        af::NodeUtilsEx::SetNodeToOperator(*op, node);
      } else {
        PyErr_SetString(PyExc_TypeError, "The start node requires the graph to be passed in as an input parameter.");
        delete op;
        return -1;
      }
    }
    auto self = ge::PtrToPtr<PyObject, Object>(self_pyobject);
    self->op_base.name = PyUnicode_FromString(name_ptr);
    ge::AscendString op_type;
    PY_ASSERT_GRAPH_SUCCESS(op->GetOpType(op_type), "Get op_type failed.");
    self->op_base.type = PyUnicode_FromString(op_type.GetString());
    self->op_base.op = op;
    self->op = op;
    self->attr = AscNodeAttr::FromAscNode<OpType>(op->attr, op_type.GetString());
    if (has_dynamic_output) {
      if (!SetupOutputs<OpType>(self_pyobject, op)) {
        delete op;
        self->op = nullptr;
        self->op_base.op = nullptr;
        self->attr = Py_None;
        return -1;
      }
      return 0;
    }
    for (size_t i = 0UL; i < op->GetOutputsSize(); ++i) {
      PY_ASSERT(i < ops_type.output_defs.size());
      PY_ASSERT(ops_type.output_defs[i].second != af::kIrOutputDynamic,
                "Op %s's %s output idx %zu is dynamic but is not inited", name_ptr, op_type.GetString(), i);
      auto attr_group = af::AscTensorAttr::GetTensorAttrPtr(op, i);
      PY_ASSERT_NOTNULL(attr_group);
      auto output = OpsOperatorOutput::FromOp(i, op, *attr_group);
      PY_ASSERT_NOTNULL(output);
      self->input_outputs[self->input_output_num++] = output;
    }
    return 0;
  }

  static OpsOperatorTypeObject CreateTypeObject() {
    OpsOperatorTypeObject ops_type;

    ops_type.member_attr_offset = offsetof(OpsOperator<OpType>::Object, attr);
    ops_type.member_input_output_offset = offsetof(OpsOperator<OpType>::Object, input_outputs);

    OpType sample_op("sample");
    auto sample_op_desc = af::OpDescUtils::GetOpDescFromOperator(sample_op);
    ops_type.input_defs = sample_op_desc->GetIrInputs();
    ops_type.output_defs = sample_op_desc->GetIrOutputs();

    ops_type.op_type = sample_op_desc->GetType();
    ops_type.object_size = sizeof(OpsOperator<OpType>::Object);
    ops_type.tp_new = OpsOperator_new;
    ops_type.tp_init = OpsOperator_init;
    ops_type.tp_dealloc = OpsOperator_dealloc;
    return ops_type;
  }
};

template <typename OpType>
bool SetupOutputs(PyObject *self_pyobject, OpType *op) {
  auto self = reinterpret_cast<typename OpsOperator<OpType>::Object *>(self_pyobject);
  auto outputs_list = PyList_New(op->GetOutputsSize());
  PY_ASSERT_NOTNULL(outputs_list);
  for (size_t i = 0; i < op->GetOutputsSize(); ++i) {
    auto attr_group = af::AscTensorAttr::GetTensorAttrPtr(op, i);
    PY_ASSERT_NOTNULL(attr_group);
    auto output = OpsOperatorOutput::FromOp(i, op, *attr_group, true);
    if (output == nullptr) {
      Py_DECREF(outputs_list);
      return false;
    }
    PyList_SetItem(outputs_list, i, output);
  }
  self->input_outputs[self->input_output_num++] = outputs_list;
  return true;
}

template <typename OpType>
bool SetupSelfAttributes(typename OpsOperator<OpType>::Object *self, const char *name, OpType *op) {
  self->op_base.name = PyUnicode_FromString(name);
  ge::AscendString op_type;
  PY_ASSERT_GRAPH_SUCCESS(op->GetOpType(op_type), "Get op_type failed.");
  self->op_base.type = PyUnicode_FromString(op_type.GetString());
  self->op_base.op = op;
  self->op = op;
  self->attr = nullptr;
  return true;
}

template <typename OpType>
int CommonOpsOperatorInit(PyObject *self_pyobject, PyObject *args, PyObject *kwargs) {
  (void)kwargs;
  const char *name = "";
  PyObject *ascgraph_object{nullptr};
  PyObject *fusedgraph_object{nullptr};
  if (PyArg_ParseTuple(args, "sO|O", &name, &ascgraph_object, &fusedgraph_object) == 0) {
    return -1;
  }
  auto op = new (std::nothrow) OpType(name);
  PY_ASSERT_NOTNULL(op);
  PY_ASSERT_NOTNULL(ascgraph_object);
  PY_ASSERT(PyObject_IsInstance(ascgraph_object, ge::PtrToPtr<PyTypeObject, PyObject>(&pyascir::HintGraph::type)) != 0,
            "The asc graph node requires hintgraph to be passed in as an input parameter.");

  auto graph = ge::PtrToPtr<PyObject, pyascir::HintGraph::Object>(ascgraph_object);
  PY_ASSERT_NOTNULL(graph->graph);
  PY_ASSERT(GraphHandler<OpType>::Handle(op, graph->graph));

  size_t input_size = 0U;
  std::set<int64_t> outputs;
  PY_ASSERT(CountInputsOutputs(graph->graph, input_size, outputs));

  auto ops_type = OpsOperatorTypeObject::GetOpsType(Py_TYPE(self_pyobject));
  PY_ASSERT_EQ(ops_type.input_defs.size(), 1U);
  PY_ASSERT_EQ(ops_type.output_defs.size(), 1U);

  op->DynamicOutputRegister(ops_type.output_defs[0U].first.c_str(), outputs.size());

  if (input_size == 0U) {
    PY_ASSERT_NOTNULL(fusedgraph_object,
                      "The asc graph node requires fusedgraph to be passed in as an input parameter"
                      " when has no inputs.");
    PY_ASSERT(
        PyObject_IsInstance(fusedgraph_object, ge::PtrToPtr<PyTypeObject, PyObject>(&pyascir::FusedGraph::type)) != 0,
        "The asc graph node requires fusedgraph type.");
    auto fused_graph = ge::PtrToPtr<PyObject, pyascir::FusedGraph::Object>(fusedgraph_object);
    PY_ASSERT_NOTNULL(fused_graph->graph, "The input param fusedgraph is nullptr.");
    auto op_desc = af::OpDescUtils::GetOpDescFromOperator(*op);
    PY_ASSERT_NOTNULL(op_desc);
    const auto &node = fused_graph->graph->AddNode(op_desc);
    af::NodeUtilsEx::SetNodeToOperator(*op, node);
  } else {
    op->DynamicInputRegister(ops_type.input_defs[0U].first.c_str(), input_size);
  }

  auto self = reinterpret_cast<typename OpsOperator<OpType>::Object *>(self_pyobject);
  PY_ASSERT(SetupSelfAttributes<OpType>(self, name, op));
  PY_ASSERT(SetupOutputs<OpType>(self_pyobject, op));
  return 0;
}

template <>
int OpsOperator<geir_op::AscBackend>::OpsOperator_init(PyObject *self_pyobject, PyObject *args, PyObject *kwargs) {
  return CommonOpsOperatorInit<geir_op::AscBackend>(self_pyobject, args, kwargs);
}

template <>
int OpsOperator<geir_op::AscGraph>::OpsOperator_init(PyObject *self_pyobject, PyObject *args, PyObject *kwargs) {
  return CommonOpsOperatorInit<geir_op::AscGraph>(self_pyobject, args, kwargs);
}
PyObject *OpsOperatorMethod::InferDtype(PyObject *self_pyobject, PyObject *args) {
  (void)args;
  auto self = ge::PtrToPtr<PyObject, OpsOperator<af::Operator>::Object>(self_pyobject);
  PY_ASSERT_NOTNULL(self);
  PY_ASSERT_NOTNULL(self->op);
  ge::AscendString op_type;
  (void)self->op->GetOpType(op_type);
  ge::AscendString op_name;
  (void)self->op->GetName(op_name);
  auto node = af::NodeUtilsEx::GetNodeFromOperator(*self->op);
  PY_ASSERT_NOTNULL(node, "node %s %s need set input before call infer dtype", op_name.GetString(),
                    op_type.GetString());
  GE_ASSERT(
      HintGraph::ProcessSingleNode(std::dynamic_pointer_cast<af::AscNode>(std::const_pointer_cast<af::Node>(node))));
  Py_RETURN_NONE;
}
int OpsOperatorInput::_setter(PyObject *self, PyObject *value, void *closure) {
  auto self_ = reinterpret_cast<OpsOperator<af::Operator>::Object *>(self);
  auto index = PyLong_AsLong(ge::PtrToPtr<void, PyObject>(closure));

  if (PyObject_IsInstance(value, ge::PtrToPtr<PyTypeObject, PyObject>(&Operator::type)) != 0) {
    auto op = reinterpret_cast<Operator::Object *>(value);
    PY_ASSERT_GRAPH_SUCCESS(af::LinkByIrIndex(*op->op, 0, *self_->op, index));
    return 0;
  } else if (PyObject_IsInstance(value, reinterpret_cast<PyObject *>(&OpsOperatorOutput::type)) != 0) {
    auto output = reinterpret_cast<OpsOperatorOutput::Object *>(value);
    PY_ASSERT_NOTNULL(output->op);
    if (output->is_dynamic_ouptut) {
      PY_ASSERT_GRAPH_SUCCESS(af::AddEdgeForNode(*output->op, output->index, *self_->op, index));
    } else {
      PY_ASSERT_GRAPH_SUCCESS(af::LinkByIrIndex(*output->op, output->index, *self_->op, index));
    }
    return 0;
  } else {
    return -1;
  }
}

int OpsOperatorInput::_setter_list(PyObject *self, PyObject *value, void *closure) {
  auto self_ = reinterpret_cast<OpsOperator<af::Operator>::Object *>(self);
  auto ir_index = PyLong_AsLong(ge::PtrToPtr<void, PyObject>(closure));

  uint32_t dynamic_num = PyList_Size(value);
  ge::AscendString op_type;
  (void)self_->op->GetOpType(op_type);
  ge::AscendString op_name;
  (void)self_->op->GetName(op_name);
  if (op_type == kAscBackendType || op_type == geir_op::AscGraph::Type) {
    PY_ASSERT(dynamic_num == self_->op->GetInputsSize(), "%s %s should have %zu input(s) but %u given",
              op_name.GetString(), op_type.GetString(), self_->op->GetInputsSize(), dynamic_num);
  } else {
    PY_ASSERT_GRAPH_SUCCESS(af::SetDynamicInputNumByIrIndex(*self_->op, ir_index, dynamic_num));
  }
  for (uint32_t i = 0U; i < dynamic_num; ++i) {
    auto item = PyList_GetItem(value, i);
    if (PyObject_IsInstance(item, reinterpret_cast<PyObject *>(&Operator::type)) != 0) {
      auto op = ge::PtrToPtr<PyObject, Operator::Object>(item);
      PY_ASSERT_GRAPH_SUCCESS(af::LinkByIrIndex(*op->op, 0, *self_->op, ir_index, i));
    } else if (PyObject_IsInstance(item, reinterpret_cast<PyObject *>(&OpsOperatorOutput::type)) != 0) {
      auto output = ge::PtrToPtr<PyObject, OpsOperatorOutput::Object>(item);
      PY_ASSERT_NOTNULL(output->op);
      if (output->is_dynamic_ouptut) {
        PY_ASSERT_GRAPH_SUCCESS(af::AddEdgeForNode(*output->op, output->index, *self_->op, ir_index + i));
      } else {
        PY_ASSERT_GRAPH_SUCCESS(af::LinkByIrIndex(*output->op, output->index, *self_->op, ir_index, i));
      }
    } else {
      PyErr_SetString(PyExc_TypeError, "Input Type is invalid.");
      return -1;
    }
  }
  return 0;
}

int OpsOperatorInput::_setter_or_setter_list(PyObject *self, PyObject *value, void *closure) {
  if (PyList_Check(value) == kPythonFail) {
    return _setter(self, value, closure);
  }
  auto self_ = ge::PtrToPtr<PyObject, OpsOperator<af::Operator>::Object>(self);
  auto ir_index = PyLong_AsLong(ge::PtrToPtr<void, PyObject>(closure));
  uint32_t dynamic_num = PyList_Size(value);
  PY_ASSERT_NOTNULL(self_->op);
  auto op_desc = af::OpDescUtils::GetOpDescFromOperator(*self_->op);
  const auto &ir_inputs = op_desc->GetIrInputs();
  PY_ASSERT(static_cast<size_t>(ir_index) < ir_inputs.size());
  af::OpDescUtils::ClearInputDesc(op_desc, ir_index);
  op_desc->AddDynamicInputDescByIndex(ir_inputs[ir_index].first, dynamic_num, ir_index);
  for (uint32_t i = 0U; i < dynamic_num; ++i) {
    auto item = PyList_GetItem(value, i);
    if (PyObject_IsInstance(item, ge::PtrToPtr<PyTypeObject, PyObject>(&Operator::type)) != 0) {
      auto op = ge::PtrToPtr<PyObject, Operator::Object>(item);
      PY_ASSERT_GRAPH_SUCCESS(af::AddEdgeForNode(*op->op, 0, *self_->op, ir_index + i));
    } else if (PyObject_IsInstance(item, ge::PtrToPtr<PyTypeObject, PyObject>(&OpsOperatorOutput::type)) != 0) {
      auto output = ge::PtrToPtr<PyObject, OpsOperatorOutput::Object>(item);
      PY_ASSERT_GRAPH_SUCCESS(af::AddEdgeForNode(*output->op, output->index, *self_->op, ir_index + i));
    } else {
      PyErr_SetString(PyExc_TypeError, "Input Type is invalid.");
      return -1;
    }
  }
  return 0;
}

template <typename OpType, typename AttrDefType>
auto GetValidatedIrAttr(PyObject *self, const char *attr_type_name) -> AttrDefType * {
  auto ir_attr_obj = reinterpret_cast<typename IrAttr<OpType>::Object *>(self);
  PY_ASSERT(ir_attr_obj != nullptr, "Internal error, has no ir attr", "");
  auto target_attr = dynamic_cast<AttrDefType *>(ir_attr_obj->ir_attr);
  PY_ASSERT(target_attr != nullptr, "Internal error, ir attr type is not %s", attr_type_name);
  return target_attr;
}

#define DEFINE_IR_ATTR_ACCESSORS(OpType, AttrType, AttrName, ValueType, CheckFunc, ConvFunc, ReverseConvFunc,      \
                                 SetMethod, GetMethod)                                                             \
  template <>                                                                                                      \
  PyObject *OpsOperatorIrAttr<af::ascir_op::OpType, AttrName>::_getter(PyObject *self, void *closure) {            \
    (void)closure;                                                                                                 \
    auto *attr = GetValidatedIrAttr<af::ascir_op::OpType, af::ascir_op::OpType::AttrType>(self, #AttrType);        \
    ValueType v{};                                                                                                 \
    return attr ? (attr->GetMethod(v), ConvFunc(v)) : nullptr;                                                     \
  }                                                                                                                \
  template <>                                                                                                      \
  int OpsOperatorIrAttr<af::ascir_op::OpType, AttrName>::_setter(PyObject *self, PyObject *value, void *closure) { \
    (void)closure;                                                                                                 \
    if (!CheckFunc(value)) {                                                                                       \
      PyErr_Format(PyExc_TypeError, "%s attr %s expected %s", #OpType, AttrName, #ValueType);                      \
      return -1;                                                                                                   \
    }                                                                                                              \
    auto *attr = GetValidatedIrAttr<af::ascir_op::OpType, af::ascir_op::OpType::AttrType>(self, #AttrType);        \
    return attr ? (attr->SetMethod(ReverseConvFunc(value)), 0) : -1;                                               \
  }

#define DEFINE_IR_EXPR_ATTR_ACCESSORS(OpType, AttrType, AttrName, SetMethod, GetMethod)                            \
  template <>                                                                                                      \
  PyObject *OpsOperatorIrAttr<af::ascir_op::OpType, AttrName>::_getter(PyObject *self, void *closure) {            \
    (void)closure;                                                                                                 \
    auto *attr = GetValidatedIrAttr<af::ascir_op::OpType, af::ascir_op::OpType::AttrType>(self, #AttrType);        \
    af::Expression value;                                                                                          \
    if (attr == nullptr) {                                                                                         \
      return nullptr;                                                                                              \
    }                                                                                                              \
    PY_ASSERT_GRAPH_SUCCESS(attr->GetMethod(value), "%s attr %s getter failed", #OpType, AttrName);                \
    return pyascir::SizeExpr::FromSizeExpr(value);                                                                 \
  }                                                                                                                \
  template <>                                                                                                      \
  int OpsOperatorIrAttr<af::ascir_op::OpType, AttrName>::_setter(PyObject *self, PyObject *value, void *closure) { \
    (void)closure;                                                                                                 \
    if (PyObject_IsInstance(value, reinterpret_cast<PyObject *>(&pyascir::SizeExpr::type)) != kPythonSuccess) {    \
      PyErr_Format(PyExc_TypeError, "%s attr %s expected Expression", #OpType, AttrName);                          \
      return -1;                                                                                                   \
    }                                                                                                              \
    auto expression = pyascir::SizeExpr::AsExpression(value);                                                      \
    if (PyErr_Occurred() != nullptr) {                                                                             \
      return -1;                                                                                                   \
    }                                                                                                              \
    if (!expression.IsValid()) {                                                                                   \
      PyErr_Format(PyExc_ValueError, "%s attr %s expected a valid Expression", #OpType, AttrName);                 \
      return -1;                                                                                                   \
    }                                                                                                              \
    auto *attr = GetValidatedIrAttr<af::ascir_op::OpType, af::ascir_op::OpType::AttrType>(self, #AttrType);        \
    return attr ? (attr->SetMethod(expression), 0) : -1;                                                           \
  }

DEFINE_IR_ATTR_ACCESSORS(Data, AscDataIrAttrDef, kIndexAttr, int64_t, PyLong_Check, PyLong_FromLong, PyLong_AsLong,
                         SetIndex, GetIndex)
DEFINE_IR_ATTR_ACCESSORS(Output, AscOutputIrAttrDef, kIndexAttr, int64_t, PyLong_Check, PyLong_FromLong, PyLong_AsLong,
                         SetIndex, GetIndex)
DEFINE_IR_ATTR_ACCESSORS(ScalarData, AscScalarDataIrAttrDef, kIndexAttr, int64_t, PyLong_Check, PyLong_FromLong,
                         PyLong_AsLong, SetIndex, GetIndex)
DEFINE_IR_EXPR_ATTR_ACCESSORS(IndexExpr, AscIndexExprIrAttrDef, kExprAttr, SetExpr, GetExpr)
DEFINE_IR_EXPR_ATTR_ACCESSORS(Arange, AscArangeIrAttrDef, kBaseAttr, SetBase, GetBase)
DEFINE_IR_EXPR_ATTR_ACCESSORS(Arange, AscArangeIrAttrDef, kStepAttr, SetStep, GetStep)
DEFINE_IR_ATTR_ACCESSORS(Gather, AscGatherIrAttrDef, kAxisAttr, int64_t, PyLong_Check, PyLong_FromLong, PyLong_AsLong,
                         SetAxis, GetAxis)
DEFINE_IR_ATTR_ACCESSORS(IndirectLoad, AscIndirectLoadIrAttrDef, kAxisAttr, int64_t, PyLong_Check, PyLong_FromLong,
                         PyLong_AsLong, SetAxis, GetAxis)
DEFINE_IR_ATTR_ACCESSORS(IndirectLoad, AscIndirectLoadIrAttrDef, kNegativeIndexSupportAttr, bool, PyBool_Check,
                         PyBool_FromLong, PyObject_IsTrue, SetNegative_index_support, GetNegative_index_support)
DEFINE_IR_ATTR_ACCESSORS(IndirectLoad, AscIndirectLoadIrAttrDef, kNeedCheckBoundAttr, bool, PyBool_Check,
                         PyBool_FromLong, PyObject_IsTrue, SetNeed_check_bound, GetNeed_check_bound)
DEFINE_IR_ATTR_ACCESSORS(MatMul, AscMatMulIrAttrDef, kHasRelu, int64_t, PyLong_Check, PyLong_FromLong, PyLong_AsLong,
                         SetHas_relu, GetHas_relu)
DEFINE_IR_ATTR_ACCESSORS(MatMul, AscMatMulIrAttrDef, kTransposeX1, int64_t, PyLong_Check, PyLong_FromLong,
                         PyLong_AsLong, SetTranspose_x1, GetTranspose_x1)
DEFINE_IR_ATTR_ACCESSORS(MatMul, AscMatMulIrAttrDef, kTransposeX2, int64_t, PyLong_Check, PyLong_FromLong,
                         PyLong_AsLong, SetTranspose_x2, GetTranspose_x2)
DEFINE_IR_ATTR_ACCESSORS(MatMul, AscMatMulIrAttrDef, kOffsetX, int64_t, PyLong_Check, PyLong_FromLong, PyLong_AsLong,
                         SetOffset_x, GetOffset_x)
DEFINE_IR_ATTR_ACCESSORS(MatMul, AscMatMulIrAttrDef, kEnableHf32, int64_t, PyLong_Check, PyLong_FromLong, PyLong_AsLong,
                         SetEnable_hf32, GetEnable_hf32)
DEFINE_IR_ATTR_ACCESSORS(MatMulBias, AscMatMulBiasIrAttrDef, kHasRelu, int64_t, PyLong_Check, PyLong_FromLong,
                         PyLong_AsLong, SetHas_relu, GetHas_relu)
DEFINE_IR_ATTR_ACCESSORS(MatMulBias, AscMatMulBiasIrAttrDef, kTransposeX1, int64_t, PyLong_Check, PyLong_FromLong,
                         PyLong_AsLong, SetTranspose_x1, GetTranspose_x1)
DEFINE_IR_ATTR_ACCESSORS(MatMulBias, AscMatMulBiasIrAttrDef, kTransposeX2, int64_t, PyLong_Check, PyLong_FromLong,
                         PyLong_AsLong, SetTranspose_x2, GetTranspose_x2)
DEFINE_IR_ATTR_ACCESSORS(MatMulBias, AscMatMulBiasIrAttrDef, kOffsetX, int64_t, PyLong_Check, PyLong_FromLong,
                         PyLong_AsLong, SetOffset_x, GetOffset_x)
DEFINE_IR_ATTR_ACCESSORS(MatMulBias, AscMatMulBiasIrAttrDef, kEnableHf32, int64_t, PyLong_Check, PyLong_FromLong,
                         PyLong_AsLong, SetEnable_hf32, GetEnable_hf32)
DEFINE_IR_ATTR_ACCESSORS(BatchMatMul, AscBatchMatMulIrAttrDef, kHasRelu, int64_t, PyLong_Check, PyLong_FromLong,
                         PyLong_AsLong, SetHas_relu, GetHas_relu)
DEFINE_IR_ATTR_ACCESSORS(BatchMatMul, AscBatchMatMulIrAttrDef, kAdjX1, int64_t, PyLong_Check, PyLong_FromLong,
                         PyLong_AsLong, SetAdj_x1, GetAdj_x1)
DEFINE_IR_ATTR_ACCESSORS(BatchMatMul, AscBatchMatMulIrAttrDef, kAdjX2, int64_t, PyLong_Check, PyLong_FromLong,
                         PyLong_AsLong, SetAdj_x2, GetAdj_x2)
DEFINE_IR_ATTR_ACCESSORS(BatchMatMul, AscBatchMatMulIrAttrDef, kOffsetX, int64_t, PyLong_Check, PyLong_FromLong,
                         PyLong_AsLong, SetOffset_x, GetOffset_x)
DEFINE_IR_ATTR_ACCESSORS(BatchMatMul, AscBatchMatMulIrAttrDef, kEnableHf32, int64_t, PyLong_Check, PyLong_FromLong,
                         PyLong_AsLong, SetEnable_hf32, GetEnable_hf32)
DEFINE_IR_ATTR_ACCESSORS(BatchMatMulBias, AscBatchMatMulBiasIrAttrDef, kHasRelu, int64_t, PyLong_Check, PyLong_FromLong,
                         PyLong_AsLong, SetHas_relu, GetHas_relu)
DEFINE_IR_ATTR_ACCESSORS(BatchMatMulBias, AscBatchMatMulBiasIrAttrDef, kAdjX1, int64_t, PyLong_Check, PyLong_FromLong,
                         PyLong_AsLong, SetAdj_x1, GetAdj_x1)
DEFINE_IR_ATTR_ACCESSORS(BatchMatMulBias, AscBatchMatMulBiasIrAttrDef, kAdjX2, int64_t, PyLong_Check, PyLong_FromLong,
                         PyLong_AsLong, SetAdj_x2, GetAdj_x2)
DEFINE_IR_ATTR_ACCESSORS(BatchMatMulBias, AscBatchMatMulBiasIrAttrDef, kOffsetX, int64_t, PyLong_Check, PyLong_FromLong,
                         PyLong_AsLong, SetOffset_x, GetOffset_x)
DEFINE_IR_ATTR_ACCESSORS(BatchMatMulBias, AscBatchMatMulBiasIrAttrDef, kEnableHf32, int64_t, PyLong_Check,
                         PyLong_FromLong, PyLong_AsLong, SetEnable_hf32, GetEnable_hf32)
DEFINE_IR_ATTR_ACCESSORS(
    Scalar, AscScalarIrAttrDef, kValueAttr, std::string, PyUnicode_Check,
    [](const std::string &str) { return PyUnicode_FromString(str.c_str()); }, PyUnicode_AsUTF8, SetValue, GetValue)
DEFINE_IR_ATTR_ACCESSORS(LeakyRelu, AscLeakyReluIrAttrDef, kNegativeSlopeAttr, float, PyFloat_Check, PyFloat_FromDouble,
                         PyFloat_AsDouble, SetNegative_slope, GetNegative_slope)
DEFINE_IR_ATTR_ACCESSORS(Axpy, AscAxpyIrAttrDef, kAlphaAttr, float, PyFloat_Check, PyFloat_FromDouble, PyFloat_AsDouble,
                         SetAlpha, GetAlpha)
DEFINE_IR_ATTR_ACCESSORS(ShiftedChebyshevPolynomialT, AscShiftedChebyshevPolynomialTIrAttrDef, kNAttr, int64_t,
                         PyLong_Check, PyLong_FromLong, PyLong_AsLong, SetN, GetN)
DEFINE_IR_ATTR_ACCESSORS(ShiftedChebyshevPolynomialU, AscShiftedChebyshevPolynomialUIrAttrDef, kNAttr, int64_t,
                         PyLong_Check, PyLong_FromLong, PyLong_AsLong, SetN, GetN)
DEFINE_IR_ATTR_ACCESSORS(ShiftedChebyshevPolynomialV, AscShiftedChebyshevPolynomialVIrAttrDef, kNAttr, int64_t,
                         PyLong_Check, PyLong_FromLong, PyLong_AsLong, SetN, GetN)
DEFINE_IR_ATTR_ACCESSORS(ShiftedChebyshevPolynomialW, AscShiftedChebyshevPolynomialWIrAttrDef, kNAttr, int64_t,
                         PyLong_Check, PyLong_FromLong, PyLong_AsLong, SetN, GetN)
DEFINE_IR_ATTR_ACCESSORS(ChebyshevPolynomialT, AscChebyshevPolynomialTIrAttrDef, kNAttr, int64_t, PyLong_Check,
                         PyLong_FromLong, PyLong_AsLong, SetN, GetN)
DEFINE_IR_ATTR_ACCESSORS(ChebyshevPolynomialU, AscChebyshevPolynomialUIrAttrDef, kNAttr, int64_t, PyLong_Check,
                         PyLong_FromLong, PyLong_AsLong, SetN, GetN)
DEFINE_IR_ATTR_ACCESSORS(ChebyshevPolynomialV, AscChebyshevPolynomialVIrAttrDef, kNAttr, int64_t, PyLong_Check,
                         PyLong_FromLong, PyLong_AsLong, SetN, GetN)
DEFINE_IR_ATTR_ACCESSORS(ChebyshevPolynomialW, AscChebyshevPolynomialWIrAttrDef, kNAttr, int64_t, PyLong_Check,
                         PyLong_FromLong, PyLong_AsLong, SetN, GetN)
DEFINE_IR_ATTR_ACCESSORS(
    Unsupported, AscUnsupportedIrAttrDef, kErrorMsgAttr, std::string, PyUnicode_Check,
    [](const std::string &str) { return PyUnicode_FromString(str.c_str()); }, PyUnicode_AsUTF8, SetError_msg,
    GetError_msg)

template <>
PyObject *OpsOperatorIrAttr<af::ascir_op::IndirectLoad, kMaxAttr>::_getter(PyObject *self, void *closure) {
  (void)closure;
  auto *attr = GetValidatedIrAttr<af::ascir_op::IndirectLoad, af::ascir_op::IndirectLoad::AscIndirectLoadIrAttrDef>(
      self, "AscIndirectLoadIrAttrDef");
  GE_ASSERT_NOTNULL(attr);
  af::Expression v;
  PY_ASSERT_GRAPH_SUCCESS(attr->GetMax(v));
  return SizeExpr::FromSizeExpr(v);
}

template <>
int OpsOperatorIrAttr<af::ascir_op::IndirectLoad, kMaxAttr>::_setter(PyObject *self, PyObject *value, void *closure) {
  (void)closure;
  auto v = SizeExpr::AsSizeExpr(value);
  if ((!v.IsValid()) || (v.Serialize().get() == nullptr)) {
    return -1;
  }
  auto *attr = GetValidatedIrAttr<af::ascir_op::IndirectLoad, af::ascir_op::IndirectLoad::AscIndirectLoadIrAttrDef>(
      self, "AscIndirectLoadIrAttrDef");
  GE_ASSERT_NOTNULL(attr);
  PY_ASSERT_GRAPH_SUCCESS(attr->SetMax(v));
  return 0;
}

template <>
PyObject *OpsOperatorIrAttr<af::ascir_op::Load, kOffsetAttr>::_getter(PyObject *self, void *closure) {
  (void)closure;
  auto *attr = GetValidatedIrAttr<af::ascir_op::Load, af::ascir_op::Load::AscLoadIrAttrDef>(self, "AscLoadIrAttrDef");
  GE_ASSERT_NOTNULL(attr);
  af::Expression v;
  PY_ASSERT_GRAPH_SUCCESS(attr->GetOffset(v));
  return SizeExpr::FromSizeExpr(v);
}
template <>
int OpsOperatorIrAttr<af::ascir_op::Load, kOffsetAttr>::_setter(PyObject *self, PyObject *value, void *closure) {
  (void)closure;
  auto v = SizeExpr::AsSizeExpr(value);
  if ((!v.IsValid()) || (v.Serialize().get() == nullptr)) {
    return -1;
  }
  auto *attr = GetValidatedIrAttr<af::ascir_op::Load, af::ascir_op::Load::AscLoadIrAttrDef>(self, "AscLoadIrAttrDef");
  GE_ASSERT_NOTNULL(attr);
  PY_ASSERT_GRAPH_SUCCESS(attr->SetOffset(v));
  return 0;
}

template <>
PyObject *OpsOperatorIrAttr<af::ascir_op::Store, kOffsetAttr>::_getter(PyObject *self, void *closure) {
  (void)closure;
  auto *attr =
      GetValidatedIrAttr<af::ascir_op::Store, af::ascir_op::Store::AscStoreIrAttrDef>(self, "AscStoreIrAttrDef");
  GE_ASSERT_NOTNULL(attr);
  af::Expression v;
  PY_ASSERT_GRAPH_SUCCESS(attr->GetOffset(v));
  return SizeExpr::FromSizeExpr(v);
}
template <>
int OpsOperatorIrAttr<af::ascir_op::Store, kOffsetAttr>::_setter(PyObject *self, PyObject *value, void *closure) {
  (void)closure;
  auto v = SizeExpr::AsSizeExpr(value);
  if ((!v.IsValid()) || (v.Serialize().get() == nullptr)) {
    return -1;
  }
  auto *attr =
      GetValidatedIrAttr<af::ascir_op::Store, af::ascir_op::Store::AscStoreIrAttrDef>(self, "AscStoreIrAttrDef");
  GE_ASSERT_NOTNULL(attr);
  PY_ASSERT_GRAPH_SUCCESS(attr->SetOffset(v));
  return 0;
}

template <typename OpType, auto... Attrs>
struct AutoRegAttrHandle {
  static_assert(sizeof...(Attrs) > 0, "at least one attr is needed");
};

template <typename OpType, auto Attr>
struct AutoRegAttrHandle<OpType, Attr> {
  static void RegHandle(std::vector<PyGetSetDef> &getsetters) {
    getsetters.push_back(
        {Attr, OpsOperatorIrAttr<OpType, Attr>::_getter, OpsOperatorIrAttr<OpType, Attr>::_setter, "", nullptr});
  }
};
// 支持后续可能出现的一个op有多个属性的情况
template <typename OpType, auto Attr, auto... Attrs>
struct AutoRegAttrHandle<OpType, Attr, Attrs...> {
  static void RegHandle(std::vector<PyGetSetDef> &getsetters) {
    AutoRegAttrHandle<OpType, Attr>::RegHandle(getsetters);
    AutoRegAttrHandle<OpType, Attrs...>::RegHandle(getsetters);
  }
};

template <typename OpType>
const std::map<std::string, typename IrAttr<OpType>::handler> IrAttr<OpType>::attr_handlers = {
    {"Data", AutoRegAttrHandle<af::ascir_op::Data, kIndexAttr>::RegHandle},
    {"Scalar", AutoRegAttrHandle<af::ascir_op::Scalar, kValueAttr>::RegHandle},
    {"ScalarData", AutoRegAttrHandle<af::ascir_op::ScalarData, kIndexAttr>::RegHandle},
    {"IndexExpr", AutoRegAttrHandle<af::ascir_op::IndexExpr, kExprAttr>::RegHandle},
    {"Arange", AutoRegAttrHandle<af::ascir_op::Arange, kBaseAttr, kStepAttr>::RegHandle},
    {"Output", AutoRegAttrHandle<af::ascir_op::Output, kIndexAttr>::RegHandle},
    {"Load", AutoRegAttrHandle<af::ascir_op::Load, kOffsetAttr>::RegHandle},
    {"Store", AutoRegAttrHandle<af::ascir_op::Store, kOffsetAttr>::RegHandle},
    {"Gather", AutoRegAttrHandle<af::ascir_op::Gather, kAxisAttr>::RegHandle},
    {"IndirectLoad", AutoRegAttrHandle<af::ascir_op::IndirectLoad, kAxisAttr, kNegativeIndexSupportAttr,
                                       kNeedCheckBoundAttr, kMaxAttr>::RegHandle},
    {"MatMul",
     AutoRegAttrHandle<af::ascir_op::MatMul, kHasRelu, kOffsetX, kTransposeX1, kTransposeX2, kEnableHf32>::RegHandle},
    {"MatMulBias", AutoRegAttrHandle<af::ascir_op::MatMulBias, kHasRelu, kOffsetX, kTransposeX1, kTransposeX2,
                                     kEnableHf32>::RegHandle},
    {"BatchMatMul",
     AutoRegAttrHandle<af::ascir_op::BatchMatMul, kHasRelu, kOffsetX, kAdjX1, kAdjX2, kEnableHf32>::RegHandle},
    {"BatchMatMulBias",
     AutoRegAttrHandle<af::ascir_op::BatchMatMulBias, kHasRelu, kOffsetX, kAdjX1, kAdjX2, kEnableHf32>::RegHandle},
    {"LeakyRelu", AutoRegAttrHandle<af::ascir_op::LeakyRelu, kNegativeSlopeAttr>::RegHandle},
    {"Axpy", AutoRegAttrHandle<af::ascir_op::Axpy, kAlphaAttr>::RegHandle},
    {"ShiftedChebyshevPolynomialT", AutoRegAttrHandle<af::ascir_op::ShiftedChebyshevPolynomialT, kNAttr>::RegHandle},
    {"ShiftedChebyshevPolynomialU", AutoRegAttrHandle<af::ascir_op::ShiftedChebyshevPolynomialU, kNAttr>::RegHandle},
    {"ShiftedChebyshevPolynomialV", AutoRegAttrHandle<af::ascir_op::ShiftedChebyshevPolynomialV, kNAttr>::RegHandle},
    {"ShiftedChebyshevPolynomialW", AutoRegAttrHandle<af::ascir_op::ShiftedChebyshevPolynomialW, kNAttr>::RegHandle},
    {"ChebyshevPolynomialT", AutoRegAttrHandle<af::ascir_op::ChebyshevPolynomialT, kNAttr>::RegHandle},
    {"ChebyshevPolynomialU", AutoRegAttrHandle<af::ascir_op::ChebyshevPolynomialU, kNAttr>::RegHandle},
    {"ChebyshevPolynomialV", AutoRegAttrHandle<af::ascir_op::ChebyshevPolynomialV, kNAttr>::RegHandle},
    {"ChebyshevPolynomialW", AutoRegAttrHandle<af::ascir_op::ChebyshevPolynomialW, kNAttr>::RegHandle},
    {"Unsupported", AutoRegAttrHandle<af::ascir_op::Unsupported, kErrorMsgAttr>::RegHandle},
};
PyTypeObject &OpsOperatorTypeObject::GetPyType() {
  return pytype;
}

OpsOperatorTypeObject &OpsOperatorTypeObject::GetOpsType(PyTypeObject *type) {
  return *(ge::PtrToPtr<PyTypeObject, OpsOperatorTypeObject>(type));
}

void OpsOperatorTypeObject::InitPyType() {
  members.push_back({"attr", T_OBJECT_EX, member_attr_offset, 0, "Operator attributes."});
  for (uint32_t i = 0U; i < output_defs.size(); ++i) {
    members.push_back(OpsOperatorOutput::CreateMember(output_defs[i].first.c_str(),
                                                      member_input_output_offset + i * sizeof(PyObject *)));
  }
  if (op_type == kOutputOpType) {
    getsetters.push_back({"x", nullptr, OpsOperatorInput::_setter_or_setter_list, "", PyLong_FromLong(0)});
  } else {
    for (uint32_t i = 0U; i < input_defs.size(); ++i) {
      if (input_defs[i].second == af::IrInputType::kIrInputDynamic) {
        getsetters.push_back(
            {input_defs[i].first.c_str(), nullptr, OpsOperatorInput::_setter_list, "", PyLong_FromLong(i)});
      } else {
        getsetters.push_back({input_defs[i].first.c_str(), nullptr, OpsOperatorInput::_setter, "", PyLong_FromLong(i)});
      }
    }
  }
  methods.push_back({"infer_dtype", OpsOperatorMethod::InferDtype, METH_NOARGS, "Infer node output dtype"});
  methods.push_back({nullptr});
  members.push_back({nullptr});
  getsetters.push_back({nullptr});

  this->pytype = {PyVarObject_HEAD_INIT(nullptr, 0)};
  pytype.tp_name = op_type.c_str();
  pytype.tp_doc = op_type.c_str();
  pytype.tp_base = &Operator::type;
  pytype.tp_basicsize = object_size + output_defs.size() * sizeof(PyObject *);
  pytype.tp_itemsize = 0;
  pytype.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
  pytype.tp_new = tp_new;
  pytype.tp_init = tp_init;
  pytype.tp_dealloc = tp_dealloc;
  pytype.tp_members = members.data();
  pytype.tp_methods = methods.data();
  pytype.tp_getset = getsetters.data();
}
}  // namespace pyascir

static PyModuleDef GraphModule = {
    PyModuleDef_HEAD_INIT,
    "graph",
    "Graph module",
    -1,
};

PyMODINIT_FUNC PyInit_graph(void) {
  PyObject *m;

  pyautofuse_type_init();
  pyascir_type_init();
  pyascir_types_type_init();

  m = PyModule_Create(&GraphModule);
  PY_ASSERT_NOTNULL(m);
  return m;
}

static PyObject *UtilsDebugStr(PyObject *self_pyobject, PyObject *args) {
  (void)self_pyobject;
  PyObject *graph_object;
  if (PyArg_ParseTuple(args, "O!", &pyascir::HintGraph::type, &graph_object) == 0) {
    return PyErr_Format(PyExc_TypeError, "Argument must be a HintGraph object");
  }

  auto graph = ge::PtrToPtr<PyObject, pyascir::HintGraph::Object>(graph_object);
  auto debug_str = ascir::utils::DebugStr(*graph->graph);
  return PyUnicode_FromString(debug_str.c_str());
}

static PyObject *UtilsDumpGraph(PyObject *self_pyobject, PyObject *args) {
  (void)self_pyobject;
  PyObject *graph_object;
  PY_ASSERT(PyArg_ParseTuple(args, "O", &graph_object) != 0);
  if (PyObject_IsInstance(graph_object, ge::PtrToPtr<PyTypeObject, PyObject>(&pyascir::HintGraph::type)) != 0) {
    // 处理 HintGraph 类型
    auto graph = ge::PtrToPtr<PyObject, pyascir::HintGraph::Object>(graph_object);
    PY_ASSERT_NOTNULL(graph->graph);
    const auto compute_graph = af::AscGraphUtils::GetComputeGraph(*(graph->graph));
    af::GraphUtils::DumpGEGraph(compute_graph, compute_graph->GetName(), true);
    af::GraphUtils::DumpGEGraphToOnnx(*compute_graph, compute_graph->GetName(), true);
    af::GraphUtils::DumpGEGraphToReadable(compute_graph, compute_graph->GetName(), true);
  } else if (PyObject_IsInstance(graph_object, ge::PtrToPtr<PyTypeObject, PyObject>(&pyascir::FusedGraph::type)) != 0) {
    // 处理 FusedGraph 类型
    auto fused_graph = ge::PtrToPtr<PyObject, pyascir::FusedGraph::Object>(graph_object);
    PY_ASSERT_NOTNULL(fused_graph->graph);
    const auto &compute_graph = fused_graph->graph;
    PY_ASSERT_NOTNULL(compute_graph);
    af::GraphUtils::DumpGEGraph(compute_graph, compute_graph->GetName(), true);
    af::GraphUtils::DumpGEGraphToOnnx(*compute_graph, compute_graph->GetName(), true);
    af::GraphUtils::DumpGEGraphToReadable(compute_graph, compute_graph->GetName(), true);
  } else {
    PyErr_Format(PyExc_TypeError, "Argument must be a HintGraph or FusedGraph object, got %s",
                 Py_TYPE(graph_object)->tp_name);
    return nullptr;
  }
  Py_RETURN_NONE;
}
namespace {
// Template function for binary operations on SizeExpr (Max, Min, Mod, etc.)
template <af::Expression (*BinaryOp)(const af::Expression &, const af::Expression &)>
static PyObject *SizeExpr_BinaryOp(PyObject *self, PyObject *args) {
  (void)self;
  PyObject *left;
  PyObject *right;
  if (PyArg_ParseTuple(args, "OO", &left, &right) == 0) {
    return nullptr;
  }
  af::Expression left_expr = pyascir::SizeExpr::AsSizeExpr(left);
  PY_ASSERT_TRUE(left_expr.IsValid(), "left operand of binary operation is not a valid SizeExpr");
  af::Expression right_expr = pyascir::SizeExpr::AsSizeExpr(right);
  PY_ASSERT_TRUE(right_expr.IsValid(), "right operand of binary operation is not a valid SizeExpr");
  return pyascir::SizeExpr::FromSizeExpr(BinaryOp(left_expr, right_expr));
}

// Global Max, Min and Mod functions for SizeExpr
static PyObject *SizeExprMax(PyObject *self, PyObject *args) {
  return SizeExpr_BinaryOp<af::sym::Max>(self, args);
}

static PyObject *SizeExprMin(PyObject *self, PyObject *args) {
  return SizeExpr_BinaryOp<af::sym::Min>(self, args);
}

static PyObject *SizeExprMod(PyObject *self, PyObject *args) {
  return SizeExpr_BinaryOp<af::sym::Mod>(self, args);
}

PyMethodDef UtilsMethods[] = {
    {"debug_str", UtilsDebugStr, METH_VARARGS, "Get graph debug string"},
    {"dump", UtilsDumpGraph, METH_VARARGS, "Dump graph"},
    {"deserialize", reinterpret_cast<PyCFunction>(pyascir::UtilsDeserialize), METH_VARARGS, "info deserialize"},
    {"duration_record", reinterpret_cast<PyCFunction>(pyascir::UtilsDurationRecord), METH_VARARGS, "duration record"},
    {"report_durations", reinterpret_cast<PyCFunction>(pyascir::UtilsReportDurations), METH_VARARGS,
     "report durations"},
    {"set_platform", reinterpret_cast<PyCFunction>(pyascir::UtilsSetPlatform), METH_VARARGS,
     "set platform for platform context"},
    {"requires_full_soc_version", reinterpret_cast<PyCFunction>(pyascir::UtilsRequiresFullSocVersion), METH_VARARGS,
     "check whether npu arch requires full soc version"},
    {NULL}};

PyMethodDef AscirMethods[] = {{"Max", SizeExprMax, METH_VARARGS, "Return the maximum of two SizeExpr values"},
                              {"Min", SizeExprMin, METH_VARARGS, "Return the minimum of two SizeExpr values"},
                              {"Mod", SizeExprMod, METH_VARARGS, "Return the modulo of two SizeExpr values"},
                              {NULL}};
}  // namespace

static PyModuleDef UtilsModule = {
    PyModuleDef_HEAD_INIT, "utils", "Utils module", -1, UtilsMethods,
};

static PyModuleDef OpsModule = {
    PyModuleDef_HEAD_INIT,
    "ops",
    "Operators that ASCIR supports",
    -1,
};

static PyModuleDef AscirModule = {
    PyModuleDef_HEAD_INIT, "ascir", "AscendC IR", -1, AscirMethods,
};

static PyModuleDef DtypesModule = {
    PyModuleDef_HEAD_INIT,
    "dtypes",
    "Data types",
    -1,
};

static int OpsModule_init(PyObject *ascir_module) {
  auto module_types = std::vector{
      std::pair{"SizeExpr", &pyascir::SizeExpr::type},
      std::pair{"Axis", &pyascir::Axis::type},
      std::pair{"OpsOperatorOutput", &pyascir::OpsOperatorOutput::type},
      std::pair{"AscNodeAttr", &pyascir::AscNodeAttr::type},
      std::pair{"ApiInfo", &pyascir::ApiInfo::type},
      std::pair{"SchedInfo", &pyascir::SchedInfo::type},
      std::pair{"Operator", &pyascir::Operator::type},
      std::pair{"HintGraph", &pyascir::HintGraph::type},
      std::pair{"HintComputeGraph", &pyascir::HintComputeGraph::type},
      std::pair{"FusedGraph", &pyascir::FusedGraph::type},
      std::pair{"ShapeInfo", &pyascir::ShapeInfo::type},
  };
  for (auto [name, type] : module_types) {
    if (PyType_Ready(type) < 0) {
      return -1;
    }
    Py_INCREF(type);
    PyModule_AddObject(ascir_module, name, ge::PtrToPtr<PyTypeObject, PyObject>(type));
  }

  return 0;
}
namespace {
std::vector<pyascir::OpsOperatorTypeObject> kOpsOperators = {
#define OP(NAME) pyascir::OpsOperator<af::ascir_op::NAME>::CreateTypeObject(),
    REGISTERED_OPS
#undef OP
};
}  // namespace

void pyascir_type_init() {
  using namespace pyascir;
  // ApiInfo::type
  ApiInfo::type.tp_name = "ApiInfo";
  ApiInfo::type.tp_basicsize = sizeof(ApiInfo::Object);
  ApiInfo::type.tp_itemsize = 0;
  ApiInfo::type.tp_dealloc = ApiInfo::Dealloc;
  ApiInfo::type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
  ApiInfo::type.tp_doc = "Api compute type attrs.";
  ApiInfo::type.tp_methods = nullptr;
  ApiInfo::type.tp_members = nullptr;
  ApiInfo::type.tp_getset = ApiInfo::GetSetters;
  ApiInfo::type.tp_init = nullptr;
  ApiInfo::type.tp_new = nullptr;
  // SchedInfo::type
  SchedInfo::type.tp_name = "SchedInfo";
  SchedInfo::type.tp_basicsize = sizeof(SchedInfo::Object);
  SchedInfo::type.tp_itemsize = 0;
  SchedInfo::type.tp_dealloc = SchedInfo::Dealloc;
  SchedInfo::type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
  SchedInfo::type.tp_doc = "Scheduled info attrs.";
  SchedInfo::type.tp_methods = nullptr;
  SchedInfo::type.tp_members = nullptr;
  SchedInfo::type.tp_getset = SchedInfo::GetSetters;
  SchedInfo::type.tp_init = nullptr;
  SchedInfo::type.tp_new = nullptr;
  // AscNodeAttr::type
  AscNodeAttr::type.tp_name = "AscNodeAttr";
  AscNodeAttr::type.tp_basicsize = sizeof(AscNodeAttr::Object);
  AscNodeAttr::type.tp_itemsize = 0;
  AscNodeAttr::type.tp_dealloc = AscNodeAttr::Dealloc;
  AscNodeAttr::type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
  AscNodeAttr::type.tp_doc = "Node attributes.";
  AscNodeAttr::type.tp_methods = nullptr;
  AscNodeAttr::type.tp_members = AscNodeAttr::members;
  AscNodeAttr::type.tp_init = nullptr;
  AscNodeAttr::type.tp_new = nullptr;
  // OpsOperatorOutput::type
  OpsOperatorOutput::type.tp_name = "OpsOperatorOutput";
  OpsOperatorOutput::type.tp_basicsize = sizeof(OpsOperatorOutput::Object);
  OpsOperatorOutput::type.tp_itemsize = 0;
  OpsOperatorOutput::type.tp_dealloc = OpsOperatorOutput::Dealloc;
  OpsOperatorOutput::type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
  OpsOperatorOutput::type.tp_getset = OpsOperatorOutput::GetSetters;
}

PyMODINIT_FUNC PyInit_ascir(void) {
  pyautofuse_type_init();
  pyascir_type_init();
  pyascir_types_type_init();

  PyObject *ascir_module = PyModule_Create(&AscirModule);
  PY_ASSERT_NOTNULL(ascir_module);

  PyObject *dtypes_module = PyModule_Create(&DtypesModule);
  PY_ASSERT_NOTNULL(dtypes_module);

  for (auto entry : dtype_entries) {
    PyModule_AddObject(dtypes_module, entry.py_name, PyLong_FromLong(entry.dtype_value));
  }
  PyModule_AddObject(ascir_module, "dtypes", dtypes_module);

  if (OpsModule_init(ascir_module) < 0) {
    return nullptr;
  }

  static auto utils_module = PyModule_Create(&UtilsModule);
  PY_ASSERT_NOTNULL(utils_module);

  PyModule_AddObject(ascir_module, "utils", utils_module);

  static auto ops_module = PyModule_Create(&OpsModule);
  PY_ASSERT_NOTNULL(ops_module);

  kOpsOperators.emplace_back(pyascir::OpsOperator<geir_op::AscBackend>::CreateTypeObject());
  kOpsOperators.emplace_back(pyascir::OpsOperator<geir_op::AscGraph>::CreateTypeObject());
  for (auto &type : kOpsOperators) {
    type.InitPyType();
    if (PyType_Ready(&type.GetPyType()) < 0) {
      return nullptr;
    }
    Py_INCREF(&type.GetPyType());
    PyModule_AddObject(ops_module, type.op_type.c_str(), ge::PtrToPtr<PyTypeObject, PyObject>(&type.GetPyType()));
  }

  PyModule_AddObject(ascir_module, "ops", ops_module);
  return ascir_module;
}
