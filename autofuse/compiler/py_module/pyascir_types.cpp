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

#include <algorithm>

#include "ascir_ops.h"
#include "ascir_ops_utils.h"
#include "ascgen_log.h"
#include "graph/utils/type_utils.h"
#include "graph/ascendc_ir/utils/asc_graph_utils.h"
#include "common/platform_context.h"

#include "pyascir_types.h"
#include "common/common_utils.h"

using ascir::ImplGraph;

/** SizeExpr */
namespace pyascir {
// 生成推导dtype的映射
using InferDtypeFunc = Status (*)(const std::vector<af::DataType> &input_dtypes,
                                  std::vector<af::DataType> &expect_output_dtypes, const std::string &npu_arch);
std::map<std::string, pyascir::InferDtypeFunc> kInferDtypeFuncs = {
#define OP(NAME) {#NAME, af::ascir_op::NAME::InferDataType},
    REGISTERED_OPS
#undef OP
};
namespace {
std::string DataTypesToString(const std::vector<af::DataType> &dtypes) {
  if (dtypes.empty()) {
    return "[]";
  }
  std::string result = "[" + ge::TypeUtils::DataTypeToSerialString(dtypes[0]);
  for (size_t i = 1U; i < dtypes.size(); ++i) {
    result += ", " + ge::TypeUtils::DataTypeToSerialString(dtypes[i]);
  }
  return result + "]";
}

InferDtypeFunc GetInferDtypeFunc(const std::string &node_type) {
  auto iter = kInferDtypeFuncs.find(node_type);
  PY_ASSERT(iter != kInferDtypeFuncs.end(), "%s has no infer dtype func", node_type.c_str());
  PY_ASSERT_NOTNULL(iter->second, "%s has invalid dtype func", node_type.c_str());
  return iter->second;
}

bool ProcessRequiredInput(const af::AscNodePtr &node, size_t index, size_t count,
                          std::vector<af::DataType> &input_dtypes) {
  PY_ASSERT_EQ(count, 1U);
  PY_ASSERT(static_cast<uint32_t>(index) < node->inputs.Size());
  const auto &tensor = node->inputs[index];
  input_dtypes.push_back(tensor.attr.dtype);
  return true;
}

bool ProcessDynamicInput(const af::AscNodePtr &node, size_t index, size_t count,
                         std::vector<af::DataType> &input_dtypes) {
  std::set<af::DataType> unique_dtypes;
  for (size_t i = index; i < index + count; ++i) {
    PY_ASSERT(static_cast<uint32_t>(i) < node->inputs.Size());
    unique_dtypes.insert(node->inputs[i].attr.dtype);
  }
  PY_ASSERT(unique_dtypes.size() == 1U, "%s dynamic_input should have uniform dtypes", node->GetOpDesc()->GetNamePtr());
  input_dtypes.push_back(*unique_dtypes.begin());
  return true;
}

bool CollectInputDtypesForOutput(const af::AscNodePtr &node, std::vector<af::DataType> &input_dtypes) {
  std::set<af::DataType> unique_dtypes;
  for (const auto input : node->inputs()) {
    unique_dtypes.insert(input->attr.dtype);
  }
  PY_ASSERT(unique_dtypes.size() == 1U, "%s %s should have uniform dtypes", node->GetNamePtr(), node->GetTypePtr());
  input_dtypes.push_back(*unique_dtypes.begin());
  return true;
}

bool CollectInputDtypes(const af::AscNodePtr &node, std::vector<af::DataType> &input_dtypes) {
  if (node->GetType() == af::ascir_op::Output::Type) {
    // Output因为前面做了一个可变ir的操作，即ir是必选输入，但是实际行为支持是动态输入或者必选两种，因此特殊处理一下
    return CollectInputDtypesForOutput(node, input_dtypes);
  }
  const auto op_desc = node->GetOpDesc();
  PY_ASSERT_NOTNULL(op_desc, "Internal error!");

  const auto &ir_inputs = op_desc->GetIrInputs();
  std::map<size_t, std::pair<size_t, size_t>> ir_input_2_range;
  PY_ASSERT_GRAPH_SUCCESS(af::OpDescUtils::GetIrInputRawDescRange(op_desc, ir_input_2_range),
                          "Op %s %s has invalid ir desc", op_desc->GetNamePtr(), op_desc->GetTypePtr());

  size_t index = 0UL;
  for (size_t ir_input_index = 0UL; ir_input_index < ir_inputs.size(); ++ir_input_index) {
    const auto &range_iter = ir_input_2_range.find(ir_input_index);
    PY_ASSERT(range_iter != ir_input_2_range.end(), "Invalid ir_input_index: %zu", ir_input_index);

    const auto &start_and_count = range_iter->second;
    const auto count = start_and_count.second;
    const auto &ir_input_type = ir_inputs[ir_input_index].second;

    switch (ir_input_type) {
      case af::IrInputType::kIrInputRequired:
        if (!ProcessRequiredInput(node, index, count, input_dtypes)) {
          return false;
        }
        break;
      case af::IrInputType::kIrInputDynamic:
        if (!ProcessDynamicInput(node, index, count, input_dtypes)) {
          return false;
        }
        break;
      default:
        PyErr_Format(PyExc_TypeError, "%s %s unsupported input type %ld at ir index %zu", op_desc->GetNamePtr(),
                     op_desc->GetTypePtr(), static_cast<int64_t>(ir_input_type), ir_input_index);
        return false;
    }
    index += count;
  }
  return true;
}

bool HasDynamicOutput(const af::OpDescPtr &op_desc) {
  const auto &ir_outputs = op_desc->GetIrOutputs();
  return std::any_of(ir_outputs.begin(), ir_outputs.end(),
                     [](const auto &output_def) { return output_def.second == af::IrOutputType::kIrOutputDynamic; });
}

bool DoDynamicOutputInference(const af::AscNodePtr &node, InferDtypeFunc infer_func,
                              const std::vector<af::DataType> &input_dtypes, std::vector<af::DataType> &output_dtyps) {
  auto op_desc = node->GetOpDesc();
  PY_ASSERT_NOTNULL(op_desc, "op_desc is null!");
  const auto &ir_outputs = op_desc->GetIrOutputs();
  std::map<size_t, std::pair<size_t, size_t>> ir_output_2_range;
  PY_ASSERT_GRAPH_SUCCESS(af::OpDescUtils::GetIrOutputDescRange(op_desc, ir_output_2_range),
                          "Op %s %s has invalid ir desc", op_desc->GetNamePtr(), op_desc->GetTypePtr());

  bool has_complete_output_dtypes = true;
  for (size_t ir_output_index = 0UL; ir_output_index < ir_outputs.size(); ++ir_output_index) {
    const auto range_iter = ir_output_2_range.find(ir_output_index);
    PY_ASSERT(range_iter != ir_output_2_range.end(), "Invalid ir_output_index: %zu", ir_output_index);

    const auto start_index = range_iter->second.first;
    const auto count = range_iter->second.second;
    std::set<af::DataType> unique_dtypes;
    for (size_t output_index = start_index; output_index < start_index + count; ++output_index) {
      PY_ASSERT(output_index < op_desc->GetOutputsSize(), "Invalid output index: %zu", output_index);
      const auto dtype = node->outputs[output_index].attr.dtype;
      if (dtype == ge::DT_UNDEFINED) {
        has_complete_output_dtypes = false;
        unique_dtypes.clear();
        break;
      }
      unique_dtypes.insert(dtype);
    }
    if (!has_complete_output_dtypes) {
      output_dtyps.clear();
      break;
    }
    if (unique_dtypes.empty()) {
      has_complete_output_dtypes = false;
      output_dtyps.clear();
      break;
    }
    PY_ASSERT(unique_dtypes.size() == 1U, "%s dynamic_output should have uniform dtypes", op_desc->GetNamePtr());
    output_dtyps.push_back(*unique_dtypes.begin());
  }
  if (!has_complete_output_dtypes) {
    output_dtyps.clear();
  }

  std::string npu_arch;
  PY_ASSERT_SUCCESS(ge::PlatformContext::GetInstance().GetCurrentPlatformString(npu_arch), "Failed to get npu_arch");

  if (has_complete_output_dtypes) {
    PY_ASSERT_SUCCESS(infer_func(input_dtypes, output_dtyps, npu_arch),
                      "Check dtype failed for %s %s; input_dtypes: %s, output_dtypes: %s", node->GetNamePtr(),
                      node->GetTypePtr(), DataTypesToString(input_dtypes).c_str(),
                      DataTypesToString(output_dtyps).c_str());
    return true;
  }

  PY_ASSERT_SUCCESS(infer_func(input_dtypes, output_dtyps, npu_arch),
                    "Infer dtype failed for %s %s; input_dtypes: %s is not supported now", node->GetNamePtr(),
                    node->GetTypePtr(), DataTypesToString(input_dtypes).c_str(),
                    DataTypesToString(output_dtyps).c_str());

  PY_ASSERT_EQ(output_dtyps.size(), ir_outputs.size());
  std::vector<af::DataType> expanded_output_dtypes;
  for (size_t ir_output_index = 0UL; ir_output_index < ir_outputs.size(); ++ir_output_index) {
    const auto range_iter = ir_output_2_range.find(ir_output_index);
    PY_ASSERT(range_iter != ir_output_2_range.end(), "Invalid ir_output_index: %zu", ir_output_index);
    expanded_output_dtypes.insert(expanded_output_dtypes.end(), range_iter->second.second,
                                  output_dtyps[ir_output_index]);
  }

  PY_ASSERT_EQ(expanded_output_dtypes.size(), op_desc->GetOutputsSize());
  for (size_t i = 0UL; i < expanded_output_dtypes.size(); ++i) {
    op_desc->MutableOutputDesc(i)->SetDataType(expanded_output_dtypes[i]);
  }
  return true;
}

bool DoInference(const af::AscNodePtr &node, InferDtypeFunc infer_func, const std::vector<af::DataType> &input_dtypes,
                 std::vector<af::DataType> &output_dtyps) {
  auto op_desc = node->GetOpDesc();
  PY_ASSERT_NOTNULL(op_desc, "op_desc is null!");
  if (HasDynamicOutput(op_desc)) {
    return DoDynamicOutputInference(node, infer_func, input_dtypes, output_dtyps);
  }
  // 获取 npu_arch
  std::string npu_arch;
  PY_ASSERT_SUCCESS(ge::PlatformContext::GetInstance().GetCurrentPlatformString(npu_arch), "Failed to get npu_arch");

  // 收集非DT_UNDEFINED的预定义输出类型
  bool for_infer = true;
  for (const auto &tensor : node->outputs()) {
    if (tensor->attr.dtype == ge::DT_UNDEFINED) {
      break;
    }
    for_infer = false;
    output_dtyps.push_back(tensor->attr.dtype);
  }

  // 执行推导或者校验
  if (!for_infer) {
    PY_ASSERT_SUCCESS(infer_func(input_dtypes, output_dtyps, npu_arch),
                      "Check dtype failed for %s %s; input_dtypes: %s, output_dtypes: %s", node->GetNamePtr(),
                      node->GetTypePtr(), DataTypesToString(input_dtypes).c_str(),
                      DataTypesToString(output_dtyps).c_str());
    return true;
  }
  PY_ASSERT_SUCCESS(infer_func(input_dtypes, output_dtyps, npu_arch),
                    "Infer dtype failed for %s %s; input_dtypes: %s is not supported now", node->GetNamePtr(),
                    node->GetTypePtr(), DataTypesToString(input_dtypes).c_str(),
                    DataTypesToString(output_dtyps).c_str());

  PY_ASSERT_EQ(output_dtyps.size(), op_desc->GetOutputsSize());
  for (size_t i = 0UL; i < output_dtyps.size(); ++i) {
    op_desc->MutableOutputDesc(i)->SetDataType(output_dtyps[i]);
  }
  return true;
}
}  // namespace

PyNumberMethods SizeExpr::NumberMethods;
PyGetSetDef SizeExpr::getseters[] = {{"expression",
                                      [](PyObject *self, void *) -> PyObject * {
                                        auto obj = reinterpret_cast<SizeExpr::Object *>(self);
                                        PY_ASSERT_NOTNULL(obj);
                                        PY_ASSERT_NOTNULL(obj->expression);
                                        // 返回 expression的可读str而非裸指针
                                        return PyUnicode_FromString(obj->expression->Serialize().get());
                                      },
                                      nullptr, "Expression string", nullptr},
                                     {nullptr}};
PyTypeObject SizeExpr::type = {PyVarObject_HEAD_INIT(nullptr, 0)};

void SizeExpr::Dealloc(PyObject *self) {
  auto self_ = reinterpret_cast<SizeExpr::Object *>(self);
  delete self_->expression;
  Py_TYPE(self)->tp_free(reinterpret_cast<SizeExpr::Object *>(self));
}

PyObject *SizeExpr::New(PyTypeObject *type, PyObject *args, PyObject *kwargs) {
  (void)args;
  (void)kwargs;
  auto self = reinterpret_cast<SizeExpr::Object *>(type->tp_alloc(type, 0));
  self->expression = nullptr;
  return reinterpret_cast<PyObject *>(self);
}

int SizeExpr::Init(PyObject *self_pyobject, PyObject *args, PyObject *kwargs) {
  (void)kwargs;
  PyObject *num = nullptr;
  if (PyArg_ParseTuple(args, "|O", &num) == kPythonFail) {
    return -1;
  }

  auto self = reinterpret_cast<SizeExpr::Object *>(self_pyobject);
  // asc.SizeExpr() = 1
  if (num == nullptr) {
    auto symbol = new (std::nothrow) af::Symbol(1);
    if (symbol == nullptr) {
      return -1;
    }
    self->expression = symbol;
    return 0;
  }

  // ascir.SizeExpr(const_var)
  if (PyLong_Check(num) == kPythonSuccess) {
    auto symbol = new (std::nothrow) af::Symbol(PyLong_AsLong(num));
    if (symbol == nullptr) {
      return -1;
    }
    self->expression = symbol;
    return 0;
  } else if (PyObject_IsInstance(num, reinterpret_cast<PyObject *>(&SizeExpr::type)) == kPythonSuccess) {
    auto size = reinterpret_cast<SizeExpr::Object *>(num);
    self->expression = new (std::nothrow) af::Expression(*size->expression);
  }
  PyErr_Format(PyExc_TypeError, "Only support type of SizeExpr or long");
  return -1;
}

int SizeExpr::Init(PyObject *self_pyobject, const af::Expression &expr) {
  auto self = reinterpret_cast<SizeExpr::Object *>(self_pyobject);
  auto expression = new (std::nothrow) af::Expression(expr);
  if (expression == nullptr) {
    return -1;
  }
  self->expression = expression;
  return 0;
}

PyObject *SizeExpr::FromSizeExpr(const af::Expression &expr) {
  auto size = New(&SizeExpr::type, nullptr, nullptr);
  PY_ASSERT_NOTNULL(size);
  Init(size, expr);
  Py_IncRef(size);
  return size;
}

const af::Expression SizeExpr::AsSizeExpr(PyObject *obj) {
  if (PyLong_Check(obj) == kPythonSuccess) {
    return af::Symbol(PyLong_AsLong(obj));
  } else if (PyObject_IsInstance(obj, reinterpret_cast<PyObject *>(&SizeExpr::type)) == kPythonSuccess) {
    auto size = reinterpret_cast<SizeExpr::Object *>(obj);
    return af::Expression(*size->expression);
  }
  PyErr_Format(PyExc_TypeError, "Only support type of SizeExpr or long");
  return af::Expression();
}

const af::Expression SizeExpr::AsExpression(PyObject *obj) {
  if (PyObject_IsInstance(obj, reinterpret_cast<PyObject *>(&SizeExpr::type)) == kPythonSuccess) {
    auto size = reinterpret_cast<SizeExpr::Object *>(obj);
    if (size->expression == nullptr) {
      PyErr_SetString(PyExc_ValueError, "Expression is not initialized");
      return af::Expression();
    }
    return af::Expression(*size->expression);
  }
  PyErr_Format(PyExc_TypeError, "Only support type of Expression");
  return af::Expression();
}

PyObject *SizeExpr::Add(PyObject *self, PyObject *args) {
  af::Expression left = SizeExpr::AsSizeExpr(self);
  PY_ASSERT_TRUE(left.IsValid(), "left operand of add is not a valid SizeExpr");
  af::Expression right = SizeExpr::AsSizeExpr(args);
  PY_ASSERT_TRUE(right.IsValid(), "right operand of add is not a valid SizeExpr");
  return FromSizeExpr(left + right);
}

PyObject *SizeExpr::Div(PyObject *self, PyObject *args) {
  af::Expression left = SizeExpr::AsSizeExpr(self);
  PY_ASSERT_TRUE(left.IsValid(), "left operand of div is not a valid SizeExpr");
  af::Expression right = SizeExpr::AsSizeExpr(args);
  PY_ASSERT_TRUE(right.IsValid(), "right operand of div is not a valid SizeExpr");
  return FromSizeExpr(left / right);
}

PyObject *SizeExpr::Mul(PyObject *self, PyObject *args) {
  af::Expression left = SizeExpr::AsSizeExpr(self);
  PY_ASSERT_TRUE(left.IsValid(), "left operand of mul is not a valid SizeExpr");
  af::Expression right = SizeExpr::AsSizeExpr(args);
  PY_ASSERT_TRUE(right.IsValid(), "right operand of mul is not a valid SizeExpr");
  return FromSizeExpr(left * right);
}

PyObject *SizeExpr::Sub(PyObject *self, PyObject *args) {
  af::Expression left = SizeExpr::AsSizeExpr(self);
  PY_ASSERT_TRUE(left.IsValid(), "left operand of sub is not a valid SizeExpr");
  af::Expression right = SizeExpr::AsSizeExpr(args);
  PY_ASSERT_TRUE(right.IsValid(), "right operand of sub is not a valid SizeExpr");
  return FromSizeExpr(left - right);
}

PyObject *SizeExpr::Negate(PyObject *self) {
  af::Expression expr = SizeExpr::AsSizeExpr(self);
  PY_ASSERT_TRUE(expr.IsValid(), "operand of negate is not a valid SizeExpr");
  return FromSizeExpr(af::sym::Neg(expr));
}

PyObject *SizeExpr::Pow(PyObject *self, PyObject *args, PyObject *modulo) {
  (void)modulo;
  af::Expression left = SizeExpr::AsSizeExpr(self);
  PY_ASSERT_TRUE(left.IsValid(), "left operand of pow is not a valid SizeExpr");
  af::Expression right = SizeExpr::AsSizeExpr(args);
  PY_ASSERT_TRUE(right.IsValid(), "right operand of pow is not a valid SizeExpr");
  return FromSizeExpr(af::sym::Pow(left, right));
}

PyObject *SizeExpr::Remainder(PyObject *self, PyObject *args) {
  af::Expression left = SizeExpr::AsSizeExpr(self);
  PY_ASSERT_TRUE(left.IsValid(), "left operand of remainder is not a valid SizeExpr");
  af::Expression right = SizeExpr::AsSizeExpr(args);
  PY_ASSERT_TRUE(right.IsValid(), "right operand of remainder is not a valid SizeExpr");
  return FromSizeExpr(af::sym::Mod(left, right));
}

PyObject *SizeExpr::FloorDiv(PyObject *self, PyObject *args) {
  af::Expression left = SizeExpr::AsSizeExpr(self);
  PY_ASSERT_TRUE(left.IsValid(), "left operand of floor division is not a valid SizeExpr");
  af::Expression right = SizeExpr::AsSizeExpr(args);
  PY_ASSERT_TRUE(right.IsValid(), "right operand of floor division is not a valid SizeExpr");
  return FromSizeExpr(af::sym::Floor(left / right));
}

PyObject *SizeExpr::Compare(PyObject *self, PyObject *other, int op) {
  if (op == Py_EQ || op == Py_NE) {
    af::Expression left = SizeExpr::AsSizeExpr(self);
    af::Expression right = SizeExpr::AsSizeExpr(other);
    // richcompare 必须返回新引用: 直接 return Py_True/Py_False 会把单例的借用引用
    // 当新引用交给调用方, 调用方释放时对 True/False 多减一次引用计数;
    // Python 3.10 及以前 True/False 非永生对象, 减穿后在解释器退出阶段触发
    // 对 _Py_TrueStruct 的非法 free (munmap_chunk invalid pointer)。
    const bool equal = (left == right);
    if (op == Py_NE) {
      if (!equal) {
        Py_RETURN_TRUE;
      }
      Py_RETURN_FALSE;
    }
    if (equal) {
      Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
  }
  Py_RETURN_NOTIMPLEMENTED;
}
}  // namespace pyascir

/** Axis */
namespace pyascir {
PyMemberDef Axis::members[] = {{"id", T_INT, offsetof(Axis::Object, id), 0, "Axis id"},
                               {"size", T_OBJECT, offsetof(Axis::Object, size), 0, "Size expression"},
                               {"name", T_OBJECT, offsetof(Axis::Object, name), 0, "Name"},
                               {"type", T_OBJECT, offsetof(Axis::Object, type), 0, "Type"},
                               {nullptr}};

PyTypeObject Axis::type = {PyVarObject_HEAD_INIT(nullptr, 0)};

void Axis::Dealloc(PyObject *self) {
  auto self_ = reinterpret_cast<Axis::Object *>(self);
  Py_XDECREF(self_->size);
  Py_XDECREF(self_->name);
  Py_XDECREF(self_->type);
  Py_TYPE(self)->tp_free(self);
}

PyObject *Axis::New(PyTypeObject *type, PyObject *args, PyObject *kwargs) {
  (void)args;
  (void)kwargs;
  auto self = reinterpret_cast<Axis::Object *>(type->tp_alloc(type, 0));
  if (self != nullptr) {
    self->id = -1;
    self->size = Py_None;
    self->name = Py_None;
    self->type = Py_None;
  }
  return reinterpret_cast<PyObject *>(self);
}

int Axis::Init(PyObject *self_pyobject, int id, const af::Expression &size, const char *name, af::Axis::Type type) {
  auto size_object = SizeExpr::New(&SizeExpr::type, nullptr, nullptr);
  if (size_object == nullptr) {
    return -1;
  }

  SizeExpr::Init(size_object, size);
  Py_IncRef(size_object);

  auto self = reinterpret_cast<Axis::Object *>(self_pyobject);
  self->id = id;
  self->size = size_object;
  self->name = PyUnicode_FromString(name);
  self->type = PyUnicode_FromString(std::to_string(static_cast<uint32_t>(type)).c_str());
  return 0;
}
}  // namespace pyascir

/** Operator */
namespace pyascir {
PyMemberDef Operator::members[] = {{"name", T_OBJECT_EX, offsetof(Operator::Object, name)},
                                   {"type", T_OBJECT_EX, offsetof(Operator::Object, type)},
                                   {nullptr}};

PyMethodDef Operator::methods[] = {{nullptr}};

PyTypeObject Operator::type = {PyVarObject_HEAD_INIT(nullptr, 0)};

void Operator::Dealloc(PyObject *self_pyobject) {
  auto self = reinterpret_cast<Operator::Object *>(self_pyobject);

  Py_XDECREF(self->name);
  Py_XDECREF(self->type);

  if (self->op != nullptr) {
    // May the derived operator class will change and delete the op
    delete self->op;
  }

  Py_TYPE(self)->tp_free(self_pyobject);
}

PyObject *Operator::New(PyTypeObject *type, PyObject *args, PyObject *kwargs) {
  (void)args;
  (void)kwargs;
  auto self = reinterpret_cast<Operator::Object *>(type->tp_alloc(type, 0));
  PY_ASSERT_NOTNULL(self);

  self->name = nullptr;
  self->type = nullptr;
  self->op = nullptr;

  return reinterpret_cast<PyObject *>(self);
}

int Operator::Init(PyObject *self_pyobject, PyObject *args, PyObject *kwargs) {
  (void)kwargs;
  auto self = reinterpret_cast<Operator::Object *>(self_pyobject);

  char *name;
  char *type;
  if (PyArg_ParseTuple(args, "ss", &name, &type) == kPythonFail) {
    return -1;
  }

  self->name = PyUnicode_FromString(name);
  self->type = PyUnicode_FromString(type);
  self->op = new af::Operator(name, type);

  return 0;
}
}  // namespace pyascir

/** HintGraph */
namespace pyascir {
void HintGraph::Dealloc(PyObject *self_pyobject) {
  auto self = reinterpret_cast<HintGraph::Object *>(self_pyobject);

  Py_XDECREF(self->name);

  delete self->graph;

  Py_TYPE(self_pyobject)->tp_free(self_pyobject);
}

PyObject *HintGraph::New(PyTypeObject *type, PyObject *args, PyObject *kwargs) {
  (void)args;
  (void)kwargs;
  auto self = reinterpret_cast<HintGraph::Object *>(type->tp_alloc(type, 0));
  PY_ASSERT_NOTNULL(self);

  self->name = nullptr;
  self->graph = nullptr;

  return reinterpret_cast<PyObject *>(self);
}

int HintGraph::Init(PyObject *self_pyobject, PyObject *args, PyObject *kwargs) {
  (void)kwargs;
  const char *graph_name = "fused_graph";
  if ((args != nullptr) && (PyArg_ParseTuple(args, "|s", &graph_name) == kPythonFail)) {
    return -1;
  }
  auto self = reinterpret_cast<HintGraph::Object *>(self_pyobject);
  self->name = PyUnicode_FromString(graph_name);
  self->graph = new (std::nothrow) af::AscGraph(graph_name);
  if (self->graph == nullptr) {
    return -1;
  }
  return 0;
}

PyObject *HintGraph::CreateSize(PyObject *self_pyobject, PyObject *args) {
  char *size_var_name = nullptr;
  if (PyArg_ParseTuple(args, "s", &size_var_name) == kPythonFail) {
    return nullptr;
  }

  auto self = reinterpret_cast<HintGraph::Object *>(self_pyobject);
  auto expression = self->graph->CreateSizeVar(size_var_name);
  if (!expression.IsValid()) {
    return nullptr;
  }

  auto new_size_pyobject = SizeExpr::New(&SizeExpr::type, nullptr, nullptr);
  PY_ASSERT_NOTNULL(new_size_pyobject);
  SizeExpr::Init(new_size_pyobject, expression);

  Py_INCREF(new_size_pyobject);
  return new_size_pyobject;
}

PyObject *HintGraph::CreateAxis(PyObject *self_pyobject, PyObject *args, PyObject *kwargs) {
  (void)kwargs;
  char *name;
  PyObject *size;
  if (PyArg_ParseTuple(args, "sO", &name, &size) == kPythonFail) {
    return nullptr;
  }
  auto self = reinterpret_cast<HintGraph::Object *>(self_pyobject);
  auto new_axis = self->graph->CreateAxis(name, SizeExpr::AsSizeExpr(size));
  auto axis_object = Axis::New(&Axis::type, nullptr, nullptr);
  PY_ASSERT_NOTNULL(axis_object);
  Axis::Init(axis_object, new_axis.id, new_axis.size, new_axis.name.c_str(), new_axis.type);
  Py_INCREF(axis_object);
  return axis_object;
}

af::Expression GetAxisSize(const std::vector<af::AxisPtr> &all_axes, af::AxisId id) {
  for (auto &ax : all_axes) {
    if (ax->id == id) {
      return ax->size;
    }
  }
  return af::Symbol(0);
}
// To be optimized.
void UpdateEquivalentAxes(const std::map<af::AxisId, vector<af::AxisId>> &axis_map, af::AscGraph &asc_graph) {
  auto all_axis = asc_graph.GetAllAxis();
  for (auto node : asc_graph.GetAllNodes()) {
    auto sched_axis = node->attr.sched.axis;
    vector<af::AxisId> updated_axis;
    for (auto axis : sched_axis) {
      if (axis_map.find(axis) != axis_map.end()) {
        for (auto it : axis_map.at(axis)) {
          updated_axis.emplace_back(it);
        }
      } else {
        updated_axis.emplace_back(axis);
      }
    }
    node->attr.sched.axis = updated_axis;

    for (auto output : node->outputs()) {
      auto tensor_axis = output->attr.axis;
      auto strides = output->attr.strides;
      auto repeats = output->attr.repeats;

      vector<af::AxisId> updated_tensor_axis;
      vector<af::Expression> updated_strides;
      vector<af::Expression> updated_repeats;
      for (size_t i = 0UL; i < tensor_axis.size(); i++) {
        auto axis = tensor_axis[i];
        auto stride = strides[i];
        auto repeat = repeats[i];

        if (axis_map.find(axis) != axis_map.end()) {
          if (axis_map.at(axis).size() == 1) {
            updated_repeats.emplace_back(repeat);
          }
          for (size_t j = 0UL; j < axis_map.at(axis).size(); j++) {
            auto map_axis_id = axis_map.at(axis)[j];
            updated_tensor_axis.emplace_back(map_axis_id);
            if (axis_map.at(axis).size() > 1) {
              updated_repeats.emplace_back(GetAxisSize(all_axis, map_axis_id));
            }

            auto new_stride = stride;
            for (size_t n = j + 1UL; n < axis_map.at(axis).size(); n++) {
              auto axis_id = axis_map.at(axis)[n];
              new_stride = new_stride * GetAxisSize(all_axis, axis_id);
            }

            updated_strides.emplace_back(new_stride);
          }
        } else {
          updated_tensor_axis.emplace_back(axis);
          updated_strides.emplace_back(stride);
          updated_repeats.emplace_back(repeat);
        }
      }
      output->attr.axis = updated_tensor_axis;
      output->attr.strides = updated_strides;
      output->attr.repeats = updated_repeats;
    }
  }
}

PyObject *HintGraph::SetAxisMap(PyObject *self_pyobject, PyObject *args) {
  PyObject *axis_map_py;
  if (PyArg_ParseTuple(args, "O", &axis_map_py) == kPythonFail) {
    return nullptr;
  }
  std::map<af::AxisId, vector<af::AxisId>> axis_map;
  PyObject *key, *value;
  Py_ssize_t pos = 0;
  while (PyDict_Next(axis_map_py, &pos, &key, &value) == kPythonSuccess) {
    if (PyObject_IsInstance(key, reinterpret_cast<PyObject *>(&Axis::type)) == kPythonSuccess) {
      auto key_axis = reinterpret_cast<Axis::Object *>(key);
      for (int i = 0; i < PyList_Size(value); i++) {
        auto val = PyList_GetItem(value, i);
        auto val_axis = reinterpret_cast<Axis::Object *>(val);
        axis_map[key_axis->id].push_back(val_axis->id);
      }
    } else {
      PyErr_SetString(PyExc_TypeError, "Input type is invalid.");
      return nullptr;
    }
  }
  auto self = reinterpret_cast<HintGraph::Object *>(self_pyobject);
  PY_ASSERT_NOTNULL(self);
  PY_ASSERT_NOTNULL(self->graph);
  UpdateEquivalentAxes(axis_map, *self->graph);
  Py_RETURN_NONE;
}

PyObject *HintGraph::FromGraph(af::AscGraph *graph) {
  auto graph_object = reinterpret_cast<HintGraph::Object *>(HintGraph::New(&HintGraph::type, nullptr, nullptr));
  if (graph_object == nullptr) {
    return nullptr;
  }

  graph_object->name = PyUnicode_FromString(graph->GetName().c_str());
  graph_object->graph = new af::AscGraph(graph->GetName().c_str());
  if (graph_object->graph == nullptr) {
    Py_DECREF(graph_object);
    return nullptr;
  }

  graph_object->graph->CopyFrom(*graph);
  Py_INCREF(graph_object);
  return reinterpret_cast<PyObject *>(graph_object);
}

PyObject *HintGraph::InferDtype(PyObject *self_pyobject) {
  auto self = reinterpret_cast<HintGraph::Object *>(self_pyobject);
  PY_ASSERT_NOTNULL(self);
  auto ascgraph = self->graph;
  PY_ASSERT_NOTNULL(ascgraph, "graph is not ready");
  for (const auto &node : ascgraph->GetAllNodes()) {
    PY_ASSERT_NOTNULL(node, "node is invalid");
    // ProcessSingleNode内部已经写了py error信息，外部再次调用PY_ASSERT会覆盖
    GE_ASSERT(ProcessSingleNode(node), "node %s %s infer dtype failed", node->GetNamePtr(), node->GetTypePtr());
  }
  Py_RETURN_NONE;
}

bool HintGraph::ProcessSingleNode(const af::AscNodePtr &node) {
  const auto node_type = node->GetType();
  auto infer_func = GetInferDtypeFunc(node_type);
  std::vector<af::DataType> input_dtypes;
  GE_ASSERT(CollectInputDtypes(node, input_dtypes));
  std::vector<af::DataType> output_dtyps;
  return DoInference(node, infer_func, input_dtypes, output_dtyps);
}

PyObject *HintGraph::GetInputNum(PyObject *self_pyobject) {
  auto self = reinterpret_cast<HintGraph::Object *>(self_pyobject);
  int64_t input_num = 0;
  if (self->graph == nullptr) {
    PyErr_SetString(PyExc_TypeError, "graph is nullptr.");
    return PyLong_FromLong(-1);
  }
  for (auto input : self->graph->GetInputNodes()) {
    if (!af::ops::IsOps<af::ascir_op::Data>(input)) {
      PyErr_SetString(PyExc_TypeError, "Input type is invalid.");
      return PyLong_FromLong(-1);
    }
    input_num++;
  }
  return PyLong_FromLong(input_num);
}

PyObject *HintGraph::GetOutputNum(PyObject *self_pyobject) {
  auto self = reinterpret_cast<HintGraph::Object *>(self_pyobject);
  int64_t output_num = 0;
  if (self->graph == nullptr) {
    PyErr_SetString(PyExc_TypeError, "graph is nullptr.");
    return PyLong_FromLong(-1);
  }
  for (auto output : self->graph->GetAllNodes()) {
    if (af::ops::IsOps<af::ascir_op::Output>(output)) {
      output_num++;
    }
  }
  return PyLong_FromLong(output_num);
}

PyObject *HintGraph::GetName(PyObject *self_pyobject) {
  auto self = reinterpret_cast<HintGraph::Object *>(self_pyobject);
  if (self->graph == nullptr) {
    PyErr_SetString(PyExc_TypeError, "graph is nullptr.");
    return PyUnicode_FromString("");
  }
  return PyUnicode_FromString(self->graph->GetName().c_str());
}

PyObject *HintGraph::SetName(PyObject *self_pyobject, PyObject *args) {
  auto self = reinterpret_cast<HintGraph::Object *>(self_pyobject);
  char *graph_name = nullptr;
  if (PyArg_ParseTuple(args, "s", &graph_name) == kPythonFail) {
    PyErr_SetString(PyExc_TypeError, "name is invalid.");
    return nullptr;
  }
  self->name = PyUnicode_FromString(graph_name);
  auto compute_graph = af::AscGraphUtils::GetComputeGraph(*(self->graph));
  GE_ASSERT_NOTNULL(compute_graph);
  compute_graph->SetName(std::string(graph_name));

  Py_RETURN_NONE;
}

PyMemberDef HintGraph::members[] = {{"name", T_OBJECT_EX, offsetof(HintGraph::Object, name), 0, nullptr}, {nullptr}};
PyGetSetDef HintGraph::getsetters[] = {{"sched", HintGraph::GetSched, nullptr, "Graph schedule attributes.", nullptr},
                                       {nullptr}};

PyMethodDef HintGraph::methods[] = {
    {"create_size", reinterpret_cast<PyCFunction>(HintGraph::CreateSize), METH_VARARGS, "Create a size variable"},
    {"create_axis", reinterpret_cast<PyCFunction>(HintGraph::CreateAxis), METH_VARARGS | METH_KEYWORDS,
     "Create an axis"},
    {"set_axis_map", reinterpret_cast<PyCFunction>(HintGraph::SetAxisMap), METH_VARARGS, "Set graph axismap"},
    {"get_input_num", reinterpret_cast<PyCFunction>(HintGraph::GetInputNum), METH_NOARGS, "Get graph input num"},
    {"get_output_num", reinterpret_cast<PyCFunction>(HintGraph::GetOutputNum), METH_NOARGS, "Get graph output num"},
    {"get_name", reinterpret_cast<PyCFunction>(HintGraph::GetName), METH_NOARGS, "Get graph name"},
    {"set_name", reinterpret_cast<PyCFunction>(HintGraph::SetName), METH_VARARGS, "Set graph name"},
    {"infer_dtypes", reinterpret_cast<PyCFunction>(HintGraph::InferDtype), METH_NOARGS, "Infer dtypes"},
    {nullptr}};

PyTypeObject HintGraph::type = {PyVarObject_HEAD_INIT(nullptr, 0)};
}  // namespace pyascir

/** HintComputeGraph */
namespace pyascir {
void HintComputeGraph::Dealloc(PyObject *self_pyobject) {
  Py_TYPE(self_pyobject)->tp_free(self_pyobject);
}

PyObject *HintComputeGraph::New(PyTypeObject *type, PyObject *args, PyObject *kwargs) {
  (void)args;
  (void)kwargs;
  auto self = reinterpret_cast<HintComputeGraph::Object *>(type->tp_alloc(type, 0));
  PY_ASSERT_NOTNULL(self);
  self->compute_graph = nullptr;
  return reinterpret_cast<PyObject *>(self);
}

int HintComputeGraph::Init(PyObject *self_pyobject, PyObject *args, PyObject *kwargs) {
  (void)args;
  (void)kwargs;
  auto self = reinterpret_cast<HintComputeGraph::Object *>(self_pyobject);
  self->compute_graph = af::ComGraphMakeShared<af::ComputeGraph>("fused_graph");
  return 0;
}

PyObject *HintComputeGraph::GetInfo(PyObject *self_pyobject) {
  auto self = reinterpret_cast<HintComputeGraph::Object *>(self_pyobject);
  if (self->compute_graph == nullptr) {
    GELOGW("HintComputeGraph is null");
    return PyUnicode_FromString("");
  }
  std::stringstream ss;
  ss << self->compute_graph->GetName() << std::endl;
  ss << "Node num:" << self->compute_graph->GetDirectNodesSize() << std::endl;
  for (auto &node : self->compute_graph->GetDirectNodePtr()) {
    ss << "Node name: " << node->GetName() << ", Node type: " << node->GetType() << std::endl;
  }
  return PyUnicode_FromString(ss.str().c_str());
}

PyObject *HintComputeGraph::GetName(PyObject *self_pyobject) {
  auto self = reinterpret_cast<HintComputeGraph::Object *>(self_pyobject);
  GE_ASSERT_NOTNULL(self->compute_graph);
  std::string graph_name = self->compute_graph->GetName();
  return PyUnicode_FromString(graph_name.c_str());
}

PyMemberDef HintComputeGraph::members[] = {{nullptr}};
PyMethodDef HintComputeGraph::methods[] = {
    {"get_info", reinterpret_cast<PyCFunction>(HintComputeGraph::GetInfo), METH_NOARGS, "Get compute graph info"},
    {"get_name", reinterpret_cast<PyCFunction>(HintComputeGraph::GetName), METH_NOARGS, "Get compute graph name"},
    {nullptr}};

PyTypeObject HintComputeGraph::type = {PyVarObject_HEAD_INIT(nullptr, 0)};
}  // namespace pyascir

/** FusedGraph */
namespace pyascir {
void FusedGraph::Dealloc(PyObject *self_pyobject) {
  Py_TYPE(self_pyobject)->tp_free(self_pyobject);
}

PyObject *FusedGraph::New(PyTypeObject *type, PyObject *args, PyObject *kwargs) {
  (void)args;
  (void)kwargs;
  auto self = reinterpret_cast<FusedGraph::Object *>(type->tp_alloc(type, 0));
  PY_ASSERT_NOTNULL(self);
  self->graph = nullptr;
  return reinterpret_cast<PyObject *>(self);
}

int FusedGraph::Init(PyObject *self_pyobject, PyObject *args, PyObject *kwargs) {
  (void)kwargs;
  const char *graph_name = "fused_graph";
  if ((args != nullptr) && (PyArg_ParseTuple(args, "|s", &graph_name) == kPythonFail)) {
    return -1;
  }
  auto self = reinterpret_cast<FusedGraph::Object *>(self_pyobject);
  GE_CHK_BOOL_RET_SPECIAL_STATUS(self == nullptr, -1, "self is nullptr");
  self->graph = af::ComGraphMakeShared<af::ComputeGraph>(graph_name);
  if (self->graph == nullptr) {
    return -1;
  }
  return 0;
}

PyMemberDef FusedGraph::members[] = {{nullptr}};
PyMethodDef FusedGraph::methods[] = {{nullptr}};

PyTypeObject FusedGraph::type = {PyVarObject_HEAD_INIT(nullptr, 0)};
}  // namespace pyascir

/** ShapeInfo */
namespace pyascir {
void ShapeInfo::Dealloc(PyObject *self_pyobject) {
  auto self = reinterpret_cast<ShapeInfo::Object *>(self_pyobject);

  self->shape_info.clear();

  Py_TYPE(self_pyobject)->tp_free(self_pyobject);
}

PyObject *ShapeInfo::New(PyTypeObject *type, PyObject *args, PyObject *kwargs) {
  (void)args;
  (void)kwargs;
  ShapeInfo::Object *self = reinterpret_cast<ShapeInfo::Object *>(type->tp_alloc(type, 0));
  if (self != nullptr) {
    self->shape_info = std::map<std::string, std::string>();
  }
  return reinterpret_cast<PyObject *>(self);
}

int ShapeInfo::Init(PyObject *self_pyobject, PyObject *args, PyObject *kwargs) {
  (void)self_pyobject;
  (void)args;
  (void)kwargs;
  return 0;
}

PyMemberDef ShapeInfo::members[] = {{nullptr}};
PyMethodDef ShapeInfo::methods[] = {{nullptr}};

PyTypeObject ShapeInfo::type = {PyVarObject_HEAD_INIT(nullptr, 0)};
}  // namespace pyascir

/** FusedScheduledResult */
namespace pyascir {
static bool SetStaticMatMulInputShapes(const af::AscNodePtr &node, PyObject *attr_dict) {
  PyObject *input_shapes = PyList_New(node->inputs.Size());
  PY_ASSERT_NOTNULL(input_shapes);
  GE_DISMISSABLE_GUARD(input_shapes_guard, [input_shapes]() { Py_DECREF(input_shapes); });
  for (size_t i = 0U; i < node->inputs.Size(); ++i) {
    const auto &repeats = node->inputs[i].attr.repeats;
    PyObject *shape = PyList_New(repeats.size());
    PY_ASSERT_NOTNULL(shape);
    GE_DISMISSABLE_GUARD(shape_guard, [shape]() { Py_DECREF(shape); });
    for (size_t j = 0U; j < repeats.size(); ++j) {
      int64_t dim = 0;
      PY_ASSERT(repeats[j].GetConstValue(dim) && dim >= 0, "Static MatMul input shape must be concrete.");
      PyObject *value = PyLong_FromLongLong(dim);
      PY_ASSERT_NOTNULL(value);
      PY_ASSERT(PyList_SetItem(shape, j, value) == 0);
    }
    GE_DISMISS_GUARD(shape_guard);
    PY_ASSERT(PyList_SetItem(input_shapes, i, shape) == 0);
  }
  PY_ASSERT(PyDict_SetItemString(attr_dict, "input_shapes", input_shapes) == 0);
  return true;
}

static bool ProcessMatMulNode(const af::AscNodePtr &node, PyObject *attr_dict, bool is_static) {
  ascgen_utils::MatMulAttr mm_attr_data;
  PY_ASSERT_GRAPH_SUCCESS(ascgen_utils::ParseMatmulAttr(node, mm_attr_data));
  uint32_t length = 0U;
  PY_ASSERT_GRAPH_SUCCESS(ascgen_utils::GetCubeOutputTypeSize(node, length));

  PyDict_SetItemString(attr_dict, "has_relu", (mm_attr_data.has_relu != 0) ? Py_True : Py_False);
  PyDict_SetItemString(attr_dict, "is_batch", mm_attr_data.is_batch ? Py_True : Py_False);
  PyDict_SetItemString(attr_dict, "transpose_x1",
                       ((mm_attr_data.transpose_x1 != 0) || (mm_attr_data.adj_x1 != 0)) ? Py_True : Py_False);
  PyDict_SetItemString(attr_dict, "transpose_x2",
                       ((mm_attr_data.transpose_x2 != 0) || (mm_attr_data.adj_x2 != 0)) ? Py_True : Py_False);
  SET_DICT_LONG(attr_dict, "offset_x", mm_attr_data.offset_x);
  if (mm_attr_data.is_batch) {
    PyDict_SetItemString(attr_dict, "enable_hf32", mm_attr_data.enable_hf32 != 0 ? Py_True : Py_False);
  } else {
    SET_DICT_LONG(attr_dict, "enable_hf32", mm_attr_data.enable_hf32);
  }
  SET_DICT_LONG(attr_dict, "type_size", length);
  uint32_t mm_input_num = 0U;
  PY_ASSERT_GRAPH_SUCCESS(ascgen_utils::GetCubeInputNum(node, mm_input_num));
  SET_DICT_LONG(attr_dict, "input_num", mm_input_num);
  if (is_static) {
    return SetStaticMatMulInputShapes(node, attr_dict);
  }
  return true;
}

static bool ProcessConv2DNode(const af::AscNodePtr &node, PyObject *attr_dict) {
  ascgen_utils::Conv2DAttr conv_attr_data;
  PY_ASSERT_GRAPH_SUCCESS(ascgen_utils::ParseConv2DAttr(node, conv_attr_data));
  uint32_t length = 0U;
  PY_ASSERT_GRAPH_SUCCESS(ascgen_utils::GetCubeOutputTypeSize(node, length));

  PyObject *strides_list = PyList_New(conv_attr_data.strides.size());
  PY_ASSERT_NOTNULL(strides_list, "strides_list is not ready");
  for (size_t i = 0; i < conv_attr_data.strides.size(); ++i) {
    PyList_SetItem(strides_list, i, PyLong_FromLong(conv_attr_data.strides[i]));
  }
  PyDict_SetItemString(attr_dict, "strides", strides_list);
  Py_DECREF(strides_list);

  PyObject *pads_list = PyList_New(conv_attr_data.pads.size());
  PY_ASSERT_NOTNULL(pads_list, "pads_list is not ready");
  for (size_t i = 0; i < conv_attr_data.pads.size(); ++i) {
    PyList_SetItem(pads_list, i, PyLong_FromLong(conv_attr_data.pads[i]));
  }
  PyDict_SetItemString(attr_dict, "pads", pads_list);
  Py_DECREF(pads_list);

  PyObject *dilations_list = PyList_New(conv_attr_data.dilations.size());
  PY_ASSERT_NOTNULL(dilations_list, "dilations_list is not ready");
  for (size_t i = 0; i < conv_attr_data.dilations.size(); ++i) {
    PyList_SetItem(dilations_list, i, PyLong_FromLong(conv_attr_data.dilations[i]));
  }
  PyDict_SetItemString(attr_dict, "dilations", dilations_list);
  Py_DECREF(dilations_list);

  SET_DICT_LONG(attr_dict, "groups", conv_attr_data.groups);
  PyDict_SetItemString(attr_dict, "round_mode", PyUnicode_FromString(conv_attr_data.round_mode.c_str()));
  PyDict_SetItemString(attr_dict, "pad_mode", PyUnicode_FromString(conv_attr_data.pad_mode.c_str()));
  PyDict_SetItemString(attr_dict, "data_format", PyUnicode_FromString(conv_attr_data.data_format.c_str()));
  SET_DICT_LONG(attr_dict, "offset_x", conv_attr_data.offset_x);
  PyDict_SetItemString(attr_dict, "enable_hf32", conv_attr_data.enable_hf32 ? Py_True : Py_False);
  SET_DICT_LONG(attr_dict, "fixed_shift_value", conv_attr_data.fixed_shift_value);
  PyDict_SetItemString(attr_dict, "enable_relu0", conv_attr_data.enable_relu0 ? Py_True : Py_False);
  PyDict_SetItemString(attr_dict, "has_bias", conv_attr_data.has_bias ? Py_True : Py_False);
  PyDict_SetItemString(attr_dict, "has_offset_w", conv_attr_data.has_offset_w ? Py_True : Py_False);
  PyDict_SetItemString(attr_dict, "has_scale0", conv_attr_data.has_scale0 ? Py_True : Py_False);
  PyDict_SetItemString(attr_dict, "is_extend_conv2d", conv_attr_data.is_extend_conv2d ? Py_True : Py_False);
  PyDict_SetItemString(attr_dict, "output_dtype", PyUnicode_FromString(conv_attr_data.output_dtype.c_str()));
  PyDict_SetItemString(attr_dict, "input_dtype", PyUnicode_FromString(conv_attr_data.input_dtype.c_str()));
  SET_DICT_LONG(attr_dict, "type_size", length);
  uint32_t conv_input_num = 0U;
  PY_ASSERT_GRAPH_SUCCESS(ascgen_utils::GetCubeInputNum(node, conv_input_num));
  SET_DICT_LONG(attr_dict, "input_num", conv_input_num);
  std::string npu_arch;
  PY_ASSERT_SUCCESS(ge::PlatformContext::GetInstance().GetCurrentPlatformString(npu_arch), "Failed to get npu_arch");
  PyDict_SetItemString(attr_dict, "npu_arch", PyUnicode_FromString(npu_arch.c_str()));
  return true;
}

static bool ProcessImplGraphs(PyObject *result_dict, const std::vector<af::AscGraph> &impl_graphs, bool is_static) {
  for (auto &impl_graph : impl_graphs) {
    for (auto node : impl_graph.GetAllNodes()) {
      if (node->attr.api.compute_type != ge::ComputeType::kComputeCube) {
        continue;
      }
      PyObject *attr_dict = PyDict_New();
      PY_ASSERT_NOTNULL(attr_dict, "attr_dict is not ready");

      GE_DISMISSABLE_GUARD(attr_dict_guard, [attr_dict]() { Py_DECREF(attr_dict); });

      if (ascgen_utils::IsConv2DGraphType(impl_graph)) {
        if (!ProcessConv2DNode(node, attr_dict)) {
          return false;
        }
      } else {
        if (!ProcessMatMulNode(node, attr_dict, is_static)) {
          return false;
        }
      }
      PyDict_SetItemString(result_dict, "cube_attributes", attr_dict);

      GE_DISMISS_GUARD(attr_dict_guard);
      Py_DECREF(attr_dict);
      return true;
    }
  }
  return false;
}

void FusedScheduledResult::Dealloc(PyObject *self_pyobject) {
  Py_TYPE(self_pyobject)->tp_free(self_pyobject);
}

PyObject *FusedScheduledResult::New(PyTypeObject *type, PyObject *args, PyObject *kwargs) {
  (void)args;
  (void)kwargs;
  auto self = reinterpret_cast<FusedScheduledResult::Object *>(type->tp_alloc(type, 0));
  PY_ASSERT_NOTNULL(self);

  return reinterpret_cast<PyObject *>(self);
}

int32_t FusedScheduledResult::Init(PyObject *self_pyobject, PyObject *args, PyObject *kwargs) {
  (void)self_pyobject;
  (void)args;
  (void)kwargs;
  return 0;
}

PyObject *FusedScheduledResult::GetName(PyObject *self_pyobject) {
  auto self = reinterpret_cast<FusedScheduledResult::Object *>(self_pyobject);
  return PyUnicode_FromString(self->fused_schedule_result.fused_graph_name.GetString());
}

PyObject *FusedScheduledResult::GetInputNum(PyObject *self_pyobject) {
  auto self = reinterpret_cast<FusedScheduledResult::Object *>(self_pyobject);
  int64_t input_num = self->fused_schedule_result.input_nodes.size();
  return PyLong_FromLong(input_num);
}

PyObject *FusedScheduledResult::GetOutputNum(PyObject *self_pyobject) {
  auto self = reinterpret_cast<FusedScheduledResult::Object *>(self_pyobject);
  int64_t output_num = self->fused_schedule_result.output_nodes.size();
  return PyLong_FromLong(output_num);
}

PyObject *FusedScheduledResult::IsConvType(PyObject *self_pyobject) {
  auto self = reinterpret_cast<FusedScheduledResult::Object *>(self_pyobject);
  if (self == nullptr) {
    Py_RETURN_FALSE;
  }
  if (ascgen_utils::IsConv2DFusedScheduled(self->fused_schedule_result)) {
    Py_RETURN_TRUE;
  }

  Py_RETURN_FALSE;
}

PyObject *FusedScheduledResult::IsCubeType(PyObject *self_pyobject) {
  auto self = reinterpret_cast<FusedScheduledResult::Object *>(self_pyobject);
  if (self == nullptr) {
    Py_RETURN_FALSE;
  }
  for (auto scheduled_results : self->fused_schedule_result.node_idx_to_scheduled_results) {
    for (auto scheduled_result : scheduled_results) {
      if (scheduled_result.cube_type != ascir::CubeTemplateType::kDefault) {
        Py_RETURN_TRUE;
      }
    }
  }
  Py_RETURN_FALSE;
}

PyObject *FusedScheduledResult::GetCubeAttributes(PyObject *self_pyobject) {
  auto self = reinterpret_cast<FusedScheduledResult::Object *>(self_pyobject);
  PY_ASSERT_NOTNULL(self, "self_pyobject is not ready");
  PyObject *result_dict = PyDict_New();
  PY_ASSERT_NOTNULL(result_dict, "result_dict is not ready");

  // 创建守卫，确保在发生异常时释放result_dict
  GE_DISMISSABLE_GUARD(result_dict_guard, [result_dict]() { Py_DECREF(result_dict); });

  for (auto &scheduled_results : self->fused_schedule_result.node_idx_to_scheduled_results) {
    for (auto &scheduled_result : scheduled_results) {
      if (scheduled_result.cube_type == ascir::CubeTemplateType::kDefault) {
        continue;
      }
      for (auto &schedule_group : scheduled_result.schedule_groups) {
        if (ProcessImplGraphs(result_dict, schedule_group.impl_graphs,
                              ascgen_utils::IsFrontendStaticSchedResult(self->fused_schedule_result))) {
          GE_DISMISS_GUARD(result_dict_guard);
          return result_dict;
        }
      }
    }
  }

  GE_DISMISS_GUARD(result_dict_guard);
  return result_dict;
}

PyMemberDef FusedScheduledResult::members[] = {{nullptr}};
PyMethodDef FusedScheduledResult::methods[] = {
    {"get_input_num", reinterpret_cast<PyCFunction>(FusedScheduledResult::GetInputNum), METH_NOARGS,
     "Get graph input num"},
    {"get_output_num", reinterpret_cast<PyCFunction>(FusedScheduledResult::GetOutputNum), METH_NOARGS,
     "Get graph output num"},
    {"get_name", reinterpret_cast<PyCFunction>(FusedScheduledResult::GetName), METH_NOARGS, "Get graph name"},
    {"is_cube_type", reinterpret_cast<PyCFunction>(FusedScheduledResult::IsCubeType), METH_NOARGS, "Check cube type"},
    {"get_cube_attributes", reinterpret_cast<PyCFunction>(FusedScheduledResult::GetCubeAttributes), METH_NOARGS,
     "Get cube attributes"},
    {"is_conv_type", reinterpret_cast<PyCFunction>(FusedScheduledResult::IsConvType), METH_NOARGS, "Check conv type"},
    {nullptr}};

PyTypeObject FusedScheduledResult::type = {PyVarObject_HEAD_INIT(nullptr, 0)};
}  // namespace pyascir

namespace {
void pyascir_graph_types_type_init() {
  using namespace pyascir;
  // FusedGraph::type
  FusedGraph::type.tp_name = "FusedGraph";
  FusedGraph::type.tp_basicsize = sizeof(FusedGraph::Object);
  FusedGraph::type.tp_itemsize = 0;
  FusedGraph::type.tp_dealloc = FusedGraph::Dealloc;
  FusedGraph::type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
  FusedGraph::type.tp_doc = "FusedGraph object";
  FusedGraph::type.tp_methods = FusedGraph::methods;
  FusedGraph::type.tp_members = FusedGraph::members;
  FusedGraph::type.tp_init = FusedGraph::Init;
  FusedGraph::type.tp_new = FusedGraph::New;
  // HintGraph::type
  HintGraph::type.tp_name = "HintGraph";
  HintGraph::type.tp_basicsize = sizeof(HintGraph::Object);
  HintGraph::type.tp_itemsize = 0;
  HintGraph::type.tp_dealloc = HintGraph::Dealloc;
  HintGraph::type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
  HintGraph::type.tp_doc = "HintGraph object";
  HintGraph::type.tp_methods = HintGraph::methods;
  HintGraph::type.tp_members = HintGraph::members;
  HintGraph::type.tp_getset = HintGraph::getsetters;
  HintGraph::type.tp_init = HintGraph::Init;
  HintGraph::type.tp_new = HintGraph::New;
  // HintComputeGraph::type
  HintComputeGraph::type.tp_name = "HintComputeGraph";
  HintComputeGraph::type.tp_basicsize = sizeof(HintComputeGraph::Object);
  HintComputeGraph::type.tp_itemsize = 0;
  HintComputeGraph::type.tp_dealloc = HintComputeGraph::Dealloc;
  HintComputeGraph::type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
  HintComputeGraph::type.tp_doc = "HintComputeGraph object";
  HintComputeGraph::type.tp_methods = HintComputeGraph::methods;
  HintComputeGraph::type.tp_members = HintComputeGraph::members;
  HintComputeGraph::type.tp_init = HintComputeGraph::Init;
  HintComputeGraph::type.tp_new = HintComputeGraph::New;
}
}  // namespace

void pyascir_types_type_init() {
  using namespace pyascir;
  pyascir_graph_types_type_init();
  // SizeExpr::NumberMethods
  SizeExpr::NumberMethods.nb_add = SizeExpr::Add;
  SizeExpr::NumberMethods.nb_subtract = SizeExpr::Sub;
  SizeExpr::NumberMethods.nb_negative = SizeExpr::Negate;
  SizeExpr::NumberMethods.nb_multiply = SizeExpr::Mul;
  SizeExpr::NumberMethods.nb_power = SizeExpr::Pow;
  SizeExpr::NumberMethods.nb_true_divide = SizeExpr::Div;
  SizeExpr::NumberMethods.nb_remainder = SizeExpr::Remainder;
  SizeExpr::NumberMethods.nb_floor_divide = SizeExpr::FloorDiv;
  // SizeExpr::type
  SizeExpr::type.tp_name = "SizeExpr";
  SizeExpr::type.tp_basicsize = sizeof(SizeExpr::Object);
  SizeExpr::type.tp_itemsize = 0;
  SizeExpr::type.tp_dealloc = SizeExpr::Dealloc;
  SizeExpr::type.tp_as_number = &SizeExpr::NumberMethods;
  SizeExpr::type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
  SizeExpr::type.tp_doc = "SizeExpr object";
  SizeExpr::type.tp_richcompare = SizeExpr::Compare;
  SizeExpr::type.tp_methods = nullptr;
  SizeExpr::type.tp_getset = SizeExpr::getseters;
  SizeExpr::type.tp_init = SizeExpr::Init;
  SizeExpr::type.tp_new = SizeExpr::New;
  // Axis::type
  Axis::type.tp_name = "Axis";
  Axis::type.tp_basicsize = sizeof(Axis::Object);
  Axis::type.tp_itemsize = 0;
  Axis::type.tp_dealloc = Axis::Dealloc;
  Axis::type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
  Axis::type.tp_doc = "Axis object";
  Axis::type.tp_methods = nullptr;
  Axis::type.tp_members = Axis::members;
  Axis::type.tp_init = nullptr;
  Axis::type.tp_new = Axis::New;
  // Operator::type
  Operator::type.tp_name = "Operator";
  Operator::type.tp_basicsize = sizeof(Operator::Object);
  Operator::type.tp_itemsize = 0;
  Operator::type.tp_dealloc = Operator::Dealloc;
  Operator::type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
  Operator::type.tp_doc = "Operator.";
  Operator::type.tp_methods = Operator::methods;
  Operator::type.tp_members = Operator::members;
  Operator::type.tp_init = Operator::Init;
  Operator::type.tp_new = Operator::New;
  // ShapeInfo::type
  ShapeInfo::type.tp_name = "ShapeInfo";
  ShapeInfo::type.tp_basicsize = sizeof(ShapeInfo::Object);
  ShapeInfo::type.tp_itemsize = 0;
  ShapeInfo::type.tp_dealloc = ShapeInfo::Dealloc;
  ShapeInfo::type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
  ShapeInfo::type.tp_doc = "ShapeInfo object";
  ShapeInfo::type.tp_methods = ShapeInfo::methods;
  ShapeInfo::type.tp_members = ShapeInfo::members;
  ShapeInfo::type.tp_init = ShapeInfo::Init;
  ShapeInfo::type.tp_new = ShapeInfo::New;
  // FusedScheduledResult::type
  FusedScheduledResult::type.tp_name = "FusedScheduledResult";
  FusedScheduledResult::type.tp_basicsize = sizeof(FusedScheduledResult::Object);
  FusedScheduledResult::type.tp_itemsize = 0;
  FusedScheduledResult::type.tp_dealloc = FusedScheduledResult::Dealloc;
  FusedScheduledResult::type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
  FusedScheduledResult::type.tp_doc = "FusedScheduledResult object";
  FusedScheduledResult::type.tp_methods = FusedScheduledResult::methods;
  FusedScheduledResult::type.tp_members = FusedScheduledResult::members;
  FusedScheduledResult::type.tp_init = FusedScheduledResult::Init;
  FusedScheduledResult::type.tp_new = FusedScheduledResult::New;
  PyType_Ready(&FusedScheduledResult::type);
}
