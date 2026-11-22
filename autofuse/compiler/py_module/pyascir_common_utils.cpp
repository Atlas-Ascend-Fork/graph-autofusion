/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "pyascir_common_utils.h"
#include <google/protobuf/text_format.h>
#include "proto/af_ir.pb.h"
#include "nlohmann/json.hpp"
#include "graph/ascendc_ir/utils/asc_graph_utils.h"
#include "graph/detail/model_serialize_imp.h"
#include "common/ge_common/debug/log.h"
#include "attribute_group/attr_group_shape_env.h"
#include "common/platform_context.h"
#include "gen_model_info/api_perf_register/soc_perf_config.h"

#include "pyascir_types.h"
#include "ascgen_log.h"
#include "common/scope_tracing_recorder.h"

namespace pyascir {
bool ShapeInfoDeserialize(const std::string to_be_deserialized, PyObject *py_obj) {
  try {
    auto shape_info = reinterpret_cast<pyascir::ShapeInfo::Object *>(py_obj);
    nlohmann::json j = nlohmann::json::parse(to_be_deserialized);
    if (j.is_object()) {
      for (const auto &[dim_name, dim_value] : j.items()) {
        if (dim_value.is_string()) {
          shape_info->shape_info[dim_name] = dim_value.get<std::string>();
          LOG_PRINT("parse shape info %s : %s", dim_name.c_str(), shape_info->shape_info[dim_name].c_str());
        } else {
          ERROR_PRINT("parse shape info not string %s : %s", dim_name.c_str(), dim_value.dump().c_str());
          return false;
        }
      }
    }
    return true;
  } catch (const nlohmann::json::parse_error &e) {
    PyErr_SetString(PyExc_RuntimeError, "ShapeInfo parse fail");
    return false;
  }
}

bool OutputSymbolShapeDeserialize(PyObject *output_shape_obj, std::vector<std::vector<std::string>> &output_shape) {
  std::vector<std::string> inner_vec;
  size_t output_shape_obj_size = PyList_Size(output_shape_obj);
  for (size_t i = 0UL; i < output_shape_obj_size; i++) {
    PyObject *inner_list = PyList_GetItem(output_shape_obj, i);
    if (PyList_Check(inner_list) == kPythonFail) {
      ERROR_PRINT("OutputSymbolShape internal error, expected a list of lists");
      return false;
    }
    size_t inner_size = PyList_Size(inner_list);
    for (size_t j = 0UL; j < inner_size; j++) {
      PyObject *item = PyList_GetItem(inner_list, j);
      if (PyUnicode_Check(item) == kPythonFail) {
        ERROR_PRINT("OutputSymbolShape internal error, expected a unicode string");
        return false;
      }
      std::string item_str = PyUnicode_AsUTF8(item);
      inner_vec.push_back(item_str);
    }
    output_shape.push_back(inner_vec);
    inner_vec.clear();
  }
  return true;
}

bool ComputeGraphDeserialize(const std::string to_be_deserialized, PyObject *py_obj) {
  // construct HintComputeGraph instance
  pyascir::HintComputeGraph::Init(py_obj, nullptr, nullptr);
  auto compute_graph = reinterpret_cast<pyascir::HintComputeGraph::Object *>(py_obj);
  GE_CHK_BOOL_RET_SPECIAL_STATUS(compute_graph->compute_graph == nullptr, false, "compute_graph is nullptr");
  af::ModelSerializeImp serialize_imp;
  af::proto::GraphDef graph_def;
  GE_CHK_BOOL_RET_SPECIAL_STATUS(!google::protobuf::TextFormat::ParseFromString(to_be_deserialized, &graph_def), false,
                                 "ComputeGraph ParseFromString fail");
  GE_CHK_BOOL_RET_SPECIAL_STATUS(!serialize_imp.UnserializeGraph(compute_graph->compute_graph, graph_def), false,
                                 "ModelSerializeImp deserialize ComputeGraph fail");
  const auto shape_env_attr = compute_graph->compute_graph->GetAttrsGroup<af::ShapeEnvAttr>();
  if (shape_env_attr != nullptr) {
    SetCurShapeEnvContext(shape_env_attr);
  }
  LOG_PRINT("ComputeGraphDeserialize finish");
  return true;
}

PyObject *UtilsDeserialize(PyObject *self_pyobject, PyObject *args, PyObject *kwds) {
  (void)self_pyobject;
  (void)kwds;
  std::string type_graph = "asc_graph";
  std::string type_shape_info = "symbol_source_info";
  std::string type_compute_graph = "compute_graph";
  const char *type = nullptr;
  const char *obj = nullptr;
  std::string type_str;
  std::string obj_str;

  if (PyArg_ParseTuple(args, "ss", &type, &obj) == kPythonFail) {
    return PyErr_Format(PyExc_TypeError, "UtilsDeserialize param parse failed");
  }

  type_str = std::string(type);
  obj_str = std::string(obj);
  LOG_PRINT("UtilsDeserialize type: %s, obj: %s", type_str.c_str(), obj_str.c_str());
  if (type_str == type_graph) {
    af::AscGraph tmp_graph("fused_graph");
    auto ret = af::AscGraphUtils::DeserializeFromReadable(obj_str, tmp_graph);
    if (ret != 0) {
      return PyErr_Format(PyExc_TypeError, "HintGraph DeserializeFromReadable fail");
    }
    PyObject *hint_graph_obj = pyascir::HintGraph::New(&pyascir::HintGraph::type, nullptr, nullptr);
    GE_CHK_BOOL_RET_SPECIAL_STATUS(hint_graph_obj == nullptr, PyErr_Format(PyExc_TypeError, "HintGraph new fail"),
                                   "HintGraph new fail");
    // construct HinGraph instance
    PyObject *name_obj = PyUnicode_FromString(tmp_graph.GetName().c_str());
    PyObject *args = PyTuple_Pack(1, name_obj);
    auto ret_init = pyascir::HintGraph::Init(hint_graph_obj, args, nullptr);
    GE_CHK_BOOL_RET_SPECIAL_STATUS(ret_init != 0, PyErr_Format(PyExc_TypeError, "HintGraph init fail"),
                                   "HintGraph init fail");
    Py_DECREF(args);
    auto hint_graph = reinterpret_cast<pyascir::HintGraph::Object *>(hint_graph_obj);
    PY_ASSERT(hint_graph->graph->CopyFrom(tmp_graph));
    return reinterpret_cast<PyObject *>(hint_graph);
  } else if (type_str == type_shape_info) {
    PyObject *shape_info = pyascir::ShapeInfo::New(&pyascir::ShapeInfo::type, nullptr, nullptr);
    GE_CHK_BOOL_RET_SPECIAL_STATUS(shape_info == nullptr, PyErr_Format(PyExc_TypeError, "ShapeInfo new fail"),
                                   "ShapeInfo new fail");
    if (!pyascir::ShapeInfoDeserialize(obj_str, shape_info)) {
      ERROR_PRINT("ShapeInfo Deserialize fail");
      PyErr_Format(PyExc_TypeError, "ShapeInfo Deserialize fail");
      return nullptr;
    }
    return shape_info;
  } else if (type_str == type_compute_graph) {
    PyObject *hint_compute_graph_obj =
        pyascir::HintComputeGraph::New(&pyascir::HintComputeGraph::type, nullptr, nullptr);
    GE_CHK_BOOL_RET_SPECIAL_STATUS(hint_compute_graph_obj == nullptr,
                                   PyErr_Format(PyExc_TypeError, "HintComputeGraph new fail"),
                                   "HintComputeGraph new fail");
    if (!pyascir::ComputeGraphDeserialize(obj_str, hint_compute_graph_obj)) {
      ERROR_PRINT("HintComputeGraph Deserialize fail");
      return PyErr_Format(PyExc_TypeError, "HintComputeGraph Deserialize fail");
    }
    return hint_compute_graph_obj;
  }

  return PyErr_Format(PyExc_TypeError, "value of type is invalid");
}

bool PyListToVector(PyObject *list, std::vector<std::string> &vec) {
  if (PyList_Check(list) == kPythonFail) {
    return false;
  }
  size_t list_size = PyList_Size(list);
  for (size_t i = 0U; i < list_size; i++) {
    PyObject *item = PyList_GetItem(list, i);
    if (PyUnicode_Check(item) == kPythonFail) {
      return false;
    }
    std::string item_str = PyUnicode_AsUTF8(item);
    vec.push_back(item_str);
  }
  return true;
}

PyObject *UtilsReportDurations(PyObject *self_pyobject, PyObject *args, PyObject *kwds) {
  (void)self_pyobject;
  (void)args;
  (void)kwds;
  ReportTracingRecordDuration(ge::TracingModule::kAutoFuseBackend);
  Py_RETURN_NONE;
}

PyObject *UtilsDurationRecord(PyObject *self_pyobject, PyObject *args, PyObject *kwds) {
  (void)self_pyobject;
  (void)kwds;
  PyObject *target_list_obj = nullptr;
  long long start;
  long long duration;
  if (PyArg_ParseTuple(args, "OLL", &target_list_obj, &start, &duration) == kPythonFail) {
    return PyErr_Format(PyExc_TypeError, "UtilsDurationRecord param parse failed");
  }
  std::vector<std::string> va_args;
  PY_ASSERT(PyListToVector(target_list_obj, va_args), "target param is invalid");

  if ((start < 0L) || (duration < 0L)) {
    return PyErr_Format(PyExc_TypeError, "duration param is invalid");
  }
  TracingRecordDuration(ge::TracingModule::kAutoFuseBackend, va_args, static_cast<uint64_t>(start),
                        static_cast<uint64_t>(duration));
  Py_RETURN_NONE;
}

PyObject *UtilsSetPlatform(const PyObject *self_pyobject, PyObject *args, const PyObject *kwds) {
  (void)self_pyobject;
  (void)kwds;
  const char *platform = nullptr;
  long long vector_core_num = 0;
  long long ub_size = 0;
  const char *soc_version = nullptr;
  if (PyArg_ParseTuple(args, "sLL|s", &platform, &vector_core_num, &ub_size, &soc_version) == kPythonFail) {
    return PyErr_Format(PyExc_TypeError, "UtilsSetPlatform param parse failed, expected string, long, long");
  }
  PY_ASSERT_NOTNULL(platform);
  ge::PlatformInfo platform_info;
  platform_info.soc_ver = std::string(platform);
  platform_info.aiv_num = static_cast<int64_t>(vector_core_num);
  platform_info.ub_size = static_cast<int64_t>(ub_size);
  const std::string full_soc_version = (soc_version == nullptr) ? std::string() : std::string(soc_version);
  if (ge::PlatformContext::GetInstance().SetPlatformTarget(platform_info, full_soc_version) != af::SUCCESS) {
    return PyErr_Format(PyExc_RuntimeError, "UtilsSetPlatform failed: npu arch and soc version are inconsistent");
  }
  Py_RETURN_NONE;
}

PyObject *UtilsRequiresFullSocVersion(const PyObject *self_pyobject, PyObject *args, const PyObject *kwds) {
  (void)self_pyobject;
  (void)kwds;
  const char *npu_arch = nullptr;
  if (PyArg_ParseTuple(args, "s", &npu_arch) == kPythonFail) {
    return PyErr_Format(PyExc_TypeError, "UtilsRequiresFullSocVersion param parse failed, expected string");
  }

  bool required = false;
  if (att::RequiresFullSocVersion(std::string(npu_arch), required) != af::SUCCESS) {
    return PyErr_Format(PyExc_RuntimeError, "invalid npu arch: %s", npu_arch);
  }
  if (required) {
    Py_RETURN_TRUE;
  }
  Py_RETURN_FALSE;
}
}  // namespace pyascir
