/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <Python.h>

#include "optimize.h"
#include "codegen.h"
#include "pyascir_common_utils.h"

#include "pyascir.h"
#include "pyascir_types.h"
#include "ascgen_log.h"
#include "ascir/meta/asc_graph_dumper_context.h"
#include "common_utils.h"
#include "gen_model_info/api_perf_register/soc_perf_config.h"

namespace pyascir {
namespace {
void AssignDefaultIoIndex(af::AscGraph &graph) {
  int32_t data_index = 0;
  int32_t output_index = 0;
  for (const auto &node : graph.GetAllNodes()) {
    if (node->GetType() == "Data") {
      auto &attr = reinterpret_cast<af::AscDataIrAttrDef &>(*node->attr.ir_attr);
      attr.SetIndex(data_index++);
    } else if (node->GetType() == "Output") {
      node->attr.ir_attr = af::AscDataIrAttrDef().Clone();
      auto &attr = reinterpret_cast<af::AscDataIrAttrDef &>(*node->attr.ir_attr);
      attr.SetIndex(output_index++);
    } else {
    }
  }
}
class DumpGraphGuard {
 public:
  DumpGraphGuard() = default;
  ~DumpGraphGuard() {
    ascir::AscGraphDumperContext::GetThreadLocalCtx().DumpWatchedGraphs();
  }
  static void ReInit() {
    ascir::AscGraphDumperContext::GetThreadLocalCtx().ClearAllWatchGraphs();
  }
};
}  // namespace
class AutofuserOptions {
 public:
  struct Object {
    PyObject_HEAD

        optimize::OptimizerOptions *optimizer;
    codegen::CodegenOptions *codegen;
  };

  static PyTypeObject type;
  static void Dealloc(PyObject *self);
  static PyObject *New(PyTypeObject *type, PyObject *args, PyObject *kwds);
  static int Init(PyObject *self, PyObject *args, PyObject *kwds);
};

PyTypeObject AutofuserOptions::type = {PyVarObject_HEAD_INIT(nullptr, 0)};

void AutofuserOptions::Dealloc(PyObject *self) {
  auto self_ = reinterpret_cast<AutofuserOptions::Object *>(self);

  delete self_->optimizer;
  delete self_->codegen;

  Py_TYPE(self)->tp_free(self);
}

PyObject *AutofuserOptions::New(PyTypeObject *type, PyObject *args, PyObject *kwds) {
  (void)args;
  (void)kwds;
  auto self = reinterpret_cast<AutofuserOptions::Object *>(type->tp_alloc(type, 0));
  PY_ASSERT_NOTNULL(self);

  self->optimizer = nullptr;
  self->codegen = nullptr;

  return reinterpret_cast<PyObject *>(self);
}

int AutofuserOptions::Init(PyObject *self, PyObject *args, PyObject *kwds) {
  (void)args;
  auto self_ = reinterpret_cast<AutofuserOptions::Object *>(self);

  self_->optimizer = new optimize::OptimizerOptions();
  self_->codegen = new codegen::CodegenOptions();

  if (kwds != nullptr) {
    PyObject *tiling_lib_path_kwarg = PyDict_GetItemString(kwds, "tiling_lib_path");
    PyObject *tiling_lib_codegen_symbol_kwarg = PyDict_GetItemString(kwds, "tiling_lib_codegen_symbol");

    if (tiling_lib_path_kwarg != nullptr && PyUnicode_Check(tiling_lib_path_kwarg)) {
      Py_ssize_t tiling_lib_path_len;
      const char *tiling_lib_path = PyUnicode_AsUTF8AndSize(tiling_lib_path_kwarg, &tiling_lib_path_len);
      self_->codegen->tiling_lib_path = std::string(tiling_lib_path, tiling_lib_path_len);
    }

    if (tiling_lib_codegen_symbol_kwarg != nullptr && PyUnicode_Check(tiling_lib_codegen_symbol_kwarg)) {
      Py_ssize_t tiling_lib_codegen_symbol_len;
      const char *tiling_lib_codegen_symbol =
          PyUnicode_AsUTF8AndSize(tiling_lib_codegen_symbol_kwarg, &tiling_lib_codegen_symbol_len);
      self_->codegen->tiling_lib_codegen_symbol = std::string(tiling_lib_codegen_symbol, tiling_lib_codegen_symbol_len);
    }

    PyObject *graph_kwarg = PyDict_GetItemString(kwds, "graph_type");
    if (graph_kwarg != nullptr && PyLong_Check(graph_kwarg)) {
      int64_t graph_type = PyLong_AsLong(graph_kwarg);
      self_->optimizer->graph_type = static_cast<optimize::GraphType>(graph_type);
    }
  }

  return 0;
}

class Autofuser {
 public:
  struct Object {
    PyObject_HEAD

        optimize::Optimizer *optimizer;
    codegen::Codegen *codegen;
  };

  static PyTypeObject type;
  static PyMethodDef methods[];

  static void Dealloc(PyObject *self);
  static PyObject *New(PyTypeObject *type, PyObject *args, PyObject *kwds);
  static int Init(PyObject *self, PyObject *args, PyObject *kwds);

  static PyObject *AutofuseBackend(PyObject *self, PyObject *args, PyObject *kwds);
  static PyObject *Schedule(PyObject *self, PyObject *args, PyObject *kwds);
  static PyObject *Codegen(PyObject *self, PyObject *args, PyObject *kwds);
};

PyMethodDef Autofuser::methods[] = {
    {"autofuse_backend", reinterpret_cast<PyCFunction>(Autofuser::AutofuseBackend), METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"schedule", reinterpret_cast<PyCFunction>(Autofuser::Schedule), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"codegen", reinterpret_cast<PyCFunction>(Autofuser::Codegen), METH_VARARGS | METH_KEYWORDS, nullptr},
    {nullptr, nullptr, 0, nullptr}};

PyTypeObject Autofuser::type = {PyVarObject_HEAD_INIT(nullptr, 0)};

void Autofuser::Dealloc(PyObject *self) {
  auto self_ = reinterpret_cast<Autofuser::Object *>(self);

  delete self_->optimizer;
  delete self_->codegen;

  Py_TYPE(self)->tp_free(self);
}

PyObject *Autofuser::New(PyTypeObject *type, PyObject *args, PyObject *kwds) {
  (void)args;
  (void)kwds;
  auto self = reinterpret_cast<Autofuser::Object *>(type->tp_alloc(type, 0));
  PY_ASSERT_NOTNULL(self);

  self->optimizer = nullptr;
  self->codegen = nullptr;

  return reinterpret_cast<PyObject *>(self);
}

int Autofuser::Init(PyObject *self, PyObject *args, PyObject *kwds) {
  (void)kwds;
  auto self_ = reinterpret_cast<Autofuser::Object *>(self);

  AutofuserOptions::Object *options = nullptr;
  if (PyArg_ParseTuple(args, "O!", &AutofuserOptions::type, &options) == kPythonFail) {
    return -1;
  }

  self_->optimizer = new optimize::Optimizer(*options->optimizer);
  self_->codegen = new codegen::Codegen(*options->codegen);
  return 0;
}

PyObject *Autofuser::Schedule(PyObject *self, PyObject *args, PyObject *kwds) {
  (void)kwds;
  ascir::AscGraphDumperContext::GetThreadLocalCtx().ClearAllWatchGraphs();
  auto self_ = reinterpret_cast<Autofuser::Object *>(self);

  PyObject *graph_obj = nullptr;
  if (PyArg_ParseTuple(args, "O", &graph_obj) == kPythonFail) {
    return PyErr_Format(PyExc_ValueError, "autofuse input args error");
  }

  PyObject *fused_schedule_result_obj =
      pyascir::FusedScheduledResult::New(&pyascir::FusedScheduledResult::type, nullptr, nullptr);
  auto ret_init = pyascir::FusedScheduledResult::Init(fused_schedule_result_obj, nullptr, nullptr);
  GE_CHK_BOOL_RET_SPECIAL_STATUS(ret_init != 0, PyErr_Format(PyExc_TypeError, "FusedScheduledResult init fail"),
                                 "FusedScheduledResult init fail");
  auto fused_schedule_result = reinterpret_cast<pyascir::FusedScheduledResult::Object *>(fused_schedule_result_obj);
  if (PyObject_IsInstance(graph_obj, reinterpret_cast<PyObject *>(&pyascir::HintGraph::type)) == kPythonSuccess) {
    auto hint_graph = reinterpret_cast<pyascir::HintGraph::Object *>(graph_obj);
    PY_ASSERT_NOTNULL(hint_graph->graph);
    AssignDefaultIoIndex(*hint_graph->graph);
    auto ret = self_->optimizer->Optimize(*(hint_graph->graph), fused_schedule_result->fused_schedule_result);
    if (ret != 0) {
      ERROR_PRINT("Optimize fail ret %d", ret);
      ascir::AscGraphDumperContext::GetThreadLocalCtx().DumpWatchedGraphs();
      return PyErr_Format(PyExc_RuntimeError, "Optimize fail");
    }
    return Py_BuildValue("O", fused_schedule_result_obj);
  } else if (PyObject_IsInstance(graph_obj, reinterpret_cast<PyObject *>(&pyascir::FusedGraph::type)) ==
             kPythonSuccess) {
    auto fused_graph = reinterpret_cast<pyascir::FusedGraph::Object *>(graph_obj);
    auto ret = self_->optimizer->Optimize(fused_graph->graph, fused_schedule_result->fused_schedule_result);
    if (ret != 0) {
      ERROR_PRINT("Optimize fail ret %d", ret);
      ascir::AscGraphDumperContext::GetThreadLocalCtx().DumpWatchedGraphs();
      return PyErr_Format(PyExc_RuntimeError, "Optimize fail");
    }
    return Py_BuildValue("O", fused_schedule_result_obj);
  } else {
    return PyErr_Format(PyExc_RuntimeError, "schedule requires hint graph or fused graph");
  }
}

PyObject *Autofuser::Codegen(PyObject *self, PyObject *args, PyObject *kwds) {
  (void)kwds;
  DumpGraphGuard guard;
  auto self_ = reinterpret_cast<Autofuser::Object *>(self);

  PyObject *list_result_result = nullptr;
  if (PyArg_ParseTuple(args, "O", &list_result_result) == kPythonFail) {
    return PyErr_Format(PyExc_ValueError, "codegen requires a list of schedule result");
  }

  auto fused_schedule_result = ge::PtrToPtr<PyObject, pyascir::FusedScheduledResult::Object>(list_result_result);
  codegen::CodegenResult result;
  af::Status ret = af::FAILED;
  try {
    ret = self_->codegen->GenerateForInductor(fused_schedule_result->fused_schedule_result, result);
  } catch (const std::runtime_error &e) {
    GELOGE(af::FAILED, "Caught a runtime_error: %s", e.what());
  }
  PY_ASSERT_SUCCESS(ret, "Codegen generate kernel failed or abort");
  DumpGraphGuard::ReInit();
  return Py_BuildValue("sss", result.tiling_data.c_str(), result.tiling.c_str(), result.kernel.c_str());
}

PyObject *Autofuser::AutofuseBackend(PyObject *self, PyObject *args, PyObject *kwds) {
  (void)kwds;
  auto self_ = reinterpret_cast<Autofuser::Object *>(self);

  PyObject *graph_obj = nullptr;
  if (PyArg_ParseTuple(args, "O", &graph_obj) == kPythonFail) {
    return PyErr_Format(PyExc_ValueError, "autofuse input args error");
  }
  if (att::ValidateSocPerfEntry() != af::SUCCESS) {
    return PyErr_Format(PyExc_RuntimeError, "Invalid SoC performance target");
  }

  ascir::FusedScheduledResult fused_schedule_result;

  if (PyObject_IsInstance(graph_obj, reinterpret_cast<PyObject *>(&pyascir::HintGraph::type)) == kPythonSuccess) {
    auto hint_graph = reinterpret_cast<pyascir::HintGraph::Object *>(graph_obj);
    PY_ASSERT_NOTNULL(hint_graph->graph);
    AssignDefaultIoIndex(*hint_graph->graph);
    auto ret = self_->optimizer->Optimize(*(hint_graph->graph), fused_schedule_result);
    if (ret != 0) {
      ERROR_PRINT("Optimize fail ret %d", ret);
      return PyErr_Format(PyExc_RuntimeError, "Optimize fail");
    }
  } else if (PyObject_IsInstance(graph_obj, reinterpret_cast<PyObject *>(&pyascir::FusedGraph::type)) ==
             kPythonSuccess) {
    auto fused_graph = reinterpret_cast<pyascir::FusedGraph::Object *>(graph_obj);
    auto ret = self_->optimizer->Optimize(fused_graph->graph, fused_schedule_result);
    if (ret != 0) {
      ERROR_PRINT("Optimize fail ret %d", ret);
      return PyErr_Format(PyExc_RuntimeError, "Optimize fail");
    }
  } else {
    return PyErr_Format(PyExc_RuntimeError, "schedule requires hint graph or fused graph");
  }

  codegen::CodegenResult result;
  try {
    const auto ret = self_->codegen->GenerateForInductor(fused_schedule_result, result);
    if (ret != af::SUCCESS) {
      return PyErr_Format(PyExc_RuntimeError, "Codegen generate kernel failed");
    }
  } catch (const std::runtime_error &e) {
    GELOGE(af::FAILED, "Caught a runtime_error: %s", e.what());
    return PyErr_Format(PyExc_RuntimeError, "Codegen generate kernel failed: %s", e.what());
  }

  return Py_BuildValue("sss", result.tiling_data.c_str(), result.tiling.c_str(), result.kernel.c_str());
}

class Schedule {
 public:
  struct Object {
    PyObject_HEAD optimize::Optimizer *optimizer;
  };

  static PyTypeObject type;
  static PyMethodDef methods[];

  static void Dealloc(PyObject *self_pyobject);
  static PyObject *New(PyTypeObject *type, PyObject *args, PyObject *kwds);
  static int Init(PyObject *self_pyobject, PyObject *args, PyObject *kwds);

  static PyObject *ScheduleV1(PyObject *self_pyobject, PyObject *args, PyObject *kwds);
  static PyObject *ScheduleV2(PyObject *self_pyobject, PyObject *args, PyObject *kwds);
};

PyMethodDef Schedule::methods[] = {
    {"schedule", reinterpret_cast<PyCFunction>(Schedule::ScheduleV1), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"scheduleV2", reinterpret_cast<PyCFunction>(Schedule::ScheduleV2), METH_VARARGS | METH_KEYWORDS, nullptr},
    {nullptr, nullptr, 0, nullptr}};

PyTypeObject Schedule::type = {PyVarObject_HEAD_INIT(nullptr, 0)};

void Schedule::Dealloc(PyObject *self_pyobject) {
  auto self = reinterpret_cast<Schedule::Object *>(self_pyobject);

  delete self->optimizer;

  Py_TYPE(self_pyobject)->tp_free(self_pyobject);
}

PyObject *Schedule::New(PyTypeObject *type, PyObject *args, PyObject *kwds) {
  (void)args;
  (void)kwds;
  auto self = reinterpret_cast<Schedule::Object *>(type->tp_alloc(type, 0));
  PY_ASSERT_NOTNULL(self);
  self->optimizer = nullptr;

  return reinterpret_cast<PyObject *>(self);
}

int Schedule::Init(PyObject *self_pyobject, PyObject *args, PyObject *kwds) {
  (void)args;
  (void)kwds;
  auto self = reinterpret_cast<Schedule::Object *>(self_pyobject);
  auto options = new optimize::OptimizerOptions();
  self->optimizer = new optimize::Optimizer(*options);
  delete options;
  return 0;
}

PyObject *Schedule::ScheduleV1(PyObject *self_pyobject, PyObject *args, PyObject *kwds) {
  (void)kwds;
  ascir::AscGraphDumperContext::GetThreadLocalCtx().ClearAllWatchGraphs();
  auto self = reinterpret_cast<Schedule::Object *>(self_pyobject);

  pyascir::HintGraph::Object *hint_graph = nullptr;
  if (PyArg_ParseTuple(args, "O!", &pyascir::HintGraph::type, &hint_graph) == kPythonFail) {
    return PyErr_Format(PyExc_ValueError, "Schedule requires a hint graph");
  }
  PyObject *fused_schedule_result_obj =
      pyascir::FusedScheduledResult::New(&pyascir::FusedScheduledResult::type, nullptr, nullptr);
  auto ret_init = pyascir::FusedScheduledResult::Init(fused_schedule_result_obj, nullptr, nullptr);
  GE_CHK_BOOL_RET_SPECIAL_STATUS(ret_init != 0, PyErr_Format(PyExc_TypeError, "FusedScheduledResult init fail"),
                                 "FusedScheduledResult init fail");
  auto fused_schedule_result = reinterpret_cast<pyascir::FusedScheduledResult::Object *>(fused_schedule_result_obj);
  PY_ASSERT_NOTNULL(hint_graph->graph);
  AssignDefaultIoIndex(*hint_graph->graph);
  if (self->optimizer->Optimize(*hint_graph->graph, fused_schedule_result->fused_schedule_result) != af::SUCCESS) {
    ascir::AscGraphDumperContext::GetThreadLocalCtx().DumpWatchedGraphs();
  }
  return Py_BuildValue("O", fused_schedule_result_obj);
}

PyObject *Schedule::ScheduleV2(PyObject *self_pyobject, PyObject *args, PyObject *kwds) {
  (void)kwds;
  ascir::AscGraphDumperContext::GetThreadLocalCtx().ClearAllWatchGraphs();
  auto self = reinterpret_cast<Schedule::Object *>(self_pyobject);

  pyascir::HintComputeGraph::Object *hint_compute_graph = nullptr;
  if (PyArg_ParseTuple(args, "O!", &pyascir::HintComputeGraph::type, &hint_compute_graph) == kPythonFail) {
    return PyErr_Format(PyExc_ValueError, "Schedule requires a hint compute graph");
  }

  PyObject *fused_schedule_result_obj =
      pyascir::FusedScheduledResult::New(&pyascir::FusedScheduledResult::type, nullptr, nullptr);

  GE_CHK_BOOL_RET_SPECIAL_STATUS(hint_compute_graph->compute_graph == nullptr,
                                 PyErr_Format(PyExc_RuntimeError, "compute_graph is nullptr"), "HintGraph new fail");
  auto ret_init = pyascir::FusedScheduledResult::Init(fused_schedule_result_obj, nullptr, nullptr);
  GE_CHK_BOOL_RET_SPECIAL_STATUS(ret_init != 0, PyErr_Format(PyExc_TypeError, "FusedScheduledResult init fail"),
                                 "FusedScheduledResult init fail");

  auto fused_schedule_result = reinterpret_cast<pyascir::FusedScheduledResult::Object *>(fused_schedule_result_obj);
  auto ret = self->optimizer->Optimize(hint_compute_graph->compute_graph, fused_schedule_result->fused_schedule_result);
  if (ret != 0) {
    ascir::AscGraphDumperContext::GetThreadLocalCtx().DumpWatchedGraphs();
    ERROR_PRINT("Optimize fail ret %d", ret);
    return PyErr_Format(PyExc_RuntimeError, "Optimize fail");
  }

  return Py_BuildValue("O", fused_schedule_result_obj);
}

class CodeGen {
 public:
  struct Object {
    PyObject_HEAD codegen::Codegen *codegen;
  };

  static PyTypeObject type;
  static PyMethodDef methods[];

  static void Dealloc(PyObject *self_pyobject);
  static PyObject *New(PyTypeObject *type, PyObject *args, PyObject *kwds);
  static int Init(PyObject *self_pyobject, PyObject *args, PyObject *kwds);

  static PyObject *device_code_generator(PyObject *self_pyobject, PyObject *args, PyObject *kwds);
  static PyObject *host_code_generator(PyObject *self_pyobject, PyObject *args, PyObject *kwds);
  static PyObject *get_kernel_and_json_generator(PyObject *self_pyobject, PyObject *args, const PyObject *kwds);
  static PyObject *pgo_code_generator(PyObject *self_pyobject, PyObject *args, const PyObject *kwds);

  static af::Status HandleDeviceCodeGenForCVFusion(CodeGen::Object *self,
                                                   const ascir::FusedScheduledResult &fused_schedule_result,
                                                   PyObject *tiling_dict, PyObject *kernel_dict);
  static af::Status HandleDeviceCodeGenForNonCVFusion(CodeGen::Object *self,
                                                      const ascir::FusedScheduledResult &fused_schedule_result,
                                                      PyObject *tiling_dict, PyObject *kernel_dict);
  static af::Status HandleHostCodeGenForCVFusion(CodeGen::Object *self,
                                                 const ascir::FusedScheduledResult &fused_schedule_result,
                                                 const std::map<std::string, std::string> &symbol_source_info,
                                                 const char *pgo_dir, const char *vector_core_num,
                                                 PyObject *py_tilings);
  static af::Status HandleHostCodeGenForNonCVFusion(CodeGen::Object *self,
                                                    const ascir::FusedScheduledResult &fused_schedule_result,
                                                    const std::map<std::string, std::string> &symbol_source_info,
                                                    const char *pgo_dir, const char *vector_core_num,
                                                    PyObject *py_tilings);
  static PyObject *GenerateHostCodeResult(CodeGen::Object *self,
                                          pyascir::FusedScheduledResult::Object *fused_schedule_result,
                                          PyObject *shape_info_obj,
                                          const std::vector<std::vector<std::string>> &output_shape,
                                          const char *pgo_dir, const char *vector_core_num);
};

PyMethodDef CodeGen::methods[] = {
    {"device_code_generator", reinterpret_cast<PyCFunction>(CodeGen::device_code_generator),
     METH_VARARGS | METH_KEYWORDS, nullptr},
    {"host_code_generator", reinterpret_cast<PyCFunction>(CodeGen::host_code_generator), METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"get_kernel_and_json_generator", reinterpret_cast<PyCFunction>(CodeGen::get_kernel_and_json_generator),
     METH_VARARGS | METH_KEYWORDS, nullptr},
    {"pgo_code_generator", reinterpret_cast<PyCFunction>(CodeGen::pgo_code_generator), METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {nullptr, nullptr, 0, nullptr}};

PyTypeObject CodeGen::type = {PyVarObject_HEAD_INIT(nullptr, 0)};

void CodeGen::Dealloc(PyObject *self_pyobject) {
  auto self = reinterpret_cast<CodeGen::Object *>(self_pyobject);

  delete self->codegen;

  Py_TYPE(self_pyobject)->tp_free(self_pyobject);
}

PyObject *CodeGen::New(PyTypeObject *type, PyObject *args, PyObject *kwds) {
  (void)args;
  (void)kwds;
  auto self = reinterpret_cast<CodeGen::Object *>(type->tp_alloc(type, 0));
  PY_ASSERT_NOTNULL(self);
  self->codegen = nullptr;

  return reinterpret_cast<PyObject *>(self);
}

int CodeGen::Init(PyObject *self_pyobject, PyObject *args, PyObject *kwds) {
  (void)args;
  auto self = reinterpret_cast<CodeGen::Object *>(self_pyobject);
  auto options = new codegen::CodegenOptions();
  GE_CHK_BOOL_RET_SPECIAL_STATUS(options == nullptr, -1, "self is nullptr");

  if (kwds != nullptr) {
    PyObject *tiling_lib_path_kwarg = PyDict_GetItemString(kwds, "tiling_lib_path");
    PyObject *tiling_lib_codegen_symbol_kwarg = PyDict_GetItemString(kwds, "tiling_lib_codegen_symbol");

    if (tiling_lib_path_kwarg != nullptr && PyUnicode_Check(tiling_lib_path_kwarg)) {
      Py_ssize_t tiling_lib_path_len;
      const char *tiling_lib_path = PyUnicode_AsUTF8AndSize(tiling_lib_path_kwarg, &tiling_lib_path_len);
      options->tiling_lib_path = std::string(tiling_lib_path, tiling_lib_path_len);
    }

    if (tiling_lib_codegen_symbol_kwarg != nullptr && PyUnicode_Check(tiling_lib_codegen_symbol_kwarg)) {
      Py_ssize_t tiling_lib_codegen_symbol_len;
      const char *tiling_lib_codegen_symbol =
          PyUnicode_AsUTF8AndSize(tiling_lib_codegen_symbol_kwarg, &tiling_lib_codegen_symbol_len);
      options->tiling_lib_codegen_symbol = std::string(tiling_lib_codegen_symbol, tiling_lib_codegen_symbol_len);
    }
  }

  self->codegen = new codegen::Codegen(*options);
  delete options;
  return 0;
}

af::Status CodeGen::HandleDeviceCodeGenForCVFusion(CodeGen::Object *self,
                                                   const ascir::FusedScheduledResult &fused_schedule_result,
                                                   PyObject *tiling_dict, PyObject *kernel_dict) {
  ascir::FusedScheduledResult ub_schedule_result = fused_schedule_result;
  ascir::FusedScheduledResult common_schedule_result = fused_schedule_result;

  GE_ASSERT_SUCCESS(ascgen_utils::FilterCVFusionUBResult(ub_schedule_result));
  GE_ASSERT_SUCCESS(ascgen_utils::FilterCVFusionCommonResult(common_schedule_result));

  // 生成UB模板的代码，存在没有UB模板的场景
  std::string ub_tiling_data;
  std::string ub_kernel;
  if (ascgen_utils::HasCubeUBFusedScheduled(ub_schedule_result)) {
    ub_tiling_data = self->codegen->GenerateTilingData(ub_schedule_result);
    af::Status ret = self->codegen->GenerateKernel(ub_schedule_result, ub_kernel, false);
    GE_CHK_STATUS_RET(ret, "codegen generate ub kernel fail");
  } else {
    GELOGI("Has no cube ub fused schedule result, no need to generate ub tiling data and kernel");
  }

  // 生成兜底模板的代码，必须有兜底模板
  GE_ASSERT_TRUE(ascgen_utils::HasCubeCommonFusedScheduled(common_schedule_result));
  std::string common_tiling_data = self->codegen->GenerateTilingData(common_schedule_result);
  std::string common_kernel;
  af::Status ret = self->codegen->GenerateKernel(common_schedule_result, common_kernel, false);
  GE_CHK_STATUS_RET(ret, "codegen generate common kernel fail");

  // 将结果添加到字典中
  PyObjectGuard g_ub_tiling(PyUnicode_FromString(ub_tiling_data.c_str()));
  PyObjectGuard g_common_tiling(PyUnicode_FromString(common_tiling_data.c_str()));
  PyObjectGuard g_ub_kernel(PyUnicode_FromString(ub_kernel.c_str()));
  PyObjectGuard g_common_kernel(PyUnicode_FromString(common_kernel.c_str()));
  if (g_ub_tiling.get() == nullptr || g_common_tiling.get() == nullptr || g_ub_kernel.get() == nullptr ||
      g_common_kernel.get() == nullptr || PyDict_SetItemString(tiling_dict, "ub", g_ub_tiling.get()) != 0 ||
      PyDict_SetItemString(tiling_dict, "common", g_common_tiling.get()) != 0 ||
      PyDict_SetItemString(kernel_dict, "ub", g_ub_kernel.get()) != 0 ||
      PyDict_SetItemString(kernel_dict, "common", g_common_kernel.get()) != 0) {
    return af::FAILED;
  }

  return af::SUCCESS;
}

af::Status CodeGen::HandleDeviceCodeGenForNonCVFusion(CodeGen::Object *self,
                                                      const ascir::FusedScheduledResult &fused_schedule_result,
                                                      PyObject *tiling_dict, PyObject *kernel_dict) {
  // 非CV融合场景，生成一种结果，使用"default"
  std::string tiling_data = self->codegen->GenerateTilingData(fused_schedule_result);
  std::string kernel;
  af::Status ret = self->codegen->GenerateKernel(fused_schedule_result, kernel, false);
  GE_CHK_STATUS_RET(ret, "codegen generate kernel fail");

  // 将结果添加到字典中，使用"default"键值
  PyObjectGuard g_tiling(PyUnicode_FromString(tiling_data.c_str()));
  PyObjectGuard g_kernel(PyUnicode_FromString(kernel.c_str()));
  if (g_tiling.get() == nullptr || g_kernel.get() == nullptr ||
      PyDict_SetItemString(tiling_dict, "default", g_tiling.get()) != 0 ||
      PyDict_SetItemString(kernel_dict, "default", g_kernel.get()) != 0) {
    return af::FAILED;
  }

  return af::SUCCESS;
}

af::Status CodeGen::HandleHostCodeGenForCVFusion(CodeGen::Object *self,
                                                 const ascir::FusedScheduledResult &fused_schedule_result,
                                                 const std::map<std::string, std::string> &symbol_source_info,
                                                 const char *pgo_dir, const char *vector_core_num,
                                                 PyObject *py_tilings) {
  ascir::FusedScheduledResult ub_schedule_result = fused_schedule_result;
  ascir::FusedScheduledResult common_schedule_result = fused_schedule_result;

  GE_ASSERT_SUCCESS(ascgen_utils::FilterCVFusionUBResult(ub_schedule_result));
  GE_ASSERT_SUCCESS(ascgen_utils::FilterCVFusionCommonResult(common_schedule_result));

  // 生成UB模板的代码,存在没有UB模板的场景
  std::map<std::string, std::string> ub_tiling_file_name_to_content;
  if (ascgen_utils::HasCubeUBFusedScheduled(ub_schedule_result)) {
    GE_CHK_STATUS_RET(self->codegen->GenerateTiling(ub_schedule_result, symbol_source_info, pgo_dir, vector_core_num,
                                                    ub_tiling_file_name_to_content),
                      "generate ub tiling failed");
  } else {
    GELOGI("Has no cube ub fused schedule result, no need to generate ub tiling");
  }

  // 生成兜底模板的代码，必须有兜底模板
  GE_ASSERT_TRUE(ascgen_utils::HasCubeCommonFusedScheduled(common_schedule_result));
  std::map<std::string, std::string> common_tiling_file_name_to_content;
  GE_CHK_STATUS_RET(self->codegen->GenerateTiling(common_schedule_result, symbol_source_info, pgo_dir, vector_core_num,
                                                  common_tiling_file_name_to_content),
                    "generate common tiling failed");

  // 创建UB模板、兜底模板的内层字典
  PyObject *ub_dict = PyDict_New();
  if (ub_dict == nullptr) {
    return af::FAILED;
  }
  GE_DISMISSABLE_GUARD(ub_dict_guard, [ub_dict]() { Py_DECREF(ub_dict); });

  PyObject *common_dict = PyDict_New();
  if (common_dict == nullptr) {
    return af::FAILED;
  }
  GE_DISMISSABLE_GUARD(common_dict_guard, [common_dict]() { Py_DECREF(common_dict); });

  // 将UB模板的结果添加到内层字典中
  for (const auto &[key, value] : ub_tiling_file_name_to_content) {
    PyObjectGuard tmp_guard(PyUnicode_FromString(value.c_str()));
    PyDict_SetItemString(ub_dict, key.c_str(), tmp_guard.get());
  }

  // 将兜底模板的结果添加到内层字典中
  for (const auto &[key, value] : common_tiling_file_name_to_content) {
    PyObjectGuard tmp_guard(PyUnicode_FromString(value.c_str()));
    PyDict_SetItemString(common_dict, key.c_str(), tmp_guard.get());
  }

  // 将内层字典添加到外层字典中
  PyDict_SetItemString(py_tilings, "ub", ub_dict);
  PyDict_SetItemString(py_tilings, "common", common_dict);

  // 释放内层字典的所有权，因为外层字典已经持有引用
  GE_DISMISS_GUARD(ub_dict_guard);
  GE_DISMISS_GUARD(common_dict_guard);

  return af::SUCCESS;
}

af::Status CodeGen::HandleHostCodeGenForNonCVFusion(CodeGen::Object *self,
                                                    const ascir::FusedScheduledResult &fused_schedule_result,
                                                    const std::map<std::string, std::string> &symbol_source_info,
                                                    const char *pgo_dir, const char *vector_core_num,
                                                    PyObject *py_tilings) {
  // 非CV融合场景，生成一种结果
  std::map<std::string, std::string> tiling_file_name_to_content;
  GE_CHK_STATUS_RET(self->codegen->GenerateTiling(fused_schedule_result, symbol_source_info, pgo_dir, vector_core_num,
                                                  tiling_file_name_to_content),
                    "generate tiling failed");

  PyObject *default_dict = PyDict_New();
  if (default_dict == nullptr) {
    return af::FAILED;
  }
  GE_DISMISSABLE_GUARD(default_dict_guard, [default_dict]() { Py_DECREF(default_dict); });

  for (const auto &[key, value] : tiling_file_name_to_content) {
    PyObjectGuard tmp_guard(PyUnicode_FromString(value.c_str()));
    PyDict_SetItemString(default_dict, key.c_str(), tmp_guard.get());
  }

  PyDict_SetItemString(py_tilings, "default", default_dict);

  // 释放内层字典的所有权，因为外层字典已经持有引用
  GE_DISMISS_GUARD(default_dict_guard);

  return af::SUCCESS;
}

PyObject *CodeGen::device_code_generator(PyObject *self_pyobject, PyObject *args, PyObject *kwds) {
  (void)kwds;
  DumpGraphGuard guard;
  auto self = reinterpret_cast<CodeGen::Object *>(self_pyobject);

  PyObject *list_result_result = nullptr;
  if (PyArg_ParseTuple(args, "O", &list_result_result) == kPythonFail) {
    return PyErr_Format(PyExc_ValueError, "codegen param parse failed");
  }

  if (PyObject_IsInstance(list_result_result, reinterpret_cast<PyObject *>(&pyascir::FusedScheduledResult::type)) !=
      kPythonSuccess) {
    return PyErr_Format(PyExc_TypeError, "device_code_generator requires FusedScheduledResult");
  }
  auto fused_schedule_result = reinterpret_cast<pyascir::FusedScheduledResult::Object *>(list_result_result);

  // 创建字典用于存储不同模板的结果
  PyObject *tiling_dict = PyDict_New();
  if (tiling_dict == nullptr) {
    return PyErr_NoMemory();
  }
  GE_DISMISSABLE_GUARD(tiling_dict_guard, [tiling_dict]() { Py_DECREF(tiling_dict); });

  PyObject *kernel_dict = PyDict_New();
  if (kernel_dict == nullptr) {
    return PyErr_NoMemory();
  }
  GE_DISMISSABLE_GUARD(kernel_dict_guard, [kernel_dict]() { Py_DECREF(kernel_dict); });

  af::Status ret = af::FAILED;

  try {
    // 如果是cv融合，需要过滤ub、兜底融合结果分别生成代码
    if (ascgen_utils::IsCubeFusedScheduled(fused_schedule_result->fused_schedule_result)) {
      ret =
          HandleDeviceCodeGenForCVFusion(self, fused_schedule_result->fused_schedule_result, tiling_dict, kernel_dict);
    } else {
      ret = HandleDeviceCodeGenForNonCVFusion(self, fused_schedule_result->fused_schedule_result, tiling_dict,
                                              kernel_dict);
    }
    GE_CHK_BOOL_RET_SPECIAL_STATUS(ret != af::SUCCESS, PyErr_Format(PyExc_ValueError, "codegen generate kernel fail"),
                                   "codegen generate kernel fail");
  } catch (const std::runtime_error &e) {
    GELOGE(af::FAILED, "Caught a runtime_error: %s", e.what());
    return PyErr_Format(PyExc_ValueError, "codegen generate kernel fail: %s", e.what());
  }

  DumpGraphGuard::ReInit();

  // 释放字典的所有权
  GE_DISMISS_GUARD(tiling_dict_guard);
  GE_DISMISS_GUARD(kernel_dict_guard);
  return Py_BuildValue("OO", tiling_dict, kernel_dict);
}

PyObject *CodeGen::GenerateHostCodeResult(CodeGen::Object *self,
                                          pyascir::FusedScheduledResult::Object *fused_schedule_result,
                                          PyObject *shape_info_obj,
                                          const std::vector<std::vector<std::string>> &output_shape,
                                          const char *pgo_dir, const char *vector_core_num) {
  af::Status ret = af::FAILED;
  std::string infer_shape;

  // 创建嵌套字典用于存储不同模板的结果
  // 外层字典的key是模板类型："common"、"ub"或"default"
  // 内层字典保持原始的key和value
  PyObject *py_tilings = PyDict_New();
  if (py_tilings == nullptr) {
    return PyErr_NoMemory();
  }
  GE_DISMISSABLE_GUARD(py_tilings_guard, [py_tilings]() { Py_DECREF(py_tilings); });

  try {
    std::map<std::string, std::string> symbol_source_info;
    if (shape_info_obj != Py_None) {
      if (PyObject_IsInstance(shape_info_obj, reinterpret_cast<PyObject *>(&pyascir::ShapeInfo::type)) !=
          kPythonSuccess) {
        return PyErr_Format(PyExc_TypeError, "host_code_generator shape_info must be ShapeInfo type");
      }
      symbol_source_info = (reinterpret_cast<pyascir::ShapeInfo::Object *>(shape_info_obj))->shape_info;
    }

    // 如果是cv融合，需要过滤ub、兜底融合结果分别生成代码
    if (ascgen_utils::IsCubeFusedScheduled(fused_schedule_result->fused_schedule_result)) {
      ret = HandleHostCodeGenForCVFusion(self, fused_schedule_result->fused_schedule_result, symbol_source_info,
                                         pgo_dir, vector_core_num, py_tilings);
    } else {
      ret = HandleHostCodeGenForNonCVFusion(self, fused_schedule_result->fused_schedule_result, symbol_source_info,
                                            pgo_dir, vector_core_num, py_tilings);
    }
    GE_CHK_BOOL_RET_SPECIAL_STATUS(ret != af::SUCCESS, PyErr_Format(PyExc_ValueError, "codegen generate host fail"),
                                   "codegen generate host fail");

    infer_shape = self->codegen->GenerateInferShape(output_shape, symbol_source_info);
    // 释放字典的所有权
    GE_DISMISS_GUARD(py_tilings_guard);
    return Py_BuildValue("(NO)", py_tilings, PyUnicode_FromString(infer_shape.c_str()));
  } catch (const std::runtime_error &e) {
    GELOGE(af::FAILED, "Caught a runtime_error: %s", e.what());
    return PyErr_Format(PyExc_ValueError, "codegen generate host fail: %s", e.what());
  }
}

PyObject *CodeGen::host_code_generator(PyObject *self_pyobject, PyObject *args, PyObject *kwds) {
  (void)kwds;
  auto self = reinterpret_cast<CodeGen::Object *>(self_pyobject);

  PyObject *list_result_result = nullptr;
  PyObject *shape_info_obj = nullptr;
  PyObject *output_shape_obj = nullptr;
  const char *pgo_dir = nullptr;
  const char *vector_core_num = "";
  if (PyArg_ParseTuple(args, "OOOss", &list_result_result, &shape_info_obj, &output_shape_obj, &pgo_dir,
                       &vector_core_num) == kPythonFail) {
    return PyErr_Format(PyExc_ValueError, "codegen param parse failed");
  }

  if (shape_info_obj == Py_None) {
    GELOGW("host_code_generator shape info is none");
  } else if (PyObject_IsInstance(shape_info_obj, reinterpret_cast<PyObject *>(&pyascir::ShapeInfo::type)) ==
             kPythonFail) {
    return PyErr_Format(PyExc_ValueError, "host_code_generator shape info type invalid");
  }

  if (PyObject_IsInstance(list_result_result, reinterpret_cast<PyObject *>(&pyascir::FusedScheduledResult::type)) !=
      kPythonSuccess) {
    return PyErr_Format(PyExc_TypeError, "host_code_generator requires FusedScheduledResult");
  }
  auto fused_schedule_result = reinterpret_cast<pyascir::FusedScheduledResult::Object *>(list_result_result);

  std::vector<std::vector<std::string>> output_shape;
  if (!pyascir::OutputSymbolShapeDeserialize(output_shape_obj, output_shape)) {
    return PyErr_Format(PyExc_ValueError, "output_symbol_shape parse fail");
  }

  return GenerateHostCodeResult(self, fused_schedule_result, shape_info_obj, output_shape, pgo_dir, vector_core_num);
}

PyObject *CodeGen::get_kernel_and_json_generator(PyObject *self_pyobject, PyObject *args, const PyObject *kwds) {
  (void)kwds;
  auto self = reinterpret_cast<CodeGen::Object *>(self_pyobject);
  std::string get_kernel;
  const char *kernel_path = nullptr;
  const char *json_path = nullptr;
  if (PyArg_ParseTuple(args, "ss", &kernel_path, &json_path) == kPythonFail) {
    return PyErr_Format(PyExc_ValueError, "codegen param parse failed");
  }

  af::Status ret = af::SUCCESS;
  get_kernel = self->codegen->GenGetKernelAndJson(kernel_path, json_path);
  if (get_kernel == "") {
    ret = af::FAILED;
  }
  GE_CHK_BOOL_RET_SPECIAL_STATUS(ret != af::SUCCESS, PyErr_Format(PyExc_ValueError, "codegen generate get_kernel fail"),
                                 "codegen generate get_kernel fail");

  return Py_BuildValue("s", get_kernel.c_str());
}

PyObject *CodeGen::pgo_code_generator(PyObject *self_pyobject, PyObject *args, const PyObject *kwds) {
  (void)kwds;
  auto self = reinterpret_cast<CodeGen::Object *>(self_pyobject);

  PyObject *list_result_result = nullptr;
  const char *pgo_dir = nullptr;
  std::string pgo_src;
  if (PyArg_ParseTuple(args, "Os", &list_result_result, &pgo_dir) == kPythonFail) {
    return PyErr_Format(PyExc_ValueError, "codegen param parse failed");
  }

  if (PyObject_IsInstance(list_result_result, reinterpret_cast<PyObject *>(&pyascir::FusedScheduledResult::type)) !=
      kPythonSuccess) {
    return PyErr_Format(PyExc_TypeError, "pgo_code_generator requires FusedScheduledResult");
  }
  auto fused_schedule_result = reinterpret_cast<pyascir::FusedScheduledResult::Object *>(list_result_result);

  af::Status ret = af::SUCCESS;
  pgo_src = self->codegen->GeneratorPgo(fused_schedule_result->fused_schedule_result, pgo_dir);
  if (pgo_src == "") {
    ret = af::FAILED;
  }
  GE_CHK_BOOL_RET_SPECIAL_STATUS(ret != af::SUCCESS, PyErr_Format(PyExc_ValueError, "codegen generate pgo fail"),
                                 "codegen generate pgo fail");

  return Py_BuildValue("s", pgo_src.c_str());
}
}  // namespace pyascir

static PyModuleDef PyAutofuseModule = {
    PyModuleDef_HEAD_INIT,
    "autofuse",
    "Autofuse module",
    -1,
};

void pyautofuse_type_init() {
  using namespace pyascir;
  // AutofuserOptions::type
  AutofuserOptions::type.tp_name = "AutofuserOptions";
  AutofuserOptions::type.tp_basicsize = sizeof(AutofuserOptions::Object);
  AutofuserOptions::type.tp_itemsize = 0;
  AutofuserOptions::type.tp_dealloc = AutofuserOptions::Dealloc;
  AutofuserOptions::type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
  AutofuserOptions::type.tp_doc = "AutofuserOptions";
  AutofuserOptions::type.tp_members = nullptr;
  AutofuserOptions::type.tp_init = AutofuserOptions::Init;
  AutofuserOptions::type.tp_new = AutofuserOptions::New;
  // Autofuser::type
  Autofuser::type.tp_name = "Autofuser";
  Autofuser::type.tp_basicsize = sizeof(Autofuser::Object);
  Autofuser::type.tp_itemsize = 0;
  Autofuser::type.tp_dealloc = Autofuser::Dealloc;
  Autofuser::type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
  Autofuser::type.tp_doc = "Autofuser";
  Autofuser::type.tp_methods = Autofuser::methods;
  Autofuser::type.tp_members = nullptr;
  Autofuser::type.tp_init = Autofuser::Init;
  Autofuser::type.tp_new = Autofuser::New;
  // Schedule::type
  Schedule::type.tp_name = "Schedule";
  Schedule::type.tp_basicsize = sizeof(Schedule::Object);
  Schedule::type.tp_itemsize = 0;
  Schedule::type.tp_dealloc = Schedule::Dealloc;
  Schedule::type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
  Schedule::type.tp_doc = "Schedule";
  Schedule::type.tp_methods = Schedule::methods;
  Schedule::type.tp_members = nullptr;
  Schedule::type.tp_init = Schedule::Init;
  Schedule::type.tp_new = Schedule::New;
  // CodeGen::type
  CodeGen::type.tp_name = "CodeGen";
  CodeGen::type.tp_basicsize = sizeof(CodeGen::Object);
  CodeGen::type.tp_itemsize = 0;
  CodeGen::type.tp_dealloc = CodeGen::Dealloc;
  CodeGen::type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
  CodeGen::type.tp_doc = "CodeGen";
  CodeGen::type.tp_methods = CodeGen::methods;
  CodeGen::type.tp_members = nullptr;
  CodeGen::type.tp_init = CodeGen::Init;
  CodeGen::type.tp_new = CodeGen::New;
}

PyMODINIT_FUNC PyInit_pyautofuse(void) {
  pyautofuse_type_init();
  pyascir_type_init();
  pyascir_types_type_init();

  auto pyautofuse_module = PyModule_Create(&PyAutofuseModule);
  PY_ASSERT_NOTNULL(pyautofuse_module);

  auto pyascir_module = PyInit_ascir();
  if (pyascir_module == nullptr) {
    Py_DECREF(pyautofuse_module);
    return nullptr;
  }
  PyModule_AddObject(pyautofuse_module, "ascir", pyascir_module);

  if (PyType_Ready(&pyascir::AutofuserOptions::type) < 0) {
    Py_DECREF(pyautofuse_module);
    Py_DECREF(pyascir_module);
    return nullptr;
  };
  PyModule_AddObject(pyautofuse_module, "AutofuserOptions",
                     reinterpret_cast<PyObject *>(&pyascir::AutofuserOptions::type));

  if (PyType_Ready(&pyascir::Autofuser::type) < 0) {
    Py_DECREF(pyautofuse_module);
    Py_DECREF(pyascir_module);
    return nullptr;
  }
  PyModule_AddObject(pyautofuse_module, "Autofuser", reinterpret_cast<PyObject *>(&pyascir::Autofuser::type));

  if (PyType_Ready(&pyascir::Schedule::type) < 0) {
    Py_DECREF(pyautofuse_module);
    Py_DECREF(pyascir_module);
    return nullptr;
  }
  PyModule_AddObject(pyautofuse_module, "Schedule", reinterpret_cast<PyObject *>(&pyascir::Schedule::type));

  if (PyType_Ready(&pyascir::CodeGen::type) < 0) {
    Py_DECREF(pyautofuse_module);
    Py_DECREF(pyascir_module);
    return nullptr;
  }
  PyModule_AddObject(pyautofuse_module, "CodeGen", reinterpret_cast<PyObject *>(&pyascir::CodeGen::type));

  // Register ascir as a top-level module in sys.modules
  // This allows "from ascir import Max, Min, Mod" to work directly
  PyObject *modules = PyImport_GetModuleDict();
  Py_INCREF(pyascir_module);
  PyDict_SetItem(modules, PyUnicode_FromString("ascir"), pyascir_module);

  // Register pyautofuse as "autofuse.pyautofuse" in sys.modules
  Py_INCREF(pyautofuse_module);
  PyDict_SetItem(modules, PyUnicode_FromString("autofuse.pyautofuse"), pyautofuse_module);

  return pyautofuse_module;
}
// Entry point for "import autofuse.pyautofuse"
// This is required by tests using "from autofuse.pyautofuse import ..."
PyMODINIT_FUNC PyInit_autofuse_pyautofuse(void) {
  return PyInit_pyautofuse();
}

// Entry point for "import autofuse"
// This is required when autofuse.so symlink is used
PyMODINIT_FUNC PyInit_autofuse(void) {
  return PyInit_pyautofuse();
}
