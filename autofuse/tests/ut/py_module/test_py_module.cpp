/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"

#include <string>

#include "ascendc_ir.h"
#include "graph/ascendc_ir/utils/asc_graph_utils.h"
#include "ascir_ops.h"

#include "pyascir.h"
#include "pyascir_types.h"
#include "pyascir_common_utils.h"
#include "common/platform_context.h"

class TestPyModule : public testing::Test {
 protected:
  static PyObject *pyautofuse_module_;

  static void SetUpTestSuite() {
    Py_Initialize();
    pyautofuse_module_ = PyInit_pyautofuse();
  }

  static void TearDownTestSuite() {
    Py_XDECREF(pyautofuse_module_);
    pyautofuse_module_ = nullptr;
    Py_Finalize();
  }
};

PyObject *TestPyModule::pyautofuse_module_ = nullptr;

TEST_F(TestPyModule, UtilsSetPlatformKeepsLegacyThreeArguments) {
  PyObject *args = Py_BuildValue("(sLL)", "5102", 48, 245760);
  ASSERT_NE(args, nullptr);

  PyObject *result = pyascir::UtilsSetPlatform(nullptr, args, nullptr);
  EXPECT_NE(result, nullptr);
  Py_DECREF(result);
  Py_DECREF(args);

  std::string npu_arch;
  std::string soc_version;
  EXPECT_EQ(ge::PlatformContext::GetInstance().GetPlatformTarget(npu_arch, soc_version), af::SUCCESS);
  EXPECT_EQ(npu_arch, "5102");
  EXPECT_EQ(soc_version, "");
  ge::PlatformContext::GetInstance().Reset();
}

TEST_F(TestPyModule, UtilsSetPlatformAcceptsFullSocVersion) {
  PyObject *args = Py_BuildValue("(sLLs)", "5102", 48, 245760, "MC62CM12AA");
  ASSERT_NE(args, nullptr);

  PyObject *result = pyascir::UtilsSetPlatform(nullptr, args, nullptr);
  EXPECT_NE(result, nullptr);
  Py_DECREF(result);
  Py_DECREF(args);

  std::string npu_arch;
  std::string soc_version;
  EXPECT_EQ(ge::PlatformContext::GetInstance().GetPlatformTarget(npu_arch, soc_version), af::SUCCESS);
  EXPECT_EQ(npu_arch, "5102");
  EXPECT_EQ(soc_version, "MC62CM12AA");
  ge::PlatformContext::GetInstance().Reset();
}

TEST_F(TestPyModule, UtilsSetPlatformRejectsFullSocVersionWithoutNpuArch) {
  PyObject *valid_args = Py_BuildValue("(sLLs)", "5102", 48, 245760, "MC62CM12AA");
  ASSERT_NE(valid_args, nullptr);
  PyObject *result = pyascir::UtilsSetPlatform(nullptr, valid_args, nullptr);
  EXPECT_NE(result, nullptr);
  Py_DECREF(result);
  Py_DECREF(valid_args);

  PyObject *invalid_args = Py_BuildValue("(sLLs)", "", 48, 245760, "MC62CM12AA");
  ASSERT_NE(invalid_args, nullptr);
  result = pyascir::UtilsSetPlatform(nullptr, invalid_args, nullptr);
  EXPECT_EQ(result, nullptr);
  PyErr_Clear();
  Py_DECREF(invalid_args);

  std::string npu_arch;
  std::string soc_version;
  EXPECT_EQ(ge::PlatformContext::GetInstance().GetPlatformTarget(npu_arch, soc_version), af::SUCCESS);
  EXPECT_EQ(npu_arch, "5102");
  EXPECT_EQ(soc_version, "MC62CM12AA");
  ge::PlatformContext::GetInstance().Reset();
}

TEST_F(TestPyModule, AutofuseBackendRejectsUnknownArchBeforeGraphValidation) {
  ge::PlatformContext::GetInstance().SetPlatform("UnknownArch");
  ASSERT_NE(pyautofuse_module_, nullptr);
  PyObject *options_type = PyObject_GetAttrString(pyautofuse_module_, "AutofuserOptions");
  ASSERT_NE(options_type, nullptr);
  PyObject *options = PyObject_CallObject(options_type, nullptr);
  ASSERT_NE(options, nullptr);
  PyObject *autofuser_type = PyObject_GetAttrString(pyautofuse_module_, "Autofuser");
  ASSERT_NE(autofuser_type, nullptr);
  PyObject *autofuser = PyObject_CallFunctionObjArgs(autofuser_type, options, NULL);
  ASSERT_NE(autofuser, nullptr);
  PyObject *method = PyObject_GetAttrString(autofuser, "autofuse_backend");
  ASSERT_NE(method, nullptr);
  PyObject *args = Py_BuildValue("(O)", Py_None);
  ASSERT_NE(args, nullptr);

  PyObject *result = PyObject_CallObject(method, args);
  EXPECT_EQ(result, nullptr);
  ASSERT_TRUE(PyErr_Occurred());

  PyObject *exc_type = nullptr;
  PyObject *exc_value = nullptr;
  PyObject *exc_traceback = nullptr;
  PyErr_Fetch(&exc_type, &exc_value, &exc_traceback);
  ASSERT_NE(exc_value, nullptr);
  PyObject *message_obj = PyObject_Str(exc_value);
  ASSERT_NE(message_obj, nullptr);
  const std::string message = PyUnicode_AsUTF8(message_obj);
  EXPECT_NE(message.find("Invalid SoC performance target"), std::string::npos) << message;

  Py_XDECREF(message_obj);
  Py_XDECREF(exc_traceback);
  Py_XDECREF(exc_value);
  Py_XDECREF(exc_type);
  Py_XDECREF(result);
  Py_XDECREF(args);
  Py_XDECREF(method);
  Py_XDECREF(autofuser);
  Py_XDECREF(autofuser_type);
  Py_XDECREF(options);
  Py_XDECREF(options_type);
  ge::PlatformContext::GetInstance().Reset();
}

TEST_F(TestPyModule, DISABLED_UtilsDeserialize_symbol_source_info) {
  PyObject *args = PyTuple_New(2);
  PyObject *kwds = PyDict_New();

  {
    PyObject *type = PyUnicode_FromString("symbol_source_info");
    PyObject *obj = PyUnicode_FromString("{\"s0\":\"GetDimValueFromGraphInputData(0, 0);\"}");

    PyTuple_SetItem(args, 0, type);  // 添加到索引 0
    PyTuple_SetItem(args, 1, obj);   // 添加到索引 1

    auto res = pyascir::UtilsDeserialize(nullptr, args, kwds);
    EXPECT_NE(res, nullptr);

    auto shape_info = (pyascir::ShapeInfo::Object *)res;
    EXPECT_EQ(shape_info->shape_info["s0"], "GetDimValueFromGraphInputData(0, 0);");
    Py_DECREF(res);
  }

  {
    PyObject *type = PyUnicode_FromString("symbol_source_info");
    PyObject *obj = PyUnicode_FromString("error_test");

    PyTuple_SetItem(args, 0, type);  // 添加到索引 0
    PyTuple_SetItem(args, 1, obj);   // 添加到索引 1

    auto res = pyascir::UtilsDeserialize(nullptr, args, kwds);
    EXPECT_EQ(res, nullptr);
  }

  {
    PyObject *type = PyUnicode_FromString("symbol_source_info");
    PyObject *obj = PyUnicode_FromString("{\"s0\": 1}");

    PyTuple_SetItem(args, 0, type);  // 添加到索引 0
    PyTuple_SetItem(args, 1, obj);   // 添加到索引 1

    auto res = pyascir::UtilsDeserialize(nullptr, args, kwds);
    EXPECT_EQ(res, nullptr);
  }

  Py_DECREF(args);
  Py_DECREF(kwds);
}

TEST_F(TestPyModule, DISABLED_Deserialize_output_symbol_shape) {
  PyObject *output_list = PyList_New(0);

  {
    PyObject *inner_list1 = PyList_New(2);
    PyList_SetItem(inner_list1, 0, PyUnicode_FromString("s0"));
    PyList_SetItem(inner_list1, 1, PyUnicode_FromString("s1"));
    PyObject *inner_list2 = PyList_New(2);
    PyList_SetItem(inner_list2, 0, PyUnicode_FromString("s2"));
    PyList_SetItem(inner_list2, 1, PyUnicode_FromString("s3"));

    PyList_Append(output_list, inner_list1);
    PyList_Append(output_list, inner_list2);

    std::vector<std::vector<std::string>> output_shape;
    auto ret = pyascir::OutputSymbolShapeDeserialize(output_list, output_shape);
    EXPECT_TRUE(ret);
    Py_DECREF(inner_list1);
    Py_DECREF(inner_list2);

    ASSERT_EQ(output_shape.size(), 2U);
    std::vector<std::string> inner_1 = output_shape.at(0);
    std::vector<std::string> inner_2 = output_shape.at(1);
    ASSERT_EQ(inner_1.size(), 2U);
    ASSERT_EQ(inner_2.size(), 2U);
    EXPECT_EQ(inner_1.at(0), "s0");
    EXPECT_EQ(inner_1.at(1), "s1");
    EXPECT_EQ(inner_2.at(0), "s2");
    EXPECT_EQ(inner_2.at(1), "s3");
  }

  PyList_SetSlice(output_list, 0, PyList_Size(output_list), NULL);
  {
    PyList_Append(output_list, PyUnicode_FromString("s0"));
    std::vector<std::vector<std::string>> output_shape;
    auto ret = pyascir::OutputSymbolShapeDeserialize(output_list, output_shape);
    EXPECT_FALSE(ret);
  }

  PyList_SetSlice(output_list, 0, PyList_Size(output_list), NULL);
  {
    PyObject *inner_list1 = PyList_New(2);
    PyList_SetItem(inner_list1, 0, PyUnicode_FromString("s0"));
    PyList_SetItem(inner_list1, 1, PyLong_FromLong(42));

    PyList_Append(output_list, inner_list1);
    std::vector<std::vector<std::string>> output_shape;
    auto ret = pyascir::OutputSymbolShapeDeserialize(output_list, output_shape);
    EXPECT_FALSE(ret);
    Py_DECREF(inner_list1);
  }

  Py_DECREF(output_list);
}

TEST_F(TestPyModule, DISABLED_UtilsDeserialize_asc_graph) {
  PyObject *args = PyTuple_New(2);
  PyObject *kwds = PyDict_New();

  af::AscGraph g("fused_graph");
  af::ascir_op::Data x("x", g);
  auto node = g.FindNode("x");
  node->inputs();
  node->outputs();
  g.SetTilingKey(0x5a5a);
  std::string to_be_deser;
  EXPECT_EQ(af::AscGraphUtils::SerializeToReadable(g, to_be_deser), ge::GRAPH_SUCCESS);

  {
    PyObject *type = PyUnicode_FromString("asc_graph");
    PyObject *obj = PyUnicode_FromString(to_be_deser.c_str());

    PyTuple_SetItem(args, 0, type);  // 添加到索引 0
    PyTuple_SetItem(args, 1, obj);   // 添加到索引 1

    auto res = pyascir::UtilsDeserialize(nullptr, args, kwds);
    EXPECT_NE(res, nullptr);

    auto graph = (pyascir::HintGraph::Object *)res;
    EXPECT_EQ(graph->graph->GetName(), "fused_graph");
    Py_DECREF(res);
  }

  {
    PyObject *type = PyUnicode_FromString("asc_graph");
    PyObject *obj = PyUnicode_FromString("error");

    PyTuple_SetItem(args, 0, type);  // 添加到索引 0
    PyTuple_SetItem(args, 1, obj);   // 添加到索引 1

    auto res = pyascir::UtilsDeserialize(nullptr, args, kwds);
    EXPECT_EQ(res, nullptr);
  }

  Py_DECREF(args);
  Py_DECREF(kwds);
}

TEST_F(TestPyModule, DISABLED_UtilsDeserialize_compute_graph) {
  PyObject *args = PyTuple_New(2);
  PyObject *kwds = PyDict_New();

  {
    PyObject *type = PyUnicode_FromString("compute_graph");
    PyObject *obj = PyUnicode_FromString("error");

    PyTuple_SetItem(args, 0, type);  // 添加到索引 0
    PyTuple_SetItem(args, 1, obj);   // 添加到索引 1

    auto res = pyascir::UtilsDeserialize(nullptr, args, kwds);
    EXPECT_EQ(res, nullptr);
  }

  Py_DECREF(args);
  Py_DECREF(kwds);
}

// TEST_F(TestPyModule, UtilsDeserialize_error_type) {
//   PyObject *args = PyTuple_New(2);
//   PyObject *kwds = PyDict_New();

//   {
//     PyObject *type = PyUnicode_FromString("error_type");
//     PyObject *obj = PyUnicode_FromString("error");

//     PyTuple_SetItem(args, 0, type); // 添加到索引 0
//     PyTuple_SetItem(args, 1, obj); // 添加到索引 1

//     auto res = pyascir::UtilsDeserialize(nullptr, args, kwds);
//     EXPECT_EQ(res, nullptr);
//   }

//   Py_DECREF(args);
//   Py_DECREF(kwds);
// }

// TEST_F(TestPyModule, UtilsDurationRecord) {
//   PyObject *args = PyTuple_New(3);
//   PyObject *kwds = PyDict_New();

//   // target list type error
//   {
//     PyObject *target_list = PyList_New(2);
//     PyList_SetItem(target_list, 0, PyLong_FromLong(111));
//     PyList_SetItem(target_list, 1, PyLong_FromLong(222));

//     PyTuple_SetItem(args, 0, target_list); // 添加到索引 0
//     PyTuple_SetItem(args, 1, PyLong_FromLong(1234)); // 添加到索引 1
//     PyTuple_SetItem(args, 2, PyLong_FromLong(1234)); // 添加到索引 2

//     auto res = pyascir::UtilsDurationRecord(nullptr, args, kwds);
//     EXPECT_EQ(res, nullptr);
//     pyascir::UtilsReportDurations(nullptr, args, kwds);
//     Py_DECREF(target_list);
//   }

//   // target type error
//   {
//     PyTuple_SetItem(args, 0, PyLong_FromLong(1234)); // 添加到索引 0
//     PyTuple_SetItem(args, 1, PyLong_FromLong(1234)); // 添加到索引 1
//     PyTuple_SetItem(args, 2, PyLong_FromLong(1234)); // 添加到索引 2

//     auto res = pyascir::UtilsDurationRecord(nullptr, args, kwds);
//     EXPECT_EQ(res, nullptr);
//   }

//   // args error
//   {
//     PyTuple_SetItem(args, 0, PyLong_FromLong(1234)); // 添加到索引 0
//     auto res = pyascir::UtilsDurationRecord(nullptr, args, kwds);
//     EXPECT_EQ(res, nullptr);
//   }

//   // time error
//   {
//     PyObject *target_list = PyList_New(2);
//     PyList_SetItem(target_list, 0, PyLong_FromLong(111));
//     PyList_SetItem(target_list, 1, PyLong_FromLong(222));

//     PyTuple_SetItem(args, 0, target_list); // 添加到索引 0
//     PyTuple_SetItem(args, 1, PyLong_FromLong(-1)); // 添加到索引 1
//     PyTuple_SetItem(args, 2, PyLong_FromLong(1234)); // 添加到索引 2

//     auto res = pyascir::UtilsDurationRecord(nullptr, args, kwds);
//     EXPECT_EQ(res, nullptr);
//     Py_DECREF(target_list);
//   }

//   {
//     PyObject *target_list = PyList_New(2);
//     PyList_SetItem(target_list, 0, PyUnicode_FromString("codegen"));
//     PyList_SetItem(target_list, 1, PyUnicode_FromString("ascgraph_0"));

//     PyTuple_SetItem(args, 0, target_list); // 添加到索引 0
//     PyTuple_SetItem(args, 1, PyLong_FromLong(1234)); // 添加到索引 1
//     PyTuple_SetItem(args, 2, PyLong_FromLong(1234)); // 添加到索引 2

//     auto res = pyascir::UtilsDurationRecord(nullptr, args, kwds);
//     EXPECT_EQ(res, Py_None);
//     Py_DECREF(target_list);
//   }

//   Py_DECREF(args);
//   Py_DECREF(kwds);
// }
