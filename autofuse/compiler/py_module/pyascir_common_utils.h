/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __PYASCIR_COMMON_UTILS_H__
#define __PYASCIR_COMMON_UTILS_H__

#include <Python.h>
#include <string>
#include <vector>

namespace pyascir {
bool ShapeInfoDeserialize(const std::string to_be_deserialized, PyObject *py_obj);
bool OutputSymbolShapeDeserialize(PyObject *output_shape_obj, std::vector<std::vector<std::string>> &output_shape);
bool ComputeGraphDeserialize(const std::string to_be_deserialized, PyObject *py_obj);
bool PyListToVector(PyObject *list, std::vector<std::string> &vec);
PyObject *UtilsDeserialize(PyObject *self_pyobject, PyObject *args, PyObject *kwds);
PyObject *UtilsDurationRecord(PyObject *self_pyobject, PyObject *args, PyObject *kwds);
PyObject *UtilsReportDurations(PyObject *self_pyobject, PyObject *args, PyObject *kwds);
PyObject *UtilsSetPlatform(const PyObject *self_pyobject, PyObject *args, const PyObject *kwds);
PyObject *UtilsRequiresFullSocVersion(const PyObject *self_pyobject, PyObject *args, const PyObject *kwds);
}  // namespace pyascir

#endif
