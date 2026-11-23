/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/utils/type_utils.h"
#include "base/utils/type_utils_impl_af.h"
#include "graph/utils/type_utils_inner.h"
#include <algorithm>
#include "graph/buffer.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "graph/types_af.h"

namespace af {
namespace {
const std::map<domi::domiTensorFormat_t, Format> kDomiFormatToGeFormat = {
    {domi::DOMI_TENSOR_NCHW, FORMAT_NCHW},
    {domi::DOMI_TENSOR_NHWC, FORMAT_NHWC},
    {domi::DOMI_TENSOR_ND, FORMAT_ND},
    {domi::DOMI_TENSOR_NC1HWC0, FORMAT_NC1HWC0},
    {domi::DOMI_TENSOR_FRACTAL_Z, FORMAT_FRACTAL_Z},
    {domi::DOMI_TENSOR_NC1C0HWPAD, FORMAT_NC1C0HWPAD},
    {domi::DOMI_TENSOR_NHWC1C0, FORMAT_NHWC1C0},
    {domi::DOMI_TENSOR_FSR_NCHW, FORMAT_FSR_NCHW},
    {domi::DOMI_TENSOR_FRACTAL_DECONV, FORMAT_FRACTAL_DECONV},
    {domi::DOMI_TENSOR_BN_WEIGHT, FORMAT_BN_WEIGHT},
    {domi::DOMI_TENSOR_CHWN, FORMAT_CHWN},
    {domi::DOMI_TENSOR_FILTER_HWCK, FORMAT_FILTER_HWCK},
    {domi::DOMI_TENSOR_NDHWC, FORMAT_NDHWC},
    {domi::DOMI_TENSOR_NCDHW, FORMAT_NCDHW},
    {domi::DOMI_TENSOR_DHWCN, FORMAT_DHWCN},
    {domi::DOMI_TENSOR_DHWNC, FORMAT_DHWNC},
    {domi::DOMI_TENSOR_RESERVED, FORMAT_RESERVED}};

const std::set<std::string> kInternalFormat = {"NC1HWC0",
                                               "FRACTAL_Z",
                                               "NC1C0HWPAD",
                                               "NHWC1C0",
                                               "FRACTAL_DECONV",
                                               "C1HWNC0",
                                               "FRACTAL_DECONV_TRANSPOSE",
                                               "FRACTAL_DECONV_SP_STRIDE_TRANS",
                                               "NC1HWC0_C04",
                                               "FRACTAL_Z_C04",
                                               "FRACTAL_DECONV_SP_STRIDE8_TRANS",
                                               "NC1KHKWHWC0",
                                               "C1HWNCoC0",
                                               "FRACTAL_ZZ",
                                               "FRACTAL_NZ",
                                               "NDC1HWC0",
                                               "FRACTAL_Z_3D",
                                               "FRACTAL_Z_3D_TRANSPOSE",
                                               "FRACTAL_ZN_LSTM",
                                               "FRACTAL_Z_G",
                                               "ND_RNN_BIAS",
                                               "FRACTAL_ZN_RNN",
                                               "NYUV",
                                               "NYUV_A"};

const std::map<domi::FrameworkType, std::string> kFmkTypeToString = {
    {domi::CAFFE, "caffe"},           {domi::MINDSPORE, "mindspore"}, {domi::TENSORFLOW, "tensorflow"},
    {domi::ANDROID_NN, "android_nn"}, {domi::ONNX, "onnx"},           {domi::FRAMEWORK_RESERVED, "framework_reserved"},
};

const std::map<domi::ImplyType, std::string> kImplyTypeToString = {
    {domi::ImplyType::BUILDIN, "buildin"}, {domi::ImplyType::TVM, "tvm"}, {domi::ImplyType::CUSTOM, "custom"},
    {domi::ImplyType::AI_CPU, "ai_cpu"},   {domi::ImplyType::CCE, "cce"}, {domi::ImplyType::GELOCAL, "gelocal"},
    {domi::ImplyType::INVALID, "invalid"}};
}  // namespace

std::string TypeUtils::DataTypeToSerialString(const DataType data_type) {
  return TypeUtilsImpl::DataTypeToAscendString(data_type).GetString();
}

DataType TypeUtils::SerialStringToDataType(const std::string &str) {
  return TypeUtilsImpl::AscendStringToDataType(str.c_str());
}

std::string TypeUtils::FormatToSerialString(const Format format) {
  return TypeUtilsImpl::FormatToAscendString(format).GetString();
}

Format TypeUtils::SerialStringToFormat(const std::string &str) {
  return TypeUtilsImpl::AscendStringToFormat(str.c_str());
}

Format TypeUtils::DataFormatToFormat(const std::string &str) {
  return TypeUtilsImpl::DataFormatToFormat(str.c_str());
}

bool TypeUtils::GetDataTypeLength(const af::DataType data_type, uint32_t &length) {
  return TypeUtilsImpl::GetDataTypeLength(data_type, length);
}

std::string TypeUtilsInner::ImplyTypeToSerialString(const domi::ImplyType imply_type) {
  const auto it = kImplyTypeToString.find(imply_type);
  if (it != kImplyTypeToString.end()) {
    return it->second;
  } else {
    REPORT_INNER_ERR_MSG("E18888",
                         "ImplyTypeToSerialString: imply_type %u is not supported, valid types are: "
                         "BUILDIN, TVM, CUSTOM, AI_CPU, CCE, GELOCAL, INVALID",
                         static_cast<uint32_t>(imply_type));
    GELOGE(GRAPH_FAILED,
           "[Check][Param] ImplyTypeToSerialString: imply_type %u is not supported, valid types are: "
           "BUILDIN, TVM, CUSTOM, AI_CPU, CCE, GELOCAL, INVALID",
           static_cast<uint32_t>(imply_type));
    return "UNDEFINED";
  }
}

bool TypeUtilsInner::IsInternalFormat(const Format format) {
  const std::string serial_format = TypeUtils::FormatToSerialString(static_cast<Format>(GetPrimaryFormat(format)));
  const auto iter = kInternalFormat.find(serial_format);
  const bool result = (iter == kInternalFormat.cend()) ? false : true;
  return result;
}

Format TypeUtilsInner::DomiFormatToFormat(const domi::domiTensorFormat_t domi_format) {
  const auto it = kDomiFormatToGeFormat.find(domi_format);
  if (it != kDomiFormatToGeFormat.end()) {
    return it->second;
  }
  GELOGW("[Check][Param] do not find domi Format %d from map", domi_format);
  return FORMAT_RESERVED;
}

std::string TypeUtilsInner::FmkTypeToSerialString(const domi::FrameworkType fmk_type) {
  const auto it = kFmkTypeToString.find(fmk_type);
  if (it != kFmkTypeToString.end()) {
    return it->second;
  } else {
    GELOGW(
        "[Util][Serialize] Framework type %d is not supported, valid types are: "
        "CAFFE, MINDSPORE, TENSORFLOW, ANDROID_NN, ONNX, FRAMEWORK_RESERVED.",
        fmk_type);
    return "";
  }
}
}  // namespace af
