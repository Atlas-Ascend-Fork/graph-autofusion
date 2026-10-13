/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include "nlohmann/json.hpp"
#include <cstdlib>

#include "acl/acl_prof.h"
#include "graph/ge_error_codes_af.h"
#include "experiment/msprof/toolchain/prof_api.h"
#include "fused_graph_test.h"

static const int32_t SUCCESS = 0;
static const int32_t FAILED = -1;

using json = nlohmann::json;
using TilingFunc = ge::graphStatus (*)(void *, uint64_t *, uint32_t *, void *);
using LaunchFunc = uint32_t (*)(uint32_t, void *, void **, int32_t, void **, int32_t, void *, void *);
using GetTilingSizeFunc = size_t (*)();

std::map<std::string, ge::DataType> kDataTypeMap{
    {"float32", ge::DT_FLOAT}, {"float16", ge::DT_FLOAT16}, {"int8", ge::DT_INT8},     {"int16", ge::DT_INT16},
    {"int32", ge::DT_INT32},   {"int64", ge::DT_INT64},     {"uint8", ge::DT_UINT8},   {"uint16", ge::DT_UINT16},
    {"uint32", ge::DT_UINT32}, {"uint64", ge::DT_UINT64},   {"double", ge::DT_DOUBLE}, {"bool", ge::DT_BOOL},
    {"string", ge::DT_STRING}};

int64_t GetShapeSize(const std::vector<int64_t> &shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

int EnvInit(int32_t deviceId, aclrtStream *stream) {
  // 固定写法 AscendCL初始化
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return 0;
}

int32_t CreateDeviceData(const std::vector<int8_t> &host_data, const std::vector<int64_t> &shape, ge::DataType dtype,
                         void **device_addr) {
  auto size = GetShapeSize(shape) * ge::GetSizeByDataType(dtype);
  // aclrtMalloc device
  auto ret = aclrtMalloc(device_addr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // aclrtMemcpy host to device
  ret = aclrtMemcpy(*device_addr, size, host_data.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  return af::SUCCESS;
}

int32_t CreateDeviceDataV2(const void *host_addr, const std::vector<int64_t> &shape, ge::DataType dtype,
                           void **device_addr) {
  auto size = GetShapeSize(shape) * ge::GetSizeByDataType(dtype);
  // aclrtMalloc device
  auto ret = aclrtMalloc(device_addr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // aclrtMemcpy host to device
  ret = aclrtMemcpy(*device_addr, size, host_addr, size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  return af::SUCCESS;
}

void Finalize(int32_t deviceId, aclrtStream stream) {
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
}

std::unique_ptr<int8_t[]> GenFromBin(const std::vector<int64_t> &shape, const std::string &file_path,
                                     ge::DataType data_type) {
  int64_t data_len = GetShapeSize(shape) * ge::GetSizeByDataType(data_type);
  std::ifstream bin_file(file_path.c_str(), std::ifstream::binary);
  if (!bin_file.is_open()) {
    LOG_PRINT("ERROR: open file %s failed.\n", file_path.c_str());
    return nullptr;
  }
  bin_file.seekg(0, std::ifstream::end);
  uint32_t file_size = bin_file.tellg();
  bin_file.seekg(0, std::ifstream::beg);
  if (data_len != file_size) {
    bin_file.close();
    LOG_PRINT("ERROR: data_len != file_size. \n");
    return nullptr;
  }
  std::unique_ptr<int8_t[]> data_holder = MakeUnique<int8_t[]>(data_len);
  bin_file.read(reinterpret_cast<char *>(data_holder.get()), data_len);
  bin_file.close();
  return data_holder;
}

int32_t SaveBinToFile(const std::string &file_path, void *data, size_t data_len) {
  std::ofstream file(file_path, std::ios::out | std::ios::binary);
  if (!file.is_open()) {
    LOG_PRINT("ERROR: open file %s failed.\n", file_path.c_str());
    return FAILED;
  }
  file.write(reinterpret_cast<char *>(data), data_len);
  file.close();
  return af::SUCCESS;
}

int32_t ReadConfigJson(const std::string &json_path, json &config_json) {
  std::ifstream config_json_file(json_path);
  if (!config_json_file.is_open()) {
    LOG_PRINT("ERROR: open file %s failed.\n", json_path.c_str());
    return FAILED;
  }
  try {
    config_json_file >> config_json;
  } catch (const nlohmann::json::exception &e) {
    LOG_PRINT("ERROR: fail to read file %s, error msg: %s.\n", json_path.c_str(), e.what());
    return FAILED;
  }
  return af::SUCCESS;
}

int32_t ParseMapConfig(const json &config_json, const std::string &key, std::map<std::string, std::string> &configs) {
  if (config_json.contains(key)) {
    try {
      std::map<std::string, std::string> buff = config_json[key];
      for (auto &it : buff) {
        configs[it.first.c_str()] = it.second.c_str();
      }
    } catch (const nlohmann::json::exception &e) {
      LOG_PRINT("ERROR: fail to parse %s from json %s, err msg: %s\n", key.c_str(), config_json[key].dump().c_str(),
                e.what());
      return FAILED;
    }
  } else {
    LOG_PRINT("ERROR: fail to parse %s from json %s\n", key.c_str(), config_json[key].dump().c_str());
    return FAILED;
  }
  return af::SUCCESS;
}

std::vector<std::vector<int64_t>> ParseDataShape(const std::string &data_shape) {
  std::vector<std::vector<int64_t>> result;
  std::string temp;
  std::istringstream ss(data_shape);

  while (getline(ss, temp, ';')) {
    std::vector<int64_t> inner_result;
    std::istringstream inner_ss(temp);

    while (getline(inner_ss, temp, ',')) {
      inner_result.emplace_back(std::stol(temp));
    }
    result.emplace_back(inner_result);
  }
  return result;
}

std::vector<ge::DataType> ParseDataType(std::string &data_type) {
  std::istringstream iss(data_type);
  std::string token;
  std::vector<ge::DataType> data_types;
  while (getline(iss, token, ',')) {
    token.erase(0, token.find_first_not_of(" "));
    token.erase(token.find_last_not_of(" ") + 1);
    if (kDataTypeMap.find(token) != kDataTypeMap.end()) {
      data_types.emplace_back(kDataTypeMap[token]);
    }
  }
  return data_types;
}

int32_t AutofuseKernelInfo::Init(void *stream, const std::string &config_path) {
  stream_ = stream;
  auto ret = InitKernelConfig(config_path);
  if (ret != af::SUCCESS) {
    LOG_PRINT("ERROR: InitKernelConfig fail\n.");
    return FAILED;
  }
  (void)InitDeviceData();
  return af::SUCCESS;
}

int32_t AutofuseKernelInfo::InitKernelConfig(const std::string &config_file_path) {
  json config_json;
  if (ReadConfigJson(config_file_path, config_json) != af::SUCCESS) {
    LOG_PRINT("ERROR: Fail to read config json: %s\n.", config_file_path.c_str());
    return FAILED;
  }
  const std::string kernel_config_key = "kernel_config";
  std::map<std::string, std::string> json_kernel_config;
  if (ParseMapConfig(config_json, kernel_config_key, json_kernel_config) != af::SUCCESS) {
    LOG_PRINT("ERROR: Fail to parse config: %s\n.", kernel_config_key.c_str());
    return FAILED;
  }
  for (auto &config : json_kernel_config) {
    LOG_PRINT("kernel_config: %s : %s\n.", config.first.c_str(), config.second.c_str());
  }
  if (SetKernelConfig(json_kernel_config) != af::SUCCESS) {
    LOG_PRINT("ERROR: Fail to set kernel config: %s\n.", kernel_config_key.c_str());
    return FAILED;
  }
  return af::SUCCESS;
}

int32_t AutofuseKernelInfo::SetKernelConfig(std::map<std::string, std::string> &src_kernel_config) {
  graph_name_snake_ = src_kernel_config["graph_name"];
  kernel_config_.input_num = std::stoi(src_kernel_config["input_num"]);
  kernel_config_.output_num = std::stoi(src_kernel_config["output_num"]);
  kernel_config_.input_shape = ParseDataShape(src_kernel_config["input_shapes"]);
  kernel_config_.output_shape = ParseDataShape(src_kernel_config["output_shapes"]);
  kernel_config_.input_data_type = ParseDataType(src_kernel_config["input_data_types"]);
  kernel_config_.output_data_type = ParseDataType(src_kernel_config["output_data_types"]);

  if (kernel_config_.input_num != kernel_config_.input_shape.size()) {
    LOG_PRINT("ERROR: input_num[%d] != input_shape size[%d]\n.", kernel_config_.input_num,
              kernel_config_.input_shape.size());
    return FAILED;
  }
  return af::SUCCESS;
}

int32_t AutofuseKernelInfo::InitDeviceData() {
  input_addr_ = MakeUnique<void *[]>(kernel_config_.input_num);
  output_addr_ = MakeUnique<void *[]>(kernel_config_.output_num);

  std::vector<std::unique_ptr<int8_t[]>> input_host_data;
  for (int i = 0; i < kernel_config_.input_num; ++i) {
    std::string file_path = "input/Data_" + std::to_string(i) + ".bin";
    auto tmp_input = GenFromBin(kernel_config_.input_shape[i], file_path, kernel_config_.input_data_type[i]);
    input_host_data.emplace_back(std::move(tmp_input));
  }

  std::vector<std::vector<int8_t>> output_host_data;
  for (int i = 0; i < kernel_config_.output_num; ++i) {
    int64_t data_size =
        GetShapeSize(kernel_config_.output_shape[i]) * ge::GetSizeByDataType(kernel_config_.output_data_type[i]);
    std::vector<int8_t> tmp_host_data;
    for (int64_t j = 0; j < data_size; ++j) {
      tmp_host_data.emplace_back((int8_t)1);
    }
    output_host_data.emplace_back(tmp_host_data);
  }

  for (int i = 0; i < kernel_config_.input_num; ++i) {
    auto ret = CreateDeviceDataV2((void *)input_host_data[i].get(), kernel_config_.input_shape[i],
                                  kernel_config_.input_data_type[i], &input_addr_[i]);
    CHECK_RET(ret == SUCCESS, LOG_PRINT("Create input device data failed. input: %d, ERROR: %d\n", i, ret);
              return FAILED);
  }

  for (int i = 0; i < kernel_config_.output_num; ++i) {
    auto ret = CreateDeviceData(output_host_data[i], kernel_config_.output_shape[i], kernel_config_.output_data_type[i],
                                &output_addr_[i]);
    CHECK_RET(ret == SUCCESS, LOG_PRINT("Create output device data failed. output: %d, ERROR: %d\n", i, ret);
              return FAILED);
  }
  return af::SUCCESS;
}

int32_t AutofuseKernelInfo::LoadSoHandles() {
  std::string so_path = graph_name_snake_ + ".so";
  LOG_PRINT("so path: %s\n", so_path.c_str());
  char real_path[MMPA_MAX_PATH] = {};
  auto ret = mmRealPath(so_path.c_str(), &real_path[0], MMPA_MAX_PATH);
  if (ret != EN_OK) {
    LOG_PRINT("mmRealPath failed, ERROR: %d, so path: %s\n", ret, so_path.c_str());
    return FAILED;
  }

  handles_ = mmDlopen(real_path, static_cast<int32_t>(MMPA_RTLD_NOW));
  if (handles_ == nullptr) {
    LOG_PRINT("so %s open failed\n", so_path.c_str());
    return FAILED;
  }
  LOG_PRINT("so %s open success, handles_:%p\n", so_path.c_str(), handles_);
  return af::SUCCESS;
}

int32_t AutofuseKernelInfo::ParseTaskRunParam() {
  // 获取tiling size
  std::string get_tiling_size_func_name = "GetTilingDataSize";
  const auto get_tiling_size_func =
      reinterpret_cast<GetTilingSizeFunc>(mmDlsym(handles_, get_tiling_size_func_name.c_str()));
  if (get_tiling_size_func == nullptr) {
    LOG_PRINT("ERROR: get_tiling_size_func is nullptr, %s\n", get_tiling_size_func_name.c_str());
    return FAILED;
  }
  tiling_size_ = get_tiling_size_func();
  LOG_PRINT("name: %s, tiling_size_: %zu, func_name: %s\n", graph_name_snake_.c_str(), tiling_size_,
            get_tiling_size_func_name.c_str());
  return af::SUCCESS;
}

int32_t AutofuseKernelInfo::DoTiling(std::unique_ptr<uint8_t[]> &tiling_data_holder, uint64_t &workspace_size) {
  std::string tiling_func_name = "AutofuseTiling";
  const auto tiling_func = reinterpret_cast<TilingFunc>(mmDlsym(handles_, tiling_func_name.c_str()));
  if (tiling_func == nullptr) {
    LOG_PRINT("ERROR: tiling_func is nullptr, %s\n", tiling_func_name.c_str());
    return FAILED;
  }
  auto tiling_data = static_cast<void *>(tiling_data_holder.get());
  std::vector<uint32_t> symbols_value;
  auto ret = tiling_func(tiling_data, &workspace_size, &block_dim_, nullptr);
  if (ret != 0) {
    LOG_PRINT("ERROR: tiling_func failed, ret: %d\n", ret);
    return FAILED;
  }
  return af::SUCCESS;
}

int32_t AutofuseKernelInfo::MallocWorkSpace(uint64_t &size) {
  if (size == 0) {
    LOG_PRINT("WARN: workspace size is %" PRIu64 "\n", size);
    return af::SUCCESS;
  }

  auto ret = aclrtMalloc(&workspace_, size, ACL_MEM_MALLOC_HUGE_FIRST);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("ERROR: allocate workspace failed, ret: %d\n", ret);
    return FAILED;
  }
  ret = aclrtMemset(workspace_, size, 0U, size);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("ERROR: memset workspace failed, ret: %d\n", ret);
    (void)aclrtFree(workspace_);
    return FAILED;
  }
  return af::SUCCESS;
}

void ReportApiInfo(const uint64_t beginTime) {
  MsprofApi info{};
  info.type = MSPROF_REPORT_NODE_LAUNCH_TYPE;
  info.itemId = 0UL;
  info.level = MSPROF_REPORT_NODE_LEVEL;
  info.threadId = gettid();
  info.beginTime = beginTime;
  info.endTime = MsprofSysCycleTime();
  info.magicNumber = MSPROF_REPORT_DATA_MAGIC_NUM;
  info.reserve = 0U;
  const int32_t res = MsprofReportApi(true, &info);
}

void ReportNodeBasicInfo(const uint64_t beginTime, uint32_t block_dim, std::string graph_name) {
  struct MsprofCompactInfo nodeInfo;
  nodeInfo.level = MSPROF_REPORT_NODE_LEVEL;
  nodeInfo.type = MSPROF_REPORT_NODE_BASIC_INFO_TYPE;
  nodeInfo.threadId = gettid();
  nodeInfo.timeStamp = beginTime;
  nodeInfo.data.nodeBasicInfo.opName = MsprofGetHashId(graph_name.c_str(), graph_name.length());
  nodeInfo.data.nodeBasicInfo.opType = MsprofGetHashId(graph_name.c_str(), graph_name.length());
  nodeInfo.data.nodeBasicInfo.taskType = MSPROF_GE_TASK_TYPE_AIV;
  nodeInfo.data.nodeBasicInfo.blockDim = block_dim;
  MsprofReportCompactInfo(false, static_cast<void *>(&nodeInfo), sizeof(MsprofCompactInfo));
}

int32_t AutofuseKernelInfo::Distribute(uint64_t *tiling_data) {
  const std::string launch_func_name = "AutofuseLaunchV2";
  const auto launch_func = reinterpret_cast<LaunchFunc>(mmDlsym(handles_, launch_func_name.c_str()));
  if (launch_func == nullptr) {
    LOG_PRINT("ERROR: launch_func is nullptr, %s\n", launch_func_name.c_str());
    return FAILED;
  }

  const uint64_t startTime = MsprofSysCycleTime();
  auto ret = launch_func(block_dim_, stream_, input_addr_.get(), kernel_config_.input_num, output_addr_.get(),
                         kernel_config_.output_num, workspace_, reinterpret_cast<void *>(tiling_data));
  CHECK_RET(ret == SUCCESS, LOG_PRINT("launch failed, ERROR: %d\n", ret); return FAILED);
  ReportNodeBasicInfo(startTime, block_dim_, graph_name_snake_);
  ReportApiInfo(startTime);

  ret = aclrtSynchronizeStream(stream_);
  CHECK_RET(ret == SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed, ERROR: %d\n", ret); return FAILED);

  for (int i = 0; i < kernel_config_.output_num; ++i) {
    auto output_size =
        GetShapeSize(kernel_config_.output_shape[i]) * ge::GetSizeByDataType(kernel_config_.output_data_type[i]);
    std::vector<int8_t> result_data(output_size, 0);
    ret = aclrtMemcpy(result_data.data(), result_data.size() * sizeof(result_data[0]), output_addr_[i], output_size,
                      ACL_MEMCPY_DEVICE_TO_HOST);
    auto file_path = "out/Output_" + std::to_string(i) + ".bin";
    SaveBinToFile(file_path, (void *)result_data.data(), output_size);
  }

  return af::SUCCESS;
}

int FuseGraphTest(int deviceId, aclrtStream &stream, const std::string config_path) {
  auto ret = EnvInit(deviceId, &stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return FAILED);

  AutofuseKernelInfo fuse_kernel;
  fuse_kernel.Init(stream, config_path);

  ret = fuse_kernel.LoadSoHandles();
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("LoadSoHandles failed. ERROR: %d\n", ret); return FAILED);

  ret = fuse_kernel.ParseTaskRunParam();
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("ParseTaskRunParam failed. ERROR: %d\n", ret); return FAILED);

  size_t tiling_size = fuse_kernel.GetTilingSize();

  // 构造tiling
  std::unique_ptr<uint8_t[]> tiling_data_holder = MakeUnique<uint8_t[]>(tiling_size);
  CHECK_FREE_RET(tiling_data_holder != nullptr, LOG_PRINT("ERROR: tiling_data_holder is nullptr\n"); return FAILED);

  // tiling & malloc workspace addr
  uint64_t workspace_size = 0U;
  ret = fuse_kernel.DoTiling(tiling_data_holder, workspace_size);
  CHECK_FREE_RET(ret == SUCCESS, LOG_PRINT("ERROR: DoTiling failed.\n"); return FAILED);
  LOG_PRINT("workspace_size: %" PRIu64 "\n", workspace_size);

  ret = fuse_kernel.MallocWorkSpace(workspace_size);
  CHECK_FREE_RET(ret == SUCCESS, LOG_PRINT("ERROR: MallocWorkSpace failed.\n"); return FAILED);

  uint64_t *tiling_data = reinterpret_cast<uint64_t *>(tiling_data_holder.get());
  ret = fuse_kernel.Distribute(tiling_data);
  CHECK_FREE_RET(ret == SUCCESS, LOG_PRINT("ERROR: Distribute failed.\n"); return FAILED);
  LOG_PRINT("Distribute success.\n");

  return ACL_SUCCESS;
}

int main(int argc, char *argv[]) {
  if ((argc != 2) || (argv[1] == nullptr)) {
    std::string cmd_demo = "./fused_graph_test /home/autofuse/asc_graph_test/config/fused_graph_kernel_config.json";
    LOG_PRINT("ERROR: Please input config path. The cmd like: %s\n", cmd_demo.c_str());
    return FAILED;
  }

  std::string kernel_config_path = argv[1];
  LOG_PRINT("DEBUG: kernel_config_path: %s\n", kernel_config_path.c_str());
  // ASCEND_DEVICE_ID
  int32_t deviceId = 0;
  const char *env_var = std::getenv("ASCEND_DEVICE_ID");
  if (env_var != nullptr) {
    try {
      deviceId = std::stoi(env_var);
    } catch (const std::exception &e) {
      std::cerr << "ASCEND_DEVICE_ID : [" << env_var << "] is error" << std::endl;
    }
  }
  LOG_PRINT("DEBUG: using device id : %d\n", deviceId);
  aclrtStream stream;
  auto ret = FuseGraphTest(deviceId, stream, kernel_config_path);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("FuseGraphTest failed. ERROR: %d\n", ret); return ret);

  Finalize(deviceId, stream);
  return af::SUCCESS;
}
