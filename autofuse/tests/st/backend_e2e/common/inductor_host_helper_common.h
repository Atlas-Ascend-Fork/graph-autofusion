/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_TESTS_ST_BACKEND_E2E_COMMON_INDUCTOR_HOST_HELPER_COMMON_H_
#define AUTOFUSE_TESTS_ST_BACKEND_E2E_COMMON_INDUCTOR_HOST_HELPER_COMMON_H_

#include <cstdint>
#include <dlfcn.h>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace autofuse::tests {

using InputConfigs = std::vector<std::map<std::string, std::string>>;

struct ResLimit {
  uint32_t valid_num = 0;
  uint32_t aiv_num = 0;
  uint32_t aic_num = 0;
  uint32_t ub_size = 0;
  uint32_t resv[10];
};

struct HostHelperOptions {
  std::string host_so;
  std::string tiling_repr_out;
  std::string input_configs_file;
  std::vector<int64_t> dynamic_shape_args;
  int64_t topn = 4;
  bool verify_empty_config = false;
  bool check_z0t_positive = false;
};

class HostCaseRunner {
 public:
  virtual ~HostCaseRunner() = default;
  virtual bool Resolve(void *handle) = 0;
  virtual int64_t GenerateTopn(const InputConfigs &input_configs, int64_t topn) = 0;
  virtual int64_t RunAutofuseTiling() = 0;
  virtual size_t ResultSize() const = 0;
  virtual int64_t ResultWorkspace(size_t index) const = 0;
  virtual int64_t ResultBlockDim(size_t index) const = 0;
  virtual std::string ResultRepr(size_t index) const = 0;
  virtual std::string DefaultRepr() const = 0;
  virtual uint64_t DefaultWorkspace() const = 0;
  virtual uint32_t DefaultBlockDim() const = 0;
  virtual bool VerifyExtraTopnResult() const {
    return true;
  }
};

template <typename Fn>
Fn ResolveSymbol(void *handle, const char *symbol) {
  dlerror();
  void *ptr = dlsym(handle, symbol);
  const char *error = dlerror();
  if (error != nullptr) {
    std::cerr << symbol << " not found: " << error << std::endl;
    return nullptr;
  }
  return reinterpret_cast<Fn>(ptr);
}

bool ParseHostHelperOptions(int argc, char **argv, HostHelperOptions *options);
int RunHostCheck(const HostHelperOptions &options, HostCaseRunner *runner);

}  // namespace autofuse::tests

#endif  // AUTOFUSE_TESTS_ST_BACKEND_E2E_COMMON_INDUCTOR_HOST_HELPER_COMMON_H_
