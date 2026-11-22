/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "acl/acl_rt_compile.h"
#include "acl/acl_base_rt.h"

#include <fcntl.h>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <string>

extern "C" aclError aclrtcKernelSpecialization(const aclrtcKernelSpecRequest *request, const char *outputPath) {
  if (std::strcmp(request->kernelEntry, "compile_fail") == 0) {
    return ACL_ERROR_FAILURE;
  }
  if (std::strcmp(request->kernelEntry, "compile_failure") == 0) {
    return 12345;
  }
  if (std::strcmp(request->kernelEntry, "missing_output") == 0) {
    return 0;
  }
  std::ofstream output(outputPath, std::ios::binary);
  if (std::strcmp(request->kernelEntry, "invalid_output") == 0) {
    output << "invalid ELF";
    return 0;
  }
  output.write("\177ELF", 4);
  output << '\n' << request->resourceId << '\n' << request->kernelEntry << '\n';
  output << "nonblock=" << ((fcntl(3, F_GETFL) & O_NONBLOCK) != 0)
         << " cloexec=" << ((fcntl(3, F_GETFD) & FD_CLOEXEC) != 0) << '\n';
  output << "arguments=" << request->argsCount << '\n';
  for (uint64_t i = 0; i < request->argsCount; ++i) {
    output << std::dec << request->argsBytes[i] << ':';
    const auto *bytes = static_cast<const unsigned char *>(request->argsAddr[i]);
    for (uint64_t j = 0; j < request->argsBytes[i]; ++j) {
      output << std::hex << std::setfill('0') << std::setw(2) << static_cast<unsigned>(bytes[j]);
    }
    output << '\n';
  }
  output << std::dec << "options=" << request->optionCount << '\n';
  for (uint64_t i = 0; i < request->optionCount; ++i) {
    output << request->options[i] << '\n';
  }
  output << "skOptions=" << request->skOptionCount << '\n';
  for (uint64_t i = 0; i < request->skOptionCount; ++i) {
    output << request->skOptions[i] << '\n';
  }
  output << "reservedZero=";
  bool reservedZero = true;
  for (unsigned char byte : request->reserved) {
    reservedZero = reservedZero && byte == 0;
  }
  output << reservedZero << '\n';
  return output.good() ? 0 : 23456;
}
