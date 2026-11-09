/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "compile_artifact.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <openssl/evp.h>

#include <array>
#include <cerrno>
#include <cstring>
#include <memory>
#include <utility>

namespace model_spec::compile {
namespace {

struct ArtifactFile {
  int fd;
  ~ArtifactFile() {
    close(fd);
  }
};

bool Fail(std::string &error, const std::string &message) {
  error = message;
  return false;
}

}  // namespace

bool InspectArtifact(const std::string &path, ArtifactInfo &info, std::string &error) {
  error.clear();
  info = {};
  ArtifactFile file{open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK)};
  if (file.fd < 0) {
    return Fail(error, std::string("open artifact: ") + std::strerror(errno));
  }
  struct stat attributes {};
  if (fstat(file.fd, &attributes) != 0 || !S_ISREG(attributes.st_mode) || attributes.st_size < 4 ||
      static_cast<uint64_t>(attributes.st_size) > MAX_ARTIFACT_BYTES) {
    return Fail(error, "artifact is not a regular ELF within size limits");
  }
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> digest(EVP_MD_CTX_new(), EVP_MD_CTX_free);
  if (digest == nullptr || EVP_DigestInit_ex(digest.get(), EVP_sha256(), nullptr) != 1) {
    return Fail(error, "initialize artifact SHA-256 failed");
  }
  std::array<unsigned char, 65536> buffer{};
  std::array<unsigned char, 4> magic{};
  uint64_t total = 0;
  while (true) {
    const auto count = read(file.fd, buffer.data(), buffer.size());
    if (count < 0 && errno == EINTR) {
      continue;
    }
    if (count < 0) {
      return Fail(error, std::string("read artifact: ") + std::strerror(errno));
    }
    if (count == 0) {
      break;
    }
    const auto size = static_cast<size_t>(count);
    for (size_t index = 0; index < size && total + index < magic.size(); ++index) {
      magic[total + index] = buffer[index];
    }
    total += size;
    if (total > MAX_ARTIFACT_BYTES || EVP_DigestUpdate(digest.get(), buffer.data(), size) != 1) {
      return Fail(error, "artifact exceeds size limit or SHA-256 update failed");
    }
  }
  if (total != static_cast<uint64_t>(attributes.st_size) ||
      magic != std::array<unsigned char, 4>{0x7f, 'E', 'L', 'F'}) {
    return Fail(error, "artifact size changed while reading or ELF magic is invalid");
  }
  std::array<unsigned char, EVP_MAX_MD_SIZE> hash{};
  unsigned int hashSize = 0;
  if (EVP_DigestFinal_ex(digest.get(), hash.data(), &hashSize) != 1 || hashSize != 32) {
    return Fail(error, "finalize artifact SHA-256 failed");
  }
  constexpr char HEX[] = "0123456789abcdef";
  std::string sha256;
  sha256.reserve(64);
  for (unsigned int index = 0; index < hashSize; ++index) {
    sha256.push_back(HEX[hash[index] >> 4]);
    sha256.push_back(HEX[hash[index] & 15]);
  }
  info = {total, std::move(sha256)};
  return true;
}

}  // namespace model_spec::compile
