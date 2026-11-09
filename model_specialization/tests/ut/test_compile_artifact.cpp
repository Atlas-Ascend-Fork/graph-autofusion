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

#include <cstdlib>
#include <fstream>
#include <gtest/gtest.h>

namespace model_spec::compile {
namespace {

class CompileArtifactTest : public testing::Test {
 protected:
  void SetUp() override {
    const char *root = std::getenv("TMPDIR");
    std::string directory = std::string(root != nullptr ? root : MODEL_SPEC_TEST_TMP_DIR) + "/artifact_XXXXXX";
    ASSERT_NE(mkdtemp(directory.data()), nullptr);
    directory_ = directory;
    path_ = directory_ + "/result.elf";
  }
  void TearDown() override {
    unlink(path_.c_str());
    unlink((directory_ + "/link").c_str());
    unlink((directory_ + "/fifo").c_str());
    rmdir(directory_.c_str());
  }
  void Write(const std::string &bytes) {
    std::ofstream stream(path_, std::ios::binary);
    stream.write(bytes.data(), bytes.size());
  }
  std::string directory_;
  std::string path_;
};

TEST_F(CompileArtifactTest, ReadsElfAndComputesSha256) {
  Write(
      std::string("\x7f"
                  "ELF",
                  4));
  ArtifactInfo info;
  std::string error;
  ASSERT_TRUE(InspectArtifact(path_, info, error)) << error;
  EXPECT_EQ(info.size, 4ULL);
  EXPECT_EQ(info.sha256, "3bdbb4fe8397cd2b842430b39ccff01a8663c751945ef5e9a09e267fb8b1d359");
}

TEST_F(CompileArtifactTest, RejectsMissingShortWrongMagicDirectorySymlinkAndFifo) {
  ArtifactInfo info;
  std::string error;
  EXPECT_FALSE(InspectArtifact(path_, info, error));
  Write("abc");
  EXPECT_FALSE(InspectArtifact(path_, info, error));
  Write("not an ELF");
  EXPECT_FALSE(InspectArtifact(path_, info, error));
  EXPECT_FALSE(InspectArtifact(directory_, info, error));
  Write(
      std::string("\x7f"
                  "ELF",
                  4));
  ASSERT_EQ(symlink(path_.c_str(), (directory_ + "/link").c_str()), 0);
  EXPECT_FALSE(InspectArtifact(directory_ + "/link", info, error));
  ASSERT_EQ(mkfifo((directory_ + "/fifo").c_str(), 0600), 0);
  EXPECT_FALSE(InspectArtifact(directory_ + "/fifo", info, error));
}

TEST_F(CompileArtifactTest, RejectsOversizedSparseArtifact) {
  const int fd = open(path_.c_str(), O_CREAT | O_RDWR, 0600);
  ASSERT_GE(fd, 0);
  ASSERT_EQ(ftruncate(fd, MAX_ARTIFACT_BYTES + 1), 0);
  close(fd);
  ArtifactInfo info;
  std::string error;
  EXPECT_FALSE(InspectArtifact(path_, info, error));
}

}  // namespace
}  // namespace model_spec::compile
