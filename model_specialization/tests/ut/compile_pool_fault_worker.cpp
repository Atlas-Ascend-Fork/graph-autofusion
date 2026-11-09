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
#include "compile_protocol.h"

#include <chrono>
#include <csignal>
#include <sys/wait.h>
#include <cstdlib>
#include <fstream>
#include <poll.h>
#include <string>
#include <thread>
#include <unistd.h>

using namespace model_spec::compile;

namespace {
bool Send(FramedChannel &channel, const Message &message) {
  std::string error;
  if (!channel.Queue(message, error)) {
    return false;
  }
  while (channel.WantsWrite()) {
    if (channel.PumpWrite(error) == IoStatus::Error) {
      return false;
    }
    pollfd descriptor{channel.Fd(), POLLOUT, 0};
    (void)poll(&descriptor, 1, 20);
  }
  return true;
}

void Hang() {
  for (;;) {
    pause();
  }
}

Message Execute(const Message &message) {
  Message result;
  result.kind = MessageKind::RESULT;
  result.jobId = message.jobId;
  const auto &job = message.job;
  if (job.kernelEntry == "crash") {
    _exit(17);
  }
  if (job.kernelEntry == "ignore-term") {
    signal(SIGTERM, SIG_IGN);
    Hang();
  }
  if (job.kernelEntry == "hang") {
    Hang();
  }
  if (job.kernelEntry == "delay") {
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
  }
  if (job.kernelEntry == "fail" || job.kernelEntry == "zero-failure") {
    result.reply.kind = WorkerResultKind::CompileFailed;
    result.reply.aclrtcCode = job.kernelEntry == "zero-failure" ? 0 : 37;
    result.reply.detail = "ordinary compilation failure";
    return result;
  }
  if (job.kernelEntry == "descendant") {
    const pid_t descendant = fork();
    if (descendant < 0) {
      _exit(19);
    }
    if (descendant == 0) {
      Hang();
    }
    std::ofstream descendantPid(job.outputElfPath + ".descendant");
    descendantPid << descendant;
  }
  std::ofstream file(job.outputElfPath, std::ios::binary);
  file << "\177ELF" << getpid();
  file.close();
  ArtifactInfo artifact;
  std::string error;
  if (!InspectArtifact(job.outputElfPath, artifact, error)) {
    _exit(18);
  }
  result.reply.kind = WorkerResultKind::Produced;
  if (job.kernelEntry == "nonzero-success") {
    result.reply.aclrtcCode = 37;
  }
  result.reply.artifactSize = artifact.size;
  result.reply.artifactSha256 = artifact.sha256;
  if (job.kernelEntry == "corrupt") {
    std::ofstream corrupt(job.outputElfPath, std::ios::binary | std::ios::trunc);
    corrupt << "broken";
  }
  if (job.kernelEntry == "wrong-id") {
    ++result.jobId;
  }
  return result;
}
}  // namespace

int main(int argc, char **argv) {
  if (argc != 3 || std::string(argv[1]) != "--control-fd") {
    return 2;
  }
  FramedChannel channel(std::stoi(argv[2]));
  const char *rawMode = std::getenv("MODEL_SPEC_FAULT_MODE");
  const std::string mode = rawMode == nullptr ? "" : rawMode;
  if (mode == "silent-startup") {
    Hang();
  }
  Message ready;
  ready.kind = MessageKind::READY;
  ready.buildId = mode == "wrong-ready" ? "wrong-build" : WORKER_BUILD_ID;
  if (!Send(channel, ready)) {
    return 3;
  }
  if (mode == "nonreader") {
    Hang();
  }
  for (;;) {
    pollfd descriptor{channel.Fd(), POLLIN, 0};
    if (poll(&descriptor, 1, -1) < 0) {
      continue;
    }
    std::vector<Message> messages;
    std::string error;
    const auto status = channel.PumpRead(messages, error);
    for (const auto &message : messages) {
      if (message.kind == MessageKind::SHUTDOWN) {
        if (mode == "shutdown-hang") {
          Hang();
        }
        return 0;
      }
      if (message.kind != MessageKind::JOB || !Send(channel, Execute(message))) {
        return 4;
      }
    }
    if (status == IoStatus::Error || status == IoStatus::PeerClosed) {
      return 0;
    }
  }
}
