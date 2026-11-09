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
#include "acl/acl_rt_compile.h"

#include <fcntl.h>
#include <poll.h>
#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

namespace model_spec::compile {
namespace {

bool ParseControlFd(int argc, char **argv, int &fd);
int RunWorker(int fd);
bool Flush(FramedChannel &channel);
bool WaitForEvent(FramedChannel &channel, short events);
Message Compile(const Message &message);
void ReportJobError(FramedChannel &channel, uint64_t jobId, const char *detail);

bool ParseControlFd(int argc, char **argv, int &fd) {
  if (argc != 3 || std::strcmp(argv[1], "--control-fd") != 0 || argv[2][0] == '\0') {
    return false;
  }
  errno = 0;
  char *end = nullptr;
  const long value = std::strtol(argv[2], &end, 10);
  if (errno != 0 || *end != '\0' || value < 0 || value > INT_MAX) {
    return false;
  }
  fd = static_cast<int>(value);
  const int statusFlags = fcntl(fd, F_GETFL);
  const int descriptorFlags = fcntl(fd, F_GETFD);
  return statusFlags >= 0 && descriptorFlags >= 0 && fcntl(fd, F_SETFL, statusFlags | O_NONBLOCK) == 0 &&
         fcntl(fd, F_SETFD, descriptorFlags | FD_CLOEXEC) == 0;
}

int RunWorker(int fd) {
  FramedChannel channel(fd);
  Message ready;
  ready.kind = MessageKind::READY;
  ready.buildId = WORKER_BUILD_ID;
  std::string error;
  if (!channel.Queue(ready, error) || !Flush(channel)) {
    return 1;
  }
  while (true) {
    std::vector<Message> messages;
    const IoStatus status = channel.PumpRead(messages, error);
    for (const auto &message : messages) {
      if (message.kind == MessageKind::SHUTDOWN) {
        return 0;
      }
      if (message.kind != MessageKind::JOB) {
        ReportJobError(channel, message.jobId, "worker expected JOB or SHUTDOWN");
        return 1;
      }
      try {
        const Message result = Compile(message);
        if (!channel.Queue(result, error) || !Flush(channel)) {
          return 1;
        }
      } catch (const std::exception &exception) {
        ReportJobError(channel, message.jobId, exception.what());
        return 1;
      } catch (...) {
        ReportJobError(channel, message.jobId, "worker request execution raised an unknown exception");
        return 1;
      }
    }
    if (status == IoStatus::PeerClosed) {
      return 0;
    }
    if (status == IoStatus::Error) {
      std::fprintf(stderr, "worker protocol error: %s\n", error.c_str());
      return 1;
    }
    if (!WaitForEvent(channel, POLLIN)) {
      return 1;
    }
  }
}

bool Flush(FramedChannel &channel) {
  std::string error;
  while (channel.WantsWrite()) {
    const IoStatus status = channel.PumpWrite(error);
    if (status == IoStatus::Error || status == IoStatus::PeerClosed) {
      return false;
    }
    if (channel.WantsWrite() && !WaitForEvent(channel, POLLOUT)) {
      return false;
    }
  }
  return true;
}

bool WaitForEvent(FramedChannel &channel, short events) {
  pollfd fd{channel.Fd(), events, 0};
  int result;
  do {
    result = poll(&fd, 1, -1);
  } while (result < 0 && errno == EINTR);
  return result > 0 && (fd.revents & POLLNVAL) == 0;
}

Message Compile(const Message &message) {
  const CompileJob &job = message.job;
  std::vector<const void *> argumentAddresses;
  std::vector<const char *> basicOptionPointers;
  std::vector<const char *> skOptionPointers;
  argumentAddresses.reserve(job.arguments.size());
  basicOptionPointers.reserve(job.basicOptions.size());
  skOptionPointers.reserve(job.skOptions.size());
  for (const auto &argument : job.arguments) {
    argumentAddresses.push_back(argument.data());
  }
  for (const auto &option : job.basicOptions) {
    basicOptionPointers.push_back(option.c_str());
  }
  for (const auto &option : job.skOptions) {
    skOptionPointers.push_back(option.c_str());
  }
  aclrtcKernelSpecRequest request{};
  request.resourceId = job.resourceId.c_str();
  request.kernelEntry = job.kernelEntry.c_str();
  request.argsCount = argumentAddresses.size();
  request.argsAddr = argumentAddresses.data();
  request.argsBytes = job.argumentBytes.data();
  request.options = basicOptionPointers.data();
  request.optionCount = basicOptionPointers.size();
  request.skOptions = skOptionPointers.data();
  request.skOptionCount = skOptionPointers.size();

  Message result;
  result.kind = MessageKind::RESULT;
  result.jobId = message.jobId;
  const std::string partialPath = job.outputElfPath + ".partial";
  result.reply.aclrtcCode = aclrtcKernelSpecialization(&request, partialPath.c_str());
  if (result.reply.aclrtcCode != 0) {
    result.reply.kind = WorkerResultKind::CompileFailed;
    result.reply.detail = "aclrtcKernelSpecialization failed";
    return result;
  }
  result.reply.kind = WorkerResultKind::ArtifactFailed;
  ArtifactInfo artifact;
  if (!InspectArtifact(partialPath, artifact, result.reply.detail)) {
    return result;
  }
  if (std::rename(partialPath.c_str(), job.outputElfPath.c_str()) != 0) {
    result.reply.detail = std::string("rename failed: ") + std::strerror(errno);
    return result;
  }
  result.reply.kind = WorkerResultKind::Produced;
  result.reply.artifactSize = artifact.size;
  result.reply.artifactSha256 = artifact.sha256;
  return result;
}

void ReportJobError(FramedChannel &channel, uint64_t jobId, const char *detail) {
  try {
    Message message;
    message.kind = MessageKind::JOB_ERROR;
    message.jobId = jobId;
    message.reply.detail = std::string(detail).substr(0, MAX_STRING_BYTES);
    std::string error;
    if (channel.Queue(message, error)) {
      Flush(channel);
    }
  } catch (...) {
    // Closing the channel is also a fatal task result when allocation itself failed.
  }
}

}  // namespace
}  // namespace model_spec::compile

int main(int argc, char **argv) {
  int controlFd = -1;
  if (!model_spec::compile::ParseControlFd(argc, argv, controlFd)) {
    std::fprintf(stderr, "usage: aclrtc_compile_worker --control-fd <open-fd>\n");
    return 1;
  }
  try {
    return model_spec::compile::RunWorker(controlFd);
  } catch (const std::exception &exception) {
    std::fprintf(stderr, "worker failed: %s\n", exception.what());
    return 1;
  } catch (...) {
    return 1;
  }
}
