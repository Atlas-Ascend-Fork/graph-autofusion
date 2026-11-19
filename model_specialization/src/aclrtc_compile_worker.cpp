/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "compile_protocol.h"
#include "acl/acl_rt_compile.h"
#include "sk_log.h"

#include <fcntl.h>
#include <poll.h>
#include <cerrno>
#include <chrono>
#include <cinttypes>
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
bool SendReady(FramedChannel &channel);
bool CompileAndReply(FramedChannel &channel, const Message &message);
bool Flush(FramedChannel &channel);
bool WaitForEvent(FramedChannel &channel, short events);
Message Compile(const Message &message);
void ReportJobError(FramedChannel &channel, uint64_t jobId, const char *detail);

bool ParseControlFd(int argc, char **argv, int &fd) {
  constexpr int WORKER_ARG_COUNT = 3;
  constexpr int CONTROL_FD_VALUE_INDEX = 2;
  constexpr int DECIMAL_BASE = 10;
  if (argc != WORKER_ARG_COUNT || std::strcmp(argv[1], "--control-fd") != 0 ||
      argv[CONTROL_FD_VALUE_INDEX][0] == '\0') {
    return false;
  }
  errno = 0;
  char *end = nullptr;
  const long value = std::strtol(argv[CONTROL_FD_VALUE_INDEX], &end, DECIMAL_BASE);
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
  const auto workerPid = static_cast<int>(getpid());
  SK_DLOGI("[SpecCompile] Compile worker started: workerPid=%d controlFd=%d", workerPid, fd);
  FramedChannel channel(fd);
  if (!SendReady(channel)) {
    SK_DLOGE("[SpecCompile] Worker failed to send READY: workerPid=%d", workerPid);
    return 1;
  }
  while (channel.Fd() >= 0) {
    std::vector<Message> messages;
    std::string error;
    const IoStatus status = channel.PumpRead(messages, error);
    // 一次读取可能同时得到完整任务和 EOF，先处理完整任务，再结束连接。
    for (const auto &message : messages) {
      if (message.kind == MessageKind::SHUTDOWN) {
        SK_DLOGI("[SpecCompile] Received SHUTDOWN from parent: workerPid=%d", workerPid);
        return 0;
      }
      if (!CompileAndReply(channel, message)) {
        return 1;
      }
    }
    if (status == IoStatus::PeerClosed) {
      SK_DLOGI("[SpecCompile] Parent closed compile channel: workerPid=%d", workerPid);
      channel.Close();
      return 0;
    }
    if (status == IoStatus::Error) {
      SK_DLOGE("[SpecCompile] Worker protocol error: workerPid=%d reason=%s", workerPid, error.c_str());
      std::fprintf(stderr, "[SpecCompile] Worker protocol error: workerPid=%d reason=%s\n", workerPid, error.c_str());
      return 1;
    }
    if (!WaitForEvent(channel, POLLIN)) {
      SK_DLOGE("[SpecCompile] Worker receive wait failed: workerPid=%d", workerPid);
      std::fprintf(stderr, "[SpecCompile] Worker receive wait failed: workerPid=%d\n", workerPid);
      return 1;
    }
  }
  return 0;
}

bool SendReady(FramedChannel &channel) {
  Message ready;
  ready.kind = MessageKind::READY;
  ready.buildId = WORKER_BUILD_ID;
  std::string error;
  if (!channel.Queue(ready, error)) {
    SK_DLOGE("[SpecCompile] Worker failed to queue READY: workerPid=%d reason=%s", static_cast<int>(getpid()),
             error.c_str());
    std::fprintf(stderr, "[SpecCompile] Worker failed to queue READY: workerPid=%d reason=%s\n",
                 static_cast<int>(getpid()), error.c_str());
    return false;
  }
  if (!Flush(channel)) {
    return false;
  }
  SK_DLOGI("[SpecCompile] Sent READY to parent: workerPid=%d", static_cast<int>(getpid()));
  return true;
}

// 返回值表示是否完成消息处理；kernel 编译失败仍发送 RESULT，worker 可以继续接任务。
bool CompileAndReply(FramedChannel &channel, const Message &message) {
  if (message.kind != MessageKind::JOB) {
    ReportJobError(channel, message.jobId, "worker expected JOB or SHUTDOWN");
    return false;
  }
  const auto workerPid = static_cast<int>(getpid());
  SK_DLOGI("[SpecCompile] Received JOB from parent: workerPid=%d jobId=%" PRIu64 " kernel=%s", workerPid, message.jobId,
           message.job.kernelEntry.c_str());
  try {
    const Message result = Compile(message);
    std::string error;
    if (!channel.Queue(result, error) || !Flush(channel)) {
      SK_DLOGE("[SpecCompile] Worker failed to send RESULT: workerPid=%d jobId=%" PRIu64 " reason=%s", workerPid,
               result.jobId, error.empty() ? "channel closed or write failed" : error.c_str());
      std::fprintf(stderr, "[SpecCompile] Worker failed to send RESULT: workerPid=%d jobId=%" PRIu64 " reason=%s\n",
                   workerPid, result.jobId, error.empty() ? "channel closed or write failed" : error.c_str());
      return false;
    }
    SK_DLOGI("[SpecCompile] Sent RESULT to parent: workerPid=%d jobId=%" PRIu64, workerPid, result.jobId);
    return true;
  } catch (const std::exception &exception) {
    ReportJobError(channel, message.jobId, exception.what());
  } catch (...) {
    ReportJobError(channel, message.jobId, "worker request execution raised an unknown exception");
  }
  return false;
}

bool Flush(FramedChannel &channel) {
  std::string error;
  while (channel.WantsWrite()) {
    const IoStatus status = channel.PumpWrite(error);
    if (status == IoStatus::Error || status == IoStatus::PeerClosed) {
      SK_DLOGE("[SpecCompile] Worker channel write failed: workerPid=%d reason=%s", static_cast<int>(getpid()),
               error.empty() ? "peer closed" : error.c_str());
      std::fprintf(stderr, "[SpecCompile] Worker channel write failed: workerPid=%d reason=%s\n",
                   static_cast<int>(getpid()), error.empty() ? "peer closed" : error.c_str());
      return false;
    }
    if (channel.WantsWrite() && !WaitForEvent(channel, POLLOUT)) {
      SK_DLOGE("[SpecCompile] Worker send wait failed: workerPid=%d", static_cast<int>(getpid()));
      std::fprintf(stderr, "[SpecCompile] Worker send wait failed: workerPid=%d\n", static_cast<int>(getpid()));
      return false;
    }
  }
  return true;
}

bool WaitForEvent(FramedChannel &channel, short events) {
  constexpr int WAIT_INDEFINITELY = -1;
  pollfd fd{channel.Fd(), events, 0};
  int result;
  do {
    result = poll(&fd, 1, WAIT_INDEFINITELY);
  } while (result < 0 && errno == EINTR);
  return result > 0 && (fd.revents & POLLNVAL) == 0;
}

Message Compile(const Message &message) {
  const CompileJob &job = message.job;
  // ACLRTC 同步读取这些指针；它们都指向 worker 的内存，调用返回前一直有效。
  std::vector<const void *> argumentAddresses;
  std::vector<uint64_t> argumentBytes;
  std::vector<const char *> basicOptionPointers;
  std::vector<const char *> skOptionPointers;
  argumentAddresses.reserve(job.arguments.size());
  argumentBytes.reserve(job.arguments.size());
  basicOptionPointers.reserve(job.basicOptions.size());
  skOptionPointers.reserve(job.skOptions.size());
  for (const auto &argument : job.arguments) {
    argumentAddresses.push_back(argument.data());
    argumentBytes.push_back(argument.size());
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
  request.argsBytes = argumentBytes.data();
  request.options = basicOptionPointers.data();
  request.optionCount = basicOptionPointers.size();
  request.skOptions = skOptionPointers.data();
  request.skOptionCount = skOptionPointers.size();

  Message result;
  result.kind = MessageKind::RESULT;
  result.jobId = message.jobId;
  const auto workerPid = static_cast<int>(getpid());
  SK_DLOGI("[SpecCompile] Kernel compilation started: workerPid=%d jobId=%" PRIu64 " kernel=%s", workerPid,
           message.jobId, job.kernelEntry.c_str());
  const auto started = std::chrono::steady_clock::now();
  result.reply.aclrtcCode = aclrtcKernelSpecialization(&request, job.outputElfPath.c_str());
  const auto elapsedUs =
      std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - started).count();
  SK_DLOGI("[SpecCompile] Kernel compilation finished: workerPid=%d jobId=%" PRIu64
           " kernel=%s aclrtcCode=%d elapsedUs=%" PRId64,
           workerPid, message.jobId, job.kernelEntry.c_str(), result.reply.aclrtcCode, static_cast<int64_t>(elapsedUs));
  if (result.reply.aclrtcCode != 0) {
    result.reply.detail = "aclrtcKernelSpecialization failed";
  }
  return result;
}

void ReportJobError(FramedChannel &channel, uint64_t jobId, const char *detail) {
  SK_DLOGW("[SpecCompile] Worker job failed: workerPid=%d jobId=%" PRIu64 " detail=%s", static_cast<int>(getpid()),
           jobId, detail);
  // 此类错误会结束 worker；stderr 作为 CANN 日志之外的兜底，保留原始失败原因。
  std::fprintf(stderr, "[SpecCompile] Worker job failed: workerPid=%d jobId=%" PRIu64 " detail=%s\n",
               static_cast<int>(getpid()), jobId, detail);
  try {
    Message message;
    message.kind = MessageKind::JOB_ERROR;
    message.jobId = jobId;
    message.reply.detail = std::string(detail).substr(0, MAX_STRING_BYTES);
    std::string error;
    if (channel.Queue(message, error)) {
      Flush(channel);
    } else {
      SK_DLOGE("[SpecCompile] Worker failed to report JOB_ERROR: workerPid=%d jobId=%" PRIu64 " reason=%s",
               static_cast<int>(getpid()), jobId, error.c_str());
      std::fprintf(stderr,
                   "[SpecCompile] Worker failed to report JOB_ERROR: workerPid=%d jobId=%" PRIu64 " reason=%s\n",
                   static_cast<int>(getpid()), jobId, error.c_str());
    }
  } catch (...) {
    SK_DLOGE("[SpecCompile] Worker failed to report JOB_ERROR: workerPid=%d jobId=%" PRIu64, static_cast<int>(getpid()),
             jobId);
    std::fprintf(stderr,
                 "[SpecCompile] Worker failed to report JOB_ERROR: workerPid=%d jobId=%" PRIu64
                 " reason=unknown exception\n",
                 static_cast<int>(getpid()), jobId);
  }
}

}  // namespace
}  // namespace model_spec::compile

int main(int argc, char **argv) {
  int controlFd = -1;
  if (!model_spec::compile::ParseControlFd(argc, argv, controlFd)) {
    SK_DLOGE("[SpecCompile] Invalid worker arguments: usage=aclrtc_compile_worker --control-fd <open-fd>");
    std::fprintf(stderr,
                 "[SpecCompile] Invalid worker arguments: usage=aclrtc_compile_worker --control-fd <open-fd>\n");
    return 1;
  }
  try {
    return model_spec::compile::RunWorker(controlFd);
  } catch (const std::exception &exception) {
    SK_DLOGE("[SpecCompile] Worker terminated by exception: reason=%s", exception.what());
    std::fprintf(stderr, "[SpecCompile] Worker terminated by exception: reason=%s\n", exception.what());
    return 1;
  } catch (...) {
    SK_DLOGE("[SpecCompile] Worker terminated by unknown exception");
    std::fprintf(stderr, "[SpecCompile] Worker terminated by unknown exception\n");
    return 1;
  }
}
