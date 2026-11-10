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

#include <chrono>
#include <csignal>
#include <fcntl.h>
#include <sys/wait.h>
#include <cstdlib>
#include <fstream>
#include <poll.h>
#include <string>
#include <thread>
#include <unistd.h>

using namespace model_spec::compile;

namespace {
int terminationLog = -1;

void RecordTermination(int) {
  const char marker = '1';
  (void)write(terminationLog, &marker, 1);
}

std::string FaultMarkerPath(const CompileJob &job) {
  return job.outputElfPath.substr(0, job.outputElfPath.find_last_of('/') + 1) + "deferred-worker";
}

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
  // 故意永不返回，供进程池测试超时、SIGTERM 无效及最终强杀；只能由父进程清理。
  for (;;) {
    pause();
  }
}

bool AwaitFaultyWorkerDisabled(const std::string &marker) {
  // 等父进程关闭故障连接后再结束本任务，下一份任务必定发生在故障处理之后。
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (std::chrono::steady_clock::now() < deadline) {
    pid_t faultPid = -1;
    std::ifstream(marker + ".pid") >> faultPid;
    if (std::ifstream(marker + ".disabled").good() || std::ifstream(marker + ".signal").peek() != EOF ||
        (faultPid > 0 && kill(faultPid, 0) != 0)) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  return false;
}

bool FaultyWorkerStillAlive(const std::string &marker) {
  pid_t faultPid = -1;
  std::ifstream(marker + ".pid") >> faultPid;
  std::string processStat;
  std::getline(std::ifstream("/proc/" + std::to_string(faultPid) + "/stat"), processStat);
  const auto stateOffset = processStat.find_last_of(')');
  const bool zombie = stateOffset != std::string::npos && processStat[stateOffset + 2] == 'Z';
  return faultPid > 0 && kill(faultPid, 0) == 0 && !zombie && std::ifstream(marker + ".signal").peek() == EOF;
}

void ApplyProcessFault(const CompileJob &job) {
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
}

void ProduceArtifact(const CompileJob &job, Message &result) {
  std::ofstream file(job.outputElfPath, std::ios::binary);
  file << "\177ELF" << getpid();
  file.close();
  if (job.kernelEntry == "nonzero-success") {
    result.reply.aclrtcCode = 37;
  }
  if (job.kernelEntry == "corrupt") {
    std::ofstream corrupt(job.outputElfPath, std::ios::binary | std::ios::trunc);
    corrupt << "broken";
  }
  if (job.kernelEntry == "missing") {
    unlink(job.outputElfPath.c_str());
  }
  if (job.kernelEntry == "wrong-id") {
    ++result.jobId;
  }
}

Message Execute(const Message &message) {
  Message result;
  result.kind = MessageKind::RESULT;
  result.jobId = message.jobId;
  const auto &job = message.job;
  const auto marker = FaultMarkerPath(job);
  if (job.kernelEntry == "wrong-id-live") {
    terminationLog = open((marker + ".signal").c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
    signal(SIGTERM, RecordTermination);
    std::ofstream(marker + ".pid") << getpid();
    ++result.jobId;
    return result;
  }
  if (job.kernelEntry == "await-disabled" && !AwaitFaultyWorkerDisabled(marker)) {
    result.reply.aclrtcCode = 37;
    result.reply.detail = "parent did not disable faulty worker";
    return result;
  }
  if (job.kernelEntry == "check-deferred" && !FaultyWorkerStillAlive(marker)) {
    result.reply.aclrtcCode = 37;
    result.reply.detail = "faulty worker was terminated before healthy jobs completed";
    return result;
  }
  ApplyProcessFault(job);
  if (job.kernelEntry == "fail") {
    result.reply.aclrtcCode = 37;
    result.reply.detail = "ordinary compilation failure";
    return result;
  }
  ProduceArtifact(job, result);
  return result;
}

void AwaitParentDisconnect(FramedChannel &channel, const CompileJob &job) {
  while (channel.Fd() >= 0) {
    pollfd peer{channel.Fd(), POLLIN, 0};
    (void)poll(&peer, 1, 20);
    std::vector<Message> ignored;
    std::string readError;
    const auto status = channel.PumpRead(ignored, readError);
    if (status == IoStatus::PeerClosed || status == IoStatus::Error) {
      channel.Close();
    }
  }
  std::ofstream(FaultMarkerPath(job) + ".disabled") << '1';
  Hang();
}

enum class MessageAction { Continue, Stop, Error };

MessageAction HandleMessage(FramedChannel &channel, const Message &message, const std::string &mode) {
  if (message.kind == MessageKind::SHUTDOWN) {
    if (mode == "shutdown-hang") {
      Hang();
    }
    return MessageAction::Stop;
  }
  if (message.kind != MessageKind::JOB || !Send(channel, Execute(message))) {
    return MessageAction::Error;
  }
  if (message.job.kernelEntry == "wrong-id-live") {
    AwaitParentDisconnect(channel, message.job);
  }
  return MessageAction::Continue;
}

int ReceiveJobs(FramedChannel &channel, const std::string &mode) {
  while (channel.Fd() >= 0) {
    pollfd descriptor{channel.Fd(), POLLIN, 0};
    if (poll(&descriptor, 1, -1) < 0) {
      continue;
    }
    std::vector<Message> messages;
    std::string error;
    const auto status = channel.PumpRead(messages, error);
    for (const auto &message : messages) {
      const auto action = HandleMessage(channel, message, mode);
      if (action != MessageAction::Continue) {
        return action == MessageAction::Stop ? 0 : 4;
      }
    }
    if (status == IoStatus::Error || status == IoStatus::PeerClosed) {
      channel.Close();
    }
  }
  return 0;
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
  return ReceiveJobs(channel, mode);
}
