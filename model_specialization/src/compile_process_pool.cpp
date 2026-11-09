/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "compile_process_pool.h"

#include <algorithm>
#include <cerrno>
#include <climits>
#include <csignal>
#include <cstring>
#include <fcntl.h>
#include <poll.h>
#include <spawn.h>
#include <stdexcept>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
#include <utility>

#include "compile_artifact.h"

extern char **environ;

namespace model_spec::compile {
namespace {
using Clock = std::chrono::steady_clock;
using TimePoint = Clock::time_point;

enum class WorkerState { Starting, Idle, Sending, Busy, Stopping, Reaped };

struct WorkerSlot {
  pid_t pid = -1;
  WorkerState state = WorkerState::Reaped;
  std::unique_ptr<FramedChannel> channel;
  std::optional<size_t> currentJob;
  TimePoint deadline{};
  TimePoint killAt{};
  TimePoint groupCheckUntil{};
  bool termSent = false;
  bool killSent = false;
  bool leaderReaped = false;
  bool exitStatusKnown = false;
  bool groupUnknown = false;
  int exitStatus = 0;

  ~WorkerSlot() {
    if (channel != nullptr) {
      channel->Close();
    }
    if (pid > 0 && !leaderReaped) {
      (void)kill(-pid, SIGKILL);
      while (waitpid(pid, nullptr, 0) < 0 && errno == EINTR) {
      }
    }
  }
};

struct OwnedFd {
  int fd;
  ~OwnedFd() {
    if (fd >= 0) {
      close(fd);
    }
  }
  int Release() {
    const int owned = fd;
    fd = -1;
    return owned;
  }
};

std::string SystemError(const char *operation, int error) {
  return std::string(operation) + ": " + std::strerror(error);
}

bool IsActive(WorkerState state) {
  return state == WorkerState::Starting || state == WorkerState::Idle || state == WorkerState::Sending ||
         state == WorkerState::Busy;
}
}  // namespace

class CompileProcessPool::Impl {
 public:
  explicit Impl(PoolConfig config) : config_(std::move(config)) {}

  PoolRunResult Run(const std::vector<CompileJob> &jobs);
  CleanupReport CloseAndReap();

 private:
  bool StartWorkers(size_t count, std::string &error);
  bool SpawnWorker(WorkerSlot &worker, size_t index, std::string &error);
  void Dispatch(const std::vector<CompileJob> &jobs, size_t &next);
  void PollWorkers();
  void HandleMessage(WorkerSlot &worker, const Message &message);
  void FailWorker(WorkerSlot &worker, CompileOutcome outcome, const std::string &detail);
  void BeginStopping(WorkerSlot &worker, bool graceful);
  void AdvanceWorkers();
  void SignalGroup(WorkerSlot &worker, int signal);
  void VerifyProduced(const std::vector<CompileJob> &jobs);
  void AppendWorkerDiagnostics();
  bool HasActiveWorkers() const;
  bool AllReaped() const;
  int PollTimeout() const;

  PoolConfig config_;
  std::vector<std::unique_ptr<WorkerSlot>> workers_;
  PoolRunResult result_;
  std::vector<WorkerSlot *> producedBy_;
  std::vector<WorkerSlot *> dispatchedBy_;
  bool used_ = false;
  bool dispatched_ = false;
  bool startupFailed_ = false;
  TimePoint startupDeadline_{};
};

PoolRunResult CompileProcessPool::Impl::Run(const std::vector<CompileJob> &jobs) {
  if (used_) {
    throw std::logic_error("CompileProcessPool::Run may only be called once");
  }
  used_ = true;
  if (jobs.empty()) {
    return {};
  }
  if (config_.workerCount == 0 || config_.reapInterval.count() <= 0) {
    throw std::invalid_argument("invalid compile pool configuration");
  }
  result_.jobs.resize(jobs.size());
  producedBy_.assign(jobs.size(), nullptr);
  dispatchedBy_.assign(jobs.size(), nullptr);
  for (size_t index = 0; index < jobs.size(); ++index) {
    if (jobs[index].jobId != index) {
      throw std::invalid_argument("compile jobId must equal its input index");
    }
    result_.jobs[index].jobId = index;
    result_.jobs[index].detail = "no worker available before task dispatch";
  }
  std::string error;
  if (!StartWorkers(std::min<uint64_t>(config_.workerCount, jobs.size()), error)) {
    for (auto &job : result_.jobs) {
      job.detail = error;
    }
    const auto cleanup = CloseAndReap();
    result_.preserveOutputDirectory = cleanup.preserveOutputDirectory;
    if (!dispatched_ && !cleanup.preserveOutputDirectory) {
      result_.disposition = PoolDisposition::RetrySerialBeforeDispatch;
      result_.jobs.clear();
    }
    return std::move(result_);
  }
  size_t next = 0;
  for (;;) {
    Dispatch(jobs, next);
    const bool pending = std::any_of(workers_.begin(), workers_.end(),
                                     [](const auto &worker) { return worker->currentJob.has_value(); });
    if (next == jobs.size() && !pending) {
      break;
    }
    if (!HasActiveWorkers()) {
      break;
    }
    PollWorkers();
    AdvanceWorkers();
  }
  const auto cleanup = CloseAndReap();
  result_.preserveOutputDirectory = cleanup.preserveOutputDirectory;
  VerifyProduced(jobs);
  AppendWorkerDiagnostics();
  return std::move(result_);
}

bool CompileProcessPool::Impl::StartWorkers(size_t count, std::string &error) {
  startupDeadline_ = Clock::now() + config_.startupTimeout;
  workers_.reserve(count);
  for (size_t index = 0; index < count; ++index) {
    workers_.push_back(std::make_unique<WorkerSlot>());
    if (!SpawnWorker(*workers_.back(), index, error)) {
      workers_.back()->state = WorkerState::Reaped;
      return false;
    }
  }
  while (!startupFailed_) {
    const bool ready = std::all_of(workers_.begin(), workers_.end(),
                                   [](const auto &worker) { return worker->state == WorkerState::Idle; });
    if (ready) {
      return true;
    }
    if (Clock::now() >= startupDeadline_) {
      error = "worker READY deadline expired";
      return false;
    }
    PollWorkers();
    AdvanceWorkers();
  }
  error = "worker failed before all READY messages were accepted";
  return false;
}

bool CompileProcessPool::Impl::SpawnWorker(WorkerSlot &worker, size_t index, std::string &error) {
  int sockets[2] = {-1, -1};
  if (socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0, sockets) != 0) {
    error = SystemError("socketpair", errno);
    return false;
  }
  OwnedFd parent{sockets[0]};
  OwnedFd childOriginal{sockets[1]};
  OwnedFd child{fcntl(childOriginal.fd, F_DUPFD_CLOEXEC, 5)};
  if (child.fd < 0) {
    error = SystemError("duplicate worker socket", errno);
    return false;
  }
  const std::string logPath = config_.outputDirectory + "/worker_" + std::to_string(index) + ".log";
  OwnedFd log{open(logPath.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600)};
  if (log.fd < 0) {
    error = SystemError("open worker log", errno);
    return false;
  }
  auto channel = std::make_unique<FramedChannel>(parent.fd);
  (void)parent.Release();
  posix_spawn_file_actions_t actions;
  posix_spawnattr_t attributes;
  int code = posix_spawn_file_actions_init(&actions);
  if (code != 0) {
    error = SystemError("spawn file actions init", code);
    return false;
  }
  code = posix_spawnattr_init(&attributes);
  if (code != 0) {
    error = SystemError("spawn attributes init", code);
    posix_spawn_file_actions_destroy(&actions);
    return false;
  }
  auto add = [&code](int nextCode) {
    if (code == 0) {
      code = nextCode;
    }
  };
  add(posix_spawn_file_actions_addclose(&actions, channel->Fd()));
  add(posix_spawn_file_actions_adddup2(&actions, log.fd, STDOUT_FILENO));
  add(posix_spawn_file_actions_adddup2(&actions, log.fd, STDERR_FILENO));
  add(posix_spawn_file_actions_adddup2(&actions, child.fd, 3));
  add(posix_spawn_file_actions_addclose(&actions, child.fd));
  if (log.fd != 3 && log.fd != STDOUT_FILENO && log.fd != STDERR_FILENO) {
    add(posix_spawn_file_actions_addclose(&actions, log.fd));
  }
  sigset_t mask;
  sigemptyset(&mask);
  add(posix_spawnattr_setsigmask(&attributes, &mask));
  sigset_t defaults;
  sigemptyset(&defaults);
  for (const int signal : {SIGTERM, SIGINT, SIGPIPE, SIGCHLD, SIGHUP}) {
    sigaddset(&defaults, signal);
  }
  add(posix_spawnattr_setsigdefault(&attributes, &defaults));
  add(posix_spawnattr_setpgroup(&attributes, 0));
  add(posix_spawnattr_setflags(&attributes, POSIX_SPAWN_SETPGROUP | POSIX_SPAWN_SETSIGMASK | POSIX_SPAWN_SETSIGDEF));
  char controlOption[] = "--control-fd";
  char controlFd[] = "3";
  char *argv[] = {const_cast<char *>(config_.workerPath.c_str()), controlOption, controlFd, nullptr};
  pid_t childPid = -1;
  if (code == 0) {
    code = posix_spawn(&childPid, config_.workerPath.c_str(), &actions, &attributes, argv, environ);
  }
  posix_spawnattr_destroy(&attributes);
  posix_spawn_file_actions_destroy(&actions);
  if (code != 0) {
    error = SystemError("posix_spawn", code);
    return false;
  }
  worker.pid = childPid;
  worker.state = WorkerState::Starting;
  worker.channel = std::move(channel);
  worker.deadline = startupDeadline_;
  return true;
}

void CompileProcessPool::Impl::Dispatch(const std::vector<CompileJob> &jobs, size_t &next) {
  for (auto &slot : workers_) {
    auto &worker = *slot;
    if (worker.state != WorkerState::Idle || next == jobs.size()) {
      continue;
    }
    const size_t index = next++;
    worker.currentJob = index;
    dispatchedBy_[index] = &worker;
    worker.deadline = Clock::now() + config_.taskTimeout;
    Message message;
    message.kind = MessageKind::JOB;
    message.jobId = jobs[index].jobId;
    message.job = jobs[index];
    message.taskDeadlineNs =
        std::chrono::duration_cast<std::chrono::nanoseconds>(worker.deadline.time_since_epoch()).count();
    std::string error;
    if (!worker.channel->Queue(message, error)) {
      FailWorker(worker, CompileOutcome::ExecutionFailed, error);
      continue;
    }
    worker.state = WorkerState::Sending;
    // Once sending begins the peer's execution state is uncertain; never fall back to serial.
    dispatched_ = true;
  }
}

void CompileProcessPool::Impl::PollWorkers() {
  std::vector<pollfd> descriptors;
  std::vector<WorkerSlot *> owners;
  for (auto &worker : workers_) {
    if (worker->channel == nullptr || worker->channel->Fd() < 0) {
      continue;
    }
    short events = POLLIN;
    if (worker->channel->WantsWrite()) {
      events |= POLLOUT;
    }
    descriptors.push_back({worker->channel->Fd(), events, 0});
    owners.push_back(worker.get());
  }
  const int count = poll(descriptors.data(), descriptors.size(), PollTimeout());
  if (count < 0) {
    if (errno == EINTR) {
      return;
    }
    throw std::runtime_error(SystemError("poll", errno));
  }
  for (size_t index = 0; index < descriptors.size(); ++index) {
    auto &worker = *owners[index];
    const short events = descriptors[index].revents;
    if (events == 0 || worker.channel->Fd() < 0) {
      continue;
    }
    std::string error;
    if ((events & (POLLIN | POLLHUP | POLLERR | POLLNVAL)) != 0) {
      std::vector<Message> messages;
      const auto status = worker.channel->PumpRead(messages, error);
      for (const auto &message : messages) {
        HandleMessage(worker, message);
        if (worker.state == WorkerState::Stopping || worker.state == WorkerState::Reaped) {
          break;
        }
      }
      if (status == IoStatus::Error || status == IoStatus::PeerClosed || (events & POLLNVAL) != 0) {
        if (IsActive(worker.state)) {
          FailWorker(worker, CompileOutcome::ExecutionFailed, error.empty() ? "worker IPC closed" : error);
        } else {
          worker.channel->Close();
        }
      }
    }
    if ((events & POLLOUT) != 0 && worker.channel->Fd() >= 0) {
      const auto status = worker.channel->PumpWrite(error);
      if (status == IoStatus::Error || status == IoStatus::PeerClosed) {
        if (IsActive(worker.state)) {
          FailWorker(worker, CompileOutcome::ExecutionFailed, error);
        } else {
          BeginStopping(worker, false);
        }
      } else if (worker.state == WorkerState::Sending && !worker.channel->WantsWrite()) {
        worker.state = WorkerState::Busy;
      }
    }
  }
}

void CompileProcessPool::Impl::HandleMessage(WorkerSlot &worker, const Message &message) {
  if (worker.state == WorkerState::Starting) {
    if (message.kind != MessageKind::READY || message.jobId != NO_JOB_ID || message.buildId != WORKER_BUILD_ID ||
        Clock::now() >= startupDeadline_) {
      startupFailed_ = true;
      FailWorker(worker, CompileOutcome::ExecutionFailed, "invalid or late worker READY");
      return;
    }
    worker.state = WorkerState::Idle;
    return;
  }
  if (worker.state != WorkerState::Busy || !worker.currentJob || message.jobId != *worker.currentJob ||
      (message.kind != MessageKind::RESULT && message.kind != MessageKind::JOB_ERROR)) {
    FailWorker(worker, CompileOutcome::ExecutionFailed, "worker message does not match active job");
    return;
  }
  if (Clock::now() >= worker.deadline) {
    FailWorker(worker, CompileOutcome::TimedOut, "worker RESULT arrived after task deadline");
    return;
  }
  if (message.kind == MessageKind::JOB_ERROR) {
    FailWorker(worker, CompileOutcome::ExecutionFailed, message.reply.detail);
    return;
  }
  if ((message.reply.kind == WorkerResultKind::Produced && message.reply.aclrtcCode != 0) ||
      (message.reply.kind == WorkerResultKind::CompileFailed && message.reply.aclrtcCode == 0)) {
    FailWorker(worker, CompileOutcome::ExecutionFailed, "worker result kind contradicts ACLRTC return code");
    return;
  }
  const size_t index = *worker.currentJob;
  auto &result = result_.jobs[index];
  result.aclrtcCode = message.reply.aclrtcCode;
  result.detail = message.reply.detail;
  result.artifactSize = message.reply.artifactSize;
  result.artifactSha256 = message.reply.artifactSha256;
  if (message.reply.kind == WorkerResultKind::Produced) {
    producedBy_[index] = &worker;
  } else {
    result.outcome = message.reply.kind == WorkerResultKind::CompileFailed ? CompileOutcome::CompileFailed
                                                                           : CompileOutcome::ArtifactFailed;
  }
  worker.currentJob.reset();
  worker.state = WorkerState::Idle;
}

void CompileProcessPool::Impl::FailWorker(WorkerSlot &worker, CompileOutcome outcome, const std::string &detail) {
  if (worker.state == WorkerState::Starting) {
    startupFailed_ = true;
  }
  if (worker.currentJob) {
    auto &result = result_.jobs[*worker.currentJob];
    result.outcome = outcome;
    result.detail = detail;
    worker.currentJob.reset();
  }
  BeginStopping(worker, false);
}

void CompileProcessPool::Impl::BeginStopping(WorkerSlot &worker, bool graceful) {
  if (worker.state == WorkerState::Reaped) {
    return;
  }
  if (worker.state != WorkerState::Stopping) {
    worker.state = WorkerState::Stopping;
    worker.deadline = Clock::now() + config_.shutdownTimeout;
  }
  if (graceful && worker.channel != nullptr && worker.channel->Fd() >= 0 && !worker.termSent) {
    Message shutdown;
    shutdown.kind = MessageKind::SHUTDOWN;
    std::string error;
    if (worker.channel->Queue(shutdown, error)) {
      return;
    }
  }
  if (!worker.termSent) {
    if (worker.channel != nullptr) {
      worker.channel->Close();
    }
    SignalGroup(worker, SIGTERM);
    worker.termSent = true;
    worker.killAt = Clock::now() + config_.termGrace;
  }
}

void CompileProcessPool::Impl::SignalGroup(WorkerSlot &worker, int signal) {
  if (worker.pid > 0 && kill(-worker.pid, signal) != 0 && errno != ESRCH) {
    worker.groupUnknown = true;
  }
}

void CompileProcessPool::Impl::AdvanceWorkers() {
  const auto now = Clock::now();
  for (auto &slot : workers_) {
    auto &worker = *slot;
    if (worker.state == WorkerState::Reaped || worker.pid <= 0) {
      continue;
    }
    if ((worker.state == WorkerState::Sending || worker.state == WorkerState::Busy) && now >= worker.deadline) {
      FailWorker(worker, CompileOutcome::TimedOut, "compile task deadline expired");
    }
    if (!worker.leaderReaped) {
      const pid_t waited = waitpid(worker.pid, &worker.exitStatus, WNOHANG);
      if (waited == worker.pid || (waited < 0 && errno == ECHILD)) {
        worker.leaderReaped = true;
        worker.exitStatusKnown = waited == worker.pid;
        if (waited < 0) {
          worker.groupUnknown = true;
        }
        if (IsActive(worker.state)) {
          std::string detail = "worker exited";
          if (waited > 0 && WIFSIGNALED(worker.exitStatus)) {
            detail += " from signal " + std::to_string(WTERMSIG(worker.exitStatus));
          } else if (waited > 0 && WIFEXITED(worker.exitStatus)) {
            detail += " with code " + std::to_string(WEXITSTATUS(worker.exitStatus));
          }
          FailWorker(worker, CompileOutcome::ExecutionFailed, detail);
        }
      } else if (waited < 0 && errno != EINTR) {
        worker.groupUnknown = true;
      }
    }
    if (worker.state != WorkerState::Stopping) {
      continue;
    }
    if (!worker.termSent && now >= worker.deadline) {
      BeginStopping(worker, false);
    }
    if (worker.leaderReaped) {
      if (kill(-worker.pid, 0) < 0 && errno == ESRCH) {
        worker.state = WorkerState::Reaped;
        if (worker.channel != nullptr) {
          worker.channel->Close();
        }
        continue;
      }
      if (!worker.termSent) {
        BeginStopping(worker, false);
      }
    }
    if (worker.termSent && !worker.killSent && now >= worker.killAt) {
      SignalGroup(worker, SIGKILL);
      worker.killSent = true;
      worker.groupCheckUntil = now + config_.groupCheckTimeout;
    }
    if (worker.leaderReaped && worker.killSent && now >= worker.groupCheckUntil) {
      worker.groupUnknown = true;
      worker.state = WorkerState::Reaped;
      if (worker.channel != nullptr) {
        worker.channel->Close();
      }
    }
  }
}

bool CompileProcessPool::Impl::HasActiveWorkers() const {
  return std::any_of(workers_.begin(), workers_.end(), [](const auto &worker) { return IsActive(worker->state); });
}

bool CompileProcessPool::Impl::AllReaped() const {
  return std::all_of(workers_.begin(), workers_.end(),
                     [](const auto &worker) { return worker->state == WorkerState::Reaped; });
}

int CompileProcessPool::Impl::PollTimeout() const {
  const auto now = Clock::now();
  auto until = now + config_.reapInterval;
  for (const auto &worker : workers_) {
    if (worker->state == WorkerState::Starting || worker->state == WorkerState::Sending ||
        worker->state == WorkerState::Busy || (worker->state == WorkerState::Stopping && !worker->termSent)) {
      until = std::min(until, worker->deadline);
    }
    if (worker->state == WorkerState::Stopping && worker->termSent) {
      if (!worker->killSent) {
        until = std::min(until, worker->killAt);
      } else if (worker->leaderReaped) {
        until = std::min(until, worker->groupCheckUntil);
      }
    }
  }
  const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(until - now).count();
  return static_cast<int>(std::max<int64_t>(0, std::min<int64_t>(remaining, INT_MAX)));
}

CleanupReport CompileProcessPool::Impl::CloseAndReap() {
  for (auto &worker : workers_) {
    if (worker->state == WorkerState::Idle) {
      BeginStopping(*worker, true);
    } else if (IsActive(worker->state)) {
      FailWorker(*worker, CompileOutcome::ExecutionFailed, "compile pool stopped before task completion");
    }
  }
  while (!AllReaped()) {
    AdvanceWorkers();
    if (!AllReaped()) {
      PollWorkers();
    }
  }
  CleanupReport report;
  for (const auto &worker : workers_) {
    if (worker->groupUnknown) {
      report.preserveOutputDirectory = true;
      report.unconfirmedWorkers.push_back(worker->pid);
    }
  }
  return report;
}

void CompileProcessPool::Impl::VerifyProduced(const std::vector<CompileJob> &jobs) {
  for (size_t index = 0; index < jobs.size(); ++index) {
    const auto *worker = producedBy_[index];
    if (worker == nullptr) {
      continue;
    }
    auto &result = result_.jobs[index];
    if (worker->groupUnknown) {
      result.outcome = CompileOutcome::ArtifactFailed;
      result.detail = "worker process group cleanup could not be confirmed";
      continue;
    }
    ArtifactInfo artifact;
    std::string error;
    if (!InspectArtifact(jobs[index].outputElfPath, artifact, error) || artifact.size != result.artifactSize ||
        artifact.sha256 != result.artifactSha256) {
      result.outcome = CompileOutcome::ArtifactFailed;
      result.detail = error.empty() ? "worker artifact size or SHA-256 mismatch" : error;
      continue;
    }
    result.outcome = CompileOutcome::Success;
    result.detail.clear();
  }
}

void CompileProcessPool::Impl::AppendWorkerDiagnostics() {
  for (size_t index = 0; index < result_.jobs.size(); ++index) {
    auto &result = result_.jobs[index];
    const auto *worker = dispatchedBy_[index];
    if (worker == nullptr || result.outcome == CompileOutcome::Success) {
      continue;
    }
    result.detail += "; worker PID=" + std::to_string(worker->pid);
    if (worker->exitStatusKnown && WIFEXITED(worker->exitStatus)) {
      result.detail += " exit code " + std::to_string(WEXITSTATUS(worker->exitStatus));
    } else if (worker->exitStatusKnown && WIFSIGNALED(worker->exitStatus)) {
      result.detail += " signal " + std::to_string(WTERMSIG(worker->exitStatus));
    } else {
      result.detail += " exit status unavailable";
    }
  }
}

CompileProcessPool::CompileProcessPool(PoolConfig config) : impl_(std::make_unique<Impl>(std::move(config))) {}

CompileProcessPool::~CompileProcessPool() {
  try {
    (void)impl_->CloseAndReap();
  } catch (...) {
    // WorkerSlot owns a final non-throwing kill/reap fallback during stack unwinding.
  }
}

PoolRunResult CompileProcessPool::Run(const std::vector<CompileJob> &jobs) {
  return impl_->Run(jobs);
}

CleanupReport CompileProcessPool::CloseAndReap() {
  return impl_->CloseAndReap();
}

}  // namespace model_spec::compile
