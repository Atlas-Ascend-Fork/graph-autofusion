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
#include <cinttypes>
#include <climits>
#include <csignal>
#include <cstring>
#include <fcntl.h>
#include <poll.h>
#include <spawn.h>
#include <stdexcept>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <utility>

#include "sk_log.h"
namespace model_spec::compile {
namespace {
using Clock = std::chrono::steady_clock;
using TimePoint = Clock::time_point;
constexpr int WORKER_CONTROL_FD = 3;
constexpr int MIN_DUPLICATED_SOCKET_FD = 5;

// 以下均为父进程记录的 worker 调度状态，不代表子进程内部的实时执行状态。
enum class WorkerState {
  Starting,  // 子进程已启动，等待接收并校验 READY 消息。
  Idle,      // 已就绪且没有绑定任务，可以派发下一份任务。
  Working,   // 已绑定任务，涵盖发送 JOB 和等待结果；发送进度由 channel->WantsWrite() 表示。
  Stopping,  // 停止派发；故障 worker 等本批结束后，再统一终止和回收。
  Reaped,    // 尚未启动，或回收流程已结束；清理是否确认完成另由 groupUnknown 记录。
};

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
      (void)killpg(pid, SIGKILL);
      pid_t waitResult;
      int waitError;
      do {
        waitResult = waitpid(pid, nullptr, 0);
        waitError = waitResult < 0 ? errno : 0;
      } while (waitResult < 0 && waitError == EINTR);
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

std::string PollErrorDetail(int error) {
  return "parent poll failed errno=" + std::to_string(error) + " reason=" + std::strerror(error);
}

bool IsActive(WorkerState state) {
  return state == WorkerState::Starting || state == WorkerState::Idle || state == WorkerState::Working;
}

bool ConfigureSpawn(posix_spawn_file_actions_t &actions, posix_spawnattr_t &attributes, int parentFd, int childFd,
                    int logFd, int &errorCode) {
  errorCode = posix_spawn_file_actions_init(&actions);
  if (errorCode != 0) {
    return false;
  }
  errorCode = posix_spawnattr_init(&attributes);
  if (errorCode != 0) {
    posix_spawn_file_actions_destroy(&actions);
    return false;
  }
  auto add = [&errorCode](int nextCode) {
    if (errorCode == 0) {
      errorCode = nextCode;
    }
  };
  // 子进程关闭父端点，将自己的控制端点固定到 fd 3，并把标准输出写入独立日志。
  add(posix_spawn_file_actions_addclose(&actions, parentFd));
  add(posix_spawn_file_actions_adddup2(&actions, logFd, STDOUT_FILENO));
  add(posix_spawn_file_actions_adddup2(&actions, logFd, STDERR_FILENO));
  add(posix_spawn_file_actions_adddup2(&actions, childFd, WORKER_CONTROL_FD));
  add(posix_spawn_file_actions_addclose(&actions, childFd));
  if (logFd != WORKER_CONTROL_FD && logFd != STDOUT_FILENO && logFd != STDERR_FILENO) {
    add(posix_spawn_file_actions_addclose(&actions, logFd));
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
  if (errorCode != 0) {
    posix_spawnattr_destroy(&attributes);
    posix_spawn_file_actions_destroy(&actions);
    return false;
  }
  return true;
}

bool SpawnWorkerProcess(const std::string &workerPath, int parentFd, int childFd, int logFd, pid_t &childPid,
                        int &errorCode) {
  char controlOption[] = "--control-fd";
  auto controlFd = std::to_string(WORKER_CONTROL_FD);
  char *argv[] = {const_cast<char *>(workerPath.c_str()), controlOption, controlFd.data(), nullptr};
  posix_spawn_file_actions_t actions;
  posix_spawnattr_t attributes;
  if (!ConfigureSpawn(actions, attributes, parentFd, childFd, logFd, errorCode)) {
    return false;
  }
  childPid = -1;
  errorCode = posix_spawn(&childPid, workerPath.c_str(), &actions, &attributes, argv, environ);
  posix_spawnattr_destroy(&attributes);
  posix_spawn_file_actions_destroy(&actions);
  return errorCode == 0;
}
}  // namespace

class CompileProcessPool::Impl {
 public:
  explicit Impl(PoolConfig config) : config_(std::move(config)) {}

  PoolRunResult Run(const std::vector<CompileJob> &jobs);
  CleanupReport CloseAndReap(bool graceful);

 private:
  bool PrepareRun(const std::vector<CompileJob> &jobs);
  bool StartWorkers(size_t count);
  bool SpawnWorker(WorkerSlot &worker, size_t index);
  void Dispatch(const std::vector<CompileJob> &jobs, size_t &next);
  int PollWorkers();
  void ReceiveWorkerMessages(WorkerSlot &worker, short events);
  void FlushWorkerMessage(WorkerSlot &worker);
  void HandleMessage(WorkerSlot &worker, const Message &message);
  void FailWorker(WorkerSlot &worker, CompileOutcome outcome, const std::string &detail);
  void BeginStopping(WorkerSlot &worker, bool graceful);
  void AdvanceWorkers();
  void CheckWorkerExit(WorkerSlot &worker);
  void TryReapWorker(WorkerSlot &worker);
  void AdvanceStoppingWorker(WorkerSlot &worker, TimePoint now);
  void SignalGroup(WorkerSlot &worker, int signal);
  void AppendWorkerDiagnostics();
  void MarkAllJobsFailed(int pollError);
  bool HasActiveWorkers() const;
  bool AllReaped() const;
  int PollTimeout() const;

  PoolConfig config_;
  std::vector<std::unique_ptr<WorkerSlot>> workers_;
  PoolRunResult result_;
  std::vector<WorkerSlot *> dispatchedBy_;
  bool used_ = false;
  bool closing_ = false;
  TimePoint startupDeadline_{};
};

PoolRunResult CompileProcessPool::Impl::Run(const std::vector<CompileJob> &jobs) {
  if (!PrepareRun(jobs)) {
    return {};
  }
  bool graceful = StartWorkers(std::min<uint64_t>(config_.workerCount, jobs.size()));
  // next 指向下一份未分配的任务；已分配但尚无结果的任务由 currentJob 跟踪。
  size_t next = 0;
  while (graceful) {
    Dispatch(jobs, next);
    const bool pending = std::any_of(workers_.begin(), workers_.end(),
                                     [](const auto &worker) { return worker->currentJob.has_value(); });
    if ((next == jobs.size() && !pending) || !HasActiveWorkers()) {
      break;
    }
    if (PollWorkers() != 0) {
      graceful = false;
      break;
    }
    AdvanceWorkers();
  }
  const auto cleanup = CloseAndReap(graceful);
  result_.preserveOutputDirectory = cleanup.preserveOutputDirectory;
  AppendWorkerDiagnostics();
  return std::move(result_);
}

bool CompileProcessPool::Impl::PrepareRun(const std::vector<CompileJob> &jobs) {
  if (used_) {
    throw std::logic_error("CompileProcessPool::Run may only be called once");
  }
  used_ = true;
  if (jobs.empty()) {
    return false;
  }
  if (config_.workerCount == 0 || config_.reapInterval.count() <= 0) {
    throw std::invalid_argument("invalid compile pool configuration");
  }
  result_.jobs.resize(jobs.size());
  dispatchedBy_.assign(jobs.size(), nullptr);
  for (size_t index = 0; index < jobs.size(); ++index) {
    if (jobs[index].jobId != index) {
      throw std::invalid_argument("compile jobId must equal its input index");
    }
    result_.jobs[index].jobId = index;
    result_.jobs[index].detail = "no worker available before task dispatch";
  }
  return true;
}

bool CompileProcessPool::Impl::StartWorkers(size_t count) {
  startupDeadline_ = Clock::now() + config_.startupTimeout;
  workers_.reserve(count);
  for (size_t index = 0; index < count; ++index) {
    workers_.push_back(std::make_unique<WorkerSlot>());
    if (!SpawnWorker(*workers_.back(), index)) {
      return false;
    }
  }
  bool ready = false;
  while (!ready) {
    if (std::any_of(workers_.begin(), workers_.end(), [](const auto &worker) {
          return worker->state != WorkerState::Starting && worker->state != WorkerState::Idle;
        })) {
      SK_DLOGW("[SpecCompile] Compile pool startup failed: worker failed before all READY messages were accepted");
      return false;
    }
    ready = std::all_of(workers_.begin(), workers_.end(),
                        [](const auto &worker) { return worker->state == WorkerState::Idle; });
    if (!ready) {
      if (Clock::now() >= startupDeadline_) {
        SK_DLOGW("[SpecCompile] Compile pool startup failed: worker READY deadline expired");
        return false;
      }
      if (PollWorkers() != 0) {
        return false;
      }
      AdvanceWorkers();
    }
  }
  SK_DLOGI("[SpecCompile] All compile workers ready: count=%zu", workers_.size());
  return true;
}

bool CompileProcessPool::Impl::SpawnWorker(WorkerSlot &worker, size_t index) {
  int sockets[2] = {-1, -1};
  if (socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0, sockets) != 0) {
    const auto detail = SystemError("socketpair", errno);
    SK_DLOGW("[SpecCompile] Compile worker startup failed: workerIndex=%zu reason=%s", index, detail.c_str());
    return false;
  }
  OwnedFd parent{sockets[0]};
  OwnedFd childOriginal{sockets[1]};
  // 复制到较高的描述符，避免子进程重定向日志和控制端点时覆盖原端点。
  OwnedFd child{fcntl(childOriginal.fd, F_DUPFD_CLOEXEC, MIN_DUPLICATED_SOCKET_FD)};
  if (child.fd < 0) {
    const auto detail = SystemError("duplicate worker socket", errno);
    SK_DLOGW("[SpecCompile] Compile worker startup failed: workerIndex=%zu reason=%s", index, detail.c_str());
    return false;
  }
  const std::string logPath = config_.outputDirectory + "/worker_" + std::to_string(index) + ".log";
  OwnedFd log{open(logPath.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, S_IRUSR | S_IWUSR)};
  if (log.fd < 0) {
    const auto detail = SystemError("open worker log", errno);
    SK_DLOGW("[SpecCompile] Compile worker startup failed: workerIndex=%zu reason=%s", index, detail.c_str());
    return false;
  }
  auto channel = std::make_unique<FramedChannel>(parent.fd);
  (void)parent.Release();
  int code = 0;
  pid_t childPid = -1;
  if (!SpawnWorkerProcess(config_.workerPath, channel->Fd(), child.fd, log.fd, childPid, code)) {
    const auto detail = SystemError("posix_spawn", code);
    SK_DLOGW("[SpecCompile] Compile worker startup failed: workerIndex=%zu reason=%s", index, detail.c_str());
    return false;
  }
  worker.pid = childPid;
  worker.state = WorkerState::Starting;
  worker.channel = std::move(channel);
  worker.deadline = startupDeadline_;
  SK_DLOGI("[SpecCompile] Compile worker spawned: workerIndex=%zu workerPid=%d logPath=%s", index,
           static_cast<int>(worker.pid), logPath.c_str());
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
    // 从任务分配时开始计时，发送请求和等待结果都计入超时。
    worker.deadline = Clock::now() + config_.taskTimeout;
    Message message;
    message.kind = MessageKind::JOB;
    message.jobId = jobs[index].jobId;
    message.job = jobs[index];
    std::string error;
    if (!worker.channel->Queue(message, error)) {
      FailWorker(worker, CompileOutcome::ExecutionFailed, error);
      continue;
    }
    worker.state = WorkerState::Working;
    SK_DLOGI("[SpecCompile] Queued JOB for worker: workerPid=%d jobId=%" PRIu64 " kernel=%s timeoutMs=%" PRId64,
             static_cast<int>(worker.pid), message.jobId, message.job.kernelEntry.c_str(),
             static_cast<int64_t>(config_.taskTimeout.count()));
  }
}

int CompileProcessPool::Impl::PollWorkers() {
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
      return 0;
    }
    const int error = errno;
    MarkAllJobsFailed(error);
    return error;
  }
  for (size_t index = 0; index < descriptors.size(); ++index) {
    auto &worker = *owners[index];
    const short events = descriptors[index].revents;
    if (events == 0 || worker.channel->Fd() < 0) {
      continue;
    }
    if ((events & (POLLIN | POLLHUP | POLLERR | POLLNVAL)) != 0) {
      ReceiveWorkerMessages(worker, events);
    }
    if ((events & POLLOUT) != 0 && worker.channel->Fd() >= 0) {
      FlushWorkerMessage(worker);
    }
  }
  return 0;
}

void CompileProcessPool::Impl::ReceiveWorkerMessages(WorkerSlot &worker, short events) {
  std::string error;
  // 对端关闭时也可能有最后一条结果，先处理已收到的完整消息。
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

void CompileProcessPool::Impl::FlushWorkerMessage(WorkerSlot &worker) {
  std::string error;
  const bool hadPendingWrite = worker.channel->WantsWrite();
  const auto status = worker.channel->PumpWrite(error);
  if (status == IoStatus::Error || status == IoStatus::PeerClosed) {
    if (IsActive(worker.state)) {
      FailWorker(worker, CompileOutcome::ExecutionFailed, error);
    } else {
      BeginStopping(worker, false);
    }
  } else if (worker.state == WorkerState::Working && hadPendingWrite && !worker.channel->WantsWrite()) {
    SK_DLOGI("[SpecCompile] Sent JOB to worker: workerPid=%d jobId=%zu", static_cast<int>(worker.pid),
             *worker.currentJob);
  }
}

void CompileProcessPool::Impl::HandleMessage(WorkerSlot &worker, const Message &message) {
  if (worker.state == WorkerState::Starting) {
    if (message.kind != MessageKind::READY || message.jobId != NO_JOB_ID || message.buildId != WORKER_BUILD_ID ||
        Clock::now() >= startupDeadline_) {
      FailWorker(worker, CompileOutcome::ExecutionFailed, "invalid or late worker READY");
      return;
    }
    worker.state = WorkerState::Idle;
    SK_DLOGI("[SpecCompile] Received READY from worker: workerPid=%d", static_cast<int>(worker.pid));
    return;
  }
  if (worker.state != WorkerState::Working || worker.channel->WantsWrite() || !worker.currentJob ||
      message.jobId != *worker.currentJob ||
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
  const size_t index = *worker.currentJob;
  auto &result = result_.jobs[index];
  result.aclrtcCode = message.reply.aclrtcCode;
  result.detail = message.reply.detail;
  result.outcome = message.reply.aclrtcCode == 0 ? CompileOutcome::Success : CompileOutcome::CompileFailed;
  const char *outcome = result.outcome == CompileOutcome::Success ? "Success" : "CompileFailed";
  SK_DLOGI("[SpecCompile] Received RESULT from worker: workerPid=%d jobId=%" PRIu64 " outcome=%s aclrtcCode=%d",
           static_cast<int>(worker.pid), message.jobId, outcome, message.reply.aclrtcCode);
  worker.currentJob.reset();
  worker.state = WorkerState::Idle;
}

void CompileProcessPool::Impl::FailWorker(WorkerSlot &worker, CompileOutcome outcome, const std::string &detail) {
  if (worker.currentJob) {
    auto &result = result_.jobs[*worker.currentJob];
    result.outcome = outcome;
    result.detail = detail;
    worker.currentJob.reset();
  }
  worker.state = WorkerState::Stopping;
  if (worker.channel != nullptr) {
    worker.channel->Close();
  }
  SK_DLOGW("[SpecCompile] Worker disabled; cleanup deferred until batch completion: workerPid=%d reason=%s",
           static_cast<int>(worker.pid), detail.c_str());
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
      SK_DLOGI("[SpecCompile] Queued SHUTDOWN for worker: workerPid=%d", static_cast<int>(worker.pid));
      return;
    }
    SK_DLOGW("[SpecCompile] Failed to queue SHUTDOWN for worker: workerPid=%d reason=%s", static_cast<int>(worker.pid),
             error.c_str());
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
  SK_DLOGI("[SpecCompile] Stopping worker process group: workerPid=%d signal=%d", static_cast<int>(worker.pid), signal);
  if (worker.pid > 0 && killpg(worker.pid, signal) != 0 && errno != ESRCH) {
    SK_DLOGW("[SpecCompile] Failed to signal worker process group: workerPid=%d signal=%d errno=%d reason=%s",
             static_cast<int>(worker.pid), signal, errno, std::strerror(errno));
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
    if (worker.state == WorkerState::Working && now >= worker.deadline) {
      FailWorker(worker, CompileOutcome::TimedOut, "compile task deadline expired");
    }
    if (!closing_) {
      if (IsActive(worker.state)) {
        CheckWorkerExit(worker);
      }
      continue;
    }
    if (!worker.leaderReaped) {
      TryReapWorker(worker);
    }
    if (worker.state == WorkerState::Stopping) {
      AdvanceStoppingWorker(worker, now);
    }
  }
}

void CompileProcessPool::Impl::CheckWorkerExit(WorkerSlot &worker) {
  // 运行期间只探测退出并保留 PID，避免提前回收后，批末再向已复用的 PID/PGID 发信号。
  siginfo_t status{};
  if (waitid(P_PID, worker.pid, &status, WEXITED | WNOHANG | WNOWAIT) != 0) {
    const int error = errno;
    if (error != EINTR) {
      SK_DLOGW("[SpecCompile] Failed to inspect worker exit: workerPid=%d errno=%d reason=%s",
               static_cast<int>(worker.pid), error, std::strerror(error));
      worker.groupUnknown = true;
      if (error == ECHILD) {
        FailWorker(worker, CompileOutcome::ExecutionFailed, "worker exit status unavailable");
      }
    }
    return;
  }
  if (status.si_pid == 0) {
    return;
  }
  const std::string detail = status.si_code == CLD_EXITED
                                 ? "worker exited with code " + std::to_string(status.si_status)
                                 : "worker exited from signal " + std::to_string(status.si_status);
  FailWorker(worker, CompileOutcome::ExecutionFailed, detail);
}

void CompileProcessPool::Impl::TryReapWorker(WorkerSlot &worker) {
  const pid_t waited = waitpid(worker.pid, &worker.exitStatus, WNOHANG);
  if (waited == 0) {
    return;
  }
  if (waited < 0) {
    const int error = errno;
    if (error != EINTR) {
      SK_DLOGW("[SpecCompile] Failed to reap worker: workerPid=%d errno=%d reason=%s", static_cast<int>(worker.pid),
               error, std::strerror(error));
      worker.groupUnknown = true;
    }
    if (error != ECHILD) {
      return;
    }
  }
  worker.leaderReaped = true;
  worker.exitStatusKnown = waited == worker.pid;
  if (waited < 0) {
    worker.groupUnknown = true;
  }
  if (!IsActive(worker.state)) {
    return;
  }
  std::string detail = "worker exited";
  if (waited > 0 && WIFSIGNALED(worker.exitStatus)) {
    detail += " from signal " + std::to_string(WTERMSIG(worker.exitStatus));
  } else if (waited > 0 && WIFEXITED(worker.exitStatus)) {
    detail += " with code " + std::to_string(WEXITSTATUS(worker.exitStatus));
  }
  FailWorker(worker, CompileOutcome::ExecutionFailed, detail);
}

void CompileProcessPool::Impl::AdvanceStoppingWorker(WorkerSlot &worker, TimePoint now) {
  if (!worker.termSent && now >= worker.deadline) {
    BeginStopping(worker, false);
  }
  if (worker.leaderReaped) {
    // waitpid 只回收 worker 本身，还要确认它的进程组里没有残留的编译进程。
    if (killpg(worker.pid, 0) < 0 && errno == ESRCH) {
      worker.state = WorkerState::Reaped;
      SK_DLOGI("[SpecCompile] Worker process group exited: workerPid=%d exitCode=%d exitSignal=%d",
               static_cast<int>(worker.pid),
               worker.exitStatusKnown && WIFEXITED(worker.exitStatus) ? WEXITSTATUS(worker.exitStatus) : -1,
               worker.exitStatusKnown && WIFSIGNALED(worker.exitStatus) ? WTERMSIG(worker.exitStatus) : 0);
      if (worker.channel != nullptr) {
        worker.channel->Close();
      }
      return;
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
    SK_DLOGW("[SpecCompile] Worker cleanup could not be confirmed: workerPid=%d", static_cast<int>(worker.pid));
    if (worker.channel != nullptr) {
      worker.channel->Close();
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
  // 即使 socket 没有事件，也要定期唤醒以检查子进程退出，并及时处理到期动作。
  auto until = now + config_.reapInterval;
  for (const auto &worker : workers_) {
    if (worker->state == WorkerState::Starting || worker->state == WorkerState::Working ||
        (closing_ && worker->state == WorkerState::Stopping && !worker->termSent)) {
      until = std::min(until, worker->deadline);
    }
    if (closing_ && worker->state == WorkerState::Stopping && worker->termSent) {
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

CleanupReport CompileProcessPool::Impl::CloseAndReap(bool graceful) {
  const bool needsCleanup = !AllReaped();
  closing_ = true;
  if (needsCleanup) {
    SK_DLOGI("[SpecCompile] Closing compile pool: workers=%zu graceful=%d", workers_.size(), graceful ? 1 : 0);
  }
  bool pollForShutdown = graceful;
  for (auto &worker : workers_) {
    if (!graceful || worker->state == WorkerState::Stopping) {
      BeginStopping(*worker, false);
    } else if (worker->state == WorkerState::Idle) {
      BeginStopping(*worker, true);
    } else if (IsActive(worker->state)) {
      FailWorker(*worker, CompileOutcome::ExecutionFailed, "compile pool stopped before task completion");
      BeginStopping(*worker, false);
    }
  }
  while (!AllReaped()) {
    AdvanceWorkers();
    if (AllReaped()) {
      break;
    }
    if (pollForShutdown) {
      const int pollError = PollWorkers();
      if (pollError != 0) {
        pollForShutdown = false;
      }
    } else {
      // poll 故障时仍需回收进程，此处只等下一轮 waitpid，不再依赖 socket。
      std::this_thread::sleep_for(config_.reapInterval);
    }
  }
  CleanupReport report;
  for (const auto &worker : workers_) {
    if (worker->groupUnknown) {
      report.preserveOutputDirectory = true;
    }
  }
  if (needsCleanup) {
    SK_DLOGI("[SpecCompile] Compile pool closed: workers=%zu preserveOutputDirectory=%d", workers_.size(),
             report.preserveOutputDirectory ? 1 : 0);
  }
  return report;
}

void CompileProcessPool::Impl::MarkAllJobsFailed(int pollError) {
  const std::string detail = PollErrorDetail(pollError);
  SK_DLOGW("[SpecCompile] Compile pool stopped: reason=%s", detail.c_str());
  for (auto &job : result_.jobs) {
    job.outcome = CompileOutcome::ExecutionFailed;
    job.aclrtcCode.reset();
    job.detail = detail;
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
    (void)CloseAndReap();
  } catch (...) {
    // 析构时不能继续抛异常，WorkerSlot 会兜底终止并回收尚未回收的子进程。
  }
}

PoolRunResult CompileProcessPool::Run(const std::vector<CompileJob> &jobs) {
  return impl_->Run(jobs);
}

CleanupReport CompileProcessPool::CloseAndReap() {
  return impl_->CloseAndReap(false);
}

}  // namespace model_spec::compile
