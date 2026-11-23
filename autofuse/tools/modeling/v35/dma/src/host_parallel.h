/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once
#include <condition_variable>
#include <array>
#include <algorithm>
#include <cstring>
#include <cstdint>
#include <exception>
#include <functional>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
constexpr uint32_t kMinHostThreads = 1U;
constexpr uint32_t kMaxHostThreads = 8U;
constexpr uint64_t kPatternPeriod = 251U;
constexpr uint64_t kSegmentBytes = 4096U;
constexpr uint32_t kPatternMultiplier = 17U;
}  // namespace

// Host only. Run returns after every task finishes; no worker runs while a GPU sample starts.
class HostJobs {
 public:
  explicit HostJobs(uint32_t count) : count_(count) {
    if (count < kMinHostThreads || count > kMaxHostThreads) {
      throw std::runtime_error("host thread count must be1..8");
    }
    if (count == 1) {
      return;
    }
    try {
      for (uint32_t id = 0; id < count_; ++id) {
        workers_.emplace_back([this, id] { Worker(id); });
      }
    } catch (...) {
      Stop();
      throw;
    }
  }
  ~HostJobs() {
    Stop();
  }
  HostJobs(const HostJobs &) = delete;
  HostJobs &operator=(const HostJobs &) = delete;
  uint32_t Count() const {
    return count_;
  }
  void Run(uint64_t jobs, std::function<void(uint64_t)> fn) {
    if (count_ == 1) {
      for (uint64_t k = 0; k < jobs; ++k) {
        fn(k);
      }
      return;
    }
    std::unique_lock<std::mutex> guard(mutex_);
    jobs_ = jobs;
    fn_ = std::move(fn);
    finished_ = 0;
    firstError_ = nullptr;
    firstErrorJob_ = std::numeric_limits<uint64_t>::max();
    ++generation_;
    ready_.notify_all();
    done_.wait(guard, [this] { return finished_ == count_; });
    auto error = firstError_;
    fn_ = {};
    guard.unlock();
    if (error) {
      std::rethrow_exception(error);
    }
  }

 private:
  bool RunGeneration(uint32_t id, uint64_t &generation) {
    std::unique_lock<std::mutex> guard(mutex_);
    ready_.wait(guard, [&] { return stopping_ || generation != generation_; });
    if (stopping_) {
      return false;
    }
    generation = generation_;
    const auto n = jobs_;
    auto fn = fn_;
    guard.unlock();
    std::exception_ptr error;
    uint64_t failing = std::numeric_limits<uint64_t>::max();
    for (uint64_t k = id; k < n; k += count_) {
      try {
        fn(k);
      } catch (...) {
        error = std::current_exception();
        failing = k;
        break;
      }
    }
    guard.lock();
    if (error && failing < firstErrorJob_) {
      firstError_ = error;
      firstErrorJob_ = failing;
    }
    ++finished_;
    if (finished_ == count_) {
      done_.notify_one();
    }
    return true;
  }
  void Worker(uint32_t id) {
    uint64_t generation = 0;
    while (RunGeneration(id, generation)) {
    }
  }
  void Stop() {
    {
      std::lock_guard<std::mutex> guard(mutex_);
      stopping_ = true;
      ready_.notify_all();
    }
    for (auto &worker : workers_) {
      if (worker.joinable()) {
        worker.join();
      }
    }
  }
  uint32_t count_, finished_ = 0;
  uint64_t generation_ = 0, jobs_ = 0, firstErrorJob_ = 0;
  bool stopping_ = false;
  std::mutex mutex_;
  std::condition_variable ready_, done_;
  std::function<void(uint64_t)> fn_;
  std::exception_ptr firstError_;
  std::vector<std::thread> workers_;
};

// Exact for the caller's <=24GiB input window: the uint64 Value polynomial never overflows.
// Within a4KiB interval both shifted terms are constant; byte values repeat every251 steps.
// Recompute the scalar seed at every4KiB boundary, also covering every16MiB shifted-term change.
inline void FillExactValue(uint8_t *out, uint64_t n, uint64_t base, uint8_t (*scalar)(uint64_t)) {
  static const auto patterns = [] {
    std::array<std::array<uint8_t, kPatternPeriod>, kPatternPeriod> result{};
    for (uint32_t seed = 0; seed < kPatternPeriod; ++seed) {
      for (uint32_t k = 0; k < kPatternPeriod; ++k) {
        result[seed][k] = static_cast<uint8_t>((seed + kPatternMultiplier * k) % kPatternPeriod);
      }
    }
    return result;
  }();
  while (n) {
    const uint64_t segment = std::min<uint64_t>(n, kSegmentBytes - base % kSegmentBytes);
    const auto &pattern = patterns[scalar(base)];
    for (uint64_t offset = 0; offset < segment; offset += kPatternPeriod) {
      const uint64_t copy_size = std::min<uint64_t>(kPatternPeriod, segment - offset);
      std::copy_n(pattern.data(), copy_size, out + offset);
    }
    out += segment;
    base += segment;
    n -= segment;
  }
}
