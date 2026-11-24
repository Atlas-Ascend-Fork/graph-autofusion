/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/utils/graph_thread_pool.h"

#include <atomic>
#include <functional>
#include <exception>
#include <stdexcept>

#include "register/register_types.h"
#include "graph/ge_context.h"
#include "mmpa/mmpa_api.h"

namespace af {
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY GraphThreadPool::GraphThreadPool(const uint32_t size)
    : is_stoped_(false) {
  idle_thrd_num_ = (size < 1U) ? 1U : size;

  for (uint32_t i = 0U; i < idle_thrd_num_; ++i) {
    pool_.emplace_back(&ThreadFunc, this);
  }
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY GraphThreadPool::~GraphThreadPool() {
  is_stoped_.store(true);
  {
    const std::unique_lock<std::mutex> lock{m_lock_};
    cond_var_.notify_all();
  }

  for (std::thread &thd : pool_) {
    if (thd.joinable()) {
      try {
        thd.join();
      } catch (const std::exception &e) {
        GELOGW("Catch exception when processing task in thread pool, reason: %s", e.what());
      } catch (...) {
        GELOGW("Catch unknown exception when processing task in thread pool");
      }
    }
  }
}

void GraphThreadPool::ThreadFunc(GraphThreadPool *const thread_pool) {
  if (thread_pool == nullptr) {
    return;
  }
  GELOGI("Thread started successfully");
  while (!thread_pool->is_stoped_) {
    std::function<void()> task;
    {
      std::unique_lock<std::mutex> lock{thread_pool->m_lock_};
      thread_pool->cond_var_.wait(
          lock, [thread_pool]() -> bool { return thread_pool->is_stoped_.load() || (!thread_pool->tasks_.empty()); });
      if (thread_pool->is_stoped_ && thread_pool->tasks_.empty()) {
        return;
      }
      task = std::move(thread_pool->tasks_.front());
      thread_pool->tasks_.pop();
    }
    --thread_pool->idle_thrd_num_;
    task();
    ++thread_pool->idle_thrd_num_;
  }
}
}  // namespace af
