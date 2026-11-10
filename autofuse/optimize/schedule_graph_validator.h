/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef OPTIMIZE_SCHEDULE_GRAPH_VALIDATOR_H_
#define OPTIMIZE_SCHEDULE_GRAPH_VALIDATOR_H_

#include <string>

#include "ascir.h"

namespace optimize {

// 所有调度图校验诊断共用此前缀，便于统一检索日志。
constexpr char kScheduleGraphValidationErrorPrefix[] = "[AF][SCHEDULE GRAPH VALIDATION ERROR]";

/**
 * @brief 校验进入 Autofuse Schedule 流程前必须满足的图契约。
 *
 * 校验过程只读图数据，应在图重写和轴索引建立前调用。
 *
 * @param graph 待校验的 ASC 图。
 * @return 校验成功返回 af::SUCCESS，否则返回 af::FAILED。
 */
af::Status ValidateScheduleGraph(const af::AscGraph &graph);

/** @brief 清理当前线程上一次 Schedule 图校验留下的诊断信息。 */
void ClearScheduleValidationError();

/** @brief 获取当前线程保存的首个 Schedule 图校验错误信息。 */
const std::string &GetScheduleValidationError();

/**
 * @brief 保存当前线程的 Schedule 图校验错误信息。
 *
 * 仅保留首个根因，避免后续包装错误覆盖更有价值的诊断。
 *
 * @param message 要保存的错误文本。
 */
void RecordScheduleValidationError(const std::string &message);

}  // namespace optimize

#endif  // OPTIMIZE_SCHEDULE_GRAPH_VALIDATOR_H_
