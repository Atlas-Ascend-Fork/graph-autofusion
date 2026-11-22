/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef AUTOFUSE_MLIR_ARTIFACT_KERNELSOURCE_AGGREGATELAUNCHDRIVER_H
#define AUTOFUSE_MLIR_ARTIFACT_KERNELSOURCE_AGGREGATELAUNCHDRIVER_H

#include <string>

#include <utility>

#include "artifact/KernelSource/GenAggregateKernelEntry.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::ascend::kernel {

// Canonical artifacts from one aggregate: device kernel body, tiling-data
// header, host representation helper, and AutofuseLaunch wrapper. The kernel
// source combines kernel_body + autofuse_launch; the Inductor host source
// includes tiling_data_repr separately from device compilation.
struct AggregateLaunchArtifacts {
  std::string kernel_body;
  std::string tiling_data;
  std::string tiling_data_repr;
  std::string autofuse_launch;
};

// Build a self-contained entry module with aggregate tiling data, validated contract,
// kernel entry, bodies and includes. It owns cloned plan bodies and outlives the inputs.
// Shared by the --afir-aggregate pass and runAggregateLaunchDriver.
FailureOr<std::pair<OwningOpRef<ModuleOp>, AggregateArgumentLayout>> buildAggregateEntryModule(
    ArrayRef<ModuleOp> planModules, StringRef fusedGraphName, std::string *diagnostic);

// Aggregate lowered per-candidate planModules, translate to AscendC, and emit the
// tiling header and AutofuseLaunch wrapper. Inputs come from addPlanCandidateLoweringPipeline;
// fusedGraphName names the aggregate entry. Shared by codegen orchestration and af-opt.
FailureOr<AggregateLaunchArtifacts> runAggregateLaunchDriver(ArrayRef<ModuleOp> planModules, StringRef fusedGraphName,
                                                             std::string *diagnostic);

}  // namespace mlir::ascend::kernel

#endif  // AUTOFUSE_MLIR_ARTIFACT_KERNELSOURCE_AGGREGATELAUNCHDRIVER_H
