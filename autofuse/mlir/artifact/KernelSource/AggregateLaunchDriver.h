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

// The three text artifacts a plan candidate set aggregates down to: the
// translated AscendC kernel body, the generated tiling-data header, and
// the AutofuseLaunch host wrapper. Concatenated in this order by callers to
// form the final kernel source (kernel_body + autofuse_launch) and the tiling
// data header respectively.
struct AggregateLaunchArtifacts {
  std::string kernel_body;
  std::string tiling_data;
  std::string autofuse_launch;
};

// Aggregates `planModules` into one self-contained entry module: tiling data,
// contract validation, kernel entry, kernel body, kernel includes. The returned
// module owns clones of the plan bodies, so it stays valid after `planModules`
// are gone. Exposed so the --afir-aggregate pass and runAggregateLaunchDriver
// drive one implementation rather than two.
FailureOr<std::pair<OwningOpRef<ModuleOp>, AggregateArgumentLayout>> buildAggregateEntryModule(
    ArrayRef<ModuleOp> planModules, StringRef fusedGraphName, std::string *diagnostic);

// Drives the aggregate + AutofuseLaunch text emission stage shared by the
// codegen orchestration path and the af-opt tool: aggregates `planModules`
// (independent per-candidate AFIR/AscendC modules already lowered via
// addPlanCandidateLoweringPipeline) into one entry module, translates it to
// AscendC, and emits the tiling-data header and AutofuseLaunch host wrapper
// for it. `fusedGraphName` names the aggregate kernel entry.
FailureOr<AggregateLaunchArtifacts> runAggregateLaunchDriver(ArrayRef<ModuleOp> planModules, StringRef fusedGraphName,
                                                             std::string *diagnostic);

}  // namespace mlir::ascend::kernel

#endif  // AUTOFUSE_MLIR_ARTIFACT_KERNELSOURCE_AGGREGATELAUNCHDRIVER_H
