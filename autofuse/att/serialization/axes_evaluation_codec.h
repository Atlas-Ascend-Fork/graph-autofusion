/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATT_SERIALIZATION_AXES_EVALUATION_CODEC_H_
#define ATT_SERIALIZATION_AXES_EVALUATION_CODEC_H_
#include <set>
#include <map>
#include <string>
#include "aftp_v1.h"
#include "axes_model_codec.h"
#include "program/tiling_result_writer.h"
#include "program/tiling_stage.h"
#include "program/axes_callback_bindings.h"
#include "program/axes_solve_topology.h"

namespace att {
struct AxesConstraintRecord {
  AxesProgramEvaluationDefinition callback;
  std::vector<uint32_t> tiling_indices;
  std::vector<uint32_t> shape_indices;
};
struct AxesUpperBoundRecord {
  AxesProgramEvaluationDefinition callback;
  uint32_t target_index = 0;
  std::vector<uint32_t> parent_indices;
};
struct AxesEvaluationDefinition {
  uint32_t tiling_count = 0;
  AxesModelDefinitions methods;
  std::vector<AxesConstraintRecord> constraints;
  std::vector<AxesUpperBoundRecord> upper_bounds;
};
struct AxesEvaluationRecords {
  uint32_t tiling_count = 0;
  AxesModelMethods methods;
  AxesCallbackBindings bindings;
};
TilingExpressionStatus GetAxesEvaluationScratchElements(const AxesEvaluationRecords &records, size_t *elements);
TilingExpressionStatus EncodeAxesEvaluationV1(const AxesEvaluationDefinition &definition, std::vector<uint8_t> *output);
TilingExpressionStatus DecodeAxesEvaluationV1(const void *data, size_t size, AxesEvaluationRecords *output);
struct AxesEvaluationEnvelopeError {
  AftpV1Status envelope = AftpV1Status::kOk;
  TilingExpressionStatus evaluation = TilingExpressionStatus::kOk;
};
struct AxesEvaluationEnvelopeRequirements {
  uint64_t schema_hash = 0;
  uint64_t supported_capabilities = 0;
  uint64_t target_constraints = 0;
};
AxesEvaluationEnvelopeError EncodeAxesEvaluationEnvelopeV1(const AxesEvaluationDefinition &definition,
                                                           const AftpV1EncodeOptions &options,
                                                           std::vector<uint8_t> *output);
AxesEvaluationEnvelopeError DecodeAxesEvaluationEnvelopeV1(const void *data, size_t size,
                                                           const AxesEvaluationEnvelopeRequirements &requirements,
                                                           AftpV1Header *header, AxesEvaluationRecords *output);
struct AxesEvaluationWithLayout {
  AxesEvaluationRecords evaluation;
  std::shared_ptr<const TilingResultLayout> layout;
  uint64_t layout_identity = 0;
};
TilingExpressionStatus EncodeAxesEvaluationWithLayoutV1(const AxesEvaluationDefinition &definition,
                                                        uint32_t payload_size,
                                                        const std::vector<TilingResultField> &fields,
                                                        std::vector<uint8_t> *output);
TilingExpressionStatus DecodeAxesEvaluationWithLayoutV1(const void *data, size_t size,
                                                        AxesEvaluationWithLayout *output);
enum AxesSolveFlag : uint32_t {
  kAxesSolveTradeOff = 1U,
  kAxesSolveTuneBlock = 2U,
  kAxesSolveTune = 4U,
};
struct AxesDerivedDefinition {
  std::vector<TilingProgramIRStageDefinition> stages;
  std::vector<TilingResultField> fields;
  uint32_t shape_count = 0;
};
struct AxesDerivedWriteback {
  // Stage inputs are shape slots, solved tile slots, then the solved block dimension and effective UB budget.
  std::vector<std::shared_ptr<const TilingProgramIRCompiledStage>> stages;
  std::shared_ptr<const TilingResultLayout> layout;
  uint32_t shape_count = 0;
};
struct AxesSolveRecords : AxesEvaluationWithLayout {
  AxesSolveTopology topology;
  AxesDerivedWriteback derived;
  uint32_t solve_flags = 0;
};
TilingExpressionStatus EncodeAxesSolveV1(const AxesEvaluationDefinition &definition, const AxesSolveTopology &topology,
                                         uint32_t payload_size, const std::vector<TilingResultField> &fields,
                                         std::vector<uint8_t> *output, uint32_t solve_flags = 0);
TilingExpressionStatus DecodeAxesSolveV1(const void *data, size_t size, AxesSolveRecords *output);
uint64_t HashAxesShapeSlots(const std::vector<std::string> &ordered_names);
struct AxesSolveManifestIdentity {
  uint64_t graph_case;
  uint64_t shape_slots;
  uint64_t companion_assets;
};
TilingExpressionStatus EncodeAxesSolveManifestV1(const AxesSolveManifestIdentity &identity, uint32_t payload_size,
                                                 uint32_t value_count, const std::vector<TilingResultField> &fields,
                                                 std::vector<uint8_t> *output);
struct AxesSolveEnvelopeOptions {
  AftpV1EncodeOptions header;
  AftpV1PayloadDescriptor manifest;
  uint32_t solve_flags = 0;
  const AxesSolveManifestIdentity *structured_identity = nullptr;
  uint64_t graph_identity = 0;
  const std::map<std::string, uint32_t> *scalar_field_offsets = nullptr;
  const AxesDerivedDefinition *derived = nullptr;
  bool derive_memory_fields = false;
  const std::map<uint32_t, uint32_t> *device_dispatch_keys = nullptr;
  const std::vector<std::string> *verified_zero_fields = nullptr;
  const std::map<uint32_t, std::set<std::string>> *device_candidate_reads = nullptr;
  bool derive_inactive_scalar_fields = false;
};
AxesEvaluationEnvelopeError EncodeAxesSolveEnvelopeV1(const AxesEvaluationDefinition &definition,
                                                      const AxesSolveTopology &topology, uint32_t payload_size,
                                                      const std::vector<TilingResultField> &fields,
                                                      const AxesSolveEnvelopeOptions &options,
                                                      std::vector<uint8_t> *output);
AxesEvaluationEnvelopeError DecodeAxesSolveEnvelopeV1(const void *data, size_t size,
                                                      const AxesEvaluationEnvelopeRequirements &requirements,
                                                      const AftpV1PayloadDescriptor &expected_manifest,
                                                      AftpV1Header *header, AxesSolveRecords *output);
}  // namespace att
#endif
