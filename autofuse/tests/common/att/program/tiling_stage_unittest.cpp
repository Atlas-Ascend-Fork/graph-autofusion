/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include <thread>
#include <array>
#include "serialization/aftp_v1.h"
#include "program/axes_callback_bindings.h"
#include "program/axes_solve_runtime.h"
#include "program/tiling_result_writer.h"
#include "serialization/axes_callback_codec.h"
#include "serialization/axes_evaluation_codec.h"
#include "serialization/axes_model_codec.h"
#include "serialization/tiling_result_layout_codec.h"
#include "serialization/tiling_stage_codec.h"
namespace att {
namespace {
using Type = TilingProgramIRValueType;
using Value = TilingProgramIRValue;
using Status = TilingExpressionStatus;
using Kind = TilingProgramIRInstructionKind;
using Op = TilingProgramIROpcode;
struct Inputs {
  std::vector<Value> values;
  std::vector<uint32_t> reads;
};
TilingProgramValueStatus Read(void *context, uint32_t slot, Value *output) {
  auto &inputs = *static_cast<Inputs *>(context);
  inputs.reads.push_back(slot);
  if (slot >= inputs.values.size()) {
    return TilingProgramValueStatus::kInvalidArgument;
  }
  *output = inputs.values[slot];
  return TilingProgramValueStatus::kOk;
}
TilingProgramIRExpressionCode SymbolCode(std::vector<Type> types, uint32_t slot, Type result) {
  TilingProgramIRExpressionCode code;
  code.input_types = std::move(types);
  TilingProgramIRInstruction symbol;
  symbol.kind = Kind::kSymbol;
  symbol.symbol_id = slot;
  code.instructions.push_back(symbol);
  TilingProgramIRInstruction cast;
  cast.kind = Kind::kCast;
  cast.value_type = result;
  code.instructions.push_back(cast);
  code.result_type = result;
  return code;
}
std::shared_ptr<const TilingProgramIRCompiledStage> Build(const TilingProgramIRStageDefinition &definition) {
  std::shared_ptr<const TilingProgramIRCompiledStage> stage;
  EXPECT_EQ(BuildTilingProgramIRStage(definition, &stage).status, Status::kOk);
  return stage;
}
Value RunStage(const TilingProgramIRCompiledStage &stage, Inputs &inputs) {
  std::vector<Value> scratch(GetTilingStageScratchElements(stage));
  Value output;
  EXPECT_EQ(EvaluateTilingProgramIRStage(stage, &inputs, Read, scratch.data(), scratch.size(), &output).status,
            Status::kOk);
  return output;
}
TilingProgramIRStageDefinition SnapshotStage() {
  TilingProgramIRStageDefinition definition;
  definition.input_types = {Type::kInt64};
  definition.expressions.push_back(SymbolCode({Type::kInt64}, 0, Type::kFloat64));
  auto sum = SymbolCode({Type::kInt64, Type::kFloat64}, 1, Type::kFloat64);
  sum.instructions.pop_back();
  sum.instructions.push_back(sum.instructions.front());
  TilingProgramIRInstruction add;
  add.kind = Kind::kOperation;
  add.opcode = Op::kAdd;
  add.arg_count = 2;
  sum.instructions.push_back(add);
  definition.expressions.push_back(sum);
  definition.expressions.push_back(SymbolCode({Type::kInt64, Type::kFloat64, Type::kFloat64}, 2, Type::kInt64));
  return definition;
}
TEST(TilingStage, ConditionalReturnUsesEarlierLocalAndSkipsLaterInput) {
  TilingProgramIRStageDefinition definition;
  definition.input_types = {Type::kInt64, Type::kBoolean, Type::kInt64};
  definition.expressions = {
      SymbolCode(definition.input_types, 0, Type::kInt64),
      SymbolCode({Type::kInt64, Type::kBoolean, Type::kInt64, Type::kInt64}, 1, Type::kBoolean),
      SymbolCode({Type::kInt64, Type::kBoolean, Type::kInt64, Type::kInt64, Type::kBoolean}, 2, Type::kInt64),
  };
  definition.conditional_returns = {{1, {}, 0}};
  std::vector<uint8_t> bytes;
  ASSERT_EQ(EncodeTilingStageV1(definition, &bytes), Status::kOk);
  TilingProgramIRStageDefinition decoded;
  ASSERT_EQ(DecodeTilingStageV1(bytes.data(), bytes.size(), &decoded), Status::kOk);
  std::vector<uint8_t> roundtrip;
  ASSERT_EQ(EncodeTilingStageV1(decoded, &roundtrip), Status::kOk);
  EXPECT_EQ(roundtrip, bytes);
  EXPECT_EQ(DecodeTilingStageV1(bytes.data(), bytes.size() - 1, &decoded), Status::kInvalidProgram);
  bytes.push_back(0);
  EXPECT_EQ(DecodeTilingStageV1(bytes.data(), bytes.size(), &decoded), Status::kInvalidProgram);
  bytes.pop_back();
  bytes[0] = 2;
  EXPECT_EQ(DecodeTilingStageV1(bytes.data(), bytes.size(), &decoded), Status::kUnsupported);
  auto stage = Build(decoded);
  ASSERT_NE(stage, nullptr);
  Inputs selected{{{Type::kInt64, 42}, {Type::kBoolean, 1}}, {}};
  EXPECT_EQ(RunStage(*stage, selected).raw_bits, 42U);
  EXPECT_EQ(selected.reads, (std::vector<uint32_t>{0, 1}));
  Inputs fallthrough{{{Type::kInt64, 42}, {Type::kBoolean, 0}, {Type::kInt64, 17}}, {}};
  EXPECT_EQ(RunStage(*stage, fallthrough).raw_bits, 17U);
  EXPECT_EQ(fallthrough.reads, (std::vector<uint32_t>{0, 1, 2}));

  const auto saved = stage;
  for (uint32_t invalid : {1U, 2U, UINT32_MAX - 1}) {
    definition.conditional_returns.front().value_step = invalid;
    EXPECT_EQ(BuildTilingProgramIRStage(definition, &stage).status, Status::kInvalidProgram);
    EXPECT_EQ(stage, saved);
  }
  definition.conditional_returns.front().value_step = 0;
  definition.expressions[0].result_type = Type::kFloat64;
  EXPECT_EQ(BuildTilingProgramIRStage(definition, &stage).status, Status::kInvalidProgram);
  EXPECT_EQ(stage, saved);
}
TEST(TilingStage, ResultWriterChecksCapacityAndValuesBeforeWriting) {
  std::shared_ptr<const TilingResultLayout> layout;
  std::vector<uint8_t> encoded;
  ASSERT_EQ(EncodeTilingResultLayoutV1(6, 2, {{1, 4, Type::kUint16}, {0, 1, Type::kBoolean}}, &encoded), Status::kOk);
  ASSERT_EQ(DecodeTilingResultLayoutV1(encoded.data(), encoded.size(), &layout), Status::kOk);
  const auto saved = layout;
  encoded[32] = 5;  // Second field now overlaps the first.
  EXPECT_EQ(DecodeTilingResultLayoutV1(encoded.data(), encoded.size(), &layout), Status::kInvalidProgram);
  EXPECT_EQ(layout, saved);
  EXPECT_EQ(DecodeTilingResultLayoutV1(encoded.data(), encoded.size() - 1, &layout), Status::kInvalidProgram);
  EXPECT_EQ(layout, saved);
  EXPECT_EQ(TilingResultLayout::Build(6, 2, {{1, 4, Type::kUint16}, {0, 5, Type::kBoolean}}, &layout),
            Status::kInvalidProgram);
  EXPECT_EQ(layout, saved);
  std::vector<uint8_t> bytes(6, 0xaa);
  const auto untouched = bytes;
  Value values[] = {{Type::kBoolean, 1}, {Type::kUint16, 0x10000}};
  EXPECT_EQ(layout->Write(values, 2, bytes.data(), 5), Status::kBufferTooSmall);
  EXPECT_EQ(layout->Write(values, 2, bytes.data(), bytes.size()), Status::kInputError);
  EXPECT_EQ(bytes, untouched);
  values[1].raw_bits = 0x1234;
  EXPECT_EQ(layout->Write(values, 2, bytes.data(), bytes.size()), Status::kOk);
  EXPECT_EQ(bytes, (std::vector<uint8_t>{0xaa, 1, 0xaa, 0xaa, 0x34, 0x12}));
}
TEST(TilingStage, IntegerTileWriteRejectsLossBeforeWriting) {
  std::shared_ptr<const TilingResultLayout> layout;
  ASSERT_EQ(TilingResultLayout::Build(8, 2, {{0, 0, Type::kInt32}, {1, 4, Type::kUint32}}, &layout), Status::kOk);
  uint8_t payload[8] = {99};
  int64_t values[] = {INT32_MIN, -1};
  EXPECT_EQ(layout->WriteTiles(values, 2, payload, sizeof(payload)), Status::kInputError);
  EXPECT_EQ(payload[0], 99);
  values[1] = static_cast<int64_t>(UINT32_MAX) + 1;
  EXPECT_EQ(layout->WriteTiles(values, 2, payload, sizeof(payload)), Status::kInputError);
  EXPECT_EQ(payload[0], 99);
  values[1] = UINT32_MAX;
  EXPECT_EQ(layout->WriteTiles(values, 2, payload, sizeof(payload)), Status::kOk);
  EXPECT_EQ((std::vector<uint8_t>(payload, payload + sizeof(payload))),
            (std::vector<uint8_t>{0, 0, 0, 128, 255, 255, 255, 255}));
}
TEST(TilingStage, CallbackRecordRetainsRoleAndRejectsInvalidLocalBinding) {
  AxesCallbackBindings bindings;
  for (const auto role : {AxesProgramEvaluationRole::kConstraint, AxesProgramEvaluationRole::kUpperBound}) {
    AxesProgramEvaluationDefinition definition;
    definition.role = role;
    if (role == AxesProgramEvaluationRole::kConstraint) {
      definition.related_tiling_count = 1;
      definition.inputs = {{AxesProgramInputKind::kRelatedTilingVariable, 0}};
    } else {
      definition.parent_count = 1;
      definition.inputs = {{AxesProgramInputKind::kParentVariable, 0}};
    }
    definition.stage.input_types = {Type::kInt64};
    definition.stage.expressions = {SymbolCode({Type::kInt64}, 0, Type::kInt64)};
    std::vector<uint8_t> bytes;
    ASSERT_EQ(EncodeAxesCallbackV1(definition, &bytes), Status::kOk);
    std::shared_ptr<const AxesProgramEvaluation> callback;
    ASSERT_EQ(DecodeAxesCallbackV1(bytes.data(), bytes.size(), &callback), Status::kOk);
    EXPECT_EQ(callback->Role(), role);
    if (role == AxesProgramEvaluationRole::kConstraint) {
      bindings.constraints.push_back({callback, {1}, {}});
    } else {
      bindings.upper_bounds.push_back({callback, 0, {1}});
    }
    EXPECT_EQ(callback->Inputs().front().kind, definition.inputs.front().kind);
    const auto saved = callback;
    bytes[28] = 1;  // Binding index equals the declared array length.
    EXPECT_EQ(DecodeAxesCallbackV1(bytes.data(), bytes.size(), &callback), Status::kInvalidProgram);
    EXPECT_EQ(callback, saved);
    bytes.clear();
    Inputs inputs{{{Type::kInt64, 42}}, {}};
    EXPECT_EQ(RunStage(callback->Stage(), inputs).raw_bits, 42U);
  }
  EXPECT_EQ(VerifyAxesCallbackBindings(bindings, 2, 0), Status::kOk);
  EXPECT_EQ(VerifyAxesCallbackBindings(bindings, 1, 0), Status::kInvalidProgram);
  bindings.upper_bounds.push_back(bindings.upper_bounds.front());
  EXPECT_EQ(VerifyAxesCallbackBindings(bindings, 2, 0), Status::kInvalidProgram);
  bindings.upper_bounds.back().target_index = 1;
  EXPECT_EQ(VerifyAxesCallbackBindings(bindings, 2, 0), Status::kOk);
  bindings.constraints.front().callback = bindings.upper_bounds.front().callback;
  EXPECT_EQ(VerifyAxesCallbackBindings(bindings, 2, 0), Status::kInvalidProgram);
}
TEST(TilingStage, ModelMethodGroupRejectsInconsistentArrayLayoutsAtomically) {
  AxesModelDefinitions definitions;
  definitions[0].writes_output = false;
  definitions[1].function = AxesModelFunction::kRealUsedCoreNum;
  definitions[1].writes_output = false;
  definitions[2].function = AxesModelFunction::kUbThreshold;
  definitions[3].function = AxesModelFunction::kUbCacheLine;
  definitions[4].function = AxesModelFunction::kPerf;
  for (size_t i = 2; i < definitions.size(); ++i) {
    TilingProgramIRExpressionCode code;
    code.result_type = i == 4 ? Type::kFloat64 : Type::kBoolean;
    code.constants = {{code.result_type, i == 4 ? UINT64_C(0x3ff0000000000000) : UINT64_C(1)}};
    code.instructions.emplace_back();
    definitions[i].stage.expressions.push_back(std::move(code));
  }
  std::vector<uint8_t> bytes;
  ASSERT_EQ(EncodeAxesModelMethodsV1(definitions, &bytes), Status::kOk);
  AxesModelMethods methods;
  ASSERT_EQ(DecodeAxesModelMethodsV1(bytes.data(), bytes.size(), &methods), Status::kOk);
  EXPECT_EQ(methods[4]->Function(), AxesModelFunction::kPerf);
  const auto saved = methods;
  ASSERT_EQ(bytes[4], 28U);
  bytes[48] = 1;  // Shape count of the second method, whose individual record remains valid.
  EXPECT_EQ(DecodeAxesModelMethodsV1(bytes.data(), bytes.size(), &methods), Status::kInvalidProgram);
  EXPECT_EQ(methods, saved);
  AxesEvaluationDefinition bundle;
  bundle.tiling_count = 2;
  bundle.methods = definitions;
  AxesUpperBoundRecord bound;
  bound.target_index = 0;
  bound.parent_indices = {1};
  bound.callback.role = AxesProgramEvaluationRole::kUpperBound;
  bound.callback.parent_count = 1;
  bound.callback.inputs = {{AxesProgramInputKind::kParentVariable, 0}};
  bound.callback.stage.input_types = {Type::kInt64};
  bound.callback.stage.expressions = {SymbolCode({Type::kInt64}, 0, Type::kInt64)};
  bundle.upper_bounds.push_back(bound);
  std::vector<uint8_t> encoded;
  ASSERT_EQ(EncodeAxesEvaluationV1(bundle, &encoded), Status::kOk);
  AxesEvaluationRecords decoded;
  ASSERT_EQ(DecodeAxesEvaluationV1(encoded.data(), encoded.size(), &decoded), Status::kOk);
  ASSERT_EQ(decoded.bindings.upper_bounds.size(), 1U);
  EXPECT_EQ(decoded.bindings.upper_bounds[0].parent_indices, (std::vector<uint32_t>{1}));
  const auto saved_callback = decoded.bindings.upper_bounds[0].callback;
  encoded[encoded.size() - 4] = 2;  // The last field is the global parent index.
  EXPECT_EQ(DecodeAxesEvaluationV1(encoded.data(), encoded.size(), &decoded), Status::kInvalidProgram);
  EXPECT_EQ(decoded.bindings.upper_bounds[0].callback, saved_callback);
  AftpV1EncodeOptions options;
  options.schema_hash = 42;
  options.required_capabilities = 2;
  options.target_constraints = 7;
  std::vector<uint8_t> envelope;
  auto envelope_error = EncodeAxesEvaluationEnvelopeV1(bundle, options, &envelope);
  ASSERT_EQ(envelope_error.envelope, AftpV1Status::kOk);
  ASSERT_EQ(envelope_error.evaluation, Status::kOk);
  AftpV1Header header{};
  envelope_error = DecodeAxesEvaluationEnvelopeV1(envelope.data(), envelope.size(), {42, 0, 7}, &header, &decoded);
  EXPECT_EQ(envelope_error.envelope, AftpV1Status::kUnsupportedFeature);
  EXPECT_EQ(header.schema_hash, 0U);
  envelope_error = DecodeAxesEvaluationEnvelopeV1(envelope.data(), envelope.size(), {42, 2, 7}, &header, &decoded);
  ASSERT_EQ(envelope_error.envelope, AftpV1Status::kOk);
  ASSERT_EQ(envelope_error.evaluation, Status::kOk);
  EXPECT_EQ(header.schema_hash, 42U);
  const auto published = decoded.bindings.upper_bounds[0].callback;
  // Wrap the corrupted inner record again so that the envelope checksum is valid.
  ASSERT_EQ(EncodeAftpSectionPayloadV1(encoded.data(), encoded.size(), AftpV1SectionType::kTypedPlanPayload, options,
                                       &envelope),
            AftpV1Status::kOk);
  envelope_error = DecodeAxesEvaluationEnvelopeV1(envelope.data(), envelope.size(), {42, 2, 7}, &header, &decoded);
  EXPECT_EQ(envelope_error.envelope, AftpV1Status::kOk);
  EXPECT_EQ(envelope_error.evaluation, Status::kInvalidProgram);
  EXPECT_EQ(decoded.bindings.upper_bounds[0].callback, published);
  std::vector<uint8_t> with_layout;
  ASSERT_EQ(EncodeAxesEvaluationWithLayoutV1(bundle, 2, {{1, 0, Type::kUint16}}, &with_layout), Status::kOk);
  AxesEvaluationWithLayout decoded_bundle;
  ASSERT_EQ(DecodeAxesEvaluationWithLayoutV1(with_layout.data(), with_layout.size(), &decoded_bundle), Status::kOk);
  const auto saved_layout = decoded_bundle.layout;
  Value results[] = {{Type::kUint16, 0}, {Type::kUint16, 0x1234}};
  uint8_t payload[2]{};
  EXPECT_EQ(saved_layout->Write(results, 2, payload, sizeof(payload)), Status::kOk);
  EXPECT_EQ(payload[0], 0x34);
  EXPECT_EQ(payload[1], 0x12);
  with_layout[with_layout.size() - 20] = 3;  // A valid layout for the wrong global value count.
  EXPECT_EQ(DecodeAxesEvaluationWithLayoutV1(with_layout.data(), with_layout.size(), &decoded_bundle),
            Status::kInvalidProgram);
  EXPECT_EQ(decoded_bundle.layout, saved_layout);
  definitions[4].input_count = 1;
  const auto original_bytes = bytes;
  EXPECT_EQ(EncodeAxesModelMethodsV1(definitions, &bytes), Status::kInvalidProgram);
  EXPECT_EQ(bytes, original_bytes);
}
AxesEvaluationDefinition SingleAxisEvaluation() {
  AxesEvaluationDefinition definition;
  definition.tiling_count = 1;
  const std::array<AxesModelFunction, 5> functions = {
      AxesModelFunction::kUsedCoreNum, AxesModelFunction::kRealUsedCoreNum, AxesModelFunction::kUbThreshold,
      AxesModelFunction::kUbCacheLine, AxesModelFunction::kPerf};
  const std::array<Value, 5> constants = {Value{Type::kFloat64, UINT64_C(0x3ff0000000000000)}, Value{Type::kInt64, 1},
                                          Value{Type::kBoolean, 1}, Value{Type::kBoolean, 1},
                                          Value{Type::kFloat64, UINT64_C(0x3ff0000000000000)}};
  for (size_t i = 0; i < definition.methods.size(); ++i) {
    auto &method = definition.methods[i];
    method.function = functions[i];
    method.input_count = 1;
    method.pure_mc_count = 1;
    TilingProgramIRExpressionCode code;
    code.result_type = constants[i].type;
    code.constants = {constants[i]};
    code.instructions.emplace_back();
    method.stage.expressions.push_back(std::move(code));
  }
  AxesUpperBoundRecord upper;
  upper.parent_indices = {kAxesShapeParentBit};
  upper.callback.role = AxesProgramEvaluationRole::kUpperBound;
  upper.callback.parent_count = 1;
  upper.callback.inputs = {{AxesProgramInputKind::kParentVariable, 0}};
  upper.callback.stage.input_types = {Type::kInt64};
  upper.callback.stage.expressions = {SymbolCode({Type::kInt64}, 0, Type::kInt64)};
  definition.upper_bounds.push_back(std::move(upper));
  return definition;
}

TEST(AxesSolveRecord, EncodesDecodesAndRunsWithoutGraphProducer) {
  const auto definition = SingleAxisEvaluation();
  AxesSolveTopology topology;
  topology.variables.resize(1);
  topology.pure_mc = {0};
  std::vector<uint8_t> bytes;
  const std::vector<TilingResultField> fields{{0, 0, Type::kInt64}};
  ASSERT_EQ(EncodeAxesSolveV1(definition, topology, 8, fields, &bytes), Status::kOk);
  AxesSolveRecords records;
  ASSERT_EQ(DecodeAxesSolveV1(bytes.data(), bytes.size(), &records), Status::kOk);
  const auto layout = records.layout;
  EXPECT_EQ(DecodeAxesSolveV1(bytes.data(), bytes.size() - 1, &records), Status::kInvalidProgram);
  EXPECT_EQ(records.layout, layout);

  // V1 stores the topology at the end. Its first variable starts after the count,
  // four int64 fields and the multi-core flag.
  constexpr size_t topology_size = 4 + 48 + 4 + 8 + 4 + 4;
  ASSERT_GE(bytes.size(), topology_size);
  const auto reserved = bytes.begin() + (bytes.size() - topology_size) + 4 + 4 * 8 + 4;
  EXPECT_EQ(*reserved, 0xffU);
  *reserved = 0;
  EXPECT_EQ(DecodeAxesSolveV1(bytes.data(), bytes.size(), &records), Status::kInvalidProgram);
  EXPECT_EQ(records.layout, layout);
  bytes.clear();

  AxesSolveResources resources;
  resources.core_num = 8;
  resources.ub_size = 128;
  std::array<uint8_t, 8> payload;
  payload.fill(0xa5);
  const auto sentinel = payload;
  auto outcome = AxesSolveOutcome::kNoSolution;
  uint32_t block_dim = 99;
  EXPECT_EQ(RunAxesSolveRecord(records, {8}, resources, {}, payload.data(), payload.size() - 1, &outcome, nullptr,
                               &block_dim),
            Status::kBufferTooSmall);
  EXPECT_EQ(payload, sentinel);
  EXPECT_EQ(outcome, AxesSolveOutcome::kNoSolution);
  EXPECT_EQ(block_dim, 99U);

  ASSERT_EQ(
      RunAxesSolveRecord(records, {0}, resources, {}, payload.data(), payload.size(), &outcome, nullptr, &block_dim),
      Status::kOk);
  EXPECT_EQ(payload, sentinel);
  EXPECT_EQ(outcome, AxesSolveOutcome::kEmpty);
  EXPECT_EQ(block_dim, 99U);

  ASSERT_EQ(
      RunAxesSolveRecord(records, {8}, resources, {}, payload.data(), payload.size(), &outcome, nullptr, &block_dim),
      Status::kOk);
  EXPECT_EQ(outcome, AxesSolveOutcome::kSolved);
  EXPECT_EQ(block_dim, 1U);
  EXPECT_EQ(payload, (std::array<uint8_t, 8>{1, 0, 0, 0, 0, 0, 0, 0}));
}

TEST(AxesSolveRecord, RuntimeUbConstraintFailurePreservesOutputs) {
  auto definition = SingleAxisEvaluation();
  AxesConstraintRecord record;
  record.tiling_indices = {0};
  record.shape_indices = {0, kAxesTilingRelatedBit};
  auto &callback = record.callback;
  callback.related_tiling_count = 1;
  callback.related_input_count = 2;
  using InputKind = AxesProgramInputKind;
  callback.inputs = {{InputKind::kRelatedTilingVariable, 0},
                     {InputKind::kRelatedInputShape, 0},
                     {InputKind::kRelatedInputShape, 1},
                     {InputKind::kResourceValue, 0}};
  callback.stage.input_types.assign(4, Type::kInt64);
  TilingProgramIRExpressionCode code;
  code.input_types = callback.stage.input_types;
  TilingProgramIRInstruction add, divide;
  add.kind = divide.kind = Kind::kOperation;
  add.arg_count = divide.arg_count = 2;
  add.opcode = Op::kAdd;
  divide.opcode = Op::kDiv;
  const auto slot = [](uint32_t id) {
    TilingProgramIRInstruction instruction;
    instruction.kind = Kind::kSymbol;
    instruction.symbol_id = id;
    return instruction;
  };
  code.instructions = {slot(0), slot(1), add, slot(2), add, slot(3), divide};
  callback.stage.expressions = {code};
  definition.constraints.push_back(std::move(record));

  AxesSolveTopology topology;
  topology.variables.resize(1);
  topology.variables[0].constraints = {0};
  topology.pure_mc = {0};
  topology.constraints = {{0, 0, AxesConstraintResource::kRuntimeUb}};
  std::vector<uint8_t> bytes;
  ASSERT_EQ(EncodeAxesSolveV1(definition, topology, 8, {{0, 0, Type::kInt64}}, &bytes), Status::kOk);
  AxesSolveRecords decoded;
  ASSERT_EQ(DecodeAxesSolveV1(bytes.data(), bytes.size(), &decoded), Status::kOk);

  AxesSolveResources resources;
  resources.core_num = 8;
  resources.ub_size = 128;
  std::array<uint8_t, 8> payload{};
  uint32_t block_dim = 99;
  auto outcome = AxesSolveOutcome::kNoSolution;
  ASSERT_EQ(
      RunAxesSolveRecord(decoded, {8}, resources, {}, payload.data(), payload.size(), &outcome, nullptr, &block_dim),
      Status::kOk);
  EXPECT_EQ(outcome, AxesSolveOutcome::kSolved);
  EXPECT_EQ(block_dim, 1U);
  EXPECT_EQ(payload, (std::array<uint8_t, 8>{1, 0, 0, 0, 0, 0, 0, 0}));

  resources.ub_size = 0;
  payload.fill(0xa5);
  const auto sentinel = payload;
  block_dim = 99;
  outcome = AxesSolveOutcome::kEmpty;
  EXPECT_EQ(
      RunAxesSolveRecord(decoded, {8}, resources, {}, payload.data(), payload.size(), &outcome, nullptr, &block_dim),
      Status::kEvaluationError);
  EXPECT_EQ(payload, sentinel);
  EXPECT_EQ(outcome, AxesSolveOutcome::kEmpty);
  EXPECT_EQ(block_dim, 99U);
}

TEST(AxesSolveRecord, DerivedWritebackAndManifestRoundTrip) {
  const auto definition = SingleAxisEvaluation();
  AxesSolveTopology topology;
  topology.variables.resize(1);
  topology.pure_mc = {0};
  const std::vector<TilingResultField> tiles{{0, 0, Type::kUint32}};
  AxesDerivedDefinition derived;
  derived.shape_count = 1;
  derived.fields = {{0, 0, Type::kUint32}, {1, 4, Type::kUint32}};
  TilingProgramIRStageDefinition stage;
  stage.input_types.assign(4, Type::kUint32);
  stage.expressions = {SymbolCode(stage.input_types, 3, Type::kUint32)};
  derived.stages = {stage};
  const auto slots = HashAxesShapeSlots({"extent"});
  EXPECT_NE(slots, 0U);
  EXPECT_NE(slots, HashAxesShapeSlots({"other"}));
  const AxesSolveManifestIdentity identity{11, slots, 13};
  AxesSolveEnvelopeOptions options;
  options.header = {1, 2, 3, 4, 5};
  options.structured_identity = &identity;
  options.derived = &derived;

  std::vector<uint8_t> manifest;
  ASSERT_EQ(EncodeAxesSolveManifestV1(identity, 8, 2, derived.fields, &manifest), Status::kOk);
  const AftpV1PayloadDescriptor expected{manifest.data(), manifest.size()};
  std::vector<uint8_t> bytes;
  const auto encoded = EncodeAxesSolveEnvelopeV1(definition, topology, 8, tiles, options, &bytes);
  ASSERT_EQ(encoded.envelope, AftpV1Status::kOk);
  ASSERT_EQ(encoded.evaluation, Status::kOk);
  AftpV1Header header{};
  AftpV1ProgramPayload envelope;
  ASSERT_EQ(DecodeAftpProgramWithManifestV1(bytes.data(), bytes.size(), &header, &envelope), AftpV1Status::kOk);
  ASSERT_FALSE(envelope.plan_payload.empty());
  EXPECT_EQ(envelope.plan_payload.front(), 6U);
  AxesSolveRecords decoded;
  auto result = DecodeAxesSolveEnvelopeV1(bytes.data(), bytes.size(), {3, 4, 5}, expected, &header, &decoded);
  ASSERT_EQ(result.envelope, AftpV1Status::kOk);
  ASSERT_EQ(result.evaluation, Status::kOk);

  AxesSolveResources resources;
  resources.core_num = 8;
  resources.ub_size = 128;
  std::array<uint8_t, 8> payload{};
  uint32_t block_dim = 99;
  auto outcome = AxesSolveOutcome::kNoSolution;
  ASSERT_EQ(
      RunAxesSolveRecord(decoded, {8}, resources, {}, payload.data(), payload.size(), &outcome, nullptr, &block_dim),
      Status::kOk);
  EXPECT_EQ(outcome, AxesSolveOutcome::kSolved);
  EXPECT_EQ(block_dim, 1U);
  EXPECT_EQ(payload, (std::array<uint8_t, 8>{1, 0, 0, 0, 128, 0, 0, 0}));

  const auto layout = decoded.derived.layout;
  header.program_id = 99;
  auto wrong_manifest = manifest;
  wrong_manifest.back() ^= 1;
  const AftpV1PayloadDescriptor wrong{wrong_manifest.data(), wrong_manifest.size()};
  EXPECT_EQ(DecodeAxesSolveEnvelopeV1(bytes.data(), bytes.size(), {30, 4, 5}, expected, &header, &decoded).envelope,
            AftpV1Status::kUnsupportedFeature);
  EXPECT_EQ(DecodeAxesSolveEnvelopeV1(bytes.data(), bytes.size(), {3, 4, 50}, expected, &header, &decoded).envelope,
            AftpV1Status::kUnsupportedFeature);
  EXPECT_EQ(DecodeAxesSolveEnvelopeV1(bytes.data(), bytes.size(), {3, 4, 5}, wrong, &header, &decoded).envelope,
            AftpV1Status::kUnsupportedFeature);
  // Rewrap the changed manifest with a valid envelope checksum to reach layout identity validation.
  ASSERT_EQ(EncodeAftpProgramWithManifestV1({{envelope.plan_payload.data(), envelope.plan_payload.size()},
                                             wrong,
                                             AftpV1SectionType::kAxesSolvePayload,
                                             options.header},
                                            &bytes),
            AftpV1Status::kOk);
  result = DecodeAxesSolveEnvelopeV1(bytes.data(), bytes.size(), {3, 4, 5}, wrong, &header, &decoded);
  EXPECT_EQ(result.envelope, AftpV1Status::kOk);
  EXPECT_EQ(result.evaluation, Status::kInvalidProgram);
  EXPECT_EQ(header.program_id, 99U);
  EXPECT_EQ(decoded.derived.layout, layout);

  derived.stages.front().expressions.front() = SymbolCode(stage.input_types, 3, Type::kInt64);
  const auto mismatched = EncodeAxesSolveEnvelopeV1(definition, topology, 8, tiles, options, &bytes);
  ASSERT_EQ(mismatched.envelope, AftpV1Status::kOk);
  ASSERT_EQ(mismatched.evaluation, Status::kOk);
  ASSERT_EQ(DecodeAxesSolveEnvelopeV1(bytes.data(), bytes.size(), {3, 4, 5}, expected, &header, &decoded).evaluation,
            Status::kOk);
  const auto previous = payload;
  block_dim = 99;
  outcome = AxesSolveOutcome::kEmpty;
  EXPECT_EQ(
      RunAxesSolveRecord(decoded, {8}, resources, {}, payload.data(), payload.size(), &outcome, nullptr, &block_dim),
      Status::kInputError);
  EXPECT_EQ(payload, previous);
  EXPECT_EQ(block_dim, 99U);
  EXPECT_EQ(outcome, AxesSolveOutcome::kEmpty);
}

TEST(TilingStage, ModelRecordVerifiesBindingsAndOwnsDecodedStage) {
  AxesModelEvaluationDefinition definition;
  definition.function = AxesModelFunction::kPerf;
  definition.input_count = 1;
  definition.inputs = {{AxesModelInputKind::kInputShape, 0}};
  definition.stage.input_types = {Type::kInt64};
  definition.stage.expressions = {SymbolCode({Type::kInt64}, 0, Type::kFloat64)};
  std::vector<uint8_t> bytes;
  ASSERT_EQ(EncodeAxesModelV1(definition, &bytes), Status::kOk);
  std::shared_ptr<const AxesModelEvaluation> model;
  ASSERT_EQ(DecodeAxesModelV1(bytes.data(), bytes.size(), &model), Status::kOk);
  const auto saved = model;
  bytes[32] = 1;  // Binding index is now outside the declared shape array.
  EXPECT_EQ(DecodeAxesModelV1(bytes.data(), bytes.size(), &model), Status::kInvalidProgram);
  EXPECT_EQ(model, saved);
  bytes.clear();
  Inputs inputs{{{Type::kInt64, 3}}, {}};
  EXPECT_EQ(RunStage(*model->Stage(), inputs).raw_bits, UINT64_C(0x4008000000000000));

  definition = {};
  definition.writes_output = false;
  ASSERT_EQ(EncodeAxesModelV1(definition, &bytes), Status::kOk);
  ASSERT_EQ(DecodeAxesModelV1(bytes.data(), bytes.size(), &model), Status::kOk);
  EXPECT_FALSE(model->WritesOutput());
  EXPECT_EQ(model->Stage(), nullptr);
  bytes.push_back(0);
  EXPECT_EQ(DecodeAxesModelV1(bytes.data(), bytes.size(), &model), Status::kInvalidProgram);
  EXPECT_FALSE(model->WritesOutput());
}
TEST(TilingStage, MaterializesOnceAndOwnsVerifiedCode) {
  auto definition = SnapshotStage();
  auto stage = Build(definition);
  ASSERT_NE(stage, nullptr);
  definition.expressions.clear();
  Inputs inputs{{{Type::kInt64, 9007199254740993ULL}}, {}};
  const auto result = RunStage(*stage, inputs);
  EXPECT_EQ(result.type, Type::kInt64);
  EXPECT_EQ(result.raw_bits, 18014398509481984ULL);
  EXPECT_EQ(inputs.reads, (std::vector<uint32_t>{0}));
}
TEST(TilingStage, RejectsForwardReferencesTypeMismatchAndCumulativeLimitAtomically) {
  auto valid = SnapshotStage();
  auto stage = Build(valid);
  auto original = stage;
  auto bad = valid;
  bad.expressions[1].instructions[0].symbol_id = 2;
  EXPECT_EQ(BuildTilingProgramIRStage(bad, &stage).status, Status::kInvalidProgram);
  EXPECT_EQ(stage, original);
  bad = valid;
  bad.expressions[1].input_types[1] = Type::kInt64;
  EXPECT_EQ(BuildTilingProgramIRStage(bad, &stage).step, 1U);
  bad = valid;
  bad.expressions[0].constants.resize(kTilingExpressionMaxItems);
  EXPECT_EQ(BuildTilingProgramIRStage(bad, &stage).status, Status::kInvalidProgram);
  EXPECT_EQ(BuildTilingProgramIRStage({}, &stage).status, Status::kInvalidProgram);
  EXPECT_EQ(BuildTilingProgramIRStage(valid, nullptr).status, Status::kInvalidArgument);
  EXPECT_EQ(stage, original);
}
TEST(TilingStage, ChecksStorageBeforeReadingAndPreservesOutputOnFailure) {
  auto stage = Build(SnapshotStage());
  ASSERT_NE(stage, nullptr);
  Inputs inputs{{{Type::kInt64, 1}}, {}};
  const size_t required = GetTilingStageScratchElements(*stage);
  std::vector<Value> scratch(required);
  Value output{Type::kUint32, 97};
  EXPECT_EQ(EvaluateTilingProgramIRStage(*stage, &inputs, Read, scratch.data(), required - 1, &output).status,
            Status::kBufferTooSmall);
  EXPECT_EQ(EvaluateTilingProgramIRStage(*stage, &inputs, Read, nullptr, required, &output).status,
            Status::kInvalidArgument);
  EXPECT_EQ(EvaluateTilingProgramIRStage(*stage, &inputs, Read, scratch.data(), required, scratch.data()).status,
            Status::kInvalidArgument);
  EXPECT_TRUE(inputs.reads.empty());
  inputs.values.clear();
  const auto error = EvaluateTilingProgramIRStage(*stage, &inputs, Read, scratch.data(), required, &output);
  EXPECT_EQ(error.status, Status::kInputError);
  EXPECT_EQ(error.step, 0U);
  EXPECT_EQ(error.expression_error.instruction, 0U);
  EXPECT_EQ(output.type, Type::kUint32);
  EXPECT_EQ(output.raw_bits, 97U);
  EXPECT_EQ(inputs.reads, (std::vector<uint32_t>{0}));
}
TEST(TilingStage, SharedImmutableCodeHasPrivateInvocationState) {
  auto stage = Build(SnapshotStage());
  ASSERT_NE(stage, nullptr);
  std::vector<std::thread> threads;
  for (uint64_t i = 1; i <= 4; ++i) {
    threads.emplace_back([stage, i] {
      Inputs inputs{{{Type::kInt64, i}}, {}};
      for (size_t repeat = 0; repeat < 100; ++repeat) {
        EXPECT_EQ(RunStage(*stage, inputs).raw_bits, 2 * i);
      }
      EXPECT_EQ(inputs.reads.size(), 100U);
    });
  }
  for (auto &thread : threads) {
    thread.join();
  }
}
TEST(TilingStage, ShortCircuitSkipsExternalReadAndKeepsFirstError) {
  TilingProgramIRStageDefinition definition;
  definition.input_types = {Type::kInt64};
  TilingProgramIRExpressionCode condition;
  condition.input_types = definition.input_types;
  condition.result_type = Type::kBoolean;
  TilingProgramIRInstruction flag;
  flag.kind = Kind::kPushBoolean;
  condition.instructions.push_back(flag);
  definition.expressions.push_back(condition);
  TilingProgramIRExpressionCode branch;
  branch.input_types = {Type::kInt64, Type::kBoolean};
  branch.constants = {{Type::kInt64, 42}};
  TilingProgramIRInstruction local;
  local.kind = Kind::kSymbol;
  local.symbol_id = 1;
  TilingProgramIRInstruction jump_false;
  jump_false.kind = Kind::kJumpIfFalse;
  jump_false.jump = 4;
  TilingProgramIRInstruction external;
  external.kind = Kind::kSymbol;
  TilingProgramIRInstruction jump;
  jump.kind = Kind::kJump;
  jump.jump = 5;
  TilingProgramIRInstruction constant;
  branch.instructions = {local, jump_false, external, jump, constant};
  definition.expressions.push_back(branch);
  auto stage = Build(definition);
  ASSERT_NE(stage, nullptr);
  Inputs inputs;
  EXPECT_EQ(RunStage(*stage, inputs).raw_bits, 42U);
  EXPECT_TRUE(inputs.reads.empty());
  definition.expressions[0].instructions[0].boolean_value = true;
  stage = Build(definition);
  ASSERT_NE(stage, nullptr);
  std::vector<Value> scratch(GetTilingStageScratchElements(*stage));
  Value result{Type::kInt64, 99};
  const auto error = EvaluateTilingProgramIRStage(*stage, &inputs, Read, scratch.data(), scratch.size(), &result);
  EXPECT_EQ(error.status, Status::kInputError);
  EXPECT_EQ(error.step, 1U);
  EXPECT_EQ(error.expression_error.instruction, 2U);
  EXPECT_EQ(inputs.reads, (std::vector<uint32_t>{0}));
  EXPECT_EQ(result.raw_bits, 99U);
}

TilingProgramIRStageDefinition ConditionalReturnStage() {
  TilingProgramIRStageDefinition definition;
  definition.input_types = {Type::kBoolean, Type::kBoolean, Type::kInt64};
  auto types = definition.input_types;
  definition.expressions.push_back(SymbolCode(types, 0, Type::kBoolean));
  types.push_back(Type::kBoolean);
  definition.expressions.push_back(SymbolCode(types, 1, Type::kBoolean));
  types.push_back(Type::kBoolean);
  definition.expressions.push_back(SymbolCode(types, 2, Type::kInt64));
  definition.conditional_returns = {{0, {Type::kInt64, UINT64_MAX}}, {1, {Type::kInt64, 17}}};
  return definition;
}
TEST(TilingStage, ConditionalReturnsOwnValuesAndSkipSubsequentReads) {
  auto definition = ConditionalReturnStage();
  const auto stage = Build(definition);
  ASSERT_NE(stage, nullptr);
  definition.conditional_returns[0].value.raw_bits = 5;
  definition.conditional_returns.clear();
  Inputs inputs{{{Type::kBoolean, 1}}, {}};
  EXPECT_EQ(RunStage(*stage, inputs).raw_bits, UINT64_MAX);
  EXPECT_EQ(inputs.reads, (std::vector<uint32_t>{0}));
  inputs = {{{Type::kBoolean, 0}, {Type::kBoolean, 1}}, {}};
  EXPECT_EQ(RunStage(*stage, inputs).raw_bits, 17U);
  EXPECT_EQ(inputs.reads, (std::vector<uint32_t>{0, 1}));
  inputs = {{{Type::kBoolean, 0}, {Type::kBoolean, 0}, {Type::kInt64, 29}}, {}};
  EXPECT_EQ(RunStage(*stage, inputs).raw_bits, 29U);
  EXPECT_EQ(inputs.reads, (std::vector<uint32_t>{0, 1, 2}));
}
TEST(TilingStage, ConditionalReturnSkipsArithmeticButDoesNotHideInvalidCode) {
  auto definition = ConditionalReturnStage();
  auto &last = definition.expressions.back();
  last.constants = {{Type::kInt64, 1}, {Type::kInt64, 0}};
  TilingProgramIRInstruction one, zero, divide;
  zero.constant_id = 1;
  divide.kind = Kind::kOperation;
  divide.opcode = Op::kDiv;
  divide.arg_count = 2;
  last.instructions = {one, zero, divide};
  auto stage = Build(definition);
  ASSERT_NE(stage, nullptr);
  Inputs inputs{{{Type::kBoolean, 1}}, {}};
  EXPECT_EQ(RunStage(*stage, inputs).raw_bits, UINT64_MAX);
  inputs = {{{Type::kBoolean, 0}, {Type::kBoolean, 0}}, {}};
  std::vector<Value> scratch(GetTilingStageScratchElements(*stage));
  Value output{Type::kInt64, 97};
  const auto error = EvaluateTilingProgramIRStage(*stage, &inputs, Read, scratch.data(), scratch.size(), &output);
  EXPECT_EQ(error.status, Status::kEvaluationError);
  EXPECT_EQ(error.step, 2U);
  EXPECT_EQ(error.expression_error.instruction, 2U);
  EXPECT_EQ(output.raw_bits, 97U);
  last.instructions[1].constant_id = 2;
  auto previous = stage;
  EXPECT_EQ(BuildTilingProgramIRStage(definition, &stage).status, Status::kInvalidProgram);
  EXPECT_EQ(stage, previous);
}
TEST(TilingStage, RejectsInvalidConditionalReturnRecordsAtomically) {
  const auto valid = ConditionalReturnStage();
  auto stage = Build(valid);
  ASSERT_NE(stage, nullptr);
  const auto original = stage;
  auto bad = valid;
  bad.conditional_returns[0].condition_step = 3;
  EXPECT_EQ(BuildTilingProgramIRStage(bad, &stage).status, Status::kInvalidProgram);
  bad = valid;
  bad.conditional_returns[1].condition_step = 0;
  EXPECT_EQ(BuildTilingProgramIRStage(bad, &stage).status, Status::kInvalidProgram);
  bad = valid;
  std::swap(bad.conditional_returns[0], bad.conditional_returns[1]);
  EXPECT_EQ(BuildTilingProgramIRStage(bad, &stage).status, Status::kInvalidProgram);
  bad = valid;
  bad.conditional_returns = {{2, {Type::kInt64, 0}}};
  EXPECT_EQ(BuildTilingProgramIRStage(bad, &stage).status, Status::kInvalidProgram);
  bad = valid;
  bad.conditional_returns[0].value.type = Type::kUint64;
  EXPECT_EQ(BuildTilingProgramIRStage(bad, &stage).status, Status::kInvalidProgram);
  bad = valid;
  bad.expressions.pop_back();
  bad.conditional_returns = {{0, {Type::kBoolean, 2}}};
  EXPECT_EQ(BuildTilingProgramIRStage(bad, &stage).status, Status::kInvalidProgram);
  EXPECT_EQ(stage, original);
}
TEST(TilingStage, ConditionalReturnPreservesStorageChecksAndFirstInputError) {
  auto stage = Build(ConditionalReturnStage());
  ASSERT_NE(stage, nullptr);
  Inputs inputs{{{Type::kBoolean, 1}}, {}};
  std::vector<Value> scratch(GetTilingStageScratchElements(*stage));
  Value output{Type::kUint32, 97};
  auto error = EvaluateTilingProgramIRStage(*stage, &inputs, Read, scratch.data(), scratch.size() - 1, &output);
  EXPECT_EQ(error.status, Status::kBufferTooSmall);
  EXPECT_TRUE(inputs.reads.empty());
  inputs.values.clear();
  error = EvaluateTilingProgramIRStage(*stage, &inputs, Read, scratch.data(), scratch.size(), &output);
  EXPECT_EQ(error.status, Status::kInputError);
  EXPECT_EQ(error.step, 0U);
  EXPECT_EQ(error.expression_error.instruction, 0U);
  EXPECT_EQ(inputs.reads, (std::vector<uint32_t>{0}));
  EXPECT_EQ(output.type, Type::kUint32);
  EXPECT_EQ(output.raw_bits, 97U);
}
TEST(TilingStage, ConditionalReturnRecordsCountTowardCumulativeLimit) {
  TilingProgramIRStageDefinition definition;
  definition.input_types.resize((kTilingExpressionMaxItems - 2) / 2, Type::kBoolean);
  TilingProgramIRExpressionCode code;
  code.input_types = definition.input_types;
  code.result_type = Type::kBoolean;
  TilingProgramIRInstruction instruction;
  instruction.kind = Kind::kPushBoolean;
  code.instructions = {instruction};
  definition.expressions = {code};
  definition.conditional_returns = {{0, {Type::kBoolean, 1}}};
  auto stage = Build(definition);
  ASSERT_NE(stage, nullptr);
  const auto original = stage;
  definition.expressions[0].constants = {{Type::kBoolean, 0}};
  EXPECT_EQ(BuildTilingProgramIRStage(definition, &stage).status, Status::kInvalidProgram);
  EXPECT_EQ(stage, original);
  definition.conditional_returns.clear();
  EXPECT_EQ(BuildTilingProgramIRStage(definition, &stage).status, Status::kOk);
}
}  // namespace
}  // namespace att
