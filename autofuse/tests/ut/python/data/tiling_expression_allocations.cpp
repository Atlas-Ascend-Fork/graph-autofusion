/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <array>
#include <cstdio>
#include <cstdlib>
#include <new>
#include "program/tiling_stage.h"
static bool track = false;
static size_t allocations = 0;
void *operator new(size_t n) {
  if (track) {
    ++allocations;
  }
  void *p = std::malloc(n ? n : 1);
  if (!p) {
    throw std::bad_alloc();
  }
  return p;
}
void *operator new[](size_t n) {
  return ::operator new(n);
}
void operator delete(void *p) noexcept {
  std::free(p);
}
void operator delete[](void *p) noexcept {
  std::free(p);
}
void operator delete(void *p, size_t) noexcept {
  std::free(p);
}
void operator delete[](void *p, size_t) noexcept {
  std::free(p);
}
struct Context {
  uint64_t value;
  bool fail;
  size_t calls;
};
att::TilingProgramValueStatus Read(void *p, uint32_t, att::TilingProgramIRValue *v) {
  auto &c = *static_cast<Context *>(p);
  ++c.calls;
  if (c.fail) {
    return att::TilingProgramValueStatus::kInvalidValue;
  }
  *v = {att::TilingProgramIRValueType::kInt64, c.value};
  return att::TilingProgramValueStatus::kOk;
}
int main() {
  using namespace att;
  track = true;
  void *control = ::operator new(16);
  track = false;
  ::operator delete(control);
  bool hook = allocations == 1;
  allocations = 0;
  TilingProgramIRExpressionCode c;
  c.constants = {{TilingProgramIRValueType::kInt64, 1}};
  c.input_types = {TilingProgramIRValueType::kInt64};
  TilingProgramIRInstruction input;
  input.kind = TilingProgramIRInstructionKind::kSymbol;
  TilingProgramIRInstruction constant;
  TilingProgramIRInstruction add;
  add.kind = TilingProgramIRInstructionKind::kOperation;
  add.opcode = TilingProgramIROpcode::kAdd;
  add.arg_count = 2;
  c.instructions = {input, constant, add};
  std::shared_ptr<const TilingProgramIRCompiledExpression> program;
  if (BuildTilingProgramIRExpression(c, &program).status != TilingExpressionStatus::kOk) {
    return 2;
  }
  TilingProgramIRStageDefinition stage_definition;
  stage_definition.input_types = c.input_types;
  stage_definition.expressions = {c};
  auto next = c;
  next.input_types.push_back(TilingProgramIRValueType::kInt64);
  next.instructions[0].symbol_id = 1;
  stage_definition.expressions.push_back(next);
  std::shared_ptr<const TilingProgramIRCompiledStage> stage;
  if (BuildTilingProgramIRStage(stage_definition, &stage).status != TilingExpressionStatus::kOk) {
    return 3;
  }
  std::array<TilingProgramIRValue, 4> stage_scratch;
  std::array<TilingProgramIRValue, 2> scratch;
  Context context{0, false, 0};
  bool ok = true;
  track = true;
  for (size_t i = 0; i < 2000; ++i) {
    context.value = i;
    context.fail = i % 2 != 0;
    TilingProgramIRValue value{TilingProgramIRValueType::kInt64, 99};
    if (GetTilingExpressionScratchElements(*program) != 2) {
      ok = false;
      break;
    }
    auto e = EvaluateTilingProgramIRExpression(*program, &context, Read, scratch.data(), scratch.size(), &value);
    if (e.status != (context.fail ? TilingExpressionStatus::kInputError : TilingExpressionStatus::kOk) ||
        value.raw_bits != (context.fail ? 99 : i + 1)) {
      ok = false;
      break;
    }
    value.raw_bits = 99;
    if (GetTilingStageScratchElements(*stage) != stage_scratch.size()) {
      ok = false;
      break;
    }
    const auto stage_error =
        EvaluateTilingProgramIRStage(*stage, &context, Read, stage_scratch.data(), stage_scratch.size(), &value);
    if (stage_error.status != (context.fail ? TilingExpressionStatus::kInputError : TilingExpressionStatus::kOk) ||
        value.raw_bits != (context.fail ? 99 : i + 2)) {
      ok = false;
      break;
    }
  }
  track = false;
  std::printf("negative_control=%d allocations=%zu reads=%zu ok=%d\n", hook, allocations, context.calls, ok);
  return hook && allocations == 0 && context.calls == 4000 && ok ? 0 : 1;
}
