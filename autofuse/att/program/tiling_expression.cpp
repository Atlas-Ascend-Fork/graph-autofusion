/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "tiling_expression.h"
#include <algorithm>
#include <utility>
namespace att {
namespace {
using Kind = TilingProgramIRInstructionKind;
using Type = TilingProgramIRValueType;
using Status = TilingExpressionStatus;
using ValueStatus = TilingProgramValueStatus;
using Stack = std::vector<Type>;
// Bounds follow from the verified control-flow stack and the entry capacity check.
class ValueStack {
 public:
  explicit ValueStack(TilingProgramIRValue *values) : values_(values) {}
  void push_back(const TilingProgramIRValue &value) {
    values_[size_++] = value;
  }
  void pop_back() {
    --size_;
  }
  TilingProgramIRValue &back() {
    return values_[size_ - 1];
  }
  TilingProgramIRValue &operator[](size_t index) {
    return values_[index];
  }
  size_t size() const {
    return size_;
  }

 private:
  TilingProgramIRValue *values_;
  size_t size_ = 0;
};

TilingExpressionError Error(Status status, size_t pc, ValueStatus detail = ValueStatus::kOk) {
  return {status, static_cast<uint32_t>(pc), detail};
}
bool KnownType(Type type) {
  return ValidateTilingProgramIRValue({type, 0}) == ValueStatus::kOk;
}
bool SameInstruction(const TilingProgramIRInstruction &a, const TilingProgramIRInstruction &b) {
  return a.kind == b.kind && a.opcode == b.opcode && a.constant_id == b.constant_id && a.symbol_id == b.symbol_id &&
         a.jump == b.jump && a.arg_count == b.arg_count && a.boolean_value == b.boolean_value &&
         a.value_type == b.value_type;
}
TilingExpressionError Verify(const TilingProgramIRExpressionCode &code, size_t &max_stack) {
  const size_t count = code.instructions.size();
  if (count == 0 || count > kTilingExpressionMaxItems || code.constants.size() > kTilingExpressionMaxItems ||
      code.input_types.size() > kTilingExpressionMaxItems) {
    return Error(Status::kInvalidProgram, UINT32_MAX);
  }
  if (!KnownType(code.result_type)) {
    return Error(Status::kUnsupported, UINT32_MAX);
  }
  for (const auto &v : code.constants) {
    const auto status = ValidateTilingProgramIRValue(v);
    if (status != ValueStatus::kOk) {
      return Error(status == ValueStatus::kUnsupportedType ? Status::kUnsupported : Status::kInvalidProgram, UINT32_MAX,
                   status);
    }
  }
  for (Type type : code.input_types) {
    if (!KnownType(type)) {
      return Error(Status::kUnsupported, UINT32_MAX);
    }
  }
  std::vector<Stack> incoming(count + 1);
  std::vector<bool> reached(count + 1, false);
  reached[0] = true;
  for (size_t pc = 0; pc < count; ++pc) {
    if (!reached[pc]) {
      return Error(Status::kInvalidProgram, pc);
    }
    auto stack = incoming[pc];
    const auto &i = code.instructions[pc];
    TilingProgramIRInstruction canonical;
    canonical.kind = i.kind;
    bool branch = false, unconditional = false;
    switch (i.kind) {
      case Kind::kConstant:
        canonical.constant_id = i.constant_id;
        if (i.constant_id >= code.constants.size()) {
          return Error(Status::kInvalidProgram, pc);
        }
        stack.push_back(code.constants[i.constant_id].type);
        break;
      case Kind::kSymbol:
        canonical.symbol_id = i.symbol_id;
        if (i.symbol_id >= code.input_types.size()) {
          return Error(Status::kInvalidProgram, pc);
        }
        stack.push_back(code.input_types[i.symbol_id]);
        break;
      case Kind::kPushBoolean:
        canonical.boolean_value = i.boolean_value;
        stack.push_back(Type::kBoolean);
        break;
      case Kind::kCast:
        canonical.value_type = i.value_type;
        if (stack.empty()) {
          return Error(Status::kInvalidProgram, pc);
        }
        if (!KnownType(i.value_type)) {
          return Error(Status::kUnsupported, pc);
        }
        stack.back() = i.value_type;
        break;
      case Kind::kOperation: {
        canonical.opcode = i.opcode;
        canonical.arg_count = i.arg_count;
        if (i.arg_count < 1 || i.arg_count > kTilingTernaryArity || stack.size() < i.arg_count ||
            (i.arg_count == kTilingTernaryArity && stack.back() != stack[stack.size() - 2])) {
          return Error(Status::kInvalidProgram, pc);
        }
        Type result;
        const auto status =
            i.arg_count == 1 ? GetTilingProgramIRUnaryType(i.opcode, stack.back(), &result)
            : i.arg_count == kTilingBinaryArity
                ? GetTilingProgramIRBinaryType(i.opcode, stack[stack.size() - 2], stack.back(), &result)
                : GetTilingProgramIRTernaryType(i.opcode, stack[stack.size() - 3], stack.back(), &result);
        if (status != ValueStatus::kOk) {
          return Error(status == ValueStatus::kInvalidArgument ? Status::kInvalidProgram : Status::kUnsupported, pc,
                       status);
        }
        for (uint32_t operand = 1; operand < i.arg_count; ++operand) {
          stack.pop_back();
        }
        stack.back() = result;
        break;
      }
      case Kind::kJumpIfFalse:
      case Kind::kJumpIfTrue:
        if (stack.empty() || stack.back() != Type::kBoolean) {
          return Error(Status::kInvalidProgram, pc);
        }
        stack.pop_back();
        branch = true;
        canonical.jump = i.jump;
        break;
      case Kind::kJump:
        unconditional = true;
        canonical.jump = i.jump;
        break;
      default:
        return Error(Status::kUnsupported, pc);
    }
    if (!SameInstruction(i, canonical) || stack.size() > kTilingExpressionMaxStack) {
      return Error(Status::kInvalidProgram, pc);
    }
    max_stack = std::max(max_stack, stack.size());
    auto propagate = [&](size_t dest) {
      if (dest <= pc || dest > count) {
        return false;
      }
      if (reached[dest]) {
        return incoming[dest] == stack;
      }
      reached[dest] = true;
      incoming[dest] = stack;
      return true;
    };
    if ((branch || unconditional) && !propagate(i.jump)) {
      return Error(Status::kInvalidProgram, pc);
    }
    if (!unconditional && !propagate(pc + 1)) {
      return Error(Status::kInvalidProgram, pc);
    }
  }
  if (!reached[count] || incoming[count] != Stack{code.result_type}) {
    return Error(Status::kInvalidProgram, count);
  }
  return {};
}
}  // namespace
TilingExpressionError BuildTilingProgramIRExpression(const TilingProgramIRExpressionCode &code,
                                                     std::shared_ptr<const TilingProgramIRCompiledExpression> *output) {
  if (output == nullptr) {
    return Error(Status::kInvalidArgument, UINT32_MAX);
  }
  if (code.instructions.size() > kTilingExpressionMaxItems || code.constants.size() > kTilingExpressionMaxItems ||
      code.input_types.size() > kTilingExpressionMaxItems) {
    return Error(Status::kInvalidProgram, UINT32_MAX);
  }
  auto owned =
      std::make_shared<TilingProgramIRCompiledExpression>(TilingProgramIRCompiledExpression::ConstructionKey{}, code);
  auto error = Verify(owned->code_, owned->stack_size_);
  if (error.status == Status::kOk) {
    *output = std::move(owned);
  }
  return error;
}
TilingExpressionError EvaluateTilingProgramIRExpression(const TilingProgramIRCompiledExpression &compiled,
                                                        void *context, TilingExpressionReadSlot reader,
                                                        TilingProgramIRValue *scratch, size_t scratch_elements,
                                                        TilingProgramIRValue *output) {
  if (output == nullptr) {
    return Error(Status::kInvalidArgument, UINT32_MAX);
  }
  if (scratch_elements < compiled.stack_size_) {
    return Error(Status::kBufferTooSmall, UINT32_MAX);
  }
  if (scratch == nullptr) {
    return Error(Status::kInvalidArgument, UINT32_MAX);
  }
  const uintptr_t start = reinterpret_cast<uintptr_t>(scratch);
  const uintptr_t result = reinterpret_cast<uintptr_t>(output);
  const size_t bytes = compiled.stack_size_ * sizeof(TilingProgramIRValue);
  const bool overlap = result >= start ? result - start < bytes : start - result < sizeof(TilingProgramIRValue);
  if (overlap) {
    return Error(Status::kInvalidArgument, UINT32_MAX);
  }
  const auto &code = compiled.code_;
  ValueStack stack(scratch);
  for (size_t pc = 0; pc < code.instructions.size();) {
    const auto &i = code.instructions[pc];
    TilingProgramIRValue result;
    ValueStatus status = ValueStatus::kOk;
    switch (i.kind) {
      case Kind::kConstant:
        stack.push_back(code.constants[i.constant_id]);
        break;
      case Kind::kPushBoolean:
        stack.push_back({Type::kBoolean, i.boolean_value ? 1U : 0U});
        break;
      case Kind::kSymbol:
        if (reader == nullptr) {
          return Error(Status::kInputError, pc, ValueStatus::kInvalidArgument);
        }
        status = reader(context, i.symbol_id, &result);
        if (status == ValueStatus::kOk) {
          status = ValidateTilingProgramIRValue(result);
        }
        if (status != ValueStatus::kOk) {
          return Error(Status::kInputError, pc, status);
        }
        if (result.type != code.input_types[i.symbol_id]) {
          return Error(Status::kInputError, pc, ValueStatus::kInvalidValue);
        }
        stack.push_back(result);
        break;
      case Kind::kCast:
        status = CastTilingProgramIRValue(stack.back(), i.value_type, &result);
        if (status != ValueStatus::kOk) {
          return Error(Status::kEvaluationError, pc, status);
        }
        stack.back() = result;
        break;
      case Kind::kOperation:
        status = i.arg_count == 1 ? EvaluateTilingProgramIRUnary(i.opcode, stack.back(), &result)
                 : i.arg_count == kTilingBinaryArity
                     ? EvaluateTilingProgramIRBinary(i.opcode, stack[stack.size() - 2], stack.back(), &result)
                     : EvaluateTilingProgramIRTernary(i.opcode, stack[stack.size() - 3], stack[stack.size() - 2],
                                                      stack.back(), &result);
        if (status != ValueStatus::kOk) {
          return Error(Status::kEvaluationError, pc, status);
        }
        for (uint32_t operand = 1; operand < i.arg_count; ++operand) {
          stack.pop_back();
        }
        stack.back() = result;
        break;
      case Kind::kJump:
        pc = i.jump;
        continue;
      case Kind::kJumpIfFalse:
      case Kind::kJumpIfTrue: {
        const bool take = i.kind == Kind::kJumpIfFalse ? stack.back().raw_bits == 0 : stack.back().raw_bits != 0;
        stack.pop_back();
        pc = take ? i.jump : pc + 1;
        continue;
      }
      default:
        return Error(Status::kUnsupported, pc);
    }
    ++pc;
  }
  *output = stack.back();
  return {};
}
size_t GetTilingExpressionScratchElements(const TilingProgramIRCompiledExpression &code) {
  return code.stack_size_;
}
}  // namespace att
