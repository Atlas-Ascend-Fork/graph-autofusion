/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "tiling_expression_lowering.h"
#include <utility>
namespace att {
namespace {
using Kind = TilingProgramIRExpressionKind;
using Instruction = TilingProgramIRInstructionKind;
using Op = TilingProgramIROpcode;
using Type = TilingProgramIRValueType;
using Status = TilingExpressionStatus;
using ValueStatus = TilingProgramValueStatus;
TilingExpressionLoweringError Error(Status status, uint32_t node = UINT32_MAX) {
  return {status, node, {}};
}
Status ValueError(ValueStatus status) {
  return status == ValueStatus::kUnsupportedOperation || status == ValueStatus::kUnsupportedType
             ? Status::kUnsupported
             : Status::kInvalidProgram;
}
bool SameNode(const TilingProgramIRExpression &left, const TilingProgramIRExpression &right) {
  return left.kind == right.kind && left.value_type == right.value_type && left.opcode == right.opcode &&
         left.constant_id == right.constant_id && left.symbol_id == right.symbol_id &&
         left.arg_count == right.arg_count && left.args == right.args;
}
TilingExpressionLoweringError ValidateNode(const TilingProgramIRExpressionDefinition &definition, uint32_t id) {
  const auto &node = definition.nodes[id];
  TilingProgramIRExpression canonical;
  canonical.kind = node.kind;
  canonical.value_type = node.value_type;
  Type actual = node.value_type;
  if (ValidateTilingProgramIRValue({node.value_type, 0}) != ValueStatus::kOk) {
    return Error(Status::kUnsupported, id);
  }
  if (node.arg_count > node.args.size()) {
    return Error(Status::kInvalidProgram, id);
  }
  for (uint32_t arg = 0; arg < node.arg_count; ++arg) {
    if (node.args[arg] >= id) {
      return Error(Status::kInvalidProgram, id);
    }
    canonical.args[arg] = node.args[arg];
  }
  auto type = [&](uint32_t arg) { return definition.nodes[node.args[arg]].value_type; };
  switch (node.kind) {
    case Kind::kConstant:
      canonical.constant_id = node.constant_id;
      if (node.constant_id >= definition.constants.size()) {
        return Error(Status::kInvalidProgram, id);
      }
      actual = definition.constants[node.constant_id].type;
      break;
    case Kind::kSymbol:
      canonical.symbol_id = node.symbol_id;
      if (node.symbol_id >= definition.input_types.size()) {
        return Error(Status::kInvalidProgram, id);
      }
      actual = definition.input_types[node.symbol_id];
      break;
    case Kind::kCast:
      canonical.arg_count = 1;
      break;
    case Kind::kConditional:
      canonical.arg_count = kTilingTernaryArity;
      if (node.arg_count != kTilingTernaryArity || type(0) != Type::kBoolean || type(1) != type(2)) {
        return Error(Status::kInvalidProgram, id);
      }
      actual = type(1);
      break;
    case Kind::kOperation: {
      canonical.arg_count = node.arg_count;
      canonical.opcode = node.opcode;
      if (node.arg_count == 0) {
        return Error(Status::kInvalidProgram, id);
      }
      if (node.opcode == Op::kLogicalAnd || node.opcode == Op::kLogicalOr) {
        if (node.arg_count != kTilingBinaryArity || type(0) != Type::kBoolean || type(1) != Type::kBoolean) {
          return Error(Status::kInvalidProgram, id);
        }
        actual = Type::kBoolean;
        break;
      }
      if (node.arg_count == kTilingTernaryArity && type(1) != type(2)) {
        return Error(Status::kInvalidProgram, id);
      }
      const auto status = node.arg_count == 1 ? GetTilingProgramIRUnaryType(node.opcode, type(0), &actual)
                          : node.arg_count == kTilingBinaryArity
                              ? GetTilingProgramIRBinaryType(node.opcode, type(0), type(1), &actual)
                              : GetTilingProgramIRTernaryType(node.opcode, type(0), type(1), &actual);
      if (status != ValueStatus::kOk) {
        return Error(ValueError(status), id);
      }
      break;
    }
    default:
      return Error(Status::kUnsupported, id);
  }
  return actual == node.value_type && SameNode(node, canonical) ? TilingExpressionLoweringError{}
                                                                : Error(Status::kInvalidProgram, id);
}
TilingExpressionLoweringError Validate(const TilingProgramIRExpressionDefinition &definition) {
  const auto count = definition.nodes.size();
  if (count == 0 || count > kTilingExpressionMaxItems || definition.constants.size() > kTilingExpressionMaxItems ||
      definition.input_types.size() > kTilingExpressionMaxItems) {
    return Error(Status::kInvalidProgram);
  }
  for (const auto &value : definition.constants) {
    const auto status = ValidateTilingProgramIRValue(value);
    if (status != ValueStatus::kOk) {
      return Error(ValueError(status));
    }
  }
  for (const auto type : definition.input_types) {
    if (ValidateTilingProgramIRValue({type, 0}) != ValueStatus::kOk) {
      return Error(Status::kUnsupported);
    }
  }
  for (uint32_t id = 0; id < count; ++id) {
    auto error = ValidateNode(definition, id);
    if (error.status != Status::kOk) {
      return error;
    }
  }
  std::vector<bool> reached(count, false);
  reached.back() = true;
  for (size_t id = count; id-- > 0;) {
    if (!reached[id]) {
      return Error(Status::kInvalidProgram, id);
    }
    for (uint32_t arg = 0; arg < definition.nodes[id].arg_count; ++arg) {
      reached[definition.nodes[id].args[arg]] = true;
    }
  }
  return {};
}
struct Frame {
  uint32_t node, next = 0, branch = 0, end = 0;
};
TilingExpressionLoweringError Emit(const TilingProgramIRExpressionDefinition &definition,
                                   TilingProgramIRExpressionCode &code, std::vector<uint32_t> &origins) {
  std::vector<Frame> frames{{static_cast<uint32_t>(definition.nodes.size() - 1)}};
  auto append = [&](TilingProgramIRInstruction instruction, uint32_t node) {
    if (code.instructions.size() >= kTilingExpressionMaxItems) {
      return false;
    }
    code.instructions.push_back(instruction);
    origins.push_back(node);
    return true;
  };
  while (!frames.empty()) {
    auto &frame = frames.back();
    const auto &node = definition.nodes[frame.node];
    const bool logical =
        node.kind == Kind::kOperation && (node.opcode == Op::kLogicalAnd || node.opcode == Op::kLogicalOr);
    TilingProgramIRInstruction instruction;
    if (logical || node.kind == Kind::kConditional) {
      constexpr uint32_t kAfterCondition = 1;
      constexpr uint32_t kAfterTrueBranch = 2;
      constexpr uint32_t kAfterFalseBranch = 3;
      if (frame.next == 0) {
        frame.next = kAfterCondition;
        frames.push_back({node.args[0]});
        continue;
      }
      if (frame.next == kAfterCondition) {
        instruction.kind =
            logical && node.opcode == Op::kLogicalOr ? Instruction::kJumpIfTrue : Instruction::kJumpIfFalse;
        frame.branch = code.instructions.size();
        if (!append(instruction, frame.node)) {
          return Error(Status::kInvalidProgram, frame.node);
        }
        frame.next = kAfterTrueBranch;
        frames.push_back({node.args[1]});
        continue;
      }
      if (frame.next == kAfterTrueBranch) {
        instruction.kind = Instruction::kJump;
        frame.end = code.instructions.size();
        if (!append(instruction, frame.node)) {
          return Error(Status::kInvalidProgram, frame.node);
        }
        code.instructions[frame.branch].jump = code.instructions.size();
        if (!logical) {
          frame.next = kAfterFalseBranch;
          frames.push_back({node.args[2]});
          continue;
        }
        instruction = {};
        instruction.kind = Instruction::kPushBoolean;
        instruction.boolean_value = node.opcode == Op::kLogicalOr;
        if (!append(instruction, frame.node)) {
          return Error(Status::kInvalidProgram, frame.node);
        }
      }
      code.instructions[frame.end].jump = code.instructions.size();
      frames.pop_back();
      continue;
    }
    if (frame.next < node.arg_count) {
      const auto arg = node.args[frame.next++];
      frames.push_back({arg});
      continue;
    }
    switch (node.kind) {
      case Kind::kConstant:
        instruction.constant_id = node.constant_id;
        break;
      case Kind::kSymbol:
        instruction.kind = Instruction::kSymbol;
        instruction.symbol_id = node.symbol_id;
        break;
      case Kind::kCast:
        instruction.kind = Instruction::kCast;
        instruction.value_type = node.value_type;
        break;
      case Kind::kOperation:
        instruction.kind = Instruction::kOperation;
        instruction.opcode = node.opcode;
        instruction.arg_count = node.arg_count;
        break;
      default:
        return Error(Status::kUnsupported, frame.node);
    }
    if (!append(instruction, frame.node)) {
      return Error(Status::kInvalidProgram, frame.node);
    }
    frames.pop_back();
  }
  return {};
}
}  // namespace
TilingExpressionLoweringError LowerTilingProgramIRExpression(const TilingProgramIRExpressionDefinition &definition,
                                                             TilingProgramIRExpressionCode *output) {
  if (output == nullptr) {
    return Error(Status::kInvalidArgument);
  }
  auto error = Validate(definition);
  if (error.status != Status::kOk) {
    return error;
  }
  TilingProgramIRExpressionCode code;
  code.constants = definition.constants;
  code.input_types = definition.input_types;
  code.result_type = definition.nodes.back().value_type;
  std::vector<uint32_t> origins;
  error = Emit(definition, code, origins);
  if (error.status != Status::kOk) {
    return error;
  }
  std::shared_ptr<const TilingProgramIRCompiledExpression> verified;
  const auto code_error = BuildTilingProgramIRExpression(code, &verified);
  if (code_error.status != Status::kOk) {
    const auto node = code_error.instruction < origins.size() ? origins[code_error.instruction] : UINT32_MAX;
    return {code_error.status, node, code_error};
  }
  *output = std::move(code);
  return {};
}
}  // namespace att
