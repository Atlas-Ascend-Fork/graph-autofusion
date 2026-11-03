/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- AFIR.h - AFIR dialect declaration -------------------------*- C++ -*-===//
//
// AFIR dialect public header.
//
//===----------------------------------------------------------------------===//

#ifndef AUTOFUSE_MLIR_DIALECT_AFIR_IR_AFIR_H
#define AUTOFUSE_MLIR_DIALECT_AFIR_IR_AFIR_H

#include "dialect/AFIR/IR/AFIRAttributes.h"
#include <limits>
#include <optional>

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/StringRef.h"

// Include dialect definition
#include "AFIR/IR/AFIRDialect.h.inc"

// Include enums (must be before attributes)
#include "AFIR/IR/AFIREnums.h.inc"

// Include type definitions (must be before attributes and ops)
#define GET_TYPEDEF_CLASSES
#include "AFIR/IR/AFIRTypes.h.inc"

// Include attribute definitions
#define GET_ATTRDEF_CLASSES
#include "AFIR/IR/AFIRAttrs.h.inc"

namespace mlir {
namespace afir {

// A definition may represent several (axis, role) bindings when sizes share an SSA value.
SmallVector<AxisBindingAttr> getAxisBindings(Operation *op);
bool hasAxisBinding(Operation *op, int64_t axis, AxisValueRole role);
void addAxisBinding(Operation *op, AxisBindingAttr binding);

// Stable verifier diagnostic categories. The field-specific explanation stays
// at the call site so diagnostics retain the relevant owner and field context.
inline constexpr llvm::StringLiteral kUnsupportedOperationDiagnostic = "UnsupportedOperation";
inline constexpr llvm::StringLiteral kDescriptorTypeMismatchDiagnostic = "DescriptorTypeMismatch";
inline constexpr llvm::StringLiteral kInvalidDescriptorDiagnostic = "InvalidDescriptor";
inline constexpr llvm::StringLiteral kDescriptorCountMismatchDiagnostic = "DescriptorCountMismatch";
inline constexpr llvm::StringLiteral kInvalidBindingDiagnostic = "InvalidBinding";
inline constexpr llvm::StringLiteral kInvalidLifetimeDiagnostic = "InvalidLifetime";
inline constexpr llvm::StringLiteral kInvalidScheduleDiagnostic = "InvalidSchedule";
// 竞争检查的诊断标签（设计文档 5.1 步 8b）。它与上面几条的性质不同：不是 IR 不变量
// 被破坏，而是两次访问之间缺少顺序保证——IR 本身完全合法。
inline constexpr llvm::StringLiteral kDataRaceDiagnostic = "DataRace";

}  // namespace afir
}  // namespace mlir

namespace mlir {
namespace OpTrait {

template <typename ConcreteType>
class AFIRTransferIn : public TraitBase<ConcreteType, AFIRTransferIn> {};

template <typename ConcreteType>
class AFIRTransferOut : public TraitBase<ConcreteType, AFIRTransferOut> {};

template <typename ConcreteType>
class AFIRRejectUnknownAttributes : public TraitBase<ConcreteType, AFIRRejectUnknownAttributes> {
 public:
  static LogicalResult verifyTrait(Operation *op) {
    auto registered = op->getRegisteredInfo();
    if (!registered) {
      return success();
    }
    for (NamedAttribute attribute : op->getDiscardableAttrs()) {
      StringRef name = attribute.getName().getValue();
      if (!isAFIROrUnqualified(name) || isRegistered(*registered, name) || isKnownAFIR(name)) {
        continue;
      }
      return op->emitError() << afir::kUnsupportedOperationDiagnostic << ": unknown AFIR operation attribute '"
                             << attribute.getName() << "'";
    }
    return success();
  }

 private:
  static bool isAFIROrUnqualified(StringRef name) {
    return !name.contains('.') || name.starts_with(afir::kAFIRAttrPrefix);
  }

  static bool isKnownAFIR(StringRef name) {
    static constexpr StringLiteral known[] = {
        afir::kSchedAttrName,           afir::kAxisBindingAttrName,    afir::kSymbolicShapeAttrName,
        afir::kShapeExprAttrName,       afir::kDimSymbolsAttrName,     afir::kVarRelationsAttrName,
        afir::kShapeGuardsAttrName,     afir::kShapeAssertsAttrName,   afir::kBufferDescAttrName,
        afir::kReductionDomainAttrName, afir::kReduceScratchRolesAttr, afir::kReduceScratchScopeAttr};
    return llvm::is_contained(known, name);
  }

  static bool isRegistered(const RegisteredOperationName &registered, StringRef name) {
    StringRef shortName = name;
    if (size_t separator = name.find('.'); separator != StringRef::npos) {
      shortName = name.drop_front(separator + 1);
    }
    return llvm::any_of(registered.getAttributeNames(), [name, shortName](StringAttr registeredName) {
      return registeredName.getValue() == name || registeredName.getValue() == shortName;
    });
  }
};

}  // namespace OpTrait
}  // namespace mlir

#include "AFIR/IR/AFIRInterfaces.h.inc"

// Include operation definitions
#define GET_OP_CLASSES
#include "AFIR/IR/AFIROps.h.inc"

namespace mlir {
namespace afir {

inline FunctionRoleAttr getFunctionRoleAttr(Operation *op) {
  return op->getAttrOfType<FunctionRoleAttr>(kFunctionRoleAttrName);
}

inline bool hasFunctionRole(Operation *op, FunctionRole role) {
  FunctionRoleAttr attr = getFunctionRoleAttr(op);
  return attr && attr.getValue() == role;
}

inline void setFunctionRole(Operation *op, FunctionRole role) {
  op->setAttr(kFunctionRoleAttrName, FunctionRoleAttr::get(op->getContext(), role));
}

// plan module 编译所针对的目标。取到空属性表示「不存在或类型不符」——对调用方是
// 同一件事：这个 module 没说自己是为谁切的。取到非空即意味着已经过 verifier，
// 字段无需再次校验（与 `afplan::getCandidate` 同一约定）。
inline TargetAttr getTargetAttr(Operation *op) {
  return op ? op->getAttrOfType<TargetAttr>(kTargetAttrName) : TargetAttr();
}

// 这个 Op 是不是 AFIR 自己的算子。
//
// 问的是方言归属，不是任何一相分析的事，所以住在这里而不是某个分析的头文件里——
// 循环生成、绑定收集与资源改写都要问这一句，各写一份就会各漂一份（D2）。
inline bool isAfirOperation(Operation *op) {
  Dialect *dialect = op->getDialect();
  return dialect != nullptr && dialect->getNamespace() == AFIRDialect::getDialectNamespace();
}

// 这个 Op 是不是在某个 `afir.vector_func` 的 region 里。
//
// region 里是二次融合的那一段，由 `AFIRVectorFuncToRegBase` 一手下降成寄存器形态，所以
// AFIR 的物化与改写 pass 都不往里走。与 `isAfirOperation` 同住在这里，理由一样：问这一句
// 的不止一处，各写一份就会各漏一处——而漏掉的那一处不报错，只在产物里多出一段东西。
bool isInsideVectorFunc(Operation *op);

//===----------------------------------------------------------------------===//
// Memory space: the address space of a value lives in the value's type
//===----------------------------------------------------------------------===//
//
// 设计文档 3.1 / D8：地址空间是 SSA 值的固有属性，MLIR 里它的标准位置是
// `memref` 自带的 memory space，而不是挂在每个使用点上、按下标对齐的 Op 属性。
// 放在类型上以后，RAUW / canonicalize / `memref.subview` 免费保持它，
// `MemoryAllocate` 也不必再维护 value→descriptor 的侧表来把它查回来。
//
// `#afir.buffer_desc` 的 `space` 字段是迁移期的旧编码。两者同时存在期间，
// `verifyCanonicalBufferDescriptors` 强制二者一致，使两份拷贝不会静默发散（D2）。

// 返回值类型携带的 AFIR 地址空间属性；类型不是 memref、没有 memory space，
// 或 memory space 不是 `#afir.space<*>`（例如后端物化后的 CCE-C 地址空间）时返回空。
inline SpaceAttr getMemorySpaceAttr(Type type) {
  auto memref = dyn_cast<BaseMemRefType>(type);
  if (!memref) {
    return {};
  }
  return dyn_cast_or_null<SpaceAttr>(memref.getMemorySpace());
}

inline std::optional<Space> getMemorySpace(Type type) {
  SpaceAttr attr = getMemorySpaceAttr(type);
  if (!attr) {
    return std::nullopt;
  }
  return attr.getValue();
}

// 在保持 shape / 元素类型 / layout 不变的前提下换上 AFIR 地址空间。
inline MemRefType withMemorySpace(MemRefType type, Space space) {
  return MemRefType::get(type.getShape(), type.getElementType(), type.getLayout(),
                         SpaceAttr::get(type.getContext(), space));
}

//===----------------------------------------------------------------------===//
// Buffer descriptor: the planning result lives at the value's definition
//===----------------------------------------------------------------------===//
//
// 设计文档 D2 / D3 / 3.2 / 5.1 步 8。一块存储的规划结果（空间、字节大小表达式、
// 对齐、资源身份）是**值**的属性，不是使用点的属性。迁移前它是 4 个按下标对齐的
// 数组属性，同一个值在每个使用点各存一份；一致性只能靠消费端把所有拷贝收集起来
// 逐个比对（`CanonicalFunctionMemoryAllocator::recordValue` 正是在做这件事）。
//
// 现在一个值一份，挂在它的定义点上。查找顺序即定义点的三种形态：
//   1. `func.func` 的 block argument -> 该参数的 arg attr；
//   2. destination-style AFIR Op 的结果 -> 顺着 tie 取对应 destination 操作数的那一份
//      （结果与 destination 是同一块存储，迁移前两份 descriptor 必须相等，是纯冗余）；
//   3. 其余定义 Op（`memref.alloc`、tensor 形态的 AFIR Op、物化后的 AscendC Op）
//      -> 该 Op 上的 `afir.buffer_desc`。
//
// 取不到返回空属性：block argument 不属于 `func.func`、定义 Op 没挂、或值来自尚未
// 改造的生产者，都是这种情况，由调用点决定是诊断还是回落。

// AFIR Op 的操作数按 inputs | destinations | tmpBuffers 三段排列。tmp buffer 是被调 API
// 要的临时空间，不描述计算里的任何值：既不进 descriptor 列表，也不参与 tie。
// 凡是要切这三段的地方都必须过这里——各推一次就是各自一个 off-by-one，而其中
// 「把 tmp buffer 当成 destination」这一类的表现是漏报，不是报错。
struct OperandPartition {
  unsigned inputCount = 0;
  unsigned destinationCount = 0;
  unsigned tmpBufferCount = 0;

  OperandRange inputs(Operation *op) const;
  OperandRange destinations(Operation *op) const;
};

// 段长之和超过操作数总数时返回 failure，说明接口实现与 Op 定义对不上。不发诊断，
// 由调用点决定措辞。
FailureOr<OperandPartition> getOperandPartition(Operation *op);

// destination-style Op 的结果与它绑定的 destination 是同一块存储。别名分析与
// descriptor 查询问的是同一件事，所以这条 tie 只写一次。非 tie 的值返回空。
Value getTiedDestination(Value value);

// 按上面那三种定义点形态取值的 buffer descriptor。
BufferDescAttr getBufferDescAttr(Value value);

// 把「取不到」变成一条带位置的诊断。`reporter` 为空时诊断挂在值自身的位置上。
FailureOr<BufferDescAttr> requireBufferDescAttr(Value value, Operation *reporter = nullptr);

// 在值的定义点写入 descriptor。仅用于生产者（bridge 与物化阶段）；对 block argument
// 无效，函数参数由 `func.setArgAttr` 直接写。
void setBufferDescAttr(Operation *definingOp, BufferDescAttr descriptor);

// 函数里是否有任何值携带规划结果。迁移前这个判据是「有没有那 4 个数组属性」，
// 现在是「有没有值在自己的定义点上挂 `afir.buffer_desc`」——参数、结果、函数体
// 里的定义 Op 三处都算。
bool hasBufferDescriptors(Operation *function);

// 同步原语作用的那件全局状态：pipe 之间的执行顺序。它不是任何一个操作数或结果，所以
// 需要一个自己的副作用资源——让「这些 Op 不可删除、不可重排、不可 CSE」成为显式声明，
// 而不是「碰巧没有 MemoryEffect 接口」的副产品。
struct PipelineResource : public SideEffects::Resource::Base<PipelineResource> {
  llvm::StringRef getName() final {
    return "AFIRPipeline";
  }
};

//===----------------------------------------------------------------------===//
// Kernel argument classification: each argument declares its own ABI role
//===----------------------------------------------------------------------===//
//
// 设计文档 3.6「Kernel 程序容器」。身份挂在参数自己的定义点（arg attr）上，一个参数
// 一份；「有几个输入」「workspace 在哪一位」这类问题一律从这组标记数出来，不另存计数、
// 不按下标推断。

// 参数声明的 ABI 身份；没挂返回空属性。空属性不等于「非法」：尚未经过 ABI 物化的
// 中间形态可以整个函数都不带标记，由调用点决定是诊断还是回落。
KernelArgAttr getKernelArgAttr(FunctionOpInterface function, unsigned index);

// 在参数的定义点写入身份。`argIndex` 只对 input / output 有意义，其余角色必须留空，
// 否则 `KernelArgAttr::verify` 会拒绝。
void setKernelArgAttr(FunctionOpInterface function, unsigned index, KernelArgRole role, int64_t argIndex = -1);

// 按角色找参数下标。`workspace` 与 `tiling` 全 kernel 至多一个，这是它们的查询入口；
// 找不到返回 nullopt，由调用点诊断。重复出现时返回第一个——重复本身是 ABI 错误，
// 在 `verifyKernelArgumentRoles` 里报。
std::optional<unsigned> findKernelArgument(FunctionOpInterface function, KernelArgRole role);

// 函数是否已经声明过参数身份。判「ABI 是否已物化」不要用它——那由 `func` 上的
// AscendC 属性回答；这里只回答「这张表在不在」。
bool hasKernelArgRoles(FunctionOpInterface function);

// 一张参数表整体是否自洽：input / output 的序号在各自角色内不重复，workspace 与
// tiling 至多各一个。单个参数的自洽性（角色与序号是否匹配）已由属性 verifier 钉死，
// 这里只查跨参数的那部分。
LogicalResult verifyKernelArgumentRoles(FunctionOpInterface function);

//===----------------------------------------------------------------------===//
// Layout: the memory layout of a value also lives in the value's type
//===----------------------------------------------------------------------===//
//
// 设计文档 3.1 / D8 / ADR-16。与地址空间同理：布局是值的固有属性，`memref` 自带
// layout 扩展点。NZ / fractal 此前表达不了，根因就是这个扩展点没被使用——这是
// Cube 路线的前置阻塞。
//
// 取不到 `#afir.layout` 不等于「没有布局」：null layout 与 identity affine map 都表示
// 连续 ND，是 Vector 路径上的常态。要判「是不是连续 ND」用 `getLayout().isIdentity()`，
// 不要用「有没有 LayoutAttr」。

// 返回值类型携带的 AFIR 布局属性；类型不是 memref，或 layout 不是 `#afir.layout<*>`
// （null、`affine_map<>`、`strided<>` 都算）时返回空。
inline LayoutAttr getLayoutAttr(Type type) {
  auto memref = dyn_cast<MemRefType>(type);
  if (!memref) {
    return {};
  }
  return dyn_cast_or_null<LayoutAttr>(memref.getLayout());
}

// 在保持 shape / 元素类型 / 地址空间不变的前提下换上 AFIR 布局。
inline MemRefType withLayout(MemRefType type, LayoutAttr layout) {
  return MemRefType::get(type.getShape(), type.getElementType(), layout, type.getMemorySpace());
}

}  // namespace afir
}  // namespace mlir

#endif  // AUTOFUSE_MLIR_DIALECT_AFIR_IR_AFIR_H
