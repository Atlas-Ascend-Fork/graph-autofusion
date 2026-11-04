/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "conversion/AFIRToAscendC/AscendCKernelABI.h"
#include "dialect/AFIR/IR/AFIRAttributes.h"

#include "conversion/AFIRToAscendC/AscendCAttributes.h"
#include "dialect/AFIR/IR/AFIRVerifierUtils.h"
#include "dialect/AFIR/IR/SymbolicBuilder.h"
#include "ascir/Dialect/Asc/IR/Asc.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/ErrorHandling.h"

#include <numeric>
#include <optional>
#include <tuple>

using namespace mlir;
using namespace mlir::afir;

// 参数身份的唯一读点。
//
// 此前这里做三件事：查一对**互斥**的整数 arg attr（`afir.input_index` /
// `afir.output_index`，两个都挂是合法 IR，只能在这里事后发现）、校验它们的位宽与
// 非负性、再在两个都没有时按**类型相等**认 workspace。现在这三件事分别落在
// `#afir.kernel_arg` 的类型、它的 verifier、和参数自己的声明上，本函数只做翻译。
FailureOr<mlir::ascend::kernel::KernelArgumentIdentity> mlir::ascend::kernel::getKernelArgumentIdentity(
    func::FuncOp function, unsigned index, StringRef diagnosticPrefix) {
  if (index >= function.getNumArguments()) {
    return function.emitError() << diagnosticPrefix << ": argument index is out of range";
  }
  afir::KernelArgAttr argument = afir::getKernelArgAttr(function, index);
  if (!argument) {
    return function.emitError() << diagnosticPrefix << ": argument " << index << " does not declare '"
                                << mlir::afir::kKernelArgAttrName << "'";
  }
  switch (argument.getRole()) {
    case afir::KernelArgRole::Input:
      return KernelArgumentIdentity{KernelArgumentKind::Input, argument.getIndex()};
    case afir::KernelArgRole::Output:
      return KernelArgumentIdentity{KernelArgumentKind::Output, argument.getIndex()};
    case afir::KernelArgRole::Workspace:
      return KernelArgumentIdentity{KernelArgumentKind::Workspace, -1};
    case afir::KernelArgRole::WorkspaceOffset:
      return KernelArgumentIdentity{KernelArgumentKind::WorkspaceOffset, argument.getIndex()};
    case afir::KernelArgRole::Tiling:
      return KernelArgumentIdentity{KernelArgumentKind::Tiling, -1};
    case afir::KernelArgRole::ScalarInput:
      return KernelArgumentIdentity{KernelArgumentKind::ScalarInput, argument.getIndex()};
  }
  llvm_unreachable("unknown kernel argument role");
}

std::string mlir::ascend::kernel::getKernelArgumentKey(const KernelArgumentIdentity &identity) {
  switch (identity.kind) {
    // 按值标量与 GM 张量共用一个序号空间，所以同一个 `index` 上出现两者是**冲突**，
    // 而不是两个各自合法的参数。同键就是这条事实的唯一编码点：去重表会当场报
    // `duplicate argument identity input:<N>`。
    case KernelArgumentKind::ScalarInput:
    case KernelArgumentKind::Input:
      return "input:" + std::to_string(identity.index);
    case KernelArgumentKind::Output:
      return "output:" + std::to_string(identity.index);
    case KernelArgumentKind::Workspace:
      return "workspace";
    case KernelArgumentKind::WorkspaceOffset:
      return "workspace_offset:" + std::to_string(identity.index);
    case KernelArgumentKind::Tiling:
      return "tiling";
  }
  llvm_unreachable("unknown kernel argument kind");
}

// Builds the permutation that changes discovered argument order to the canonical
// input, output, workspace order. Each result entry stores the old index.
namespace {
// 排序上的秩。`ScalarInput` 折成 `Input`：两者共用一个 index 序号空间，先后完全由 index
// 决定。**分秩会把所有标量整体排到指针形参之后**，与 legacy 按 `input_nodes` 位置交错排
// 的顺序对不上——而图里只有一个标量、且它的 index 恰好最靠后时，分秩与同秩给出的顺序
// 一模一样，所以这条只有「标量夹在两个 Data 中间」的形状才测得出来。
int kernelArgumentRank(mlir::ascend::kernel::KernelArgumentKind kind) {
  using mlir::ascend::kernel::KernelArgumentKind;
  return static_cast<int>(kind == KernelArgumentKind::ScalarInput ? KernelArgumentKind::Input : kind);
}
}  // namespace

SmallVector<unsigned> mlir::ascend::kernel::getKernelArgumentPermutation(ArrayRef<KernelArgumentIdentity> arguments) {
  SmallVector<unsigned> permutation(arguments.size());
  std::iota(permutation.begin(), permutation.end(), 0);
  llvm::stable_sort(permutation, [&](unsigned lhs, unsigned rhs) {
    auto lhsKey = std::make_tuple(kernelArgumentRank(arguments[lhs].kind), arguments[lhs].index);
    auto rhsKey = std::make_tuple(kernelArgumentRank(arguments[rhs].kind), arguments[rhs].index);
    return lhsKey < rhsKey;
  });
  return permutation;
}

Attribute mlir::ascend::kernel::getAscendCMemorySpaceAttr(MLIRContext *context, Space space) {
  std::optional<ascendc::AddressSpace> addressSpace;
  switch (space) {
    case Space::GM:
      addressSpace = ascendc::AddressSpace::gm;
      break;
    case Space::UB:
      addressSpace = ascendc::AddressSpace::ubuf;
      break;
    case Space::L1:
      addressSpace = ascendc::AddressSpace::cbuf;
      break;
    case Space::L0A:
      addressSpace = ascendc::AddressSpace::ca;
      break;
    case Space::L0B:
      addressSpace = ascendc::AddressSpace::cb;
      break;
    case Space::L0C:
      addressSpace = ascendc::AddressSpace::cc;
      break;
    case Space::L2:
      // L2 是 cache，不是 CCE-C 的地址空间：没有可映射的编码。
      break;
  }
  if (!addressSpace) {
    return {};
  }
  return IntegerAttr::get(IntegerType::get(context, 32), static_cast<int64_t>(*addressSpace));
}

Attribute mlir::ascend::kernel::getAscendCGlobalMemorySpaceAttr(MLIRContext *context) {
  return getAscendCMemorySpaceAttr(context, Space::GM);
}

MemRefType mlir::ascend::kernel::getAscendCGlobalMemRefType(RankedTensorType tensorType) {
  return MemRefType::get(tensorType.getShape(), tensorType.getElementType(), MemRefLayoutAttrInterface{},
                         getAscendCGlobalMemorySpaceAttr(tensorType.getContext()));
}

MemRefType mlir::ascend::kernel::getAscendCWorkspaceType(MLIRContext *context) {
  Type ui8 = IntegerType::get(context, 8, IntegerType::Unsigned);
  return MemRefType::get({}, ui8, MemRefLayoutAttrInterface{}, getAscendCGlobalMemorySpaceAttr(context));
}

IntegerType mlir::ascend::kernel::getAscendCWorkspaceOffsetType(MLIRContext *context) {
  return IntegerType::get(context, 64, IntegerType::Unsigned);
}

Type mlir::ascend::kernel::getAscendCAbiArgumentType(Type type) {
  auto memref = dyn_cast<MemRefType>(type);
  if (!memref) {
    return type;
  }
  return UnrankedMemRefType::get(memref.getElementType(), memref.getMemorySpace());
}

Type mlir::ascend::kernel::getAscendCTilingDataType(func::FuncOp func) {
  MLIRContext *context = func.getContext();
  Type i64 = IntegerType::get(context, 64);
  SmallVector<Attribute> fieldTypes;
  SmallVector<Attribute> fieldNames;
  llvm::StringSet<> seen;
  auto appendField = [&](StringRef name) {
    if (!seen.insert(name).second) {
      return;
    }
    fieldTypes.push_back(TypeAttr::get(i64));
    fieldNames.push_back(StringAttr::get(context, name));
  };

  if (auto graph = func->getAttrOfType<AscGraphAttrGroupsAttr>(mlir::afir::kAscGraphAttrName)) {
    appendField("block_dim");
    for (ExprAttr sizeVariable : graph.getSizeVar()) {
      appendField(sizeVariable.getValue().getValue());
    }
  } else {
    appendField("tile_len");
    appendField("total");
  }
  return mlir::ascend::kernel::TilingSchema::get(context, StringAttr::get(context, "TilingData"),
                                                 ArrayAttr::get(context, fieldTypes),
                                                 ArrayAttr::get(context, fieldNames));
}

// 绑定的是 `afir.tiling_param`，不是 `emitasc.member`。
//
// 此前这里当场建一个 `emitasc.member` —— 一个**后端**算子，而本函数服务的是循环生成，
// 那还是 AFIR 阶段。于是 `--afir-loop-generate` 之后的函数体里就躺着一个 emitasc 算子，
// 判据 C3（函数体里只出现 AFIR 与它复用的标准方言）当场不成立。现在取值这件事由 AFIR
// 自己的 `afir.tiling_param` 表达，`emitasc.member` 只在 lowering 里出现一次。
//
// 保留的是**校验**：字段在不在 TilingData 的 schema 里、类型是不是整数，这两条属于
// ABI，而本文件就是 ABI 的所在地。挪走的只是「建后端算子」这一个动作。
//
// `builder` 因此不再被用到：`afir.tiling_param` 的位置由 `getOrCreateTilingParam` 统一
// 定在入口块顶部（见 SymbolicBuilder.h 里为什么必须是那里），调用点的插入点无从参与。
FailureOr<mlir::afir::SymbolBindingMap> mlir::ascend::kernel::materializeAscendCTilingDataBindings(
    OpBuilder &, func::FuncOp func, ArrayRef<ExprAttr> sizeVariables) {
  FailureOr<Value> tiling = getCanonicalTilingArgument(func);
  if (failed(tiling)) {
    return failure();
  }
  // 「哪个参数是 tiling」由 `#afir.kernel_arg<role = tiling>` 回答（getCanonicalTilingArgument），
  // 这里只要它的类型确实是个 tiling schema 结构体。**不看结构体叫什么名字**：`GenTilingData`
  // 会把它改名成 `<module>TilingData`，靠名字认就等于给整条流水线加一条隐式排序约束，而
  // 违反它的失败形态是「找不到 tiling 参数」，不是「顺序错了」。
  TilingSchema type = getTilingSchema(tiling->getType());
  if (!type) {
    return failure();
  }

  SymbolBindingMap bindings;
  ArrayAttr names = type.getNamesAttr();
  ArrayAttr types = type.getTypesAttr();
  for (auto [nameAttr, typeAttr] : llvm::zip(names, types)) {
    auto name = cast<StringAttr>(nameAttr);
    auto fieldType = cast<TypeAttr>(typeAttr);
    if (!isa<IntegerType, IndexType>(fieldType.getValue())) {
      return func.emitError("TilingData axis fields must have integer types");
    }
    auto sizeVariable = llvm::find_if(
        sizeVariables, [&](ExprAttr expression) { return expression.getValue().getValue() == name.getValue(); });
    if (sizeVariable == sizeVariables.end()) {
      continue;
    }
    bindings.insert({*sizeVariable, afir::getOrCreateTilingParam(func.front(), name.getValue()).getResult()});
  }
  for (ExprAttr sizeVariable : sizeVariables) {
    if (!bindings.count(sizeVariable)) {
      return func.emitError("TilingData is missing graph size variable '") << sizeVariable.getValue().getValue() << "'";
    }
  }
  return bindings;
}

// AscendC kernel ABI 的 TilingData 实参。
//
// 此前这里靠**位置 + 形状**反推：倒数第二个参数类型恰好等于 `getAscendCWorkspaceType`
// 就算 workspace，倒数第一个参数是名字叫 "TilingData" 的结构体就算 tiling。两条规则
// 都不在任何对外文档里，而且一个碰巧同型的普通 GM 参数排在倒数第二位就会被当成
// workspace。现在身份由参数自己声明（`#afir.kernel_arg`），位置与类型降级为**对声明的
// 校验**：顺序仍是 AscendC 生成的 C++ 签名要求的，但它是被检查的约定，不是推断依据。
FailureOr<Value> mlir::ascend::kernel::getCanonicalTilingArgument(func::FuncOp func) {
  if (func.isExternal() || func.front().getNumArguments() != func.getNumArguments()) {
    func.emitError("canonical AscendC kernel ABI requires a function body with matching arguments");
    return failure();
  }
  if (failed(afir::verifyKernelArgumentRoles(func))) {
    return failure();
  }
  std::optional<unsigned> workspaceIndex = afir::findKernelArgument(func, afir::KernelArgRole::Workspace);
  std::optional<unsigned> tilingIndex = afir::findKernelArgument(func, afir::KernelArgRole::Tiling);
  if (!workspaceIndex || !tilingIndex) {
    func.emitError("canonical AscendC kernel ABI requires workspace and TilingData arguments");
    return failure();
  }
  // 位置约定：`..., workspace, workspace_offset*, tiling`。workspace 与 tiling 之间**只**允许
  // workspace 偏移参数（可以没有），与 legacy 的
  // `(..., GM_ADDR workspace, uint64_t workspace<ID>..., const T *t)` 同形。
  bool tailIsCanonical = *tilingIndex + 1 == func.getNumArguments() && *workspaceIndex < *tilingIndex;
  for (unsigned index = *workspaceIndex + 1; tailIsCanonical && index < *tilingIndex; ++index) {
    afir::KernelArgAttr argument = afir::getKernelArgAttr(func, index);
    tailIsCanonical = argument && argument.getRole() == afir::KernelArgRole::WorkspaceOffset;
  }
  if (!tailIsCanonical) {
    func.emitError(
        "canonical AscendC kernel ABI requires the workspace, its offsets and the TilingData argument "
        "last, in that order");
    return failure();
  }
  Value workspace = func.getArgument(*workspaceIndex);
  if (workspace.getType() != getAscendCWorkspaceType(func.getContext())) {
    func.emitError("canonical AscendC kernel ABI has an invalid workspace argument");
    return failure();
  }
  Value tiling = func.getArgument(*tilingIndex);
  if (!getTilingSchema(tiling.getType())) {
    func.emitError("canonical AscendC kernel ABI requires the tiling argument to be a tiling schema struct");
    return failure();
  }
  return tiling;
}

namespace {

LogicalResult verifyCanonicalScalarArgument(func::FuncOp func, unsigned index, TensorDescAttr tensor,
                                            Attribute declaredBuffer) {
  if (isa<ShapedType>(func.getArgumentTypes()[index])) {
    return func.emitError("canonical kernel argument ")
           << index << " declares the scalar_input role and cannot be a shaped type";
  }
  if (!tensor || tensor.getRole().getValue() != Role::Input) {
    return func.emitError("canonical kernel argument ") << index << " must have input role";
  }
  if (declaredBuffer) {
    return func.emitError("canonical kernel argument ")
           << index << " declares the scalar_input role and cannot carry '" << kBufferDescAttrName << "'";
  }
  return success();
}

LogicalResult verifyCanonicalMemrefArgument(func::FuncOp func, unsigned index, TensorDescAttr tensor,
                                            Attribute declaredBuffer, BufferDescAttr buffer, MemRefType memref) {
  if (!memref) {
    return func.emitError("canonical kernel argument ") << index << " must be a memref";
  }
  // 写错类型与压根没写是两回事：前者说明有人在参数的定义点上放了别的东西，报出来要
  // 指向类型本身，而不是含糊地说「不是外部 GM 存储」。
  if (declaredBuffer && !buffer) {
    return func.emitError("canonical kernel argument ")
           << index << " requires a BufferDescAttr at '" << kBufferDescAttrName << "'";
  }
  if (!buffer || buffer.getSpace().getValue() != Space::GM) {
    return func.emitError("canonical kernel argument ") << index << " must describe external GM storage";
  }
  if (!tensor || (tensor.getRole().getValue() != Role::Input && tensor.getRole().getValue() != Role::Output)) {
    return func.emitError("canonical kernel argument ") << index << " must have input or output role";
  }
  if (buffer.getSlotCount() != -1 || buffer.getQueueDepth() != -1) {
    return func.emitError("canonical kernel argument ") << index << " cannot carry queue fields";
  }
  // kernel 参数的地址空间在这条流水线上有两种合法形态：AFIR 语义层的
  // `#afir.space<gm>`（步 3 起由 bridge 写在类型上），以及 `rewriteCanonicalSignature`
  // 物化后的 AscendC CCE-C 地址空间。本函数在物化前后都会被调用，两种都要放行；
  // 除此之外的任何地址空间都是冲突。
  if (Attribute memorySpace = memref.getMemorySpace()) {
    bool isAfirGlobal = afir::getMemorySpace(memref) == Space::GM;
    bool isMaterializedGlobal = memorySpace == mlir::ascend::kernel::getAscendCGlobalMemorySpaceAttr(func.getContext());
    if (!isAfirGlobal && !isMaterializedGlobal) {
      return func.emitError("canonical kernel argument ") << index << " has a conflicting memory space";
    }
  }
  // kernel 参数是主机侧交进来的 GM 张量，AscendC ABI 按连续 ND 解释它们。步 6 让
  // `memref` 能表达 NZ / fractal 之后，这条闸门保证这些片上布局不会无声地漏到 kernel
  // 边界上——那里没有对应的 ABI 约定，错了也不会报错，只会算错。Cube 路线打通 GM 侧
  // NZ 输入时，这里要随之放开并补上边界上的布局约定。
  if (!memref.getLayout().isIdentity()) {
    return func.emitError("canonical kernel argument ")
           << index << " must have an identity layout at the AscendC ABI boundary, got " << memref.getLayout();
  }
  return success();
}

}  // namespace

LogicalResult mlir::ascend::kernel::verifyCanonicalKernelArguments(func::FuncOp func, unsigned dataArgumentCount) {
  auto tensors = func->getAttrOfType<ArrayAttr>(kArgDescsAttrName);
  if (dataArgumentCount > func.getNumArguments() || !tensors || tensors.size() != dataArgumentCount) {
    return func.emitError("canonical argument descriptor arrays must match the external data arguments");
  }
  for (unsigned index = 0; index < dataArgumentCount; ++index) {
    auto tensor = dyn_cast<TensorDescAttr>(tensors[index]);
    Attribute declaredBuffer = func.getArgAttr(index, kBufferDescAttrName);
    auto buffer = dyn_cast_or_null<BufferDescAttr>(declaredBuffer);
    auto memref = dyn_cast<MemRefType>(func.getArgumentTypes()[index]);
    // 角色与类型是一条**双向**不变量：`scalar_input` 的参数不能是 shaped 类型，其余数据
    // 角色的参数必须是 memref。只判一个方向不够——只放行不回拦的话，一个本该是 GM 张量
    // 的参数被写成标量类型时会被静默收下，而没有任何用例会红。
    const bool declaresScalar =
        getKernelArgAttr(func, index) && getKernelArgAttr(func, index).getRole() == KernelArgRole::ScalarInput;
    if (declaresScalar) {
      if (failed(verifyCanonicalScalarArgument(func, index, tensor, declaredBuffer))) {
        return failure();
      }
      continue;
    }
    if (failed(verifyCanonicalMemrefArgument(func, index, tensor, declaredBuffer, buffer, memref))) {
      return failure();
    }
  }
  return success();
}

static LogicalResult verifyMaterializedDataArguments(func::FuncOp func, unsigned dataArgumentCount) {
  for (auto [index, type] : llvm::enumerate(func.getArgumentTypes().take_front(dataArgumentCount))) {
    // 按值标量形参（`ScalarData`）不是 GM 存储。放行按**角色**判而不是按类型判：否则
    // 一个漏标了角色、类型又碰巧不是 memref 的参数会被当成标量收下。
    afir::KernelArgAttr argument = afir::getKernelArgAttr(func, index);
    if (argument && argument.getRole() == afir::KernelArgRole::ScalarInput) {
      continue;
    }
    // 收 `BaseMemRefType` 而不是 `MemRefType`：判据是「必须是 GM 存储」，秩不在其中。
    // template 保留 ranked 视图，Plan 与聚合入口使用 unranked ABI 类型；两者都表示合法的
    // GM 实参，发射出来仍是 `__gm__ float*`。
    auto memref = dyn_cast<BaseMemRefType>(type);
    if (!memref ||
        memref.getMemorySpace() != mlir::ascend::kernel::getAscendCGlobalMemorySpaceAttr(func.getContext())) {
      return func.emitError("materialized kernel argument ") << index << " must be a GM memref";
    }
  }
  return success();
}

FailureOr<bool> mlir::ascend::kernel::checkAscendCKernelABI(func::FuncOp func, FunctionRole role) {
  FailureOr<bool> aiCoreAttrState = hasAscendCAiCoreAttr(func);
  FailureOr<IntegerAttr> backendInputCountAttr = getAscendCNumInputs(func);
  FailureOr<bool> globalAttrState = hasAscendCGlobalAttr(func);
  if (failed(aiCoreAttrState) || failed(backendInputCountAttr) || failed(globalAttrState)) {
    func.emitError("malformed AscendC kernel attributes");
    return failure();
  }
  bool hasAiCore = *aiCoreAttrState;
  bool hasBackendInputCount = static_cast<bool>(*backendInputCountAttr);
  bool hasGlobal = *globalAttrState;
  if (!hasAiCore && !hasBackendInputCount && !hasGlobal) {
    return false;
  }
  if (!hasAiCore || !hasBackendInputCount || !afir::hasKernelArgRoles(func)) {
    func.emitError("partially materialized AscendC kernel attributes");
    return failure();
  }
  // ABI 物化前有几个参数，此前在 `afir.num_inputs` 与 `ascendc.num_inputs` 里各存一份，
  // 这里把两份比对一遍（D2：同一事实的第二份编码，连校验都要多写一道）。AFIR 侧那份
  // 已经删掉——它记的是后端 ABI 的形参个数，本来就该由后端属性单独持有；AFIR 侧要知道
  // 「谁是什么」，问 `#afir.kernel_arg` 即可，不需要计数。
  int64_t backendInputCount = backendInputCountAttr->getInt();
  if (!backendInputCountAttr->getType().isSignlessInteger(32) || backendInputCount < 0 ||
      static_cast<uint64_t>(backendInputCount) + 2 > func.getNumArguments() || func.getNumResults() != 0) {
    func.emitError("invalid materialized AscendC kernel signature");
    return failure();
  }

  if (failed(getCanonicalTilingArgument(func))) {
    return failure();
  }
  const unsigned dataArgumentCount = func.getNumArguments() - 2;
  if (failed(verifyMaterializedDataArguments(func, dataArgumentCount))) {
    return failure();
  }
  if (afir::hasBufferDescriptors(func.getOperation())) {
    // Canonical buffer form counts inputs and output destinations together.
    // Tensor form retains its original input count, with output arguments after it.
    if (static_cast<uint64_t>(backendInputCount) != dataArgumentCount) {
      return func.emitError("canonical kernel argument count must include every external data argument");
    }
    if (failed(verifyCanonicalKernelArguments(func, dataArgumentCount))) {
      return failure();
    }
  }

  bool expectsGlobal = role == FunctionRole::KernelEntry;
  if (hasGlobal != expectsGlobal) {
    func.emitError("materialized AscendC kernel attributes do not match AFIR function role");
    return failure();
  }
  return true;
}

void mlir::ascend::kernel::setAscendCKernelAttributes(func::FuncOp func, FunctionRole role, unsigned inputCount) {
  setAscendCAiCore(func);
  setAscendCNumInputs(func, inputCount);
  setAscendCGlobal(func, role == FunctionRole::KernelEntry);
}
