/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- AfirBuilder.h - Build an AFIR module from neutral schedule info -*- C++ -*-===//
//
// Autofuse MLIR schedule-to-AFIR bridge core.
//
// This is the *backend-neutral* half of the schedule-to-plan bridge: it builds
// an AFIR MLIR module from plain structs (no autofuse backend types). The
// in-process adapter (ScheduleToAfir, compiled into the codegen lib) fills
// these structs directly from one ascir::ScheduledResult (a plan) and calls
// BuildAfirModule — zero serialization, all in memory.
//
// Keeping the MLIR-construction logic here (depending only on MLIR + the AFIR
// dialect) makes it buildable/testable in the standalone mlir/ build, and is
// the piece codegen-migration engineers actually need to learn.
//
//===----------------------------------------------------------------------===//

#ifndef AUTOFUSE_MLIR_BRIDGE_AFIRBUILDER_H
#define AUTOFUSE_MLIR_BRIDGE_AFIRBUILDER_H

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "dialect/AFIR/IR/AFIR.h"
#include "dialect/AFIR/IR/ReduceScratch.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace afir {
namespace bridge {

// Shared dtype conversion for AFIR values and their GM view byte sizes.
Type parseDtype(MLIRContext *ctx, llvm::StringRef dtype);

// Which storage forms of an AFIR operation the bridge can build. The production
// path (ScheduleToAfir) only ever asks for Buffer; Tensor is the legacy DTO path
// and the af-afir-gen demo, so a Tensor-only operation is not yet usable in a
// generated kernel.
bool isReduceNodeType(llvm::StringRef type);

enum class BridgeOpForm : uint8_t {
  Tensor,
  Buffer,
  Both,
};

struct AFIRBridgeOpMapping {
  llvm::StringRef scheduler_name;
  llvm::StringRef afir_operation_name;
  BridgeOpForm form;
};

// Return the scheduler compute nodes materialized as AFIR operations by
// BuildAfirModule. This is the source of truth for the bridgeable operation set,
// its `form` column included.
llvm::ArrayRef<AFIRBridgeOpMapping> getAFIRBridgeableOperations();

// 这张表在头文件里而不是实现文件里，是因为它是**唯一一份可遍历的 op 类名单**。
// `getAFIRBridgeableOperations()` 只给得出名字，给不出类，所以凡是要对每个 op 调一个
// **静态成员**的断言（`OdsElementTypesMatchRegistrations` 要调
// `getOdsElementTypeSpellings()`）都只能展开这张表。放在实现文件里的话，那种断言就得自己
// 再抄一份 op 类名单——而那份名单漏掉一个 op 的后果是那个 op 不被查，静默。
// The single source of truth for what the Schedule bridge can build, in three
// columns: the scheduler's node type, the AFIR operation, and which storage
// forms that operation can be built in.
//
// The form column matters because the two forms are not interchangeable: the
// production path (ScheduleToAfir) always asks for the buffer form, while the
// tensor form only survives for the legacy DTO path and the af-afir-gen demo.
// An operation listed as `Tensor` is therefore *not* usable in production, and
// before this column existed that distinction lived in a second hand-maintained
// StringSwitch that had already drifted from this table.
//
// Adding an operation means: a row here, an ODS definition, and an AscendC
// implementation registration -- three places, checked by AfirBuilderTest.
#define AFIR_BRIDGE_OPERATION_REGISTRY(M)                                                   \
  M("Load", afir::LoadOp, BridgeOpForm::Both)                                               \
  M("Store", afir::StoreOp, BridgeOpForm::Both)                                             \
  M("Nddma", afir::NddmaOp, BridgeOpForm::Both)                                             \
  M("Broadcast", afir::BroadcastOp, BridgeOpForm::Both)                                     \
  M("RemovePad", afir::RemovePadOp, BridgeOpForm::Both)                                     \
  M("Transpose", afir::TransposeOp, BridgeOpForm::Both)                                     \
  M("Pad", afir::PadOp, BridgeOpForm::Both)                                                 \
  M("Cast", afir::CastOp, BridgeOpForm::Both)                                               \
  M("Add", afir::AddOp, BridgeOpForm::Both)                                                 \
  M("Sub", afir::SubOp, BridgeOpForm::Both)                                                 \
  M("Mul", afir::MulOp, BridgeOpForm::Both)                                                 \
  M("Maximum", afir::MaximumOp, BridgeOpForm::Both)                                         \
  M("Minimum", afir::MinimumOp, BridgeOpForm::Both)                                         \
  M("Div", afir::DivOp, BridgeOpForm::Both)                                                 \
  M("Abs", afir::AbsOp, BridgeOpForm::Both)                                                 \
  M("Exp", afir::ExpOp, BridgeOpForm::Both)                                                 \
  M("Ln", afir::LnOp, BridgeOpForm::Both)                                                   \
  M("Relu", afir::ReluOp, BridgeOpForm::Both)                                               \
  M("Sqrt", afir::SqrtOp, BridgeOpForm::Both)                                               \
  M("Neg", afir::NegOp, BridgeOpForm::Both)                                                 \
  M("Reciprocal", afir::ReciprocalOp, BridgeOpForm::Both)                                   \
  M("Sigmoid", afir::SigmoidOp, BridgeOpForm::Both)                                         \
  M("Sum", afir::SumOp, BridgeOpForm::Both)                                                 \
  M("Max", afir::MaxOp, BridgeOpForm::Both)                                                 \
  M("Min", afir::MinOp, BridgeOpForm::Both)                                                 \
  M("Mean", afir::MeanOp, BridgeOpForm::Both)                                               \
  M("Prod", afir::ProdOp, BridgeOpForm::Both)                                               \
  M("Any", afir::AnyOp, BridgeOpForm::Both)                                                 \
  M("All", afir::AllOp, BridgeOpForm::Both)                                                 \
  M("Isnan", afir::IsnanOp, BridgeOpForm::Both)                                             \
  M("IsFinite", afir::IsFiniteOp, BridgeOpForm::Both)                                       \
  M("IsInf", afir::IsInfOp, BridgeOpForm::Both)                                             \
  M("Select", afir::SelectOp, BridgeOpForm::Both)                                           \
  M("Where", afir::WhereOp, BridgeOpForm::Both)                                             \
  M("ClipByValue", afir::ClipByValueOp, BridgeOpForm::Both)                                 \
  M("LeakyRelu", afir::LeakyReluOp, BridgeOpForm::Both)                                     \
  M("BitwiseAnd", afir::BitwiseAndOp, BridgeOpForm::Both)                                   \
  M("FloorDiv", afir::FloorDivOp, BridgeOpForm::Both)                                       \
  M("Remainder", afir::RemainderOp, BridgeOpForm::Both)                                     \
  M("Gelu", afir::GeluOp, BridgeOpForm::Both)                                               \
  M("Sign", afir::SignOp, BridgeOpForm::Both)                                               \
  M("LogicalAnd", afir::LogicalAndOp, BridgeOpForm::Both)                                   \
  M("LogicalOr", afir::LogicalOrOp, BridgeOpForm::Both)                                     \
  M("Axpy", afir::AxpyOp, BridgeOpForm::Both)                                               \
  M("TrueDiv", afir::TrueDivOp, BridgeOpForm::Both)                                         \
  M("Tanh", afir::TanhOp, BridgeOpForm::Both)                                               \
  M("Erf", afir::ErfOp, BridgeOpForm::Both)                                                 \
  M("Pow", afir::PowOp, BridgeOpForm::Both)                                                 \
  M("Rsqrt", afir::RsqrtOp, BridgeOpForm::Both)                                             \
  M("Ndtr", afir::NdtrOp, BridgeOpForm::Both)                                               \
  M("Ndtri", afir::NdtriOp, BridgeOpForm::Both)                                             \
  M("LogNdtr", afir::LogNdtrOp, BridgeOpForm::Both)                                         \
  M("I0", afir::I0Op, BridgeOpForm::Both)                                                   \
  M("AiryAi", afir::AiryAiOp, BridgeOpForm::Both)                                           \
  M("BesselJ0", afir::BesselJ0Op, BridgeOpForm::Both)                                       \
  M("BesselJ1", afir::BesselJ1Op, BridgeOpForm::Both)                                       \
  M("I0e", afir::I0eOp, BridgeOpForm::Both)                                                 \
  M("I1e", afir::I1eOp, BridgeOpForm::Both)                                                 \
  M("ModifiedBesselI0", afir::ModifiedBesselI0Op, BridgeOpForm::Both)                       \
  M("ModifiedBesselI1", afir::ModifiedBesselI1Op, BridgeOpForm::Both)                       \
  M("ModifiedBesselK0", afir::ModifiedBesselK0Op, BridgeOpForm::Both)                       \
  M("ModifiedBesselK1", afir::ModifiedBesselK1Op, BridgeOpForm::Both)                       \
  M("Erfinv", afir::ErfinvOp, BridgeOpForm::Both)                                           \
  M("Exp2", afir::Exp2Op, BridgeOpForm::Both)                                               \
  M("Log2", afir::Log2Op, BridgeOpForm::Both)                                               \
  M("Floor", afir::FloorOp, BridgeOpForm::Both)                                             \
  M("Lgamma", afir::LgammaOp, BridgeOpForm::Both)                                           \
  M("LogicalXor", afir::LogicalXorOp, BridgeOpForm::Both)                                   \
  M("Fmod", afir::FmodOp, BridgeOpForm::Both)                                               \
  M("Mod", afir::ModOp, BridgeOpForm::Both)                                                 \
  M("NextAfter", afir::NextAfterOp, BridgeOpForm::Both)                                     \
  M("Acos", afir::AcosOp, BridgeOpForm::Both)                                               \
  M("Acosh", afir::AcoshOp, BridgeOpForm::Both)                                             \
  M("Asin", afir::AsinOp, BridgeOpForm::Both)                                               \
  M("Asinh", afir::AsinhOp, BridgeOpForm::Both)                                             \
  M("Atan", afir::AtanOp, BridgeOpForm::Both)                                               \
  M("Atanh", afir::AtanhOp, BridgeOpForm::Both)                                             \
  M("Ceil", afir::CeilOp, BridgeOpForm::Both)                                               \
  M("Cos", afir::CosOp, BridgeOpForm::Both)                                                 \
  M("Cosh", afir::CoshOp, BridgeOpForm::Both)                                               \
  M("Erfcx", afir::ErfcxOp, BridgeOpForm::Both)                                             \
  M("Round", afir::RoundOp, BridgeOpForm::Both)                                             \
  M("Sin", afir::SinOp, BridgeOpForm::Both)                                                 \
  M("Sinh", afir::SinhOp, BridgeOpForm::Both)                                               \
  M("Tan", afir::TanOp, BridgeOpForm::Both)                                                 \
  M("BesselY0", afir::BesselY0Op, BridgeOpForm::Both)                                       \
  M("BesselY1", afir::BesselY1Op, BridgeOpForm::Both)                                       \
  M("SphericalBesselJ0", afir::SphericalBesselJ0Op, BridgeOpForm::Both)                     \
  M("ScaledModifiedBesselK0", afir::ScaledModifiedBesselK0Op, BridgeOpForm::Both)           \
  M("ScaledModifiedBesselK1", afir::ScaledModifiedBesselK1Op, BridgeOpForm::Both)           \
  M("Digamma", afir::DigammaOp, BridgeOpForm::Both)                                         \
  M("Erfc", afir::ErfcOp, BridgeOpForm::Both)                                               \
  M("Trunc", afir::TruncOp, BridgeOpForm::Both)                                             \
  M("Expm1", afir::Expm1Op, BridgeOpForm::Both)                                             \
  M("Log1p", afir::Log1pOp, BridgeOpForm::Both)                                             \
  M("CopySign", afir::CopySignOp, BridgeOpForm::Both)                                       \
  M("TruncDiv", afir::TruncDivOp, BridgeOpForm::Both)                                       \
  M("Atan2", afir::Atan2Op, BridgeOpForm::Both)                                             \
  M("Zeta", afir::ZetaOp, BridgeOpForm::Both)                                               \
  M("Igamma", afir::IgammaOp, BridgeOpForm::Both)                                           \
  M("Igammac", afir::IgammacOp, BridgeOpForm::Both)                                         \
  M("LogicalNot", afir::LogicalNotOp, BridgeOpForm::Both)                                   \
  M("Eq", afir::EqOp, BridgeOpForm::Both)                                                   \
  M("Ne", afir::NeOp, BridgeOpForm::Both)                                                   \
  M("Lt", afir::LtOp, BridgeOpForm::Both)                                                   \
  M("Le", afir::LeOp, BridgeOpForm::Both)                                                   \
  M("Gt", afir::GtOp, BridgeOpForm::Both)                                                   \
  M("Ge", afir::GeOp, BridgeOpForm::Both)                                                   \
  M("Log10", afir::Log10Op, BridgeOpForm::Both)                                             \
  M("Hypot", afir::HypotOp, BridgeOpForm::Both)                                             \
  M("ChebyshevPolynomialT", afir::ChebyshevPolynomialTOp, BridgeOpForm::Both)               \
  M("ChebyshevPolynomialU", afir::ChebyshevPolynomialUOp, BridgeOpForm::Both)               \
  M("ChebyshevPolynomialV", afir::ChebyshevPolynomialVOp, BridgeOpForm::Both)               \
  M("ChebyshevPolynomialW", afir::ChebyshevPolynomialWOp, BridgeOpForm::Both)               \
  M("ShiftedChebyshevPolynomialT", afir::ShiftedChebyshevPolynomialTOp, BridgeOpForm::Both) \
  M("ShiftedChebyshevPolynomialU", afir::ShiftedChebyshevPolynomialUOp, BridgeOpForm::Both) \
  M("ShiftedChebyshevPolynomialV", afir::ShiftedChebyshevPolynomialVOp, BridgeOpForm::Both) \
  M("ShiftedChebyshevPolynomialW", afir::ShiftedChebyshevPolynomialWOp, BridgeOpForm::Both) \
  M("Ceil2Int", afir::Ceil2IntOp, BridgeOpForm::Both)                                       \
  M("FloorToInt", afir::FloorToIntOp, BridgeOpForm::Both)                                   \
  M("Fma", afir::FmaOp, BridgeOpForm::Both)                                                 \
  M("BitwiseNot", afir::BitwiseNotOp, BridgeOpForm::Both)                                   \
  M("BitwiseOr", afir::BitwiseOrOp, BridgeOpForm::Both)                                     \
  M("BitwiseXor", afir::BitwiseXorOp, BridgeOpForm::Both)

// AscGraph 上一个节点自有的 float `ir_attr`，在 AFIR 上成为 op 属性。它不是操作数：
// 不进三段划分、不占一块缓冲，也不走上面那张操作数表。
//
// **两个名字都要说出来**：两边的拼写不保证一致——`LeakyRelu` 在 AscGraph 上是
// `negative_slope`，在 AFIR 上是 `negative_slop`（ODS 那边就这么写的）；`Axpy` 两边
// 都是 `alpha`。只留一个名字的话，第二个拼写不一致的 op 就会悄悄读到空值。
//
// 收成一张表是 `Axpy` 逼出来的：在它之前只有 `LeakyRelu` 一个，bridge 两处各写一个
// `if (type == "LeakyRelu")` 还说得清；第二个出现时两处都要再加一个分支，而两处的
// 分支条件必须一致——表驱动之后它们读的是同一行。
struct FloatIrAttrMapping {
  llvm::StringRef scheduler_name;
  llvm::StringRef ascgraph_attr_name;
  llvm::StringRef afir_attr_name;
};

llvm::ArrayRef<FloatIrAttrMapping> getFloatIrAttrMappings();

// 这个节点类型带不带这样一个属性。返回 nullptr 表示不带。
const FloatIrAttrMapping *findFloatIrAttr(llvm::StringRef schedulerName);

// 节点自有的 int64 `ir_attr`，在 AFIR 上成为 op 属性。与 `FloatIrAttrMapping` 是同一个
// 形状、同一套两列名字（两边拼写不保证一致，所以两个名字都要说出来），**只有值的类型
// 不同**。
//
// 为什么不把两张表合成一张带 dtype 标记的：合了之后每个取用点都要先 switch 一次类型才
// 能往下走，而两条路的落点本来就不同——float 那个成为 `F32FloatAttr`、这个成为
// `I64IntegerAttr`，其中一条还要走到模板实参位上去。两张表各自是平的，读的人不必先判类型。
struct IntIrAttrMapping {
  llvm::StringRef scheduler_name;
  llvm::StringRef ascgraph_attr_name;
  llvm::StringRef afir_attr_name;
};

llvm::ArrayRef<IntIrAttrMapping> getIntIrAttrMappings();

// 这个节点类型带不带这样一个属性。返回 nullptr 表示不带。
const IntIrAttrMapping *findIntIrAttr(llvm::StringRef schedulerName);

mlir::FailureOr<mlir::afir::ExprAttr> makeExprAttr(mlir::MLIRContext *context, llvm::StringRef canonicalExpression);

// Raw scheduler binding facts. These fields are retained only for validation
// and diagnostics; the canonical AFIR descriptor stores the role-derived
// buffer id and memory space instead.
enum class BindingDomain : uint8_t {
  Buffer,
  Queue,
};

// 调度给出的资源绑定。`position` / `alloc_type` **不在**这里：它们各自的信息已经被
// 转成了别的字段——position 决定 `BufferDescDTO::space`，alloc_type 决定 `domain`
// ——再原样带一份就是第二份事实源，两份不一致时没人知道该信哪个。
struct ScheduleBindingDTO {
  int64_t tensorId = -1;
  int64_t queueId = -1;
  int64_t bufferId = -1;
  int64_t reuseId = -1;
  int64_t queueDepth = -1;
  int64_t queueBufferCount = -1;
  BindingDomain domain = BindingDomain::Buffer;
};

// Canonical tensor and schedule facts. Shape is kept here because the bridge
// still emits the logical tensor form for operations that do not yet have a
// buffer-form ODS definition. The shape itself remains the MLIR type fact.
struct TensorDescDTO {
  std::vector<int64_t> shape;  // static dims (ShapedType::kDynamic for dynamic)
  // Canonical legacy Expression for every dimension. Dynamic dimensions keep
  // their symbolic identity here while their runtime value is represented by
  // tensor.dim/index SSA in the generated function.
  std::vector<std::string> shape_exprs;
  std::string dtype;  // "float16" | "float32" | "int32" | "int16" | "uint8" | ...
  std::vector<int64_t> axis;
  std::vector<std::string> repeats;
  std::vector<std::string> strides;
  std::vector<int64_t> vectorized_axis;  // axis ids stored in the buffer (API vector-length axes)
  std::vector<std::string> vectorized_strides;
  afir::Role role = afir::Role::Compute;
};

// Canonical resource facts. A missing lifetime axis is represented by the
// AFIR optional-parameter sentinel so ordinary tensor bindings do not invent
// a temporary-buffer lifetime.
struct BufferDescDTO {
  afir::Space space = afir::Space::GM;
  int64_t bufferId = -1;
  int64_t reuseId = -1;
  int64_t slotCount = -1;
  int64_t queueDepth = -1;
  std::string sizeBytesExpr;
  // 与 `sizeBytesExpr` 互斥：大小取自本资源的 canonical 尺寸字段（host 侧 tiling 填的
  // `q<id>_size` / `b<id>_size`），字段名不进 IR，由消费侧按 id 派生。
  bool sizeFromResource = false;
  int64_t alignmentBytes = -1;
  int64_t lifeTimeAxisId = std::numeric_limits<int64_t>::min();
};

// One scheduler tensor with its canonical descriptor pair and raw source
// binding. The builder consumes this value object and never receives an
// ascir::AscTensorAttr or scheduler Position/AllocType directly.
struct TensorValueDTO {
  TensorDescDTO tensor;
  BufferDescDTO buffer;
  ScheduleBindingDTO schedule;
};

// One node of the schedule graph. Inputs reference producer values by name
// (either a kernel input tensor name or an earlier node's name).
// A tmp buffer the scheduler allocated for one node. Only the identity is
// carried across: the byte count is not AFIR's to know, because the value that
// is actually reserved is aggregated on the ATT side across every node sharing
// the id and floored at 8 KB. AFIR references it by the canonical resource name
// `b{id}_size` instead -- see the tmp buffer section of
// docs/design/2026-09-17-afir-op-completion-elementwise-slice-design.md.
struct TmpBufferDTO {
  int64_t id = -1;
  int64_t life_time_axis_id = -1;
  std::optional<ReduceScratchRole> reduce_role = std::nullopt;
};

struct NodeInfo;

// 二次融合（`VectorFunc`）节点挂着的那张嵌套子图，已经翻成这一层的 DTO。
//
// **边界在这里就已经解掉了**，这是本结构存在的全部理由。子图的边界形状是
// `Data` + `Load`（入）与 `Store` + `Output`（出），而那两对表达的是「从 UB 取进寄存器」
// 与「从寄存器写回 UB」——与父图里的 `afir.load` / `afir.store`（GM <-> UB）不是一件事。
// 把它们原样翻成 AFIR 算子会多出两次搬运，而且**两种都编得过**，所以这一处的转换必须
// 在进 MLIR 之前完成：
//
//   父算子 srcs 的第 i 个  ->  `boundary_input_names[i]`（子图里引用这段数据的那个名字，
//                             通常是边界 `Load` 的名字，因为下游算子按它引用）
//   父算子 dsts 的第 i 个  ->  `boundary_output_names[i]`（子图里产出它的那个名字）
//
// `nodes` 只剩子图内的计算节点，拓扑序（分区器已经为子图做过
// `TopologicalSortingForVfGraph`，这里不重排，也不许依赖容器顺序）。
struct VectorFuncBodyDTO {
  std::vector<std::string> boundary_input_names;
  std::vector<std::string> boundary_output_names;
  std::vector<NodeInfo> nodes;
};

struct NodeInfo {
  std::string name;
  std::string type;  // Scheduler op name; see getAFIRBridgeableOperations().
  std::vector<std::string> input_names;
  std::vector<TensorValueDTO> inputs;
  // Scheduler nodes may expose multiple outputs. The singular fields below
  // remain as the first-result compatibility view for the currently
  // registered one-result AFIR operations; builders must reject an
  // unsupported multi-result node instead of silently dropping outputs.
  std::vector<std::string> output_names;
  std::vector<TensorValueDTO> outputs;
  TensorValueDTO output;
  std::vector<int64_t> sched_axis;  // loop axis ids, outer -> inner (SchedInfo.axis)
  int64_t loop_axis = -1;           // innermost loop axis id (SchedInfo.loop_axis)
  // 缓存复用周期（`SchedInfo.exec_condition`）的规范拼写，空串按 `no_cache` 处理。
  // 与 `type` / `dtype` / 轴类型同样走字符串：这一层不认识 AFIR 的枚举，把上游枚举
  // 翻译成规范拼写的那一处（`ScheduleToAfir` 的 `detail::ExecConditionStr`）也是拒收
  // 不支持取值的唯一闸门。
  std::string exec_condition;
  // Empty for the operations whose API needs no tmp buffer, which is most of them
  // on the reg-based family and about a third on the mem-based one.
  std::vector<TmpBufferDTO> tmp_buffers;
  // 编译期标量的字面值，只有 `Scalar` 节点有。AscGraph 那边它是一个字符串
  // （legacy 直接把这串文本拼进生成的 C++），这里按 `output.tensor.dtype` 解析成一个
  // 真正的常量——**不原样拼接**：AFIR 里它要成为一个有类型的 SSA 值，才谈得上被
  // verifier 和 capability 检查。解析不了的写法在 bridge 就报出来，不留到设备编译器。
  std::optional<std::string> scalar_value;
  // `IndexExpr` 节点的表达式，AFIR 规范串形式。与 `scalar_value` 同类——都是「不产生
  // 计算、只产生一个值」的标量源——差别只在值是**编译期字面量**还是**由 size var 算出
  // 来的运行期常量**。
  //
  // legacy 把两者收进同一张表（`kernel.constant_tensors`），在 kernel 顶部、所有循环
  // 之前一起物化（`AppendConstTensorInit`，跑在 `root_loop.Generate` 前面），渲染时
  // 只把 size var 换成 TilingData 字段、轴不参与。所以**它不需要归纳变量**，落点与
  // `Scalar` 同一处。
  std::optional<std::string> index_expr;
  // 节点自有的 float `ir_attr` 的值（`LeakyRelu` 的 `negative_slope`、`Axpy` 的
  // `alpha`）。读哪个名字、挂成 AFIR 上的哪个名字由 `getFloatIrAttrMappings()` 那张表
  // 说，这里只带一个值——**一个节点至多一个**，今天两个 op 都是如此。
  //
  // 与 `Scalar` 的 `value` 同类（都是节点自己的一个数，不是操作数），但那边是字符串
  // 字面量、要按 dtype 解析成一个有类型的 SSA 常量，这边是 float、成为一个属性。
  std::optional<float> float_ir_attr;
  // 节点自有的 int64 `ir_attr` 的值（多项式族的 `n`）。与上面那个同类，读哪个名字由
  // `getIntIrAttrMappings()` 说。**一个节点至多一个**，而且今天没有哪个节点同时带两种。
  std::optional<int64_t> int_ir_attr;
  // 只有 `VectorFunc` 节点有：它挂的那张嵌套子图。见 `VectorFuncBodyDTO`。
  std::shared_ptr<VectorFuncBodyDTO> vector_func_body;
  // host tiling 字段的节点级引用（`afir.transpose` 的 `api_tiling_field`）：字段名由
  // `ScheduleToAfir` 的 `GetApiTilingInfo` 按节点名生成，func 级
  // `afir.api_tiling_fields` 声明布局，这一份挂在 op 上说「本节点取哪个字段」。空表示
  // 这个节点的 API 不需要 host tiling（绝大多数节点如此）。同一个生成点保证两处不会
  // 说岔——这里只透传，不重算。
  std::optional<std::string> api_tiling_field;
};

struct ApiTilingInfo {
  std::string type_name;
  std::string field_name;
  std::vector<std::string> fields;
};

// A loop axis (carried into AFIR as attributes for later tiling).
struct AxisInfo {
  int64_t id = -1;
  std::string name;
  std::string type = "original";  // afir::AxisType spelling
  std::string size;               // symbolic size expression string
  bool bind_block = false;
  std::string align;
  std::vector<int64_t> from;
  int64_t split_pair_other_id = -1;
};

// Named graph input/output tensors. In buffer form both become func arguments;
// in logical form outputs remain function results.
struct IoTensor {
  std::string name;
  TensorValueDTO info;
  int64_t index = -1;
  // AscGraph 的 `ScalarData`：主机按值传进来的标量输入，kernel 形参是元素类型本身而不是
  // memref，`index` 与普通输入共用一个序号空间。
  //
  // **必须是显式的一位，不能从 `info.tensor` 的形状全 1 反推**——一个 `Data` 完全可以是
  // `[1]` 的张量，两者反推不出来。
  bool by_value = false;
};

// One impl graph -> one func.func.
struct TemplateInfo {
  std::string name;
  std::string original_name;
  std::vector<AxisInfo> axes;
  std::vector<std::string> size_vars;
  std::vector<IoTensor> inputs;
  std::vector<IoTensor> outputs;
  std::vector<NodeInfo> nodes;  // topological order
  std::vector<int64_t> queue_ids;
  std::vector<int64_t> buffer_ids;
  std::vector<ApiTilingInfo> api_tiling_data;
  int64_t tiling_key = 0;
  int64_t stage_id = -1;
  int64_t template_id = -1;
  bool double_buffer = false;
  std::string score_func;
  std::string workspace_size;
  // ScheduleToAfir emits the canonical memref destination-style form. The
  // standalone DTO demo may keep the legacy tensor form while migration
  // fixtures are being updated.
  bool buffer_form = false;
};

struct VarRelationInfo {
  int64_t dst_stage = -1;
  int64_t src_stage = -1;
  std::string dst_var;
  std::string src_expr;
};

// One plan -> one builtin ModuleOp. ScheduleGroup is not an
// MLIR container op: stage/template coordinates are flat func.func attributes.
//
// subgraph_id -- 对应 FusedScheduledResult 二维数组的第 1 维。Inductor 场景下会把多个
// AscGraph 组织成一个 Kernel（类似 SuperKernel，减少 Kernel Launch 次数），其中每个元素是一个
// 融合子图，在 MLIR 中称为一个 subgraph。
//
// plan_id -- 对应原来的 ScheduledResult。Schedule 模块针对同一个融合子图，可能给出多种不同的
// 切分策略，每种切分策略在 MLIR 中称为 plan。Reduce 场景可有双切分或
// 单切分（R 轴切核）策略，最终由 ATT 挑选出一个 plan 下发执行。
//
// stage_id（见 TemplateInfo）-- 对应原来的 ScheduleGroup。少量场景会把子图
// 拆成多个阶段执行（串行或并行），在 MLIR 中称为一个 stage。所有 stage 组合才是完整的
// 计算图。
//
// template_id（见 TemplateInfo）-- 对应 ImplGraph。一个 stage 可按不同轴切分生成
// 循环逻辑，每种切分结果称为 template。它们互斥，每个 stage 最终只挑选一个最优的
// template 执行。
struct PlanInfo {
  std::string name;
  int64_t subgraph_id = -1;
  int64_t plan_id = -1;
  bool is_reduce_mem_reuse = false;
  bool enable_group_parallel = false;
  int64_t cube_type = -1;
  std::string score_func;
  std::vector<int64_t> template_counts_per_stage;
  std::vector<std::string> origin_vars;
  std::vector<std::string> frontend_shape_vars;
  std::vector<std::string> shape_guards;
  std::vector<std::string> shape_asserts;
  std::vector<VarRelationInfo> var_relations;
  // Workspace requirements have no SSA owner. Keep their stable scheduler
  // order at function scope so the ABI and later resource passes can consume
  // the same canonical descriptors.
  std::vector<BufferDescDTO> workspace_plan;
  std::vector<TemplateInfo> templates;
};

// Build one AFIR module from one neutral Plan. Returns null on error
// (diagnostics are emitted to the context).
mlir::OwningOpRef<mlir::ModuleOp> BuildAfirModule(mlir::MLIRContext &context, const PlanInfo &plan);

}  // namespace bridge
}  // namespace afir
}  // namespace mlir

#endif  // AUTOFUSE_MLIR_BRIDGE_AFIRBUILDER_H
