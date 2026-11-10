/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AFIR_VECTOR_FUNC_OPS
#define AFIR_VECTOR_FUNC_OPS

include "AFIRDialect.td"
include "AFIROpBase.td"
include "AFIRAttrs.td"
include "AFIRTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// 二次融合：一段在寄存器里逐 lane 算完的子图
//===----------------------------------------------------------------------===//
//
// 这是 AFIR 的第一个带 region 的算子。 它的来源是 A5 调度在 buffer 分配阶段做的二次
// 融合（`v35/optimize/partition/vector_func_partitioner.cpp`，门槛 `kMinVfNodesNum = 2`）：
// 相邻且都支持寄存器形态的节点被聚成一个 `VectorFunc` 节点，节点体是一张嵌套 AscGraph。
// 设计见 docs/development/2026-10-18-afir-vector-func-implementation-design.md。
//
// 为什么 region 内必须是 tensor 形态（值语义）而不是 buffer 形态：VF 的中间结果留在
// 向量寄存器里，一块 UB 都没有分配——UB 的队列容量由 legacy 的 buffer 分配与 ATT 定，
// MLIR 只读 tiling 里的 `q*_size`。buffer 形态是 destination-style，每个中间结果都要一个
// `outs` 指向的存储，那等于要求 MLIR 自己变出没人分配过的 UB。tensor 形态（`outs` 不给，
// 见 `Elementwise.td` 顶部那段）正好表达「有数据流、没有存储」。
//
// 边界则相反，是真的 UB 张量：`srcs` / `dsts` 是调度分配好的队列张量，所以它们是
// shaped（LocalTensor / memref）。于是这个算子的语义正是两者的接缝：
//
//     对这块 tile 的每一段 lane：从 srcs 取一段进寄存器 -> 跑一遍 region -> 把 yield
//     出来的结果写回 dsts 的同一段
//
// lane 这一维在 AFIR 里不出现：一段有多长（`GetVecLen() / sizeof(T)`）是目标属性，
// 由下降那一侧算，AFIR 这层只说「逐 lane」。
//
//===----------------------------------------------------------------------===//

// 元素类型域取 AFIR 今天建模的那 6 位（`Implementations/ElementwiseApiFamily.h`）。
// 这里不收窄：这个算子是个容器，收不收某个 dtype 由 region 内算子自己的 ODS 说，
// 而「这一侧有没有寄存器形态的实现」由能力层说（见 AFIRVectorFuncToRegBase）。
defvar AFIR_VectorFuncElementTypes = [UI8, UI16, I16, I32, F16, F32, BF16];

def AFIR_VectorFuncOp : AFIR_Op<"vector_func", [
    SingleBlockImplicitTerminator<"::mlir::afir::VectorFuncYieldOp">,
    IsolatedFromAbove,
    AttrSizedOperandSegments,
    // 接口方法在 `extraClassDeclaration` 里就地实现（三行），所以这里直接列接口，
    // 不用 `DeclareOpInterfaceMethods`——后者会再声明一遍，与就地实现撞成重载。
    AFIRDestinationStyleInterface
]>, AFIR_CommonAttrs {
  let afirStability = AFIR_Internal;
  let summary = "Evaluate a fused sub-graph lane by lane, keeping intermediates in registers";
  let description = [{
    把 region 里的子图对 `srcs` 的每一段 lane 算一遍，结果写进 `dsts` 的同一段。region 的
    块参数与 `srcs` 一一对应、`afir.vector_func_yield` 的操作数与 `dsts` 一一对应，两侧
    按下标配对——序号来自调度的边界序号，不是按名字猜的。

    region 内是**tensor 形态**的 AFIR 算子：没有 `outs`，也就没有存储，中间结果在下降之后
    是向量寄存器。这与 buffer 形态的区别不是写法偏好：buffer 形态的每个中间结果都要一块
    UB，而 VF 这一段的 UB 预算里没有它们。

    `dsts` 同时作为结果值产出，消费者用结果值串依赖（与其余 buffer 形态算子一致）。

    ```mlir
    %out = afir.vector_func srcs(%a, %b) dsts(%y) {
    ^bb0(%la: tensor<?xf32>, %lb: tensor<?xf32>):
      %sum = afir.add %la, %lb : tensor<?xf32>, tensor<?xf32> -> tensor<?xf32>
      %res = afir.abs %sum : tensor<?xf32> -> tensor<?xf32>
      afir.vector_func_yield %res : tensor<?xf32>
    } : (!ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>)
        -> (!ascendc.local_tensor<?xf32>) -> !ascendc.local_tensor<?xf32>
    ```

    **verifier 只查结构**（块参数/yield 与边界数目和元素类型对得上、region 内不再嵌 VF）。
    「这一期支持几个输出、支持哪些 dtype、region 内的算子有没有寄存器形态」都**不在
    verifier 里**——那些是能力问题，verifier 拒收是硬失败，而能力不足必须干净回落。
  }];

  // `commonArgs` 带的是使用点的 tensor descriptor（`operand_descs` / `result_descs` /
  // `destination_descs`）。**它们不是装饰**：流水同步那一步按「有没有 destination descs」
  // 认一个算子是不是 buffer 形态的（`AFIRPipelineSync.cpp` 的 `isBufferizedOperation`），
  // 认不出来就不会给这段插 enque / deque——产物在 CPU 模拟器上照样算对，到真机上才是
  // 偶发错数据。
  // 副作用显式声明，不用 `RecursiveMemoryEffects`。 region 里是 tensor 形态的算子，
  // 它们是纯的，于是「递归推导」推出来的是「整个算子无副作用」；而这个算子的全部作用正是
  // 把结果写进 `dsts` 那块 UB。无副作用的后果是：destination-style 的结果一旦被折叠成
  // destination，这个算子就没有使用者了，随后被当成死代码回收——**产物里只剩搬运、
  // 计算整段消失，而且照样编得过**。实测在资源物化那一阶段踩过一次。
  let arguments = !con((ins
    Arg<Variadic<AFIR_BufferizableShapedType<AFIR_VectorFuncElementTypes>>, "", [MemRead]>:$srcs,
    Arg<Variadic<AFIR_BufferizableShapedType<AFIR_VectorFuncElementTypes>>, "", [MemWrite]>:$dsts
  ), commonArgs);
  let results = (outs Variadic<AFIR_BufferizableShapedType<AFIR_VectorFuncElementTypes>>:$results);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    `srcs` `(` $srcs `)` `dsts` `(` $dsts `)` attr-dict-with-keyword $body
    `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    /// region 的那一个块。
    ::mlir::Block &getBodyBlock() { return getBody().front(); }

    /// yield 算子。`SingleBlockImplicitTerminator` 保证它存在且类型正确。
    ::mlir::afir::VectorFuncYieldOp getYield();

    //=== AFIRDestinationStyleInterface ===//
    // 每个 `dsts` 就是一个 destination，并且与同下标的结果绑定（`verify` 查这一条）。
    // tmp buffer 恒为 0：寄存器形态的 API 不要 tmp（`MicroAPI` 那一类自足）。
    unsigned getNumDestinations() { return static_cast<unsigned>(getDsts().size()); }
    ::mlir::FailureOr<unsigned> getTiedResultIndex(unsigned destinationIndex) {
      if (destinationIndex >= getDsts().size()) {
        return ::mlir::failure();
      }
      return destinationIndex;
    }
    unsigned getNumTmpBuffers() { return 0U; }
  }];
}

def AFIR_VectorFuncYieldOp : AFIR_Op<"vector_func_yield", [
    Terminator,
    HasParent<"::mlir::afir::VectorFuncOp">,
    Pure
]> {
  let afirStability = AFIR_Internal;
  let summary = "Terminate an afir.vector_func body, naming the per-lane results";
  let description = [{
    结束 `afir.vector_func` 的 region。操作数是这一段 lane 算出来的结果，按下标写进父算子
    `dsts` 的同一段。操作数数目与元素类型由父算子的 verifier 核对。
  }];

  let arguments = (ins Variadic<AnyType>:$results);
  let assemblyFormat = "($results^ `:` type($results))? attr-dict";

  // 空参构造是 `SingleBlockImplicitTerminator` 要的：父算子的 `ensureTerminator` 用它补
  // 终结算子。`Variadic` 生成的构造要一个 `ValueRange`，不补这一个的失败方式是模板
  // 实例化报「no matching function for call to build」，指向的是 MLIR 的头文件。
  let builders = [
    OpBuilder<(ins), [{ build($_builder, $_state, ::mlir::ValueRange{}); }]>
  ];
}

#endif // AFIR_VECTOR_FUNC_OPS
