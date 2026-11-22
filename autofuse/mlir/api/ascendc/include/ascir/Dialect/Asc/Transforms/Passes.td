/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASC_PASSES_TD
#define ASC_PASSES_TD

include "mlir/Pass/PassBase.td"

def DeclarePyStruct : Pass<"ascendc-declare-py-struct", "ModuleOp"> {
  let summary = "Insert emitasc.declare_py_struct operations for all Python struct types used in the module";
  let constructor = "mlir::ascendc::createDeclarePyStructPass()";
  let dependentDialects = ["emitasc::EmitAscDialect"];
  let description = [{
    Collects all `emitasc.py_struct` types referenced in function arguments, block arguments, and operation results,
    then inserts `emitasc.declare_py_struct` operations at the module beginning to declare them for C++ code emission.

    The pass deduplicates struct types while preserving the order of first occurrence. Nested struct types
    (structs containing other structs) are also declared.
  }];
}

def DefineCubeOnly : Pass<"ascendc-define-cube-only", "ModuleOp"> {
  let summary = "Define ASCENDC_CUBE_ONLY macro for cube-only (matrix multiplication) kernels";
  let constructor = "mlir::ascendc::createDefineCubeOnlyPass()";
  let dependentDialects = ["emitc::EmitCDialect"];
  let description = [{
    Inserts `#define ASCENDC_CUBE_ONLY` verbatim at module start and sets `matmul_cube_only` attribute.
    This macro enables cube-only optimizations in the Ascend C runtime for kernels that only perform
    matrix multiplication without vector operations.
  }];
}

def DetectEnableDebug : Pass<"ascendc-detect-enable-debug", "ModuleOp"> {
  let summary = "Detect debug utility usage (printf, dump_tensor) and set enable_debug attribute";
  let constructor = "mlir::ascendc::createDetectEnableDebugPass()";
  let description = [{
    Scans the module for `PrintfOp` or `DumpTensorOp` operations. If any are found, sets the `enable_debug`
    unit attribute on the module to enable debug runtime support during kernel execution.
  }];
}

def DetectKernelType : Pass<"ascendc-detect-kernel-type", "ModuleOp"> {
  let summary = "Classify kernel as vector, cube, or mixed based on operation types present";
  let constructor = "mlir::ascendc::createDetectKernelTypePass()";
  let description = [{
    Analyzes operations in the module to determine kernel type and sets `kernel_type` string attribute:
    - "vector": Only vector operations present (no `MmadOp` or `RegistMatmulObjOp`)
    - "cube": Only matrix multiplication operations present (no `VectorOp`)
    - "mixed": Both vector and cube operations present

    This classification affects synchronization strategy and runtime configuration.
  }];
}

def EraseSync : Pass<"ascendc-erase-sync", "func::FuncOp"> {
  let summary = "Remove TQueBind synchronization operations and replace deque_tensor with allocated tensors";
  let constructor = "mlir::ascendc::createEraseSyncPass()";
  let description = [{
    Removes intra-core synchronization infrastructure for static allocation mode:
    - Erases `TQueBindEnqueTensorOp`, `SetFlagOp`, `WaitFlagOp`, `PipeBarrierOp`
    - Replaces `TQueBindDequeTensorOp` with the tensor from corresponding `TQueBindAllocTensorOp`

    This pass is used when tensors are statically allocated and queue-based synchronization is not needed.
  }];
}

def GenerateBoilerplate : Pass<"ascendc-generate-boilerplate", "ModuleOp"> {
  let summary = "Insert required C++ include headers based on operations used in the kernel";
  let constructor = "mlir::ascendc::createGenerateBoilerplatePass()";
  let dependentDialects = ["emitc::EmitCDialect"];
  let description = [{
    Adds necessary Ascend C header includes at module start:
    - Always includes `kernel_operator.h` (core runtime)
    - If `ListTensorDescOp` or `ListTensorDescV2Op` present: includes `kernel_operator_list_tensor_intf.h`
    - If `RegistMatmulObjOp` present: includes `lib/matmul_intf.h`
    - If `TensorDescOp` present: includes `kernel_operator_list_tensor_intf.h`
  }];
}

def HoistQueBind : Pass<"ascendc-hoist-que-bind", "func::FuncOp"> {
  let summary = "Move queue and buffer initialization operations to the function entry block";
  let constructor = "mlir::ascendc::createHoistQueBindPass()";
  let description = [{
    Hoists TQueBind-related initialization ops from loops to function root:
    - `QueueOp`, `QueBindOp`, `TBufOp` declarations
    - `TPipeInitQueueOp`, `TPipeInitBufferOp` initialization
    - `TBufGetTensorOp` tensor retrieval

    This optimization ensures queue/buffer setup happens once at kernel entry rather than repeatedly in loops.
  }];
}

def HoistTensorAllocation : Pass<"ascendc-hoist-tensor-allocation", "func::FuncOp"> {
  let summary = "Move LocalTensorAutoOp allocations from loops to the function entry block";
  let constructor = "mlir::ascendc::createHoistTensorAllocationPass()";
  let options = [
    Option<"excludeInOut", "exclude-in-out", "bool", "false",
           "Keep input/output tensors inside loops, only hoist temporary tensors">,
  ];
  let description = [{
    Hoists `LocalTensorAutoOp` tensor allocations to the function root to reduce repeated allocations in loops.
    When `exclude-in-out=true`: only hoists tensors without `input` or `output` flags (pure temporary tensors).
    When `exclude-in-out=false`: hoists all tensors regardless of input/output status.

    Input/output tensors are those connected to GM via `DataCopyOp` with `GlobalToLocal`/`LocalToGlobal` direction.
  }];
}

def InputOutputTensor : Pass<"ascendc-input-output-tensor", "func::FuncOp"> {
  let summary = "Mark tensors as input/output based on GM data copy direction and handle input-only to output copies";
  let constructor = "mlir::ascendc::createInputOutputTensorPass()";
  let description = [{
    Analyzes `DataCopyOp` to set `input`/`output` flags on `LocalTensorAutoOp`:
    - `gm_ubuf` direction → tensor is input (loaded from GM)
    - `ubuf_gm` direction → tensor is output (stored to GM)

    Also handles special case: if an input-only tensor is used for `ubuf_gm` copy, creates a separate output tensor
    and inserts `DataCopyL2Op` to copy input tensor to output tensor before the GM store.

    Example transformation:
    ```mlir
    // Before: unmarked tensors
    %0 = ascendc.local_tensor_auto vecin() : <64xf32>
    ascendc.data_copy_l2 %0, %gm_in, %c64  // gm→ub (input)
    %1 = ascendc.local_tensor_auto vecout() : <64xf32>
    ascendc.data_copy_l2 %gm_out, %1, %c64  // ub→gm (output)

    // After: tensors marked with input/output flags
    %0 = ascendc.local_tensor_auto vecin() input : <64xf32>
    %1 = ascendc.local_tensor_auto vecout() output : <64xf32>
    ```
  }];
}

def InsertQueSync : Pass<"ascendc-insert-que-sync", "func::FuncOp"> {
  let summary = "Insert TQueBind enqueue/dequeue synchronization and scalar get/set_value barriers";
  let constructor = "mlir::ascendc::createInsertQueSyncPass()";
  let dependentDialects = ["ascendc::AscendCDialect"];
  let description = [{
    Inserts intra-core synchronization for TQueBind-managed tensors and scalar operations:
    1. **Enqueue tensors**: After each `OpWithDst`, inserts `TQueBindEnqueTensorOp` or `PipeBarrierOp(pipe_v)`
    2. **Dequeue tensors**: For enqueued tensors, inserts `TQueBindDequeTensorOp` before first user
    3. **Get/Set value sync**: Wraps `get_value`/`set_value` with V_S/S_V event sync (fetch_event_id, set_flag, wait_flag)
    4. **Loop handling**: Special sync placement for `set_value` inside single-op loops
    5. **Canonicalization**: Adds `PipeBarrierOp(pipe_all)` at function end and runs canonicalization

    TQueBind synchronization enables double-buffering and overlapping compute with data transfer.

    Example: enqueue/dequeue synchronization:
    ```mlir
    // Before: tensor used directly after allocation
    %tensor = ascendc.que_bind.alloc_tensor %queue
    ascendc.add_l2 %dst, %tensor, %tensor, %count

    // After: tensor dequeued before use, enqueued after compute
    %tensor = ascendc.que_bind.alloc_tensor %queue
    %dequeued = ascendc.que_bind.deque_tensor %queue
    ascendc.add_l2 %dst, %dequeued, %dequeued, %count
    ascendc.que_bind.enque_tensor %queue, %dst
    ```

    Example: scalar get/set_value with V_S/S_V sync:
    ```mlir
    // Before: direct scalar access
    %val = ascendc.local_tensor.get_value %tensor, %offset

    // After: wrapped with event sync
    %event = ascendc.pipe.fetch_event_id %pipe, v_s
    ascendc.set_flag v_s, %event
    ascendc.wait_flag v_s, %event
    %val = ascendc.local_tensor.get_value %tensor, %offset
    %event2 = ascendc.pipe.fetch_event_id %pipe, s_v
    ascendc.set_flag s_v, %event2
    ascendc.wait_flag s_v, %event2
    ```
  }];
}

def MaterializeTensor : Pass<"ascendc-materialize-tensor", "func::FuncOp"> {
  let summary = "Convert LocalTensorAutoOp placeholders to concrete TBuf or TQueBind allocations";
  let constructor = "mlir::ascendc::createMaterializeTensorPass()";
  let dependentDialects = ["arith::ArithDialect", "ascendc::AscendCDialect"];
  let options = [
    Option<"alwaysBuf", "always-buf", "bool", "false",
           "Use TBuf for all tensors; otherwise use TQueBind for input/output tensors">,
  ];
  let description = [{
    Materializes tensor allocation placeholders into concrete Ascend C buffer objects:
    - **TQueBind path** (for input/output tensors when `alwaysBuf=false`):
      Creates `QueueOp` + `TPipeInitQueueOp` + `TQueBindAllocTensorOp` + `TQueBindFreeTensorOp` at function end
    - **TBuf path** (for temporary tensors or when `alwaysBuf=true`):
      Creates `TBufOp` + `TPipeInitBufferOp` + `TBufGetTensorOp`

    Input tensors use `VECIN` queue position, output tensors use `VECOUT`. Temporary tensors use `VECCALC` TBuf.
    Buffer size computed from tensor shape (static or dynamic via shape operands).

    Example transformation (TQueBind path for input tensor):
    ```mlir
    // Before: placeholder
    %0 = ascendc.local_tensor_auto vecin() input : <64xf32>

    // After: queue-based allocation
    %pipe = ascendc.pipe
    %queue = ascendc.queue : <vecin, 1>
    ascendc.pipe.init_queue %pipe, %queue, %c1, %c256
    %0 = ascendc.que_bind.alloc_tensor %queue : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<64xf32>
    // ... at function end:
    ascendc.que_bind.free_tensor %queue, %0
    ```

    Example transformation (TBuf path for temporary tensor):
    ```mlir
    // Before: placeholder
    %0 = ascendc.local_tensor_auto veccalc() : <64xf32>

    // After: buffer-based allocation
    %pipe = ascendc.pipe
    %tbuf = ascendc.tbuf : <veccalc>
    ascendc.pipe.init_buffer %pipe, %tbuf, %c256
    %0 = ascendc.tbuf.get_tensor %tbuf : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<64xf32>
    ```
  }];
}

def LegalizeKernelArgs : Pass<"ascendc-legalize-kernel-args", "ModuleOp"> {
  let summary = "Attach kernel argument attributes and insert ffts_addr handling for multi-core kernels";
  let constructor = "mlir::ascendc::createLegalizeKernelArgsPass()";
  let dependentDialects = ["arith::ArithDialect", "ascendc::AscendCDialect", "emitasc::EmitAscDialect", "scf::SCFDialect"];
  let options = [
    Option<"setFftsAddr", "set-ffts-addr", "bool", "false",
           "Append ffts_addr kernel argument and call set_ffts_base_addr for cross-core sync">,
  ];
  let description = [{
    Processes kernel entry functions (marked with `ascendc.global` attribute):
    1. Marks all existing arguments as `Explicit` kernel arguments
    2. If `setFftsAddr=true`: adds `ffts_addr` memref argument and calls `SetFftsBaseAddrOp`
    3. If matmul present and not cube-only: inserts `AscendIsAICOp` check with `FftsCrossCoreSyncOp` for AIC mode

    Kernel argument attributes (`emitasc.kernel_arg`) control how arguments are handled by the runtime.
  }];
}

def Noop : Pass<"ascendc-noop", "func::FuncOp"> {
  let summary = "Placeholder pass that performs no transformations";
  let constructor = "mlir::ascendc::createNoopPass()";
  let description = [{
    A no-operation pass useful for testing, pipeline debugging, or as a placeholder in pass schedules.
    The pass walks the function but performs no modifications to the IR.
  }];
}

def PrivatizeFunc : Pass<"ascendc-privatize-func", "ModuleOp"> {
  let summary = "Mark non-kernel functions as private and kernel functions as public for emission";
  let constructor = "mlir::ascendc::createPrivatizeFuncPass()";
  let description = [{
    Adjusts function visibility based on kernel status:
    - Functions without `ascendc.global` attribute → set as `private` (internal helper functions)
    - Functions with `ascendc.global` and body → set as `public` (kernel entry points)
    - Functions with `ascendc.global` but no body (declarations) → remain as-is

    This ensures only kernel entry points are exported while helper functions remain internal.
  }];
}

def UnifyPipe : Pass<"ascendc-unify-pipe", "func::FuncOp"> {
  let summary = "Replace multiple PipeOp instances with a single unified pipe at function entry";
  let constructor = "mlir::ascendc::createUnifyPipePass()";
  let description = [{
    Consolidates multiple `PipeOp` declarations into a single pipe object at function entry.
    All uses of individual pipes are replaced with the unified pipe, and original `PipeOp` instances are erased.

    This ensures consistent pipe management across the kernel and reduces redundant TPipe object creation.
  }];
}

def VerifySync : Pass<"ascendc-verify-sync", "func::FuncOp"> {
  let summary = "Validate TQueBind synchronization correctness and emit warnings for mismatches";
  let constructor = "mlir::ascendc::createVerifySyncPass()";
  let description = [{
    Verifies proper pairing of TQueBind operations and emits warnings for incorrect usage:
    - `alloc_tensor` without corresponding `free_tensor`
    - `free_tensor` for already-freed tensor or without `alloc_tensor`
    - `enque_tensor` without corresponding `deque_tensor`
    - `deque_tensor` without `enque_tensor` in queue
    - Unexpected tensor uses between `enque_tensor` and `deque_tensor`

    This is a verification-only pass that helps detect synchronization bugs during development.
  }];
}

#endif // ASC_PASSES_TD
