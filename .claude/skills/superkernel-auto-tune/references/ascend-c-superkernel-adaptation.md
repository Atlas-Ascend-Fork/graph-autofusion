# Ascend C SuperKernel Operator Adaptation

Use this offline reference when `OP_UNSUPPORT`, a hang, or wrong output points to an
Ascend C custom operator, especially a direct-launch (`<<<>>>`) kernel. It distills the
operator-facing constraints needed for `npugraph_ex`; it is not a complete Ascend C
programming guide.

## Provenance And Authority

- Upstream repository: <https://gitcode.com/cann/asc-devkit.git>
- Source revision: `a5ba8c38fb6d8c06fa95624c8a6b9d07736f5eb2`
- Snapshot checked: 2026-08-01
- Source directory: `docs/zh/guide/.../SuperKernel/`
- API source: `docs/zh/api/Utils-API/SuperKernel/SK_BIND.md`

The relevant source pages are the principle, operator adaptation, direct-kernel
adaptation, and operator self-validation documents. CANN headers and compiler behavior
from the user's installed release remain authoritative. Confirm APIs and compile the
operator in that environment before claiming support.

## Contents

1. Common operator constraints
2. Direct-launch kernel adaptation
3. Validation ladder
4. Failure checklist

## Common Operator Constraints

### Core count and synchronization

The SuperKernel launch core count is the maximum launch core count among its child
kernels. A child using full-core synchronization is safe only when its expected core
count equals the SuperKernel core count, normally the device's full-core count. If the
child launches fewer cores but waits for every SuperKernel core, it can hang.

For `KERNEL_TYPE_MIX_AIC_1_1`, the child must also tolerate a SuperKernel 1:2
AIC-to-AIV launch ratio:

- every participating AIV core must execute matching hard-sync operations;
- only AIV0 should call the Matmul high-level API when that API expects one vector-side
  participant.

### GM scalar access and DCache

A standalone kernel may receive a compiler-inserted whole-DCache refresh at kernel
end. A SuperKernel child is compiled as a function, so that whole-kernel safeguard is
not automatic.

- `GlobalTensor::GetValue` and `SetValue` can receive cache-line refresh handling in
  supported `npugraph_ex` aclnn/GE paths.
- A scalar write through `GlobalTensor::operator()` does not automatically establish
  write-side consistency. The operator must do so explicitly.
- DCCI before/after options can trade frequent cache-line refreshes for an entire-cache
  refresh at a child boundary. Apply them only after correctness is established, and
  match the compiled symbol name rather than a bare op type.

### Block APIs and TPipe lifetime

- Use `AscendC::GetBlockIdx()` and `AscendC::GetBlockNum()` in ordinary Ascend C
  children. Do not read low-level `block_idx` or `block_num` variables directly.
- In SuperKernel children, `TPipe::Destroy` may have its internal
  `PipeBarrier<PIPE_ALL>()` removed. If multiple `TPipe` lifetimes or manual destroys
  rely on that ordering, insert the required barrier explicitly between them.
- For a direct-launch `__sk__` child that owns a `TPipe`, call
  `DestroyWithoutPipeAll()` before natural destruction as required by the direct
  adaptation contract.
- `SetNextTaskStart` and `WaitPreTaskEnd` can overlap child work only when the operator
  and `SK_BIND` mask are adapted consistently and the accepted `early_start` option is
  enabled.

## Direct-Launch Kernel Adaptation

Kernels invoked with `<<<>>>` require an additional SuperKernel entry. In this source
snapshot they can enter SuperKernel only through `npugraph_ex`, not the GE graph path.

Adapt in this order:

1. Keep the original `__global__` function unchanged for non-SuperKernel execution.
2. Define an argument struct with fields in the exact original parameter order. Apply
   `alignas(4)` to fields smaller than four bytes, such as `int8_t` or `int16_t`.
3. Add an `__sk__` function with the same `__cube__`, `__vector__`, or `__mix__(c, v)`
   kernel-type annotation as the original.
4. Read arguments from the struct and preserve all original initialization,
   computation, and output logic.
5. Add `sk::SkSystemArgs *sysArgs` only when block-count information is needed. Use
   `sysArgs->skNumBlocks` or `sysArgs->SkGetNumBlocks()` in that child. In MIX 1:2,
   vector-side `skNumBlocks` is twice the Cube block count.
6. If the child creates a `TPipe`, call `DestroyWithoutPipeAll()` before return.
7. Bind the original function to one to four child symbols with `SK_BIND`.

Minimal shape of the adaptation:

```cpp
#include "kernel_operator.h"

__global__ __vector__ void custom(GM_ADDR x, uint32_t length, int16_t flag)
{
    // Original implementation remains available.
}

struct CustomArgs {
    GM_ADDR x;
    uint32_t length;
    alignas(4) int16_t flag;
};

template <uint32_t Variant>
__sk__ __vector__ void custom_sk(
    const CustomArgs *args, sk::SkSystemArgs *sysArgs)
{
    const uint32_t block_num = sysArgs->SkGetNumBlocks();
    // Reproduce the original implementation with args and block_num.
}

SK_BIND(custom, 4, custom_sk<0>, custom_sk<1>);
```

Omit `sysArgs` when the child does not need block count. Instantiate only the child
symbols needed; `SK_BIND` supports at most four.

The second `SK_BIND` argument is a bit mask. Combine only features the kernel actually
implements:

| Bit value | Meaning |
|---:|---|
| 1 | early-start wait flag |
| 2 | early-start set flag |
| 4 | disable DCCI |
| 8 | disable batch-mode check |

Do not cargo-cult mask value `4` from examples: it disables DCCI and therefore has a
correctness contract.

## Validation Ladder

1. Build two modules with identical inputs: baseline uses the original operator; the
   candidate places one operator inside a balanced SuperKernel scope.
2. Compile both with `backend="npugraph_ex"`; enable
   `static_kernel_compile=True` for the controlled baseline and candidate, and enable
   `super_kernel_optimize=True` only for the candidate.
3. Compare exact output when exact equality is expected; otherwise use an explicitly
   justified tolerance. Confirm metadata and profiler evidence that the candidate
   actually used the SuperKernel path.
4. If core-count behavior is relevant, run an accepted
   `debug_per_op_max_core_num=1` diagnostic. A one-op scope alone may not launch at the
   real model's maximum core count.
5. Fuse the operator with another known-supported child, or invoke it repeatedly in a
   dependency chain. This can expose cross-child Cache and synchronization faults that
   a one-op test misses.
6. Run the real model and repeat correctness tests before any performance benchmark.

## Failure Checklist

For `OP_UNSUPPORT`, wrong output, timeout, or a fault inside the custom child, verify:

- the custom binary exports a matching SuperKernel binding;
- argument order, widths, `alignas(4)`, and template instantiations match the launch;
- `__sk__` and original kernel-type annotations match;
- block count comes from supported APIs and MIX 1:2 semantics are handled;
- full-core and Cube/AIV synchronization participant counts match;
- GM scalar writes and DCCI policy preserve Cache consistency;
- every `TPipe` lifetime has the required destruction and barriers;
- early-start calls and mask bits agree;
- one-op, maximum-core, multi-op, and real-model tests all pass.
