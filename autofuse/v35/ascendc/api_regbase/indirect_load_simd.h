/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_V35_ASCENDC_API_REGBASE_INDIRECT_LOAD_SIMD_H_
#define AUTOFUSE_V35_ASCENDC_API_REGBASE_INDIRECT_LOAD_SIMD_H_

#include <type_traits>
#include <utility>

#ifndef AUTOFUSE_V35_ASCENDC_API_REGBASE_INDIRECT_LOAD_SIMD_POLICY_H_
#include "indirect_load_simd_policy.h"
#endif

namespace AscendC {
enum class IndirectLoadSimdFallback : uint8_t {
  kRegisterGather = 0,
  kGatherApi = 1,
  kStrided = 2,
};

template <IndirectLoadSimdFallback Fallback, bool TryEmbedding>
struct IndirectLoadSimdCaseTag {};

template <int32_t Rank>
struct IndirectLoadSimdStridedParams {
  uint32_t logical_size;
  uint32_t physical_size;
  int64_t output_offset;
  int64_t index_sizes[Rank];
  int64_t input_strides[Rank];
  int64_t index_strides[Rank];
  int64_t output_strides[Rank];
};

template <int32_t Rank>
struct IndirectLoadSimdDenseParams {
  uint32_t physical_size;
  uint32_t input_actual_size;
  int64_t output_offset;
  int64_t input_axis;
  int64_t index_sizes[Rank];
  int64_t input_strides[Rank];
};

template <int32_t Rank>
struct IndirectLoadSimdEmbeddingParams {
  uint32_t logical_size;
  uint32_t physical_size;
  uint32_t input_actual_size;
  int64_t output_offset;
  int64_t input_axis;
  int64_t index_sizes[Rank];
  int64_t input_strides[Rank];
  int64_t index_strides[Rank];
  int64_t output_strides[Rank];
};

namespace Internal {
template <int32_t Dim, int32_t Axis, int32_t StrideBase>
struct IndirectLoadSimdInnerOffset {
  __aicore__ inline static int64_t Call(int64_t inner, const int64_t *shape) {
    int64_t offset = inner % shape[Dim] * shape[StrideBase + Dim];
    if constexpr (Dim > Axis + 1) {
      offset += IndirectLoadSimdInnerOffset<Dim - 1, Axis, StrideBase>::Call(inner / shape[Dim], shape);
    }
    return offset;
  }
};

template <int32_t Dim, int32_t StrideBase>
struct IndirectLoadSimdOuterOffset {
  __aicore__ inline static int64_t Call(int64_t outer, const int64_t *shape) {
    int64_t offset = outer % shape[Dim] * shape[StrideBase + Dim];
    if constexpr (Dim > 0) {
      offset += IndirectLoadSimdOuterOffset<Dim - 1, StrideBase>::Call(outer / shape[Dim], shape);
    }
    return offset;
  }
};

template <int32_t Dim, int32_t Axis>
struct IndirectLoadSimdInnerSize {
  __aicore__ inline static int64_t Call(const int64_t *shape) {
    if constexpr (Dim == Axis) {
      return 1;
    } else {
      return shape[Dim] * IndirectLoadSimdInnerSize<Dim - 1, Axis>::Call(shape);
    }
  }
};

template <typename X>
struct IndirectLoadSimdGatherAction {
  template <typename ValuePolicy>
  __simd_callee__ inline static void Commit(__ubuf__ X *x, __ubuf__ X *y, __ubuf__ uint32_t *, uint32_t input_size,
                                            uint32_t base, MicroAPI::RegTensor<uint32_t> &source_index,
                                            MicroAPI::MaskReg mask, uint32_t count) {
    ValuePolicy::GatherAndStore(x, y + base, source_index, mask, mask, count, input_size);
  }

  template <typename ValuePolicy>
  __simd_callee__ inline static void CommitPair(__ubuf__ X *x, __ubuf__ X *y, __ubuf__ uint32_t *, uint32_t input_size,
                                                uint32_t base, MicroAPI::RegTensor<uint32_t> &index0,
                                                MicroAPI::RegTensor<uint32_t> &index1, MicroAPI::MaskReg mask0,
                                                MicroAPI::MaskReg mask1, uint32_t count) {
    ValuePolicy::GatherAndStorePair(x, y + base, index0, index1, mask0, mask1, count, input_size);
  }
};

template <typename X>
struct IndirectLoadSimdOffsetAction {
  template <typename ValuePolicy>
  __simd_callee__ inline static void Commit(__ubuf__ X *, __ubuf__ X *, __ubuf__ uint32_t *offsets, uint32_t,
                                            uint32_t base, MicroAPI::RegTensor<uint32_t> &source_index,
                                            MicroAPI::MaskReg mask, uint32_t) {
    IndirectLoadSimdStoreByteOffsets<X>(offsets + base, source_index, mask);
  }

  template <typename ValuePolicy>
  __simd_callee__ inline static void CommitPair(__ubuf__ X *, __ubuf__ X *, __ubuf__ uint32_t *offsets, uint32_t,
                                                uint32_t base, MicroAPI::RegTensor<uint32_t> &index0,
                                                MicroAPI::RegTensor<uint32_t> &index1, MicroAPI::MaskReg mask0,
                                                MicroAPI::MaskReg mask1, uint32_t count) {
    constexpr uint32_t elements_per_reg = VECTOR_REG_WIDTH / sizeof(uint32_t);
    IndirectLoadSimdStoreByteOffsets<X>(offsets + base, index0, mask0);
    if (count > elements_per_reg) {
      IndirectLoadSimdStoreByteOffsets<X>(offsets + base + elements_per_reg, index1, mask1);
    }
  }
};

template <typename X, typename Index>
struct IndirectLoadSimdArgs {
  __ubuf__ X *x;
  __ubuf__ Index *index;
  __ubuf__ X *y;
  __ubuf__ uint32_t *offsets;
  uint32_t actual_size;
};

template <IndirectLoadSimdAddressMode Mode, typename X, typename Index, int32_t Rank, int32_t Axis, typename Action,
          typename... ShapeArgs>
__simd_callee__ inline void IndirectLoadSimdRunRepeat(IndirectLoadSimdArgs<X, Index> &args,
                                                      typename IndirectLoadSimdIndexPolicy<Index>::LoadState &state,
                                                      uint32_t base, uint32_t count, MicroAPI::MaskReg mask,
                                                      MicroAPI::RegTensor<uint32_t> &input_inner,
                                                      MicroAPI::RegTensor<uint32_t> &address_invariant,
                                                      const IndirectLoadSimdAddressContext &context,
                                                      ShapeArgs... shape_args) {
  using Traits = IndirectLoadSimdModeTraits<Mode, X, Index, Rank, Axis>;
  using IndexPolicy = typename Traits::IndexPolicy;
  using ValuePolicy = typename Traits::ValuePolicy;
  MicroAPI::RegTensor<uint32_t> source_index;
  IndexPolicy::Load(source_index, state, count);
  IndirectLoadSimdApplyAddress<Mode, Rank, Axis>(source_index, base, context, input_inner, address_invariant, mask,
                                                 shape_args...);
  Action::template Commit<ValuePolicy>(args.x, args.y, args.offsets, context.input_actual_size, base, source_index,
                                       mask, count);
}

template <typename X, typename Index, typename Action>
__simd_callee__ inline void IndirectLoadSimdRunPair(IndirectLoadSimdArgs<X, Index> &args,
                                                    typename IndirectLoadSimdIndexPolicy<Index>::LoadState &state,
                                                    uint32_t base, uint32_t count,
                                                    const IndirectLoadSimdAddressContext &context) {
  using Traits = IndirectLoadSimdRegTraits<X, Index, 1, 0>;
  using IndexPolicy = typename Traits::IndexPolicy;
  using ValuePolicy = typename Traits::ValuePolicy;
  MicroAPI::RegTensor<uint32_t> index0, index1;
  MicroAPI::MaskReg mask0, mask1;
  IndexPolicy::LoadPair(index0, index1, mask0, mask1, state, count);
  Action::template CommitPair<ValuePolicy>(args.x, args.y, args.offsets, context.input_actual_size, base, index0,
                                           index1, mask0, mask1, count);
}

template <IndirectLoadSimdAddressMode Mode, typename X, typename Index, int32_t Rank, int32_t Axis, typename Action,
          typename... ShapeArgs>
__aicore__ inline void IndirectLoadSimdRunMode(IndirectLoadSimdArgs<X, Index> &args,
                                               const IndirectLoadSimdAddressContext &context, ShapeArgs... shape_args) {
  using Traits = IndirectLoadSimdModeTraits<Mode, X, Index, Rank, Axis>;
  using IndexPolicy = typename Traits::IndexPolicy;
  constexpr uint32_t elements_per_repeat = Traits::kElementsPerRepeat;
  const uint16_t full_repeats = static_cast<uint16_t>(args.actual_size / elements_per_repeat);
  const uint32_t tail_count = args.actual_size % elements_per_repeat;
  __VEC_SCOPE__ {
    typename IndexPolicy::LoadState state;
    IndexPolicy::Init(state, args.index);
    if constexpr (sizeof(X) == sizeof(uint16_t) && Mode == IndirectLoadSimdAddressMode::kDirect) {
      for (uint16_t repeat = 0U; repeat < full_repeats; ++repeat) {
        const uint32_t base = static_cast<uint32_t>(repeat) * elements_per_repeat;
        IndirectLoadSimdRunPair<X, Index, Action>(args, state, base, elements_per_repeat, context);
      }
      if (tail_count != 0U) {
        IndirectLoadSimdRunPair<X, Index, Action>(args, state, full_repeats * elements_per_repeat, tail_count, context);
      }
    } else {
      MicroAPI::MaskReg full_mask = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
      MicroAPI::RegTensor<uint32_t> input_inner;
      MicroAPI::RegTensor<uint32_t> address_invariant;
      IndirectLoadSimdInitInvariants<Mode>(input_inner, address_invariant, context, full_mask);
      for (uint16_t repeat = 0U; repeat < full_repeats; ++repeat) {
        const uint32_t base = static_cast<uint32_t>(repeat) * elements_per_repeat;
        IndirectLoadSimdRunRepeat<Mode, X, Index, Rank, Axis, Action>(
            args, state, base, elements_per_repeat, full_mask, input_inner, address_invariant, context, shape_args...);
      }
      if (tail_count != 0U) {
        uint32_t mask_count = tail_count;
        MicroAPI::MaskReg tail_mask = MicroAPI::UpdateMask<uint32_t>(mask_count);
        IndirectLoadSimdRunRepeat<Mode, X, Index, Rank, Axis, Action>(args, state, full_repeats * elements_per_repeat,
                                                                      tail_count, tail_mask, input_inner,
                                                                      address_invariant, context, shape_args...);
      }
    }
  }
}

template <typename X, typename Index, typename Action>
__aicore__ inline void IndirectLoadSimdRunReuse(IndirectLoadSimdArgs<X, Index> &args,
                                                const IndirectLoadSimdAddressContext &context) {
  using Traits = IndirectLoadSimdRegTraits<X, Index, 1, 0>;
  using IndexPolicy = typename Traits::IndexPolicy;
  using ValuePolicy = typename Traits::ValuePolicy;
  constexpr uint32_t elements_per_repeat = VECTOR_REG_WIDTH / sizeof(uint32_t);
  const uint16_t full_repeats = static_cast<uint16_t>(args.actual_size / elements_per_repeat);
  const uint32_t tail_count = args.actual_size % elements_per_repeat;
  __VEC_SCOPE__ {
    typename IndexPolicy::LoadState state;
    IndexPolicy::Init(state, args.index);
    MicroAPI::RegTensor<uint32_t> inner_offset;
    MicroAPI::RegTensor<uint32_t> input_inner;
    MicroAPI::MaskReg full_mask = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    IndirectLoadSimdInitInnerOffset(inner_offset, context, full_mask);
    MicroAPI::Duplicate(input_inner, context.input_inner, full_mask);
    for (uint16_t repeat = 0U; repeat < full_repeats; ++repeat) {
      const uint32_t base = repeat * elements_per_repeat;
      MicroAPI::RegTensor<uint32_t> source_index;
      IndexPolicy::Load(source_index, state, elements_per_repeat);
      MicroAPI::Mul(source_index, source_index, input_inner, full_mask);
      MicroAPI::Add(source_index, source_index, inner_offset, full_mask);
      Action::template Commit<ValuePolicy>(args.x, args.y, args.offsets, context.input_actual_size, base, source_index,
                                           full_mask, elements_per_repeat);
    }
    if (tail_count != 0U) {
      const uint32_t base = full_repeats * elements_per_repeat;
      uint32_t mask_count = tail_count;
      MicroAPI::MaskReg tail_mask = MicroAPI::UpdateMask<uint32_t>(mask_count);
      MicroAPI::RegTensor<uint32_t> source_index;
      IndexPolicy::Load(source_index, state, tail_count);
      MicroAPI::Mul(source_index, source_index, input_inner, tail_mask);
      MicroAPI::Add(source_index, source_index, inner_offset, tail_mask);
      Action::template Commit<ValuePolicy>(args.x, args.y, args.offsets, context.input_actual_size, base, source_index,
                                           tail_mask, tail_count);
    }
  }
}

template <typename X, typename Index, int32_t Rank, int32_t Axis, typename Action>
struct IndirectLoadSimdDispatchPolicy {
  using Args = IndirectLoadSimdArgs<X, Index>;

  __aicore__ inline static void RunReuse(Args &args, const IndirectLoadSimdAddressContext &context) {
    IndirectLoadSimdRunReuse<X, Index, Action>(args, context);
  }

  template <IndirectLoadSimdAddressMode Mode, typename... ShapeArgs>
  __aicore__ inline static void RunMode(Args &args, const IndirectLoadSimdAddressContext &context,
                                        ShapeArgs... shape_args) {
    IndirectLoadSimdRunMode<Mode, X, Index, Rank, Axis, Action>(args, context, shape_args...);
  }
};

template <int32_t Rank, int32_t Axis, typename DispatchPolicy, typename... ShapeArgs>
__aicore__ inline void IndirectLoadSimdDispatch(typename DispatchPolicy::Args &args,
                                                const IndirectLoadSimdAddressContext &context,
                                                ShapeArgs... shape_args) {
  constexpr uint32_t reuse_elements = VECTOR_REG_WIDTH / sizeof(uint32_t);
  if constexpr (Axis + 1 == Rank) {
    DispatchPolicy::template RunMode<IndirectLoadSimdAddressMode::kDirect>(args, context, shape_args...);
    return;
  }
  if (!context.inner_layout_matches) {
    DispatchPolicy::template RunMode<IndirectLoadSimdAddressMode::kStrided>(args, context, shape_args...);
    return;
  }
  if (IndirectLoadSimdIsPowerOfTwo(context.index_inner)) {
    if (args.actual_size > reuse_elements && (reuse_elements & (context.index_inner - 1U)) == 0U) {
      DispatchPolicy::RunReuse(args, context);
    } else {
      DispatchPolicy::template RunMode<IndirectLoadSimdAddressMode::kDensePow2>(args, context, shape_args...);
    }
    return;
  }
  DispatchPolicy::template RunMode<IndirectLoadSimdAddressMode::kDenseGeneric>(args, context, shape_args...);
}

template <typename X, typename Index, int32_t Rank, int32_t Axis, typename... ShapeArgs>
__aicore__ inline void IndirectLoadSimdRegGather(__ubuf__ X *x, __ubuf__ Index *index, __ubuf__ X *y,
                                                 uint32_t actual_size, const IndirectLoadSimdAddressContext &context,
                                                 ShapeArgs... shape_args) {
  static_assert(IndirectLoadSimdRegTraits<X, Index, Rank, Axis>::kSupported,
                "IndirectLoad SIMD register Gather specialization is not implemented.");
  IndirectLoadSimdArgs<X, Index> args{x, index, y, nullptr, actual_size};
  using Policy = IndirectLoadSimdDispatchPolicy<X, Index, Rank, Axis, IndirectLoadSimdGatherAction<X>>;
  IndirectLoadSimdDispatch<Rank, Axis, Policy>(args, context, shape_args...);
}

template <typename X, typename Index, int32_t Rank, int32_t Axis, typename... ShapeArgs>
__aicore__ inline void IndirectLoadSimdBuildOffsets(__ubuf__ Index *index, __ubuf__ uint32_t *offsets,
                                                    uint32_t actual_size, const IndirectLoadSimdAddressContext &context,
                                                    ShapeArgs... shape_args) {
  IndirectLoadSimdArgs<X, Index> args{nullptr, index, nullptr, offsets, actual_size};
  using Policy = IndirectLoadSimdDispatchPolicy<X, Index, Rank, Axis, IndirectLoadSimdOffsetAction<X>>;
  IndirectLoadSimdDispatch<Rank, Axis, Policy>(args, context, shape_args...);
}

template <int32_t Rank, int32_t Axis, typename... ShapeArgs>
__aicore__ inline IndirectLoadSimdAddressContext InitIndirectLoadSimdAddressContext(int64_t output_offset,
                                                                                    uint32_t input_actual_size,
                                                                                    int64_t input_axis,
                                                                                    ShapeArgs... shape_args) {
  (void)input_axis;
  if constexpr (Axis + 1 == Rank) {
    return {0U, input_actual_size, 1U, 1U, true};
  }
  const int64_t shape[] = {static_cast<int64_t>(shape_args)...};
  uint32_t index_inner = 1U;
  uint32_t expected_input_stride = 1U;
  bool inner_layout_matches = true;
  for (int32_t dim = Rank - 1; dim > Axis; --dim) {
    inner_layout_matches &= static_cast<uint32_t>(shape[Rank + dim]) == expected_input_stride;
    expected_input_stride *= static_cast<uint32_t>(shape[dim]);
    index_inner *= static_cast<uint32_t>(shape[dim]);
  }
  const uint32_t input_inner = static_cast<uint32_t>(shape[Rank + Axis]);
  const uint32_t output_slice = static_cast<uint32_t>(shape[Axis]) * index_inner;
  return {static_cast<uint32_t>(output_offset % static_cast<int64_t>(output_slice)), input_actual_size, input_inner,
          index_inner, inner_layout_matches};
}

template <typename X, typename Index, int32_t Rank, int32_t Axis, typename... ShapeArgs>
__aicore__ inline void IndirectLoadSimdDenseImpl(const LocalTensor<X> &x, const LocalTensor<Index> &index,
                                                 const LocalTensor<X> &y, uint32_t actual_size, int64_t output_offset,
                                                 uint32_t input_actual_size, int64_t input_axis,
                                                 ShapeArgs... shape_args) {
  static_assert(Rank > 0 && Axis >= 0 && Axis < Rank, "IndirectLoad SIMD rank or axis is invalid.");
  static_assert(sizeof...(ShapeArgs) == static_cast<size_t>(2 * Rank), "IndirectLoad SIMD shape is invalid.");
  __ubuf__ X *x_address = (__ubuf__ X *)x.GetPhyAddr();
  __ubuf__ Index *index_address = (__ubuf__ Index *)index.GetPhyAddr();
  __ubuf__ X *y_address = (__ubuf__ X *)y.GetPhyAddr();
  const Internal::IndirectLoadSimdAddressContext context = Internal::InitIndirectLoadSimdAddressContext<Rank, Axis>(
      output_offset, input_actual_size, input_axis, shape_args...);
  Internal::IndirectLoadSimdRegGather<X, Index, Rank, Axis>(x_address, index_address, y_address, actual_size, context,
                                                            shape_args...);
}

template <typename X>
__aicore__ inline void IndirectLoadSimdCopyUnaligned(__ubuf__ X *dst, __ubuf__ X *src, uint32_t count) {
  constexpr uint32_t elements_per_vector = VECTOR_REG_WIDTH / sizeof(X);
  for (uint32_t offset = 0U; offset < count; offset += elements_per_vector) {
    const uint32_t copy_count = (count - offset < elements_per_vector) ? count - offset : elements_per_vector;
    __VEC_SCOPE__ {
      MicroAPI::UnalignReg src_unalign;
      MicroAPI::UnalignReg dst_unalign;
      MicroAPI::RegTensor<X> value;
      __ubuf__ X *src_ptr = src + offset;
      __ubuf__ X *dst_ptr = dst + offset;
      MicroAPI::DataCopyUnAlignPre(src_unalign, src_ptr);
      MicroAPI::DataCopyUnAlign(value, src_unalign, src_ptr, copy_count);
      MicroAPI::DataCopyUnAlign(dst_ptr, value, dst_unalign, copy_count);
      MicroAPI::DataCopyUnAlignPost(dst_ptr, dst_unalign, 0);
    }
  }
}

template <typename X>
__aicore__ inline void IndirectLoadSimdCopyContiguous(const LocalTensor<X> &dst_tensor,
                                                      const LocalTensor<X> &src_tensor, uint32_t count,
                                                      int64_t source_offset, int64_t output_offset) {
  const int64_t block_elements = static_cast<int64_t>(AscendC::ONE_BLK_SIZE / sizeof(X));
  if (count == 0U || block_elements <= 0) {
    return;
  }
  const int64_t source_alignment = source_offset % block_elements;
  const int64_t output_alignment = output_offset % block_elements;
  const bool can_align_block = source_alignment == output_alignment;
  uint32_t head = 0U;
  if (can_align_block) {
    head = static_cast<uint32_t>((block_elements - source_alignment) % block_elements);
    if (head > count) {
      head = count;
    }
  }
  __ubuf__ X *dst = (__ubuf__ X *)dst_tensor.GetPhyAddr() + output_offset;
  __ubuf__ X *src = (__ubuf__ X *)src_tensor.GetPhyAddr() + source_offset;
  IndirectLoadSimdCopyUnaligned<X>(dst, src, head);
  const uint32_t aligned_count =
      ((count - head) / static_cast<uint32_t>(block_elements)) * static_cast<uint32_t>(block_elements);
  const bool block_addresses_aligned = ((source_offset + static_cast<int64_t>(head)) % block_elements == 0) &&
                                       ((output_offset + static_cast<int64_t>(head)) % block_elements == 0);
  if (aligned_count != 0U && block_addresses_aligned) {
    AscendC::DataCopy(dst_tensor[output_offset + head], src_tensor[source_offset + head], aligned_count);
  } else {
    IndirectLoadSimdCopyUnaligned<X>(dst + head, src + head, aligned_count);
  }
  IndirectLoadSimdCopyUnaligned<X>(dst + head + aligned_count, src + head + aligned_count,
                                   count - head - aligned_count);
}

template <typename X, typename Index, int32_t Rank, int32_t Axis, typename... ShapeArgs>
__aicore__ inline bool TryIndirectLoadSimdEmbedding(const LocalTensor<X> &x, const LocalTensor<Index> &index,
                                                    const LocalTensor<X> &y, uint32_t logical_size,
                                                    int64_t output_offset, ShapeArgs... shape_args) {
  static_assert(sizeof...(ShapeArgs) == static_cast<size_t>(4 * Rank), "IndirectLoad SIMD Embedding shape is invalid.");
  if constexpr (Rank > 1 && Axis >= 0 && Axis + 1 < Rank &&
                (std::is_same_v<Index, int32_t> || std::is_same_v<Index, int64_t>) &&
                sizeof(X) <= AscendC::ONE_BLK_SIZE) {
    const int64_t shape[] = {static_cast<int64_t>(shape_args)...};
    const int32_t input_stride_base = Rank;
    const int32_t index_stride_base = 2 * Rank;
    const int32_t output_stride_base = 3 * Rank;
    int64_t payload_span = 1;
    int64_t expected_input_stride = 1;
    for (int32_t dim = Rank - 1; dim > Axis; --dim) {
      if (shape[dim] <= 0 || shape[index_stride_base + dim] != 0 ||
          shape[input_stride_base + dim] != expected_input_stride || shape[output_stride_base + dim] != payload_span) {
        return false;
      }
      payload_span *= shape[dim];
      expected_input_stride *= shape[dim];
    }
    const int64_t output_axis_stride = shape[output_stride_base + Axis];
    if (payload_span <= 0 || shape[input_stride_base + Axis] != expected_input_stride || output_offset < 0 ||
        output_axis_stride <= 0 || output_axis_stride < payload_span) {
      return false;
    }
    const int64_t index_axis_stride = shape[index_stride_base + Axis];
    if (index_axis_stride <= 0) {
      return false;
    }
    const int64_t output_slice_span = shape[Axis] * payload_span;
    if (shape[Axis] <= 0 || output_slice_span <= 0) {
      return false;
    }
    int64_t outer_index = output_offset / output_slice_span;
    int64_t input_outer_base = 0;
    for (int32_t dim = Axis - 1; dim >= 0; --dim) {
      if (shape[dim] <= 0) {
        return false;
      }
      if (shape[index_stride_base + dim] != 0) {
        // The index source is compacted into UB by the producer-side copy.  This
        // fast path only supports one index row per local tile; varying outer
        // index dimensions require the general strided implementation.
        return false;
      }
      const int64_t coordinate = outer_index % shape[dim];
      outer_index /= shape[dim];
      input_outer_base += coordinate * shape[input_stride_base + dim];
    }

    uint32_t copied = 0U;
    int64_t output_local_offset = 0;
    while (copied < logical_size) {
      const int64_t output_index = output_offset + static_cast<int64_t>(copied);
      const int64_t payload_offset = output_index % payload_span;
      int64_t lookup_index = output_index / payload_span;
      int64_t index_axis_coordinate = 0;
      int64_t input_offset = -input_outer_base;
      for (int32_t dim = Axis; dim >= 0; --dim) {
        if (shape[dim] <= 0) {
          return false;
        }
        const int64_t coordinate = lookup_index % shape[dim];
        lookup_index /= shape[dim];
        if (dim == Axis) {
          index_axis_coordinate = coordinate;
        }
        if (dim != Axis) {
          input_offset += coordinate * shape[input_stride_base + dim];
        }
      }
      const int64_t index_value = static_cast<int64_t>(index.GetValue(index_axis_coordinate * index_axis_stride));
      const int64_t source_offset = index_value * shape[input_stride_base + Axis] + input_offset + payload_offset;
      const int64_t payload_remaining = payload_span - payload_offset;
      const uint32_t copy_count = static_cast<uint32_t>(
          (static_cast<int64_t>(logical_size - copied) < payload_remaining) ? logical_size - copied
                                                                            : payload_remaining);
      if constexpr (Axis == 0) {
        // With axis 0 there are no outer coordinates.  The destination offset
        // can therefore be advanced directly across payload slices.
        IndirectLoadSimdCopyContiguous<X>(y, x, copy_count, source_offset, output_local_offset);
        copied += copy_count;
        output_local_offset += copy_count;
        if (static_cast<int64_t>(copy_count) == payload_remaining && copied < logical_size) {
          output_local_offset += shape[output_stride_base + Axis] - payload_span;
        }
      } else {
        int64_t output_coordinate = output_index;
        int64_t output_base = output_offset;
        output_local_offset = 0;
        for (int32_t dim = Rank - 1; dim >= 0; --dim) {
          const int64_t coordinate = output_coordinate % shape[dim];
          const int64_t base_coordinate = output_base % shape[dim];
          output_coordinate /= shape[dim];
          output_base /= shape[dim];
          output_local_offset += (coordinate - base_coordinate) * shape[output_stride_base + dim];
        }
        IndirectLoadSimdCopyContiguous<X>(y, x, copy_count, source_offset, output_local_offset);
        copied += copy_count;
      }
    }
    return true;
  }
  return false;
}

template <int32_t Rank, int32_t Axis>
struct IndirectLoadSimdStridedContext {
  const int64_t *shape;
  const int64_t *output_strides;
  uint32_t logical_size;
  uint32_t physical_size;
  int64_t output_offset;
  int64_t index_inner;
  int64_t output_slice_count;
  int64_t input_window_base;
  int64_t index_window_base;
};

template <int32_t Rank>
__aicore__ inline void BuildIndirectLoadSimdStridedShape(int64_t (&shape)[3 * Rank],
                                                         const IndirectLoadSimdStridedParams<Rank> &params) {
  for (int32_t dim = 0; dim < Rank; ++dim) {
    shape[dim] = params.index_sizes[dim];
    shape[Rank + dim] = params.input_strides[dim];
    shape[2 * Rank + dim] = params.index_strides[dim];
  }
}

template <int32_t Rank, int32_t Axis>
__aicore__ inline IndirectLoadSimdStridedContext<Rank, Axis> MakeIndirectLoadSimdStridedContext(
    const int64_t *shape, const IndirectLoadSimdStridedParams<Rank> &params) {
  const int64_t index_inner = Internal::IndirectLoadSimdInnerSize<Rank - 1, Axis>::Call(shape);
  const int64_t output_slice_count = shape[Axis] * index_inner;
  const int64_t outer_begin = params.output_offset / output_slice_count;
  int64_t input_window_base = 0;
  int64_t index_window_base = 0;
  if constexpr (Axis > 0) {
    input_window_base = Internal::IndirectLoadSimdOuterOffset<Axis - 1, Rank>::Call(outer_begin, shape);
    index_window_base = Internal::IndirectLoadSimdOuterOffset<Axis - 1, 2 * Rank>::Call(outer_begin, shape);
  }
  return {shape,       params.output_strides, params.logical_size, params.physical_size, params.output_offset,
          index_inner, output_slice_count,    input_window_base,   index_window_base};
}

template <int32_t Rank, int32_t Axis>
__aicore__ inline int64_t GetIndirectLoadSimdStridedIndexOffset(int64_t global_idx,
                                                                const IndirectLoadSimdStridedContext<Rank, Axis> &ctx) {
  const int64_t outer_global = global_idx / ctx.output_slice_count;
  const int64_t tail = global_idx % ctx.output_slice_count;
  const int64_t axis_coord = tail / ctx.index_inner;
  const int64_t inner = tail % ctx.index_inner;
  int64_t index_offset = axis_coord * ctx.shape[2 * Rank + Axis];
  if constexpr (Axis > 0) {
    index_offset += Internal::IndirectLoadSimdOuterOffset<Axis - 1, 2 * Rank>::Call(outer_global, ctx.shape) -
                    ctx.index_window_base;
  }
  if constexpr (Axis + 1 < Rank) {
    index_offset += Internal::IndirectLoadSimdInnerOffset<Rank - 1, Axis, 2 * Rank>::Call(inner, ctx.shape);
  }
  return index_offset;
}

template <int32_t Rank, int32_t Axis>
__aicore__ inline int64_t GetIndirectLoadSimdStridedInputOffset(int64_t global_idx, int64_t index_value,
                                                                const IndirectLoadSimdStridedContext<Rank, Axis> &ctx) {
  const int64_t outer_global = global_idx / ctx.output_slice_count;
  const int64_t tail = global_idx % ctx.output_slice_count;
  const int64_t inner = tail % ctx.index_inner;
  int64_t input_inner_offset = 0;
  int64_t input_outer_offset = 0;
  if constexpr (Axis + 1 < Rank) {
    input_inner_offset = Internal::IndirectLoadSimdInnerOffset<Rank - 1, Axis, Rank>::Call(inner, ctx.shape);
  }
  if constexpr (Axis > 0) {
    input_outer_offset =
        Internal::IndirectLoadSimdOuterOffset<Axis - 1, Rank>::Call(outer_global, ctx.shape) - ctx.input_window_base;
  }
  return input_outer_offset + index_value * ctx.shape[Rank + Axis] + input_inner_offset;
}

template <typename X, typename Index, int32_t Rank, int32_t Axis>
__aicore__ inline void BuildIndirectLoadSimdStridedOffsets(const LocalTensor<Index> &index,
                                                           const LocalTensor<uint32_t> &offsets,
                                                           const IndirectLoadSimdStridedContext<Rank, Axis> &ctx) {
  // 仿射快路径：inner 域塌缩、tile 为整行对齐窗口（单行即 rows=1 特例）、index 在 tile
  // 内连续时，逐元素坐标折叠退化为常量三元组，offsets 满足线性模型，可按行用向量仿射构建：
  //   offsets[seg * slice + j] = (input_base(seg) + idx[index_base(seg) + j] * gather_stride) * sizeof(X)
  // 行数 R 次 Cast/Muls/Adds 替代 R*slice 次标量折叠；未整行对齐/跨步 index 形态走通用路径。
  if (ctx.index_inner == 1) {
    const int64_t slice_count = ctx.output_slice_count;
    const int64_t logical = static_cast<int64_t>(ctx.logical_size);
    const int64_t rows =
        (ctx.output_offset % slice_count == 0 && logical % slice_count == 0) ? logical / slice_count : 0;
    if (rows > 0 && ctx.shape[2 * Rank + Axis] == 1 && ctx.shape[2 * Rank + Rank - 1] == 1) {
      const int64_t outer_base = ctx.output_offset / slice_count;
      // offsets 为字节偏移（与通用路径 source_offset × sizeof(X) 一致）。
      const int64_t stride_bytes = ctx.shape[Rank + Axis] * static_cast<int64_t>(sizeof(X));
      // 一次 Cast + 一次 Muls + 一次 Adds 批量构建：offsets 与 int32 视图同址，Gather 按
      // uint32 读；Index 为 32 位时无需窄化 Cast，直接按 int32 视图仿射。
      auto offsets32 = offsets.template ReinterpretCast<int32_t>();
      const int64_t seg_count = rows;
      const int64_t seg_len = slice_count;
      for (int64_t seg = 0; seg < seg_count; ++seg) {
        const int64_t outer_global = outer_base + seg;
        int64_t index_base = 0;
        int64_t input_base = 0;
        if constexpr (Axis > 0) {
          index_base += Internal::IndirectLoadSimdOuterOffset<Axis - 1, 2 * Rank>::Call(outer_global, ctx.shape) -
                        ctx.index_window_base;
          input_base = Internal::IndirectLoadSimdOuterOffset<Axis - 1, Rank>::Call(outer_global, ctx.shape) -
                       ctx.input_window_base;
        }
        const int64_t base_bytes = input_base * static_cast<int64_t>(sizeof(X));
        const auto index_view = index[index_base];
        const auto seg_view = offsets32[seg * seg_len];
        if constexpr (sizeof(Index) == sizeof(int64_t)) {
          Cast(seg_view, index_view, AscendC::RoundMode::CAST_NONE, seg_len);
          Muls(seg_view, seg_view, static_cast<int32_t>(stride_bytes), seg_len);
          Adds(seg_view, seg_view, static_cast<int32_t>(base_bytes), seg_len);
        } else {
          Muls(seg_view, index_view.template ReinterpretCast<int32_t>(), static_cast<int32_t>(stride_bytes), seg_len);
          Adds(seg_view, seg_view, static_cast<int32_t>(base_bytes), seg_len);
        }
      }
      return;
    }
  }
  // —— 通用路径：逐元素坐标折叠（非尾轴 gather 含 inner 残留 / 跨 slice / index
  // 跨步等形态，无法塌缩为常量仿射基址）。
  for (uint32_t i = 0; i < ctx.logical_size; ++i) {
    const int64_t global_idx = ctx.output_offset + static_cast<int64_t>(i);
    const int64_t index_offset = GetIndirectLoadSimdStridedIndexOffset<Rank, Axis>(global_idx, ctx);
    const int64_t index_value = static_cast<int64_t>(index.GetValue(index_offset));
    const int64_t source_offset = GetIndirectLoadSimdStridedInputOffset<Rank, Axis>(global_idx, index_value, ctx);
    offsets.SetValue(i, static_cast<uint32_t>(source_offset * sizeof(X)));
  }
}

template <int32_t Rank, int32_t Axis>
__aicore__ inline int64_t GetIndirectLoadSimdStridedOutputOffset(
    int64_t global_idx, const IndirectLoadSimdStridedContext<Rank, Axis> &ctx) {
  int64_t current = global_idx;
  int64_t base = ctx.output_offset;
  int64_t output_offset = 0;
  for (int32_t dim = Rank - 1; dim >= 0; --dim) {
    const int64_t coord = current % ctx.shape[dim];
    const int64_t base_coord = base % ctx.shape[dim];
    current /= ctx.shape[dim];
    base /= ctx.shape[dim];
    output_offset += (coord - base_coord) * ctx.output_strides[dim];
  }
  return output_offset;
}

template <typename X, int32_t Rank, int32_t Axis>
__aicore__ inline void ScatterIndirectLoadSimdStridedOutput(const LocalTensor<X> &y,
                                                            const IndirectLoadSimdStridedContext<Rank, Axis> &ctx) {
  // 整行对齐窗口（Rank=2 尾轴 gather，单行即 logical==slice_count 特例）且
  // output_strides == {slice_count, 1} 时，行内与行间均线性连续，逐元素输出偏移恒等
  // 于 i，Gather 写入的紧凑结果已在最终位置，无需重排；其余形态逐元素重排。
  if constexpr (Rank == 2 && Axis == 1) {
    if (ctx.index_inner == 1 && ctx.output_strides[0] == ctx.output_slice_count && ctx.output_strides[1] == 1 &&
        ctx.output_offset % ctx.output_slice_count == 0 &&
        static_cast<int64_t>(ctx.logical_size) % ctx.output_slice_count == 0) {
      return;
    }
  }
  // Gather writes packed logical values first. Repack from the end so padding lanes do not
  // overwrite a source value that is still needed by the supported compact output layouts.
  for (int64_t i = static_cast<int64_t>(ctx.logical_size) - 1; i >= 0; --i) {
    const int64_t output_offset = GetIndirectLoadSimdStridedOutputOffset<Rank, Axis>(ctx.output_offset + i, ctx);
    y.SetValue(output_offset, y.GetValue(i));
  }
}

// Strided SIMD output may contain alignment holes (for example, a logical 23-element row
// is stored with a physical stride of 24). Keep the gather packed and expand it into the
// padded destination layout afterwards.
template <typename X, typename Index, int32_t Rank, int32_t Axis>
__aicore__ inline void IndirectLoadSimdStridedImpl(const LocalTensor<X> &x, const LocalTensor<Index> &index,
                                                   const LocalTensor<X> &y, const LocalTensor<uint8_t> &tmp,
                                                   const IndirectLoadSimdStridedParams<Rank> &params) {
  static_assert(Rank > 0 && Axis >= 0 && Axis < Rank, "IndirectLoad SIMD rank or axis is invalid.");
  int64_t shape[3 * Rank];
  BuildIndirectLoadSimdStridedShape(shape, params);
  const auto context = MakeIndirectLoadSimdStridedContext<Rank, Axis>(shape, params);
  LocalTensor<uint32_t> offsets = tmp.template ReinterpretCast<uint32_t>();
  BuildIndirectLoadSimdStridedOffsets<X, Index, Rank, Axis>(index, offsets, context);
  int32_t offset_event_id = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_V));
  AscendC::SetFlag<AscendC::HardEvent::S_V>(offset_event_id);
  AscendC::WaitFlag<AscendC::HardEvent::S_V>(offset_event_id);
  Gather(y, x, offsets, static_cast<uint32_t>(0), context.logical_size);
  ScatterIndirectLoadSimdStridedOutput<X, Rank, Axis>(y, context);
}
}  // namespace Internal

template <typename X, typename Index, int32_t Rank, int32_t Axis, typename... ShapeArgs>
__aicore__ inline void IndirectLoadSimd(const LocalTensor<X> &x, const LocalTensor<Index> &index,
                                        const LocalTensor<X> &y, uint32_t actual_size, int64_t output_offset,
                                        uint32_t input_actual_size, int64_t input_axis, ShapeArgs... shape_args) {
  static_assert(Rank > 0 && Axis >= 0 && Axis < Rank, "IndirectLoad SIMD rank or axis is invalid.");
  static_assert(sizeof...(ShapeArgs) == static_cast<size_t>(2 * Rank), "IndirectLoad SIMD shape is invalid.");
  Internal::IndirectLoadSimdDenseImpl<X, Index, Rank, Axis>(x, index, y, actual_size, output_offset, input_actual_size,
                                                            input_axis, shape_args...);
}

template <typename X, typename Index, int32_t Rank, int32_t Axis>
__aicore__ inline void IndirectLoadSimdStrided(const LocalTensor<X> &x, const LocalTensor<Index> &index,
                                               const LocalTensor<X> &y, const LocalTensor<uint8_t> &tmp,
                                               const IndirectLoadSimdStridedParams<Rank> &params) {
  static_assert(Rank > 0 && Axis >= 0 && Axis < Rank, "IndirectLoad SIMD rank or axis is invalid.");
  Internal::IndirectLoadSimdStridedImpl<X, Index, Rank, Axis>(x, index, y, tmp, params);
}

template <typename X, typename Index, int32_t Rank, int32_t Axis, typename... ShapeArgs>
__aicore__ inline void IndirectLoadSimdGatherApi(const LocalTensor<X> &x, const LocalTensor<Index> &index,
                                                 const LocalTensor<X> &y, uint32_t actual_size, int64_t output_offset,
                                                 uint32_t input_actual_size, int64_t input_axis,
                                                 ShapeArgs... shape_args) {
  static_assert(Rank > 0 && Axis >= 0 && Axis < Rank, "IndirectLoad SIMD rank or axis is invalid.");
  static_assert(sizeof...(ShapeArgs) == static_cast<size_t>(2 * Rank), "IndirectLoad SIMD shape is invalid.");
  // Gather consumes uint32 byte offsets, so reuse the dead index UB instead of allocating another buffer.
  __ubuf__ Index *index_address = (__ubuf__ Index *)index.GetPhyAddr();
  __ubuf__ uint32_t *offset_address = (__ubuf__ uint32_t *)index.GetPhyAddr();
  const Internal::IndirectLoadSimdAddressContext context = Internal::InitIndirectLoadSimdAddressContext<Rank, Axis>(
      output_offset, input_actual_size, input_axis, shape_args...);
  Internal::IndirectLoadSimdBuildOffsets<X, Index, Rank, Axis>(index_address, offset_address, actual_size, context,
                                                               shape_args...);
  PipeBarrier<PIPE_V>();
  const LocalTensor<uint32_t> offsets = index.template ReinterpretCast<uint32_t>();
  Gather(y, x, offsets, 0U, actual_size);
}

namespace Internal {
template <typename CaseTag, int32_t Rank>
struct IndirectLoadSimdCaseSelector;

template <IndirectLoadSimdFallback Fallback, bool TryEmbedding, int32_t Rank>
struct IndirectLoadSimdCaseSelector<IndirectLoadSimdCaseTag<Fallback, TryEmbedding>, Rank> {
  using Params =
      std::conditional_t<TryEmbedding, IndirectLoadSimdEmbeddingParams<Rank>,
                         std::conditional_t<Fallback == IndirectLoadSimdFallback::kStrided,
                                            IndirectLoadSimdStridedParams<Rank>, IndirectLoadSimdDenseParams<Rank>>>;

  template <typename X, typename Index, int32_t Axis>
  __aicore__ inline static void RunDense(const LocalTensor<X> &x, const LocalTensor<Index> &index,
                                         const LocalTensor<X> &y, const Params &params) {
    static_assert(Fallback != IndirectLoadSimdFallback::kStrided,
                  "IndirectLoad SIMD dense facade requires a dense fallback.");
    if constexpr (TryEmbedding) {
      if (TryEmbeddingImpl<X, Index, Axis>(x, index, y, params, std::make_index_sequence<Rank>{})) {
        return;
      }
    }
    RunDenseFallback<X, Index, Axis>(x, index, y, params, std::make_index_sequence<Rank>{});
  }

  template <typename X, typename Index, int32_t Axis>
  __aicore__ inline static void RunStrided(const LocalTensor<X> &x, const LocalTensor<Index> &index,
                                           const LocalTensor<X> &y, const LocalTensor<uint8_t> &tmp,
                                           const Params &params) {
    static_assert(Fallback == IndirectLoadSimdFallback::kStrided,
                  "IndirectLoad SIMD strided facade requires a strided fallback.");
    if constexpr (TryEmbedding) {
      if (TryEmbeddingImpl<X, Index, Axis>(x, index, y, params, std::make_index_sequence<Rank>{})) {
        return;
      }
    }
    if constexpr (TryEmbedding) {
      IndirectLoadSimdStridedParams<Rank> strided_params{
          params.logical_size, params.physical_size, params.output_offset, {}, {}, {}, {}};
      for (int32_t dim = 0; dim < Rank; ++dim) {
        strided_params.index_sizes[dim] = params.index_sizes[dim];
        strided_params.input_strides[dim] = params.input_strides[dim];
        strided_params.index_strides[dim] = params.index_strides[dim];
        strided_params.output_strides[dim] = params.output_strides[dim];
      }
      AscendC::IndirectLoadSimdStrided<X, Index, Rank, Axis>(x, index, y, tmp, strided_params);
    } else {
      AscendC::IndirectLoadSimdStrided<X, Index, Rank, Axis>(x, index, y, tmp, params);
    }
  }

 private:
  template <typename X, typename Index, int32_t Axis, size_t... Dims>
  __aicore__ inline static bool TryEmbeddingImpl(const LocalTensor<X> &x, const LocalTensor<Index> &index,
                                                 const LocalTensor<X> &y, const Params &params,
                                                 std::index_sequence<Dims...>) {
    return TryIndirectLoadSimdEmbedding<X, Index, Rank, Axis>(
        x, index, y, params.logical_size, params.output_offset, params.index_sizes[Dims]...,
        params.input_strides[Dims]..., params.index_strides[Dims]..., params.output_strides[Dims]...);
  }

  template <typename X, typename Index, int32_t Axis, size_t... Dims>
  __aicore__ inline static void RunDenseFallback(const LocalTensor<X> &x, const LocalTensor<Index> &index,
                                                 const LocalTensor<X> &y, const Params &params,
                                                 std::index_sequence<Dims...>) {
    if constexpr (Fallback == IndirectLoadSimdFallback::kRegisterGather) {
      AscendC::IndirectLoadSimd<X, Index, Rank, Axis>(x, index, y, params.physical_size, params.output_offset,
                                                      params.input_actual_size, params.input_axis,
                                                      params.index_sizes[Dims]..., params.input_strides[Dims]...);
    } else {
      AscendC::IndirectLoadSimdGatherApi<X, Index, Rank, Axis>(
          x, index, y, params.physical_size, params.output_offset, params.input_actual_size, params.input_axis,
          params.index_sizes[Dims]..., params.input_strides[Dims]...);
    }
  }
};
}  // namespace Internal

template <typename CaseTag, int32_t Rank>
using IndirectLoadSimdParams = typename Internal::IndirectLoadSimdCaseSelector<CaseTag, Rank>::Params;

template <typename X, typename Index, int32_t Rank, int32_t Axis, bool TryEmbedding>
__aicore__ inline void IndirectLoadSimd(
    const LocalTensor<X> &x, const LocalTensor<Index> &index, const LocalTensor<X> &y,
    IndirectLoadSimdCaseTag<IndirectLoadSimdFallback::kRegisterGather, TryEmbedding>,
    const IndirectLoadSimdParams<IndirectLoadSimdCaseTag<IndirectLoadSimdFallback::kRegisterGather, TryEmbedding>, Rank>
        &params) {
  using CaseTag = IndirectLoadSimdCaseTag<IndirectLoadSimdFallback::kRegisterGather, TryEmbedding>;
  Internal::IndirectLoadSimdCaseSelector<CaseTag, Rank>::template RunDense<X, Index, Axis>(x, index, y, params);
}

template <typename X, typename Index, int32_t Rank, int32_t Axis, bool TryEmbedding>
__aicore__ inline void IndirectLoadSimdGatherApi(
    const LocalTensor<X> &x, const LocalTensor<Index> &index, const LocalTensor<X> &y,
    IndirectLoadSimdCaseTag<IndirectLoadSimdFallback::kGatherApi, TryEmbedding>,
    const IndirectLoadSimdParams<IndirectLoadSimdCaseTag<IndirectLoadSimdFallback::kGatherApi, TryEmbedding>, Rank>
        &params) {
  using CaseTag = IndirectLoadSimdCaseTag<IndirectLoadSimdFallback::kGatherApi, TryEmbedding>;
  Internal::IndirectLoadSimdCaseSelector<CaseTag, Rank>::template RunDense<X, Index, Axis>(x, index, y, params);
}

template <typename X, typename Index, int32_t Rank, int32_t Axis, bool TryEmbedding>
__aicore__ inline void IndirectLoadSimdStrided(
    const LocalTensor<X> &x, const LocalTensor<Index> &index, const LocalTensor<X> &y, const LocalTensor<uint8_t> &tmp,
    IndirectLoadSimdCaseTag<IndirectLoadSimdFallback::kStrided, TryEmbedding>,
    const IndirectLoadSimdParams<IndirectLoadSimdCaseTag<IndirectLoadSimdFallback::kStrided, TryEmbedding>, Rank>
        &params) {
  using CaseTag = IndirectLoadSimdCaseTag<IndirectLoadSimdFallback::kStrided, TryEmbedding>;
  Internal::IndirectLoadSimdCaseSelector<CaseTag, Rank>::template RunStrided<X, Index, Axis>(x, index, y, tmp, params);
}

}  // namespace AscendC

#endif  // AUTOFUSE_V35_ASCENDC_API_REGBASE_INDIRECT_LOAD_SIMD_H_
