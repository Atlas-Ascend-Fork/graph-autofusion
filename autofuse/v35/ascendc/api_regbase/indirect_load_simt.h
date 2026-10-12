/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_V35_ASCENDC_API_REGBASE_INDIRECT_LOAD_SIMT_H_
#define AUTOFUSE_V35_ASCENDC_API_REGBASE_INDIRECT_LOAD_SIMT_H_

#include <type_traits>

// The SIMT runtime (async_invoke thread stacks and launch metadata) occupies the top of UB.
// Kernels containing this header must keep temporary UB buffers below that reserve, e.g. the
// ReduceInit maskBuf selected by this macro in reduce_init.h.
#ifndef AUTOFUSE_SIMT_RESERVED_UB_SIZE
#define AUTOFUSE_SIMT_RESERVED_UB_SIZE 40960  // 32 * 1024 + 8 * 1024
#endif

namespace AscendC {
enum class IndirectLoadSimtCase : uint8_t {
  kStaticPowerOfTwo = 0,
  kStaticInner = 1,
  kStructuredMagic = 2,
  kEmbedding = 3,
  kRecursive = 4,
  kStrided = 5,
};

template <IndirectLoadSimtCase Case, typename OffsetT, int32_t Rank, int32_t Axis, uint64_t InnerSpan = 0U,
          uint64_t OutputAxisSpan = 0U, uint64_t InputAxisStride = 0U, uint64_t InputAxisSpan = 0U,
          uint64_t InputStrideMask = 0U, uint64_t IndexStrideMask = 0U>
struct IndirectLoadSimtCaseTag {};

template <int32_t Rank>
struct IndirectLoadSimtShapeLayout {
  static constexpr int32_t kInputStrideBase = Rank;
  static constexpr int32_t kIndexStrideBase = kInputStrideBase + Rank;
  static constexpr int32_t kRecursiveShapeSize = kIndexStrideBase;
  static constexpr int32_t kStridedShapeSize = kIndexStrideBase + Rank;
};

struct IndirectLoadSimtStaticPowerOfTwoParams {};

template <typename OffsetT>
struct IndirectLoadSimtStaticInnerParams {
  OffsetT output_axis_span;
};

template <typename OffsetT>
struct IndirectLoadSimtStructuredMagicParams {
  OffsetT inner_span;
  OffsetT output_axis_span;
  OffsetT input_axis_stride;
  OffsetT input_axis_span;
};

template <typename OffsetT, int32_t Rank>
struct IndirectLoadSimtEmbeddingParams {
  OffsetT shape[IndirectLoadSimtShapeLayout<Rank>::kStridedShapeSize];
};

template <typename OffsetT, int32_t Rank>
struct IndirectLoadSimtRecursiveParams {
  OffsetT shape[IndirectLoadSimtShapeLayout<Rank>::kRecursiveShapeSize];
};

template <typename OffsetT, int32_t Rank>
struct IndirectLoadSimtStridedParams {
  OffsetT shape[IndirectLoadSimtShapeLayout<Rank>::kStridedShapeSize];
};

namespace Internal {
__aicore__ inline uint64_t IndirectLoadGetUintDivMagic(uint64_t dividend, uint64_t divisor) {
  uint64_t quotient = 0U;
  uint64_t remainder = dividend;
  constexpr uint64_t kHighestBit = 1ULL << 63U;
  for (uint32_t bit = 0U; bit < 64U; ++bit) {
    quotient <<= 1U;
    const bool borrow = (remainder & kHighestBit) != 0U;
    remainder <<= 1U;
    if (borrow) {
      remainder = ~uint64_t{0U} - divisor + 1U + remainder;
      quotient |= 1U;
    } else if (remainder >= divisor) {
      remainder -= divisor;
      quotient |= 1U;
    }
  }
  return quotient + 1U;
}

template <typename T>
__aicore__ inline void IndirectLoadGetUintDivMagicAndShift(T &magic, T &shift, T divisor) {
  static_assert(std::is_same<T, uint32_t>::value || std::is_same<T, uint64_t>::value,
                "IndirectLoad SIMT division only supports uint32_t and uint64_t.");
  uint32_t position = 0U;
  for (T value = divisor; value != 0U; value >>= 1U) {
    ++position;
  }
  shift = static_cast<T>((divisor & (divisor - 1U)) == 0U ? position - 1U : position);
  if constexpr (std::is_same<T, uint32_t>::value) {
    magic = static_cast<T>((1ULL << 32U) * ((1ULL << shift) - divisor) / divisor + 1U);
  } else {
    const uint64_t dividend = shift < 64U ? (1ULL << shift) - divisor : ~uint64_t{0U} - divisor + 1U;
    magic = static_cast<T>(IndirectLoadGetUintDivMagic(dividend, divisor));
  }
}

template <uint64_t Value>
inline __aicore__ constexpr uint32_t IndirectLoadLog2() {
  static_assert(Value > 0U && (Value & (Value - 1U)) == 0U, "IndirectLoad SIMT span must be a power of two.");
  uint32_t shift = 0U;
  uint64_t value = Value;
  while (value > 1U) {
    value >>= 1U;
    ++shift;
  }
  return shift;
}

template <typename OffsetT>
struct IndirectLoadSimtAddress {
  OffsetT index_offset;
  OffsetT input_base;
};

template <typename OffsetT, uint64_t InnerSpan, uint64_t InputAxisSpan>
__simt_callee__ __aicore__ inline IndirectLoadSimtAddress<OffsetT> BuildInnerPowerOfTwoAddress(OffsetT output_index,
                                                                                               OffsetT outer) {
  const OffsetT inner = output_index & static_cast<OffsetT>(InnerSpan - 1U);
  return {output_index, outer * static_cast<OffsetT>(InputAxisSpan) + inner};
}

template <int32_t Dim, typename OffsetT, typename AddressPolicy>
struct IndirectLoadSimtAddressDecoder {
  __simt_callee__ __aicore__ inline static void Call(OffsetT linear_index, const AddressPolicy &policy,
                                                     IndirectLoadSimtAddress<OffsetT> &address) {
    const OffsetT quotient = Simt::UintDiv(linear_index, policy.magic[Dim], policy.shift[Dim]);
    const OffsetT coordinate = linear_index - quotient * policy.shape[Dim];
    policy.template AddCoordinate<Dim>(coordinate, address);
    if constexpr (Dim > 0) {
      IndirectLoadSimtAddressDecoder<Dim - 1, OffsetT, AddressPolicy>::Call(quotient, policy, address);
    }
  }
};
}  // namespace Internal

template <typename OffsetT, uint64_t InnerSpan, uint64_t OutputAxisSpan, uint64_t InputAxisStride,
          uint64_t InputAxisSpan>
struct IndirectLoadSimtStaticPowerOfTwoPolicy {
  using OffsetType = OffsetT;
  static constexpr bool kStructured = true;
  static constexpr bool kUsesInputAxis = true;
  static constexpr bool kEmbedding = false;

  __simt_callee__ __aicore__ inline Internal::IndirectLoadSimtAddress<OffsetT> GetAddress(OffsetT output_index) const {
    constexpr uint32_t output_axis_shift = Internal::IndirectLoadLog2<OutputAxisSpan>();
    const OffsetT outer = output_index >> output_axis_shift;
    return Internal::BuildInnerPowerOfTwoAddress<OffsetT, InnerSpan, InputAxisSpan>(output_index, outer);
  }

  static constexpr OffsetT input_axis_stride = static_cast<OffsetT>(InputAxisStride);
};

template <typename OffsetT, uint64_t InnerSpan, uint64_t InputAxisStride, uint64_t InputAxisSpan>
struct IndirectLoadSimtStaticInnerPolicy {
  using OffsetType = OffsetT;
  static constexpr bool kStructured = true;
  static constexpr bool kUsesInputAxis = true;
  static constexpr bool kEmbedding = false;

  __aicore__ explicit IndirectLoadSimtStaticInnerPolicy(OffsetT output_axis_span) {
    Internal::IndirectLoadGetUintDivMagicAndShift(output_axis_magic, output_axis_shift, output_axis_span);
  }

  __simt_callee__ __aicore__ inline Internal::IndirectLoadSimtAddress<OffsetT> GetAddress(OffsetT output_index) const {
    const OffsetT outer = Simt::UintDiv(output_index, output_axis_magic, output_axis_shift);
    return Internal::BuildInnerPowerOfTwoAddress<OffsetT, InnerSpan, InputAxisSpan>(output_index, outer);
  }

  static constexpr OffsetT input_axis_stride = static_cast<OffsetT>(InputAxisStride);
  OffsetT output_axis_magic{0U};
  OffsetT output_axis_shift{0U};
};

template <typename OffsetT>
struct IndirectLoadSimtStructuredMagicPolicy {
  using OffsetType = OffsetT;
  static constexpr bool kStructured = true;
  static constexpr bool kUsesInputAxis = true;
  static constexpr bool kEmbedding = false;

  __aicore__ explicit IndirectLoadSimtStructuredMagicPolicy(OffsetT inner_span_arg, OffsetT output_axis_span_arg,
                                                            OffsetT input_axis_stride_arg, OffsetT input_axis_span_arg)
      : inner_span(inner_span_arg), input_axis_stride(input_axis_stride_arg), input_axis_span(input_axis_span_arg) {
    Internal::IndirectLoadGetUintDivMagicAndShift(inner_magic, inner_shift, inner_span);
    Internal::IndirectLoadGetUintDivMagicAndShift(output_axis_magic, output_axis_shift, output_axis_span_arg);
  }

  __simt_callee__ __aicore__ inline Internal::IndirectLoadSimtAddress<OffsetT> GetAddress(OffsetT output_index) const {
    const OffsetT outer = Simt::UintDiv(output_index, output_axis_magic, output_axis_shift);
    const OffsetT inner_quotient = Simt::UintDiv(output_index, inner_magic, inner_shift);
    const OffsetT inner = output_index - inner_quotient * inner_span;
    return {output_index, outer * input_axis_span + inner};
  }

  OffsetT inner_span;
  OffsetT input_axis_stride;
  OffsetT input_axis_span;
  OffsetT inner_magic{0U};
  OffsetT inner_shift{0U};
  OffsetT output_axis_magic{0U};
  OffsetT output_axis_shift{0U};
};

// Embedding-like views use a zero-stride index payload.  Keep the canonical
// rank-2 path compact and use the same stride-aware decoder for higher ranks.
template <typename OffsetT, int32_t Rank = 2, int32_t Axis = 0, uint64_t InputStrideMask = 0U,
          uint64_t IndexStrideMask = 0U>
struct IndirectLoadSimtEmbeddingPolicy {
  using OffsetType = OffsetT;
  using ShapeLayout = IndirectLoadSimtShapeLayout<Rank>;
  static constexpr bool kStructured = true;
  static constexpr bool kUsesInputAxis = true;
  static constexpr bool kEmbedding = Rank == 2 && Axis == 0 && InputStrideMask == 0U && IndexStrideMask == 0U;

  template <typename... ShapeArgs>
  __aicore__ explicit IndirectLoadSimtEmbeddingPolicy(ShapeArgs... shape_args)
      : shape{static_cast<OffsetT>(shape_args)...} {
    static_assert(Rank > 0 && Axis >= 0 && Axis < Rank, "IndirectLoad SIMT embedding rank or axis is invalid.");
    static_assert(sizeof...(ShapeArgs) == static_cast<size_t>(ShapeLayout::kStridedShapeSize),
                  "IndirectLoad SIMT embedding shape is invalid.");
    if constexpr (Rank == 2 && Axis == 0 && InputStrideMask == 0U && IndexStrideMask == 0U) {
      Internal::IndirectLoadGetUintDivMagicAndShift(magic[1], shift[1], shape[1]);
    } else {
      for (int32_t dim = 0; dim < Rank; ++dim) {
        Internal::IndirectLoadGetUintDivMagicAndShift(magic[dim], shift[dim], shape[dim]);
      }
    }
    input_axis_stride = shape[ShapeLayout::kInputStrideBase + Axis];
  }

  __aicore__ explicit IndirectLoadSimtEmbeddingPolicy(const IndirectLoadSimtEmbeddingParams<OffsetT, Rank> &params)
      : shape{} {
    for (int32_t dim = 0; dim < ShapeLayout::kStridedShapeSize; ++dim) {
      shape[dim] = params.shape[dim];
    }
    if constexpr (Rank == 2 && Axis == 0 && InputStrideMask == 0U && IndexStrideMask == 0U) {
      Internal::IndirectLoadGetUintDivMagicAndShift(magic[1], shift[1], shape[1]);
    } else {
      for (int32_t dim = 0; dim < Rank; ++dim) {
        Internal::IndirectLoadGetUintDivMagicAndShift(magic[dim], shift[dim], shape[dim]);
      }
    }
    input_axis_stride = shape[ShapeLayout::kInputStrideBase + Axis];
  }

  __simt_callee__ __aicore__ inline Internal::IndirectLoadSimtAddress<OffsetT> GetAddress(OffsetT output_index) const {
    if constexpr (Rank == 2 && Axis == 0 && InputStrideMask == 0U && IndexStrideMask == 0U) {
      const OffsetT row = Simt::UintDiv(output_index, magic[1], shift[1]);
      const OffsetT inner = output_index - row * shape[1];
      return {row * shape[ShapeLayout::kIndexStrideBase], inner * shape[ShapeLayout::kInputStrideBase + 1]};
    } else {
      Internal::IndirectLoadSimtAddress<OffsetT> address{0U, 0U};
      Internal::IndirectLoadSimtAddressDecoder<Rank - 1, OffsetT, IndirectLoadSimtEmbeddingPolicy>::Call(
          output_index, *this, address);
      return address;
    }
  }

  template <int32_t Dim>
  __simt_callee__ __aicore__ inline void AddCoordinate(OffsetT coordinate,
                                                       Internal::IndirectLoadSimtAddress<OffsetT> &address) const {
    if constexpr ((IndexStrideMask & (1ULL << Dim)) != 0U) {
      address.index_offset += coordinate * shape[ShapeLayout::kIndexStrideBase + Dim];
    }
    if constexpr (Dim != Axis && (InputStrideMask & (1ULL << Dim)) != 0U) {
      address.input_base += coordinate * shape[ShapeLayout::kInputStrideBase + Dim];
    }
  }

  OffsetT shape[ShapeLayout::kStridedShapeSize];
  OffsetT magic[Rank];
  OffsetT shift[Rank];
  OffsetT input_axis_stride{0U};
};

template <typename OffsetT, int32_t Rank, int32_t Axis>
struct IndirectLoadSimtRecursivePolicy {
  using OffsetType = OffsetT;
  using ShapeLayout = IndirectLoadSimtShapeLayout<Rank>;
  static constexpr bool kStructured = false;
  static constexpr bool kUsesInputAxis = true;
  static constexpr bool kEmbedding = false;

  template <typename... ShapeArgs>
  __aicore__ explicit IndirectLoadSimtRecursivePolicy(ShapeArgs... shape_args)
      : shape{static_cast<OffsetT>(shape_args)...} {
    static_assert(Rank > 0 && Axis >= 0 && Axis < Rank, "IndirectLoad SIMT rank or axis is invalid.");
    static_assert(sizeof...(ShapeArgs) == static_cast<size_t>(ShapeLayout::kRecursiveShapeSize),
                  "IndirectLoad SIMT shape is invalid.");
    for (int32_t dim = 0; dim < Rank; ++dim) {
      Internal::IndirectLoadGetUintDivMagicAndShift(magic[dim], shift[dim], shape[dim]);
    }
    input_axis_stride = shape[ShapeLayout::kInputStrideBase + Axis];
  }

  __aicore__ explicit IndirectLoadSimtRecursivePolicy(const IndirectLoadSimtRecursiveParams<OffsetT, Rank> &params)
      : shape{} {
    for (int32_t dim = 0; dim < ShapeLayout::kRecursiveShapeSize; ++dim) {
      shape[dim] = params.shape[dim];
    }
    for (int32_t dim = 0; dim < Rank; ++dim) {
      Internal::IndirectLoadGetUintDivMagicAndShift(magic[dim], shift[dim], shape[dim]);
    }
    input_axis_stride = shape[ShapeLayout::kInputStrideBase + Axis];
  }

  __simt_callee__ __aicore__ inline Internal::IndirectLoadSimtAddress<OffsetT> GetAddress(OffsetT output_index) const {
    Internal::IndirectLoadSimtAddress<OffsetT> address{output_index, 0U};
    Internal::IndirectLoadSimtAddressDecoder<Rank - 1, OffsetT, IndirectLoadSimtRecursivePolicy>::Call(output_index,
                                                                                                       *this, address);
    return address;
  }

  template <int32_t Dim>
  __simt_callee__ __aicore__ inline void AddCoordinate(OffsetT coordinate,
                                                       Internal::IndirectLoadSimtAddress<OffsetT> &address) const {
    if constexpr (Dim != Axis) {
      address.input_base += coordinate * shape[ShapeLayout::kInputStrideBase + Dim];
    }
  }

  OffsetT shape[ShapeLayout::kRecursiveShapeSize];
  OffsetT magic[Rank];
  OffsetT shift[Rank];
  OffsetT input_axis_stride{0U};
};

template <typename OffsetT, int32_t Rank, int32_t Axis, uint64_t InputStrideMask, uint64_t IndexStrideMask>
struct IndirectLoadSimtStridedPolicy {
  using OffsetType = OffsetT;
  using ShapeLayout = IndirectLoadSimtShapeLayout<Rank>;
  static constexpr bool kStructured = false;
  static constexpr bool kUsesInputAxis = (InputStrideMask & (1ULL << Axis)) != 0U;
  static constexpr bool kEmbedding = false;

  template <typename... ShapeArgs>
  __aicore__ explicit IndirectLoadSimtStridedPolicy(ShapeArgs... shape_args)
      : shape{static_cast<OffsetT>(shape_args)...} {
    static_assert(Rank > 0 && Axis >= 0 && Axis < Rank, "IndirectLoad SIMT rank or axis is invalid.");
    static_assert(sizeof...(ShapeArgs) == static_cast<size_t>(ShapeLayout::kStridedShapeSize),
                  "IndirectLoad SIMT shape is invalid.");
    for (int32_t dim = 0; dim < Rank; ++dim) {
      Internal::IndirectLoadGetUintDivMagicAndShift(magic[dim], shift[dim], shape[dim]);
    }
    input_axis_stride = shape[ShapeLayout::kInputStrideBase + Axis];
  }

  __aicore__ explicit IndirectLoadSimtStridedPolicy(const IndirectLoadSimtStridedParams<OffsetT, Rank> &params)
      : shape{} {
    for (int32_t dim = 0; dim < ShapeLayout::kStridedShapeSize; ++dim) {
      shape[dim] = params.shape[dim];
    }
    for (int32_t dim = 0; dim < Rank; ++dim) {
      Internal::IndirectLoadGetUintDivMagicAndShift(magic[dim], shift[dim], shape[dim]);
    }
    input_axis_stride = shape[ShapeLayout::kInputStrideBase + Axis];
  }

  __simt_callee__ __aicore__ inline Internal::IndirectLoadSimtAddress<OffsetT> GetAddress(OffsetT output_index) const {
    Internal::IndirectLoadSimtAddress<OffsetT> address{0U, 0U};
    if constexpr (Rank == 3 && Axis == 1 && IndexStrideMask == 2U) {
      const OffsetT axis_and_inner = Simt::UintDiv(output_index, magic[2], shift[2]);
      const OffsetT inner = output_index - axis_and_inner * shape[2];
      const OffsetT outer = Simt::UintDiv(axis_and_inner, magic[1], shift[1]);
      const OffsetT axis = axis_and_inner - outer * shape[1];
      if constexpr ((InputStrideMask & 1U) != 0U) {
        address.input_base += outer * shape[ShapeLayout::kInputStrideBase];
      }
      if constexpr ((InputStrideMask & 4U) != 0U) {
        address.input_base += inner * shape[ShapeLayout::kInputStrideBase + 2];
      }
      address.index_offset = axis * shape[ShapeLayout::kIndexStrideBase + 1];
      return address;
    }
    if constexpr (Rank == 2 && Axis == 0 && IndexStrideMask == 1U) {
      const OffsetT row = Simt::UintDiv(output_index, magic[1], shift[1]);
      const OffsetT column = output_index - row * shape[1];
      address.index_offset = row * shape[ShapeLayout::kIndexStrideBase];
      if constexpr ((InputStrideMask & 2U) != 0U) {
        address.input_base += column * shape[ShapeLayout::kInputStrideBase + 1];
      }
      return address;
    }
    Internal::IndirectLoadSimtAddressDecoder<Rank - 1, OffsetT, IndirectLoadSimtStridedPolicy>::Call(output_index,
                                                                                                     *this, address);
    return address;
  }

  template <int32_t Dim>
  __simt_callee__ __aicore__ inline void AddCoordinate(OffsetT coordinate,
                                                       Internal::IndirectLoadSimtAddress<OffsetT> &address) const {
    if constexpr ((IndexStrideMask & (1ULL << Dim)) != 0U) {
      address.index_offset += coordinate * shape[ShapeLayout::kIndexStrideBase + Dim];
    }
    if constexpr (Dim != Axis && (InputStrideMask & (1ULL << Dim)) != 0U) {
      address.input_base += coordinate * shape[ShapeLayout::kInputStrideBase + Dim];
    }
  }

  OffsetT shape[ShapeLayout::kStridedShapeSize];
  OffsetT magic[Rank];
  OffsetT shift[Rank];
  OffsetT input_axis_stride{0U};
};

namespace Internal {
template <typename...>
using IndirectLoadVoidT = void;

template <typename FusedBody, typename = void>
struct IndirectLoadHasAicoreIndex {
  static constexpr bool kValue = false;
};

template <typename FusedBody>
struct IndirectLoadHasAicoreIndex<FusedBody, IndirectLoadVoidT<decltype(FusedBody::kSupportsAicoreIndex)>> {
  static constexpr bool kValue = FusedBody::kSupportsAicoreIndex;
};

template <typename FusedBody, typename = void>
struct IndirectLoadHasIdentityOutput {
  static constexpr bool kValue = false;
};

template <typename FusedBody>
struct IndirectLoadHasIdentityOutput<FusedBody, IndirectLoadVoidT<decltype(FusedBody::kSimtOutputIdentity)>> {
  static constexpr bool kValue = FusedBody::kSimtOutputIdentity;
};

template <typename CaseTag>
struct IndirectLoadSimtCaseSelector;

template <typename OffsetT, int32_t Rank, int32_t Axis, uint64_t InnerSpan, uint64_t OutputAxisSpan,
          uint64_t InputAxisStride, uint64_t InputAxisSpan, uint64_t InputStrideMask, uint64_t IndexStrideMask>
struct IndirectLoadSimtCaseSelector<
    IndirectLoadSimtCaseTag<IndirectLoadSimtCase::kStaticPowerOfTwo, OffsetT, Rank, Axis, InnerSpan, OutputAxisSpan,
                            InputAxisStride, InputAxisSpan, InputStrideMask, IndexStrideMask>> {
  using OffsetType = OffsetT;
  using Params = IndirectLoadSimtStaticPowerOfTwoParams;
  using Policy =
      IndirectLoadSimtStaticPowerOfTwoPolicy<OffsetT, InnerSpan, OutputAxisSpan, InputAxisStride, InputAxisSpan>;

  __aicore__ inline static Policy MakePolicy(const Params &) {
    return {};
  }
};

template <typename OffsetT, int32_t Rank, int32_t Axis, uint64_t InnerSpan, uint64_t OutputAxisSpan,
          uint64_t InputAxisStride, uint64_t InputAxisSpan, uint64_t InputStrideMask, uint64_t IndexStrideMask>
struct IndirectLoadSimtCaseSelector<
    IndirectLoadSimtCaseTag<IndirectLoadSimtCase::kStaticInner, OffsetT, Rank, Axis, InnerSpan, OutputAxisSpan,
                            InputAxisStride, InputAxisSpan, InputStrideMask, IndexStrideMask>> {
  using OffsetType = OffsetT;
  using Params = IndirectLoadSimtStaticInnerParams<OffsetT>;
  using Policy = IndirectLoadSimtStaticInnerPolicy<OffsetT, InnerSpan, InputAxisStride, InputAxisSpan>;

  __aicore__ inline static Policy MakePolicy(const Params &params) {
    return Policy{params.output_axis_span};
  }
};

template <typename OffsetT, int32_t Rank, int32_t Axis, uint64_t InnerSpan, uint64_t OutputAxisSpan,
          uint64_t InputAxisStride, uint64_t InputAxisSpan, uint64_t InputStrideMask, uint64_t IndexStrideMask>
struct IndirectLoadSimtCaseSelector<
    IndirectLoadSimtCaseTag<IndirectLoadSimtCase::kStructuredMagic, OffsetT, Rank, Axis, InnerSpan, OutputAxisSpan,
                            InputAxisStride, InputAxisSpan, InputStrideMask, IndexStrideMask>> {
  using OffsetType = OffsetT;
  using Params = IndirectLoadSimtStructuredMagicParams<OffsetT>;
  using Policy = IndirectLoadSimtStructuredMagicPolicy<OffsetT>;

  __aicore__ inline static Policy MakePolicy(const Params &params) {
    return Policy{params.inner_span, params.output_axis_span, params.input_axis_stride, params.input_axis_span};
  }
};

template <typename OffsetT, int32_t Rank, int32_t Axis, uint64_t InnerSpan, uint64_t OutputAxisSpan,
          uint64_t InputAxisStride, uint64_t InputAxisSpan, uint64_t InputStrideMask, uint64_t IndexStrideMask>
struct IndirectLoadSimtCaseSelector<
    IndirectLoadSimtCaseTag<IndirectLoadSimtCase::kEmbedding, OffsetT, Rank, Axis, InnerSpan, OutputAxisSpan,
                            InputAxisStride, InputAxisSpan, InputStrideMask, IndexStrideMask>> {
  using OffsetType = OffsetT;
  using Params = IndirectLoadSimtEmbeddingParams<OffsetT, Rank>;
  using Policy = IndirectLoadSimtEmbeddingPolicy<OffsetT, Rank, Axis, InputStrideMask, IndexStrideMask>;

  __aicore__ inline static Policy MakePolicy(const Params &params) {
    return Policy{params};
  }
};

template <typename OffsetT, int32_t Rank, int32_t Axis, uint64_t InnerSpan, uint64_t OutputAxisSpan,
          uint64_t InputAxisStride, uint64_t InputAxisSpan, uint64_t InputStrideMask, uint64_t IndexStrideMask>
struct IndirectLoadSimtCaseSelector<
    IndirectLoadSimtCaseTag<IndirectLoadSimtCase::kRecursive, OffsetT, Rank, Axis, InnerSpan, OutputAxisSpan,
                            InputAxisStride, InputAxisSpan, InputStrideMask, IndexStrideMask>> {
  using OffsetType = OffsetT;
  using Params = IndirectLoadSimtRecursiveParams<OffsetT, Rank>;
  using Policy = IndirectLoadSimtRecursivePolicy<OffsetT, Rank, Axis>;

  __aicore__ inline static Policy MakePolicy(const Params &params) {
    return Policy{params};
  }
};

template <typename OffsetT, int32_t Rank, int32_t Axis, uint64_t InnerSpan, uint64_t OutputAxisSpan,
          uint64_t InputAxisStride, uint64_t InputAxisSpan, uint64_t InputStrideMask, uint64_t IndexStrideMask>
struct IndirectLoadSimtCaseSelector<
    IndirectLoadSimtCaseTag<IndirectLoadSimtCase::kStrided, OffsetT, Rank, Axis, InnerSpan, OutputAxisSpan,
                            InputAxisStride, InputAxisSpan, InputStrideMask, IndexStrideMask>> {
  using OffsetType = OffsetT;
  using Params = IndirectLoadSimtStridedParams<OffsetT, Rank>;
  using Policy = IndirectLoadSimtStridedPolicy<OffsetT, Rank, Axis, InputStrideMask, IndexStrideMask>;

  __aicore__ inline static Policy MakePolicy(const Params &params) {
    return Policy{params};
  }
};

template <uint32_t ThreadNum, typename X, typename FusedBody, typename Context, typename AddressPolicy>
__simt_vf__ __aicore__ LAUNCH_BOUND(ThreadNum) inline void IndirectLoadSimtKernel(
    __gm__ X *x, typename FusedBody::OutputTargets targets, Context context, uint32_t actual_size,
    typename AddressPolicy::OffsetType output_offset, AddressPolicy address_policy) {
  using OffsetT = typename AddressPolicy::OffsetType;
  for (uint32_t i = threadIdx.x; i < actual_size; i += blockDim.x) {
    const OffsetT output_index = output_offset + static_cast<OffsetT>(i);
    const auto address = address_policy.GetAddress(output_index);
    const OffsetT indirect_index = static_cast<OffsetT>(FusedBody::Index(output_index, address.index_offset, context));
    OffsetT input_offset = address.input_base;
    if constexpr (AddressPolicy::kUsesInputAxis) {
      input_offset += indirect_index * address_policy.input_axis_stride;
    }
    const X value = x[input_offset];
    const typename FusedBody::OutputPack outputs =
        FusedBody::Outputs(value, output_index, address.index_offset, context);
    FusedBody::Store(targets, output_index, static_cast<OffsetT>(i), context, outputs);
  }
}

template <uint32_t ThreadNum, typename X, typename FusedBody, typename Context, typename AddressPolicy>
__aicore__ inline void LaunchIndirectLoadSimt(__gm__ X *x, typename FusedBody::OutputTargets targets, Context context,
                                              uint32_t actual_size, typename AddressPolicy::OffsetType output_offset,
                                              AddressPolicy address_policy) {
  Simt::VF_CALL<IndirectLoadSimtKernel<ThreadNum, X, FusedBody, Context, AddressPolicy>>(
      Simt::Dim3(ThreadNum), x, targets, context, actual_size, output_offset, address_policy);
}

// Embedding rows share one index across the payload dimension.  Load that
// index once per warp so boundary fragments do not fall back to one index load
// for every payload element when a flattened tile cuts through a row.
template <uint32_t ThreadNum, typename X, typename FusedBody, typename Context, typename AddressPolicy>
__simt_vf__ __aicore__ LAUNCH_BOUND(ThreadNum) inline void IndirectLoadSimtEmbeddingKernel(
    __gm__ X *x, typename FusedBody::OutputTargets targets, Context context, uint32_t actual_size,
    typename AddressPolicy::OffsetType output_offset, AddressPolicy address_policy) {
  using OffsetT = typename AddressPolicy::OffsetType;
  const OffsetT inner_size = address_policy.shape[1];
  const OffsetT index_stride = address_policy.shape[4];
  const OffsetT input_axis_stride = address_policy.input_axis_stride;
  const OffsetT payload_stride = address_policy.shape[3];
  const OffsetT output_end = output_offset + static_cast<OffsetT>(actual_size);
  const OffsetT first_row = output_offset / inner_size;
  const OffsetT last_row = (output_end - 1U) / inner_size;
  const uint32_t warp_id = static_cast<uint32_t>(threadIdx.x) / static_cast<uint32_t>(warpSize);
  const uint32_t lane_id = static_cast<uint32_t>(threadIdx.x) % static_cast<uint32_t>(warpSize);
  const uint32_t warp_count =
      (static_cast<uint32_t>(blockDim.x) + static_cast<uint32_t>(warpSize) - 1U) / static_cast<uint32_t>(warpSize);
  for (OffsetT row = first_row + static_cast<OffsetT>(warp_id); row <= last_row;
       row += static_cast<OffsetT>(warp_count)) {
    const OffsetT row_begin = row * inner_size;
    const OffsetT begin = output_offset > row_begin ? output_offset : row_begin;
    const OffsetT row_end = row_begin + inner_size;
    const OffsetT end = output_end < row_end ? output_end : row_end;
    const OffsetT index_offset = row * index_stride;
    OffsetT indirect_index = 0U;
    if (lane_id == 0U) {
      indirect_index = static_cast<OffsetT>(FusedBody::Index(row_begin, index_offset, context));
    }
    indirect_index = Simt::WarpShflSync(indirect_index, 0);
    const OffsetT input_row = indirect_index * input_axis_stride;
    OffsetT output_index = begin + static_cast<OffsetT>(lane_id);
    OffsetT input_offset = input_row + (output_index - row_begin) * payload_stride;
    for (; output_index < end; output_index += static_cast<OffsetT>(warpSize),
                               input_offset += static_cast<OffsetT>(warpSize) * payload_stride) {
      const X value = x[input_offset];
      const typename FusedBody::OutputPack outputs = FusedBody::Outputs(value, output_index, index_offset, context);
      FusedBody::Store(targets, output_index, output_index - output_offset, context, outputs);
    }
  }
}

// 根据选embedding行，选择择合适的线程数
template <typename X, typename FusedBody, typename Context, typename AddressPolicy>
__aicore__ inline void LaunchIndirectLoadSimtEmbeddingByRows(__gm__ X *x, typename FusedBody::OutputTargets targets,
                                                             Context context, uint32_t row_count,
                                                             typename AddressPolicy::OffsetType output_offset,
                                                             AddressPolicy address_policy) {
  const uint32_t actual_size = row_count * static_cast<uint32_t>(address_policy.shape[1]);
  if (row_count <= 1U) {
    Simt::VF_CALL<IndirectLoadSimtEmbeddingKernel<32U, X, FusedBody, Context, AddressPolicy>>(
        Simt::Dim3(32U), x, targets, context, actual_size, output_offset, address_policy);
  } else if (row_count <= 2U) {
    Simt::VF_CALL<IndirectLoadSimtEmbeddingKernel<64U, X, FusedBody, Context, AddressPolicy>>(
        Simt::Dim3(64U), x, targets, context, actual_size, output_offset, address_policy);
  } else if (row_count <= 4U) {
    Simt::VF_CALL<IndirectLoadSimtEmbeddingKernel<128U, X, FusedBody, Context, AddressPolicy>>(
        Simt::Dim3(128U), x, targets, context, actual_size, output_offset, address_policy);
  } else if (row_count <= 8U) {
    Simt::VF_CALL<IndirectLoadSimtEmbeddingKernel<256U, X, FusedBody, Context, AddressPolicy>>(
        Simt::Dim3(256U), x, targets, context, actual_size, output_offset, address_policy);
  } else if (row_count <= 16U) {
    Simt::VF_CALL<IndirectLoadSimtEmbeddingKernel<512U, X, FusedBody, Context, AddressPolicy>>(
        Simt::Dim3(512U), x, targets, context, actual_size, output_offset, address_policy);
  } else {
    Simt::VF_CALL<IndirectLoadSimtEmbeddingKernel<1024U, X, FusedBody, Context, AddressPolicy>>(
        Simt::Dim3(1024U), x, targets, context, actual_size, output_offset, address_policy);
  }
}

// 处理 embedding 输出中的“残片”——不是完整一行的数据
template <typename X, typename FusedBody, typename Context, typename AddressPolicy>
__aicore__ inline void LaunchIndirectLoadSimtEmbeddingFragment(__gm__ X *x, typename FusedBody::OutputTargets targets,
                                                               Context context, uint32_t actual_size,
                                                               typename AddressPolicy::OffsetType output_offset,
                                                               AddressPolicy address_policy) {
  Simt::VF_CALL<IndirectLoadSimtEmbeddingKernel<32U, X, FusedBody, Context, AddressPolicy>>(
      Simt::Dim3(32U), x, targets, context, actual_size, output_offset, address_policy);
}

template <typename AddressPolicy>
inline __aicore__ constexpr bool IndirectLoadUse2048Threads() {
  return sizeof(typename AddressPolicy::OffsetType) == sizeof(uint32_t);
}

template <typename X, typename FusedBody, typename Context, typename AddressPolicy>
__aicore__ inline bool TryIndirectLoadSimtEmbeddingMte(__gm__ X *x, typename FusedBody::OutputTargets targets,
                                                       Context context, uint32_t actual_size,
                                                       typename AddressPolicy::OffsetType output_offset,
                                                       AddressPolicy address_policy) {
  if constexpr (FusedBody::kGmOutputCount != 1U || FusedBody::kUbOutputCount != 0U ||
                !IndirectLoadHasAicoreIndex<FusedBody>::kValue || !IndirectLoadHasIdentityOutput<FusedBody>::kValue ||
                !AddressPolicy::kEmbedding || !std::is_same<X, typename FusedBody::PrimaryOutputType>::value ||
                (sizeof(X) != sizeof(uint16_t) && sizeof(X) != sizeof(uint32_t))) {
    return false;
  } else {
    using OffsetT = typename AddressPolicy::OffsetType;
    if (actual_size == 0U) {
      return true;
    }
    constexpr uint32_t kBufferBytes = 192U * 1024U - 256U;
    constexpr uint32_t kAlignmentBytes = 32U;
    const OffsetT inner_size = address_policy.shape[1];
    const OffsetT input_axis_stride = address_policy.input_axis_stride;
    const OffsetT payload_stride = address_policy.shape[3];
    const OffsetT index_stride = address_policy.shape[4];
    const uint64_t row_bytes = static_cast<uint64_t>(inner_size) * sizeof(X);
    if (inner_size <= static_cast<OffsetT>(warpSize) || payload_stride != 1U || input_axis_stride != inner_size ||
        row_bytes == 0U || row_bytes > kBufferBytes || row_bytes % kAlignmentBytes != 0U) {
      return false;
    }
    const OffsetT output_row_offset = output_offset % inner_size;
    const uint32_t prefix_elements =
        output_row_offset == 0U
            ? 0U
            : static_cast<uint32_t>(inner_size - output_row_offset < actual_size ? inner_size - output_row_offset
                                                                                 : actual_size);
    const uint32_t remaining_elements = actual_size - prefix_elements;
    const uint32_t full_row_elements = remaining_elements - remaining_elements % static_cast<uint32_t>(inner_size);
    const uint32_t row_count = full_row_elements / static_cast<uint32_t>(inner_size);
    const uint32_t batch_capacity = kBufferBytes / static_cast<uint32_t>(row_bytes);
    if (batch_capacity == 0U) {
      return false;
    }
    if (row_count == 0U) {
      LaunchIndirectLoadSimtEmbeddingFragment<X, FusedBody>(x, targets, context, actual_size, output_offset,
                                                            address_policy);
      return true;
    }
    if (prefix_elements != 0U) {
      LaunchIndirectLoadSimtEmbeddingFragment<X, FusedBody>(x, targets, context, prefix_elements, output_offset,
                                                            address_policy);
    }

    TQue<QuePosition::VECIN, 1> input_queue;
    if (!GetTPipePtr()->InitBuffer(input_queue, 1U, kBufferBytes)) {
      return false;
    }
    GlobalTensor<X> input_gm;
    input_gm.SetGlobalBuffer(x);
    GlobalTensor<X> output_gm;
    output_gm.SetGlobalBuffer((__gm__ X *)targets.output0);
    const OffsetT aligned_output_offset = output_offset + static_cast<OffsetT>(prefix_elements);
    const OffsetT first_row = aligned_output_offset / inner_size;
    const uint32_t mte_row_count = row_count;
    for (uint32_t row_offset = 0U; row_offset < mte_row_count; row_offset += batch_capacity) {
      const uint32_t remaining_rows = mte_row_count - row_offset;
      const uint32_t batch_rows = remaining_rows < batch_capacity ? remaining_rows : batch_capacity;
      LocalTensor<X> input_local = input_queue.template AllocTensor<X>();
      for (uint32_t local_row = 0U; local_row < batch_rows; ++local_row) {
        const OffsetT row = first_row + static_cast<OffsetT>(row_offset + local_row);
        const OffsetT indirect_index = static_cast<OffsetT>(FusedBody::AicoreIndex(row * index_stride, context));
        DataCopy(input_local[static_cast<OffsetT>(local_row) * inner_size],
                 input_gm[indirect_index * input_axis_stride], static_cast<uint32_t>(inner_size));
      }
      input_queue.EnQue(input_local);
      input_local = input_queue.template DeQue<X>();
      const int32_t input_ready_event_id = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
      SetFlag<HardEvent::MTE2_MTE3>(input_ready_event_id);
      WaitFlag<HardEvent::MTE2_MTE3>(input_ready_event_id);
      DataCopy(output_gm[aligned_output_offset + static_cast<OffsetT>(row_offset) * inner_size], input_local,
               batch_rows * static_cast<uint32_t>(inner_size));
      if (row_offset + batch_rows < mte_row_count) {
        const int32_t output_done_event_id = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(output_done_event_id);
        WaitFlag<HardEvent::MTE3_MTE2>(output_done_event_id);
      }
      input_queue.FreeTensor(input_local);
    }

    const uint32_t suffix_elements = remaining_elements - full_row_elements;
    if (suffix_elements != 0U) {
      const OffsetT suffix_offset = aligned_output_offset + static_cast<OffsetT>(full_row_elements);
      LaunchIndirectLoadSimtEmbeddingFragment<X, FusedBody>(x, targets, context, suffix_elements, suffix_offset,
                                                            address_policy);
    }
    return true;
  }
}

template <typename X, typename FusedBody, typename Context, typename AddressPolicy>
__aicore__ inline void DispatchIndirectLoadSimt(__gm__ X *x, typename FusedBody::OutputTargets targets, Context context,
                                                uint32_t actual_size, typename AddressPolicy::OffsetType output_offset,
                                                AddressPolicy address_policy) {
  if constexpr (AddressPolicy::kEmbedding) {
    if (TryIndirectLoadSimtEmbeddingMte<X, FusedBody>(x, targets, context, actual_size, output_offset,
                                                      address_policy)) {
      return;
    }
  }
  if (actual_size <= 128U) {
    LaunchIndirectLoadSimt<128U, X, FusedBody>(x, targets, context, actual_size, output_offset, address_policy);
    return;
  }
  if (actual_size <= 256U) {
    LaunchIndirectLoadSimt<256U, X, FusedBody>(x, targets, context, actual_size, output_offset, address_policy);
    return;
  }
  if (actual_size <= 512U) {
    LaunchIndirectLoadSimt<512U, X, FusedBody>(x, targets, context, actual_size, output_offset, address_policy);
    return;
  }
  if (actual_size <= 1024U) {
    LaunchIndirectLoadSimt<1024U, X, FusedBody>(x, targets, context, actual_size, output_offset, address_policy);
    return;
  }
  if constexpr (IndirectLoadUse2048Threads<AddressPolicy>()) {
    LaunchIndirectLoadSimt<2048U, X, FusedBody>(x, targets, context, actual_size, output_offset, address_policy);
  } else {
    LaunchIndirectLoadSimt<1024U, X, FusedBody>(x, targets, context, actual_size, output_offset, address_policy);
  }
}
}  // namespace Internal

template <typename CaseTag>
using IndirectLoadSimtParams = typename Internal::IndirectLoadSimtCaseSelector<CaseTag>::Params;

template <typename X, typename FusedBody, typename CaseTag, typename Context>
__aicore__ inline void IndirectLoadSimt(
    __gm__ X *x, typename FusedBody::OutputTargets targets, Context context, uint32_t actual_size,
    typename Internal::IndirectLoadSimtCaseSelector<CaseTag>::OffsetType output_offset,
    const IndirectLoadSimtParams<CaseTag> &params) {
  static_assert(FusedBody::kGmOutputCount + FusedBody::kUbOutputCount > 0U,
                "IndirectLoad SIMT requires at least one output target.");
  static_assert(FusedBody::kUbOutputCount <= 1U, "IndirectLoad SIMT supports at most one UB output target.");
  using Selector = Internal::IndirectLoadSimtCaseSelector<CaseTag>;
  if (actual_size != 0U) {
    const typename Selector::Policy address_policy = Selector::MakePolicy(params);
    Internal::DispatchIndirectLoadSimt<X, FusedBody>(x, targets, context, actual_size, output_offset, address_policy);
  }
  if constexpr (FusedBody::kGmOutputCount > 0U) {
    const int32_t event_id = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(event_id);
    WaitFlag<HardEvent::V_MTE3>(event_id);
  } else {
    PipeBarrier<PIPE_V>();
  }
}

template <typename X, typename FusedBody, typename CaseTag, typename Context>
__aicore__ inline void IndirectLoadSimtMulti(
    __gm__ X *x, typename FusedBody::OutputTargets targets, Context context, uint32_t actual_size,
    typename Internal::IndirectLoadSimtCaseSelector<CaseTag>::OffsetType output_offset,
    const IndirectLoadSimtParams<CaseTag> &params) {
  IndirectLoadSimt<X, FusedBody, CaseTag>(x, targets, context, actual_size, output_offset, params);
}

}  // namespace AscendC

#endif  // AUTOFUSE_V35_ASCENDC_API_REGBASE_INDIRECT_LOAD_SIMT_H_
