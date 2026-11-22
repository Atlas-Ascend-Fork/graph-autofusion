/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCIR_TARGET_ASC_UTILS_H
#define ASCIR_TARGET_ASC_UTILS_H

#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {

/// Convenience functions to produce interleaved output with functions returning a LogicalResult.
/// This is different than those in STLExtras as functions used on each element doesn't return a string.
template <typename ForwardIterator, typename UnaryFunctor, typename NullaryFunctor>
inline LogicalResult interleaveWithError(
    ForwardIterator beginIter, ForwardIterator endIter, UnaryFunctor eachFn, NullaryFunctor betweenFn)
{
    if (beginIter == endIter)
        return success();
    if (failed(eachFn(*beginIter)))
        return failure();
    ++beginIter;
    for (; beginIter != endIter; ++beginIter) {
        betweenFn();
        if (failed(eachFn(*beginIter)))
            return failure();
    }
    return success();
}

template <typename Container, typename UnaryFunctor, typename NullaryFunctor>
inline LogicalResult interleaveWithError(const Container& container, UnaryFunctor eachFn, NullaryFunctor betweenFn)
{
    return interleaveWithError(std::cbegin(container), std::cend(container), eachFn, betweenFn);
}

template <typename Container, typename UnaryFunctor>
inline LogicalResult interleaveCommaWithError(const Container& c, raw_ostream& os, UnaryFunctor eachFn)
{
    return interleaveWithError(std::cbegin(c), std::cend(c), eachFn, [&]() { os << ", "; });
}

} // namespace mlir

#endif // ASCIR_TARGET_ASC_UTILS_H
