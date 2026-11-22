/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_MLIR_TARGET_ASCEND_CAPABILITY_IMPLEMENTATIONREGISTRY_H
#define AUTOFUSE_MLIR_TARGET_ASCEND_CAPABILITY_IMPLEMENTATIONREGISTRY_H

#include "target/Ascend/Capability/CapabilityProvider.h"

#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "llvm/ADT/StringRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
class MLIRContext;
class RewritePatternSet;
class TypeConverter;
}  // namespace mlir

namespace mlir::ascend::instruction {

struct PatternBuildContext {
  TypeConverter &typeConverter;
  const TargetProfile &target;
};

class CapabilityQuery;

class ImplementationRegistry {
 public:
  explicit ImplementationRegistry(MLIRContext &context);

  template <typename Mapping>
  using PatternRegistration = std::function<void(RewritePatternSet &, const PatternBuildContext &,
                                                 const ImplementationKey &, std::shared_ptr<const Mapping>)>;

  template <typename SourceOp, typename Mapping, typename Provider>
  LogicalResult registerImplementation(ImplementationKind kind, ImplementationId id, Mapping mapping,
                                       PatternRegistration<Mapping> addPattern) {
    static_assert(std::is_base_of_v<CapabilityProvider, Provider>, "Provider must implement CapabilityProvider");
    if (frozen_ || id.value.empty() || !addPattern) {
      return failure();
    }

    llvm::StringRef sourceOpName = SourceOp::getOperationName();
    if (sourceOpName.empty()) {
      return failure();
    }

    ImplementationKey key{OperationName(sourceOpName, &context_), kind, id};
    std::string ownedSourceOpName = sourceOpName.str();
    auto typedMapping = std::make_shared<const Mapping>(std::move(mapping));
    auto provider = std::make_unique<Provider>(typedMapping);
    auto erasedPattern = [registration = std::move(addPattern), typedMapping, kind, id](
                             RewritePatternSet &patterns, const PatternBuildContext &context) {
      ImplementationKey callbackKey{OperationName(SourceOp::getOperationName(), patterns.getContext()), kind, id};
      registration(patterns, context, callbackKey, typedMapping);
    };
    RegistrationEntry entry{std::move(key), typedMapping, std::move(provider), std::move(erasedPattern),
                            std::move(ownedSourceOpName)};
    entries_.push_back(std::move(entry));
    return success();
  }

  LogicalResult freeze();
  bool isFrozen() const;
  bool contains(const ImplementationKey &key) const;
  bool containsSourceOp(OperationName sourceOp) const;

 private:
  struct RegistrationEntry {
    ImplementationKey key;
    std::shared_ptr<const void> mappingOwner;
    std::unique_ptr<const CapabilityProvider> provider;
    std::function<void(RewritePatternSet &, const PatternBuildContext &)> addPattern;
    std::string sourceOpName;
  };

  friend class CapabilityQuery;
  friend void populateEnabledPatterns(RewritePatternSet &patterns, const ImplementationRegistry &registry,
                                      const PatternBuildContext &context);

  MLIRContext &context_;
  std::vector<RegistrationEntry> entries_;
  std::map<std::string, std::pair<size_t, size_t>> entryRanges_;
  bool frozen_ = false;
};

}  // namespace mlir::ascend::instruction

#endif  // AUTOFUSE_MLIR_TARGET_ASCEND_CAPABILITY_IMPLEMENTATIONREGISTRY_H
