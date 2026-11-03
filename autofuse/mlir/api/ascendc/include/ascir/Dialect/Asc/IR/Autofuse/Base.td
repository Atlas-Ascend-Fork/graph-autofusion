/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASC_AUTOFUSE_BASE_TD
#define ASC_AUTOFUSE_BASE_TD

// Spells out the `ApiExtensionOpInterface` methods for an operation whose API
// is declared in Autofuse API extension headers rather than in the stock
// AscendC library. `headers` are the registry keys the kernel-source layer looks
// the header texts up by. Use it alongside the interface in the trait list:
//
//   def AscendC_AbsExtendOp
//       : UnaryOp<"abs_extend", "AbsExtend", [ApiExtensionOpInterface]> {
//     let extraClassDeclaration =
//         extraClassDeclarationBase # ApiExtensionHeader<["abs.h"]>.decl;
//   }
//
// **`headers` is ordered, and the order is a dependency order.** The texts are
// inlined in the order the include block lists them, and one wrapper header may
// call into another: `where.h` calls `Duplicate`, which only `duplicate.h`
// defines, so it must be written `["duplicate.h", "where.h"]`. Getting it
// backwards does not fail here -- the include is emitted, the key resolves, the
// text is inlined -- it fails in the device compiler, reporting a line inside
// the inlined text.
//
// The kernel-source layer writes the include block in dictionary order, so today
// a list must already be sorted; every wrapper pair Autofuse ships happens to be
// (`duplicate.h` < `where.h`, `compare.h` < `compare_v2.h`, `cast.h` < `sign.h`).
// Declaring one that is not sorted is a diagnostic there, not a silent reorder --
// the day a wrapper genuinely needs the reverse, that layer is what changes.
//
// `preludes` name headers that must be inlined *before all* extension headers,
// which is a stronger constraint than the pairwise order above -- that is why
// they are a separate list rather than a first entry in `headers`. Some Autofuse
// wrappers are not self-contained translation units: they spell AscendC names
// without the `AscendC::` qualifier, or reach for `KernelUtils` / `MAX_REPEAT_TIME`
// which only `utils.h` declares. Legacy codegen papered over this by writing the
// same prelude into every kernel; here the operation says what it needs, because
// the kernel-source layer collects names and must not know which family a name
// belongs to. Leave it empty for a wrapper that stands on its own.
class ApiExtensionHeader<list<string> headers, list<string> preludes = []> {
  code decl = [{
    static ::llvm::ArrayRef<::llvm::StringRef> getExtensionHeaders() {
      static const ::llvm::StringRef kHeaders[] = {"}] # !interleave(headers, "\", \"") # [{"};
      return kHeaders;
    }
    static ::llvm::ArrayRef<::llvm::StringRef> getExtensionPreludeHeaders() {
      }] # !if(!empty(preludes), [{return {};}], [{static const ::llvm::StringRef kPreludes[] = {"}]
          # !interleave(preludes, "\", \"") # [{"};
      return kPreludes;}]) # [{
    }
  }];
}

// `UnaryOp` / `VecScalarOp` hand `OpWithSrcInterface::getSrcTensors` down through
// their own `extraClassDeclaration`, so an operation that replaces that field --
// every API extension wrapper does, to spell out its header -- loses the method
// and fails to build. Paste this back in alongside the replacement.
class SrcTensorsFromSrc {
  code decl = [{
    SmallVector<Value> getSrcTensors() { return {getSrc()}; }
  }];
}

#endif // ASC_AUTOFUSE_BASE_TD
