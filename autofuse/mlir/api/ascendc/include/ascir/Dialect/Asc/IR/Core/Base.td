/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASC_CORE_BASE_TD
#define ASC_CORE_BASE_TD

include "Dialect.td"
include "Core/Interfaces.td"

class AscendC_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<AscendC_Dialect, name, [MemRefElementTypeInterface] # traits> {
  let mnemonic = typeMnemonic;
}

class AscendC_BaseQueueType<string name, string typeMnemonic, list<Trait> traits = []>
    : AscendC_Type<name, typeMnemonic, [AscendC_BaseQueueTypeInterface] # traits> {
}

class AscendC_BaseLoadDataParamsType<string name, string typeMnemonic, list<Trait> traits = []>
    : AscendC_Type<name, typeMnemonic, [AscendC_BaseLoadDataParamsTypeInterface] # traits> {
}

class AscendC_BaseTensorType<string name, string typeMnemonic, list<Trait> traits = []>
    : AscendC_Type<name, typeMnemonic, [AscendC_BaseTensorTypeInterface,
        DeclareTypeInterfaceMethods<ShapedTypeInterface>] # traits> {
  let parameters = (ins ArrayRefParameter<"int64_t">:$shape,
                        "Type":$elementType);

  let builders = [
    TypeBuilderWithInferredContext<(ins
      "ArrayRef<int64_t>":$shape, "Type":$elementType)>,
    TypeBuilderWithInferredContext<(ins "Type":$elementType)>,
    TypeBuilderWithInferredContext<(ins
      "::mlir::ascendc::BaseTensorType":$baseType)>,
  ];

  let hasCustomAssemblyFormat = 1;
  let skipDefaultBuilders = 1;
}

#endif // ASC_CORE_BASE_TD
