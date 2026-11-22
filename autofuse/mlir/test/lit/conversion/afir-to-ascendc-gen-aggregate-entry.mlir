// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This fixture documents the aggregate Entry contract. The multi-root API is
// exercised by the in-process codegen tests; af-opt checks the resulting Entry module.
// RUN: af-opt --verify-diagnostics %s

module @AutofuseEntry attributes {
  afir.module_kind = "entry",
  ascendc.tiling_schema = !emitasc.py_struct<"AutofuseTilingData", [
      !emitasc.py_struct<"Plan0TilingData", [ui32], ["tiling_key"]>],
  ["plan0_tiling_data"]>
} {
}
