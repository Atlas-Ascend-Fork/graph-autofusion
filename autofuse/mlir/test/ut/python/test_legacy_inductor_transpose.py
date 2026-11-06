#!/usr/bin/env python3
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import sys

# Inductor binding cases copied from both legacy UT/ST test_python_ascir.py.
# Those two files contain the same three cases. RuntimeError is not swallowed.
import argparse
import importlib
import os
from pathlib import Path
import shutil
import tempfile
import unittest

ascir = None
Autofuser = None
AutofuserOptions = None


class TestAutofuseLoadTransposeStore(unittest.TestCase):
    def setUp(self):
        self.work = tempfile.TemporaryDirectory(prefix="mlir-inductor-transpose-")
        self.previous = os.getcwd()
        os.chdir(self.work.name)
        ascir.utils.set_platform("2201", 1, 245760)

    def tearDown(self):
        os.chdir(self.previous)
        self.work.cleanup()

    @staticmethod
    def construct_invalid_graph():
        graph = ascir.HintGraph("LoadTransposeStore")
        s0 = 100
        s1 = 200
        s2 = 300
        z0 = graph.create_axis("z0", s0)
        assert z0.size.expression == "100"
        z1 = graph.create_axis("z1", s1)
        z2 = graph.create_axis("z2", s2)

        arg3_1 = ascir.ops.Data("arg3_1", graph)
        arg3_1.attr.ir_attr.index = 0
        arg3_1.attr.sched.axis = [z0, z1, z2]
        arg3_1.y.dtype = ascir.dtypes.float16
        arg3_1.y.axis = [z0, z1, z2]
        arg3_1.y.size = [s0, s1, s2]
        arg3_1.y.strides = [s1 * s2, s2, ascir.SizeExpr(1)]
        return graph

    @staticmethod
    def construct_graph():
        graph = ascir.HintGraph("LoadTransposeStore")
        s0 = 100
        s1 = 200
        s2 = 300
        z0 = graph.create_axis("z0", s0)
        assert z0.size.expression == "100"
        z1 = graph.create_axis("z1", s1)
        z2 = graph.create_axis("z2", s2)
        buf_z0 = graph.create_axis("buf_z0", s0)
        buf_z1 = graph.create_axis("buf_z1", s1)
        buf_z2 = graph.create_axis("buf_z2", s2)

        arg3_1 = ascir.ops.Data("arg3_1", graph)
        arg3_1.attr.ir_attr.index = 0
        arg3_1.attr.sched.axis = [z0, z1, z2]
        arg3_1.y.dtype = ascir.dtypes.float16
        arg3_1.y.axis = [z0, z1, z2]
        arg3_1.y.size = [s0, s1, s2]
        arg3_1.y.strides = [s1 * s2, s2, ascir.SizeExpr(1)]

        load = ascir.ops.Load("load")
        try:
            load.attr.ir_attr.offset = "3"
        except Exception as e:
            assert e.args[0] == "Only support type of SizeExpr or long"
        offset_of_0 = ascir.SizeExpr(0)
        load.attr.ir_attr.offset = offset_of_0
        assert load.attr.ir_attr.offset.expression == "0"
        load.x = arg3_1
        load.attr.sched.axis = [z0, z1, z2]
        load.y.dtype = ascir.dtypes.float16
        load.y.axis = [z0, z1, z2]
        load.y.size = [s0, s1, s2]
        load.y.strides = [s1 * s2, s2, ascir.SizeExpr(1)]

        transpose0_op = ascir.ops.Transpose("Transpose0")
        transpose0_op.x = load
        transpose0_op.attr.sched.axis = [z0, z1, z2]
        transpose0_op.y.dtype = ascir.dtypes.float16
        transpose0_op.y.axis = [z1, z0, z2]
        transpose0_op.y.size = [s1, s0, s2]
        transpose0_op.y.strides = [s0 * s2, s2, ascir.SizeExpr(1)]

        store = ascir.ops.Store("store")
        try:
            store.attr.ir_attr.offset = "4"
        except Exception as e:
            assert e.args[0] == "Only support type of SizeExpr or long"
        store.attr.ir_attr.offset = offset_of_0 + 1
        assert store.attr.ir_attr.offset.expression == "1"
        store.x = transpose0_op
        store.attr.sched.axis = [z0, z1, z2]
        store.y.dtype = ascir.dtypes.float16
        store.y.axis = [z1, z0, z2]
        store.y.size = [s1, s0, s2]
        store.y.strides = [s0 * s2, s2, ascir.SizeExpr(1)]

        buf1 = ascir.ops.Output("buf1", graph)
        buf1.attr.ir_attr.index = 0
        assert buf1.attr.ir_attr.index == 0
        buf1.x = store
        buf1.attr.sched.axis = [z0, z1, z2]
        buf1.y.dtype = ascir.dtypes.float16
        buf1.y.axis = [z1, z0, z2]
        buf1.y.size = [s1, s0, s2]
        buf1.y.strides = [s0 * s2, s2, ascir.SizeExpr(1)]
        graph.set_axis_map({z0: [buf_z0], z1: [buf_z1], z2: [buf_z2]})
        return graph

    def test_construct_graph(self):
        self.assertTrue(ascir.utils.debug_str(self.construct_graph()))

    def test_autofuse_backend(self):
        fuser = Autofuser(AutofuserOptions())
        scheduled = fuser.schedule(self.construct_graph())
        tiling_def, host_tiling, op_kernel = fuser.codegen(scheduled)
        self.assertTrue(tiling_def)
        self.assertTrue(host_tiling)
        self.assertIn("_subgraph0_plan", op_kernel)
        self.assertIn("GetTilingDataRepr", host_tiling)
        self.assertNotIn('#include ""', op_kernel)

    def test_autofuse_backend_faild_dump_graph(self):
        fuser = Autofuser(AutofuserOptions())
        with self.assertRaisesRegex(RuntimeError, r"^Optimize fail$"):
            fuser.schedule(self.construct_invalid_graph())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--python-source", type=Path, required=True)
    args = parser.parse_args()
    if not args.extension.is_file():
        parser.error("build the matching pyautofuse target first")
    os.environ["AUTOFUSE_DFX_FLAGS"] = "--mlir_codegen_config=mlir"
    with tempfile.TemporaryDirectory(prefix="mlir-transpose-overlay-") as overlay:
        package = Path(overlay) / "autofuse"
        package.mkdir()
        for path in args.python_source.glob("*.py"):
            shutil.copy2(path, package / path.name)
        (package / "pyautofuse.so").symlink_to(args.extension.resolve())
        sys.path.insert(0, overlay)
        binding = importlib.import_module("autofuse.pyautofuse")
        if Path(binding.__file__).resolve() != args.extension.resolve():
            raise RuntimeError("pyautofuse did not resolve to the requested build")
        ascir, Autofuser, AutofuserOptions = (
            binding.ascir,
            binding.Autofuser,
            binding.AutofuserOptions,
        )
        unittest.main(argv=[sys.argv[0]], verbosity=2)
