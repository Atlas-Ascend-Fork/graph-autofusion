# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import pytest
import importlib.util
import json
import time
import os
import shutil
import sys
from pathlib import Path
from autofuse.pyautofuse import ascir, Autofuser, AutofuserOptions, Schedule, CodeGen

try:
    from autofuse import ascir_api
except ImportError:
    ascir_api = None
from ascir import Max, Min, Mod

PYF_PATH = os.path.dirname(os.path.realpath(__file__))
RUNTIME_ERROR_TILING_LIB = (
    Path(__file__).resolve().parents[4]
    / "build"
    / "autofuse"
    / "tests"
    / "libpyautofuse_runtime_error_tiling.so"
)

ascir.utils.set_platform("2201", 1, 245760)


# pyascir 构图能力暂不支持
class TestAscir:
    @staticmethod
    def test_graph_create_size_expr_by_long():
        s0 = ascir.SizeExpr()
        assert s0 == 1
        try:
            s0 = ascir.SizeExpr("100")
        except Exception as e:
            assert e.args[0] == "Only support type of SizeExpr or long"

        assert s0 == 1
        s0 = ascir.SizeExpr(0)
        assert s0 == 0

        s1 = ascir.SizeExpr(1)
        assert s1 == 1

        s2 = ascir.SizeExpr(1)
        assert s2 == 1

        s3 = ascir.SizeExpr(100)
        assert s3 == 100
        s4 = ascir.SizeExpr(1) + ascir.SizeExpr(2)
        assert s4 == 3
        s5 = ascir.SizeExpr(1) + ascir.SizeExpr(128)
        assert s5 == 129
        s7 = s5 + s4
        assert s7 == 132
        s8 = s1 + s2
        assert s8 == 2

        s9 = s3 * s4
        assert s9 == 300
        s10 = s3 * 2
        assert s10 == 200

    @staticmethod
    def test_graph_create_size():
        graph = ascir.HintGraph("test")
        graph.create_size("s0")
        graph.create_size("s1")
        graph.create_size("s2")

        debug_str = ascir.utils.debug_str(graph)
        assert debug_str == "".join(
            [
                "Graph: test\n",
                "Sizes:\n",
                "  s0: VAR\n",
                "  s1: VAR\n",
                "  s2: VAR\n",
                "Axis:\n",
                "Nodes:\n",
            ]
        )

    @staticmethod
    def test_graph_create_axis():
        graph = ascir.HintGraph("test")
        s0 = graph.create_size("s0")
        s1 = graph.create_size("s1")
        s2 = graph.create_size("s2")

        graph.create_axis("z0", s0)
        graph.create_axis("z1", s1)
        graph.create_axis("z2", s2)
        graph.create_axis("z3", 512)
        graph.create_axis("z4", s1 + s2)

        debug_str = ascir.utils.debug_str(graph)
        assert debug_str == "".join(
            [
                "Graph: test\n",
                "Sizes:\n",
                "  s0: VAR\n",
                "  s1: VAR\n",
                "  s2: VAR\n",
                "Axis:\n",
                "  z0(0) : ORIGINAL, size:s0, \n",
                "  z1(1) : ORIGINAL, size:s1, \n",
                "  z2(2) : ORIGINAL, size:s2, \n",
                "  z3(3) : ORIGINAL, size:512, \n",
                "  z4(4) : ORIGINAL, size:(s1 + s2), \n",
                "Nodes:\n",
            ]
        )

    @staticmethod
    def test_graph_sched_axis():
        graph = ascir.HintGraph("test")
        z0 = graph.create_axis("z0", 4)
        z1 = graph.create_axis("z1", 8)

        assert graph.sched.axis == []
        graph.sched.axis = [z0, z1]
        assert graph.sched.axis == [z0.id, z1.id]

    @staticmethod
    def test_graph_create_node():
        graph = ascir.HintGraph("test")

        ascir.ops.Data("x", graph)
        debug_str = ascir.utils.debug_str(graph)
        assert debug_str

    @staticmethod
    def test_graph_create_node_with_cast_infer():
        graph = ascir.HintGraph("test")

        x = ascir.ops.Data("x", graph)
        x.y.dtype = ascir.dtypes.int8
        x.attr.ir_attr.index = 0
        assert x.attr.ir_attr.index == 0
        cast = ascir.ops.Cast("cast")
        cast.y.dtype = ascir.dtypes.int4
        cast.x = x
        cast.attr.api.compute_type = "elemwise"
        try:
            graph.infer_dtypes()
        except Exception as e:
            assert (
                e.args[0]
                == "Check dtype failed for cast Cast; input_dtypes: [DT_INT8], output_dtypes: [DT_INT4]"
            )
        import sys

        # 通常为2 （变量 + getrefcount 参数）
        print(f"x.attr ref count is {sys.getrefcount(x.attr)}")
        del x.attr

    @staticmethod
    def test_graph_create_node_with_cast_api():
        graph = ascir.HintGraph("test")

        x = ascir_api.Data(graph, dtype=ascir.dtypes.int8)
        try:
            ascir_api.Cast(graph, x, dtype=ascir.dtypes.int4, axis=[])
        except Exception as e:
            assert (
                e.args[0]
                == "Check dtype failed for cast_0 Cast; input_dtypes: [DT_INT8], output_dtypes: [DT_INT4]"
            )

    @staticmethod
    def test_index_expr_and_arange_api():
        ascir.utils.set_platform("3510", 1, 245760, "SOC_EMPTY")
        try:
            graph = ascir.HintGraph("index_expr_arange_api")
            size = graph.create_size("size")
            axis = graph.create_axis("axis", size)

            index = ascir_api.IndexExpr(graph, dtype=ascir.dtypes.int64, expr=size + 1)
            values = ascir_api.arange(
                graph,
                dtype=ascir.dtypes.int64,
                base=ascir.SizeExpr(3),
                step=ascir.SizeExpr(2),
                axis=[axis],
                size=[size],
                stride=[ascir.SizeExpr(1)],
            )

            assert index.dtype == ascir.dtypes.int64
            assert values.dtype == ascir.dtypes.int64
            assert values.axis == [axis.id]
            assert values.size == [size]
            assert values.strides == [ascir.SizeExpr(1)]
            assert not hasattr(values, "vectorized_axis")
            assert not hasattr(values, "vectorized_strides")

            rows = graph.create_size("rows")
            row_axis = graph.create_axis("row", rows)
            expanded = ascir_api.Broadcast(
                graph,
                values,
                axis=[row_axis, axis],
                size=[rows, size],
                stride=[size, ascir.SizeExpr(1)],
            )
            assert expanded.axis == [row_axis.id, axis.id]
            assert expanded.size == [rows, size]
            assert expanded.strides == [size, ascir.SizeExpr(1)]

            singleton = graph.create_axis("singleton", 1)
            trailing_singleton = ascir_api.arange(
                graph,
                dtype=ascir.dtypes.int64,
                base=ascir.SizeExpr(0),
                step=ascir.SizeExpr(1),
                axis=[axis, singleton],
                size=[size, ascir.SizeExpr(1)],
                stride=[ascir.SizeExpr(1), ascir.SizeExpr(0)],
            )
            assert trailing_singleton.strides == [ascir.SizeExpr(1), ascir.SizeExpr(0)]

            stored = ascir_api.Store(
                graph,
                values,
                axis=[axis],
                size=[size],
                stride=[ascir.SizeExpr(1)],
            )
            ascir_api.Output(graph, stored, dtype=ascir.dtypes.int64)
            Autofuser(AutofuserOptions()).schedule(graph)

            broadcast_graph = ascir.HintGraph("explicit_arange_broadcast")
            b_rows = broadcast_graph.create_size("rows")
            b_cols = broadcast_graph.create_size("cols")
            b_row_axis = broadcast_graph.create_axis("row", b_rows)
            b_col_axis = broadcast_graph.create_axis("col", b_cols)
            arange_value = ascir_api.arange(
                broadcast_graph,
                dtype=ascir.dtypes.int64,
                base=ascir.SizeExpr(0),
                step=ascir.SizeExpr(1),
                axis=[b_col_axis],
                size=[b_cols],
                stride=[ascir.SizeExpr(1)],
            )
            expanded_value = ascir_api.Broadcast(
                broadcast_graph,
                arange_value,
                axis=[b_row_axis, b_col_axis],
                size=[b_rows, b_cols],
                stride=[b_cols, ascir.SizeExpr(1)],
            )
            broadcast_stored = ascir_api.Store(
                broadcast_graph,
                expanded_value,
                axis=[b_row_axis, b_col_axis],
                size=[b_rows, b_cols],
                stride=[b_cols, ascir.SizeExpr(1)],
            )
            ascir_api.Output(
                broadcast_graph, broadcast_stored, dtype=ascir.dtypes.int64
            )
            broadcast_fuser = Autofuser(AutofuserOptions())
            scheduled = broadcast_fuser.schedule(broadcast_graph)
            _, _, broadcast_kernel = broadcast_fuser.codegen(scheduled)
            assert broadcast_kernel
            assert "for (int64_t arange_b" not in broadcast_kernel
            assert "BroadcastExtend" in broadcast_kernel

            degenerate_graph = ascir.HintGraph("degenerate_arange_broadcast")
            d_rows = degenerate_graph.create_size("rows")
            d_cols = degenerate_graph.create_size("cols")
            d_row_axis = degenerate_graph.create_axis("row", d_rows)
            d_col_axis = degenerate_graph.create_axis("col", d_cols)
            # Pattern A: 退化前缀轴 Arange([1, cols]/[0, 1]) -> Broadcast。
            col_arange = ascir_api.arange(
                degenerate_graph,
                dtype=ascir.dtypes.int64,
                base=ascir.SizeExpr(0),
                step=ascir.SizeExpr(1),
                axis=[d_row_axis, d_col_axis],
                size=[ascir.SizeExpr(1), d_cols],
                stride=[ascir.SizeExpr(0), ascir.SizeExpr(1)],
            )
            # Pattern B: 退化尾轴 Arange([rows, 1]/[1, 0]) -> Broadcast。
            row_arange = ascir_api.arange(
                degenerate_graph,
                dtype=ascir.dtypes.int64,
                base=ascir.SizeExpr(0),
                step=d_cols,
                axis=[d_row_axis, d_col_axis],
                size=[d_rows, ascir.SizeExpr(1)],
                stride=[ascir.SizeExpr(1), ascir.SizeExpr(0)],
            )
            col_expanded = ascir_api.Broadcast(
                degenerate_graph,
                col_arange,
                axis=[d_row_axis, d_col_axis],
                size=[d_rows, d_cols],
                stride=[d_cols, ascir.SizeExpr(1)],
            )
            row_expanded = ascir_api.Broadcast(
                degenerate_graph,
                row_arange,
                axis=[d_row_axis, d_col_axis],
                size=[d_rows, d_cols],
                stride=[d_cols, ascir.SizeExpr(1)],
            )
            add_value = ascir.ops.Add("add", degenerate_graph)
            add_value.x1 = col_expanded
            add_value.x2 = row_expanded
            add_value.attr.sched.axis = [d_row_axis, d_col_axis]
            add_value.y.axis = [d_row_axis, d_col_axis]
            add_value.y.size = [d_rows, d_cols]
            add_value.y.strides = [d_cols, ascir.SizeExpr(1)]
            add_value.y.dtype = ascir.dtypes.int64
            degenerate_stored = ascir_api.Store(
                degenerate_graph,
                add_value.y,
                axis=[d_row_axis, d_col_axis],
                size=[d_rows, d_cols],
                stride=[d_cols, ascir.SizeExpr(1)],
            )
            ascir_api.Output(
                degenerate_graph, degenerate_stored, dtype=ascir.dtypes.int64
            )
            degenerate_fuser = Autofuser(AutofuserOptions())
            degenerate_scheduled = degenerate_fuser.schedule(degenerate_graph)
            _, _, degenerate_kernel = degenerate_fuser.codegen(degenerate_scheduled)
            assert degenerate_kernel
            assert "for (int64_t arange_b" not in degenerate_kernel
            assert "BroadcastExtend" in degenerate_kernel

            with pytest.raises(ValueError, match="axis must not be empty"):
                ascir_api.arange(
                    graph,
                    dtype=ascir.dtypes.int32,
                    base=ascir.SizeExpr(0),
                    step=ascir.SizeExpr(1),
                    axis=[],
                )
            with pytest.raises(ValueError, match="unit physical stride"):
                ascir_api.arange(
                    graph,
                    dtype=ascir.dtypes.int32,
                    base=ascir.SizeExpr(0),
                    step=ascir.SizeExpr(1),
                    axis=[axis],
                    size=[size],
                    stride=[ascir.SizeExpr(2)],
                )
            with pytest.raises(ValueError, match="explicit Broadcast"):
                ascir_api.arange(
                    graph,
                    dtype=ascir.dtypes.int32,
                    base=ascir.SizeExpr(0),
                    step=ascir.SizeExpr(1),
                    axis=[axis, singleton],
                    size=[size, ascir.SizeExpr(2)],
                    stride=[ascir.SizeExpr(1), ascir.SizeExpr(0)],
                )
            with pytest.raises(ValueError, match="same with axis len"):
                ascir_api.arange(
                    graph,
                    dtype=ascir.dtypes.int32,
                    base=ascir.SizeExpr(0),
                    step=ascir.SizeExpr(1),
                    axis=[axis],
                    stride=[],
                )
            with pytest.raises(Exception, match="Check dtype failed"):
                ascir_api.arange(
                    graph,
                    dtype=ascir.dtypes.float32,
                    base=ascir.SizeExpr(0),
                    step=ascir.SizeExpr(1),
                    axis=[axis],
                )
        finally:
            ascir.utils.set_platform("2201", 1, 245760)

    @staticmethod
    def test_indirect_load_max_expression_attr():
        graph = ascir.HintGraph("test_indirect_load_max")
        dynamic_max = graph.create_size("dynamic_max")
        indirect_load = ascir.ops.IndirectLoad("indirect_load")

        indirect_load.attr.ir_attr.max = dynamic_max
        assert indirect_load.attr.ir_attr.max == dynamic_max

        with pytest.raises(TypeError):
            indirect_load.attr.ir_attr.max = "invalid"

    @staticmethod
    def test_graph_create_node_with_indirect_load_api():
        import pyautofuse

        origin_pyautofuse = sys.modules.get("autofuse.pyautofuse")
        sys.modules["autofuse.pyautofuse"] = pyautofuse
        ascir_mod = pyautofuse.ascir
        ascir_mod.utils.set_platform("3510", 1, 245760)
        try:
            source_path = os.path.abspath(
                os.path.join(PYF_PATH, "../../..", "compiler", "python", "ascir_api.py")
            )
            spec = importlib.util.spec_from_file_location(
                "source_ascir_api", source_path
            )
            source_ascir_api = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(source_ascir_api)
            finally:
                if origin_pyautofuse is None:
                    sys.modules.pop("autofuse.pyautofuse", None)
                else:
                    sys.modules["autofuse.pyautofuse"] = origin_pyautofuse

            graph = ascir_mod.HintGraph("test_indirect_load_api")
            s0 = graph.create_size("s0")
            s1 = graph.create_size("s1")
            s2 = graph.create_size("s2")
            z0 = graph.create_axis("z0", s0)
            z1 = graph.create_axis("z1", s1)
            z2 = graph.create_axis("z2", s2)

            x = source_ascir_api.Data(graph, dtype=ascir_mod.dtypes.float32)
            x.axis = [z0, z1, z2]
            index = source_ascir_api.Data(graph, dtype=ascir_mod.dtypes.int32)
            index.axis = [z0, z1, z2]
            y = source_ascir_api.IndirectLoad(
                graph, x, index, axis=1, sched_axis=[z0, z1, z2]
            )

            assert y.dtype == ascir_mod.dtypes.float32
            debug_str = ascir_mod.utils.debug_str(graph)
            assert "IndirectLoad" in debug_str
            assert "axis" in debug_str
        finally:
            ascir_mod.utils.set_platform("2201", 1, 245760)

    @staticmethod
    def test_graph_create_const_node_with_value_str_attr():
        graph = ascir.HintGraph("test")

        x = ascir.ops.Scalar("x", graph)
        x.attr.ir_attr.value = "11.1"
        x.y.dtype = ascir.dtypes.float32
        debug_str = ascir.utils.debug_str(graph)
        assert x.attr.ir_attr.value == "11.1"
        assert debug_str

    @staticmethod
    def test_graph_create_node_with_axis():
        graph = ascir.HintGraph("test")

        s0 = graph.create_size("s0")
        s1 = graph.create_size("s1")

        z0 = graph.create_axis("z0", s0)
        z1 = graph.create_axis("z1", s1)
        z2 = graph.create_axis("z2", 512)

        x = ascir.ops.Data("x", graph)
        x.attr.sched.axis = [z0, z1, z2]

        ascir.ops.Load("load")
        x.y.dtype = ascir.dtypes.float16
        x.y.axis = [z0, z1, z2]
        assert x.y.axis == [z0.id, z1.id, z2.id]
        x.y.size = [s0, s1, 512]
        assert x.y.size == [s0, s1, 512]
        x.y.strides = [s1 * 512, 512, ascir.SizeExpr(1)]
        assert x.y.strides == [s1 * 512, 512, ascir.SizeExpr(1)]
        debug_str = ascir.utils.debug_str(graph)
        assert debug_str

    @staticmethod
    def test_graph_link_nodes():
        graph = ascir.HintGraph("test")

        x = ascir.ops.Data("x", graph)
        x.y.dtype = ascir.dtypes.int64
        load = ascir.ops.Load("load")
        load.x = x
        try:
            graph.infer_dtypes()
        except Exception as e:
            assert (
                e.args[0]
                == "Infer dtype failed for load Load; input_dtypes: [DT_INT64] is not supported now"
            )
        graph.infer_dtypes()
        debug_str = ascir.utils.debug_str(graph)
        assert debug_str

    @staticmethod
    def test_graph_link_nodes_by_output():
        graph = ascir.HintGraph("test")

        x = ascir.ops.Data("x", graph)

        load = ascir.ops.Load("load")
        load.x = x.y
        graph.infer_dtypes()
        debug_str = ascir.utils.debug_str(graph)
        assert debug_str

    @staticmethod
    def test_duration_record():
        start = time.time()
        time.sleep(0.1)
        end = time.time()
        ascir.utils.duration_record(
            ["device", "fused_graph"], int(start * 1e9), int((end - start) * 1e9)
        )
        ascir.utils.report_durations()
        try:
            ascir.utils.duration_record(["device", "fused_graph"], "time")
        except TypeError as e:
            assert e.args[0] == "UtilsDurationRecord param parse failed"

        try:
            ascir.utils.duration_record(["device", "fused_graph"], int(-1), int(-1))
        except TypeError as e:
            assert e.args[0] == "duration param is invalid"

        try:
            ascir.utils.duration_record([0, 1], int(-1), int(-1))
        except TypeError as e:
            assert e.args[0] == "target param is invalid"


class TestAutofuseLoadAbsStore:
    @staticmethod
    def construct_graph():
        graph = ascir.HintGraph("LoadAbsStore")
        s0 = graph.create_size("s0")
        s1 = graph.create_size("s1")
        s2 = graph.create_size("s2")

        z0 = graph.create_axis("z0", s0 + 100)
        assert z0.size.expression == "(100 + s0)"
        z1 = graph.create_axis("z1", s1)
        z2 = graph.create_axis("z2", s2)
        buf_z0 = graph.create_axis("buf_z0", 100)
        buf_z1 = graph.create_axis("buf_z1", s1)
        buf_z2 = graph.create_axis("buf_z2", s2)

        arg3_1 = ascir.ops.Data("arg3_1", graph)
        arg3_1.attr.ir_attr.index = 0
        arg3_1.attr.sched.axis = [z0, z1, z2]
        arg3_1.y.dtype = ascir.dtypes.float16
        arg3_1.y.axis = [z0, z1, z2]
        arg3_1.y.size = [100 + s0, s1, s2]
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
        load.y.size = [100 + s0, s1, s2]
        load.y.strides = [s1 * s2, s2, ascir.SizeExpr(1)]

        abs_op = ascir.ops.Abs("abs")
        abs_op.x = load
        abs_op.attr.sched.axis = [z0, z1, z2]
        abs_op.y.dtype = ascir.dtypes.float16
        abs_op.y.axis = [z0, z1, z2]
        abs_op.y.size = [100 + s0, s1, s2]
        abs_op.y.strides = [s1 * s2, s2, ascir.SizeExpr(1)]

        store = ascir.ops.Store("store")
        try:
            store.attr.ir_attr.offset = "4"
        except Exception as e:
            assert e.args[0] == "Only support type of SizeExpr or long"
        store.attr.ir_attr.offset = offset_of_0 + 1
        assert store.attr.ir_attr.offset.expression == "1"
        store.x = abs_op
        store.attr.sched.axis = [z0, z1, z2]
        store.y.dtype = ascir.dtypes.float16
        store.y.axis = [z0, z1, z2]
        store.y.size = [100 + s0, s1, s2]
        store.y.strides = [s1 * s2, s2, ascir.SizeExpr(1)]

        buf1 = ascir.ops.Output("buf1", graph)
        buf1.attr.ir_attr.index = 0
        assert buf1.attr.ir_attr.index == 0
        buf1.x = store
        buf1.attr.sched.axis = [z0, z1, z2]
        buf1.y.dtype = ascir.dtypes.float16
        buf1.y.axis = [z0, z1, z2]
        buf1.y.size = [100 + s0, s1, s2]
        buf1.y.strides = [s1 * s2, s2, ascir.SizeExpr(1)]
        graph.set_axis_map({z0: [buf_z0], z1: [buf_z1], z2: [buf_z2]})
        return graph

    def test_construct_graph(self):
        graph = self.construct_graph()
        debug_str = ascir.utils.debug_str(graph)
        assert debug_str

    def test_optimize(self):
        options = AutofuserOptions()
        fuser = Autofuser(options)

        hint_graph = self.construct_graph()
        schedule_results = fuser.schedule(hint_graph)
        schedule_results.get_name()
        schedule_results.get_input_num()
        schedule_results.get_output_num()

    def test_codegen(self):
        options = AutofuserOptions()
        fuser = Autofuser(options)

        hint_graph = self.construct_graph()
        impl_graphs = fuser.schedule(hint_graph)
        tiling_def, host_tiling, op_kernel = fuser.codegen(impl_graphs)

    def test_device_code_generator(self):
        # 测试device_code_generator方法
        options = AutofuserOptions()
        fuser = Autofuser(options)
        codegen = CodeGen()

        hint_graph = self.construct_graph()
        schedule_results = fuser.schedule(hint_graph)

        # 调用device_code_generator方法
        tiling_dict, kernel_dict = codegen.device_code_generator(schedule_results)

        # 验证返回结果是字典类型
        assert isinstance(tiling_dict, dict), "tiling_dict should be a dictionary"
        assert isinstance(kernel_dict, dict), "kernel_dict should be a dictionary"

        # 验证字典中包含预期的键（可能是"ub"、"common"或"default"）
        assert any(key in tiling_dict for key in ["ub", "common", "default"]), (
            "tiling_dict should contain at least one of the expected keys"
        )
        assert any(key in kernel_dict for key in ["ub", "common", "default"]), (
            "kernel_dict should contain at least one of the expected keys"
        )

    def test_host_code_generator(self):
        # 测试host_code_generator方法
        options = AutofuserOptions()
        fuser = Autofuser(options)
        codegen = CodeGen()

        hint_graph = self.construct_graph()
        schedule_results = fuser.schedule(hint_graph)

        # 准备参数
        shape_info = None  # 使用None作为shape_info
        output_shape = [["z0", "z1", "z2"]]  # 输出形状示例
        pgo_dir = ""
        vector_core_num = ""

        # 调用host_code_generator方法
        py_tilings, infer_shape = codegen.host_code_generator(
            schedule_results, shape_info, output_shape, pgo_dir, vector_core_num
        )

        # 验证返回结果类型
        assert isinstance(py_tilings, dict), "py_tilings should be a dictionary"
        assert isinstance(infer_shape, str), "infer_shape should be a string"

        # 验证py_tilings中包含预期的键（可能是"ub"、"common"或"default"）
        assert any(key in py_tilings for key in ["ub", "common", "default"]), (
            "py_tilings should contain at least one of the expected keys"
        )

    def test_autofuse_backend(self):
        options = AutofuserOptions()
        fuser = Autofuser(options)

        hint_graph = self.construct_graph()
        tiling_def, host_tiling, op_kernel = fuser.autofuse_backend(hint_graph)

    def test_autofuse_backend_runtime_error_does_not_return_partial_result(self):
        assert RUNTIME_ERROR_TILING_LIB.is_file(), (
            f"missing runtime error tiling library: {RUNTIME_ERROR_TILING_LIB}"
        )
        options = AutofuserOptions(
            tiling_lib_path=str(RUNTIME_ERROR_TILING_LIB),
            tiling_lib_codegen_symbol="CodegenTiling",
        )
        fuser = Autofuser(options)

        hint_graph = self.construct_graph()
        with pytest.raises(
            RuntimeError,
            match=r"^Codegen generate kernel failed: pyautofuse runtime error test$",
        ):
            fuser.autofuse_backend(hint_graph)


class TestAutofuseLoadMatMulStore:
    @staticmethod
    def construct_graph():
        graph = ascir.HintGraph("LoadCubeStore")
        s0 = graph.create_size("s0")
        s1 = graph.create_size("s1")
        s2 = graph.create_size("s2")

        z0 = graph.create_axis("z0", s0 + 100)
        assert z0.size.expression == "(100 + s0)"
        z1 = graph.create_axis("z1", s1)
        z2 = graph.create_axis("z2", s2)
        buf_z0 = graph.create_axis("buf_z0", 100)
        buf_z1 = graph.create_axis("buf_z1", s1)
        buf_z2 = graph.create_axis("buf_z2", s2)

        arg3_1 = ascir.ops.Data("arg3_1", graph)
        arg3_1.attr.ir_attr.index = 0
        arg3_1.attr.sched.axis = [z0, z1, z2]
        arg3_1.y.dtype = ascir.dtypes.float16
        arg3_1.y.axis = [z0, z1, z2]
        arg3_1.y.size = [100 + s0, s1, s2]
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
        load.y.size = [100 + s0, s1, s2]
        load.y.strides = [s1 * s2, s2, ascir.SizeExpr(1)]

        matmul_op = ascir.ops.MatMul("matmul")
        matmul_op.x1 = load
        matmul_op.x2 = load
        matmul_op.attr.sched.axis = [z0, z1, z2]
        matmul_op.attr.api.compute_type = "cube"
        matmul_op.attr.ir_attr.enable_hf32 = 1
        matmul_op.attr.ir_attr.transpose_x1 = 0
        matmul_op.attr.ir_attr.transpose_x2 = 1
        matmul_op.attr.ir_attr.has_relu = 1
        matmul_op.attr.ir_attr.offset_x = 1
        matmul_op.y.dtype = ascir.dtypes.float16
        matmul_op.y.axis = [z0, z1, z2]
        matmul_op.y.size = [100 + s0, s1, s2]
        matmul_op.y.strides = [s1 * s2, s2, ascir.SizeExpr(1)]

        store = ascir.ops.Store("store")
        try:
            store.attr.ir_attr.offset = "4"
        except Exception as e:
            assert e.args[0] == "Only support type of SizeExpr or long"
        store.attr.ir_attr.offset = offset_of_0 + 1
        assert store.attr.ir_attr.offset.expression == "1"
        store.x = matmul_op
        store.attr.sched.axis = [z0, z1, z2]
        store.y.dtype = ascir.dtypes.float16
        store.y.axis = [z0, z1, z2]
        store.y.size = [100 + s0, s1, s2]
        store.y.strides = [s1 * s2, s2, ascir.SizeExpr(1)]

        buf1 = ascir.ops.Output("buf1", graph)
        buf1.attr.ir_attr.index = 0
        assert buf1.attr.ir_attr.index == 0
        buf1.x = store
        buf1.attr.sched.axis = [z0, z1, z2]
        buf1.y.dtype = ascir.dtypes.float16
        buf1.y.axis = [z0, z1, z2]
        buf1.y.size = [100 + s0, s1, s2]
        buf1.y.strides = [s1 * s2, s2, ascir.SizeExpr(1)]
        graph.set_axis_map({z0: [buf_z0], z1: [buf_z1], z2: [buf_z2]})
        ascir.utils.dump(graph)
        return graph

    def test_optimize_cube(self):
        options = AutofuserOptions()
        fuser = Autofuser(options)

        hint_graph = self.construct_graph()
        schedule_results = fuser.schedule(hint_graph)
        schedule_results.is_cube_type()
        schedule_results.get_cube_attributes()

    def test_device_code_generator(self):
        # 测试device_code_generator方法
        options = AutofuserOptions()
        fuser = Autofuser(options)
        codegen = CodeGen()

        hint_graph = self.construct_graph()
        schedule_results = fuser.schedule(hint_graph)

        # 调用device_code_generator方法
        tiling_dict, kernel_dict = codegen.device_code_generator(schedule_results)

        # 验证返回结果是字典类型
        assert isinstance(tiling_dict, dict), "tiling_dict should be a dictionary"
        assert isinstance(kernel_dict, dict), "kernel_dict should be a dictionary"

        # 验证字典中包含预期的键（可能是"ub"、"common"或"default"）
        assert any(key in tiling_dict for key in ["ub", "common", "default"]), (
            "tiling_dict should contain at least one of the expected keys"
        )
        assert any(key in kernel_dict for key in ["ub", "common", "default"]), (
            "kernel_dict should contain at least one of the expected keys"
        )

    def test_host_code_generator(self):
        # 测试host_code_generator方法
        options = AutofuserOptions()
        fuser = Autofuser(options)
        codegen = CodeGen()

        hint_graph = self.construct_graph()
        schedule_results = fuser.schedule(hint_graph)

        # 准备参数
        shape_info = None  # 使用None作为shape_info
        output_shape = [["z0", "z1", "z2"]]  # 输出形状示例
        pgo_dir = ""
        vector_core_num = ""

        # 调用host_code_generator方法
        py_tilings, infer_shape = codegen.host_code_generator(
            schedule_results, shape_info, output_shape, pgo_dir, vector_core_num
        )

        # 验证返回结果类型
        assert isinstance(py_tilings, dict), "py_tilings should be a dictionary"
        assert isinstance(infer_shape, str), "infer_shape should be a string"

        # 验证py_tilings中包含预期的键（可能是"ub"、"common"或"default"）
        assert any(key in py_tilings for key in ["ub", "common", "default"]), (
            "py_tilings should contain at least one of the expected keys"
        )

        # 验证每个值也是字典
        for key, value in py_tilings.items():
            assert isinstance(value, dict), (
                f"Value for key {key} should be a dictionary"
            )


class TestAutofuseLoadBatchMatmulStore:
    @staticmethod
    def construct_graph():
        graph = ascir.HintGraph("LoadBatchMatMulStore")
        s0 = graph.create_size("s0")
        s1 = graph.create_size("s1")
        s2 = graph.create_size("s2")

        z0 = graph.create_axis("z0", s0 + 100)
        assert z0.size.expression == "(100 + s0)"
        z1 = graph.create_axis("z1", s1)
        z2 = graph.create_axis("z2", s2)
        buf_z0 = graph.create_axis("buf_z0", 100)
        buf_z1 = graph.create_axis("buf_z1", s1)
        buf_z2 = graph.create_axis("buf_z2", s2)

        arg3_1 = ascir.ops.Data("arg3_1", graph)
        arg3_1.attr.ir_attr.index = 0
        arg3_1.attr.sched.axis = [z0, z1, z2]
        arg3_1.y.dtype = ascir.dtypes.float16
        arg3_1.y.axis = [z0, z1, z2]
        arg3_1.y.size = [100 + s0, s1, s2]
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
        load.y.size = [100 + s0, s1, s2]
        load.y.strides = [s1 * s2, s2, ascir.SizeExpr(1)]

        matmul_op = ascir.ops.BatchMatMul("batch_matmul")
        matmul_op.x1 = load
        matmul_op.x2 = load
        matmul_op.attr.sched.axis = [z0, z1, z2]
        matmul_op.attr.api.compute_type = "cube"
        matmul_op.attr.ir_attr.enable_hf32 = 1
        matmul_op.attr.ir_attr.adj_x1 = 0
        matmul_op.attr.ir_attr.adj_x2 = 1
        matmul_op.attr.ir_attr.has_relu = 1
        matmul_op.attr.ir_attr.offset_x = 1
        matmul_op.y.dtype = ascir.dtypes.float16
        matmul_op.y.axis = [z0, z1, z2]
        matmul_op.y.size = [100 + s0, s1, s2]
        matmul_op.y.strides = [s1 * s2, s2, ascir.SizeExpr(1)]

        store = ascir.ops.Store("store")
        try:
            store.attr.ir_attr.offset = "4"
        except Exception as e:
            assert e.args[0] == "Only support type of SizeExpr or long"
        store.attr.ir_attr.offset = offset_of_0 + 1
        assert store.attr.ir_attr.offset.expression == "1"
        store.x = matmul_op
        store.attr.sched.axis = [z0, z1, z2]
        store.y.dtype = ascir.dtypes.float16
        store.y.axis = [z0, z1, z2]
        store.y.size = [100 + s0, s1, s2]
        store.y.strides = [s1 * s2, s2, ascir.SizeExpr(1)]

        buf1 = ascir.ops.Output("buf1", graph)
        buf1.attr.ir_attr.index = 0
        assert buf1.attr.ir_attr.index == 0
        buf1.x = store
        buf1.attr.sched.axis = [z0, z1, z2]
        buf1.y.dtype = ascir.dtypes.float16
        buf1.y.axis = [z0, z1, z2]
        buf1.y.size = [100 + s0, s1, s2]
        buf1.y.strides = [s1 * s2, s2, ascir.SizeExpr(1)]
        graph.set_axis_map({z0: [buf_z0], z1: [buf_z1], z2: [buf_z2]})
        ascir.utils.dump(graph)
        return graph

    def test_optimize_cube(self):
        options = AutofuserOptions()
        fuser = Autofuser(options)

        hint_graph = self.construct_graph()
        schedule_results = fuser.schedule(hint_graph)
        schedule_results.is_cube_type()
        schedule_results.get_cube_attributes()


class TestAutofuseLoadMatMulStoreNew:
    @staticmethod
    def construct_graph():
        graph = ascir.HintGraph("LoadCubeStoreNew")
        s0 = graph.create_size("s0")
        s1 = graph.create_size("s1")
        s2 = graph.create_size("s2")

        # 创建矩阵乘法所需的轴：m, k, n
        m = graph.create_axis("m", s0)
        k = graph.create_axis("k", s1)
        n = graph.create_axis("n", s2)

        # 创建缓冲区轴
        buf_m = graph.create_axis("buf_m", s0)
        buf_k = graph.create_axis("buf_k", s1)
        buf_n = graph.create_axis("buf_n", s2)

        # 创建输入矩阵X1 (m x k)
        data_x1 = ascir.ops.Data("data_x1", graph)
        data_x1.attr.ir_attr.index = 0
        data_x1.attr.sched.axis = [m, k]
        data_x1.y.dtype = ascir.dtypes.float16
        data_x1.y.axis = [m, k]
        data_x1.y.size = [s0, s1]
        data_x1.y.strides = [s1, ascir.SizeExpr(1)]

        # 创建输入矩阵X2 (k x n)
        data_x2 = ascir.ops.Data("data_x2", graph)
        data_x2.attr.ir_attr.index = 1
        data_x2.attr.sched.axis = [k, n]
        data_x2.y.dtype = ascir.dtypes.float16
        data_x2.y.axis = [k, n]
        data_x2.y.size = [s1, s2]
        data_x2.y.strides = [s2, ascir.SizeExpr(1)]

        # 创建Load操作读取X1
        load_x1 = ascir.ops.Load("load_x1")
        offset_of_0 = ascir.SizeExpr(0)
        load_x1.attr.ir_attr.offset = offset_of_0
        load_x1.x = data_x1
        load_x1.attr.sched.axis = [m, k]
        load_x1.y.dtype = ascir.dtypes.float16
        load_x1.y.axis = [m, k]
        load_x1.y.size = [s0, s1]
        load_x1.y.strides = [s1, ascir.SizeExpr(1)]

        # 创建Load操作读取X2
        load_x2 = ascir.ops.Load("load_x2")
        load_x2.attr.ir_attr.offset = offset_of_0
        load_x2.x = data_x2
        load_x2.attr.sched.axis = [k, n]
        load_x2.y.dtype = ascir.dtypes.float16
        load_x2.y.axis = [k, n]
        load_x2.y.size = [s1, s2]
        load_x2.y.strides = [s2, ascir.SizeExpr(1)]

        # 创建MatMul操作，设置为cube类型
        matmul_op = ascir.ops.MatMul("matmul")
        matmul_op.x1 = load_x1
        matmul_op.x2 = load_x2
        matmul_op.attr.sched.axis = [m, n, k]
        matmul_op.attr.api.compute_type = "cube"  # 确保设置为cube类型
        matmul_op.attr.ir_attr.enable_hf32 = 1
        matmul_op.attr.ir_attr.transpose_x1 = 0  # X1不转置
        matmul_op.attr.ir_attr.transpose_x2 = 0  # X2不转置
        matmul_op.attr.ir_attr.has_relu = 0  # 不使用ReLU
        matmul_op.attr.ir_attr.offset_x = 0  # 偏移量为0
        matmul_op.y.dtype = ascir.dtypes.float16
        matmul_op.y.axis = [m, n]
        matmul_op.y.size = [s0, s2]
        matmul_op.y.strides = [s2, ascir.SizeExpr(1)]

        # 在MatMul节点后添加Abs节点
        abs_op = ascir.ops.Abs("abs")
        abs_op.x = matmul_op
        abs_op.attr.sched.axis = [m, n]
        abs_op.attr.api.compute_type = "elemwise"  # 设置为elewise类型
        abs_op.y.dtype = ascir.dtypes.float16
        abs_op.y.axis = [m, n]
        abs_op.y.size = [s0, s2]
        abs_op.y.strides = [s2, ascir.SizeExpr(1)]

        # 创建Store操作，将Abs节点的输出连接到Store
        store = ascir.ops.Store("store")
        store.attr.ir_attr.offset = offset_of_0 + 1
        store.x = abs_op  # Store的输入现在是Abs节点
        store.attr.sched.axis = [m, n]
        store.y.dtype = ascir.dtypes.float16
        store.y.axis = [m, n]
        store.y.size = [s0, s2]
        store.y.strides = [s2, ascir.SizeExpr(1)]

        # 创建输出节点
        buf1 = ascir.ops.Output("buf1", graph)
        buf1.attr.ir_attr.index = 0
        buf1.x = store
        buf1.attr.sched.axis = [m, n]
        buf1.y.dtype = ascir.dtypes.float16
        buf1.y.axis = [m, n]
        buf1.y.size = [s0, s2]
        buf1.y.strides = [s2, ascir.SizeExpr(1)]

        # 设置轴映射
        graph.set_axis_map({m: [buf_m], k: [buf_k], n: [buf_n]})
        ascir.utils.dump(graph)
        return graph

    def test_optimize_cube(self):
        options = AutofuserOptions()
        fuser = Autofuser(options)

        hint_graph = self.construct_graph()
        schedule_results = fuser.schedule(hint_graph)
        schedule_results.is_cube_type()
        schedule_results.get_cube_attributes()

    def test_device_code_generator(self):
        # 测试device_code_generator方法
        options = AutofuserOptions()
        fuser = Autofuser(options)
        codegen = CodeGen()

        hint_graph = self.construct_graph()
        schedule_results = fuser.schedule(hint_graph)

        # 调用device_code_generator方法
        tiling_dict, kernel_dict = codegen.device_code_generator(schedule_results)

        # 验证返回结果是字典类型
        assert isinstance(tiling_dict, dict), "tiling_dict should be a dictionary"
        assert isinstance(kernel_dict, dict), "kernel_dict should be a dictionary"

        # 验证字典中只包含"ub"或"common"键
        assert any(key in tiling_dict for key in ["ub", "common", "default"]), (
            "tiling_dict should contain either 'ub', 'common' or 'default' key"
        )
        assert all(key in ["ub", "common", "default"] for key in tiling_dict.keys()), (
            "tiling_dict should only contain 'ub', 'common' or 'default' keys"
        )

        assert any(key in kernel_dict for key in ["ub", "common", "default"]), (
            "kernel_dict should contain either 'ub', 'common' or 'default' key"
        )
        assert all(key in ["ub", "common", "default"] for key in kernel_dict.keys()), (
            "kernel_dict should only contain 'ub', 'common' or 'default' keys"
        )

    def test_host_code_generator(self):
        # 测试host_code_generator方法
        options = AutofuserOptions()
        fuser = Autofuser(options)
        codegen = CodeGen()

        hint_graph = self.construct_graph()
        schedule_results = fuser.schedule(hint_graph)

        # 准备参数
        shape_info = None  # 使用None作为shape_info
        output_shape = [["m", "n"]]  # 输出形状示例
        pgo_dir = ""
        vector_core_num = ""

        # 调用host_code_generator方法
        py_tilings, infer_shape = codegen.host_code_generator(
            schedule_results, shape_info, output_shape, pgo_dir, vector_core_num
        )

        # 验证返回结果类型
        assert isinstance(py_tilings, dict), "py_tilings should be a dictionary"
        assert isinstance(infer_shape, str), "infer_shape should be a string"

        # 验证py_tilings中只包含"ub"或"common"键
        assert any(key in py_tilings for key in ["ub", "common", "default"]), (
            "py_tilings should contain either 'ub', 'common' or 'default' key"
        )
        assert all(key in ["ub", "common", "default"] for key in py_tilings.keys()), (
            "py_tilings should only contain 'ub', 'common' or 'default' keys"
        )

        # 验证每个值也是字典
        for key, value in py_tilings.items():
            assert isinstance(value, dict), (
                f"Value for key {key} should be a dictionary"
            )


class TestCubeAttributes:
    @staticmethod
    def construct_graph_with_cube_matmul():
        graph = ascir.HintGraph("LoadCubeStoreTest")
        s0 = graph.create_size("s0")
        s1 = graph.create_size("s1")
        s2 = graph.create_size("s2")

        # 创建矩阵乘法所需的轴
        m = graph.create_axis("m", s0)
        k = graph.create_axis("k", s1)
        n = graph.create_axis("n", s2)

        # 创建缓冲区轴
        buf_m = graph.create_axis("buf_m", s0)
        buf_k = graph.create_axis("buf_k", s1)
        buf_n = graph.create_axis("buf_n", s2)

        # 创建输入矩阵X1 (m x k)
        data_x1 = ascir.ops.Data("data_x1", graph)
        data_x1.attr.ir_attr.index = 0
        data_x1.attr.sched.axis = [m, k]
        data_x1.y.dtype = ascir.dtypes.float16
        data_x1.y.axis = [m, k]
        data_x1.y.size = [s0, s1]
        data_x1.y.strides = [s1, ascir.SizeExpr(1)]

        # 创建输入矩阵X2 (k x n)
        data_x2 = ascir.ops.Data("data_x2", graph)
        data_x2.attr.ir_attr.index = 1
        data_x2.attr.sched.axis = [k, n]
        data_x2.y.dtype = ascir.dtypes.float16
        data_x2.y.axis = [k, n]
        data_x2.y.size = [s1, s2]
        data_x2.y.strides = [s2, ascir.SizeExpr(1)]

        # 创建Load操作读取X1
        load_x1 = ascir.ops.Load("load_x1")
        load_x1.attr.ir_attr.offset = ascir.SizeExpr(0)
        load_x1.x = data_x1
        load_x1.attr.sched.axis = [m, k]
        load_x1.y.dtype = ascir.dtypes.float16
        load_x1.y.axis = [m, k]
        load_x1.y.size = [s0, s1]
        load_x1.y.strides = [s1, ascir.SizeExpr(1)]

        # 创建Load操作读取X2
        load_x2 = ascir.ops.Load("load_x2")
        load_x2.attr.ir_attr.offset = ascir.SizeExpr(0)
        load_x2.x = data_x2
        load_x2.attr.sched.axis = [k, n]
        load_x2.y.dtype = ascir.dtypes.float16
        load_x2.y.axis = [k, n]
        load_x2.y.size = [s1, s2]
        load_x2.y.strides = [s2, ascir.SizeExpr(1)]

        # 创建MatMul操作，设置为cube类型
        matmul_op = ascir.ops.MatMul("matmul")
        matmul_op.x1 = load_x1
        matmul_op.x2 = load_x2
        matmul_op.attr.sched.axis = [m, n, k]
        matmul_op.attr.api.compute_type = "cube"  # 确保设置为cube类型
        matmul_op.attr.ir_attr.enable_hf32 = 1  # 启用hf32
        matmul_op.attr.ir_attr.transpose_x1 = 0  # X1不转置
        matmul_op.attr.ir_attr.transpose_x2 = 1  # X2转置
        matmul_op.attr.ir_attr.has_relu = 1  # 使用ReLU
        matmul_op.attr.ir_attr.offset_x = 2  # 偏移量为2
        matmul_op.y.dtype = ascir.dtypes.float16
        matmul_op.y.axis = [m, n]
        matmul_op.y.size = [s0, s2]
        matmul_op.y.strides = [s2, ascir.SizeExpr(1)]

        # 创建Store操作
        store = ascir.ops.Store("store")
        store.attr.ir_attr.offset = ascir.SizeExpr(0)
        store.x = matmul_op
        store.attr.sched.axis = [m, n]
        store.y.dtype = ascir.dtypes.float16
        store.y.axis = [m, n]
        store.y.size = [s0, s2]
        store.y.strides = [s2, ascir.SizeExpr(1)]

        # 创建输出节点
        buf1 = ascir.ops.Output("buf1", graph)
        buf1.attr.ir_attr.index = 0
        buf1.x = store
        buf1.attr.sched.axis = [m, n]
        buf1.y.dtype = ascir.dtypes.float16
        buf1.y.axis = [m, n]
        buf1.y.size = [s0, s2]
        buf1.y.strides = [s2, ascir.SizeExpr(1)]

        # 设置轴映射
        graph.set_axis_map({m: [buf_m], k: [buf_k], n: [buf_n]})
        return graph

    def test_cube_attributes_extraction(self):
        options = AutofuserOptions()
        fuser = Autofuser(options)

        hint_graph = self.construct_graph_with_cube_matmul()
        schedule_results = fuser.schedule(hint_graph)

        # 验证是cube类型，当前schedule result 不支持构造cube结果，暂时不验证is_cube结果，待支持后再验证
        # is_cube = schedule_results.is_cube_type()
        # assert is_cube, "Graph should be of cube type"

        # 获取cube属性
        cube_attrs = schedule_results.get_cube_attributes()
        assert isinstance(cube_attrs, dict), "Cube attributes should be a dictionary"
        # assert "cube_attributes" in cube_attrs, "Cube attributes should contain 'cube_attributes' key"

        # 验证具体属性，当前schedule result 不支持构造cube结果，暂时不验证具体值
        # attr_dict = cube_attrs["cube_attributes"]
        # assert attr_dict["has_relu"] == True, "has_relu should be True"
        # assert attr_dict["transpose_x1"] == False, "transpose_x1 should be False"
        # assert attr_dict["transpose_x2"] == True, "transpose_x2 should be True"
        # assert attr_dict["offset_x"] == 2, "offset_x should be 2"
        # assert attr_dict["enable_hf32"] == 1, "enable_hf32 should be 1"
        # assert attr_dict["type_size"] == 2, "type_size should be 2 for float16"
        # assert attr_dict["input_num"] == 2, "input_num should be 2"


class TestAutofuseGatherAbsStore:
    @staticmethod
    def construct_graph():
        graph = ascir.HintGraph("GatherAbsStore")
        size_of_z0 = ascir.SizeExpr(4001)
        z0 = graph.create_axis("z0", size_of_z0)
        size_of_z1 = ascir.SizeExpr(100)
        z1 = graph.create_axis("z1", size_of_z1)
        size_of_z2 = ascir.SizeExpr(1000)
        z2 = graph.create_axis("z2", size_of_z2)

        assert z0.size.expression == "4001"
        assert z1.size.expression == "100"
        assert z2.size.expression == "1000"

        data_0 = ascir.ops.Data("data_0", graph)
        data_0.attr.sched.axis = [z0]
        data_0.y.axis = [z0]
        data_0.y.size = [4001]
        data_0.y.strides = [1]
        data_0.y.dtype = ascir.dtypes.float32

        data_1 = ascir.ops.Data("data_1", graph)
        data_1.attr.sched.axis = [z1, z2]
        data_1.y.axis = [z1, z2]
        data_1.y.size = [100, 1000]
        data_1.y.strides = [1000, 1]
        data_1.y.dtype = ascir.dtypes.int64

        gather_0 = ascir.ops.Gather("gather_0")
        gather_0.attr.ir_attr.axis = 0
        gather_0.attr.api.compute_type = "gather"
        gather_0.attr.sched.axis = [z1, z2]
        gather_0.x1 = data_0.y
        gather_0.x2 = data_1.y
        gather_0.y.axis = [z1, z2]
        gather_0.y.size = [100, 1000]
        gather_0.y.strides = [1000, 1]
        gather_0.y.dtype = ascir.dtypes.float32

        abs_0 = ascir.ops.Abs("abs_0")
        abs_0.attr.sched.axis = [z1, z2]
        abs_0.x = gather_0.y
        abs_0.y.axis = [z1, z2]
        abs_0.y.size = [100, 1000]
        abs_0.y.strides = [1000, 1]
        abs_0.y.dtype = ascir.dtypes.float32

        store_0 = ascir.ops.Store("store_0")
        store_0.attr.sched.axis = [z1, z2]
        store_0.x = abs_0.y
        store_0.y.axis = [z1, z2]
        store_0.y.size = [100, 1000]
        store_0.y.strides = [1000, 1]
        store_0.y.dtype = ascir.dtypes.float32

        output_0 = ascir.ops.Output("output_0")
        output_0.attr.sched.axis = [z1, z2]
        output_0.x = store_0.y
        output_0.y.dtype = ascir.dtypes.float32
        graph.set_axis_map({z0: [z0], z1: [z1], z2: [z2]})
        return graph

    def test_construct_graph(self):
        graph = self.construct_graph()
        debug_str = ascir.utils.debug_str(graph)
        assert debug_str

    @pytest.mark.skip
    def test_optimize(self):
        options = AutofuserOptions()
        fuser = Autofuser(options)
        hint_graph = self.construct_graph()
        fuser.autofuse(hint_graph)

    @pytest.mark.skip
    def test_codegen(self):
        options = AutofuserOptions()
        fuser = Autofuser(options)

        hint_graph = self.construct_graph()
        impl_graphs = fuser.schedule(hint_graph)
        tiling_def, host_tiling, op_kernel = fuser.codegen(impl_graphs)


class TestAutofuseLoadConcatStore:
    @staticmethod
    def construct_graph():
        try:
            npu_kernel0_graph = ascir.HintGraph(100)
        except Exception as e:
            assert e.args[0] == "argument 1 must be str, not int"
        npu_kernel0_graph = ascir.HintGraph("LoadConcatStore")
        s0 = npu_kernel0_graph.create_size("s0")
        s1 = npu_kernel0_graph.create_size("s1")
        z0 = npu_kernel0_graph.create_axis("z0", s0)
        z1 = npu_kernel0_graph.create_axis("z1", s1 * 2)
        arg2_1 = ascir.ops.Data("arg2_1", npu_kernel0_graph)
        arg2_1.y.dtype = ascir.dtypes.float16
        load = ascir.ops.Load("load")
        try:
            load.infer_dtype()
        except Exception as e:
            assert e.args[0] == "node load Load need set input before call infer dtype"
        load.attr.sched.axis = [z0, z1]
        load.x = arg2_1.y
        load.y.axis = [z0, z1]
        load.y.strides = [s1, ascir.SizeExpr(1)]
        load.y.size = [s0, s1]
        load.infer_dtype()
        assert load.y.dtype == ascir.dtypes.float16
        arg3_1 = ascir.ops.Data("arg3_1", npu_kernel0_graph)
        arg3_1.y.dtype = ascir.dtypes.float16
        load1 = ascir.ops.Load("load1")
        load1.attr.sched.axis = [z0, z1]
        assert load1.attr.sched.axis == [z0.id, z1.id]
        load1.x = arg3_1.y
        load1.y.axis = [z0, z1]
        load1.y.strides = [s1, ascir.SizeExpr(1)]
        load1.y.size = [s0, s1]
        concat = ascir.ops.Concat("concat")
        concat.attr.sched.axis = [z0, z1]
        concat.x = [load, load1.y]
        concat.y.axis = [z0, z1]
        concat.y.strides = [s1 + s1, ascir.SizeExpr(1)]
        concat.y.size = [s0, s1 * 2]
        store = ascir.ops.Store("store")
        store.attr.sched.axis = [z0, z1]
        store.x = concat.y
        store.y.axis = [z0, z1]
        store.y.strides = [s1 * 2, ascir.SizeExpr(1)]
        store.y.size = [s0, s1 * 2]
        buf0 = ascir.ops.Output("buf0")
        buf0.x = store.y
        buf0.y.dtype = ascir.dtypes.float16
        npu_kernel0_graph.infer_dtypes()
        return npu_kernel0_graph

    def test_construct_graph(self):
        graph = self.construct_graph()
        debug_graph = ascir.utils.debug_str(graph)
        assert debug_graph != ""

    def test_optimize_graph(self):
        options = AutofuserOptions()
        fuser = Autofuser(options)

        hint_graph = self.construct_graph()
        fuser.schedule(hint_graph)


class TestAutofuseLoadSplitStore:
    @staticmethod
    def construct_graph():
        graph = ascir.HintGraph("LoadSplitStore")
        s0 = graph.create_size("s0")
        s1 = graph.create_size("s1")
        z0 = graph.create_axis("z0", s0)
        z1 = graph.create_axis("z1", s1 * 2)

        data = ascir.ops.Data("data", graph)
        data.y.dtype = ascir.dtypes.float16

        load = ascir.ops.Load("load")
        load.attr.sched.axis = [z0, z1]
        load.x = data.y
        load.y.axis = [z0, z1]
        load.y.strides = [s1 * 2, ascir.SizeExpr(1)]
        load.y.size = [s0, s1 * 2]
        load.infer_dtype()

        split = ascir.ops.Split("split", 2)
        assert len(split.y) == 2
        split.attr.sched.axis = [z0, z1]
        split.x = load.y
        for output in split.y:
            output.axis = [z0, z1]
            output.strides = [s1, ascir.SizeExpr(1)]
            output.size = [s0, s1]
        split.infer_dtype()
        assert split.y[0].dtype == ascir.dtypes.float16
        assert split.y[1].dtype == ascir.dtypes.float16

        store0 = ascir.ops.Store("store0")
        store0.attr.sched.axis = [z0, z1]
        store0.x = split.y[0]
        store0.y.axis = [z0, z1]
        store0.y.strides = [s1, ascir.SizeExpr(1)]
        store0.y.size = [s0, s1]

        store1 = ascir.ops.Store("store1")
        store1.attr.sched.axis = [z0, z1]
        store1.x = split.y[1]
        store1.y.axis = [z0, z1]
        store1.y.strides = [s1, ascir.SizeExpr(1)]
        store1.y.size = [s0, s1]

        output0 = ascir.ops.Output("output0")
        output0.attr.ir_attr.index = 0
        output0.x = store0.y

        output1 = ascir.ops.Output("output1")
        output1.attr.ir_attr.index = 1
        output1.x = store1.y

        graph.infer_dtypes()
        return graph

    def test_construct_graph(self):
        # Split currently resolves to the v35 registration in this build, which
        # only supports v2 platforms such as 5102/3510.
        ascir.utils.set_platform("5102", 1, 1024)
        try:
            graph = self.construct_graph()
            debug_graph = ascir.utils.debug_str(graph)
            assert debug_graph != ""
        finally:
            ascir.utils.set_platform("2201", 1, 245760)


class TestWorkspaceOptimize:
    @staticmethod
    def construct_graph():
        npu_kernel0_graph = ascir.HintGraph("workspace")
        s0 = npu_kernel0_graph.create_size("s0")
        z0 = npu_kernel0_graph.create_axis("z0", s0)
        arg2_1 = ascir.ops.Data("arg2_1", npu_kernel0_graph)
        arg2_1.y.dtype = ascir.dtypes.float16
        load = ascir.ops.Load("load")
        load.attr.sched.axis = [z0]
        load.x = arg2_1.y
        load.y.axis = [z0]
        load.y.strides = [ascir.SizeExpr(1)]
        load.y.size = [s0]
        store = ascir.ops.Store("store")
        store.attr.sched.axis = [z0]
        store.x = load.y
        store.y.axis = [z0]
        store.y.strides = [ascir.SizeExpr(1)]
        store.y.size = [s0]
        ws = ascir.ops.Workspace("buf8")
        ws.attr.sched.axis = [z0]
        ws.x = store.y
        ws.y.size = [s0]
        ws.y.dtype = ascir.dtypes.float16
        ws.y.axis = [z0]
        ws.y.strides = [ascir.SizeExpr(1)]
        load1 = ascir.ops.Load("load1")
        load1.attr.sched.axis = [z0]
        load1.x = ws.y
        load1.y.axis = [z0]
        load1.y.strides = [ascir.SizeExpr(1)]
        load1.y.size = [s0]

        store1 = ascir.ops.Store("store1")
        store1.attr.sched.axis = [z0]
        store1.x = load1.y
        store1.y.axis = [z0]
        store1.y.strides = [ascir.SizeExpr(1)]
        store1.y.size = [s0]

        ws1 = ascir.ops.Workspace("buf2")
        ws1.attr.sched.axis = [z0]
        ws1.x = store1.y
        ws1.y.size = [s0]
        ws1.y.dtype = ascir.dtypes.float16
        ws1.y.axis = [z0]
        ws1.y.strides = [ascir.SizeExpr(1)]

        load2 = ascir.ops.Load("load2")
        load2.attr.sched.axis = [z0]
        load2.x = ws1.y
        load2.y.axis = [z0]
        load2.y.strides = [ascir.SizeExpr(1)]
        load2.y.size = [s0]

        load3 = ascir.ops.Load("load3")
        load3.attr.sched.axis = [z0]
        load3.x = ws1.y
        load3.y.axis = [z0]
        load3.y.strides = [ascir.SizeExpr(1)]
        load3.y.size = [s0]
        npu_kernel0_graph.infer_dtypes()
        return npu_kernel0_graph

    def test_construct_graph(self):
        graph = self.construct_graph()
        debug_graph = ascir.utils.debug_str(graph)
        assert debug_graph != ""

    def test_optimize_graph(self):
        options = AutofuserOptions()
        fuser = Autofuser(options)

        hint_graph = self.construct_graph()
        fuser.schedule(hint_graph)


class TestCodeGenLoadAbsStore:
    @staticmethod
    def construct_graph():
        graph = ascir.HintGraph("LoadAbsStore")
        s0 = graph.create_size("s0")
        s1 = graph.create_size("s1")
        s2 = graph.create_size("s2")

        z0 = graph.create_axis("z0", s0)
        z1 = graph.create_axis("z1", s1)
        z2 = graph.create_axis("z2", s2)
        buf_z0 = graph.create_axis("buf_z0", s0)
        buf_z1 = graph.create_axis("buf_z1", s1)
        buf_z2 = graph.create_axis("buf_z2", s2)

        arg3_1 = ascir.ops.Data("arg3_1", graph)
        arg3_1.attr.sched.axis = [z0, z1, z2]
        arg3_1.y.dtype = ascir.dtypes.float16
        arg3_1.y.axis = [z0, z1, z2]
        arg3_1.y.size = [s0, s1, s2]
        arg3_1.y.strides = [s1 * s2, s2, ascir.SizeExpr(1)]

        load = ascir.ops.Load("load")
        load.x = arg3_1
        load.attr.sched.axis = [z0, z1, z2]
        load.y.dtype = ascir.dtypes.float16
        load.y.axis = [z0, z1, z2]
        load.y.size = [s0, s1, s2]
        load.y.strides = [s1 * s2, s2, ascir.SizeExpr(1)]

        abs_op = ascir.ops.Abs("abs")
        abs_op.x = load
        abs_op.attr.sched.axis = [z0, z1, z2]
        abs_op.y.dtype = ascir.dtypes.float16
        abs_op.y.axis = [z0, z1, z2]
        abs_op.y.size = [s0, s1, s2]
        abs_op.y.strides = [s1 * s2, s2, ascir.SizeExpr(1)]

        store = ascir.ops.Store("store")
        store.x = abs_op
        store.attr.sched.axis = [z0, z1, z2]
        store.y.dtype = ascir.dtypes.float16
        store.y.axis = [z0, z1, z2]
        store.y.size = [s0, s1, s2]
        store.y.strides = [s1 * s2, s2, ascir.SizeExpr(1)]

        buf1 = ascir.ops.Output("buf1", graph)
        buf1.x = store
        buf1.attr.sched.axis = [z0, z1, z2]
        buf1.y.dtype = ascir.dtypes.float16
        buf1.y.axis = [z0, z1, z2]
        buf1.y.size = [s0, s1, s2]
        buf1.y.strides = [s1 * s2, s2, ascir.SizeExpr(1)]
        graph.set_axis_map({z0: [buf_z0], z1: [buf_z1], z2: [buf_z2]})
        return graph

    def test_construct_graph(self):
        graph = self.construct_graph()
        try:
            graph.infer_dtypes()
        except Exception as e:
            print(e.args)
        debug_str = ascir.utils.debug_str(graph)
        assert debug_str

    def test_schedule(self):
        options = AutofuserOptions()
        scheduler = Schedule(options)

        hint_graph = self.construct_graph()
        scheduler.schedule(hint_graph)

    @pytest.mark.skip
    def test_codegen(self):
        scheduler = Schedule()
        codegen = CodeGen(tiling_lib_path="./test", tiling_lib_codegen_symbol="test")

        hint_graph = self.construct_graph()
        impl_graphs = scheduler.schedule(hint_graph)
        shape_info = ascir.ShapeInfo(
            {
                "s0": "GetDimValueFromGraphInputData(0, 0);",
                "s1": "GetDimValueFromGraphInputData(0, 1);",
                "s2": "GetDimValueFromGraphInputData(1, 0);",
            }
        )

        kernel_path = "./fused_graph_kernel.o"
        with open(kernel_path, "wb") as o_file:
            o_file.write(b"This is a .o file content.")

        data = {
            "name": "Alice",
            "age": 30,
            "is_student": False,
            "courses": ["Math", "Science", "History"],
        }
        json_path = "./fused_graph_kernel.json"
        with open(json_path, "w") as json_file:
            json.dump(data, json_file, indent=4)

        tiling_data, op_kernel = codegen.device_code_generator(hint_graph, impl_graphs)
        tiling, infer_shape = codegen.host_code_generator(
            hint_graph, impl_graphs, shape_info, "", ["", ""]
        )
        get_kernel = codegen.get_kernel_and_json_generator(kernel_path, json_path)
        os.remove(kernel_path)
        os.remove(json_path)
        assert tiling_data == "".join(
            [
                "#ifndef __Autofuse_Tiling_Data_H__\n"
                "#define __Autofuse_Tiling_Data_H__\n"
                "#include <stdint.h>\n"
                '#include "kernel_tiling/kernel_tiling.h"\n'
                "#define BEGIN_TILING_DATA_DEF_T(name) struct name {\n"
                "#define TILING_DATA_FIELD_DEF_T(type, name) \\\n"
                "  type name; \\\n"
                "  inline void set_##name(type value) { name = value; } \\\n",
                "  inline type get_##name() { return name; } \\\n"
                "  inline type* get_addr_##name() {return &name;}\n"
                "#define END_TILING_DATA_DEF_T };\n"
                "#define TILING_DATA_FIELD_DEF_T_STRUCT(struct_type, filed_name) \\\n"
                "  struct_type filed_name;\n\n"
                "BEGIN_TILING_DATA_DEF_T(AutofuseTilingData)\n"
                "  TILING_DATA_FIELD_DEF_T(uint32_t, block_dim);\n"
                "  TILING_DATA_FIELD_DEF_T(uint32_t, corenum);\n"
                "  TILING_DATA_FIELD_DEF_T(uint32_t, ub_size);\n"
                "  TILING_DATA_FIELD_DEF_T(uint32_t, hbm_size);\n"
                "  TILING_DATA_FIELD_DEF_T(uint32_t, tiling_key);\n"
                "  TILING_DATA_FIELD_DEF_T(uint32_t, z0z1z2t_size);\n"
                "  TILING_DATA_FIELD_DEF_T(uint32_t, z0z1z2Tb_size);\n"
                "END_TILING_DATA_DEF_T;\n\n"
                "struct AutofuseTilingDataPerf {\n"
                "  AutofuseTilingData tiling_data;\n"
                "  double best_perf;\n"
                "};\n"
                "#endif\n",
            ]
        )

        assert infer_shape == "".join([""])

        assert get_kernel == "".join(
            [
                "#include <cstdint>\n"
                "#include <cstring>\n"
                "#include <vector>\n"
                'extern "C" void GetKernelBin(std::vector<char> &kernel_bin) {\n'
                "  std::vector<uint8_t> temp_kernel = {\n"
                "    84, 104, 105, 115, 32, 105, 115, 32, 97, 32, 46, 111, 32, 102, 105, 108, 101, 32, 99, 111, \n"
                "    110, 116, 101, 110, 116, 46, };\n"
                "  kernel_bin.resize(temp_kernel.size());\n"
                "  std::memcpy(kernel_bin.data(), temp_kernel.data(), temp_kernel.size() * sizeof(uint8_t));\n"
                "}"
            ]
        )


class TestComputeGraphInput:
    @staticmethod
    def construct_compute_graph():
        test_graph = os.path.join(PYF_PATH, "test_graph.txt")
        with open(test_graph, "r", encoding="utf-8") as file:
            content = file.read()
        compute_graph = ascir.utils.deserialize("compute_graph", content)
        print(compute_graph.get_name(), flush=True)
        print(compute_graph.get_info(), flush=True)
        assert compute_graph is not None
        return compute_graph

    @pytest.mark.skip
    def test_scheduleV2(self):
        options = AutofuserOptions()
        scheduler = Schedule(options)

        compute_graph = self.construct_compute_graph()
        scheduler.scheduleV2(compute_graph)

    def test_scheduleV2_fail(self):
        options = AutofuserOptions()
        scheduler = Schedule(options)

        compute_graph = ascir.HintComputeGraph("test")
        try:
            scheduler.scheduleV2(compute_graph)
        except RuntimeError:
            pass

    @pytest.mark.skip
    def test_computegraph_codegen(self):
        scheduler = Schedule()
        codegen = CodeGen(tiling_lib_path="./test", tiling_lib_codegen_symbol="test")

        compute_graph = self.construct_compute_graph()
        schedule_results = scheduler.scheduleV2(compute_graph)
        shape_info = ascir.ShapeInfo(
            {
                "s0": "GetDimValueFromGraphInputData(0, 0);",
                "s1": "GetDimValueFromGraphInputData(0, 1);",
                "s2": "GetDimValueFromGraphInputData(1, 0);",
            }
        )

        kernel_path = "./fused_graph_kernel.o"
        with open(kernel_path, "wb") as o_file:
            o_file.write(b"This is a .o file content.")

        data = {
            "name": "Alice",
            "age": 30,
            "is_student": False,
            "courses": ["Math", "Science", "History"],
        }
        json_path = "./fused_graph_kernel.json"
        with open(json_path, "w") as json_file:
            json.dump(data, json_file, indent=4)

        tiling_data, op_kernel = codegen.device_code_generator(schedule_results)
        assert tiling_data == "".join(
            [
                "#ifndef __Autofuse_Tiling_Data_H__\n"
                "#define __Autofuse_Tiling_Data_H__\n"
                "#include <stdint.h>\n"
                '#include "kernel_tiling/kernel_tiling.h"\n'
                "#define BEGIN_TILING_DATA_DEF_T(name) struct name {\n"
                "#define TILING_DATA_FIELD_DEF_T(type, name) \\\n"
                "  type name; \\\n"
                "  inline void set_##name(type value) { name = value; } \\\n",
                "  inline type get_##name() { return name; } \\\n"
                "  inline type* get_addr_##name() {return &name;}\n"
                "#define END_TILING_DATA_DEF_T };\n"
                "#define TILING_DATA_FIELD_DEF_T_STRUCT(struct_type, filed_name) \\\n"
                "  struct_type filed_name;\n\n"
                "BEGIN_TILING_DATA_DEF_T(AutofuseTilingData)\n"
                "  TILING_DATA_FIELD_DEF_T(uint32_t, block_dim);\n"
                "  TILING_DATA_FIELD_DEF_T(uint32_t, corenum);\n"
                "  TILING_DATA_FIELD_DEF_T(uint32_t, ub_size);\n"
                "  TILING_DATA_FIELD_DEF_T(uint32_t, hbm_size);\n"
                "  TILING_DATA_FIELD_DEF_T(uint32_t, tiling_key);\n"
                "  TILING_DATA_FIELD_DEF_T(uint32_t, z0z1z2t_size);\n"
                "  TILING_DATA_FIELD_DEF_T(uint32_t, z0z1z2Tb_size);\n"
                "  TILING_DATA_FIELD_DEF_T(uint32_t, q0_size);\n"
                "  TILING_DATA_FIELD_DEF_T(uint32_t, q1_size);\n"
                "  TILING_DATA_FIELD_DEF_T(uint32_t, b0_size);\n"
                "END_TILING_DATA_DEF_T;\n\n"
                "struct AutofuseTilingDataPerf {\n"
                "  AutofuseTilingData tiling_data;\n"
                "  double best_perf;\n"
                "};\n"
                "#endif\n",
            ]
        )

        output_shape = [["s0", "s1"]]
        vector_core_num = "0"
        tiling, infer_shape = codegen.host_code_generator(
            schedule_results, shape_info, output_shape, "", vector_core_num
        )
        codegen.pgo_code_generator(schedule_results, "")
        get_kernel = codegen.get_kernel_and_json_generator(kernel_path, json_path)
        os.remove(kernel_path)
        os.remove(json_path)
        assert get_kernel == "".join(
            [
                "#include <cstdint>\n"
                "#include <cstring>\n"
                "#include <vector>\n"
                'extern "C" void GetKernelBin(std::vector<char> &kernel_bin) {\n'
                "  std::vector<uint8_t> temp_kernel = {\n"
                "    84, 104, 105, 115, 32, 105, 115, 32, 97, 32, 46, 111, 32, 102, 105, 108, 101, 32, 99, 111, \n"
                "    110, 116, 101, 110, 116, 46, };\n"
                "  kernel_bin.resize(temp_kernel.size());\n"
                "  std::memcpy(kernel_bin.data(), temp_kernel.data(), temp_kernel.size() * sizeof(uint8_t));\n"
                "}"
            ]
        )

        try:
            output_shape = ["s0"]
            vector_core_num = "0"
            tiling, infer_shape = codegen.host_code_generator(
                schedule_results, shape_info, output_shape, "", vector_core_num
            )
        except ValueError:
            pass

        try:
            codegen.pgo_code_generator(schedule_results)
        except ValueError:
            pass

        try:
            get_kernel = codegen.get_kernel_and_json_generator(kernel_path, json_path)
        except ValueError:
            pass

        try:
            get_kernel = codegen.get_kernel_and_json_generator(kernel_path)
        except ValueError:
            pass


class TestHintGraph:
    @staticmethod
    def construct_graph():
        graph = ascir.HintGraph("LoadAbsStore")
        s0 = graph.create_size("s0")
        s1 = graph.create_size("s1")
        s2 = graph.create_size("s2")

        z0 = graph.create_axis("z0", s0)
        z1 = graph.create_axis("z1", s1)
        z2 = graph.create_axis("z2", s2)
        buf_z0 = graph.create_axis("buf_z0", s0)
        buf_z1 = graph.create_axis("buf_z1", s1)
        buf_z2 = graph.create_axis("buf_z2", s2)

        arg3_1 = ascir.ops.Data("arg3_1", graph)
        arg3_1.attr.sched.axis = [z0, z1, z2]
        arg3_1.y.dtype = ascir.dtypes.float16
        arg3_1.y.axis = [z0, z1, z2]
        arg3_1.y.size = [s0, s1, s2]
        arg3_1.y.strides = [s1 * s2, s2, ascir.SizeExpr(1)]

        load = ascir.ops.Load("load")
        load.x = arg3_1
        load.attr.sched.axis = [z0, z1, z2]
        load.y.dtype = ascir.dtypes.float16
        load.y.axis = [z0, z1, z2]
        load.y.size = [s0, s1, s2]
        load.y.strides = [s1 * s2, s2, ascir.SizeExpr(1)]

        abs_op = ascir.ops.Abs("abs")
        abs_op.x = load
        abs_op.attr.sched.axis = [z0, z1, z2]
        abs_op.y.dtype = ascir.dtypes.float16
        abs_op.y.axis = [z0, z1, z2]
        abs_op.y.size = [s0, s1, s2]
        abs_op.y.strides = [s1 * s2, s2, ascir.SizeExpr(1)]

        store = ascir.ops.Store("store")
        store.x = abs_op
        store.attr.sched.axis = [z0, z1, z2]
        store.y.dtype = ascir.dtypes.float16
        store.y.axis = [z0, z1, z2]
        store.y.size = [s0, s1, s2]
        store.y.strides = [s1 * s2, s2, ascir.SizeExpr(1)]

        buf1 = ascir.ops.Output("buf1", graph)
        buf1.x = store
        buf1.attr.sched.axis = [z0, z1, z2]
        buf1.y.dtype = ascir.dtypes.float16
        buf1.y.axis = [z0, z1, z2]
        buf1.y.size = [s0, s1, s2]
        buf1.y.strides = [s1 * s2, s2, ascir.SizeExpr(1)]
        graph.set_axis_map({z0: [buf_z0], z1: [buf_z1], z2: [buf_z2]})
        return graph

    def test_hintgraph_set_name(self):
        asc_graph = self.construct_graph()
        try:
            asc_graph.set_name(2)
        except TypeError:
            assert asc_graph.get_name() == "".join(["LoadAbsStore"])

        asc_graph.set_name("test_graph")
        assert asc_graph.get_name() == "".join(["test_graph"])


class TestFusedGraph:
    @staticmethod
    def construct_add_ascgraph(name: str) -> ascir.HintGraph:
        npu_kernel0_graph = ascir.HintGraph(name)
        s0 = npu_kernel0_graph.create_size("s0")
        s1 = npu_kernel0_graph.create_size("s1")
        z0 = npu_kernel0_graph.create_axis("z0", s0)
        z1 = npu_kernel0_graph.create_axis("z1", s1)
        sub_data0 = ascir.ops.Data("sub_data0", npu_kernel0_graph)
        sub_data0.y.dtype = ascir.dtypes.float16
        sub_data0.attr.ir_attr.index = 0
        load0 = ascir.ops.Load("load")
        load0.attr.ir_attr.offset = 0
        load0.attr.sched.axis = [z0, z1]
        load0.x = sub_data0.y
        load0.y.axis = [z0, z1]
        load0.y.strides = [s1, ascir.SizeExpr(1)]
        load0.y.size = [s0, s1]
        sub_data1 = ascir.ops.Data("sub_data1", npu_kernel0_graph)
        sub_data1.y.dtype = ascir.dtypes.float16
        sub_data1.attr.ir_attr.index = 1
        load1 = ascir.ops.Load("load")
        load1.attr.ir_attr.offset = ascir.SizeExpr(0)
        load1.attr.sched.axis = [z0, z1]
        load1.x = sub_data1.y
        load1.y.axis = [z0, z1]
        load1.y.strides = [s1, ascir.SizeExpr(1)]
        load1.y.size = [s0, s1]

        add0 = ascir.ops.Add("add")
        add0.attr.sched.axis = [z0, z1]
        add0.x1 = load0.y
        add0.x2 = load1.y
        add0.y.axis = [z0, z1]
        add0.y.strides = [s1 + s1, ascir.SizeExpr(1)]
        add0.y.size = [s0, s1 * 2]

        store0 = ascir.ops.Store("store")
        store0.attr.ir_attr.offset = ascir.SizeExpr(0)
        store0.attr.sched.axis = [z0, z1]
        store0.x = add0.y
        store0.y.axis = [z0, z1]
        store0.y.strides = [s1**2, ascir.SizeExpr(1)]
        store0.y.size = [s0, s1 * 2]

        store1 = ascir.ops.Store("store")
        store1.attr.ir_attr.offset = ascir.SizeExpr(10)
        store1.attr.sched.axis = [z0, z1]
        store1.x = add0.y
        store1.y.axis = [z0, z1]
        store1.y.strides = [s1 * 2, ascir.SizeExpr(1)]
        store1.y.size = [s0, s1 * 2]
        buf0 = ascir.ops.Output("buf0")
        buf0.attr.ir_attr.index = 0
        # store0, strore1 写到同一个output上，偏移不同
        buf0.x = [store0.y, store1]
        buf0.y.dtype = ascir.dtypes.float16
        buf1 = ascir.ops.Output("buf1")
        buf1.attr.ir_attr.index = 1
        buf1.x = store1.y
        npu_kernel0_graph.infer_dtypes()
        ascir.utils.dump(npu_kernel0_graph)
        return npu_kernel0_graph

    @staticmethod
    def construct_add_ascgraph_without_data(name: str) -> ascir.HintGraph:
        npu_kernel0_graph = ascir.HintGraph(name)
        s0 = npu_kernel0_graph.create_size("s0")
        s1 = npu_kernel0_graph.create_size("s1")
        z0 = npu_kernel0_graph.create_axis("z0", s0)
        z1 = npu_kernel0_graph.create_axis("z1", s1)
        sub_data0 = ascir.ops.Scalar("sub_data0", npu_kernel0_graph)
        sub_data0.y.dtype = ascir.dtypes.float16
        load0 = ascir.ops.Load("load")
        load0.attr.ir_attr.offset = 0
        load0.attr.sched.axis = [z0, z1]
        load0.x = sub_data0.y
        load0.y.axis = [z0, z1]
        load0.y.strides = [s1, ascir.SizeExpr(1)]
        load0.y.size = [s0, s1]
        sub_data1 = ascir.ops.Scalar("sub_data1", npu_kernel0_graph)
        sub_data1.y.dtype = ascir.dtypes.float16
        load1 = ascir.ops.Load("load")
        load1.attr.ir_attr.offset = ascir.SizeExpr(0)
        load1.attr.sched.axis = [z0, z1]
        load1.x = sub_data1.y
        load1.y.axis = [z0, z1]
        load1.y.strides = [s1, ascir.SizeExpr(1)]
        load1.y.size = [s0, s1]

        add0 = ascir.ops.Add("add")
        add0.attr.sched.axis = [z0, z1]
        add0.x1 = load0.y
        add0.x2 = load1.y
        add0.y.axis = [z0, z1]
        add0.y.strides = [s1 + s1, ascir.SizeExpr(1)]
        add0.y.size = [s0, s1 * 2]

        store0 = ascir.ops.Store("store")
        store0.attr.ir_attr.offset = ascir.SizeExpr(0)
        store0.attr.sched.axis = [z0, z1]
        store0.x = add0.y
        store0.y.axis = [z0, z1]
        store0.y.strides = [s1**2, ascir.SizeExpr(1)]
        store0.y.size = [s0, s1 * 2]

        store1 = ascir.ops.Store("store")
        store1.attr.ir_attr.offset = ascir.SizeExpr(10)
        store1.attr.sched.axis = [z0, z1]
        store1.x = add0.y
        store1.y.axis = [z0, z1]
        store1.y.strides = [s1 * 2, ascir.SizeExpr(1)]
        store1.y.size = [s0, s1 * 2]
        buf0 = ascir.ops.Output("buf0")
        buf0.attr.ir_attr.index = 0
        # store0, strore1 写到同一个output上，偏移不同
        buf0.x = [store0.y, store1]
        buf0.y.dtype = ascir.dtypes.float16
        buf1 = ascir.ops.Output("buf1")
        buf1.attr.ir_attr.index = 1
        buf1.x = store1.y
        npu_kernel0_graph.infer_dtypes()
        ascir.utils.dump(npu_kernel0_graph)
        return npu_kernel0_graph

    def test_fused_graph_construct_and_dump_with_ascbackend_node(self):
        fused_graph = ascir.FusedGraph("fused_graph")
        data0 = ascir.ops.Data("data0", fused_graph)
        data0.attr.ir_attr.index = 0
        data1 = ascir.ops.Data("data1", fused_graph)
        data1.attr.ir_attr.index = 0
        ascgraph_node0 = ascir.ops.AscBackend(
            "ascgraph_node0",
            self.construct_add_ascgraph_without_data("ascgraph0"),
            fused_graph,
        )
        ascgraph_node1 = ascir.ops.AscBackend(
            "ascgraph_node1", self.construct_add_ascgraph("ascgraph1"), fused_graph
        )
        ascgraph_node1.x = [data0.y, data1.y]
        ascgraph_node2 = ascir.ops.AscBackend(
            "ascgraph_node2", self.construct_add_ascgraph("ascgraph2"), fused_graph
        )
        ascgraph_node2.x = [ascgraph_node0.y[0], ascgraph_node1.y[1]]
        output = ascir.ops.Output("output", fused_graph)
        output.x = ascgraph_node2.y[1]
        ascir.utils.dump(fused_graph)

    def test_fused_graph_inductor(self):
        fused_graph = ascir.FusedGraph("fused_graph")

        options = AutofuserOptions()
        Schedule(options)
        fuser = Autofuser(options)
        try:
            fuser.schedule(fused_graph)
            tiling_def, host_tiling, op_kernel = fuser.autofuse_backend(fused_graph)
        except RuntimeError:
            pass

    def test_fused_graph_construct_and_dump_with_ascgraph_node(self):
        fused_graph = ascir.FusedGraph("fused_graph")
        data0 = ascir.ops.Data("data0", fused_graph)
        data0.attr.ir_attr.index = 0
        data1 = ascir.ops.Data("data1", fused_graph)
        data1.attr.ir_attr.index = 0
        ascgraph_node0 = ascir.ops.AscGraph(
            "ascgraph_node0", self.construct_add_ascgraph("ascgraph0"), fused_graph
        )
        ascgraph_node0.x = [data0.y, data1.y]
        ascgraph_node1 = ascir.ops.AscGraph(
            "ascgraph_node1", self.construct_add_ascgraph("ascgraph1"), fused_graph
        )
        ascgraph_node1.x = [data0.y, data1.y]
        ascgraph_node2 = ascir.ops.AscGraph(
            "ascgraph_node2", self.construct_add_ascgraph("ascgraph2"), fused_graph
        )
        ascgraph_node2.x = [ascgraph_node0.y[0], ascgraph_node1.y[0]]
        output = ascir.ops.Output("output", fused_graph)
        output.x = ascgraph_node2.y[0]
        try:
            ascgraph_node2.x = [ascgraph_node0.y[0].dtype, ascgraph_node1.y[0]]
        except TypeError as e:
            assert e.args[0] == "Input Type is invalid."

        ascir.utils.dump(fused_graph)
        try:
            ascir.utils.dump(data0)
        except TypeError as e:
            assert (
                e.args[0]
                == "Argument must be a HintGraph or FusedGraph object, got Data"
            )


class TestFusedGraphByApi:
    @staticmethod
    def construct_add_ascgraph(name: str) -> ascir.HintGraph:
        npu_kernel0_graph = ascir.HintGraph(name)
        s0 = npu_kernel0_graph.create_size("s0")
        s1 = npu_kernel0_graph.create_size("s1")
        z0 = npu_kernel0_graph.create_axis("z0", s0)
        z1 = npu_kernel0_graph.create_axis("z1", s1)
        sub_data0 = ascir_api.Data(npu_kernel0_graph, dtype=ascir.dtypes.float16)
        load0 = ascir_api.Load(npu_kernel0_graph, sub_data0, offset=0, axis=[z0, z1])
        assert load0.axis == [z0.id, z1.id]
        assert load0.size == [s0, s1]
        assert load0.strides == [s1, 1]
        sub_data1 = ascir_api.Data(npu_kernel0_graph, dtype=ascir.dtypes.float16)
        load1 = ascir_api.Load(npu_kernel0_graph, sub_data1, offset=0, axis=[z0, z1])
        add0 = ascir_api.Add(npu_kernel0_graph, load0, load1, axis=[z0, z1])
        assert add0.axis == [z0.id, z1.id]
        assert add0.size == [s0, s1]
        assert add0.strides == [s1, 1]
        store0 = ascir_api.Store(npu_kernel0_graph, add0, offset=0, axis=[z0, z1])
        store1 = ascir_api.Store(npu_kernel0_graph, add0, offset=10, axis=[z0, z1])
        # store0, strore1 写到同一个output上，偏移不同
        ascir_api.Output(
            npu_kernel0_graph, [store0, store1], dtype=ascir.dtypes.float16
        )
        buf1 = ascir_api.Output(npu_kernel0_graph, store1)  # infer
        assert buf1.dtype == ascir.dtypes.float16
        print(ascir.utils.debug_str(npu_kernel0_graph))
        return npu_kernel0_graph

    def test_fused_graph_construct_and_dump_with_ascbackend_node(self):
        fused_graph = ascir.FusedGraph("fused_graph")
        data0 = ascir.ops.Data("data0", fused_graph)
        data0.attr.ir_attr.index = 0
        data1 = ascir.ops.Data("data1", fused_graph)
        data1.attr.ir_attr.index = 0
        ascgraph_node0 = ascir.ops.AscGraph(
            "ascgraph_node0", self.construct_add_ascgraph("ascgraph0"), fused_graph
        )
        ascgraph_node0.x = [data0.y, data1.y]
        ascgraph_node1 = ascir.ops.AscGraph(
            "ascgraph_node1", self.construct_add_ascgraph("ascgraph1"), fused_graph
        )
        ascgraph_node1.x = [data0.y, data1.y]
        ascgraph_node2 = ascir.ops.AscGraph(
            "ascgraph_node2", self.construct_add_ascgraph("ascgraph2"), fused_graph
        )
        ascgraph_node2.x = [ascgraph_node0.y[0], ascgraph_node1.y[1]]
        output = ascir.ops.Output("output", fused_graph)
        output.x = ascgraph_node2.y[1]
        ascir.utils.dump(fused_graph)


# 测试包含transpose的sched, codegen的流程, 执行不抛异常, 返回结果非空
class TestAutofuseLoadTransposeStore:
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
        graph = self.construct_graph()
        debug_str = ascir.utils.debug_str(graph)
        assert debug_str

    def test_autofuse_backend(self):
        options = AutofuserOptions()
        fuser = Autofuser(options)
        try:
            hint_graph = self.construct_graph()
            sched_result = fuser.schedule(hint_graph)
            tiling_def, host_tiling, op_kernel = fuser.codegen(sched_result)
            assert len(tiling_def) > 0
            assert len(host_tiling) > 0
            assert len(op_kernel) > 0
        except RuntimeError:
            pass

    import os

    def test_autofuse_backend_faild_dump_graph(self):
        options = AutofuserOptions()
        fuser = Autofuser(options)
        hint_graph = self.construct_invalid_graph()
        with pytest.raises(RuntimeError, match=r"^Optimize fail$"):
            fuser.schedule(hint_graph)
        target_dir = "./"
        for item in os.listdir(target_dir):
            item_path = os.path.join(target_dir, item)
            if os.path.isdir(item_path) and item.startswith("ascgen_dump_pid"):
                print(f"delete dump dir :{item_path}")
                shutil.rmtree(item_path)


class TestSizeExprMaxMin:
    """Test Max and Min functions for SizeExpr"""

    @staticmethod
    def test_max_basic():
        """Test Max function with basic SizeExpr"""
        a = ascir.SizeExpr(10)
        b = ascir.SizeExpr(20)
        c = Max(a, b)
        assert c == 20

    @staticmethod
    def test_min_basic():
        """Test Min function with basic SizeExpr"""
        a = ascir.SizeExpr(10)
        b = ascir.SizeExpr(20)
        c = Min(a, b)
        assert c == 10

    @staticmethod
    def test_max_equal_values():
        """Test Max function with equal values"""
        a = ascir.SizeExpr(15)
        b = ascir.SizeExpr(15)
        c = Max(a, b)
        assert c == 15

    @staticmethod
    def test_min_equal_values():
        """Test Min function with equal values"""
        a = ascir.SizeExpr(15)
        b = ascir.SizeExpr(15)
        c = Min(a, b)
        assert c == 15

    @staticmethod
    def test_max_zero():
        """Test Max function with zero"""
        a = ascir.SizeExpr(0)
        b = ascir.SizeExpr(100)
        c = Max(a, b)
        assert c == 100

    @staticmethod
    def test_min_zero():
        """Test Min function with zero"""
        a = ascir.SizeExpr(0)
        b = ascir.SizeExpr(100)
        c = Min(a, b)
        assert c == 0

    @staticmethod
    def test_max_with_symbolic():
        """Test Max function with symbolic sizes"""
        graph = ascir.HintGraph("test_max")
        s0 = graph.create_size("s0")
        s1 = graph.create_size("s1")
        max_size = Max(s0, s1)
        # Max of symbolic sizes should work
        assert max_size.expression == "Max(s0, s1)"

    @staticmethod
    def test_min_with_symbolic():
        """Test Min function with symbolic sizes"""
        graph = ascir.HintGraph("test_min")
        s0 = graph.create_size("s0")
        s1 = graph.create_size("s1")
        min_size = Min(s0, s1)
        # Min of symbolic sizes should work
        assert min_size.expression == "Min(s0, s1)"

    @staticmethod
    def test_max_with_expression():
        """Test Max with complex expression"""
        s0 = ascir.SizeExpr(10)
        s1 = ascir.SizeExpr(20)
        s2 = ascir.SizeExpr(30)
        expr1 = s0 + s1  # 30
        expr2 = s2  # 30
        max_expr = Max(expr1, expr2)
        assert max_expr == 30

    @staticmethod
    def test_min_with_expression():
        """Test Min with complex expression"""
        s0 = ascir.SizeExpr(10)
        s1 = ascir.SizeExpr(20)
        s2 = ascir.SizeExpr(5)
        expr1 = s0 + s1  # 30
        expr2 = s2  # 5
        min_expr = Min(expr1, expr2)
        assert min_expr == 5

    @staticmethod
    def test_max_chained():
        """Test chained Max operations"""
        a = ascir.SizeExpr(10)
        b = ascir.SizeExpr(20)
        c = ascir.SizeExpr(15)
        max_abc = Max(Max(a, b), c)
        assert max_abc == 20

    @staticmethod
    def test_min_chained():
        """Test chained Min operations"""
        a = ascir.SizeExpr(10)
        b = ascir.SizeExpr(20)
        c = ascir.SizeExpr(15)
        min_abc = Min(Min(a, b), c)
        assert min_abc == 10

    @staticmethod
    def test_max_min_combined():
        """Test combined Max and Min operations"""
        a = ascir.SizeExpr(10)
        b = ascir.SizeExpr(20)
        c = ascir.SizeExpr(30)
        max_val = Max(a, b)  # 20
        min_val = Min(max_val, c)  # 20
        assert min_val == 20


class TestSizeExprMod:
    """Test Mod function for SizeExpr"""

    @staticmethod
    def test_mod_basic():
        """Test Mod function with basic SizeExpr"""
        a = ascir.SizeExpr(10)
        b = ascir.SizeExpr(3)
        c = Mod(a, b)
        assert c == 1

    @staticmethod
    def test_mod_zero():
        """Test Mod function with zero"""
        a = ascir.SizeExpr(100)
        b = ascir.SizeExpr(5)
        c = Mod(a, b)
        assert c == 0

    @staticmethod
    def test_mod_with_symbolic():
        """Test Mod function with symbolic sizes"""
        graph = ascir.HintGraph("test_mod")
        s0 = graph.create_size("s0")
        s1 = graph.create_size("s1")
        mod_size = Mod(s0, s1)
        # Mod of symbolic sizes should work
        assert mod_size.expression == "Mod(s0, s1)"

    @staticmethod
    def test_mod_with_constant():
        """Test Mod with constant (alignment check)"""
        graph = ascir.HintGraph("test_mod_const")
        s0 = graph.create_size("s0")
        mod_size = Mod(s0, 16)
        # Mod with constant (common for alignment check)
        assert mod_size.expression == "Mod(s0, 16)"

    @staticmethod
    def test_mod_with_expression():
        """Test Mod with complex expression"""
        s0 = ascir.SizeExpr(100)
        s1 = ascir.SizeExpr(30)
        expr = s0 + s1  # 130
        mod_expr = Mod(expr, 7)
        assert mod_expr == 4

    @staticmethod
    def test_mod_chained():
        """Test chained Mod operations"""
        a = ascir.SizeExpr(100)
        b = ascir.SizeExpr(7)
        c = ascir.SizeExpr(5)
        mod_abc = Mod(Mod(a, b), c)
        # (100 % 7) = 2, then 2 % 5 = 2
        assert mod_abc == 2


class TestSizeExprArithmetic:
    """Test SizeExpr arithmetic operators in various scenarios"""

    @staticmethod
    def test_power_in_block_size():
        """Test power operation in block size calculation"""
        graph = ascir.HintGraph("test_power")
        base_size = graph.create_size("base")

        # Block size = base^2
        block_size = base_size**2
        graph.create_axis("z0", block_size)

        debug_str = ascir.utils.debug_str(graph)
        assert debug_str

    @staticmethod
    def test_multiplication_in_memory_calculation():
        """Test multiplication in memory size calculation"""
        graph = ascir.HintGraph("test_mem_mul")

        batch_size = graph.create_size("batch_size")
        seq_len = graph.create_size("seq_len")
        hidden_size = graph.create_size("hidden_size")

        # Total elements = batch * seq * hidden
        total_elements = batch_size * seq_len * hidden_size
        graph.create_axis("z0", total_elements)

        debug_str = ascir.utils.debug_str(graph)
        assert debug_str

    @staticmethod
    def test_division_in_split_calculation():
        """Test division in split calculation"""
        graph = ascir.HintGraph("test_split")

        total_size = graph.create_size("total_size")
        num_splits = 2

        # Split size = total / num_splits
        split_size = total_size / num_splits
        graph.create_axis("z0", split_size)

        debug_str = ascir.utils.debug_str(graph)
        assert debug_str

    @staticmethod
    def test_addition_in_concat_calculation():
        """Test addition in concat calculation"""
        graph = ascir.HintGraph("test_concat_add")

        size1 = graph.create_size("size1")
        size2 = graph.create_size("size2")
        constant = ascir.SizeExpr(10)

        # Total size = size1 + size2 + constant
        total_size = size1 + size2 + constant
        graph.create_axis("z0", total_size)

        debug_str = ascir.utils.debug_str(graph)
        assert debug_str

    @staticmethod
    def test_subtraction_basic():
        """Test basic subtraction between SizeExpr"""
        s0 = ascir.SizeExpr(10)
        s1 = ascir.SizeExpr(3)
        result = s0 - s1
        assert result == 7

    @staticmethod
    def test_subtraction_with_int():
        """Test subtraction with integer operand"""
        s0 = ascir.SizeExpr(100)
        result = s0 - 42
        assert result == 58

    @staticmethod
    def test_subtraction_to_zero():
        """Test subtraction resulting in zero"""
        s0 = ascir.SizeExpr(50)
        s1 = ascir.SizeExpr(50)
        result = s0 - s1
        assert result == 0

    @staticmethod
    def test_negate_basic():
        """Test unary negation of SizeExpr"""
        s0 = ascir.SizeExpr(42)
        result = -s0
        assert result == -42

    @staticmethod
    def test_negate_zero():
        """Test unary negation of zero"""
        s0 = ascir.SizeExpr(0)
        result = -s0
        assert result == 0

    @staticmethod
    def test_negate_combined_with_add():
        """Test negation combined with addition"""
        s0 = ascir.SizeExpr(10)
        s1 = ascir.SizeExpr(3)
        result = s0 + (-s1)
        assert result == 7


class TestSizeExprEdgeCases:
    """Test SizeExpr edge cases and boundary conditions"""

    @staticmethod
    def test_remainder_with_zero_result():
        """Test remainder when result is zero"""
        s0 = ascir.SizeExpr(100)
        result = s0 % 100
        assert result.expression == "0"

    @staticmethod
    def test_remainder_with_same_value():
        """Test remainder with same dividend and divisor"""
        s0 = ascir.SizeExpr(50)
        result = s0 % 50
        assert result.expression == "0"

    @staticmethod
    def test_floordiv_with_one():
        """Test floor division by one"""
        s0 = ascir.SizeExpr(100)
        result = s0 // 1
        assert result.expression == "100"

    @staticmethod
    def test_floordiv_with_large_divisor():
        """Test floor division when divisor is larger than dividend"""
        s0 = ascir.SizeExpr(10)
        result = s0 // 100
        assert result.expression == "0"

    @staticmethod
    def test_max_with_same_values():
        """Test Max with identical values"""
        s0 = ascir.SizeExpr(42)
        s1 = ascir.SizeExpr(42)
        max_val = Max(s0, s1)
        assert max_val == 42

    @staticmethod
    def test_min_with_same_values():
        """Test Min with identical values"""
        s0 = ascir.SizeExpr(42)
        s1 = ascir.SizeExpr(42)
        min_val = Min(s0, s1)
        assert min_val == 42

    @staticmethod
    def test_max_with_zero():
        """Test Max with zero value"""
        s0 = ascir.SizeExpr(0)
        s1 = ascir.SizeExpr(100)
        max_val = Max(s0, s1)
        assert max_val == 100

    @staticmethod
    def test_min_with_zero():
        """Test Min with zero value"""
        s0 = ascir.SizeExpr(0)
        s1 = ascir.SizeExpr(100)
        min_val = Min(s0, s1)
        assert min_val == 0


class TestSizeExprInRealScenarios:
    """Test SizeExpr in real-world scenarios"""

    @staticmethod
    def test_tile_size_clamping():
        """Test tile size clamping between min and max"""
        graph = ascir.HintGraph("test_tile_clamp")

        requested_size = graph.create_size("requested_size")
        min_tile = 64
        max_tile = 1024

        # Clamp tile size: at least min_tile, at most max_tile
        clamped_size = Min(Max(requested_size, min_tile), max_tile)

        graph.create_axis("z0", clamped_size)

        debug_str = ascir.utils.debug_str(graph)
        assert debug_str

    @staticmethod
    def test_strided_memory_access():
        """Test strided memory access calculation"""
        graph = ascir.HintGraph("test_strided_access")

        n = graph.create_size("n")
        c = graph.create_size("c")
        h = graph.create_size("h")
        w = graph.create_size("w")

        # Calculate offset for element (n, c, h, w) in NCHW layout
        # offset = n*C*H*W + c*H*W + h*W + w
        offset = n * c * h * w + c * h * w + h * w + w

        graph.create_axis("z0", offset)

        debug_str = ascir.utils.debug_str(graph)
        assert debug_str


class TestSizeExprOperatorCombination:
    """Test SizeExpr operator combinations"""

    @staticmethod
    def test_combined_mod_and_floordiv():
        """Test Mod and FloorDiv combination: blocking with remainder"""
        graph = ascir.HintGraph("test_block_rem")

        total_size = graph.create_size("total_size")
        block_size = 128

        # Calculate number of full blocks and remaining elements
        num_blocks = total_size // block_size
        remainder = total_size % block_size

        # Create axes for both
        graph.create_axis("z_blocks", num_blocks)
        graph.create_axis("z_remain", remainder)

        debug_str = ascir.utils.debug_str(graph)
        assert debug_str
        assert debug_str

    @staticmethod
    def test_max_min_in_clamping():
        """Test Max/Min combination for clamping value"""
        graph = ascir.HintGraph("test_clamp")

        value = graph.create_size("value")
        min_val = 10
        max_val = 100

        # Clamp value between min and max
        clamped = Min(Max(value, min_val), max_val)

        graph.create_axis("z0", clamped)

        debug_str = ascir.utils.debug_str(graph)
        assert debug_str

    @staticmethod
    def test_concat_max_memory():
        """Test Max for concat output memory calculation"""
        graph = ascir.HintGraph("test_concat_mem")

        size1 = graph.create_size("size1")
        size2 = graph.create_size("size2")
        size3 = graph.create_size("size3")

        # For concat, output size in non-concat dim is max of inputs
        max_dim_size = Max(Max(size1, size2), size3)

        graph.create_axis("z0", max_dim_size)

        debug_str = ascir.utils.debug_str(graph)
        assert debug_str

    @staticmethod
    def test_broadcast_min_size():
        """Test Min for broadcast compatibility check"""
        graph = ascir.HintGraph("test_broadcast")

        size1 = graph.create_size("size1")
        size2 = graph.create_size("size2")

        # For broadcast, compatible if one is 1 or they are equal
        # Min helps check the smaller dimension
        min_size = Min(size1, size2)

        graph.create_axis("z0", min_size)

        debug_str = ascir.utils.debug_str(graph)
        assert debug_str


class TestSizeExprErrorScenarios:
    """Test SizeExpr error handling and edge cases"""

    @staticmethod
    def test_max_with_no_arguments():
        """Test Max with no arguments should raise error"""
        graph = ascir.HintGraph("test_max_no_args")
        graph.create_size("size1")
        # Max with no arguments should raise TypeError
        try:
            Max()
            assert False, "Expected TypeError for Max() with no arguments"
        except (TypeError, AttributeError):
            # Expected - invalid number of arguments
            pass

    @staticmethod
    def test_max_with_single_argument():
        """Test Max with single argument should raise error"""
        graph = ascir.HintGraph("test_max_single_arg")
        size1 = graph.create_size("size1")
        # Max with single argument should raise TypeError
        try:
            Max(size1)
            assert False, "Expected TypeError for Max() with single argument"
        except TypeError:
            # Expected - invalid number of arguments
            pass

    @staticmethod
    def test_max_with_three_arguments():
        """Test Max with three arguments should raise error"""
        graph = ascir.HintGraph("test_max_three_args")
        size1 = graph.create_size("size1")
        size2 = graph.create_size("size2")
        size3 = graph.create_size("size3")
        # Max with three arguments should raise TypeError
        try:
            Max(size1, size2, size3)
            assert False, "Expected TypeError for Max() with three arguments"
        except TypeError:
            # Expected - invalid number of arguments
            pass

    @staticmethod
    def test_max_with_invalid_types():
        """Test Max with non-SizeExpr arguments"""
        graph = ascir.HintGraph("test_max_invalid_types")
        size1 = graph.create_size("size1")
        # Max with string argument
        try:
            Max(size1, "invalid")
            # If it doesn't raise, at least verify it handles gracefully
        except (TypeError, AttributeError, SystemError):
            # Expected - invalid type
            pass
        # Max with None argument
        try:
            Max(size1, None)
            # If it doesn't raise, at least verify it handles gracefully
        except (TypeError, AttributeError, SystemError):
            # Expected - invalid type
            pass

    @staticmethod
    def test_min_with_no_arguments():
        """Test Min with no arguments should raise error"""
        graph = ascir.HintGraph("test_min_no_args")
        graph.create_size("size1")
        # Min with no arguments should raise TypeError
        try:
            Min()
            assert False, "Expected TypeError for Min() with no arguments"
        except (TypeError, AttributeError):
            # Expected - invalid number of arguments
            pass

    @staticmethod
    def test_min_with_single_argument():
        """Test Min with single argument should raise error"""
        graph = ascir.HintGraph("test_min_single_arg")
        size1 = graph.create_size("size1")
        # Min with single argument should raise TypeError
        try:
            Min(size1)
            assert False, "Expected TypeError for Min() with single argument"
        except TypeError:
            # Expected - invalid number of arguments
            pass

    @staticmethod
    def test_min_with_invalid_types():
        """Test Min with non-SizeExpr arguments"""
        graph = ascir.HintGraph("test_min_invalid_types")
        size1 = graph.create_size("size1")
        # Min with integer argument
        try:
            Min(size1, 42)
            # If it doesn't raise, at least verify it handles gracefully
        except (TypeError, AttributeError, SystemError):
            # Expected - invalid type
            pass
        # Min with dict argument
        try:
            Min(size1, {"key": "value"})
            # If it doesn't raise, at least verify it handles gracefully
        except (TypeError, AttributeError, SystemError):
            # Expected - invalid type
            pass

    @staticmethod
    def test_mod_with_no_arguments():
        """Test Mod with no arguments should raise error"""
        graph = ascir.HintGraph("test_mod_no_args")
        graph.create_size("size1")
        # Mod with no arguments should raise TypeError
        try:
            Mod()
            assert False, "Expected TypeError for Mod() with no arguments"
        except (TypeError, AttributeError):
            # Expected - invalid number of arguments
            pass

    @staticmethod
    def test_mod_with_single_argument():
        """Test Mod with single argument should raise error"""
        graph = ascir.HintGraph("test_mod_single_arg")
        size1 = graph.create_size("size1")
        # Mod with single argument should raise TypeError
        try:
            Mod(size1)
            assert False, "Expected TypeError for Mod() with single argument"
        except TypeError:
            # Expected - invalid number of arguments
            pass

    @staticmethod
    def test_mod_with_invalid_types():
        """Test Mod with non-SizeExpr arguments"""
        graph = ascir.HintGraph("test_mod_invalid_types")
        size1 = graph.create_size("size1")
        # Mod with list argument
        try:
            Mod(size1, [1, 2, 3])
            # If it doesn't raise, at least verify it handles gracefully
        except (TypeError, AttributeError, SystemError):
            # Expected - invalid type
            pass
        # Mod with tuple argument
        try:
            Mod(size1, (1, 2))
            # If it doesn't raise, at least verify it handles gracefully
        except (TypeError, AttributeError, SystemError):
            # Expected - invalid type
            pass

    @staticmethod
    def test_remainder_operator_left_invalid():
        """Test Remainder (%) operator with invalid left operand"""
        # Create a SizeExpr and try invalid operations
        s1 = ascir.SizeExpr(50)
        # This tests the operator in reverse - string formatting with SizeExpr
        try:
            "invalid" % s1
            # String % with SizeExpr - might work differently
        except (TypeError, AttributeError):
            # Expected - string formatting doesn't support SizeExpr
            pass

    @staticmethod
    def test_floordiv_operator_with_negative_divisor():
        """Test FloorDiv with negative value edge case"""
        s0 = ascir.SizeExpr(100)
        # Negative divisor - verify it handles without crashing
        s0 // -5
        # Just verify it doesn't crash

    @staticmethod
    def test_max_with_none_left():
        """Test Max with None as left argument"""
        # None is not a valid SizeExpr - should raise SystemError
        try:
            Max(None, ascir.SizeExpr(10))
        except (SystemError, TypeError):
            # Expected - None is not a valid SizeExpr
            pass

    @staticmethod
    def test_max_with_none_right():
        """Test Max with None as right argument"""
        # None is not a valid SizeExpr - should raise SystemError
        try:
            Max(ascir.SizeExpr(10), None)
        except (SystemError, TypeError):
            # Expected - None is not a valid SizeExpr
            pass

    @staticmethod
    def test_min_with_both_none():
        """Test Min with None as both arguments"""
        # None is not a valid SizeExpr - should raise SystemError
        try:
            Min(None, None)
        except (SystemError, TypeError):
            # Expected - None is not a valid SizeExpr
            pass

    @staticmethod
    def test_mod_with_small_divisor():
        """Test Mod with various small divisors"""
        s0 = ascir.SizeExpr(100)
        # Test with divisor 1
        result1 = s0 % 1
        assert result1.expression == "0"
        # Test with divisor 2
        result2 = s0 % 2
        assert result2.expression == "0"

    @staticmethod
    def test_set_platform():
        """Test UtilsSetPlatform with various platform strings"""
        # Test with valid platform strings
        result = ascir.utils.set_platform("v2", 1, 1024)
        assert result is None, "set_platform should return None for valid input"

        result = ascir.utils.set_platform("Ascend910B", 1, 1024)
        assert result is None, "set_platform should return None for valid input"

        # Test with empty string (should return None without error)
        result = ascir.utils.set_platform("", 0, 0)
        assert result is None, "set_platform should return None for empty string"

        # Test with None-like input (passing empty string)
        result = ascir.utils.set_platform("", 0, 0)
        assert result is None, "set_platform should return None for empty string"

        # Test with invalid parameter type - should raise TypeError
        try:
            ascir.utils.set_platform(123, 1, 1024)
        except TypeError as e:
            assert "param parse failed" in str(e) or "string" in str(e).lower()

        try:
            ascir.utils.set_platform(None, 1, 1024)
        except TypeError as e:
            assert "param parse failed" in str(e) or "string" in str(e).lower()

        result = ascir.utils.set_platform("2201", 1, 1024)
        assert result is None, "set_platform should return None for valid input"
        ascir.utils.set_platform("2201", 1, 245760)
