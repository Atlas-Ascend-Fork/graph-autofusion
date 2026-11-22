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
import json
import os
from autofuse.pyautofuse import ascir, AutofuserOptions, Schedule, CodeGen
from ascir import Max, Min, Mod

PYF_PATH = os.path.dirname(os.path.realpath(__file__))


class TestComputeGraphInput:
    @staticmethod
    def construct_compute_graph():
        test_graph = os.path.join(PYF_PATH, "test_seri_compute_graph.txt")
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
        except RuntimeError as e:
            print(f"Caught a RuntimeError: {e}", flush=True)
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
                "  TILING_DATA_FIELD_DEF_T(uint16_t, block_dim);\n"
                "  TILING_DATA_FIELD_DEF_T(uint16_t, tiling_key);\n"
                "  TILING_DATA_FIELD_DEF_T(uint32_t, ub_size);\n"
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
            output_shape = output_shape = [["s0", 1]]
            vector_core_num = "20"
            tiling, infer_shape = codegen.host_code_generator(
                schedule_results, shape_info, output_shape, "", vector_core_num
            )
        except ValueError:
            pass

        try:
            get_kernel = codegen.get_kernel_and_json_generator(kernel_path, json_path)
        except ValueError:
            pass


class TestUtilsDeserialize:
    @staticmethod
    def get_serialize_asc_graph():
        test_graph = os.path.join(PYF_PATH, "test_seri_asc_graph.txt")
        with open(test_graph, "r", encoding="utf-8") as file:
            content = file.read()
        return content

    def test_symbol_source_info(self):
        symbol_source = '{"s0":"GetDimValueFromGraphInputData(0, 0);","s1":"GetDimValueFromGraphInputData(0,1)"}'
        symbol_obj = ascir.utils.deserialize("symbol_source_info", symbol_source)
        assert symbol_obj is not None

        try:
            symbol_source = '{"s0":"GetDimValueFromGraphInputData(0, 0);","s1":0}'
            symbol_obj = ascir.utils.deserialize("symbol_source_info", symbol_source)
        except TypeError:
            pass

        try:
            symbol_source = "test"
            symbol_obj = ascir.utils.deserialize("symbol_source_info", symbol_source)
        except TypeError:
            pass

    @pytest.mark.skip
    def test_asc_graph(self):
        asc_graph = self.get_serialize_asc_graph()
        asc_graph_obj = ascir.utils.deserialize("asc_graph", asc_graph)
        assert asc_graph_obj is not None

        try:
            asc_graph_obj = ascir.utils.deserialize("asc_graph", "error")
        except TypeError:
            pass


class TestSizeExprOperators:
    """Test SizeExpr % (remainder) and // (floor division) operators"""

    @staticmethod
    def test_size_expr_remainder_operator():
        """Test SizeExpr % (remainder) operator with constants"""
        s0 = ascir.SizeExpr(10)
        s1 = ascir.SizeExpr(3)

        # Test remainder operation
        result = s0 % s1
        assert result.expression == "1"

        # Test with integers
        result2 = s0 % 4
        assert result2.expression == "2"

    @staticmethod
    def test_size_expr_floordiv_operator():
        """Test SizeExpr // (floor division) operator with constants"""
        s0 = ascir.SizeExpr(10)
        s1 = ascir.SizeExpr(3)

        # Test floor division operation
        result = s0 // s1
        assert result.expression == "3"

        # Test with integers
        result2 = s0 // 4
        assert result2.expression == "2"

    @staticmethod
    def test_size_expr_remainder_symbolic():
        """Test SizeExpr % (remainder) operator with symbolic expressions"""
        graph = ascir.HintGraph("test_mod")
        s0 = graph.create_size("s0")
        s1 = graph.create_size("s1")

        # Test symbolic remainder: s0 % 16 (alignment check)
        remainder = s0 % 16
        assert remainder.expression == "Mod(s0, 16)"

        # Test symbolic remainder: s0 % s1
        remainder2 = s0 % s1
        assert remainder2.expression == "Mod(s0, s1)"

    @staticmethod
    def test_size_expr_floordiv_symbolic():
        """Test SizeExpr // (floor division) operator with symbolic expressions"""
        graph = ascir.HintGraph("test_floordiv")
        s0 = graph.create_size("s0")
        s1 = graph.create_size("s1")

        # Test symbolic floor division: s0 // 16 (block count calculation)
        blocks = s0 // 16
        assert blocks.expression == "Floor((Rational(1 , 16) * s0))"

        # Test symbolic floor division: s0 // s1
        blocks2 = s0 // s1
        assert blocks2.expression == "Floor((s0 / (s1)))"


class TestSizeExprMaxMinST:
    """System Test: Max and Min functions for SizeExpr in real graph scenarios"""

    @staticmethod
    def test_max_in_graph_construction():
        """Test Max function in graph construction scenario"""
        graph = ascir.HintGraph("test_max_graph")
        s0 = graph.create_size("s0")
        s1 = graph.create_size("s1")

        # Use Max to create axis size
        max_size = Max(s0, s1)
        graph.create_axis("z0", max_size)

        # Verify axis is created with Max expression
        debug_str = ascir.utils.debug_str(graph)
        assert "z0" in debug_str
        assert "max" in debug_str.lower()

    @staticmethod
    def test_min_in_graph_construction():
        """Test Min function in graph construction scenario"""
        graph = ascir.HintGraph("test_min_graph")
        s0 = graph.create_size("s0")
        s1 = graph.create_size("s1")

        # Use Min to create axis size
        min_size = Min(s0, s1)
        graph.create_axis("z0", min_size)

        # Verify axis is created with Min expression
        debug_str = ascir.utils.debug_str(graph)
        assert "z0" in debug_str
        assert "min" in debug_str.lower()

    @staticmethod
    def test_max_with_constant_and_symbolic():
        """Test Max with constant and symbolic sizes"""
        graph = ascir.HintGraph("test_max_const_sym")
        s0 = graph.create_size("s0")

        # Max of symbolic and constant (common for padding)
        max_size = Max(s0, 128)
        graph.create_axis("z0", max_size)

        # Verify the max expression (Max函数会重新排序参数)
        assert max_size.expression == "Max(128, s0)"

    @staticmethod
    def test_min_with_constant_and_symbolic():
        """Test Min with constant and symbolic sizes"""
        graph = ascir.HintGraph("test_min_const_sym")
        s0 = graph.create_size("s0")

        # Min of symbolic and constant (common for capping)
        min_size = Min(s0, 1024)
        graph.create_axis("z0", min_size)

        # Verify the min expression (Min函数会重新排序参数)
        assert min_size.expression == "Min(1024, s0)"

    @staticmethod
    def test_max_min_in_memory_calculation():
        """Test Max/Min in memory size calculation scenario"""
        graph = ascir.HintGraph("test_memory_calc")

        # Create symbolic sizes for different dimensions
        batch_size = graph.create_size("batch_size")
        seq_len = graph.create_size("seq_len")
        hidden_size = graph.create_size("hidden_size")

        # Calculate memory with alignment
        base_size = batch_size * seq_len * hidden_size
        aligned_size = Max(base_size, 512)  # At least 512 bytes

        # Verify the expression (Max函数会重新排序参数)
        assert (
            aligned_size.expression == "Max(512, (batch_size * hidden_size * seq_len))"
        )

    @staticmethod
    def test_max_min_in_tiling_scenario():
        """Test Max/Min in tiling scenario"""
        graph = ascir.HintGraph("test_tiling")

        # Tile size with min/max bounds
        input_size = graph.create_size("input_size")
        max_tile = 1024
        min_tile = 128

        # Clamp tile size between min and max
        tile_size = Min(Max(input_size, min_tile), max_tile)

        # Create axis with clamped size
        graph.create_axis("z0", tile_size)

        # Verify clamping works
        debug_str = ascir.utils.debug_str(graph)
        assert "z0" in debug_str

    @staticmethod
    def test_max_in_concat_scenario():
        """Test Max in concat output size calculation"""
        graph = ascir.HintGraph("test_concat_max")

        s0 = graph.create_size("s0")
        s1 = graph.create_size("s1")
        s2 = graph.create_size("s2")

        # Scenario: concat along dim 1, output dim 0 is max of inputs
        z0 = graph.create_axis("z0", Max(Max(s0, s1), s2))

        # Verify axis creation (Max函数会重新排序参数)
        assert z0.size.expression == "Max(Max(s0, s1), s2)"

    @staticmethod
    def test_min_in_reduce_scenario():
        """Test Min in reduce scenario"""
        graph = ascir.HintGraph("test_reduce_min")

        s0 = graph.create_size("s0")
        s1 = graph.create_size("s1")

        # Scenario: reduce output size is min of dimensions
        reduced_size = Min(s0, s1)
        z0 = graph.create_axis("z0", reduced_size)

        # Verify axis creation
        assert z0.size.expression == "Min(s0, s1)"

    @staticmethod
    def test_max_min_with_arithmetic():
        """Test Max/Min combined with arithmetic operations"""
        graph = ascir.HintGraph("test_arithmetic")

        s0 = graph.create_size("s0")
        s1 = graph.create_size("s1")

        # Max of sums: max(s0 + 100, s1)
        max_expr = Max(s0 + 100, s1)
        assert max_expr.expression == "Max(s1, (100 + s0))"

        # Min of products: min(s0 * 2, s1 * 3)
        min_expr = Min(s0 * 2, s1 * 3)
        assert min_expr.expression == "Min((2 * s0), (3 * s1))"


class TestSizeExprModST:
    """System Test: Mod function for SizeExpr in real graph scenarios"""

    @staticmethod
    def test_mod_in_graph_construction():
        """Test Mod function in graph construction scenario"""
        graph = ascir.HintGraph("test_mod_graph")
        s0 = graph.create_size("s0")
        s1 = graph.create_size("s1")

        # Use Mod to create axis size
        mod_size = Mod(s0, s1)
        graph.create_axis("z0", mod_size)

        # Verify axis is created with Mod expression
        debug_str = ascir.utils.debug_str(graph)
        assert "z0" in debug_str
        assert "mod" in debug_str.lower()

    @staticmethod
    def test_mod_with_constant_alignment():
        """Test Mod with constant for alignment check"""
        graph = ascir.HintGraph("test_mod_align")
        s0 = graph.create_size("s0")

        # Mod with constant 16 (common for alignment check)
        aligned_size = Mod(s0, 16)
        graph.create_axis("z0", aligned_size)

        # Verify the mod expression
        assert aligned_size.expression == "Mod(s0, 16)"

    @staticmethod
    def test_mod_in_padding_calculation():
        """Test Mod in padding calculation scenario"""
        graph = ascir.HintGraph("test_padding")

        # Create symbolic size
        input_size = graph.create_size("input_size")
        tile_size = 32

        # Calculate remainder when divided by tile_size
        remainder = Mod(input_size, tile_size)

        # Verify the mod expression
        assert remainder.expression == "Mod(input_size, 32)"

    @staticmethod
    def test_mod_in_tiling_scenario():
        """Test Mod in tiling scenario"""
        graph = ascir.HintGraph("test_tiling_mod")

        # Tile size with alignment check
        input_size = graph.create_size("input_size")
        alignment = 64

        # Check if input is aligned
        remainder = Mod(input_size, alignment)

        # Create axis with remainder
        graph.create_axis("z0", remainder)

        # Verify axis creation
        debug_str = ascir.utils.debug_str(graph)
        assert "z0" in debug_str

    @staticmethod
    def test_mod_with_arithmetic():
        """Test Mod combined with arithmetic operations"""
        graph = ascir.HintGraph("test_mod_arithmetic")

        s0 = graph.create_size("s0")
        s1 = graph.create_size("s1")

        # Mod of sum: Mod(s0 + 16, s1)
        mod_expr = Mod(s0 + 16, s1)
        assert mod_expr.expression == "Mod((16 + s0), s1)"

    @staticmethod
    def test_mod_in_memory_alignment():
        """Test Mod in memory alignment calculation"""
        graph = ascir.HintGraph("test_mem_align")

        # Create symbolic sizes
        batch_size = graph.create_size("batch_size")
        elem_size = 4

        # Calculate total size in bytes
        total_bytes = batch_size * elem_size

        # Check alignment to 32 bytes
        alignment_remainder = Mod(total_bytes, 32)

        # Verify the mod expression
        assert alignment_remainder.expression == "Mod((4 * batch_size), 32)"
