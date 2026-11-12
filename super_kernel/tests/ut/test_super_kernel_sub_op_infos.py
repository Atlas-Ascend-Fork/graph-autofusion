#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------------------

"""Minimal smoke test ensuring the UT harness executes."""

import logging
from unittest import mock
from ut.test_super_kernel import sub_op_add_json, info_dict
from superkernel.super_kernel_sub_op_infos import SubOperatorInfos
import pytest
from asc_op_compile_base.asc_op_compiler.super_kernel_utility import CommonUtility
from superkernel.super_kernel_constants import (
    SuperKernelEarlyStartMode,
    SuperKernelKernelType,
    SuperKernelProfilingMode,
)


class TestSuperKernelSubOpInfos:
    @staticmethod
    @pytest.mark.parametrize("schedule_mode", [None, 0, 1])
    def test_schedule_mode_from_json(schedule_mode):
        sub_op_json = dict(sub_op_add_json)
        if schedule_mode is not None:
            sub_op_json["schedule_mode"] = schedule_mode
        with (
            mock.patch("builtins.open", mock.mock_open(read_data="{}")),
            mock.patch("json.load", return_value=sub_op_json),
        ):
            sub_op = SubOperatorInfos(0, info_dict, 0, {})
            sub_op.init_of_sub_operator_info()
        assert sub_op.schedule_mode == (schedule_mode or 0)

    @staticmethod
    def setup_method():
        logging.info("---------------SetUp---------------")

    @staticmethod
    def teardown_method():
        logging.info("--------------TearDown-------------")

    @staticmethod
    def test_gen_profiling_for_notify():
        with mock.patch("json.load", return_value=sub_op_add_json):
            sub_op = SubOperatorInfos(0, info_dict, 0, {})
            sub_op.profiling_mode = SuperKernelProfilingMode.ProfilingDisable
            code_gen = sub_op.gen_profiling_for_notify(1, False)
            assert code_gen == ""

            sub_op.profiling_mode = SuperKernelProfilingMode.ProfilingEnable
            code_gen = sub_op.gen_profiling_for_notify(1, False)
            assert "RecordProfiling(1, 0x4, true);" in code_gen

            code_gen = sub_op.gen_profiling_for_notify(1, True)
            assert "RecordProfiling(1, 0x4, false);" in code_gen

    @staticmethod
    def test_gen_profiling_for_wait():
        with mock.patch("json.load", return_value=sub_op_add_json):
            sub_op = SubOperatorInfos(0, info_dict, 0, {})
            sub_op.profiling_mode = SuperKernelProfilingMode.ProfilingDisable
            code_gen = sub_op.gen_profiling_for_wait(1, False)
            assert code_gen == ""

            sub_op.profiling_mode = SuperKernelProfilingMode.ProfilingEnable
            code_gen = sub_op.gen_profiling_for_wait(1, False)
            assert "RecordProfiling(1, 12, true);\n" == code_gen

            code_gen = sub_op.gen_profiling_for_wait(1, True)
            assert "RecordProfiling(1, 12, false);\n" == code_gen

    @staticmethod
    def test_gen_notify_from_outside(tmp_dir):
        with mock.patch("json.load", return_value=sub_op_add_json):
            with mock.patch.object(
                CommonUtility, "get_kernel_meta_dir", return_value=tmp_dir
            ):
                sub_op = SubOperatorInfos(0, info_dict, 0, {})
                sub_op.send_event_list = [100, 101]
                sub_op.notify_param_offset = 5
                sub_op.kernel_name = "test_kernel"

                sub_op.kernel_type = SuperKernelKernelType.KERNEL_TYPE_AIC_ONLY
                inner_event_id_set = []
                sub_op.gen_notify_from_outside(inner_event_id_set, False)
                goden_code = """\
if ASCEND_IS_AIC {
    // kernel=test_kernel, ev=100, param_offset=5
    NotifyFunc<true>(param_base[5]);
    // kernel=test_kernel, ev=101, param_offset=6
    NotifyFunc<true>(param_base[6]);
}
"""
                assert goden_code == sub_op.notify_block

                sub_op.kernel_type = SuperKernelKernelType.KERNEL_TYPE_AIV_ONLY
                sub_op.gen_notify_from_outside(inner_event_id_set, False)
                goden_code = """\
if ASCEND_IS_AIV {
    // kernel=test_kernel, ev=100, param_offset=5
    NotifyFunc<false>(param_base[5]);
    // kernel=test_kernel, ev=101, param_offset=6
    NotifyFunc<false>(param_base[6]);
}
"""
                assert goden_code == sub_op.notify_block

    @staticmethod
    def test_gen_wait_from_outside(tmp_dir):
        with mock.patch("json.load", return_value=sub_op_add_json):
            with mock.patch.object(
                CommonUtility, "get_kernel_meta_dir", return_value=tmp_dir
            ):
                sub_op = SubOperatorInfos(0, info_dict, 0, {})
                sub_op.recv_event_list = [200, 201]
                sub_op.wait_param_offset = 8
                sub_op.kernel_name = "test_kernel"

                sub_op.kernel_type = SuperKernelKernelType.KERNEL_TYPE_AIC_ONLY
                inner_event_id_set = []
                sub_op.gen_wait_from_outside(inner_event_id_set, False)
                goden_code = """\
if ASCEND_IS_AIC {
    // kernel=test_kernel, ev=200, param_offset=8
    WaitFunc<true>(param_base[8]);
    // kernel=test_kernel, ev=201, param_offset=9
    WaitFunc<true>(param_base[9]);
}
"""
                assert goden_code == sub_op.wait_block

                sub_op.kernel_type = SuperKernelKernelType.KERNEL_TYPE_AIV_ONLY
                sub_op.gen_wait_from_outside(inner_event_id_set, False)
                goden_code = """\
if ASCEND_IS_AIV {
    // kernel=test_kernel, ev=200, param_offset=8
    WaitFunc<false>(param_base[8]);
    // kernel=test_kernel, ev=201, param_offset=9
    WaitFunc<false>(param_base[9]);
}
"""
                assert goden_code == sub_op.wait_block

    @staticmethod
    def test_init_of_sub_operator_info(tmp_dir):
        with (
            mock.patch("builtins.open", new_callable=mock.mock_open, read_data="{}"),
            mock.patch.object(
                CommonUtility, "ascendc_raise_python_err"
            ) as mock_ascendc_raise_python_err,
            mock.patch.object(
                CommonUtility, "get_kernel_meta_dir", return_value=tmp_dir
            ),
        ):
            new_json = sub_op_add_json
            new_json["sub_operator_early_start_set_flag"] = True
            with mock.patch("json.load", return_value=new_json):
                sub_op = SubOperatorInfos(
                    0,
                    info_dict,
                    0,
                    {"early-start": SuperKernelEarlyStartMode.EarlyStartDisable},
                )
                sub_op.init_of_sub_operator_info()
                mock_ascendc_raise_python_err.assert_called()

            new_json["split_mode"] = 1
            with mock.patch("json.load", return_value=new_json):
                sub_op.split_mode = 4
                sub_op.init_of_sub_operator_info()
                mock_ascendc_raise_python_err.assert_called()

    @staticmethod
    def test_gen_select_addr_code():
        with mock.patch("json.load", return_value=sub_op_add_json):
            sub_op = SubOperatorInfos(0, info_dict, 0, {})

            sub_op.split_mode = 1
            code_gen = sub_op.gen_select_addr_code("func_addr", "kernel_name")
            assert "func_addr = (uint64_t)(kernel_name);" == code_gen

            sub_op.split_mode = 3
            code_gen = sub_op.gen_select_addr_code("func_addr", "kernel_name")
            assert (
                "func_addr = (uint64_t)(kernel_name);\
\n    func_addr_split1 = (uint64_t)(kernel_name_split1);\
\n    func_addr_split2 = (uint64_t)(kernel_name_split2);"
                == code_gen
            )

    @staticmethod
    def test_gen_switch_case_block_of_dynamic_op_only():
        with mock.patch("json.load", return_value=sub_op_add_json):
            with mock.patch.object(
                CommonUtility, "get_chip_version", return_value="c220"
            ):
                sub_op = SubOperatorInfos(0, info_dict, 0, {})
                kernel_info = {
                    "AiCore": "test_kernel",
                    "dav-c220-vec": "test_kernel_aiv",
                    "dav-c220-cube": "test_kernel_aic",
                }
                sub_op.kernel_params = ["kernel_params"]
                code_gen = sub_op.gen_switch_case_block_of_dynamic_op(
                    kernel_info,
                    "tiling_key",
                    SuperKernelKernelType.KERNEL_TYPE_AIV_ONLY,
                )
                goden_code = """
aiv_func_addr = (uint64_t)(test_kernel);
    aiv_func_addr_split1 = (uint64_t)(test_kernel_split1);
    aiv_func_addr_split2 = (uint64_t)(test_kernel_split2);
    aiv_func_addr_split3 = (uint64_t)(test_kernel_split3);
dy_block_num = ((uint64_t)0) << 32 | (*blockNumAddr);
"""
                assert code_gen == goden_code

                code_gen = sub_op.gen_switch_case_block_of_dynamic_op(
                    kernel_info,
                    "tiling_key",
                    SuperKernelKernelType.KERNEL_TYPE_AIC_ONLY,
                )
                goden_code = """
aic_func_addr = (uint64_t)(test_kernel);
    aic_func_addr_split1 = (uint64_t)(test_kernel_split1);
    aic_func_addr_split2 = (uint64_t)(test_kernel_split2);
    aic_func_addr_split3 = (uint64_t)(test_kernel_split3);
dy_block_num = ((uint64_t)1) << 32 | (*blockNumAddr);
"""
                assert code_gen == goden_code

    @staticmethod
    def test_gen_switch_case_block_of_dynamic_op_mix():
        with mock.patch("json.load", return_value=sub_op_add_json):
            with mock.patch.object(
                CommonUtility, "get_chip_version", return_value="c220"
            ):
                sub_op = SubOperatorInfos(0, info_dict, 0, {})
                kernel_info = {
                    "AiCore": "test_kernel",
                    "dav-c220-vec": "test_kernel_aiv",
                    "dav-c220-cube": "test_kernel_aic",
                }
                sub_op.kernel_params = ["kernel_params"]

                code_gen = sub_op.gen_switch_case_block_of_dynamic_op(
                    kernel_info,
                    "tiling_key",
                    SuperKernelKernelType.KERNEL_TYPE_MIX_AIV_1_0,
                )
                goden_code = """
aiv_func_addr = (uint64_t)(test_kernel_aiv);
    aiv_func_addr_split1 = (uint64_t)(test_kernel_aiv_split1);
    aiv_func_addr_split2 = (uint64_t)(test_kernel_aiv_split2);
    aiv_func_addr_split3 = (uint64_t)(test_kernel_aiv_split3);
dy_block_num = ((uint64_t)4) << 32 | (*blockNumAddr);
"""
                assert code_gen == goden_code

                code_gen = sub_op.gen_switch_case_block_of_dynamic_op(
                    kernel_info,
                    "tiling_key",
                    SuperKernelKernelType.KERNEL_TYPE_MIX_AIC_1_0,
                )
                goden_code = """
aic_func_addr = (uint64_t)(test_kernel_aic);
    aic_func_addr_split1 = (uint64_t)(test_kernel_aic_split1);
    aic_func_addr_split2 = (uint64_t)(test_kernel_aic_split2);
    aic_func_addr_split3 = (uint64_t)(test_kernel_aic_split3);
dy_block_num = ((uint64_t)5) << 32 | (*blockNumAddr);
"""
                assert code_gen == goden_code

                code_gen = sub_op.gen_switch_case_block_of_dynamic_op(
                    kernel_info,
                    "tiling_key",
                    SuperKernelKernelType.KERNEL_TYPE_MIX_AIC_1_2,
                )
                goden_code = """
aiv_func_addr = (uint64_t)(test_kernel_aiv);
    aiv_func_addr_split1 = (uint64_t)(test_kernel_aiv_split1);
    aiv_func_addr_split2 = (uint64_t)(test_kernel_aiv_split2);
    aiv_func_addr_split3 = (uint64_t)(test_kernel_aiv_split3);
aic_func_addr = (uint64_t)(test_kernel_aic);
    aic_func_addr_split1 = (uint64_t)(test_kernel_aic_split1);
    aic_func_addr_split2 = (uint64_t)(test_kernel_aic_split2);
    aic_func_addr_split3 = (uint64_t)(test_kernel_aic_split3);
dy_block_num = ((uint64_t)7) << 32 | (*blockNumAddr);
"""
                assert code_gen == goden_code

    @staticmethod
    def test_gen_param_code():
        with mock.patch("json.load", return_value=sub_op_add_json):
            sub_op = SubOperatorInfos(0, info_dict, 0, {})
            sub_op.split_mode = 1
            code_gen = sub_op.gen_param_code("param")
            assert code_gen == "param"

            sub_op.split_mode = 3
            code_gen = sub_op.gen_param_code("param")

            assert "param, param_split1, param_split2" in code_gen

    @staticmethod
    def test_gen_binary_search_block():
        with mock.patch("json.load", return_value=sub_op_add_json):
            sub_op = SubOperatorInfos(0, info_dict, 0, {})

            blocks = [["code1", 100]]
            code_gen = sub_op.gen_binary_search_block(blocks)
            goden_code = """\
if (*tilingKeyAddr == 100) {
    code1
}\
"""
            assert code_gen == goden_code

            blocks = [["code1", 100], ["code2", 200]]
            code_gen = sub_op.gen_binary_search_block(blocks)
            goden_code = """\
if (*tilingKeyAddr == 100) {
    code1
} else {
    code2
}\
"""
            assert code_gen == goden_code

            blocks = [["code1", 100], ["code2", 200], ["code3", 300]]
            code_gen = sub_op.gen_binary_search_block(blocks)
            goden_code = """\
if (*tilingKeyAddr < 200) {
    if (*tilingKeyAddr == 100) {
        code1
    }
} else {
    if (*tilingKeyAddr == 200) {
        code2
    } else {
        code3
    }
}\
"""
            assert code_gen == goden_code

    @staticmethod
    def test_gen_switch_code_of_dynamic_op():
        with mock.patch("json.load", return_value=sub_op_add_json):
            sub_op = SubOperatorInfos(0, info_dict, 0, {})
            sub_op.kernel_params = ["param1"]
            sub_op.split_mode = 1
            sub_op.kernel_name = "test_kernel"
            sub_op.called_kernel_name = {
                "dynamic_func_names": {
                    "0": {"AiCore": "kernel", "kernel_type": "KERNEL_TYPE_AIV_ONLY"}
                }
            }

            sub_op.gen_switch_code_of_dynamic_op()
            goden_code = """
// begin implement of dynamic op test_kernel
static __aicore__ void switch_func_of_test_kernel(GM_ADDR __ac_dynamic_tiling_key_0, GM_ADDR __ac_dynamic_block_num_0, GM_ADDR __ac_wait_lock_0, uint64_t& aiv_func_addr, uint64_t& aic_func_addr, uint64_t& dy_block_num) {
    __gm__ uint64_t* tilingKeyAddr = reinterpret_cast<__gm__ uint64_t*>(__ac_dynamic_tiling_key_0);
    __gm__ uint64_t* blockNumAddr = reinterpret_cast<__gm__ uint64_t*>(__ac_dynamic_block_num_0);
    __gm__ volatile uint64_t* lockAddr = reinterpret_cast<__gm__ uint64_t*>(__ac_wait_lock_0);
    dcci(lockAddr, 0, 2);
    while(*lockAddr != 1) {
        dcci(lockAddr, 0, 2);
    }

    if (*tilingKeyAddr == 0) {

    aiv_func_addr = (uint64_t)(kernel);
    dy_block_num = ((uint64_t)0) << 32 | (*blockNumAddr);

}
    return;
}
"""
        assert sub_op.dynamic_impl_func_block == goden_code

    @staticmethod
    def test_dynamic_gen_split_call_code():
        with mock.patch("json.load", return_value=sub_op_add_json):
            sub_op = SubOperatorInfos(0, info_dict, 0, {})
            sub_op.split_mode = 1
            code_gen = sub_op.dynamic_gen_split_call_code("test_func", "args")
            assert "test_func(args);" == code_gen

            sub_op.split_mode = 3
            code_gen = sub_op.dynamic_gen_split_call_code("test_func", "args")
            goden_code = """\
if ((coreid % 3) == 0) {

                test_func(args);
            } else if ((coreid % 3) == 1) {
            test_func_split1(args);
            } else {


                test_func_split2(args);
            }\
"""
            assert goden_code == code_gen

    @staticmethod
    def test_gen_kernel_call_block():
        with mock.patch("json.load", return_value=sub_op_add_json):
            sub_op = SubOperatorInfos(0, info_dict, 0, {})
            sub_op.kernel_type = SuperKernelKernelType.KERNEL_TYPE_AIV_ONLY
            sub_op.param_offset = 0
            sub_op.kernel_params = ["param1"]
            sub_op.extra_kernel_params = ["__ac_wait_lock_0"]
            sub_op.split_mode = 1

            sub_op.gen_kernel_call_block(False)
            goden_code = """\
call_func_of_(0, aiv_func_addr, aic_func_addr, dy_blockNum);
if ASCEND_IS_AIV {

    if (AscendC::GetBlockIdx() == 0) {
        __gm__ volatile uint64_t* lockAddr = reinterpret_cast<__gm__ uint64_t*>(param_base[1]);
        *lockAddr = 0;
        dcci(lockAddr, 0, 2);
    }
}
"""
            assert goden_code == sub_op.kernel_call_block

            sub_op.gen_kernel_call_block(True)
            goden_code = """\
    AscendC::SyncAll<false>(); // reason: double stream need syncall to wait switch func
call_func_of_(0, aiv_func_addr, aic_func_addr, dy_blockNum);
if ASCEND_IS_AIV {

    if (AscendC::GetBlockIdx() == 0) {
        __gm__ volatile uint64_t* lockAddr = reinterpret_cast<__gm__ uint64_t*>(param_base[1]);
        *lockAddr = 0;
        dcci(lockAddr, 0, 2);
    }
}
"""
            assert goden_code == sub_op.kernel_call_block


if __name__ == "__main__":
    pytest.main()
