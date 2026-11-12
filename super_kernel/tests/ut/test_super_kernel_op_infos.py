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
import os
import subprocess
from unittest import mock
import pytest
from asc_op_compile_base.asc_op_compiler.super_kernel_utility import CommonUtility
from superkernel.super_kernel_constants import (
    SuperKernelEarlyStartMode,
    SubOperatorType,
    SuperKernelStreamFusionMode,
    SuperKernelFeedSyncAllMode,
    SuperKernelKernelType,
)
from superkernel.super_kernel_sub_op_infos import SubOperatorInfos
from superkernel.super_kernel_op_infos import (
    SuperOperatorInfos,
    gen_symbol_rename_file,
    split_dynamic_o_in_super_kernel,
    get_sub_op_streamid,
)
from ut.test_super_kernel import sub_op_add_json, kernel_info, info_dict


class TestSuperKernelOpInfos:
    @staticmethod
    def setup_method():
        logging.info("---------------SetUp---------------")

    @staticmethod
    def teardown_method():
        logging.info("--------------TearDown-------------")

    @staticmethod
    def test_gen_symbol_rename_file(tmp_dir):
        sub_operator = SubOperatorInfos(0, info_dict, 0, {})
        with (
            mock.patch("json.load", return_value=sub_op_add_json),
            mock.patch.object(
                CommonUtility, "get_kernel_meta_dir", return_value=tmp_dir
            ),
            mock.patch.object(CommonUtility, "get_chip_version", return_value="c220"),
        ):
            rename_file_path_list = []
            kernel_meta_dir = CommonUtility.get_kernel_meta_dir()
            sub_operator.split_mode = 2
            for i in range(1, sub_operator.split_mode):
                rename_file_name = f"{sub_operator.kernel_name}_rename_file_{i}.txt"
                rename_file_path_list.append(
                    os.path.join(kernel_meta_dir, rename_file_name)
                )

            dynamic_func_names = {"tiling_key": {"dav-c220-cube": "cube_kernel_name"}}
            new_kernel_names_list = gen_symbol_rename_file(
                dynamic_func_names, rename_file_path_list, sub_operator.split_mode
            )
            assert new_kernel_names_list == [["cube_kernel_name_split1"]]

    @staticmethod
    def test_split_dynamic_o_in_super_kernel(tmp_dir):
        super_operator = SuperOperatorInfos(
            kernel_info, "test_split_dynamic_o_in_super_kernel"
        )
        tmp_dir_str = str(tmp_dir)
        orign_bin_path = os.path.join(tmp_dir_str, "original.o")
        rename_file = os.path.join(tmp_dir_str, "rename.txt")
        with mock.patch.object(
            CommonUtility, "get_kernel_meta_dir", return_value=tmp_dir_str
        ):
            with mock.patch("subprocess.run"):
                with mock.patch("os.path.exists"):
                    filename = os.path.basename(orign_bin_path)
                    kernel_meta_dir = CommonUtility.get_kernel_meta_dir()
                    new_bin_path_exist_mock = os.path.join(
                        kernel_meta_dir, filename[:-2] + "_split1.o"
                    )
                    new_bin_path = split_dynamic_o_in_super_kernel(
                        orign_bin_path, rename_file, 1, super_operator.compile_log_path
                    )
                    assert new_bin_path_exist_mock == new_bin_path

            with mock.patch(
                "subprocess.run",
                side_effect=subprocess.CalledProcessError(1, "test_command"),
            ):
                with mock.patch.object(
                    CommonUtility, "ascendc_raise_python_err"
                ) as mock_raise:
                    new_bin_path = split_dynamic_o_in_super_kernel(
                        orign_bin_path, rename_file, 1, super_operator.compile_log_path
                    )
                    mock_raise.assert_called()
                    assert new_bin_path_exist_mock == new_bin_path

    @staticmethod
    def test_get_sub_op_streamid():
        op_info = {}
        op_info["stream_id"] = 1
        streamid = get_sub_op_streamid(op_info)
        assert streamid == 1

    @staticmethod
    def test_gen_op_options():
        super_operator = SuperOperatorInfos(kernel_info, "test_gen_op_options")
        super_operator.enable_double_stream = True
        super_operator.gen_op_options()
        assert (
            super_operator.early_start_mode.value
            == SuperKernelEarlyStartMode.EarlyStartDisable.value
        )

    @staticmethod
    def test_split_op_by_kernel_type():
        with mock.patch("json.load", return_value=sub_op_add_json):
            super_operator = SuperOperatorInfos(kernel_info, "test_split_by_type")
            sub_op1 = SubOperatorInfos(0, info_dict, 0, {})
            sub_op1.kernel_type = SuperKernelKernelType.KERNEL_TYPE_AIC_ONLY
            sub_op2 = SubOperatorInfos(1, info_dict, 0, {})
            sub_op2.kernel_type = SuperKernelKernelType.KERNEL_TYPE_AIV_ONLY
            super_operator.info_base = [sub_op1, sub_op2]

            super_operator.split_op_by_kernel_type()
            assert super_operator.cub_op_list == [sub_op1]
            assert super_operator.vec_op_list == [sub_op2]

    @staticmethod
    def test_get_task_type():
        with mock.patch("json.load", return_value=sub_op_add_json):
            super_operator = SuperOperatorInfos(kernel_info, "test_get_task_type")
            op = SubOperatorInfos(0, info_dict, 0, {})

            op.kernel_type = SuperKernelKernelType.KERNEL_TYPE_AIC_ONLY
            assert super_operator.get_task_type(op) == "cub"

            op.kernel_type = SuperKernelKernelType.KERNEL_TYPE_AIV_ONLY
            assert super_operator.get_task_type(op) == "vec"

            op.kernel_type = SuperKernelKernelType.KERNEL_TYPE_MIX_AIC_1_1
            assert super_operator.get_task_type(op) == "mix"

    @staticmethod
    def test_gen_sync_name():
        with mock.patch("json.load", return_value=sub_op_add_json):
            super_operator = SuperOperatorInfos(kernel_info, "test_gen_sync_name")
            pre_op = SubOperatorInfos(0, info_dict, 0, {})
            current_op = SubOperatorInfos(1, info_dict, 0, {})

            pre_op.kernel_type = SuperKernelKernelType.KERNEL_TYPE_MIX_AIC_1_1
            current_op.kernel_type = SuperKernelKernelType.KERNEL_TYPE_MIX_AIC_1_1
            assert super_operator.gen_sync_name(pre_op, current_op) == "cub:vec;vec:cub"

            pre_op.kernel_type = SuperKernelKernelType.KERNEL_TYPE_MIX_AIC_1_1
            current_op.kernel_type = SuperKernelKernelType.KERNEL_TYPE_AIC_ONLY
            assert super_operator.gen_sync_name(pre_op, current_op) == "vec:cub"

            pre_op.kernel_type = SuperKernelKernelType.KERNEL_TYPE_MIX_AIC_1_1
            current_op.kernel_type = SuperKernelKernelType.KERNEL_TYPE_AIV_ONLY
            assert super_operator.gen_sync_name(pre_op, current_op) == "cub:vec"

            pre_op.kernel_type = SuperKernelKernelType.KERNEL_TYPE_AIC_ONLY
            current_op.kernel_type = SuperKernelKernelType.KERNEL_TYPE_MIX_AIC_1_1
            assert super_operator.gen_sync_name(pre_op, current_op) == "cub:vec"

            pre_op.kernel_type = SuperKernelKernelType.KERNEL_TYPE_AIV_ONLY
            current_op.kernel_type = SuperKernelKernelType.KERNEL_TYPE_MIX_AIC_1_1
            assert super_operator.gen_sync_name(pre_op, current_op) == "vec:cub"

            pre_op.kernel_type = SuperKernelKernelType.KERNEL_TYPE_AIC_ONLY
            current_op.kernel_type = SuperKernelKernelType.KERNEL_TYPE_AIV_ONLY
            assert super_operator.gen_sync_name(pre_op, current_op) == "cub:vec"

    @staticmethod
    def test_insert_sync_event():
        with mock.patch("json.load", return_value=sub_op_add_json):
            super_operator = SuperOperatorInfos(kernel_info, "test_insert_sync_event")
            pre_op = SubOperatorInfos(0, info_dict, 0, {})
            current_op = SubOperatorInfos(1, info_dict, 0, {})

            pre_op.kernel_type = SuperKernelKernelType.KERNEL_TYPE_AIC_ONLY
            current_op.kernel_type = SuperKernelKernelType.KERNEL_TYPE_AIV_ONLY
            pre_op.kernel_name_for_multi_stream = "pre_op_kernel_name"
            current_op.kernel_name_for_multi_stream = "current_op_kernel_name"
            super_operator.cub_op_list = [pre_op]
            super_operator.vec_op_list = [current_op]
            super_operator.insert_sync_event(pre_op, current_op)
            assert pre_op.send_info["current_op_kernel_name"] == "cub:vec"
            assert current_op.recv_info["pre_op_kernel_name"] == "cub:vec"

            pre_op.kernel_type = SuperKernelKernelType.KERNEL_TYPE_AIV_ONLY
            current_op.kernel_type = SuperKernelKernelType.KERNEL_TYPE_AIC_ONLY
            super_operator.cub_op_list = [current_op]
            super_operator.vec_op_list = [pre_op]
            super_operator.insert_sync_event(pre_op, current_op)
            assert pre_op.send_info["current_op_kernel_name"] == "vec:cub"
            assert current_op.recv_info["pre_op_kernel_name"] == "vec:cub"

    @staticmethod
    def test_insert_sync_by_stream_idx():
        with mock.patch("json.load", return_value=sub_op_add_json):
            super_operator = SuperOperatorInfos(
                kernel_info, "test_insert_sync_by_stream_idx"
            )
            op1 = SubOperatorInfos(0, info_dict, 0, {})
            op1.stream_index = 0
            op1.kernel_type = SuperKernelKernelType.KERNEL_TYPE_AIC_ONLY
            op2 = SubOperatorInfos(1, info_dict, 0, {})
            op2.stream_index = 0
            op2.kernel_type = SuperKernelKernelType.KERNEL_TYPE_AIV_ONLY

            op1.kernel_name_for_multi_stream = "pre_op_kernel_name"
            op2.kernel_name_for_multi_stream = "current_op_kernel_name"
            super_operator.info_base = [op1, op2]
            super_operator.cub_op_list = [op1]
            super_operator.vec_op_list = [op2]

            super_operator.insert_sync_by_stream_idx()
            assert op1.send_info["current_op_kernel_name"] == "cub:vec"
            assert op2.recv_info["pre_op_kernel_name"] == "cub:vec"

    @staticmethod
    def test_insert_sync_by_event():
        with mock.patch("json.load", return_value=sub_op_add_json):
            super_operator = SuperOperatorInfos(
                kernel_info, "test_insert_sync_by_event"
            )

            op1 = SubOperatorInfos(0, info_dict, 0, {})
            op1.send_event_list = [100]
            op1.kernel_type = SuperKernelKernelType.KERNEL_TYPE_AIC_ONLY

            op2 = SubOperatorInfos(0, info_dict, 0, {})
            op2.recv_event_list = [100]
            op2.kernel_type = SuperKernelKernelType.KERNEL_TYPE_AIV_ONLY

            op1.kernel_name_for_multi_stream = "pre_op_kernel_name"
            op2.kernel_name_for_multi_stream = "current_op_kernel_name"
            super_operator.info_base = [op1, op2]
            super_operator.cub_op_list = [op1]
            super_operator.vec_op_list = [op2]

            super_operator.insert_sync_by_event()
            assert op1.send_info["current_op_kernel_name"] == "cub:vec"
            assert op2.recv_info["pre_op_kernel_name"] == "cub:vec"

    @staticmethod
    def test_insert_sync_for_notify():
        with mock.patch("json.load", return_value=sub_op_add_json):
            super_operator = SuperOperatorInfos(kernel_info, "test_sync_notify")

            op1 = SubOperatorInfos(0, info_dict, 0, {})
            op1.kernel_type = SuperKernelKernelType.KERNEL_TYPE_MIX_AIC_1_1
            op1.notify_block = {"aic": "notify_code", "aiv": "notify_code"}
            op1.tmp_notify_block = {"aic": "aic_tmp_notify", "aiv": "aiv_tmp_notify"}
            op1.stream_index = 0
            op1.send_info = ["cub"]

            op2 = SubOperatorInfos(1, info_dict, 0, {})
            op2.kernel_type = SuperKernelKernelType.KERNEL_TYPE_AIC_ONLY
            op2.stream_index = 1
            op2.kernel_name_for_multi_stream = "cub"

            super_operator.info_base = [op1, op2]
            super_operator.cub_op_list = [op1]
            super_operator.vec_op_list = [op2]

            super_operator.insert_sync_for_notify()
            assert op1.notify_block["aic"] == "aic_tmp_notify"
            assert op1.notify_block["aiv"] == "aiv_tmp_notify"

    @staticmethod
    def test_remove_info_by_name():
        with mock.patch("json.load", return_value=sub_op_add_json):
            super_operator = SuperOperatorInfos(kernel_info, "test_remove_info_by_name")

            op1 = SubOperatorInfos(0, info_dict, 0, {})
            op1.kernel_name_for_multi_stream = "op1"
            op1.send_info = {"op1": "cub:vec", "op2": "cub:vec"}

            op2 = SubOperatorInfos(1, info_dict, 0, {})
            op2.recv_info = {"op2": "cub:vec"}

            super_operator.info_base = [op1, op2]

            super_operator.remove_info_by_name("op1", "op2", True, "")
            assert "op2" not in op1.recv_info

            super_operator.remove_info_by_name("op1", "op2", False, "")
            assert "op2" not in op1.send_info

            super_operator.remove_info_by_name("op1", "op2", True, "new_content")
            assert op1.recv_info.get("op2") == "new_content"

            super_operator.remove_info_by_name("op1", "op2", False, "new_content")
            assert op1.send_info.get("op2") == "new_content"

    @staticmethod
    def test_get_remain_events():
        with mock.patch("json.load", return_value=sub_op_add_json):
            super_operator = SuperOperatorInfos(kernel_info, "test_get_remain_events")

            origin_events = "cub:vec;vec:cub;cub:cub"
            delete_event = "vec:cub"
            result = super_operator.get_remain_events(origin_events, delete_event)
            assert result == "cub:vec;cub:cub"

    @staticmethod
    def test_get_idx():
        with mock.patch("json.load", return_value=sub_op_add_json):
            super_operator = SuperOperatorInfos(kernel_info, "test_get_idx")

            op1 = SubOperatorInfos(0, info_dict, 0, {})
            op1.kernel_name_for_multi_stream = "op1"

            op2 = SubOperatorInfos(1, info_dict, 0, {})
            op2.kernel_name_for_multi_stream = "op2"

            super_operator.vec_op_list = [op1]
            super_operator.cub_op_list = [op2]

            idx = super_operator.get_idx("op1", True)
            assert idx == 0

            idx = super_operator.get_idx("op2", False)
            assert idx == 0

    @staticmethod
    def test_judge_remove():
        with mock.patch("json.load", return_value=sub_op_add_json):
            super_operator = SuperOperatorInfos(kernel_info, "test_judge_remove")

            op1 = SubOperatorInfos(0, info_dict, 0, {})
            op1.kernel_name_for_multi_stream = "op1"
            op1.recv_info = {"op2": "cub:vec"}
            op2 = SubOperatorInfos(1, info_dict, 0, {})
            op2.kernel_name_for_multi_stream = "op2"
            super_operator.cub_op_list = [op1, op2]
            super_operator.vec_op_list = [op1, op2]
            result = super_operator.judge_remove("op1", "op2", True)
            assert result

            op1.recv_info = {"op1": "cub:vec"}
            result = super_operator.judge_remove("op1", "op2", True)
            assert not result

            op1 = SubOperatorInfos(0, info_dict, 0, {})
            op1.kernel_name_for_multi_stream = "op1"
            op1.recv_info = {"op2": "vec:cub"}
            op2 = SubOperatorInfos(1, info_dict, 0, {})
            op2.kernel_name_for_multi_stream = "op2"
            super_operator.cub_op_list = [op1, op2]
            super_operator.vec_op_list = [op1, op2]
            result = super_operator.judge_remove("op1", "op2", False)
            assert result

            op1.recv_info = {"op1": "vec:cub"}
            result = super_operator.judge_remove("op1", "op2", False)
            assert not result

    @staticmethod
    def test_remove_crossed_line_sync():
        with mock.patch("json.load", return_value=sub_op_add_json):
            super_operator = SuperOperatorInfos(
                kernel_info, "test_remove_crossed_line_sync"
            )

            op1 = SubOperatorInfos(0, info_dict, 0, {})
            op1.kernel_name_for_multi_stream = "op1"
            op1.kernel_type = SuperKernelKernelType.KERNEL_TYPE_AIC_ONLY

            op2 = SubOperatorInfos(1, info_dict, 0, {})
            op2.kernel_name_for_multi_stream = "op2"
            op2.kernel_type = SuperKernelKernelType.KERNEL_TYPE_AIV_ONLY

            op3 = SubOperatorInfos(2, info_dict, 0, {})
            op3.kernel_name_for_multi_stream = "op3"
            op3.kernel_type = SuperKernelKernelType.KERNEL_TYPE_AIC_ONLY

            op4 = SubOperatorInfos(3, info_dict, 0, {})
            op4.kernel_name_for_multi_stream = "op4"
            op4.kernel_type = SuperKernelKernelType.KERNEL_TYPE_AIV_ONLY

            super_operator.cub_op_list = [op1, op3]
            super_operator.vec_op_list = [op2, op4]
            super_operator.info_base = [op1, op2, op3, op4]

            op1.send_info = {"op4": "cub:vec"}
            op1.recv_info = {}
            op4.send_info = {}
            op4.recv_info = {"op1": "cub:vec"}

            op3.send_info = {"op2": "cub:vec"}
            op3.recv_info = {}
            op2.send_info = {}
            op2.recv_info = {"op3": "cub:vec"}

            super_operator.remove_crossed_line_sync()
            assert "op4" not in op1.send_info
            assert "op1" not in op4.recv_info

            op2.send_info = {"op3": "vec:cub"}
            op2.recv_info = {}
            op3.send_info = {}
            op3.recv_info = {"op2": "vec:cub"}

            op4.send_info = {"op1": "vec:cub"}
            op4.recv_info = {}
            op1.send_info = {}
            op1.recv_info = {"op4": "vec:cub"}

            super_operator.remove_crossed_line_sync()
            assert "op3" not in op2.send_info
            assert "op2" not in op3.recv_info

    @staticmethod
    def test_remove_multi_send_info():
        with mock.patch("json.load", return_value=sub_op_add_json):
            super_operator = SuperOperatorInfos(
                kernel_info, "test_remove_multi_send_info"
            )
            op1 = SubOperatorInfos(0, info_dict, 0, {})
            op1.kernel_name_for_multi_stream = "vec_op1"
            op1.kernel_type = SuperKernelKernelType.KERNEL_TYPE_AIV_ONLY
            op2 = SubOperatorInfos(1, info_dict, 0, {})
            op2.kernel_name_for_multi_stream = "cub_op2"
            op2.kernel_type = SuperKernelKernelType.KERNEL_TYPE_AIC_ONLY
            op3 = SubOperatorInfos(2, info_dict, 0, {})
            op3.kernel_name_for_multi_stream = "cub_op3"
            op3.kernel_type = SuperKernelKernelType.KERNEL_TYPE_AIC_ONLY

            op1.send_info = {"cub_op2": "vec:cub", "cub_op3": "vec:cub"}
            op1.recv_info = {}
            op2.recv_info = {"vec_op1": "vec:cub"}
            op3.recv_info = {"vec_op1": "vec:cub"}
            super_operator.vec_op_list = [op1]
            super_operator.cub_op_list = [op2, op3]
            super_operator.info_base = [op1, op2, op3]
            super_operator.remove_multi_send_info()

            assert "cub_op3" not in op1.send_info
            assert "vec_op1" not in op3.recv_info

            op4 = SubOperatorInfos(3, info_dict, 0, {})
            op4.kernel_name_for_multi_stream = "cub_op4"
            op4.kernel_type = SuperKernelKernelType.KERNEL_TYPE_AIC_ONLY
            op5 = SubOperatorInfos(4, info_dict, 0, {})
            op5.kernel_name_for_multi_stream = "vec_op5"
            op5.kernel_type = SuperKernelKernelType.KERNEL_TYPE_AIV_ONLY
            op6 = SubOperatorInfos(5, info_dict, 0, {})
            op6.kernel_name_for_multi_stream = "vec_op6"
            op6.kernel_type = SuperKernelKernelType.KERNEL_TYPE_AIV_ONLY

            op4.send_info = {"vec_op5": "cub:vec", "vec_op6": "cub:vec"}
            op4.recv_info = {}
            op5.recv_info = {"cub_op4": "cub:vec"}
            op6.recv_info = {"cub_op4": "cub:vec"}
            super_operator.cub_op_list = [op4]
            super_operator.vec_op_list = [op5, op6]
            super_operator.info_base = [op4, op5, op6]
            super_operator.remove_multi_send_info()

            assert "vec_op6" not in op4.send_info
            assert "cub_op4" not in op6.recv_info

    @staticmethod
    def test_remove_multi_recv_info():
        with mock.patch("json.load", return_value=sub_op_add_json):
            super_operator = SuperOperatorInfos(
                kernel_info, "test_remove_multi_recv_info"
            )
            op1 = SubOperatorInfos(0, info_dict, 0, {})
            op1.kernel_name_for_multi_stream = "cub_op1"
            op1.kernel_type = SuperKernelKernelType.KERNEL_TYPE_AIC_ONLY
            op2 = SubOperatorInfos(1, info_dict, 0, {})
            op2.kernel_name_for_multi_stream = "cub_op2"
            op2.kernel_type = SuperKernelKernelType.KERNEL_TYPE_AIC_ONLY
            op3 = SubOperatorInfos(2, info_dict, 0, {})
            op3.kernel_name_for_multi_stream = "vec_op3"
            op3.kernel_type = SuperKernelKernelType.KERNEL_TYPE_AIV_ONLY

            op1.send_info = {"vec_op3": "cub:vec"}
            op2.send_info = {"vec_op3": "cub:vec"}
            op3.recv_info = {"cub_op1": "cub:vec", "cub_op2": "cub:vec"}
            op3.send_info = {}
            super_operator.vec_op_list = [op3]
            super_operator.cub_op_list = [op1, op2]
            super_operator.info_base = [op1, op2, op3]
            super_operator.remove_multi_recv_info()
            assert "vec_op3" not in op1.send_info
            assert "cub_op1" not in op3.recv_info

            op4 = SubOperatorInfos(3, info_dict, 0, {})
            op4.kernel_name_for_multi_stream = "vec_op4"
            op4.kernel_type = SuperKernelKernelType.KERNEL_TYPE_AIV_ONLY
            op5 = SubOperatorInfos(4, info_dict, 0, {})
            op5.kernel_name_for_multi_stream = "vec_op5"
            op5.kernel_type = SuperKernelKernelType.KERNEL_TYPE_AIV_ONLY
            op6 = SubOperatorInfos(5, info_dict, 0, {})
            op6.kernel_name_for_multi_stream = "cub_op6"
            op6.kernel_type = SuperKernelKernelType.KERNEL_TYPE_AIC_ONLY

            op4.send_info = {"cub_op6": "vec:cub"}
            op5.send_info = {"cub_op6": "vec:cub"}
            op6.recv_info = {"vec_op4": "vec:cub", "vec_op5": "vec:cub"}
            op6.send_info = {}
            super_operator.cub_op_list = [op6]
            super_operator.vec_op_list = [op4, op5]
            super_operator.info_base = [op4, op5, op6]
            super_operator.remove_multi_recv_info()
            assert "cub_op6" not in op4.send_info
            assert "vec_op4" not in op6.recv_info

    @staticmethod
    def test_print_vec_cub_list_info():
        with mock.patch.object(
            CommonUtility, "print_compile_log"
        ) as mock_print_compile_log:
            with mock.patch("json.load", return_value=sub_op_add_json):
                super_operator = SuperOperatorInfos(
                    kernel_info, "test_print_vec_cub_list_info"
                )
                op1 = SubOperatorInfos(0, info_dict, 0, {})
                op1.kernel_name_for_multi_stream = "op1"
                op1.stream_index = 0
                op1.send_info = {"op2": "cub:vec"}
                op1.recv_info = {"op3": "vec:cub"}

                op2 = SubOperatorInfos(1, info_dict, 0, {})
                op2.kernel_name_for_multi_stream = "op2"
                op2.stream_index = 1
                op2.send_info = {}
                op2.recv_info = {"op1": "cub:vec"}

                super_operator.vec_op_list = [op1]
                super_operator.cub_op_list = [op2]

                super_operator.print_vec_cub_list_info()
                assert mock_print_compile_log.called

                call_args_list = mock_print_compile_log.call_args_list
                assert "[VEC LIST OP]:" == call_args_list[0][0][1]
                goden_vec_log_str = """\
op_name: op1,                 \
stream_idx: 0, \
send_info: {'op2': 'cub:vec'},                 \
recv_info: {'op3': 'vec:cub'}\
"""
                assert call_args_list[1][0][1] == goden_vec_log_str
                assert "[CUB LIST OP]:" == call_args_list[2][0][1]
                goden_cub_log_str = """\
op_name: op2,                 \
stream_idx: 1, \
send_info: {},                 \
recv_info: {'op1': 'cub:vec'}\
"""
                assert call_args_list[3][0][1] == goden_cub_log_str

    @staticmethod
    def test_optimize_sync_pass():
        with mock.patch("json.load", return_value=sub_op_add_json):
            with mock.patch.object(
                CommonUtility, "print_compile_log"
            ) as mock_print_compile_log:
                super_operator = SuperOperatorInfos(
                    kernel_info, "test_optimize_sync_pass"
                )

                vec_op0 = SubOperatorInfos(0, info_dict, 0, {})
                vec_op0.kernel_name_for_multi_stream = "vec_op0"
                vec_op0.kernel_type = SuperKernelKernelType.KERNEL_TYPE_AIV_ONLY
                cub_op1 = SubOperatorInfos(1, info_dict, 0, {})
                cub_op1.kernel_name_for_multi_stream = "cub_op1"
                cub_op1.kernel_type = SuperKernelKernelType.KERNEL_TYPE_AIC_ONLY
                cub_op2 = SubOperatorInfos(2, info_dict, 0, {})
                cub_op2.kernel_name_for_multi_stream = "cub_op2"
                cub_op2.kernel_type = SuperKernelKernelType.KERNEL_TYPE_AIC_ONLY
                vec_op3 = SubOperatorInfos(3, info_dict, 0, {})
                vec_op3.kernel_name_for_multi_stream = "vec_op3"
                vec_op3.kernel_type = SuperKernelKernelType.KERNEL_TYPE_AIV_ONLY

                vec_op0.send_info = {"cub_op1": "vec:cub", "cub_op2": "vec:cub"}
                vec_op0.recv_info = {}
                cub_op1.send_info = {"vec_op3": "cub:vec"}
                cub_op1.recv_info = {"vec_op0": "vec:cub"}
                cub_op2.send_info = {"vec_op3": "cub:vec"}
                cub_op2.recv_info = {"vec_op0": "vec:cub"}
                vec_op3.send_info = {}
                vec_op3.recv_info = {"cub_op1": "cub:vec", "cub_op2": "cub:vec"}

                super_operator.vec_op_list = [vec_op0, vec_op3]
                super_operator.cub_op_list = [cub_op1, cub_op2]
                super_operator.info_base = [vec_op0, cub_op1, cub_op2, vec_op3]

                super_operator.optimize_sync_pass()
                mock_print_compile_log.assert_called()
                assert vec_op0.send_info == {"cub_op1": "vec:cub"}
                assert vec_op0.recv_info == {}
                assert cub_op1.send_info == {}
                assert cub_op1.recv_info == {"vec_op0": "vec:cub"}
                assert cub_op2.send_info == {"vec_op3": "cub:vec"}
                assert cub_op2.recv_info == {}
                assert vec_op3.send_info == {}
                assert vec_op3.recv_info == {"cub_op2": "cub:vec"}

    @staticmethod
    def test_creat_compile_log(tmp_dir):
        distinct_tag = "distinct_tag_mock"
        with (
            mock.patch("json.load", return_value=sub_op_add_json),
            mock.patch.object(
                CommonUtility, "get_kernel_meta_dir", return_value=tmp_dir
            ),
            mock.patch.object(
                CommonUtility, "get_distinct_filename_tag", return_value=distinct_tag
            ),
            mock.patch(
                "superkernel.super_kernel_op_infos.get_current_build_config",
                return_value="dump_cce",
            ),
        ):
            super_operator = SuperOperatorInfos(kernel_info, "test_creat_compile_log")
            super_operator.creat_compile_log()
            goden_path = os.path.join(
                tmp_dir, super_operator.kernel_name + distinct_tag + ".log"
            )
            assert super_operator.compile_log_path == goden_path

    @staticmethod
    def test_sub_op_connect_set():
        with mock.patch("json.load", return_value=sub_op_add_json):
            super_operator = SuperOperatorInfos(kernel_info, "test_sub_op_connect_set")

            op1 = SubOperatorInfos(0, info_dict, 0, {})
            op1.send_event_list = [100, 101]
            op2 = SubOperatorInfos(1, info_dict, 0, {})
            op2.recv_event_list = [101, 102]

            result = super_operator.sub_op_connect_set(op1, op2)
            assert result == {101}

    @staticmethod
    def test_find_all_inner_event_id_set():
        with mock.patch("json.load", return_value=sub_op_add_json):
            super_operator = SuperOperatorInfos(
                kernel_info, "test_find_all_inner_event_id_set"
            )

            op1 = SubOperatorInfos(0, info_dict, 0, {})
            op1.send_event_list = [100, 101]
            op1.stream_index = 0
            op2 = SubOperatorInfos(1, info_dict, 1, {})
            op2.recv_event_list = [101, 102]
            op2.stream_index = 1
            super_operator.info_base = [op1, op2]
            super_operator.find_all_inner_event_id_set()
            assert super_operator.inner_event_id_set == {101}

            op1.stream_index = 0
            op2.stream_index = 0
            with mock.patch.object(
                CommonUtility, "ascendc_raise_python_err"
            ) as mock_raise:
                super_operator.find_all_inner_event_id_set()
                mock_raise.assert_called()

    @staticmethod
    def test_check_sp_has_two_real_stream():
        with mock.patch("json.load", return_value=sub_op_add_json):
            with mock.patch.object(
                CommonUtility, "ascendc_raise_python_err"
            ) as mock_ascendc_raise_python_err:
                super_operator = SuperOperatorInfos(
                    kernel_info, "test_check_sp_has_two_real_stream"
                )
                op1 = SubOperatorInfos(0, info_dict, 0, {})
                op2 = SubOperatorInfos(1, info_dict, 1, {})
                op1.send_event_list = []
                op1.recv_event_list = []
                op2.send_event_list = []
                op2.recv_event_list = []
                op1.stream_index = 0
                op2.stream_index = 1
                super_operator.info_base = [op1, op2]

                super_operator.stream_fusin_mode = (
                    SuperKernelStreamFusionMode.StreamFusionEnable
                )
                super_operator.check_sp_has_two_real_stream()
                assert super_operator.enable_double_stream

                op1.send_event_list = [100, 101]
                op1.recv_event_list = [100, 101]
                op2.send_event_list = [100, 101]
                op2.recv_event_list = [100, 101]
                op1.stream_index = 0
                op2.stream_index = 0
                super_operator.stream_fusin_mode = (
                    SuperKernelStreamFusionMode.StreamFusionDisable
                )
                super_operator.check_sp_has_two_real_stream()
                mock_ascendc_raise_python_err.assert_called()

    @staticmethod
    def test_get_finale_type_and_block_num():
        with mock.patch("json.load", return_value=sub_op_add_json):
            super_operator = SuperOperatorInfos(
                kernel_info, "test_get_finale_type_and_block_num"
            )

            super_operator.get_finale_type_and_block_num(0b1, 10, 20)
            assert (
                super_operator.kernel_type
                == SuperKernelKernelType.KERNEL_TYPE_MIX_AIV_1_0
            )
            assert super_operator.block_num == 20

            super_operator.get_finale_type_and_block_num(0b10, 10, 20)
            assert (
                super_operator.kernel_type
                == SuperKernelKernelType.KERNEL_TYPE_MIX_AIC_1_0
            )
            assert super_operator.block_num == 10

            super_operator.get_finale_type_and_block_num(0b100, 10, 20)
            assert (
                super_operator.kernel_type
                == SuperKernelKernelType.KERNEL_TYPE_MIX_AIV_1_0
            )
            assert super_operator.block_num == 20

            super_operator.get_finale_type_and_block_num(0b1000, 10, 20)
            assert (
                super_operator.kernel_type
                == SuperKernelKernelType.KERNEL_TYPE_MIX_AIC_1_0
            )
            assert super_operator.block_num == 10

            super_operator.get_finale_type_and_block_num(0b10000, 10, 20)
            assert (
                super_operator.kernel_type
                == SuperKernelKernelType.KERNEL_TYPE_MIX_AIC_1_1
            )
            assert super_operator.block_num == 10

            super_operator.get_finale_type_and_block_num(0b100000, 10, 20)
            assert (
                super_operator.kernel_type
                == SuperKernelKernelType.KERNEL_TYPE_MIX_AIC_1_2
            )

    @staticmethod
    @pytest.mark.parametrize(
        "vector_cores,device_cube_cores,schedule_mode,expected_type,expected_blocks",
        [
            (5, 8, 1, SuperKernelKernelType.KERNEL_TYPE_MIX_AIC_1_1, 5),
            (5, 4, 1, SuperKernelKernelType.KERNEL_TYPE_MIX_AIC_1_2, 3),
            (6, 8, 1, SuperKernelKernelType.KERNEL_TYPE_MIX_AIC_1_2, 3),
            (5, 8, 0, SuperKernelKernelType.KERNEL_TYPE_MIX_AIC_1_2, 3),
        ],
    )
    def test_odd_vector_batchmode_type_selection(
        vector_cores, device_cube_cores, schedule_mode, expected_type, expected_blocks
    ):
        super_operator = SuperOperatorInfos(kernel_info, "test_odd_vector_batchmode")
        super_operator.info_base = [mock.Mock(schedule_mode=schedule_mode)]
        with mock.patch(
            "superkernel.super_kernel_op_infos.get_soc_spec",
            return_value=device_cube_cores,
        ):
            super_operator.get_finale_type_and_block_num(0b11, 3, vector_cores)
        assert super_operator.kernel_type == expected_type
        assert super_operator.block_num == expected_blocks

    @staticmethod
    def test_get_summary_type_and_options(tmp_dir):
        with mock.patch("json.load", return_value=sub_op_add_json):
            with mock.patch.object(
                CommonUtility, "get_kernel_meta_dir", return_value=tmp_dir
            ):
                super_operator = SuperOperatorInfos(
                    kernel_info, "test_get_summary_type_and_options"
                )
                op1 = SubOperatorInfos(0, info_dict, 0, {})
                op1.block_num = 10
                op1.timestamp_option = True
                op1.debug_size = 2
                super_operator.debug_size = 1
                op1.debug_option = "debug_option1,debug_option2"
                super_operator.debug_option = "debug_option1,debug_option3"
                super_operator.info_base = [op1]

                op1.kernel_type = SuperKernelKernelType.KERNEL_TYPE_AIV_ONLY
                super_operator.get_summary_type_and_options()
                assert (
                    super_operator.kernel_type
                    == SuperKernelKernelType.KERNEL_TYPE_MIX_AIV_1_0
                )
                assert (
                    super_operator.debug_option
                    == "debug_option1,debug_option3,debug_option2"
                )

                op1.kernel_type = SuperKernelKernelType.KERNEL_TYPE_AIC_ONLY
                super_operator.get_summary_type_and_options()
                assert (
                    super_operator.kernel_type
                    == SuperKernelKernelType.KERNEL_TYPE_MIX_AIC_1_0
                )
                assert (
                    super_operator.debug_option
                    == "debug_option1,debug_option3,debug_option2"
                )

                op1.kernel_type = SuperKernelKernelType.KERNEL_TYPE_MIX_AIV_1_0
                super_operator.get_summary_type_and_options()
                assert (
                    super_operator.kernel_type
                    == SuperKernelKernelType.KERNEL_TYPE_MIX_AIV_1_0
                )
                assert (
                    super_operator.debug_option
                    == "debug_option1,debug_option3,debug_option2"
                )

                op1.kernel_type = SuperKernelKernelType.KERNEL_TYPE_MIX_AIC_1_0
                super_operator.get_summary_type_and_options()
                assert (
                    super_operator.kernel_type
                    == SuperKernelKernelType.KERNEL_TYPE_MIX_AIC_1_0
                )
                assert (
                    super_operator.debug_option
                    == "debug_option1,debug_option3,debug_option2"
                )

                op1.kernel_type = SuperKernelKernelType.KERNEL_TYPE_MIX_AIC_1_1
                super_operator.get_summary_type_and_options()
                assert (
                    super_operator.kernel_type
                    == SuperKernelKernelType.KERNEL_TYPE_MIX_AIC_1_1
                )
                assert (
                    super_operator.debug_option
                    == "debug_option1,debug_option3,debug_option2"
                )

                op1.kernel_type = SuperKernelKernelType.KERNEL_TYPE_MIX_AIC_1_2
                super_operator.debug_option = ""
                super_operator.get_summary_type_and_options()
                assert (
                    super_operator.kernel_type
                    == SuperKernelKernelType.KERNEL_TYPE_MIX_AIC_1_2
                )
                assert super_operator.debug_option == "debug_option1,debug_option2"

    @staticmethod
    def test_find_sub_kernel_name():
        with mock.patch("json.load", return_value=sub_op_add_json):
            super_operator = SuperOperatorInfos(kernel_info, "test_find_sub_name")
            origin_names = ["kernel_mix_aiv_func", "kernel_mix_aic_func"]
            aiv_name, aic_name = super_operator.find_sub_kernel_name(origin_names)

            assert "kernel_mix_aiv_func" == aiv_name
            assert "kernel_mix_aic_func" == aic_name

    @staticmethod
    def test_split_o_in_super_kernel(tmp_dir):
        super_operator = SuperOperatorInfos(kernel_info, "test_split_o_in_super_kernel")
        tmp_dir_str = str(tmp_dir)
        orign_bin_path = os.path.join(tmp_dir_str, "original.o")
        origin_kernel_name = "origin_kernel_name_mock"

        with (
            mock.patch.object(
                CommonUtility, "get_kernel_meta_dir", return_value=tmp_dir_str
            ),
            mock.patch("subprocess.run"),
            mock.patch("os.path.exists"),
            mock.patch.object(
                CommonUtility, "dump_compile_log"
            ) as mock_dump_compile_log,
        ):
            new_bin_path_goden = os.path.join(tmp_dir_str, "original_split1.o")
            new_kernel_name_goden = f"{origin_kernel_name}_split1"
            new_bin_path, new_kernel_name = super_operator.split_o_in_super_kernel(
                orign_bin_path, origin_kernel_name, 1
            )

            mock_dump_compile_log.assert_called()
            assert new_bin_path_goden == new_bin_path
            assert new_kernel_name_goden == new_kernel_name

        with (
            mock.patch.object(
                CommonUtility, "get_kernel_meta_dir", return_value=tmp_dir_str
            ),
            mock.patch(
                "subprocess.run",
                side_effect=subprocess.CalledProcessError(1, "test_command"),
            ),
            mock.patch.object(CommonUtility, "ascendc_raise_python_err") as mock_raise,
            mock.patch.object(
                CommonUtility, "dump_compile_log"
            ) as mock_dump_compile_log,
        ):
            new_bin_path, new_kernel_name = super_operator.split_o_in_super_kernel(
                orign_bin_path, origin_kernel_name, 1
            )
            mock_dump_compile_log.assert_called()
            mock_raise.assert_called()

    @staticmethod
    def test_gen_super_kernel_params():
        with mock.patch("json.load", return_value=sub_op_add_json):
            super_operator = SuperOperatorInfos(
                kernel_info, "test_gen_super_kernel_params"
            )
            op1 = SubOperatorInfos(0, info_dict, 0, {})
            super_operator.info_base = [op1]
            super_operator.super_kernel_params = ["super_kernel_params"]
            op1.kernel_params = ["sub_kernel_params"]
            op1.extra_kernel_params = ["sub_extra_kernel_params"]
            op1.sub_op_task_type = SubOperatorType.DYNAMIC_OP
            super_operator.gen_super_kernel_params()
            assert super_operator.super_kernel_params == [
                "super_kernel_params",
                "sub_kernel_params",
                "sub_extra_kernel_params",
            ]

            super_operator.super_kernel_params = ["super_kernel_params"]
            op1.kernel_params = ["sub_kernel_params"]
            op1.extra_kernel_params = ["sub_extra_kernel_params"]
            op1.sub_op_task_type = SubOperatorType.STATIC_OP
            super_operator.gen_super_kernel_params()
            assert super_operator.super_kernel_params == [
                "super_kernel_params",
                "sub_kernel_params",
                "sub_extra_kernel_params",
            ]

    @staticmethod
    def test_get_ws_size():
        with mock.patch("json.load", return_value=sub_op_add_json):
            super_operator = SuperOperatorInfos(kernel_info, "test_get_ws_size")
            op1 = SubOperatorInfos(0, info_dict, 0, {})
            op2 = SubOperatorInfos(0, info_dict, 0, {})
            op3 = SubOperatorInfos(0, info_dict, 0, {})
            op4 = SubOperatorInfos(0, info_dict, 0, {})
            super_operator.info_base = [op1, op2, op3, op4]

            super_operator.get_ws_size(1)
            assert super_operator.workspace_size == 1024

    @staticmethod
    def test_calc_workspace_size():
        with mock.patch("json.load", return_value=sub_op_add_json):
            super_operator = SuperOperatorInfos(kernel_info, "test_calc_workspace_size")
            super_operator.kernel_type = SuperKernelKernelType.KERNEL_TYPE_AIC_ONLY
            op1 = SubOperatorInfos(0, info_dict, 0, {})
            op2 = SubOperatorInfos(0, info_dict, 0, {})
            op3 = SubOperatorInfos(0, info_dict, 0, {})
            op4 = SubOperatorInfos(0, info_dict, 0, {})
            super_operator.info_base = [op1, op2, op3, op4]
            super_operator.block_num = 1
            super_operator.feed_sync_all_mode = (
                SuperKernelFeedSyncAllMode.FeedSyncAllEnable
            )

            super_operator.calc_workspace_size()
            assert super_operator.workspace_size == 1024

            super_operator.kernel_type = SuperKernelKernelType.KERNEL_TYPE_MIX_AIC_1_2
            super_operator.calc_workspace_size()
            assert super_operator.workspace_size == 1024

    @staticmethod
    def test_add_define_options():
        with mock.patch("json.load", return_value=sub_op_add_json):
            exist_dynamic_sub_ops = True
            super_operator = SuperOperatorInfos(kernel_info, "test_add_define_options")
            options = []

            super_operator.early_start_mode = (
                SuperKernelEarlyStartMode.EarlyStartEnableV1
            )
            super_operator.feed_sync_all_mode = (
                SuperKernelFeedSyncAllMode.FeedSyncAllEnable
            )
            super_operator.timestamp_option = True
            super_operator.op_options["compile-options"] = "part_option1,part_option2"
            super_operator.add_define_options(exist_dynamic_sub_ops, options)
            assert "-D__ASCENDC_SUPERKERNEL_EARLY_START_V1" in options
            assert "-DASCENDC_DUMP" in options
            assert "-D__ASCENDC_SUPERKERNEL_AUTO_SYNC_ALL__" in options

            super_operator.early_start_mode = (
                SuperKernelEarlyStartMode.EarlyStartEnableV2
            )
            super_operator.timestamp_option = False
            super_operator.add_define_options(exist_dynamic_sub_ops, options)
            assert "-D__ASCENDC_SUPERKERNEL_EARLY_START_V2" in options
            assert "-DASCENDC_DUMP=0" in options

    @staticmethod
    def test_gen_compile_info_aic(tmp_dir):
        with (
            mock.patch("json.load", return_value=sub_op_add_json),
            mock.patch.dict(os.environ, {"BISHENG_REAL_PATH": "/mock/bisheng"}),
            mock.patch("subprocess.run"),
            mock.patch.object(
                CommonUtility, "get_kernel_meta_dir", return_value=tmp_dir
            ),
        ):
            super_operator = SuperOperatorInfos(
                kernel_info, "test_gen_compile_info_aic"
            )
            sub_operator = SubOperatorInfos(0, info_dict, 0, {})
            sub_operator.param_offset = "param_offset1"
            super_operator.info_base = [sub_operator]
            sub_operator.dynamic_bin = None
            sub_operator.split_mode = 2
            sub_operator.split_mode_in_json = 2
            sub_operator.sub_kernel_names = ["kernel_mix_aic"]
            sub_operator.aic_bin = os.path.join("aic_bin_mock", "original.o")
            super_operator.gen_compile_info()
            assert (
                sub_operator.aic_bin
                == super_operator.compile_info["sub_operator"][0]["aic_bin"]
            )
            assert (
                sub_operator.aic_bin[:-2] + "_split1.o"
                == super_operator.compile_info["sub_operator"][1]["aic_bin"]
            )
            assert (
                sub_operator.sub_kernel_names
                == super_operator.compile_info["sub_operator"][0]["sub_kernel_names"]
            )

            sub_operator.split_mode_in_json = None
            sub_operator.aic_bin = os.path.join("aic_bin_mock", "original.o")
            filename = os.path.basename(sub_operator.aic_bin)
            aic_split_o_path_goden = os.path.join(tmp_dir, filename[:-2] + "_split1.o")
            super_operator.gen_compile_info()
            assert (
                aic_split_o_path_goden
                == super_operator.compile_info["sub_operator"][1]["aic_bin"]
            )
            assert (
                "kernel_mix_aic_split1"
                in super_operator.compile_info["sub_operator"][1]["sub_kernel_names"]
            )

            sub_operator.called_kernel_name = {}
            sub_operator.called_kernel_name["dynamic_func_names"] = {
                "tiling_key": {"dav-c220-cube": "cube_kernel_name"}
            }
            tmp_dir_str = str(tmp_dir)
            sub_operator.dynamic_bin = os.path.join(tmp_dir_str, "original.o")
            with (
                mock.patch("os.environ.get", return_value=None),
                mock.patch(
                    "superkernel.super_kernel_op_infos.get_current_build_config",
                    return_value="dump_cce",
                ),
            ):
                super_operator.gen_compile_info()
                assert (
                    sub_operator.dynamic_bin
                    == super_operator.compile_info["sub_operator"][0]["dynamic_bin"]
                )
                assert (
                    sub_operator.dynamic_bin[:-2] + "_split1.o"
                    == super_operator.compile_info["sub_operator"][1]["dynamic_bin"]
                )
                assert ["cube_kernel_name_split1"] == super_operator.compile_info[
                    "sub_operator"
                ][1]["sub_kernel_names"]

    @staticmethod
    def test_gen_compile_info_aiv(tmp_dir):
        with (
            mock.patch("json.load", return_value=sub_op_add_json),
            mock.patch.dict(os.environ, {"BISHENG_REAL_PATH": "/mock/bisheng"}),
            mock.patch("subprocess.run"),
            mock.patch.object(
                CommonUtility, "get_kernel_meta_dir", return_value=tmp_dir
            ),
        ):
            super_operator = SuperOperatorInfos(
                kernel_info, "test_gen_compile_info_aiv"
            )
            sub_operator = SubOperatorInfos(0, info_dict, 0, {})
            sub_operator.param_offset = "param_offset1"
            super_operator.info_base = [sub_operator]
            sub_operator.dynamic_bin = None
            sub_operator.split_mode = 2
            sub_operator.split_mode_in_json = 2

            sub_operator.sub_kernel_names = ["kernel_mix_aiv"]
            sub_operator.aiv_bin = os.path.join("aiv_bin_mock", "original.o")
            super_operator.gen_compile_info()
            assert (
                sub_operator.aiv_bin
                == super_operator.compile_info["sub_operator"][0]["aiv_bin"]
            )
            assert (
                sub_operator.aiv_bin[:-2] + "_split1.o"
                == super_operator.compile_info["sub_operator"][1]["aiv_bin"]
            )
            assert (
                sub_operator.sub_kernel_names
                == super_operator.compile_info["sub_operator"][0]["sub_kernel_names"]
            )

            sub_operator.split_mode_in_json = None
            sub_operator.aiv_bin = os.path.join("aiv_bin_mock", "original.o")
            filename = os.path.basename(sub_operator.aiv_bin)
            aiv_split_o_path_goden = os.path.join(tmp_dir, filename[:-2] + "_split1.o")
            super_operator.gen_compile_info()
            assert (
                aiv_split_o_path_goden
                == super_operator.compile_info["sub_operator"][1]["aiv_bin"]
            )
            assert (
                "kernel_mix_aiv_split1"
                in super_operator.compile_info["sub_operator"][1]["sub_kernel_names"]
            )


if __name__ == "__main__":
    pytest.main()
