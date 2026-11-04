#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms of
# the CANN Open Software License Agreement Version 2.0 (the "License").
# ----------------------------------------------------------------------------------------------------------------------

"""AllLoad 分组离线回归：不导入 torch，也不需要 NPU。"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cases
import run as entry
from support import case_status
import load_min_store_allload_tail1 as tail1


class AllLoadCasesTest(unittest.TestCase):
    def test_tail1_registered_as_ar_allload_plan2(self):
        case = cases.by_id()["allload_ar_tail1"]
        self.assertEqual(case.script, "load_min_store_allload_tail1.py")
        self.assertEqual(case.env, cases.dfx(2))
        result = case_status.CaseResult(
            "LOAD_MIN_STORE_ALLLOAD_TAIL1", "", "", case_status.PASS, "", "", ""
        )
        outcome = entry.Outcome(case, 0, [result], Path("unused.log"), 0)
        self.assertEqual(
            entry.reconcile(outcome),
            [(case.id, case_status.PASS, case_status.PASS, "match")],
        )
        self.assertEqual(tail1.INPUT_SHAPE_CASES, (((4, 15, 1),),))

        # 验证归约仍为原始图的 dim=1，并保持输出 rank；无需 torch。
        class FakeTorch:
            class nn:
                Module = object

            @staticmethod
            def amin(x, **kwargs):
                return x, kwargs

        self.assertEqual(
            tail1.build_module(FakeTorch).forward("x"),
            ("x", {"dim": 1, "keepdim": True}),
        )


if __name__ == "__main__":
    unittest.main()
