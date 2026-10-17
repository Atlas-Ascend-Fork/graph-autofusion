# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Register shared fixtures for multistream UT and ST."""

from pathlib import Path

import pytest

from multistream.common import runtime_platform

from common.runtime_stub import install


pytest_plugins = [
    "fixtures.graphs",
    "fixtures.profiling",
]


@pytest.fixture(autouse=True)
def stub_runtime(monkeypatch):
    monkeypatch.setattr(runtime_platform, "_get_soc_spec", None)
    query = install(monkeypatch)
    yield query
    monkeypatch.setattr(runtime_platform, "_get_soc_spec", None)


def pytest_collection_modifyitems(items):
    root = Path(__file__).parent
    for item in items:
        path = Path(item.fspath)
        if root / "ut" in path.parents:
            item.add_marker(pytest.mark.ut)
        elif root / "st" in path.parents:
            item.add_marker(pytest.mark.st)
