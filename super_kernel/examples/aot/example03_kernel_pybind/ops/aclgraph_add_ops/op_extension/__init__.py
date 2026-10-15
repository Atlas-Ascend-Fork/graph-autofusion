# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# ----------------------------------------------------------------------------------------------------------

from pathlib import Path

from . import custom_ops_lib
from ._torch_library import register_torch_ops as register_torch_ops


LOADED_LIBRARY_PATH = Path(custom_ops_lib.__file__).as_posix()
run_add_custom = custom_ops_lib.run_add_custom
