# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details and restrictions.

"""ATT DFX configuration helpers for device regression scripts."""

import os


def flags(environment):
    return dict(
        item.split("=", 1)
        for item in environment.get("AUTOFUSE_DFX_FLAGS", "").split(";")
        if "=" in item
    )


def selected_plan():
    return flags(os.environ).get("--force_schedule_result", "<unpinned>")


def environment_with_option(name, value):
    environment = dict(os.environ)
    options = [
        item
        for item in environment.get("AUTOFUSE_DFX_FLAGS", "").split(";")
        if item and item.split("=", 1)[0] != f"--{name}"
    ]
    options.append(f"--{name}={value}")
    environment["AUTOFUSE_DFX_FLAGS"] = ";".join(options)
    return environment


def environment_for_plan(plan):
    return environment_with_option("force_schedule_result", plan)


def environment_for_mlir_config(config):
    return environment_with_option("mlir_codegen_config", config)
