#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""
super kernel feature manager
"""

import threading
from abc import ABC, abstractmethod
from typing import Dict, Type, Callable, Optional
from asc_op_compile_base.common.utils.log_utils import LogUtil, AscendCLogLevel


class BaseFeature(ABC):
    """
    base feature class
    """

    def __init__(self, feature_name: str, feature_version: int):
        if not isinstance(feature_name, str):
            raise TypeError(
                f"feature_name's type must be str, current type is {type(feature_name).__name__}"
            )
        if not isinstance(feature_version, int):
            raise TypeError(
                f"feature_version's type must be int, current type is {type(feature_version).__name__}"
            )
        self.feature_name = feature_name
        self.feature_version = feature_version

    def get_intersection_version(self, others_value: int) -> int:
        """
        Get the minimum version between current feature version and another version.

        Args:
            other_version: The version to compare with.

        Return:
            The minimum version of the two.
        """
        return min(self.feature_version, others_value)

    @abstractmethod
    def get_feature_version_of_value(self, feature_value: any) -> int:
        """
        Abstract method to get feature version from feature value.
        """
        pass

    @abstractmethod
    def get_feature_value_of_version(self, feature_version: int) -> any:
        """
        Abstract method to get feature value from feature version.
        """
        pass


class SuperKernelFeatureManager:
    """
    super kernel feature manager
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if not cls._instance:
                cls._instance = super(SuperKernelFeatureManager, cls).__new__(
                    cls, *args, **kwargs
                )
        return cls._instance

    def __init__(self, use_ordered: bool = False):
        # store registered feature class: {feature_name, feature_class_instance}
        self._support_features_instances: Dict[str, BaseFeature] = {}
        self._available_feature_version_map: dict = {}
        self._enable_features_base: dict = {}

    def init_available_and_enable_features(self):
        input_features = {}
        try:
            from importlib import import_module

            ascendc_compile_impl = (
                "asc_op_compile_base.asc_op_compiler.ascendc_kernel_feature_manager"
            )
            ascendc_compile_module = import_module(ascendc_compile_impl)
            get_ascendc_feature_versions_func: Optional[Callable] = getattr(
                ascendc_compile_module, "get_features", None
            )
            if get_ascendc_feature_versions_func is None:
                LogUtil.print_compile_log(
                    "Super Kernel Feature Manager",
                    f"Module '{ascendc_compile_impl}' does not have a 'get_features' function.",
                    AscendCLogLevel.LOG_ERROR,
                )
                raise AttributeError(
                    f"Module '{ascendc_compile_impl}' does not have a 'get_features' function."
                )
            input_features = get_ascendc_feature_versions_func()
            LogUtil.print_compile_log(
                "Super Kernel Feature Manager",
                f"get ascendc feature versions success, {input_features}",
                AscendCLogLevel.LOG_INFO,
            )
        except Exception as e:
            LogUtil.print_compile_log(
                "Super Kernel Feature Manager",
                f"import {ascendc_compile_impl} failed, \
                please install the related software version, i.e. cann, opp, graph-autofusion.",
                AscendCLogLevel.LOG_ERROR,
            )
            raise e

        for feature_name in input_features.keys():
            if feature_name not in self._support_features_instances.keys():
                LogUtil.print_compile_log(
                    "Super Kernel Feature Manager",
                    f"{feature_name} is not supported in ascendc compile, please upgrade the CANN version",
                    AscendCLogLevel.LOG_INFO,
                )
            else:
                feature_instance = self._support_features_instances[feature_name]
                self._available_feature_version_map[feature_name] = (
                    feature_instance.get_intersection_version(
                        input_features[feature_name]
                    )
                )

                self._enable_features_base[feature_name] = (
                    feature_instance.get_feature_value_of_version(
                        self._available_feature_version_map[feature_name]
                    )
                )
                LogUtil.print_compile_log(
                    "Super Kernel Feature Manager",
                    f"{feature_name} available feature version is {self._available_feature_version_map[feature_name]}, \
                    enable feature value is {self._enable_features_base[feature_name]}",
                    AscendCLogLevel.LOG_INFO,
                )

    def check_feature_valid(self, feature_name: str, feature_value: any):
        available_feature_verison = self._available_feature_version_map[feature_name]
        feature_instance = self._support_features_instances[feature_name]
        return (
            feature_instance.get_feature_version_of_value(feature_value)
            <= available_feature_verison
        )

    def register_feature(self, feature_cls: Type[BaseFeature]) -> None:
        """
        register feature class
        :param feature_cls: feature class base on BaseFeature
        :raises ValueError: if duplicated or unimplemented
        """
        if not issubclass(feature_cls, BaseFeature):
            raise ValueError(
                f"class {feature_cls.__name__} must be inherited from BaseFeature"
            )

        instance = feature_cls()
        if instance.feature_name in self._support_features_instances:
            raise ValueError(f"feature {instance.feature_name} already existed")
        self._support_features_instances[instance.feature_name] = instance

    def unregister_feature(self, feature_name: str) -> None:
        if feature_name not in self._support_features_instances:
            raise KeyError(f"feature {feature_name} does not exist")

        self._support_features_instances.pop(feature_name)

    def set_feature_value(self, feature_name: str, feature_value: any):
        """
        set feature value
        if the input passes the verification, the input value is used,
        otherwise, the default value is used.
        """
        if feature_name not in self._enable_features_base.keys():
            LogUtil.print_compile_log(
                "Super Kernel Feature Manager",
                f"{feature_name} is not supported in ascendc compile, \
                    please upgrade the related software version, i.e. cann, opp, graph-autofusion.",
                AscendCLogLevel.LOG_INFO,
            )
            return
        if self.check_feature_valid(feature_name, feature_value):
            self._enable_features_base[feature_name] = feature_value
        else:
            LogUtil.print_compile_log(
                "Super Kernel Feature Manager",
                f"feature {feature_name} does not support {feature_value}, \
                currently supported version is {self._available_feature_version_map[feature_name]}, \
                please upgrade the related software version, \
                i.e. cann, opp, graph-autofusion.",
                AscendCLogLevel.LOG_INFO,
            )

    def get_feature_value(self, feature_name: str) -> any:
        """
        get feature value according to feature name
        """
        if feature_name not in self._enable_features_base.keys():
            LogUtil.print_compile_log(
                "Super Kernel Feature Manager",
                f"{feature_name} is not supported in ascendc compile, \
                    please upgrade the related software version, i.e. cann, opp, graph-autofusion.",
                AscendCLogLevel.LOG_INFO,
            )
            return None
        return self._enable_features_base[feature_name]

    def get_available_feature_versions(self) -> dict:
        return self._available_feature_version_map.copy()


def register_feature(manager: SuperKernelFeatureManager):
    """
    decorator: used to quickly register a feature class to a specified manager
    :param manager: feature manager instance
    example
    @register_feature(global_super_kernel_feature_manager)
    class FeatureExample(BaseFeature):
        def __init__(self):
            super().__init__("test_feature", 1) # feature value

        def get_feature_version_of_value(self, feature_value: any) -> int:
            if feature_value == True:
                return 1
            else:
                return 0

        def get_feature_value_of_version(self, feature_version: int) -> any:
            if feature_version == 1:
                return True
            elif feature_version == 0:
                return False
            else:
                raise ValueError(f"{self.feature_name} does not support {feature_version}")
    """

    def decorator(feature_cls: Type[BaseFeature]) -> Type[BaseFeature]:
        manager.register_feature(feature_cls)
        return feature_cls

    return decorator


global_super_kernel_feature_manager = SuperKernelFeatureManager()


def get_features():
    global_super_kernel_feature_manager.init_available_and_enable_features()
    return global_super_kernel_feature_manager.get_available_feature_versions()
