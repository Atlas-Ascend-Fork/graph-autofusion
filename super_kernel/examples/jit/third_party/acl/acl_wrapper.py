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
# fmt: off

import ctypes
import os
from typing import Type, Any

ACL_SUCCESS = 0


def check_acl_result(func_name=None):
    def decorator(func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            assert result == ACL_SUCCESS, f"{func_name or func.__name__} failed! Return code: {result}"
            return result

        return wrapper

    return decorator


class SharedLibrary:
    """共享库加载基类，提供基础的库加载和函数绑定功能"""

    def __init__(self, lib_name: str, search_paths: list = None):
        self.lib = self._load_library(lib_name, search_paths)
        self._function_cache = {}

    @staticmethod
    def _load_library(lib_name: str, search_paths: list = None) -> ctypes.CDLL:
        """加载共享库，支持指定搜索路径"""
        search_paths = search_paths or []

        # 尝试直接加载
        try:
            return ctypes.CDLL(lib_name)
        except OSError:
            pass

        # 尝试搜索路径
        for path in search_paths:
            full_path = os.path.join(path, lib_name)
            if os.path.exists(full_path):
                try:
                    return ctypes.CDLL(full_path)
                except OSError as e:
                    raise RuntimeError(f"无法加载共享库 {full_path}: {str(e)}")

        # 尝试系统默认路径
        try:
            return ctypes.CDLL(lib_name)
        except OSError as e:
            raise RuntimeError(f"无法找到或加载共享库 {lib_name}: {str(e)}")

    def bind_function(self, func_name: str, argtypes: list, restype: Type, doc: str = "") -> Any:
        """绑定函数并设置参数类型和返回类型"""
        if func_name in self._function_cache:
            return self._function_cache[func_name]

        try:
            func = getattr(self.lib, func_name)
        except AttributeError:
            raise AttributeError(f"共享库中未找到函数 {func_name}")

        func.argtypes = argtypes
        func.restype = restype
        func.__doc__ = doc

        self._function_cache[func_name] = func
        return func


class AscendCLLibrary(SharedLibrary):
    """AscendCL库封装，包含acl相关接口"""

    def __init__(self):
        # 定义libascendcl.so的搜索路径
        search_paths = os.environ["LD_LIBRARY_PATH"].split(":")
        super().__init__("libascendcl.so", search_paths)

        # 绑定常量和枚举
        self._define_constants()
        self._define_enums()

        # 绑定函数
        self._bind_functions()

    def _define_constants(self):
        """定义ACL相关常量"""
        self.acl_success = ACL_SUCCESS
        # 可以添加更多状态码常量

    def _define_enums(self):
        """定义ACL相关枚举类型"""

        # 内存拷贝类型枚举
        class AclrtMemcpyKind:
            ACL_MEMCPY_HOST_TO_HOST = 0
            ACL_MEMCPY_HOST_TO_DEVICE = 1
            ACL_MEMCPY_DEVICE_TO_HOST = 2
            ACL_MEMCPY_DEVICE_TO_DEVICE = 3
            ACL_MEMCPY_DEFAULT = 4
            ACL_MEMCPY_HOST_TO_BUF_TO_DEVICE = 5
            ACL_MEMCPY_INNER_DEVICE_TO_DEVICE = 6
            ACL_MEMCPY_INTER_DEVICE_TO_DEVICE = 7

        # 内存分配策略枚举
        class AclrtMemMallocPolicy:
            ACL_MEM_MALLOC_HUGE_FIRST = 0
            ACL_MEM_MALLOC_HUGE_ONLY = 1
            ACL_MEM_MALLOC_NORMAL_ONLY = 2
            ACL_MEM_MALLOC_HUGE_FIRST_P2P = 3
            ACL_MEM_MALLOC_HUGE_ONLY_P2P = 4
            ACL_MEM_MALLOC_NORMAL_ONLY_P2P = 5
            ACL_MEM_MALLOC_HUGE1G_ONLY = 6
            ACL_MEM_MALLOC_HUGE1G_ONLY_P2P = 7
            ACL_MEM_TYPE_LOW_BAND_WIDTH = 0x0100
            ACL_MEM_TYPE_HIGH_BAND_WIDTH = 0x1000
            ACL_MEM_ACCESS_USER_SPACE_READONLY = 0x100000

        # 捕获模式枚举
        class AclMdlRICaptureMode:
            ACL_MODEL_RI_CAPTURE_MODE_GLOBAL = 0
            ACL_MODEL_RI_CAPTURE_MODE_THREAD_LOCAL = 1
            ACL_MODEL_RI_CAPTURE_MODE_RELAXED = 2

        # 捕获状态枚举
        class ACLMdlRICaptureStatus:
            ACL_MODEL_RI_CAPTURE_STATUS_NONE = 0
            ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE = 1
            ACL_MODEL_RI_CAPTURE_STATUS_INVALIDATED = 2

        self.aclrt_memcpy_kind = AclrtMemcpyKind
        self.aclrt_mem_malloc_policy = AclrtMemMallocPolicy
        self.aclmdl_ri_capture_status = ACLMdlRICaptureStatus
        self.aclmdl_ri_capture_mode = AclMdlRICaptureMode

        # 定义aclmdlRI类型（void*）
        self.aclmdl_ri = ctypes.c_void_p
        self.aclrt_stream = ctypes.c_void_p
        self.aclrt_context = ctypes.c_void_p

    def _bind_functions(self):
        """绑定ACL相关函数"""
        self.acl_init = self.bind_function("aclInit",
                                           [ctypes.c_char_p],  # configPath (const char*，可 NULL)
                                           ctypes.c_int,  # 返回aclError
                                           "初始化 ACL，configPath 为配置路径（可传 None），返回错误码，0 表示成功"
                                           )

        self.acl_finalize = self.bind_function("aclFinalize",
                                               [],  # 无参数
                                               ctypes.c_int,
                                               "释放 ACL 所有资源，进程退出前调用，返回错误码，0 表示成功")

        self.aclrt_set_device = self.bind_function("aclrtSetDevice",
                                                   [ctypes.c_int32],  # deviceId
                                                   ctypes.c_int,
                                                   "指定操作的设备，返回错误码，0 表示成功")

        self.aclrt_reset_device = self.bind_function("aclrtResetDevice",
                                                     [ctypes.c_int32],  # deviceId
                                                     ctypes.c_int,
                                                     "重置设备并释放资源，返回错误码，0 表示成功")

        self.aclrt_create_stream = self.bind_function("aclrtCreateStream",
                                                      [ctypes.POINTER(ctypes.c_void_p)],  # stream*
                                                      ctypes.c_int,
                                                      "创建流实例，返回错误码，0 表示成功")

        self.aclrt_synchronize_stream = self.bind_function("aclrtSynchronizeStream",
                                                           [ctypes.c_void_p],  # stream
                                                           ctypes.c_int,
                                                           "阻塞主机直到流中所有任务完成，返回错误码，0表示成功")

        self.aclrt_destroy_stream = self.bind_function("aclrtDestroyStream",
                                                       [ctypes.c_void_p],  # stream
                                                       ctypes.c_int,  # 返回aclError
                                                       "销毁流实例（仅支持 aclrtCreateStream 创建的流），返回错误码，0 表示成功"
                                                       )

        self.aclrt_create_context = self.bind_function("aclrtCreateContext",
                                                       [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int32],
                                                       # context*, deviceId
                                                       ctypes.c_int, "创建上下文（绑定设备），返回错误码，0 表示成功")

        self.aclrt_destroy_context = self.bind_function("aclrtDestroyContext",
                                                        [ctypes.c_void_p],  # context
                                                        ctypes.c_int,  # 返回aclError
                                                        "销毁上下文（仅支持 aclrtCreateContext 创建的上下文），返回错误码，0 表示成功")

        self.aclrt_malloc_host = self.bind_function("aclrtMallocHost",
                                                    [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t],
                                                    # hostPtr*, size
                                                    ctypes.c_int,
                                                    "分配主机内存（需用 aclrtFree 释放），返回错误码，0 表示成功")

        self.aclrt_malloc_align32 = self.bind_function("aclrtMallocAlign32",
                                                       [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_int],
                                                       # devPtr*, size, policy
                                                       ctypes.c_int,
                                                       "在设备上分配 32 字节对齐内存，返回错误码，0 表示成功")

        self.aclrt_free = self.bind_function("aclrtFree",
                                             [ctypes.c_void_p],  # devPtr
                                             ctypes.c_int,
                                             "释放设备内存（需是 aclrtMalloc 系列分配的内存），返回错误码，0 表示成功")

        self.aclrt_memcpy = self.bind_function("aclrtMemcpy",
                                               [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t,
                                                ctypes.c_int],
                                               # dst, destMax, src, count, kind
                                               ctypes.c_int,
                                               "同步内存拷贝（阻塞直到完成），返回错误码，0 表示成功")
        self.aclrt_memcpy_async = self.bind_function("aclrtMemcpyAsync",
                                                     [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
                                                      ctypes.c_size_t,
                                                      ctypes.c_int, ctypes.c_void_p],
                                                     # dst, destMax, src, count, kind, stream
                                                     ctypes.c_int, "异步内存拷贝（需配合流同步），返回错误码，0 表示成功")

        # aclmdlRICaptureBegin函数
        self.aclmdl_ri_capture_begin = self.bind_function("aclmdlRICaptureBegin",
                                                          [ctypes.c_void_p, ctypes.c_int],  # stream, mode
                                                          ctypes.c_int,  # 返回aclError
                                                          "开始捕获，返回错误码，0表示成功"
                                                          )

        # aclmdlRICaptureEnd函数
        self.aclmdl_ri_capture_end = self.bind_function("aclmdlRICaptureEnd",
                                                        [
                                                            ctypes.c_void_p,  # stream
                                                            ctypes.POINTER(ctypes.c_void_p)  # modelRI (aclmdlRI*)
                                                        ],
                                                        ctypes.c_int,  # 返回aclError
                                                        "结束流捕获并获取模型，返回错误码，0表示成功"
                                                        )

        # aclmdlRIExecuteAsync函数
        self.aclmdl_ri_execute_async = self.bind_function("aclmdlRIExecuteAsync",
                                                          [
                                                              ctypes.c_void_p,  # modelRI
                                                              ctypes.c_void_p  # stream
                                                          ],
                                                          ctypes.c_int,  # 返回aclError
                                                          "异步执行模型推理，返回错误码，0表示成功"
                                                          )

        # aclmdlRIExecute函数
        self.aclmdl_ri_execute = self.bind_function("aclmdlRIExecute",
                                                    [ctypes.c_void_p,  # modelRI
                                                     ctypes.c_int32  # timeout (ms)
                                                     ],
                                                    ctypes.c_int,  # 返回aclError
                                                    "同步执行模型实例，返回错误码，0表示成功"
                                                    )

        # aclmdlRIDestroy函数
        self.aclmdl_ri_destroy = self.bind_function("aclmdlRIDestroy",
                                                    [ctypes.c_void_p],  # modelRI
                                                    ctypes.c_int,  # 返回aclError
                                                    "销毁模型，返回错误码，0表示成功"
                                                    )


class RuntimeLibrary(SharedLibrary):
    """Runtime库封装，包含libruntime.so中的接口"""

    def __init__(self):
        # 定义libruntime.so的搜索路径
        search_paths = os.environ["LD_LIBRARY_PATH"].split(":")
        super().__init__("libruntime.so", search_paths)

        # 绑定常量和枚举
        self._define_constants()
        self._define_enums()

        # 绑定函数
        self._bind_functions()

    def _define_constants(self):
        """定义Runtime相关常量"""
        pass

    def _define_enums(self):
        """定义Runtime相关枚举类型"""

        class RtDevBinary_t(ctypes.Structure):
            """rtDevBinary_t结构体定义"""
            _fields_ = [
                ("magic", ctypes.c_uint32),  # magic number
                ("version", ctypes.c_uint32),  # version of binary
                ("data", ctypes.c_void_p),  # binary data
                ("length", ctypes.c_uint64)  # binary length
            ]

        class RtFuncModeType:
            FUNC_MODE_NORMAL = 0
            FUNC_MODE_PCTRACE_USERPROFILE_RECORDLOOP = 1
            FUNC_MODE_PCTRACE_USERPROFILE_SKIPLOOP = 2
            FUNC_MODE_PCTRACE_CYCLECNT_RECORDLOOP = 3
            FUNC_MODE_PCTRACE_CYCLECNT_SKIPLOOP = 4
            FUNC_MODE_BUTT = 5

        class RtSmData(ctypes.Structure):
            """rtSmData_t结构体占位定义（具体字段未提供）"""
            # 实际使用时需要根据完整定义补充字段
            _fields_ = [
                ("L2_mirror_addr", ctypes.c_uint64),  # preload or swap source addr
                ("L2_data_section_size", ctypes.c_uint32),  # every data size
                ("L2_preload", ctypes.c_uint8),  # 1 - preload from mirrorAddr, 0 - no preload
                ("modified", ctypes.c_uint8),  # 1 - data will be modified by kernel, 0 - no modified
                ("priority", ctypes.c_uint8),  # data priority
                ("prev_L2_page_offset_base", ctypes.c_int8),  # remap source section offset
                ("L2_page_offset_base", ctypes.c_uint8),  # remap destination section offset
                ("L2_load_to_ddr", ctypes.c_uint8),  # 1 - need load out, 0 - no need
                ("reserved", ctypes.c_uint8 * 2),  # reserved
            ]

        class RtSmDesc(ctypes.Structure):
            """rtSmDesc_t结构体定义"""
            _fields_ = [
                ("data", RtSmData * 8),  # 数据描述符数组
                ("size", ctypes.c_uint64),  # 最大页数量
                ("remap", ctypes.c_uint8 * 64),  # 重映射数组
                ("l2_in_main", ctypes.c_uint8),  # 0-DDR, 1-L2
                ("reserved", ctypes.c_uint8 * 3)  # 预留字段
            ]

        self.rt_dev_binary = RtDevBinary_t
        self.rt_func_mode_type = RtFuncModeType
        self.rt_sm_data = RtSmData
        self.rt_sm_desc = RtSmDesc

    def _bind_functions(self):
        """绑定Runtime相关函数"""
        self.rt_dev_binary_register = self.bind_function(
            "rtDevBinaryRegister",
            [
                ctypes.POINTER(self.rt_dev_binary),  # const rtDevBinary_t *bin
                ctypes.POINTER(ctypes.c_void_p)  # void **hdl
            ],
            ctypes.c_int,  # 返回rtError_t
            "注册算子二进制，返回错误码，RT_ERROR_NONE表示成功"
        )

        self.rt_function_register = self.bind_function(
            "rtFunctionRegister",
            [
                ctypes.c_void_p,  # void *binHandle
                ctypes.c_void_p,  # const void *stubFunc
                ctypes.POINTER(ctypes.c_char),  # const char_t *stubName
                ctypes.c_void_p,  # const void *kernelInfoExt
                ctypes.c_uint32  # uint32_t funcMode
            ],
            ctypes.c_int,  # 返回rtError_t
            "注册设备函数，返回错误码，RT_ERROR_NONE表示成功"
        )

        self.rt_get_function_by_name = self.bind_function(
            "rtGetFunctionByName",
            [
                ctypes.POINTER(ctypes.c_char),  # const char_t *stubName
                ctypes.POINTER(ctypes.c_void_p)  # void **stubFunc
            ],
            ctypes.c_int,  # 返回rtError_t
            "根据函数名查找device上算子执行函数，返回错误码，RT_ERROR_NONE表示成功"
        )

        # 绑定rtKernelLaunch函数
        self.rt_kernel_launch = self.bind_function(
            "rtKernelLaunch",
            [
                ctypes.c_void_p,  # rtFuncHandle funcHandle
                ctypes.c_uint32,  # uint32_t blockDim
                ctypes.c_void_p,  # args
                ctypes.c_uint32,  # argsSize
                ctypes.POINTER(self.rt_sm_desc),  # rtSmDesc_t *smDesc
                ctypes.c_void_p  # rtStream_t stm
            ],
            ctypes.c_int,  # 返回rtError_t
            "启动内核函数，返回错误码，RT_ERROR_NONE表示成功"
        )

        self.rt_get_c2c_ctrl_addr = self.bind_function(
            "rtGetC2cCtrlAddr",
            [
                ctypes.POINTER(ctypes.c_uint64),  # uint64_t *addr
                ctypes.POINTER(ctypes.c_uint32)  # uint32_t *len
            ],
            ctypes.c_int,  # 返回rtError_t
            "获取C2C控制地址，返回错误码，RT_ERROR_NONE表示成功"
        )

        """注销算子二进制，返回错误码，RT_ERROR_NONE表示成功"""
        self.rt_dev_binary_unregister = self.bind_function(
            "rtDevBinaryUnRegister",
            [
                ctypes.c_void_p  # void *hdl
            ],
            ctypes.c_int,  # 返回rtError_t
            "注销算子二进制，返回错误码，RT_ERROR_NONE表示成功"
        )


class ACLWrapper:
    """ACL接口封装入口，主要用于封装Runtime接口以及acl接口，后续如果runtime接口通过acl暴露，可以做到外层脚本侧不感知"""

    def __init__(self):
        # 初始化各个共享库
        self.ascend_cl = AscendCLLibrary()
        self.runtime = RuntimeLibrary()

        self.aclrt_dev_binary = self.runtime.rt_dev_binary
        self.aclrt_func_mode_type = self.runtime.rt_func_mode_type
        self.aclrt_sm_desc = self.runtime.rt_sm_desc

        self.acl_success = self.ascend_cl.acl_success

        self.aclmdl_ri_capture_mode = self.ascend_cl.aclmdl_ri_capture_mode
        self.aclmdl_ri_capture_status = self.ascend_cl.aclmdl_ri_capture_status
        self.aclmdl_ri = self.ascend_cl.aclmdl_ri
        self.aclrt_mem_malloc_policy = self.ascend_cl.aclrt_mem_malloc_policy
        self.aclrt_memcpy_kind = self.ascend_cl.aclrt_memcpy_kind

    @check_acl_result("acl_init")
    def acl_init(self, config_path=None):
        """初始化 ACL：config_path 为配置文件路径（字符串），None 表示使用默认配置"""
        # 转换 Python 字符串为 C 兼容的 char*（None 对应 NULL）
        c_config_path = ctypes.c_char_p(config_path.encode('utf-8')) if config_path else None
        return self.ascend_cl.acl_init(c_config_path)

    @check_acl_result("acl_finalize")
    def acl_finalize(self):
        """释放 ACL 所有资源：进程退出前必须调用，之后无法再使用 ACL 服务"""
        return self.ascend_cl.acl_finalize()

    @check_acl_result("aclrt_set_device")
    def aclrt_set_device(self, device_id):
        """指定操作的设备"""
        return self.ascend_cl.aclrt_set_device(ctypes.c_int32(device_id))

    @check_acl_result("aclrt_reset_device")
    def aclrt_reset_device(self, device_id):
        """重置指定设备：释放设备上所有资源（含默认上下文 / 流），建议先销毁显式创建的流 / 上下文"""
        return self.ascend_cl.aclrt_reset_device(ctypes.c_int32(device_id))

    @check_acl_result("aclrt_create_stream")
    def aclrt_create_stream(self, stream):
        """创建流实例，返回 (错误码，流实例)"""
        return self.ascend_cl.aclrt_create_stream(stream)

    @check_acl_result("aclrt_synchronize_stream")
    def aclrt_synchronize_stream(self, stream):
        """阻塞主机，直到指定流中所有任务完成"""
        return self.ascend_cl.aclrt_synchronize_stream(stream)

    @check_acl_result("aclrt_destroy_stream")
    def aclrt_destroy_stream(self, stream):
        """销毁流实例（销毁前需确保流中任务已完成，建议先调用 aclrt_synchronize_stream）"""
        return self.ascend_cl.aclrt_destroy_stream(stream)

    @check_acl_result("aclrt_malloc_host")
    def aclrt_malloc_host(self, host_ptr, size):
        """分配host内存（可用于 Host 与 Device 间数据交互）返回：(错误码，主机内存指针)，内存需用 aclrt_free 释放"""
        return self.ascend_cl.aclrt_malloc_host(host_ptr, ctypes.c_size_t(size))

    @check_acl_result("aclrt_create_context")
    def aclrt_create_context(self, context, device_id):
        """创建上下文（上下文是设备资源管理的容器，绑定到指定设备）返回：(错误码，上下文实例)"""
        return self.ascend_cl.aclrt_create_context(context, ctypes.c_int32(device_id))

    @check_acl_result("aclrt_destroy_context")
    def aclrt_destroy_context(self, context):
        """销毁上下文（销毁前需确保：1. 上下文内所有流已销毁 2. 流中任务已完成）"""
        return self.ascend_cl.aclrt_destroy_context(context)

    @check_acl_result("aclrt_malloc_align32")
    def aclrt_malloc_align32(self, dev_ptr, size, policy):
        """分配 32 字节对齐的设备内存，返回 (错误码，设备内存指针)"""
        return self.ascend_cl.aclrt_malloc_align32(dev_ptr, ctypes.c_size_t(size), policy)

    @check_acl_result("aclrt_free")
    def aclrt_free(self, dev_ptr):
        """释放设备内存（仅支持 aclrtMalloc 系列分配的内存）"""
        return self.ascend_cl.aclrt_free(dev_ptr)

    @check_acl_result("aclrt_memcpy")
    def aclrt_memcpy(self, dst_ptr, dst_max, src_ptr, count, kind):
        """同步内存拷贝（阻塞），支持 Host/Device 间各种组合"""
        return self.ascend_cl.aclrt_memcpy(dst_ptr, ctypes.c_size_t(dst_max), src_ptr, ctypes.c_size_t(count), kind)

    @check_acl_result("aclrt_memcpy_async")
    def aclrt_memcpy_async(self, dst_ptr, dst_max, src_ptr, count, kind, stream):
        """异步内存拷贝（非阻塞），需调用 aclrtSynchronizeStream 确保完成"""
        return self.ascend_cl.aclrt_memcpy_async(dst_ptr, ctypes.c_size_t(dst_max), src_ptr, ctypes.c_size_t(count),
                                                 kind, stream)

    @check_acl_result("aclmdl_ri_capture_begin")
    def aclmdl_ri_capture_begin(self, stream, mode):
        """开始捕获"""
        return self.ascend_cl.aclmdl_ri_capture_begin(stream, mode)

    @check_acl_result("aclmdl_ri_capture_end")
    def aclmdl_ri_capture_end(self, stream, model_ri):
        """结束流捕获并获取模型"""
        return self.ascend_cl.aclmdl_ri_capture_end(stream, model_ri)

    @check_acl_result("aclmdl_ri_execute_async")
    def aclmdl_ri_execute_async(self, model_ri, stream):
        """异步执行模型推理"""
        return self.ascend_cl.aclmdl_ri_execute_async(model_ri, stream)

    @check_acl_result("aclmdl_ri_execute")
    def aclmdl_ri_execute(self, model_ri, timeout):
        """同步执行模型实例"""
        return self.ascend_cl.aclmdl_ri_execute(model_ri, ctypes.c_int32(timeout))

    @check_acl_result("aclmdl_ri_destroy")
    def aclmdl_ri_destroy(self, model_ri):
        """销毁模型"""
        return self.ascend_cl.aclmdl_ri_destroy(model_ri)

    @check_acl_result("aclrt_dev_binary_register")
    def aclrt_dev_binary_register(self, binary_ptr, handle):
        """算子二进制注册到device"""
        return self.runtime.rt_dev_binary_register(binary_ptr, handle)

    @check_acl_result("aclrt_function_register")
    def aclrt_function_register(self, bin_handle, stub_func, stub_name, kernel_info_ext, func_mode):
        """注册内核"""
        return self.runtime.rt_function_register(bin_handle, stub_func, stub_name, kernel_info_ext, func_mode)

    @check_acl_result("aclrt_get_function_by_name")
    def aclrt_get_function_by_name(self, stub_name, stub_func):
        """根据函数名字获取函数句柄"""
        return self.runtime.rt_get_function_by_name(stub_name, stub_func)

    @check_acl_result("aclrt_get_c2c_ctrl_addr")
    def aclrt_get_c2c_ctrl_addr(self, addr, length):
        """获取C2C控制地址"""
        return self.runtime.rt_get_c2c_ctrl_addr(addr, length)

    @check_acl_result("aclrt_kernel_launch")
    def aclrt_kernel_launch(self, func_handle, block_dim, args, args_size, sm_desc, stream):
        """启动内核函数"""
        return self.runtime.rt_kernel_launch(func_handle, block_dim, args, args_size, sm_desc, stream)

    @check_acl_result("aclrt_dev_binary_unregister")
    def aclrt_dev_binary_unregister(self, binary_handle):
        """算子二进制卸载"""
        return self.runtime.rt_dev_binary_unregister(binary_handle)


acl = ACLWrapper()
# fmt: on
