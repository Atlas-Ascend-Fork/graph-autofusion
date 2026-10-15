/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file acl_rt_compile.h
 * @brief Stub file for aclrtc interface (Unit Test)
 *
 * This file provides stub implementations for aclrtc functions used in unit tests.
 * Based on /usr/local/Ascend/cann/include/acl/acl_rt_compile.h
 */

#ifndef ASCENDC_ACL_RT_COMPILE_H
#define ASCENDC_ACL_RT_COMPILE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int aclError;

// Stub error codes
#define ACL_SUCCESS 0
#define ACL_ERROR_INVALID_PARAM 100000

// Stub types
typedef void *aclrtcProg;
typedef void *aclrtBinHandle;
typedef void *aclrtFuncHandle;

// Binary load options (from acl.h)
typedef enum aclrtBinaryLoadOptionType { ACL_RT_BINARY_LOAD_OPT_LAZY_MAGIC = 0 } aclrtBinaryLoadOptionType;

#define ACL_RT_BINARY_MAGIC_ELF_AICORE 0x12345678

typedef struct aclrtBinaryLoadOption {
  aclrtBinaryLoadOptionType type;
  union {
    uint64_t magic;
  } value;
} aclrtBinaryLoadOption;

typedef struct aclrtBinaryLoadOptions {
  uint32_t numOpt;
  aclrtBinaryLoadOption *options;
} aclrtBinaryLoadOptions;

/**
 * @brief Creates an instance of aclrtcProg with the given input parameters.
 * @param[out] prog Runtime Compilation program.
 * @param[in] src Program source.
 * @param[in] name Program name. "default_program" is used when name is "".
 * @param[in] numHeaders Currently must be 0. Header support not implemented.
 * @param[in] headers Currently must be NULL.
 * @param[in] includeNames Currently must be NULL.
 * @return aclError: ACL_RTC_SUCCESS or ACL_ERROR_RTC_XXX
 */
static inline aclError aclrtcCreateProg(aclrtcProg *prog, const char *src, const char *name, int numHeaders,
                                        const char **headers, const char **includeNames) {
  (void)src;
  (void)name;
  (void)numHeaders;
  (void)headers;
  (void)includeNames;
  if (prog) *prog = nullptr;
  return ACL_SUCCESS;
}

/**
 * @brief Compiles the given program.
 * @param[in] prog Runtime Compilation program.
 * @param[in] numOptions Number of compiler options.
 * @param[in] options Array of option strings.
 * @return aclError: ACL_RTC_SUCCESS or ACL_ERROR_RTC_XXX
 */
static inline aclError aclrtcCompileProg(aclrtcProg prog, int numOptions, const char **options) {
  (void)prog;
  (void)numOptions;
  (void)options;
  return ACL_SUCCESS;
}

/**
 * @brief Destroys the given program.
 * @param[in,out] prog Runtime Compilation program.
 * @return aclError: ACL_RTC_SUCCESS or ACL_ERROR_RTC_XXX
 */
static inline aclError aclrtcDestroyProg(aclrtcProg *prog) {
  if (prog) *prog = nullptr;
  return ACL_SUCCESS;
}

/**
 * @brief Retrieves the compiled device ELF binary.
 * @param[in] prog Runtime Compilation program.
 * @param[out] binData Compiled result.
 * @return aclError: ACL_RTC_SUCCESS or ACL_ERROR_RTC_XXX
 */
static inline aclError aclrtcGetBinData(aclrtcProg prog, char *binData) {
  (void)prog;
  (void)binData;
  return ACL_SUCCESS;
}

/**
 * @brief Retrieves the size of the compiled device ELF binary.
 * @param[in] prog Runtime Compilation program.
 * @param[out] binDataSizeRet Size of the ELF binary.
 * @return aclError: ACL_RTC_SUCCESS or ACL_ERROR_RTC_XXX
 */
static inline aclError aclrtcGetBinDataSize(aclrtcProg prog, size_t *binDataSizeRet) {
  (void)prog;
  if (binDataSizeRet) *binDataSizeRet = 0;
  return ACL_SUCCESS;
}

/**
 * @brief Retrieves the size of the compilation log.
 * @param[in] prog Runtime Compilation program.
 * @param[out] logSizeRet Size of the log string.
 * @return aclError: ACL_RTC_SUCCESS or ACL_ERROR_RTC_XXX
 */
static inline aclError aclrtcGetCompileLogSize(aclrtcProg prog, size_t *logSizeRet) {
  (void)prog;
  if (logSizeRet) *logSizeRet = 0;
  return ACL_SUCCESS;
}

/**
 * @brief Retrieves the compilation log.
 * @param[in] prog Runtime Compilation program.
 * @param[out] log Compilation log.
 * @return aclError: ACL_RTC_SUCCESS or ACL_ERROR_RTC_XXX
 */
static inline aclError aclrtcGetCompileLog(aclrtcProg prog, char *log) {
  (void)prog;
  (void)log;
  return ACL_SUCCESS;
}

/**
 * @brief Loads binary data with options (from acl.h).
 * @param[in] data Binary data pointer.
 * @param[in] size Binary data size.
 * @param[in] options Load options.
 * @param[out] bin Binary handle.
 * @return aclError: ACL_SUCCESS or error code
 */
static inline aclError aclrtBinaryLoadFromData(void *data, size_t size, aclrtBinaryLoadOptions *options,
                                               aclrtBinHandle *bin) {
  (void)data;
  (void)size;
  (void)options;
  if (bin) *bin = reinterpret_cast<aclrtBinHandle>(0x1000);
  return ACL_SUCCESS;
}

/**
 * @brief Unloads binary.
 * @param[in] bin Binary handle.
 * @return aclError: ACL_SUCCESS or error code
 */
static inline aclError aclrtBinaryUnLoad(aclrtBinHandle bin) {
  (void)bin;
  return ACL_SUCCESS;
}

#ifdef __cplusplus
}
#endif

#endif  // ASCENDC_ACL_RT_COMPILE_H
