/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_INC_COMMON_UTIL_H_
#define AIR_INC_COMMON_UTIL_H_

#include <cmath>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "graph/types_af.h"
#include "graph/ge_error_codes_af.h"
#include "common/ge_common/debug/ge_log.h"
#include "common/ge_common/string_util.h"
#include "common/ge_common/scope_guard.h"

using AddrGetter = std::function<const void *(size_t)>;

#define GE_CHECK_POSITIVE_SIZE_RANGE(size)                                                                 \
  do {                                                                                                     \
    if ((size) <= 0) {                                                                                     \
      GELOGE(af::FAILED, "param[%s][%ld] is not a positive number, which should be greater than 0", #size, \
             static_cast<int64_t>(size));                                                                  \
      return PARAM_INVALID;                                                                                \
    }                                                                                                      \
  } while (false)

#define CHECK_FALSE_EXEC(expr, exec_expr, ...) \
  {                                            \
    const bool b = (expr);                     \
    if (!b) {                                  \
      exec_expr;                               \
    }                                          \
  }

// new ge marco
// Encapsulate common resource releases
#define GE_MAKE_GUARD_RTMEM(var)     \
  GE_MAKE_GUARD(var, [&var]() {      \
    if ((var) != nullptr) {          \
      GE_CHK_RT(aclrtFreeHost(var)); \
    }                                \
  })

#define GE_MAKE_GUARD_RTSTREAM(var)    \
  GE_MAKE_GUARD(var, [&]() {           \
    if ((var) != nullptr) {            \
      GE_CHK_RT(rtStreamDestroy(var)); \
    }                                  \
  })

#define GE_MAKE_GUARD_ACLRTSTREAM(var)    \
  GE_MAKE_GUARD(var, [&]() {              \
    if ((var) != nullptr) {               \
      GE_CHK_RT(aclrtDestroyStream(var)); \
    }                                     \
  })

// For propagating errors when calling a function.
#define GE_RETURN_IF_ERROR(expr)           \
  do {                                     \
    const af::Status _chk_status = (expr); \
    if (_chk_status != af::SUCCESS) {      \
      return _chk_status;                  \
    }                                      \
  } while (false)

#define GE_RETURN_WITH_LOG_IF_ERROR(expr, ...) \
  do {                                         \
    const af::Status _chk_status = (expr);     \
    if (_chk_status != af::SUCCESS) {          \
      GELOGE(af::FAILED, __VA_ARGS__);         \
      return _chk_status;                      \
    }                                          \
  } while (false)

// check whether the parameter is true. If it is, return FAILED and record the error log
#define GE_RETURN_WITH_LOG_IF_TRUE(condition, ...) \
  do {                                             \
    if (condition) {                               \
      GELOGE(af::FAILED, __VA_ARGS__);             \
      return af::FAILED;                           \
    }                                              \
  } while (false)

// Check if the parameter is false. If yes, return FAILED and record the error log
#define GE_RETURN_WITH_LOG_IF_FALSE(condition, ...) \
  do {                                              \
    const bool _condition = (condition);            \
    if (!_condition) {                              \
      GELOGE(af::FAILED, __VA_ARGS__);              \
      return af::FAILED;                            \
    }                                               \
  } while (false)

// Checks whether the parameter is true. If so, returns PARAM_INVALID and records the error log
#define GE_RT_PARAM_INVALID_WITH_LOG_IF_TRUE(condition, ...) \
  do {                                                       \
    if (condition) {                                         \
      GELOGE(af::FAILED, __VA_ARGS__);                       \
      return af::PARAM_INVALID;                              \
    }                                                        \
  } while (false)

// Check if the parameter is false. If yes, return PARAM_INVALID and record the error log
#define GE_RT_PARAM_INVALID_WITH_LOG_IF_FALSE(condition, ...) \
  do {                                                        \
    const bool _condition = (condition);                      \
    if (!_condition) {                                        \
      GELOGE(af::FAILED, __VA_ARGS__);                        \
      return af::PARAM_INVALID;                               \
    }                                                         \
  } while (false)

// Check if the parameter is null. If yes, return PARAM_INVALID and record the error
#define GE_CHECK_NOTNULL(val, ...)                                                            \
  do {                                                                                        \
    if ((val) == nullptr) {                                                                   \
      REPORT_INNER_ERR_MSG("E19999", "Param:" #val " is nullptr, check invalid" __VA_ARGS__); \
      GELOGE(af::FAILED, "[Check][Param:" #val "] null is invalid" __VA_ARGS__);              \
      return af::PARAM_INVALID;                                                               \
    }                                                                                         \
  } while (false)

// Check if the parameter is null. If yes, just return and record the error
#define GE_CHECK_NOTNULL_JUST_RETURN(val)                      \
  do {                                                         \
    if ((val) == nullptr) {                                    \
      GELOGE(af::FAILED, "param[%s] must not be null.", #val); \
      return;                                                  \
    }                                                          \
  } while (false)

// Check whether the parameter is null. If so, execute the exec_expr expression and record the error log
#define GE_CHECK_NOTNULL_EXEC(val, exec_expr)                  \
  do {                                                         \
    if ((val) == nullptr) {                                    \
      GELOGE(af::FAILED, "param[%s] must not be null.", #val); \
      exec_expr;                                               \
    }                                                          \
  } while (false)

// Check whether the parameter is null. If yes, return directly and record the error log
#define GE_RT_VOID_CHECK_NOTNULL(val)                          \
  do {                                                         \
    if ((val) == nullptr) {                                    \
      GELOGE(af::FAILED, "param[%s] must not be null.", #val); \
      return;                                                  \
    }                                                          \
  } while (false)

// Check if the parameter is null. If yes, return false and record the error log
#define GE_RT_FALSE_CHECK_NOTNULL(val)                         \
  do {                                                         \
    if ((val) == nullptr) {                                    \
      GELOGE(af::FAILED, "param[%s] must not be null.", #val); \
      return false;                                            \
    }                                                          \
  } while (false)

// Check if the parameter is out of bounds
#define GE_CHECK_SIZE(size)                                                                                          \
  do {                                                                                                               \
    if ((size) == 0U) {                                                                                              \
      GELOGE(af::FAILED, "param[%s][%zu] is out of range, which should not be 0", #size, static_cast<size_t>(size)); \
      return af::PARAM_INVALID;                                                                                      \
    }                                                                                                                \
  } while (false)

// Check if the value on the left is greater than or equal to the value on the right
#define GE_CHECK_GE(lhs, rhs)                                                                           \
  do {                                                                                                  \
    if ((lhs) < (rhs)) {                                                                                \
      GELOGE(af::FAILED, "param[%s][%ld] is less than[%s][%ld]", #lhs, static_cast<int64_t>(lhs), #rhs, \
             static_cast<int64_t>(rhs));                                                                \
      return af::PARAM_INVALID;                                                                         \
    }                                                                                                   \
  } while (false)

// Check if the value on the left is less than or equal to the value on the right
#define GE_CHECK_LE(lhs, rhs)                                                                              \
  do {                                                                                                     \
    if ((lhs) > (rhs)) {                                                                                   \
      GELOGE(af::FAILED, "param[%s][%ld] is greater than[%s][%ld]", #lhs, static_cast<int64_t>(lhs), #rhs, \
             static_cast<int64_t>(rhs));                                                                   \
      return af::PARAM_INVALID;                                                                            \
    }                                                                                                      \
  } while (false)

#define GE_DELETE_NEW_SINGLE(var) \
  do {                            \
    if ((var) != nullptr) {       \
      delete (var);               \
      (var) = nullptr;            \
    }                             \
  } while (false)

#define GE_DELETE_NEW_ARRAY(var) \
  do {                           \
    if ((var) != nullptr) {      \
      delete[] (var);            \
      (var) = nullptr;           \
    }                            \
  } while (false)

#define GE_FREE_RT_LOG(addr)                                               \
  do {                                                                     \
    if ((addr) != nullptr) {                                               \
      const aclError error = aclrtFree(addr);                              \
      if (error != ACL_SUCCESS) {                                          \
        GELOGE(ge::RT_FAILED, "Call aclrtFree failed, error: %#x", error); \
      }                                                                    \
      (addr) = nullptr;                                                    \
    }                                                                      \
  } while (false)

namespace af {
/**
 * @ingroup domi_common
 * @brief version of om.proto file
 */
constexpr int32_t OM_PROTO_VERSION = 2;

/// @ingroup domi_common
/// @brief onverts Vector of a number to a string.
/// @param [in] v  Vector of a number
/// @return string
template <typename T>
GE_FUNC_VISIBILITY std::string ToString(const std::vector<T> &v) {
  bool first = true;
  std::stringstream ss;
  ss << "[";
  for (const T &x : v) {
    if (first) {
      first = false;
      ss << x;
    } else {
      ss << ", " << x;
    }
  }
  ss << "]";
  return ss.str();
}

/// @ingroup: domi_common
/// @brief: get length of file
/// @param [in] input_file: path of file
/// @return int64_t： File length. If the file length fails to be obtained, the value -1 is returned.
GE_FUNC_VISIBILITY extern int64_t GetFileLength(const std::string &input_file);

/// @ingroup domi_common
/// @brief Reads all data from a binary file.
/// @param [in] file_name  path of file
/// @param [out] buffer  Output memory address, which needs to be released by the caller.
/// @param [out] length  Output memory size
/// @return false fail
/// @return true success
GE_FUNC_VISIBILITY bool ReadBytesFromBinaryFile(const char_t *const file_name, char_t **const buffer, int32_t &length);

///@ingroup domi_common
/// @brief  Get binary file from file
/// @param [in] path  file path.
/// @param [out] buffer char[] used to store file data
/// @param [out] data_len store read size
/// @return graphStatus GRAPH_SUCCESS: success, OTHERS: fail
GE_FUNC_VISIBILITY ge::graphStatus GetBinFromFile(const std::string &path, char_t *buffer, size_t &data_len);

/// @ingroup domi_common
/// @brief Recursively Creating a Directory
/// @param [in] directory_path  Path, which can be a multi-level directory.
/// @return 0 success
/// @return -1 fail
GE_FUNC_VISIBILITY extern int32_t CreateDirectory(const std::string &directory_path);

/// @ingroup domi_common
/// @brief Obtains the current time string.
/// @return Time character string in the format ： %Y%m%d%H%M%S, eg: 20171011083555
GE_FUNC_VISIBILITY std::string CurrentTimeInStr();

/// @ingroup domi_common
/// @brief Obtains the absolute time (timestamp) of the current system.
/// @return Timestamp, in microseconds (US)
GE_FUNC_VISIBILITY uint64_t GetCurrentTimestamp();

/// @ingroup domi_common
/// @brief Obtains the absolute time (timestamp) of the current system.
/// @return Timestamp, in seconds (US)
GE_FUNC_VISIBILITY uint32_t GetCurrentSecondTimestap();

/// @ingroup domi_common
/// @brief Absolute path for obtaining files.
/// @param [in] path of input file
/// @param [out] Absolute path of a file. If the absolute path cannot be obtained, an empty string is returned
GE_FUNC_VISIBILITY std::string RealPath(const char_t *path);

/// @ingroup domi_common
/// @brief Check whether the specified input file path is valid.
/// 1.  The specified path cannot be empty.
/// 2.  The path can be converted to an absolute path.
/// 3.  The file path exists and is readable.
/// @param [in] file_path path of input file
/// @param [out] result
GE_FUNC_VISIBILITY bool CheckInputPathValid(const std::string &file_path, const std::string &atc_param = "");

/// @ingroup domi_common
/// @brief Checks whether the specified output file path is valid.
/// @param [in] file_path path of output file
/// @param [out] result
GE_FUNC_VISIBILITY bool CheckOutputPathValid(const std::string &file_path, const std::string &atc_param = "");

/// @ingroup domi_common
/// @brief Check whether the file path meets the whitelist verification requirements.
/// @param [in] str file path
/// @param [out] result
GE_FUNC_VISIBILITY bool ValidateStr(const std::string &file_path, const std::string &mode);

GE_FUNC_VISIBILITY Status ConvertToInt32(const std::string &str, int32_t &val);

inline std::string FormatErrnoReason(const int32_t error_num, const char_t *err_msg) {
  std::string reason = "[Errno " + std::to_string(error_num) + "]";
  if ((err_msg != nullptr) && (err_msg[0] != '\0')) {
    reason += " ";
    reason += err_msg;
  }
  return reason;
}

GE_FUNC_VISIBILITY std::string GetErrorNumStr(const int32_t errorNum);

GE_FUNC_VISIBILITY void SplitStringByComma(const std::string &str, std::vector<std::string> &sub_str_vec);

/// @ingroup domi_common
/// @brief Parse output reuse input memory indexes from config string
/// @param [in] reuse_indexes_str Config string, format: "input_idx,output_idx|input_idx,output_idx|..." e.g., "1,1|2,3"
/// @param [out] io_same_addr_pairs Parsed pairs list (input_idx, output_idx)
GE_FUNC_VISIBILITY void ParseOutputReuseInputMemIndexes(const std::string &reuse_indexes_str,
                                                        std::vector<std::pair<size_t, size_t>> &io_same_addr_pairs);

/// @ingroup domi_common
/// @brief Check IO reuse address pairs
/// @param [in] io_same_addr_pairs Pairs list (input_idx, output_idx)
/// @param [in] get_input_addr Callback function to retrieve input address by index
/// @param [in] input_num Number of inputs
/// @param [in] get_output_addr Callback function to retrieve output address by index
/// @param [in] output_num Number of outputs
/// @return SUCCESS if all addresses match, FAILED otherwise
GE_FUNC_VISIBILITY Status CheckIoReuseAddrPairs(const std::vector<std::pair<size_t, size_t>> &io_same_addr_pairs,
                                                const AddrGetter &get_input_addr, size_t input_num,
                                                const AddrGetter &get_output_addr, size_t output_num);
}  // namespace af

#endif  // AIR_INC_COMMON_UTIL_H_
