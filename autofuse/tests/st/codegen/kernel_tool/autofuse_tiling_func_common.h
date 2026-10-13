/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __AUTOFUSE_TILING_FUNC_COMMON_H__
#define __AUTOFUSE_TILING_FUNC_COMMON_H__
#include <stdexcept>
#include <sstream>
#include <cmath>
#include <vector>
#include "autofuse_tiling_data.h"
#ifndef __CCE_KT_TEST__
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/kernel_context.h"
#include "exe_graph/runtime/continuous_vector.h"
#include "platform/platform_infos_def.h"
#include "platform_ascendc.h"
#endif

#include <cfloat>

#include <cstdint>
#include <memory>
#include <cmath>
#include <cstdlib>
#include <memory.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cfloat>
#include <algorithm>
#include <set>
#include <unordered_map>
#include <array>
#include <functional>
#include <chrono>
#include <cstdint>
#include <string>

#include <cinttypes>
#include <sys/syscall.h>
#include <unistd.h>
#include <cstdint>
#include <stdio.h>
#include <stdlib.h>
#include <map>
#include "dlog_pub.h"

#define GE_MODULE_NAME static_cast<int32_t>(45)
inline bool IsLogPrintStdout() {
  static int32_t stdout_flag = -1;
  if (stdout_flag == -1) {
    const char *env_ret = getenv("ASCEND_SLOG_PRINT_TO_STDOUT");
    const bool print_stdout = ((env_ret != nullptr) && (strcmp(env_ret, "1") == 0));
    stdout_flag = print_stdout ? 1 : 0;
  }
  return (stdout_flag == 1) ? true : false;
}

inline uint64_t GetTid() {
  return static_cast<uint64_t>(syscall(__NR_gettid));
}

#define GELOGE(ERROR_CODE, fmt, ...)                                                                               \
  do {                                                                                                             \
    dlog_error(GE_MODULE_NAME, "%" PRIu64 " %s: ErrorNo: %" PRIuLEAST8 "(%s) %s" fmt, GetTid(), &__FUNCTION__[0U], \
               (ERROR_CODE), "", "", ##__VA_ARGS__);                                                               \
  } while (false)

#define GELOGW(fmt, ...)                                                                          \
  do {                                                                                            \
    dlog_warn(GE_MODULE_NAME, "%" PRIu64 " %s:" fmt, GetTid(), &__FUNCTION__[0U], ##__VA_ARGS__); \
  } while (false)

#define GELOGI(fmt, ...)                                                                          \
  do {                                                                                            \
    dlog_info(GE_MODULE_NAME, "%" PRIu64 " %s:" fmt, GetTid(), &__FUNCTION__[0U], ##__VA_ARGS__); \
  } while (false)

#define GELOGD(fmt, ...)                                                                           \
  do {                                                                                             \
    dlog_debug(GE_MODULE_NAME, "%" PRIu64 " %s:" fmt, GetTid(), &__FUNCTION__[0U], ##__VA_ARGS__); \
  } while (false)

#define GEEVENT(fmt, ...)                                                                                        \
  do {                                                                                                           \
    dlog_info(static_cast<int32_t>(static_cast<uint32_t>(RUN_LOG_MASK) | static_cast<uint32_t>(GE_MODULE_NAME)), \
              "%" PRIu64 " %s:" fmt, GetTid(), &__FUNCTION__[0U], ##__VA_ARGS__);                                \
    if (!IsLogPrintStdout()) {                                                                                   \
      dlog_info(GE_MODULE_NAME, "%" PRIu64 " %s:" fmt, GetTid(), &__FUNCTION__[0U], ##__VA_ARGS__);              \
    }                                                                                                            \
  } while (false)

#define OP_LOGD(name, fmt, ...)
#define OP_LOGI(name, fmt, ...)
#define OP_LOGW(name, fmt, ...)
#define OP_LOGE(name, fmt, ...)
#define OP_EVENT(name, fmt, ...)
#define Max(a, b) ((double)(a) > (double)(b) ? (a) : (b))
#define Min(a, b) ((double)(a) < (double)(b) ? (a) : (b))
#define Log(a) (log((double)(a)))
#define Pow(a, b) pow(a, b)
#define Rational(a, b) ((double)(a) / (double)(b))
#define ExpectEq(a, b) ((a) == (b))
#define ExpectNe(a, b) ((a) != (b))
#define ExpectLe(a, b) ((a) <= (b))
#define ExpectLt(a, b) ((a) < (b))
#define LogicAnd(a, b) ((a) && (b))
#define LogicOr(a, b) ((a) || (b))
#define True true
#define False false
#define MAX_SOLUTION 50
#define OP_NAME "autofuse_pointwise_0_Abs_Add"

namespace optiling {};
using namespace optiling;
uint64_t GetWorkspaceSize(const AutofuseTilingData &tiling_data);
namespace optiling {
using namespace std;

// ATT缓存相关常量
constexpr size_t kAttInputShapeSize = 64;         // 输入Shape数组大小
constexpr size_t kAttOperatorCacheCapacity = 24;  // 算子级缓存容量
constexpr double kAttLoadFactorThreshold = 0.8;   // 负载因子阈值
constexpr uint32_t kAttHashPrime = 0x9e3779b9;    // Hash混合黄金比例常量
inline bool IsEqual(double a, double b) {
  const double epsilon = 1e-8;
  double abs = (a > b) ? (a - b) : (b - a);
  return abs < epsilon;
}
template <typename T1, typename T2>
inline double TernaryOp(bool cond, T1 a, T2 b) {
  return static_cast<double>(cond ? a : b);
}
template <typename T>
inline T Ceiling(T a) {
  T value = static_cast<T>(static_cast<int64_t>(a));
  return (IsEqual(value, a)) ? value : (value + 1);
}
template <typename T>
inline T Floor(T a) {
  return static_cast<T>(static_cast<int64_t>(a));
}
template <typename T1, typename T2>
inline auto Mod(T1 a, T2 b) -> decltype(a % b) {
  return a % b;
}
template <typename T1, typename T2>
inline auto Mod(T1 a, T2 b) ->
    typename std::enable_if<std::is_floating_point<T1>::value || std::is_floating_point<T2>::value,
                            decltype(std::fmod(a, b))>::type {
  return std::fmod(a, b);
}
template <typename TI, typename TO>
inline TO &RefToRef(TI &ptr) {
  return *(reinterpret_cast<TO *>(reinterpret_cast<void *>(&ptr)));
}

struct TilingDataCopy {
  uint32_t b0_size;
  void set_b0_size(uint32_t val) {
    b0_size = val;
  }
  inline uint32_t get_b0_size() {
    return b0_size;
  }
  uint32_t block_dim;
  void set_block_dim(uint32_t val) {
    block_dim = val;
  }
  inline uint32_t get_block_dim() {
    return block_dim;
  }
  uint32_t q0_size;
  void set_q0_size(uint32_t val) {
    q0_size = val;
  }
  inline uint32_t get_q0_size() {
    return q0_size;
  }
  uint32_t q1_size;
  void set_q1_size(uint32_t val) {
    q1_size = val;
  }
  inline uint32_t get_q1_size() {
    return q1_size;
  }
  uint32_t q2_size;
  void set_q2_size(uint32_t val) {
    q2_size = val;
  }
  inline uint32_t get_q2_size() {
    return q2_size;
  }
  uint32_t s2;
  void set_s2(uint32_t val) {
    s2 = val;
  }
  inline uint32_t get_s2() {
    return s2;
  }
  uint32_t s3;
  void set_s3(uint32_t val) {
    s3 = val;
  }
  inline uint32_t get_s3() {
    return s3;
  }
  uint32_t tiling_key;
  void set_tiling_key(uint32_t val) {
    tiling_key = val;
  }
  inline uint32_t get_tiling_key() {
    return tiling_key;
  }
  uint32_t ub_size;
  void set_ub_size(uint32_t val) {
    ub_size = val;
  }
  inline uint32_t get_ub_size() {
    return ub_size;
  }
  uint32_t z0z1Tb_size;
  void set_z0z1Tb_size(uint32_t val) {
    z0z1Tb_size = val;
  }
  inline uint32_t get_z0z1Tb_size() {
    return z0z1Tb_size;
  }
  uint32_t z0z1t_size;
  void set_z0z1t_size(uint32_t val) {
    z0z1t_size = val;
  }
  inline uint32_t get_z0z1t_size() {
    return z0z1t_size;
  }
};
template <size_t INPUT_VARS_SIZE>
struct InputKeyHash {
  size_t operator()(const std::array<uint32_t, INPUT_VARS_SIZE> &key) const {
    size_t hash = 0;
    for (uint32_t value : key) {
      hash ^= value + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    }
    return hash;
  }
};

template <size_t INPUT_VARS_SIZE, size_t CAPACITY>
class FixedSizeHashMap {
 private:
  using Key = std::array<uint32_t, INPUT_VARS_SIZE>;
  using Value = TilingDataCopy;
  enum BucketState { EMPTY, OCCUPIED, DELETED };
  struct Bucket {
    Key key;
    Value value;
    BucketState state;
    Bucket() : state(EMPTY) {}
  };
  std::array<Bucket, CAPACITY> buckets;
  InputKeyHash<INPUT_VARS_SIZE> hasher;
  size_t size_ = 0;
  size_t find_index(const Key &key) const {
    size_t hash = hasher(key) % CAPACITY;
    size_t start = hash;
    do {
      if (buckets[hash].state == EMPTY) {
        return hash;
      } else if (buckets[hash].state == OCCUPIED && buckets[hash].key == key) {
        return hash;
      }
      hash = (hash + 1) % CAPACITY;
    } while (hash != start);
    return CAPACITY;
  }

 public:
  bool insert(const Key &key, const Value &value) {
    if (size_ >= CAPACITY * 0.8) {
      return false;
    }
    size_t index = find_index(key);
    if (index >= CAPACITY) {
      return false;
    }
    if (buckets[index].state != OCCUPIED) {
      buckets[index].key = key;
      buckets[index].value = value;
      buckets[index].state = OCCUPIED;
      size_++;
    } else {
      buckets[index].value = value;
    }
    return true;
  }
  Value *find(const Key &key) {
    size_t index = find_index(key);
    if (index < CAPACITY && buckets[index].state == OCCUPIED) {
      return &buckets[index].value;
    }
    return nullptr;
  }
  const Value *find(const Key &key) const {
    return const_cast<FixedSizeHashMap *>(this)->find(key);
  }
  bool erase(const Key &key) {
    size_t index = find_index(key);
    if (index < CAPACITY && buckets[index].state == OCCUPIED) {
      buckets[index].state = DELETED;
      size_--;
      return true;
    }
    return false;
  }
  void clear() {
    for (auto &bucket : buckets) {
      bucket.state = EMPTY;
    }
    size_ = 0;
  }
  size_t size() const {
    return size_;
  }
  bool empty() const {
    return size_ == 0;
  }
};

enum class PipeType : uint8_t {
  AIV_VEC,
  AIV_MTE2,
  AIV_MTE3,
};

struct TilingOption {
  int32_t tiling_case_id{-1};
  int32_t algorithm_index{0};
};
static TilingOption tiling_option_default{};

/*
ConstraintType:约束类型
  LOCAL_BUFFER:仅与内存占用相关的约束, 如s1t * s2t < UB
  LB_MIXED:与内存占用相关的约束
  MC_MIXED:纯多核相关的约束
*/
enum class ConstraintType {
  LOCAL_BUFFER = 0,
  LB_MIXED = 1,
  MC_MIXED = 2,
};

struct Variable;
struct TilingVariable;
struct Constraint;
using ConsEvalFuncPtr = int64_t (*)(TilingVariable **rel_tiling_vars, Variable **rel_input_shapes, int32_t rel_hw_spec);
using GetUpperBoundFuncPtr = int64_t (*)(Variable **rel_ori_dims);

struct Variable {
  int32_t value = -1;
};

struct Constraint {
  int32_t rel_hw_spec = 0;
  uint32_t rel_tiling_vars_size = 0u;
  uint32_t rel_in_shapes_size = 0u;
  TilingVariable **rel_tiling_vars = nullptr;
  Variable **rel_in_shapes = nullptr;
  ConsEvalFuncPtr eval = nullptr;
  ConstraintType type;
};

struct TilingVariable : public Variable {
  int32_t align = 1;
  int32_t prompt_align = 1;
  int32_t data_type_size = 4;
  bool notail = false;
  bool mc_related = false;
  TilingVariable *notail_var = nullptr;
  uint32_t rel_cons_size = 0u;
  uint32_t upper_bound_vars_size = 0u;
  Variable **upper_bound_vars = nullptr;
  Constraint **rel_cons = nullptr;
  GetUpperBoundFuncPtr upper_bound = nullptr;
  __attribute__((always_inline)) bool SetValue(int32_t val) noexcept {
    return (val > 0) ? (value = val, true) : false;
  }
};

struct AxesReorderSolverInput {
  uint32_t core_num = 0u;
  uint32_t ub_size = 0u;
  uint32_t input_vars_size = 0u;
  uint32_t tiling_vars_size = 0u;
  uint32_t pure_mc_vars_size = 0u;
  uint32_t local_buffer_vars_size = 0u;
  uint32_t all_cons_size = 0u;
  double ub_threshold = 0.2f;
  double corenum_threshold = 0.4f;
  double perf_threshold = 0000.0f;
  Variable **input_vars = nullptr;
  TilingVariable **tiling_vars = nullptr;
  TilingVariable **pure_mc_vars = nullptr;
  TilingVariable **local_buffer_vars = nullptr;
  Constraint **all_cons = nullptr;

  std::string DebugString() const {
    std::stringstream ss;
    ss << "core_num: " << core_num;
    ss << ", ub_size: " << ub_size;
    if (input_vars != nullptr) {
      for (uint32_t i = 0; i < input_vars_size; i++) {
        if (input_vars[i] != nullptr) {
          ss << ", input_vars[" << i << "]: " << input_vars[i]->value;
        }
      }
    }
    if (tiling_vars != nullptr) {
      for (uint32_t i = 0; i < tiling_vars_size; i++) {
        if (tiling_vars[i] != nullptr) {
          ss << ", tiling_vars[" << i << "]: " << tiling_vars[i]->value;
        }
      }
    }
    if (pure_mc_vars != nullptr) {
      for (uint32_t i = 0; i < pure_mc_vars_size; i++) {
        if (pure_mc_vars[i] != nullptr) {
          ss << ", pure_mc_vars[" << i << "]: " << pure_mc_vars[i]->value;
        }
      }
    }
    if (local_buffer_vars != nullptr) {
      for (uint32_t i = 0; i < local_buffer_vars_size; i++) {
        if (local_buffer_vars[i] != nullptr) {
          ss << ", local_buffer_vars[" << i << "]: " << local_buffer_vars[i]->value;
        }
      }
    }
    ss << ", all_cons_size: " << all_cons_size;
    ss << ", ub_threshold: " << ub_threshold;
    ss << ", corenum_threshold: " << corenum_threshold;
    ss << ", perf_threshold: " << perf_threshold;
    return ss.str();
  }
};

class AxesReorderSolver {
 public:
  explicit AxesReorderSolver(const AxesReorderSolverInput &input) : input_(input) {}
  ~AxesReorderSolver() {}

  bool Run(const bool is_trade_off, const bool enable_auto_tune);
  bool PgoSolverGenerateAllTilingData();
  std::vector<std::vector<uint32_t>> GetTilingDataList() {
    return availiable_tiling_data_list_;
  }

 protected:
  virtual bool CalUsedCoreNum(double &used_core_num) = 0;
  virtual bool CalRealUsedCoreNum(int32_t &used_core_num) = 0;
  virtual double GetPerf() = 0;
  virtual bool SatisfyThresholdUBSize() = 0;
  AxesReorderSolverInput input_;
  std::vector<std::vector<uint32_t>> availiable_tiling_data_list_;

 private:
  inline bool GetTiling(const bool is_tuning, const bool enable_workload_balance);
  inline bool GetMaxBlockDimTiling(const uint32_t block_dim);
  inline bool AutoTuning(const bool is_trade_off);
  inline bool FindNextUpperBlockDim(const uint32_t block_dim, uint32_t &next_lower_block_dim) const;
  inline bool FindNextLowerBlockDim(const uint32_t block_dim, uint32_t &next_upper_block_dim) const;
  inline void SaveInputTilingVars(TilingVariable *tiling_vars, TilingVariable *pure_mc_vars,
                                  TilingVariable *local_buffer_vars) const;
  inline void RestoreInputTilingVars(const TilingVariable *tiling_vars, const TilingVariable *pure_mc_vars,
                                     const TilingVariable *local_buffer_vars) const;
  inline void FindBetterSolutionByUpperBlockDim(double next_upper_perf, uint32_t next_upper_block_dim);
  inline void FindBetterSolutionByLowerBlockDim(double next_lower_perf, uint32_t next_lower_block_dim);
  inline bool WorkloadBalance();
  bool TuneNotailVar(TilingVariable *var);
  bool SatisfyCons(ConstraintType cons_type);
  bool SatisfyCons(TilingVariable *var, ConstraintType cons_type);
  bool SatisfyMCCons();
  bool InitLocalBufferVars();
  bool InitMulticoreVars();
  bool GetMinMulticoreVars();
  bool MulticoreTiling(bool enable_workload_balance = false);
  bool NaiveLocalBufTiling();
  bool BinaryLocalBufTiling();
  void ApplyPromptAlign(TilingVariable *var);
  bool LocalBufTiling(const bool is_tuning);
  void PgoSolverGenerateAllTilingDataInner(const uint32_t index, std::vector<uint32_t> &ans_item,
                                           std::vector<std::vector<uint32_t>> &ans, int32_t step_max = 16);
  int64_t pgo_step_max_{16};
};

bool GetTiling(AutofuseTilingData &tiling_data, TilingOption *tiling_option);
bool GetTiling(AutofuseTilingData &tiling_data, int32_t tilingCaseId);
}  // namespace optiling
#endif  // __AUTOFUSE_TILING_FUNC_COMMON_H__
