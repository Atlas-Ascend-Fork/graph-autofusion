/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "autofuse_tiling_func_common.h"
namespace optiling {
class TilingCaseImpl {
 public:
  TilingCaseImpl(uint32_t corenum) : corenum_(corenum) {}
  virtual ~TilingCaseImpl() = default;
  bool GetTiling(AutofuseTilingData &tiling_data, double &cur_ub_ratio) {
    OP_LOGD(OP_NAME, "Execute DoTiling.");
    if (!DoTiling(tiling_data)) {
      OP_LOGW(OP_NAME, "Failed to do tiling.");
      return false;
    }
    DoApiTiling(tiling_data);
    GeneralTiling(tiling_data);
    TilingSummary(tiling_data, cur_ub_ratio);
    return true;
  }
  virtual double GetPerf(AutofuseTilingData &tiling_data) {
    return 0.0;
  }
  virtual void TilingSummary(AutofuseTilingData &tiling_data, double &cur_ub_ratio) = 0;
  virtual void GetTilingData(TilingDataCopy &from_tiling, AutofuseTilingData &to_tiling) {};
  virtual void SetTilingData(AutofuseTilingData &from_tiling, TilingDataCopy &to_tiling) {};

 protected:
  virtual bool DoTiling(AutofuseTilingData &tiling_data) = 0;
  virtual void DoApiTiling(AutofuseTilingData &tiling_data) {}
  virtual void GeneralTiling(AutofuseTilingData &tiling_data) {}
  uint32_t corenum_;
};
using TilingCaseImplPtr = std::shared_ptr<TilingCaseImpl>;

class AxesReorderSolvercase0 : public AxesReorderSolver {
 public:
  explicit AxesReorderSolvercase0(const AxesReorderSolverInput input) : AxesReorderSolver(input) {}
  ~AxesReorderSolvercase0() = default;
  bool CalUsedCoreNum(double &used_core_num) override;
  bool CalRealUsedCoreNum(int32_t &used_corenum) override;
  bool SatisfyThresholdUBSize() override;
  double GetPerf() override;
};

double AxesReorderSolvercase0::GetPerf() {
  double s2 = static_cast<double>(input_.input_vars[0]->value);
  double s3 = static_cast<double>(input_.input_vars[1]->value);
  double block_dim = 1;
  CalUsedCoreNum(block_dim);
  double z0z1Tb_size = static_cast<double>(input_.pure_mc_vars[0]->value);
  double z0z1t_size = static_cast<double>(input_.local_buffer_vars[0]->value);
  double AIV_MTE2 =
      ((15.8900003433228 * block_dim) +
       (2 *
        TernaryOp(
            (4 * z0z1t_size) < 25000,
            TernaryOp(IsEqual(False, 0),
                      ((4 * z0z1t_size / (((7.30999994277954 / (block_dim)) + 7.90520000457764))) + 27.0100002288818),
                      ((512 / (((7.30999994277954 / (block_dim)) + 7.90520000457764))) + 27.0100002288818)),
            ((4 * z0z1t_size / (((15.8959999084473 / (block_dim)) + 9.90740013122559))) + 27.0100002288818)) *
        z0z1Tb_size) +
       882.090026855469);
  double AIV_MTE3 =
      ((TernaryOp(
            IsEqual(Mod(z0z1t_size, 8), 0),
            ((4 * z0z1t_size / (((3.78999996185303 / (block_dim)) + 9.96000003814697))) + 12.0900001525879),
            TernaryOp(
                (-512 + z0z1t_size) < 0,
                TernaryOp((-8 + z0z1t_size) < 0,
                          (((-2.20000004768372 - (0.101000003516674 * block_dim)) * 4 * z0z1t_size) +
                           (8.89000034332275 * block_dim) + 108.329998016357),
                          ((2.0 * z0z1t_size) + 1.29999995231628)),
                (((256 - ((512 / (((3.78999996185303 / (block_dim)) + 9.96000003814697))) + 12.0900001525879)) * 1.0) +
                 (4 * z0z1t_size / (((3.78999996185303 / (block_dim)) + 9.96000003814697))) + 12.0900001525879))) *
        z0z1Tb_size) +
       497.359985351562);
  double AIV_VEC = ((((0.0147000001743436 * z0z1t_size) + 20.0592002868652) * z0z1Tb_size) +
                    (((0.0206000003963709 * z0z1t_size) + 23.2224998474121) * z0z1Tb_size) + 37.3699989318848);
  return ((300.0 * Max(0, ((((s2 * s3 / (z0z1t_size))) / (z0z1Tb_size))))) + Max(Max(AIV_VEC, AIV_MTE2), AIV_MTE3));
}

bool AxesReorderSolvercase0::SatisfyThresholdUBSize() {
  double s2 = static_cast<double>(input_.input_vars[0]->value);
  double s3 = static_cast<double>(input_.input_vars[1]->value);
  double z0z1t_size = static_cast<double>(input_.local_buffer_vars[0]->value);
  auto temp0 = (4 * z0z1t_size);
  double tensor_0 = temp0;
  double tensor_1 = temp0;
  double tensor_2 = temp0;
  double tensor_3 = temp0;
  auto temp1 = Rational(1, 32);
  auto temp2 = (temp1 * tensor_3);
  auto temp3 = Ceiling(temp2);
  auto temp4 = (32 * temp3);
  auto temp5 = (temp1 * tensor_0);
  auto temp6 = Ceiling(temp5);
  auto temp7 = (64 * temp6);
  auto temp8 = (temp4 + temp7);
  auto temp9 = (temp1 * tensor_1);
  auto temp10 = Ceiling(temp9);
  auto temp11 = (64 * temp10);
  auto temp12 = (temp8 + temp11);
  auto temp13 = (temp1 * tensor_2);
  auto temp14 = Ceiling(temp13);
  auto temp15 = (64 * temp14);
  auto temp16 = (temp12 + temp15);
  auto temp17 = (temp16 + 8192);
  int32_t ub_size = temp17;
  return ub_size > static_cast<int32_t>(input_.ub_threshold * input_.ub_size);
}

bool AxesReorderSolvercase0::CalUsedCoreNum(double &used_core_num) {
  double s2 = static_cast<double>(input_.input_vars[0]->value);
  double s3 = static_cast<double>(input_.input_vars[1]->value);
  double z0z1Tb_size = static_cast<double>(input_.pure_mc_vars[0]->value);
  double z0z1t_size = static_cast<double>(input_.local_buffer_vars[0]->value);
  used_core_num = Max(0, ((((s2 * s3 / (z0z1t_size))) / (z0z1Tb_size))));
  return true;
}
bool AxesReorderSolvercase0::CalRealUsedCoreNum(int32_t &used_core_num) {
  double s2 = static_cast<double>(input_.input_vars[0]->value);
  double s3 = static_cast<double>(input_.input_vars[1]->value);
  double z0z1Tb_size = static_cast<double>(input_.pure_mc_vars[0]->value);
  double z0z1t_size = static_cast<double>(input_.local_buffer_vars[0]->value);
  used_core_num = Max(0, Ceiling((Ceiling((s2 * s3 / (z0z1t_size))) / (z0z1Tb_size))));
  return true;
};

/*
 Tensor used for tiling case 0 is:
  tensor_0:Add_out0_graph/Load_0_output_0
  tensor_1:Add_out0_graph/Load_1_output_0
  tensor_2:Add_out0_graph/Add_0_output_0
  tensor_3:Add_out0_graph/Abs_0_output_0
 Exe time & Perf time used for tiling case 0 is:
  Add_out0_graph/Abs_0_AIV_VEC_perf:((0.0147000001743436 * z0z1t_size) + 20.0592002868652)
  Add_out0_graph/Abs_0_exe_time:z0z1Tb_size
  Add_out0_graph/Add_0_AIV_VEC_perf:((0.0206000003963709 * z0z1t_size) + 23.2224998474121)
  Add_out0_graph/Add_0_exe_time:z0z1Tb_size
  Add_out0_graph/Load_0_AIV_MTE2_perf:TernaryOp((4 * z0z1t_size) < 25000, TernaryOp(IsEqual(False, 0), ((4 * z0z1t_size
 / (((7.30999994277954 / (block_dim)) + 7.90520000457764))) + 27.0100002288818), ((512 / (((7.30999994277954 /
 (block_dim)) + 7.90520000457764))) + 27.0100002288818)), ((4 * z0z1t_size / (((15.8959999084473 / (block_dim))
 + 9.90740013122559))) + 27.0100002288818)) Add_out0_graph/Load_0_exe_time:z0z1Tb_size
  Add_out0_graph/Load_1_AIV_MTE2_perf:TernaryOp((4 * z0z1t_size) < 25000, TernaryOp(IsEqual(False, 0), ((4 * z0z1t_size
 / (((7.30999994277954 / (block_dim)) + 7.90520000457764))) + 27.0100002288818), ((512 / (((7.30999994277954 /
 (block_dim)) + 7.90520000457764))) + 27.0100002288818)), ((4 * z0z1t_size / (((15.8959999084473 / (block_dim))
 + 9.90740013122559))) + 27.0100002288818)) Add_out0_graph/Load_1_exe_time:z0z1Tb_size
  Add_out0_graph/Store_0_AIV_MTE3_perf:TernaryOp(IsEqual(Mod(z0z1t_size, 8), 0), ((4 * z0z1t_size / (((3.78999996185303
 / (block_dim)) + 9.96000003814697))) + 12.0900001525879), TernaryOp((-512 + z0z1t_size) < 0, TernaryOp((-8 +
 z0z1t_size) < 0, (((-2.20000004768372 - (0.101000003516674 * block_dim)) * 4 * z0z1t_size) + (8.89000034332275 *
 block_dim) + 108.329998016357), ((2.0 * z0z1t_size) + 1.29999995231628)), (((256 - ((512 / (((3.78999996185303 /
 (block_dim)) + 9.96000003814697))) + 12.0900001525879)) * 1.0) + (4 * z0z1t_size / (((3.78999996185303 / (block_dim))
 + 9.96000003814697))) + 12.0900001525879))) Add_out0_graph/Store_0_exe_time:z0z1Tb_size

*/
class TilingCase0Impl : public TilingCaseImpl {
 public:
  TilingCase0Impl(uint32_t corenum) : TilingCaseImpl(corenum) {}

 protected:
  std::unordered_map<std::string, std::vector<AutofuseTilingData>> filter_map{};
  void GetTilingData(TilingDataCopy &from_tiling, AutofuseTilingData &to_tiling) {
    to_tiling.set_s2(from_tiling.get_s2());
    to_tiling.set_s3(from_tiling.get_s3());
    to_tiling.set_z0z1Tb_size(from_tiling.get_z0z1Tb_size());
    to_tiling.set_z0z1t_size(from_tiling.get_z0z1t_size());
    to_tiling.set_b0_size(from_tiling.get_b0_size());
    to_tiling.set_q0_size(from_tiling.get_q0_size());
    to_tiling.set_q1_size(from_tiling.get_q1_size());
    to_tiling.set_q2_size(from_tiling.get_q2_size());
    to_tiling.set_block_dim(from_tiling.get_block_dim());
    to_tiling.set_tiling_key(from_tiling.get_tiling_key());
  }
  void SetTilingData(AutofuseTilingData &from_tiling, TilingDataCopy &to_tiling) {
    to_tiling.set_s2(from_tiling.get_s2());
    to_tiling.set_s3(from_tiling.get_s3());
    to_tiling.set_z0z1Tb_size(from_tiling.get_z0z1Tb_size());
    to_tiling.set_z0z1t_size(from_tiling.get_z0z1t_size());
    to_tiling.set_b0_size(from_tiling.get_b0_size());
    to_tiling.set_q0_size(from_tiling.get_q0_size());
    to_tiling.set_q1_size(from_tiling.get_q1_size());
    to_tiling.set_q2_size(from_tiling.get_q2_size());
    to_tiling.set_block_dim(from_tiling.get_block_dim());
    to_tiling.set_tiling_key(from_tiling.get_tiling_key());
  }
  bool ExecuteAxesReorderSolver(AutofuseTilingData &tiling_data) {
    Variable s2;
    s2.value = tiling_data.get_s2();
    Variable s3;
    s3.value = tiling_data.get_s3();
    TilingVariable z0z1Tb_size;
    TilingVariable z0z1t_size;
    int64_t ub_size = tiling_data.get_ub_size();
    Constraint cons0;
    auto cons0Eval = [](TilingVariable **rel_tiling_vars, Variable **rel_in_shapes, int32_t rel_hw_spec) {
      double z0z1t_size = rel_tiling_vars[0]->value;
      auto temp0 = (4 * z0z1t_size);
      double tensor_0 = temp0;
      double tensor_1 = temp0;
      double tensor_2 = temp0;
      double tensor_3 = temp0;
      auto temp1 = Rational(1, 32);
      auto temp2 = (temp1 * tensor_3);
      auto temp3 = Ceiling(temp2);
      auto temp4 = (32 * temp3);
      auto temp5 = (temp1 * tensor_0);
      auto temp6 = Ceiling(temp5);
      auto temp7 = (64 * temp6);
      auto temp8 = (temp4 + temp7);
      auto temp9 = (temp1 * tensor_1);
      auto temp10 = Ceiling(temp9);
      auto temp11 = (64 * temp10);
      auto temp12 = (temp8 + temp11);
      auto temp13 = (temp1 * tensor_2);
      auto temp14 = Ceiling(temp13);
      auto temp15 = (64 * temp14);
      auto temp16 = (temp12 + temp15);
      auto temp17 = (temp16 + 8192);
      int64_t value = temp17 - rel_hw_spec;
      return value;
    };
    TilingVariable *cons_0rel_tiling_vars[1] = {
        &z0z1t_size,
    };
    cons0.rel_tiling_vars = cons_0rel_tiling_vars;
    cons0.rel_tiling_vars_size = 1u;
    cons0.rel_hw_spec = ub_size;
    cons0.type = ConstraintType::LOCAL_BUFFER;
    cons0.eval = cons0Eval;
    Constraint cons1;
    auto cons1Eval = [](TilingVariable **rel_tiling_vars, Variable **rel_in_shapes, int32_t rel_hw_spec) {
      double z0z1t_size = rel_tiling_vars[0]->value;
      double s2 = rel_in_shapes[0]->value;
      double s3 = rel_in_shapes[1]->value;
      int64_t value = (z0z1t_size - (s2 * s3));
      return value;
    };
    TilingVariable *cons_1rel_tiling_vars[1] = {
        &z0z1t_size,
    };
    cons1.rel_tiling_vars = cons_1rel_tiling_vars;
    cons1.rel_tiling_vars_size = 1u;
    Variable *cons_1rel_in_shapes[2] = {
        &s2,
        &s3,
    };
    cons1.rel_in_shapes = cons_1rel_in_shapes;
    cons1.rel_in_shapes_size = 2u;
    cons1.type = ConstraintType::LB_MIXED;
    cons1.eval = cons1Eval;
    Constraint cons2;
    auto cons2Eval = [](TilingVariable **rel_tiling_vars, Variable **rel_in_shapes, int32_t rel_hw_spec) {
      double z0z1t_size = rel_tiling_vars[0]->value;
      double s2 = rel_in_shapes[0]->value;
      double s3 = rel_in_shapes[1]->value;
      double z0z1Tb_size = rel_in_shapes[2]->value;
      int64_t value = (z0z1Tb_size - Ceiling((s2 * s3 / (z0z1t_size))));
      return value;
    };
    TilingVariable *cons_2rel_tiling_vars[1] = {
        &z0z1t_size,
    };
    cons2.rel_tiling_vars = cons_2rel_tiling_vars;
    cons2.rel_tiling_vars_size = 1u;
    Variable *cons_2rel_in_shapes[3] = {
        &s2,
        &s3,
        &z0z1Tb_size,
    };
    cons2.rel_in_shapes = cons_2rel_in_shapes;
    cons2.rel_in_shapes_size = 3u;
    cons2.type = ConstraintType::MC_MIXED;
    cons2.eval = cons2Eval;
    GetUpperBoundFuncPtr z0z1Tb_size_upper_bound = [](Variable **parent_vars) {
      int64_t upper_bound = 1;
      double s2 = parent_vars[0]->value;
      double s3 = parent_vars[1]->value;
      double z0z1t_size = parent_vars[2]->value;
      if (parent_vars[0]->value == -1 || parent_vars[1]->value == -1 || parent_vars[2]->value == -1) {
        return static_cast<int64_t>(-1);
      }
      upper_bound *= Ceiling((s2 * s3 / (z0z1t_size)));
      return upper_bound;
    };
    z0z1Tb_size.upper_bound = z0z1Tb_size_upper_bound;
    Variable *z0z1Tb_size_upper_bound_vars[3] = {
        &s2,
        &s3,
        &z0z1t_size,
    };
    z0z1Tb_size.upper_bound_vars = z0z1Tb_size_upper_bound_vars;
    z0z1Tb_size.upper_bound_vars_size = 3u;
    Constraint *z0z1Tb_size_rel_cons[1] = {
        &cons2,
    };
    z0z1Tb_size.rel_cons = z0z1Tb_size_rel_cons;
    z0z1Tb_size.rel_cons_size = 1u;
    z0z1t_size.align = 1;
    z0z1t_size.data_type_size = 4;
    z0z1t_size.prompt_align = 8;
    z0z1t_size.mc_related = true;
    GetUpperBoundFuncPtr z0z1t_size_upper_bound = [](Variable **parent_vars) {
      int64_t upper_bound = 1;
      double s2 = parent_vars[0]->value;
      double s3 = parent_vars[1]->value;
      if (parent_vars[0]->value == -1 || parent_vars[1]->value == -1) {
        return static_cast<int64_t>(-1);
      }
      upper_bound *= (s2 * s3);
      return upper_bound;
    };
    z0z1t_size.upper_bound = z0z1t_size_upper_bound;
    Variable *z0z1t_size_upper_bound_vars[2] = {
        &s2,
        &s3,
    };
    z0z1t_size.upper_bound_vars = z0z1t_size_upper_bound_vars;
    z0z1t_size.upper_bound_vars_size = 2u;
    Constraint *z0z1t_size_rel_cons[3] = {
        &cons0,
        &cons1,
        &cons2,
    };
    z0z1t_size.rel_cons = z0z1t_size_rel_cons;
    z0z1t_size.rel_cons_size = 3u;
    AxesReorderSolverInput input;
    Variable *input_vars[2] = {
        &s2,
        &s3,
    };
    input.input_vars = input_vars;
    input.input_vars_size = 2u;
    TilingVariable *tiling_vars[2] = {
        &z0z1Tb_size,
        &z0z1t_size,
    };
    input.tiling_vars = tiling_vars;
    input.tiling_vars_size = 2u;
    Constraint *all_cons[3] = {
        &cons0,
        &cons1,
        &cons2,
    };
    input.all_cons_size = 3u;
    input.all_cons = all_cons;
    TilingVariable *pure_mc_vars[1] = {
        &z0z1Tb_size,
    };
    input.pure_mc_vars_size = 1u;
    input.pure_mc_vars = pure_mc_vars;
    TilingVariable *local_buffer_vars[1] = {
        &z0z1t_size,
    };
    input.local_buffer_vars_size = 1u;
    input.local_buffer_vars = local_buffer_vars;
    input.core_num = corenum_;
    input.ub_threshold = 0.200000;
    input.corenum_threshold = 0.400000;
    input.ub_size = tiling_data.get_ub_size();
    AxesReorderSolvercase0 solver(input);
    if (!solver.Run(false, true)) {
      return false;
    }
    tiling_data.set_z0z1Tb_size(input.pure_mc_vars[0]->value);
    tiling_data.set_z0z1t_size(input.local_buffer_vars[0]->value);
    return true;
  }

  bool DoTiling(AutofuseTilingData &tiling_data) {
    OP_LOGI(OP_NAME, "Set input params for tiling case 0 of AscGraph0ScheduleResult0G0.  s2 = %lld. s3 = %lld.",
            static_cast<long long>(tiling_data.get_s2()), static_cast<long long>(tiling_data.get_s3()));
    OP_LOGI(OP_NAME,
            "Set ub_size for tiling case 0 of AscGraph0ScheduleResult0G0 to ((224 * Ceiling((Rational(1 , 8) * "
            "z0z1t_size))) + 8192)");
    OP_LOGI(OP_NAME,
            "Set block_dim for tiling case 0 of AscGraph0ScheduleResult0G0 to Max(0, Ceiling((Ceiling((s2 * s3 / "
            "(z0z1t_size))) / (z0z1Tb_size))))");

    OP_LOGD(OP_NAME, "Set hardware params. ub_size = %u. block_dim = %u.", tiling_data.get_ub_size(), corenum_);

    if (!ExecuteAxesReorderSolver(tiling_data)) {
      OP_LOGW(OP_NAME, "Failed to execute axes reorder solver for tilingCaseId case0.");
      return false;
    }
    OP_LOGD(OP_NAME, "Execute axes reorder solver for tilingCaseId case0 successfully.");

    return true;
  }

  void DoApiTiling(AutofuseTilingData &tiling_data) {}
  void GeneralTiling(AutofuseTilingData &tiling_data) {
    double s2 = static_cast<double>(tiling_data.get_s2());
    double s3 = static_cast<double>(tiling_data.get_s3());
    double z0z1Tb_size = static_cast<double>(tiling_data.get_z0z1Tb_size());
    double z0z1t_size = static_cast<double>(tiling_data.get_z0z1t_size());
    tiling_data.set_block_dim(Max(0, Ceiling((Ceiling((s2 * s3 / (z0z1t_size))) / (z0z1Tb_size)))));
    ComputeMemoryParam(tiling_data);
  }

  int Getub_size(AutofuseTilingData &tiling_data) {
    double z0z1t_size = tiling_data.get_z0z1t_size();
    auto temp0 = (4 * z0z1t_size);
    double tensor_0 = temp0;
    double tensor_1 = temp0;
    double tensor_2 = temp0;
    double tensor_3 = temp0;
    auto temp1 = Rational(1, 32);
    auto temp2 = (temp1 * tensor_3);
    auto temp3 = Ceiling(temp2);
    auto temp4 = (32 * temp3);
    auto temp5 = (temp1 * tensor_0);
    auto temp6 = Ceiling(temp5);
    auto temp7 = (64 * temp6);
    auto temp8 = (temp4 + temp7);
    auto temp9 = (temp1 * tensor_1);
    auto temp10 = Ceiling(temp9);
    auto temp11 = (64 * temp10);
    auto temp12 = (temp8 + temp11);
    auto temp13 = (temp1 * tensor_2);
    auto temp14 = Ceiling(temp13);
    auto temp15 = (64 * temp14);
    auto temp16 = (temp12 + temp15);
    auto temp17 = (temp16 + 8192);
    return temp17;
  }

  int Getblock_dim(AutofuseTilingData &tiling_data) {
    double s2 = tiling_data.get_s2();
    double s3 = tiling_data.get_s3();
    double z0z1Tb_size = tiling_data.get_z0z1Tb_size();
    double z0z1t_size = tiling_data.get_z0z1t_size();
    auto temp0 = (s2 * s3);
    auto temp1 = (temp0 / z0z1t_size);
    auto temp2 = Ceiling(temp1);
    auto temp3 = (temp2 / z0z1Tb_size);
    auto temp4 = Ceiling(temp3);
    auto temp5 = Max(0, temp4);
    return temp5;
  }

  double GetAIV_MTE2(AutofuseTilingData &tiling_data) {
    double block_dim = tiling_data.get_block_dim();
    double z0z1Tb_size = tiling_data.get_z0z1Tb_size();
    double z0z1t_size = tiling_data.get_z0z1t_size();

    return (
        (15.8900003433228 * block_dim) +
        (2 *
         TernaryOp(
             (4 * z0z1t_size) < 25000,
             TernaryOp(IsEqual(False, 0),
                       ((4 * z0z1t_size / (((7.30999994277954 / (block_dim)) + 7.90520000457764))) + 27.0100002288818),
                       ((512 / (((7.30999994277954 / (block_dim)) + 7.90520000457764))) + 27.0100002288818)),
             ((4 * z0z1t_size / (((15.8959999084473 / (block_dim)) + 9.90740013122559))) + 27.0100002288818)) *
         z0z1Tb_size) +
        882.090026855469);
  }

  double GetAIV_MTE3(AutofuseTilingData &tiling_data) {
    double block_dim = tiling_data.get_block_dim();
    double z0z1Tb_size = tiling_data.get_z0z1Tb_size();
    double z0z1t_size = tiling_data.get_z0z1t_size();

    return (
        (TernaryOp(
             IsEqual(Mod(z0z1t_size, 8), 0),
             ((4 * z0z1t_size / (((3.78999996185303 / (block_dim)) + 9.96000003814697))) + 12.0900001525879),
             TernaryOp(
                 (-512 + z0z1t_size) < 0,
                 TernaryOp((-8 + z0z1t_size) < 0,
                           (((-2.20000004768372 - (0.101000003516674 * block_dim)) * 4 * z0z1t_size) +
                            (8.89000034332275 * block_dim) + 108.329998016357),
                           ((2.0 * z0z1t_size) + 1.29999995231628)),
                 (((256 - ((512 / (((3.78999996185303 / (block_dim)) + 9.96000003814697))) + 12.0900001525879)) * 1.0) +
                  (4 * z0z1t_size / (((3.78999996185303 / (block_dim)) + 9.96000003814697))) + 12.0900001525879))) *
         z0z1Tb_size) +
        497.359985351562);
  }

  double GetAIV_VEC(AutofuseTilingData &tiling_data) {
    double z0z1Tb_size = tiling_data.get_z0z1Tb_size();
    double z0z1t_size = tiling_data.get_z0z1t_size();

    return ((((0.0147000001743436 * z0z1t_size) + 20.0592002868652) * z0z1Tb_size) +
            (((0.0206000003963709 * z0z1t_size) + 23.2224998474121) * z0z1Tb_size) + 37.3699989318848);
  }

  double GetPerf(AutofuseTilingData &tiling_data) {
    double block_dim = tiling_data.get_block_dim();
    double s2 = tiling_data.get_s2();
    double s3 = tiling_data.get_s3();
    double z0z1Tb_size = tiling_data.get_z0z1Tb_size();
    double z0z1t_size = tiling_data.get_z0z1t_size();

    double AIV_MTE2 =
        ((15.8900003433228 * block_dim) +
         (2 *
          TernaryOp(
              (4 * z0z1t_size) < 25000,
              TernaryOp(IsEqual(False, 0),
                        ((4 * z0z1t_size / (((7.30999994277954 / (block_dim)) + 7.90520000457764))) + 27.0100002288818),
                        ((512 / (((7.30999994277954 / (block_dim)) + 7.90520000457764))) + 27.0100002288818)),
              ((4 * z0z1t_size / (((15.8959999084473 / (block_dim)) + 9.90740013122559))) + 27.0100002288818)) *
          z0z1Tb_size) +
         882.090026855469);
    double AIV_MTE3 =
        ((TernaryOp(
              IsEqual(Mod(z0z1t_size, 8), 0),
              ((4 * z0z1t_size / (((3.78999996185303 / (block_dim)) + 9.96000003814697))) + 12.0900001525879),
              TernaryOp(
                  (-512 + z0z1t_size) < 0,
                  TernaryOp((-8 + z0z1t_size) < 0,
                            (((-2.20000004768372 - (0.101000003516674 * block_dim)) * 4 * z0z1t_size) +
                             (8.89000034332275 * block_dim) + 108.329998016357),
                            ((2.0 * z0z1t_size) + 1.29999995231628)),
                  (((256 - ((512 / (((3.78999996185303 / (block_dim)) + 9.96000003814697))) + 12.0900001525879)) *
                    1.0) +
                   (4 * z0z1t_size / (((3.78999996185303 / (block_dim)) + 9.96000003814697))) + 12.0900001525879))) *
          z0z1Tb_size) +
         497.359985351562);
    double AIV_VEC = ((((0.0147000001743436 * z0z1t_size) + 20.0592002868652) * z0z1Tb_size) +
                      (((0.0206000003963709 * z0z1t_size) + 23.2224998474121) * z0z1Tb_size) + 37.3699989318848);

    return ((300.0 * Max(0, Ceiling((Ceiling((s2 * s3 / (z0z1t_size))) / (z0z1Tb_size))))) +
            Max(Max(AIV_VEC, AIV_MTE2), AIV_MTE3));
  }

  void Setb0_size(AutofuseTilingData &tiling_data) {
    const auto z0z1t_size = tiling_data.get_z0z1t_size();
    auto temp0 = Rational(1, 8);
    auto temp1 = (temp0 * z0z1t_size);
    auto temp2 = Ceiling(temp1);
    auto temp3 = (32 * temp2);
    tiling_data.set_b0_size(temp3);
  }

  void Setq0_size(AutofuseTilingData &tiling_data) {
    const auto z0z1t_size = tiling_data.get_z0z1t_size();
    auto temp0 = Rational(1, 8);
    auto temp1 = (temp0 * z0z1t_size);
    auto temp2 = Ceiling(temp1);
    auto temp3 = (32 * temp2);
    tiling_data.set_q0_size(temp3);
  }

  void Setq1_size(AutofuseTilingData &tiling_data) {
    const auto z0z1t_size = tiling_data.get_z0z1t_size();
    auto temp0 = Rational(1, 8);
    auto temp1 = (temp0 * z0z1t_size);
    auto temp2 = Ceiling(temp1);
    auto temp3 = (32 * temp2);
    tiling_data.set_q1_size(temp3);
  }

  void Setq2_size(AutofuseTilingData &tiling_data) {
    const auto z0z1t_size = tiling_data.get_z0z1t_size();
    auto temp0 = Rational(1, 8);
    auto temp1 = (temp0 * z0z1t_size);
    auto temp2 = Ceiling(temp1);
    auto temp3 = (32 * temp2);
    tiling_data.set_q2_size(temp3);
  }

  void ComputeMemoryParam(AutofuseTilingData &tiling_data) {
    Setb0_size(tiling_data);
    Setq0_size(tiling_data);
    Setq1_size(tiling_data);
    Setq2_size(tiling_data);
  }
  void TilingSummary(AutofuseTilingData &tiling_data, double &cur_ub_ratio) {
    OP_LOGI(OP_NAME, "[PROF]The value of z0z1Tb_size is %u in graph0_result0_g0_0.", tiling_data.get_z0z1Tb_size());
    OP_LOGI(OP_NAME, "[PROF]The value of z0z1t_size is %u in graph0_result0_g0_0.", tiling_data.get_z0z1t_size());
    OP_LOGI(OP_NAME, "[PROF]The value of ub_size is %d in graph0_result0_g0_0.", Getub_size(tiling_data));
    OP_LOGI(OP_NAME, "[PROF]The value of block_dim is %d in graph0_result0_g0_0.", Getblock_dim(tiling_data));
    OP_LOGI(OP_NAME, "[PROF]The value of b0_size is %u in graph0_result0_g0_0.", tiling_data.get_b0_size());
    OP_LOGI(OP_NAME, "[PROF]The value of q0_size is %u in graph0_result0_g0_0.", tiling_data.get_q0_size());
    OP_LOGI(OP_NAME, "[PROF]The value of q1_size is %u in graph0_result0_g0_0.", tiling_data.get_q1_size());
    OP_LOGI(OP_NAME, "[PROF]The value of q2_size is %u in graph0_result0_g0_0.", tiling_data.get_q2_size());
    OP_LOGI(OP_NAME, "[PROF]The value of AIV_MTE2 is %f in graph0_result0_g0_0.", GetAIV_MTE2(tiling_data));
    OP_LOGI(OP_NAME, "[PROF]The value of AIV_MTE3 is %f in graph0_result0_g0_0.", GetAIV_MTE3(tiling_data));
    OP_LOGI(OP_NAME, "[PROF]The value of AIV_VEC is %f in graph0_result0_g0_0.", GetAIV_VEC(tiling_data));
    OP_LOGI(OP_NAME, "[PROF]The objective value of the tiling data is %f in graph0_result0_g0_0.",
            GetPerf(tiling_data));
    cur_ub_ratio = static_cast<double>(Getub_size(tiling_data)) / tiling_data.get_ub_size();
    if (std::isnan(cur_ub_ratio)) {
      cur_ub_ratio = 1;
      OP_LOGI(OP_NAME, "The ub ratio is NaN, set it to 1.");
    }
  }
};

TilingCaseImplPtr GetTilingImplPtr(uint32_t tilingCaseId, uint32_t corenum) {
  TilingCaseImplPtr tilingCaseImplPtr = nullptr;
  if (tilingCaseId == 0u) {
    tilingCaseImplPtr = std::make_shared<TilingCase0Impl>(corenum);
  }
  return tilingCaseImplPtr;
}
void UpdateBetterTiling(TilingCaseImpl *tilingCaseImplPtr, TilingDataCopy &tmp_tiling, AutofuseTilingData &tiling_data,
                        uint32_t tilingCaseId) {
  OP_LOGD(OP_NAME, "The solution for tilingCaseId %u is better, updating the tiling data.", tilingCaseId);
  tiling_data.set_tiling_key(tilingCaseId);
  tilingCaseImplPtr->SetTilingData(tiling_data, tmp_tiling);
  OP_LOGD(OP_NAME, "Set the output tiling data.");
  OP_LOGD(OP_NAME, "Updated the best tilingCaseId to %u.", tilingCaseId);
}

bool FindPerfBetterTilingbyCaseId(TilingCaseImpl *tilingCaseImplPtr, double &obj, double &ub_ratio,
                                  TilingDataCopy &tmp_tiling, AutofuseTilingData &tiling_data, uint32_t tilingCaseId,
                                  bool is_sub_case, bool &sub_case_flag) {
  double cur_obj;
  double cur_ub_ratio;
  if (tilingCaseImplPtr == nullptr) {
    OP_LOGE(OP_NAME, "Pointer for tilingCaseId is null.");
    return false;
  }
  tilingCaseImplPtr->SetTilingData(tiling_data, tmp_tiling);
  if (tilingCaseImplPtr->GetTiling(tiling_data, cur_ub_ratio)) {
    cur_obj = tilingCaseImplPtr->GetPerf(tiling_data);
    OP_LOGD(OP_NAME, "The ub ratio for tilingCaseId %u is %f.", tilingCaseId, cur_ub_ratio);
    OP_LOGD(OP_NAME, "The optimal objective for tilingCaseId %u is %f.", tilingCaseId, cur_obj);
    if (obj < 0) {
      UpdateBetterTiling(tilingCaseImplPtr, tmp_tiling, tiling_data, tilingCaseId);
      sub_case_flag = is_sub_case;
      obj = cur_obj;
      ub_ratio = cur_ub_ratio;
      return true;
    }
    double ub_ratio_diff = cur_ub_ratio > ub_ratio ? (cur_ub_ratio - ub_ratio) : (ub_ratio - cur_ub_ratio);
    if ((cur_obj - obj > 5000)) {
      tilingCaseImplPtr->GetTilingData(tmp_tiling, tiling_data);
    } else if ((obj - cur_obj > 5000)) {
      UpdateBetterTiling(tilingCaseImplPtr, tmp_tiling, tiling_data, tilingCaseId);
      sub_case_flag = is_sub_case;
      obj = cur_obj;
      ub_ratio = cur_ub_ratio;
    } else if (cur_ub_ratio < 0.190000 && ub_ratio >= 0.190000) {
      tilingCaseImplPtr->GetTilingData(tmp_tiling, tiling_data);
    } else if (cur_ub_ratio >= 0.190000 && ub_ratio < 0.190000) {
      UpdateBetterTiling(tilingCaseImplPtr, tmp_tiling, tiling_data, tilingCaseId);
      sub_case_flag = is_sub_case;
      obj = cur_obj;
      ub_ratio = cur_ub_ratio;
    } else if (cur_ub_ratio < 0.190000 && ub_ratio < 0.190000 && !IsEqual(cur_ub_ratio, ub_ratio)) {
      if (cur_ub_ratio > ub_ratio) {
        UpdateBetterTiling(tilingCaseImplPtr, tmp_tiling, tiling_data, tilingCaseId);
        sub_case_flag = is_sub_case;
        obj = cur_obj;
        ub_ratio = cur_ub_ratio;
      } else {
        tilingCaseImplPtr->GetTilingData(tmp_tiling, tiling_data);
      }
    } else {
      if (cur_obj < obj) {
        UpdateBetterTiling(tilingCaseImplPtr, tmp_tiling, tiling_data, tilingCaseId);
        sub_case_flag = is_sub_case;
        obj = cur_obj;
        ub_ratio = cur_ub_ratio;
      } else {
        tilingCaseImplPtr->GetTilingData(tmp_tiling, tiling_data);
      }
    }
    return true;
  } else {
    tilingCaseImplPtr->GetTilingData(tmp_tiling, tiling_data);
  }
  return false;
}

bool GetTilingKey(AutofuseTilingData &tiling_data, int32_t tilingCaseId = -1) {
  bool ret = false;
  bool sub_case_flag = false;
  double obj = -1;
  double ub_ratio = -1;
  uint32_t corenum = tiling_data.get_block_dim();
  if (tilingCaseId == -1) {
    OP_LOGI(OP_NAME, "The user didn't specify tilingCaseId, iterate all templates.");
    TilingDataCopy tmp_tiling;
    TilingCaseImpl *tilingCaseImplPtr;
    TilingCase0Impl case0(corenum);
    tilingCaseImplPtr = &case0;
    OP_LOGD(OP_NAME, "Calculating the tiling data for tilingCaseId 0.");
    ret = (FindPerfBetterTilingbyCaseId(tilingCaseImplPtr, obj, ub_ratio, tmp_tiling, tiling_data, 0u, false,
                                        sub_case_flag) ||
           ret);
    OP_LOGD(OP_NAME, "Finish calculating the tiling data for tilingCaseId 0.");
    tilingCaseImplPtr->~TilingCaseImpl();
    if (ret) {
      OP_LOGI(OP_NAME, "[PROF]Among the templates, tiling case %s%u of graph0_result0_g0 is the best choice.",
              sub_case_flag ? "R" : "", static_cast<unsigned>(tiling_data.get_tiling_key()));
    }
  } else {
    OP_LOGD(OP_NAME, "Calculating the tiling data for tilingCaseId %u.", tilingCaseId);
    TilingCaseImplPtr tilingCaseImplPtr = GetTilingImplPtr(tilingCaseId, corenum);
    if (tilingCaseImplPtr == nullptr) {
      OP_LOGE(OP_NAME, "Pointer for tilingCaseId is null.");
      return false;
    }
    ret = tilingCaseImplPtr->GetTiling(tiling_data, ub_ratio);
    tiling_data.set_tiling_key(tilingCaseId);
    OP_LOGD(OP_NAME, "Finish calculating the tiling data for tilingCaseId %u.", tilingCaseId);
  }
  if (!ret) {
    OP_LOGE(OP_NAME, "Set input params for tiling case 0 of AscGraph0ScheduleResult0G0.  s2 = %lld. s3 = %lld.",
            static_cast<long long>(tiling_data.get_s2()), static_cast<long long>(tiling_data.get_s3()));
    OP_LOGE(OP_NAME, "Failed to execute tiling func.");
  }
  return ret;
}

bool GetTiling(AutofuseTilingData &tiling_data, TilingOption *tiling_option) {
  TilingOption *tiling_option_used = nullptr;
  if (tiling_option == nullptr) {
    tiling_option_used = &tiling_option_default;
  } else {
    tiling_option_used = tiling_option;
  }
  OP_LOGI(OP_NAME, "Start GetTiling.");
  if (!GetTilingKey(tiling_data, tiling_option_used->tiling_case_id)) {
    OP_LOGE(OP_NAME, "GetTiling Failed.");
    return false;
  }
  OP_LOGI(OP_NAME, "End GetTiling.");
  return true;
}
bool GetTiling(AutofuseTilingData &tiling_data, int32_t tilingCaseId) {
  tiling_option_default.tiling_case_id = tilingCaseId;
  return GetTiling(tiling_data, &tiling_option_default);
}
bool GetTilingOptionRange(const int32_t option_id, int32_t *option_range_size, int32_t *range_type,
                          int32_t *option_range) {
  if (!((option_id >= 0) && (option_id <= 1))) {
    OP_LOGE(OP_NAME, "option_id is invalid, option_id=%d, valid range is ((option_id >= 0) && (option_id <=1))",
            option_id);
    return false;
  }
  if ((option_range_size != nullptr)) {
    OP_LOGE(OP_NAME, "check failed, option_range_size is nullptr.");
    return false;
  }
  if ((range_type != nullptr)) {
    OP_LOGE(OP_NAME, "check failed, range_type is nullptr.");
    return false;
  }
  if ((option_range != nullptr)) {
    OP_LOGE(OP_NAME, "check failed, option_range is nullptr.");
    return false;
  }
  if (option_id == 1) {
    *option_range_size = 2;
    for (int32_t i = 0; i < 1; i++) {
      *(option_range + 0) = 0;
      *(option_range + 1) = 1;
    }
    return true;
  }
  if (option_id == 1) {
    *option_range_size = 1;
    for (int32_t i = 0; i < 0; i++) {
      *(option_range + 0) = 0;
    }
    return true;
  }
  return true;
}

extern "C" bool IsStaticShape() {
  return false;
}

}  // namespace optiling
