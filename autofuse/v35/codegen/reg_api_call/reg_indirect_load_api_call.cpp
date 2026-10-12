/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reg_indirect_load_api_call.h"

#include <algorithm>
#include <cctype>
#include <map>
#include <sstream>
#include <unordered_map>
#include "api_call/utils/api_call_factory.h"
#include "ascir_ops.h"
#include "ascir_ops_utils.h"
#include "common/checker.h"
#include "common_utils.h"
#include "indirect_load_utils.h"
#include "v35/ascir/ascir_codegen_v2.h"

namespace codegen {
namespace {
constexpr size_t kIndirectLoadInputCount = 2UL;
constexpr size_t kIndirectLoadOutputCount = 1UL;
constexpr char kSimtContextNamePrefix[] = "IndirectLoadSimtContext_";
constexpr char kSimtBodyNamePrefix[] = "IndirectLoadSimtBody_";
constexpr char kSimtBodyGuardPrefix[] = "AUTOFUSE_INDIRECT_LOAD_SIMT_BODY_DEFINED_";
constexpr char kGlobalTensorNamePrefix[] = "global_";
constexpr char kSimtGmFieldNamePrefix[] = "gm_";
constexpr char kSimtValueNamePrefix[] = "v_";

Status GenerateSimtContextDefinition(const std::string &context_name,
                                     const std::vector<ascgen_utils::indirect_load::SimtGmTensorMetadata> &gm_tensors,
                                     bool has_output_inner_size, std::stringstream &ss) {
  ss << "struct " << context_name << " {" << std::endl;
  for (const auto &tensor : gm_tensors) {
    std::string dtype;
    GE_ASSERT_SUCCESS(Tensor::DtypeName(tensor.dtype, dtype));
    if (tensor.is_scalar) {
      ss << "  " << dtype << " " << kSimtValueNamePrefix << tensor.value_tensor_id << ";" << std::endl;
    } else {
      ss << "  __gm__ " << dtype << " *" << kSimtGmFieldNamePrefix << tensor.value_tensor_id << ";" << std::endl;
    }
  }
  // 后置 Reduce 的局部输出按 Inner size 对齐写回；该 size 可能是运行期符号
  // （如 t->ksN），不能内联进文件作用域的 Store，需经 Context 传入。
  if (has_output_inner_size) {
    ss << "  uint64_t output_inner_size;" << std::endl;
  }
  ss << "};" << std::endl;
  return af::SUCCESS;
}

af::Status EmitSimtScalarExpr(const ascir::NodeView &node, const std::vector<std::string> &inputs, std::string &expr) {
  GE_ASSERT_NOTNULL(node, "SIMT scalar node is null.");
  GE_ASSERT_TRUE(!inputs.empty() && inputs.size() == node->inputs.Size(),
                 "SIMT scalar node %s[%s] expects %zu non-empty inputs, but got %zu.", node->GetTypePtr(),
                 node->GetNamePtr(), node->inputs.Size(), inputs.size());
  const auto impl = ascgen_utils::GetAscIrCodegenImpl(node->GetType());
  GE_ASSERT_NOTNULL(impl, "SIMT scalar codegen is not registered for node %s[%s].", node->GetTypePtr(),
                    node->GetNamePtr());
  const auto *v2_impl = dynamic_cast<af::ascir::AscIrCodegenV2 *>(impl.get());
  GE_ASSERT_NOTNULL(v2_impl, "SIMT scalar codegen for node %s[%s] is not a V2 implementation.", node->GetTypePtr(),
                    node->GetNamePtr());
  GE_ASSERT_TRUE(v2_impl->IsSimtScalarSupported(*node), "SIMT scalar codegen is not supported for node %s[%s].",
                 node->GetTypePtr(), node->GetNamePtr());
  return v2_impl->GenerateSimtScalarExpr(*node, inputs, expr);
}

bool IsAxisDerivedFrom(const TPipe &tpipe, ascir::AxisId axis_id, ascir::AxisId ancestor_axis_id) {
  if (axis_id == ancestor_axis_id) {
    return true;
  }
  for (ascir::AxisId from_axis_id : tpipe.tiler.GetAxis(axis_id).from) {
    if (IsAxisDerivedFrom(tpipe, from_axis_id, ancestor_axis_id)) {
      return true;
    }
  }
  return false;
}

af::Status BuildTensorWindowInfo(const ascgen_utils::indirect_load::IndirectLoadTensorLayout &layout,
                                 const Tensor &tensor, size_t axis_pos, LogicalTensorInfo &info) {
  GE_ASSERT_TRUE(layout.axis_ids.size() == layout.sizes.size() && layout.sizes.size() == layout.strides.size(),
                 "IndirectLoad tensor window layout rank mismatch.");
  GE_ASSERT_TRUE(axis_pos < layout.sizes.size(), "IndirectLoad tensor window axis %zu is out of range [0, %zu).",
                 axis_pos, layout.sizes.size());
  GE_ASSERT_TRUE(tensor.vectorized_axis.size() == tensor.vectorized_strides.size(),
                 "IndirectLoad tensor vectorized axis/stride rank mismatch.");
  info = LogicalTensorInfo(layout);
  // Dense layouts keep their logical row-major strides after the local window is built. For a zero-stride-compact
  // view, preserve zero-stride axes but derive non-zero window strides from the physical vectorized tensor view;
  // bitwidth-changing producers can introduce padding between logical elements.
  if (layout.kind == ascgen_utils::indirect_load::IndirectLoadLayoutKind::kDense) {
    return af::SUCCESS;
  }
  af::Expression compact_stride = af::sym::kSymbolOne;
  for (size_t index = info.sizes.size(); index > axis_pos; --index) {
    const size_t dim = index - 1UL;
    if (af::SymbolicUtils::StaticCheckEq(layout.strides[dim], af::sym::kSymbolZero) == af::TriBool::kTrue) {
      info.strides[dim] = af::sym::kSymbolZero;
      continue;
    }
    const auto vectorized_axis =
        std::find(tensor.vectorized_axis.begin(), tensor.vectorized_axis.end(), layout.axis_ids[dim]);
    if (vectorized_axis != tensor.vectorized_axis.end()) {
      info.strides[dim] =
          tensor
              .vectorized_strides[static_cast<size_t>(std::distance(tensor.vectorized_axis.begin(), vectorized_axis))];
    } else {
      info.strides[dim] = compact_stride;
    }
    if (af::SymbolicUtils::StaticCheckEq(info.strides[dim], af::sym::kSymbolZero) != af::TriBool::kTrue) {
      compact_stride = af::sym::Mul(info.strides[dim], info.sizes[dim]);
    }
  }
  return af::SUCCESS;
}

std::string PromoteSizeExpr(const std::string &expr, const std::string &type) {
  if (type == "uint32_t") {
    return expr;
  }
  const size_t begin = expr.find("t->");
  if (begin == std::string::npos) {
    return "static_cast<" + type + ">(" + expr + ")";
  }
  size_t end = begin + 3U;
  while (end < expr.size() && (std::isalnum(static_cast<unsigned char>(expr[end])) != 0 || expr[end] == '_')) {
    ++end;
  }
  return expr.substr(0U, begin) + "static_cast<" + type + ">(" + expr.substr(begin, end - begin) + ")" +
         expr.substr(end);
}

std::string JoinSizeExprs(const std::vector<ascir::SizeExpr> &exprs, const TPipe &tpipe,
                          const std::string &type = "uint32_t") {
  std::stringstream ss;
  for (size_t i = 0; i < exprs.size(); ++i) {
    if (i > 0) {
      ss << ", ";
    }
    ss << PromoteSizeExpr(tpipe.tiler.Size(exprs[i]), type);
  }
  return ss.str();
}

// [strided 视图扁平化] 调度器把节点视图 merge/split 到模板轴空间（如 a0 拆为
// [128/Tb, Tb, 1]）后，strided index load 的物理视图与逻辑输出 rank 不再对齐，
// 坐标折叠循环无法直接消费。去掉退化轴（size==1；stride==0 且 size>1 的广播轴
// 承载重复语义、不可去）后，把满足 strides[i] == strides[i+1] * sizes[i+1] 的
// 相邻轴合并（merged size = sizes[i] * sizes[i+1]，merged stride = strides[i+1]），
// 得到遍历顺序不变的语义等价低 rank 视图。符号乘积化简（如 (128/Tb)*Tb == 128）
// 与 IsDenseEquivalentView 的总量比较同源，已被生产 dense 等价路径验证。
void FlattenSimtStridedView(ascgen_utils::indirect_load::LogicalTensorView &view) {
  if (view.sizes.size() != view.strides.size()) {
    return;
  }
  const bool has_axis_ids = view.axis_ids.size() == view.sizes.size();
  std::vector<af::AxisId> axis_ids;
  std::vector<af::Expression> sizes;
  std::vector<af::Expression> strides;
  for (size_t dim = 0UL; dim < view.sizes.size(); ++dim) {
    if (af::SymbolicUtils::StaticCheckEq(view.sizes[dim], af::ops::One) == af::TriBool::kTrue) {
      continue;
    }
    if (has_axis_ids) {
      axis_ids.emplace_back(view.axis_ids[dim]);
    }
    sizes.emplace_back(view.sizes[dim]);
    strides.emplace_back(view.strides[dim]);
  }
  bool merged = true;
  while (merged && sizes.size() > 1UL) {
    merged = false;
    for (size_t dim = 0UL; dim + 1UL < sizes.size(); ++dim) {
      if (af::SymbolicUtils::StaticCheckEq(strides[dim], af::sym::Mul(strides[dim + 1UL], sizes[dim + 1UL])) !=
          af::TriBool::kTrue) {
        continue;
      }
      sizes[dim] = af::sym::Mul(sizes[dim], sizes[dim + 1UL]);
      strides[dim] = strides[dim + 1UL];
      if (has_axis_ids) {
        axis_ids.erase(axis_ids.begin() + static_cast<int64_t>(dim) + 1);
      }
      sizes.erase(sizes.begin() + static_cast<int64_t>(dim) + 1);
      strides.erase(strides.begin() + static_cast<int64_t>(dim) + 1);
      merged = true;
      break;
    }
  }
  view.axis_ids = std::move(axis_ids);
  view.sizes = std::move(sizes);
  view.strides = std::move(strides);
}

af::Status BuildSimtPerLoadIndexOffsetExpressions(const ascgen_utils::indirect_load::TemplateLogicalView &logical_view,
                                                  const std::vector<af::AscNodePtr> &nodes,
                                                  const SimtLoadMetadataMap &load_metadata, const TPipe &tpipe,
                                                  const std::string &offset_type, const std::string &output_index_expr,
                                                  const std::string &direct_offset_expr, bool has_post_reduce,
                                                  SimtLoadIndexOffsetExpressions &expressions, std::stringstream &ss) {
  expressions.clear();
  const size_t rank = logical_view.output.sizes.size();
  GE_ASSERT_TRUE(logical_view.output.strides.size() == rank, "SIMT output view rank mismatch for index offsets.");
  std::map<std::pair<size_t, std::string>, std::string> coordinates;
  for (const af::AscNodePtr &node : nodes) {
    const auto load = load_metadata.find(node->GetName());
    if (load == load_metadata.end()) {
      continue;
    }
    af::Expression load_offset = af::ops::Zero;
    if (node->attr.ir_attr != nullptr) {
      (void)node->attr.ir_attr->GetAttrValue("offset", load_offset);
    }
    const bool has_load_offset = af::SymbolicUtils::StaticCheckEq(load_offset, af::ops::Zero) != af::TriBool::kTrue;
    const std::string load_offset_expr =
        has_load_offset ? PromoteSizeExpr(tpipe.tiler.Size(load_offset), offset_type) : "";
    const auto append_load_offset = [&load_offset_expr, has_load_offset](std::string offset) {
      if (has_load_offset) {
        offset += " + " + load_offset_expr;
      }
      return offset;
    };
    if (!load->second.use_logical_offset) {
      if (has_load_offset) {
        std::string base_offset;
        switch (load->second.address_source) {
          case ascgen_utils::indirect_load::SimtLoadAddressSource::kZeroOffset:
            base_offset = "0";
            break;
          case ascgen_utils::indirect_load::SimtLoadAddressSource::kIndexOffset:
            base_offset = "index_offset";
            break;
          case ascgen_utils::indirect_load::SimtLoadAddressSource::kOutputOffset:
            base_offset = direct_offset_expr;
            break;
        }
        expressions[node->GetName()] = append_load_offset(base_offset);
      }
      continue;
    }
    ascgen_utils::indirect_load::LogicalTensorView view = load->second.physical_view;
    // 调度会把普通节点的 tensor view merge/split 到模板轴空间（如 a0 拆为
    // [64/a0Tb, a0Tb]），视图 rank 可与输出逻辑视图不同但语义 dense 同构：等价时
    // 直接使用 output_index，避免在 SIMT 静态成员函数（无 tiling data 参数）内生成
    // 引用 t-> 求解变量的坐标表达式。dense 判定不要求 rank 相等，置于 rank 断言之前。
    if (ascgen_utils::indirect_load::IsDenseEquivalentView(view, logical_view.output)) {
      expressions[node->GetName()] = append_load_offset(output_index_expr);
      continue;
    }
    if (view.sizes.size() != rank || view.strides.size() != rank) {
      // gather+norm 复合图的 side-input Load 可能是退化视图（全部轴 size==1，任意
      // 位置读同一元素）：偏移恒为 0，无需坐标重建。
      const bool degenerate = view.sizes.size() == view.strides.size() &&
                              std::all_of(view.sizes.begin(), view.sizes.end(), [](const af::Expression &size) {
                                return af::SymbolicUtils::StaticCheckEq(size, af::ops::One) == af::TriBool::kTrue;
                              });
      if (degenerate) {
        expressions[node->GetName()] = append_load_offset("0");
        continue;
      }
      // 归约为『零贡献前缀（stride==0 或 size==1）+ 常量尺寸的稠密尾段』的广播
      // side-input（如生产 softmax 图 load4 的 [1,1,1,2048]）：正确寻址是对尾段
      // 尺寸取模（每 tail_size 个输出元素重复一轮）。尾段尺寸必须为编译期常量：
      // SIMT body 是静态成员函数，无 tiling data 参数，不能引用 t-> 运行时变量。
      // （该形态源于 SIMT 角色节点视图 split 产生的 rank 差与 develop ea563ac0
      // 强制 GM load 坐标重建的交互，与 tile 形态无关，固定 tile 同样触发。）
      int64_t dense_tail_size = 1;
      bool has_dense_tail = true;
      for (size_t dim = 0UL; dim < view.sizes.size(); ++dim) {
        const bool zero_contribution =
            (af::SymbolicUtils::StaticCheckEq(view.strides[dim], af::ops::Zero) == af::TriBool::kTrue) ||
            (af::SymbolicUtils::StaticCheckEq(view.sizes[dim], af::ops::One) == af::TriBool::kTrue);
        if (zero_contribution) {
          continue;
        }
        const bool unit_stride =
            af::SymbolicUtils::StaticCheckEq(view.strides[dim], af::ops::One) == af::TriBool::kTrue;
        int64_t tail_const = 0;
        if (!unit_stride || !view.sizes[dim].GetConstValue(tail_const)) {
          has_dense_tail = false;
          break;
        }
        dense_tail_size *= tail_const;
      }
      if (has_dense_tail && dense_tail_size > 1) {
        expressions[node->GetName()] = append_load_offset(output_index_expr + " % " + std::to_string(dense_tail_size));
        continue;
      }
      // [行级广播 side-input] 尾轴零贡献（stride==0 且 size==1、其余轴稠密）的广播
      // 形态（如生产 gather+softmax 图 load3 [8,2048,1]/[2048,1,0]）：每行读一个
      // 值，正确寻址=行号×行宽。调度期 SIMT 边界的轴 split 会把视图改写为 rank
      // 不匹配且尺寸符号化（tiling 变量）形态——稠密尾段取模兜底要求编译期常量而
      // 失效（历史回归：带 arange/matmul 的图 e46cedb6）。行号定位不依赖被 split
      // 改写的尺寸：行宽=logical 输出除首轴外的元素积（logical_view 与 split 无关）。
      {
        const auto load_it = load_metadata.find(node->GetName());
        if (load_it != load_metadata.end() && load_it->second.is_row_broadcast && !logical_view.output.sizes.empty()) {
          // 语义：读 [行,列] 的值沿尾轴广播（原始视图 [行,列,1]/strides=[行宽,1,0]）。
          // 正确寻址 = output_index 去掉零贡献尾维：/ 尾轴宽（logical 输出尾轴，与
          // split 改写无关）。行首×行宽的折叠是错误语义（会把列方向也折叠）。
          const auto &tail_size = logical_view.output.sizes.back();
          int64_t tail_const = 0;
          if (tail_size.GetConstValue(tail_const) && tail_const > 0) {
            expressions[node->GetName()] =
                append_load_offset("(" + output_index_expr + ") / " + std::to_string(tail_const));
            continue;
          }
        }
      }
      int64_t payload_size = 0;
      if (has_post_reduce && !view.sizes.empty() && !view.strides.empty() &&
          af::SymbolicUtils::StaticCheckEq(view.sizes.back(), af::ops::One) == af::TriBool::kTrue &&
          af::SymbolicUtils::StaticCheckEq(view.strides.back(), af::ops::Zero) == af::TriBool::kTrue &&
          !logical_view.output.sizes.empty() && logical_view.output.sizes.back().GetConstValue(payload_size) &&
          payload_size > 1) {
        expressions[node->GetName()] = append_load_offset(output_index_expr + " / " + std::to_string(payload_size));
        continue;
      }
      // [strided 视图扁平化] 调度器 split 改写产生的 rank 差 strided 视图（如
      // [128]/[39]+offset 列切片被改写为 [128/Tb, Tb, 1]/[39*Tb, 39, 39]）此前
      // 均走 dense 线性路径，折叠路径从未覆盖：先符号化扁平化（去退化轴 + 相邻
      // 轴合并）再与输出 rank 对齐。折叠循环按输出轴逐维取模，要求扁平化后各维
      // 尺寸与输出逻辑视图一致，否则语义未对齐（如过度合并成 [4,4]/[8,1] 对
      // [2,8] 输出），保留原断言兜底。
      FlattenSimtStridedView(view);
      bool flattened_aligns = view.sizes.size() == rank && view.strides.size() == rank;
      for (size_t dim = 0UL; flattened_aligns && dim < rank; ++dim) {
        flattened_aligns =
            af::SymbolicUtils::StaticCheckEq(view.sizes[dim], logical_view.output.sizes[dim]) == af::TriBool::kTrue;
      }
      if (!flattened_aligns) {
        GE_ASSERT_TRUE(false, "SIMT index Load[%s] physical view rank mismatch.", node->GetNamePtr());
        continue;
      }
    }

    std::string offset;
    for (size_t dim = 0UL; dim < rank; ++dim) {
      if (af::SymbolicUtils::StaticCheckEq(view.strides[dim], af::ops::Zero) == af::TriBool::kTrue ||
          af::SymbolicUtils::StaticCheckEq(view.sizes[dim], af::ops::One) == af::TriBool::kTrue) {
        continue;
      }
      const std::string output_stride =
          PromoteSizeExpr(tpipe.tiler.Size(logical_view.output.strides[dim]), offset_type);
      const std::string output_size = PromoteSizeExpr(tpipe.tiler.Size(view.sizes[dim]), offset_type);
      const std::string load_stride = PromoteSizeExpr(tpipe.tiler.Size(view.strides[dim]), offset_type);
      // Different physical extents on the same axis must not share a modulo expression.
      const auto key = std::make_pair(dim, output_size);
      auto found = coordinates.find(key);
      if (found == coordinates.end()) {
        const std::string name = "index_coord_" + std::to_string(coordinates.size());
        const bool unit_output_stride =
            af::SymbolicUtils::StaticCheckEq(logical_view.output.strides[dim], af::ops::One) == af::TriBool::kTrue;
        const std::string dividend =
            unit_output_stride ? output_index_expr : "((" + output_index_expr + ") / " + output_stride + ")";
        ss << "    const " << offset_type << " " << name << " = static_cast<" << offset_type << ">(" << dividend
           << " % " << output_size << ");" << std::endl;
        found = coordinates.emplace(key, name).first;
      }
      const bool unit_load_stride =
          af::SymbolicUtils::StaticCheckEq(view.strides[dim], af::ops::One) == af::TriBool::kTrue;
      const std::string term = unit_load_stride ? found->second : found->second + " * " + load_stride;
      offset += (offset.empty() ? "" : " + ") + term;
    }
    expressions[node->GetName()] = append_load_offset(offset.empty() ? "0" : offset);
  }
  return af::SUCCESS;
}

std::string GetSimtOffsetType(const ascgen_utils::indirect_load::SimtPolicyMetadata &policy) {
  return policy.offset_width == ascgen_utils::indirect_load::SimtOffsetWidth::kUint32 ? "uint32_t" : "uint64_t";
}

std::string GetSimtCaseTag(const ascgen_utils::indirect_load::SimtPolicyMetadata &policy, size_t rank, int64_t axis) {
  const std::string offset_type = GetSimtOffsetType(policy);
  std::stringstream ss;
  ss << "AscendC::IndirectLoadSimtCaseTag<static_cast<AscendC::IndirectLoadSimtCase>("
     << static_cast<int64_t>(policy.policy) << "), " << offset_type << ", " << rank << ", " << axis << ", "
     << policy.inner_span << "ULL, " << policy.output_axis_span << "ULL, " << policy.input_axis_stride << "ULL, "
     << policy.input_axis_span << "ULL, " << policy.input_stride_mask << "ULL, " << policy.index_stride_mask << "ULL>";
  return ss.str();
}

std::string GetSimtPolicyParams(const ascgen_utils::indirect_load::SimtPolicyMetadata &policy, const TPipe &tpipe) {
  const std::string offset_type = GetSimtOffsetType(policy);
  std::stringstream ss;
  for (size_t i = 0UL; i < policy.runtime_params.size(); ++i) {
    if (i != 0UL) {
      ss << ", ";
    }
    ss << "static_cast<" << offset_type << ">(";
    ss << PromoteSizeExpr(tpipe.tiler.Size(policy.runtime_params[i]), offset_type) << ")";
  }
  return ss.str();
}

bool FindCurrentAxisVar(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis, Axis::Type axis_type,
                        ascir::AxisId ancestor_axis_id, std::string &axis_var) {
  for (ascir::AxisId axis_id : current_axis) {
    const Axis &axis = tpipe.tiler.GetAxis(axis_id);
    if (axis.type == axis_type && IsAxisDerivedFrom(tpipe, axis_id, ancestor_axis_id)) {
      axis_var = axis.Str();
      return true;
    }
  }
  return false;
}

af::Status CheckDenseStrides(const LogicalTensorInfo &tensor, const char *tensor_name) {
  af::Expression expected_stride = af::ops::One;
  for (int64_t i = static_cast<int64_t>(tensor.sizes.size()) - 1; i >= 0; --i) {
    if (af::SymbolicUtils::StaticCheckEq(tensor.sizes[i], af::ops::One) == af::TriBool::kTrue) {
      continue;
    }
    GE_ASSERT_TRUE(af::SymbolicUtils::StaticCheckEq(tensor.strides[i], expected_stride) == af::TriBool::kTrue,
                   "IndirectLoad %s must be dense contiguous.", tensor_name);
    expected_stride = af::sym::Mul(expected_stride, tensor.sizes[i]);
  }
  return af::SUCCESS;
}

af::Status CheckIndirectLoadShape(const ascgen_utils::indirect_load::TemplateLogicalView &logical_view,
                                  const Tensor &output_tensor) {
  const auto &input = logical_view.input;
  const auto &index = logical_view.index;
  const auto &output = logical_view.output;
  GE_ASSERT_TRUE(input.sizes.size() == index.sizes.size(), "Invalid IndirectLoad logical rank, input:%zu, index:%zu.",
                 input.sizes.size(), index.sizes.size());
  GE_ASSERT_TRUE(index.sizes.size() == output.sizes.size(), "Invalid IndirectLoad logical rank, index:%zu, output:%zu.",
                 index.sizes.size(), output.sizes.size());
  GE_ASSERT_TRUE(input.kind != ascgen_utils::indirect_load::IndirectLoadLayoutKind::kUnsupported &&
                     index.kind != ascgen_utils::indirect_load::IndirectLoadLayoutKind::kUnsupported,
                 "IndirectLoad input or index layout is unsupported.");
  GE_ASSERT_TRUE(input.sizes.size() == input.strides.size() && index.sizes.size() == index.strides.size(),
                 "IndirectLoad logical sizes/strides rank mismatch.");
  GE_ASSERT_SUCCESS(CheckDenseStrides(LogicalTensorInfo(output.sizes, output.strides), "output logical view"));
  GE_ASSERT_TRUE(output_tensor.axis_size.size() == output_tensor.axis_strides.size(),
                 "IndirectLoad output tensor sizes/strides rank mismatch.");
  GE_ASSERT_SUCCESS(
      CheckDenseStrides(LogicalTensorInfo(output_tensor.axis_size, output_tensor.axis_strides), "output tensor"));
  return af::SUCCESS;
}

af::Status EmitSimtScalarInput(const af::AscNodePtr &node, std::map<ascir::TensorId, std::string> &values) {
  std::string value;
  GE_ASSERT_NOTNULL(node->attr.ir_attr, "IndirectLoad SIMT Scalar node[%s] has no IR attr.", node->GetNamePtr());
  GE_ASSERT_GRAPH_SUCCESS(node->attr.ir_attr->GetAttrValue("value", value));
  const auto output = node->outputs()[0];
  std::string dtype;
  GE_ASSERT_SUCCESS(Tensor::DtypeName(output->attr.dtype, dtype));
  std::string processed_value;
  GE_ASSERT_SUCCESS(ascgen_utils::ScalarValuePreProcess(value, dtype, processed_value));
  values[output->attr.mem.tensor_id] = "static_cast<" + dtype + ">(" + processed_value + ")";
  return af::SUCCESS;
}

af::Status EmitSimtTransform(const af::AscNodePtr &node, std::map<ascir::TensorId, std::string> &values,
                             std::stringstream &ss) {
  std::vector<std::string> inputs;
  inputs.reserve(node->inputs.Size());
  for (size_t i = 0UL; i < node->inputs.Size(); ++i) {
    const auto found = values.find(node->inputs()[i]->attr.mem.tensor_id);
    GE_ASSERT_TRUE(found != values.end(), "SIMT node[%s] input[%zu] has no scalar value.", node->GetNamePtr(), i);
    inputs.emplace_back(found->second);
  }
  std::string expr;
  GE_ASSERT_SUCCESS(EmitSimtScalarExpr(node, inputs, expr));
  const auto output = node->outputs()[0];
  // Scalar identity expressions need no intermediate. Keep GM loads materialized to avoid duplicate reads.
  const bool is_gm_load = expr.rfind("context.gm_", 0UL) == 0UL && expr.find('[') != std::string::npos;
  if (inputs.size() == 1UL && expr == inputs.front() && !is_gm_load) {
    values[output->attr.mem.tensor_id] = expr;
    return af::SUCCESS;
  }
  std::string output_dtype;
  GE_ASSERT_SUCCESS(Tensor::DtypeName(output->attr.dtype, output_dtype));
  const std::string variable = kSimtValueNamePrefix + std::to_string(output->attr.mem.tensor_id);
  ss << "    " << output_dtype << " " << variable << " = " << expr << ";" << std::endl;
  values[output->attr.mem.tensor_id] = variable;
  return af::SUCCESS;
}

af::Status EmitSimtArangeExpr(const af::AscNodePtr &node,
                              const ascgen_utils::indirect_load::TemplateLogicalView &logical_view,
                              const std::string &offset_expr, std::map<ascir::TensorId, std::string> &values) {
  const auto impl = ascgen_utils::GetAscIrCodegenImpl(node->GetType());
  GE_ASSERT_NOTNULL(impl, "SIMT zero-input codegen is not registered for node %s[%s].", node->GetTypePtr(),
                    node->GetNamePtr());
  const auto *v2_impl = dynamic_cast<af::ascir::AscIrCodegenV2 *>(impl.get());
  GE_ASSERT_NOTNULL(v2_impl, "SIMT zero-input codegen for node %s[%s] is not V2.", node->GetTypePtr(),
                    node->GetNamePtr());
  GE_ASSERT_TRUE(v2_impl->IsSimtScalarSupported(*node), "SIMT scalar codegen is not supported for node %s[%s].",
                 node->GetTypePtr(), node->GetNamePtr());
  const auto &output = node->outputs()[0]->attr;
  const auto varying = std::find_if(output.strides.begin(), output.strides.end(), [](const af::Expression &stride) {
    return af::SymbolicUtils::StaticCheckEq(stride, af::sym::kSymbolZero) != af::TriBool::kTrue;
  });
  GE_ASSERT_TRUE(varying != output.strides.end(), "Arange has no varying axis.");
  const auto axis = output.axis[static_cast<size_t>(std::distance(output.strides.begin(), varying))];
  const auto logical_axis = std::find(logical_view.output.axis_ids.begin(), logical_view.output.axis_ids.end(), axis);
  GE_ASSERT_TRUE(logical_axis != logical_view.output.axis_ids.end(), "Arange varying axis is absent from output view.");
  const size_t logical_pos = static_cast<size_t>(std::distance(logical_view.output.axis_ids.begin(), logical_axis));
  GE_ASSERT_TRUE(logical_pos < logical_view.output.sizes.size() && logical_pos < logical_view.output.strides.size(),
                 "Arange output view metadata is incomplete.");
  std::string coordinate = std::string("(") + offset_expr;
  if (af::SymbolicUtils::StaticCheckEq(logical_view.output.strides[logical_pos], af::sym::kSymbolOne) !=
      af::TriBool::kTrue) {
    coordinate += std::string(" / ") + logical_view.output.strides[logical_pos].Str().get();
  }
  coordinate += std::string(") % (") + logical_view.output.sizes[logical_pos].Str().get() + ")";
  std::string expr;
  GE_ASSERT_GRAPH_SUCCESS(v2_impl->GenerateSimtScalarExpr(*node, {coordinate}, expr));
  values[node->outputs()[0]->attr.mem.tensor_id] = expr;
  return af::SUCCESS;
}

af::Status EmitSimtEvaluatorNodes(const std::vector<af::AscNodePtr> &nodes, const SimtLoadMetadataMap &load_metadata,
                                  const SimtLoadIndexOffsetExpressions *index_offset_expressions,
                                  const ascgen_utils::indirect_load::TemplateLogicalView &logical_view,
                                  const std::string &output_offset_expr, const std::string &arange_offset_expr,
                                  std::map<ascir::TensorId, std::string> &values, std::stringstream &ss) {
  for (const af::AscNodePtr &node : nodes) {
    const auto load = load_metadata.find(node->GetName());
    if (load != load_metadata.end()) {
      const auto output = node->outputs()[0];
      if (values.count(output->attr.mem.tensor_id) != 0UL) {
        continue;
      }
      const char *offset = nullptr;
      std::string custom_offset;
      if (index_offset_expressions != nullptr) {
        const auto expression = index_offset_expressions->find(node->GetName());
        if (expression != index_offset_expressions->end()) {
          custom_offset = expression->second;
          offset = custom_offset.c_str();
        }
      }
      if (offset == nullptr) {
        switch (load->second.address_source) {
          case ascgen_utils::indirect_load::SimtLoadAddressSource::kZeroOffset:
            offset = "0";
            break;
          case ascgen_utils::indirect_load::SimtLoadAddressSource::kIndexOffset:
            offset = "index_offset";
            break;
          case ascgen_utils::indirect_load::SimtLoadAddressSource::kOutputOffset:
            offset = output_offset_expr.c_str();
            break;
        }
      }
      values[output->attr.mem.tensor_id] = "context." + std::string(kSimtGmFieldNamePrefix) +
                                           std::to_string(output->attr.mem.tensor_id) + "[" + offset + "]";
      continue;
    }
    if (af::ops::IsOps<af::ascir_op::Scalar>(node)) {
      GE_ASSERT_SUCCESS(EmitSimtScalarInput(node, values));
      continue;
    }
    if (af::ops::IsOps<af::ascir_op::ScalarData>(node)) {
      const auto output = node->outputs()[0];
      const std::string variable =
          "context." + std::string(kSimtValueNamePrefix) + std::to_string(output->attr.mem.tensor_id);
      values[output->attr.mem.tensor_id] = variable;
      continue;
    }
    if (af::ops::IsOps<af::ascir_op::Arange>(node)) {
      GE_ASSERT_SUCCESS(EmitSimtArangeExpr(node, logical_view, arange_offset_expr, values));
      continue;
    }
    if (af::ops::IsOps<af::ascir_op::Store>(node)) {
      continue;
    }
    if (af::ops::IsOps<af::ascir_op::Transpose>(node)) {
      GE_ASSERT_TRUE(node->inputs.Size() == 1UL && !node->outputs().empty(), "SIMT transpose node[%s] must be unary.",
                     node->GetNamePtr());
      const auto input = node->inputs()[0];
      const auto output = node->outputs()[0];
      const auto found = values.find(input->attr.mem.tensor_id);
      GE_ASSERT_TRUE(found != values.end(), "SIMT transpose node[%s] input has no scalar value.", node->GetNamePtr());
      // Load->Transpose is represented as a layout-only boundary in SIMT;
      // each lane already addresses the corresponding scalar, so preserve
      // the value while transferring it to the transpose output tensor id.
      values[output->attr.mem.tensor_id] = found->second;
      continue;
    }
    GE_ASSERT_SUCCESS(EmitSimtTransform(node, values, ss));
  }
  return af::SUCCESS;
}

af::Status GenerateSimtEvaluatorBody(const std::vector<af::AscNodePtr> &nodes,
                                     std::map<ascir::TensorId, std::string> &values, ascir::TensorId result_tensor_id,
                                     const SimtLoadMetadataMap &load_metadata, std::stringstream &ss,
                                     const ascgen_utils::indirect_load::TemplateLogicalView &logical_view,
                                     const std::string &output_offset_expr, const std::string &arange_offset_expr,
                                     const SimtLoadIndexOffsetExpressions *index_offset_expressions = nullptr) {
  GE_ASSERT_SUCCESS(EmitSimtEvaluatorNodes(nodes, load_metadata, index_offset_expressions, logical_view,
                                           output_offset_expr, arange_offset_expr, values, ss));
  const auto result = values.find(result_tensor_id);
  GE_ASSERT_TRUE(result != values.end(), "SIMT evaluator result tensor[%ld] has no scalar value.", result_tensor_id);
  ss << "    return " << result->second << ";" << std::endl;
  ss << "  }" << std::endl;
  return af::SUCCESS;
}

bool IsAicoreScalarCastSupported(af::DataType input_dtype, af::DataType output_dtype) {
  const bool input_supported = input_dtype == af::DT_FLOAT || input_dtype == af::DT_DOUBLE ||
                               input_dtype == af::DT_INT8 || input_dtype == af::DT_INT16 ||
                               input_dtype == af::DT_INT32 || input_dtype == af::DT_INT64 ||
                               input_dtype == af::DT_UINT8 || input_dtype == af::DT_UINT16 ||
                               input_dtype == af::DT_UINT32 || input_dtype == af::DT_UINT64;
  const bool output_supported = output_dtype == af::DT_FLOAT || output_dtype == af::DT_DOUBLE ||
                                output_dtype == af::DT_INT8 || output_dtype == af::DT_INT16 ||
                                output_dtype == af::DT_INT32 || output_dtype == af::DT_INT64 ||
                                output_dtype == af::DT_UINT8 || output_dtype == af::DT_UINT16 ||
                                output_dtype == af::DT_UINT32 || output_dtype == af::DT_UINT64;
  return input_supported && output_supported;
}

bool CanGenerateAicoreIndexEvaluator(const std::vector<af::AscNodePtr> &nodes,
                                     const SimtLoadMetadataMap &load_metadata) {
  for (const af::AscNodePtr &node : nodes) {
    if (load_metadata.count(node->GetName()) != 0UL || af::ops::IsOps<af::ascir_op::Scalar>(node) ||
        af::ops::IsOps<af::ascir_op::ScalarData>(node) || af::ops::IsOps<af::ascir_op::Store>(node) ||
        af::ops::IsOps<af::ascir_op::Transpose>(node)) {
      continue;
    }
    if (!af::ops::IsOps<af::ascir_op::Cast>(node) || node->inputs().size() != 1UL || node->outputs().empty() ||
        !IsAicoreScalarCastSupported(node->inputs()[0]->attr.dtype, node->outputs()[0]->attr.dtype)) {
      return false;
    }
  }
  return true;
}

af::Status EmitAicoreIndexEvaluatorNodes(const std::vector<af::AscNodePtr> &nodes,
                                         const SimtLoadMetadataMap &load_metadata,
                                         const SimtLoadIndexOffsetExpressions &index_offset_expressions,
                                         std::map<ascir::TensorId, std::string> &values, std::stringstream &ss) {
  for (const af::AscNodePtr &node : nodes) {
    const auto load = load_metadata.find(node->GetName());
    if (load != load_metadata.end()) {
      const auto output = node->outputs()[0];
      std::string offset;
      const auto custom_offset = index_offset_expressions.find(node->GetName());
      if (custom_offset != index_offset_expressions.end()) {
        offset = custom_offset->second;
      } else {
        switch (load->second.address_source) {
          case ascgen_utils::indirect_load::SimtLoadAddressSource::kZeroOffset:
            offset = "0";
            break;
          case ascgen_utils::indirect_load::SimtLoadAddressSource::kIndexOffset:
            offset = "index_offset";
            break;
          case ascgen_utils::indirect_load::SimtLoadAddressSource::kOutputOffset:
            offset = "output_index";
            break;
        }
      }
      values[output->attr.mem.tensor_id] = "context." + std::string(kSimtGmFieldNamePrefix) +
                                           std::to_string(output->attr.mem.tensor_id) + "[" + offset + "]";
      continue;
    }
    if (af::ops::IsOps<af::ascir_op::Scalar>(node)) {
      GE_ASSERT_SUCCESS(EmitSimtScalarInput(node, values));
      continue;
    }
    if (af::ops::IsOps<af::ascir_op::ScalarData>(node)) {
      const auto output = node->outputs()[0];
      values[output->attr.mem.tensor_id] =
          "context." + std::string(kSimtValueNamePrefix) + std::to_string(output->attr.mem.tensor_id);
      continue;
    }
    if (af::ops::IsOps<af::ascir_op::Store>(node)) {
      continue;
    }
    if (af::ops::IsOps<af::ascir_op::Transpose>(node)) {
      const auto found = values.find(node->inputs()[0]->attr.mem.tensor_id);
      GE_ASSERT_TRUE(found != values.end(), "AICore index Transpose node[%s] input has no scalar value.",
                     node->GetNamePtr());
      values[node->outputs()[0]->attr.mem.tensor_id] = found->second;
      continue;
    }

    GE_ASSERT_TRUE(af::ops::IsOps<af::ascir_op::Cast>(node) && node->inputs().size() == 1UL && !node->outputs().empty(),
                   "AICore index node[%s] is unsupported.", node->GetNamePtr());
    const auto found = values.find(node->inputs()[0]->attr.mem.tensor_id);
    GE_ASSERT_TRUE(found != values.end(), "AICore index Cast node[%s] input has no scalar value.", node->GetNamePtr());
    std::string output_dtype;
    GE_ASSERT_SUCCESS(Tensor::DtypeName(node->outputs()[0]->attr.dtype, output_dtype));
    const std::string variable = kSimtValueNamePrefix + std::to_string(node->outputs()[0]->attr.mem.tensor_id);
    ss << "    " << output_dtype << " " << variable << " = static_cast<" << output_dtype << ">(" << found->second
       << ");" << std::endl;
    values[node->outputs()[0]->attr.mem.tensor_id] = variable;
  }
  return af::SUCCESS;
}

af::Status GenerateAicoreIndexEvaluatorBody(const std::vector<af::AscNodePtr> &nodes,
                                            std::map<ascir::TensorId, std::string> &values,
                                            ascir::TensorId result_tensor_id, const SimtLoadMetadataMap &load_metadata,
                                            const SimtLoadIndexOffsetExpressions &index_offset_expressions,
                                            std::stringstream &ss) {
  GE_ASSERT_SUCCESS(EmitAicoreIndexEvaluatorNodes(nodes, load_metadata, index_offset_expressions, values, ss));
  const auto result = values.find(result_tensor_id);
  GE_ASSERT_TRUE(result != values.end(), "AICore index evaluator result tensor[%ld] has no scalar value.",
                 result_tensor_id);
  ss << "    return " << result->second << ";" << std::endl;
  ss << "  }" << std::endl;
  return af::SUCCESS;
}

bool IsSimtFloatingType(af::DataType dtype) {
  return dtype == af::DT_FLOAT16 || dtype == af::DT_BF16 || dtype == af::DT_FLOAT || dtype == af::DT_DOUBLE;
}

bool IsSimtSignedIntegerType(af::DataType dtype) {
  return dtype == af::DT_INT8 || dtype == af::DT_INT16 || dtype == af::DT_INT32 || dtype == af::DT_INT64;
}

bool IsSimtUnsignedIntegerType(af::DataType dtype) {
  return dtype == af::DT_UINT8 || dtype == af::DT_UINT16 || dtype == af::DT_UINT32 || dtype == af::DT_UINT64;
}

bool IsSameSimtCastCategory(af::DataType lhs, af::DataType rhs) {
  return (IsSimtFloatingType(lhs) && IsSimtFloatingType(rhs)) ||
         (IsSimtSignedIntegerType(lhs) && IsSimtSignedIntegerType(rhs)) ||
         (IsSimtUnsignedIntegerType(lhs) && IsSimtUnsignedIntegerType(rhs));
}

bool IsLosslessSimtCastChain(const std::vector<af::AscNodePtr> &nodes, ascir::TensorId value_tensor_id,
                             ascir::TensorId result_tensor_id) {
  ascir::TensorId current_tensor_id = value_tensor_id;
  af::DataType source_dtype = af::DT_UNDEFINED;
  std::vector<af::DataType> cast_dtypes;
  for (const af::AscNodePtr &node : nodes) {
    if (af::ops::IsOps<af::ascir_op::Store>(node)) {
      continue;
    }
    if (!af::ops::IsOps<af::ascir_op::Cast>(node) || node->inputs().size() != 1UL || node->outputs().empty() ||
        node->inputs()[0]->attr.mem.tensor_id != current_tensor_id) {
      return false;
    }
    if (cast_dtypes.empty()) {
      source_dtype = node->inputs()[0]->attr.dtype;
    }
    current_tensor_id = node->outputs()[0]->attr.mem.tensor_id;
    cast_dtypes.emplace_back(node->outputs()[0]->attr.dtype);
  }
  if (current_tensor_id != result_tensor_id) {
    return false;
  }
  if (cast_dtypes.empty()) {
    return true;
  }
  if (source_dtype != cast_dtypes.back() || !IsSameSimtCastCategory(source_dtype, cast_dtypes.front())) {
    return false;
  }
  const uint32_t source_size = af::GetSizeByDataType(source_dtype);
  for (const af::DataType dtype : cast_dtypes) {
    const uint32_t dtype_size = af::GetSizeByDataType(dtype);
    if (!IsSameSimtCastCategory(source_dtype, dtype) || dtype_size < source_size ||
        (dtype_size == source_size && dtype != source_dtype)) {
      return false;
    }
  }
  return true;
}

namespace {
// 后置 Reduce 的局部输出按 Inner size 对齐写回。该 size 仅当不是编译期常量（如动态 shape
// 下的 t->ksN）时才需经 Context 运行期传入；编译期常量应直接内联，避免每个输出元素做
// 一次运行期整型除/模而破坏常量折叠。
bool UseRuntimeOutputInnerSize(bool has_post_reduce,
                               const ascgen_utils::indirect_load::TemplateLogicalView &logical_view) {
  return has_post_reduce && !logical_view.output.sizes.empty() && !logical_view.output.sizes.back().IsConstExpr();
}
}  // namespace

af::Status GenerateSimtOutputsEvaluator(const std::string &input_dtype, const std::string &offset_type,
                                        ascir::TensorId value_tensor_id, const std::vector<SimtOutputChain> &chains,
                                        const std::vector<af::AscNodePtr> &nodes,
                                        const SimtLoadMetadataMap &load_metadata,
                                        const ascgen_utils::indirect_load::TemplateLogicalView &logical_view,
                                        const TPipe &tpipe, bool has_post_reduce, std::stringstream &ss) {
  GE_ASSERT_TRUE(!chains.empty(), "IndirectLoad SIMT output chains are empty.");
  std::string primary_output_dtype;
  GE_ASSERT_SUCCESS(Tensor::DtypeName(chains[0].dtype, primary_output_dtype));
  ss << "  using PrimaryOutputType = " << primary_output_dtype << ";" << std::endl;
  // 只有输出为gm，纯搬运操作，无element算子操作，才走mte快速搬运路径
  if (chains.size() == 1UL && !chains[0].local_target &&
      IsLosslessSimtCastChain(nodes, value_tensor_id, chains[0].result_tensor_id)) {
    ss << "  static constexpr bool kSimtOutputIdentity = true;" << std::endl;
  }
  ss << "  struct OutputPack {" << std::endl;
  for (size_t i = 0UL; i < chains.size(); ++i) {
    std::string dtype;
    GE_ASSERT_SUCCESS(Tensor::DtypeName(chains[i].dtype, dtype));
    ss << "    " << dtype << " output" << i << ";" << std::endl;
  }
  ss << "  };" << std::endl;
  ss << "  struct OutputTargets {" << std::endl;
  for (size_t i = 0UL; i < chains.size(); ++i) {
    std::string dtype;
    GE_ASSERT_SUCCESS(Tensor::DtypeName(chains[i].dtype, dtype));
    ss << "    " << (chains[i].local_target ? "__ubuf__ " : "__gm__ ") << dtype << " *output" << i << ";" << std::endl;
  }
  ss << "  };" << std::endl;
  std::map<ascir::TensorId, std::string> values{{value_tensor_id, "value"}};
  ss << "  __simt_callee__ __aicore__ inline static OutputPack Outputs(" << input_dtype << " value, " << offset_type
     << " output_index, " << offset_type << " index_offset, const Context &context) {" << std::endl;
  SimtLoadIndexOffsetExpressions offsets;
  GE_ASSERT_SUCCESS(BuildSimtPerLoadIndexOffsetExpressions(logical_view, nodes, load_metadata, tpipe, offset_type,
                                                           "output_index", "output_index", has_post_reduce, offsets,
                                                           ss));
  GE_ASSERT_SUCCESS(
      EmitSimtEvaluatorNodes(nodes, load_metadata, &offsets, logical_view, "output_index", "output_index", values, ss));
  // index 生产链尾部的外部 store 收为空节点输出链：链值由 Index() 求值器结果注入
  // （与 gather 值的 "value" 种子同构）。已由值侧求值产出（含输出侧 GM load 产出）
  // 的张量不注入，避免合流链双写。
  for (size_t i = 0UL; i < chains.size(); ++i) {
    const ascir::TensorId chain_result_tensor_id = chains[i].result_tensor_id;
    if (chain_result_tensor_id == value_tensor_id || values.count(chain_result_tensor_id) != 0UL) {
      continue;
    }
    const bool produced_by_output_load = std::any_of(nodes.begin(), nodes.end(), [&](const af::AscNodePtr &node) {
      return load_metadata.count(node->GetName()) != 0UL && !node->outputs().empty() &&
             node->outputs()[0]->attr.mem.tensor_id == chain_result_tensor_id;
    });
    if (produced_by_output_load) {
      continue;
    }
    values[chain_result_tensor_id] = "Index(output_index, index_offset, context)";
  }
  ss << "    OutputPack outputs;" << std::endl;
  for (size_t i = 0UL; i < chains.size(); ++i) {
    const auto found = values.find(chains[i].result_tensor_id);
    GE_ASSERT_TRUE(found != values.end(), "SIMT output chain[%zu] result tensor[%ld] has no scalar value.", i,
                   chains[i].result_tensor_id);
    ss << "    outputs.output" << i << " = " << found->second << ";" << std::endl;
  }
  ss << "    return outputs;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  __simt_callee__ __aicore__ inline static void Store(const OutputTargets &targets, " << offset_type
     << " output_index, " << offset_type << " local_index, const Context &context, const OutputPack &outputs) {"
     << std::endl;
  for (size_t i = 0UL; i < chains.size(); ++i) {
    std::string output_offset = "output_index";
    if (chains[i].local_target) {
      if (!logical_view.output.sizes.empty()) {
        // 编译期常量直接内联；仅运行期符号（如 t->ksN）经 Context 传入，避免在静态 Store 中引用 t->。
        const std::string inner_size = UseRuntimeOutputInnerSize(has_post_reduce, logical_view)
                                           ? "static_cast<" + offset_type + ">(context.output_inner_size)"
                                           : tpipe.tiler.Size(logical_view.output.sizes.back());
        std::string dtype;
        GE_ASSERT_SUCCESS(Tensor::DtypeName(chains[i].dtype, dtype));
        const auto row_stride = "KernelUtils::SizeAlign(" + inner_size + ", 32/sizeof(" + dtype + "))";
        output_offset =
            "(local_index / (" + inner_size + ")) * (" + row_stride + ") + local_index % (" + inner_size + ")";
      } else {
        output_offset = "local_index";
      }
    }
    ss << "    targets.output" << i << "[" << output_offset << "] = outputs.output" << i << ";" << std::endl;
  }
  ss << "  }" << std::endl;
  return af::SUCCESS;
}

std::string GetSimdLogicalOutputSize(const TPipe &tpipe, const Tensor &tensor) {
  std::stringstream ss;
  ss << "1";
  for (size_t i = 0; i < tensor.vectorized_axis.size(); ++i) {
    const uint32_t axis_pos = tensor.vectorized_axis_pos[i];
    if (axis_pos >= tensor.axis_size.size()) {
      continue;
    }
    const auto &axis = tpipe.tiler.GetAxis(tensor.vectorized_axis[i]);
    const auto &axis_size = tensor.axis_size[axis_pos];
    const bool use_actual = axis.type == Axis::Type::kAxisTypeTileInner ||
                            af::SymbolicUtils::StaticCheckEq(axis_size, axis.size_expr) == af::TriBool::kTrue;
    ss << " * " << (use_actual ? "(" + axis.actual_size.Str() + ")" : "(" + tpipe.tiler.Size(axis_size) + ")");
  }
  return ss.str();
}

af::Status GenSimtIndexEvaluator(const std::string &index_dtype, const std::string &offset_type,
                                 ascir::TensorId result_tensor_id, const std::vector<af::AscNodePtr> &nodes,
                                 const SimtLoadMetadataMap &load_metadata,
                                 const ascgen_utils::indirect_load::TemplateLogicalView &logical_view,
                                 const TPipe &tpipe, bool has_post_reduce, std::stringstream &ss) {
  std::map<ascir::TensorId, std::string> values;
  ss << "  __simt_callee__ __aicore__ inline static " << index_dtype << " Index(" << offset_type << " output_index, "
     << offset_type << " index_offset, const Context &context) {" << std::endl;
  SimtLoadIndexOffsetExpressions offsets;
  GE_ASSERT_SUCCESS(BuildSimtPerLoadIndexOffsetExpressions(logical_view, nodes, load_metadata, tpipe, offset_type,
                                                           "output_index", "index_offset", has_post_reduce, offsets,
                                                           ss));
  return GenerateSimtEvaluatorBody(nodes, values, result_tensor_id, load_metadata, ss, logical_view, "index_offset",
                                   "output_index", &offsets);
}

af::Status GenAicoreIndexEvaluator(const std::string &index_dtype, const std::string &offset_type,
                                   ascir::TensorId result_tensor_id, const std::vector<af::AscNodePtr> &nodes,
                                   const SimtLoadMetadataMap &load_metadata,
                                   const ascgen_utils::indirect_load::TemplateLogicalView &logical_view,
                                   const TPipe &tpipe, bool has_post_reduce, std::stringstream &ss) {
  std::map<ascir::TensorId, std::string> values;
  ss << "  static constexpr bool kSupportsAicoreIndex = true;" << std::endl;
  ss << "  __aicore__ inline static " << index_dtype << " AicoreIndex(" << offset_type
     << " output_index, const Context &context) {" << std::endl;
  SimtLoadIndexOffsetExpressions offsets;
  GE_ASSERT_SUCCESS(BuildSimtPerLoadIndexOffsetExpressions(logical_view, nodes, load_metadata, tpipe, offset_type,
                                                           "output_index", "output_index", has_post_reduce, offsets,
                                                           ss));
  return GenerateAicoreIndexEvaluatorBody(nodes, values, result_tensor_id, load_metadata, offsets, ss);
}

af::Status CalcVectorizedElementCount(const Tensor &tensor, af::Expression &element_count) {
  element_count = af::ops::One;
  for (uint32_t axis_pos : tensor.vectorized_axis_pos) {
    GE_ASSERT_TRUE(axis_pos < tensor.axis_size.size(), "IndirectLoad SIMT output axis is invalid.");
    element_count = af::sym::Mul(element_count, tensor.axis_size[axis_pos]);
  }
  return af::SUCCESS;
}

std::vector<ascir::SizeExpr> GetSimdOutputStrides(const Tensor &tensor,
                                                  const ascgen_utils::indirect_load::LogicalTensorView &layout) {
  std::vector<ascir::SizeExpr> strides(layout.axis_ids.size(), af::sym::kSymbolOne);
  for (size_t dim = 0; dim < layout.axis_ids.size(); ++dim) {
    const auto it = std::find(tensor.vectorized_axis.begin(), tensor.vectorized_axis.end(), layout.axis_ids[dim]);
    if (it != tensor.vectorized_axis.end()) {
      strides[dim] = tensor.vectorized_strides[static_cast<size_t>(std::distance(tensor.vectorized_axis.begin(), it))];
      continue;
    }
    const auto axis_it = std::find(tensor.axis.begin(), tensor.axis.end(), layout.axis_ids[dim]);
    if (axis_it != tensor.axis.end()) {
      strides[dim] = tensor.axis_strides[static_cast<size_t>(std::distance(tensor.axis.begin(), axis_it))];
    }
  }
  return strides;
}

const char *GetSimdApiName(ascgen_utils::indirect_load::SimdFallback fallback) {
  switch (fallback) {
    case ascgen_utils::indirect_load::SimdFallback::kStrided:
      return "IndirectLoadSimdStrided";
    case ascgen_utils::indirect_load::SimdFallback::kGatherApi:
      return "IndirectLoadSimdGatherApi";
    case ascgen_utils::indirect_load::SimdFallback::kRegisterGather:
      return "IndirectLoadSimd";
  }
  return "IndirectLoadSimd";
}

void EmitSimdParams(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis, const Tensor &input,
                    const Tensor &output, const LogicalTensorInfo &input_info, const LogicalTensorInfo &index_info,
                    const ascgen_utils::indirect_load::LogicalTensorView &output_layout,
                    const ascgen_utils::indirect_load::SimdLoweringMetadata &metadata, size_t axis_pos,
                    std::stringstream &ss) {
  const size_t rank = input_info.sizes.size();
  ss << "  using IndirectLoadSimdCase = AscendC::IndirectLoadSimdCaseTag<"
     << "static_cast<AscendC::IndirectLoadSimdFallback>(" << static_cast<int64_t>(metadata.fallback) << "), "
     << (metadata.try_embedding ? "true" : "false") << ">;" << std::endl;
  ss << "  AscendC::IndirectLoadSimdParams<IndirectLoadSimdCase, " << rank << "> indirect_load_simd_params{";
  const std::string output_offset = tpipe.tiler.Offset(current_axis, output.axis, output.axis_strides);
  if (metadata.try_embedding) {
    const std::string logical_output_size = GetSimdLogicalOutputSize(tpipe, output);
    const auto output_strides = GetSimdOutputStrides(output, output_layout);
    ss << "static_cast<uint32_t>(" << logical_output_size << "), static_cast<uint32_t>(" << output.actual_size
       << "), static_cast<uint32_t>(" << input.actual_size << "), " << output_offset << ", "
       << tpipe.tiler.Size(input_info.sizes[axis_pos]) << ", {" << JoinSizeExprs(index_info.sizes, tpipe) << "}, {"
       << JoinSizeExprs(input_info.strides, tpipe) << "}, {" << JoinSizeExprs(index_info.strides, tpipe) << "}, {"
       << JoinSizeExprs(output_strides, tpipe) << "}";
  } else if (metadata.fallback == ascgen_utils::indirect_load::SimdFallback::kStrided) {
    const std::string logical_output_size = GetSimdLogicalOutputSize(tpipe, output);
    const auto output_strides = GetSimdOutputStrides(output, output_layout);
    ss << "static_cast<uint32_t>(" << logical_output_size << "), static_cast<uint32_t>(" << output.actual_size << "), "
       << output_offset << ", {" << JoinSizeExprs(index_info.sizes, tpipe) << "}, {"
       << JoinSizeExprs(input_info.strides, tpipe) << "}, {" << JoinSizeExprs(index_info.strides, tpipe) << "}, {"
       << JoinSizeExprs(output_strides, tpipe) << "}";
  } else {
    ss << "static_cast<uint32_t>(" << output.actual_size << "), static_cast<uint32_t>(" << input.actual_size << "), "
       << output_offset << ", " << tpipe.tiler.Size(input_info.sizes[axis_pos]) << ", {"
       << JoinSizeExprs(index_info.sizes, tpipe) << "}, {" << JoinSizeExprs(input_info.strides, tpipe) << "}";
  }
  ss << "};" << std::endl;
}

void EmitSimdInvocation(const Tensor &input, const Tensor &index, const Tensor &output,
                        const LogicalTensorInfo &input_info, int64_t axis,
                        ascgen_utils::indirect_load::SimdFallback fallback, const std::string &input_dtype,
                        const std::string &index_dtype, const std::string &tmp_name, std::stringstream &ss) {
  ss << "  AscendC::" << GetSimdApiName(fallback) << "<" << input_dtype << ", " << index_dtype << ", "
     << input_info.sizes.size() << ", " << axis << ">(\n";
  ss << "      " << input << ", " << index << ", " << output << ", ";
  if (fallback == ascgen_utils::indirect_load::SimdFallback::kStrided) {
    ss << tmp_name << ", ";
  }
  ss << "IndirectLoadSimdCase{}, indirect_load_simd_params);" << std::endl;
}

void EmitSimtPolicyParams(const TPipe &tpipe, const ascgen_utils::indirect_load::SimtPolicyMetadata &policy,
                          size_t rank, int64_t axis, std::stringstream &ss) {
  ss << "  using IndirectLoadSimtCase = " << GetSimtCaseTag(policy, rank, axis) << ";" << std::endl;
  ss << "  AscendC::IndirectLoadSimtParams<IndirectLoadSimtCase> indirect_load_simt_params{"
     << GetSimtPolicyParams(policy, tpipe) << "};" << std::endl;
}

af::Status GenerateSimtContextInitializer(
    const std::string &context_name, const std::vector<ascgen_utils::indirect_load::SimtGmTensorMetadata> &gm_tensors,
    const TPipe &tpipe, const std::string &output_inner_size_expr, std::stringstream &ss) {
  ss << "  " << context_name << " context{";
  for (size_t i = 0UL; i < gm_tensors.size(); ++i) {
    const auto &gm_tensor = gm_tensors[i];
    std::string dtype;
    GE_ASSERT_SUCCESS(Tensor::DtypeName(gm_tensor.dtype, dtype));
    if (gm_tensor.is_scalar) {
      const Tensor *scalar = tpipe.GetTensor(gm_tensor.value_tensor_id);
      GE_ASSERT_NOTNULL(scalar, "IndirectLoad SIMT ScalarData tensor is missing.");
      ss << (i == 0UL ? "" : ", ") << scalar->name;
    } else {
      ss << (i == 0UL ? "" : ", ") << "(__gm__ " << dtype << " *)" << kGlobalTensorNamePrefix << gm_tensor.gm_tensor_id
         << ".GetPhyAddr()";
    }
  }
  if (!output_inner_size_expr.empty()) {
    ss << (gm_tensors.empty() ? "" : ", ") << output_inner_size_expr;
  }
  ss << "};" << std::endl;
  return af::SUCCESS;
}

af::Status BuildSimtGraphNodeMap(const ascir::NodeView &node, SimtGraphNodeMap &node_map) {
  const auto owner_graph = node->GetOwnerComputeGraph();
  GE_ASSERT_NOTNULL(owner_graph, "IndirectLoad SIMT node has no owner graph.");
  for (const auto &graph_node : owner_graph->GetDirectNode()) {
    const auto asc_node = std::dynamic_pointer_cast<af::AscNode>(graph_node);
    GE_ASSERT_NOTNULL(asc_node, "IndirectLoad SIMT graph contains invalid node.");
    GE_ASSERT_TRUE(node_map.emplace(asc_node->GetName(), asc_node).second,
                   "IndirectLoad SIMT graph contains duplicate node name[%s].", asc_node->GetNamePtr());
  }
  return af::SUCCESS;
}

bool ContainsAllNodeNames(const std::vector<std::string> &node_names, const SimtGraphNodeMap &node_map) {
  return std::all_of(node_names.begin(), node_names.end(),
                     [&node_map](const std::string &node_name) { return node_map.count(node_name) != 0UL; });
}

bool HasCompleteSimtTensorIds(const ascgen_utils::indirect_load::SimtLoweringMetadata &simt) {
  const auto is_valid_tensor_id = [](ascir::TensorId tensor_id) { return tensor_id != af::kIdNone; };
  if (!is_valid_tensor_id(simt.index_result_tensor_id) || !is_valid_tensor_id(simt.value_tensor_id)) {
    return false;
  }
  const bool output_result_required = simt.has_post_reduce || simt.output_chains.size() == 1UL;
  if (output_result_required && !is_valid_tensor_id(simt.output_result_tensor_id)) {
    return false;
  }
  const bool gm_tensor_ids_valid =
      std::all_of(simt.gm_tensors.begin(), simt.gm_tensors.end(), [&is_valid_tensor_id](const auto &gm_tensor) {
        return is_valid_tensor_id(gm_tensor.value_tensor_id) && is_valid_tensor_id(gm_tensor.gm_tensor_id);
      });
  if (!gm_tensor_ids_valid) {
    return false;
  }
  return std::all_of(simt.output_chains.begin(), simt.output_chains.end(), [&is_valid_tensor_id](const auto &chain) {
    return is_valid_tensor_id(chain.result_tensor_id) && is_valid_tensor_id(chain.target_tensor_id);
  });
}

bool HasCurrentMetadataReferences(ascir::TemplateId template_id,
                                  const ascgen_utils::indirect_load::IndirectLoadLoweringMetadata &metadata,
                                  const SimtGraphNodeMap &node_map) {
  if (template_id == ascir::TemplateId::kIndirectLoadSimd) {
    return true;
  }
  const auto &simt = metadata.simt;
  if (!ContainsAllNodeNames(simt.index_node_names, node_map) ||
      !ContainsAllNodeNames(simt.output_node_names, node_map)) {
    return false;
  }
  const auto load_exists = [&node_map](const ascgen_utils::indirect_load::SimtLoadMetadata &load) {
    return node_map.count(load.node_name) != 0UL;
  };
  if (!std::all_of(simt.index_loads.begin(), simt.index_loads.end(), load_exists) ||
      !std::all_of(simt.output_loads.begin(), simt.output_loads.end(), load_exists)) {
    return false;
  }
  return std::all_of(simt.output_chains.begin(), simt.output_chains.end(),
                     [&node_map](const ascgen_utils::indirect_load::SimtOutputChainMetadata &chain) {
                       return ContainsAllNodeNames(chain.node_names, node_map);
                     });
}

af::Status LoadCurrentLoweringMetadata(const ascir::NodeView &node, ascir::TemplateId template_id,
                                       ascgen_utils::indirect_load::IndirectLoadLoweringMetadata &metadata,
                                       SimtGraphNodeMap &node_map) {
  if (template_id == ascir::TemplateId::kIndirectLoadSimt) {
    GE_ASSERT_SUCCESS(BuildSimtGraphNodeMap(node, node_map));
  }
  bool needs_refresh = !ascgen_utils::indirect_load::HasLoweringMetadata(node);
  if (!needs_refresh) {
    GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::GetLoweringMetadata(node, metadata));
    needs_refresh = !HasCurrentMetadataReferences(template_id, metadata, node_map);
    if (template_id == ascir::TemplateId::kIndirectLoadSimt && !HasCompleteSimtTensorIds(metadata.simt)) {
      needs_refresh = true;
    }
  }
  if (needs_refresh) {
    bool metadata_supported = false;
    GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::FinalizeLoweringMetadata(node, metadata_supported));
    GE_ASSERT_TRUE(metadata_supported, "IndirectLoad lowering metadata is unsupported, node[%s].", node->GetNamePtr());
    GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::GetLoweringMetadata(node, metadata));
  }
  GE_ASSERT_TRUE(HasCurrentMetadataReferences(template_id, metadata, node_map),
                 "IndirectLoad lowering metadata contains stale node references after refresh, node[%s].",
                 node->GetNamePtr());
  if (template_id == ascir::TemplateId::kIndirectLoadSimt) {
    GE_ASSERT_TRUE(HasCompleteSimtTensorIds(metadata.simt),
                   "IndirectLoad SIMT lowering metadata tensor IDs remain unassigned after refresh, node[%s].",
                   node->GetNamePtr());
  }
  return af::SUCCESS;
}

af::Status ResolveSimtNodes(const std::vector<std::string> &node_names, const SimtGraphNodeMap &node_map,
                            std::vector<af::AscNodePtr> &nodes) {
  nodes.clear();
  nodes.reserve(node_names.size());
  for (const auto &node_name : node_names) {
    const auto node = node_map.find(node_name);
    GE_ASSERT_TRUE(node != node_map.end(), "IndirectLoad SIMT metadata node[%s] is missing.", node_name.c_str());
    nodes.emplace_back(node->second);
  }
  return af::SUCCESS;
}

}  // namespace

Status IndirectLoadRegApiCall::ParseAttr(const ascir::NodeView &node) {
  template_id_ = ::ascir::GetTemplateIdOrDefault(*node);
  GE_ASSERT_TRUE(
      template_id_ == ascir::TemplateId::kIndirectLoadSK || template_id_ == ascir::TemplateId::kIndirectLoadSimd ||
          template_id_ == ascir::TemplateId::kIndirectLoadSimt,
      "IndirectLoad node[%s] has invalid template id[%d].", node->GetNamePtr(), static_cast<int32_t>(template_id_));
  if (template_id_ == ascir::TemplateId::kIndirectLoadSK) {
    int64_t axis = 0L;
    GE_CHK_GRAPH_STATUS_RET(node->attr.ir_attr->GetAttrValue("axis", axis),
                            "Failed to get IndirectLoad axis attr, node = %s", node->GetNamePtr());
    ascgen_utils::indirect_load::TemplateAxes template_axes;
    GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::GetTemplateAxes(node, template_axes));
    outer_axis_ = template_axes.outer_axis;
    has_post_reduce_ =
        ascgen_utils::indirect_load::GetPostReduceConsumer(std::dynamic_pointer_cast<af::AscNode>(node)) != nullptr;
    GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::GetTemplateLogicalView(node, logical_view_));
    GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::AnalyzeIndirectLoadAccess(node, logical_view_, access_info_));
    GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::GetImplementation(node, implementation_));
    const int64_t rank = static_cast<int64_t>(logical_view_.input.sizes.size());
    GE_ASSERT_TRUE(axis >= -rank && axis < rank, "IndirectLoad axis %ld is out of range [%ld, %ld).", axis, -rank,
                   rank);
    axis_ = axis < 0L ? axis + rank : axis;
  } else {
    ascgen_utils::indirect_load::IndirectLoadLoweringMetadata metadata;
    SimtGraphNodeMap node_map;
    GE_ASSERT_SUCCESS(LoadCurrentLoweringMetadata(node, template_id_, metadata, node_map));
    axis_ = metadata.axis;
    outer_axis_ = metadata.outer_axis;
    logical_view_ = metadata.logical_view;
    access_info_ = metadata.access_info;
    simd_metadata_ = metadata.simd;
    has_post_reduce_ = metadata.simt.has_post_reduce;
    if (template_id_ == ascir::TemplateId::kIndirectLoadSimt) {
      GE_ASSERT_SUCCESS(ResolveSimtMetadata(node, metadata.simt, node_map));
    }
  }
  GELOGI("[IndirectLoad] Parse codegen attrs for node[%s], axis[%ld], template_id[%d].", node->GetNamePtr(), axis_,
         static_cast<int32_t>(template_id_));
  return af::SUCCESS;
}

Status IndirectLoadRegApiCall::ResolveSimtMetadata(const ascir::NodeView &node,
                                                   const ascgen_utils::indirect_load::SimtLoweringMetadata &metadata,
                                                   const SimtGraphNodeMap &node_map) {
  simt_gm_tensors_.clear();
  index_nodes_.clear();
  output_nodes_.clear();
  output_chains_.clear();
  simt_index_loads_.clear();
  simt_output_loads_.clear();
  GE_ASSERT_TRUE(node->inputs.Size() == kIndirectLoadInputCount, "Invalid IndirectLoad SIMT input number:%zu.",
                 node->inputs.Size());
  GE_ASSERT_TRUE(node->outputs().size() == kIndirectLoadOutputCount, "Invalid IndirectLoad SIMT output number:%zu.",
                 node->outputs().size());
  GE_ASSERT_TRUE(logical_view_.input.sizes.size() < 64UL, "IndirectLoad SIMT rank must be smaller than 64.");
  has_post_reduce_ = metadata.has_post_reduce;
  simt_policy_ = metadata.policy;
  simt_gm_tensors_ = metadata.gm_tensors;
  index_result_tensor_id_ = metadata.index_result_tensor_id;
  simt_value_tensor_id_ = metadata.value_tensor_id;
  output_result_tensor_id_ = metadata.output_result_tensor_id;
  index_dtype_ = metadata.index_dtype;

  GE_ASSERT_SUCCESS(ResolveSimtNodes(metadata.index_node_names, node_map, index_nodes_));
  GE_ASSERT_SUCCESS(ResolveSimtNodes(metadata.output_node_names, node_map, output_nodes_));
  for (const auto &load : metadata.index_loads) {
    GE_ASSERT_TRUE(node_map.count(load.node_name) != 0UL, "IndirectLoad SIMT index Load[%s] is missing.",
                   load.node_name.c_str());
    GE_ASSERT_TRUE(simt_index_loads_.emplace(load.node_name, load).second,
                   "IndirectLoad SIMT index Load[%s] metadata is duplicated.", load.node_name.c_str());
  }
  for (const auto &load : metadata.output_loads) {
    GE_ASSERT_TRUE(node_map.count(load.node_name) != 0UL, "IndirectLoad SIMT output Load[%s] is missing.",
                   load.node_name.c_str());
    GE_ASSERT_TRUE(simt_output_loads_.emplace(load.node_name, load).second,
                   "IndirectLoad SIMT output Load[%s] metadata is duplicated.", load.node_name.c_str());
  }
  for (const auto &chain_metadata : metadata.output_chains) {
    SimtOutputChain chain;
    GE_ASSERT_SUCCESS(ResolveSimtNodes(chain_metadata.node_names, node_map, chain.nodes));
    chain.result_tensor_id = chain_metadata.result_tensor_id;
    chain.target_tensor_id = chain_metadata.target_tensor_id;
    chain.dtype = chain_metadata.dtype;
    chain.local_target = chain_metadata.local_target;
    output_chains_.emplace_back(std::move(chain));
  }
  GE_ASSERT_TRUE(!output_chains_.empty(), "IndirectLoad SIMT output chains are empty.");
  const size_t local_target_count = static_cast<size_t>(std::count_if(
      output_chains_.begin(), output_chains_.end(), [](const SimtOutputChain &chain) { return chain.local_target; }));
  GE_ASSERT_TRUE(local_target_count <= 1UL, "IndirectLoad SIMT supports at most one local output target.");
  GE_ASSERT_TRUE(has_post_reduce_ == (local_target_count == 1UL),
                 "IndirectLoad SIMT post Reduce and local output target must appear together.");
  if (has_post_reduce_) {
    GE_ASSERT_TRUE(outputs.size() == kIndirectLoadOutputCount, "Invalid IndirectLoad SIMT output number:%zu.",
                   outputs.size());
    outputs[0].id = output_result_tensor_id_;
  }
  return af::SUCCESS;
}

Status IndirectLoadRegApiCall::GenerateFuncDefinition(const TPipe &tpipe, const Tiler &tiler,
                                                      std::stringstream &ss) const {
  (void)tiler;
  if (template_id_ != ascir::TemplateId::kIndirectLoadSimt) {
    return af::SUCCESS;
  }

  GE_ASSERT_TRUE(inputs.size() == kIndirectLoadInputCount, "Invalid IndirectLoad SIMT input number:%zu.",
                 inputs.size());
  const Tensor *input_tensor = tpipe.GetTensor(inputs[ascgen_utils::indirect_load::kInputTensorIndex]->id);
  GE_ASSERT_NOTNULL(input_tensor, "IndirectLoad SIMT input tensor is missing.");
  const size_t rank = logical_view_.input.sizes.size();
  const std::string offset_type = GetSimtOffsetType(simt_policy_);
  std::string input_dtype;
  GE_ASSERT_SUCCESS(Tensor::DtypeName(input_tensor->dtype, input_dtype));
  const std::string valid_node_name = ascgen_utils::GenValidName(node_name);
  const std::string context_name = kSimtContextNamePrefix + valid_node_name;
  const std::string body_name = kSimtBodyNamePrefix + valid_node_name;
  const std::string body_guard = kSimtBodyGuardPrefix + valid_node_name;
  std::string index_dtype;
  GE_ASSERT_SUCCESS(Tensor::DtypeName(index_dtype_, index_dtype));
  GELOGI(
      "[IndirectLoad] Generate SIMT body for node[%s], rank[%zu], axis[%ld], index_nodes[%zu], output_nodes[%zu], "
      "gm_inputs[%zu].",
      node_name.c_str(), rank, axis_, index_nodes_.size(),
      has_post_reduce_ ? output_nodes_.size() : output_chains_.size(), simt_gm_tensors_.size());

  // A generated source file can contain several kernel variants for the same graph node
  // (for example, a SIMT variant and an NDDMA variant).  They share the evaluator type
  // name, so guard the definition while keeping every variant's invocation available.
  ss << "#ifndef " << body_guard << std::endl;
  ss << "#define " << body_guard << std::endl;
  GE_ASSERT_SUCCESS(GenerateSimtContextDefinition(context_name, simt_gm_tensors_,
                                                  UseRuntimeOutputInnerSize(has_post_reduce_, logical_view_), ss));
  ss << "struct " << body_name << " {" << std::endl;
  ss << "  using Context = " << context_name << ";" << std::endl;
  const size_t ub_output_count = static_cast<size_t>(std::count_if(
      output_chains_.begin(), output_chains_.end(), [](const SimtOutputChain &chain) { return chain.local_target; }));
  ss << "  static constexpr uint32_t kGmOutputCount = " << output_chains_.size() - ub_output_count << "U;" << std::endl;
  ss << "  static constexpr uint32_t kUbOutputCount = " << ub_output_count << "U;" << std::endl;

  GE_ASSERT_SUCCESS(GenSimtIndexEvaluator(index_dtype, offset_type, index_result_tensor_id_, index_nodes_,
                                          simt_index_loads_, logical_view_, tpipe, has_post_reduce_, ss));
  if (simt_policy_.policy == ascgen_utils::indirect_load::SimtAddressPolicy::kEmbedding &&
      CanGenerateAicoreIndexEvaluator(index_nodes_, simt_index_loads_)) {
    GE_ASSERT_SUCCESS(GenAicoreIndexEvaluator(index_dtype, offset_type, index_result_tensor_id_, index_nodes_,
                                              simt_index_loads_, logical_view_, tpipe, has_post_reduce_, ss));
  }
  GE_ASSERT_SUCCESS(GenerateSimtOutputsEvaluator(input_dtype, offset_type, simt_value_tensor_id_, output_chains_,
                                                 output_nodes_, simt_output_loads_, logical_view_, tpipe,
                                                 has_post_reduce_, ss));
  ss << "};" << std::endl;
  ss << "#endif" << std::endl;
  return af::SUCCESS;
}

Status IndirectLoadRegApiCall::Generate(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                                        const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                                        const std::vector<std::reference_wrapper<const Tensor>> &outputs,
                                        std::string &result) const {
  GE_ASSERT_TRUE(inputs.size() == kIndirectLoadInputCount, "Invalid IndirectLoad input number:%zu.", inputs.size());
  GE_ASSERT_TRUE(outputs.size() == kIndirectLoadOutputCount, "Invalid IndirectLoad output number:%zu.", outputs.size());
  GE_ASSERT_TRUE(
      template_id_ == ascir::TemplateId::kIndirectLoadSK || template_id_ == ascir::TemplateId::kIndirectLoadSimd,
      "IndirectLoad tensor-based Generate only supports SK and SIMD.");
  GE_ASSERT_SUCCESS(CheckIndirectLoadShape(logical_view_, outputs[0].get()));
  (void)RegisterBasicDumpParam(this->api_name_, inputs, outputs);
  if (template_id_ == ascir::TemplateId::kIndirectLoadSK) {
    GELOGI("[IndirectLoad] Generate SK API body for node[%s].", node_name.c_str());
    return GenerateSk(tpipe, current_axis, inputs, outputs, result);
  }
  GELOGI("[IndirectLoad] Generate SIMD API body for node[%s].", node_name.c_str());
  return GenerateSimd(tpipe, current_axis, inputs, outputs, result);
}

Status IndirectLoadRegApiCall::Generate(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                                        std::string &result) const {
  if (template_id_ != ascir::TemplateId::kIndirectLoadSimt) {
    return ApiCall::Generate(tpipe, current_axis, result);
  }
  GE_ASSERT_TRUE(inputs.size() == kIndirectLoadInputCount, "Invalid IndirectLoad SIMT input number:%zu.",
                 inputs.size());
  const Tensor *input_tensor = tpipe.GetTensor(inputs[ascgen_utils::indirect_load::kInputTensorIndex]->id);
  GE_ASSERT_NOTNULL(input_tensor, "IndirectLoad SIMT input tensor is missing.");
  return GenerateSimt(tpipe, current_axis, *input_tensor, result);
}

Status IndirectLoadRegApiCall::GenerateSk(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                                          const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                                          const std::vector<std::reference_wrapper<const Tensor>> &outputs,
                                          std::string &result) const {
  const Tensor &x = inputs[0].get();
  const Tensor &index = inputs[1].get();
  const Tensor &y = outputs[0].get();
  const auto tmp_iter = tmp_buf_id.find(-1L);
  GE_ASSERT_TRUE(tmp_iter != tmp_buf_id.end(), "IndirectLoad SK requires an API-level tmp buffer.");

  const size_t axis_pos = static_cast<size_t>(axis_);
  const LogicalTensorInfo x_info(logical_view_.input);
  LogicalTensorInfo index_info;
  GE_ASSERT_SUCCESS(BuildTensorWindowInfo(logical_view_.index, index, axis_pos, index_info));
  std::string x_dtype_name;
  std::string index_dtype_name;
  GE_ASSERT_SUCCESS(Tensor::DtypeName(x.dtype, x_dtype_name));
  GE_ASSERT_SUCCESS(Tensor::DtypeName(index.dtype, index_dtype_name));
  GE_ASSERT_TRUE(x.dtype == y.dtype, "IndirectLoad SK input/output dtype must match.");
  GELOGD("[IndirectLoad] Generate SK body for node[%s], rank[%zu], axis[%zu].", node_name.c_str(), x_info.sizes.size(),
         axis_pos);

  std::stringstream ss;
  ss << "// IndirectLoad SK" << std::endl;
  ss << "{" << std::endl;
  ss << "  AscendC::IndirectLoadSk<" << x_dtype_name << ", " << index_dtype_name << ", " << x_info.sizes.size() << ", "
     << axis_ << ">(" << std::endl;
  ss << "      " << x << ", " << index << ", " << y << ", " << tpipe.tmp_buf.name << "_" << tmp_iter->second << ", "
     << y.actual_size << ", " << tpipe.tiler.Offset(current_axis, y.axis, y.axis_strides) << ", "
     << tpipe.tiler.Size(x_info.sizes[axis_pos]) << ", " << JoinSizeExprs(index_info.sizes, tpipe) << ", "
     << JoinSizeExprs(x_info.strides, tpipe) << ", " << JoinSizeExprs(index_info.strides, tpipe) << ");" << std::endl;
  ss << "}" << std::endl;
  result = ss.str();
  return af::SUCCESS;
}

Status IndirectLoadRegApiCall::GenerateSimd(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                                            const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                                            const std::vector<std::reference_wrapper<const Tensor>> &outputs,
                                            std::string &result) const {
  const Tensor &input = inputs[ascgen_utils::indirect_load::kInputTensorIndex].get();
  const Tensor &index = inputs[ascgen_utils::indirect_load::kIndexTensorIndex].get();
  const Tensor &output = outputs[0].get();
  const size_t axis_pos = static_cast<size_t>(axis_);
  LogicalTensorInfo input_info;
  LogicalTensorInfo index_info;
  GE_ASSERT_SUCCESS(BuildTensorWindowInfo(logical_view_.input, input, axis_pos, input_info));
  GE_ASSERT_SUCCESS(BuildTensorWindowInfo(logical_view_.index, index, axis_pos, index_info));
  const auto tmp_iter = tmp_buf_id.find(-1L);
  if (simd_metadata_.fallback == ascgen_utils::indirect_load::SimdFallback::kStrided) {
    GE_ASSERT_TRUE(tmp_iter != tmp_buf_id.end(), "IndirectLoad SIMD requires an API-level tmp buffer.");
  }
  // [padded 视图兜底降级] codegen 期为最终视图（通用对齐已完成），不依赖 lowering
  // metadata 序列化时的视图时机：输出 tensor 的向量化跨度（与 Tiler::TensorActualSize
  // 同源）大于逻辑元素积（Π尾轴前有效宽）即视图被 pad——dense facade（RegGather
  // RunReuse）线性稠密写出与 padded 视图布局冲突（index 按窗口线性读越过真实数据
  // → 越界读表 AIC 341；strided 解释稠密数据 → 数值错乱），强制降级 strided facade
  // （按视图 strides 逐行写，行内 pad 由 ReduceInit OptImpl 清中性值）。SIMD 模板
  // 的 tmp 恒分配（CalcTmpBufSize 只判模板），降级无 tmp 缺口。
  {
    // codegen::Tensor 直接持有 axis/axis_size/axis_strides 与向量化视图字段，
    // axis_size 即视图轴宽（Tiler::TensorActualSize 同源）。
    af::Expression y_span = af::sym::kSymbolOne;
    af::Expression y_product = af::sym::kSymbolOne;
    bool y_view_ok = output.vectorized_axis.size() == output.vectorized_strides.size() &&
                     !output.vectorized_axis.empty() && output.axis.size() == output.axis_size.size();
    if (y_view_ok) {
      for (size_t dim = 0UL; dim < output.vectorized_axis.size(); ++dim) {
        const auto axis_it = std::find(output.axis.begin(), output.axis.end(), output.vectorized_axis[dim]);
        if (axis_it == output.axis.end()) {
          y_view_ok = false;
          break;
        }
        const size_t axis_pos_y = static_cast<size_t>(std::distance(output.axis.begin(), axis_it));
        const auto &y_stride = output.vectorized_strides[dim];
        if (af::SymbolicUtils::StaticCheckEq(y_stride, af::sym::kSymbolZero) == af::TriBool::kTrue) {
          continue;
        }
        y_span = y_span + (output.axis_size[axis_pos_y] - af::sym::kSymbolOne) * y_stride;
        y_product = y_product * output.axis_size[axis_pos_y];
      }
    }
    const bool y_padded = y_view_ok && output.vectorized_axis.size() > 1UL &&
                          af::SymbolicUtils::StaticCheckGt(y_span, y_product) == af::TriBool::kTrue;
    GELOGI("[IndirectLoad] SIMD fallback check: node[%s] span[%s] product[%s] padded[%d] fallback[%d].",
           node_name.c_str(), y_span.Str().get(), y_product.Str().get(), static_cast<int>(y_padded),
           static_cast<int>(simd_metadata_.fallback));
    if (y_padded && simd_metadata_.fallback != ascgen_utils::indirect_load::SimdFallback::kStrided) {
      GELOGW("[IndirectLoad] SIMD dense fallback downgraded to strided: node[%s] padded view span[%s] > product[%s].",
             node_name.c_str(), y_span.Str().get(), y_product.Str().get());
      simd_metadata_.fallback = ascgen_utils::indirect_load::SimdFallback::kStrided;
      if (tmp_iter == tmp_buf_id.end()) {
        GE_ASSERT_TRUE(false, "IndirectLoad SIMD strided downgrade requires an API-level tmp buffer, node[%s].",
                       node_name.c_str());
      }
    }
  }

  std::string input_dtype;
  std::string index_dtype;
  GE_ASSERT_SUCCESS(Tensor::DtypeName(input.dtype, input_dtype));
  GE_ASSERT_SUCCESS(Tensor::DtypeName(index.dtype, index_dtype));
  GE_ASSERT_TRUE(input.dtype == output.dtype,
                 "IndirectLoad SIMD input/output dtype must match after input preprocess.");
  GELOGD("[IndirectLoad] Generate SIMD body for node[%s], rank[%zu], axis[%zu].", node_name.c_str(),
         input_info.sizes.size(), axis_pos);

  std::stringstream ss;
  ss << "// IndirectLoad SIMD" << std::endl;
  ss << "{" << std::endl;
  EmitSimdParams(tpipe, current_axis, input, output, input_info, index_info, logical_view_.output, simd_metadata_,
                 axis_pos, ss);
  const std::string tmp_name = simd_metadata_.fallback == ascgen_utils::indirect_load::SimdFallback::kStrided
                                   ? tpipe.tmp_buf.name + "_" + std::to_string(tmp_iter->second)
                                   : "";
  EmitSimdInvocation(input, index, output, input_info, axis_, simd_metadata_.fallback, input_dtype, index_dtype,
                     tmp_name, ss);
  ss << "}" << std::endl;
  result = ss.str();
  return af::SUCCESS;
}

Status IndirectLoadRegApiCall::GenerateSimtInvocation(const TPipe &tpipe, const std::string &input_dtype,
                                                      const std::string &outer_tb_var, std::stringstream &ss) const {
  const size_t rank = logical_view_.input.sizes.size();
  const std::string offset_type = GetSimtOffsetType(simt_policy_);
  const std::string body_name = kSimtBodyNamePrefix + ascgen_utils::GenValidName(node_name);
  std::string actual_size_expr;
  std::string output_offset_expr;
  if (has_post_reduce_) {
    const auto local_chain = std::find_if(output_chains_.begin(), output_chains_.end(),
                                          [](const SimtOutputChain &chain) { return chain.local_target; });
    GE_ASSERT_TRUE(local_chain != output_chains_.end(), "IndirectLoad SIMT post Reduce local output is missing.");
    const Tensor *local_target = tpipe.GetTensor(local_chain->target_tensor_id);
    GE_ASSERT_NOTNULL(local_target, "IndirectLoad SIMT post Reduce local target is missing.");
    const Tensor *value_tensor = tpipe.GetTensor(simt_value_tensor_id_);
    GE_ASSERT_NOTNULL(value_tensor, "IndirectLoad SIMT value tensor is missing.");
    // 后置 Reduce 的目标是 IndirectLoad value tensor，而 API 调用的输出会重定向到最终
    // GM result tensor。自行管理 UB 生命周期的模板已暴露该 local tensor；其他模板则需在
    // 提升后的调用前显式分配。
    const bool skips_ub_lifecycle =
        node != nullptr && ascgen_utils::indirect_load::GetTemplateBehavior(node).skips_ub_lifecycle;
    if (!skips_ub_lifecycle && !local_target->no_need_realloc) {
      std::string local_target_alloc;
      GE_ASSERT_SUCCESS(tpipe.TensorAlloc(*local_target, local_target_alloc));
      ss << local_target_alloc;
    }
    af::Expression local_element_count;
    GE_ASSERT_SUCCESS(CalcVectorizedElementCount(*value_tensor, local_element_count));
    actual_size_expr = tpipe.tiler.ActualSize(local_element_count);
    const std::string local_tile_size_expr = tpipe.tiler.Size(local_element_count);
    output_offset_expr = "(static_cast<" + offset_type + ">(block_dim_offset) + static_cast<" + offset_type + ">(" +
                         outer_tb_var + ")) * " + PromoteSizeExpr(local_tile_size_expr, offset_type);
  } else {
    actual_size_expr = outer_tb_var + "_loop_size";
    output_offset_expr = "static_cast<" + offset_type + ">(block_dim_offset)";
  }
  if (!has_post_reduce_ && simt_policy_.policy == ascgen_utils::indirect_load::SimtAddressPolicy::kEmbedding &&
      rank == 2UL && axis_ == 0L) {
    const std::string row_count_expr = PromoteSizeExpr(tpipe.tiler.Size(logical_view_.output.sizes[0]), offset_type);
    const std::string inner_size_expr = PromoteSizeExpr(tpipe.tiler.Size(logical_view_.output.sizes[1]), offset_type);
    ss << "  const " << offset_type << " embedding_rows_per_block = (" << row_count_expr << " + static_cast<"
       << offset_type << ">(t->block_dim) - 1U) / static_cast<" << offset_type << ">(t->block_dim);" << std::endl;
    ss << "  const " << offset_type << " embedding_row_offset = static_cast<" << offset_type
       << ">(block_dim) * embedding_rows_per_block;" << std::endl;
    ss << "  const " << offset_type << " embedding_actual_rows = embedding_row_offset < " << row_count_expr << " ? ("
       << row_count_expr << " - embedding_row_offset < embedding_rows_per_block ? " << row_count_expr
       << " - embedding_row_offset : embedding_rows_per_block) : 0U;" << std::endl;
    actual_size_expr = "embedding_actual_rows * (" + inner_size_expr + ")";
    output_offset_expr = "embedding_row_offset * (" + inner_size_expr + ")";
  }
  EmitSimtPolicyParams(tpipe, simt_policy_, rank, axis_, ss);
  ss << "  AscendC::IndirectLoadSimt<" << input_dtype << ", " << body_name << ", IndirectLoadSimtCase>(\n";
  ss << "      input_ptr, " << body_name << "::OutputTargets{";
  for (size_t i = 0UL; i < output_chains_.size(); ++i) {
    const auto &chain = output_chains_[i];
    std::string dtype;
    GE_ASSERT_SUCCESS(Tensor::DtypeName(chain.dtype, dtype));
    if (i != 0UL) {
      ss << ", ";
    }
    if (chain.local_target) {
      const Tensor *output_tensor = tpipe.GetTensor(chain.target_tensor_id);
      GE_ASSERT_NOTNULL(output_tensor, "IndirectLoad SIMT local output tensor[%ld] is missing.",
                        chain.target_tensor_id);
      ss << "(__ubuf__ " << dtype << " *)" << output_tensor->name << ".GetPhyAddr()";
    } else {
      ss << "(__gm__ " << dtype << " *)" << kGlobalTensorNamePrefix << chain.target_tensor_id << ".GetPhyAddr()";
    }
  }
  ss << "}, context, static_cast<uint32_t>(" << actual_size_expr << "), ";
  ss << output_offset_expr << ", indirect_load_simt_params);" << std::endl;
  return af::SUCCESS;
}

Status IndirectLoadRegApiCall::GenerateSimt(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                                            const Tensor &input, std::string &result) const {
  GE_ASSERT_TRUE(logical_view_.input.kind != ascgen_utils::indirect_load::IndirectLoadLayoutKind::kUnsupported &&
                     logical_view_.index.kind != ascgen_utils::indirect_load::IndirectLoadLayoutKind::kUnsupported,
                 "IndirectLoad SIMT input or index layout is unsupported.");
  GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::ValidateIndirectLoadOutputLayout(logical_view_.output));
  GELOGD("[IndirectLoad] Generate SIMT body for node[%s], output_chains[%zu].", node_name.c_str(),
         output_chains_.size());

  std::string outer_tb_var;
  const bool has_outer_tb =
      FindCurrentAxisVar(tpipe, current_axis, Axis::Type::kAxisTypeBlockInner, outer_axis_, outer_tb_var);
  GE_ASSERT_TRUE(has_outer_tb, "IndirectLoad SIMT current axes must contain the output block-inner axis.");
  std::string input_dtype;
  GE_ASSERT_SUCCESS(Tensor::DtypeName(input.dtype, input_dtype));
  const std::string valid_node_name = ascgen_utils::GenValidName(node_name);
  const std::string context_name = kSimtContextNamePrefix + valid_node_name;
  std::stringstream ss;
  ss << "// IndirectLoad SIMT" << std::endl;
  ss << "{" << std::endl;
  ss << "  __gm__ " << input_dtype << " *input_ptr = (__gm__ " << input_dtype << " *)" << input << ".GetPhyAddr();"
     << std::endl;
  std::string output_inner_size_expr;
  if (UseRuntimeOutputInnerSize(has_post_reduce_, logical_view_)) {
    output_inner_size_expr = "static_cast<uint64_t>(" + tpipe.tiler.Size(logical_view_.output.sizes.back()) + ")";
  }
  GE_ASSERT_SUCCESS(GenerateSimtContextInitializer(context_name, simt_gm_tensors_, tpipe, output_inner_size_expr, ss));
  GE_ASSERT_SUCCESS(GenerateSimtInvocation(tpipe, input_dtype, outer_tb_var, ss));
  ss << "}" << std::endl;
  result = ss.str();
  return af::SUCCESS;
}

static ApiCallRegister<IndirectLoadRegApiCall> register_indirect_load_reg_api_call("IndirectLoadRegApiCall");
}  // namespace codegen
