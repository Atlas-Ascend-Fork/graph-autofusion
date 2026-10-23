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
namespace semantic = ascgen_utils::template_semantic;
namespace {
constexpr size_t kIndirectLoadInputCount = 2UL;
constexpr size_t kIndirectLoadOutputCount = 1UL;
constexpr char kSimtContextNamePrefix[] = "IndirectLoadSimtContext_";
constexpr char kSimtBodyNamePrefix[] = "IndirectLoadSimtBody_";
constexpr char kSimtBodyGuardPrefix[] = "AUTOFUSE_INDIRECT_LOAD_SIMT_BODY_DEFINED_";
constexpr char kGlobalTensorNamePrefix[] = "global_";
constexpr char kSimtGmFieldNamePrefix[] = "gm_";
constexpr char kSimtValueNamePrefix[] = "v_";

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

// [SIMT 动态 shape 坐标分解] SIMT 静态方法体（Index/AicoreIndex/Outputs/Store）作用域
// 内没有 tiling 数据指针 t，坐标分解涉及的运行期符号尺寸/偏移（t->ksN）不能内联——与
// 后置 Reduce 的 output_inner_size 同法，经 Context 字段传入。注册表按 tiler 表达式串
// 去重；空跑收集（CollectSimtRuntimeSizeExprs / fused 空跑）产出表达式列表，Context 字
// 段声明（GenerateSimtContextDefinition，文件作用域）与初始化
// （GenerateSimtContextInitializer，kernel 作用域，t 可见）按同序复现。
class SimtRuntimeSizeRegistry {
 public:
  SimtRuntimeSizeRegistry() = default;
  // 预填已收集条目：正式发射阶段复用收集结果（同输入确定性，查找不新增条目）。
  explicit SimtRuntimeSizeRegistry(std::vector<std::string> prefill) : entries_(std::move(prefill)) {}

  // 编译期常量直接内联渲染；运行期符号注册进 Context 并返回字段引用表达式。
  std::string InlineOrContext(const af::Expression &expr, const TPipe &tpipe, const std::string &offset_type) {
    if (expr.IsConstExpr()) {
      return PromoteSizeExpr(tpipe.tiler.Size(expr), offset_type);
    }
    const std::string tiler_expr = tpipe.tiler.Size(expr);
    for (size_t i = 0UL; i < entries_.size(); ++i) {
      if (entries_[i] == tiler_expr) {
        return ContextFieldRef(i, offset_type);
      }
    }
    entries_.emplace_back(tiler_expr);
    return ContextFieldRef(entries_.size() - 1UL, offset_type);
  }

  const std::vector<std::string> &Entries() const {
    return entries_;
  }

  // 表达式已注册为运行期字段时返回 true 并给出序号（与 InlineOrContext 同一比
  // 较基准：tiler 渲染串）。调用前提：该表达式已被 InlineOrContext 注册过。
  bool TryGetContextIndex(const af::Expression &expr, const TPipe &tpipe, size_t &index) const {
    if (expr.IsConstExpr()) {
      return false;
    }
    const std::string tiler_expr = tpipe.tiler.Size(expr);
    for (size_t i = 0UL; i < entries_.size(); ++i) {
      if (entries_[i] == tiler_expr) {
        index = i;
        return true;
      }
    }
    return false;
  }

  // [SIMT 动态 shape 坐标分解加速] 运行期除数的热循环 div/mod 若直译为软件除法
  // （AIV 无硬件除法器，数十 cycle/次），逐元素坐标分解（fused 消费者的按维取
  // 模）会成为数倍于常量形态（aicc 将字面量除模折叠为乘移）的性能差源。与
  // SIMT 模板 Policy 的 AddressDecoder 同源：每个 rt_size_N 配对预计算一次
  // magic/shift（Context 初始化处），热循环 div/mod 改用 Simt::UintDiv 乘移
  // 序列；静态除数保持字面量除法（编译期折叠）不变。
  static std::string MagicFieldName(size_t index) {
    return "rt_div_magic_" + std::to_string(index);
  }
  static std::string ShiftFieldName(size_t index) {
    return "rt_div_shift_" + std::to_string(index);
  }
  static std::string MagicFieldRef(size_t index) {
    return "context." + MagicFieldName(index);
  }
  static std::string ShiftFieldRef(size_t index) {
    return "context." + ShiftFieldName(index);
  }

  // 除数静态：字面量除法（aicc 折叠）；除数运行期：magic 乘移除法。返回值为商表
  // 达式；quotient_var 非空时改为「先声明商变量再引用」两段式（避免上层取模展开
  // 中重复内联同一除法表达式）。先渲染一次（含注册副作用：运行期除数进入
  // Context 字段表），再查索引分流。
  std::string MakeRuntimeDiv(const std::string &dividend, const af::Expression &divisor, const TPipe &tpipe,
                             const std::string &offset_type, std::string *quotient_var = nullptr) {
    const std::string rendered = InlineOrContext(divisor, tpipe, offset_type);
    size_t index = 0UL;
    const bool is_runtime = TryGetContextIndex(divisor, tpipe, index);
    if (quotient_var != nullptr && is_runtime) {
      *quotient_var = "index_div_" + std::to_string(div_quotient_count_++);
      quotient_stmts_.push_back("    const " + offset_type + " " + *quotient_var + " = Simt::UintDiv(" + dividend +
                                ", " + MagicFieldRef(index) + ", " + ShiftFieldRef(index) + ");");
      return *quotient_var;
    }
    if (!is_runtime) {
      return "(" + dividend + ") / " + rendered;
    }
    return "Simt::UintDiv(" + dividend + ", " + MagicFieldRef(index) + ", " + ShiftFieldRef(index) + ")";
  }

  // 取模展开为 x - div(x) * divisor（乘法廉价；div(x) 的除数与本层除数一致，直
  // 接内联 UintDiv——若 value 已是上层商变量则无重复展开）；除数静态保持字面量
  // 取模。注意：不能复用上层「不同除数」的商变量（语义错配）。
  std::string MakeRuntimeMod(const std::string &value, const af::Expression &divisor, const TPipe &tpipe,
                             const std::string &offset_type) {
    const std::string rendered = InlineOrContext(divisor, tpipe, offset_type);
    size_t index = 0UL;
    if (!TryGetContextIndex(divisor, tpipe, index)) {
      return "(" + value + ") % " + rendered;
    }
    return "(" + value + " - Simt::UintDiv(" + value + ", " + MagicFieldRef(index) + ", " + ShiftFieldRef(index) +
           ") * " + rendered + ")";
  }

  // 一次坐标分解内累积的商变量声明（发射坐标行前先落地，避免重复内联）。
  void TakeQuotientStmts(std::vector<std::string> &stmts) {
    stmts = std::move(quotient_stmts_);
    quotient_stmts_.clear();
  }

  size_t div_quotient_count_ = 0UL;
  std::vector<std::string> quotient_stmts_;
  std::vector<std::string> TakeEntries() {
    return std::move(entries_);
  }

  static std::string FieldName(size_t index) {
    return "rt_size_" + std::to_string(index);
  }
  static std::string ContextFieldRef(size_t index, const std::string &offset_type) {
    return "static_cast<" + offset_type + ">(context." + FieldName(index) + ")";
  }

 private:
  std::vector<std::string> entries_;
};

Status GenerateSimtContextDefinition(const std::string &context_name,
                                     const std::vector<ascgen_utils::indirect_load::SimtGmTensorMetadata> &gm_tensors,
                                     bool has_output_inner_size, bool has_idx_table, const std::string &offset_type,
                                     const std::vector<std::string> &runtime_size_exprs, std::stringstream &ss) {
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
  // [SIMT index 闭包预计算] 索引表（驻留 value tensor 存储的 int32 视图）与行寻
  // 址参数；Index() 退化为一次表读。表在 UB：成员须带 __ubuf__ 地址空间限定，
  // 否则 bisheng 拒绝用 (__ubuf__ int32_t *) 实参初始化。
  if (has_idx_table) {
    ss << "  const __ubuf__ int32_t *idx_table;" << std::endl;
    ss << "  " << offset_type << " idx_base;" << std::endl;
    ss << "  " << offset_type << " idx_row_stride;" << std::endl;
  }
  // [SIMT 动态 shape 坐标分解] 静态方法体内的运行期符号尺寸/偏移（t->ksN）经以下字
  // 段传入；字段序与 GenerateSimtContextInitializer 的初始化序一致。
  for (size_t i = 0UL; i < runtime_size_exprs.size(); ++i) {
    ss << "  uint64_t " << SimtRuntimeSizeRegistry::FieldName(i) << ";" << std::endl;
  }
  // [SIMT 动态 shape 坐标分解加速] 与 rt_size_N 一一配对的除法 magic/shift（初
  // 始化一次，热循环 UintDiv 消费），类型随偏移宽度。
  for (size_t i = 0UL; i < runtime_size_exprs.size(); ++i) {
    ss << "  " << offset_type << " " << SimtRuntimeSizeRegistry::MagicFieldName(i) << ";" << std::endl;
    ss << "  " << offset_type << " " << SimtRuntimeSizeRegistry::ShiftFieldName(i) << ";" << std::endl;
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

// 判断张量向量化视图是否越出 gather 窗口：存在未被窗口维（布局轴 [axis_pos, rank)）
// 派生覆盖的轴（如窗口跨行时由模板 outer 轴派生的 tile 轴出现在视图首部）。
bool TensorViewSpansRows(const TPipe &tpipe, const Tensor &tensor,
                         const ascgen_utils::indirect_load::IndirectLoadTensorLayout &layout, size_t axis_pos) {
  for (ascir::AxisId vec_axis : tensor.vectorized_axis) {
    bool inside_window = false;
    for (size_t dim = axis_pos; dim < layout.axis_ids.size(); ++dim) {
      if (IsAxisDerivedFrom(tpipe, vec_axis, layout.axis_ids[dim])) {
        inside_window = true;
        break;
      }
    }
    if (!inside_window) {
      return true;
    }
  }
  return false;
}

af::Status BuildTensorWindowInfo(const TPipe &tpipe,
                                 const ascgen_utils::indirect_load::IndirectLoadTensorLayout &layout,
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
  for (size_t index = info.sizes.size(); index > 0UL; --index) {
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
      // 可求解多行 tile：逻辑视图轴在张量中被分裂为 TileInner 后代轴（如 a0 -> a0t），
      // 取后代轴的向量化步长（staging 视图步长）。行维（dim < axis_pos）的 staging 按
      // 紧凑布局写入（DMA 剥离行间 padding），行步长必须取 staging 视图步长而非逻辑
      // （GM）步长，否则多行窗口按 padding 步长寻址紧凑缓冲会读错行。gather 轴
      // （dim == axis_pos）步长描述 staged 源区内被 gather 行的寻址几何：staging 按
      // GM 布局整块拷贝源区（含行间 gap，如 rank3_axis1_input_index_gap 用例的
      // 16x8 源区仅内层 5 有效），必须保留逻辑步长，不得取输出侧 tile 后代（如 a2t）
      // 的步长 1（mixed_rank4_axis2 用例回归：行 gather 退化为滑动窗口 gather）；
      // 轴内维度（dim > axis_pos）无后代时沿用 compact_stride（紧凑写回语义）。
      // 固定单行 tile 无向量化后代，保留逻辑步长：单行窗口下行步长不参与寻址（由
      // 窗口基址吸收），维持既有产物不变。
      bool derived = false;
      if (dim < axis_pos) {
        for (size_t vec_pos = 0UL; vec_pos < tensor.vectorized_axis.size(); ++vec_pos) {
          if (IsAxisDerivedFrom(tpipe, tensor.vectorized_axis[vec_pos], layout.axis_ids[dim])) {
            info.strides[dim] = tensor.vectorized_strides[vec_pos];
            derived = true;
            break;
          }
        }
        if (!derived && TensorViewSpansRows(tpipe, tensor, layout, axis_pos)) {
          // 窗口跨行但行维后代轴不可达（tile 轴派生自模板 outer 轴——其谱系根为新建的
          // 模板轴而非本维布局轴，派生查找失败）：staging 将各行按窗口跨度紧凑写入
          // （多块 DMA，dstStride=0），行步长取紧凑累积步长。模板 OuterOffset 按
          // strides 逐维分解 merged outer 行号，紧凑布局下各 row 维步长恰为
          // compact_stride 的逐级累积（最内 row 维为窗口跨度，更外层逐级乘以内层
          // 尺寸），与 staging 逐块紧凑写入的布局一致（如 torch_gather_frontend
          // 用例：GM 64-stride 行 staged 为 32 紧凑块，行步长须 32 而非 64）。
          info.strides[dim] = compact_stride;
          derived = true;
        }
      }
      if (!derived && dim > axis_pos) {
        info.strides[dim] = compact_stride;
      }
    }
    if (af::SymbolicUtils::StaticCheckEq(info.strides[dim], af::sym::kSymbolZero) != af::TriBool::kTrue) {
      compact_stride = af::sym::Mul(info.strides[dim], info.sizes[dim]);
    }
  }
  return af::SUCCESS;
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
                                                  int64_t gather_axis, SimtRuntimeSizeRegistry &runtime_sizes,
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
        has_load_offset ? runtime_sizes.InlineOrContext(load_offset, tpipe, offset_type) : "";
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
      const std::string output_size = runtime_sizes.InlineOrContext(view.sizes[dim], tpipe, offset_type);
      const std::string load_stride = runtime_sizes.InlineOrContext(view.strides[dim], tpipe, offset_type);
      // Different physical extents on the same axis must not share a modulo expression.
      const auto key = std::make_pair(dim, output_size);
      auto found = coordinates.find(key);
      if (found == coordinates.end()) {
        const std::string name = "index_coord_" + std::to_string(coordinates.size());
        // [SIMT 分解共享] fused 视图与输出视图按维对齐时直接复用 Policy 分解结果
        // （随 Address 透传），坐标零除法：gather 轴维取 address.coord_axis（要求
        // 同轴宽）；轴内单维（dim == gather_axis + 1 == rank - 1，两侧单位步长）
        // 用差分 output_index - address.axis_prefix * inner_size。其余形态回退
        // magic div/mod（除数运行期时 UintDiv 乘移序列，静态字面量除模由 aicc
        // 折叠）；商先落局部变量供取模复用。
        bool shared_coord = false;
        if (gather_axis >= 0L && !has_load_offset &&
            af::SymbolicUtils::StaticCheckEq(view.sizes[dim], logical_view.output.sizes[dim]) == af::TriBool::kTrue) {
          if (static_cast<int64_t>(dim) == gather_axis) {
            ss << "    const " << offset_type << " " << name << " = coord_axis;" << std::endl;
            shared_coord = true;
          } else if (static_cast<int64_t>(dim) == gather_axis + 1L && dim + 1UL == rank &&
                     af::SymbolicUtils::StaticCheckEq(view.strides[dim], af::ops::One) == af::TriBool::kTrue &&
                     af::SymbolicUtils::StaticCheckEq(logical_view.output.strides[dim], af::ops::One) ==
                         af::TriBool::kTrue) {
            const std::string inner_total = runtime_sizes.InlineOrContext(view.sizes[dim], tpipe, offset_type);
            ss << "    const " << offset_type << " " << name << " = " << output_index_expr << " - axis_prefix"
               << " * " << inner_total << ";" << std::endl;
            shared_coord = true;
          }
        }
        if (!shared_coord) {
          // [SIMT 32 位双实例] magic 表按 64 位公式预算，32 位 body 的除法回退在
          // 宽类型上求值后截断回 offset_type（共享坐标路径无除法，不受影响）。
          const std::string wide_type = (offset_type == "uint32_t") ? "uint64_t" : offset_type;
          const bool unit_output_stride =
              af::SymbolicUtils::StaticCheckEq(logical_view.output.strides[dim], af::ops::One) == af::TriBool::kTrue;
          std::string quotient_var;
          const std::string dividend =
              unit_output_stride ? "static_cast<" + wide_type + ">(" + output_index_expr + ")"
                                 : runtime_sizes.MakeRuntimeDiv(output_index_expr, logical_view.output.strides[dim],
                                                                tpipe, wide_type, &quotient_var);
          const std::string mod_expr = runtime_sizes.MakeRuntimeMod(dividend, view.sizes[dim], tpipe, wide_type);
          std::vector<std::string> quotient_stmts;
          runtime_sizes.TakeQuotientStmts(quotient_stmts);
          for (const std::string &stmt : quotient_stmts) {
            ss << stmt << std::endl;
          }
          ss << "    const " << offset_type << " " << name << " = static_cast<" << offset_type << ">(" << mod_expr
             << ");" << std::endl;
        }
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

std::string GetSimtCaseTag(const ascgen_utils::indirect_load::SimtPolicyMetadata &policy, size_t rank, int64_t axis,
                           const std::string &offset_type_override = std::string()) {
  const std::string offset_type = offset_type_override.empty() ? GetSimtOffsetType(policy) : offset_type_override;
  std::stringstream ss;
  ss << "AscendC::IndirectLoadSimtCaseTag<static_cast<AscendC::IndirectLoadSimtCase>("
     << static_cast<int64_t>(policy.policy) << "), " << offset_type << ", " << rank << ", " << axis << ", "
     << policy.inner_span << "ULL, " << policy.output_axis_span << "ULL, " << policy.input_axis_stride << "ULL, "
     << policy.input_axis_span << "ULL, " << policy.input_stride_mask << "ULL, " << policy.index_stride_mask << "ULL>";
  return ss.str();
}

std::string GetSimtPolicyParams(const ascgen_utils::indirect_load::SimtPolicyMetadata &policy, const TPipe &tpipe,
                                const std::string &offset_type_override = std::string()) {
  const std::string offset_type = offset_type_override.empty() ? GetSimtOffsetType(policy) : offset_type_override;
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
                              const std::string &offset_expr, const TPipe &tpipe,
                              SimtRuntimeSizeRegistry &runtime_sizes, const std::string &offset_type,
                              std::map<ascir::TensorId, std::string> &values) {
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
  // [SIMT 动态 shape 节点链] Arange 坐标的视图步长/尺寸与坐标分解同源：动态 shape 下
  // 是运行期符号（ksN），静态方法体内既无 tiling 指针 t 也不能裸渲染符号名，须与坐标
  // 分解同法经 Context（rt_size_N）传入；静态 shape 保持常量内联（uint32 产物不变）。
  std::string coordinate = std::string("(") + offset_expr;
  if (af::SymbolicUtils::StaticCheckEq(logical_view.output.strides[logical_pos], af::sym::kSymbolOne) !=
      af::TriBool::kTrue) {
    coordinate += std::string(" / ") +
                  runtime_sizes.InlineOrContext(logical_view.output.strides[logical_pos], tpipe, offset_type);
  }
  coordinate += std::string(") % (") +
                runtime_sizes.InlineOrContext(logical_view.output.sizes[logical_pos], tpipe, offset_type) + ")";
  std::string expr;
  GE_ASSERT_GRAPH_SUCCESS(v2_impl->GenerateSimtScalarExpr(*node, {coordinate}, expr));
  values[node->outputs()[0]->attr.mem.tensor_id] = expr;
  return af::SUCCESS;
}

af::Status EmitSimtEvaluatorNodes(const std::vector<af::AscNodePtr> &nodes, const SimtLoadMetadataMap &load_metadata,
                                  const SimtLoadIndexOffsetExpressions *index_offset_expressions,
                                  const ascgen_utils::indirect_load::TemplateLogicalView &logical_view,
                                  const std::string &output_offset_expr, const std::string &arange_offset_expr,
                                  const TPipe &tpipe, SimtRuntimeSizeRegistry &runtime_sizes,
                                  const std::string &offset_type, std::map<ascir::TensorId, std::string> &values,
                                  std::stringstream &ss) {
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
      GE_ASSERT_SUCCESS(
          EmitSimtArangeExpr(node, logical_view, arange_offset_expr, tpipe, runtime_sizes, offset_type, values));
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
                                     const TPipe &tpipe, SimtRuntimeSizeRegistry &runtime_sizes,
                                     const std::string &offset_type,
                                     const SimtLoadIndexOffsetExpressions *index_offset_expressions = nullptr) {
  GE_ASSERT_SUCCESS(EmitSimtEvaluatorNodes(nodes, load_metadata, index_offset_expressions, logical_view,
                                           output_offset_expr, arange_offset_expr, tpipe, runtime_sizes, offset_type,
                                           values, ss));
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

// [非连续 Store 目标寻址] 尾部 Store 目标视图与逻辑输出视图同 rank 同 sizes 且 strides 不同
// （NonOwningLayout 切片 / cat 输出切片，行 stride 大于逻辑行宽）时，逻辑线性 output_index 不再
// 等于物理元素偏移：按输出轴逐维折叠坐标（offset + Σ coord×stride）。坐标按逻辑 stride 取商、
// 按目标维宽取模，商/模与 Load 侧同法（静态除数编译期折叠，运行期除数走 magic 乘移，符号尺寸
// 经 Context 传入，见 SimtRuntimeSizeRegistry）。目标视图与逻辑视图不一致（ApplyMerge 合并视图、
// post Reduce 并行写回）时保持既有线性寻址，其稠密序不变性由调度门控保证。
std::string BuildSimtStoreTargetOffsetExpr(const SimtOutputChain &chain,
                                           const ascgen_utils::indirect_load::TemplateLogicalView &logical_view,
                                           const TPipe &tpipe, const std::string &offset_type,
                                           SimtRuntimeSizeRegistry &runtime_sizes,
                                           std::map<std::pair<size_t, std::string>, std::string> &coordinates,
                                           std::stringstream &ss) {
  const size_t rank = logical_view.output.sizes.size();
  std::string offset;
  if (af::SymbolicUtils::StaticCheckEq(chain.target_offset, af::ops::Zero) != af::TriBool::kTrue) {
    offset = runtime_sizes.InlineOrContext(chain.target_offset, tpipe, offset_type);
  }
  const std::string linear = offset.empty() ? "output_index" : offset + " + output_index";
  const bool view_matches_logical = chain.target_sizes.size() == rank && chain.target_strides.size() == rank &&
                                    chain.target_sizes == logical_view.output.sizes;
  if (!view_matches_logical) {
    return linear;
  }
  if (chain.target_strides == logical_view.output.strides) {
    return linear;
  }
  std::string expression;
  for (size_t dim = 0UL; dim < rank; ++dim) {
    if (af::SymbolicUtils::StaticCheckEq(chain.target_strides[dim], af::ops::Zero) == af::TriBool::kTrue ||
        af::SymbolicUtils::StaticCheckEq(chain.target_sizes[dim], af::ops::One) == af::TriBool::kTrue) {
      continue;
    }
    const std::string target_size = runtime_sizes.InlineOrContext(chain.target_sizes[dim], tpipe, offset_type);
    // Different target extents on the same axis must not share a modulo expression.
    const auto key = std::make_pair(dim, target_size);
    auto found = coordinates.find(key);
    if (found == coordinates.end()) {
      const std::string name = "store_coord_" + std::to_string(coordinates.size());
      // [SIMT 32 位双实例] 与 Load 侧同法：div/mod 在宽类型上求值后截断回 offset_type。
      const std::string wide_type = (offset_type == "uint32_t") ? "uint64_t" : offset_type;
      const bool unit_logical_stride =
          af::SymbolicUtils::StaticCheckEq(logical_view.output.strides[dim], af::ops::One) == af::TriBool::kTrue;
      std::string quotient_var;
      const std::string dividend = unit_logical_stride
                                       ? "static_cast<" + wide_type + ">(output_index)"
                                       : runtime_sizes.MakeRuntimeDiv("output_index", logical_view.output.strides[dim],
                                                                      tpipe, wide_type, &quotient_var);
      const std::string mod_expr = runtime_sizes.MakeRuntimeMod(dividend, chain.target_sizes[dim], tpipe, wide_type);
      std::vector<std::string> quotient_stmts;
      runtime_sizes.TakeQuotientStmts(quotient_stmts);
      for (const std::string &stmt : quotient_stmts) {
        ss << stmt << std::endl;
      }
      ss << "    const " << offset_type << " " << name << " = static_cast<" << offset_type << ">(" << mod_expr << ");"
         << std::endl;
      found = coordinates.emplace(key, name).first;
    }
    const bool unit_target_stride =
        af::SymbolicUtils::StaticCheckEq(chain.target_strides[dim], af::ops::One) == af::TriBool::kTrue;
    std::string target_stride;
    if (!unit_target_stride) {
      target_stride = runtime_sizes.InlineOrContext(chain.target_strides[dim], tpipe, offset_type);
    }
    const std::string term = unit_target_stride ? found->second : found->second + " * " + target_stride;
    expression += (expression.empty() ? "" : " + ") + term;
  }
  if (!offset.empty()) {
    expression = expression.empty() ? offset : offset + " + " + expression;
  }
  return expression.empty() ? "0" : expression;
}

af::Status GenerateSimtOutputsEvaluator(const std::string &input_dtype, const std::string &offset_type,
                                        int64_t gather_axis, ascir::TensorId value_tensor_id,
                                        const std::vector<SimtOutputChain> &chains,
                                        const std::vector<af::AscNodePtr> &nodes,
                                        const SimtLoadMetadataMap &load_metadata,
                                        const ascgen_utils::indirect_load::TemplateLogicalView &logical_view,
                                        const TPipe &tpipe, bool has_post_reduce,
                                        SimtRuntimeSizeRegistry &runtime_sizes, std::stringstream &ss) {
  GE_ASSERT_TRUE(!chains.empty(), "IndirectLoad SIMT output chains are empty.");
  std::string primary_output_dtype;
  GE_ASSERT_SUCCESS(Tensor::DtypeName(chains[0].dtype, primary_output_dtype));
  ss << "  using PrimaryOutputType = " << primary_output_dtype << ";" << std::endl;
  // 只有输出为gm，纯搬运操作，无element算子操作，且目标为 dense 零偏移布局，才走mte快速搬运路径
  const bool identity_target_dense =
      chains.size() == 1UL && !chains[0].local_target && chains[0].target_strides == logical_view.output.strides &&
      af::SymbolicUtils::StaticCheckEq(chains[0].target_offset, af::ops::Zero) == af::TriBool::kTrue;
  if (identity_target_dense && IsLosslessSimtCastChain(nodes, value_tensor_id, chains[0].result_tensor_id)) {
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
  // [SIMT 分解共享] Policy 分解的 gather 轴坐标与轴前压平商以标量透传（bisheng
  // -x asc 的 SIMT 通道限制 __simt_callee__ 签名引用 Internal 类模板），fused
  // consumer 可差分免除逐元素除法（不可差分维回退除法）。
  ss << "  __simt_callee__ __aicore__ inline static OutputPack Outputs(" << input_dtype << " value, " << offset_type
     << " output_index, " << offset_type << " index_offset, " << offset_type << " coord_axis, " << offset_type
     << " axis_prefix, const Context &context) {" << std::endl;
  SimtLoadIndexOffsetExpressions offsets;
  GE_ASSERT_SUCCESS(BuildSimtPerLoadIndexOffsetExpressions(logical_view, nodes, load_metadata, tpipe, offset_type,
                                                           "output_index", "output_index", has_post_reduce, gather_axis,
                                                           runtime_sizes, offsets, ss));
  GE_ASSERT_SUCCESS(EmitSimtEvaluatorNodes(nodes, load_metadata, &offsets, logical_view, "output_index", "output_index",
                                           tpipe, runtime_sizes, offset_type, values, ss));
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
  std::map<std::pair<size_t, std::string>, std::string> coordinates;
  std::vector<std::string> store_exprs(chains.size());
  for (size_t i = 0UL; i < chains.size(); ++i) {
    if (chains[i].local_target) {
      continue;
    }
    store_exprs[i] =
        BuildSimtStoreTargetOffsetExpr(chains[i], logical_view, tpipe, offset_type, runtime_sizes, coordinates, ss);
  }
  for (size_t i = 0UL; i < chains.size(); ++i) {
    std::string output_offset = "local_index";
    if (!chains[i].local_target) {
      output_offset = store_exprs[i];
    } else if (!logical_view.output.sizes.empty()) {
      // 编译期常量直接内联；仅运行期符号（如 t->ksN）经 Context 传入，避免在静态 Store 中引用 t->。
      const std::string inner_size = UseRuntimeOutputInnerSize(has_post_reduce, logical_view)
                                         ? "static_cast<" + offset_type + ">(context.output_inner_size)"
                                         : tpipe.tiler.Size(logical_view.output.sizes.back());
      std::string dtype;
      GE_ASSERT_SUCCESS(Tensor::DtypeName(chains[i].dtype, dtype));
      const auto row_stride = "KernelUtils::SizeAlign(" + inner_size + ", 32/sizeof(" + dtype + "))";
      output_offset =
          "(local_index / (" + inner_size + ")) * (" + row_stride + ") + local_index % (" + inner_size + ")";
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
                                 const TPipe &tpipe, bool has_post_reduce, bool use_idx_table,
                                 const std::string &idx_inner_expr, SimtRuntimeSizeRegistry &runtime_sizes,
                                 std::stringstream &ss) {
  std::map<ascir::TensorId, std::string> values;
  ss << "  __simt_callee__ __aicore__ inline static " << index_dtype << " Index(" << offset_type << " output_index, "
     << offset_type << " index_offset, const Context &context) {" << std::endl;
  if (use_idx_table) {
    // [SIMT index 闭包预计算] 链值已按 tile 预计算进索引表：本调用点元素相对表基址
    // 定位（行布局与 Store 一致），Index() 退化为一次表读。
    ss << "    const " << offset_type << " local_index = output_index - context.idx_base;" << std::endl;
    ss << "    return context.idx_table[static_cast<uint64_t>((local_index / " << idx_inner_expr
       << ") * "
          "context.idx_row_stride + local_index % "
       << idx_inner_expr << ")];" << std::endl;
    ss << "  }" << std::endl;
    return af::SUCCESS;
  }
  SimtLoadIndexOffsetExpressions offsets;
  GE_ASSERT_SUCCESS(BuildSimtPerLoadIndexOffsetExpressions(logical_view, nodes, load_metadata, tpipe, offset_type,
                                                           "output_index", "index_offset", has_post_reduce, -1L,
                                                           runtime_sizes, offsets, ss));
  return GenerateSimtEvaluatorBody(nodes, values, result_tensor_id, load_metadata, ss, logical_view, "index_offset",
                                   "output_index", tpipe, runtime_sizes, offset_type, &offsets);
}

af::Status GenAicoreIndexEvaluator(const std::string &index_dtype, const std::string &offset_type,
                                   ascir::TensorId result_tensor_id, const std::vector<af::AscNodePtr> &nodes,
                                   const SimtLoadMetadataMap &load_metadata,
                                   const ascgen_utils::indirect_load::TemplateLogicalView &logical_view,
                                   const TPipe &tpipe, bool has_post_reduce, SimtRuntimeSizeRegistry &runtime_sizes,
                                   std::stringstream &ss) {
  std::map<ascir::TensorId, std::string> values;
  ss << "  static constexpr bool kSupportsAicoreIndex = true;" << std::endl;
  ss << "  __aicore__ inline static " << index_dtype << " AicoreIndex(" << offset_type
     << " output_index, const Context &context) {" << std::endl;
  SimtLoadIndexOffsetExpressions offsets;
  GE_ASSERT_SUCCESS(BuildSimtPerLoadIndexOffsetExpressions(logical_view, nodes, load_metadata, tpipe, offset_type,
                                                           "output_index", "output_index", has_post_reduce, -1L,
                                                           runtime_sizes, offsets, ss));
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

// [SIMT index 闭包预计算] 发射 pre-pass：按行（逻辑输出尾轴宽度）向量化重放 index
// 链。值为 fp32 域（区间分析已证 int 值逐位精确；float→int 截断经 int32 槽往返，
// 用向量 Cast 支持的 CAST_TRUNC（to-zero）模式），比较落 uint8 mask 槽，Select 回
// 值域；行末将链结果截断写入
// 驻留 value tensor 存储的 int32 索引表（行布局与 Store 一致，行宽 = SizeAlign(inner,
// 32B/dtype_size)，表容量由 value tensor 自身的行对齐需求保证）。scratch 槽布局与
// CalcTmpBufSize 的申请一致：槽 i 位于 tmp 偏移 i*Align(inner*4,32B)，mask 槽位于
// 4B 槽区之后按 4B 槽宽对齐覆盖。
af::Status EmitSimtIndexPrepass(const ascgen_utils::indirect_load::SimtIndexPrepassPlan &plan, const TPipe &tpipe,
                                const std::string &offset_type, const std::string &local_target_name, int64_t tmp_id,
                                const std::string &inner_expr, const std::string &slot_stride_expr,
                                const std::string &row_stride_expr, const std::string &actual_size_expr,
                                const std::string &output_offset_expr, std::stringstream &ss) {
  using ascgen_utils::indirect_load::SimtIndexPrepassNodeKind;
  const std::string tmp_name = tpipe.tmp_buf.name + "_" + std::to_string(tmp_id);
  const auto four_byte_slot = [&slot_stride_expr](size_t slot) {
    return "pp_val[" + std::to_string(slot) + " * (" + slot_stride_expr + ")]";
  };
  const auto int32_slot = [&slot_stride_expr](size_t slot) {
    return "pp_i32[" + std::to_string(slot) + " * (" + slot_stride_expr + ")]";
  };
  const auto mask_slot = [&plan, &tmp_name, &slot_stride_expr](size_t slot) {
    return tmp_name + "[" + std::to_string(plan.four_byte_slot_count + slot) + " * (4U * (" + slot_stride_expr + "))]";
  };
  const auto scalar_expr = [](const std::string &value) { return "static_cast<float>(" + value + ")"; };
  // Where 掩码来源：沿透传/同域 Cast 回溯到比较节点的 mask 槽。
  const auto resolve_mask_slot = [&plan](const ascgen_utils::indirect_load::SimtIndexPrepassOperand &operand) {
    size_t cursor = operand.node_index;
    for (size_t hop = 0UL; hop < plan.nodes.size() && cursor < plan.nodes.size(); ++hop) {
      const auto &current = plan.nodes[cursor];
      if (current.kind == SimtIndexPrepassNodeKind::kOpCompare) {
        return current.slot;
      }
      if ((current.kind == SimtIndexPrepassNodeKind::kPassthrough ||
           current.kind == SimtIndexPrepassNodeKind::kCastNoop) &&
          !current.operands.empty() && current.operands.front().is_node) {
        cursor = current.operands.front().node_index;
        continue;
      }
      break;
    }
    return static_cast<size_t>(-1);
  };
  bool has_row_constant = false;
  for (const auto &node_plan : plan.nodes) {
    has_row_constant = has_row_constant || node_plan.kind == SimtIndexPrepassNodeKind::kLeafArangeRowConstant;
  }
  ss << "  // [IndirectLoad SIMT] index 闭包向量化预计算（fp32 域重放，表写 value tensor 存储）。" << std::endl;
  ss << "  {" << std::endl;
  ss << "    const uint32_t pp_inner = " << inner_expr << ";" << std::endl;
  ss << "    const uint32_t pp_rows = (static_cast<uint32_t>(" << actual_size_expr << ") + pp_inner - 1U) / pp_inner;"
     << std::endl;
  ss << "    const uint32_t pp_stride = " << slot_stride_expr << ";" << std::endl;
  ss << "    const uint32_t pp_row_stride = " << row_stride_expr << ";" << std::endl;
  ss << "    LocalTensor<float> pp_val = " << tmp_name << ".ReinterpretCast<float>();" << std::endl;
  ss << "    LocalTensor<int32_t> pp_i32 = " << tmp_name << ".ReinterpretCast<int32_t>();" << std::endl;
  ss << "    LocalTensor<int32_t> pp_idx = " << local_target_name << ".ReinterpretCast<int32_t>();" << std::endl;
  ss << "    for (uint32_t pp_r = 0U; pp_r < pp_rows; ++pp_r) {" << std::endl;
  if (has_row_constant) {
    ss << "      const " << offset_type << " pp_row_base = " << output_offset_expr << " + static_cast<" << offset_type
       << ">(pp_r) * static_cast<" << offset_type << ">(pp_inner);" << std::endl;
  }
  for (const auto &node_plan : plan.nodes) {
    switch (node_plan.kind) {
      case SimtIndexPrepassNodeKind::kLeafArangeInnerRamp: {
        // 列坐标 ramp [0, inner)：int32 ramp → fp32，叠加 base/step。
        const size_t ramp_slot = node_plan.temp_slots.front();
        ss << "      GenOneInnerDimTransposeIndex<int32_t>((__ubuf__ int32_t *)" << int32_slot(ramp_slot)
           << ".GetPhyAddr(), 1, pp_inner);" << std::endl;
        ss << "      AscendC::Cast(" << four_byte_slot(node_plan.slot) << ", " << int32_slot(ramp_slot)
           << ", AscendC::RoundMode::CAST_NONE, pp_inner);" << std::endl;
        int64_t base_const = 0L;
        int64_t step_const = 1L;
        (void)node_plan.arange.base.GetConstValue(base_const);
        (void)node_plan.arange.step.GetConstValue(step_const);
        if (step_const != 1L) {
          ss << "      AscendC::Muls(" << four_byte_slot(node_plan.slot) << ", " << four_byte_slot(node_plan.slot)
             << ", " << scalar_expr(std::to_string(step_const)) << ", pp_inner);" << std::endl;
        }
        if (base_const != 0L) {
          ss << "      AscendC::Adds(" << four_byte_slot(node_plan.slot) << ", " << four_byte_slot(node_plan.slot)
             << ", " << scalar_expr(std::to_string(base_const)) << ", pp_inner);" << std::endl;
        }
        break;
      }
      case SimtIndexPrepassNodeKind::kLeafArangeRowConstant: {
        // 行常量坐标：coord = (row_base / stride) % size，value = base + coord*step。
        int64_t base_const = 0L;
        int64_t step_const = 1L;
        (void)node_plan.arange.base.GetConstValue(base_const);
        (void)node_plan.arange.step.GetConstValue(step_const);
        const std::string stride_expr = PromoteSizeExpr(tpipe.tiler.Size(node_plan.arange.view_stride), offset_type);
        const std::string size_expr = PromoteSizeExpr(tpipe.tiler.Size(node_plan.arange.view_size), offset_type);
        ss << "      AscendC::Duplicate(" << four_byte_slot(node_plan.slot) << ", static_cast<float>((static_cast<"
           << "int64_t>(" << base_const << ") + static_cast<int64_t>((pp_row_base / " << stride_expr << ") % "
           << size_expr << ") * static_cast<int64_t>(" << step_const << "))), pp_inner);" << std::endl;
        break;
      }
      case SimtIndexPrepassNodeKind::kOpBinary: {
        const auto &lhs = node_plan.operands[0];
        const auto &rhs = node_plan.operands[1];
        const std::string &dst = four_byte_slot(node_plan.slot);
        static const char *kTensorApis[] = {"Add", "Sub", "Mul", "Min", "Max"};
        static const char *kScalarApis[] = {"Adds", "Subs", "Muls", "Mins", "Maxs"};
        if (lhs.is_node && rhs.is_node) {
          ss << "      AscendC::" << kTensorApis[node_plan.op_code] << "(" << dst << ", "
             << four_byte_slot(plan.nodes[lhs.node_index].slot) << ", "
             << four_byte_slot(plan.nodes[rhs.node_index].slot) << ", pp_inner);" << std::endl;
        } else {
          // 标量操作数走 *s 变体；Add/Mul/Min/Max 可交换（标量在左时换边），Sub 的
          // 被减数为标量的形态在计划期已拒绝。
          const auto &tensor_operand = lhs.is_node ? lhs : rhs;
          const auto &scalar_operand = lhs.is_node ? rhs : lhs;
          ss << "      AscendC::" << kScalarApis[node_plan.op_code] << "(" << dst << ", "
             << four_byte_slot(plan.nodes[tensor_operand.node_index].slot) << ", "
             << scalar_expr(scalar_operand.scalar_value) << ", pp_inner);" << std::endl;
        }
        break;
      }
      case SimtIndexPrepassNodeKind::kOpUnary: {
        const std::string api = node_plan.op_code == 0UL ? "Abs" : "Ln";
        ss << "      AscendC::" << api << "(" << four_byte_slot(node_plan.slot) << ", "
           << four_byte_slot(plan.nodes[node_plan.operands.front().node_index].slot) << ", pp_inner);" << std::endl;
        break;
      }
      case SimtIndexPrepassNodeKind::kOpCompare: {
        static const char *kCmpModes[] = {"GT", "GE", "LT", "LE", "EQ", "NE"};
        static const char *kMirroredModes[] = {"LT", "LE", "GT", "GE", "EQ", "NE"};
        const auto &lhs = node_plan.operands[0];
        const auto &rhs = node_plan.operands[1];
        if (lhs.is_node && rhs.is_node) {
          ss << "      AscendC::Compare(" << mask_slot(node_plan.slot) << ", "
             << four_byte_slot(plan.nodes[lhs.node_index].slot) << ", "
             << four_byte_slot(plan.nodes[rhs.node_index].slot)
             << ", AscendC::CMPMODE::" << kCmpModes[node_plan.op_code] << ", pp_inner);" << std::endl;
        } else {
          // 标量比较走 Compares；标量在左时按镜像模式换边（a>b 等价 b<a）。
          const auto &tensor_operand = lhs.is_node ? lhs : rhs;
          const auto &scalar_operand = lhs.is_node ? rhs : lhs;
          const char *mode = lhs.is_node ? kCmpModes[node_plan.op_code] : kMirroredModes[node_plan.op_code];
          ss << "      AscendC::Compares(" << mask_slot(node_plan.slot) << ", "
             << four_byte_slot(plan.nodes[tensor_operand.node_index].slot) << ", "
             << scalar_expr(scalar_operand.scalar_value) << ", AscendC::CMPMODE::" << mode << ", pp_inner);"
             << std::endl;
        }
        break;
      }
      case SimtIndexPrepassNodeKind::kOpWhere: {
        const size_t mask_slot_index = resolve_mask_slot(node_plan.operands[0]);
        GE_ASSERT_TRUE(mask_slot_index != static_cast<size_t>(-1),
                       "IndirectLoad SIMT prepass Where mask slot is unresolved.");
        size_t temp_cursor = 0UL;
        std::string branch_exprs[2];
        for (size_t branch_pos = 0UL; branch_pos < 2UL; ++branch_pos) {
          const auto &operand = node_plan.operands[1UL + branch_pos];
          if (operand.is_node) {
            branch_exprs[branch_pos] = four_byte_slot(plan.nodes[operand.node_index].slot);
          } else {
            // 标量分支物化到节点内临时槽。
            const size_t temp = node_plan.temp_slots[temp_cursor++];
            ss << "      AscendC::Duplicate(" << four_byte_slot(temp) << ", " << scalar_expr(operand.scalar_value)
               << ", pp_inner);" << std::endl;
            branch_exprs[branch_pos] = four_byte_slot(temp);
          }
        }
        ss << "      AscendC::Select(" << four_byte_slot(node_plan.slot) << ", " << mask_slot(mask_slot_index) << ", "
           << branch_exprs[0] << ", " << branch_exprs[1] << ", AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, pp_inner);"
           << std::endl;
        break;
      }
      case SimtIndexPrepassNodeKind::kCastTruncToInt: {
        // float→int 截断：向量 Cast 无 CAST_ZERO 模式（dav 仅 SIMT 标量路径支持，
        // 其余落入 impl 的 default 分支并 ASCENDC_ASSERT 报 illegal cast mode），
        // CAST_TRUNC（to-zero）与参考链 (int64_t)float 截断语义一致。
        const size_t roundtrip = node_plan.temp_slots.front();
        ss << "      AscendC::Cast(" << int32_slot(roundtrip) << ", "
           << four_byte_slot(plan.nodes[node_plan.operands.front().node_index].slot)
           << ", AscendC::RoundMode::CAST_TRUNC, pp_inner);" << std::endl;
        ss << "      AscendC::Cast(" << four_byte_slot(node_plan.slot) << ", " << int32_slot(roundtrip)
           << ", AscendC::RoundMode::CAST_NONE, pp_inner);" << std::endl;
        break;
      }
      case SimtIndexPrepassNodeKind::kPassthrough:
      case SimtIndexPrepassNodeKind::kCastNoop:
      case SimtIndexPrepassNodeKind::kLeafScalar:
        break;
    }
  }
  ss << "      AscendC::Cast(pp_idx[pp_r * pp_row_stride], " << four_byte_slot(plan.result_slot)
     << ", AscendC::RoundMode::CAST_TRUNC, pp_inner);" << std::endl;
  ss << "    }" << std::endl;
  ss << "    AscendC::PipeBarrier<PIPE_V>();" << std::endl;
  ss << "  }" << std::endl;
  return af::SUCCESS;
}

std::vector<ascir::SizeExpr> GetSimdOutputStrides(const TPipe &tpipe, const Tensor &tensor,
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
      continue;
    }
    // 可求解多行 tile：逻辑视图轴在张量中被分裂为 TileInner 后代轴（如 a0 -> a0t），
    // 该维输出步长取后代轴在紧凑 tile 视图中的向量化步长（多行窗口的行步长）。
    for (size_t vec_pos = 0UL; vec_pos < tensor.vectorized_axis.size(); ++vec_pos) {
      if (IsAxisDerivedFrom(tpipe, tensor.vectorized_axis[vec_pos], layout.axis_ids[dim])) {
        strides[dim] = tensor.vectorized_strides[vec_pos];
        break;
      }
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
    const auto output_strides = GetSimdOutputStrides(tpipe, output, output_layout);
    ss << "static_cast<uint32_t>(" << logical_output_size << "), static_cast<uint32_t>(" << output.actual_size
       << "), static_cast<uint32_t>(" << input.actual_size << "), " << output_offset << ", "
       << tpipe.tiler.Size(input_info.sizes[axis_pos]) << ", {" << JoinSizeExprs(index_info.sizes, tpipe) << "}, {"
       << JoinSizeExprs(input_info.strides, tpipe) << "}, {" << JoinSizeExprs(index_info.strides, tpipe) << "}, {"
       << JoinSizeExprs(output_strides, tpipe) << "}";
  } else if (metadata.fallback == ascgen_utils::indirect_load::SimdFallback::kStrided) {
    const std::string logical_output_size = GetSimdLogicalOutputSize(tpipe, output);
    const auto output_strides = GetSimdOutputStrides(tpipe, output, output_layout);
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
                          size_t rank, int64_t axis, std::stringstream &ss,
                          const std::string &offset_type_override = std::string(),
                          const std::string &name_suffix = std::string()) {
  ss << "  using IndirectLoadSimtCase" << name_suffix << " = "
     << GetSimtCaseTag(policy, rank, axis, offset_type_override) << ";" << std::endl;
  ss << "  AscendC::IndirectLoadSimtParams<IndirectLoadSimtCase" << name_suffix << "> indirect_load_simt_params"
     << name_suffix << "{" << GetSimtPolicyParams(policy, tpipe, offset_type_override) << "};" << std::endl;
}

af::Status GenerateSimtContextInitializer(
    const std::string &context_name, const std::vector<ascgen_utils::indirect_load::SimtGmTensorMetadata> &gm_tensors,
    const TPipe &tpipe, const std::string &output_inner_size_expr, const std::string &idx_table_expr,
    const std::string &idx_base_expr, const std::string &idx_row_stride_expr,
    const std::vector<std::string> &runtime_size_exprs, const std::string &offset_type, std::stringstream &ss) {
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
  // [SIMT index 闭包预计算] 索引表与行寻址参数（表基址 = 本 tile 的 value tensor 局
  // 部目标首地址，分配已提升到本初始化之前）。
  if (!idx_table_expr.empty()) {
    ss << (gm_tensors.empty() && output_inner_size_expr.empty() ? "" : ", ") << idx_table_expr << ", " << idx_base_expr
       << ", " << idx_row_stride_expr;
  }
  // [SIMT 动态 shape 坐标分解] 运行期符号尺寸在此（kernel 作用域，t 可见）取 tiler 表达
  // 式初始化，供静态方法体经 Context 读取。
  for (size_t i = 0UL; i < runtime_size_exprs.size(); ++i) {
    ss << (gm_tensors.empty() && output_inner_size_expr.empty() && idx_table_expr.empty() && i == 0UL ? "" : ", ")
       << "static_cast<uint64_t>(" << runtime_size_exprs[i] << ")";
  }
  ss << "};" << std::endl;
  // [SIMT 动态 shape 坐标分解加速] 聚合初始化（magic/shift 尾部零值初始化）后逐
  // 字段预算除法 magic（每 kernel 一次，几十 cycle），替代热循环逐元素软件除法。
  for (size_t i = 0UL; i < runtime_size_exprs.size(); ++i) {
    ss << "  AscendC::Internal::IndirectLoadGetUintDivMagicAndShift<" << offset_type << ">(context."
       << SimtRuntimeSizeRegistry::MagicFieldName(i) << ", context." << SimtRuntimeSizeRegistry::ShiftFieldName(i)
       << ", static_cast<" << offset_type << ">(context." << SimtRuntimeSizeRegistry::FieldName(i) << "));"
       << std::endl;
  }
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
    chain.target_sizes = chain_metadata.target_sizes;
    chain.target_strides = chain_metadata.target_strides;
    chain.target_offset = chain_metadata.target_offset;
    output_chains_.emplace_back(std::move(chain));
  }
  GE_ASSERT_TRUE(!output_chains_.empty(), "IndirectLoad SIMT output chains are empty.");
  const size_t local_target_count = static_cast<size_t>(std::count_if(
      output_chains_.begin(), output_chains_.end(), [](const SimtOutputChain &chain) { return chain.local_target; }));
  GE_ASSERT_TRUE(local_target_count <= 1UL, "IndirectLoad SIMT supports at most one local output target.");
  // 与 schedule 侧 ValidateSimtOutputPlan 的放宽对齐：单 gather 的 post Reduce 与 local
  // target 成对出现；多 gather 组员 value 直接写 UB 局部目标（归并/输出链为普通向量代码），
  // 允许 local target 不伴随 post Reduce——双向等价断言会使组员（含 fused primary）在
  // ParseAttr 阶段（ApiCall::Init 逐节点调用，先于 Generate 的成员跳过）断言失败。
  GE_ASSERT_TRUE(!has_post_reduce_ || local_target_count == 1UL,
                 "IndirectLoad SIMT post Reduce requires a local output target.");
  if (has_post_reduce_) {
    GE_ASSERT_TRUE(outputs.size() == kIndirectLoadOutputCount, "Invalid IndirectLoad SIMT output number:%zu.",
                   outputs.size());
    outputs[0].id = output_result_tensor_id_;
  }
  return af::SUCCESS;
}

bool IndirectLoadRegApiCall::SimtSupportsDualOffsetBody() const {
  // 仅动态（uint64）形态双发：静态 shape 已由常量求值直接判 uint32，双发只会
  // 产生一套冗余的同类型 body 与恒真的运行时判定。
  if (GetSimtOffsetType(simt_policy_) != "uint64_t") {
    return false;
  }
  if (has_post_reduce_ || template_id_ != ascir::TemplateId::kIndirectLoadSimt) {
    return false;
  }
  if (simt_policy_.policy != ascgen_utils::indirect_load::SimtAddressPolicy::kStrided) {
    return false;
  }
  const size_t rank = logical_view_.input.sizes.size();
  return rank == 3UL && axis_ == 1L && output_chains_.size() == 1UL;
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
  // [SIMT index 闭包预计算] Index() 形态与调用点 pre-pass 发射共用同一位点。
  ascgen_utils::indirect_load::SimtIndexPrepassPlan prepass_plan;
  const bool use_prepass = CanUseSimtIndexPrepass(tpipe, prepass_plan);
  // [SIMT 32 位双实例] 动态 shape 的偏移域在编译期无法证明 < 2^32（uint32 判定依
  // 赖常量求值必失败），全链路退化为 uint64：AIV 上 64 位乘加/比较成本显著高于
  // 32 位，与静态字面量版本（全 uint32）实测差数倍。此处对「偏移域运行时判定可
  // 32 位」的形态同时发射 uint64/uint32 两套 body，调用点按实际尺寸选择（保守域：
  // 各视图元素总数 <= INT32_MAX，覆盖元素偏移上界）。32 位 body 的坐标差分路径
  // 与静态版同构（零除法），仅不可差分维的回退除法在宽类型上求值后截断。
  const bool dual_offset_body = SimtSupportsDualOffsetBody();
  const std::vector<std::string> body_offset_types =
      dual_offset_body ? std::vector<std::string>{offset_type, "uint32_t"} : std::vector<std::string>{offset_type};
  // [SIMT 动态 shape 坐标分解] 先空跑求值器收集静态方法体引用的运行期符号尺寸
  // （t->ksN 不能内联进静态方法，须走 Context）；Context 结构体先于方法体发射，故收集
  // 先行。prepass 行宽首注册（idx_inner_expr 带出复用）；正式发射以收集结果预填注册
  // 表（同输入确定性，查找不新增条目）。Context 尺寸字段统一 64 位，32 位 body 引用
  // 时显式截断（注册键为 tiler 渲染串，与类型无关，两套 body 字段引用一致）。
  std::string idx_inner_expr;
  const std::vector<std::string> runtime_size_exprs = CollectSimtRuntimeSizeExprs(
      input_dtype, index_dtype, offset_type, tpipe, use_prepass, prepass_plan, idx_inner_expr);
  GE_ASSERT_SUCCESS(GenerateSimtContextDefinition(context_name, simt_gm_tensors_,
                                                  UseRuntimeOutputInnerSize(has_post_reduce_, logical_view_),
                                                  use_prepass, offset_type, runtime_size_exprs, ss));
  const size_t ub_output_count = static_cast<size_t>(std::count_if(
      output_chains_.begin(), output_chains_.end(), [](const SimtOutputChain &chain) { return chain.local_target; }));
  for (const std::string &body_offset_type : body_offset_types) {
    const bool is_dual_u32 = dual_offset_body && body_offset_type == "uint32_t";
    const std::string suffix = is_dual_u32 ? "_u32" : (dual_offset_body ? "_u64" : "");
    ss << "#ifndef " << body_guard << suffix << std::endl;
    ss << "#define " << body_guard << suffix << std::endl;
    SimtRuntimeSizeRegistry runtime_sizes(runtime_size_exprs);
    ss << "struct " << body_name << suffix << " {" << std::endl;
    ss << "  using Context = " << context_name << ";" << std::endl;
    ss << "  static constexpr uint32_t kGmOutputCount = " << output_chains_.size() - ub_output_count << "U;"
       << std::endl;
    ss << "  static constexpr uint32_t kUbOutputCount = " << ub_output_count << "U;" << std::endl;
    GE_ASSERT_SUCCESS(GenSimtIndexEvaluator(index_dtype, body_offset_type, index_result_tensor_id_, index_nodes_,
                                            simt_index_loads_, logical_view_, tpipe, has_post_reduce_, use_prepass,
                                            idx_inner_expr, runtime_sizes, ss));
    if (simt_policy_.policy == ascgen_utils::indirect_load::SimtAddressPolicy::kEmbedding &&
        CanGenerateAicoreIndexEvaluator(index_nodes_, simt_index_loads_)) {
      GE_ASSERT_SUCCESS(GenAicoreIndexEvaluator(index_dtype, body_offset_type, index_result_tensor_id_, index_nodes_,
                                                simt_index_loads_, logical_view_, tpipe, has_post_reduce_,
                                                runtime_sizes, ss));
    }
    GE_ASSERT_SUCCESS(GenerateSimtOutputsEvaluator(input_dtype, body_offset_type, axis_, simt_value_tensor_id_,
                                                   output_chains_, output_nodes_, simt_output_loads_, logical_view_,
                                                   tpipe, has_post_reduce_, runtime_sizes, ss));
    ss << "};" << std::endl;
    ss << "#endif" << std::endl;
  }
  return af::SUCCESS;
}

std::vector<std::string> IndirectLoadRegApiCall::CollectSimtRuntimeSizeExprs(
    const std::string &input_dtype, const std::string &index_dtype, const std::string &offset_type, const TPipe &tpipe,
    bool use_prepass, const ascgen_utils::indirect_load::SimtIndexPrepassPlan &prepass_plan,
    std::string &idx_inner_expr) const {
  // 空跑三个求值器（与 GenerateFuncDefinition 正式发射同参同序）：静态方法体内的
  // tiler 渲染经注册表收集为 Context 字段。输出写入临时流丢弃。
  SimtRuntimeSizeRegistry runtime_sizes;
  idx_inner_expr = use_prepass ? runtime_sizes.InlineOrContext(prepass_plan.inner_size, tpipe, offset_type) : "";
  std::stringstream scratch;
  GE_ASSERT_SUCCESS(GenSimtIndexEvaluator(index_dtype, offset_type, index_result_tensor_id_, index_nodes_,
                                          simt_index_loads_, logical_view_, tpipe, has_post_reduce_, use_prepass,
                                          idx_inner_expr, runtime_sizes, scratch));
  if (simt_policy_.policy == ascgen_utils::indirect_load::SimtAddressPolicy::kEmbedding &&
      CanGenerateAicoreIndexEvaluator(index_nodes_, simt_index_loads_)) {
    GE_ASSERT_SUCCESS(GenAicoreIndexEvaluator(index_dtype, offset_type, index_result_tensor_id_, index_nodes_,
                                              simt_index_loads_, logical_view_, tpipe, has_post_reduce_, runtime_sizes,
                                              scratch));
  }
  GE_ASSERT_SUCCESS(GenerateSimtOutputsEvaluator(input_dtype, offset_type, axis_, simt_value_tensor_id_, output_chains_,
                                                 output_nodes_, simt_output_loads_, logical_view_, tpipe,
                                                 has_post_reduce_, runtime_sizes, scratch));
  return runtime_sizes.TakeEntries();
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
  // seq-V1（L2/L3）：fused primary 统一发射全组，携带 fused 代发标记的非 primary 成员跳过；
  // L4（V2 多 evaluator）无该标记——各成员走标准单 gather invocation。
  if (node != nullptr && ascgen_utils::indirect_load::IsSimtFusedMemberEmitted(node)) {
    result = "// IndirectLoad SIMT group member (emitted by fused primary)\n";
    return af::SUCCESS;
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
  GE_ASSERT_SUCCESS(BuildTensorWindowInfo(tpipe, logical_view_.index, index, axis_pos, index_info));
  std::string x_dtype_name;
  std::string index_dtype_name;
  GE_ASSERT_SUCCESS(Tensor::DtypeName(x.dtype, x_dtype_name));
  GE_ASSERT_SUCCESS(Tensor::DtypeName(index.dtype, index_dtype_name));
  GE_ASSERT_TRUE(x.dtype == y.dtype, "IndirectLoad SK input/output dtype must match.");
  GELOGD("[IndirectLoad] Generate SK body for node[%s], rank[%zu], axis[%zu].", node_name.c_str(), x_info.sizes.size(),
         axis_pos);

  std::stringstream ss;
  ss << "// IndirectLoad SK" << std::endl;
  GE_ASSERT_SUCCESS(SetSemanticFeature(semantic::kSk));
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
  GE_ASSERT_SUCCESS(BuildTensorWindowInfo(tpipe, logical_view_.input, input, axis_pos, input_info));
  GE_ASSERT_SUCCESS(BuildTensorWindowInfo(tpipe, logical_view_.index, index, axis_pos, index_info));
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
  std::string simd_feature = semantic::kSimdPrefix;
  switch (simd_metadata_.fallback) {
    case ascgen_utils::indirect_load::SimdFallback::kGatherApi:
      simd_feature += semantic::kGatherApi;
      break;
    case ascgen_utils::indirect_load::SimdFallback::kStrided:
      simd_feature += semantic::kStrided;
      break;
    case ascgen_utils::indirect_load::SimdFallback::kRegisterGather:
    default:
      simd_feature += simd_metadata_.try_embedding ? semantic::kAdaptive : semantic::kRegisterGather;
      break;
  }
  GE_ASSERT_SUCCESS(SetSemanticFeature(simd_feature));
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

Status IndirectLoadRegApiCall::GenerateSimtInvocation(
    const TPipe &tpipe, const std::string &input_dtype, const std::string &outer_tb_var,
    const ascgen_utils::indirect_load::SimtIndexPrepassPlan *prepass_plan, std::stringstream &ss) const {
  const size_t rank = logical_view_.input.sizes.size();
  const std::string offset_type = GetSimtOffsetType(simt_policy_);
  const std::string body_name = kSimtBodyNamePrefix + ascgen_utils::GenValidName(node_name);
  std::string actual_size_expr;
  std::string output_offset_expr;
  // 局部目标（UB）：post Reduce 合成链（单 gather）或多 gather 组员的 value 直写链——
  // 两者均按 value tensor 元素计数与 tile 偏移寻址（组员 value 即 IL 输出，同构复用）。
  const auto local_chain_iter = std::find_if(output_chains_.begin(), output_chains_.end(),
                                             [](const SimtOutputChain &chain) { return chain.local_target; });
  const bool has_local_target = local_chain_iter != output_chains_.end();
  if (has_local_target) {
    GE_ASSERT_TRUE(has_post_reduce_ || local_chain_iter->target_tensor_id == simt_value_tensor_id_,
                   "IndirectLoad SIMT local output must be the post Reduce chain or the group member value tensor.");
    const Tensor *local_target = tpipe.GetTensor(local_chain_iter->target_tensor_id);
    GE_ASSERT_NOTNULL(local_target, "IndirectLoad SIMT local target is missing.");
    const Tensor *value_tensor = tpipe.GetTensor(simt_value_tensor_id_);
    GE_ASSERT_NOTNULL(value_tensor, "IndirectLoad SIMT value tensor is missing.");
    // 后置 Reduce 的目标是 IndirectLoad value tensor，而 API 调用的输出会重定向到最终
    // GM result tensor。自行管理 UB 生命周期的模板已暴露该 local tensor；其他模板则需在
    // 提升后的调用前显式分配（pre-pass 开启时分配已提升到 Context 初始化之前，表基
    // 址需先于 Context 就位）。
    const bool skips_ub_lifecycle =
        node != nullptr && ascgen_utils::indirect_load::GetTemplateBehavior(node).skips_ub_lifecycle;
    if (prepass_plan == nullptr && !skips_ub_lifecycle && !local_target->no_need_realloc) {
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
    if (prepass_plan != nullptr) {
      // [SIMT index 闭包预计算] 行/槽寻址参数与 Store 行布局一致；scratch 槽宽与
      // CalcTmpBufSize 的申请同源（Align(inner*4,32B)/4 == Align(inner,8)）。
      const std::string inner_expr = PromoteSizeExpr(tpipe.tiler.Size(prepass_plan->inner_size), "uint32_t");
      const std::string slot_stride_expr =
          PromoteSizeExpr(tpipe.tiler.Size(af::sym::Align(prepass_plan->inner_size, 8U)), "uint32_t");
      const uint32_t dtype_size = static_cast<uint32_t>(af::GetSizeByDataType(value_tensor->dtype));
      const std::string row_stride_expr =
          PromoteSizeExpr(tpipe.tiler.Size(af::sym::Align(prepass_plan->inner_size, 32U / dtype_size)), "uint32_t");
      const auto tmp_iter = tmp_buf_id.find(-1L);
      GE_ASSERT_TRUE(tmp_iter != tmp_buf_id.end(), "IndirectLoad SIMT prepass requires an API-level tmp buffer.");
      GE_ASSERT_SUCCESS(EmitSimtIndexPrepass(*prepass_plan, tpipe, offset_type, local_target->name, tmp_iter->second,
                                             inner_expr, slot_stride_expr, row_stride_expr, actual_size_expr,
                                             output_offset_expr, ss));
    }
  } else {
    actual_size_expr = outer_tb_var + "_loop_size";
    output_offset_expr = "static_cast<" + offset_type + ">(block_dim_offset)";
  }
  if (!has_local_target && !has_post_reduce_ &&
      simt_policy_.policy == ascgen_utils::indirect_load::SimtAddressPolicy::kEmbedding && rank == 2UL && axis_ == 0L) {
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
  if (!SimtSupportsDualOffsetBody()) {
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
  // [SIMT 32 位双实例] 运行时判定偏移域（各视图元素总数即元素偏移上界）是否
  // 可 32 位；可行走 uint32 body（与静态字面量版本同构的 32 位全链路），否则
  // 回退 uint64 body。两套 body 共享同一 Context（字段 64 位，32 位引用截断）。
  {
    auto total_elements = [](const auto &view) {
      af::Expression total = af::sym::kSymbolOne;
      for (const af::Expression &size : view.sizes) {
        total = af::sym::Mul(total, size);
      }
      return total;
    };
    std::stringstream guard;
    guard << "(static_cast<uint64_t>("
          << PromoteSizeExpr(tpipe.tiler.Size(total_elements(logical_view_.input)), "uint64_t")
          << ") <= 0x7FFFFFFFULL)";
    guard << " && (static_cast<uint64_t>("
          << PromoteSizeExpr(tpipe.tiler.Size(total_elements(logical_view_.index)), "uint64_t")
          << ") <= 0x7FFFFFFFULL)";
    guard << " && (static_cast<uint64_t>("
          << PromoteSizeExpr(tpipe.tiler.Size(total_elements(logical_view_.output)), "uint64_t")
          << ") <= 0x7FFFFFFFULL)";
    EmitSimtPolicyParams(tpipe, simt_policy_, rank, axis_, ss, "uint64_t", "_u64");
    EmitSimtPolicyParams(tpipe, simt_policy_, rank, axis_, ss, "uint32_t", "_u32");
    ss << "  const bool indirect_load_simt_use_u32 = " << guard.str() << ";" << std::endl;
    // 目标表达式与 body 名解耦预计算（GE_ASSERT_SUCCESS 宏含 return，不能放进
    // lambda——返回类型推导冲突触发 -Werror=return-type）。
    std::string targets_fields;
    {
      std::stringstream tss;
      for (size_t i = 0UL; i < output_chains_.size(); ++i) {
        const auto &chain = output_chains_[i];
        std::string dtype;
        GE_ASSERT_SUCCESS(Tensor::DtypeName(chain.dtype, dtype));
        if (i != 0UL) {
          tss << ", ";
        }
        if (chain.local_target) {
          const Tensor *output_tensor = tpipe.GetTensor(chain.target_tensor_id);
          GE_ASSERT_NOTNULL(output_tensor, "IndirectLoad SIMT local output tensor[%ld] is missing.",
                            chain.target_tensor_id);
          tss << "(__ubuf__ " << dtype << " *)" << output_tensor->name << ".GetPhyAddr()";
        } else {
          tss << "(__gm__ " << dtype << " *)" << kGlobalTensorNamePrefix << chain.target_tensor_id << ".GetPhyAddr()";
        }
      }
      targets_fields = tss.str();
    }
    ss << "  if (indirect_load_simt_use_u32) {" << std::endl;
    ss << "    AscendC::IndirectLoadSimt<" << input_dtype << ", " << body_name << "_u32, IndirectLoadSimtCase_u32>(\n";
    ss << "      input_ptr, " << body_name << "_u32::OutputTargets{" << targets_fields << "}, context, "
       << "static_cast<uint32_t>(" << actual_size_expr << "), ";
    ss << "static_cast<uint32_t>(" << output_offset_expr << "), indirect_load_simt_params_u32);" << std::endl;
    ss << "  } else {" << std::endl;
    ss << "    AscendC::IndirectLoadSimt<" << input_dtype << ", " << body_name << "_u64, IndirectLoadSimtCase_u64>(\n";
    ss << "      input_ptr, " << body_name << "_u64::OutputTargets{" << targets_fields << "}, context, "
       << "static_cast<uint32_t>(" << actual_size_expr << "), ";
    ss << "static_cast<uint32_t>(" << output_offset_expr << "), indirect_load_simt_params_u64);" << std::endl;
    ss << "  }" << std::endl;
  }
  return af::SUCCESS;
}

bool IndirectLoadRegApiCall::CanUseSimtIndexPrepass(const TPipe &tpipe,
                                                    ascgen_utils::indirect_load::SimtIndexPrepassPlan &plan) const {
  plan = ascgen_utils::indirect_load::SimtIndexPrepassPlan{};
  if (template_id_ != ascir::TemplateId::kIndirectLoadSimt || !has_post_reduce_ || node == nullptr ||
      logical_view_.output.sizes.empty()) {
    return false;
  }
  if (ascgen_utils::indirect_load::BuildSimtIndexPrepassPlan(*node, plan) != af::SUCCESS || !plan.eligible) {
    return false;
  }
  // scratch 必须已由调度期 CalcTmpBufSize 申请（同一判定；两处图副本结构漂移时以此
  // 为准回退逐元素路径）。
  if (tmp_buf_id.find(-1L) == tmp_buf_id.end()) {
    return false;
  }
  // [漂移防护] 调度期与 codegen 期各构建一次 plan，两份图副本之间图可能被改写
  // （真机日志实证 chain[40]→[36]）：scratch 预留尺寸以调度期 stash 的需求为准，
  // 重建需求超出预留时越界写会穿透同 b0 的其他 tmp 用户，必须回退逐元素路径。
  if (!ascgen_utils::indirect_load::SimtIndexPrepassPlanFitsTmpDemand(
          plan, ascgen_utils::indirect_load::GetSimtIndexPrepassTmpDemand(*node))) {
    return false;
  }
  // tile 整行对齐：向量化元素数须被尾轴整除（tile 基址 = 整数×元素数），否则
  // ramp/行常量叶的行内一致性不成立。
  const Tensor *value_tensor = tpipe.GetTensor(simt_value_tensor_id_);
  if (value_tensor == nullptr) {
    return false;
  }
  af::Expression element_count;
  GE_ASSERT_SUCCESS(CalcVectorizedElementCount(*value_tensor, element_count));
  const af::Expression row_aligned = af::sym::Mul(af::sym::Div(element_count, plan.inner_size), plan.inner_size);
  return af::SymbolicUtils::StaticCheckEq(row_aligned, element_count) == af::TriBool::kTrue;
}

Status IndirectLoadRegApiCall::GenerateSimtLocalTargetAlloc(const TPipe &tpipe, std::stringstream &ss) const {
  if (!has_post_reduce_) {
    return af::SUCCESS;
  }
  const auto local_chain = std::find_if(output_chains_.begin(), output_chains_.end(),
                                        [](const SimtOutputChain &chain) { return chain.local_target; });
  GE_ASSERT_TRUE(local_chain != output_chains_.end(), "IndirectLoad SIMT post Reduce local output is missing.");
  const Tensor *local_target = tpipe.GetTensor(local_chain->target_tensor_id);
  GE_ASSERT_NOTNULL(local_target, "IndirectLoad SIMT post Reduce local target is missing.");
  const bool skips_ub_lifecycle =
      node != nullptr && ascgen_utils::indirect_load::GetTemplateBehavior(node).skips_ub_lifecycle;
  if (skips_ub_lifecycle || local_target->no_need_realloc) {
    return af::SUCCESS;
  }
  std::string local_target_alloc;
  GE_ASSERT_SUCCESS(tpipe.TensorAlloc(*local_target, local_target_alloc));
  ss << local_target_alloc;
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
  // [SIMT index 闭包预计算] 判定与计划：Context 的表字段、Index() 表读形态与调用点
  // pre-pass 发射共用同一位点，避免形态漂移。
  ascgen_utils::indirect_load::SimtIndexPrepassPlan prepass_plan;
  const bool use_prepass = CanUseSimtIndexPrepass(tpipe, prepass_plan);
  const std::string offset_type = GetSimtOffsetType(simt_policy_);
  std::stringstream ss;
  const std::string simt_feature = std::string(semantic::kSimtPrefix) +
                                   ascgen_utils::GetSimtSemanticRole(simt_policy_.policy) + semantic::kOffsetPrefix +
                                   (offset_type == "uint32_t" ? "32" : "64");
  GE_ASSERT_SUCCESS(SetSemanticFeature(simt_feature));
  // [seq-V1] fused primary（六级降级阶梯 L2/L3，见设计文档）：单 evaluator 统一发射全组——
  // 一个 VF_CALL（IndirectLoadSimtFused）启动线程块，每线程逐路串行 index 链求值 →
  // AddressPolicy 寻址 → load。L2（寄存器归并）：归并树在 FusedEval 内以 N 个 value 为种子
  // 经 EmitSimtEvaluatorNodes 按拓扑序折叠为标量表达式（任意 element-wise 算子，不可交换
  // 算子按操作数序），结果直写 GM 输出链；L3（UB 归并）：各路 value 写各自 UB 局部目标
  // （行对齐寻址），归并链由向量代码执行。相比 V2（N 个 VF_CALL + N 个 PipeBarrier）→
  // 1 个 VF_CALL + 1 个 PipeBarrier。L4（V2）不进入本分支（无 fused primary 标记）。
  if (node != nullptr && ascgen_utils::indirect_load::IsSimtFusedPrimary(node)) {
    const auto member_names = ascgen_utils::indirect_load::GetSimtFusedMemberNames(node);
    GE_ASSERT_TRUE(!member_names.empty(), "IndirectLoad SIMT fused primary has no members.");
    const auto owner_graph = node->GetOwnerComputeGraph();
    GE_ASSERT_NOTNULL(owner_graph, "IndirectLoad SIMT fused primary has no owner graph.");
    const auto merge_info = ascgen_utils::indirect_load::GetSimtFusedMergeInfo(node);
    const size_t member_count = member_names.size();
    ss << "// IndirectLoad SIMT Fused (seq-V1: single evaluator, serial loads, " << member_count << " paths, "
       << (merge_info.register_merge ? "register merge" : "UB merge") << ")\n";
    ss << "{\n";
    // Phase 1：收集全部成员数据（input 指针、index 链节点与负载元数据、地址策略、UB 目标）。
    SimtGraphNodeMap node_map;
    GE_ASSERT_SUCCESS(BuildSimtGraphNodeMap(node, node_map));
    struct FusedMemberData {
      const Tensor *input_tensor;
      const Tensor *ub_target;
      ascir::TensorId value_tensor_id;
      ascir::TensorId index_result_tensor_id;
      std::string dtype_name;
      std::vector<af::AscNodePtr> index_nodes;
      std::vector<ascgen_utils::indirect_load::SimtLoadMetadata> index_loads;
      std::vector<ascgen_utils::indirect_load::SimtGmTensorMetadata> gm_tensors;
      ascgen_utils::indirect_load::SimtPolicyMetadata policy;
      ascgen_utils::indirect_load::TemplateLogicalView logical_view;
      size_t rank;
      int64_t axis;
    };
    std::vector<FusedMemberData> members;
    members.reserve(member_count);
    for (const std::string &member_name : member_names) {
      const af::AscNodePtr member_node =
          std::dynamic_pointer_cast<af::AscNode>(owner_graph->FindNode(member_name.c_str()));
      GE_ASSERT_NOTNULL(member_node, "Fused group member[%s] not found.", member_name.c_str());
      const auto member_inputs = member_node->inputs();
      GE_ASSERT_TRUE(member_inputs.size() >= kIndirectLoadInputCount, "Member[%s] invalid input count.",
                     member_name.c_str());
      ascgen_utils::indirect_load::IndirectLoadLoweringMetadata meta;
      GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::GetLoweringMetadata(member_node, meta));
      FusedMemberData md;
      md.input_tensor =
          tpipe.GetTensor(member_inputs[ascgen_utils::indirect_load::kInputTensorIndex]->attr.mem.tensor_id);
      GE_ASSERT_NOTNULL(md.input_tensor, "Member[%s] input tensor missing.", member_name.c_str());
      GE_ASSERT_TRUE(!meta.simt.output_chains.empty() && meta.simt.output_chains[0].local_target,
                     "Member[%s] has no local_target output chain.", member_name.c_str());
      md.ub_target = tpipe.GetTensor(meta.simt.output_chains[0].target_tensor_id);
      GE_ASSERT_NOTNULL(md.ub_target, "Member[%s] UB target tensor missing.", member_name.c_str());
      md.value_tensor_id = meta.simt.value_tensor_id;
      md.index_result_tensor_id = meta.simt.index_result_tensor_id;
      Tensor::DtypeName(meta.simt.output_dtype, md.dtype_name);
      md.index_loads = meta.simt.index_loads;
      md.gm_tensors = meta.simt.gm_tensors;
      md.policy = meta.simt.policy;
      md.logical_view = meta.logical_view;
      md.rank = meta.logical_view.input.sizes.size();
      md.axis = meta.axis;
      GE_ASSERT_SUCCESS(ResolveSimtNodes(meta.simt.index_node_names, node_map, md.index_nodes));
      members.emplace_back(std::move(md));
    }
    // 归并树节点解析（仅 L2）。输出链 tensor id 与 side load 寻址分类均在当前图上重建——
    // 节点名跨图副本稳定（唯一可信引用），视图/张量 id 在调度期与 codegen 期两份图副本间
    // 会漂移（见 utils「漂移防护」注释），调度期存下的 id 不可直接消费。
    std::vector<af::AscNodePtr> merge_nodes;
    std::vector<ascgen_utils::indirect_load::SimtOutputChainMetadata> merge_chains;
    SimtLoadMetadataMap merge_load_map;
    std::vector<af::AscNodePtr> merge_side_loads;
    std::vector<af::AscNodePtr> emission_nodes;
    if (merge_info.register_merge) {
      GE_ASSERT_SUCCESS(ResolveSimtNodes(merge_info.merge_node_names, node_map, merge_nodes));
      // 输出链：Store 终端按名重解析，result/target tensor id 取当前图。
      for (const std::string &store_name : merge_info.merge_store_names) {
        const auto store_iter = node_map.find(store_name);
        GE_ASSERT_TRUE(store_iter != node_map.end(), "Fused merge store terminal[%s] is missing.", store_name.c_str());
        const auto store_inputs = store_iter->second->inputs();
        GE_ASSERT_TRUE(!store_inputs.empty() && !store_iter->second->outputs().empty() &&
                           store_iter->second->outputs()[0] != nullptr,
                       "Fused merge store terminal[%s] has invalid inputs/outputs.", store_name.c_str());
        ascgen_utils::indirect_load::SimtOutputChainMetadata chain;
        chain.result_tensor_id = store_inputs[0]->attr.mem.tensor_id;
        chain.target_tensor_id = store_iter->second->outputs()[0]->attr.mem.tensor_id;
        chain.dtype = store_inputs[0]->attr.dtype;
        chain.local_target = false;
        merge_chains.emplace_back(std::move(chain));
      }
      // side GM load 发现 + 寻址分类（当前图，与调度期门控同一分类函数——非 GM load 的
      // side input 在此出现即为两份图副本间的结构漂移，显式断言暴露而非静默错算）。
      // merge_nodes 含 ③ side 链节点（merge_node_names 含两者），链上 Load 由同一遍历发现。
      const auto is_merge_node = [&merge_info](const std::string &name) {
        return std::find(merge_info.merge_node_names.begin(), merge_info.merge_node_names.end(), name) !=
               merge_info.merge_node_names.end();
      };
      const auto is_member_node = [&member_names](const std::string &name) {
        return std::find(member_names.begin(), member_names.end(), name) != member_names.end();
      };
      for (const af::AscNodePtr &merge_node : merge_nodes) {
        for (const auto &in_node : merge_node->GetInDataNodes()) {
          const auto producer = std::dynamic_pointer_cast<af::AscNode>(in_node);
          if (producer == nullptr || is_merge_node(producer->GetName()) || is_member_node(producer->GetName()) ||
              af::ops::IsOps<af::ascir_op::Scalar>(producer) || af::ops::IsOps<af::ascir_op::ScalarData>(producer) ||
              af::ops::IsOps<af::ascir_op::Arange>(producer)) {
            continue;
          }
          GE_ASSERT_TRUE(ascgen_utils::indirect_load::IsSimtDirectGmLoadNode(producer),
                         "Fused merge side input[%s] is not a direct GM load (schedule/codegen drift).",
                         producer->GetNamePtr());
          if (std::find(merge_side_loads.begin(), merge_side_loads.end(), producer) != merge_side_loads.end()) {
            continue;
          }
          merge_side_loads.emplace_back(producer);
        }
      }
      for (const af::AscNodePtr &side_load : merge_side_loads) {
        ascgen_utils::indirect_load::SimtLoadMetadata side_meta;
        GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::BuildSimtMergeSideLoadMetadata(side_load, side_meta));
        merge_load_map.emplace(side_meta.node_name, std::move(side_meta));
      }
      // 发射节点列表 = 归并树 + ③ side 链节点 + side load（①②③），按图拓扑序（GetDirectNode）
      // 排列——load/链节点先于消费者，EmitSimtEvaluatorNodes 的 values map 传播依赖此序
      // （side load 不在闭包内，缺席会使归并节点输入无表达式）。
      const auto is_side_load = [&merge_side_loads](const std::string &name) {
        return std::find_if(merge_side_loads.begin(), merge_side_loads.end(), [&name](const af::AscNodePtr &side_load) {
                 return side_load->GetName() == name;
               }) != merge_side_loads.end();
      };
      for (const auto &graph_node : owner_graph->GetDirectNode()) {
        const auto asc_node = std::dynamic_pointer_cast<af::AscNode>(graph_node);
        if (asc_node != nullptr && (is_merge_node(asc_node->GetName()) || is_side_load(asc_node->GetName()))) {
          emission_nodes.emplace_back(asc_node);
        }
      }
      GE_ASSERT_TRUE(!emission_nodes.empty(), "Fused merge emission node list is empty.");
    }
    // context 张量清单（成员序优先、归并树 side input 随后，去重）：index 链 GM load /
    // ScalarData + 归并树 ScalarData side input。字段命名与单 gather Context 一致，
    // EmitSimtEvaluatorNodes 的 context.gm_N / context.v_N 引用直接生效。
    std::vector<ascgen_utils::indirect_load::SimtGmTensorMetadata> context_tensors;
    const auto seen_context_tensor = [&context_tensors](ascir::TensorId tensor_id) {
      return std::find_if(context_tensors.begin(), context_tensors.end(),
                          [tensor_id](const ascgen_utils::indirect_load::SimtGmTensorMetadata &tensor) {
                            return tensor.value_tensor_id == tensor_id;
                          }) != context_tensors.end();
    };
    for (const FusedMemberData &member : members) {
      for (const auto &gm_tensor : member.gm_tensors) {
        if (!seen_context_tensor(gm_tensor.value_tensor_id)) {
          context_tensors.emplace_back(gm_tensor);
        }
      }
    }
    if (merge_info.register_merge) {
      for (const af::AscNodePtr &merge_node : merge_nodes) {
        for (const auto &in_node : merge_node->GetInDataNodes()) {
          const auto producer = std::dynamic_pointer_cast<af::AscNode>(in_node);
          if (producer == nullptr || !af::ops::IsOps<af::ascir_op::ScalarData>(producer) ||
              producer->outputs().empty() || producer->outputs()[0] == nullptr ||
              seen_context_tensor(producer->outputs()[0]->attr.mem.tensor_id)) {
            continue;
          }
          ascgen_utils::indirect_load::SimtGmTensorMetadata tensor;
          tensor.value_tensor_id = producer->outputs()[0]->attr.mem.tensor_id;
          tensor.gm_tensor_id = producer->outputs()[0]->attr.mem.tensor_id;
          tensor.dtype = producer->outputs()[0]->attr.dtype;
          tensor.is_scalar = true;
          context_tensors.emplace_back(std::move(tensor));
        }
      }
      // side GM load 的 context 张量（镜像单 gather AppendSimtGmTensorMetadata：Load 的
      // gm 字段名取其输入 GM 张量 id，与 evaluator 表达式引用的输出 id 同源——direct GM
      // 边界两 tensor 同一）。
      for (const af::AscNodePtr &side_load : merge_side_loads) {
        const auto output = side_load->outputs()[0];
        GE_ASSERT_TRUE(output != nullptr && !side_load->inputs().empty(),
                       "Fused merge side load[%s] has invalid inputs/outputs.", side_load->GetNamePtr());
        if (seen_context_tensor(output->attr.mem.tensor_id)) {
          continue;
        }
        ascgen_utils::indirect_load::SimtGmTensorMetadata tensor;
        tensor.value_tensor_id = output->attr.mem.tensor_id;
        tensor.gm_tensor_id = side_load->inputs()[0]->attr.mem.tensor_id;
        tensor.dtype = output->attr.dtype;
        tensor.is_scalar = false;
        context_tensors.emplace_back(std::move(tensor));
      }
    }
    // [SIMT 动态 shape 坐标分解] 空跑 FusedEval 内的 per-member index 与归并 side-load
    // 坐标分解，收集运行期符号尺寸（t->ksN 不能内联进 FusedBody 静态方法，须经 FusedCtx
    // 字段传入；FusedCtx 先于 FusedBody 发射，故收集先行；空跑输出丢弃，正式发射复用
    // 同一注册表查找）。offset_type 只影响丢弃的引用串包装，不影响收集条目（键为 tiler
    // 表达式），此处取首成员位宽。
    SimtRuntimeSizeRegistry fused_runtime_sizes;
    {
      const std::string collect_offset_type = GetSimtOffsetType(members.front().policy);
      std::stringstream scratch;
      SimtLoadIndexOffsetExpressions scratch_offsets;
      for (size_t k = 0; k < member_count; ++k) {
        SimtLoadMetadataMap load_map;
        for (const auto &load : members[k].index_loads) {
          load_map.emplace(load.node_name, load);
        }
        GE_ASSERT_SUCCESS(BuildSimtPerLoadIndexOffsetExpressions(
            members[k].logical_view, members[k].index_nodes, load_map, tpipe, GetSimtOffsetType(members[k].policy),
            "output_index", "index_offset", false, -1L, fused_runtime_sizes, scratch_offsets, scratch));
        std::map<ascir::TensorId, std::string> scratch_values;
        GE_ASSERT_SUCCESS(EmitSimtEvaluatorNodes(
            members[k].index_nodes, load_map, &scratch_offsets, members[k].logical_view, "index_offset", "output_index",
            tpipe, fused_runtime_sizes, GetSimtOffsetType(members[k].policy), scratch_values, scratch));
      }
      if (merge_info.register_merge) {
        GE_ASSERT_SUCCESS(BuildSimtPerLoadIndexOffsetExpressions(
            members.front().logical_view, emission_nodes, merge_load_map, tpipe, collect_offset_type, "output_index",
            "output_index", false, -1L, fused_runtime_sizes, scratch_offsets, scratch));
        std::map<ascir::TensorId, std::string> scratch_values;
        for (size_t k = 0; k < member_count; ++k) {
          scratch_values[members[k].value_tensor_id] = "value" + std::to_string(k);
        }
        GE_ASSERT_SUCCESS(EmitSimtEvaluatorNodes(emission_nodes, merge_load_map, &scratch_offsets,
                                                 members.front().logical_view, "output_index", "output_index", tpipe,
                                                 fused_runtime_sizes, collect_offset_type, scratch_values, scratch));
      }
    }
    // Phase 2：FusedCtx——N 路 input 指针 + 逐成员策略参数（常量驻留）+ context 张量。
    for (size_t k = 0; k < member_count; ++k) {
      ss << "  using FusedCase" << k << " = " << GetSimtCaseTag(members[k].policy, members[k].rank, members[k].axis)
         << ";\n";
    }
    ss << "  struct FusedCtx {\n";
    for (size_t k = 0; k < member_count; ++k) {
      ss << "    __gm__ " << members[k].dtype_name << " *in" << k << ";\n";
      ss << "    AscendC::IndirectLoadSimtParams<FusedCase" << k << "> params" << k << ";\n";
    }
    for (const auto &gm_tensor : context_tensors) {
      std::string dtype;
      GE_ASSERT_SUCCESS(Tensor::DtypeName(gm_tensor.dtype, dtype));
      if (gm_tensor.is_scalar) {
        ss << "    " << dtype << " " << kSimtValueNamePrefix << gm_tensor.value_tensor_id << ";\n";
      } else {
        ss << "    __gm__ " << dtype << " *" << kSimtGmFieldNamePrefix << gm_tensor.gm_tensor_id << ";\n";
      }
    }
    // [SIMT 动态 shape 坐标分解] FusedBody 静态方法体内的运行期符号尺寸（t->ksN）字段；
    // 初始化序与下方 fctx 一致。
    for (size_t i = 0UL; i < fused_runtime_sizes.Entries().size(); ++i) {
      ss << "    uint64_t " << SimtRuntimeSizeRegistry::FieldName(i) << ";\n";
    }
    ss << "  };\n";
    ss << "  FusedCtx fctx = {\n";
    for (size_t k = 0; k < member_count; ++k) {
      ss << "    (__gm__ " << members[k].dtype_name << " *)" << members[k].input_tensor->name << ".GetPhyAddr(),\n";
      ss << "    {" << GetSimtPolicyParams(members[k].policy, tpipe) << "},\n";
    }
    for (const auto &gm_tensor : context_tensors) {
      std::string dtype;
      GE_ASSERT_SUCCESS(Tensor::DtypeName(gm_tensor.dtype, dtype));
      if (gm_tensor.is_scalar) {
        const Tensor *scalar = tpipe.GetTensor(gm_tensor.value_tensor_id);
        GE_ASSERT_NOTNULL(scalar, "IndirectLoad SIMT fused ScalarData tensor is missing.");
        ss << "    " << scalar->name << ",\n";
      } else {
        ss << "    (__gm__ " << dtype << " *)" << kGlobalTensorNamePrefix << gm_tensor.gm_tensor_id
           << ".GetPhyAddr(),\n";
      }
    }
    // [SIMT 动态 shape 坐标分解] 本作用域（kernel 函数内，t 可见）取 tiler 表达式初始化。
    for (size_t i = 0UL; i < fused_runtime_sizes.Entries().size(); ++i) {
      ss << "    static_cast<uint64_t>(" << fused_runtime_sizes.Entries()[i] << "),\n";
    }
    ss << "  };\n";
    // Phase 3：FusedBody。OffsetType 按成员策略位宽选优（全 uint32 → 32 位偏移）；线程档
    // 上限 = 资源压力模型档位（schedule 侧推得）与 offset 位宽门槛（64 位封顶 1024，对齐
    // 单 gather IndirectLoadUse2048Threads）取小（模型规格见设计文档「寄存器压力模型」）。
    bool all_offset_uint32 = true;
    for (const FusedMemberData &member : members) {
      if (member.policy.offset_width != ascgen_utils::indirect_load::SimtOffsetWidth::kUint32) {
        all_offset_uint32 = false;
        break;
      }
    }
    const std::string fused_offset_type = all_offset_uint32 ? "uint32_t" : "uint64_t";
    uint32_t max_thread_tier = merge_info.max_thread_tier;
    if (!all_offset_uint32) {
      max_thread_tier = std::min(max_thread_tier, 1024U);
    }
    ss << "  struct FusedBody {\n";
    ss << "    using OffsetType = " << fused_offset_type << ";\n";
    ss << "    static constexpr uint32_t kMaxThreadTier = " << max_thread_tier << "U;\n";
    if (merge_info.register_merge) {
      ss << "    struct OutputTargets {";
      for (size_t i = 0; i < merge_chains.size(); ++i) {
        std::string dtype;
        GE_ASSERT_SUCCESS(Tensor::DtypeName(merge_chains[i].dtype, dtype));
        ss << " __gm__ " << dtype << " *out" << i << ";";
      }
      ss << " };\n";
      ss << "    struct OutputPack {";
      for (size_t i = 0; i < merge_chains.size(); ++i) {
        std::string dtype;
        GE_ASSERT_SUCCESS(Tensor::DtypeName(merge_chains[i].dtype, dtype));
        ss << " " << dtype << " out" << i << ";";
      }
      ss << " };\n";
    } else {
      ss << "    struct OutputTargets {";
      for (size_t k = 0; k < member_count; ++k) {
        ss << " __ubuf__ " << members[k].dtype_name << " *val" << k << ";";
      }
      ss << " };\n";
      ss << "    struct OutputPack {";
      for (size_t k = 0; k < member_count; ++k) {
        ss << " " << members[k].dtype_name << " v" << k << ";";
      }
      ss << " };\n";
    }
    ss << "    static OutputPack FusedEval(OffsetType output_index, const FusedCtx& context) {\n";
    ss << "      OutputPack out;\n";
    for (size_t k = 0; k < member_count; ++k) {
      ss << "      " << members[k].dtype_name << " value" << k << ";\n";
      ss << "      {\n";
      ss << "        using Policy" << k << " = AscendC::Internal::IndirectLoadSimtCaseSelector<FusedCase" << k
         << ">::Policy;\n";
      ss << "        const Policy" << k << " policy" << k
         << " = AscendC::Internal::IndirectLoadSimtCaseSelector<FusedCase" << k << ">::MakePolicy(context.params" << k
         << ");\n";
      ss << "        const auto addr" << k << " = policy" << k << ".GetAddress(output_index);\n";
      ss << "        const OffsetType index_offset = static_cast<OffsetType>(addr" << k << ".index_offset);\n";
      SimtLoadMetadataMap load_map;
      for (const auto &load : members[k].index_loads) {
        load_map.emplace(load.node_name, load);
      }
      std::map<ascir::TensorId, std::string> index_values;
      SimtLoadIndexOffsetExpressions index_offsets;
      GE_ASSERT_SUCCESS(BuildSimtPerLoadIndexOffsetExpressions(
          members[k].logical_view, members[k].index_nodes, load_map, tpipe, GetSimtOffsetType(members[k].policy),
          "output_index", "index_offset", false, -1L, fused_runtime_sizes, index_offsets, ss));
      GE_ASSERT_SUCCESS(EmitSimtEvaluatorNodes(
          members[k].index_nodes, load_map, &index_offsets, members[k].logical_view, "index_offset", "output_index",
          tpipe, fused_runtime_sizes, GetSimtOffsetType(members[k].policy), index_values, ss));
      const auto index_result = index_values.find(members[k].index_result_tensor_id);
      GE_ASSERT_TRUE(index_result != index_values.end(), "Fused member[%zu] index result tensor has no scalar value.",
                     k);
      ss << "        const OffsetType idx" << k << " = static_cast<OffsetType>(" << index_result->second << ");\n";
      ss << "        OffsetType input_offset" << k << " = static_cast<OffsetType>(addr" << k << ".input_base);\n";
      ss << "        if constexpr (Policy" << k << "::kUsesInputAxis) {\n";
      ss << "          input_offset" << k << " += idx" << k << " * static_cast<OffsetType>(policy" << k
         << ".input_axis_stride);\n";
      ss << "        }\n";
      ss << "        value" << k << " = context.in" << k << "[input_offset" << k << "];\n";
      ss << "      }\n";
    }
    if (merge_info.register_merge) {
      // L2：N 个 value 为种子，发射列表（归并树 + side 链 + side load，图拓扑序）按序折叠
      // （EmitSimtEvaluatorNodes：任意 element-wise 算子经各自 V2 标量表达式发射，操作数按
      // 输入序——不可交换算子语义正确）。side GM load 经 use_logical_offset 坐标折叠寻址
      // （BuildSimtPerLoadIndexOffsetExpressions，与单 gather Outputs() 同构——strided/行广播/
      // 视图匹配 index 的 ② 形态由物理 view 承载正确寻址）。
      std::map<ascir::TensorId, std::string> merge_values;
      for (size_t k = 0; k < member_count; ++k) {
        merge_values[members[k].value_tensor_id] = "value" + std::to_string(k);
      }
      SimtLoadIndexOffsetExpressions merge_offsets;
      GE_ASSERT_SUCCESS(BuildSimtPerLoadIndexOffsetExpressions(
          members.front().logical_view, emission_nodes, merge_load_map, tpipe, fused_offset_type, "output_index",
          "output_index", false, -1L, fused_runtime_sizes, merge_offsets, ss));
      GE_ASSERT_SUCCESS(EmitSimtEvaluatorNodes(emission_nodes, merge_load_map, &merge_offsets,
                                               members.front().logical_view, "output_index", "output_index", tpipe,
                                               fused_runtime_sizes, fused_offset_type, merge_values, ss));
      for (size_t i = 0; i < merge_chains.size(); ++i) {
        const auto chain_result = merge_values.find(merge_chains[i].result_tensor_id);
        GE_ASSERT_TRUE(chain_result != merge_values.end(), "Fused merge chain[%zu] result tensor has no scalar value.",
                       i);
        ss << "      out.out" << i << " = " << chain_result->second << ";\n";
      }
    } else {
      for (size_t k = 0; k < member_count; ++k) {
        ss << "      out.v" << k << " = value" << k << ";\n";
      }
    }
    ss << "      return out;\n";
    ss << "    }\n";
    ss << "    static void Store(const OutputTargets& t, OffsetType output_index, OffsetType local_index, "
          "const FusedCtx& context, const OutputPack& o) {\n";
    if (merge_info.register_merge) {
      for (size_t i = 0; i < merge_chains.size(); ++i) {
        ss << "      t.out" << i << "[output_index] = o.out" << i << ";\n";
      }
    } else {
      // L3：UB 局部目标行对齐寻址（与单 gather GenerateSimtOutputsEvaluator 的 local target
      // Store 一致；组员无 post-Reduce ⇒ inner size 恒为编译期常量，直接内联）。
      std::string inner_size_expr;
      if (!members.front().logical_view.output.sizes.empty()) {
        inner_size_expr = tpipe.tiler.Size(members.front().logical_view.output.sizes.back());
      }
      for (size_t k = 0; k < member_count; ++k) {
        if (inner_size_expr.empty()) {
          ss << "      t.val" << k << "[local_index] = o.v" << k << ";\n";
        } else {
          const std::string row_stride =
              "KernelUtils::SizeAlign(" + inner_size_expr + ", 32/sizeof(" + members[k].dtype_name + "))";
          ss << "      t.val" << k << "[(local_index / (" << inner_size_expr << ")) * (" << row_stride
             << ") + local_index % (" << inner_size_expr << ")] = o.v" << k << ";\n";
        }
      }
    }
    ss << "    }\n";
    ss << "  };\n";
    // Phase 4：OutputTargets 初始化 + fused 调用。actual_size/output_offset 语义对齐单 gather：
    // L2（GM 直出）= loop size 与 block 偏移；L3（UB 局部目标）= tile 元素计数与 tile 基址偏移
    // （Store 仅用 local_index，基址供 index 链 GM load 寻址）。
    ss << "  FusedBody::OutputTargets ftgt = {";
    if (merge_info.register_merge) {
      for (size_t i = 0; i < merge_chains.size(); ++i) {
        std::string dtype;
        GE_ASSERT_SUCCESS(Tensor::DtypeName(merge_chains[i].dtype, dtype));
        if (i != 0) {
          ss << ", ";
        }
        ss << "(__gm__ " << dtype << " *)" << kGlobalTensorNamePrefix << merge_chains[i].target_tensor_id
           << ".GetPhyAddr()";
      }
    } else {
      for (size_t k = 0; k < member_count; ++k) {
        if (k != 0) {
          ss << ", ";
        }
        ss << "(__ubuf__ " << members[k].dtype_name << " *)" << members[k].ub_target->name << ".GetPhyAddr()";
      }
    }
    ss << "};\n";
    std::string fused_actual_size_expr;
    std::string fused_output_offset_expr;
    if (merge_info.register_merge) {
      fused_actual_size_expr = outer_tb_var + "_loop_size";
      fused_output_offset_expr = "static_cast<" + fused_offset_type + ">(block_dim_offset)";
    } else {
      const Tensor *value_tensor = tpipe.GetTensor(members.front().value_tensor_id);
      GE_ASSERT_NOTNULL(value_tensor, "IndirectLoad SIMT fused value tensor is missing.");
      af::Expression local_element_count;
      GE_ASSERT_SUCCESS(CalcVectorizedElementCount(*value_tensor, local_element_count));
      const std::string local_tile_size_expr = tpipe.tiler.Size(local_element_count);
      fused_actual_size_expr = tpipe.tiler.ActualSize(local_element_count);
      fused_output_offset_expr = "(static_cast<" + fused_offset_type + ">(block_dim_offset) + static_cast<" +
                                 fused_offset_type + ">(" + outer_tb_var + ")) * " +
                                 PromoteSizeExpr(local_tile_size_expr, fused_offset_type);
    }
    ss << "  AscendC::IndirectLoadSimtFused<FusedBody>(ftgt, fctx, static_cast<uint32_t>(" << fused_actual_size_expr
       << "), " << fused_output_offset_expr << ");\n";
    ss << "}\n";
    result = ss.str();
    return af::SUCCESS;
  }
  ss << "// IndirectLoad SIMT" << std::endl;
  ss << "{" << std::endl;
  ss << "  __gm__ " << input_dtype << " *input_ptr = (__gm__ " << input_dtype << " *)" << input << ".GetPhyAddr();"
     << std::endl;
  std::string output_inner_size_expr;
  std::string idx_table_expr;
  std::string idx_base_expr;
  std::string idx_row_stride_expr;
  if (UseRuntimeOutputInnerSize(has_post_reduce_, logical_view_)) {
    output_inner_size_expr = "static_cast<uint64_t>(" + tpipe.tiler.Size(logical_view_.output.sizes.back()) + ")";
  }
  if (use_prepass) {
    // local target 分配提升到 Context 之前：索引表基址需先于 Context 就位。
    GE_ASSERT_SUCCESS(GenerateSimtLocalTargetAlloc(tpipe, ss));
    const auto local_chain = std::find_if(output_chains_.begin(), output_chains_.end(),
                                          [](const SimtOutputChain &chain) { return chain.local_target; });
    GE_ASSERT_TRUE(local_chain != output_chains_.end(), "IndirectLoad SIMT post Reduce local output is missing.");
    const Tensor *local_target = tpipe.GetTensor(local_chain->target_tensor_id);
    GE_ASSERT_NOTNULL(local_target, "IndirectLoad SIMT post Reduce local target is missing.");
    const Tensor *value_tensor = tpipe.GetTensor(simt_value_tensor_id_);
    GE_ASSERT_NOTNULL(value_tensor, "IndirectLoad SIMT value tensor is missing.");
    af::Expression local_element_count;
    GE_ASSERT_SUCCESS(CalcVectorizedElementCount(*value_tensor, local_element_count));
    const std::string local_tile_size_expr = tpipe.tiler.Size(local_element_count);
    const uint32_t dtype_size = static_cast<uint32_t>(af::GetSizeByDataType(value_tensor->dtype));
    idx_table_expr = "(__ubuf__ int32_t *)" + local_target->name + ".GetPhyAddr()";
    idx_base_expr = "(static_cast<" + offset_type + ">(block_dim_offset) + static_cast<" + offset_type + ">(" +
                    outer_tb_var + ")) * " + PromoteSizeExpr(local_tile_size_expr, offset_type);
    idx_row_stride_expr =
        PromoteSizeExpr(tpipe.tiler.Size(af::sym::Align(prepass_plan.inner_size, 32U / dtype_size)), offset_type);
  }
  // [SIMT 动态 shape 坐标分解] 与 GenerateFuncDefinition 同参同序重算（纯函数性输入，
  // 两处结果一致），Context 初始化在本作用域（kernel 函数内，t 可见）取 tiler 表达式。
  std::vector<std::string> runtime_size_exprs;
  {
    std::string index_dtype;
    GE_ASSERT_SUCCESS(Tensor::DtypeName(index_dtype_, index_dtype));
    std::string collect_idx_inner;
    runtime_size_exprs = CollectSimtRuntimeSizeExprs(input_dtype, index_dtype, offset_type, tpipe, use_prepass,
                                                     prepass_plan, collect_idx_inner);
  }
  GE_ASSERT_SUCCESS(GenerateSimtContextInitializer(context_name, simt_gm_tensors_, tpipe, output_inner_size_expr,
                                                   idx_table_expr, idx_base_expr, idx_row_stride_expr,
                                                   runtime_size_exprs, offset_type, ss));
  GE_ASSERT_SUCCESS(
      GenerateSimtInvocation(tpipe, input_dtype, outer_tb_var, use_prepass ? &prepass_plan : nullptr, ss));
  ss << "}" << std::endl;
  result = ss.str();
  return af::SUCCESS;
}

static ApiCallRegister<IndirectLoadRegApiCall> register_indirect_load_reg_api_call("IndirectLoadRegApiCall");
}  // namespace codegen
