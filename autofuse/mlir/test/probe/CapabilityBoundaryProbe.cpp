/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- CapabilityBoundaryProbe.cpp - 哪条「报不支持」在真实图上命中 ---------===//
//
// 这不是测试，是一个测量工具。 它不断言任何事，产物是一张表：形状 → 状态 → 第一条诊断。
// 所以它是一个独立可执行文件、不接任何 `check-*` 目标，要量就手动跑：
//
//   make -C build af-capability-probe -j 6
//   AUTOFUSE_DFX_FLAGS="--mlir_codegen_config=mlir" LD_LIBRARY_PATH="<见下>" \
//     build/autofuse/mlir/test/probe/af-capability-probe > table.txt
//
// `LD_LIBRARY_PATH` 必须与 CPU E2E 用的那一份一致，照抄
// `build/autofuse/mlir/test/e2e/cpu/cases/<case>/CTestTestfile.cmake` 里那一行即可
// （含 CANN 的 `devlib` 与 metadef 的几个 build 目录）。**用别的路径时失败方式是段错误，
// 不是报错**：调度在 `ScheduleUtils::ClearAllSizeVar` 里挂掉，而且第一张图就挂，看起来像
// 建图建错了。实测换成 E2E 那一份就正常。具体是哪个库被解析错了没有查到底——记在这里是为了
// 省下下一轮排查，不是解释机理。
//
// 链接方式照 CPU E2E 的生成器来（只链 `MlirE2eCpuGraphBuilder` + runtime stub，MLIR / AFIR
// 全部从 `libaihac_codegen.so` 里走），因为那是已经被 65 条用例证过的配置。**没有做成 UT 里
// 的一个 `DISABLED_` 用例**：它不是测试，不该挂在 CI 的目标上。
//
// 要回答的问题来自 `docs/development/2026-09-28-afir-vectorized-axis-collapse-design.md`
// 「下一步」第 3 条：那几条保守边界各自对应什么真实场景。今天它们都是「判据推得出来、
// 但没有用例，于是宁可拒收」，而收哪一条要先知道它对应的图长什么样——不知道就开工，是在
// 给一个可能不存在的形状铺砖（2026-09-23 那轮已经栽过一次：`exec_condition` 被当成卡点，
// 重测发现不是）。
//
// 每一行还带一列 `vf=<a>/<b>`：调度产物里带 `VectorFunc` 节点的 impl graph 数 / 总数。
// 加它是因为一格的第一条诊断不足以说明这一格修了就能进来——VF 在 `Optimize` 里就已经
// 融出来，bridge 拒不拒它是后面一道门禁的事，于是诊断可能记在别的 op 上而图里仍带着 VF。
// 判法与 legacy 接管表那一侧同源，成因见 `CountVectorFuncGraphs` 上的说明。
// 这一列只在 `--reg-base` 下会非 0：V1 平台不产生 `VectorFunc`，mem 侧是个自带的对照组。
//
// 走的是和 CPU E2E 生成器完全相同的两步（`test/e2e/cpu/support/Generator.cpp`）：
// `optimize::Optimizer::Optimize` 之后直接调 `RunMlirCodegen`。**不走
// `Codegen::GenerateForInductor`**——那一层会在 MLIR 报不支持时接着跑 legacy 并返回成功，
// 把要量的东西盖掉。
//
//===----------------------------------------------------------------------===//

#include <array>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <unistd.h>

#include "bridge/CodegenOrchestration/CodegenOrchestration.h"

#include "asc_graph_builder.h"
#include "codegen.h"
#include "common/platform_context.h"
#include "optimize.h"

namespace {

constexpr int kDtypeColumnWidth = 4;
constexpr int kTopologyColumnWidth = 3;
constexpr int kGraphColumnWidth = 40;
constexpr int kBroadcastGraphColumnWidth = 34;
constexpr int kTopologyGraphColumnWidth = 26;
constexpr int kStatusColumnWidth = 14;
constexpr int kTailBytesColumnWidth = 5;
constexpr int kBroadcastColumnWidth = 11;
constexpr int kVectorFuncColumnWidth = 7;
constexpr int kBucketColumnWidth = 56;
constexpr int kTableSeparatorWidth = 120;
constexpr int kBucketSeparatorWidth = 70;

// 量哪一侧。reg 侧（950）要同时改两处才算真的换过去，少一处调度与 MLIR 两边对不上：
// `soc_version`（MLIR 按它查 TargetProfile、决定选哪一侧 implementation）与 **NpuArch**
// （legacy 那一半全按它分流，`REG_ASC_IR(...).Impl` 注册的键就是它）。数值与
// `test/e2e/cpu/AddCpuCase.cmake` 的 `reg_base` 那一支一致，那里抄的是
// `target/Ascend/Profiles/platform_config/Ascend950PR_9599.ini`。
struct ProbeTarget {
  const char *soc_version;
  const char *npu_arch;
  int64_t aiv_num;
  int64_t ub_size;
  int64_t l2_size;
};

constexpr ProbeTarget kMemBased{"Ascend910B1", "2201", 48, 245760, 134217728};
constexpr ProbeTarget kRegBased{"Ascend950PR_9599", "3510", 72, 253952, 134217728};

const ProbeTarget *g_target = &kMemBased;

// 换平台走 `SetPlatformInfo`，不是 `SetPlatform`——后者只写 `soc_ver` 就把
// `initialized_` 置了位，其余字段留 0，调度拿到的是一台没有 UB 的机器。
// mem 侧不调它：那一侧由 runtime stub 懒初始化，手填只填得出四个字段，
// `PlatformInfo` 比这多（见 Generator.cpp 上那段）。
void SelectTarget(const ProbeTarget &target) {
  g_target = &target;
  if (&target == &kMemBased) {
    return;
  }
  ge::PlatformInfo info;
  info.soc_ver = target.npu_arch;
  info.aiv_num = target.aiv_num;
  info.ub_size = target.ub_size;
  info.l2_size = target.l2_size;
  ge::PlatformContext::GetInstance().SetPlatformInfo(info);
}

// 量到的一行。
struct ProbeOutcome {
  std::string status;
  std::string diagnostic;
  // 调度产物里有多少张 impl graph 带 `VectorFunc` 节点，以及一共多少张。
  // 这一列不能由状态列推出来，见 `CountVectorFuncGraphs` 上的说明。
  int vfGraphs = 0;
  int implGraphs = 0;
};

// MLIR 的诊断走默认 handler 打到 fd 2，而 `RunMlirCodegen` 自己建 MLIRContext，外面插不进
// handler。所以这里按 fd 收。
//
// 只有测量工具才这么干。 产品代码里要分类失败用的是诊断上的标记
// （`support/CapabilityShortfall.h`），不是文本。
class StderrCapture {
 public:
  StderrCapture() {
    saved_ = ::dup(STDERR_FILENO);
    file_ = std::tmpfile();
    if (file_ != nullptr) {
      ::dup2(::fileno(file_), STDERR_FILENO);
    }
  }

  ~StderrCapture() {
    if (saved_ >= 0) {
      ::dup2(saved_, STDERR_FILENO);
      ::close(saved_);
    }
    if (file_ != nullptr) {
      std::fclose(file_);
    }
  }

  StderrCapture(const StderrCapture &) = delete;
  StderrCapture &operator=(const StderrCapture &) = delete;

  // 第一条 `error:` 行。后面那些通常是同一次失败的连带（`failed to legalize` 之后还会
  // 跟一串），第一条才说明是哪道门禁拦下的。
  std::string FirstErrorLine() {
    if (file_ == nullptr) {
      return {};
    }
    std::fflush(stderr);
    std::rewind(file_);
    std::array<char, 4096> line{};
    while (std::fgets(line.data(), static_cast<int>(line.size()), file_) != nullptr) {
      std::string text(line.data());
      while (!text.empty() && (text.back() == '\n' || text.back() == '\r')) {
        text.pop_back();
      }
      if (text.find("error:") != std::string::npos) {
        return text;
      }
    }
    return {};
  }

 private:
  int saved_ = -1;
  std::FILE *file_ = nullptr;
};

// VF 那一列的拼法，三张表共用一份——两处写法不一致时，量出来的表就不可比了。
std::string VfText(const ProbeOutcome &outcome) {
  return std::to_string(outcome.vfGraphs) + "/" + std::to_string(outcome.implGraphs);
}

std::string ShapeText(const std::vector<int64_t> &shape) {
  std::string text = "[";
  for (size_t index = 0; index < shape.size(); ++index) {
    text += (index == 0 ? "" : ",") + std::to_string(shape[index]);
  }
  return text + "]";
}

std::string StatusText(mlir::bridge::MlirCodegenStatus status) {
  switch (status) {
    case mlir::bridge::MlirCodegenStatus::kSucceeded:
      return "ok";
    case mlir::bridge::MlirCodegenStatus::kNotSupported:
      return "unsupported";
    case mlir::bridge::MlirCodegenStatus::kFailed:
      return "FAILED";
  }
  return "?";
}

// 诊断行归到哪一桶。桶按是哪道门禁拦下的分，不按 op 分——同一道门禁拦下不同 op 是一件
// 事，不同门禁拦下同一个 op 是两件事。
std::string Bucket(const std::string &diagnostic) {
  if (diagnostic.empty()) {
    return "(none)";
  }
  if (diagnostic.find("has no supported Ascend implementation") != std::string::npos) {
    return "capability: no implementation";
  }
  if (diagnostic.find("does not support node") != std::string::npos) {
    return "bridge: node type not modeled";
  }
  // 「能力不足」不止 capability 那一处会报，bridge 也会（`emitCapabilityShortfall`）。
  // 它与上面两桶一样是干净回落，不是硬失败——落进 "other" 会让汇总那一栏读起来像还挂着。
  if (diagnostic.find("broadcast tile and a full-shape tile") != std::string::npos) {
    return "capability: broadcast tile shares a resource";
  }
  // 这一桶应当恒为 0：capability 说支持而 pattern 失败，就是两侧判据漂移
  // （`localTileIsSingleSegment` 那条修的正是它）。非 0 说明又漂了一处。
  if (diagnostic.find("failed to legalize") != std::string::npos) {
    return "DRIFT: capability said yes, pattern said no";
  }
  if (diagnostic.find("constant-cost plan") != std::string::npos) {
    return "plan selection: no constant-cost plan";
  }
  // 二次融合（`afir.vector_func`）那一段收不了。与上面几桶一样是干净回落，分开一桶是因为
  // 它问的是另一张支持表：「这个 op 有没有寄存器形态的下降」，与「平台供不供得起这个
  // tile 级 API」是两件事。落进 "other" 的话，M1 之后那 20 格看起来像是没人管的杂项。
  if (diagnostic.find("has no register-form lowering") != std::string::npos ||
      diagnostic.find("afir.vector_func") != std::string::npos ||
      diagnostic.find("VectorFunc subgraph") != std::string::npos) {
    return "vector func: no register-form lowering";
  }
  return "other";
}

// 调度产物里带 `VectorFunc` 节点的 impl graph 数 / impl graph 总数。
//
// 为什么要单列这一列，而不是从拒收原因里读：VF 是 A5 调度在 buffer 分配阶段做的二次融合
// （`v35/optimize/partition/vector_func_partitioner.cpp`，门槛 `kMinVfNodesNum = 2`，调用点
// `optimize/buffer_allocate/buf_que_allocator.cpp` 对每张 impl graph 无条件调用），它在
// `Optimize` 里就已经发生；bridge 认不认识 `VectorFunc` 是后面一道门禁的事。于是**一格的
// 第一条诊断完全可能记在别的 op 上，而图里仍然带着 VF**——那种格子把记下来那条原因修掉
// 也进不来，因为 VF 还在后面等着。2026-10-17 的 legacy 接管表上这个差额是 93 格
// （reg 侧 200 格里 110 格含 VF，只有 17 格把 VF 记成首因，见
// `docs/development/2026-10-17-vf-node-census-reg.txt`）。
//
// 数的是 `node_idx_to_scheduled_results` 里所有候选模板，不是 plan selection 选中的那一个：
// VF 由调度产生、与选哪个模板无关，而选型发生在这之后。`a/b` 两个数都要：`a == b` 说明这一格
// 没有不带 VF 的后路，`0 < a < b` 说明有。
//
// mem 侧（NpuArch 2201）这一列应当恒为 `0/b`——V1 平台不产生 `VectorFunc`。非 0 说明平台
// 分流那一处又对不上了（`SelectTarget` 上那段说的正是它）。
std::pair<int, int> CountVectorFuncGraphs(const ascir::FusedScheduledResult &fused) {
  int withVf = 0;
  int total = 0;
  for (const auto &nodeResults : fused.node_idx_to_scheduled_results) {
    for (const auto &scheduled : nodeResults) {
      for (const auto &group : scheduled.schedule_groups) {
        for (const auto &graph : group.impl_graphs) {
          ++total;
          for (const auto &node : graph.GetAllNodes()) {
            if (node != nullptr && node->GetType() == "VectorFunc") {
              ++withVf;
              break;
            }
          }
        }
      }
    }
  }
  return {withVf, total};
}

ProbeOutcome RunOneGraph(af::AscGraph &graph) {
  // 进度打到 stderr 并立刻 flush：表是最后一次性打到 stdout 的，中途挂了就看不出挂在哪
  // 一张图上。两边分开，`> table.txt` 拿到的就是干净的表。
  std::fprintf(stderr, "[probe] %s\n", graph.GetName().c_str());
  std::fflush(stderr);
  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  ascir::FusedScheduledResult fused;
  codegen::CodegenResult result;
  ProbeOutcome outcome;
  {
    StderrCapture capture;
    if (optimizer.Optimize(graph, fused) != af::SUCCESS) {
      outcome.status = "schedule-failed";
      outcome.diagnostic = capture.FirstErrorLine();
      return outcome;
    }
    std::tie(outcome.vfGraphs, outcome.implGraphs) = CountVectorFuncGraphs(fused);
    const auto status =
        mlir::bridge::RunMlirCodegen(fused, mlir::bridge::MlirCallerScene::kInductor, g_target->soc_version, result);
    outcome.status = StatusText(status);
    outcome.diagnostic = capture.FirstErrorLine();
  }
  return outcome;
}

std::vector<af::Expression> ToExpressions(const std::vector<int64_t> &shape) {
  std::vector<af::Expression> expressions;
  expressions.reserve(shape.size());
  for (int64_t extent : shape) {
    expressions.push_back(af::Symbol(extent));
  }
  return expressions;
}

// 广播轴与 Load 的 stride，八种拓扑共用这一份。广播轴由 `in` 那一维是 1 而 `out` 不是
// 定出来；广播轴上 Load 的 stride 取 0，其余维按输入自己的行主序算。与
// `test/e2e/cpu/cases/load_broadcast_segmented_store/Case.cpp` 同一写法。
//
// 抽出来而不是各拓扑抄一遍：这条约定一旦两处不一致，量出来的两张表就不可比，而不可比的
// 表看起来仍然像一张表。
struct BroadcastBinding {
  std::vector<af::Expression> inputShape;
  std::vector<af::Expression> outputShape;
  std::vector<af::Expression> loadStrides;
  // 广播轴，内 → 外（下面的遍历次序）。`Broadcast()` 只按下标改 repeats，与次序无关；
  // T4 要按这个次序切两半，所以记下来方向。
  std::vector<int64_t> brcAxes;
};

BroadcastBinding BindBroadcast(const std::vector<int64_t> &out, const std::vector<int64_t> &in) {
  BroadcastBinding binding;
  binding.inputShape = ToExpressions(in);
  binding.outputShape = ToExpressions(out);
  binding.loadStrides.assign(in.size(), af::sym::kSymbolZero);
  int64_t stride = 1;
  for (size_t position = in.size(); position > 0U; --position) {
    const size_t axis = position - 1U;
    if (in[axis] == 1 && out[axis] != 1) {
      binding.brcAxes.push_back(static_cast<int64_t>(axis));
      continue;
    }
    binding.loadStrides[axis] = af::Symbol(stride);
    stride *= in[axis];
  }
  return binding;
}

// `load -> brc -> store`（T1）。
af::AscGraph BuildBroadcastGraph(const std::string &name, const std::vector<int64_t> &out,
                                 const std::vector<int64_t> &in, af::DataType dtype) {
  using af::testing::AscGraphBuilder;
  const BroadcastBinding binding = BindBroadcast(out, in);
  AscGraphBuilder builder(name);
  builder.Loops(binding.outputShape)
      .Data("input0", 0, binding.inputShape, {}, dtype)
      .Load("load0", "input0", binding.inputShape, binding.loadStrides);
  return builder.Broadcast("brc", "load0", binding.brcAxes)
      .Store("store", "brc")
      .Output("output", "store", 0, dtype)
      .Build();
}

// 对照组（T0）：同样的形状、同样的两头搬运，中间不是广播。区分「这条边界是广播独有的」与
// 「它是搬运层的」——`Load / Store 合轴后段数 > 1` 那条就属于后者。
af::AscGraph BuildElementwiseGraph(const std::string &name, const std::vector<int64_t> &out, af::DataType dtype) {
  using af::testing::AscGraphBuilder;
  const std::vector<af::Expression> shape = ToExpressions(out);
  // `Load` 走四参那一支（显式形状 + 空 stride），与 `load_abs_store/Case.cpp` 一致。两参那
  // 一支不给形状，调度在 `ClearAllSizeVar` 里当场段错误 —— 探针不是在量这个。
  AscGraphBuilder builder(name);
  builder.Loops(shape).Data("input0", 0, shape, {}, dtype).Load("load0", "input0", shape, {});
  af::AscGraph graph = builder.Abs("abs", "load0").Store("store", "abs").Output("output", "store", 0, dtype).Build();
  graph.SetTilingKey(0);
  return graph;
}

//===----------------------------------------------------------------------===//
// 级联拓扑（F8，设计见 `docs/development/2026-10-12-afir-elementwise-broadcast-
// generalization-test-design.md` 的功能需求 1）
//
// 当前的 sweep 只有 T1（`load -> brc -> store`），也就是广播的结果直接写回 GM。真实
// 模型里广播几乎总是中间结果，而「广播之后还有人用它」会动到几件 T1 碰不到的事：
// 下游算子与 brc 争 tmp buffer、一个 SSA 上的 descriptor 必须一致、段与段之间的流水同步、
// 以及广播作为某个二元算子的一个 operand（`classifySegments` 的「源与目的地各恰好
// 一个」正是按 T1 那种形状写的）。
//
// 每种拓扑只加一件事，不叠加——一格同时违反两条判据会互相遮蔽（前序文档的变异验证栽过
// 一次）。
//===----------------------------------------------------------------------===//

// T2 `load -> brc -> abs -> store`：广播结果进计算。
af::AscGraph BuildBroadcastThenComputeGraph(const std::string &name, const std::vector<int64_t> &out,
                                            const std::vector<int64_t> &in, af::DataType dtype) {
  using af::testing::AscGraphBuilder;
  const BroadcastBinding binding = BindBroadcast(out, in);
  AscGraphBuilder builder(name);
  builder.Loops(binding.outputShape)
      .Data("input0", 0, binding.inputShape, {}, dtype)
      .Load("load0", "input0", binding.inputShape, binding.loadStrides);
  return builder.Broadcast("brc", "load0", binding.brcAxes)
      .Abs("abs", "brc")
      .Store("store", "abs")
      .Output("output", "store", 0, dtype)
      .Build();
}

// T3 `add(load(full), brc(load(small))) -> store`：广播作为二元算子的一个 operand。
// 这是真实模型里最常见的那一种，今天零覆盖。
af::AscGraph BuildBroadcastOperandGraph(const std::string &name, const std::vector<int64_t> &out,
                                        const std::vector<int64_t> &in, af::DataType dtype) {
  using af::testing::AscGraphBuilder;
  const BroadcastBinding binding = BindBroadcast(out, in);
  AscGraphBuilder builder(name);
  builder.Loops(binding.outputShape)
      .Data("input0", 0, binding.inputShape, {}, dtype)
      .Data("input1", 1, binding.outputShape, {}, dtype)
      .Load("load0", "input0", binding.inputShape, binding.loadStrides)
      .Load("load1", "input1", binding.outputShape, {});
  return builder.Broadcast("brc", "load0", binding.brcAxes)
      .Add("add", "load1", "brc")
      .Store("store", "add")
      .Output("output", "store", 0, dtype)
      .Build();
}

// T4 `load -> brc -> brc -> store`：两步广播，第二个 brc 的源是片上张量而不是 Load 的
// 结果。要两个广播轴才造得出来（`ProbeTopologies` 里按 `needsTwoBroadcastAxes` 跳过）。
af::AscGraph BuildChainedBroadcastGraph(const std::string &name, const std::vector<int64_t> &out,
                                        const std::vector<int64_t> &in, af::DataType dtype) {
  using af::testing::AscGraphBuilder;
  const BroadcastBinding binding = BindBroadcast(out, in);
  // 内侧那一半先广播，其余交给第二趟。`brcAxes` 是内 → 外，所以前半是内侧那些。
  const size_t split = binding.brcAxes.size() / 2U;
  const std::vector<int64_t> inner(binding.brcAxes.begin(), binding.brcAxes.begin() + split);
  const std::vector<int64_t> outer(binding.brcAxes.begin() + split, binding.brcAxes.end());
  AscGraphBuilder builder(name);
  builder.Loops(binding.outputShape)
      .Data("input0", 0, binding.inputShape, {}, dtype)
      .Load("load0", "input0", binding.inputShape, binding.loadStrides);
  return builder.Broadcast("brc0", "load0", inner)
      .Broadcast("brc1", "brc0", outer)
      .Store("store", "brc1")
      .Output("output", "store", 0, dtype)
      .Build();
}

// T5 菱形：`brc` 的结果同时喂两个算子，所以它在第二个消费者结束前一直活着。写法与
// `load_add_diamond_store/Case.cpp` 同形，只是分叉点换成广播。
af::AscGraph BuildBroadcastDiamondGraph(const std::string &name, const std::vector<int64_t> &out,
                                        const std::vector<int64_t> &in, af::DataType dtype) {
  using af::testing::AscGraphBuilder;
  const BroadcastBinding binding = BindBroadcast(out, in);
  AscGraphBuilder builder(name);
  builder.Loops(binding.outputShape)
      .Data("input0", 0, binding.inputShape, {}, dtype)
      .Load("load0", "input0", binding.inputShape, binding.loadStrides);
  return builder.Broadcast("brc", "load0", binding.brcAxes)
      .Abs("abs", "brc")
      .Neg("neg", "brc")
      .Add("add", "abs", "neg")
      .Store("store", "add")
      .Output("output", "store", 0, dtype)
      .Build();
}

// T6 `load -> brc -> cast -> store`：dtype 在图中途变，于是「32B 一档装多少个元素」在同一
// 张图里有两个值。目标 dtype 取与源不同的那一个。
af::AscGraph BuildBroadcastThenCastGraph(const std::string &name, const std::vector<int64_t> &out,
                                         const std::vector<int64_t> &in, af::DataType dtype) {
  using af::testing::AscGraphBuilder;
  const BroadcastBinding binding = BindBroadcast(out, in);
  // 目标 dtype 取与源不同、且对表里真有这一对的那一个。原先是 `f16 ? f32 : f16`，
  // 那是只有 f16 / f32 两种 dtype 时写的；bf16 进来会被折到 f16 上，而 `bf16 -> f16`
  // 没有任何一层供得起——不是对表漏了一行，是硬件没有这条转换
  // （`kernel_operator_vec_vconv_impl.h` 把合法组合列死：`src: bfloat16_t, dst: float /
  // int32_t`），legacy 两张 `kCastTypePairs` 因此都没有这一对。于是这一格问的不是
  // 「dtype 中途变」而是一个不存在的转换，答案只能是 ODS verifier 拒收 = 硬失败
  // （2026-10-26 两侧各 5 格）。
  //
  // `f32` 是四个非 f32 源（f16 / i32 / ui8 / bf16）**共同**有的目标，两侧对表都收，
  // 所以规则写成「不是 f32 就转 f32，是 f32 就转 f16」，而不是逐 dtype 列一张表。
  const af::DataType target = dtype == af::DT_FLOAT ? af::DT_FLOAT16 : af::DT_FLOAT;
  AscGraphBuilder builder(name);
  builder.Loops(binding.outputShape)
      .Data("input0", 0, binding.inputShape, {}, dtype)
      .Load("load0", "input0", binding.inputShape, binding.loadStrides);
  return builder.Broadcast("brc", "load0", binding.brcAxes)
      .Cast("cast", "brc", target)
      .Store("store", "cast")
      .Output("output", "store", 0, target)
      .Build();
}

// T3c 对照：与 T3 同形，但两个 operand 都是整形状的 Load——不含广播。
// T3 的选定格（`[8,1] -> [8,128]`）当前是 ok；这格继续把「两输入二元算子本身」
// 与「其中一个输入是广播轴 stride 为 0 的 Load」分开，防止二元算子成为新的能力边界。
af::AscGraph BuildBinaryControlGraph(const std::string &name, const std::vector<int64_t> &out,
                                     const std::vector<int64_t> &in, af::DataType dtype) {
  (void)in;
  using af::testing::AscGraphBuilder;
  const std::vector<af::Expression> shape = ToExpressions(out);
  AscGraphBuilder builder(name);
  builder.Loops(shape)
      .Data("input0", 0, shape, {}, dtype)
      .Data("input1", 1, shape, {}, dtype)
      .Load("load0", "input0", shape, {})
      .Load("load1", "input1", shape, {});
  af::AscGraph graph =
      builder.Add("add", "load0", "load1").Store("store", "add").Output("output", "store", 0, dtype).Build();
  graph.SetTilingKey(0);
  return graph;
}

// T5c 对照：菱形，但分叉点是普通 elementwise 而不是广播——不含广播。
// T5 当时十格全是 `InvalidBinding: conflicting descriptors for one canonical resource`，
// 而 `load_add_diamond_store` 那条既有 E2E（分叉点是 `Add`）是绿的。这一格把「菱形本身」
// 与「分叉点是广播」分开，答案是前者无关、判据在「分叉点是不是单输入单输出」。
//
// 那一条已经修掉（`alignment_bytes` 不再是资源身份字段，见 `canonicalResourceFacts`），
// T5c / T5e 现在十格全过，E2E 是 `load_abs_diamond_store`。这是 mem 侧的事实：
// `--reg-base` 下这四个菱形拓扑共 40 格全部被 `VectorFunc` 拦在 bridge
// （`vf=a/a`，三个算子够 `kMinVfNodesNum`），所以 reg 侧今天量不到菱形自己的判据。
// 这三格留着不是为了再修一次，
// 是因为「binary 分叉过、unary 分叉挂」这个对照本身是判据的证据，去掉之后下一次漂移就
// 没有参照了。T5 则换了一条原因继续挂（非队列资源一组多于一个成员），今天报能力不足。
af::AscGraph BuildDiamondControlGraph(const std::string &name, const std::vector<int64_t> &out,
                                      const std::vector<int64_t> &in, af::DataType dtype) {
  (void)in;
  using af::testing::AscGraphBuilder;
  const std::vector<af::Expression> shape = ToExpressions(out);
  AscGraphBuilder builder(name);
  builder.Loops(shape).Data("input0", 0, shape, {}, dtype).Load("load0", "input0", shape, {});
  af::AscGraph graph = builder.Abs("fork", "load0")
                           .Abs("abs", "fork")
                           .Neg("neg", "fork")
                           .Add("add", "abs", "neg")
                           .Store("store", "add")
                           .Output("output", "store", 0, dtype)
                           .Build();
  graph.SetTilingKey(0);
  return graph;
}

// T5d 对照：与既有 E2E `load_add_diamond_store/Case.cpp` 逐字同形（四个 Load、分叉点是
// `Add`、两个消费者也是 `Add`）。那条 E2E 是绿的，所以这一格应当 ok——它验的是探针复现得出
// 既有用例，不是在量新东西。它要是也挂，说明问题在探针建图而不在编译器。
af::AscGraph BuildDiamondBinaryForkGraph(const std::string &name, const std::vector<int64_t> &out,
                                         const std::vector<int64_t> &in, af::DataType dtype) {
  (void)in;
  using af::testing::AscGraphBuilder;
  const std::vector<af::Expression> shape = ToExpressions(out);
  AscGraphBuilder builder(name);
  builder.Loops(shape)
      .Data("input0", 0, shape, {}, dtype)
      .Data("input1", 1, shape, {}, dtype)
      .Data("input2", 2, shape, {}, dtype)
      .Data("input3", 3, shape, {}, dtype)
      .Load("load0", "input0", shape, {})
      .Load("load1", "input1", shape, {})
      .Load("load2", "input2", shape, {})
      .Load("load3", "input3", shape, {});
  af::AscGraph graph = builder.Add("add0", "load0", "load1")
                           .Add("add1", "add0", "load2")
                           .Add("add2", "add0", "load3")
                           .Add("add3", "add1", "add2")
                           .Store("store", "add3")
                           .Output("output", "store", 0, dtype)
                           .Build();
  graph.SetTilingKey(0);
  return graph;
}

// T5e 对照：分叉点是 unary，但两个消费者是 binary（各带一个额外 Load）。
// 与 T5c（分叉点 unary、消费者也 unary）和 T5d（两头都是 binary）三格一比，就把
// 「分叉点是什么」与「消费者是什么」分开了——一格只差一件事。
af::AscGraph BuildDiamondUnaryForkGraph(const std::string &name, const std::vector<int64_t> &out,
                                        const std::vector<int64_t> &in, af::DataType dtype) {
  (void)in;
  using af::testing::AscGraphBuilder;
  const std::vector<af::Expression> shape = ToExpressions(out);
  AscGraphBuilder builder(name);
  builder.Loops(shape)
      .Data("input0", 0, shape, {}, dtype)
      .Data("input1", 1, shape, {}, dtype)
      .Data("input2", 2, shape, {}, dtype)
      .Load("load0", "input0", shape, {})
      .Load("load1", "input1", shape, {})
      .Load("load2", "input2", shape, {});
  af::AscGraph graph = builder.Abs("fork", "load0")
                           .Add("add1", "fork", "load1")
                           .Add("add2", "fork", "load2")
                           .Add("add3", "add1", "add2")
                           .Store("store", "add3")
                           .Output("output", "store", 0, dtype)
                           .Build();
  graph.SetTilingKey(0);
  return graph;
}

// T7 多输出：同一个 `brc` 的结果经两条 Store 写到两个输出。
af::AscGraph BuildBroadcastMultiOutputGraph(const std::string &name, const std::vector<int64_t> &out,
                                            const std::vector<int64_t> &in, af::DataType dtype) {
  using af::testing::AscGraphBuilder;
  const BroadcastBinding binding = BindBroadcast(out, in);
  AscGraphBuilder builder(name);
  builder.Loops(binding.outputShape)
      .Data("input0", 0, binding.inputShape, {}, dtype)
      .Load("load0", "input0", binding.inputShape, binding.loadStrides);
  return builder.Broadcast("brc", "load0", binding.brcAxes)
      .Store("store0", "brc")
      .Output("output0", "store0", 0, dtype)
      .Store("store1", "brc")
      .Output("output1", "store1", 1, dtype)
      .Build();
}

// 量哪些形状。挑的不是随手挑的，每一组各管一类：
//   `[128]` / `[1024]`        —— 既有用例的形状，对照用（应当全 ok）
//   `[8,128]` / `[128,8]`     —— 广播轴在外 / 在内，两种段方向
//   `[33,5]` / `[2,3,5,7]`    —— 尾轴 ×4B 不是 32 的倍数（20B / 28B），带填充的 descriptor
//   `[4,8,4,8]`              —— 四轴，广播段之间可能相邻
//   `[128,128,128]`          —— 元素数够大，强制一个块里多趟（小形状对多趟类问题全盲）
const std::vector<std::vector<int64_t>> &ProbeOutputShapes() {
  static const std::vector<std::vector<int64_t>> shapes = {
      {128},
      {1024},
      {8, 128},
      {128, 8},
      {33, 5},
      {4, 8, 4, 8},
      {2, 3, 5, 7},
      {128, 128, 128},
      // 2026-10-11 扩展：尾轴字节数跨过 32 的几档（dtype 一起扫，所以同一个形状在 f16 / ui8
      // 上尾轴字节数不同），以及一个秩 5。
      //   `[7,8]`      尾 32B（f32 刚好一档）
      //   `[64,4]`     尾 16B（f32 不足一档）
      //   `[8,32]`     尾 128B
      //   `[3,5,32]`   中间维不对齐、尾维对齐
      //   `[2,3,5,7,11]` 秩 5，尾 44B
      {7, 8},
      {64, 4},
      {8, 32},
      {3, 5, 32},
      {2, 3, 5, 7, 11},
  };
  return shapes;
}

// 扫哪几种 dtype。不是为了凑数：32B 一档装多少个元素由它定，而「目的地在窗口里稠密」
// 那条判据查的正是 `Align(Z, 32/sizeof(T))` 与 `Z` 相不相等——同一个形状换 dtype 会跨过那条
// 边界。`ui8` 另有一层：分段两档收它，而常量源那一档的集合里没有它
// （`kDuplicateElementTypes`），所以整张都是广播轴的那一格应当在 ui8 上被拒。
struct ProbeDataType {
  const char *name;
  af::DataType dtype;
  int64_t bytes;
};

const std::vector<ProbeDataType> &ProbeDataTypes() {
  static const std::vector<ProbeDataType> types = {
      {"f32", af::DT_FLOAT, 4},
      {"f16", af::DT_FLOAT16, 2},
      {"i32", af::DT_INT32, 4},
      {"ui8", af::DT_UINT8, 1},
      // bf16（2026-10-20 加）。2026-10-22 起 bridge 的暂缓名单空了，这一列量的是真实状态
      // ——在那之前它整列都是「不在 AFIR 支持表里」，与这个测量工具自己无关。
      //
      // 门禁关着时量过一次开闸后的样子：95 格里 17 格 ok、26 格能力不足、18 格硬失败
      // （cast 对表 5、`DescriptorCountMismatch` 5、无诊断 8）。那组数不含接管表上
      // 那 12 格 store 两端 dtype 不一致（首因是 legacy 用例样板漏设 `Output` 的 dtype，
      // 已补，见 `ScheduleToAfir.cpp` 的 `AFIR_ELEMENT_TYPE_HELD_BACK` 那段）——两张表的
      // 行不是同一批图，别把两组数对齐着读。
      {"bf16", af::DT_BF16, 2},
  };
  return types;
}

void PrintTable(const std::vector<std::string> &rows, const std::map<std::string, int> &buckets) {
  std::printf("\n%-*s %-*s %s\n", kGraphColumnWidth, "graph", kStatusColumnWidth, "status", "first diagnostic");
  std::printf("%s\n", std::string(kTableSeparatorWidth, '-').c_str());
  for (const std::string &row : rows) {
    std::printf("%s\n", row.c_str());
  }
  std::printf("\n%-*s %s\n", kBucketColumnWidth, "bucket", "count");
  std::printf("%s\n", std::string(kBucketSeparatorWidth, '-').c_str());
  for (const auto &[bucket, count] : buckets) {
    std::printf("%-*s %d\n", kBucketColumnWidth, bucket.c_str(), count);
  }
  std::printf("\n");
}

// 广播：每个 dtype × 每个输出形状 × 每个非空的广播轴子集。
//
// 行尾多打两列，是为了按签名分簇而不是按形状猜（2026-10-11 之前那张表就是按形状读出来的，
// 结果把一簇归错了判据）：
//
//   `tail=<N>B`  尾轴字节数，`Align` 那条判据跨不跨边界全看它；
//   `inner=brc`  最内那根轴是不是广播轴——实测这一位把被拒的那些分成了两簇，而且它与
//                「尾轴对不对齐」是两件独立的事。
void ProbeBroadcast() {
  std::vector<std::string> rows;
  std::map<std::string, int> buckets;
  int index = 0;
  for (const ProbeDataType &type : ProbeDataTypes()) {
    for (const std::vector<int64_t> &out : ProbeOutputShapes()) {
      const size_t rank = out.size();
      for (uint32_t mask = 1; mask < (1U << rank); ++mask) {
        std::vector<int64_t> in = out;
        for (size_t axis = 0; axis < rank; ++axis) {
          if ((mask & (1U << axis)) != 0) {
            in[axis] = 1;
          }
        }
        if (in == out) {
          continue;  // 该维本来就是 1，构不出广播轴
        }
        const std::string name = "probe_brc_" + std::to_string(index++);
        af::AscGraph graph = BuildBroadcastGraph(name, out, in, type.dtype);
        const ProbeOutcome outcome = RunOneGraph(graph);
        const int64_t tailBytes = out.back() * type.bytes;
        const bool innerIsBroadcast = in.back() == 1 && out.back() != 1;
        std::ostringstream row;
        row << std::left << std::setw(kDtypeColumnWidth) << type.name << " " << std::setw(kBroadcastGraphColumnWidth)
            << (ShapeText(in) + " -> " + ShapeText(out)) << " " << std::setw(kStatusColumnWidth) << outcome.status
            << " tail=" << std::setw(kTailBytesColumnWidth) << static_cast<long>(tailBytes) << " "
            << std::setw(kBroadcastColumnWidth) << (innerIsBroadcast ? "inner=brc" : "inner=normal")
            << " vf=" << std::setw(kVectorFuncColumnWidth) << VfText(outcome) << " " << outcome.diagnostic;
        rows.emplace_back(row.str());
        buckets[outcome.status == "ok" ? "ok" : Bucket(outcome.diagnostic)] += 1;
      }
    }
  }
  PrintTable(rows, buckets);
}

// 对照组：同形状的 elementwise。
void ProbeElementwise() {
  std::vector<std::string> rows;
  std::map<std::string, int> buckets;
  int index = 0;
  for (const std::vector<int64_t> &out : ProbeOutputShapes()) {
    const std::string name = "probe_ew_" + std::to_string(index++);
    af::AscGraph graph = BuildElementwiseGraph(name, out, af::DT_FLOAT);
    const ProbeOutcome outcome = RunOneGraph(graph);
    std::ostringstream row;
    row << std::left << std::setw(kGraphColumnWidth) << ShapeText(out) << " " << std::setw(kStatusColumnWidth)
        << outcome.status << " vf=" << std::setw(kVectorFuncColumnWidth) << VfText(outcome) << " "
        << outcome.diagnostic;
    rows.emplace_back(row.str());
    buckets[outcome.status == "ok" ? "ok" : Bucket(outcome.diagnostic)] += 1;
  }
  PrintTable(rows, buckets);
}

//===----------------------------------------------------------------------===//
// 拓扑切片（F8）
//===----------------------------------------------------------------------===//

// 拓扑一条。`needsTwoBroadcastAxes` 的那一种（T4）在只有一个广播轴的形状上造不出来，
// 跳过而不是硬造——硬造出来的是另一种图，混在同一张表里就不可比了。
struct ProbeTopology {
  const char *name;
  af::AscGraph (*build)(const std::string &, const std::vector<int64_t> &, const std::vector<int64_t> &, af::DataType);
  bool needsTwoBroadcastAxes;
  const char *what;  // 它比 T1 多测的那一件事
};

// T0 的构图函数签名要对上，`in` 用不到（对照组不含广播）。
af::AscGraph BuildTopologyControlGraph(const std::string &name, const std::vector<int64_t> &out,
                                       const std::vector<int64_t> &in, af::DataType dtype) {
  (void)in;
  return BuildElementwiseGraph(name, out, dtype);
}

const std::vector<ProbeTopology> &ProbeTopologies() {
  static const std::vector<ProbeTopology> topologies = {
      {"T0", BuildTopologyControlGraph, false, "对照：同形状不含广播"},
      {"T1", BuildBroadcastGraph, false, "基准：广播直接写回 GM"},
      {"T2", BuildBroadcastThenComputeGraph, false, "广播结果进计算（tmp 争用 + PIPE_V 同步）"},
      {"T3", BuildBroadcastOperandGraph, false, "广播作二元算子的一个 operand"},
      {"T3c", BuildBinaryControlGraph, false, "对照：二元算子两输入都不含广播"},
      {"T4", BuildChainedBroadcastGraph, true, "两步广播，第二个的源在片上"},
      {"T5", BuildBroadcastDiamondGraph, false, "一个广播两个消费者"},
      {"T5c", BuildDiamondControlGraph, false, "对照：菱形，分叉点 unary、消费者 unary"},
      {"T5d", BuildDiamondBinaryForkGraph, false, "对照：菱形，两头都是 binary（= 既有 E2E）"},
      {"T5e", BuildDiamondUnaryForkGraph, false, "对照：菱形，分叉点 unary、消费者 binary"},
      {"T6", BuildBroadcastThenCastGraph, false, "dtype 中途变"},
      {"T7", BuildBroadcastMultiOutputGraph, false, "多输出"},
  };
  return topologies;
}

// 拓扑切片用哪几格形状。不与 380 格的形状 sweep 叉乘：那一维量的是形状，这一维量的是
// 拓扑，叉乘出来上万格而判据只有十几条。按「尾轴 32B 对齐 × 最内轴是否广播轴」四象限各取
// 一格（那张签名表的四格），外加一格两个广播轴的给 T4。
struct TopologyCell {
  std::vector<int64_t> out;
  std::vector<int64_t> in;
  const char *what;
};

const std::vector<TopologyCell> &ProbeTopologyCells() {
  static const std::vector<TopologyCell> cells = {
      {{128, 8}, {1, 8}, "对齐 + 最内非广播（f32 下尾 32B）——唯一全通的那一象限"},
      {{8, 128}, {8, 1}, "对齐 + 最内是广播轴（尾 512B）"},
      {{33, 5}, {1, 5}, "不对齐 + 最内非广播（尾 20B）"},
      {{2, 3, 5, 7}, {2, 3, 5, 1}, "不对齐 + 最内是广播轴（尾 28B）"},
      {{4, 4, 8}, {1, 1, 8}, "两个广播轴且相邻（尾 32B）——T4 只在这一格造得出来"},
  };
  return cells;
}

// 拓扑 × 形状格 × dtype。dtype 只取 f32 / f16：这一维量的是拓扑，而 32B 档位的边界由形状
// sweep 那 380 格管，这里留两档够看出「dtype 换了结论变不变」。
void ProbeTopologySlice() {
  std::vector<std::string> rows;
  std::map<std::string, int> buckets;
  // 留三档：f32 / f16 够看出「dtype 换了结论变不变」，bf16 多一档「只有一侧供得起」。
  const std::vector<ProbeDataType> types = {
      {"f32", af::DT_FLOAT, 4}, {"f16", af::DT_FLOAT16, 2}, {"bf16", af::DT_BF16, 2}};
  int index = 0;
  for (const ProbeDataType &type : types) {
    for (const TopologyCell &cell : ProbeTopologyCells()) {
      const BroadcastBinding binding = BindBroadcast(cell.out, cell.in);
      for (const ProbeTopology &topology : ProbeTopologies()) {
        if (topology.needsTwoBroadcastAxes && binding.brcAxes.size() < 2U) {
          continue;
        }
        const std::string name = "probe_topo_" + std::to_string(index++);
        af::AscGraph graph = topology.build(name, cell.out, cell.in, type.dtype);
        const ProbeOutcome outcome = RunOneGraph(graph);
        const int64_t tailBytes = cell.out.back() * type.bytes;
        const bool innerIsBroadcast = cell.in.back() == 1 && cell.out.back() != 1;
        std::ostringstream row;
        row << std::left << std::setw(kTopologyColumnWidth) << topology.name << " " << std::setw(kDtypeColumnWidth)
            << type.name << " " << std::setw(kTopologyGraphColumnWidth)
            << (ShapeText(cell.in) + " -> " + ShapeText(cell.out)) << " " << std::setw(kStatusColumnWidth)
            << outcome.status << " tail=" << std::setw(kTailBytesColumnWidth) << static_cast<long>(tailBytes) << " "
            << std::setw(kBroadcastColumnWidth) << (innerIsBroadcast ? "inner=brc" : "inner=normal")
            << " vf=" << std::setw(kVectorFuncColumnWidth) << VfText(outcome) << " " << outcome.diagnostic;
        rows.emplace_back(row.str());
        // 桶按「拓扑 + 判决」分：这一维要答的是「换了拓扑结论变不变」，而不是「哪道门禁拦
        // 下的」（那个由形状 sweep 的桶答）。
        buckets[std::string(topology.name) + ": " + (outcome.status == "ok" ? "ok" : Bucket(outcome.diagnostic))] += 1;
      }
    }
  }
  std::printf("\n");
  for (const ProbeTopology &topology : ProbeTopologies()) {
    std::printf("%-*s %s\n", kTopologyColumnWidth, topology.name, topology.what);
  }
  for (const TopologyCell &cell : ProbeTopologyCells()) {
    std::printf("%-*s %s\n", kTopologyGraphColumnWidth, (ShapeText(cell.in) + " -> " + ShapeText(cell.out)).c_str(),
                cell.what);
  }
  PrintTable(rows, buckets);
}

}  // namespace

int main(int argc, char **argv) {
  // `--topo` 只跑拓扑切片。形状 sweep 那 380 格一侧要 90 秒，而拓扑这一维改动时要反复跑，
  // 两者分开省的是迭代时间，不是别的。
  bool topologyOnly = false;
  for (int index = 1; index < argc; ++index) {
    const std::string argument(argv[index]);
    if (argument == "--reg-base") {
      SelectTarget(kRegBased);
    } else if (argument == "--topo") {
      topologyOnly = true;
    } else {
      std::printf("unknown argument '%s'; expected --reg-base and/or --topo\n", argument.c_str());
      return 2;
    }
  }
  const auto options = mlir::bridge::ReadMlirCodegenOptions();
  if (options.config != mlir::bridge::MlirCodegenConfig::kMlir) {
    std::printf("set AUTOFUSE_DFX_FLAGS=--%s=mlir to run the probe; MLIR codegen is opt-in\n",
                mlir::bridge::kMlirCodegenConfigOptionName.data());
    return 2;
  }
  // 平台由 runtime stub 懒初始化（与 CPU E2E 生成器同一条路），这里不手动填：
  // `PlatformInfo` 的字段比看起来多（`padding_size` 等），只填四个会让调度拿到半张表。
  if (!topologyOnly) {
    std::printf("=== broadcast (%s) ===\n", g_target->soc_version);
    ProbeBroadcast();
    std::printf("=== elementwise control (%s) ===\n", g_target->soc_version);
    ProbeElementwise();
  }
  std::printf("=== cascade topologies (%s) ===\n", g_target->soc_version);
  ProbeTopologySlice();
  return 0;
}
