/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sk_scope_split.h
 * \brief Super Kernel Scope Splitter - splits multi-stream graph into scopes
 *
 * This module implements the scope splitting algorithm for multi-stream graphs.
 * A "scope" represents a group of nodes that can be fused into a single Super Kernel.
 *
 * Architecture:
 * The scope splitting is implemented as a multi-pass pipeline, similar to compiler passes:
 * - Pass 0: Initial scope splitting based on fusibility and scopeBitFlags
 * - Pass 1: ScheMode core compatibility refinement
 * - Pass 2: Progressive deadlock detection
 * - Pass 3: Remove event-only streams from scopes
 * - Finalization: Calculate final SK core info after scope boundaries are stable
 *
 * Key Concepts:
 * - Scope: A collection of nodes from potentially multiple streams that can be fused together
 * - ScopeBitFlags: Bit flags that identify which scope a node belongs to
 * - Deadlock Detection: Prevents circular dependencies between streams
 */

#ifndef __SK_SCOPE_SPLIT_H__
#define __SK_SCOPE_SPLIT_H__

#include <optional>
#include <queue>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "sk_candidate_heap.h"
#include "sk_graph.h"
#include "sk_log.h"
#include "sk_lock_detector.h"
#include "sk_options_manager.h"
#include "sk_scope_info.h"
#include "sk_scope_split_result_reporter.h"

/*!
 * \struct StreamState
 * \brief Processing state for a single stream during multi-stream splitting
 */
struct StreamState {
  uint64_t currentNodeId;     ///< Current node being processed
  bool isSuspended;           ///< True if waiting for Notify event
  uint64_t waitingForNotify;  ///< Event ID being waited for (if suspended)
  bool isTerminated;          ///< True if stream cannot continue in current scope

  StreamState()
      : currentNodeId(INVALID_TASK_ID), isSuspended(false), waitingForNotify(INVALID_TASK_ID), isTerminated(false) {}

  /*!
   * \brief Format stream state information for logging
   * \return Formatted string describing the stream state
   */
  std::string FormatStreamStateInfo() const {
    std::string info = "currentNodeId=";
    if (currentNodeId == INVALID_TASK_ID) {
      info += "INVALID";
    } else {
      info += std::to_string(currentNodeId);
    }

    info += ", isSuspended=";
    info += (isSuspended ? "true" : "false");

    info += ", waitingForNotify=";
    if (waitingForNotify == INVALID_TASK_ID) {
      info += "INVALID";
    } else {
      info += "0x" + std::to_string(waitingForNotify);
    }

    info += ", isTerminated=";
    info += (isTerminated ? "true" : "false");

    return info;
  }
};

// Forward declaration
class SuperKernelScopeSplitter;

class ScopeCoreInfoCalculator {
 public:
  ScopeCoreInfoCalculator(int64_t maxDeviceCubeNum, int64_t maxDeviceVectorNum)
      : maxDeviceCubeNum_(maxDeviceCubeNum), maxDeviceVectorNum_(maxDeviceVectorNum) {}

  // Use this overload when only scope validity is needed.
  bool GetValidScopeCoreInfo(const std::vector<SuperKernelBaseNode *> &nodes) const;

  // On failure, scopeCoreInfo contains the preferred attempted shape for mismatch diagnostics.
  bool GetValidScopeCoreInfo(const std::vector<SuperKernelBaseNode *> &nodes, ScopeCoreInfo &scopeCoreInfo) const;

  // On failure, scopeCoreInfo contains the attempted shape and the accumulated constraints remain unchanged.
  bool UpdateScopeCoreInfo(const SuperKernelBaseNode &node, ScopeCoreInfo &scopeCoreInfo);
  void Reset();

 private:
  struct ScopeCoreConstraints {
    uint32_t minCubeNum = 0;
    uint32_t minVectorNum = 0;
    std::optional<uint32_t> exactCubeNum;
    std::optional<uint32_t> exactVectorNum;
    bool requireMix12 = false;
  };

  static bool UpdateScopeCoreConstraints(const SuperKernelBaseNode &node, ScopeCoreConstraints &constraints);
  bool GetScopeCoreCandidate(const ScopeCoreConstraints &constraints, ScopeCoreInfo &scopeCoreInfo) const;

  int64_t maxDeviceCubeNum_ = 0;
  int64_t maxDeviceVectorNum_ = 0;
  ScopeCoreConstraints constraints_;
};

// ============ Pass Base Class ============

/*!
 * \class ScopeSplitPass
 * \brief Base class for scope splitting passes
 */
class ScopeSplitPass {
 public:
  explicit ScopeSplitPass(SuperKernelGraph &inputGraph)
      : graph_(inputGraph), splitter_(nullptr), scopeSplitResultReporter_(nullptr) {}
  virtual ~ScopeSplitPass() = default;

  /*!
   * \brief Execute the pass on the given scopes
   * \param scopes Input/output scope list
   * \return true on success
   */
  virtual bool Run(std::vector<SuperKernelScopeInfo> &scopes) = 0;

  virtual std::string GetName() const = 0;

  /*!
   * \brief Set the splitter reference for re-split requests
   * \param splitter Reference to the splitter
   */
  void SetSplitter(SuperKernelScopeSplitter *splitter) {
    splitter_ = splitter;
  }

  void SetScopeSplitResultReporter(ScopeSplitResultReporter *scopeSplitResultReporter) {
    scopeSplitResultReporter_ = scopeSplitResultReporter;
  }

  /*!
   * \brief Request a re-split of the graph
   * Called by passes when they modify node fusibility and need the pipeline to re-run
   */
  void RequestResplit();

  // ============ Debug/Logging Utilities (public static for reuse) ============

  /*!
   * \brief Print scope splitting results for debugging
   * \param scopes Scope list to print
   * \param graph Graph for scope name lookup
   */
  static void PrintScopeResults(const std::vector<SuperKernelScopeInfo> &scopes, const SuperKernelGraph &graph,
                                const char *passName);

  /*!
   * \brief Get scope names string from scopeBitFlags
   * \param scopeBitFlags Bit flags to convert
   * \param graph Graph for scope name lookup
   * \return Comma-separated scope names string
   */
  static std::string GetScopeNamesFromBitFlags(const std::bitset<MAX_SCOPE_NUM> &scopeBitFlags,
                                               const SuperKernelGraph &graph);

  /*!
   * \brief Print nodes in a scope
   * \param scopeIdx Scope index
   * \param scope Scope info
   */
  static void PrintScopeNodes(size_t scopeIdx, const SuperKernelScopeInfo &scope);

  /*!
   * \brief Print stream infos in a scope
   * \param scopeIdx Scope index
   * \param scope Scope info
   */
  static void PrintScopeStreamInfos(size_t scopeIdx, const SuperKernelScopeInfo &scope);

  /*!
   * \brief Rebuild stream infos for a scope
   * \param scope Scope to rebuild
   */
  static void RebuildStreamInfos(SuperKernelScopeInfo &scope);

  /*!
   * \brief Get kernel nodes from a scope
   * \param scope Scope to extract kernel nodes from
   * \return Vector of kernel node IDs
   */
  static std::vector<uint64_t> GetKernelNodeIds(const SuperKernelScopeInfo &scope);

 private:
  /*!
   * \brief Print detailed scope information to current log context
   * \param scopes Scope list to print
   * \param graph Graph for scope name lookup
   */
  static void PrintScopeDetails(const std::vector<SuperKernelScopeInfo> &scopes, const SuperKernelGraph &graph,
                                const char *passName);

 protected:
  SuperKernelGraph &graph_;
  SuperKernelScopeSplitter *splitter_;  ///< Reference to splitter for re-split requests
  ScopeSplitResultReporter *scopeSplitResultReporter_;
};

// ============ Pass 3: Event-Only Stream Remove (after SK core calculation) ============

/*!
 * \class EventOnlyStreamRemovePass
 * \brief Pass 3: Mark event-only stream nodes as non-fusible to trigger re-split
 *
 * This pass runs after ScheModeKernelSplitPass to detect scopes that contain streams
 * with only event nodes (NODE_NOTIFY, NODE_WAIT, NODE_RESET, NODE_MEMORY_WRITE,
 * NODE_MEMORY_WAIT). Instead of removing such streams directly, it marks all event
 * nodes in those streams as non-fusible and signals the pipeline to re-run scope
 * splitting from the beginning.
 *
 * This approach prevents potential deadlocks that could occur if nodes are removed
 * from scopes after deadlock detection has already been performed.
 *
 * Processing logic:
 * - For each scope, collect nodes by stream
 * - If a stream's nodes are ALL event nodes, mark them as non-fusible
 * - Clear scopes to signal re-split is needed
 * - The pipeline will re-run from Pass 0 with updated fusibility
 */
class EventOnlyStreamRemovePass : public ScopeSplitPass {
 public:
  explicit EventOnlyStreamRemovePass(SuperKernelGraph &inputGraph);
  ~EventOnlyStreamRemovePass() = default;

  bool Run(std::vector<SuperKernelScopeInfo> &scopes) override;
  std::string GetName() const override {
    return "EventOnlyStreamRemovePass";
  }

 private:
  /*!
   * \brief Check if a node is an event node type
   * \param node Node to check
   * \return true if the node is an event node type
   */
  bool IsEventNode(SuperKernelBaseNode *node) const;

  /*!
   * \brief Collect nodes grouped by stream index for a scope
   * \param scope Scope to analyze
   * \param streamNodes Output: map from stream index to list of nodes
   */
  void CollectNodesPerStream(const SuperKernelScopeInfo &scope,
                             std::unordered_map<uint32_t, std::vector<SuperKernelBaseNode *>> &streamNodes);

  /*!
   * \brief Check if all nodes in a vector are event nodes
   * \param nodes Vector of nodes to check
   * \return true if all nodes are event nodes (and vector is non-empty)
   */
  bool IsStreamAllEventNodes(const std::vector<SuperKernelBaseNode *> &nodes) const;

  /*!
   * \brief Process a single scope and mark event-only stream nodes as non-fusible
   * \param scope Scope to process
   * \return Number of nodes marked as non-fusible
   */
  uint32_t ProcessScope(SuperKernelScopeInfo &scope);

  uint32_t markedCount_;  ///< Total number of nodes marked as non-fusible
};

// ============ Pass: Per-Op Max Core Debug Split ============

class PerOpMaxCoreSplitPass : public ScopeSplitPass {
 public:
  explicit PerOpMaxCoreSplitPass(SuperKernelGraph &inputGraph);
  ~PerOpMaxCoreSplitPass() = default;

  bool Run(std::vector<SuperKernelScopeInfo> &scopes) override;
  std::string GetName() const override {
    return "PerOpMaxCoreSplitPass";
  }
};

// ============ Pass 0: Initial Scope Split ============

/*!
 * \class InitialScopeSplitPass
 * \brief Pass 0: Split graph into initial scopes based on fusibility and scopeBitFlags
 *
 * This pass performs coarse-grained scope splitting:
 * - Groups nodes by scopeBitFlags
 * - Respects fusibility constraints
 * - Handles Wait/Notify synchronization
 * - Does NOT perform deadlock detection
 */
class InitialScopeSplitPass : public ScopeSplitPass {
 public:
  explicit InitialScopeSplitPass(SuperKernelGraph &inputGraph, SkHeapType heapType);
  ~InitialScopeSplitPass() = default;

  bool Run(std::vector<SuperKernelScopeInfo> &scopes) override;
  std::string GetName() const override {
    return "InitialScopeSplitPass";
  }

 private:
  // ============ Stream State Management ============
  void InitStreamStates();
  bool ResetStreamStates();   // Changed from void to bool to propagate errors
  bool SkipUnfusibleNodes();  // Changed from void to bool to propagate errors
  bool SkipUnfusibleNodesForStream(uint32_t streamIdx);
  bool ProcessUnfusibleNodeForSkip(uint32_t streamIdx, SuperKernelBaseNode *node);
  bool ProcessUnfusibleWaitNode(uint32_t streamIdx, SuperKernelBaseNode *waitNode);
  bool AllStreamsFinished() const;
  bool DetermineCurrentScopeBitFlags();

  // ============ Node Processing ============
  void InitNodeHeap();
  void TryAddNodeToHeap(uint32_t streamIdx);
  void HandleWaitNode(SuperKernelBaseNode *waitNode, uint32_t streamIdx);
  void ProcessNotifyNode(SuperKernelBaseNode *notifyNode);
  void ProcessResetNode(SuperKernelBaseNode *resetNode);
  bool HandleUnfusibleNotifyNode(SuperKernelBaseNode *notifyNode, uint32_t streamIdx);
  bool ResumeSuspendedWaitStreams(SuperKernelBaseNode *notifyNode, uint32_t notifyStreamIdx);

  // ============ Scope Building ============
  void AddStreamInfoToScope(SuperKernelScopeInfo &scopeInfo, SuperKernelBaseNode *node);
  bool BuildCurrentScope(SuperKernelScopeInfo &scopeInfo);

  // ============ Diagnostic Logging ============
  void LogFusibleNodeSearchResult();

  // ============ Member Variables ============
  std::unordered_map<uint32_t, StreamState> streamStates_;
  std::set<uint64_t> visitedNotifies_;
  std::set<uint64_t> processedNodes_;
  SkCandidateHeap nodeHeap_;
  std::bitset<MAX_SCOPE_NUM> currentScopeBitFlags_;
  std::unordered_map<uint32_t, ScopeBreakInfo> streamBreakInfos_;  // Per-stream break info (last one wins)
  ScopeBreakInfo currentScopeBreakInfo_;                           // Current scope's primary break reason
  ScopeBreakInfo scopeStartBreakInfo_;                             // Break reason from skipped nodes at scope start
  SuperKernelScopeInfo *currentScope_ = nullptr;                   // Pointer to scope currently being built
};

// ============ Pass 2: Deadlock Detection and Refinement ============

/*!
 * \enum ScopeProcessResult
 * \brief Result of processing a single scope for deadlock detection
 */
enum class ScopeProcessResult {
  NO_DEADLOCK,          ///< No deadlock found, scope is safe and added to outputScopes
  DEADLOCK_RESOLVED,    ///< Deadlock found and successfully split, remaining part returned via pendingScope
  DEADLOCK_UNRESOLVED,  ///< Deadlock found but cannot split (no Wait node)
};

/*!
 * \class DeadlockRefinePass
 * \brief Pass 2: Detect deadlocks and refine scopes
 *
 * This pass performs fine-grained scope refinement:
 * - Detects deadlocks within each scope
 * - Splits scopes at Wait nodes that would cause deadlocks
 * - The Wait split point is NOT included in either split scope
 */
class DeadlockRefinePass : public ScopeSplitPass {
 public:
  DeadlockRefinePass(SuperKernelGraph &inputGraph, SuperKernelOptionsManager &opts);
  ~DeadlockRefinePass() = default;

  bool Run(std::vector<SuperKernelScopeInfo> &scopes) override;
  std::string GetName() const override {
    return "DeadlockRefinePass";
  }

 protected:
  virtual void RecordDeadlockSplitResult(SuperKernelBaseNode *deadlockNode, SuperKernelBaseNode *deadlockWaitNode) {
    (void)deadlockNode;
    (void)deadlockWaitNode;
  }

 private:
  /*!
   * \brief Find deadlock point in a scope
   * \param scope Scope to check
   * \param deadlockNode Output: node that causes deadlock (nullptr if no deadlock)
   * \param deadlockWaitNode Output: nearest Wait node before deadlock (nullptr if no deadlock)
   * \param checkedScopeCoreInfo Output: core info used by deadlock detection for the checked scope prefix
   * \return true if a deadlock is found
   */
  bool FindDeadlockInScope(const SuperKernelScopeInfo &scope, SuperKernelBaseNode **deadlockNode,
                           SuperKernelBaseNode **deadlockWaitNode, ScopeCoreInfo *checkedScopeCoreInfo);

  /*!
   * \brief Split a scope at a Wait node
   * \param scope Original scope
   * \param waitNode Wait node to split at (not included in either result)
   * \param scopeBefore Output: scope before Wait node
   * \param scopeAfter Output: scope after Wait node
   */
  void SplitScopeAtWaitNode(const SuperKernelScopeInfo &scope, SuperKernelBaseNode *waitNode,
                            SuperKernelScopeInfo &scopeBefore, SuperKernelScopeInfo &scopeAfter);

  /*!
   * \brief Process a single scope with deadlock detection and one-step splitting.
   *
   * This function either:
   * - Detects no deadlock and moves the whole scope into outputScopes; or
   * - Detects a deadlock and splits at the nearest preceding Wait node, moving
   *   the front part into outputScopes and returning the remaining part via
   *   pendingScope; or
   * - Detects an unresolvable deadlock (no suitable Wait node).
   *
   * \param scopeToProcess The scope to process (moved)
   * \param outputScopes Vector to store deadlock-free scopes produced from the front part
   * \param pendingScope Optional: remaining part of the scope that still needs processing
   * \return Result of the processing
   */
  ScopeProcessResult ProcessSingleScope(SuperKernelScopeInfo &&scopeToProcess,
                                        std::vector<SuperKernelScopeInfo> &outputScopes,
                                        std::optional<SuperKernelScopeInfo> &pendingScope);

  /**
   * @brief Handle deadlock split logic
   * @param workingScope scope to split
   * @param deadlockNode node that caused deadlock
   * @param deadlockWaitNode wait node to split at
   * @param scopeBeforeCoreInfo core info used to model the scope before the wait node during deadlock detection
   * @param outputScopes output scopes
   * @param pendingScope pending scope for next iteration
   * @return split result
   */
  ScopeProcessResult HandleDeadlockSplit(SuperKernelScopeInfo &workingScope, SuperKernelBaseNode *deadlockNode,
                                         SuperKernelBaseNode *deadlockWaitNode,
                                         const ScopeCoreInfo &scopeBeforeCoreInfo,
                                         std::vector<SuperKernelScopeInfo> &outputScopes,
                                         std::optional<SuperKernelScopeInfo> &pendingScope);
  LockDetector lockDetector_;
  int64_t maxCubeNum_ = 0;
  int64_t maxVectorNum_ = 0;
};

// ============ Pass 1: ScheMode Kernel Core Split Refinement ============

/*!
 * \enum ScheModeScopeProcessResult
 * \brief Result of processing a single scope for ScheMode kernel split refinement
 */
enum class ScheModeScopeProcessResult {
  NO_SPLIT,       ///< No split required, scope is output as-is
  SPLIT_RESOLVED  ///< Split required and completed, remaining part returned via pendingScope
};

/*!
 * \class ScheModeKernelSplitPass
 * \brief Explore SK type/core candidates and refine incompatible scopes
 *
 * Rules:
 * - Explore legal AIC/AIV/MIX1:1/MIX1:2 core results after each kernel is appended
 * - Non-ScheMode tasks require no more C/V cores than the candidate SK core result
 * - ScheMode vector-only tasks require an exact V match; other ScheMode tasks require an exact C match
 * - Split before the first kernel that makes all launch-shape candidates invalid
 */
class ScheModeKernelSplitPass : public ScopeSplitPass {
 public:
  explicit ScheModeKernelSplitPass(SuperKernelGraph &inputGraph);
  ~ScheModeKernelSplitPass() = default;

  bool Run(std::vector<SuperKernelScopeInfo> &scopes) override;
  std::string GetName() const override {
    return "ScheModeKernelSplitPass";
  }

 protected:
  virtual void RecordScheModeSplitResult(SuperKernelBaseNode *splitNode) {
    (void)splitNode;
  }

 private:
  /*!
   * \brief Process a single scope and split if ScheMode rule requires
   * \param scopeToProcess Input scope to process (moved)
   * \param outputScopes Output refined scopes
   * \param pendingScope Remaining part after split, if any
   * \return Processing result
   */
  ScheModeScopeProcessResult ProcessSingleScope(SuperKernelScopeInfo &&scopeToProcess,
                                                std::vector<SuperKernelScopeInfo> &outputScopes,
                                                std::optional<SuperKernelScopeInfo> &pendingScope);

  /*!
   * \brief Split a scope at the split node (split node belongs to scopeAfter)
   * \param scope Original scope
   * \param splitNode Node where split happens
   * \param scopeBefore Output: nodes before splitNode
   * \param scopeAfter Output: nodes from splitNode to end
   */
  void SplitScopeAtNode(const SuperKernelScopeInfo &scope, SuperKernelBaseNode *splitNode,
                        SuperKernelScopeInfo &scopeBefore, SuperKernelScopeInfo &scopeAfter);

  std::string BuildScheModeCoreMismatchDetail(const ScopeCoreInfo &scopeBeforeCoreInfo,
                                              const ScopeCoreInfo &candidateScopeCoreInfo,
                                              const SuperKernelScopeInfo &scopeBefore,
                                              const SuperKernelBaseNode &triggerNode) const;

  int64_t maxCubeNum_ = 0;
  int64_t maxVectorNum_ = 0;
};

// ============ Default Node Process Pass ============

/*!
 * \struct StreamDefaultInfo
 * \brief Information about kernel and default nodes in a stream
 */
struct StreamDefaultInfo {
  bool hasKernel = false;                       ///< Whether stream has kernel nodes
  std::vector<SuperKernelBaseNode *> defaults;  ///< Default nodes in this stream
};

/*!
 * \class DefaultNodeProcessPass
 * \brief Pass for processing default nodes in scopes
 *
 * Processing logic (each scope independently):
 * 1. One-pass collect: gather kernel and default info per stream
 * 2. Classify defaults:
 *    - In stream with kernel: mark unfusible -> trigger resplit
 *    - In stream without kernel: remove -> execute via original task
 *
 * Optimization: single traversal to collect all stream info
 */
class DefaultNodeProcessPass : public ScopeSplitPass {
 public:
  explicit DefaultNodeProcessPass(SuperKernelGraph &inputGraph);
  ~DefaultNodeProcessPass() = default;

  bool Run(std::vector<SuperKernelScopeInfo> &scopes) override;
  std::string GetName() const override {
    return "DefaultNodeProcessPass";
  }

 private:
  uint32_t ProcessSingleScope(SuperKernelScopeInfo &scope);

  std::unordered_map<uint32_t, StreamDefaultInfo> CollectStreamInfo(const SuperKernelScopeInfo &scope);
  void MarkDefaultsUnfusible(const std::vector<SuperKernelBaseNode *> &defaultNodes);
  void RemoveDefaultsAndStreams(SuperKernelScopeInfo &scope, const std::vector<SuperKernelBaseNode *> &defaultsToRemove,
                                const std::unordered_set<uint32_t> &streamsToRemove);
  void RemoveStreamFromScope(SuperKernelScopeInfo &scope, uint32_t streamIdx);
};

// ============ Main Splitter Class ============

/*!
 * \class SuperKernelScopeSplitter
 * \brief Main class that orchestrates scope splitting passes
 *
 * Usage:
 *   SuperKernelScopeSplitter splitter(graph);
 *   splitter.SplitGraph();
 *   const auto& scopes = splitter.GetScopeInfos();
 */
class SuperKernelScopeSplitter {
 public:
  explicit SuperKernelScopeSplitter(SuperKernelGraph &inputGraph, SuperKernelOptionsManager &opts);
  ~SuperKernelScopeSplitter() = default;
  SuperKernelScopeSplitter(const SuperKernelScopeSplitter &) = delete;
  SuperKernelScopeSplitter &operator=(const SuperKernelScopeSplitter &) = delete;
  SuperKernelScopeSplitter(SuperKernelScopeSplitter &&) = default;
  SuperKernelScopeSplitter &operator=(SuperKernelScopeSplitter &&) = default;

  /*!
   * \brief Split graph into scopes using multi-pass pipeline
   * \return true on success
   */
  bool SplitGraph();

  /*!
   * \brief Get the resulting scope information
   * \return Reference to vector of scope information
   */
  std::vector<SuperKernelScopeInfo> &GetScopeInfos() noexcept {
    return scopeInfos_;
  }

  /*!
   * \brief Request a re-split of the graph
   * Called by passes when they modify node fusibility and need the pipeline to re-run
   */
  void RequestResplit() {
    needResplit_ = true;
  }

  /*!
   * \brief Check if a re-split has been requested
   * \return true if re-split is needed
   */
  bool NeedsResplit() const {
    return needResplit_;
  }

  bool IsDebugPerOpMaxCoreEnabled() const {
    return enablePerOpMaxCore_;
  }

  /*!
   * \brief Print scope break reason report for debugging
   */
  void PrintScopeBreakReasonReport();

  /*!
   * \brief Print all scopes detail
   */
  static void PrintAllScopesDetail(const std::vector<SuperKernelScopeInfo> &scopeInfos);

  SuperKernelGraph &graph_;
  std::vector<SuperKernelScopeInfo> scopeInfos_;
  std::vector<std::unique_ptr<ScopeSplitPass>> passes_;
  ScopeSplitResultReporter splitResultReporter_;
  bool needResplit_ = false;

 private:
  bool FinalizeScopeCoreInfo();
  void InitDefaultNodeFusibility();
  SuperKernelOptionsManager *opts_ = nullptr;
  bool enableTaskBreakerBypass_ = false;
  bool enablePerOpMaxCore_ = false;
};

// Declaration for FindRootBreakInfo
ScopeBreakInfo FindRootBreakInfo(const SuperKernelScopeInfo &scope,
                                 const std::unordered_map<uint16_t, size_t> &scopeIdToIdx,
                                 const std::vector<SuperKernelScopeInfo> &scopeInfos);

#endif  // __SK_SCOPE_SPLIT_H__
