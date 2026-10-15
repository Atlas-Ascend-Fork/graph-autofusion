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
 * \file sk_scope_info.h
 * \brief Super Kernel Scope Info - data structures for scope splitting
 */

#ifndef __SK_SCOPE_INFO_H__
#define __SK_SCOPE_INFO_H__

#include <atomic>
#include <bitset>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include "sk_types.h"
#include "sk_node.h"

struct ScopeCoreInfo {
  SkKernelType type = SkKernelType::DEFAULT;
  uint32_t numBlocks = 0;

  bool IsValid() const {
    return type != SkKernelType::DEFAULT && numBlocks != 0;
  }

  uint32_t GetCubeNum() const {
    switch (type) {
      case SkKernelType::AIC_ONLY:
      case SkKernelType::MIX_AIC_1_0:
      case SkKernelType::MIX_AIC_1_1:
      case SkKernelType::MIX_AIC_1_2:
        return numBlocks;
      default:
        return 0;
    }
  }

  uint32_t GetVectorNum() const {
    switch (type) {
      case SkKernelType::AIV_ONLY:
      case SkKernelType::MIX_AIV_1_0:
      case SkKernelType::MIX_AIC_1_1:
        return numBlocks;
      case SkKernelType::MIX_AIC_1_2:
        return numBlocks * 2U;
      default:
        return 0;
    }
  }
};

/*!
 * \struct OriginalScopeInfo
 * \brief Simple structure to store original scope info (for dumping)
 *        Used to preserve scope information before any splitting
 */
struct OriginalScopeInfo {
  uint16_t scopeId;
  std::bitset<MAX_SCOPE_NUM> scopeBitFlags;
  std::vector<uint64_t> nodeIds;

  const std::bitset<MAX_SCOPE_NUM> &GetScopeBitFlags() const {
    return scopeBitFlags;
  }
};

/*!
 * \struct ScopeStreamInfo
 * \brief Information about a single stream within a scope
 */
struct ScopeStreamInfo {
  uint32_t streamIdx = 0;                  ///< Stream index in the graph
  uint64_t headNodeIdx = INVALID_TASK_ID;  ///< First node of this stream in scope
  uint64_t tailNodeIdx = INVALID_TASK_ID;  ///< Last node of this stream in scope
  uint64_t nodeSize = 0;                   ///< Number of nodes from this stream in scope
};

/*!
 * \enum ScopeProcessStatus
 * \brief Processing result status for scope
 */
enum class ScopeProcessStatus : uint8_t {
  INIT = 0,               // Scope has not been processed
  SUCCESS,                // Scope processing succeeded
  RESOURCE_INSUFFICIENT,  // Insufficient stream task slots or event memory resources
  NO_TARGET_NODE,         // No target node remains after filtering
  UNRECOVERABLE_FAIL,     // Unrecoverable failure that cannot be skipped
};

/*!
 * \brief Convert ScopeProcessStatus to string
 */
inline const char *to_string(ScopeProcessStatus status) {
  switch (status) {
    case ScopeProcessStatus::INIT:
      return "INIT";
    case ScopeProcessStatus::SUCCESS:
      return "SUCCESS";
    case ScopeProcessStatus::RESOURCE_INSUFFICIENT:
      return "RESOURCE_INSUFFICIENT";
    case ScopeProcessStatus::NO_TARGET_NODE:
      return "NO_TARGET_NODE";
    case ScopeProcessStatus::UNRECOVERABLE_FAIL:
      return "UNRECOVERABLE_FAIL";
    default:
      return "UNKNOWN";
  }
}

/*!
 * \brief Convert ScopeProcessStatus to user-facing detail string
 */
inline const char *ScopeProcessStatusDetail(ScopeProcessStatus status) {
  switch (status) {
    case ScopeProcessStatus::INIT:
      return "Scope has not been processed";
    case ScopeProcessStatus::RESOURCE_INSUFFICIENT:
      return "Insufficient stream task slots or event memory resources";
    case ScopeProcessStatus::NO_TARGET_NODE:
      return "No target node remains after filtering";
    case ScopeProcessStatus::UNRECOVERABLE_FAIL:
      return "Unrecoverable failure that cannot be skipped";
    default:
      return "";
  }
}

/*!
 * \enum ScopeBreakReason
 * \brief Reasons why a scope boundary was created (scope was split)
 */
enum class ScopeBreakReason : uint8_t {
  NONE,
  UNFUSIBLE_NODE,
  DEADLOCK_DETECTED,
  SYNCALL_OP_DROP,
  DEBUG_PER_OP_MAX_CORE,
};

/*!
 * \brief Convert ScopeBreakReason to enum name string
 */
inline const char *to_string(ScopeBreakReason reason) {
  switch (reason) {
    case ScopeBreakReason::NONE:
      return "NONE";
    case ScopeBreakReason::UNFUSIBLE_NODE:
      return "UNFUSIBLE_NODE";
    case ScopeBreakReason::DEADLOCK_DETECTED:
      return "DEADLOCK_DETECTED";
    case ScopeBreakReason::SYNCALL_OP_DROP:
      return "SYNCALL_OP_DROP";
    case ScopeBreakReason::DEBUG_PER_OP_MAX_CORE:
      return "DEBUG_PER_OP_MAX_CORE";
    default:
      return "UNKNOWN_SCOPE_BREAK_REASON";
  }
}

/*!
 * \class ScopeBreakInfo
 * \brief Information about why a scope boundary was created
 */
class ScopeBreakInfo {
 public:
  ScopeBreakInfo() = default;

  // Builder-style setters for convenient construction
  ScopeBreakInfo &SetReason(ScopeBreakReason r) {
    reason = r;
    return *this;
  }

  ScopeBreakInfo &SetTriggerNode(uint64_t nodeId, uint32_t streamIdx = 0) {
    triggerNodeId = nodeId;
    triggerStreamIdx = streamIdx;
    return *this;
  }

  ScopeBreakInfo &SetTriggerStreamIdx(uint32_t streamIdx) {
    triggerStreamIdx = streamIdx;
    return *this;
  }

  ScopeBreakInfo &SetParentScopeId(uint16_t id) {
    parentScopeId = id;
    return *this;
  }

  ScopeBreakInfo &SetSyncAllNodeIds(const std::vector<uint64_t> &nodeIds) {
    syncAllNodeIds = nodeIds;
    return *this;
  }

  ScopeBreakInfo &SetSyncAllNodeIds(std::vector<uint64_t> &&nodeIds) {
    syncAllNodeIds = std::move(nodeIds);
    return *this;
  }

  ScopeBreakInfo &SetDetail(const std::string &d) {
    detail = d;
    return *this;
  }

  ScopeBreakInfo &SetDetail(std::string &&d) {
    detail = std::move(d);
    return *this;
  }

  // Getters
  ScopeBreakReason GetReason() const {
    return reason;
  }
  uint64_t GetTriggerNodeId() const {
    return triggerNodeId;
  }
  uint32_t GetTriggerStreamIdx() const {
    return triggerStreamIdx;
  }
  uint16_t GetParentScopeId() const {
    return parentScopeId;
  }
  const std::string &GetDetail() const {
    return detail;
  }
  const std::vector<uint64_t> &GetSyncAllNodeIds() const {
    return syncAllNodeIds;
  }

  std::string Format() const {
    std::string result;
    if (reason != ScopeBreakReason::NONE) {
      result = "breakReason=" + std::string(to_string(reason));
    }
    if (triggerNodeId != INVALID_TASK_ID) {
      result += ", triggerNode=" + std::to_string(triggerNodeId);
    }
    if (triggerStreamIdx != 0 || triggerNodeId != INVALID_TASK_ID) {
      result += ", triggerStream=" + std::to_string(triggerStreamIdx);
    }
    if (parentScopeId != INVALID_SCOPE_ID) {
      result += ", parentScope=" + std::to_string(parentScopeId);
    }
    if (!syncAllNodeIds.empty()) {
      result += ", syncAllNodes=[";
      for (size_t i = 0; i < syncAllNodeIds.size(); ++i) {
        if (i != 0) {
          result += ",";
        }
        result += std::to_string(syncAllNodeIds[i]);
      }
      result += "]";
    }
    if (!detail.empty()) {
      result += ", detail=\"" + detail + "\"";
    }
    return result;
  }

 private:
  ScopeBreakReason reason = ScopeBreakReason::NONE;
  uint64_t triggerNodeId = INVALID_TASK_ID;   // Node that triggered the break
  uint32_t triggerStreamIdx = 0;              // Stream of trigger node
  uint16_t parentScopeId = INVALID_SCOPE_ID;  // Parent scope ID (split from, for tracing split chain)
  std::vector<uint64_t> syncAllNodeIds;       // SyncAll kernel nodes contained in the scope before break
  std::string detail;                         // Human-readable description
};

/*!
 * \class ScopeIdGenerator
 * \brief Singleton class for generating unique scope IDs
 *
 * This generator ensures each scope gets a unique ID at creation time.
 * The ID is not recycled when scope is destroyed, allowing full lifecycle tracking.
 * Uses uint16_t for compact storage; logs INFO when wraparound occurs (every 65536 scopes).
 */
class ScopeIdGenerator {
 public:
  static ScopeIdGenerator &Instance() {
    static ScopeIdGenerator instance;
    return instance;
  }

  uint16_t NextId() {
    uint16_t id = nextId_.fetch_add(1, std::memory_order_relaxed);
    if (static_cast<uint16_t>(id + 1U) == 0U) {
      SK_LOGI("ScopeId wraparound detected: scope ID counter reset to 0 after reaching maximum");
    }
    return id;
  }

 private:
  ScopeIdGenerator() = default;
  ScopeIdGenerator(const ScopeIdGenerator &) = delete;
  ScopeIdGenerator &operator=(const ScopeIdGenerator &) = delete;

  std::atomic<uint16_t> nextId_{0};
};

/*!
 * \struct ScopeExtInfo
 * \brief Extended information for scope processing and scheduling
 */
struct ScopeExtInfo {
  std::vector<std::vector<aclmdlRITaskParams>> customParamsList;  ///< Custom parameters for each stream
  std::vector<SuperKernelBaseNode *> filteredNodes;               ///< Filtered nodes used for scheduling
  std::vector<std::unique_ptr<SuperKernelBaseNode>> eventNodes;   ///< Synthesized event nodes for stream sync
  uint64_t skMainNodeId = INVALID_TASK_ID;                        ///< Main launch node ID for this scope
  std::string scopeName;
  ScopeProcessStatus processStatus = ScopeProcessStatus::INIT;  ///< Processing result status for this scope

  ScopeExtInfo() = default;
  ScopeExtInfo(const ScopeExtInfo &) = delete;
  ScopeExtInfo &operator=(const ScopeExtInfo &) = delete;
  ScopeExtInfo(ScopeExtInfo &&) = default;
  ScopeExtInfo &operator=(ScopeExtInfo &&) = default;
};

/*!
 * \class SuperKernelScopeInfo
 * \brief Information about a single scope in Super Kernel fusion
 *
 * A scope represents a collection of nodes from potentially multiple streams
 * that can be fused into a single Super Kernel.
 */
class SuperKernelScopeInfo {
 public:
  SuperKernelScopeInfo() : scopeId_(ScopeIdGenerator::Instance().NextId()) {}
  ~SuperKernelScopeInfo() {
    ClearNodeScopeIds();
  }
  SuperKernelScopeInfo(const SuperKernelScopeInfo &) = delete;
  SuperKernelScopeInfo &operator=(const SuperKernelScopeInfo &) = delete;
  SuperKernelScopeInfo(SuperKernelScopeInfo &&other) noexcept
      : scopeId_(other.scopeId_),
        scopeStreamInfos_(std::move(other.scopeStreamInfos_)),
        nodes_(std::move(other.nodes_)),
        scopeBitFlags_(other.scopeBitFlags_),
        breakInfo_(std::move(other.breakInfo_)),
        scopeCoreInfo_(other.scopeCoreInfo_),
        extInfo_(std::move(other.extInfo_)) {
    other.nodes_.clear();
  }
  SuperKernelScopeInfo &operator=(SuperKernelScopeInfo &&other) noexcept {
    if (this != &other) {
      ClearNodeScopeIds();
      scopeId_ = other.scopeId_;
      scopeStreamInfos_ = std::move(other.scopeStreamInfos_);
      nodes_ = std::move(other.nodes_);
      scopeBitFlags_ = other.scopeBitFlags_;
      breakInfo_ = std::move(other.breakInfo_);
      scopeCoreInfo_ = other.scopeCoreInfo_;
      extInfo_ = std::move(other.extInfo_);
      other.nodes_.clear();
    }
    return *this;
  }

  uint16_t GetScopeId() const {
    return scopeId_;
  }

  // ============ ScopeStreamInfos ============
  const std::vector<ScopeStreamInfo> &GetScopeStreamInfos() const {
    return scopeStreamInfos_;
  }
  void SetScopeStreamInfos(std::vector<ScopeStreamInfo> infos) {
    scopeStreamInfos_ = std::move(infos);
  }
  void AddScopeStreamInfo(const ScopeStreamInfo &info) {
    scopeStreamInfos_.push_back(info);
  }

  // ============ Nodes ============
  const std::vector<SuperKernelBaseNode *> &GetNodes() const {
    return nodes_;
  }
  void ClearNodeScopeIds() const {
    for (auto *node : nodes_) {
      if (node != nullptr && node->GetScopeId() == scopeId_) {
        node->ClearScopeId();
      }
    }
  }
  void SetNodes(std::vector<SuperKernelBaseNode *> nodeList) {
    ClearNodeScopeIds();
    nodes_ = std::move(nodeList);
    for (auto *node : nodes_) {
      if (node != nullptr) {
        node->SetScopeId(scopeId_);
      }
    }
  }
  void AddNode(SuperKernelBaseNode *node) {
    nodes_.push_back(node);
    if (node != nullptr) {
      node->SetScopeId(scopeId_);
    }
  }

  // ============ ScopeBitFlags ============
  const std::bitset<MAX_SCOPE_NUM> &GetScopeBitFlags() const {
    return scopeBitFlags_;
  }
  void SetScopeBitFlags(const std::bitset<MAX_SCOPE_NUM> &flags) {
    scopeBitFlags_ = flags;
  }

  // ============ BreakInfo ============
  const ScopeBreakInfo &GetBreakInfo() const {
    return breakInfo_;
  }
  ScopeBreakInfo &MutableBreakInfo() {
    return breakInfo_;
  }
  void SetBreakInfo(const ScopeBreakInfo &info) {
    breakInfo_ = info;
  }
  void SetBreakInfo(ScopeBreakInfo &&info) {
    breakInfo_ = std::move(info);
  }

  // Convenience method to set break info with builder pattern
  void SetBreakReason(ScopeBreakReason reason, uint64_t triggerNodeId = INVALID_TASK_ID,
                      uint32_t triggerStreamIdx = 0) {
    breakInfo_.SetReason(reason).SetTriggerNode(triggerNodeId, triggerStreamIdx);
  }

  // ============ Final SK Core Result ============
  const ScopeCoreInfo &GetScopeCoreInfo() const {
    return scopeCoreInfo_;
  }
  void SetScopeCoreInfo(const ScopeCoreInfo &scopeCoreInfo) {
    scopeCoreInfo_ = scopeCoreInfo;
  }
  void ClearScopeCoreInfo() {
    scopeCoreInfo_ = {};
  }

  // ============ ExtInfo ============
  const ScopeExtInfo &GetExtInfo() const {
    return extInfo_;
  }
  void SetExtInfo(ScopeExtInfo &&extInfo) {
    extInfo_ = std::move(extInfo);
  }
  ScopeExtInfo &MutableExtInfo() {
    return extInfo_;
  }

 private:
  uint16_t scopeId_;                               ///< Unique scope ID assigned at creation
  std::vector<ScopeStreamInfo> scopeStreamInfos_;  ///< Per-stream information
  std::vector<SuperKernelBaseNode *> nodes_;       ///< All nodes in this scope
  std::bitset<MAX_SCOPE_NUM> scopeBitFlags_;       ///< Scope bit flags
  ScopeBreakInfo breakInfo_;                       ///< Break reason for this scope boundary
  ScopeCoreInfo scopeCoreInfo_;                    ///< Final SK type and numBlocks
  ScopeExtInfo extInfo_;                           ///< Extended info for scope processing and scheduling
};

#endif  // __SK_SCOPE_INFO_H__
