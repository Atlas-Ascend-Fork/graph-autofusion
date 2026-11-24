/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/anchor.h"
#include <algorithm>
#include <cstring>
#include <sstream>
#include "graph_metadef/graph/debug/ge_util.h"
#include "framework/common/debug/ge_log.h"
#include "graph/node.h"
#include "common/util/trace_manager/trace_manager.h"

namespace {
constexpr size_t kAnchorTypeMaxLen = 1024U;
bool CanAddPeer(const af::AnchorPtr &anchor) {
  if (anchor->IsTypeIdOf<af::InDataAnchor>() && (anchor->GetPeerAnchorsSize() != 0U)) {
    REPORT_INNER_ERR_MSG("E18888", "anchor is type of InDataAnchor, its peer is not empty.");
    GELOGE(ge::GRAPH_FAILED, "[Check][Param] anchor is type of InDataAnchor, its peer is not empty.");
    return false;
  }
  return true;
}
bool IsSameType(const af::Anchor::TYPE &lh, const af::Anchor::TYPE &rh) {
  if (lh == rh) {
    return true;
  }

  return (strncmp(lh, rh, kAnchorTypeMaxLen) == 0);
}
};  // namespace

namespace af {
class AnchorImpl {
 public:
  AnchorImpl(const NodePtr &owner_node, const int32_t idx);
  ~AnchorImpl() = default;
  size_t GetPeerAnchorsSize() const;
  Anchor::Vistor<AnchorPtr> GetPeerAnchors(const std::shared_ptr<ConstAnchor> &anchor_ptr) const;
  std::vector<Anchor *> GetPeerAnchorsPtr() const;
  AnchorPtr GetFirstPeerAnchor() const;
  NodePtr GetOwnerNode() const;
  Node *GetOwnerNodeBarePtr() const;
  int32_t GetIdx() const;
  void SetIdx(const int32_t index);

 private:
  // All peer anchors connected to current anchor
  std::vector<std::weak_ptr<Anchor>> peer_anchors_;
  // The owner node of anchor
  std::weak_ptr<Node> owner_node_;
  // The bare ptr of owner node,
  Node *const owner_node_ptr_;
  // The index of current anchor
  int32_t idx_;

  friend class Anchor;
  friend class OutControlAnchor;
  friend class InControlAnchor;
  friend class OutDataAnchor;
  friend class InDataAnchor;
};

AnchorImpl::AnchorImpl(const NodePtr &owner_node, const int32_t idx)
    : owner_node_(owner_node), owner_node_ptr_(owner_node_.lock().get()), idx_(idx) {}

size_t AnchorImpl::GetPeerAnchorsSize() const {
  return peer_anchors_.size();
}

Anchor::Vistor<AnchorPtr> AnchorImpl::GetPeerAnchors(const std::shared_ptr<ConstAnchor> &anchor_ptr) const {
  std::vector<AnchorPtr> ret;
  ret.resize(peer_anchors_.size());
  (void)std::transform(peer_anchors_.begin(), peer_anchors_.end(), ret.begin(),
                       [](const std::weak_ptr<Anchor> &anchor) { return anchor.lock(); });
  return Anchor::Vistor<AnchorPtr>(anchor_ptr, ret);
}

std::vector<Anchor *> AnchorImpl::GetPeerAnchorsPtr() const {
  std::vector<Anchor *> ret;
  ret.resize(peer_anchors_.size());
  (void)std::transform(peer_anchors_.begin(), peer_anchors_.end(), ret.begin(),
                       [](const std::weak_ptr<Anchor> &anchor) { return anchor.lock().get(); });
  return ret;
}

AnchorPtr AnchorImpl::GetFirstPeerAnchor() const {
  if (peer_anchors_.empty()) {
    return nullptr;
  } else {
    return Anchor::DynamicAnchorCast<Anchor>(peer_anchors_.begin()->lock());
  }
}

NodePtr AnchorImpl::GetOwnerNode() const {
  return owner_node_.lock();
}
Node *AnchorImpl::GetOwnerNodeBarePtr() const {
  return owner_node_ptr_;
}

int32_t AnchorImpl::GetIdx() const {
  return idx_;
}

void AnchorImpl::SetIdx(const int32_t index) {
  idx_ = index;
}

Anchor::Anchor(const NodePtr &owner_node, const int32_t idx)
    : enable_shared_from_this(), impl_(ComGraphMakeShared<AnchorImpl>(owner_node, idx)) {}

Anchor::~Anchor() = default;

bool Anchor::IsTypeOf(const TYPE type) const {
  return strncmp(Anchor::TypeOf<Anchor>(), type, kAnchorTypeMaxLen) == 0;
}

bool Anchor::IsTypeIdOf(const TypeId &type) const {
  return ge::GetTypeId<Anchor>() == type;
}

Anchor::TYPE Anchor::GetSelfType() const {
  return TypeOf<Anchor>();
}

size_t Anchor::GetPeerAnchorsSize() const {
  if (impl_ == nullptr) {
    GELOGE(ge::GRAPH_FAILED, "[Check][Param] impl_ of anchor is nullptr.");
    return 0UL;
  }
  return impl_->GetPeerAnchorsSize();
}

Anchor::Vistor<AnchorPtr> Anchor::GetPeerAnchors() const {
  if (impl_ == nullptr) {
    GELOGE(ge::GRAPH_FAILED, "[Check][Param] impl_ of anchor is nullptr.");
    const std::vector<AnchorPtr> ret;
    return Anchor::Vistor<AnchorPtr>(shared_from_this(), ret);
  }
  return impl_->GetPeerAnchors(shared_from_this());
}

std::vector<Anchor *> Anchor::GetPeerAnchorsPtr() const {
  if (impl_ == nullptr) {
    GELOGE(ge::GRAPH_FAILED, "[Check][Param] impl_ of anchor is nullptr.");
    std::vector<Anchor *> ret;
    return ret;
  }
  return impl_->GetPeerAnchorsPtr();
}

AnchorPtr Anchor::GetFirstPeerAnchor() const {
  if (impl_ == nullptr) {
    GELOGE(ge::GRAPH_FAILED, "[Check][Param] impl_ of anchor is nullptr.");
    return nullptr;
  }
  return impl_->GetFirstPeerAnchor();
}

NodePtr Anchor::GetOwnerNode() const {
  if (impl_ == nullptr) {
    GELOGE(ge::GRAPH_FAILED, "[Check][Param] impl_ of anchor is nullptr.");
    return nullptr;
  }
  return impl_->GetOwnerNode();
}

Node *Anchor::GetOwnerNodeBarePtr() const {
  if (impl_ == nullptr) {
    GELOGE(ge::GRAPH_FAILED, "[Check][Param] impl_ of anchor is nullptr.");
    return nullptr;
  }
  return impl_->GetOwnerNodeBarePtr();
}

void Anchor::UnlinkAll() noexcept {
  if (impl_ == nullptr) {
    GELOGE(ge::GRAPH_FAILED, "[Check][Param] impl_ of anchor is nullptr.");
    return;
  }
  if (!impl_->peer_anchors_.empty()) {
    do {
      const auto peer_anchor_ptr = impl_->peer_anchors_.begin()->lock();
      (void)Unlink(peer_anchor_ptr);
    } while (!impl_->peer_anchors_.empty());
  }
}

graphStatus Anchor::Unlink(const AnchorPtr &peer) {
  if (peer == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "param peer is nullptr, check invalid.");
    GELOGE(ge::GRAPH_FAILED, "[Check][Param] peer anchor is invalid.");
    return ge::GRAPH_FAILED;
  }
  GE_CHK_BOOL_RET_STATUS(impl_ != nullptr, ge::GRAPH_FAILED, "[Check][Param] impl_ of anchor is nullptr");
  const auto it =
      std::find_if(impl_->peer_anchors_.begin(), impl_->peer_anchors_.end(), [peer](const std::weak_ptr<Anchor> &an) {
        const auto anchor = an.lock();
        return peer->Equal(anchor);
      });
  if (it == impl_->peer_anchors_.end()) {
    GELOGW("[Check][Param] Unlink failed , as this anchor is not connected to peer.");
    return ge::GRAPH_FAILED;
  }

  const auto it_peer = std::find_if(peer->impl_->peer_anchors_.begin(), peer->impl_->peer_anchors_.end(),
                                    [this](const std::weak_ptr<Anchor> &an) {
                                      const auto anchor = an.lock();
                                      return Equal(anchor);
                                    });

  GE_CHK_BOOL_RET_STATUS(it_peer != peer->impl_->peer_anchors_.end(), ge::GRAPH_FAILED,
                         "[Check][Param] peer(%s, %d) is not connected to this anchor(%s, %d)",
                         peer->GetOwnerNode()->GetName().c_str(), peer->GetIdx(),
                         this->GetOwnerNode()->GetName().c_str(), this->GetIdx());
  if ((this->GetOwnerNode() != nullptr) && (peer->GetOwnerNode() != nullptr)) {
    TRACE_GEN_RECORD(TraceManager::GetTraceHeader(), "delete", TraceManager::GetOutGraphName(),
                     this->GetOwnerNode()->GetName(), "output:" << this->GetIdx(), "", "",
                     peer->GetOwnerNode()->GetName() << ":input:" << peer->GetIdx());
  }
  (void)impl_->peer_anchors_.erase(it);
  (void)peer->impl_->peer_anchors_.erase(it_peer);
  return GRAPH_SUCCESS;
}

graphStatus Anchor::Insert(const AnchorPtr &old_peer, const AnchorPtr &first_peer, const AnchorPtr &second_peer) {
  GE_CHECK_NOTNULL(old_peer);
  GE_CHECK_NOTNULL(first_peer);
  GE_CHECK_NOTNULL(second_peer);
  GE_CHECK_NOTNULL(impl_);

  if (!IsSameType(old_peer->GetSelfType(), first_peer->GetSelfType())) {
    REPORT_INNER_ERR_MSG("E18888", "the type of old_peer[%s] and first_peer[%s] is not the same.",
                         old_peer->GetSelfType(), first_peer->GetSelfType());
    GELOGE(ge::GRAPH_FAILED, "[Check][Param] the type of old_peer[%s] and first_peer[%s] is not the same.",
           old_peer->GetSelfType(), first_peer->GetSelfType());
    return ge::GRAPH_FAILED;
  }

  if (!IsSameType(second_peer->GetSelfType(), this->GetSelfType())) {
    REPORT_INNER_ERR_MSG("E18888", "the type of second_peer[%s] and current anchor[%s] is not the same.",
                         second_peer->GetSelfType(), this->GetSelfType());
    GELOGE(ge::GRAPH_FAILED, "[Check][Param] the type of second_peer[%s] and current anchor[%s] is not the same.",
           second_peer->GetSelfType(), this->GetSelfType());
    return ge::GRAPH_FAILED;
  }

  if ((!CanAddPeer(first_peer)) || (!CanAddPeer(second_peer))) {
    REPORT_INNER_ERR_MSG("E18888", "first_peer[%s] or second_peer[%s] check failed", first_peer->GetSelfType(),
                         second_peer->GetSelfType());
    GELOGE(ge::GRAPH_FAILED, "[Check][Param] first_peer[%s] or second_peer[%s] check failed", first_peer->GetSelfType(),
           second_peer->GetSelfType());
    return ge::GRAPH_FAILED;
  }

  const auto this_it = std::find_if(impl_->peer_anchors_.begin(), impl_->peer_anchors_.end(),
                                    [old_peer](const std::weak_ptr<Anchor> &an) {
                                      const auto anchor = an.lock();
                                      return old_peer->Equal(anchor);
                                    });

  GE_CHK_BOOL_RET_STATUS(this_it != impl_->peer_anchors_.end(), ge::GRAPH_FAILED,
                         "[Check][Param] this anchor(%s, %d) is not connected to old_peer(%s, %d)",
                         this->GetOwnerNode()->GetName().c_str(), this->GetIdx(),
                         old_peer->GetOwnerNode()->GetName().c_str(), old_peer->GetIdx());

  const auto old_it = std::find_if(old_peer->impl_->peer_anchors_.begin(), old_peer->impl_->peer_anchors_.end(),
                                   [this](const std::weak_ptr<Anchor> &an) {
                                     const auto anchor = an.lock();
                                     return Equal(anchor);
                                   });
  GE_CHK_BOOL_RET_STATUS(old_it != old_peer->impl_->peer_anchors_.end(), ge::GRAPH_FAILED,
                         "[Check][Param] old_peer(%s, %d) is not connected to this anchor(%s, %d)",
                         old_peer->GetOwnerNode()->GetName().c_str(), old_peer->GetIdx(),
                         this->GetOwnerNode()->GetName().c_str(), this->GetIdx());
  *this_it = first_peer;
  first_peer->impl_->peer_anchors_.push_back(shared_from_this());
  *old_it = second_peer;
  second_peer->impl_->peer_anchors_.push_back(old_peer);
  return GRAPH_SUCCESS;
}

bool Anchor::IsLinkedWith(const AnchorPtr &peer) const {
  if (impl_ == nullptr) {
    GELOGE(ge::GRAPH_FAILED, "[Check][Param] impl_ of anchor is nullptr.");
    return false;
  }
  const auto it =
      std::find_if(impl_->peer_anchors_.begin(), impl_->peer_anchors_.end(), [peer](const std::weak_ptr<Anchor> &an) {
        const auto anchor = an.lock();
        if (peer == nullptr) {
          GELOGE(ge::GRAPH_FAILED, "[Check][Param] this old peer anchor is nullptr");
          return false;
        }
        return peer->Equal(anchor);
      });
  return (it != impl_->peer_anchors_.end());
}

int32_t Anchor::GetIdx() const {
  if (impl_ == nullptr) {
    GELOGE(ge::GRAPH_FAILED, "[Check][Param] impl_ of anchor is nullptr.");
    return 0;
  }
  return impl_->GetIdx();
}

void Anchor::SetIdx(const int32_t index) {
  if (impl_ == nullptr) {
    GELOGE(ge::GRAPH_FAILED, "[Check][Param] impl_ of anchor is nullptr.");
    return;
  }
  impl_->SetIdx(index);
}

DataAnchor::DataAnchor(const NodePtr &owner_node, const int32_t idx) : Anchor(owner_node, idx) {}

bool DataAnchor::IsTypeOf(const TYPE type) const {
  if (strncmp(Anchor::TypeOf<DataAnchor>(), type, kAnchorTypeMaxLen) == 0) {
    return true;
  }
  return Anchor::IsTypeOf(type);
}

Anchor::TYPE DataAnchor::GetSelfType() const {
  return Anchor::TypeOf<DataAnchor>();
}

bool DataAnchor::IsTypeIdOf(const TypeId &type) const {
  if (ge::GetTypeId<DataAnchor>() == type) {
    return true;
  }
  return Anchor::IsTypeIdOf(type);
}

InDataAnchor::InDataAnchor(const NodePtr &owner_node, const int32_t idx) : DataAnchor(owner_node, idx) {}

OutDataAnchorPtr InDataAnchor::GetPeerOutAnchor() const {
  if ((impl_ == nullptr) || impl_->peer_anchors_.empty()) {
    return nullptr;
  } else {
    return Anchor::DynamicAnchorCast<OutDataAnchor>(impl_->peer_anchors_.begin()->lock());
  }
}

graphStatus InDataAnchor::LinkFrom(const OutDataAnchorPtr &src) {
  // InDataAnchor must be only linkfrom once
  if ((src == nullptr) || (src->impl_ == nullptr) || (impl_ == nullptr) || (!impl_->peer_anchors_.empty())) {
    REPORT_INNER_ERR_MSG("E18888", "src anchor is invalid or the peerAnchors is not empty.");
    GELOGE(ge::GRAPH_FAILED, "[Check][Param] src anchor is invalid or the peerAnchors is not empty.");
    return ge::GRAPH_FAILED;
  }
  impl_->peer_anchors_.push_back(src);
  src->impl_->peer_anchors_.push_back(shared_from_this());
  // src->impl_->GetOwnerNode() is null:  peer->GetOwnerNode() is null:
  if ((src->impl_->GetOwnerNodeBarePtr() == nullptr) || (impl_->GetOwnerNodeBarePtr() == nullptr)) {
    GELOGW("[Check][Param] src->impl_->GetOwnerNode() or impl_->GetOwnerNode() is null.");
  } else {
    TRACE_GEN_RECORD(TraceManager::GetTraceHeader(), "add", TraceManager::GetOutGraphName(),
                     src->impl_->GetOwnerNode()->GetName(), "output:" << src->impl_->GetIdx(), "", "",
                     impl_->GetOwnerNode()->GetName() << ":input:" << impl_->GetIdx());
  }
  return GRAPH_SUCCESS;
}

bool InDataAnchor::Equal(const AnchorPtr anchor) const {
  const auto in_data_anchor = Anchor::DynamicAnchorCast<InDataAnchor>(anchor);
  if (in_data_anchor != nullptr) {
    if ((GetOwnerNodeBarePtr() == in_data_anchor->GetOwnerNodeBarePtr()) && (GetIdx() == in_data_anchor->GetIdx())) {
      return true;
    }
  }
  return false;
}

bool InDataAnchor::IsTypeOf(const TYPE type) const {
  if (strncmp(Anchor::TypeOf<InDataAnchor>(), type, kAnchorTypeMaxLen) == 0) {
    return true;
  }
  return DataAnchor::IsTypeOf(type);
}

Anchor::TYPE InDataAnchor::GetSelfType() const {
  return Anchor::TypeOf<InDataAnchor>();
}

bool InDataAnchor::IsTypeIdOf(const TypeId &type) const {
  if (ge::GetTypeId<InDataAnchor>() == type) {
    return true;
  }
  return DataAnchor::IsTypeIdOf(type);
}

OutDataAnchor::OutDataAnchor(const NodePtr &owner_node, const int32_t idx) : DataAnchor(owner_node, idx) {}

OutDataAnchor::Vistor<InDataAnchorPtr> OutDataAnchor::GetPeerInDataAnchors() const {
  std::vector<InDataAnchorPtr> ret;
  if (impl_ != nullptr) {
    ret.reserve(impl_->peer_anchors_.size());
    for (const auto &anchor : impl_->peer_anchors_) {
      const auto in_data_anchor = Anchor::DynamicAnchorCast<InDataAnchor>(anchor.lock());
      if (in_data_anchor != nullptr) {
        ret.push_back(in_data_anchor);
      }
    }
  }
  return OutDataAnchor::Vistor<InDataAnchorPtr>(shared_from_this(), ret);
}

std::vector<InDataAnchor *> OutDataAnchor::GetPeerInDataAnchorsPtr() const {
  std::vector<InDataAnchor *> ret;
  if (impl_ != nullptr) {
    ret.reserve(impl_->peer_anchors_.size());
    for (const auto &anchor : impl_->peer_anchors_) {
      const auto in_data_anchor = Anchor::DynamicAnchorPtrCast<InDataAnchor>(anchor.lock().get());
      if (in_data_anchor != nullptr) {
        ret.push_back(in_data_anchor);
      }
    }
  }
  return ret;
}

uint32_t OutDataAnchor::GetPeerInDataNodesSize() const {
  uint32_t out_nums = 0U;
  if (impl_ != nullptr) {
    for (const auto &anchor : impl_->peer_anchors_) {
      const auto in_data_anchor = Anchor::DynamicAnchorCast<InDataAnchor>(anchor.lock());
      if ((in_data_anchor != nullptr) && (in_data_anchor->GetOwnerNodeBarePtr() != nullptr)) {
        out_nums++;
      }
    }
  }
  return out_nums;
}

OutDataAnchor::Vistor<InControlAnchorPtr> OutDataAnchor::GetPeerInControlAnchors() const {
  std::vector<InControlAnchorPtr> ret;
  if (impl_ != nullptr) {
    ret.reserve(impl_->peer_anchors_.size());
    for (const auto &anchor : impl_->peer_anchors_) {
      const auto in_control_anchor = Anchor::DynamicAnchorCast<InControlAnchor>(anchor.lock());
      if (in_control_anchor != nullptr) {
        ret.push_back(in_control_anchor);
      }
    }
  }
  return OutDataAnchor::Vistor<InControlAnchorPtr>(shared_from_this(), ret);
}

graphStatus OutDataAnchor::LinkTo(const InDataAnchorPtr &dest) {
  if ((dest == nullptr) || (dest->impl_ == nullptr) || (!dest->impl_->peer_anchors_.empty())) {
    REPORT_INNER_ERR_MSG("E18888", "dest anchor is nullptr or the peerAnchors is not empty.");
    GELOGE(ge::GRAPH_FAILED, "[Check][Param] dest anchor is nullptr or the peerAnchors is not empty.");
    return ge::GRAPH_FAILED;
  }
  if (impl_ == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "anchor param is nullptr, check invalid");
    GELOGE(ge::GRAPH_FAILED, "[Check][Param] owner anchor is invalid.");
    return ge::GRAPH_FAILED;
  }
  impl_->peer_anchors_.push_back(dest);
  dest->impl_->peer_anchors_.push_back(shared_from_this());

  TRACE_GEN_RECORD(TraceManager::GetTraceHeader(), "add", TraceManager::GetOutGraphName(),
                   impl_->GetOwnerNode()->GetName(), "output:" << impl_->GetIdx(), "", "",
                   dest->impl_->GetOwnerNode()->GetName() << ":input:" << dest->impl_->GetIdx());
  return GRAPH_SUCCESS;
}

graphStatus OutDataAnchor::LinkTo(const InControlAnchorPtr &dest) {
  if ((dest == nullptr) || (dest->impl_ == nullptr)) {
    REPORT_INNER_ERR_MSG("E18888", "param dest is nullptr, check invalid");
    GELOGE(ge::GRAPH_FAILED, "[Check][Param] dest anchor is invalid.");
    return ge::GRAPH_FAILED;
  }
  if (impl_ == nullptr) {
    GELOGE(ge::GRAPH_FAILED, "src anchor is invalid.");
    return ge::GRAPH_FAILED;
  }
  impl_->peer_anchors_.push_back(dest);
  dest->impl_->peer_anchors_.push_back(shared_from_this());

  TRACE_GEN_RECORD(TraceManager::GetTraceHeader(), "add", TraceManager::GetOutGraphName(),
                   impl_->GetOwnerNode()->GetName(), "output:" << impl_->GetIdx(), "", "",
                   dest->impl_->GetOwnerNode()->GetName() << ":input:" << dest->impl_->GetIdx());
  return GRAPH_SUCCESS;
}

graphStatus OutControlAnchor::LinkTo(const InDataAnchorPtr &dest) {
  if ((dest == nullptr) || (dest->impl_ == nullptr)) {
    REPORT_INNER_ERR_MSG("E18888", "param dest is nullptr, check invalid");
    GELOGE(ge::GRAPH_FAILED, "[Check][Param] dest anchor is invalid.");
    return ge::GRAPH_FAILED;
  }
  if (impl_ == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "anchor param is nullptr, check invalid");
    GELOGE(ge::GRAPH_FAILED, "[Check][Param] owner anchor is invalid.");
    return ge::GRAPH_FAILED;
  }
  impl_->peer_anchors_.push_back(dest);
  dest->impl_->peer_anchors_.push_back(shared_from_this());

  TRACE_GEN_RECORD(TraceManager::GetTraceHeader(), "add", TraceManager::GetOutGraphName(),
                   impl_->GetOwnerNode()->GetName(), "output:" << impl_->GetIdx(), "", "",
                   dest->impl_->GetOwnerNode()->GetName() << ":input:" << dest->impl_->GetIdx());
  return GRAPH_SUCCESS;
}

bool OutDataAnchor::Equal(const AnchorPtr anchor) const {
  CHECK_FALSE_EXEC(anchor != nullptr, return false);
  const auto out_data_anchor = Anchor::DynamicAnchorCast<OutDataAnchor>(anchor);
  if (out_data_anchor != nullptr) {
    if ((GetOwnerNodeBarePtr() == out_data_anchor->GetOwnerNodeBarePtr()) && (GetIdx() == out_data_anchor->GetIdx())) {
      return true;
    }
  }
  return false;
}

bool OutDataAnchor::IsTypeOf(const TYPE type) const {
  if (strncmp(Anchor::TypeOf<OutDataAnchor>(), type, kAnchorTypeMaxLen) == 0) {
    return true;
  }
  return DataAnchor::IsTypeOf(type);
}

Anchor::TYPE OutDataAnchor::GetSelfType() const {
  return Anchor::TypeOf<OutDataAnchor>();
}

bool OutDataAnchor::IsTypeIdOf(const TypeId &type) const {
  if (ge::GetTypeId<OutDataAnchor>() == type) {
    return true;
  }
  return DataAnchor::IsTypeIdOf(type);
}

ControlAnchor::ControlAnchor(const NodePtr &owner_node) : Anchor(owner_node, -1) {}

ControlAnchor::ControlAnchor(const NodePtr &owner_node, const int32_t idx) : Anchor(owner_node, idx) {}

bool ControlAnchor::IsTypeOf(const TYPE type) const {
  if (strncmp(Anchor::TypeOf<ControlAnchor>(), type, kAnchorTypeMaxLen) == 0) {
    return true;
  }
  return Anchor::IsTypeOf(type);
}

Anchor::TYPE ControlAnchor::GetSelfType() const {
  return Anchor::TypeOf<ControlAnchor>();
}

bool ControlAnchor::IsTypeIdOf(const TypeId &type) const {
  if (ge::GetTypeId<ControlAnchor>() == type) {
    return true;
  }
  return Anchor::IsTypeIdOf(type);
}

InControlAnchor::InControlAnchor(const NodePtr &owner_node) : ControlAnchor(owner_node) {}

InControlAnchor::InControlAnchor(const NodePtr &owner_node, const int32_t idx) : ControlAnchor(owner_node, idx) {}

InControlAnchor::Vistor<OutControlAnchorPtr> InControlAnchor::GetPeerOutControlAnchors() const {
  std::vector<OutControlAnchorPtr> ret;
  if (impl_ != nullptr) {
    ret.reserve(impl_->peer_anchors_.size());
    for (const auto &anchor : impl_->peer_anchors_) {
      const auto out_control_anchor = Anchor::DynamicAnchorCast<OutControlAnchor>(anchor.lock());
      if (out_control_anchor != nullptr) {
        ret.push_back(out_control_anchor);
      }
    }
  }
  return InControlAnchor::Vistor<OutControlAnchorPtr>(shared_from_this(), ret);
}

std::vector<OutControlAnchor *> InControlAnchor::GetPeerOutControlAnchorsPtr() const {
  std::vector<OutControlAnchor *> ret;
  if (impl_ != nullptr) {
    ret.reserve(impl_->peer_anchors_.size());
    for (const auto &anchor : impl_->peer_anchors_) {
      const auto out_control_anchor = Anchor::DynamicAnchorPtrCast<OutControlAnchor>(anchor.lock().get());
      if (out_control_anchor != nullptr) {
        ret.push_back(out_control_anchor);
      }
    }
  }
  return ret;
}

bool InControlAnchor::IsPeerOutAnchorsEmpty() const {
  if (impl_ == nullptr) {
    return false;
  }
  return impl_->peer_anchors_.empty();
}

InControlAnchor::Vistor<OutDataAnchorPtr> InControlAnchor::GetPeerOutDataAnchors() const {
  std::vector<OutDataAnchorPtr> ret;
  if (impl_ != nullptr) {
    for (const auto &anchor : impl_->peer_anchors_) {
      const auto out_data_anchor = Anchor::DynamicAnchorCast<OutDataAnchor>(anchor.lock());
      if (out_data_anchor != nullptr) {
        ret.push_back(out_data_anchor);
      }
    }
  }
  return InControlAnchor::Vistor<OutDataAnchorPtr>(shared_from_this(), ret);
}

graphStatus InControlAnchor::LinkFrom(const OutControlAnchorPtr &src) {
  if ((src == nullptr) || (src->impl_ == nullptr)) {
    REPORT_INNER_ERR_MSG("E18888", "param src is nullptr, check invalid");
    GELOGE(ge::GRAPH_FAILED, "[Check][Param] src anchor is invalid.");
    return ge::GRAPH_FAILED;
  }
  if (impl_ == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "anchor param is nullptr, check invalid");
    GELOGE(ge::GRAPH_FAILED, "[Check][Param] owner anchor is invalid.");
    return ge::GRAPH_FAILED;
  }
  impl_->peer_anchors_.push_back(src);
  src->impl_->peer_anchors_.push_back(shared_from_this());

  TRACE_GEN_RECORD(TraceManager::GetTraceHeader(), "add", TraceManager::GetOutGraphName(),
                   src->impl_->GetOwnerNode()->GetName(), "output:" << src->impl_->GetIdx(), "", "",
                   impl_->GetOwnerNode()->GetName() << ":input:" << impl_->GetIdx());
  return GRAPH_SUCCESS;
}

bool InControlAnchor::Equal(const AnchorPtr anchor) const {
  CHECK_FALSE_EXEC(anchor != nullptr, REPORT_INNER_ERR_MSG("E18888", "param anchor is nullptr, check invalid");
                   GELOGE(ge::GRAPH_FAILED, "[Check][Param] anchor is invalid."); return false);
  const auto in_control_anchor = Anchor::DynamicAnchorCast<InControlAnchor>(anchor);
  if (in_control_anchor != nullptr) {
    if (GetOwnerNodeBarePtr() == in_control_anchor->GetOwnerNodeBarePtr()) {
      return true;
    }
  }
  return false;
}

bool InControlAnchor::IsTypeOf(const TYPE type) const {
  if (strncmp(Anchor::TypeOf<InControlAnchor>(), type, kAnchorTypeMaxLen) == 0) {
    return true;
  }
  return ControlAnchor::IsTypeOf(type);
}

Anchor::TYPE InControlAnchor::GetSelfType() const {
  return Anchor::TypeOf<InControlAnchor>();
}

bool InControlAnchor::IsTypeIdOf(const TypeId &type) const {
  if (ge::GetTypeId<InControlAnchor>() == type) {
    return true;
  }
  return ControlAnchor::IsTypeIdOf(type);
}

OutControlAnchor::OutControlAnchor(const NodePtr &owner_node) : ControlAnchor(owner_node) {}

OutControlAnchor::OutControlAnchor(const NodePtr &owner_node, const int32_t idx) : ControlAnchor(owner_node, idx) {}

OutControlAnchor::Vistor<InControlAnchorPtr> OutControlAnchor::GetPeerInControlAnchors() const {
  std::vector<InControlAnchorPtr> ret;
  if (impl_ != nullptr) {
    ret.reserve(impl_->peer_anchors_.size());
    for (const auto &anchor : impl_->peer_anchors_) {
      const auto in_control_anchor = Anchor::DynamicAnchorCast<InControlAnchor>(anchor.lock());
      if (in_control_anchor != nullptr) {
        ret.push_back(in_control_anchor);
      }
    }
  }
  return OutControlAnchor::Vistor<InControlAnchorPtr>(shared_from_this(), ret);
}

std::vector<InControlAnchor *> OutControlAnchor::GetPeerInControlAnchorsPtr() const {
  std::vector<InControlAnchor *> ret;
  if (impl_ != nullptr) {
    ret.reserve(impl_->peer_anchors_.size());
    for (const auto &anchor : impl_->peer_anchors_) {
      const auto in_control_anchor = Anchor::DynamicAnchorPtrCast<InControlAnchor>(anchor.lock().get());
      if (in_control_anchor != nullptr) {
        ret.push_back(in_control_anchor);
      }
    }
  }
  return ret;
}

OutControlAnchor::Vistor<InDataAnchorPtr> OutControlAnchor::GetPeerInDataAnchors() const {
  std::vector<InDataAnchorPtr> ret;
  if (impl_ != nullptr) {
    for (const auto &anchor : impl_->peer_anchors_) {
      const auto in_data_anchor = Anchor::DynamicAnchorCast<InDataAnchor>(anchor.lock());
      if (in_data_anchor != nullptr) {
        ret.push_back(in_data_anchor);
      }
    }
  }
  return OutControlAnchor::Vistor<InDataAnchorPtr>(shared_from_this(), ret);
}

graphStatus OutControlAnchor::LinkTo(const InControlAnchorPtr &dest) {
  if ((dest == nullptr) || (dest->impl_ == nullptr)) {
    REPORT_INNER_ERR_MSG("E18888", "param dest is nullptr, check invalid");
    GELOGE(ge::GRAPH_FAILED, "[Check][Param] dest anchor is invalid.");
    return ge::GRAPH_FAILED;
  }
  if (impl_ == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "anchor param is nullptr, check invalid");
    GELOGE(ge::GRAPH_FAILED, "[Check][Param] owner anchor is invalid.");
    return ge::GRAPH_FAILED;
  }
  impl_->peer_anchors_.push_back(dest);
  dest->impl_->peer_anchors_.push_back(shared_from_this());

  TRACE_GEN_RECORD(TraceManager::GetTraceHeader(), "add", TraceManager::GetOutGraphName(),
                   impl_->GetOwnerNode()->GetName(), "output:" << impl_->GetIdx(), "", "",
                   dest->impl_->GetOwnerNode()->GetName() << ":input:" << dest->impl_->GetIdx());
  return GRAPH_SUCCESS;
}

bool OutControlAnchor::Equal(const AnchorPtr anchor) const {
  const auto out_control_anchor = Anchor::DynamicAnchorCast<OutControlAnchor>(anchor);
  if (out_control_anchor != nullptr) {
    if (GetOwnerNodeBarePtr() == out_control_anchor->GetOwnerNodeBarePtr()) {
      return true;
    }
  }
  return false;
}

bool OutControlAnchor::IsTypeOf(const TYPE type) const {
  if (strncmp(Anchor::TypeOf<OutControlAnchor>(), type, kAnchorTypeMaxLen) == 0) {
    return true;
  }
  return ControlAnchor::IsTypeOf(type);
}

Anchor::TYPE OutControlAnchor::GetSelfType() const {
  return Anchor::TypeOf<OutControlAnchor>();
}

bool OutControlAnchor::IsTypeIdOf(const TypeId &type) const {
  if (ge::GetTypeId<OutControlAnchor>() == type) {
    return true;
  }
  return ControlAnchor::IsTypeIdOf(type);
}

}  // namespace af

namespace ge {
template <>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<af::Anchor>() {
  return reinterpret_cast<TypeId>(1);
}

template <>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<af::DataAnchor>() {
  return reinterpret_cast<TypeId>(2);
}

template <>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<af::ControlAnchor>() {
  return reinterpret_cast<TypeId>(3);
}

template <>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<af::InDataAnchor>() {
  return reinterpret_cast<TypeId>(4);
}

template <>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<af::OutDataAnchor>() {
  return reinterpret_cast<TypeId>(5);
}

template <>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<af::InControlAnchor>() {
  return reinterpret_cast<TypeId>(6);
}

template <>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<af::OutControlAnchor>() {
  return reinterpret_cast<TypeId>(7);
}
}  // namespace ge
