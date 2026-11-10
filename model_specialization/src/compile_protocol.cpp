/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "compile_protocol.h"

#include <fcntl.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <climits>
#include <cstring>
#include <utility>

namespace model_spec::compile {
namespace {

// 协议整数均为小端序。4 字节长度前缀不计入帧体；帧头依次是版本、消息类型和任务 ID。
constexpr size_t FRAME_LENGTH_BYTES = sizeof(uint32_t);
constexpr size_t BODY_HEADER_BYTES = sizeof(PROTOCOL_VERSION) + sizeof(MessageKind) + sizeof(uint64_t);
constexpr size_t READ_BUFFER_BYTES = 16U * 1024U;

bool Fail(std::string &error, const std::string &detail) {
  error = detail;
  return false;
}

class Encoder {
 public:
  bool Integer(uint64_t value, size_t width) {
    if (bytes.size() > MAX_FRAME_BYTES - width) {
      return false;
    }
    for (size_t index = 0; index < width; ++index) {
      bytes.push_back(static_cast<unsigned char>(value >> (index * CHAR_BIT)));
    }
    return true;
  }

  bool Blob(const unsigned char *data, size_t size, size_t limit) {
    if (size > limit || !Integer(size, sizeof(uint64_t)) || size > MAX_FRAME_BYTES - bytes.size()) {
      return false;
    }
    if (size != 0) {
      bytes.insert(bytes.end(), data, data + size);
    }
    return true;
  }

  bool String(const std::string &value) {
    return value.find('\0') == std::string::npos &&
           Blob(reinterpret_cast<const unsigned char *>(value.data()), value.size(), MAX_STRING_BYTES);
  }

  bool Strings(const std::vector<std::string> &values) {
    if (values.size() > MAX_OPTION_COUNT || !Integer(values.size(), sizeof(uint64_t))) {
      return false;
    }
    for (const auto &value : values) {
      if (!String(value)) {
        return false;
      }
    }
    return true;
  }

  Bytes bytes;
};

class Decoder {
 public:
  explicit Decoder(const Bytes &bytes) : bytes_(bytes) {}

  bool Integer(uint64_t &value, size_t width) {
    if (width > bytes_.size() - offset_) {
      return false;
    }
    value = 0;
    for (size_t index = 0; index < width; ++index) {
      value |= static_cast<uint64_t>(bytes_[offset_++]) << (index * CHAR_BIT);
    }
    return true;
  }

  bool Blob(Bytes &value, size_t limit) {
    uint64_t size = 0;
    if (!Integer(size, sizeof(size)) || size > limit || size > bytes_.size() - offset_) {
      return false;
    }
    value.assign(bytes_.begin() + offset_, bytes_.begin() + offset_ + static_cast<size_t>(size));
    offset_ += static_cast<size_t>(size);
    return true;
  }

  bool String(std::string &value) {
    uint64_t size = 0;
    if (!Integer(size, sizeof(size)) || size > MAX_STRING_BYTES || size > bytes_.size() - offset_) {
      return false;
    }
    value.assign(reinterpret_cast<const char *>(bytes_.data() + offset_), static_cast<size_t>(size));
    offset_ += static_cast<size_t>(size);
    return value.find('\0') == std::string::npos;
  }

  bool Strings(std::vector<std::string> &values) {
    uint64_t count = 0;
    if (!Integer(count, sizeof(count)) || count > MAX_OPTION_COUNT ||
        count > (bytes_.size() - offset_) / sizeof(uint64_t)) {
      return false;
    }
    values.resize(static_cast<size_t>(count));
    for (auto &value : values) {
      if (!String(value)) {
        return false;
      }
    }
    return true;
  }

  bool Finished() const {
    return offset_ == bytes_.size();
  }

 private:
  const Bytes &bytes_;
  size_t offset_ = 0;
};

bool EncodeJob(const Message &message, Encoder &encoder) {
  const auto &job = message.job;
  if (job.jobId != message.jobId || job.arguments.size() > MAX_ARGUMENT_COUNT || !encoder.String(job.resourceId) ||
      !encoder.String(job.kernelEntry) || !encoder.Integer(job.arguments.size(), sizeof(uint64_t))) {
    return false;
  }
  for (size_t index = 0; index < job.arguments.size(); ++index) {
    const auto &argument = job.arguments[index];
    if (!encoder.Blob(argument.data(), argument.size(), MAX_ARGUMENT_BYTES)) {
      return false;
    }
  }
  return encoder.Strings(job.basicOptions) && encoder.Strings(job.skOptions) && encoder.String(job.outputElfPath);
}

bool DecodeJob(Decoder &decoder, Message &message) {
  auto &job = message.job;
  job.jobId = message.jobId;
  uint64_t count = 0;
  if (!decoder.String(job.resourceId) || !decoder.String(job.kernelEntry) || !decoder.Integer(count, sizeof(count)) ||
      count > MAX_ARGUMENT_COUNT) {
    return false;
  }
  for (uint64_t index = 0; index < count; ++index) {
    Bytes argument;
    if (!decoder.Blob(argument, MAX_ARGUMENT_BYTES)) {
      return false;
    }
    job.arguments.push_back(std::move(argument));
  }
  return decoder.Strings(job.basicOptions) && decoder.Strings(job.skOptions) && decoder.String(job.outputElfPath);
}

bool EncodeReply(const CompileReply &reply, Encoder &encoder) {
  return encoder.Integer(static_cast<uint32_t>(reply.aclrtcCode), sizeof(uint32_t)) && encoder.String(reply.detail);
}

bool DecodeReply(Decoder &decoder, CompileReply &reply) {
  uint64_t code = 0;
  if (!decoder.Integer(code, sizeof(uint32_t)) || !decoder.String(reply.detail)) {
    return false;
  }
  const uint32_t unsignedCode = static_cast<uint32_t>(code);
  reply.aclrtcCode = static_cast<int32_t>(unsignedCode);
  return true;
}

std::string SystemError(const char *operation) {
  return std::string(operation) + ": " + std::strerror(errno);
}

}  // namespace

bool EncodeMessage(const Message &message, Bytes &fullFrame, std::string &error) {
  error.clear();
  fullFrame.clear();
  Encoder encoder;
  encoder.Integer(PROTOCOL_VERSION, sizeof(PROTOCOL_VERSION));
  encoder.Integer(static_cast<uint32_t>(message.kind), sizeof(MessageKind));
  encoder.Integer(message.jobId, sizeof(message.jobId));
  bool valid = false;
  switch (message.kind) {
    case MessageKind::READY:
      valid = encoder.String(message.buildId);
      break;
    case MessageKind::JOB:
      valid = EncodeJob(message, encoder);
      break;
    case MessageKind::RESULT:
    case MessageKind::JOB_ERROR:
      valid = EncodeReply(message.reply, encoder);
      break;
    case MessageKind::SHUTDOWN:
      valid = true;
      break;
  }
  if (!valid) {
    return Fail(error, "invalid message fields or payload exceeds protocol budget");
  }
  const auto size = static_cast<uint32_t>(encoder.bytes.size());
  fullFrame.reserve(FRAME_LENGTH_BYTES + size);
  for (size_t index = 0; index < FRAME_LENGTH_BYTES; ++index) {
    fullFrame.push_back(static_cast<unsigned char>(size >> (index * CHAR_BIT)));
  }
  fullFrame.insert(fullFrame.end(), encoder.bytes.begin(), encoder.bytes.end());
  return true;
}

bool DecodeMessage(const Bytes &body, Message &message, std::string &error) {
  error.clear();
  if (body.size() < BODY_HEADER_BYTES || body.size() > MAX_FRAME_BYTES) {
    return Fail(error, "invalid frame body size");
  }
  Decoder decoder(body);
  uint64_t version = 0;
  uint64_t kind = 0;
  Message decoded;
  if (!decoder.Integer(version, sizeof(PROTOCOL_VERSION)) || version != PROTOCOL_VERSION ||
      !decoder.Integer(kind, sizeof(MessageKind)) || !decoder.Integer(decoded.jobId, sizeof(decoded.jobId))) {
    return Fail(error, "invalid protocol header");
  }
  decoded.kind = static_cast<MessageKind>(kind);
  bool valid = false;
  switch (decoded.kind) {
    case MessageKind::READY:
      valid = decoder.String(decoded.buildId);
      break;
    case MessageKind::JOB:
      valid = DecodeJob(decoder, decoded);
      break;
    case MessageKind::RESULT:
    case MessageKind::JOB_ERROR:
      valid = DecodeReply(decoder, decoded.reply);
      break;
    case MessageKind::SHUTDOWN:
      valid = true;
      break;
  }
  if (!valid || !decoder.Finished()) {
    return Fail(error, "malformed message payload or trailing bytes");
  }
  message = std::move(decoded);
  return true;
}

FramedChannel::FramedChannel(int fd) : fd_(fd) {
  const int flags = fcntl(fd_, F_GETFL);
  const int fdFlags = fcntl(fd_, F_GETFD);
  if (flags < 0 || fdFlags < 0 || fcntl(fd_, F_SETFL, flags | O_NONBLOCK) < 0 ||
      fcntl(fd_, F_SETFD, fdFlags | FD_CLOEXEC) < 0) {
    setupError_ = SystemError("configure IPC fd");
  }
}

FramedChannel::~FramedChannel() {
  Close();
}

FramedChannel::FramedChannel(FramedChannel &&other) noexcept
    : fd_(std::exchange(other.fd_, -1)),
      setupError_(std::move(other.setupError_)),
      rx_(std::move(other.rx_)),
      tx_(std::move(other.tx_)),
      txOffset_(other.txOffset_) {}

FramedChannel &FramedChannel::operator=(FramedChannel &&other) noexcept {
  if (this != &other) {
    Close();
    fd_ = std::exchange(other.fd_, -1);
    setupError_ = std::move(other.setupError_);
    rx_ = std::move(other.rx_);
    tx_ = std::move(other.tx_);
    txOffset_ = other.txOffset_;
  }
  return *this;
}

bool FramedChannel::Queue(const Message &message, std::string &error) {
  if (fd_ < 0 || !setupError_.empty()) {
    return Fail(error, setupError_.empty() ? "IPC fd is closed" : setupError_);
  }
  if (WantsWrite()) {
    return Fail(error, "previous IPC frame is still pending");
  }
  txOffset_ = 0;
  return EncodeMessage(message, tx_, error);
}

IoStatus FramedChannel::PumpRead(std::vector<Message> &messages, std::string &error) {
  error.clear();
  if (fd_ < 0 || !setupError_.empty()) {
    error = setupError_.empty() ? "IPC fd is closed" : setupError_;
    return IoStatus::Error;
  }
  std::array<unsigned char, READ_BUFFER_BYTES> buffer{};
  size_t processed = 0;
  while (processed < IO_BYTES_PER_PUMP) {
    const auto size = recv(fd_, buffer.data(), std::min(buffer.size(), IO_BYTES_PER_PUMP - processed), 0);
    if (size > 0) {
      processed += static_cast<size_t>(size);
      rx_.insert(rx_.end(), buffer.begin(), buffer.begin() + size);
      if (!ExtractMessages(messages, error)) {
        return IoStatus::Error;
      }
      continue;
    }
    if (size == 0) {
      if (!rx_.empty()) {
        error = "IPC peer closed with a truncated frame";
        return IoStatus::Error;
      }
      return IoStatus::PeerClosed;
    }
    if (errno == EINTR) {
      continue;
    }
    if (errno == EAGAIN || errno == EWOULDBLOCK) {
      return processed == 0 ? IoStatus::WouldBlock : IoStatus::Progress;
    }
    error = SystemError("receive IPC frame");
    return IoStatus::Error;
  }
  return IoStatus::Progress;
}

IoStatus FramedChannel::PumpWrite(std::string &error) {
  error.clear();
  if (fd_ < 0 || !setupError_.empty()) {
    error = setupError_.empty() ? "IPC fd is closed" : setupError_;
    return IoStatus::Error;
  }
  size_t processed = 0;
  while (WantsWrite() && processed < IO_BYTES_PER_PUMP) {
    // 对端退出时让 send 返回错误，避免 SIGPIPE 终止当前进程。
    const auto size = send(fd_, tx_.data() + txOffset_, std::min(tx_.size() - txOffset_, IO_BYTES_PER_PUMP - processed),
                           MSG_NOSIGNAL);
    if (size > 0) {
      txOffset_ += static_cast<size_t>(size);
      processed += static_cast<size_t>(size);
      continue;
    }
    if (size < 0 && errno == EINTR) {
      continue;
    }
    if (size < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
      return processed == 0 ? IoStatus::WouldBlock : IoStatus::Progress;
    }
    if (size == 0 || errno == EPIPE || errno == ECONNRESET) {
      return IoStatus::PeerClosed;
    }
    error = SystemError("send IPC frame");
    return IoStatus::Error;
  }
  if (!WantsWrite()) {
    tx_.clear();
    txOffset_ = 0;
  }
  return processed == 0 ? IoStatus::WouldBlock : IoStatus::Progress;
}

bool FramedChannel::WantsWrite() const {
  return txOffset_ < tx_.size();
}

int FramedChannel::Fd() const {
  return fd_;
}

void FramedChannel::Close() {
  if (fd_ >= 0) {
    close(fd_);
    fd_ = -1;
  }
  rx_.clear();
  tx_.clear();
  txOffset_ = 0;
}

bool FramedChannel::ExtractMessages(std::vector<Message> &messages, std::string &error) {
  size_t consumed = 0;
  while (rx_.size() - consumed >= FRAME_LENGTH_BYTES) {
    uint32_t length = 0;
    for (size_t index = 0; index < FRAME_LENGTH_BYTES; ++index) {
      length |= static_cast<uint32_t>(rx_[consumed + index]) << (index * CHAR_BIT);
    }
    if (length < BODY_HEADER_BYTES || length > MAX_FRAME_BYTES) {
      return Fail(error, "IPC frame length is outside protocol budget");
    }
    if (rx_.size() - consumed - FRAME_LENGTH_BYTES < length) {
      // 保留尾部的不完整帧，下次 PumpRead 再接着解析。
      break;
    }
    Message message;
    const auto begin = rx_.begin() + consumed + FRAME_LENGTH_BYTES;
    if (!DecodeMessage(Bytes(begin, begin + length), message, error)) {
      return false;
    }
    messages.push_back(std::move(message));
    consumed += FRAME_LENGTH_BYTES + length;
  }
  rx_.erase(rx_.begin(), rx_.begin() + consumed);
  return true;
}

}  // namespace model_spec::compile
