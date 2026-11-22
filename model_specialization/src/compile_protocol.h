/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MODEL_SPEC_COMPILE_PROTOCOL_H
#define MODEL_SPEC_COMPILE_PROTOCOL_H

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#ifndef MODEL_SPEC_WORKER_BUILD_ID
#define MODEL_SPEC_WORKER_BUILD_ID "model-spec-worker-dev-v1"
#endif

namespace model_spec::compile {

using Bytes = std::vector<unsigned char>;
inline constexpr uint32_t PROTOCOL_VERSION = 3;
inline constexpr const char *WORKER_BUILD_ID = MODEL_SPEC_WORKER_BUILD_ID;
inline constexpr uint64_t NO_JOB_ID = std::numeric_limits<uint64_t>::max();
inline constexpr size_t MAX_FRAME_BYTES = 64U * 1024U * 1024U;
inline constexpr size_t MAX_ARGUMENT_BYTES = 16U * 1024U * 1024U;
inline constexpr size_t MAX_STRING_BYTES = 64U * 1024U;
inline constexpr size_t MAX_ARGUMENT_COUNT = 65536;
inline constexpr size_t MAX_OPTION_COUNT = 4096;
// 限制单次读写的工作量，避免一个大帧独占父进程的 poll 循环。
inline constexpr size_t IO_BYTES_PER_PUMP = 256U * 1024U;

// 跨进程传递参数的字节快照；worker 用自己的存储重建 ACLRTC 请求中的指针。
struct CompileJob {
  uint64_t jobId = NO_JOB_ID;
  std::string resourceId;
  std::string kernelEntry;
  std::vector<Bytes> arguments;
  std::vector<std::string> basicOptions;
  std::vector<std::string> skOptions;
  std::string outputElfPath;
};

enum class MessageKind : uint32_t { READY = 1, JOB = 2, RESULT = 3, JOB_ERROR = 4, SHUTDOWN = 5 };

struct CompileReply {
  int32_t aclrtcCode = 0;
  std::string detail;
};

struct Message {
  MessageKind kind = MessageKind::SHUTDOWN;
  uint64_t jobId = NO_JOB_ID;
  std::string buildId;
  CompileJob job;
  CompileReply reply;
};

// 编码输出含长度前缀；解码只接收去掉前缀的帧体。
bool EncodeMessage(const Message &message, Bytes &fullFrame, std::string &error);
bool DecodeMessage(const Bytes &body, Message &message, std::string &error);

enum class IoStatus { Progress, WouldBlock, PeerClosed, Error };

class FramedChannel final {
 public:
  // 接管 fd，并将它设为非阻塞和 exec 时关闭。
  explicit FramedChannel(int fd);
  ~FramedChannel();
  FramedChannel(const FramedChannel &) = delete;
  FramedChannel &operator=(const FramedChannel &) = delete;
  FramedChannel(FramedChannel &&other) noexcept;
  FramedChannel &operator=(FramedChannel &&other) noexcept;

  // 通道只保留一帧待发送数据；短写后继续发送剩余部分。
  bool Queue(const Message &message, std::string &error);
  // PeerClosed 只表示读到了 EOF；本次调用仍可能已向 messages 追加完整帧。
  IoStatus PumpRead(std::vector<Message> &messages, std::string &error);
  IoStatus PumpWrite(std::string &error);
  bool WantsWrite() const;
  int Fd() const;
  void Close();

 private:
  bool ExtractMessages(std::vector<Message> &messages, std::string &error);
  int fd_ = -1;
  std::string setupError_;
  Bytes rx_;
  Bytes tx_;
  size_t txOffset_ = 0;
};

}  // namespace model_spec::compile
#endif
