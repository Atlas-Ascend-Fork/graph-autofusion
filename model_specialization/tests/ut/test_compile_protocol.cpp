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
#include <utility>
#include <gtest/gtest.h>

namespace model_spec::compile {
namespace {

Message JobMessage() {
  Message message;
  message.kind = MessageKind::JOB;
  message.jobId = 7;
  message.job.jobId = 7;
  message.job.resourceId = "resource";
  message.job.kernelEntry = "kernel";
  message.job.arguments = {{0, 1, 255}, {7, 0}};
  message.job.basicOptions = {"--first", "--second"};
  message.job.skOptions = {"--sk"};
  message.job.outputElfPath = "/data/hxj/task.elf";
  return message;
}

Bytes Body(const Bytes &frame) {
  return Bytes(frame.begin() + 4, frame.end());
}

TEST(CompileProtocolTest, JobRoundTripPreservesRequestBytesAndOptions) {
  const auto sent = JobMessage();
  Bytes frame;
  Message received;
  std::string error;
  ASSERT_TRUE(EncodeMessage(sent, frame, error)) << error;
  ASSERT_TRUE(DecodeMessage(Body(frame), received, error)) << error;
  EXPECT_EQ(received.kind, MessageKind::JOB);
  EXPECT_EQ(received.jobId, 7ULL);
  EXPECT_EQ(received.job.jobId, 7ULL);
  EXPECT_EQ(received.job.resourceId, sent.job.resourceId);
  EXPECT_EQ(received.job.kernelEntry, sent.job.kernelEntry);
  EXPECT_EQ(received.job.arguments, sent.job.arguments);
  EXPECT_EQ(received.job.basicOptions, sent.job.basicOptions);
  EXPECT_EQ(received.job.skOptions, sent.job.skOptions);
  EXPECT_EQ(received.job.outputElfPath, sent.job.outputElfPath);
}

TEST(CompileProtocolTest, HeaderUsesFixedLittleEndianFields) {
  Message message;
  message.kind = MessageKind::SHUTDOWN;
  message.jobId = 0x0102030405060708ULL;
  Bytes frame;
  std::string error;
  ASSERT_TRUE(EncodeMessage(message, frame, error));
  const Bytes expected = {16, 0, 0, 0, 3, 0, 0, 0, 5, 0, 0, 0, 8, 7, 6, 5, 4, 3, 2, 1};
  EXPECT_EQ(frame, expected);
}

TEST(CompileProtocolTest, ReadyAndResultReturnCodesRoundTrip) {
  for (const int32_t code : {0, -71}) {
    for (const auto kind : {MessageKind::READY, MessageKind::RESULT, MessageKind::JOB_ERROR}) {
      Message source;
      source.kind = kind;
      source.buildId = WORKER_BUILD_ID;
      source.jobId = 4;
      source.reply = {code, "compile result"};
      Bytes frame;
      std::string error;
      Message result;
      ASSERT_TRUE(EncodeMessage(source, frame, error)) << error;
      ASSERT_TRUE(DecodeMessage(Body(frame), result, error)) << error;
      EXPECT_EQ(result.kind, source.kind);
      EXPECT_EQ(result.jobId, source.jobId);
      if (kind == MessageKind::READY) {
        EXPECT_EQ(result.buildId, source.buildId);
      } else {
        EXPECT_EQ(result.reply.aclrtcCode, source.reply.aclrtcCode);
        EXPECT_EQ(result.reply.detail, source.reply.detail);
      }
    }
  }
}

TEST(CompileProtocolTest, RejectsOversizedInput) {
  auto message = JobMessage();
  Bytes frame;
  std::string error;
  message.job.arguments[0].assign(MAX_ARGUMENT_BYTES + 1, 0);
  EXPECT_FALSE(EncodeMessage(message, frame, error));
  message = JobMessage();
  message.job.resourceId.assign(MAX_STRING_BYTES + 1, 'x');
  EXPECT_FALSE(EncodeMessage(message, frame, error));
}

TEST(CompileProtocolTest, RejectsVersionKindTruncationTrailingBytesAndHugeCount) {
  Bytes frame;
  std::string error;
  ASSERT_TRUE(EncodeMessage(JobMessage(), frame, error));
  Message message;
  auto body = Body(frame);
  body[0] = 127;
  EXPECT_FALSE(DecodeMessage(body, message, error));
  body = Body(frame);
  body[4] = 127;
  EXPECT_FALSE(DecodeMessage(body, message, error));
  body = Body(frame);
  body.pop_back();
  EXPECT_FALSE(DecodeMessage(body, message, error));
  body = Body(frame);
  body.push_back(0);
  EXPECT_FALSE(DecodeMessage(body, message, error));
  body = Body(frame);
  // Header, resource string, entry string, then argument count.
  const size_t countOffset = 16 + 8 + 8 + 8 + 6;
  std::fill(body.begin() + countOffset, body.begin() + countOffset + 8, 255);
  EXPECT_FALSE(DecodeMessage(body, message, error));
}

TEST(CompileProtocolTest, FramedChannelHandlesFragmentedFramesAndCleanEof) {
  int sockets[2];
  ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets), 0);
  FramedChannel receiver(sockets[0]);
  EXPECT_NE(fcntl(receiver.Fd(), F_GETFL) & O_NONBLOCK, 0);
  EXPECT_NE(fcntl(receiver.Fd(), F_GETFD) & FD_CLOEXEC, 0);
  Bytes frame;
  std::string error;
  ASSERT_TRUE(EncodeMessage(JobMessage(), frame, error));
  ASSERT_EQ(write(sockets[1], frame.data(), 2), 2);
  std::vector<Message> messages;
  EXPECT_EQ(receiver.PumpRead(messages, error), IoStatus::Progress);
  EXPECT_TRUE(messages.empty());
  ASSERT_EQ(write(sockets[1], frame.data() + 2, frame.size() - 2), static_cast<ssize_t>(frame.size() - 2));
  close(sockets[1]);
  EXPECT_EQ(receiver.PumpRead(messages, error), IoStatus::PeerClosed);
  ASSERT_EQ(messages.size(), 1U);
  EXPECT_EQ(messages[0].jobId, 7ULL);
}

TEST(CompileProtocolTest, FramedChannelRejectsIncompleteOrOversizedFrames) {
  int sockets[2];
  ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets), 0);
  FramedChannel receiver(sockets[0]);
  const unsigned char prefix[] = {255, 255, 255, 127};
  ASSERT_EQ(write(sockets[1], prefix, sizeof(prefix)), 4);
  std::vector<Message> messages;
  std::string error;
  EXPECT_EQ(receiver.PumpRead(messages, error), IoStatus::Error);
  close(sockets[1]);

  ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets), 0);
  FramedChannel partial(sockets[0]);
  ASSERT_EQ(write(sockets[1], prefix, 2), 2);
  close(sockets[1]);
  EXPECT_EQ(partial.PumpRead(messages, error), IoStatus::Error);
}

TEST(CompileProtocolTest, QueuedLargeFrameMakesBoundedProgressAndRoundTrips) {
  int sockets[2];
  ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets), 0);
  FramedChannel sender(sockets[0]);
  FramedChannel receiver(sockets[1]);
  auto source = JobMessage();
  source.job.arguments[0].assign(IO_BYTES_PER_PUMP * 3, 42);
  std::string error;
  ASSERT_TRUE(sender.Queue(source, error));
  EXPECT_FALSE(sender.Queue(source, error));
  std::vector<Message> messages;
  for (int iteration = 0; iteration < 100 && messages.empty(); ++iteration) {
    EXPECT_NE(sender.PumpWrite(error), IoStatus::Error) << error;
    EXPECT_NE(receiver.PumpRead(messages, error), IoStatus::Error) << error;
  }
  ASSERT_EQ(messages.size(), 1U);
  EXPECT_EQ(messages[0].job.arguments, source.job.arguments);
  EXPECT_FALSE(sender.WantsWrite());
}

TEST(CompileProtocolTest, PeerCloseDuringSendDoesNotRaiseSigpipeAndMoveOwnsFd) {
  int sockets[2];
  ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets), 0);
  FramedChannel original(sockets[0]);
  FramedChannel moved(std::move(original));
  EXPECT_EQ(original.Fd(), -1);
  close(sockets[1]);
  std::string error;
  ASSERT_TRUE(moved.Queue(JobMessage(), error));
  EXPECT_EQ(moved.PumpWrite(error), IoStatus::PeerClosed);
  moved.Close();
  EXPECT_EQ(moved.Fd(), -1);
}

}  // namespace
}  // namespace model_spec::compile
