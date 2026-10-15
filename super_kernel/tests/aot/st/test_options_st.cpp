/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <fstream>
#include <nlohmann/json.hpp>

#include "st_fixture.h"
#include "st_process.h"

namespace {
using Json = nlohmann::ordered_json;

class OptionsDumpDirectory {
 public:
  OptionsDumpDirectory() : previous_(std::filesystem::current_path()) {
    std::filesystem::current_path(directory_.Path());
    setenv("ASCEND_OP_COMPILE_SAVE_KERNEL_META", "1", 1);
  }
  ~OptionsDumpDirectory() {
    // Reinitialize the public logging lifecycle before deleting its files,
    // including when a fatal assertion exits a test early.
    unsetenv("ASCEND_OP_COMPILE_SAVE_KERNEL_META");
    sk::test::Model quiet;
    quiet.AddKernel(quiet.AddStream(), "quiet_options");
    EXPECT_EQ(aclskOptimize(quiet.Handle(), nullptr), ACL_SUCCESS);
    std::error_code error;
    std::filesystem::current_path(previous_, error);
  }

 private:
  sk::test::TemporaryDirectory directory_;
  std::filesystem::path previous_;
};

Json OptimizeAndReadOptions(aclskOptions &options, const char *expectedEntry = "sk_entry_aiv") {
  sk::test::Model model;
  const auto stream = model.AddStream();
  const auto first = model.AddKernel(stream, "options_first");
  const auto second = model.AddKernel(stream, "options_second");
  const auto dumpIndex = SkUtGetDebugJsonPrintCallCount();
  EXPECT_EQ(aclskOptimize(model.Handle(), &options), ACL_SUCCESS);
  EXPECT_EQ(model.SuccessfulUpdates(), 1U);
  EXPECT_NE(model.Snapshot(first).disabled, model.Snapshot(second).disabled);
  const auto entry = model.Snapshot(model.Snapshot(first).disabled ? second : first);
  EXPECT_EQ(entry.function, expectedEntry);
  EXPECT_FALSE(entry.args.empty());
  EXPECT_EQ(entry.setParamsCount, 1U);
  if (SkUtGetDebugJsonPrintCallCount() != dumpIndex + 2) {
    ADD_FAILURE() << "Missing original/updated graph dumps";
    return Json::object();
  }
  const auto directory = std::filesystem::path(SkUtGetDebugJsonPrintPath(dumpIndex)).parent_path();
  std::ifstream originalFile(directory / "sk_graph_origin.json");
  std::ifstream updatedFile(directory / "sk_graph_updated.json");
  const auto original = Json::parse(originalFile);
  const auto updated = Json::parse(updatedFile);
  EXPECT_EQ(original.at("options"), updated.at("options"));
  return original.at("options");
}

void ExpectOptionsLog(const std::string &message) {
  EXPECT_NE(ut_log::LogBuffer::Instance().GetContent().find(message), std::string::npos) << message;
}
}  // namespace

TEST_F(AotSystemTest, OptionsMalformedExtendDiscardsWholeMapAndPreservesFusion) {
  OptionsDumpDirectory directory;
  struct Case {
    std::string value;
    const char *warning;
  };
  const std::vector<Case> cases = {
      {" \t ", "raw extend value is empty after trim"},
      {std::string(1025, 'a'), "raw extend value is too long: 1025"},
      {"valid=one::next=two", "extend pair is empty"},
      {"valid=one:next", "extend pair format is invalid"},
      {"valid=one:=two", "extend pair format is invalid"},
      {"valid=one:next=", "extend pair format is invalid"},
      {"valid=one:next=two=three", "extend pair format is invalid"},
      {"valid=one:bad/key=two", "extend key is invalid"},
      {"valid=one: =two", "extend pair format is invalid"},
      {"valid=one:valid=two", "extend key is duplicated"},
      {"valid=one:next=two,", "extend value is invalid"},
      {"valid=one:next=bad@value", "extend value is invalid"},
  };
  for (const auto type : {aclskOptionType::OPT_EXTEND_OPTION, aclskOptionType::DEBUG_EXTEND_OPTION}) {
    const char *name = type == aclskOptionType::OPT_EXTEND_OPTION ? "opt_extend_option" : "debug_extend_option";
    for (auto item : cases) {
      SCOPED_TRACE(name);
      SCOPED_TRACE(item.value);
      ut_log::LogBuffer::Instance().Clear();
      aclskOption option{};
      option.optionType = type;
      if (type == aclskOptionType::OPT_EXTEND_OPTION) {
        option.optExtend.value = item.value.data();
      } else {
        option.debugExtend.value = item.value.data();
      }
      aclskOptions options{&option, 1};
      EXPECT_EQ(OptimizeAndReadOptions(options).at(name).at("value"), Json::object());
      ExpectOptionsLog(item.warning);
    }
    aclskOption nullOption{};
    nullOption.optionType = type;
    aclskOptions options{&nullOption, 1};
    ut_log::LogBuffer::Instance().Clear();
    EXPECT_EQ(OptimizeAndReadOptions(options).at(name).at("value"), Json::object());
    ExpectOptionsLog("raw extend value is nullptr");
  }
}

TEST_F(AotSystemTest, OptionsExtendPersistsTrimmedListsAndMaximumLength) {
  OptionsDumpDirectory directory;
  char debug[] = "  trace-mode.v1 = /tmp/trace-one , trace_two : enabled = 1  ";
  std::string maximum = "key=" + std::string(1020, 'a');
  aclskOption values[2]{};
  values[0].optionType = aclskOptionType::DEBUG_EXTEND_OPTION;
  values[0].debugExtend.value = debug;
  values[1].optionType = aclskOptionType::OPT_EXTEND_OPTION;
  values[1].optExtend.value = maximum.data();
  aclskOptions options{values, 2};
  const auto saved = OptimizeAndReadOptions(options);
  EXPECT_EQ(saved.at("debug_extend_option").at("value").at("trace-mode.v1"),
            Json::array({"/tmp/trace-one", "trace_two"}));
  EXPECT_EQ(saved.at("debug_extend_option").at("value").at("enabled"), Json::array({"1"}));
  EXPECT_EQ(saved.at("opt_extend_option").at("value").at("key"), Json::array({std::string(1020, 'a')}));
}

TEST_F(AotSystemTest, OptionsNumericBoundariesAndDuplicateFirstValuePersist) {
  OptionsDumpDirectory directory;
  for (uint32_t split = 1; split <= 4; ++split) {
    SCOPED_TRACE(split);
    aclskOption values[5]{};
    values[0].optionType = aclskOptionType::PRELOAD_CODE;
    values[0].preload.preloadMode = split % 2 == 0 ? 2 : 0;
    values[1].optionType = aclskOptionType::SPLIT_MODE;
    values[1].splitMode.splitCnt = split;
    values[2].optionType = aclskOptionType::AUTO_OP_PARALLEL;
    values[2].autoOpParallel.enableAutoOpParallel = 1;
    values[3].optionType = aclskOptionType::EARLY_START;
    values[3].earlyStart.enableEarlyStart = 1;
    values[4] = values[1];
    values[4].splitMode.splitCnt = 5;
    aclskOptions options{values, 5};
    ut_log::LogBuffer::Instance().Clear();
    const auto saved = OptimizeAndReadOptions(options, "sk_entry_aiv_early_start");
    EXPECT_EQ(saved.at("preload_code").at("value"), values[0].preload.preloadMode);
    EXPECT_EQ(saved.at("split_mode").at("value"), split);
    EXPECT_EQ(saved.at("auto_op_parallel").at("value"), 1);
    EXPECT_EQ(saved.at("early_start").at("value"), 1);
    ExpectOptionsLog("OptionName split_mode already parsed");
  }
}

TEST_F(AotSystemTest, OptionsInvalidNumbersFallBackToDocumentedDefaults) {
  OptionsDumpDirectory directory;
  aclskOption values[5]{};
  values[0].optionType = aclskOptionType::PRELOAD_CODE;
  values[0].preload.preloadMode = 3;
  values[1].optionType = aclskOptionType::SPLIT_MODE;
  values[1].splitMode.splitCnt = 0;
  values[2].optionType = aclskOptionType::AUTO_OP_PARALLEL;
  values[2].autoOpParallel.enableAutoOpParallel = 2;
  values[3].optionType = aclskOptionType::EARLY_START;
  values[3].earlyStart.enableEarlyStart = 2;
  values[4].optionType = aclskOptionType::AGGRESSIVE_OPT_STRATEGIES;
  values[4].aggressiveOpts = {0, 0, 2};
  aclskOptions options{values, 5};
  const auto saved = OptimizeAndReadOptions(options);
  EXPECT_EQ(saved.at("preload_code").at("value"), 1);
  EXPECT_EQ(saved.at("split_mode").at("value"), 4);
  EXPECT_EQ(saved.at("auto_op_parallel").at("value"), 0);
  EXPECT_EQ(saved.at("early_start").at("value"), 0);
  EXPECT_EQ(saved.at("aggressive_opt_strategies").at("value").at("taskBreakerBypass"), 0);
  ExpectOptionsLog("task_breaker_bypass");
}

TEST_F(AotSystemTest, OptionsKernelListsSkipNullEntriesAndRejectMissingArrays) {
  OptionsDumpDirectory directory;
  const std::vector<std::pair<aclskOptionType, const char *>> types = {
      {aclskOptionType::DCCI_DISABLE_ON_KERNEL, "dcci_disable_on_kernel"},
      {aclskOptionType::DCCI_BEFORE_KERNEL_START, "dcci_before_kernel_start"},
      {aclskOptionType::DCCI_AFTER_KERNEL_END, "dcci_after_kernel_end"},
      {aclskOptionType::UBUF_LOCK_IGNORE_KERNEL, "ubuf_lock_ignore_kernel"},
  };
  char pattern[] = "options_.*";
  char *names[] = {nullptr, pattern};
  for (const auto &[type, name] : types) {
    for (uint32_t mode = 0; mode < 3; ++mode) {
      SCOPED_TRACE(name);
      SCOPED_TRACE(mode);
      aclskOption option{};
      option.optionType = type;
      char **list = mode == 2 ? names : nullptr;
      const uint32_t count = mode == 0 ? 0 : 2;
      switch (type) {
        case aclskOptionType::DCCI_DISABLE_ON_KERNEL:
          option.disableKernelDcci = {list, count};
          break;
        case aclskOptionType::DCCI_BEFORE_KERNEL_START:
          option.dcciBeforeKernelStart = {list, count};
          break;
        case aclskOptionType::DCCI_AFTER_KERNEL_END:
          option.dcciAfterKernelEnd = {list, count};
          break;
        default:
          option.ubufLockIgnoreKernel = {count, list};
          break;
      }
      aclskOptions options{&option, 1};
      ut_log::LogBuffer::Instance().Clear();
      const auto saved = OptimizeAndReadOptions(options);
      EXPECT_EQ(saved.at(name).at("value"), mode == 2 ? Json::array({pattern}) : Json::array());
      ExpectOptionsLog(mode == 0 ? "value is empty" : mode == 1 ? "is nullptr while kernelCnt" : "is nullptr, skip");
      if (mode == 2 && type == aclskOptionType::DCCI_DISABLE_ON_KERNEL) {
        ExpectOptionsLog("Apply dcci_after_kernel_end fallback for options_first");
        ExpectOptionsLog("Apply dcci_after_kernel_end fallback for options_second");
      }
    }
  }
}

TEST_F(AotSystemTest, OptionsInvalidKernelPatternsWarnWithoutPreventingFusion) {
  OptionsDumpDirectory directory;
  for (const auto &[pattern, warning] :
       std::vector<std::pair<std::string, std::string>>{{" \t ", "pattern is empty after trim"},
                                                        {"options_[a-z]", "pattern contains invalid characters"},
                                                        {"*options", "invalid pattern starts with '*'"}}) {
    std::string input = pattern;
    char *names[] = {input.data()};
    aclskOption option{};
    option.optionType = aclskOptionType::DCCI_BEFORE_KERNEL_START;
    option.dcciBeforeKernelStart = {names, 1};
    aclskOptions options{&option, 1};
    ut_log::LogBuffer::Instance().Clear();
    const auto saved = OptimizeAndReadOptions(options);
    EXPECT_EQ(saved.at("dcci_before_kernel_start").at("value"), Json::array({pattern}));
    ExpectOptionsLog(warning);
  }
}
