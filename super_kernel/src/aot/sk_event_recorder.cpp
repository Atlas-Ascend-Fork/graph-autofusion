/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sk_event_recorder.h"
#include "sk_options_manager.h"
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <fstream>
#include "acl/acl.h"
#include "sk_log.h"
#include "aprof_pub.h"
#include "sk_file_guard.h"

#define CHECK_ACL_RETURN(x)                                 \
  do {                                                      \
    aclError __ret = x;                                     \
    if (__ret != ACL_ERROR_NONE) {                          \
      SK_LOGE("[sk time profiling] acl Error: %d.", __ret); \
      return;                                               \
    }                                                       \
  } while (0)

uint32_t SkEventRecorder::coreSize_ = SK_EVENT_DEFAULT_CORE_SIZE;
uint32_t SkEventRecorder::totalSize_ = 0;  // 运行时由 ParseEnvAndSetSize 设置

// ==================== Msprof 回调相关 ====================
constexpr size_t ASCENDC_KERNEL_ID = 69;

// 代表profiling开和关，初始值为0，每次回调一次，原子变量值就加1，然后非0的偶数表示关闭
static std::atomic<uint32_t> g_profSignal{0};

static int32_t AscendProfilingCallBack(uint32_t type, void *data, uint32_t len) {
  (void)data;
  (void)len;

  SK_LOGI("[sk time profiling] AscendProfilingCallBack triggered, type=%u\n", type);

  uint32_t val = g_profSignal.fetch_add(1, std::memory_order_relaxed) + 1;
  SK_LOGI("[sk time profiling] Prof signal updated to %u (even=%d)\n", val, (val % 2 == 0 && val != 0));

  return 0;
}

namespace {
void SetupProfilingModelContext(SkLaunchInfo &launchInfo, const SuperKernelScopeInfo &scopeInfo,
                                const std::string &modelId) {
  if (launchInfo.devArgs.Get() == nullptr) {
    SK_LOGI("[sk time profiling] devArgs is null, skip setting modelIdIndexAndSkScopeId");
    return;
  }

  uint16_t modelIdIdx = SkEventRecorder::Instance().RegisterModelId(modelId);
  uint16_t skScopeId = scopeInfo.GetScopeId();
  launchInfo.devArgs.Get()->skHeader.modelIdIndexAndSkScopeId =
      (static_cast<uint64_t>(modelIdIdx) << 32) | (static_cast<uint64_t>(skScopeId) << 16);
}

bool SetupProfilingRuntime(const std::vector<SuperKernelBaseNode *> &taskNodes, SkLaunchInfo &launchInfo,
                           const SuperKernelScopeInfo &scopeInfo, const std::string &modelId) {
  int32_t deviceId = 0;
  aclError aclRet = aclrtGetDevice(&deviceId);
  if (aclRet != ACL_SUCCESS) {
    SK_LOGE("[sk time profiling] Failed to get device id, ret=%d\n", aclRet);
    return false;
  }
  launchInfo.eventGmAddr = SkEventRecorder::Instance().GetGmAddrForDevice(deviceId);
  // 获取gm地址失败后，只通知 kernel 侧禁用 profiling（enabled=0），不中断后续host调度流程
  const bool profilingEnabled = (launchInfo.eventGmAddr != nullptr);
  if (!profilingEnabled) {
    SK_LOGE("[sk time profiling] Failed to get event GM address\n");
  }

  launchInfo.modelIdIndex = SkEventRecorder::Instance().RegisterModelId(modelId);
  launchInfo.skId = static_cast<uint32_t>(scopeInfo.GetScopeId());
  SK_LOGI("[sk time profiling] Event recording %s, gm_addr=%p, modelId=%s, skId=%u\n",
          profilingEnabled ? "enabled" : "disabled", launchInfo.eventGmAddr, modelId.c_str(), launchInfo.skId);
  // 更新 devArgs 中的事件配置（通过 profilingEnabled 控制 enabled 值，防止kernel侧往错误的地址写入）
  if (launchInfo.devArgs.Get() != nullptr && launchInfo.devArgs.Get()->skHeader.eventConfigOffset != 0) {
    uint8_t *base = reinterpret_cast<uint8_t *>(launchInfo.devArgs.Get());
    SkEventConfig *eventConfig =
        reinterpret_cast<SkEventConfig *>(base + launchInfo.devArgs.Get()->skHeader.eventConfigOffset);
    eventConfig->eventGmAddr = reinterpret_cast<uint64_t>(launchInfo.eventGmAddr);
    eventConfig->modelIdIndex = launchInfo.modelIdIndex;
    eventConfig->skId = launchInfo.skId;
    eventConfig->enabled = profilingEnabled ? 1 : 0;
    eventConfig->coreSize = SkEventRecorder::Instance().GetCoreSize();
  }
  SK_LOGI("[sk time profiling] Event recording %s, gm_addr=%p, modelId=%lu, skId=%u\n",
          profilingEnabled ? "enabled" : "disabled", launchInfo.eventGmAddr, launchInfo.modelIdIndex, launchInfo.skId);
  // 建立 modelId -> skId -> nodeId -> (nodeName, numBlocks) 映射
  for (size_t nodeId = 0; nodeId < taskNodes.size(); ++nodeId) {
    SuperKernelBaseNode *node = taskNodes[nodeId];
    if (node != nullptr && node->GetNodeType() == SkNodeType::NODE_KERNEL) {
      const NodeInfos &nodeInfos = node->GetNodeInfos();
      const std::string &funcName = nodeInfos.kernelInfos.funcName;
      SkEventRecorder::Instance().AddNodeInfoMapping(modelId, launchInfo.skId, static_cast<uint32_t>(nodeId), funcName,
                                                     nodeInfos.kernelInfos.numBlocks);
    }
  }
  SkEventRecorder::Instance().AddSkNameMapping(modelId, launchInfo.skId, launchInfo.skFuncName);
  return true;
}
}  // namespace

SkEventRecorder &SkEventRecorder::Instance() {
  static SkEventRecorder instance;
  return instance;
}

SkEventRecorder::~SkEventRecorder() {
  SkProfilingShutdown();
}

std::string SkEventRecorder::CreateOutputDir() {
  // 获取当前进程 pid
  pid_t pid = getpid();

  // 检查并创建 sk_meta 文件夹
  const char *skMetaDir = "sk_meta";
  struct stat st;
  if (stat(skMetaDir, &st) != 0) {
    // sk_meta 不存在，创建它
    if (mkdir(skMetaDir, SK_DIRECTORY_MODE) != 0 && errno != EEXIST) {
      SK_LOGE("[sk time profiling] Failed to create sk_meta directory, errno=%d\n", errno);
      return "";
    }
    SK_LOGI("[sk time profiling] Created sk_meta directory\n");
  }

  // 创建 ./sk_meta/<pid> 文件夹
  char pidDir[SPRINT_LEN_BUFFER];
  errno_t snpRet = snprintf_s(pidDir, sizeof(pidDir), sizeof(pidDir) - 1, "./%s/%d", skMetaDir, pid);
  if (snpRet < 0) {
    SK_LOGE("[sk time profiling] snprintf_s pidDir failed, ret=%d\n", snpRet);
    return "";
  }

  if (stat(pidDir, &st) != 0) {
    // sk_meta/<pid> 不存在，创建它
    if (mkdir(pidDir, SK_DIRECTORY_MODE) != 0 && errno != EEXIST) {
      SK_LOGE("[sk time profiling] Failed to create pid directory %s, errno=%d\n", pidDir, errno);
      return "";
    }
    SK_LOGI("[sk time profiling] Created pid directory: %s\n", pidDir);
  }

  return std::string(pidDir);
}

bool SkEventRecorder::Init() {
  // 已经初始化过，直接返回
  if (enabled.load()) {
    SK_LOGI("[sk time profiling] Dump the time of superkernel has already been initialized, skip re-initialization");
    return true;
  }
  std::call_once(initFlag_, [this]() {
    SK_LOGI("[sk time profiling] ===================== Start dump the time of superkernel =======================");
    // 检查打点环境变量是否开启（值代表每个core申请了几KB的空间，0表示关闭）
    if (!ParseEnvAndSetSize()) {
      return;
    }

    globalRunning.store(true);
    aclError aclRet = aclrtGetDevice(&dumpDeviceId);
    if (aclRet != ACL_SUCCESS) {
      SK_LOGE("[sk time profiling] Failed to get device id, ret=%d\n", aclRet);
      globalRunning.store(false);
      return;
    }

    // 启动单个全局后台线程用于搬运解析记录事件
    int ret = pthread_create(&dumpThread, nullptr, DumpThreadFunc, this);
    if (ret != 0) {
      SK_LOGE("[sk time profiling] Failed to create dump thread, ret=%d\n", ret);
      globalRunning.store(false);
      return;
    }
    // 所有初始化完成后再标记为启用，确保其他线程看到 enabled==true 时一切就绪
    enabled.store(true);

    // 注册 Msprof 回调，profiling 结束时获取输出路径
    int32_t profRet = MsprofRegisterCallback(ASCENDC_KERNEL_ID, AscendProfilingCallBack);
    if (profRet != 0) {
      SK_LOGW("[sk time profiling] MsprofRegisterCallback failed, ret=%d\n", profRet);
    } else {
      SK_LOGI("[sk time profiling] MsprofRegisterCallback success\n");
    }

    SK_LOGI("[sk time profiling] Event recorder enabled with single dump thread\n");
  });
  return enabled.load();
}

bool SkEventRecorder::ParseEnvAndSetSize() {
  // 检查打点环境变量是否开启（值代表每个core申请了几KB的空间，0表示关闭）
  const char *env = std::getenv(ENV_SK_EVENT_RECORD);
  if (env == nullptr || std::string(env) == "0") {
    SK_LOGI("[sk time profiling] ASCEND_PROF_SK_ON is not set or is 0, profiling disabled\n");
    return false;
  }

  // 解析环境变量值为 KB
  char *end = nullptr;
  long val = std::strtol(env, &end, 10);
  if (end == env || *end != '\0' || val <= 0) {
    SK_LOGE("[sk time profiling] Invalid ASCEND_PROF_SK_ON value: %s, expected a positive number (KB)\n", env);
    return false;
  }
  if (val > 1024 * 5) {  // 超过 5MB 的 coreSize 不合理，可能是输入错误
    SK_LOGE("[sk time profiling] Invalid ASCEND_PROF_SK_ON value is too large: %ld\n", val);
    return false;
  }
  if (val > 1024) {  // 超过 1MB 的 coreSize 没必要，提示用户不需要这么大
    SK_LOGW(
        "[sk time profiling] It is not recommended to set ASCEND_PROF_SK_ON above 1024; allocating excessive profiling "
        "buffer may be unnecessary.\n");
  }

  // 向上取整到 64KB 的倍数（输入单位为 KB，对齐到 64）
  uint32_t coreSizeKB = static_cast<uint32_t>((val + 63U) / 64U * 64U);
  coreSize_ = coreSizeKB * 1024U;
  totalSize_ = GetSkRuntimeConfig().eventCoreNum * coreSize_;

  SK_LOGI(
      "[sk time profiling] ASCEND_PROF_SK_ON=%s, coreSize=%u KB (aligned), totalSize=%u MB for each device profiling\n",
      env, coreSizeKB, totalSize_ / (1024U * 1024U));
  return true;
}

void *SkEventRecorder::GetGmAddrForDevice(uint32_t deviceId) {
  SK_LOGI("[sk time profiling] Start getting device gm addr for device %u\n", deviceId);
  if (!enabled.load() || deviceId >= SK_EVENT_MAX_DEVICE_NUM) {
    SK_LOGE("[sk time profiling] Printing has not started or deviceId %u is out of range\n", deviceId);
    SK_LOGE("[sk time profiling] End get device gm addr on device: %u\n", deviceId);
    return nullptr;
  }

  SkEventDeviceCtx *ctx = &deviceCtxs;

  // Double-check locking：确保每个 device 只初始化一次
  if (ctx->active.load()) {
    SK_LOGI("[sk time profiling] End get device gm addr on device: %u, addr: %p\n", deviceId, ctx->gmAddr.get());
    return ctx->gmAddr.get();
  }

  std::lock_guard<std::mutex> lock(mutex);
  if (ctx->active.load() || ctx->gmAddr != nullptr) {
    SK_LOGI("[sk time profiling] Device gm addr already exists for device %u, addr: %p\n", deviceId, ctx->gmAddr.get());
    SK_LOGI("[sk time profiling] End get device gm addr on device: %u, addr: %p\n", deviceId, ctx->gmAddr.get());
    return ctx->gmAddr.get();
  } else {
    SK_LOGI("[sk time profiling] Device buffer not allocated yet, allocating now for device %u\n", deviceId);
    // 创建上下文
    ctx = CreateDeviceCtx(deviceId);
    if (ctx == nullptr) {
      SK_LOGE("[sk time profiling] Failed to create device ctx for device %u\n", deviceId);
      SK_LOGI("[sk time profiling] End get device gm addr on device: %u\n", deviceId);
      return nullptr;
    }
    SK_LOGI("[sk time profiling] Device gm addr created for device: %u, addr: %p\n", deviceId, ctx->gmAddr.get());
    SK_LOGI("[sk time profiling] End get device gm addr on device: %u\n", deviceId);
  }

  return ctx ? ctx->gmAddr.get() : nullptr;
}

bool SkEventRecorder::InitDeviceOutputFile(SkEventDeviceCtx *ctx, uint32_t deviceId) {
  char filename[SPRINT_LEN_BUFFER];
  errno_t snpRet = snprintf_s(filename, sizeof(filename), sizeof(filename) - 1, "%s/sk_prof_device_%u.json",
                              ctx->outputDir.c_str(), deviceId);
  if (snpRet < 0) {
    SK_LOGE("[sk time profiling] snprintf_s filename failed for device %u, ret=%d\n", deviceId, snpRet);
    SkProfilingShutdown();
    return false;
  }

  if (!ctx->outputFp.Open(filename, "w+b")) {
    return true;
  }

  const char *jsonStart = "[{}]";
  size_t written = fwrite(jsonStart, 1, strlen(jsonStart), ctx->outputFp.Get());
  if (written != strlen(jsonStart)) {
    SK_LOGE("[sk time profiling] Failed to write JSON start to file\n");
    SkProfilingShutdown();
    return false;
  }

  return true;
}

SkEventDeviceCtx *SkEventRecorder::CreateDeviceCtx(uint32_t deviceId) {
  SkEventDeviceCtx *ctx = &deviceCtxs;
  ctx->recorder = this;  // 设置回调指针

  // 1. 初始化输出目录：创建 sk_meta/<pid> 文件夹
  ctx->outputDir = CreateOutputDir();
  if (ctx->outputDir.empty()) {
    SK_LOGE("[sk time profiling] Failed to create output directory for device %u\n", deviceId);
    SkProfilingShutdown();
    return nullptr;
  }

  // 2. 分配 GM 内存
  void *rawGmAddr = nullptr;
  aclError allocRet = aclrtMalloc(&rawGmAddr, totalSize_, ACL_MEM_MALLOC_HUGE_FIRST);
  if (allocRet != ACL_SUCCESS || rawGmAddr == nullptr) {
    SK_LOGE("[sk time profiling] Failed to malloc GM for device %u, ret=%d\n", deviceId, allocRet);
    SkProfilingShutdown();
    return nullptr;
  }
  ctx->gmAddr.reset(rawGmAddr);
  allocRet = aclrtMemset(rawGmAddr, totalSize_, 0, totalSize_);
  if (allocRet != ACL_SUCCESS) {
    SK_LOGE("[sk time profiling] Failed to memset GM for device %u, ret=%d\n", deviceId, allocRet);
    SkProfilingShutdown();
    return nullptr;
  }
  SK_LOGI("[sk time profiling] Malloc device gm addr, device %u, addr: %p\n", deviceId, ctx->gmAddr.get());

  // 3. 分配 host 缓冲区（使用智能指针，RAII 管理）
  ctx->hostBuf = std::make_unique<uint8_t[]>(totalSize_);
  if (ctx->hostBuf == nullptr) {
    SK_LOGE("[sk time profiling] Failed to allocate host buf for device %u\n", deviceId);
    SkProfilingShutdown();
    return nullptr;
  }
  SK_LOGI("[sk time profiling] Malloc host buf addr, device %u, addr: %p\n", deviceId, ctx->hostBuf.get());
  errno_t memRet = memset_s(ctx->hostBuf.get(), totalSize_, 0, totalSize_);
  if (memRet != EOK) {
    SK_LOGE("[sk time profiling] memset_s hostBuf failed for device %u, ret=%d\n", deviceId, memRet);
    SkProfilingShutdown();
    return nullptr;
  }
  // 4. 创建临时输出文件（用于存储 node 事件）
  if (!InitDeviceOutputFile(ctx, deviceId)) {
    return nullptr;
  }

  // 5. 初始化host偏移量数组
  ctx->lastOffset.assign(GetSkRuntimeConfig().eventCoreNum, sizeof(SkKernelEventCoreBuf));

  // 6. 设置该device基本信息并标记为激活
  ctx->deviceId = deviceId;
  ctx->totalSize = totalSize_;
  ctx->active.store(1);

  SK_LOGI("[sk time profiling] Created context for device %u, GM addr=%p\n", deviceId, ctx->gmAddr.get());

  return ctx;
}

void SkEventRecorder::SetProfSignal(uint32_t val) {
  g_profSignal.store(val, std::memory_order_relaxed);
}

void SkEventRecorder::CopyOutputToProfPath(SkEventDeviceCtx *ctx) {
  if (!ctx->outputFp.IsValid()) {
    SK_LOGI("[sk time profiling] Output file is not valid, skip copy\n");
    return;
  }

  // 获取并缓存 profiling 路径
  {
    std::lock_guard<std::mutex> lock(profBasePathMutex);
    if (profBasePath.empty()) {
      std::string path = GetBasePath();
      if (!path.empty()) {
        profBasePath = std::move(path);
        SK_LOGI("[sk time profiling] Cached profiling base path: %s\n", profBasePath.c_str());
      }
    }
  }
  if (profBasePath.empty()) {
    SK_LOGW("[sk time profiling] No profiling base path available, skip copy output file\n");
    return;
  }

  // 先 flush 源文件
  fflush(ctx->outputFp.Get());

  // 构造源文件和目标文件路径
  std::string srcFilename = ctx->outputDir + "/sk_prof_device_" + std::to_string(ctx->deviceId) + ".json";
  std::string dstFilename = profBasePath + "/sk_prof_device_" + std::to_string(ctx->deviceId) + ".json";

  // 使用 C++ fstream 复制文件
  std::ifstream src(srcFilename, std::ios::binary);
  if (!src.is_open()) {
    SK_LOGE("[sk time profiling] Failed to open source file for copy: %s\n", srcFilename.c_str());
    return;
  }

  std::ofstream dst(dstFilename, std::ios::binary);
  if (!dst.is_open()) {
    SK_LOGE("[sk time profiling] Failed to open destination file for copy: %s\n", dstFilename.c_str());
    return;
  }

  dst << src.rdbuf();

  if (!dst) {
    SK_LOGE("[sk time profiling] Failed to copy file content\n");
  } else {
    SK_LOGI("[sk time profiling] Copied output file from %s to %s\n", srcFilename.c_str(), dstFilename.c_str());
  }
}

void *SkEventRecorder::DumpThreadFunc(void *arg) {
  SkEventRecorder *recorder = static_cast<SkEventRecorder *>(arg);
  SK_LOGI("[sk time profiling] New global dump thread setdevice: %d\n", recorder->dumpDeviceId);
  aclError result = aclrtSetDevice(recorder->dumpDeviceId);
  if (result != 0) {
    SK_LOGE("[sk time profiling] Acl set device failed, ERROR: %ld, deviceId: %d\n", result, recorder->dumpDeviceId);
    return nullptr;
  }
  SK_LOGI("[sk time profiling] Global dump thread started write on deviceId: %d\n", recorder->dumpDeviceId);
  while (recorder->globalRunning.load()) {
    // 处理激活的 device
    SkEventDeviceCtx *ctx = &recorder->deviceCtxs;
    if (ctx->active.load()) {
      // 写入node的耗时信息
      recorder->DumpDeviceData(ctx);
    }
    // 检查 profiling 关闭信号（非0偶数）
    uint32_t sig = g_profSignal.load(std::memory_order_relaxed);
    if (sig != 0 && (sig % 2 == 0)) {
      recorder->CopyOutputToProfPath(ctx);
      // 重置信号，避免重复复制
      g_profSignal.store(0, std::memory_order_relaxed);
    }
    usleep(100000);  // 100ms 轮询间隔
  }
  // 最后一次刷新所有 device
  SkEventDeviceCtx *ctx = &recorder->deviceCtxs;
  if (ctx->active.load()) {
    recorder->DumpDeviceData(ctx);
  }

  // 终止读取时再检查 profiling 关闭信号, 如果没有检测到profiling关闭信号，就在线程结束复制一份
  uint32_t sig = g_profSignal.load(std::memory_order_relaxed);
  if (sig != 0) {
    recorder->CopyOutputToProfPath(ctx);
    g_profSignal.store(0, std::memory_order_relaxed);
    SK_LOGI("[sk time profiling] The end signal of profiling missing, re-copying\n");
  }

  // 如果profiling路径下的json文件为空, 复制完整版json到profiling路径
  {
    std::string cachedProfBasePath;
    {
      std::lock_guard<std::mutex> lock(recorder->profBasePathMutex);
      cachedProfBasePath = recorder->profBasePath;
    }
    if (!cachedProfBasePath.empty()) {
      std::string dstFile = cachedProfBasePath + "/sk_prof_device_" + std::to_string(ctx->deviceId) + ".json";
      struct stat fileStat;
      // 检查目标 json 文件是否存在且大于 10 字节
      if (stat(dstFile.c_str(), &fileStat) != 0 || fileStat.st_size <= 10) {
        SK_LOGI("[sk time profiling] Target json file missing or too small (%s), re-copying\n", dstFile.c_str());
        recorder->CopyOutputToProfPath(ctx);
      }
    }
  }

  SK_LOGI("[sk time profiling] Global dump thread stopped\n");
  return nullptr;
}

bool SkEventRecorder::WriteNodeEventToJson(SkEventDeviceCtx *ctx, const SkKernelEventRecord *record, uint32_t core,
                                           const SkNodeInfo &nodeInfo) {
  if (!ctx->outputFp.IsValid()) {
    SK_LOGE("[sk time profiling] Failed to open the output file\n");
    return false;  // 文件未打开，跳过写入
  }

  std::string modelId = SkEventRecorder::Instance().GetModelIdByIndex(static_cast<uint16_t>(record->modelIdIndex));

  if (record->endTime > record->startTime) {
    // 移动到 "}]" 之前
    if (fseek(ctx->outputFp.Get(), -2, SEEK_END) != 0) {
      SK_LOGE("[sk time profiling] Failed to seek in output file\n");
      ctx->outputFp.Close();
      SkProfilingShutdown();
      return false;
    }

    double tsStart = (double)record->startTime / GetSkRuntimeConfig().tickUsMultiplier;
    double tsEnd = (double)record->endTime / GetSkRuntimeConfig().tickUsMultiplier;
    char jsonLine[SPRINT_LEN_BUFFER];
    int len = snprintf_s(jsonLine, sizeof(jsonLine), sizeof(jsonLine) - 1,
                         "\"ph\":\"X\",\"name\":\"[%u/%u]%s\",\"pid\":\"%s\",\"tid\":%u,"
                         "\"ts\":%f,\"dur\":%f,\"args\":{\"modelId\":\"%s\",\"skId\":%u,\"nodeId\":%u}},\n{}]",
                         record->blockIdx, record->blockNum, nodeInfo.nodeName.c_str(), CoreIsAiv(core) ? "AIV" : "AIC",
                         core, tsStart, (tsEnd - tsStart), modelId.c_str(), record->skId, record->nodeId);

    if (len < 0) {
      SK_LOGE("[sk time profiling] snprintf_s failed for JSON line, modelId=%s, skId=%u, nodeId=%u\n", modelId.c_str(),
              record->skId, record->nodeId);
      ctx->outputFp.Close();
      SkProfilingShutdown();
      return false;
    } else if (len > 0 && len < sizeof(jsonLine)) {
      size_t written = fwrite(jsonLine, 1, len, ctx->outputFp.Get());
      if (written != static_cast<size_t>(len)) {
        SK_LOGE("[sk time profiling] Failed to write JSON line\n");
        ctx->outputFp.Close();
        SkProfilingShutdown();
        return false;
      }
    } else {
      SK_LOGW("[sk time profiling] JSON line too long or truncated, modelId=%s, skId=%u, nodeName=%s,len=%d\n",
              modelId.c_str(), record->skId, nodeInfo.nodeName.c_str(), len);
    }
  } else {
    SK_LOGW(
        "[sk time profiling] Invalid event record with endTime <= startTime, "
        "modelId=%s, skId=%u, nodeId=%u, nodeInfo.nodeName=%s\n",
        modelId.c_str(), record->skId, record->nodeId, nodeInfo.nodeName.c_str());
  }

  return true;
}

bool SkEventRecorder::WriteSkEventToJson(SkEventDeviceCtx *ctx, const SkKernelEventRecord *record, uint32_t core) {
  if (!ctx->outputFp.IsValid()) {
    SK_LOGE("[sk time profiling] SK event failed to open the output file\n");
    return false;  // 文件未打开，跳过写入
  }

  std::string modelId = SkEventRecorder::Instance().GetModelIdByIndex(static_cast<uint16_t>(record->modelIdIndex));

  if (record->endTime > record->startTime) {
    // 移动到 "}]" 之前
    if (fseek(ctx->outputFp.Get(), -2, SEEK_END) != 0) {
      SK_LOGE("[sk time profiling] SK event failed to seek in output file\n");
      ctx->outputFp.Close();
      SkProfilingShutdown();
      return false;
    }

    double tsStart = (double)record->startTime / GetSkRuntimeConfig().tickUsMultiplier;
    double tsEnd = (double)record->endTime / GetSkRuntimeConfig().tickUsMultiplier;
    char jsonLine[SPRINT_LEN_BUFFER];
    std::string skName = SkEventRecorder::Instance().GetSkName(modelId, record->skId);
    int len = snprintf_s(jsonLine, sizeof(jsonLine), sizeof(jsonLine) - 1,
                         "\"ph\":\"X\",\"name\":\"[%u/%u] %s\",\"pid\":\"%s\",\"tid\":%u,"
                         "\"ts\":%f,\"dur\":%f,\"args\":{\"modelId\":\"%s\",\"skId\":%u}},\n{}]",
                         record->blockIdx, record->blockNum, skName.c_str(), CoreIsAiv(core) ? "AIV" : "AIC", core,
                         tsStart, (tsEnd - tsStart), modelId.c_str(), record->skId);

    if (len < 0) {
      SK_LOGE("[sk time profiling] SK event snprintf_s failed for JSON line, modelId=%s, skId=%u", modelId.c_str(),
              record->skId);
      ctx->outputFp.Close();
      SkProfilingShutdown();
      return false;
    } else if (len > 0 && len < sizeof(jsonLine)) {
      size_t written = fwrite(jsonLine, 1, len, ctx->outputFp.Get());
      if (written != static_cast<size_t>(len)) {
        SK_LOGE("[sk time profiling] SK event failed to write sk event JSON line\n");
        ctx->outputFp.Close();
        SkProfilingShutdown();
        return false;
      }
    } else {
      SK_LOGW("[sk time profiling] sk event JSON line too long or truncated, modelId=%s, skId=%u, len=%d\n",
              modelId.c_str(), record->skId, len);
    }
  } else {
    SK_LOGW("[sk time profiling] Invalid event record with endTime <= startTime, modelId=%s, skId=%u, skName=%s\n",
            modelId.c_str(), record->skId, SkEventRecorder::Instance().GetSkName(modelId, record->skId).c_str());
  }

  return true;
}

bool SkEventRecorder::DumpEventRecord(SkEventDeviceCtx *ctx, const SkKernelEventRecord *record, uint32_t core) {
  if (record->nodeId != UINT32_MAX) {
    std::string modelId = SkEventRecorder::Instance().GetModelIdByIndex(static_cast<uint16_t>(record->modelIdIndex));
    SkNodeInfo nodeInfo = SkEventRecorder::Instance().GetNodeInfo(modelId, record->skId, record->nodeId);
    if (nodeInfo.nodeName != "" && !WriteNodeEventToJson(ctx, record, core, nodeInfo)) {
      SK_LOGE("[sk time profiling] Failed to write node event to json, device %u\n", ctx->deviceId);
      return false;
    }
    return true;
  }

  if (!WriteSkEventToJson(ctx, record, core)) {
    SK_LOGE("[sk time profiling] Failed to write sk event to json, device %u\n", ctx->deviceId);
    return false;
  }
  return true;
}

void SkEventRecorder::DumpDeviceData(SkEventDeviceCtx *ctx) {
  if (ctx->gmAddr == nullptr || ctx->hostBuf == nullptr) {
    return;
  }

  aclmdlRICaptureMode mode =
      ACL_MODEL_RI_CAPTURE_MODE_RELAXED;  // support CAPTURE MODE GLOBAL on host, when using device printf
  CHECK_ACL_RETURN(aclmdlRICaptureThreadExchangeMode(&mode));
  CHECK_ACL_RETURN(
      aclrtMemcpy(ctx->hostBuf.get(), ctx->totalSize, ctx->gmAddr.get(), ctx->totalSize, ACL_MEMCPY_DEVICE_TO_HOST));
  CHECK_ACL_RETURN(aclmdlRICaptureThreadExchangeMode(&mode));

  uint8_t *hostBuf = ctx->hostBuf.get();
  bool hasNewData = false;

  // 遍历每个 core 的数据
  uint32_t coreNum = static_cast<uint32_t>(ctx->lastOffset.size());
  for (uint32_t core = 0; core < coreNum; core++) {
    SK_LOGI("[sk time profiling] Wait 100 ms then start add some node event on device %u, core %u\n", ctx->deviceId,
            core);
    SkKernelEventCoreBuf *coreBuf = reinterpret_cast<SkKernelEventCoreBuf *>(hostBuf + core * coreSize_);

    uint32_t curOffset = coreBuf->offset;
    uint64_t lastOff = ctx->lastOffset[core];

    // 读取新数据
    while (lastOff + sizeof(SkKernelEventRecord) <= curOffset && lastOff + sizeof(SkKernelEventRecord) <= coreSize_) {
      SkKernelEventRecord *record = reinterpret_cast<SkKernelEventRecord *>(hostBuf + core * coreSize_ + lastOff);
      if (!DumpEventRecord(ctx, record, core)) {
        return;
      }
      hasNewData = true;
      lastOff += sizeof(SkKernelEventRecord);
    }

    ctx->lastOffset[core] = lastOff;
    if (lastOff + sizeof(SkKernelEventRecord) > coreSize_) {
      // 缓冲区已满，后续不再打印
      SK_LOGW(
          "[sk time profiling]  device %u core %u buffer is full, stop dump the time of nodes on device %u core %u\n",
          ctx->deviceId, core, ctx->deviceId, core);
    }
  }

  if (hasNewData && ctx->outputFp.IsValid()) {
    fflush(ctx->outputFp.Get());
  }
}

void SkEventRecorder::SkProfilingShutdown() {
  if (!enabled.load()) {
    return;
  }

  // 停止全局后台线程
  globalRunning.store(false);
  pthread_join(dumpThread, nullptr);

  // 释放所有资源
  SkEventDeviceCtx *ctx = &deviceCtxs;
  if (ctx->active.load()) {
    // hostBuf unique_ptr 自动释放，无需手动 free
    ctx->hostBuf.reset();
    ctx->active.store(0);
  }

  enabled.store(false);
}

void SkEventRecorder::AddNodeInfoMapping(const std::string &modelId, uint32_t skId, uint32_t nodeId,
                                         const std::string &nodeName, uint32_t numBlocks) {
  std::lock_guard<std::mutex> lock(nodeInfoMapMutex);
  SkNodeInfo info;
  info.nodeName = nodeName;
  info.numBlocks = numBlocks;
  nodeInfoMap[modelId][skId][nodeId] = info;
}

SkNodeInfo SkEventRecorder::GetNodeInfo(const std::string &modelId, uint32_t skId, uint32_t nodeId) const {
  std::lock_guard<std::mutex> lock(nodeInfoMapMutex);
  SkNodeInfo emptyInfo;

  auto modelIt = nodeInfoMap.find(modelId);
  if (modelIt == nodeInfoMap.end()) {
    return emptyInfo;
  }
  auto skIt = modelIt->second.find(skId);
  if (skIt == modelIt->second.end()) {
    return emptyInfo;
  }
  auto nodeIt = skIt->second.find(nodeId);
  if (nodeIt == skIt->second.end()) {
    return emptyInfo;
  }
  return nodeIt->second;
}

void SkEventRecorder::AddSkNameMapping(const std::string &modelId, uint32_t skId, const std::string &skName) {
  std::lock_guard<std::mutex> lock(nodeInfoMapMutex);
  skNameMap[modelId][skId] = skName;
}

std::string SkEventRecorder::GetSkName(const std::string &modelId, uint32_t skId) const {
  std::lock_guard<std::mutex> lock(nodeInfoMapMutex);
  auto modelIt = skNameMap.find(modelId);
  if (modelIt == skNameMap.end()) {
    return "";
  }
  auto skIt = modelIt->second.find(skId);
  if (skIt == modelIt->second.end()) {
    return "";
  }
  return skIt->second;
}

uint16_t SkEventRecorder::RegisterModelId(const std::string &modelId) {
  std::lock_guard<std::mutex> lock(modelIdIndexMapMutex);
  // 如果已注册，返回已有 index
  auto it = modelIdToIndexMap.find(modelId);
  if (it != modelIdToIndexMap.end()) {
    return it->second;
  }
  // 新注册，index 为当前 vector 大小
  uint16_t index = static_cast<uint16_t>(modelIdIndexMap.size());
  if (modelIdIndexMap.size() >= UINT16_MAX) {
    SK_LOGE("[sk] modelIdIndexMap is full (max=%u), cannot register more modelId\n", UINT16_MAX);
    return 0;  // 溢出保护
  }
  modelIdIndexMap.push_back(modelId);
  modelIdToIndexMap[modelId] = index;
  return index;
}

std::string SkEventRecorder::GetModelIdByIndex(uint16_t index) const {
  std::lock_guard<std::mutex> lock(modelIdIndexMapMutex);
  if (index >= modelIdIndexMap.size()) {
    SK_LOGE("[sk] GetModelIdByIndex: index=%u out of range (size=%zu), modelId not registered or header corrupted",
            index, modelIdIndexMap.size());
    return "";
  }
  return modelIdIndexMap[index];
}

void SkEventRecorder::PrintModelIdIndexMap() const {
  std::lock_guard<std::mutex> lock(modelIdIndexMapMutex);
  SK_LOGE("=== modelId Index Map (total=%zu) ===", modelIdIndexMap.size());
  for (size_t i = 0; i < modelIdIndexMap.size(); ++i) {
    SK_LOGE("  [%zu] modelId=%s", i, modelIdIndexMap[i].c_str());
  }
}

// ==================== 性能分析相关函数 ====================
bool SkProfiling(const SuperKernelScopeInfo &scopeInfo, SkLaunchInfo &launchInfo, SuperKernelGraph &graph) {
  SK_LOGI("[sk shape profiling] =============== Start shape profiling ===================");
  SkHostEntryInfo &skEntryInfo = launchInfo.entryInfo;

  uint32_t opFlag = 0;  // 记录op属性标记的bitmap，bit0代表是否使能了HF32
  std::string combinedAttrIdStr;
  uint32_t maxTensorNum = SHAPE_MAX_TENSOR_NUM;

  // ====== 第一遍遍历：计算总 tensor 数量，并收集 NODE_KERNEL 类型的节点 ======
  uint32_t totalTensorNum = 0;
  std::vector<SuperKernelBaseNode *> kernelNodes;
  const auto &filteredNodes = scopeInfo.GetExtInfo().filteredNodes;
  for (size_t i = 0; i < filteredNodes.size(); ++i) {
    SuperKernelBaseNode *node = filteredNodes[i];
    if (node == nullptr) {
      SK_LOGE("[sk shape profiling] Failed to get node, node is nullptr");
      return false;
    }
    SkNodeType nodeType = node->GetNodeType();
    if (nodeType == SkNodeType::NODE_KERNEL) {
      const NodeInfos &nodeInfos = node->GetNodeInfos();
      const KernelInfos &kernelInfos = nodeInfos.kernelInfos;
      if (kernelInfos.opInfoPtr != nullptr) {
        const CacheopInfoBasic *cacheInfo = static_cast<const CacheopInfoBasic *>(kernelInfos.opInfoPtr);
        totalTensorNum += cacheInfo->tensorNum;
      }

      if (totalTensorNum > maxTensorNum) {
        totalTensorNum = maxTensorNum;  // 截断到最大值
        break;
      }
      kernelNodes.push_back(node);
    }
  }

  size_t totalSize = sizeof(CacheopInfoBasic) + totalTensorNum * sizeof(MsrofTensorData);
  auto ptr = std::make_unique<uint8_t[]>(totalSize);
  if (ptr == nullptr) {
    SK_LOGE("[sk shape profiling] Failed to allocate memory for launchInfo.cacheInfo\n");
    return false;
  }

  // ptr生命周期跟随graph，因为runtime在aclmdlRIUpdate才会把shape信息copy走
  launchInfo.cacheInfo = ptr.get();
  graph.AddShapeInfoPtr(std::move(ptr));
  errno_t ret = memset_s(launchInfo.cacheInfo, totalSize, 0, totalSize);
  if (ret != EOK) {
    SK_LOGE("[sk shape profiling] memset_s launchInfo.cacheInfo failed, ret=%d\n", ret);
  }
  launchInfo.cacheopInfoSize = totalSize;
  uint8_t *dest = static_cast<uint8_t *>(launchInfo.cacheInfo);
  uint64_t destOffset = sizeof(CacheopInfoBasic);  // 预留 CacheopInfoBasic 的空间，后续填充

  // ====== 第二遍遍历：从 kernelNodes 收集数据 ======
  for (size_t i = 0; i < kernelNodes.size(); ++i) {
    SuperKernelBaseNode *node = kernelNodes[i];

    const NodeInfos &nodeInfos = node->GetNodeInfos();

    const KernelInfos &kernelInfos = nodeInfos.kernelInfos;
    if (kernelInfos.opInfoPtr != nullptr) {
      const CacheopInfoBasic *cacheInfo = static_cast<const CacheopInfoBasic *>(kernelInfos.opInfoPtr);
      if (kernelInfos.opInfoSize >= (sizeof(CacheopInfoBasic) + sizeof(MsrofTensorData) * cacheInfo->tensorNum)) {
        // attrId
        char *attrIdStr = MsprofId2Str(cacheInfo->attrId);
        if (!combinedAttrIdStr.empty()) {
          combinedAttrIdStr += "|";
        }
        combinedAttrIdStr += attrIdStr;

        // opFlag 取值只有0和1
        opFlag = opFlag || cacheInfo->opFlag;

        // ====== 复制 tensorData 到 launchInfo.cacheInfo ======
        for (uint32_t t = 0; t < cacheInfo->tensorNum; ++t) {
          const MsrofTensorData &tensor = cacheInfo->tensorData[t];
          MsrofTensorData msTensor;
          msTensor.tensorType = tensor.tensorType;
          msTensor.format = tensor.format;
          msTensor.dataType = tensor.dataType;
          for (int s = 0; s < MSPROF_GE_TENSOR_DATA_SHAPE_LEN; ++s) {
            msTensor.shape[s] = tensor.shape[s];
          }
          errno_t ret = memcpy_s(dest + destOffset, totalSize - destOffset, &msTensor, sizeof(MsrofTensorData));
          if (ret != EOK) {
            SK_LOGE("[sk shape profiling] memcpy_s failed, ret=%d\n", ret);
            return false;
          }
          destOffset += sizeof(MsrofTensorData);
        }
      } else {
        SK_LOGE(
            "[sk shape profiling] warning: kernelInfos.opInfoSize should be greater than or equal to "
            "kernelInfos.opInfoPtr \n");
      }
    }
  }

  // 将拼接后的字符串转换回 uint64_t
  uint64_t combinedAttrId =
      combinedAttrIdStr.empty() ? 0 : MsprofStr2Id(combinedAttrIdStr.c_str(), combinedAttrIdStr.length());

  CacheopInfoBasic cacheopInfoBasic;
  if (skEntryInfo.entryType == SkKernelType::AIV_ONLY || skEntryInfo.entryType == SkKernelType::MIX_AIV_1_0) {
    cacheopInfoBasic.taskType = MSPROF_GE_TASK_TYPE_MIX_AIV;
  } else {
    cacheopInfoBasic.taskType = MSPROF_GE_TASK_TYPE_MIX_AIC;
  }
  // numBlocks 编码：高16位表示mix模式类型，低16位为实际numBlocks
  // sk_entry_mix11 -> 高16位=1, sk_entry_mix12 -> 高16位=2
  uint32_t numBlocks = skEntryInfo.numBlocks;
  if (skEntryInfo.entryType == SkKernelType::MIX_AIC_1_1) {
    cacheopInfoBasic.numBlocks = (1U << 16) | (numBlocks & 0xFFFF);
  } else if (skEntryInfo.entryType == SkKernelType::MIX_AIC_1_2) {
    cacheopInfoBasic.numBlocks = (2U << 16) | (numBlocks & 0xFFFF);
  } else {
    cacheopInfoBasic.numBlocks = numBlocks;
  }
  const char *skEntryFuncName = launchInfo.skFuncName.c_str();
  if (skEntryFuncName != nullptr) {
    cacheopInfoBasic.nodeId = MsprofStr2Id(skEntryFuncName, strlen(skEntryFuncName));
  } else {
    SK_LOGE("[sk shape profiling] Failed to get entry func name\n");
    return false;
  }
  const char *opTypeStr = "SuperKernel";
  cacheopInfoBasic.opType = MsprofStr2Id(opTypeStr, strlen(opTypeStr));
  cacheopInfoBasic.attrId = combinedAttrId;
  cacheopInfoBasic.opFlag = opFlag;
  cacheopInfoBasic.tensorNum = totalTensorNum;
  ret = memcpy_s(dest, sizeof(CacheopInfoBasic), &cacheopInfoBasic, sizeof(CacheopInfoBasic));
  if (ret != EOK) {
    SK_LOGE("[sk shape profiling] memcpy_s cacheopInfoBasic failed, ret=%d\n", ret);
  }
  const CacheopInfoBasic *cacheInfoPtr = static_cast<const CacheopInfoBasic *>(launchInfo.cacheInfo);
  SK_LOGI(
      "[sk shape profiling] sk shape information verify: taskType=%u, numBlocks=%u, nodeId=%lu, opType=%lu, "
      "attrId=%lu, opFlag=%u, tensorNum=%u, infoSize=%lu\n",
      cacheInfoPtr->taskType, cacheInfoPtr->numBlocks, cacheInfoPtr->nodeId, cacheInfoPtr->opType, cacheInfoPtr->attrId,
      cacheInfoPtr->opFlag, cacheInfoPtr->tensorNum, launchInfo.cacheopInfoSize);
  SK_LOGI("[sk shape profiling] =============== End shape profiling ===================");
  return true;
}

bool DumpProfilingDetail(const std::vector<SuperKernelBaseNode *> &taskNodes, SkLaunchInfo &launchInfo,
                         const SuperKernelScopeInfo &scopeInfo, const SuperKernelGraph &graph) {
  // 获取事件记录 GM 地址并更新 devArgs 中的事件配置
  // 填充 devArgs 中的 modelIdIndexAndSkScopeId（不依赖 profiling 开关）
  const std::string &modelId = graph.GetModelId();
  SetupProfilingModelContext(launchInfo, scopeInfo, modelId);

  // skName 映射不依赖 profiling 开关：异常 handler 在 profiling 关闭时也要靠这张表。
  SkEventRecorder::Instance().AddSkNameMapping(modelId, static_cast<uint32_t>(scopeInfo.GetScopeId()),
                                               launchInfo.skFuncName);

  if (SkEventRecorder::Instance().IsEnabled()) {
    return SetupProfilingRuntime(taskNodes, launchInfo, scopeInfo, modelId);
  } else {
    launchInfo.eventGmAddr = nullptr;
    launchInfo.modelIdIndex = 0;
    launchInfo.skId = 0;
  }
  return true;
}

std::string GetSkFuncName(const std::vector<SuperKernelBaseNode *> &nodes, uint16_t scopeId,
                          const std::string &scopeName) {
  const SuperKernelBaseNode *startKernelNode = nullptr;
  const SuperKernelBaseNode *endKernelNode = nullptr;
  for (const auto *node : nodes) {
    if (node->GetNodeType() == SkNodeType::NODE_KERNEL) {
      if (startKernelNode == nullptr) {
        startKernelNode = node;
      }
      endKernelNode = node;
    }
  }

  std::string scopePrefix = scopeName.empty() ? "" : scopeName;
  if (startKernelNode == nullptr || endKernelNode == nullptr) {
    return "sk_" + std::to_string(scopeId) + "_no_kernel_scope_" + scopePrefix;
  }

  const NodeInfos &startNodeInfos = startKernelNode->GetNodeInfos();
  const NodeInfos &endNodeInfos = endKernelNode->GetNodeInfos();
  return "sk_" + std::to_string(scopeId) + "_" + scopePrefix + "_start_" + startNodeInfos.kernelInfos.funcName +
         "_end_" + endNodeInfos.kernelInfos.funcName;
}
