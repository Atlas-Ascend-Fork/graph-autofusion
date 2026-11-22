# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Execute the actual Tensor CV epilogue with a host pipeline-order checker.

The checker requires ready -> ACK ordering either on one pipe or through the
vector/store dependency supplied by the callback. It does not simulate MMAD,
device timing, or hardware synchronization counters.
"""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


HEADER = (
    Path(__file__).resolve().parents[3]
    / "v35/ascendc/api_cube/matmul/mat_mul_pingpong_tensor_cv.h"
)

PROLOGUE = r"""
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <tuple>
#define __aicore__
using GM_ADDR = void *;
enum pipe_t { PIPE_V, PIPE_MTE3 };
int subblock = 0, ratio = 2, ready_pipe = -1, ready_flag = -1;
int acknowledgements = 0, stores = 0, vector_stages = 0;
int stage_rows = 2;
bool store_dependency = false, pingpong = false;
namespace AscendC {
constexpr int TOTAL_UB_SIZE = 256 * 1024;
enum class TPosition { VECIN };
template <typename T> struct LocalTensor {
  LocalTensor() = default;
  LocalTensor(TPosition, int, int) {}
  LocalTensor operator[](int64_t) const { return {}; }
};
inline int64_t GetTaskRation() { return ratio; }
inline int64_t GetSubBlockIdx() { return subblock; }
}
namespace asc { namespace te {
template <typename... T> using shape = std::tuple<T...>;
using std::get;
template <typename T> constexpr int c0_element = 32 / sizeof(T);
}}
namespace Blaze { namespace Gemm {
constexpr int MNK_M = 0, MNK_N = 1, DOUBLE_BUFFER_COUNT = 2;
template <typename T> T Min(T a, T b) { return std::min(a, b); }
template <typename T> T CeilDiv(T a, T b) { return (a + b - 1) / b; }
template <typename T> T CeilAlign(T a, T b) { return CeilDiv(a, b) * b; }
namespace Sync {
template <int Mode, pipe_t Pipe> void WaitForCube(int flag) {
  if (ready_pipe != -1) { throw std::logic_error("unmatched ready wait"); }
  ready_pipe = Pipe;
  ready_flag = flag;
  store_dependency = false;
}
template <int Mode, pipe_t Pipe> void NotifyCube(int flag) {
  if (ready_pipe == -1 || (ready_pipe != Pipe && !store_dependency)) {
    throw std::logic_error("ACK is not ordered after ready");
  }
  const int slot = pingpong ? (acknowledgements & 1) : 0;
  if (ready_flag != 6 + slot || flag != 4 + slot) {
    throw std::logic_error("wrong ping-pong synchronization slot");
  }
  ready_pipe = -1;
  ++acknowledgements;
}
}
}}
struct DispatchPolicy {};
struct Callback {
  struct Params {};
  template <typename T>
  void Init(const Params &, AscendC::LocalTensor<T> &, int64_t, int64_t, int64_t, int64_t &rows) {
    rows = stage_rows;
  }
  template <typename... T>
  void operator()(int64_t, int64_t rows, int64_t cols, int64_t, int64_t,
                  int64_t, int64_t size, int64_t, uint8_t stage, T...) {
    if (rows <= 0 || cols <= 0 || size <= 0) {
      throw std::logic_error("empty callback");
    }
    if (stage == 1) {
      if (ready_pipe != PIPE_V) { throw std::logic_error("vector work is not ordered after ready"); }
      ++vector_stages;
    }
    if (stage == 2) { store_dependency = true; ++stores; }
  }
};
"""

EPILOGUE = r"""
template <typename T, bool DistributeScalar>
void Check(int64_t rows, int64_t cols, int64_t baseM, int64_t baseN, bool splitM, bool replicateScalar, int db,
           int stageRows = 2) {
  using Op = Blaze::Epilogue::Block::BlockEpilogueTensorCV<T, T, DispatchPolicy, Callback, DistributeScalar>;
  stage_rows = stageRows;
  for (int sb = 0; sb < ratio; ++sb) {
    subblock = sb;
    ready_pipe = -1;
    acknowledgements = stores = vector_stages = 0;
    pingpong = db > 1;
    Op op;
    typename Op::ProblemShape problem{rows, cols, 32, 76};
    op.Init(typename Op::Params{}, problem);
    typename Op::BlockShape block{rows, cols, 32, 76};
    for (int repeat = 0; repeat < 64; ++repeat) {
      op(block, 0, splitM, baseM, baseN, db, replicateScalar);
    }
    const int expected = 64 * ((cols + baseN - 1) / baseN);
    const int64_t curM = std::min(rows, baseM);
    const int64_t aivRows = splitM && !replicateScalar ? (curM + ratio - 1) / ratio - ((curM & 1) ? sb : 0) : curM;
    if (acknowledgements != expected || ready_pipe != -1 || stores != vector_stages ||
        (aivRows == 0 || stageRows == 0 ? stores != 0 : stores < expected)) {
      throw std::logic_error("unbalanced epilogue handshake or callbacks");
    }
  }
}
int main() {
  for (int taskRatio : {1, 2}) {
    ratio = taskRatio;
    for (int db : {1, 2}) {
      for (int64_t rows : {1, 2, 3, 17, 33}) {
        for (int64_t cols : {1, 7, 63, 512, 513}) {
          for (bool split : {false, true}) {
            Check<float, false>(rows, cols, 16, 128, split, false, db);
            Check<uint16_t, false>(rows, cols, 16, 128, split, false, db);
          }
        }
      }
      Check<float, true>(1, 1, 16, 128, true, true, db);
      Check<float, false>(3, 513, 16, 128, true, false, db, 0);
    }
  }
}
"""


class TestTensorCvEpilogueSync(unittest.TestCase):
    def test_ready_precedes_ack_for_empty_and_active_aivs(self):
        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("A host C++ compiler is required")
        header = HEADER.read_text()
        # Compile the real class, excluding unrelated Tensor API MMAD/kernel
        # definitions, whose dependencies require the device toolchain.
        start = header.index("namespace Blaze {")
        end = header.index("namespace MatmulV3Advanced {")
        with tempfile.TemporaryDirectory(prefix="tensor-cv-epilogue-") as root:
            source = Path(root) / "test.cpp"
            binary = Path(root) / "test"
            source.write_text(PROLOGUE + header[start:end] + EPILOGUE)
            result = subprocess.run(
                [compiler, "-std=c++17", "-O2", str(source), "-o", str(binary)],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            result = subprocess.run(
                [str(binary)], capture_output=True, text=True, timeout=10, check=False
            )
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
