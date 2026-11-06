# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Portable regression tests for snapshot warnings and host-side input validation."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
HARNESS = ROOT / "Modeling/Data_collection/common/executables/standalone_nddma"


class ReviewRegressions(unittest.TestCase):
    def test_predictors_report_unvalidated_snapshots(self):
        tools = ROOT / "Modeling/Data_collection/common/tools"
        for name, args in [
            ("predict_nddma_ad_hoc.py", ["8x16", "16x1", "16x1", "8", "int16_t"]),
            (
                "predict_nddma_2d_transpose_ad_hoc.py",
                ["63x2", "1x2", "2x1", "8", "int16_t"],
            ),
        ]:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                result = subprocess.run(
                    [sys.executable, str(tools / name), *args, "--output-dir", tmp],
                    cwd=tmp,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("metrics_valid=false", result.stderr)
                self.assertTrue(list(Path(tmp).glob("*.json")))

    def test_collection_dry_run_preserves_files(self):
        scripts = ROOT / "Modeling/Data_collection"
        for name in ("round1", "round2", "round3", "round4", "round5", "ad_hoc"):
            for existing in (False, True):
                with (
                    self.subTest(name=name, existing=existing),
                    tempfile.TemporaryDirectory() as tmp,
                ):
                    base = Path(tmp)
                    output = base / "output"
                    factor = base / "factors.csv"
                    if existing:
                        (output / "experiment").mkdir(parents=True)
                        (output / "experiment/keep.log").write_text("keep me")
                        (output / "configure.log").write_text("old log")
                        factor.write_text("old factors")

                    def snapshot():
                        return {
                            str(p.relative_to(base)): p.read_bytes()
                            if p.is_file()
                            else None
                            for p in base.rglob("*")
                        }

                    before = snapshot()
                    if name == "ad_hoc":
                        command = [
                            str(scripts / "common/tools/run_nddma_ad_hoc.py"),
                            "8x16",
                            "16x1",
                            "16x1",
                            "8",
                            "int16_t",
                        ]
                    else:
                        command = [
                            str(scripts / name / "scripts/collect.py"),
                            "--factor-csv",
                            str(factor),
                        ]
                    result = subprocess.run(
                        [
                            sys.executable,
                            *command,
                            "--output-dir",
                            str(output),
                            "--msprof-bin",
                            "/nonexistent/msprof",
                            "--dry-run",
                        ],
                        cwd=tmp,
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(
                        result.returncode, 0, result.stdout + result.stderr
                    )
                    self.assertEqual(before, snapshot())
                    self.assertIn("cmake", result.stdout)

    def test_missing_arch_reports_configuration_before_writing(self):
        scripts = ROOT / "Modeling/Data_collection"
        environment = dict(os.environ)
        environment.pop("NPU_ARCH", None)
        for name in ("round1", "round2", "round3", "round4", "round5", "ad_hoc"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                if name == "ad_hoc":
                    command = [
                        str(scripts / "common/tools/run_nddma_ad_hoc.py"),
                        "8x16",
                        "16x1",
                        "16x1",
                        "8",
                        "int16_t",
                    ]
                else:
                    command = [
                        str(scripts / name / "scripts/collect.py"),
                        "--factor-csv",
                        str(Path(tmp) / "factor.csv"),
                    ]
                command = [
                    sys.executable,
                    *command,
                    "--output-dir",
                    str(Path(tmp) / "output"),
                ]
                result = subprocess.run(
                    command, env=environment, capture_output=True, text=True
                )
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertIn("export NPU_ARCH=dav-3510", result.stderr)
                self.assertIn("--npu-arch dav-3510", result.stderr)
                self.assertEqual(list(Path(tmp).iterdir()), [])
                for options, env in [
                    (["--npu-arch", "test-v35"], environment),
                    ([], dict(environment, NPU_ARCH="test-v35")),
                ]:
                    preview = subprocess.run(
                        command + options + ["--dry-run"],
                        env=env,
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(preview.returncode, 0, preview.stderr)
                    self.assertIn("-DNPU_ARCH=test-v35", preview.stdout)
                    self.assertNotIn("NPU_ARCH is not configured", preview.stdout)

    def test_e2e_architecture_is_only_forwarded_to_collection(self):
        for round_id in range(1, 6):
            script = ROOT / f"Modeling/Data_collection/round{round_id}/scripts/e2e.py"
            for action in ("all", "collect", "fit", "draw"):
                with self.subTest(round=round_id, action=action):
                    result = subprocess.run(
                        [
                            sys.executable,
                            str(script),
                            action,
                            "--npu-arch",
                            "test-v35",
                            "--dry-run",
                        ],
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    commands = [
                        line
                        for line in result.stdout.splitlines()
                        if line.startswith("[INFO]")
                    ]
                    for command in commands:
                        self.assertEqual(
                            "--npu-arch=test-v35" in command, "/collect.py" in command
                        )

    def test_snapshot_metadata(self):
        for path in (ROOT / "Modeling/Ana").glob("round*/*.json"):
            model = json.loads(path.read_text())
            self.assertIs(model["snapshot"]["metrics_valid"], False)
            self.assertNotIn("/home/", path.read_text())
        model = json.loads(
            (ROOT / "Modeling/Ana/round3/round3_multidim_model.json").read_text()
        )
        self.assertNotIn("block_dim/T", model["formula"]["n_base"])

    @unittest.skipUnless(shutil.which("c++"), "C++ compiler required")
    def test_host_validation_boundaries(self):
        source = r"""
#include "nddma_validation.h"
#include <cassert>
#include <functional>

void Rejected(const std::function<void()> &operation) {
  bool rejected = false;
  try { operation(); } catch (const std::invalid_argument &) { rejected = true; }
  assert(rejected);
}
int main() {
  for (uint64_t dim : {0ULL, 6ULL, 4294967297ULL}) {
    Rejected([=] { nddma::CheckedDimension(dim); });
  }
  assert(nddma::CheckedDimension(1) == 1);
  assert(nddma::CheckedDimension(5) == 5);
  nddma::ValidateShape(2, {2, 8}, {0, 1}, {8, 1});
  nddma::ValidateShape(5, {2, 2, 2, 2, 2}, {16, 8, 4, 2, 1}, {16, 8, 4, 2, 1});
  Rejected([] { nddma::ValidateShape(2, {2}, {1}, {1}); });
  Rejected([] { nddma::ValidateShape(1, {0}, {1}, {1}); });
  Rejected([] { nddma::ValidateShape(1, {-1}, {1}, {1}); });
  Rejected([] { nddma::ValidateShape(1, {2}, {-1}, {1}); });
  Rejected([] { nddma::ValidateShape(1, {2}, {1}, {0}); });
  Rejected([] { nddma::CheckedUint32(4294967296ULL); });
  assert(nddma::CheckedUbBufferBytes(0, 16, 2) == 160);
  assert(nddma::CheckedUbBufferBytes(0, UINT32_MAX - 64, 1) == UINT32_MAX);
  Rejected([] { nddma::CheckedUbBufferBytes(0, UINT32_MAX - 63, 1); });
  Rejected([] { nddma::CheckedUbBufferBytes(UINT32_MAX, UINT32_MAX, 8); });
  Rejected([] { nddma::CheckedUbBufferBytes(0, UINT32_MAX / 8 - 63, 8); });
}
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "validation.cpp"
            path.write_text(source)
            binary = Path(tmp) / "validation"
            subprocess.run(
                [
                    "c++",
                    "-std=c++17",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-I",
                    str(HARNESS),
                    str(path),
                    "-o",
                    str(binary),
                ],
                check=True,
                capture_output=True,
            )
            subprocess.run([str(binary)], check=True, capture_output=True)


if __name__ == "__main__":
    unittest.main()
