# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "compact_sk_prof_trace.py"


class CompactSkProfTraceTest(unittest.TestCase):
    def test_collapses_cores_and_splits_occurrences(self):
        raw_events = [
            {
                "ph": "X",
                "name": "[0/2]static_kernel_RmsNorm_abcdef0123456789abcdef0123456789_9_d0",
                "pid": "AIV",
                "tid": 25,
                "ts": 100,
                "dur": 5,
                "args": {"modelId": "48_1", "skId": 7, "nodeId": 3},
            },
            {
                "ph": "X",
                "name": "[1/2]static_kernel_RmsNorm_abcdef0123456789abcdef0123456789_9_d0",
                "pid": "AIV",
                "tid": 26,
                "ts": 102,
                "dur": 12,
                "args": {"modelId": "48_1", "skId": 7, "nodeId": 3},
            },
            {
                "ph": "X",
                "name": "[0/2]static_kernel_RmsNorm_abcdef0123456789abcdef0123456789_9_d0",
                "pid": "AIC",
                "tid": 0,
                "ts": 99,
                "dur": 9,
                "args": {"modelId": "48_1", "skId": 7, "nodeId": 3},
            },
            {
                "ph": "X",
                "name": "[1/2]static_kernel_RmsNorm_abcdef0123456789abcdef0123456789_9_d0",
                "pid": "AIC",
                "tid": 1,
                "ts": 101,
                "dur": 10,
                "args": {"modelId": "48_1", "skId": 7, "nodeId": 3},
            },
            {
                "ph": "X",
                "name": "[0/2]static_kernel_RmsNorm_abcdef0123456789abcdef0123456789_9_d0",
                "pid": "AIV",
                "tid": 25,
                "ts": 300,
                "dur": 4,
                "args": {"modelId": "48_1", "skId": 7, "nodeId": 3},
            },
            {
                "ph": "X",
                "name": "[0/2] sk_7_decode.block.0_start_static_kernel_RmsNorm",
                "pid": "AIV",
                "tid": 25,
                "ts": 90,
                "dur": 30,
                "args": {"modelId": "48_1", "skId": 7},
            },
            {
                "ph": "X",
                "name": "[1/2] sk_7_decode.block.0_start_static_kernel_RmsNorm",
                "pid": "AIC",
                "tid": 0,
                "ts": 92,
                "dur": 29,
                "args": {"modelId": "48_1", "skId": 7},
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.json"
            output = root / "compact.json"
            source.write_text(json.dumps(raw_events), encoding="utf-8")
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--input",
                    str(source),
                    "--output",
                    str(output),
                    "--occurrence-gap-us",
                    "50",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            trace = json.loads(output.read_text(encoding="utf-8"))
        self.assertIsInstance(trace, list)
        events = [event for event in trace if event.get("ph") == "X"]
        self.assertEqual(len(events), 4)
        by_component = {
            (event["args"]["component"], event["args"]["occurrence"]): event
            for event in events
        }
        self.assertEqual(by_component[("Vector", 1)]["ts"], 100.0)
        self.assertEqual(by_component[("Vector", 1)]["dur"], 14.0)
        self.assertEqual(
            by_component[("Vector", 1)]["args"]["merged_core_event_count"], 2
        )
        self.assertEqual(by_component[("Cube", 1)]["ts"], 99.0)
        self.assertEqual(by_component[("Cube", 1)]["dur"], 12.0)
        self.assertEqual(by_component[("Vector", 2)]["ts"], 300.0)
        parent = by_component[("SK E2E", 1)]
        self.assertEqual(parent["tid"], 0)
        self.assertEqual(by_component[("Cube", 1)]["tid"], 1)
        self.assertEqual(by_component[("Vector", 1)]["tid"], 2)
        sort_indices = [
            event["args"]["sort_index"]
            for event in trace
            if event.get("name") == "thread_sort_index"
        ]
        self.assertEqual(sort_indices, [0, 1, 2])
        self.assertEqual(parent["ts"], 90.0)
        self.assertEqual(parent["dur"], 31.0)
        self.assertEqual(parent["args"]["merged_core_event_count"], 2)


if __name__ == "__main__":
    unittest.main()
