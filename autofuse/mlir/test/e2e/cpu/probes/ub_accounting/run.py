"""Record allocator acceptance of deliberately inconsistent UB descriptors.

This is a diagnostic probe, not a production gate or an NPU safety test.
Run from the repository root in the existing MLIR development container.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--af-opt", type=Path, default=Path("build/autofuse/mlir/tools/af-opt/af-opt")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("build/ub-accounting-20261019")
    )
    args = parser.parse_args()
    root = Path.cwd()
    fixture = (
        root
        / "autofuse/mlir/test/lit/demo/afir_memory_allocate_shared_queue_members.mlir"
    )
    original = fixture.read_text()
    assert original.count('size_bytes_expr = <"64">') == 2
    assert original.count("slot_count = 1") == 2
    # Each member is 16xf16 = 32 bytes. They occupy offsets 0 and 32 in one slot.
    target = '#afir.target<soc = "Ascend910B1", aiv_num = 48, ub_size = 196608>'
    original = original.replace(
        "module {", f"module attributes {{afir.target = {target}}} {{", 1
    )
    cases = {
        "valid": (64, 1),
        "undersized_slot": (32, 1),
        "over_budget": (131072, 2),
        "zero_size_control": (0, 1),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
        "cases": {},
    }
    for name, (slot_bytes, slots) in cases.items():
        source = original.replace(
            'size_bytes_expr = <"64">', f'size_bytes_expr = <"{slot_bytes}">'
        )
        source = source.replace("slot_count = 1", f"slot_count = {slots}")
        input_path = args.output / f"{name}.mlir"
        input_path.write_text(source)
        command = [
            str(args.af_opt),
            "--afir-kernel-body-signature",
            "--afir-loop-generate",
            "--afir-memory-allocate",
            str(input_path),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        (args.output / f"{name}.out.mlir").write_text(result.stdout)
        (args.output / f"{name}.stderr").write_text(result.stderr)
        pipeline = subprocess.run(
            [
                str(args.af_opt),
                "--afir-kernel-body-signature",
                "--afir-to-ascendc-pipeline",
                "--target=Ascend910B1",
                str(input_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        (args.output / f"{name}.pipeline.mlir").write_text(pipeline.stdout)
        (args.output / f"{name}.pipeline.stderr").write_text(pipeline.stderr)
        report["cases"][name] = {
            "returncode": result.returncode,
            "pipeline_returncode": pipeline.returncode,
            "slot_bytes": slot_bytes,
            "slots": slots,
            "allocated_bytes": slot_bytes * slots,
            "member_required_bytes": 64,
            "target_ub_bytes": 196608,
            "contains_init_queue": "ascendc.pipe.init_queue" in result.stdout,
            "contains_subindex": "ascendc.local_tensor.subindex" in result.stdout,
            "runtime_asserts": [
                line.strip()
                for line in result.stdout.splitlines()
                if "cf.assert" in line
            ],
        }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    assert report["cases"]["valid"]["returncode"] == 0
    assert report["cases"]["valid"]["pipeline_returncode"] == 0
    assert report["cases"]["zero_size_control"]["returncode"] != 0
    assert report["cases"]["zero_size_control"]["pipeline_returncode"] != 0
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
