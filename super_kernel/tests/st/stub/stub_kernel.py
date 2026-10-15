#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and contiditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------------------

"""Stub sub-kernels for SuperKernel system tests.

The SuperKernel code generator consumes two inputs per sub-operator:

* ``bin_path``  -- the sub-kernel object file. Only two properties matter:
  - it is an ``ar`` archive whose members can be extracted (``ar x``);
  - ``llvm-objdump -h`` reports a ``.text`` section, whose size is turned into
    ``preload(ptr, N)`` (``N = ceil(text_size / 2048)``).
* ``json_path`` -- the sub-kernel metadata (kernel type, split mode, ...).

Neither input needs real Ascend machine code, so both are generated here at
test time. This keeps the repository free of binaries and makes the expected
outputs reproducible: the only observable property of the stub object is the
size of its ``.text`` section, which is pinned by ``STUB_TEXT_SIZE``.
"""

import json
import shutil
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

# ---------------------------------------------------------------------------
# Contract with the golden files
# ---------------------------------------------------------------------------

# ``.text`` size of each stub sub-kernel. ``get_text_section_size`` converts it
# to the ``preload`` count written into the generated kernel source, so changing
# a value here changes the golden files and must be done deliberately.
STUB_TEXT_SIZE = {
    "is_inf": 4096,  # ceil(4096 / 2048) = 2
    "is_finite": 4096,  # ceil(4096 / 2048) = 2
    "pows": 10240,  # ceil(10240 / 2048) = 5
}

# Number of ``_splitN`` members generated next to the base object.
SPLIT_MEMBER_COUNT = 3

# Chip version used to build the ``dav-<chip>-<core>`` metadata keys.
CHIP_VERSION = "c220"

# Machine field of the generated ELF. Only ``llvm-objdump`` has to understand
# it, so a value known to every LLVM build is used (EM_AARCH64).
ELF_MACHINE = 183

# ---------------------------------------------------------------------------
# Sub-kernel metadata
# ---------------------------------------------------------------------------

# Fields that are constant for a given sub-kernel. They are replayed as-is;
# only ``sub_operator_kernel_type`` / ``split_mode`` / ``sub_op_with_sync_all``
# vary between scenarios (see ``SCENARIO_OVERRIDES``).
SUB_KERNEL_BASE = {
    "is_inf": {
        "binFileName": "is_inf",
        "binFileSuffix": ".o",
        "blockDim": 32,
        "coreType": "VectorCore",
        "core_type": "AIV",
        "intercoreSync": 0,
        "kernelName": "is_inf__kernel0",
        "localMemorySize": 0,
        "magic": "RT_DEV_BINARY_MAGIC_ELF_AIVEC",
        "memoryStamping": [],
        "opParaSize": 32,
        "parameters": [None, None, None],
        "sub_operator_call_dcci_after_kernel_end": False,
        "sub_operator_call_dcci_before_kernel_start": False,
        "sub_operator_early_start_set_flag": False,
        "sub_operator_early_start_wait_flag": False,
        "sub_operator_params": ["x_in__", "y_out_", "workspace"],
        "supportSuperKernel": 1,
        "workspace": {"num": 1, "size": [33554432], "type": [0]},
    },
    "is_finite": {
        "binFileName": "is_finite",
        "binFileSuffix": ".o",
        "blockDim": 32,
        "coreType": "VectorCore",
        "core_type": "AIV",
        "intercoreSync": 0,
        "kernelName": "is_finite__kernel0",
        "localMemorySize": 0,
        "magic": "RT_DEV_BINARY_MAGIC_ELF_AIVEC",
        "memoryStamping": [],
        "opParaSize": 48,
        "parameters": [None, None, None],
        "sub_operator_call_dcci_after_kernel_end": False,
        "sub_operator_call_dcci_before_kernel_start": False,
        "sub_operator_early_start_set_flag": False,
        "sub_operator_early_start_wait_flag": False,
        "sub_operator_params": ["x_in__", "y_out_", "workspace"],
        "supportSuperKernel": 1,
        "workspace": {"num": 1, "size": [33554432], "type": [0]},
    },
    "pows": {
        "binFileName": "pows",
        "binFileSuffix": ".o",
        "blockDim": 1,
        "coreType": "VectorCore",
        "core_type": "AIV",
        "intercoreSync": 0,
        "kernelName": "pows__kernel0",
        "localMemorySize": 0,
        "magic": "RT_DEV_BINARY_MAGIC_ELF_AIVEC",
        "memoryStamping": [],
        "opParaSize": 88,
        "parameters": [None, None, None, None],
        "sub_operator_call_dcci_after_kernel_end": False,
        "sub_operator_call_dcci_before_kernel_start": False,
        "sub_operator_early_start_set_flag": False,
        "sub_operator_early_start_wait_flag": False,
        "sub_operator_params": ["x1_in__", "x2_in__", "y_out_", "workspace"],
        "supportSuperKernel": 1,
        "workspace": {"num": 1, "size": [1568], "type": [0]},
    },
}

# Extra metadata carried by sub-kernels compiled in dynamic mode.
DYNAMIC_BASE_PATCH = {
    "is_inf": {
        "debugBufSize": 78643200,
        "debugOptions": "printf",
        "timestamp_option": True,
    },
}

# Metadata keys describing the callable kernel, per kernel type.
KERNEL_NAME_KEYS = {
    "KERNEL_TYPE_AIC_ONLY": ("AiCore",),
    "KERNEL_TYPE_AIV_ONLY": ("AiCore",),
    "KERNEL_TYPE_MIX_AIC_1_0": ("AiCore", f"dav-{CHIP_VERSION}-cube"),
    "KERNEL_TYPE_MIX_AIC_1_1": (
        "AiCore",
        f"dav-{CHIP_VERSION}-cube",
        f"dav-{CHIP_VERSION}-vec",
    ),
    "KERNEL_TYPE_MIX_AIC_1_2": (
        "AiCore",
        f"dav-{CHIP_VERSION}-cube",
        f"dav-{CHIP_VERSION}-vec",
    ),
    "KERNEL_TYPE_MIX_AIV_1_0": ("AiCore", f"dav-{CHIP_VERSION}-vec"),
}


@dataclass(frozen=True)
class SubKernelSpec:
    """Per-scenario variation of a sub-kernel metadata set."""

    kernel_type: str = "KERNEL_TYPE_AIC_ONLY"
    split_mode: Optional[int] = 4
    sync_all: bool = False
    variant: str = "static"


DEFAULT_SPEC = SubKernelSpec()

# Scenarios that differ from ``DEFAULT_SPEC``. Everything else uses the default
# for every sub-kernel, which keeps this table small and reviewable.
SCENARIO_OVERRIDES = {
    "test_sk_1_stream_2_ops_json_split_none_aic_only": {
        "is_inf": SubKernelSpec(
            "KERNEL_TYPE_AIC_ONLY", split_mode=None, sync_all=False, variant="static"
        ),
    },
    "test_sk_1_stream_2_ops_split_mode_1_aic_only": {
        "is_finite": SubKernelSpec(
            "KERNEL_TYPE_AIC_ONLY", split_mode=1, sync_all=False, variant="static"
        ),
        "is_inf": SubKernelSpec(
            "KERNEL_TYPE_AIC_ONLY", split_mode=1, sync_all=False, variant="static"
        ),
    },
    "test_sk_2_stream_2_ops_debug_sync_all_1_aiv_only": {
        "is_inf": SubKernelSpec(
            "KERNEL_TYPE_AIV_ONLY", split_mode=4, sync_all=False, variant="static"
        ),
        "pows": SubKernelSpec(
            "KERNEL_TYPE_AIV_ONLY", split_mode=4, sync_all=False, variant="static"
        ),
    },
    "test_sk_2_stream_2_ops_default_aic_1_0": {
        "is_inf": SubKernelSpec(
            "KERNEL_TYPE_MIX_AIC_1_0", split_mode=4, sync_all=False, variant="static"
        ),
        "pows": SubKernelSpec(
            "KERNEL_TYPE_MIX_AIC_1_0", split_mode=4, sync_all=False, variant="static"
        ),
    },
    "test_sk_2_stream_2_ops_default_aic_1_1": {
        "is_inf": SubKernelSpec(
            "KERNEL_TYPE_MIX_AIC_1_1", split_mode=4, sync_all=False, variant="static"
        ),
        "pows": SubKernelSpec(
            "KERNEL_TYPE_MIX_AIC_1_1", split_mode=4, sync_all=False, variant="static"
        ),
    },
    "test_sk_2_stream_2_ops_default_aic_1_2": {
        "is_inf": SubKernelSpec(
            "KERNEL_TYPE_MIX_AIC_1_2", split_mode=4, sync_all=False, variant="static"
        ),
        "pows": SubKernelSpec(
            "KERNEL_TYPE_MIX_AIC_1_2", split_mode=4, sync_all=False, variant="static"
        ),
    },
    "test_sk_2_stream_2_ops_default_aiv_1_0": {
        "is_inf": SubKernelSpec(
            "KERNEL_TYPE_MIX_AIV_1_0", split_mode=4, sync_all=False, variant="static"
        ),
        "pows": SubKernelSpec(
            "KERNEL_TYPE_MIX_AIV_1_0", split_mode=4, sync_all=False, variant="static"
        ),
    },
    "test_sk_2_stream_2_ops_default_send_recv_aiv_only": {
        "is_inf": SubKernelSpec(
            "KERNEL_TYPE_AIV_ONLY", split_mode=4, sync_all=False, variant="static"
        ),
        "pows": SubKernelSpec(
            "KERNEL_TYPE_AIV_ONLY", split_mode=4, sync_all=False, variant="static"
        ),
    },
    "test_sk_2_stream_2_ops_dynamic_send_recv_default_aic_only": {
        "is_inf": SubKernelSpec(
            "KERNEL_TYPE_AIC_ONLY", split_mode=4, sync_all=True, variant="dynamic"
        ),
    },
    "test_sk_2_stream_2_ops_dynamic_send_recv_default_aiv_only": {
        "is_finite": SubKernelSpec(
            "KERNEL_TYPE_AIV_ONLY", split_mode=4, sync_all=False, variant="static"
        ),
        "is_inf": SubKernelSpec(
            "KERNEL_TYPE_AIV_ONLY", split_mode=4, sync_all=True, variant="dynamic"
        ),
    },
    "test_sk_2_stream_2_ops_dynamic_send_recv_intersect_aic_only": {
        "is_inf": SubKernelSpec(
            "KERNEL_TYPE_AIC_ONLY", split_mode=4, sync_all=True, variant="dynamic"
        ),
    },
    "test_sk_2_stream_2_ops_dynamic_send_recv_with_synal_all_1_aic_only": {
        "is_inf": SubKernelSpec(
            "KERNEL_TYPE_AIC_ONLY", split_mode=4, sync_all=True, variant="dynamic"
        ),
    },
    "test_sk_2_stream_4_ops_remove_crossed_cub_to_vec": {
        "pows": SubKernelSpec(
            "KERNEL_TYPE_AIV_ONLY", split_mode=4, sync_all=False, variant="static"
        ),
    },
    "test_sk_2_stream_4_ops_remove_crossed_vec_to_cub": {
        "is_inf": SubKernelSpec(
            "KERNEL_TYPE_AIV_ONLY", split_mode=4, sync_all=False, variant="static"
        ),
    },
}


def resolve_spec(scenario: str, kernel: str) -> SubKernelSpec:
    """Return the metadata spec of ``kernel`` used by ``scenario``."""
    return SCENARIO_OVERRIDES.get(scenario, {}).get(kernel, DEFAULT_SPEC)


def render_sub_kernel_json(kernel: str, spec: SubKernelSpec, bin_path: Path) -> Dict:
    """Build the sub-kernel metadata consumed by the SuperKernel compiler."""
    meta = dict(SUB_KERNEL_BASE[kernel])
    if spec.variant == "dynamic":
        meta.update(DYNAMIC_BASE_PATCH.get(kernel, {}))

    meta["sub_operator_kernel_type"] = spec.kernel_type
    meta["sub_op_with_sync_all"] = spec.sync_all
    if spec.split_mode is not None:
        meta["split_mode"] = spec.split_mode

    obj_files = {
        "func_name": f"{kernel}__kernel0_middle",
        "obj_files": str(bin_path),
    }
    for index in range(1, SPLIT_MEMBER_COUNT + 1):
        obj_files[f"obj_files_split{index}"] = str(bin_path)[:-2] + f"_split{index}.o"

    kernel_names = {key: dict(obj_files) for key in KERNEL_NAME_KEYS[spec.kernel_type]}
    if spec.variant == "dynamic":
        # Dynamic sub-kernels carry the symbol names used by the switch function.
        kernel_names["dynamic_func_names"] = {
            "2": {
                "AiCore": f"{kernel}__kernel0_middle",
                f"dav-{CHIP_VERSION}-cube": "cube_kernel_name",
                "kernel_type": spec.kernel_type,
            }
        }

    meta["sub_operator_kernel_name"] = kernel_names
    return meta


# ---------------------------------------------------------------------------
# Stub object files
# ---------------------------------------------------------------------------

_ELF_HEADER_FORMAT = "<16sHHIQQQIHHHHHH"
_SECTION_HEADER_FORMAT = "<IIQQQQIIQQ"
_ELF_HEADER_SIZE = 64
_SECTION_HEADER_SIZE = 64
_SHT_PROGBITS = 1
_SHT_STRTAB = 3
_SHF_ALLOC = 0x2
_SHF_EXECINSTR = 0x4


def _build_elf(text_size: int) -> bytes:
    """Return a minimal relocatable ELF64 with a single ``.text`` section."""
    section_names = b"\0.text\0.shstrtab\0"
    text_offset = _ELF_HEADER_SIZE
    strtab_offset = text_offset + text_size
    section_header_offset = strtab_offset + len(section_names)

    header = struct.pack(
        _ELF_HEADER_FORMAT,
        b"\x7fELF\x02\x01\x01" + b"\0" * 9,  # e_ident
        1,  # e_type: ET_REL
        ELF_MACHINE,  # e_machine
        1,  # e_version
        0,  # e_entry
        0,  # e_phoff
        section_header_offset,  # e_shoff
        0,  # e_flags
        _ELF_HEADER_SIZE,  # e_ehsize
        0,  # e_phentsize
        0,  # e_phnum
        _SECTION_HEADER_SIZE,  # e_shentsize
        3,  # e_shnum: null + .text + .shstrtab
        2,  # e_shstrndx
    )
    text_section = struct.pack(
        _SECTION_HEADER_FORMAT,
        1,  # sh_name: ".text"
        _SHT_PROGBITS,
        _SHF_ALLOC | _SHF_EXECINSTR,
        0,  # sh_addr
        text_offset,
        text_size,
        0,  # sh_link
        0,  # sh_info
        16,  # sh_addralign
        0,  # sh_entsize
    )
    strtab_section = struct.pack(
        _SECTION_HEADER_FORMAT,
        7,  # sh_name: ".shstrtab"
        _SHT_STRTAB,
        0,
        0,
        strtab_offset,
        len(section_names),
        0,
        0,
        1,
        0,
    )
    null_section = struct.pack(_SECTION_HEADER_FORMAT, *([0] * 10))
    return b"".join(
        (
            header,
            b"\0" * text_size,
            section_names,
            null_section,
            text_section,
            strtab_section,
        )
    )


def _build_archive(archive_path: Path, members: Dict[str, bytes]) -> None:
    """Create an ``ar`` archive containing ``members`` (name -> content)."""
    staging_dir = archive_path.parent / f"_{archive_path.stem}_members"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)

    try:
        for name, content in members.items():
            (staging_dir / name).write_bytes(content)
        subprocess.run(
            ["ar", "rc", str(archive_path), *members.keys()],
            cwd=str(staging_dir),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)


def write_stub_object(kernel_meta_dir: Path, kernel: str) -> Path:
    """Write the stub sub-kernel archive and return its path."""
    kernel_meta_dir.mkdir(parents=True, exist_ok=True)
    archive_path = kernel_meta_dir / f"{kernel}.o"

    payload = _build_elf(STUB_TEXT_SIZE[kernel])
    members = {f"{kernel}.o": payload}
    for index in range(1, SPLIT_MEMBER_COUNT + 1):
        members[f"{kernel}_split{index}.o"] = payload

    _build_archive(archive_path, members)
    return archive_path


# ---------------------------------------------------------------------------
# Materialisation helpers used by the fixtures
# ---------------------------------------------------------------------------


class SubKernelPath:
    """Paths of a generated stub sub-kernel."""

    def __init__(self, root: Path, name: str):
        self.root = Path(root)
        self.name = name

    def o(self) -> Path:
        return self.root / "kernel_meta" / (self.name + ".o")

    def json(self) -> str:
        return self.name + ".json"

    def json_path(self) -> Path:
        return self.root / "kernel_meta" / self.json()


def materialize_sub_kernels(tmp_root: Path) -> Dict[str, SubKernelPath]:
    """Generate every stub sub-kernel once per session."""
    sub_kernels = {}
    for kernel in SUB_KERNEL_BASE:
        root = Path(tmp_root) / f"stub_subkernel_{kernel}"
        write_stub_object(root / "kernel_meta", kernel)
        sub_kernels[kernel] = SubKernelPath(root, kernel)
    return sub_kernels


def materialize_scenario_jsons(
    tmp_root: Path,
    scenarios: Iterable[str],
    sub_kernels: Dict[str, SubKernelPath],
) -> Path:
    """Render the sub-kernel metadata of every scenario, return the root dir."""
    json_root = Path(tmp_root) / "stub_json"
    for scenario in scenarios:
        scenario_dir = json_root / scenario
        scenario_dir.mkdir(parents=True, exist_ok=True)
        for kernel, path in sub_kernels.items():
            spec = resolve_spec(scenario, kernel)
            meta = render_sub_kernel_json(kernel, spec, path.o())
            (scenario_dir / f"{kernel}.json").write_text(
                json.dumps(meta, indent=2) + "\n", encoding="utf-8"
            )
    return json_root


def scenario_names(data_dir: Path) -> List[str]:
    """List the scenarios that have golden data."""
    return sorted(path.name for path in Path(data_dir).iterdir() if path.is_dir())
