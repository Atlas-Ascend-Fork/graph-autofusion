# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Correct the CANN CPU Compact DMA bounds check in a private header copy.

CANN 9.2 invokes the Normal-only overflow checker even for PaddingMode::Compact.
Only those two ExtParams overloads are changed. Actual DMA execution and all
Normal checks remain supplied by CANN; Compact still checks its real byte span.
"""

from pathlib import Path
import re
import shutil
import sys


def patch_header(source):
    pattern = re.compile(
        r"template <typename T, PaddingMode mode>\n"
        r"__aicore__ inline __inout_pipe__\(MTE[23]\) void DataCopyPad\(\n"
        r"(?:(?!\n}).)*?const DataCopyExtParams& dataCopyParams(?:(?!\n}).)*?\n}",
        re.DOTALL,
    )

    def patch_function(match):
        function = match.group()
        load = "const LocalTensor<T>& dst" in function
        call = "CheckDataCopyPadTensorSizeOverflow(dst, src, dataCopyParams" + (
            ", padParams)" if load else ")"
        )
        original = f"    ASCENDC_REPORT_OVERFLOW_MEM(({call}));"
        if original not in function:
            return function  # A newer SDK may already have mode-aware checks.
        local = "dst" if load else "src"
        spaces = (
            "Hardware::GM, Hardware::UB, 0, localBytes, 0, compactBytes"
            if load
            else "Hardware::UB, Hardware::GM, localBytes, 0, compactBytes, 0"
        )
        replacement = f"""    if constexpr (mode == PaddingMode::Compact) {{
        const uint64_t compactBytes = static_cast<uint64_t>(dataCopyParams.blockCount) * dataCopyParams.blockLen;
        const uint64_t localBytes = static_cast<uint64_t>({local}.GetSize()) * sizeof(PrimT<T>);
        ASCENDC_REPORT_OVERFLOW_MEM((check::ReportTensorSizeOverflow(
            {spaces}, "DataCopyPad Compact")));
    }} else {{
{original}
    }}"""
        return function.replace(original, replacement)

    return pattern.sub(patch_function, source)


def prepare_sdk(asc_root, output):
    relative = Path("impl/basic_api/kernel_operator_data_copy_intf_impl.h")
    # The caller selects the host architecture's asc tree. Resolve the toolkit
    # prefix and ascendc mirror within that same architecture.
    install_root = asc_root.parent.parent
    architecture = asc_root.parent.name
    tree_candidates = (
        asc_root,
        install_root / "toolkit" / architecture / "asc",
    )
    header_candidates = [tree / relative for tree in tree_candidates] + [
        install_root
        / "toolkit"
        / architecture
        / "ascendc"
        / "include"
        / "basic_api"
        / "impl"
        / relative.name,
        install_root
        / architecture
        / "ascendc"
        / "include"
        / "basic_api"
        / "impl"
        / relative.name,
    ]
    tree = next((t for t in tree_candidates if (t / relative).is_file()), None)
    source_path = next((p for p in header_candidates if p.is_file()), None)
    if source_path is None:
        listing = "\n".join(f"  {p}" for p in header_candidates)
        print(
            f"ERROR: cannot locate {relative.name}; Compact DMA SDK preparation failed.\n"
            f"Checked:\n{listing}",
            file=sys.stderr,
        )
        return False
    if tree is None:
        tree = next(
            (
                t
                for t in tree_candidates
                if (t / "include").is_dir() and (t / "impl").is_dir()
            ),
            None,
        )
        if tree is None:
            listing = "\n".join(f"  {t}" for t in tree_candidates)
            print(
                f"ERROR: cannot locate the AscendC asc tree; Compact DMA SDK preparation failed.\n"
                f"Checked:\n{listing}",
                file=sys.stderr,
            )
            return False
    patched = patch_header(source_path.read_text())

    def keep_patched_header(directory, names):
        return {relative.name} if Path(directory) == tree / relative.parent else set()

    # Refresh every other SDK header too: an in-place SDK update must not leave
    # a mixture of old and new headers in this private tree. copy2 preserves
    # source timestamps; the corrected header is rewritten only on real changes.
    for directory in ("include", "impl"):
        shutil.copytree(
            tree / directory,
            output / directory,
            dirs_exist_ok=True,
            ignore=keep_patched_header,
        )
    target = output / relative
    if not target.exists() or target.read_text() != patched:
        target.write_text(patched)
    for item in output.rglob("*"):
        item.chmod(0o755 if item.is_dir() else 0o644)
    return True


if __name__ == "__main__":
    sys.exit(0 if prepare_sdk(*map(Path, sys.argv[1:])) else 1)
