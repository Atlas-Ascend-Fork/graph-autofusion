# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Smoke tests for graph-autofusion rpm/deb Python package hooks."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

_HOOK_DIR = (
    Path(__file__).parents[2]
    / ".."
    / "scripts"
    / "package"
    / "graph_autofusion"
    / "rpm_deb"
)
_POSTINST = _HOOK_DIR / "custom_postinst.sh"
_PRERM = _HOOK_DIR / "custom_prerm.sh"


def _run_hook(script: Path, install_path: Path | str, *, extra_env=None):
    environment = os.environ.copy()
    environment["INSTALL_PATH"] = str(install_path)
    if extra_env:
        environment.update(extra_env)
    return subprocess.run(
        ["bash", str(script)], capture_output=True, text=True, env=environment
    )


def _fake_python(tmp_path: Path):
    """Return an isolated python3 shim and invocation log for hook tests."""

    bin_dir = tmp_path / "fake-bin"
    bin_dir.mkdir()
    log_path = tmp_path / "fake-python.log"
    executable = bin_dir / "python3"
    executable.write_text(
        f"#!/bin/sh\nprintf '%s\\n' \"$*\" >> {log_path}\nexit 0\n", encoding="utf-8"
    )
    executable.chmod(0o755)
    return bin_dir, log_path


def test_empty_install_path_is_a_safe_noop():
    environment = os.environ.copy()
    environment["INSTALL_PATH"] = ""
    for script in (_POSTINST, _PRERM):
        result = subprocess.run(
            ["bash", str(script)], capture_output=True, text=True, env=environment
        )
        assert result.returncode == 0


def test_postinst_fails_when_required_multistream_wheel_is_missing(tmp_path):
    install_root = tmp_path / "cann-a"
    (install_root / "graph_autofusion" / "lib").mkdir(parents=True)
    result = _run_hook(_POSTINST, install_root)
    assert result.returncode != 0
    assert "required multistream wheel is missing" in result.stderr


def test_postinst_installs_multistream_from_the_staged_root(tmp_path):
    install_root = tmp_path / "cann-a"
    wheel_dir = install_root / "graph_autofusion" / "lib"
    wheel_dir.mkdir(parents=True)
    wheel = wheel_dir / "multistream-0.1.0-py3-none-any.whl"
    wheel.write_bytes(b"wheel fixture")
    fake_bin, invocation_log = _fake_python(tmp_path)
    result = _run_hook(
        _POSTINST,
        install_root,
        extra_env={"PATH": str(fake_bin) + os.pathsep + os.environ["PATH"]},
    )
    assert result.returncode == 0
    invocations = invocation_log.read_text(encoding="utf-8")
    assert "-m pip install" in invocations
    assert str(wheel) in invocations
    assert f"-t {install_root}/python/site-packages" in invocations


@pytest.mark.parametrize("failed_package", ["superkernel", "multistream"])
def test_postinst_preserves_package_failure_policies(tmp_path, failed_package):
    install_root = tmp_path / "cann-a"
    wheel_dir = install_root / "graph_autofusion" / "lib"
    wheel_dir.mkdir(parents=True)
    for package in ("superkernel", "multistream"):
        (wheel_dir / f"{package}-0.1.0-py3-none-any.whl").write_bytes(b"wheel fixture")
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    for command in ("python3", "pip3"):
        executable = fake_bin / command
        executable.write_text(
            '#!/bin/sh\ncase "$*" in\n'
            f"  *{failed_package}-*.whl*) exit 1 ;;\n"
            "esac\nexit 0\n",
            encoding="utf-8",
        )
        executable.chmod(0o755)
    result = _run_hook(
        _POSTINST,
        install_root,
        extra_env={"PATH": str(fake_bin) + os.pathsep + os.environ["PATH"]},
    )
    assert (result.returncode == 0) == (failed_package == "superkernel")


def test_prerm_keeps_superkernel_lookup_when_local_packages_are_absent(tmp_path):
    install_root = tmp_path / "cann-a"
    (install_root / "python" / "site-packages").mkdir(parents=True)
    fake_bin, invocation_log = _fake_python(tmp_path)
    result = _run_hook(
        _PRERM,
        install_root,
        extra_env={"PATH": str(fake_bin) + os.pathsep + os.environ["PATH"]},
    )
    assert result.returncode == 0
    invocations = invocation_log.read_text(encoding="utf-8")
    assert "-m pip uninstall -y superkernel" in invocations
    assert "multistream" not in invocations


def test_prerm_does_not_touch_a_second_install_root(tmp_path):
    first = tmp_path / "cann-a" / "python" / "site-packages"
    second = tmp_path / "cann-b" / "python" / "site-packages"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "multistream").mkdir()
    (first / "multistream" / "owned.py").write_text("owned\n", encoding="utf-8")
    (first / "multistream-0.1.0.dist-info").mkdir()
    (second / "multistream").mkdir()
    (second / "multistream" / "foreign.py").write_text("foreign\n", encoding="utf-8")
    fake_bin, invocation_log = _fake_python(tmp_path)
    result = _run_hook(
        _PRERM,
        first.parents[1],
        extra_env={"PATH": str(fake_bin) + os.pathsep + os.environ["PATH"]},
    )
    assert result.returncode == 0
    assert not (first / "multistream").exists()
    assert (second / "multistream" / "foreign.py").exists()
    assert "-m pip uninstall -y multistream" in invocation_log.read_text(
        encoding="utf-8"
    )


@pytest.mark.parametrize(
    "retry_state", ["complete", "missing_bytecode", "missing_module", "new_stage"]
)
def test_postinst_retry_validates_local_install(tmp_path, installed_wheel, retry_state):
    wheel, _ = installed_wheel
    install_root = tmp_path / "cann-a"
    wheel_dir = install_root / "graph_autofusion" / "lib"
    wheel_dir.mkdir(parents=True)
    shutil.copy2(wheel, wheel_dir / wheel.name)
    first = _run_hook(_POSTINST, install_root)
    assert first.returncode == 0, first.stdout + first.stderr
    assert not wheel_dir.parent.exists()
    package = install_root / "python" / "site-packages" / "multistream"
    assert (package / "__init__.py").is_file()
    if retry_state == "missing_bytecode":
        for cache in package.rglob("__pycache__"):
            cache.parent.chmod(0o750)
            cache.chmod(0o750)
            shutil.rmtree(cache)
    elif retry_state == "missing_module":
        module = package / "common" / "log.py"
        module.parent.chmod(0o750)
        module.unlink()
    elif retry_state == "new_stage":
        wheel_dir.mkdir(parents=True)
    second = _run_hook(_POSTINST, install_root)
    if retry_state in ("complete", "missing_bytecode"):
        assert second.returncode == 0, second.stdout + second.stderr
    else:
        assert second.returncode != 0
        assert "required multistream wheel is missing" in second.stderr


def test_postinst_does_not_accept_package_from_another_root(tmp_path, installed_wheel):
    _, other_site_packages = installed_wheel
    result = _run_hook(
        _POSTINST,
        tmp_path / "empty-install",
        extra_env={"PYTHONPATH": str(other_site_packages)},
    )
    assert result.returncode != 0
    assert "required multistream wheel is missing" in result.stderr
