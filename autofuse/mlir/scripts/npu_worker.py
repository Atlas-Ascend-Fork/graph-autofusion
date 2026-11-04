#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
# See LICENSE in the repository root for details.
"""Device worker: hold a workspace lock across checkout, build, overlay and tests."""

import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def command(argv, **kwargs):
    return subprocess.check_output([str(a) for a in argv], text=True, **kwargs).strip()


def checkout(repo, bundle, commit):
    created = not repo.exists()
    try:
        if created:
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
        if command(["git", "-C", repo, "status", "--porcelain"]):
            raise RuntimeError(
                "Managed source has local changes; preserve or remove them before retrying"
            )
        subprocess.run(
            ["git", "-C", str(repo), "fetch", str(bundle), "refs/heads/npu-snapshot"],
            check=True,
        )
        fetched = command(["git", "-C", repo, "rev-parse", "FETCH_HEAD"])
        if fetched != commit:
            raise RuntimeError("Bundle commit does not match request")
        # LFS assets are documentation images; bundles contain their tracked pointers.
        # Do not contact a remote LFS service during an offline source checkout.
        checkout_env = dict(os.environ, GIT_LFS_SKIP_SMUDGE="1")
        subprocess.run(
            ["git", "-C", str(repo), "checkout", "--detach", commit],
            env=checkout_env,
            check=True,
        )
    except Exception:
        # An interrupted first checkout must not strand an unborn, partially populated tree.
        # Existing workspaces are never erased on failure.
        if created and repo.exists():
            shutil.rmtree(repo)
        raise


def build_config(request, env):
    cann = Path(request["cann"]).resolve()
    llvm = Path(request["llvm"]).resolve()
    version_files = [
        cann / "version.info",
        cann / "set_env.sh",
        cann / "compiler/version.info",
        cann / "opp/version.info",
    ]
    version_files.extend(sorted((cann / "include/version").glob("*.h")))
    versions = {str(p): digest(p) for p in version_files if p.is_file()}
    return dict(
        cann=str(cann),
        llvm=str(llvm),
        deps=str(Path(request["deps"]).resolve()),
        cann_cmake=str(Path(request["cann_cmake"]).resolve()),
        versions=versions,
        cmake=command(["cmake", "--version"], env=env),
        compiler=command([env.get("CXX", "c++"), "--version"], env=env),
        llvm_version=command([llvm / "bin/llvm-config", "--version"], env=env),
        python=request["python"],
        python_home=request["python_home"],
        python_version=command([request["python"], "-E", "--version"], env=env),
    )


def probe_runtime(python, env, output):
    with (output / "runtime-import.log").open("w") as log:
        subprocess.run(
            [
                python,
                "-c",
                "import torch, torch_npu, autofuse.pyautofuse; "
                "torch.npu.set_device(0); print(torch.__version__, torch_npu.__version__)",
            ],
            cwd=output,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )


def execute(request, root, output, state, save):
    repo, build, overlay = (root / name for name in ("source", "build", "overlay"))
    if any(path.is_symlink() for path in (repo, build, overlay)):
        raise RuntimeError("Managed source, build and overlay must not be symlinks")
    bundle = root / "incoming" / request["run_id"] / "source.bundle"
    if digest(bundle) != request["bundle_sha256"]:
        raise RuntimeError("Bundle checksum mismatch")
    if digest(__file__) != request["worker_sha256"]:
        raise RuntimeError("Worker checksum mismatch")
    state["phase"] = "sync"
    save()
    checkout(repo, bundle, request["commit"])
    bundle.unlink()  # Git objects persist in the single source checkout.
    state["source_commit"] = command(["git", "-C", repo, "rev-parse", "HEAD"])
    save()
    # A failed/replaced build must never retain a success stamp from an older commit.
    stamp = root / "build-state.json"
    previous = json.loads(stamp.read_text()) if stamp.exists() else None
    stamp.unlink(missing_ok=True)
    env = dict(os.environ)
    for key in (
        "PYTHONPATH",
        "PYTHONHOME",
        "LD_LIBRARY_PATH",
        "AUTOFUSE_BUILD_DIR",
        "AUTOFUSE_DFX_FLAGS",
        "AF_MLIR_PINNED_PLAN_ID",
        "AF_MLIR_ENABLE_PLAN_PINNING",
    ):
        env.pop(key, None)
    env_result = subprocess.check_output(
        [
            "bash",
            "-c",
            'source "$1" >/dev/null 2>&1 && env -0',
            "npu-env",
            str(Path(request["cann"]) / "set_env.sh"),
        ],
        env=env,
    )
    env = dict(item.decode().split("=", 1) for item in env_result.split(b"\0") if item)
    env["LLVM_BUILD_DIR"] = request["llvm"]
    env["ASCEND_HOME_PATH"] = request["cann"]
    env["AF_NPU_PYTHON"] = request["python"]
    # Device Python may retain a build-time prefix that is absent/incompatible at runtime.
    # AGENTS.md requires an explicit standard-library prefix, not sys.base_prefix.
    env["PYTHONHOME"] = request["python_home"]
    config = build_config(request, env)
    # Keep the fingerprint even on failed builds so toolchain changes invalidate partial objects.
    config_path = root / "build-config.json"
    old_config = json.loads(config_path.read_text()) if config_path.exists() else None
    if build.exists() and (request["clean_build"] or old_config != config):
        shutil.rmtree(build)
    write_json(config_path, config)
    state.update(phase="build", config=config, previous_build=previous)
    save()
    cmake = [
        "cmake",
        "-S",
        str(repo),
        "-B",
        str(build),
        "-DENABLE_AUTOFUSE_MLIR=ON",
        "-DENABLE_AUTOFUSE_MLIR_TESTS=OFF",
        "-DRUN_TEST=0",
        "-DPython3_EXECUTABLE=" + request["python"],
        "-DASCEND_INSTALL_PATH=" + request["cann"],
        "-DCANN_3RD_LIB_PATH=" + request["deps"],
        "-DFETCHCONTENT_SOURCE_DIR_CANN-CMAKE=" + request["cann_cmake"],
    ]
    for filename, argv in (
        ("configure.log", cmake),
        (
            "build.log",
            ["cmake", "--build", str(build), "--target", "pyautofuse", "-j", "8"],
        ),
    ):
        with (output / filename).open("w") as log:
            subprocess.run(
                argv,
                cwd=repo,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
    library = build / "autofuse/libaihac_codegen.so"
    module = build / "autofuse/compiler/py_module/pyautofuse.so"
    build_state = dict(
        commit=request["commit"],
        config=config,
        binaries={str(p): digest(p) for p in (library, module)},
    )
    write_json(stamp, build_state)
    state.update(phase="preflight", build=build_state)
    save()
    # Use the selected commit's existing environment/preflight implementation.
    spec = importlib.util.spec_from_file_location(
        "npu_device_env", repo / "autofuse/mlir/test/e2e/npu/support/device_env.py"
    )
    device_env = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(device_env)
    os.environ.clear()
    os.environ.update(env)
    device_env.ensure_overlay(repo, build / "autofuse", overlay, rebuild=True)
    test_env = device_env.environ_for_run(repo, build / "autofuse", overlay)
    origins = device_env.preflight(test_env, overlay, request["python"])
    resolved = command(
        ["ldd", overlay / "python/site-packages/autofuse/pyautofuse.so"], env=test_env
    )
    (output / "preflight.json").write_text(json.dumps(origins, indent=2) + "\n")
    (output / "libraries.log").write_text(resolved + "\n")
    if "libaihac_codegen.so => " + str(library) + " " not in resolved:
        raise RuntimeError("Codegen library resolved outside the managed build")
    if digest(overlay / "python/site-packages/autofuse/pyautofuse.so") != digest(
        module
    ):
        raise RuntimeError("Overlay extension does not match build")
    probe_runtime(request["python"], test_env, output)
    if command(["git", "-C", repo, "rev-parse", "HEAD"]) != request["commit"]:
        raise RuntimeError("Source commit changed during build")
    if command(["git", "-C", repo, "diff", "HEAD", "--"]):
        raise RuntimeError("Tracked source changed during build")
    state["phase"] = "test"
    save()
    argv = [
        request["python"],
        "-u",
        str(repo / "autofuse/mlir/test/e2e/npu/run.py"),
        "--mode",
        "mlir",
        "--jobs",
        "1",
        "--build-dir",
        str(build / "autofuse"),
        "--overlay",
        str(overlay),
        "--work-dir",
        str(output / "cases"),
    ]
    for key in ("group", "id"):
        if request[key]:
            argv.extend(["--" + key, request[key]])
    with (output / "test.log").open("w") as log:
        result = subprocess.run(
            argv, cwd=repo, env=test_env, stdout=log, stderr=subprocess.STDOUT
        )
    state.update(phase="complete", returncode=result.returncode)
    return result.returncode


def main(request_file):
    request = json.loads(Path(request_file).read_text())
    root = Path(request["root"]).resolve()
    assert (root / ".npu-workspace").read_text() == "autofuse-npu-v1\n"
    assert Path(request["run_id"]).name == request["run_id"] and request[
        "run_id"
    ] not in (".", "..")
    output = root / "runs" / request["run_id"]
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "request.json", request)
    state = dict(
        commit=request["commit"], pid=os.getpid(), phase="locking", started=time.time()
    )

    def save():
        write_json(output / "status.json", state)

    save()
    with (root / "session.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            code = execute(request, root, output, state, save)
        except Exception as error:
            state.update(
                failed_phase=state["phase"],
                phase="failed",
                error=str(error),
                returncode=1,
            )
            code = 1
        finally:
            state["finished"] = time.time()
            save()
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
