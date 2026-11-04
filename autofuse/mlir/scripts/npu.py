#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
# See LICENSE in the repository root for details.
"""Submit a commit to one persistent NPU workspace without pushing to a remote."""

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import uuid


def run(argv, **kwargs):
    return subprocess.run(argv, check=True, **kwargs)


def ssh(host, argv):
    return run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=15",
            host,
            shlex.join([str(v) for v in argv]),
        ],
        text=True,
        capture_output=True,
    ).stdout


def make_bundle(repo, commit, destination):
    sha = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "--verify", commit + "^{commit}"],
        text=True,
    ).strip()
    # A private ref names arbitrary commits for git bundle; never alter caller refs.
    with tempfile.TemporaryDirectory() as temp:
        run(["git", "clone", "--bare", "--shared", "--quiet", str(repo), temp])
        run(["git", "-C", temp, "update-ref", "refs/heads/npu-snapshot", sha])
        run(
            [
                "git",
                "-C",
                temp,
                "bundle",
                "create",
                str(destination),
                "refs/heads/npu-snapshot",
            ]
        )
    return sha


def initialize_workspace(host, root, python, run_id=""):
    bootstrap = """from pathlib import Path
import sys
p=Path(sys.argv[1]); p.mkdir(parents=True, exist_ok=True)
m=p/'.npu-workspace'
if m.exists():
    assert m.read_text() == 'autofuse-npu-v1\\n', 'unknown workspace format'
else:
    assert not any(p.iterdir()), 'refusing nonempty unmanaged workspace'
    m.write_text('autofuse-npu-v1\\n')
if sys.argv[2]:
    (p/'incoming'/sys.argv[2]).mkdir(parents=True)
else:
    (p/'deps').mkdir(exist_ok=True)
"""
    ssh(host, [python, "-c", bootstrap, root, run_id])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, help="SSH config alias")
    parser.add_argument(
        "--root", required=True, help="Dedicated absolute device workspace"
    )
    sub = parser.add_subparsers(dest="action", required=True)
    initialize = sub.add_parser("init")
    initialize.add_argument("--python", default="/usr/bin/python3.12")
    launch = sub.add_parser("run")
    launch.add_argument("--commit", default="HEAD")
    launch.add_argument(
        "--cann", required=True, help="Device CANN root containing set_env.sh"
    )
    launch.add_argument("--llvm", required=True, help="Device LLVM installation")
    launch.add_argument(
        "--deps", required=True, help="Persistent CANN third-party cache"
    )
    launch.add_argument(
        "--cann-cmake", required=True, help="Persistent cann-cmake sources"
    )
    launch.add_argument(
        "--python", default="/usr/bin/python3.12", help="Device Python with torch_npu"
    )
    launch.add_argument(
        "--python-home", default="/usr", help="Runtime standard-library prefix"
    )
    launch.add_argument("--group", default="")
    launch.add_argument("--id", default="")
    launch.add_argument("--clean-build", action="store_true")
    status = sub.add_parser("status")
    status.add_argument("run_id")
    args = parser.parse_args()
    if not args.root.startswith("/") or args.root == "/" or args.host.startswith("-"):
        parser.error("Use an absolute dedicated root and a valid SSH alias")
    if args.action == "init":
        initialize_workspace(args.host, args.root, args.python)
        print("Initialized:", args.root)
        return
    if args.action == "status":
        if "/" in args.run_id or args.run_id in (".", ".."):
            parser.error("run_id must be a single directory name")
        print(
            ssh(args.host, ["cat", args.root + "/runs/" + args.run_id + "/status.json"])
        )
        return
    repo = Path(__file__).resolve().parents[3]
    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp)
        bundle = temp / "source.bundle"
        sha = make_bundle(repo, args.commit, bundle)
        run_id = sha[:12] + "-" + uuid.uuid4().hex[:12]
        request = vars(args).copy()
        request.update(
            commit=sha,
            run_id=run_id,
            bundle_sha256=hashlib.sha256(bundle.read_bytes()).hexdigest(),
        )
        worker = Path(__file__).with_name("npu_worker.py")
        request["worker_sha256"] = hashlib.sha256(worker.read_bytes()).hexdigest()
        (temp / "request.json").write_text(json.dumps(request, indent=2) + "\n")
        remote = args.root + "/incoming/" + run_id
        initialize_workspace(args.host, args.root, args.python, run_id)
        for source, name in (
            (bundle, "source.bundle"),
            (temp / "request.json", "request.json"),
            (worker, "worker.py"),
        ):
            with source.open("rb") as stream:
                run(
                    [
                        "ssh",
                        "-o",
                        "BatchMode=yes",
                        "-o",
                        "ConnectTimeout=15",
                        args.host,
                        "cat > " + shlex.quote(remote + "/" + name),
                    ],
                    stdin=stream,
                )
        command = shlex.join(
            [args.python, remote + "/worker.py", remote + "/request.json"]
        )
        output = ssh(
            args.host,
            [
                "bash",
                "-c",
                "nohup "
                + command
                + " > "
                + shlex.quote(remote + "/launcher.log")
                + " 2>&1 < /dev/null & echo $!",
            ],
        )
        print("commit:", sha, "\nrun_id:", run_id, "\npid:", output.strip())
        print("status:", args.root + "/runs/" + run_id + "/status.json")
        print("logs:", args.root + "/runs/" + run_id)
        print("Submission only; use status to confirm build and test results.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        print(error.stderr or str(error), file=sys.stderr)
        sys.exit(error.returncode or 1)
