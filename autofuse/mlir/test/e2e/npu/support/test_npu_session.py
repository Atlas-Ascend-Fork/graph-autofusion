#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
# See LICENSE in the repository root for details.
"""Offline regression for commit transport and exclusive device sessions."""

import fcntl
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[4] / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


client, worker = load("npu"), load("npu_worker")


class SessionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / "local"
        subprocess.run(["git", "init", "-q", str(self.repo)], check=True)
        self.git("config", "user.name", "Test")
        self.git("config", "user.email", "test@example.invalid")
        (self.repo / "first").write_text("first\n")
        self.git("add", ".")
        self.git("commit", "-qm", "first")
        self.first = self.git("rev-parse", "HEAD")
        (self.repo / "second").write_text("second\n")
        self.git("add", ".")
        self.git("commit", "-qm", "second")
        self.second = self.git("rev-parse", "HEAD")

    def git(self, *args):
        return worker.command(["git", "-C", self.repo, *args])

    def bundle(self, commit, name="source.bundle"):
        path = self.root / name
        client.make_bundle(self.repo, commit, path)
        return path

    def test_exact_commit_excludes_local_edits(self):
        (self.repo / "first").write_text("dirty\n")
        (self.repo / "untracked").write_text("local only\n")
        remote = self.root / "remote"
        worker.checkout(remote, self.bundle(self.first), self.first)
        self.assertEqual((remote / "first").read_text(), "first\n")
        self.assertFalse((remote / "second").exists())
        self.assertFalse((remote / "untracked").exists())

    def test_switching_commit_removes_obsolete_tracked_files(self):
        remote = self.root / "remote"
        worker.checkout(remote, self.bundle(self.second), self.second)
        worker.checkout(remote, self.bundle(self.first, "old.bundle"), self.first)
        self.assertFalse((remote / "second").exists())
        self.assertEqual(
            worker.command(["git", "-C", remote, "rev-parse", "HEAD"]), self.first
        )

    def test_dirty_remote_is_preserved(self):
        remote = self.root / "remote"
        bundle = self.bundle(self.first)
        worker.checkout(remote, bundle, self.first)
        (remote / "first").write_text("do not erase")
        with self.assertRaisesRegex(RuntimeError, "local changes"):
            worker.checkout(remote, bundle, self.first)
        self.assertEqual((remote / "first").read_text(), "do not erase")

    def test_wrong_commit_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            worker.checkout(self.root / "remote", self.bundle(self.first), self.second)
        self.assertFalse((self.root / "remote").exists())

    def test_checkout_disables_external_asset_download(self):
        (self.repo / ".gitattributes").write_text("*.asset filter=download\n")
        (self.repo / "image.asset").write_text("tracked pointer\n")
        self.git("add", ".")
        self.git("commit", "-qm", "asset")
        sha = self.git("rev-parse", "HEAD")
        remote = self.root / "remote"
        subprocess.run(["git", "init", "-q", str(remote)], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(remote),
                "config",
                "filter.download.smudge",
                'test "$GIT_LFS_SKIP_SMUDGE" = 1 && cat',
            ],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(remote), "config", "filter.download.required", "true"],
            check=True,
        )
        worker.checkout(remote, self.bundle(sha), sha)
        self.assertEqual((remote / "image.asset").read_text(), "tracked pointer\n")

    def test_toolchain_metadata_participates_in_build_fingerprint(self):
        cann = self.root / "cann"
        (cann / "compiler").mkdir(parents=True)
        version = cann / "compiler/version.info"
        version.write_text("version one")
        request = dict(
            cann=str(cann),
            llvm=str(self.root / "llvm"),
            deps=str(self.root / "deps"),
            cann_cmake=str(self.root / "cmake"),
            python="/usr/bin/python3",
            python_home="/usr",
        )
        with patch.object(worker, "command", return_value="tool version"):
            first = worker.build_config(request, {})
            version.write_text("version two")
            second = worker.build_config(request, {})
        self.assertNotEqual(first, second)

    def test_init_refuses_unmanaged_contents(self):
        root = self.root / "existing"
        root.mkdir()
        (root / "keep").write_text("user data")

        def local_ssh(host, argv):
            return subprocess.check_output(argv, text=True, stderr=subprocess.STDOUT)

        with patch.object(client, "ssh", side_effect=local_ssh):
            with self.assertRaises(subprocess.CalledProcessError):
                client.initialize_workspace(
                    "unused", str(root), __import__("sys").executable
                )
        self.assertEqual((root / "keep").read_text(), "user data")

    def test_runtime_import_failure_is_fatal_and_logged(self):
        interpreter = self.root / "broken-python"
        interpreter.write_text('#!/bin/sh\necho "runtime import failed" >&2\nexit 41\n')
        interpreter.chmod(0o700)
        with self.assertRaises(subprocess.CalledProcessError) as error:
            worker.probe_runtime(str(interpreter), {}, self.root)
        self.assertEqual(error.exception.returncode, 41)
        self.assertIn(
            "runtime import failed", (self.root / "runtime-import.log").read_text()
        )

    def request(self):
        root = self.root / "device"
        root.mkdir()
        (root / ".npu-workspace").write_text("autofuse-npu-v1\n")
        request = root / "request.json"
        request.write_text(
            json.dumps(dict(root=str(root), run_id="test", commit=self.first))
        )
        return root, request

    def test_busy_workspace_cannot_start_sync(self):
        root, request = self.request()
        with (root / "session.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with patch.object(worker, "execute") as execute:
                self.assertEqual(worker.main(request), 1)
                execute.assert_not_called()
        self.assertEqual(
            json.loads((root / "runs/test/status.json").read_text())["phase"], "failed"
        )

    def test_build_failure_is_recorded_and_releases_lock(self):
        root, request = self.request()
        with patch.object(worker, "execute", side_effect=RuntimeError("build failed")):
            self.assertEqual(worker.main(request), 1)
        state = json.loads((root / "runs/test/status.json").read_text())
        self.assertEqual(state["error"], "build failed")
        with (root / "session.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)


if __name__ == "__main__":
    unittest.main()
