# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import copy
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PACKAGE_ROOT / "scripts"
SCRIPT = SCRIPTS / "artifact_contract.py"
sys.path.insert(0, str(SCRIPTS))

import artifact_contract  # noqa: E402 - load sibling scripts after sys.path setup


CAPTURE_STARTED_NS = 1
FINALIZED_NS = 9_000_000_000_000_000_000
ROLES = tuple(artifact_contract.ROLE_FILES)


class WorkloadContractTest(unittest.TestCase):
    def test_normalizes_aliases_to_one_canonical_workload(self):
        value = {
            "model_name": "deepseek_v4",
            "sequence_length": 1,
            "batch_size": 1,
            "world_size": 16,
            "inference_mode": "decode",
            "warmup_runs": 2,
            "measurement_runs": 3,
            "runtime_options": {
                "static_compile": True,
                "backend": "npugraph_ex",
            },
        }

        self.assertEqual(
            artifact_contract.normalize_workload(value),
            {
                "model": "deepseek_v4",
                "input": 1,
                "batch": 1,
                "rank": 16,
                "mode": "decode",
                "warmup": 2,
                "iterations": 3,
                "runtime": {
                    "backend": "npugraph_ex",
                    "static_compile": True,
                },
            },
        )

    def test_accepts_identical_aliases_and_returns_detached_json_data(self):
        runtime = {"options": ["static", {"cache": True}]}
        value = {
            "model": "m",
            "model_name": "m",
            "input": [1, 2],
            "inputs": [1, 2],
            "batch": 1,
            "batch_size": 1,
            "rank": 8,
            "rank_size": 8,
            "mode": "decode",
            "prefill_decode_mode": "decode",
            "warmup": 0,
            "warmup_iterations": 0,
            "iterations": 3,
            "iteration_count": 3,
            "runtime": runtime,
            "runtime_parameters": copy.deepcopy(runtime),
        }

        normalized = artifact_contract.normalize_workload(value)
        runtime["options"].append("changed")

        self.assertEqual(
            normalized["runtime"], {"options": ["static", {"cache": True}]}
        )
        self.assertEqual(
            artifact_contract.canonical_sha256(normalized),
            artifact_contract.canonical_sha256(
                json.loads(artifact_contract.canonical_json(normalized))
            ),
        )

    def test_rejects_missing_and_conflicting_aliases(self):
        with self.assertRaisesRegex(ValueError, "workload.runtime is required"):
            artifact_contract.normalize_workload(
                {
                    "model": "m",
                    "input": 1,
                    "batch": 1,
                    "rank": 1,
                    "mode": "decode",
                    "warmup": 2,
                    "iterations": 3,
                }
            )

        with self.assertRaisesRegex(ValueError, "workload.batch aliases conflict"):
            artifact_contract.normalize_workload(
                {
                    "model": "m",
                    "input": 1,
                    "batch": 1,
                    "batch_size": 2,
                    "rank": 1,
                    "mode": "decode",
                    "warmup": 2,
                    "iterations": 3,
                    "runtime": {},
                }
            )

    def test_requires_an_object_and_rejects_non_standard_json(self):
        with self.assertRaisesRegex(ValueError, "workload must be an object"):
            artifact_contract.normalize_workload([])

        valid = {
            "model": "m",
            "input": 1,
            "batch": 1,
            "rank": 1,
            "mode": "decode",
            "warmup": 0,
            "iterations": 1,
            "runtime": {},
        }
        for bad_value in (float("nan"), float("inf"), (1, 2), {1: "value"}):
            with self.subTest(value=repr(bad_value)):
                changed = dict(valid)
                changed["runtime"] = {"bad": bad_value}
                with self.assertRaisesRegex(ValueError, "standard JSON"):
                    artifact_contract.normalize_workload(changed)

    def test_rejects_cyclic_and_excessively_deep_json_without_recursion_error(self):
        self.assertTrue(hasattr(artifact_contract, "MAX_JSON_DEPTH"))
        cyclic = []
        cyclic.append(cyclic)
        with self.assertRaisesRegex(ValueError, "cyclic") as cyclic_error:
            artifact_contract.canonical_json(cyclic)
        self.assertLess(len(str(cyclic_error.exception)), 300)

        long_location_cycle = {}
        long_location_cycle["x" * 500] = long_location_cycle
        with self.assertRaisesRegex(ValueError, "cyclic") as long_cycle_error:
            artifact_contract.canonical_json(long_location_cycle)
        self.assertLess(len(str(long_cycle_error.exception)), 300)

        deeply_nested = 0
        for _ in range(artifact_contract.MAX_JSON_DEPTH + 1):
            deeply_nested = [deeply_nested]
        with self.assertRaisesRegex(ValueError, "maximum JSON depth") as depth_error:
            artifact_contract.canonical_json(deeply_nested)
        self.assertLess(len(str(depth_error.exception)), 300)


class RoundFamilyTest(unittest.TestCase):
    def test_accepts_supported_round_suffixes_for_matching_candidate(self):
        for round_id in ("S1-AUTO", "S3-BASE", "S3-P1", "S3-P2", "S3-FINAL"):
            with self.subTest(round_id=round_id):
                self.assertTrue(
                    artifact_contract.round_belongs_to_candidate(
                        round_id.split("-", 1)[0], round_id
                    )
                )

    def test_rejects_invalid_suffix_whitespace_and_candidate_mismatch(self):
        cases = (
            ("S3", "S3-P0"),
            ("S1", "S1-AUTO-X"),
            (" S3", "S3-P1"),
            ("S3", " S3-P1"),
            ("S3", "S3-P1 "),
            ("S2", "S3-P1"),
            ("", "-AUTO"),
        )
        for candidate, round_id in cases:
            with self.subTest(candidate=candidate, round_id=round_id):
                self.assertFalse(
                    artifact_contract.round_belongs_to_candidate(candidate, round_id)
                )


class CollectionContractTest(unittest.TestCase):
    def setUp(self):
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.base = Path(self._temporary_directory.name)
        self.default_workload = {
            "model": "deepseek_v4",
            "input": 1,
            "batch": 1,
            "rank": 16,
            "mode": "decode",
            "warmup": 2,
            "iterations": 3,
            "runtime": {"backend": "npugraph_ex"},
        }

    def tearDown(self):
        self._temporary_directory.cleanup()

    def _write_json(self, name, value):
        path = self.base / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def _prepare_capture(
        self,
        role,
        *,
        suffix="",
        capture_started_ns=CAPTURE_STARTED_NS,
        workload=None,
        config=None,
        control=None,
        source_revision="revision-a",
        native_pid=None,
        producer_started_ns=None,
        producer_ended_ns=None,
        command_fingerprint=None,
    ):
        tag = f"{role}{suffix}"
        root = self.base / f"{tag}-root"
        session = self.base / f"{tag}-session.json"
        with mock.patch.object(
            artifact_contract.time, "time_ns", return_value=capture_started_ns
        ):
            session_record = artifact_contract.begin_collection_session(
                role=role,
                root=root,
                session_out=session,
            )

        specs = []
        for file_role in sorted(artifact_contract.ROLE_FILES[role]):
            relative = Path("artifacts") / f"{file_role}.dat"
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"{tag}:{file_role}".encode("ascii"))
            specs.append(f"{file_role}={relative.as_posix()}")

        role_index = ROLES.index(role)
        producer_started_ns = (
            producer_started_ns
            if producer_started_ns is not None
            else capture_started_ns + 10 + role_index * 10
        )
        producer_ended_ns = (
            producer_ended_ns
            if producer_ended_ns is not None
            else producer_started_ns + 5
        )
        command_fingerprint = command_fingerprint or artifact_contract.canonical_sha256(
            {"command": tag}
        )
        native_pid = native_pid if native_pid is not None else 1000 + role_index

        return {
            "role": role,
            "root": root,
            "session": session,
            "session_record": session_record,
            "source_revision": source_revision,
            "native_pid": native_pid,
            "producer_started_ns": producer_started_ns,
            "producer_ended_ns": producer_ended_ns,
            "command_fingerprint": command_fingerprint,
            "workload_path": self._write_json(
                f"inputs/{tag}-workload.json",
                self.default_workload if workload is None else workload,
            ),
            "config_path": self._write_json(
                f"inputs/{tag}-config.json",
                {"superkernel": False} if config is None else config,
            ),
            "control_path": self._write_json(
                f"inputs/{tag}-control.json",
                {"rank": 0} if control is None else control,
            ),
            "specs": specs,
            "out": root / "collection-manifest.json",
        }

    def _finalize_capture(self, prepared, *, specs=None, out=None):
        with mock.patch.object(
            artifact_contract.time, "time_ns", return_value=FINALIZED_NS
        ):
            return artifact_contract.finalize_collection_manifest(
                session=prepared["session"],
                source_revision=prepared["source_revision"],
                native_pid=prepared["native_pid"],
                started_ns=prepared["producer_started_ns"],
                ended_ns=prepared["producer_ended_ns"],
                command_fingerprint=prepared["command_fingerprint"],
                workload=prepared["workload_path"],
                config=prepared["config_path"],
                control=prepared["control_path"],
                file_specs=prepared["specs"] if specs is None else specs,
                out=prepared["out"] if out is None else out,
            )

    def _create_manifest(self, role, **kwargs):
        prepared = self._prepare_capture(role, **kwargs)
        self._finalize_capture(prepared)
        return prepared["out"]

    def _create_manifest_set(self, *, role_overrides=None):
        role_overrides = role_overrides or {}
        return {
            role: self._create_manifest(role, **role_overrides.get(role, {}))
            for role in ROLES
        }

    def _load_manifest(self, path):
        return json.loads(path.read_text(encoding="utf-8"))

    def _rewrite_manifest(self, path, change):
        manifest = self._load_manifest(path)
        change(manifest)
        payload = dict(manifest)
        payload.pop("manifest_fingerprint", None)
        manifest["manifest_fingerprint"] = artifact_contract.canonical_sha256(payload)
        path.write_text(
            artifact_contract.canonical_json(manifest) + "\n", encoding="utf-8"
        )

    def _run_cli(self, *args):
        return subprocess.run(
            [sys.executable, str(SCRIPT), *map(str, args)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    def _validate_cli_args(self, manifests):
        args = []
        for role, path in reversed(tuple(manifests.items())):
            args.extend(("--manifest", f"{role}={path}"))
        return args

    def test_begin_creates_session_for_new_or_empty_root(self):
        empty_root = self.base / "empty"
        empty_root.mkdir()
        for index, root in enumerate((self.base / "new", empty_root)):
            with self.subTest(root=root):
                session_out = self.base / f"session-{index}.json"
                with mock.patch.object(
                    artifact_contract.time,
                    "time_ns",
                    return_value=CAPTURE_STARTED_NS + index,
                ):
                    session = artifact_contract.begin_collection_session(
                        role="baseline_profile", root=root, session_out=session_out
                    )
                self.assertEqual(session["capture_root"], str(root.resolve()))
                self.assertTrue(session["root_was_empty"])
                self.assertEqual(session, self._load_manifest(session_out))
                marker = root / ".collection-session.json"
                self.assertTrue(marker.is_file())
                self.assertEqual(session, self._load_manifest(marker))

    def test_begin_rejects_content_raced_in_before_marker_publication(self):
        self.assertTrue(hasattr(artifact_contract, "_create_collection_marker"))
        original = artifact_contract._create_collection_marker

        def create_raced_content(root_fd, record):
            raced_fd = os.open(
                "raced-before-marker.dat",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=root_fd,
            )
            os.close(raced_fd)
            return original(root_fd, record)

        root = self.base / "raced-root"
        session_out = self.base / "raced-session.json"
        with mock.patch.object(
            artifact_contract,
            "_create_collection_marker",
            side_effect=create_raced_content,
        ):
            with self.assertRaisesRegex(ValueError, "changed before collection marker"):
                artifact_contract.begin_collection_session(
                    role="baseline_profile", root=root, session_out=session_out
                )
        self.assertTrue(session_out.is_file())
        self.assertEqual(
            self._load_manifest(root / artifact_contract.COLLECTION_MARKER),
            self._load_manifest(session_out),
        )

    def test_begin_rejects_reused_nonempty_root_and_session_inside_root(self):
        root = self.base / "capture"
        root.mkdir()
        (root / "old-artifact").write_text("old", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "capture root must be empty"):
            artifact_contract.begin_collection_session(
                role="baseline_profile",
                root=root,
                session_out=self.base / "session.json",
            )

        empty_root = self.base / "other-capture"
        with self.assertRaisesRegex(ValueError, "session record must be outside"):
            artifact_contract.begin_collection_session(
                role="baseline_profile",
                root=empty_root,
                session_out=empty_root / "session.json",
            )

    def test_begin_rejects_foreign_session_and_keeps_root_empty_on_publish_race(self):
        existing_out = self.base / "existing-session.json"
        existing_out.write_bytes(b"existing-session")
        existing_root = self.base / "existing-session-root"
        with self.assertRaisesRegex(ValueError, "recoverable session record"):
            artifact_contract.begin_collection_session(
                role="baseline_profile",
                root=existing_root,
                session_out=existing_out,
            )
        self.assertEqual(existing_out.read_bytes(), b"existing-session")
        self.assertEqual(list(existing_root.iterdir()), [])

        raced_out = self.base / "raced-publication-session.json"
        raced_root = self.base / "raced-publication-root"
        original_publish = artifact_contract._atomic_write_json

        def publish_after_foreign_create(path, value):
            Path(path).write_bytes(b"foreign-session")
            return original_publish(path, value)

        with mock.patch.object(
            artifact_contract,
            "_atomic_write_json",
            side_effect=publish_after_foreign_create,
        ):
            with self.assertRaisesRegex(ValueError, "output already exists"):
                artifact_contract.begin_collection_session(
                    role="baseline_profile",
                    root=raced_root,
                    session_out=raced_out,
                )
        self.assertEqual(raced_out.read_bytes(), b"foreign-session")
        self.assertEqual(list(raced_root.iterdir()), [])

        raced_out.unlink()
        retried = artifact_contract.begin_collection_session(
            role="baseline_profile",
            root=raced_root,
            session_out=raced_out,
        )
        self.assertEqual(retried, self._load_manifest(raced_out))

    def test_begin_recovers_after_marker_publication_failure(self):
        root = self.base / "marker-failure-root"
        session_out = self.base / "marker-failure-session.json"

        with mock.patch.object(
            artifact_contract,
            "_create_collection_marker",
            side_effect=OSError("injected marker publication failure"),
        ):
            with self.assertRaisesRegex(OSError, "marker publication failure"):
                artifact_contract.begin_collection_session(
                    role="baseline_profile",
                    root=root,
                    session_out=session_out,
                )
        self.assertTrue(session_out.is_file())
        published_session = self._load_manifest(session_out)
        self.assertEqual(list(root.iterdir()), [])

        recovered = artifact_contract.begin_collection_session(
            role="baseline_profile",
            root=root,
            session_out=session_out,
        )
        self.assertEqual(recovered, published_session)
        self.assertEqual(
            self._load_manifest(root / artifact_contract.COLLECTION_MARKER),
            published_session,
        )

        with mock.patch.object(
            artifact_contract.time,
            "time_ns",
            return_value=CAPTURE_STARTED_NS + 99,
        ):
            repeated = artifact_contract.begin_collection_session(
                role="baseline_profile",
                root=root,
                session_out=session_out,
            )
        self.assertEqual(repeated, published_session)

    def test_begin_rejects_mismatched_existing_session(self):
        root = self.base / "mismatched-session-root"
        root.mkdir()
        session_out = self.base / "mismatched-session.json"
        mismatch = artifact_contract._session_record(
            "candidate_compat", root.resolve(), CAPTURE_STARTED_NS
        )
        session_out.write_text(
            artifact_contract.canonical_json(mismatch) + "\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "does not match requested collection"):
            artifact_contract.begin_collection_session(
                role="baseline_profile",
                root=root,
                session_out=session_out,
            )
        self.assertEqual(list(root.iterdir()), [])
        self.assertEqual(self._load_manifest(session_out), mismatch)

    def test_begin_preserves_replaced_or_foreign_marker(self):
        root = self.base / "replaced-marker-root"
        session_out = self.base / "replaced-marker-session.json"
        marker = root / artifact_contract.COLLECTION_MARKER
        original_marker_validation = artifact_contract._validate_collection_marker
        replaced = False

        def replace_marker_then_validate(root_fd, record):
            nonlocal replaced
            if not replaced:
                replaced = True
                os.unlink(artifact_contract.COLLECTION_MARKER, dir_fd=root_fd)
            else:
                return original_marker_validation(root_fd, record)
            marker_fd = os.open(
                artifact_contract.COLLECTION_MARKER,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=root_fd,
            )
            os.write(marker_fd, b"foreign-marker")
            os.close(marker_fd)
            return original_marker_validation(root_fd, record)

        with mock.patch.object(
            artifact_contract,
            "_validate_collection_marker",
            side_effect=replace_marker_then_validate,
        ):
            with self.assertRaisesRegex(ValueError, "collection marker"):
                artifact_contract.begin_collection_session(
                    role="baseline_profile",
                    root=root,
                    session_out=session_out,
                )
        self.assertTrue(replaced)
        self.assertEqual(marker.read_bytes(), b"foreign-marker")
        self.assertTrue(session_out.is_file())
        published_session = self._load_manifest(session_out)

        with self.assertRaisesRegex(ValueError, "collection marker"):
            artifact_contract.begin_collection_session(
                role="baseline_profile",
                root=root,
                session_out=session_out,
            )
        self.assertEqual(marker.read_bytes(), b"foreign-marker")
        self.assertEqual(self._load_manifest(session_out), published_session)

    def test_json_loader_uses_no_follow_descriptors_and_returns_evidence(self):
        real_dir = self.base / "json-real"
        real_dir.mkdir()
        json_path = real_dir / "input.json"
        json_path.write_text('{"value":1}', encoding="utf-8")

        value, evidence = artifact_contract._load_json(json_path, "input")
        self.assertEqual(value, {"value": 1})
        self.assertIsInstance(evidence, dict)
        self.assertEqual(
            set(evidence),
            {"path", "dev", "ino", "size", "ctime_ns", "mtime_ns", "sha256"},
        )
        self.assertEqual(evidence["path"], str(json_path.absolute()))
        self.assertEqual(
            evidence["sha256"],
            artifact_contract.hashlib.sha256(json_path.read_bytes()).hexdigest(),
        )

        symlink_dir = self.base / "json-link"
        symlink_dir.symlink_to(real_dir, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            artifact_contract._load_json(symlink_dir / "input.json", "input")

        previous_cwd = Path.cwd()
        try:
            os.chdir(self.base)
            relative_value, relative_evidence = artifact_contract._load_json(
                Path("json-real/input.json"), "relative input"
            )
        finally:
            os.chdir(previous_cwd)
        self.assertEqual(relative_value, value)
        self.assertEqual(relative_evidence["path"], str(json_path.absolute()))

    def test_stable_reader_rejects_oversize_input_before_reading(self):
        target = self.base / "oversize-input.json"
        target.write_bytes(b'{"value":1}')

        with mock.patch.object(
            artifact_contract,
            "_read_input_descriptor_bytes",
            wraps=artifact_contract._read_input_descriptor_bytes,
        ) as reader:
            with self.assertRaisesRegex(ValueError, "byte limit"):
                artifact_contract._load_json(target, "bounded input", max_bytes=4)

        reader.assert_not_called()

    def test_stable_reader_enforces_limit_during_descriptor_read(self):
        target = self.base / "growing-input.json"
        target.write_bytes(b"{}")

        with mock.patch.object(
            artifact_contract.os,
            "read",
            side_effect=(b"123", b""),
        ):
            with self.assertRaisesRegex(ValueError, "byte limit"):
                artifact_contract._read_stable_input_bytes(
                    target, "growing input", max_bytes=2
                )

    def test_json_loader_rejects_changes_to_the_open_descriptor_during_read(self):
        target = self._write_json("changing-input.json", {"before": True})
        original_read = artifact_contract.os.read
        triggered = False

        def mutate_then_read(descriptor, size):
            nonlocal triggered
            if not triggered:
                triggered = True
                target.write_text('{"after":true}', encoding="utf-8")
            return original_read(descriptor, size)

        with mock.patch.object(
            artifact_contract.os,
            "read",
            side_effect=mutate_then_read,
        ):
            with self.assertRaisesRegex(ValueError, "changed while reading"):
                artifact_contract._load_json(target, "changing input")
        self.assertTrue(triggered)

    def test_json_loader_rejects_leaf_replacement_after_descriptor_read(self):
        target = self._write_json("replaced-input.json", {"stable": True})
        replacement = self._write_json("replacement-input.json", {"replacement": True})
        original_reader = artifact_contract._read_input_descriptor_bytes
        triggered = False

        def read_then_replace(descriptor, label):
            nonlocal triggered
            data = original_reader(descriptor, label)
            os.replace(replacement, target)
            triggered = True
            return data

        with mock.patch.object(
            artifact_contract,
            "_read_input_descriptor_bytes",
            side_effect=read_then_replace,
        ):
            with self.assertRaisesRegex(ValueError, "path binding changed"):
                artifact_contract._load_json(target, "replaced input")
        self.assertTrue(triggered)

    def test_finalize_rejects_absolute_traversal_and_symlink_paths(self):
        mutations = {}

        def absolute(prepared):
            outside = self.base / "absolute.dat"
            outside.write_text("outside", encoding="utf-8")
            return [
                f"kernel_details={outside}"
                if spec.startswith("kernel_details=")
                else spec
                for spec in prepared["specs"]
            ]

        def traversal(prepared):
            outside = prepared["root"].parent / "outside.dat"
            outside.write_text("outside", encoding="utf-8")
            return [
                "kernel_details=../outside.dat"
                if spec.startswith("kernel_details=")
                else spec
                for spec in prepared["specs"]
            ]

        def symlink(prepared):
            outside = self.base / "symlink-target.dat"
            outside.write_text("outside", encoding="utf-8")
            link = prepared["root"] / "artifacts" / "linked.dat"
            link.symlink_to(outside)
            return [
                "kernel_details=artifacts/linked.dat"
                if spec.startswith("kernel_details=")
                else spec
                for spec in prepared["specs"]
            ]

        def collection_marker(prepared):
            return [
                "kernel_details=.collection-session.json"
                if spec.startswith("kernel_details=")
                else spec
                for spec in prepared["specs"]
            ]

        mutations["absolute"] = (absolute, "relative")
        mutations["traversal"] = (traversal, "traversal")
        mutations["symlink"] = (symlink, "symlink")
        mutations["collection_marker"] = (collection_marker, "collection marker")

        for index, (name, (mutate, message)) in enumerate(mutations.items()):
            with self.subTest(case=name):
                prepared = self._prepare_capture("baseline_profile", suffix=f"-{index}")
                with self.assertRaisesRegex(ValueError, message):
                    self._finalize_capture(prepared, specs=mutate(prepared))

    def test_finalize_accepts_candidate_profile_without_optional_sk_prof(self):
        prepared = self._prepare_capture("candidate_profile")
        specs = [spec for spec in prepared["specs"] if not spec.startswith("sk_prof=")]
        self._finalize_capture(prepared, specs=specs)
        manifest = self._load_manifest(prepared["out"])
        self.assertNotIn("sk_prof", manifest["files"])

    def test_finalize_rejects_files_that_predate_session(self):
        future_start = 8_000_000_000_000_000_000
        prepared = self._prepare_capture(
            "baseline_profile",
            capture_started_ns=future_start,
            producer_started_ns=future_start + 1,
            producer_ended_ns=future_start + 2,
        )
        with self.assertRaisesRegex(ValueError, "predates capture session"):
            self._finalize_capture(prepared)

    def test_finalize_requires_matching_collection_marker(self):
        prepared = self._prepare_capture("baseline_profile")
        marker = prepared["root"] / ".collection-session.json"
        marker.write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "collection marker"):
            self._finalize_capture(prepared)

    def test_finalize_rejects_unlisted_symlinks_and_special_files(self):
        prepared = self._prepare_capture("baseline_profile", suffix="-symlink")
        outside = self.base / "outside-unlisted.dat"
        outside.write_text("outside", encoding="utf-8")
        (prepared["root"] / "unlisted-link").symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "symlink"):
            self._finalize_capture(prepared)

        prepared = self._prepare_capture("baseline_profile", suffix="-fifo")
        os.mkfifo(prepared["root"] / "unlisted-fifo")
        with self.assertRaisesRegex(ValueError, "unsafe special file"):
            self._finalize_capture(prepared)

    def test_finalize_rejects_invalid_strings_times_and_output_overlap(self):
        cases = (
            ("source_revision", " revision-a ", "trimmed non-empty string"),
            ("native_pid", 0, "native_pid must be a positive integer"),
            ("producer_started_ns", -1, "started_ns must be a nonnegative integer"),
            ("producer_ended_ns", 10, "ended_ns must be greater than started_ns"),
        )
        for index, (field, value, message) in enumerate(cases):
            with self.subTest(field=field):
                prepared = self._prepare_capture("baseline_profile", suffix=f"-{index}")
                if field == "producer_ended_ns":
                    prepared["producer_started_ns"] = 10
                prepared[field] = value
                with self.assertRaisesRegex(ValueError, message):
                    self._finalize_capture(prepared)

        prepared = self._prepare_capture("baseline_profile", suffix="-overlap")
        artifact_path = prepared["root"] / prepared["specs"][0].split("=", 1)[1]
        with self.assertRaisesRegex(ValueError, "output overlaps an input"):
            self._finalize_capture(prepared, out=artifact_path)

    def test_finalize_rejects_non_sha256_command_fingerprints(self):
        invalid_fingerprints = (
            "short",
            "g" * 64,
            "A" * 64,
        )
        for index, fingerprint in enumerate(invalid_fingerprints):
            with self.subTest(fingerprint=fingerprint):
                prepared = self._prepare_capture(
                    "baseline_profile",
                    suffix=f"-command-{index}",
                    command_fingerprint=fingerprint,
                )
                with self.assertRaisesRegex(
                    ValueError,
                    "command_fingerprint must be a lowercase SHA-256 fingerprint",
                ):
                    self._finalize_capture(prepared)

    def test_finalize_revalidates_every_json_input_after_parse(self):
        fields = {
            "session": "session",
            "workload": "workload_path",
            "config": "config_path",
            "control": "control_path",
        }
        original_normalize = artifact_contract.normalize_workload
        for index, (label, prepared_key) in enumerate(fields.items()):
            with self.subTest(label=label):
                prepared = self._prepare_capture(
                    "baseline_profile", suffix=f"-input-binding-{index}"
                )
                target = prepared[prepared_key]
                triggered = False

                def mutate_after_parse(value):
                    nonlocal triggered
                    if not triggered:
                        triggered = True
                        target.write_text('{"mutated":true}', encoding="utf-8")
                    return original_normalize(value)

                with mock.patch.object(
                    artifact_contract,
                    "normalize_workload",
                    side_effect=mutate_after_parse,
                ):
                    with self.assertRaisesRegex(ValueError, "input changed"):
                        self._finalize_capture(prepared)
                self.assertTrue(triggered)
                self.assertFalse(prepared["out"].exists())

    def test_finalize_uses_input_inode_identity_for_distinctness(self):
        prepared = self._prepare_capture(
            "baseline_profile", suffix="-hardlinked-inputs"
        )
        prepared["config_path"].unlink()
        os.link(prepared["workload_path"], prepared["config_path"])
        with self.assertRaisesRegex(ValueError, "inputs must be distinct"):
            self._finalize_capture(prepared)

    def test_profile_manifest_contains_every_required_file_role(self):
        path = self._create_manifest("candidate_profile")
        manifest = self._load_manifest(path)

        self.assertEqual(manifest["schema_version"], "1.0")
        self.assertEqual(manifest["capture_role"], "candidate_profile")
        self.assertEqual(manifest["capture_root"], ".")
        self.assertEqual(
            set(manifest["files"]), artifact_contract.ROLE_FILES["candidate_profile"]
        )
        self.assertEqual(
            manifest["workload"],
            artifact_contract.normalize_workload(self.default_workload),
        )
        self.assertIn("config", manifest)
        self.assertIn("control", manifest)
        self.assertEqual(manifest["config"], {"superkernel": False})
        self.assertEqual(manifest["control"], {"rank": 0})
        self.assertEqual(
            manifest["config_fingerprint"],
            artifact_contract.canonical_sha256(manifest["config"]),
        )
        self.assertEqual(
            manifest["control_fingerprint"],
            artifact_contract.canonical_sha256(manifest["control"]),
        )
        for records in manifest["files"].values():
            for record in records:
                self.assertEqual(
                    set(record),
                    {"path", "sha256", "size", "ctime_ns", "mtime_ns"},
                )

    def test_validate_set_rejects_changed_content_and_stat_fingerprints(self):
        for index, mutation in enumerate(("content", "stat")):
            with self.subTest(mutation=mutation):
                manifests = self._create_manifest_set(
                    role_overrides={role: {"suffix": f"-{index}"} for role in ROLES}
                )
                baseline = self._load_manifest(manifests["baseline_profile"])
                record = baseline["files"]["kernel_details"][0]
                artifact = manifests["baseline_profile"].parent / record["path"]
                if mutation == "content":
                    original = artifact.read_bytes()
                    artifact.write_bytes(bytes(byte ^ 1 for byte in original))
                else:
                    stat = artifact.stat()
                    os.utime(
                        artifact,
                        ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000),
                    )
                with self.assertRaisesRegex(
                    ValueError,
                    "content fingerprint changed"
                    if mutation == "content"
                    else "stat fingerprint changed",
                ):
                    artifact_contract.validate_manifest_set(manifests)

    def test_finalize_rejects_path_and_file_swaps_after_descriptor_open(self):
        self.assertTrue(hasattr(artifact_contract, "_read_descriptor_sha256"))
        original_reader = artifact_contract._read_descriptor_sha256

        for index, mutation in enumerate(("replace", "symlink", "in_place")):
            with self.subTest(mutation=mutation):
                prepared = self._prepare_capture(
                    "baseline_profile", suffix=f"-descriptor-{index}"
                )
                relative = next(
                    spec.split("=", 1)[1]
                    for spec in prepared["specs"]
                    if spec.startswith("kernel_details=")
                )
                target = prepared["root"] / relative
                triggered = False

                def race_descriptor(fd, label):
                    nonlocal triggered
                    if label == relative and not triggered:
                        triggered = True
                        if mutation == "replace":
                            replacement = target.with_name("replacement.dat")
                            replacement.write_bytes(b"replacement")
                            os.replace(replacement, target)
                        elif mutation == "symlink":
                            outside = self.base / f"outside-{index}.dat"
                            outside.write_bytes(b"outside")
                            target.unlink()
                            target.symlink_to(outside)
                        else:
                            target.write_bytes(b"changed while descriptor is open")
                    return original_reader(fd, label)

                with mock.patch.object(
                    artifact_contract,
                    "_read_descriptor_sha256",
                    side_effect=race_descriptor,
                ):
                    with self.assertRaisesRegex(
                        ValueError,
                        "artifact (?:path )?changed while hashing",
                    ):
                        self._finalize_capture(prepared)
                self.assertTrue(triggered)

    def test_validate_set_reuses_descriptor_safe_artifact_reader(self):
        self.assertTrue(hasattr(artifact_contract, "_read_descriptor_sha256"))
        manifests = self._create_manifest_set()
        baseline = self._load_manifest(manifests["baseline_profile"])
        relative = baseline["files"]["kernel_details"][0]["path"]
        target = manifests["baseline_profile"].parent / relative
        outside = self.base / "validate-outside.dat"
        outside.write_bytes(b"outside")
        original_reader = artifact_contract._read_descriptor_sha256
        triggered = False

        def swap_to_symlink(fd, label):
            nonlocal triggered
            if label == relative and not triggered:
                triggered = True
                target.unlink()
                target.symlink_to(outside)
            return original_reader(fd, label)

        with mock.patch.object(
            artifact_contract,
            "_read_descriptor_sha256",
            side_effect=swap_to_symlink,
        ):
            with self.assertRaisesRegex(
                ValueError, "artifact path changed while hashing"
            ):
                artifact_contract.validate_manifest_set(manifests)
        self.assertTrue(triggered)

    def test_validate_set_revalidates_manifest_bindings_after_parse(self):
        for index, mutation in enumerate(("content", "same-content-replacement")):
            with self.subTest(mutation=mutation):
                manifests = self._create_manifest_set(
                    role_overrides={
                        role: {"suffix": f"-manifest-binding-{index}"} for role in ROLES
                    }
                )
                target = manifests["baseline_profile"]
                original_content = target.read_bytes()
                original_validate = artifact_contract._validate_manifest
                triggered = False

                def validate_then_mutate(manifest, expected_role, root):
                    nonlocal triggered
                    result = original_validate(manifest, expected_role, root)
                    if expected_role == "baseline_profile" and not triggered:
                        triggered = True
                        if mutation == "content":
                            target.write_bytes(b"{}\n")
                        else:
                            replacement = target.with_name("replacement.json")
                            replacement.write_bytes(original_content)
                            os.replace(replacement, target)
                    return result

                with mock.patch.object(
                    artifact_contract,
                    "_validate_manifest",
                    side_effect=validate_then_mutate,
                ):
                    with self.assertRaisesRegex(ValueError, "manifest input changed"):
                        artifact_contract.validate_manifest_set(manifests)
                self.assertTrue(triggered)

    def test_validate_set_rechecks_manifest_bindings_before_json_publication(self):
        manifests = self._create_manifest_set(
            role_overrides={role: {"suffix": "-json-output-binding"} for role in ROLES}
        )
        target = manifests["baseline_profile"]
        json_out = self.base / "binding-summary.json"
        original_output_check = artifact_contract._validate_summary_output_path
        triggered = False

        def mutate_after_output_check(path, manifest_inputs):
            nonlocal triggered
            output = original_output_check(path, manifest_inputs)
            target.write_bytes(b"{}\n")
            triggered = True
            return output

        stderr = io.StringIO()
        stdout = io.StringIO()
        with (
            mock.patch.object(
                artifact_contract,
                "_validate_summary_output_path",
                side_effect=mutate_after_output_check,
            ),
            mock.patch.object(sys, "stderr", stderr),
            mock.patch.object(sys, "stdout", stdout),
        ):
            result = artifact_contract.main(
                [
                    "validate-set",
                    *self._validate_cli_args(manifests),
                    "--json-out",
                    str(json_out),
                ]
            )
        self.assertTrue(triggered)
        self.assertEqual(result, 2)
        self.assertFalse(json_out.exists())
        self.assertIn("manifest input changed", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_validate_set_rejects_tampered_embedded_config_and_control(self):
        for index, field in enumerate(("config", "control")):
            with self.subTest(field=field):
                manifests = self._create_manifest_set(
                    role_overrides={
                        role: {"suffix": f"-tamper-{field}-{index}"} for role in ROLES
                    }
                )
                manifest = self._load_manifest(manifests["candidate_verify"])
                self.assertIn(field, manifest)
                self._rewrite_manifest(
                    manifests["candidate_verify"],
                    lambda value: value[field].__setitem__("tampered", True),
                )
                with self.assertRaisesRegex(
                    ValueError, f"{field} fingerprint mismatch"
                ):
                    artifact_contract.validate_manifest_set(manifests)

    def test_validate_set_rejects_overlapping_roots_sessions_and_producers(self):
        manifests = self._create_manifest_set()
        copied = manifests["candidate_compat"].parent / "verify-manifest.json"
        copied.write_bytes(manifests["candidate_verify"].read_bytes())
        overlapping = dict(manifests, candidate_verify=copied)
        with self.assertRaisesRegex(
            ValueError, "capture roots must be pairwise distinct"
        ):
            artifact_contract.validate_manifest_set(overlapping)

        manifests = self._create_manifest_set(
            role_overrides={role: {"suffix": "-session"} for role in ROLES}
        )
        compat = self._load_manifest(manifests["candidate_compat"])
        self._rewrite_manifest(
            manifests["candidate_verify"],
            lambda value: value["capture"].__setitem__(
                "session_fingerprint", compat["capture"]["session_fingerprint"]
            ),
        )
        with self.assertRaisesRegex(
            ValueError, "capture session fingerprints must be pairwise distinct"
        ):
            artifact_contract.validate_manifest_set(manifests)

        manifests = self._create_manifest_set(
            role_overrides={role: {"suffix": "-producer"} for role in ROLES}
        )
        compat = self._load_manifest(manifests["candidate_compat"])
        self._rewrite_manifest(
            manifests["candidate_verify"],
            lambda value: value.__setitem__("producer", compat["producer"]),
        )
        with self.assertRaisesRegex(
            ValueError, "producer identities must be pairwise distinct"
        ):
            artifact_contract.validate_manifest_set(manifests)

    def test_validate_set_requires_exactly_all_four_roles(self):
        manifests = self._create_manifest_set()
        manifests.pop("candidate_verify")
        with self.assertRaisesRegex(ValueError, "exactly one manifest for roles"):
            artifact_contract.validate_manifest_set(manifests)

    def test_validate_set_accepts_profile_comparison_role_subset(self):
        manifests = self._create_manifest_set()
        profile_manifests = {
            role: manifests[role] for role in ("baseline_profile", "candidate_profile")
        }

        summary = artifact_contract.validate_manifest_set(
            profile_manifests,
            expected_roles=("baseline_profile", "candidate_profile"),
        )

        self.assertEqual(
            set(summary["roles"]),
            {"baseline_profile", "candidate_profile"},
        )

        cli = self._run_cli(
            "validate-set",
            *self._validate_cli_args(profile_manifests),
        )
        self.assertEqual(cli.returncode, 0, cli.stderr)
        self.assertEqual(json.loads(cli.stdout), summary)

    def test_validate_set_cli_rejects_partial_or_audit_only_role_sets(self):
        manifests = self._create_manifest_set()
        for roles in (
            ("baseline_profile",),
            ("candidate_compat", "candidate_verify"),
            ("baseline_profile", "candidate_profile", "candidate_compat"),
        ):
            with self.subTest(roles=roles):
                selected = {role: manifests[role] for role in roles}
                cli = self._run_cli(
                    "validate-set",
                    *self._validate_cli_args(selected),
                )
                self.assertEqual(cli.returncode, 2)
                self.assertIn("complete four-role audit set", cli.stderr)

    def test_validate_set_rejects_source_workload_config_and_control_mismatch(self):
        variants = {
            "source_revision": {"source_revision": "revision-b"},
            "workload": {
                "workload": dict(self.default_workload, iterations=4),
            },
            "config": {"config": {"superkernel": True}},
            "control": {"control": {"rank": 1}},
        }
        for index, (field, override) in enumerate(variants.items()):
            with self.subTest(field=field):
                common_suffixes = {
                    role: {"suffix": f"-{field}-{index}"} for role in ROLES
                }
                common_suffixes["candidate_verify"].update(override)
                manifests = self._create_manifest_set(role_overrides=common_suffixes)
                with self.assertRaisesRegex(ValueError, f"{field} mismatch"):
                    artifact_contract.validate_manifest_set(manifests)

    def test_validate_set_output_is_deterministic_for_function_and_cli(self):
        manifests = self._create_manifest_set()
        first = artifact_contract.validate_manifest_set(manifests)
        second = artifact_contract.validate_manifest_set(
            dict(reversed(manifests.items()))
        )
        self.assertEqual(first, second)
        self.assertEqual(
            artifact_contract.canonical_json(first),
            artifact_contract.canonical_json(second),
        )

        cli_args = self._validate_cli_args(manifests)
        first_cli = self._run_cli("validate-set", *cli_args)
        second_cli = self._run_cli("validate-set", *cli_args)
        self.assertEqual(first_cli.returncode, 0, first_cli.stderr)
        self.assertEqual(first_cli.stdout, second_cli.stdout)
        self.assertEqual(json.loads(first_cli.stdout), first)

        json_out = self.base / "validation-summary.json"
        written = self._run_cli("validate-set", *cli_args, "--json-out", json_out)
        self.assertEqual(written.returncode, 0, written.stderr)
        self.assertEqual(written.stdout, "")
        self.assertEqual(json.loads(json_out.read_text(encoding="utf-8")), first)

    def test_validate_set_json_output_never_overwrites_or_overlaps_inputs(self):
        cases = ("existing", "manifest", "artifact", "capture_root")
        for index, case in enumerate(cases):
            with self.subTest(case=case):
                manifests = self._create_manifest_set(
                    role_overrides={
                        role: {"suffix": f"-output-{case}-{index}"} for role in ROLES
                    }
                )
                if case == "existing":
                    target = self.base / "existing-summary.json"
                    target.write_bytes(b"do-not-replace")
                elif case == "manifest":
                    target = manifests["baseline_profile"]
                elif case == "artifact":
                    baseline = self._load_manifest(manifests["baseline_profile"])
                    relative = baseline["files"]["kernel_details"][0]["path"]
                    target = manifests["baseline_profile"].parent / relative
                else:
                    target = manifests["baseline_profile"].parent

                before = (
                    target.read_bytes()
                    if target.is_file()
                    else sorted(target.iterdir())
                )
                result = self._run_cli(
                    "validate-set",
                    *self._validate_cli_args(manifests),
                    "--json-out",
                    target,
                )
                after = (
                    target.read_bytes()
                    if target.is_file()
                    else sorted(target.iterdir())
                )
                self.assertEqual(result.returncode, 2)
                self.assertNotIn("Traceback", result.stderr)
                self.assertEqual(after, before)
                if case != "existing":
                    self.assertIn("overlaps capture input", result.stderr)

    def test_cli_errors_are_concise_and_have_no_traceback(self):
        root = self.base / "nonempty"
        root.mkdir()
        (root / "artifact").write_text("old", encoding="utf-8")
        result = self._run_cli(
            "begin",
            "--role",
            "baseline_profile",
            "--root",
            root,
            "--session-out",
            self.base / "session.json",
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("capture root must be empty", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

        result = self._run_cli(
            "validate-set",
            "--manifest",
            f"baseline_profile={self.base / 'missing.json'}",
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("complete four-role audit set", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

        deep_json = self.base / "deep.json"
        deep_json.write_text("[" * 1200 + "0" + "]" * 1200, encoding="utf-8")
        deep_args = []
        for role in ROLES:
            deep_args.extend(("--manifest", f"{role}={deep_json}"))
        result = self._run_cli("validate-set", *deep_args)
        self.assertEqual(result.returncode, 2)
        self.assertIn("JSON nesting", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertLess(len(result.stderr), 300)

    def test_machine_readable_schema_documents_manifest_and_roles(self):
        schema_path = PACKAGE_ROOT / "references" / "collection-manifest-schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.assertEqual(
            schema["$schema"], "https://json-schema.org/draft/2020-12/schema"
        )
        self.assertEqual(schema["properties"]["schema_version"]["const"], "1.0")
        self.assertIn("config", schema["required"])
        self.assertIn("control", schema["required"])
        self.assertEqual(
            schema["$defs"]["producer"]["properties"]["command_fingerprint"],
            {"$ref": "#/$defs/sha256"},
        )
        self.assertEqual(
            {role: set(files) for role, files in schema["x-role-files"].items()},
            artifact_contract.ROLE_FILES,
        )
        self.assertIn("manifest_fingerprint", schema["required"])
        self.assertFalse(schema["additionalProperties"])
        self.assertIn("x-collection-marker", schema)
        self.assertEqual(
            schema["x-collection-marker"]["path"], ".collection-session.json"
        )

    def test_schema_artifact_path_pattern_matches_runtime_path_contract(self):
        schema_path = PACKAGE_ROOT / "references" / "collection-manifest-schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        pattern = re.compile(
            schema["$defs"]["fileRecord"]["properties"]["path"]["pattern"]
        )
        cases = {
            "kernel_details.csv": True,
            "profiler/kernel_details.csv": True,
            ".hidden": True,
            "nested/.hidden/file": True,
            "path with spaces/file.dat": True,
            "": False,
            "/absolute": False,
            "../outside": False,
            "nested/../outside": False,
            "./relative": False,
            "nested/./file": False,
            "nested//file": False,
            "nested/": False,
            ".collection-session.json": False,
            " leading-space/file": False,
            "trailing-space/file ": False,
            "nested\\windows-path": False,
            "nested/\x00nul": False,
            "nested/\ttab": False,
            "nested/\x7fdelete": False,
            "nested/\u0085next-line": False,
            "nested/\ufeffbom": False,
            "nested/non-ascii-é": False,
            "name=part/printable !#$%&'()+,-;@[]^_`{}~.dat": True,
        }
        node_script = (
            "const pattern = new RegExp(process.argv[1]);"
            "const cases = JSON.parse(process.argv[2]);"
            "process.stdout.write(JSON.stringify(cases.map(v => pattern.test(v))));"
        )
        node_result = subprocess.run(
            ["node", "-e", node_script, pattern.pattern, json.dumps(list(cases))],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(node_result.returncode, 0, node_result.stderr)
        ecmascript_results = json.loads(node_result.stdout)
        for value, expected in cases.items():
            with self.subTest(value=value):
                schema_accepts = pattern.fullmatch(value) is not None
                try:
                    artifact_contract._parse_file_specs([f"kernel_details={value}"])
                except ValueError:
                    runtime_accepts = False
                else:
                    runtime_accepts = True
                self.assertEqual(schema_accepts, expected)
                self.assertEqual(runtime_accepts, expected)
                self.assertEqual(ecmascript_results[list(cases).index(value)], expected)


if __name__ == "__main__":
    unittest.main()
