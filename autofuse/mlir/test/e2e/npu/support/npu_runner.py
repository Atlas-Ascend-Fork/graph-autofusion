#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details and restrictions.
# ----------------------------------------------------------------------------------------------------------------------

import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import sys
import tempfile
import traceback
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

from support import att_selection
from support import case_status


@contextmanager
def capture_fd_stderr() -> Iterator[Dict[str, str]]:
    """在 fd 层面捕获 stderr，并在退出时原样回放。

    为什么必须在 fd 层面而不是 `contextlib.redirect_stderr`：下降失败的真正原因
    （`'afir.broadcast' op has no supported Ascend implementation`、`BindingConflict`、
    `UnsupportedOperation` 等）是 MLIR 的诊断，由 C++ 直接写 fd 2；Python 侧
    `sys.stderr` 的替换对它完全无效。而 inductor 抛出来的异常里**只有**包装文本
    （`Failed to execute graph code: ...`），不含任何拒收原因——只看异常消息会把
    「已知支持边界」全部误判成未知错误。

    退出后把捕获到的文本写回真 stderr，否则诊断就从日志里消失了，失败也就没法复查。
    """
    box: Dict[str, str] = {"text": ""}
    sys.stderr.flush()
    saved = os.dup(2)
    holder = tempfile.TemporaryFile(mode="w+b")
    try:
        os.dup2(holder.fileno(), 2)
        yield box
    finally:
        sys.stderr.flush()
        os.dup2(saved, 2)
        os.close(saved)
        holder.seek(0)
        box["text"] = holder.read().decode("utf-8", "replace")
        holder.close()
        sys.stderr.write(box["text"])
        sys.stderr.flush()


class NpuRunner:
    """Common environment, execution, and validation helpers for NPU E2E cases."""

    def __init__(
        self,
        device: str = "npu:0",
        work_dir: str = "npu_e2e_run",
        profile_steps: int = 100,
        backend_mode: str = "mlir",
        dynamic: bool = False,
    ):
        self.device = device
        self.work_dir = Path(work_dir).resolve()
        self.profile_steps = profile_steps
        self.backend_mode = backend_mode
        self.dynamic = dynamic
        self.torch = None
        self.torch_npu = None

    @staticmethod
    def add_common_arguments(
        parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        default_work_dir = Path(parser.prog).stem
        parser.add_argument(
            "--device", default="npu:0", help="NPU device used by torch_npu."
        )
        parser.add_argument(
            "--work-dir",
            default=default_work_dir,
            help="Directory used for Inductor cache and compile debug artifacts.",
        )
        parser.add_argument(
            "--steps",
            type=int,
            default=100,
            help="Number of compiled NPU iterations recorded by profiling; 0 disables profiling.",
        )
        parser.add_argument(
            "--mode",
            choices=("mlir", "legacy", "both"),
            default="mlir",
            help="Run MLIR codegen, legacy codegen, or both routes for comparison.",
        )
        parser.add_argument(
            "--dynamic",
            action="store_true",
            help="Compile with dynamic=True and reuse one compiled callable across shape cases.",
        )
        return parser

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "NpuRunner":
        return cls(
            device=args.device,
            work_dir=args.work_dir,
            profile_steps=args.steps,
            backend_mode=args.mode,
            dynamic=args.dynamic,
        )

    @classmethod
    def parse_args(cls, description: Optional[str] = None) -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            description=description or "Run an NPU E2E case through Inductor."
        )
        cls.add_common_arguments(parser)
        return parser.parse_args()

    @classmethod
    def run(
        cls,
        case_name: str,
        module_factory: Callable[[Any], Any],
        input_shapes: Optional[Sequence[Sequence[int]]] = None,
        *,
        input_shape_cases: Optional[Sequence[Sequence[Sequence[int]]]] = None,
        description: Optional[str] = None,
        dtype: Any = None,
        rtol: float = 0.0,
        atol: float = 0.0,
        profile_steps: Optional[int] = None,
        dynamic: Optional[bool] = None,
        **kwargs: Any,
    ) -> Any:
        args = cls.parse_args(description)
        runner = cls.from_args(args).setup()
        if dynamic is not None:
            runner.dynamic = dynamic
        module = module_factory(runner.torch)
        return runner.run_case(
            case_name,
            module,
            input_shapes,
            input_shape_cases=input_shape_cases,
            dtype=dtype,
            rtol=rtol,
            atol=atol,
            profile_steps=profile_steps,
            **kwargs,
        )

    @classmethod
    def main(
        cls,
        case_name: str,
        module_factory: Callable[[Any], Any],
        input_shapes: Optional[Sequence[Sequence[int]]] = None,
        **kwargs: Any,
    ) -> int:
        """单用例脚本的统一入口：跑 `run`，判级，打机读结果行，返回退出码。

        与直接调 `run` 的区别只在**失败怎么记账**。`run` 把异常原样抛出去，于是脚本带
        traceback 非零退出，所有失败长得一样——已记录的支持边界（`REFUSED`）和真缺陷
        （`MISMATCH` / `ERROR`）分不开。矩阵表早就不是这样记的，单用例这一侧一直没跟上，
        统一入口要把十几个脚本放进一张表里对账，就必须先让两侧用同一套词汇。

        所以这里做三件事，判级规则与矩阵表共用 `case_status` 的那一份：

        1. 在 fd 层面捕获 stderr——下降失败的真正原因是 MLIR 诊断，由 C++ 直接写 fd 2，
           异常文本里只有包装文本。细节见 `capture_fd_stderr`。
        2. 判级并打一行机读结果，给 `run.py` 跨脚本对账用。人读的
           `<CASE>_NPU_E2E_PASS ...` 行照旧由 `report_pass` 打，一个字没改。
        3. 退出码只在真缺陷上非零，与矩阵表一致。

        粒度是**整个脚本一格**，而不是每个 shape case 一格：判级的依据是一次捕获里累积
        的诊断文本，拆成多格就得为每格单独捕获，而 `run_case` 的编译与 route 循环是按
        脚本组织的（`--dynamic` 还要求一个 compiled callable 跨 shape 复用）。为了判级去
        改执行结构，代价比收益大；矩阵表之所以能逐格判，是因为它本来就一格一次调用。
        """
        args = cls.parse_args(kwargs.pop("description", None))
        error = None
        with capture_fd_stderr() as diagnostics:
            try:
                runner = cls.from_args(args).setup()
                dynamic = kwargs.pop("dynamic", None)
                if dynamic is not None:
                    runner.dynamic = dynamic
                module = module_factory(runner.torch)
                runner.run_case(case_name, module, input_shapes, **kwargs)
            except Exception as caught:  # noqa: BLE001 - 判级要的就是把它接住
                error = caught
                traceback.print_exc()

        status, tag = case_status.classify(error, diagnostics["text"])
        reason = ""
        if error is not None:
            text = str(error).strip()
            reason = text.splitlines()[0] if text else ""
        shapes = (
            input_shapes
            if input_shapes is not None
            else kwargs.get("input_shape_cases", "")
        )
        print(
            case_status.emit_case(case_name, "", status, tag, shapes, reason),
            flush=True,
        )
        print(case_status.emit_suite(case_name, {status: 1}), flush=True)
        label = f"{status}({tag})" if tag else status
        print(f"{case_name}_STATUS {label}", flush=True)
        return 1 if case_status.is_defect(status) else 0

    def setup(self) -> "NpuRunner":
        if self.profile_steps < 0:
            raise ValueError("--steps must be non-negative.")
        self.work_dir.mkdir(parents=True, exist_ok=True)
        device_id = self.device.rsplit(":", 1)[-1] if ":" in self.device else "0"
        os.environ.setdefault("ASCEND_DEVICE_ID", device_id)
        if "--mlir_codegen_config=" not in os.environ.get("AUTOFUSE_DFX_FLAGS", ""):
            os.environ.update(att_selection.environment_for_mlir_config("mlir"))
        self.configure_work_dir(self.work_dir)
        os.chdir(self.work_dir)

        import torch
        import torch_npu

        self.torch = torch
        self.torch_npu = torch_npu
        torch.npu.set_device(self.device)
        return self

    def configure_work_dir(self, work_dir: Path) -> None:
        work_dir.mkdir(parents=True, exist_ok=True)
        os.environ["TORCH_COMPILE_DEBUG"] = "1"
        os.environ["TORCH_COMPILE_DEBUG_DIR"] = str(work_dir)
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(work_dir / "torchinductor")
        os.environ["INDUCTOR_CACHE_DIR"] = str(work_dir / "inductor_cache")
        if self.torch is not None:
            self.torch._dynamo.config.debug_dir_root = str(
                work_dir / "torch_compile_debug"
            )

    def make_inputs(
        self, input_shapes: Sequence[Sequence[int]], dtype: Any = None
    ) -> Tuple[Any, ...]:
        """Generate reproducible CPU inputs from the shapes declared by an E2E case."""
        if self.torch is None:
            raise RuntimeError(
                "NpuRunner.setup() must be called before generating inputs."
            )
        dtype = self.torch.float32 if dtype is None else dtype
        generator = self.torch.Generator(device="cpu")
        generator.manual_seed(0)
        return tuple(
            self.torch.randn(tuple(shape), dtype=dtype, generator=generator)
            for shape in input_shapes
        )

    @staticmethod
    def _normalize_shape(shape: Sequence[int]) -> Tuple[int, ...]:
        normalized = tuple(shape)
        if not normalized:
            raise ValueError("input tensor shape must contain at least one dimension.")
        if any(
            isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0
            for dim in normalized
        ):
            raise ValueError(
                f"input tensor shape must contain positive integer dimensions: {normalized}"
            )
        return normalized

    def normalize_shape_cases(
        self,
        input_shapes: Optional[Sequence[Sequence[int]]],
        input_shape_cases: Optional[Sequence[Sequence[Sequence[int]]]],
    ) -> Tuple[Tuple[Tuple[int, ...], ...], ...]:
        if input_shapes is not None and input_shape_cases is not None:
            raise ValueError(
                "input_shapes and input_shape_cases are mutually exclusive."
            )
        if input_shape_cases is None:
            if input_shapes is None:
                raise ValueError(
                    "one of input_shapes or input_shape_cases must be provided."
                )
            input_shape_cases = (input_shapes,)

        normalized_cases = tuple(
            tuple(self._normalize_shape(shape) for shape in shape_case)
            for shape_case in input_shape_cases
        )
        if not normalized_cases:
            raise ValueError("input_shape_cases must contain at least one shape case.")
        input_count = len(normalized_cases[0])
        if input_count == 0:
            raise ValueError("each shape case must contain at least one input tensor.")
        if any(len(shape_case) != input_count for shape_case in normalized_cases):
            raise ValueError(
                "all shape cases must contain the same number of input tensors."
            )
        if self.dynamic:
            ranks = tuple(len(shape) for shape in normalized_cases[0])
            if any(
                tuple(len(shape) for shape in shape_case) != ranks
                for shape_case in normalized_cases[1:]
            ):
                raise ValueError(
                    "dynamic shape cases must keep the rank of every input tensor unchanged."
                )
        return normalized_cases

    def to_npu(self, value: Any) -> Any:
        if self.torch is not None and isinstance(value, self.torch.Tensor):
            return value.to(self.device)
        if isinstance(value, tuple):
            return tuple(self.to_npu(item) for item in value)
        if isinstance(value, list):
            return [self.to_npu(item) for item in value]
        if isinstance(value, dict):
            return {key: self.to_npu(item) for key, item in value.items()}
        return value

    def to_cpu(self, value: Any) -> Any:
        if self.torch is not None and isinstance(value, self.torch.Tensor):
            return value.cpu()
        if isinstance(value, tuple):
            return tuple(self.to_cpu(item) for item in value)
        if isinstance(value, list):
            return [self.to_cpu(item) for item in value]
        if isinstance(value, dict):
            return {key: self.to_cpu(item) for key, item in value.items()}
        return value

    def synchronize(self) -> None:
        self.torch.npu.synchronize()

    def assert_close(self, actual: Any, expected_cpu: Any, **kwargs: Any) -> None:
        self.synchronize()
        self.torch.testing.assert_close(self.to_cpu(actual), expected_cpu, **kwargs)

    def run_eager(
        self,
        fn: Any,
        *inputs: Any,
        expected_cpu: Any,
        rtol: float = 0.0,
        atol: float = 0.0,
        **kwargs: Any,
    ) -> Any:
        actual = fn(*self.to_npu(inputs), **self.to_npu(kwargs))
        self.assert_close(actual, expected_cpu, rtol=rtol, atol=atol)
        return actual

    def compile(self, module: Any, **kwargs: Any) -> Any:
        options = {"npu_backend": "ascendc"}
        options.update(kwargs.pop("options", {}))
        kwargs.setdefault("dynamic", self.dynamic)
        kwargs.setdefault("fullgraph", True)
        return self.torch.compile(module.to(self.device), options=options, **kwargs)

    def reset_compile_cache(self) -> None:
        reset = getattr(getattr(self.torch, "_dynamo", None), "reset", None)
        if reset is not None:
            reset()

    def backend_routes(self) -> Tuple[str, ...]:
        if self.backend_mode == "both":
            return ("legacy", "mlir")
        return (self.backend_mode,)

    @contextmanager
    def use_backend_route(self, route: str):
        if route not in ("mlir", "legacy"):
            raise ValueError("backend route must be mlir or legacy.")
        old_work_dir = self.work_dir
        old_cwd = os.getcwd()
        old_debug_dir_root = self.torch._dynamo.config.debug_dir_root
        old_env = {
            key: os.environ.get(key)
            for key in (
                "AUTOFUSE_DFX_FLAGS",
                "TORCH_COMPILE_DEBUG",
                "TORCH_COMPILE_DEBUG_DIR",
                "TORCHINDUCTOR_CACHE_DIR",
                "INDUCTOR_CACHE_DIR",
            )
        }

        route_work_dir = (
            old_work_dir / route if self.backend_mode == "both" else old_work_dir
        )
        self.work_dir = route_work_dir.resolve()
        self.configure_work_dir(self.work_dir)
        mlir_config = "mlir"
        if att_selection.flags(os.environ).get("--mlir_codegen_config") == "fallback":
            mlir_config = "fallback"
        os.environ.update(
            att_selection.environment_for_mlir_config(
                mlir_config if route == "mlir" else "none"
            )
        )
        os.chdir(self.work_dir)
        self.reset_compile_cache()
        try:
            yield
        finally:
            self.reset_compile_cache()
            self.work_dir = old_work_dir
            self.torch._dynamo.config.debug_dir_root = old_debug_dir_root
            os.chdir(old_cwd)
            for key, value in old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def profile(self, fn: Any, *inputs: Any, steps: int) -> Any:
        """Record repeated execution of an already-compiled NPU callable."""
        self._validate_profile_steps(steps)
        if steps == 0:
            return fn(*inputs)
        if self.torch_npu is None:
            raise RuntimeError("NpuRunner.setup() must be called before profiling.")

        profiling_dir = self.work_dir / "profiling"
        profiling_dir.mkdir(parents=True, exist_ok=True)
        profiler = self.torch_npu.profiler
        experimental_config = profiler._ExperimentalConfig(
            export_type=[profiler.ExportType.Text],
            profiler_level=profiler.ProfilerLevel.Level2,
            msprof_tx=False,
            aic_metrics=profiler.AiCMetrics.PipeUtilization,
            l2_cache=False,
            op_attr=False,
            data_simplification=False,
            record_op_args=False,
            gc_detect_threshold=None,
        )
        with profiler.profile(
            activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU],
            schedule=profiler.schedule(wait=0, warmup=1, active=steps, repeat=1),
            on_trace_ready=profiler.tensorboard_trace_handler(str(profiling_dir)),
            record_shapes=True,
            profile_memory=False,
            with_stack=False,
            with_modules=False,
            with_flops=False,
            experimental_config=experimental_config,
        ) as prof:
            actual = None
            for _ in range(steps + 1):
                actual = fn(*inputs)
                prof.step()
        self.synchronize()
        return actual

    @staticmethod
    def _validate_profile_steps(steps: int) -> None:
        if steps < 0:
            raise ValueError("profile steps must be non-negative.")

    def run_inductor(
        self,
        module: Any,
        *inputs: Any,
        expected_cpu: Any,
        rtol: float = 0.0,
        atol: float = 0.0,
        profile_steps: Optional[int] = None,
        compiled: Any = None,
        **kwargs: Any,
    ) -> Any:
        profile_steps = self.profile_steps if profile_steps is None else profile_steps
        self._validate_profile_steps(profile_steps)
        if compiled is None:
            compiled = self.compile(module, **kwargs)
        actual = compiled(*self.to_npu(inputs))
        self.assert_close(actual, expected_cpu, rtol=rtol, atol=atol)
        if profile_steps > 0:
            actual = self.profile(compiled, *self.to_npu(inputs), steps=profile_steps)
            self.assert_close(actual, expected_cpu, rtol=rtol, atol=atol)
        return actual

    def run_case(
        self,
        case_name: str,
        module: Any,
        input_shapes: Optional[Sequence[Sequence[int]]] = None,
        *,
        input_shape_cases: Optional[Sequence[Sequence[Sequence[int]]]] = None,
        dtype: Any = None,
        rtol: float = 0.0,
        atol: float = 0.0,
        profile_steps: Optional[int] = None,
        **kwargs: Any,
    ) -> Any:
        """Run a case whose inputs and CPU oracle are derived from its module and shapes."""
        if self.torch is None:
            raise RuntimeError(
                "NpuRunner.setup() must be called before running a case."
            )

        shape_cases = self.normalize_shape_cases(input_shapes, input_shape_cases)
        module.eval()
        test_cases = []
        with self.torch.no_grad():
            for shape_case in shape_cases:
                inputs_cpu = self.make_inputs(shape_case, dtype=dtype)
                expected_cpu = module(*inputs_cpu)
                test_cases.append((shape_case, inputs_cpu, expected_cpu))

        module = module.to(self.device)
        for _, inputs_cpu, expected_cpu in test_cases:
            self.run_eager(
                module, *inputs_cpu, expected_cpu=expected_cpu, rtol=rtol, atol=atol
            )

        results = {}
        for route in self.backend_routes():
            with self.use_backend_route(route):
                compiled = self.compile(module, **kwargs)
                route_results = []
                for shape_case, inputs_cpu, expected_cpu in test_cases:
                    artifact_count_before = len(self.collect_autofuse_artifacts())
                    actual = self.run_inductor(
                        module,
                        *inputs_cpu,
                        expected_cpu=expected_cpu,
                        rtol=rtol,
                        atol=atol,
                        profile_steps=profile_steps,
                        compiled=compiled,
                    )
                    artifact_count_after = len(self.collect_autofuse_artifacts())
                    self.report_pass(
                        case_name,
                        route,
                        actual,
                        shape_case=shape_case,
                        new_artifact_count=artifact_count_after - artifact_count_before,
                    )
                    route_results.append(actual)
                results[route] = (
                    route_results[0]
                    if len(route_results) == 1
                    else tuple(route_results)
                )
        if len(results) == 1:
            return next(iter(results.values()))
        return results

    def collect_autofuse_artifacts(self) -> List[Path]:
        debug_root = self.work_dir / "torch_compile_debug"
        if not debug_root.exists():
            return []
        return sorted(
            path
            for path in debug_root.rglob("*")
            if path.is_dir() and path.name.startswith("autofused_")
        )

    def collect_profiling_runs(self) -> List[Path]:
        profiling_root = self.work_dir / "profiling"
        if not profiling_root.exists():
            return []
        return sorted(
            path
            for path in profiling_root.iterdir()
            if path.is_dir()
            and (path.name.startswith("PROF_") or path.name.endswith("_ascend_pt"))
        )

    def report_pass(
        self,
        case_name: str,
        backend_route: str,
        actual: Any,
        *,
        shape_case: Optional[Sequence[Sequence[int]]] = None,
        new_artifact_count: Optional[int] = None,
    ) -> None:
        artifacts = self.collect_autofuse_artifacts()
        profiling_runs = self.collect_profiling_runs()
        actual_shape = getattr(actual, "shape", None)
        shape = tuple(actual_shape) if actual_shape is not None else None
        dtype = getattr(actual, "dtype", None)
        device = getattr(actual, "device", None)
        input_shape_text = (
            tuple(tuple(shape) for shape in shape_case)
            if shape_case is not None
            else None
        )
        print(
            f"{case_name}_NPU_E2E_PASS backend_route={backend_route} "
            f"input_shapes={input_shape_text} output_shape={shape} dtype={dtype} device={device} "
            f"dynamic={self.dynamic}"
        )
        print(f"backend_route={backend_route}")
        print(
            "mlir_codegen_config="
            + att_selection.flags(os.environ).get("--mlir_codegen_config", "none")
        )
        print(f"autofused_artifact_count={len(artifacts)}")
        if new_artifact_count is not None:
            print(f"new_autofused_artifact_count={new_artifact_count}")
        print(f"profiling_dir={self.work_dir / 'profiling'}")
        print(f"profiling_run_count={len(profiling_runs)}")
        for profiling_run in profiling_runs[:10]:
            print(f"profiling_run={profiling_run.relative_to(self.work_dir)}")
        for artifact in artifacts[:10]:
            print(f"autofused_artifact={artifact.relative_to(self.work_dir)}")
