# Static Kernel Compile Process Safety

This contract applies whenever an adaptation run launches, observes, diagnoses, times
out, or cleans up static-kernel compilation. It is a process-safety gate, not a
performance heuristic.

## Why `*_compile_error.log` Is Provisional

Some CANN static-compilation implementations create `*_compile_error.log` before
launching the corresponding `opc` subprocess. On successful return they rewrite or
rename that file to `*_compile_succ.log`. Therefore an error-named file can coexist
with a healthy live compiler and does not prove failure.

Treat these observations as non-terminal on their own:

- one or many `*_compile_error.log` files exist;
- an `opc` or `op_compiler` process has run longer than an expected duration;
- output is temporarily quiet, CPU utilization changes, or file counts stop changing;
- a child compiler is older than an arbitrary cleanup threshold.

Process age is not hang evidence. File-name state is not process terminal state.

## Allowed State Classification

Classify a compile attempt in this order:

1. **In progress**: any experiment-owned compiler or its inference/rank parent remains
   alive and the frozen top-level runner has not timed out. Keep waiting and collect
   read-only observations. Do not kill anything.
2. **Succeeded**: the compiler stack returned naturally and the final aggregate summary
   reports all inputs successful, with a zero top-level exit status.
3. **Failed**: the compiler stack returned naturally and a nonzero compiler/top-level
   exit or final aggregate summary reports failures. Only now may retained error logs be
   cited as final compiler evidence.
4. **Timed out / externally interrupted**: the frozen top-level timeout fired or an
   external signal ended the process tree. Preserve runner logs and the exact ownership
   record. Mark compile evidence invalid or interrupted, not as a proven operator compile
   failure. A clean retry is required before judging compiler support.
5. **Unknown**: ownership, terminal status, or final summary is missing or contradictory.
   Block the conclusion and collect missing evidence; do not infer failure.

The final summary is runtime-specific. Archive the exact `[summary]` lines or summary
artifact, top-level return code, timeout flag, process ownership manifest, and counts of
final success/error artifacts.

## Termination And Ownership Rules

- Never issue `kill`, `pkill`, `killall`, process-name filtering, or PID-age cleanup
  because `*_compile_error.log` exists. This prohibition applies to the parent Agent,
  child Agents, watchdogs, diagnostic scripts, and manual cleanup steps.
- Never terminate an unowned or pre-existing compiler process. If another workload is
  present, treat it as a lease/resource blocker and wait or choose an isolated resource.
- Normal failure handling waits for natural compiler return. If the frozen experiment
  runner reaches its declared top-level timeout, it may terminate only the exact process
  group it created. Do not add a second parent-side watchdog that independently kills
  compiler children.
- Preserve the interrupted attempt before retry or cleanup. A later successful attempt
  is appended and does not rewrite the interrupted scene.

## Compile Concurrency

Probe and freeze an explicit bounded `NPU_STATIC_KERNEL_COMPILE_JOBS` value when the
installed runtime otherwise derives compile concurrency from a high host CPU count.
Record that value in environment/control evidence and keep it identical across compared
runs. Select the value from runtime and host evidence; do not silently change it to
manufacture a performance or reliability result.

## Profiler Schedule And Python Startup Safety

Treat a profiler schedule change as a launcher/runtime change even when it only changes
the intended number of recorded steps. `active`, `skip_first`, `warmup`, and `repeat`
must be passed through the framework's ordinary profiler configuration or construction
path after the approved CANN environment setup and at the same initialization point as
the unmodified launcher.

Do not implement a schedule override with any Python interpreter-startup hook,
including:

- `sitecustomize.py` or `usercustomize.py`;
- a `.pth` import hook or `PYTHONSTARTUP`;
- prepending a monkey-patch directory to `PYTHONPATH`;
- another startup shim that imports application modules or `torch_npu` before the
  launcher's normal multiprocessing/static-compiler initialization.

This prohibition applies even when profiling is disabled: merely loading the hook can
change import order and process state. A schedule value such as a larger `active` window
is not itself a compiler option and must not require global monkey patching. If the
installed framework exposes no supported way to express the requested schedule, retain
its default schedule and record the limitation instead of injecting a startup override.

Before launching static compilation, archive the effective profiler schedule, relevant
environment delta, and ordered `PYTHONPATH`. Block the run when an unapproved startup
hook is present. After any intentional change to profiler construction or import order,
run one isolated fresh-root compile-health regression with the same source, workload,
compile concurrency, and runtime setup. Accept later profiling only after the compiler
returns naturally, every aggregate compile stage reports all inputs successful, the
top-level exit status is zero, and inference reaches its normal completion point.

When diagnosing a regression, vary startup override, framework profiler enablement,
optional SK trace, cache seed, and compile concurrency independently. A result that
fails only with a startup hook identifies that launch path as the trigger; it does not
prove the schedule value, profiler feature, SK trace, or named operators are defective.

## Required Report Language

Reports must distinguish:

- provisional error-named files observed while compilation was live;
- naturally returned final compiler failures;
- top-level timeout or external interruption;
- valid clean-retry result.

Never write "compile failed" solely because an error-named file existed before the
compiler's natural terminal state.
