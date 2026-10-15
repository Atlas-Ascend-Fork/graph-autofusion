# Adaptation Intake Checklist

Use this checklist before reading implementation details, editing source, or launching
the model. Ask only for fields the user has not already supplied. For a
machine-detectable value the user may write `unknown; detect it`; a blank value is
still missing.

## User Response Template

```text
Target:
- repository and working directory:
- model source directory:
- model/checkpoint path:
- configuration/YAML and overrides:
- exact inference command:
- environment activation and CANN setup command:
- allowed source/config changes:

Claimed runtime:
- Python version:
- CANN version and installation path:
- PyTorch version:
- torch_npu version:
- NPU model and device count:
- visible-device variables:
- backend and static-kernel setting:
- precision and parallel topology:

Experiment:
- batch size and input/output lengths:
- prompt or dataset:
- baseline command, logs, and metric:
- correctness method:
- runtime/device/profiler limits:
- artifact location:
- required success threshold:
```

## Confirmation

Inspect the approved setup command for side effects, then run the probe in an isolated
subprocess after loading the same CANN runtime environment used by inference. Prefer
the active CANN `bin/setenv.bash` when the repository launcher mixes environment setup
with cleanup or other mutations:

```bash
source /path/to/active/cann/bin/setenv.bash
python3 <skill-dir>/scripts/check_environment.py --json
```

Before accepting the result, require:

- `runtime_environment.loaded=true` and CANN/HCCL paths in `LD_LIBRARY_PATH`;
- `ready=true`, required APIs present, and the expected device topology;
- the tested option value appears unchanged in `accepted_values`;
- the JSON output is archived as experiment environment evidence.
- the Stage-A model glue does not hard-code or inject a non-empty SuperKernel optimize
  or debug option when the candidate config supplies explicit empty maps; freeze the
  glue revision/hash as control evidence.

`cann_environment_not_loaded`, `torch_npu_runtime_load_failed`, or
`option_probe_status=not_run` means wrapper validation never ran. Record option
acceptance as unknown and rerun after fixing the environment; never translate these
states into rejected option values.

Present machine-verifiable fields as:

| Field | User claim | Detected | Status |
|---|---|---|---|
| CANN | 9.x at path | probe result | match/mismatch/unknown |
| PyTorch / torch_npu | versions | probe result | match/mismatch |
| NPU topology | model and count | probe result | match/mismatch |
| Visible devices | variables | probe result | match/mismatch |

Stop before adaptation when a critical field is missing, environment setup fails,
required APIs or A2/A3 devices are unavailable, or a material mismatch is unresolved.
An undetectable optional version-file path is non-blocking only when the active runtime
and user-approved command are otherwise confirmed.

## External Requirements And Boundaries

The skill folder is self-contained. The external run requires a target repository,
model artifacts, compatible CANN/PyTorch/torch_npu installation, Ascend hardware, and
generated logs/profiler/`sk_meta` evidence. Bundled references and scripts are aids,
not runtime dependencies. Online documentation is optional context; the installed
wrapper controls accepted options.

This skill does not install the stack or allocate hardware. Its optimization objective
is to find a repeatable SuperKernel performance improvement within the approved search
budget. A fusion record is not a performance result, and a candidate that fails the
promotion gate must not be recommended as the default. Execution, correctness, fusion,
and repeatable performance gates all remain mandatory.
