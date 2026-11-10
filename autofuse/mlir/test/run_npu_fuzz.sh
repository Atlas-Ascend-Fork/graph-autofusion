#!/usr/bin/env bash
# 持续按批运行，无总时限；Ctrl+C 停止，已落盘的结果保留。
# NPU 上运行；先从指定源码干净构建 AF/MLIR，再创建 overlay 并持续 fuzz。不安装 AF 包。
set -eo pipefail

# ===== 环境配置：迁移机器时优先检查；均可用同名环境变量覆盖 =====
# CANN_ENV 留空表示已在当前 shell 初始化 CANN。
CANN_ENV=${CANN_ENV:-}
PYTHON_BIN=${PYTHON_BIN:-python3}
# 每次启动拉取该分支，安装到任务私有目录，不覆盖全局 debugger。
DEBUGGER_REPO_URL=${DEBUGGER_REPO_URL:-https://gitcode.com/xingzhixiong/Autofuse-Debugger.git}
DEBUGGER_REF=${DEBUGGER_REF:-gser/fuzz-generation-error}
# 普通 Python/venv 通常留空；独立安装的 Python 可填其安装前缀。
PYTHON_HOME=${PYTHON_HOME:-}
OUT_BASE=${OUT_BASE:-/mnt/workspace/af-fuzz}
SOC_TIER=${SOC_TIER:-A3}
NPU_DEVICES=${NPU_DEVICES:-0,1}
NPU_JOBS=${NPU_JOBS:-2}                 # 随可用卡数调整
COMPILE_THREADS=${COMPILE_THREADS:-4} # 每个 worker 的 kernel 编译并发

# 开发构建：默认使用系统 LLD 14，可通过 LLD_BIN 覆盖。
AF_REPO=${AF_REPO:-/mnt/workspace/af-npu/source/}
AF_LLVM_ROOT=${AF_LLVM_ROOT:-${LLVM_BUILD_DIR:-/mnt/workspace/llvm}}
BUILD_JOBS=${BUILD_JOBS:-40}           # 按 CPU/内存容量调整
LLD_BIN=${LLD_BIN:-/usr/bin/ld.lld-14}
CANN_3RD_LIB_PATH=${CANN_3RD_LIB_PATH:-/mnt/workspace/af-npu/deps/third_party}

# ===== Fuzz 配置：N_CASES 是所有卡合计，动态批次按启动次数计算 =====
N_CASES=${N_CASES:-1000}
MAX_NODES=${MAX_NODES:-10}
DYNAMIC_BATCHES=${DYNAMIC_BATCHES:-3}
GEN_TIMEOUT_MS=${GEN_TIMEOUT_MS:-30000}
RETRY_DELAY_SECONDS=${RETRY_DELAY_SECONDS:-10}
BASE_SEED=${BASE_SEED:-2026110302}

# ===== 执行逻辑：以下通常无需随环境修改 =====
[[ -z "$CANN_ENV" ]] || source "$CANN_ENV"
if [[ -n "$PYTHON_HOME" ]]; then
    export PYTHONHOME="$PYTHON_HOME"
    export LD_LIBRARY_PATH="$PYTHON_HOME/lib:${LD_LIBRARY_PATH:-}"
fi
set -u
PYTHON_BIN=$(command -v "$PYTHON_BIN") || {
    echo "找不到 Python 解释器，请检查 PYTHON_BIN" >&2; exit 2;
}
unset AF_MLIR_PINNED_PLAN_ID AF_MLIR_ENABLE_PLAN_PINNING AF_ENABLE_MLIR_CODEGEN
# 每次启动创建新的 build，不使用外部 AF_BUILD。
AF_REPO=$(cd "$AF_REPO" && pwd)
export AF_REPO AF_LLVM_ROOT
export LLVM_BUILD_DIR="$AF_LLVM_ROOT"
mkdir -p "$OUT_BASE"
TASK_ROOT=$(mktemp -d "$OUT_BASE/af-fuzz-overlay-$(date +%Y%m%d-%H%M%S)-XXXXXXXX")
export TASK_ROOT
cp -- "${BASH_SOURCE[0]}" "$TASK_ROOT/controller.sh"
printf '任务目录: %s\n' "$TASK_ROOT"

# 先检查基础依赖和所选设备，避免完成 AF 构建后才发现运行环境不可用。
echo environment_check > "$TASK_ROOT/status.txt"
if ! "$PYTHON_BIN" - "$NPU_DEVICES" > "$TASK_ROOT/environment-check.log" 2>&1 <<'PY'
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys

errors = []
report = {'python': sys.version, 'executable': sys.executable, 'packages': {}, 'devices': []}
print(f'Python: {sys.version}\nExecutable: {sys.executable}', flush=True)
if sys.version_info < (3, 9):
    errors.append('需要 Python >= 3.9')
# 每个模块独立导入，收集全部缺失项，也保留动态库/ABI 导入失败的原始信息。
packages = [('pip', 'pip'), ('setuptools', 'setuptools'), ('wheel', 'wheel'),
            ('numpy', 'numpy'), ('torch', 'torch'), ('torch-npu', 'torch_npu'),
            ('nnsmith', 'nnsmith'), ('z3-solver', 'z3'),
            ('multipledispatch', 'multipledispatch')]
for distribution, module in packages:
    try:
        version = metadata.version(distribution)
        code = f'import {module} as m; print(getattr(m, "__file__", "built-in"))'
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=60)
        report['packages'][distribution] = {'version': version, 'import_output': result.stdout.strip()}
        if result.returncode:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip() or f'exit={result.returncode}')
        print(f'OK {distribution} {version}: {result.stdout.strip()}', flush=True)
    except Exception as exc:
        errors.append(f'{distribution} ({module}): {type(exc).__name__}: {exc}')
try:
    device_ids = [int(value.strip()) for value in sys.argv[1].split(',')]
    if not device_ids or any(device < 0 for device in device_ids):
        raise ValueError('设备编号必须是非负整数')
except ValueError as exc:
    device_ids = []
    errors.append(f'NPU_DEVICES={sys.argv[1]!r} 无效: {exc}')
if not errors:
    try:
        import torch
        import torch_npu
        if not torch.npu.is_available():
            raise RuntimeError('torch.npu.is_available() 为 False；检查 CANN 初始化、驱动及设备挂载')
        count = torch.npu.device_count()
        report['visible_npu_count'] = count
        print(f'Visible NPU count: {count}', flush=True)
        for device in device_ids:
            try:
                if device >= count:
                    raise RuntimeError(f'设备编号超出当前可见范围 0..{count - 1}')
                torch.npu.set_device(device)
                x = torch.ones(8, device=f'npu:{device}')
                actual = (x + x).cpu()
                torch.npu.synchronize()
                if not torch.equal(actual, torch.full((8,), 2.0)):
                    raise RuntimeError('NPU 小算子结果不正确')
                report['devices'].append({'index': device, 'status': 'passed'})
                del x, actual
                print(f'OK npu:{device}: allocation, add, copy-back, synchronize', flush=True)
            except Exception as exc:
                errors.append(f'npu:{device}: {type(exc).__name__}: {exc}')
    except Exception as exc:
        errors.append(f'NPU runtime: {type(exc).__name__}: {exc}')
report['errors'] = errors
Path(os.environ['TASK_ROOT'], 'environment-check.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
for error in errors:
    print(f'ERROR: {error}', file=sys.stderr, flush=True)
raise SystemExit(1 if errors else 0)
PY
then
    echo environment_error > "$TASK_ROOT/status.txt"
    cat "$TASK_ROOT/environment-check.log" >&2
    echo 'Python/NPU 环境检查失败；请针对上述 Python 路径补齐依赖或修复设备环境。未构建 AF、未启动 fuzz。' >&2
    exit 2
fi
cat "$TASK_ROOT/environment-check.log"

# 普通 wheel 安装到私有目录，复用当前 Python 的 Torch/NNSmith 等依赖。
# 不使用 editable/snapshot 导入；运行期间不再拉取或更新。
echo debugger_install > "$TASK_ROOT/status.txt"
if ! (
    git clone --depth 1 --single-branch --branch "$DEBUGGER_REF" -- "$DEBUGGER_REPO_URL" "$TASK_ROOT/debugger-source" || exit $?
    git -C "$TASK_ROOT/debugger-source" rev-parse HEAD > "$TASK_ROOT/debugger-commit.txt" || exit $?
    "$PYTHON_BIN" -m pip install --no-deps --no-build-isolation \
        --target "$TASK_ROOT/debugger-site" "$TASK_ROOT/debugger-source"
) > "$TASK_ROOT/debugger-install.log" 2>&1; then
    echo debugger_install_error > "$TASK_ROOT/status.txt"
    echo "debugger 更新失败，未构建、未启动批次: $TASK_ROOT/debugger-install.log" >&2
    exit 2
fi
export PYTHONPATH="$TASK_ROOT/debugger-site${PYTHONPATH:+:$PYTHONPATH}"
printf 'debugger commit: %s\n' "$(cat "$TASK_ROOT/debugger-commit.txt")"

# 预检和实际运行复用同一份参数，避免参数变化后只更新其中一处。
set_fuzz_args() {
    FUZZ_ARGS=(af-fuzz --out-dir "$1" --n-cases "$N_CASES" --max-nodes "$MAX_NODES"
        --seed "$2" --op-categories elewise,reduce --timeout-ms "$GEN_TIMEOUT_MS"
        --method symbolic-cinit --tensor-scale default --value-dist-mode uniform
        --target npu --soc-tier "$SOC_TIER" --jobs "$NPU_JOBS" --devices "$NPU_DEVICES"
        --compile-threads "$COMPILE_THREADS")
    if [[ "$3" == dynamic ]]; then
        FUZZ_ARGS+=(--dynamic-shapes --dynamic-variants 3)
    fi
}
# 在昂贵的构建之前仅解析参数，不调用 fuzz handler，不生成/执行用例。
echo preflight > "$TASK_ROOT/status.txt"
if [[ ! "$DYNAMIC_BATCHES" =~ ^[0-9]+$ ]]; then
    echo preflight_error > "$TASK_ROOT/status.txt"
    echo 'DYNAMIC_BATCHES 必须是非负整数' >&2
    exit 2
fi
PREFLIGHT_SHAPE=static
if (( DYNAMIC_BATCHES > 0 )); then
    PREFLIGHT_SHAPE=dynamic
fi
set_fuzz_args "$TASK_ROOT/preflight-results" "$BASE_SEED" "$PREFLIGHT_SHAPE"
if ! "$PYTHON_BIN" - "${FUZZ_ARGS[@]}" > "$TASK_ROOT/preflight.log" 2>&1 <<'PY'
import importlib.metadata
import sys
import os
from pathlib import Path
from autofusedebugger import cli
print('debugger version:', importlib.metadata.version('autofusedebugger'), flush=True)
print('debugger CLI:', cli.__file__, flush=True)
expected = Path(os.environ['TASK_ROOT']) / 'debugger-site'
if not Path(cli.__file__).resolve().is_relative_to(expected.resolve()):
    raise RuntimeError(f'debugger 未加载本任务安装包: {cli.__file__}')
cli.build_parser().parse_args(sys.argv[1:])
print('Fuzz CLI parameter check passed', flush=True)
PY
then
    echo preflight_error > "$TASK_ROOT/status.txt"
    cat "$TASK_ROOT/preflight.log" >&2
    echo '启动参数检查失败，未构建、未启动批次；动态 shape 需要支持该功能的 debugger，静态测试可设置 DYNAMIC_BATCHES=0。' >&2
    exit 2
fi

# 从空目录构建运行需要的完整 AF/MLIR 依赖图，不重编 LLVM/CANN，不编测试和 SuperKernel。
: "${ASCEND_HOME_PATH:?请先初始化 CANN 环境，或设置 CANN_ENV}"
[[ -f "$AF_LLVM_ROOT/lib/cmake/llvm/LLVMConfig.cmake" && -f "$AF_LLVM_ROOT/lib/cmake/mlir/MLIRConfig.cmake" ]] || {
    echo "LLVM/MLIR 安装不完整: $AF_LLVM_ROOT" >&2; exit 2;
}
LLD_BIN=$(command -v "$LLD_BIN")
# 保留 ld.lld 文件名：解析到通用 lld 会丢失 Unix 链接器模式。
LLD_BIN="$(cd "$(dirname "$LLD_BIN")" && pwd)/$(basename "$LLD_BIN")"
LLD_VERSION=$("$LLD_BIN" --version)
printf '链接器: %s (%s)\n' "$LLD_BIN" "$LLD_VERSION"
BUILD_ROOT="$TASK_ROOT/build"
export AF_BUILD="$BUILD_ROOT/autofuse"
mkdir "$TASK_ROOT/linker"
# 保持 ld.lld 名称，使 GCC -fuse-ld=lld 与 CMAKE_LINKER 使用同一工具。
ln -s "$LLD_BIN" "$TASK_ROOT/linker/ld.lld"
LINK_FLAGS="-B$TASK_ROOT/linker -fuse-ld=lld"
printf '%s\n' "$LLD_VERSION" > "$TASK_ROOT/toolchain.log"
command -v ccache >/dev/null || { echo '需要安装 ccache' >&2; exit 2; }
ccache --show-stats > "$TASK_ROOT/ccache-before.log"
echo configuring > "$TASK_ROOT/status.txt"
CONFIGURE=(cmake -S "$AF_REPO" -B "$BUILD_ROOT"
    -DCMAKE_BUILD_TYPE=Release -DBUILD_AUTOFUSE=ON -DENABLE_AUTOFUSE_MLIR=ON
    -DENABLE_AUTOFUSE_MLIR_TESTS=OFF -DENABLE_CPP_UTEST=OFF -DENABLE_CPP_STEST=OFF
    -DENABLE_CCACHE=ON
    "-DASCEND_INSTALL_PATH=$ASCEND_HOME_PATH" "-DCANN_3RD_LIB_PATH=$CANN_3RD_LIB_PATH"
    "-DPython3_EXECUTABLE=$PYTHON_BIN" "-DLLVM_BUILD_DIR=$AF_LLVM_ROOT"
    "-DLLVM_DIR=$AF_LLVM_ROOT/lib/cmake/llvm" "-DMLIR_DIR=$AF_LLVM_ROOT/lib/cmake/mlir"
    "-DCMAKE_LINKER=$TASK_ROOT/linker/ld.lld"
    "-DCMAKE_EXE_LINKER_FLAGS=$LINK_FLAGS" "-DCMAKE_MODULE_LINKER_FLAGS=$LINK_FLAGS"
    "-DCMAKE_SHARED_LINKER_FLAGS=$LINK_FLAGS -Wl,--gc-sections")
printf '%q ' "${CONFIGURE[@]}" > "$TASK_ROOT/configure-command.sh"
printf '\n' >> "$TASK_ROOT/configure-command.sh"
if ! "${CONFIGURE[@]}" > "$TASK_ROOT/configure.log" 2>&1; then
    echo configure_error > "$TASK_ROOT/status.txt"
    echo "配置失败: $TASK_ROOT/configure.log" >&2
    exit 2
fi
echo building > "$TASK_ROOT/status.txt"
printf '干净构建 AF/MLIR，-j %s；日志: %s/build.log\n' "$BUILD_JOBS" "$TASK_ROOT"
if ! cmake --build "$BUILD_ROOT" --target pyautofuse -j "$BUILD_JOBS" > "$TASK_ROOT/build.log" 2>&1; then
    echo build_error > "$TASK_ROOT/status.txt"
    echo "编译失败: $TASK_ROOT/build.log；未启动 fuzz。" >&2
    exit 2
fi

ccache --show-stats > "$TASK_ROOT/ccache-after.log"

# 只做一次环境组装、动态库检查和版本留档。
"$PYTHON_BIN" - <<'PY'
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

out = Path(os.environ['TASK_ROOT'])
env = dict(os.environ)
repo = Path(os.environ['AF_REPO']).resolve()
build = Path(os.environ['AF_BUILD']).resolve()
sys.path.insert(0, str(repo / 'autofuse/mlir/test/e2e/npu'))
from support import device_env
overlay = out / 'overlay'
device_env.ensure_overlay(repo, build, overlay)
env = device_env.environ_for_run(repo, build, overlay)
if 'PYTHONHOME' not in os.environ:
    env.pop('PYTHONHOME', None)

probe = r"""
import os
import autofusedebugger
os.environ['TORCHINDUCTOR_ASCENDC_DEBUG'] = ','.join(
    x for x in os.environ.get('TORCHINDUCTOR_ASCENDC_DEBUG', '').split(',')
    if x.strip() and x.strip().lstrip('+') != 'gather')
import importlib, json
names = ('torch', 'torch_npu', 'autofuse.pyautofuse', 'autofuse.ascendc_compile',
         'autofuse.compile_adapter', 'autofusedebugger')
print('AF_PROBE=' + json.dumps({n: importlib.import_module(n).__file__ for n in names}))
"""
raw = subprocess.check_output([sys.executable, '-c', probe], env=env, text=True)
origins = json.loads(next(s[len('AF_PROBE='):] for s in raw.splitlines() if s.startswith('AF_PROBE=')))
libs = subprocess.check_output(['ldd', origins['autofuse.pyautofuse']], env=env, text=True)
(out / 'libraries.log').write_text(libs)
if 'not found' in libs:
    raise RuntimeError('AF 存在未解析的动态库，见 libraries.log')
if not Path(origins['autofuse.pyautofuse']).is_relative_to(overlay):
    raise RuntimeError('pyautofuse 未加载本任务 overlay')
if 'libaihac_codegen.so => ' + str(build / 'libaihac_codegen.so') + ' ' not in libs:
    raise RuntimeError('codegen 动态库不来自指定 build')
versions = {}
for name in ('autofusedebugger', 'nnsmith', 'torch', 'torch-npu'):
    try:
        versions[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        versions[name] = 'unknown'
metadata = dict(mode='overlay', python=sys.executable, origins=origins, versions=versions)
metadata.update(af_repo=str(repo), af_build=str(build),
    af_commit=subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
    af_diff=subprocess.check_output(['git', '-C', str(repo), 'diff', 'HEAD'], text=True))
root = Path(origins['autofusedebugger']).parent
metadata['debugger_sha256'] = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in root.rglob('*') if p.suffix in ('.py', '.patch')}
(out / 'environment.json').write_text(json.dumps(metadata, indent=2))
# 只保存启动子进程需要的路径变量，不落盘整份 shell 环境。
keys = ('PYTHONPATH', 'LD_LIBRARY_PATH', 'PYTHONHOME')
(out / 'paths.json').write_text(json.dumps({k: env[k] for k in keys if k in env}, indent=2))
print(json.dumps(origins, indent=2), flush=True)
PY

# 真实文件入口供双卡 spawn 使用；兼容处理在子进程中也执行。
cat > "$TASK_ROOT/launch.py" <<'PY'
import json
import os
from pathlib import Path
os.environ.update(json.loads(Path(__file__).with_name('paths.json').read_text()))
# PYTHONPATH 对当前进程还需同步到 sys.path；子进程会从环境继承。
import sys
sys.path[:0] = [p for p in os.environ.get('PYTHONPATH', '').split(os.pathsep) if p]
import autofusedebugger
os.environ['TORCHINDUCTOR_ASCENDC_DEBUG'] = ','.join(
    x for x in os.environ.get('TORCHINDUCTOR_ASCENDC_DEBUG', '').split(',')
    if x.strip() and x.strip().lstrip('+') != 'gather')
if __name__ == '__main__':
    from autofusedebugger.cli import main
    raise SystemExit(main())
PY
# 动态库路径必须在启动 Python 之前生效，不能只在 launch.py 中修改。
eval "$("$PYTHON_BIN" - "$TASK_ROOT/paths.json" <<'PY'
import json, shlex, sys
for k, v in json.load(open(sys.argv[1])).items():
    print('export ' + k + '=' + shlex.quote(v))
PY
)"

# 为每批创建独立进程组；异常退出后清理残留 worker，避免下一批抢占同一张卡。
cat > "$TASK_ROOT/supervise.py" <<'PY'
import os
import signal
import subprocess
import sys
import time

child = None
stopped = 0

def stop(signum, frame):
    global stopped
    stopped = signum

signal.signal(signal.SIGINT, stop)
signal.signal(signal.SIGTERM, stop)
try:
    child = subprocess.Popen(sys.argv[1:], start_new_session=True)
    while child.poll() is None and not stopped:
        time.sleep(0.2)
    code = child.returncode if child.returncode is not None else 0
finally:
    if child is not None:
        # 主进程退出也可能留下 worker；进程组不复用时才可安全重启。
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            child.poll()
            try:
                os.killpg(child.pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.1)
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait()
if stopped:
    raise SystemExit(128 + stopped)
raise SystemExit(code if code >= 0 else 128 - code)
PY

START=$(date +%s)
printf 'n_cases=%s\nmax_nodes=%s\ndynamic_batches=%s\ngen_timeout_ms=%s\nbase_seed=%s\nstarted_epoch=%s\n' \
    "$N_CASES" "$MAX_NODES" "$DYNAMIC_BATCHES" "$GEN_TIMEOUT_MS" "$BASE_SEED" "$START" > "$TASK_ROOT/task.conf"
printf 'retry_delay_seconds=%s\n' "$RETRY_DELAY_SECONDS" >> "$TASK_ROOT/task.conf"
printf 'batch\tseed\tshape\tstarted_epoch\tfinished_epoch\texit_code\n' > "$TASK_ROOT/batches.tsv"
printf 'running\n' > "$TASK_ROOT/status.txt"
BATCH_PID=''
stop_task() {
    trap '' INT TERM
    if [[ -n "$BATCH_PID" ]]; then
        kill -TERM "$BATCH_PID" 2>/dev/null || true
        wait "$BATCH_PID" 2>/dev/null || true
    fi
    echo stopped_by_user > "$TASK_ROOT/status.txt"
    exit "$1"
}
trap 'stop_task 130' INT
trap 'stop_task 143' TERM
BATCH=0
while true; do
    BATCH=$((BATCH + 1))
    RUN_DIR=$(printf '%s/batch-%04d' "$TASK_ROOT" "$BATCH")
    mkdir "$RUN_DIR"
    SEED=$((BASE_SEED + BATCH))
    SHAPE=static
    if (( BATCH <= DYNAMIC_BATCHES )); then
        SHAPE=dynamic
    fi
    export AUTOFUSE_DFX_FLAGS="--mlir_codegen_config=mlir;--codegen_compile_debug=true;--debug_dir=$RUN_DIR/dump;"
    export TORCHINDUCTOR_COMPILE_THREADS="$COMPILE_THREADS"
    export TORCHINDUCTOR_CACHE_DIR="$RUN_DIR/cache"
    export INDUCTOR_CACHE_DIR="$RUN_DIR/inductor_cache"
    export TORCH_COMPILE_DEBUG_DIR="$RUN_DIR/torch_compile_debug"
    export TORCHINDUCTOR_ASCENDC_DEBUG=''
    set_fuzz_args "$RUN_DIR/results" "$SEED" "$SHAPE"
    CMD=("$PYTHON_BIN" "$TASK_ROOT/launch.py" "${FUZZ_ARGS[@]}")
    printf '%q ' "${CMD[@]}" > "$RUN_DIR/command.sh"
    printf '\n' >> "$RUN_DIR/command.sh"
    BEFORE=$(date +%s)
    printf 'batch-%04d seed=%s shape=%s\n' "$BATCH" "$SEED" "$SHAPE"
    set +e
    (cd "$RUN_DIR" && exec "$PYTHON_BIN" "$TASK_ROOT/supervise.py" "${CMD[@]}") > "$RUN_DIR/run.log" 2>&1 &
    BATCH_PID=$!
    wait "$BATCH_PID"
    RC=$?
    BATCH_PID=''
    set -e
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$BATCH" "$SEED" "$SHAPE" "$BEFORE" "$(date +%s)" "$RC" >> "$TASK_ROOT/batches.tsv"
    if [[ "$RC" == 130 || "$RC" == 143 ]]; then
        echo stopped_by_user > "$TASK_ROOT/status.txt"
        exit "$RC"
    fi
    if "$PYTHON_BIN" - "$RUN_DIR" "$N_CASES" "$RC" <<'PY'
import json, sys
from pathlib import Path
out, expected, rc = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
record = {'exit_code': rc, 'expected_cases': expected, 'completed_cases': None, 'status': 'batch_error'}
try:
    summary = json.loads((out / 'results/summary.json').read_text())
    record['completed_cases'] = summary['total']
    if rc not in (0, 1):
        record['reason'] = '异常退出码'
    elif summary['total'] != expected:
        record['reason'] = '批次未完整结束'
    else:
        record['status'] = 'complete'
except (OSError, ValueError, KeyError, TypeError) as exc:
    record['reason'] = str(exc)
(out / 'batch_status.json').write_text(json.dumps(record, ensure_ascii=False, indent=2))
print(json.dumps(record, ensure_ascii=False), flush=True)
raise SystemExit(0 if record['status'] == 'complete' else 1)
PY
    then
        echo running > "$TASK_ROOT/status.txt"
    else
        echo retry_wait > "$TASK_ROOT/status.txt"
        echo "批次异常；保留 ${RUN_DIR}，${RETRY_DELAY_SECONDS}s 后换 seed 启动下一批。" >&2
        sleep "$RETRY_DELAY_SECONDS" &
        BATCH_PID=$!
        wait "$BATCH_PID" || true
        BATCH_PID=''
        echo running > "$TASK_ROOT/status.txt"
    fi
done
