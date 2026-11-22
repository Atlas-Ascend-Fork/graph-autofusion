# PyTorch torch_npu Daily Environment Setup

## Prerequisites

Installing `torch_npu Daily` mainly depends on the Python environment. Running NPU cases also requires the already-installed CANN Toolkit, OPS package, and NPU driver.

Complete the following prerequisites:

1. The CANN package version must be `9.0.0` or later. Install the toolkit and ops packages correctly through [CANN Quick Installation](https://www.hiascend.com/cann/download?versionId=745&ids=d802%2Ch0501%2Ch0602%2Ch0701). For details, see the [Installation Guide](../../../docs/en/quick_install.md).

2. Set up the PyTorch environment by following the steps below.

> **Note**: `/mnt/workspace` in the following sections is the mount directory of the Huawei Cloud development environment. Replace it with the actual path in your environment.

---

## 1. Creating a Virtual Environment

### 1. Installing Python Build Dependencies

`pyenv install` compiles Python on the current machine, so you must install the system dependencies required for compiling Python in advance. If the machine already has a complete Python build environment, you can skip this step.

First, check the operating system:

```bash
cat /etc/os-release
```

#### Ubuntu / Debian

```bash
sudo apt-get update

sudo apt-get install -y \
    build-essential \
    git \
    wget \
    curl \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncurses-dev \
    xz-utils \
    tk-dev \
    libffi-dev \
    liblzma-dev
```

#### openEuler / CentOS / RHEL

```bash
sudo yum install -y \
    gcc \
    gcc-c++ \
    make \
    git \
    wget \
    curl \
    openssl-devel \
    zlib-devel \
    bzip2-devel \
    readline-devel \
    sqlite-devel \
    ncurses-devel \
    xz-devel \
    tk-devel \
    libffi-devel
```

Run only the set of commands that matches your operating system.

### 2. Installing pyenv

```bash
mkdir -p /mnt/workspace/env

cd /mnt/workspace/env

git clone https://github.com/pyenv/pyenv.git
```

Configure the pyenv environment variables:

```bash
export PYENV_ROOT="/mnt/workspace/env/pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
```

### 3. Installing the Specified Python Version

`torch_npu Daily 2.10.0` requires Python 3.11. This document uses Python 3.11.4, which you can change as needed.

Install Python 3.11.4:

```bash
PYTHON_BUILD_MIRROR_URL="https://mirrors.huaweicloud.com/python" \
PYTHON_BUILD_MIRROR_URL_SKIP_CHECKSUM=1 \
pyenv install 3.11.4
```

### 4. Creating and Activating the Virtual Environment

Create the virtual environment:

```bash
mkdir -p /mnt/workspace/env/venv

/mnt/workspace/env/pyenv/versions/3.11.4/bin/python \
    -m venv /mnt/workspace/env/venv/torch210_daily
```

Activate the virtual environment:

```bash
source /mnt/workspace/env/venv/torch210_daily/bin/activate
```

> **Note**: All subsequent commands are executed in the activated virtual environment by default. Do not exit this environment.

---

## 2. Installing PyTorch Environment Dependencies

### 1. Installing NumPy

```bash
python -m pip install numpy
```

---

### 2. Installing torch_npu Daily

Set the installation parameters:

```bash
cd /mnt/workspace/env

# Version 2.9.0 or later is required; 2.10.0 is used here and can be changed as needed.
TORCH_NPU_VERSION="2.10.0"

DATE="$(date +%Y%m%d)"

PY_TAG="$(
    python -c 'import sys; print(f"py{sys.version_info.major}{sys.version_info.minor}")'
)"

ARCH="$(uname -m)"

PKG_DIR="/mnt/workspace/env/torch_npu_pkg"
```

Generate the download URL based on the version, date, and Python version:

```bash
URL="https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/pta/Daily/v${TORCH_NPU_VERSION}/${DATE}.1/pytorch_v${TORCH_NPU_VERSION}_${PY_TAG}.tar.gz"
```

Check whether the daily package for the current date exists:

```bash
wget --spider "$URL"
```

Create the download directory:

```bash
rm -rf "$PKG_DIR"
mkdir -p "$PKG_DIR"
```

Download and decompress:

```bash
wget -O "$PKG_DIR/pytorch.tar.gz" "$URL"

tar -xzf "$PKG_DIR/pytorch.tar.gz" \
    -C "$PKG_DIR"
```

Find the `torch_npu` wheel:

```bash
WHEEL="$(
    find "$PKG_DIR" \
        -type f \
        -name "torch_npu*${ARCH}.whl" |
        head -n 1
)"
```

Install:

```bash
python -m pip install --force-reinstall "$WHEEL"
```

After installation, the Python packages are located at:

```text
/mnt/workspace/env/venv/torch210_daily/lib/python3.11/site-packages
```

### 3. Installing Sample Runtime Dependencies

When running the [Autofuse PyTorch examples](../../../autofuse/examples/pytorch/README_en.md) with this environment, the fused operator code generation phase invokes generated scripts such as `asc_tilling.py`. These scripts depend on Python packages such as `decorator`, `scipy`, and `attr` (`attrs`), which are not included in the `NumPy` and `torch_npu` installation scope above.

If the following error occurs:

```text
RuntimeError: Failed to build ascend kernel. trigger bug by following command: '.../asc_tilling.py'
```

This error wraps the real cause as a "trigger bug" command and does not directly reveal the missing dependency. In this case, manually run the `.py` file shown in the error (for example, `python3 .../asc_tilling.py`) to see the actual error (such as `ModuleNotFoundError: No module named 'decorator'`), and then install the missing package accordingly. Alternatively, you can switch to the repository root and install all dependencies at once:

```bash
python -m pip install -r super_kernel/requirements-dev.txt
```

> **Note**: `requirements-dev.txt` is located under the `super_kernel/` directory in the repository root. Run the command above from the repository root, or replace the path with an absolute path.

---

## 3. Loading the CANN Environment

Before running NPU cases, load the installed CANN Toolkit and OPS:

```bash
# Set the CANN package installation path based on the actual installation location.
source /home/developer/Ascend/cann/set_env.sh
```

Set the device to run on:

```bash
# Assume the case runs on device 0.
export ASCEND_DEVICE_ID=0
```

---

## 4. Verifying the Environment

Verify PyTorch and `torch_npu`:

```bash
python - <<EOF
import torch
import torch_npu

print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
EOF
```

---

## 5. One-Click Configuration Script

You can also use the [One-Click Configuration Script](../../../scripts/env_install/pytorch/setup_torch_npu_daily.sh) to automatically create the virtual environment and install the PyTorch environment dependencies.
Run the following command in the root directory of the Graph-AutoFusion repository:

```bash
bash scripts/env_install/pytorch/setup_torch_npu_daily.sh
```

After the script finishes, activate the environment:

```bash
source /mnt/workspace/env/venv/torch210_daily/bin/activate
```

Load CANN:

```bash
source /home/developer/Ascend/cann/set_env.sh
```
