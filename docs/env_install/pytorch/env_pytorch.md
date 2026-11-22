# PyTorch torch_npu Daily 环境部署

## 前置准备

`torch_npu Daily` 的安装主要依赖 Python 环境，运行 NPU 用例时依赖已经安装好的 CANN Toolkit、OPS 包和 NPU 驱动。

请根据以下步骤完成前置准备：

1. CANN 包版本要求为 `9.0.0` 及以上，通过 [CANN 快速安装](https://www.hiascend.com/cann/download?versionId=745&ids=d802%2Ch0501%2Ch0602%2Ch0701) 正确安装 toolkit 和 ops 包，可以参考[ 安装指导 ](../../../docs/zh/quick_install.md)。

2. 通过下方步骤搭建 PyTorch 环境。

> **注意**：下文中的 `/mnt/workspace` 为华为云开发环境挂载目录，可根据实际环境替换。

---

## 一、创建虚拟环境


### 1. 安装 Python 编译依赖

`pyenv install` 会在当前机器上编译 Python，因此需要提前安装 Python 编译所需的系统依赖。如果机器已经具备完整的 Python 编译环境，可以跳过此步骤。

先查看操作系统：

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

只需要执行与当前操作系统对应的一组命令。

### 2. 安装 pyenv

```bash
mkdir -p /mnt/workspace/env

cd /mnt/workspace/env

git clone https://github.com/pyenv/pyenv.git
```

配置 pyenv 环境变量：

```bash
export PYENV_ROOT="/mnt/workspace/env/pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
```

### 3. 安装指定 Python 版本

`torch_npu Daily 2.10.0` 需要 Python 3.11，本文使用 Python 3.11.4，可自行更改。

安装 Python 3.11.4：

```bash
PYTHON_BUILD_MIRROR_URL="https://mirrors.huaweicloud.com/python" \
PYTHON_BUILD_MIRROR_URL_SKIP_CHECKSUM=1 \
pyenv install 3.11.4
```

### 4. 创建并激活虚拟环境

创建虚拟环境：

```bash
mkdir -p /mnt/workspace/env/venv

/mnt/workspace/env/pyenv/versions/3.11.4/bin/python \
    -m venv /mnt/workspace/env/venv/torch210_daily
```

激活虚拟环境：

```bash
source /mnt/workspace/env/venv/torch210_daily/bin/activate
```

> **注意**：后续所有命令默认都在已激活的虚拟环境中执行，请勿退出该环境。

---

## 二、安装 PyTorch 环境依赖

### 1. 安装 NumPy

```bash
python -m pip install numpy
```

---

### 2. 安装 torch_npu Daily

设置安装参数：

```bash
cd /mnt/workspace/env

# 版本要求为 2.9.0 及以上，这里使用 2.10.0，可自行更改。
TORCH_NPU_VERSION="2.10.0"

DATE="$(date +%Y%m%d)"

PY_TAG="$(
    python -c 'import sys; print(f"py{sys.version_info.major}{sys.version_info.minor}")'
)"

ARCH="$(uname -m)"

PKG_DIR="/mnt/workspace/env/torch_npu_pkg"
```

根据版本、日期和 Python 版本生成下载地址：

```bash
URL="https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/pta/Daily/v${TORCH_NPU_VERSION}/${DATE}.2/pytorch_v${TORCH_NPU_VERSION}_${PY_TAG}.tar.gz"
```

检查当天 Daily 包是否存在：

```bash
wget --spider "$URL"
```

创建下载目录：

```bash
rm -rf "$PKG_DIR"
mkdir -p "$PKG_DIR"
```

下载并解压：

```bash
wget -O "$PKG_DIR/pytorch.tar.gz" "$URL"

tar -xzf "$PKG_DIR/pytorch.tar.gz" \
    -C "$PKG_DIR"
```

查找 `torch_npu` wheel：

```bash
WHEEL="$(
    find "$PKG_DIR" \
        -type f \
        -name "torch_npu*${ARCH}.whl" |
        head -n 1
)"
```

安装：

```bash
python -m pip install --force-reinstall "$WHEEL"
```

安装完成后，Python 包位于：

```text
/mnt/workspace/env/venv/torch210_daily/lib/python3.11/site-packages
```

### 3. 安装样例运行依赖

使用本环境运行 [Autofuse PyTorch 用例](../../../autofuse/examples/pytorch/README.md)时，融合算子代码生成阶段会调用 `asc_tilling.py` 等生成脚本，这些脚本依赖 `decorator`、`scipy`、`attr`（`attrs`）等 Python 包，未包含在上述 `NumPy`、`torch_npu` 安装范围内。

若运行时出现如下报错：

```text
RuntimeError: Failed to build ascend kernel. trigger bug by following command: '.../asc_tilling.py'
```

该报错会把真实失败原因包装为“触发 bug 的命令”，并不直接暴露缺失的依赖。此时请手动执行报错中给出的 `.py` 文件（例如 `python3 .../asc_tilling.py`），即可看到真实的报错（如 `ModuleNotFoundError: No module named 'decorator'`），再按提示安装缺失的依赖包。也可以切换到仓库根目录，一次性安装全部依赖：

```bash
python -m pip install -r super_kernel/requirements-dev.txt
```

> **注意**：`requirements-dev.txt` 位于仓库根目录下的 `super_kernel/` 目录，请在仓库根目录执行上述命令，或将路径替换为绝对路径。

---

## 三、加载 CANN 环境

运行 NPU 用例前，需要加载已经安装好的 CANN Toolkit 和 OPS：

```bash
# CANN 包安装路径根据实际安装位置确定。
source /home/developer/Ascend/cann/set_env.sh
```

设置运行设备：

```bash
#假设跑在 device0
export ASCEND_DEVICE_ID=0
```

---

## 四、验证环境

验证 PyTorch 和 `torch_npu`：

```bash
python - <<EOF
import torch
import torch_npu

print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
EOF
```

---

## 五、一键配置脚本

也可使用[ 一键配置脚本 ](../../../scripts/env_install/pytorch/setup_torch_npu_daily.sh)自动完成虚拟环境创建和 PyTorch 环境依赖安装。
在 Graph-AutoFusion 仓库根目录执行：

```bash
bash scripts/env_install/pytorch/setup_torch_npu_daily.sh
```

脚本完成后激活环境：

```bash
source /mnt/workspace/env/venv/torch210_daily/bin/activate
```

加载 CANN：

```bash
source /home/developer/Ascend/cann/set_env.sh
```
