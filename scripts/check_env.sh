#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# ==============================================================================
# graph-autofusion 编译环境依赖检查脚本
#
# 仓库:   https://gitcode.com/cann/graph-autofusion
# 依据:   https://gitcode.com/cann/graph-autofusion/blob/master/docs/zh/build.md
#         super_kernel/requirements-dev.txt
# 用法:   bash scripts/check_env.sh
#
# 本脚本所有检查项和版本约束严格来源于 docs/zh/build.md 和
# super_kernel/requirements-dev.txt，如文档更新请同步修改。
# ==============================================================================

# ==================== 颜色定义 ====================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# ==================== 计数器 ====================
ERROR_COUNT=0
WARNING_COUNT=0
PASS_COUNT=0

# ==============================================================================
# 版本要求 (严格来源于 docs/zh/build.md)
#
# build.md 原文:
#   - Python3 >= 3.8.0
#   - CMake >= 3.16.0 （建议使用3.20.0版本）
#   - GCC >= 7.3.0
#   - CANN Toolkit (必需)
#   - CANN ops    (必需)
#
# super_kernel/requirements-dev.txt:
#   29 个 Python 包，均有版本下限，部分有上限
# ==============================================================================
REQUIRED_PYTHON_MIN="3.8.0"
REQUIRED_CMAKE_MIN="3.16.0"
REQUIRED_CMAKE_RECOMMEND="3.20.0"
REQUIRED_GCC_MIN="7.3.0"

# ==================== 工具函数 ====================
log_pass() {
    PASS_COUNT=$((PASS_COUNT + 1))
    echo -e "  ${GREEN}[PASS]${NC}    $1"
}

log_warn() {
    WARNING_COUNT=$((WARNING_COUNT + 1))
    echo -e "  ${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    ERROR_COUNT=$((ERROR_COUNT + 1))
    echo -e "  ${RED}[ERROR]${NC}   $1"
}

log_info() {
    echo -e "  ${BLUE}[INFO]${NC}    $1"
}

version_ge() {
    [ "$(printf '%s\n' "$1" "$2" | sort -V | head -n1)" = "$2" ]
}

version_le() {
    [ "$(printf '%s\n' "$1" "$2" | sort -V | head -n1)" = "$1" ]
}

check_command() {
    command -v "$1" &>/dev/null
}

# 安全提取版本号 (支持 1/2/3/4 段, 失败返回 0.0.0)
extract_version() {
    local input="$1"
    local ver

    ver=$(echo "$input" | grep -oP '\d+\.\d+(\.\d+)*' | head -n1 || true)

    if [ -z "$ver" ]; then
        ver=$(echo "$input" | grep -oP '\d+' | head -n1 || true)
        if [ -n "$ver" ]; then
            ver="${ver}.0.0"
        else
            echo "0.0.0"
            return
        fi
    fi

    if [[ "$ver" =~ ^[0-9]+\.[0-9]+$ ]]; then
        ver="${ver}.0"
    fi

    echo "$ver"
}

print_header() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

# ==============================================================================
echo ""
echo "=================================================================="
echo "  graph-autofusion 编译环境依赖检查"
echo "  仓库: https://gitcode.com/cann/graph-autofusion"
echo "  依据: docs/zh/build.md + super_kernel/requirements-dev.txt"
echo "  时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  系统: $(uname -s) $(uname -m)"
echo "=================================================================="

# ==================== 1. 操作系统 ====================
print_header "1. 操作系统 [可选]"

OS_NAME=$(uname -s)
OS_ARCH=$(uname -m)

if [ "$OS_NAME" = "Linux" ]; then
    log_info "操作系统: Linux"
else
    log_info "其他操作系统: $OS_NAME"
fi

if [ "$OS_ARCH" = "x86_64" ] || [ "$OS_ARCH" = "aarch64" ]; then
    log_info "CPU 架构: $OS_ARCH"
else
    log_info "其他CPU架构: $OS_ARCH (build.md 验证过 x86_64/aarch64)"
fi

if [ -f /etc/os-release ]; then
    DISTRO=$(grep "^ID=" /etc/os-release | cut -d'=' -f2 | tr -d '"')
    DISTRO_VER=$(grep "^VERSION_ID=" /etc/os-release | cut -d'=' -f2 | tr -d '"')
    log_info "发行版: $DISTRO $DISTRO_VER"
fi

# ==================== 2. Python3 ====================
# build.md: Python3 >= 3.8.0
print_header "2. Python3 [build.md: >= $REQUIRED_PYTHON_MIN]"

PYTHON_CMD=""
if check_command python3; then
    PYTHON_CMD="python3"
    PYTHON_RAW=$($PYTHON_CMD --version 2>&1)
    PYTHON_VER=$(extract_version "$PYTHON_RAW")
elif check_command python; then
    PYTHON_CMD="python"
    PYTHON_RAW=$($PYTHON_CMD --version 2>&1)
    PYTHON_VER=$(extract_version "$PYTHON_RAW")
fi

if [ -n "$PYTHON_CMD" ]; then
    log_info "$PYTHON_CMD --version: $PYTHON_RAW → $PYTHON_VER"

    if version_ge "$PYTHON_VER" "$REQUIRED_PYTHON_MIN"; then
        log_pass "Python $PYTHON_VER (>= $REQUIRED_PYTHON_MIN)"
    else
        log_error "Python 版本过低: $PYTHON_VER (build.md 要求 >= $REQUIRED_PYTHON_MIN)"
        log_info "请安装 Python >= $REQUIRED_PYTHON_MIN"
    fi

    # Python 开发头文件
    PYTHON_INCLUDE=$($PYTHON_CMD -c "import sysconfig; print(sysconfig.get_path('include'))" 2>/dev/null || true)
    if [ -n "$PYTHON_INCLUDE" ] && [ -f "$PYTHON_INCLUDE/Python.h" ]; then
        log_pass "Python 开发头文件: $PYTHON_INCLUDE/Python.h"
    else
        log_warn "缺少 Python.h (编译 C 扩展可能需要)"
        log_info "安装 (Ubuntu): sudo apt-get install python3-dev"
    fi

    # pip
    if $PYTHON_CMD -m pip --version &>/dev/null; then
        PIP_RAW=$($PYTHON_CMD -m pip --version)
        PIP_VER=$(extract_version "$PIP_RAW")
        log_pass "pip $PIP_VER"
    else
        log_error "pip 未安装 (安装 requirements-dev.txt 需要)"
        log_info "安装: $PYTHON_CMD -m ensurepip --upgrade"
    fi

    # venv (build.md 建议使用虚拟环境)
    if $PYTHON_CMD -c "import venv" &>/dev/null; then
        log_pass "venv 模块可用 (build.md 建议使用虚拟环境)"
    else
        log_warn "venv 模块不可用"
        log_info "安装 (Ubuntu): sudo apt-get install python3-venv"
        log_info "build.md: python3 -m venv venv && source venv/bin/activate"
    fi
else
    log_error "未安装 Python3 (build.md 要求 >= $REQUIRED_PYTHON_MIN)"
    log_info "安装 (Ubuntu): sudo apt-get install python3 python3-dev python3-pip python3-venv"
fi

# ==================== 3. CMake ====================
# build.md: CMake >= 3.16.0 (建议使用 3.20.0 版本)
print_header "3. CMake [build.md: >= $REQUIRED_CMAKE_MIN, 建议 $REQUIRED_CMAKE_RECOMMEND]"

if check_command cmake; then
    CMAKE_RAW=$(cmake --version | head -n1)
    CMAKE_VER=$(extract_version "$CMAKE_RAW")
    log_info "cmake --version: $CMAKE_RAW → $CMAKE_VER"

    if version_ge "$CMAKE_VER" "$REQUIRED_CMAKE_RECOMMEND"; then
        log_pass "CMake $CMAKE_VER (>= 建议版本 $REQUIRED_CMAKE_RECOMMEND)"
    elif version_ge "$CMAKE_VER" "$REQUIRED_CMAKE_MIN"; then
        log_pass "CMake $CMAKE_VER (>= $REQUIRED_CMAKE_MIN)"
        log_warn "build.md 建议使用 CMake $REQUIRED_CMAKE_RECOMMEND, 当前 $CMAKE_VER"
    else
        log_error "CMake 版本过低: $CMAKE_VER (build.md 要求 >= $REQUIRED_CMAKE_MIN)"
        log_info "安装 (Ubuntu): sudo apt-get install cmake"
        log_info "升级 (pip):    pip install cmake --upgrade"
    fi
else
    log_error "未安装 CMake (build.md 要求 >= $REQUIRED_CMAKE_MIN)"
    log_info "安装 (Ubuntu): sudo apt-get install cmake"
    log_info "安装 (pip):    pip install cmake"
fi

# ==================== 4. GCC/G++ ====================
# build.md: GCC >= 7.3.0
print_header "4. GCC/G++ [build.md: >= $REQUIRED_GCC_MIN]"

if check_command gcc; then
    GCC_RAW=$(gcc -dumpversion)
    GCC_VER=$(extract_version "$GCC_RAW")
    if version_ge "$GCC_VER" "$REQUIRED_GCC_MIN"; then
        log_pass "GCC $GCC_VER (>= $REQUIRED_GCC_MIN)"
    else
        log_error "GCC 版本过低: $GCC_VER (build.md 要求 >= $REQUIRED_GCC_MIN)"
        log_info "请安装 GCC >= $REQUIRED_GCC_MIN"
    fi
else
    log_error "未安装 GCC (build.md 要求 >= $REQUIRED_GCC_MIN)"
    log_info "安装 (Ubuntu): sudo apt-get install build-essential"
fi

if check_command g++; then
    GPP_RAW=$(g++ -dumpversion)
    GPP_VER=$(extract_version "$GPP_RAW")
    log_pass "G++ $GPP_VER"
else
    log_warn "未安装 G++"
    log_info "安装 (Ubuntu): sudo apt-get install g++"
fi

log_info "如需切换到 gcc15/gcc16，请在构建前显式设置 CC/CXX 或 GCC_VERSION，例如 CC=gcc-15 CXX=g++-15 bash build.sh --pkg -j 8"

# ==================== 5. CANN Toolkit ====================
# build.md: 安装社区版 cann-toolkit 包 (必需)
print_header "5. CANN Toolkit [build.md: 必需]"

ASCEND_HOME=""
if [ -n "$ASCEND_HOME_PATH" ]; then
    ASCEND_HOME="$ASCEND_HOME_PATH"
elif [ -d "/usr/local/Ascend/ascend-toolkit/latest" ]; then
    ASCEND_HOME="/usr/local/Ascend/ascend-toolkit/latest"
elif [ -d "$HOME/Ascend/ascend-toolkit/latest" ]; then
    ASCEND_HOME="$HOME/Ascend/ascend-toolkit/latest"
elif [ -d "/usr/local/Ascend/latest" ]; then
    ASCEND_HOME="/usr/local/Ascend/latest"
fi

if [ -n "$ASCEND_HOME" ] && [ -d "$ASCEND_HOME" ]; then
    log_pass "CANN Toolkit 路径: $ASCEND_HOME"

    if [ -f "$ASCEND_HOME/version.cfg" ]; then
        CANN_VER=$(head -n1 "$ASCEND_HOME/version.cfg")
        log_info "CANN 版本: $CANN_VER"
    fi

    # 环境变量 (build.md: source ${ascend_install_path}/cann/bin/setenv.bash)
    SET_ENV_FOUND=false
    for env_script in \
        "/usr/local/Ascend/cann/bin/setenv.bash" \
        "$HOME/Ascend/cann/bin/setenv.bash" \
        "/usr/local/Ascend/cann/set_env.sh" \
        "$HOME/Ascend/cann/set_env.sh" \
        "$ASCEND_HOME/bin/setenv.bash"; do
        if [ -f "$env_script" ]; then
            SET_ENV_FOUND=true
            log_pass "setenv 脚本: $env_script"
            break
        fi
    done

    if ! $SET_ENV_FOUND; then
        log_warn "未找到 setenv.bash / set_env.sh"
        log_info "build.md: source \${ascend_install_path}/cann/bin/setenv.bash"
    fi

    if [ -n "$ASCEND_TOOLKIT_HOME" ]; then
        log_pass "ASCEND_TOOLKIT_HOME: $ASCEND_TOOLKIT_HOME"
    else
        log_warn "ASCEND_TOOLKIT_HOME 未设置 (请先 source setenv.bash)"
    fi

    if echo "$LD_LIBRARY_PATH" | grep -q "Ascend" 2>/dev/null; then
        log_pass "LD_LIBRARY_PATH 包含 Ascend"
    else
        log_warn "LD_LIBRARY_PATH 未包含 Ascend (请先 source setenv.bash)"
    fi
else
    log_error "未检测到 CANN Toolkit"
    log_info "build.md: 安装社区版 cann-toolkit 包"
    log_info "下载: https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/master/"
    log_info "安装: ./Ascend-cann-toolkit_xxx.run --full --force"
    log_info "环境: source \${ascend_install_path}/cann/bin/setenv.bash"
fi

# ==================== 6. CANN ops ====================
# build.md: 安装社区版 CANN ops 包 (必需)
print_header "6. CANN ops [build.md: 必需]"

OPP_PATH=""
if [ -n "$ASCEND_OPP_PATH" ]; then
    OPP_PATH="$ASCEND_OPP_PATH"
elif [ -n "$ASCEND_HOME" ]; then
    for opp_dir in "$ASCEND_HOME/opp" "$ASCEND_HOME/../opp" \
                   "/usr/local/Ascend/opp" "$HOME/Ascend/opp"; do
        if [ -d "$opp_dir" ]; then
            OPP_PATH="$opp_dir"
            break
        fi
    done
fi

if [ -n "$OPP_PATH" ] && [ -d "$OPP_PATH" ]; then
    log_pass "CANN ops (OPP) 路径: $OPP_PATH"
else
    log_error "未检测到 CANN ops 包"
    log_info "build.md: 安装社区版 CANN ops 包"
    log_info "下载: https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/master/"
    log_info "安装: ./Ascend-cann-xxx-ops_xxx.run --install"
fi

# ==================== 7. git ====================
print_header "7. git [build.md: 源码下载需要]"

if check_command git; then
    GIT_RAW=$(git --version)
    GIT_VER=$(extract_version "$GIT_RAW")
    log_pass "git $GIT_VER"
else
    log_error "未安装 git (build.md 源码下载需要 git clone)"
    log_info "安装 (Ubuntu): sudo apt-get install git"
fi

# make
if check_command make; then
    MAKE_RAW=$(make --version | head -n1)
    MAKE_VER=$(extract_version "$MAKE_RAW")
    log_pass "make $MAKE_VER"
else
    log_warn "未安装 make (编译可能需要)"
    log_info "安装 (Ubuntu): sudo apt-get install make"
fi

# ==================== 8. Python 包 (requirements-dev.txt) ====================
# super_kernel/requirements-dev.txt 中列出的所有依赖
print_header "8. Python 包 [super_kernel/requirements-dev.txt]"

if [ -n "$PYTHON_CMD" ]; then

    # 检查带版本下限的包
    check_py_pkg_min() {
        local pkg_name=$1
        local import_name=$2
        local min_ver=$3
        local max_ver=$4  # 可选

        if $PYTHON_CMD -c "import $import_name" &>/dev/null; then
            local ver
            ver=$($PYTHON_CMD -c "
import $import_name
v = getattr($import_name, '__version__', None)
if v is None:
    try:
        import importlib.metadata
        v = importlib.metadata.version('$pkg_name')
    except:
        v = 'unknown'
print(v)
" 2>/dev/null || echo "unknown")

            if [ "$ver" = "unknown" ]; then
                log_pass "$pkg_name (已安装, 版本未知)"
                return
            fi

            local parsed_ver
            parsed_ver=$(extract_version "$ver")
            local ok=true

            if [ -n "$min_ver" ] && ! version_ge "$parsed_ver" "$min_ver"; then
                ok=false
            fi

            if [ -n "$max_ver" ] && ! version_le "$parsed_ver" "$max_ver"; then
                ok=false
            fi

            if $ok; then
                log_pass "$pkg_name $ver"
            else
                local range_str=">=$min_ver"
                [ -n "$max_ver" ] && range_str="$range_str,<$max_ver"
                log_error "$pkg_name $ver 不满足版本要求 ($range_str)"
                log_info "安装: pip install '$pkg_name$range_str'"
            fi
        else
            local range_str=">=$min_ver"
            [ -n "$max_ver" ] && range_str="$range_str,<$max_ver"
            log_error "缺少: $pkg_name ($range_str)"
            log_info "安装: pip install '$pkg_name$range_str'"
        fi
    }

    echo ""
    log_info "检查 super_kernel/requirements-dev.txt 中的 29 个依赖..."
    echo ""

    # ===== 按 requirements-dev.txt 顺序逐项检查 =====

    # absl-py>=2.1.0
    check_py_pkg_min "absl-py" "absl" "2.1.0" ""

    # attrs>=24.2.0
    check_py_pkg_min "attrs" "attr" "24.2.0" ""

    # build>=1.1.1
    check_py_pkg_min "build" "build" "1.1.1" ""

    # cloudpickle>=2.2.1
    check_py_pkg_min "cloudpickle" "cloudpickle" "2.2.1" ""

    # coverage[toml]>=7.2.7
    check_py_pkg_min "coverage" "coverage" "7.2.7" ""

    # decorator>=5.1.1
    check_py_pkg_min "decorator" "decorator" "5.1.1" ""

    # exceptiongroup>=1.3.0
    check_py_pkg_min "exceptiongroup" "exceptiongroup" "1.3.0" ""

    # importlib-metadata>=6.7.0
    check_py_pkg_min "importlib-metadata" "importlib_metadata" "6.7.0" ""

    # iniconfig>=2.0.0
    check_py_pkg_min "iniconfig" "iniconfig" "2.0.0" ""

    # jinja2>=3.1.6
    check_py_pkg_min "jinja2" "jinja2" "3.1.6" ""

    # markupsafe>=2.1.5
    check_py_pkg_min "markupsafe" "markupsafe" "2.1.5" ""

    # ml-dtypes>=0.2.0
    check_py_pkg_min "ml-dtypes" "ml_dtypes" "0.2.0" ""

    # mpmath>=1.3.0
    check_py_pkg_min "mpmath" "mpmath" "1.3.0" ""

    # numpy>=1.21.6
    check_py_pkg_min "numpy" "numpy" "1.21.6" ""

    # packaging>=24.0
    check_py_pkg_min "packaging" "packaging" "24.0.0" ""

    # pluggy>=1.2.0
    check_py_pkg_min "pluggy" "pluggy" "1.2.0" ""

    # psutil>=7.1.1
    check_py_pkg_min "psutil" "psutil" "7.1.1" ""

    # pyproject-hooks>=1.2.0
    check_py_pkg_min "pyproject-hooks" "pyproject_hooks" "1.2.0" ""

    # pytest>=7.4.4
    check_py_pkg_min "pytest" "pytest" "7.4.4" ""

    # pytest-cov>=4.1.0
    check_py_pkg_min "pytest-cov" "pytest_cov" "4.1.0" ""

    # pytest-timeout>=2.4.0
    check_py_pkg_min "pytest-timeout" "pytest_timeout" "2.4.0" ""

    # pytest-xdist>=3.0.0
    check_py_pkg_min "pytest-xdist" "xdist" "3.0.0" ""

    # scipy>=1.7.3
    check_py_pkg_min "scipy" "scipy" "1.7.3" ""

    # six>=1.17.0
    check_py_pkg_min "six" "six" "1.17.0" ""

    # sympy>=1.10.1
    check_py_pkg_min "sympy" "sympy" "1.10.1" ""

    # tomli>=2.0.1
    check_py_pkg_min "tomli" "tomli" "2.0.1" ""

    # tornado>=6.2
    check_py_pkg_min "tornado" "tornado" "6.2.0" ""

    # typing-extensions>=4.7.1
    check_py_pkg_min "typing-extensions" "typing_extensions" "4.7.1" ""

    # wheel>=0.42.0
    check_py_pkg_min "wheel" "wheel" "0.42.0" ""

    # zipp>=3.15.0
    check_py_pkg_min "zipp" "zipp" "3.15.0" ""

    # setuptools>=68.0.0,<80.0.0
    check_py_pkg_min "setuptools" "setuptools" "68.0.0" "80.0.0"

    echo ""
    log_info "一键安装: pip3 install -r super_kernel/requirements-dev.txt"

else
    log_error "Python 未安装, 无法检查 requirements-dev.txt 依赖"
fi

# ==================== 9. 系统资源 ====================
print_header "9. 系统资源 [可选]"

# 磁盘
DISK_AVAIL=$(df -BG . 2>/dev/null | tail -1 | awk '{print $4}' | tr -d 'G' || echo "0")
if [ "$DISK_AVAIL" -gt 20 ] 2>/dev/null; then
    log_info "可用磁盘: ${DISK_AVAIL}GB"
elif [ "$DISK_AVAIL" -gt 10 ] 2>/dev/null; then
    log_info "磁盘偏少: ${DISK_AVAIL}GB (建议 >20GB)"
else
    log_info "磁盘不足: ${DISK_AVAIL}GB"
fi

# 内存
if [ -f /proc/meminfo ]; then
    MEM_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
    MEM_GB=$((MEM_KB / 1024 / 1024))
    if [ "$MEM_GB" -ge 8 ]; then
        log_info "内存: ${MEM_GB}GB"
    elif [ "$MEM_GB" -ge 4 ]; then
        log_info "内存偏少: ${MEM_GB}GB (推荐 >=8GB)"
    else
        log_info "内存不足: ${MEM_GB}GB"
    fi
fi

# CPU
CPU_CORES=$(nproc 2>/dev/null || echo "unknown")
log_info "CPU 核心数: $CPU_CORES"

# ==============================================================================
#                            汇总报告
# ==============================================================================
echo ""
echo "=================================================================="
echo "  graph-autofusion 环境检查完成"
echo "  依据: docs/zh/build.md + super_kernel/requirements-dev.txt"
echo "=================================================================="
echo ""

[ $ERROR_COUNT -gt 0 ]   && echo -e "  ${RED}✗ ERRORS:   $ERROR_COUNT${NC}  (必须修复才能编译)"
[ $WARNING_COUNT -gt 0 ] && echo -e "  ${YELLOW}⚠ WARNINGS: $WARNING_COUNT${NC}  (建议修复)"
echo -e "  ${GREEN}✓ PASSED:   $PASS_COUNT${NC}"
echo ""

if [ $ERROR_COUNT -gt 0 ]; then
    echo -e "  ${RED}结论: 环境不满足 graph-autofusion 编译要求，请修复上述 ERROR 项。${NC}"
    echo ""
    echo "  系统依赖 (Ubuntu):"
    echo "    sudo apt-get update && sudo apt-get install -y \\"
    echo "      cmake git make python3 python3-dev python3-pip python3-venv"
    echo ""
    echo "  Python 依赖:"
    echo "    python3 -m venv venv && source venv/bin/activate"
    echo "    pip3 install -r super_kernel/requirements-dev.txt"
    echo ""
    echo "  CANN Toolkit + ops:"
    echo "    下载: https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/master/"
    echo "    安装: ./Ascend-cann-toolkit_xxx.run --full --force"
    echo "           ./Ascend-cann-xxx-ops_xxx.run --install"
    echo "    环境: source \${ascend_install_path}/cann/bin/setenv.bash"
    echo ""
    echo "  详细说明: docs/zh/build.md"
    echo ""
    exit 1
elif [ $WARNING_COUNT -gt 0 ]; then
    echo -e "  ${YELLOW}结论: 环境基本可用，建议关注上述 WARNING 项。${NC}"
    exit 0
else
    echo -e "  ${GREEN}结论: 环境检查全部通过！可以编译 graph-autofusion。${NC}"
    echo -e "  ${GREEN}编译命令: bash build.sh --pkg${NC}"
    exit 0
fi
