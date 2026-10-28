# Build from Source

## 1. Environment Preparation

Complete the basic environment setup before compiling from source. See [Quick Install](quick_install.md) for detailed instructions.

## 2. Environment Verification

Verify the environment after installing the CANN packages.

```bash
# Check CANN Toolkit version (default installation path). <arch> represents CPU architecture (aarch64 or x86_64). For WebIDE scenarios, replace /usr/local with /home/developer.
cat /usr/local/Ascend/cann/<arch>-linux/ascend_toolkit_install.info
# Check CANN ops version (default installation path). <opsname> represents the ops subpackage name to query. For WebIDE scenarios, replace /usr/local with /home/developer.
cat /usr/local/Ascend/cann/<arch>-linux/ascend_ops_install.info
```

## 3. Environment Variable Configuration

Select the appropriate command based on your scenario:

  ```bash
  # Default installation path (root user example). For non-root users, replace /usr/local with ${HOME}.
  source /usr/local/Ascend/cann/set_env.sh
  # Custom installation path
  source ${install_path}/cann/set_env.sh
  ```

## 4. Source Code Compilation

### 4.1 Download Source Code

If your compilation environment can access the network, open-source third-party software downloads automatically during compilation. Use the following command to download the source code:

```bash
git clone https://gitcode.com/cann/graph-autofusion.git
 ```

If your compilation environment cannot access the network, download the source code and open-source software packages in a networked environment. Upload them manually to your compilation environment:

1. In a networked environment, visit the [project homepage](https://gitcode.com/cann/graph-autofusion). Use the `Download zip` or `Clone` button to download the source code.

2. Download the `makeself` and `cann-cmake` third-party open-source software. Extract the source code on the networked machine first, then read the version required by that source checkout:

   | Open-Source Software | Version | Download Link |
   |---|---|---|
   | makeself | 2.5.0 | [makeself-release-2.5.0-patch1.tar.gz](https://gitcode.com/cann-src-third-party/makeself/releases/download/release-2.5.0-patch1.0/makeself-release-2.5.0-patch1.tar.gz) |
   | cann-cmake | Read `CANN_CMAKE_TAG` from the source checkout | Use the command below |

   Run from the parent directory of the extracted `graph-autofusion` source repository:

   ```bash
   CANN_CMAKE_TAG=$(sed -n 's/^[[:space:]]*set(CANN_CMAKE_TAG "\([^"]*\)").*/\1/p' graph-autofusion/cmake/fetch_cann_cmake.cmake)
   test -n "${CANN_CMAKE_TAG}"
   curl -fLO "https://cann-3rd.obs.cn-north-4.myhuaweicloud.com/cmake/cmake-${CANN_CMAKE_TAG}.tar.gz"
   ```

3. Upload the source code and third-party open-source software to the offline compilation environment. Extract the source code.

   ```bash
   # Extract source code
   unzip graph-autofusion.zip
   ```

4. Create an open_source directory at the same level as the source code for makeself and extract it there. Place the unextracted cann-cmake archive in output/third_party inside the source repository.

   ```bash
   # Create open_source directory
   mkdir open_source
   # Move third-party open-source software packages to open_source directory
   mv makeself-release-2.5.0-patch1.tar.gz open_source/
   mkdir -p graph-autofusion/output/third_party
   CANN_CMAKE_TAG=$(sed -n 's/^[[:space:]]*set(CANN_CMAKE_TAG "\([^"]*\)").*/\1/p' graph-autofusion/cmake/fetch_cann_cmake.cmake)
   test -n "${CANN_CMAKE_TAG}"
   mv "cmake-${CANN_CMAKE_TAG}.tar.gz" graph-autofusion/output/third_party/
   # Extract third-party open-source software packages and rename (cann-cmake package remains unchanged)
   cd open_source
   tar -zxvf makeself-release-2.5.0-patch1.tar.gz && mv makeself-release-2.5.0 makeself
   ```

5. The file directory structure after completion:

   ```bash
   ├── graph-autofusion           # Extracted source repository
   │  ├── cmake
   │  ├── output/third_party
   │  │  └── cmake-${CANN_CMAKE_TAG}.tar.gz
   │  └── ...
   ├── open_source                # Third-party open-source software
   │  └── makeself
   │     ├── COPYING
   │     ├── Makefile
   │     ├── README.md
   │     └── ...
   ```

### 4.2 Install Dependencies

#### 4.2.1 Install Dependencies

The following lists dependencies used for source code compilation. Pay attention to version requirements:
> [!NOTE] Note
> If you use the mirror method for project experience, all dependencies are included in [init_env.sh](../../scripts/init_env.sh). You can skip this dependency installation step.

- Python3 >= 3.8.0 (Python virtual environment recommended)
  > [!NOTE] Note
  > - Python announced 3.8.x EOL. CANN will stop support for this version in 10.0.0. Please upgrade to >= 3.9.x.

  1. Create and activate a virtual environment:

     ```shell
     python3 -m venv venv
     source venv/bin/activate
     ```

  2. Install dependencies:

     ```shell
     cd graph-autofusion
     pip3 install -r super_kernel/requirements-dev.txt
     ```

- patch

   ```shell
   # Ubuntu/Debian installation example. For other operating systems, install manually.
   sudo apt-get install patch
   ```

- CMake >= 3.16.0  (version 3.20.0 recommended)

   ```shell
   # Ubuntu/Debian installation example. For other operating systems, install manually.
   sudo apt-get install cmake
   ```

- GCC >= 7.3.0

  > [!NOTE] Note
  > - The default build uses the gcc/g++ already installed in the environment and does not change the system default compiler.
  > - To switch to gcc15/gcc16, explicitly set `CC/CXX` before building, for example `export CC=gcc-15 CXX=g++-15`.
  > - You can also set `GCC_VERSION=15` or `GCC_VERSION=16`; the scripts will generate the matching compiler commands. Note: `GCC_VERSION` only takes effect for test scripts (e.g. `scripts/test/run_autofuse_test.sh`) and CI builds. The `build.sh --pkg` main build does not read this variable; set `CC/CXX` explicitly to switch compilers for the main build.
  > - After switching compilers, clean `build/` before reconfiguring so CMake does not reuse the old compiler cache.

#### 4.2.2 Check Compilation Environment

After environment preparation, execute the environment check script. Confirm whether the current environment meets compilation requirements.

```bash
bash scripts/check_env.sh
```

Check result descriptions:

| Status | Meaning | Recommendation |
|---|---|---|
| **[PASS]** | Check passed | No action required |
| **[WARNING]** | Non-critical dependency missing or version deviation | Fix recommended, does not affect core compilation |
| **[ERROR]** | Critical dependency missing or version incompatible | Must fix, otherwise compilation cannot proceed |

> [!NOTE] Note
> All check items and version constraints in the environment check script come strictly from docs/en/build.md and super_kernel/requirements-dev.txt. If documentation and dependencies update, synchronize changes to the [script](../../scripts/check_env.sh).

### 4.3 Compilation

Navigate to the source repository root directory. Execute the following command to compile:

```shell
bash build.sh --pkg
```

View more compilation parameters through `bash build.sh -h`. After successful execution, `cann-graph-autofusion_${cann_version}_linux-${arch}.run` generates in the `build_out` directory.

- --pkg indicates building a run package.
- ${cann_version} indicates the cann version number.
- ${arch} indicates CPU architecture, for example, aarch64 or x86_64.

During compilation, CMake automatically downloads third-party source packages (abseil-cpp, boost, json, protobuf, symengine, googletest, mockcpp, and so on) required by autofuse and superkernel AOT UT from external networks (for example, `https://gitcode.com/cann-src-third-party/` and Huawei Cloud OBS) through `ExternalProject_Add`. If your compilation environment cannot directly access external networks (for example, enterprise intranet or Docker container default bridge network), select the appropriate solution:

**Solution 1: Configure Network Proxy (Recommended)**
If the environment has an HTTP proxy that can access external networks, set the `http_proxy` / `https_proxy` environment variables before compilation. `build.sh` automatically passes these proxy variables to CMake subprocesses:

```shell
# Set proxy (modify according to actual proxy address)
export http_proxy=http://user:password@proxy-server:port
export https_proxy=http://user:password@proxy-server:port

# Execute compilation
bash build.sh --pkg
```

> [!NOTE] Note
> - `build.sh` automatically detects and inherits `http_proxy` and `https_proxy` environment variables in the current shell. No additional configuration required.
> - If you use git proxy (configured through `git config --global http.proxy`), confirm that the shell environment also sets corresponding environment variables. Verify through `echo $http_proxy`.

**Solution 2: Manually Pre-download Third-Party Packages**
In environments without any external network access, pre-download third-party source packages on a networked machine. Copy them to the specified directory in the compilation environment for offline compilation.

1. Download the following third-party packages on a networked machine:

   Run from the source root of the same checkout that will be built offline:

   ```bash
   CANN_CMAKE_TAG=$(sed -n 's/^[[:space:]]*set(CANN_CMAKE_TAG "\([^"]*\)").*/\1/p' cmake/fetch_cann_cmake.cmake)
   test -n "${CANN_CMAKE_TAG}"
   curl -fLO "https://cann-3rd.obs.cn-north-4.myhuaweicloud.com/cmake/cmake-${CANN_CMAKE_TAG}.tar.gz"
   ```

   | Third-Party Package | Version | Download Link |
   |---------|------|---------|
   | cann-cmake | Read `CANN_CMAKE_TAG` from the source checkout | Use the command below |
   | abseil-cpp | 20230802.1 | https://gitcode.com/cann-src-third-party/abseil-cpp/releases/download/20230802.1/abseil-cpp-20230802.1.tar.gz |
   | json | 3.11.3 | https://gitcode.com/cann-src-third-party/json/releases/download/v3.11.3/json-3.11.3.tar.gz |
   | boost | 1.87.0 | https://gitcode.com/cann-src-third-party/boost/releases/download/v1.87.0/boost_1_87_0.tar.gz |
   | protobuf | 25.1 | https://gitcode.com/cann-src-third-party/protobuf/releases/download/v25.1/protobuf-25.1.tar.gz |
   | symengine | 0.12.0 | https://gitcode.com/cann-src-third-party/symengine/releases/download/v0.12.0/symengine-0.12.0.tar.gz |
   | googletest | 1.14.0 | https://gitcode.com/cann-src-third-party/googletest/releases/download/v1.14.0/googletest-1.14.0.tar.gz |
   | mockcpp | 2.7 | https://cann-3rd.obs.cn-north-4.myhuaweicloud.com/mockcpp/mockcpp-2.7.tar.gz |
   | mockcpp patch | 2.7-h5 | https://gitcode.com/cann-src-third-party/mockcpp/releases/download/v2.7-h5/mockcpp-2.7-h5.patch |
   | makeself | 2.5.0 | https://gitcode.com/cann-src-third-party/makeself/releases/download/release-2.5.0-patch1.0/makeself-release-2.5.0-patch1.tar.gz |

2. Copy the downloaded packages to the corresponding subdirectories under `output/third_party/` in the compilation environment (create if not exist):

   ```shell
   # Create directory structure under source root
   mkdir -p output/third_party/{abseil-cpp,json,boost,protobuf,symengine,gtest,mockcpp,makeself}

   # Place downloaded packages in corresponding directories (filenames must match the table below)
   # cmake-${CANN_CMAKE_TAG}.tar.gz              → output/third_party/
   # abseil-cpp-20230802.1.tar.gz           → output/third_party/abseil-cpp/
   # json-3.11.3.tar.gz                     → output/third_party/json/
   # boost_1_87_0.tar.gz                    → output/third_party/boost/
   # protobuf-25.1.tar.gz                   → output/third_party/protobuf/
   # symengine-0.12.0.tar.gz                → output/third_party/symengine/
   # googletest-1.14.0.tar.gz               → output/third_party/gtest/
   # mockcpp-2.7.tar.gz                     → output/third_party/mockcpp/
   # mockcpp-2.7-h5.patch                   → output/third_party/              # preferred path
   # mockcpp-2.7-h5.patch                   → output/third_party/pkg/          # fallback path
   # makeself-release-2.5.0-patch1.tar.gz   → output/third_party/makeself/
   ```

   Example placement for the mockcpp package and patch file:

   ```shell
   # Preferred placement
   mkdir -p output/third_party/mockcpp
   cp mockcpp-2.7.tar.gz output/third_party/mockcpp/
   cp mockcpp-2.7-h5.patch output/third_party/

   # Alternative fallback placement
   mkdir -p output/third_party/mockcpp output/third_party/pkg
   cp mockcpp-2.7.tar.gz output/third_party/mockcpp/
   cp mockcpp-2.7-h5.patch output/third_party/pkg/
   ```

   To manually verify that the mockcpp patch can be applied before compilation, run the following git commands in a temporary directory. This is not required for normal builds because CMake performs this step automatically.

   ```shell
   (
     # Run from the source root. This example uses the preferred patch path.
     GRAPH_AUTOFUSION_HOME=$(pwd)
     rm -rf /tmp/mockcpp_patch_check
     mkdir -p /tmp/mockcpp_patch_check
     tar -zxf output/third_party/mockcpp/mockcpp-2.7.tar.gz -C /tmp/mockcpp_patch_check --strip-components=1
     cd /tmp/mockcpp_patch_check
     git init
     git apply --check ${GRAPH_AUTOFUSION_HOME}/output/third_party/mockcpp-2.7-h5.patch

     # To apply the patch manually, continue with:
     git apply ${GRAPH_AUTOFUSION_HOME}/output/third_party/mockcpp-2.7-h5.patch
   )
   ```
   The check runs in a subshell, so the calling shell remains at the repository root for the subsequent build command.


   If the patch is placed in the fallback path, replace the patch path above with `${GRAPH_AUTOFUSION_HOME}/output/third_party/pkg/mockcpp-2.7-h5.patch`.

   > [!NOTE] Note
   > - `mockcpp-2.7-h5.patch` is required to build mockcpp. When `add_cann_third_party(mockcpp)` is used, CMake automatically applies this patch with `git init && git apply ${PATCH_FILE}` after extracting `mockcpp-2.7.tar.gz`. No manual patch step is required.
   > - For offline compilation, prepare both `mockcpp-2.7.tar.gz` and `mockcpp-2.7-h5.patch`. If the patch file is missing, CMake attempts to download it from the network, which causes offline builds to fail.
   > - The preferred mockcpp patch lookup path is `output/third_party/mockcpp-2.7-h5.patch`. If that path does not exist, CMake uses `output/third_party/pkg/mockcpp-2.7-h5.patch`.
   > - The patch is applied automatically during the CMake ExternalProject patch stage. Do not pre-extract and modify the `output/third_party/mockcpp` source directory manually, because later builds may clean or overwrite it.

   > The `cann-cmake` bootstrap dependency must match `CANN_CMAKE_TAG` in `cmake/fetch_cann_cmake.cmake`. Put the archive directly in `output/third_party/` (or the directory specified by `--cann_3rd_lib_path`) without extracting it. The current bootstrap does not recognize older-version archives.

3. During compilation, specify the local path through `--cann_3rd_lib_path` to skip the download step:

   ```shell
   bash build.sh --pkg --cann_3rd_lib_path=$(pwd)/output/third_party
   ```

> [!NOTE] Note
> - If you do not specify `--cann_3rd_lib_path`, the default search path is `./output/third_party`. Therefore, you can omit this parameter when packages exist in the default path.
> - CMake build scripts prioritize checking whether corresponding tarballs already exist in the local path. If they exist, the download skips.

### 4.4 Installation and Uninstallation

> [!WARNING] Important
> You must install the compiled `.run` package before running UT/ST tests. Otherwise, `LD_LIBRARY_PATH` will load older versions of dynamic libraries from the CANN installation path, causing runtime errors such as `undefined symbol`.

#### Installation

After local verification completes, execute the following command to install the compiled package. Ensure the installation user has execute permission for the package.

```shell
# Specify installation path if needed: --install-path=${install_path}
./build_out/cann-graph-autofusion_${cann_version}_linux-${arch}.run --full --quiet --pylocal
```

> [!NOTE] Note
> - The installation path (default or specified) must match the path where you installed the cann-toolkit package.
> - --full indicates full installation mode.
> - --install-path specifies the installation path. If not specified, the default installation path is `/usr/local/Ascend` (root user) or `${HOME}/Ascend` (non-root user).
> - --quiet indicates silent installation. It skips human-computer interaction.
> - --pylocal determines whether to install .whl files inside the package along the run package installation path.
>   - If you select this parameter, .whl installs in the `${ascend_install_path}/cann/python/site-packages` path.
>   - If you do not select this parameter, .whl installs in the local python path, for example, `/usr/local/python3.x/lib/python3.x/site-packages`.
> - For more installation options, use the --help option to view.

#### Uninstallation

If you want to uninstall the installed package, execute the following command:

```shell
# Add --install-path=${install_path} if installed to a specified path
./build_out/cann-graph-autofusion_${cann_version}_linux-${arch}.run --uninstall
```

### 4.5 Test Verification

After installation, you can perform developer testing. Ensure you have completed [Environment Preparation](./quick_install.md#1-environment-preparation) and installed the compiled `.run` package before executing operations in this section.

The `.run` package provides the shared libraries required by the tests. SuperKernel Python UT/ST loads `superkernel` directly from the current source tree and does not replace or uninstall the wheel installed in the CANN path by `--pylocal`.

When only Autofuse test modules are selected without `--pkg`, the test entry point builds its target dependencies directly,
without a preliminary full product build. `-u --module=autofuse_ascendc_api` runs numeric API UT for both architectures
without building the test `pyautofuse/optimize/codegen` libraries. Broadcast code-generation and kernel-execution cases
run under `-s --module=autofuse_e2e` and remain included in the complete Autofuse UT suite invoked with
`bash scripts/test/run_autofuse_test.sh -u -m all -j 8`.
Default parallelism uses the CPUs available to the current process, respecting affinity and container CPU quota,
rather than the host's exposed CPU count or a fixed limit of 14.

- UT Verification

   ```bash
   bash build.sh -u --cann_3rd_lib_path=$(pwd)/output/third_party
   ```

   After execution, check the UT test execution status through the output log. Successful test case execution prints `passed` without any `failed` print. Confirm all test cases pass.

- ST Verification

   ```bash
   bash build.sh -s --cann_3rd_lib_path=$(pwd)/output/third_party
   ```

   After execution, check the ST test execution status through the output log. Successful test case execution prints `passed` without any `failed` print. Confirm all test cases pass.

- Coverage Verification

   ```bash
   bash build.sh -u -c --cann_3rd_lib_path=$(pwd)/output/third_party  # UT coverage
   bash build.sh -s -c --cann_3rd_lib_path=$(pwd)/output/third_party  # ST coverage
   bash build.sh -c --cann_3rd_lib_path=$(pwd)/output/third_party     # UT coverage + ST coverage
   ```

   After execution, check the coverage status through the output log. Confirm all test cases pass.

**After the installation is complete, you can refer to the [SuperKernel sample](../../super_kernel/examples/README_en.md) or the [AutoFusion sample](../../autofuse/examples/pytorch/README_en.md) to run the examples.**
