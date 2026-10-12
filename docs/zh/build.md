# 源码构建

## 1. 环境准备

在源码编译前，请先完成基础环境搭建。具体操作请参见[快速安装](quick_install.md)。

## 2. 环境验证

安装完CANN包后，需验证环境是否正常。

```bash
# 查看CANN Toolkit的version字段提供的版本信息（默认路径安装），<arch>表示CPU架构（aarch64或x86_64）。WebIDE场景下，请将/usr/local替换为/home/developer。
cat /usr/local/Ascend/cann/<arch>-linux/ascend_toolkit_install.info
# 查看CANN ops的version字段提供的版本信息（默认路径安装），<opsname>表示待查询的ops子包的名称，请用户根据实际安装路径替换。WebIDE场景下，请将/usr/local替换为/home/developer。
cat /usr/local/Ascend/cann/<arch>-linux/ascend_ops_install.info
```

## 3. 环境变量配置

根据实际场景，选择合适的命令：

  ```bash
  # 默认路径安装，以root用户为例（非root用户，将/usr/local替换为${HOME}）
  source /usr/local/Ascend/cann/set_env.sh
  # 指定路径安装
  source ${install_path}/cann/set_env.sh
  ```

## 4. 源码编译

### 4.1 下载源码

若您的编译环境可以访问网络，编译过程中将自动下载开源第三方软件，可以使用如下命令进行源码下载：

```bash
git clone https://gitcode.com/cann/graph-autofusion.git
 ```

若您的编译环境无法访问网络，您需要通过下列步骤在联网环境中下载源码及开源软件压缩包，并手动上传至您的编译环境中：

1. 在联网环境中，进入[本项目主页](https://gitcode.com/cann/graph-autofusion)，通过`下载zip`或`Clone`按钮，根据指导，完成源码下载。

2. 下载`makeself`，`cann-cmake`第三方开源软件。

   | 开源软件 | 版本 | 下载地址 |
   |---|---|---|
   | makeself | 2.5.0 | [makeself-release-2.5.0-patch1.tar.gz](https://gitcode.com/cann-src-third-party/makeself/releases/download/release-2.5.0-patch1.0/makeself-release-2.5.0-patch1.tar.gz) |
   | cann-cmake | master-052 | [cmake-master-052.tar.gz](https://cann-3rd.obs.cn-north-4.myhuaweicloud.com/cmake/cmake-master-052.tar.gz) |

3. 将源码及第三方开源软件上传到离线编译环境中，解压缩源码。

   ```bash
   # 解压缩源码
   unzip graph-autofusion.zip
   ```

4. 在源码同级目录创建open_source目录，将第三方开源软件压缩包放到open_source目录下解压缩后改名。

   ```bash
   # 创建open_source目录
   mkdir open_source
   # 将第三方开源软件压缩包放到open_source目录下
   mv makeself-release-2.5.0-patch1.tar.gz open_source/
   mv cmake-master-052.tar.gz open_source/
   # 解压缩第三方开源软件压缩包并改名（cann-cmake压缩包保持不变）
   cd open_source
   tar -zxvf makeself-release-2.5.0-patch1.tar.gz && mv makeself-release-2.5.0 makeself
   ```

5. 完成后文件目录如下：

   ```bash
   ├── graph-autofusion           # 解压后的源码仓
   │  ├── cmake
   │  └── ...
   ├── open_source                # 三方开源软件
   │  └── makeself
   │     ├── COPYING
   │     ├── Makefile
   │     ├── README.md
   │     └── ...
   │  └── cmake-master-052.tar.gz
   ```

### 4.2 安装依赖

#### 4.2.1 安装依赖

以下所列为源码编译用到的依赖，请注意版本要求：
> [!NOTE] 注意
> 如使用镜像方式进行项目体验，所有依赖已包含在[init_env.sh](../../scripts/init_env.sh)中，可跳过此安装依赖步骤。

- Python3 >= 3.8.0 （建议使用Python虚拟环境）
  > [!NOTE] 说明
  > - python宣布3.8.x已经EOL，CANN将在10.0.0停止对该版本的支持，请升级到>=3.9.x的版本

  1. 创建虚拟环境并激活：

     ```shell
     python3 -m venv venv
     source venv/bin/activate
     ```

  2. 安装依赖：

     ```shell
     cd graph-autofusion
     pip3 install -r super_kernel/requirements-dev.txt
     ```

- patch

   ```shell
   # Ubuntu/Debian操作系统安装命令示例如下，其他操作系统请自行安装
   sudo apt-get install patch
   ```

- CMake >= 3.16.0  （建议使用3.20.0版本）

   ```shell
   # Ubuntu/Debian操作系统安装命令示例如下，其他操作系统请自行安装
   sudo apt-get install cmake
   ```

- GCC >= 7.3.0

  > [!NOTE] 说明
  > - 默认使用当前环境已安装的 gcc/g++，不会修改系统默认编译器。
  > - 如需切换到 gcc15/gcc16，请在编译前显式设置 `CC/CXX`，例如 `export CC=gcc-15 CXX=g++-15`。
  > - 也可以设置 `GCC_VERSION=15` 或 `GCC_VERSION=16`，由脚本生成对应的编译器命令。注意：`GCC_VERSION` 仅对测试脚本（如 `scripts/test/run_autofuse_test.sh`）和 CI 构建生效，`build.sh --pkg` 主构建不读取该变量，主构建切换编译器请显式设置 `CC/CXX`。
  > - 切换编译器后请清理 `build/` 再重新配置，避免 CMake 缓存沿用旧编译器。

#### 4.2.2 检查编译环境

环境准备完成后，建议执行环境检查脚本，确认当前环境是否满足编译要求。

```bash
bash scripts/check_env.sh
```

检查结果说明如下：

| 状态 | 含义 | 处理建议 |
|---|---|---|
| **[PASS]** | 检查通过 | 无需处理 |
| **[WARNING]** | 非关键依赖缺失或版本存在偏差 | 建议修复，不影响核心编译 |
| **[ERROR]** | 关键依赖缺失或版本不兼容 | 必须修复，否则无法编译 |

> [!NOTE] 注意
> 环境检查脚本中所有的检查项和版本约束严格来源于 docs/zh/build.md 和 super_kernel/requirements-dev.txt，如文档和依赖更新，请同步修改[脚本](../../scripts/check_env.sh)。

### 4.3 编译

进入源码仓根目录，执行以下命令进行编译：

```shell
bash build.sh --pkg
```

更多编译参数可以通过`bash build.sh -h`查看，执行成功后会在`build_out`目录下生成`cann-graph-autofusion_${cann_version}_linux-${arch}.run`。

- --pkg 表示构建 run 包。
- ${cann_version} 表示 cann 版本号。
- ${arch} 表示 CPU 架构，如 aarch64、x86_64。

编译过程中，CMake 会通过 `ExternalProject_Add` 从外网（如 `https://gitcode.com/cann-src-third-party/`、华为云 OBS 等）自动下载 autofuse 和 superkernel AOT UT 所需的第三方源码包（abseil-cpp、boost、json、protobuf、symengine、googletest、mockcpp 等）。如果您的编译环境无法直接访问外网（如企业内网、Docker 容器默认 bridge 网络），请根据实际情况选择以下解决方案：

**方案一：配置网络代理（推荐）**
如果环境中有 HTTP 代理可访问外网，设置 `http_proxy` / `https_proxy` 环境变量后执行编译。`build.sh` 会自动将这些代理变量传递给 CMake 子进程：

```shell
# 设置代理（根据实际代理地址修改）
export http_proxy=http://user:password@proxy-server:port
export https_proxy=http://user:password@proxy-server:port

# 执行编译
bash build.sh --pkg
```

> [!NOTE] 说明
> - `build.sh` 会自动检测并继承当前 shell 中的 `http_proxy` 和 `https_proxy` 环境变量，无需额外配置。
> - 如果使用 git 代理（通过 `git config --global http.proxy` 配置），需确认 shell 环境中也设置了对应的环境变量。可通过 `echo $http_proxy` 验证。

**方案二：手动预下载第三方包**
在不具备任何外网访问能力的环境中，可在联网机器上预先下载第三方源码包，拷贝到编译环境指定目录后离线编译。

1. 在联网机器上下载以下第三方包：

   | 第三方包 | 版本 | 下载地址 |
   |---------|------|---------|
   | abseil-cpp | 20230802.1 | https://gitcode.com/cann-src-third-party/abseil-cpp/releases/download/20230802.1/abseil-cpp-20230802.1.tar.gz |
   | json | 3.11.3 | https://gitcode.com/cann-src-third-party/json/releases/download/v3.11.3/json-3.11.3.tar.gz |
   | boost | 1.87.0 | https://gitcode.com/cann-src-third-party/boost/releases/download/v1.87.0/boost_1_87_0.tar.gz |
   | protobuf | 25.1 | https://gitcode.com/cann-src-third-party/protobuf/releases/download/v25.1/protobuf-25.1.tar.gz |
   | symengine | 0.12.0 | https://gitcode.com/cann-src-third-party/symengine/releases/download/v0.12.0/symengine-0.12.0.tar.gz |
   | googletest | 1.14.0 | https://gitcode.com/cann-src-third-party/googletest/releases/download/v1.14.0/googletest-1.14.0.tar.gz |
   | mockcpp | 2.7 | https://cann-3rd.obs.cn-north-4.myhuaweicloud.com/mockcpp/mockcpp-2.7.tar.gz |
   | mockcpp patch | 2.7-h5 | https://gitcode.com/cann-src-third-party/mockcpp/releases/download/v2.7-h5/mockcpp-2.7-h5.patch |
   | makeself | 2.5.0 | https://gitcode.com/cann-src-third-party/makeself/releases/download/release-2.5.0-patch1.0/makeself-release-2.5.0-patch1.tar.gz |

2. 将下载的包拷贝到编译环境的 `output/third_party/` 对应子目录下（如不存在则创建）：

   ```shell
   # 在源码根目录下创建目录结构
   mkdir -p output/third_party/{abseil-cpp,json,boost,protobuf,symengine,gtest,mockcpp,makeself}

   # 将下载的包放入对应目录（文件名须与下表一致）
   # abseil-cpp-20230802.1.tar.gz           → output/third_party/abseil-cpp/
   # json-3.11.3.tar.gz                     → output/third_party/json/
   # boost_1_87_0.tar.gz                    → output/third_party/boost/
   # protobuf-25.1.tar.gz                   → output/third_party/protobuf/
   # symengine-0.12.0.tar.gz                → output/third_party/symengine/
   # googletest-1.14.0.tar.gz               → output/third_party/gtest/
   # mockcpp-2.7.tar.gz                     → output/third_party/mockcpp/
   # mockcpp-2.7-h5.patch                   → output/third_party/              # 推荐路径
   # mockcpp-2.7-h5.patch                   → output/third_party/pkg/          # 备用路径
   # makeself-release-2.5.0-patch1.tar.gz   → output/third_party/makeself/
   ```

   mockcpp 离线包和补丁文件放置示例：

   ```shell
   # 推荐放置方式
   mkdir -p output/third_party/mockcpp
   cp mockcpp-2.7.tar.gz output/third_party/mockcpp/
   cp mockcpp-2.7-h5.patch output/third_party/

   # 如需放到备用路径，则使用如下方式
   mkdir -p output/third_party/mockcpp output/third_party/pkg
   cp mockcpp-2.7.tar.gz output/third_party/mockcpp/
   cp mockcpp-2.7-h5.patch output/third_party/pkg/
   ```

   如需在编译前手动验证 mockcpp 补丁是否可以应用，可在临时目录中执行以下 git 命令。正常编译时无需执行，CMake 会自动完成该步骤。

   ```shell
   (
     # 在源码根目录下执行，使用推荐 patch 放置路径进行验证
     GRAPH_AUTOFUSION_HOME=$(pwd)
     rm -rf /tmp/mockcpp_patch_check
     mkdir -p /tmp/mockcpp_patch_check
     tar -zxf output/third_party/mockcpp/mockcpp-2.7.tar.gz -C /tmp/mockcpp_patch_check --strip-components=1
     cd /tmp/mockcpp_patch_check
     git init
     git apply --check ${GRAPH_AUTOFUSION_HOME}/output/third_party/mockcpp-2.7-h5.patch

     # 如需手动应用补丁，可继续执行
     git apply ${GRAPH_AUTOFUSION_HOME}/output/third_party/mockcpp-2.7-h5.patch
   )
   ```
   上述校验在子Shell中执行，退出后仍位于源码根目录，可继续执行后续构建命令。


   如果 patch 放在备用路径，请将上述命令中的 patch 路径替换为 `${GRAPH_AUTOFUSION_HOME}/output/third_party/pkg/mockcpp-2.7-h5.patch`。

   > [!NOTE] 说明
   > - `mockcpp-2.7-h5.patch` 是 mockcpp 编译必需的补丁文件。使用 `add_cann_third_party(mockcpp)` 时，CMake 会在解压 `mockcpp-2.7.tar.gz` 后自动执行 `git init && git apply ${PATCH_FILE}` 应用该补丁，无需手动打补丁。
   > - 离线编译时需同时准备 `mockcpp-2.7.tar.gz` 和 `mockcpp-2.7-h5.patch`。如果缺少 patch 文件，CMake 会尝试从网络下载，离线环境下会导致构建失败。
   > - mockcpp patch 优先查找路径为 `output/third_party/mockcpp-2.7-h5.patch`；如该路径不存在，会使用 `output/third_party/pkg/mockcpp-2.7-h5.patch`。
   > - patch 会在 CMake 的 ExternalProject patch 阶段自动应用，不建议提前手动解压并修改 `output/third_party/mockcpp` 源码目录，避免被后续构建清理或覆盖。

3. 执行编译时，通过 `--cann_3rd_lib_path` 指定本地路径，跳过下载步骤：

   ```shell
   bash build.sh --pkg --cann_3rd_lib_path=$(pwd)/output/third_party
   ```

> [!NOTE] 说明
> - 如果不指定 `--cann_3rd_lib_path`，默认查找路径为 `./output/third_party`，因此只要包放在该默认路径下，编译时也可省略此参数。
> - CMake 构建脚本会优先检查本地路径是否已存在对应 tarball，存在则跳过下载。

### 4.4 安装与卸载

> [!WARNING] 重要
> 执行UT/ST测试前，必须先安装编译生成的 `.run` 包。否则测试运行时 `LD_LIBRARY_PATH` 会加载到 CANN 安装路径下的旧版本动态库，导致 `undefined symbol` 等运行时错误。

#### 安装

本地验证完成后，可执行如下命令安装编译生成的软件包，执行安装命令时，请确保安装用户对软件包具有可执行权限。

```shell
# 如果需要指定安装路径，则加上 --install-path=${install_path}
./build_out/cann-graph-autofusion_${cann_version}_linux-${arch}.run --full --quiet --pylocal
```

> [!NOTE]说明
> - 此处的安装路径（无论默认还是指定）需与前面安装 cann-toolkit 包时的路径保持一致。
> - --full          全量模式安装。
> - --install-path  指定安装路径，不指定则默认安装在`/usr/local/Ascend`（root 用户）或`${HOME}/Ascend`（非 root 用户）目录。
> - --quiet         静默安装，跳过人机交互环节。
> - --pylocal       安装 run 包时，是否将包内的 .whl 跟随 run 包安装路径来安装。
>   - 若选择该参数，则 .whl 安装在`${ascend_install_path}/cann/python/site-packages`路径下。
>   - 若不选择该参数，则 .whl 安装在本地 python 路径下，例如`/usr/local/python3.x/lib/python3.x/site-packages`。
> - 更多安装选项请使用 --help 选项查看。

#### 卸载

若您想卸载安装的软件包，可执行如下命令：

```shell
# 如果是安装到指定路径情况，则加上 --install-path=${install_path}
./build_out/cann-graph-autofusion_${cann_version}_linux-${arch}.run --uninstall
```

### 4.5 测试验证

安装完成后，用户可以进行开发者测试，在执行本章节操作之前，确保已完成[环境准备](./quick_install.md#1-环境准备)并已安装编译生成的 `.run` 包。

其中，`.run` 包用于提供测试依赖的动态库。SuperKernel Python UT/ST 直接从当前源码目录加载 `superkernel`，不会替换或卸载通过 `--pylocal` 安装到 CANN 路径下的 wheel 包。

- UT 验证

   ```bash
   bash build.sh -u --cann_3rd_lib_path=$(pwd)/output/third_party
   ```

   执行完成后根据输出日志查看UT测试执行情况，用例执行成功会打印`passed`并且无`failed`打印，确认所有测试用例通过。

- ST 验证

   ```bash
   bash build.sh -s --cann_3rd_lib_path=$(pwd)/output/third_party
   ```

   执行完成后根据输出日志查看ST测试执行情况，用例执行成功会打印`passed`并且无`failed`打印，确认所有测试用例通过。

- 覆盖率验证

   ```bash
   bash build.sh -u -c --cann_3rd_lib_path=$(pwd)/output/third_party  # UT 覆盖率
   bash build.sh -s -c --cann_3rd_lib_path=$(pwd)/output/third_party  # ST 覆盖率
   bash build.sh -c --cann_3rd_lib_path=$(pwd)/output/third_party     # UT 覆盖率 + ST 覆盖率
   ```

   执行完成后根据输出日志查看覆盖率情况，确认所有测试用例通过。

**安装完成后可参考[super_kernel样例运行](../../super_kernel/examples/README.md)或[autofusion样例运行](../../autofuse/examples/pytorch/README.md)尝试运行样例**。
