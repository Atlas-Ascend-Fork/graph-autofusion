# 环境部署

简体中文 | [English](../en/quick_install.md)

## 1. 环境准备

本项目支持源码编译，在源码编译前，需要确保已经安装CANN软件（Toolkit开发套件包和ops包（可选）），若运行样例，还需要安装NPU驱动和固件。

软件安装方式请根据如下描述进行选择：

| 安装方式       | 说明                                                         | 使用场景                                                     |
| :------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| CANNLab | 一站式开发平台，提供在线直接运行的昇腾环境，无需手动安装。<br>当前可提供单机算力，**默认安装最新商发版CANN包**。 | 适用于没有昇腾设备的开发者。                                 |
|  Docker  | Docker镜像是一种高效部署方式，一键部署CANN包和必备依赖。<br>当前OS仅支持Ubuntu操作系统，**默认安装最新商发版CANN包**。 |适用有昇腾设备，需要快速搭建环境的开发者。|
| 手动安装 | 手动安装CANN包和基础依赖，灵活性高。 | 适用有昇腾设备，想体验手动安装CANN包或体验最新master分支能力的开发者。 |

### 方式一：CANNLab

对于无昇腾设备的开发者，可直接使用CANNLab云开发环境，即“**一站式开发平台**”，该平台为您提供在线可直接运行的昇腾环境，环境中已安装必备的驱动固件、软件包和依赖，无需手动安装。

> [!NOTE]说明
> 环境默认安装最新商发版CANN包，源码下载时注意与软件配套。体验master版本能力或基于master版本进行开发，请参考[**方式三-场景1**](#方式三手动安装软件包)安装最新CANN包依赖。<br>更多关于开发平台的介绍请参考[CANNLab指导](https://gitcode.com/org/cann/discussions/54)。

1. 进入开源项目，单击“`CANNLab`”按钮，使用已认证过的华为云账号登录。若未注册或认证，请根据页面提示进行注册和认证。

   <img src="../figures/cloudIDE.png" alt="云平台"  width="750px" height="90px">

2. 根据页面提示创建NPU环境并配置规格，启动云开发环境后，单击“`连接 > WebIDE`”进入一站式开发平台。

   当前开源项目资源目录根据环境创建方式有所不同：
   - 如果是从CANN社区仓库创建的环境，资源默认在`/mnt/workspace/gitCode/cann`目录下。
   - 如果是从个人仓库fork的CANN仓库创建的环境，资源默认在`/mnt/workspace/gitCode/{fork仓库}`目录下。

   <img src="../figures/webIDE.png" alt="云平台"  width="1000px" height="150px">

### 方式二：Docker部署

对于不依赖昇腾设备的开发者，若您想快速搭建编译构建环境，可使用Docker镜像部署。

> [!NOTE]说明
> 镜像文件比较大，下载需要一定时间，请您耐心等待。关于docker命令的选项介绍可通过`docker --help`查询。

1.**安装驱动与固件（运行态依赖）**

宿主机上昇腾驱动与固件的下载和安装操作请参考《[CANN软件安装指南](https://www.hiascend.com/document/redirect/CannCommunityInstWizard)》中“准备软件包”和“安装NPU驱动和固件”章节。

2.**下载镜像**

- 步骤1：以root用户登录宿主机。确保宿主机已安装Docker引擎（版本1.11.2及以上）。
- 步骤2：从[昇腾镜像仓库](https://www.hiascend.com/developer/ascendhub/detail/17da20d1c2b6493cb38765adeba85884)拉取已预集成CANN软件包及开发所需依赖的镜像。命令如下，请根据实际架构选择（以Atlas A2系列产品为例，仅编译场景无需关注）：

    ```bash
    # 示例：拉取ARM架构的CANN开发镜像
    docker pull --platform=arm64 swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.0.0-beta.1-910-ubuntu22.04-py3.11
    # 示例：拉取X86架构的CANN开发镜像
    docker pull --platform=amd64 swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.0.0-beta.1-910-ubuntu22.04-py3.11
    ```

3.**运行Docker**
根据使用场景选择不同的启动方式：

- **场景1：仅编译构建（不需要运行样例）**

  如果只需要进行代码编译构建，无需访问NPU设备，使用以下简化命令：

  ```bash
  docker run --name cann_container -it -u root --privileged=true -v /home/gaf/:/home/gaf swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.0.0-beta.1-910-ubuntu22.04-py3.11 bash
  ```

- **场景2：需要运行样例（需要访问NPU设备）**

  如果需要运行样例或测试，容器需要访问宿主机的NPU设备。以Atlas A2系列产品为例：

  ```bash
  docker run --name cann_container \
    --device /dev/davinci0 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -it -u root --privileged=true \
    swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.0.0-beta.1-910-ubuntu22.04-py3.11 bash
  ```

  | 参数 | 说明 | 注意事项 |
  | :--- | :--- | :--- |
  | `--name cann_container` | 为容器指定名称，便于管理。 | 可选（取值可自定义）。 |
  | `--device /dev/davinci0` | 将NPU设备文件映射到容器内。 | 必选（运行样例时）。多个设备可多次使用此参数，如 `/dev/davinci0` `/dev/davinci1`。 |
  | `--device /dev/davinci_manager` | 昇腾设备管理器，负责设备资源管理。 | 必选（运行样例时）。 |
  | `--device /dev/devmm_svm` | 设备内存管理单元。 | 必选（运行样例时）。 |
  | `--device /dev/hisi_hdc` | 昇腾高清编解码设备。 | 必选（运行样例时）。 |
  | `-v /usr/local/dcmi:/usr/local/dcmi` | 挂载DCMI（Device Communication Management Interface）目录。 | 必选（运行样例时）。 |
  | `-v /usr/local/bin/npu-smi:...` | 挂载NPU监控工具，用于查看NPU状态。 | 必选（运行样例时）。 |
  | `-v /usr/local/Ascend/driver/...` | 挂载NPU驱动库和版本信息。 | 必选（运行样例时）。 |
  | `-v /etc/ascend_install.info:...` | 挂载昇腾软件安装信息。 | 必选（运行样例时）。 |
  | `-it` | `-i`（交互式）和 `-t`（分配伪终端）的组合参数。 | 必选 |
  | `-u root` | 以 root（管理员）身份进入容器。 | 建议 |
  | `--privileged=true` | 开启容器最高特权模式。 | 建议 |
  | `swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:...` | 指定要运行的Docker镜像。 | 必选，请确保此镜像名和标签（tag）与您通过`docker pull`拉取的镜像完全一致。 |
  | `bash` | 容器启动后立即执行的命令。 | 必选 |

  > [!NOTE]说明
  > - 场景1适用于仅编译构建graph-autofusion的场合，无需NPU设备支持。
  > - 场景2适用于需要运行样例或进行NPU相关测试的场合，需要宿主机已安装NPU驱动和固件。
  > - 如果使用其他型号芯片（如Ascend950PR、Atlas A3系列产品），请相应调整 `--device` 参数中的设备名称。

4.**初始化环境**
进入容器后，执行以下命令初始化环境：

  ```bash
  curl -fsSL https://raw.gitcode.com/cann/graph-autofusion/raw/master/scripts/init_env.sh | bash
  ```

> [!NOTE]说明
>
> - 默认进行CANN包和算子包的安装。
> - 对于其他芯片型号，请使用 `--chip-type` 参数指定对应的型号（如 `950`、`A3` ）。可选参数具体参考[init_env.sh](../../scripts/init_env.sh)

### 方式三：手动安装软件包

1. **安装驱动与固件（可选，仅运行[样例](../../autofuse/examples/pytorch/README.md)依赖）**

    驱动与固件的下载和安装操作请参考《[CANN软件安装指南](https://www.hiascend.com/document/redirect/CannCommunityInstWizard)》中“准备软件包”和“安装NPU驱动和固件”章节。驱动与固件是运行样例依赖，若仅编译环境，可以不安装。

2. **安装CANN包**

   **场景1：体验master版本能力或基于master版本进行开发**

     请单击[下载链接](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/master)获取最新时间版本，并根据产品型号和环境架构下载对应包。安装命令如下，更多指导请参考[CANN软件安装指南](https://www.hiascend.com/document/redirect/CannCommunityInstWizard)。

     1. 安装CANN Toolkit开发套件包。

        ```bash
        # 确保安装包具有可执行权限
        chmod +x Ascend-cann-toolkit_${cann_version}_linux-${arch}.run
        # 安装命令
        ./Ascend-cann-toolkit_${cann_version}_linux-${arch}.run --install --install-path=${install_path}
        ```

        - `${cann_version}`：表示CANN包版本号。
        - `${arch}`：表示CPU架构，如`aarch64`、`x86_64`。
        - `${install_path}`：表示指定安装路径，需要与Toolkit包安装在相同路径，root用户默认安装在`/usr/local/Ascend`目录。

     2. 安装CANN ops算子包。

        ```bash
        # 确保安装包具有可执行权限
        chmod +x Ascend-cann-${soc_name}-ops_${cann_version}_linux-${arch}.run
        # 安装命令
        ./Ascend-cann-${soc_name}-ops_${cann_version}_linux-${arch}.run --install --install-path=${install_path}
        ```

        其中`${soc_name}`表示NPU型号名称。

   **场景2：体验已发布版本能力或基于已发布版本进行开发**

    如果您想体验**官网正式发布的CANN包**能力，请访问[CANN官网下载中心](https://www.hiascend.com/cann/download)，选择对应版本CANN软件包（仅支持CANN 8.5.0及后续版本）进行安装。
