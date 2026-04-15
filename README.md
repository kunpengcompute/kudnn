# KuDNN

# 🔥release-notes
- [2026/03] KuDNN首次上线，支持linear、softmax和normal等算子，使用KML BLAS高性能库和JIT动态代码生成kernel作为底层计算方法

# 🚀概述
KDNN鲲鹏深度神经网络算子库，结合鲲鹏CPU微架构特性与软优化手段，优化AI算子性能。KDNN可以通过算子库插件形式集成进开源软件oneDNN框架，也可以单独作为后端加速算子集成到TensorFlow或者Pytorch。


# 📝版本配套

- 运行平台
    - 鲲鹏 920Pro
- 系统规格
    - openEuler 20.03（LTS-SP3）AArch64
    - openEuler 22.03（LTS-SP2）AArch64
    - openEuler 22.03（LTS-SP3）AArch64
    - openEuler 22.03（LTS-SP4）AArch64
    - openEuler 24.03（LTS-SP3）AArch64


# ⚡️编译安装与测试

## 1. [获取 HPCKit 软件包](https://www.hikunpeng.com/document/detail/zh/kunpenghpcs/hpckit/instg/KunpengHPCKit_install_007.html)
> https://www.hikunpeng.com/developer/hpc/hpckit-download
## 2. [安装 HPCKit](https://www.hikunpeng.com/document/detail/zh/kunpenghpcs/hpckit/instg/KunpengHPCKit_install_012.html)
### 解压 HPCKit 软件包（HPCKit 版本号根据实际情况调整）
```
tar xvf HPCKit_26.0.RC1_Linux-aarch64.tar.gz
```
### 安装 HPCKit
```
sh HPCKit_26.0.RC1_Linux-aarch64/install.sh -y --prefix=[HPCKit安装目录]
```

## 3. KuDNN编译安装

KuDNN依赖Bisheng/GCC编译器、KML BLAS和KUPL，需要预先加载，具体版本号根据HPCKit实际情况调整。这里以26.0.RC1版本HPCKit为例。

### 1. Bisheng编译KuDNN

#### 加载 module
``` shell
module use [HPCKit安装目录]/HPCKit/latest/modulefiles   #添加HPCkit到MODULEPATH环境变量
module avail                                            #查看HPCKit各组件名称
```
#### 加载编译器环境变量
- 若使用Bisheng（编译器版本号根据实际情况调整）
``` shell
module load bisheng/compiler5.1.0/bishengmodule #加载Bisheng编译器，版本号根据实际情况调整
```
#### 加载KML BLAS和KUPL环境变量
``` shell
module load bisheng/kml26.0.RC1/kblas/multi     #加载KML BLAS
module load bisheng/kupl26.0.RC1/release        #加载Kupl
```

#### 编译KuDNN
进入KuDNN根目录执行如下命令
``` shell
sh build/build_kudnn.sh --only_kudnn=on --build_type=Release --compiler=clang --install_path=[KuDNN安装目录]
# --only_kudnn=on 表示只编译kudnn，不编译测试程序
# --compiler=clang 表示指定Bisheng编译器
# --install_path 指定kudnn安装目录
```
### 2. GCC编译KuDNN

#### 加载 module
``` shell
module use [HPCKit安装目录]/HPCKit/latest/modulefiles   #添加HPCkit到MODULEPATH环境变量
module avail                                            #查看HPCKit各组件名称
```
#### 加载编译器环境变量
- 若使用GCC（编译器版本号根据实际情况调整）
``` shell
module load gcc/compiler12.3.1/gccmodule    #加载gcc编译器，版本号根据实际情况调整
```
#### 加载KML BLAS和KUPL环境变量
``` shell
module load gcc/kml26.0.RC1/kblas/multi     #加载KML BLAS
module load gcc/kupl26.0.RC1/release        #加载Kupl
```


#### 编译KuDNN
进入KuDNN根目录执行如下命令
``` shell
sh build/build_kudnn.sh --only_kudnn=on --build_type=Release --compiler=gcc --install_path=[KuDNN安装目录]
# --only_kudnn=on 表示只编译kudnn，不编译测试程序
# --compiler=gcc 表示指定gcc编译器
# --install_path 指定kudnn安装目录
```

## 4. 测试

### 1. 编译测试程序
KuDNN的gtest测试程序依赖googletest，进入KuDNN根目录执行脚本下载安装googletest。
``` shell
sh build/pre_env_dnn.sh
```

安装完成后会在opensource/install/googletest/lib64下生成gtest相关库
```
libgmock.a  libgmock_main.a  libgtest.a  libgtest_main.a
```

在kudnn编译完成的基础上，编译测试程序，测试程序只支持在Bisheng编译KuDNN的基础上编译。
``` shell
sh build/build_kudnn.sh --only_tests=on --compiler=clang --install_path=[KuDNN安装目录]
# --only_test=on 表示只编译测试程序
# --compiler=clang 测试程序只支持Bisheng编译器
# --install_path 指示kudnn安装目录位置
```
生成的测试程序kudnn_gtest位于out/test/dnn/llt/gtest。编译好测试程序后，可通过重新用GCC编译KuDNN来测试GCC版KuDNN。

### 2. 运行测试程序
进入out/test/dnn/llt/gtest目录，有测试程序kudnn_gtest，可用于测试指定用例，例如
``` shell
./kudnn_gtest --gtest_filter=LinearActivationTests.FWD_LINEAR_RELU_2D_FP32 #测试float32数据类型的linear 2D接口
```

得到结果
``` shell
Note: Google Test filter = LinearActivationTests.FWD_LINEAR_RELU_2D_FP32
[==========] Running 1 test from 1 test suite.
[----------] Global test environment set-up.
[----------] 1 test from LinearActivationTests
[ RUN      ] LinearActivationTests.FWD_LINEAR_RELU_2D_FP32
[       OK ] LinearActivationTests.FWD_LINEAR_RELU_2D_FP32 (84 ms)
[----------] 1 test from LinearActivationTests (84 ms total)

[----------] Global test environment tear-down
[==========] 1 test from 1 test suite ran. (85 ms total)
[  PASSED  ] 1 test.
```

查看全部测试用例
``` shell
./kudnn_gtest --gtest_list_tests
```
更多用法详见
``` shell
./kudnn_gtest --help
```

# 📖学习教程

## 算子说明
KuDNN目前支持的算子如表 KuDNN支持的算子如下所示。

|算子名称|算子说明|
|:---:|:---|
|GEMM & Linear|矩阵乘法算子|
|Linear+postops|在Linear算子计算后进行relu、silu、gelu、resext、resmul及residential等操作|
|Softmax|Softmax归一化算子|
|Conv2d/Conv3d|2d/3d卷积算子|
|Group Normalization|组归一化算子|
|Layer Normalization|层归一化算子|
|Embedding|映射|
|Root Mean Square Normalization|均方根归一化算子|
|Rotary Position Embedding|旋转位置编码|

学习教程见HPCkit开发指南的KuDNN章节。

[https://www.hikunpeng.com/document/detail/zh/kunpenghpcs/hpckit/devg/KunpengHPCKit_developer_158.html](https://www.hikunpeng.com/document/detail/zh/kunpenghpcs/hpckit/devg/KunpengHPCKit_developer_158.html)

# 🔍目录结构

``` python
kuDNN/
├── build/
│  ├── build_dnn.sh              # kudnn编译脚本
│  ├── pre_env_dnn_compile.sh    # 测试程序依赖库Googletest编译脚本
|  └── pre_env_dnn.sh            # 测试程序依赖库Googletest下载和编译脚本
├── src/dnn/                        
│  ├── include/                  # 头文件
│  └── src/                      # kudnn库源码
└── test/dnn/llt/gtest
   ├── cmake/                    # 测试程序cmake编译配置
   └── src/                      # 测试程序源码
```

# 🤝联系我们
本项目功能和文档正在持续更新和完善中，建议您关注最新版本。

- **问题反馈**：通过 [【Issues】](https://atomgit.com/kunpengcompute/kudnn/issues) 提交问题。
- **社区互动**：通过 [【鲲鹏论坛（HPC专区）】](https://www.hikunpeng.com/forum/forum-0187135482144798003-1.html) 参与交流。
- **技术专栏**：通过 [【鲲鹏社区】](https://www.hikunpeng.com/developer/techArticles) 获取技术文章，如系列化教程、优秀实践等。