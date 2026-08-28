# KuDNN

## 🔥release-notes
- [2026/03] KuDNN首次上线，支持linear、softmax和normal等算子，使用KML BLAS高性能库和JIT动态代码生成kernel作为底层计算方法
- [2026/08] 新增bf16和fp16的JIT支持，优化GEMM和卷积性能；删除KML BLAS依赖

## 🚀概述
KuDNN鲲鹏深度神经网络算子库，结合鲲鹏CPU微架构特性与软优化手段，优化AI算子性能。KuDNN可以通过算子库插件形式集成进开源软件oneDNN框架，也可以单独作为后端加速算子集成到TensorFlow或者Pytorch。


## 📝版本配套

- 运行平台
    - 鲲鹏 920 专业版
- 系统规格
    - openEuler 20.03（LTS-SP3）AArch64
    - openEuler 22.03（LTS-SP2）AArch64
    - openEuler 22.03（LTS-SP3）AArch64
    - openEuler 22.03（LTS-SP4）AArch64
    - openEuler 24.03（LTS-SP3）AArch64


## ⚡️编译安装与测试

### 1. [获取 HPCKit 软件包](https://www.hikunpeng.com/document/detail/zh/kunpenghpcs/hpckit/instg/KunpengHPCKit_install_007.html)
#### 下载 HPCKit 软件包（HPCKit 版本号根据实际情况调整）
```
wget https://mirrors.huaweicloud.com/kunpeng/archive/HPC/HPCKit/HPCKit_26.0.RC1_Linux-aarch64.tar.gz
```
### 2. [安装 HPCKit](https://www.hikunpeng.com/document/detail/zh/kunpenghpcs/hpckit/instg/KunpengHPCKit_install_012.html)
#### 解压 HPCKit 软件包（HPCKit 版本号根据实际情况调整）
```
tar xvf HPCKit_26.0.RC1_Linux-aarch64.tar.gz
```
#### 安装 HPCKit
```
sh HPCKit_26.0.RC1_Linux-aarch64/install.sh -y --prefix=[HPCKit安装目录]
```

### 3. KuDNN编译安装

KuDNN依赖Bisheng/GCC编译器，需要预先加载，具体版本号根据HPCKit实际情况调整。这里以26.0.RC1版本HPCKit为例。

#### (1) Bisheng编译KuDNN

##### 加载 module
``` shell
module use [HPCKit安装目录]/HPCKit/latest/modulefiles   #添加HPCkit到MODULEPATH环境变量
module avail                                            #查看HPCKit各组件名称
```
##### 加载编译器环境变量
``` shell
module load bisheng/compiler5.1.0/bishengmodule #加载Bisheng编译器，版本号根据实际情况调整
```

##### 编译KuDNN
上传并解压 KuDNN 项目源码包（分支名根据实际情况调整）
```
unzip kudnn-main.zip
```
进入KuDNN项目源码根目录（分支名根据实际情况调整）
```
cd kudnn-main
```
执行如下命令
``` shell
sh build/build_kudnn.sh --only_kudnn=on --build_type=Release --compiler=clang --install_path=[KuDNN安装目录]
# --only_kudnn=on 表示只编译kudnn，不编译测试程序
# --compiler=clang 表示指定Bisheng编译器
# --install_path 指定kudnn安装目录
```
#### (2) GCC编译KuDNN

##### 加载 module
``` shell
module use [HPCKit安装目录]/HPCKit/latest/modulefiles   #添加HPCkit到MODULEPATH环境变量
module avail                                            #查看HPCKit各组件名称
```
##### 加载编译器环境变量
``` shell
module load gcc/compiler12.3.1/gccmodule    #加载gcc编译器，版本号根据实际情况调整
```

##### 编译KuDNN

上传并解压 KuDNN 项目源码包（分支名根据实际情况调整）
```
unzip kudnn-main.zip
```
进入KuDNN项目源码根目录（分支名根据实际情况调整）
```
cd kudnn-main
```
执行如下命令
``` shell
sh build/build_kudnn.sh --only_kudnn=on --build_type=Release --compiler=gcc --install_path=[KuDNN安装目录]
# --only_kudnn=on 表示只编译kudnn，不编译测试程序
# --compiler=gcc 表示指定gcc编译器
# --install_path 指定kudnn安装目录
```

### 4. 测试

#### (1) 编译测试程序

在kudnn编译完成的基础上，编译测试程序。gtest测试程序只支持Bisheng编译器，因为bf16类型的基础符号运算。
``` shell
sh build/build_kudnn.sh --only_tests=on --compiler=clang --install_path=[KuDNN安装目录]
# --only_test=on 表示只编译测试程序
# --compiler=clang 测试程序只支持Bisheng编译器
# --install_path 指示kudnn安装目录位置
```
生成的gtest测试程序kudnn_gtest位于out/test/dnn/llt/gtest。

性能测试程序benchdnn位于out/test/dnn/oneDNN-3.4/build/tests/benchdnn；
如果指定--oneDNN_ver=3.7，则benchdnn位于out/test/dnn/oneDNN-3.7/build/tests/benchdnn。


#### (2) 运行测试程序

##### gtest测试
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

##### benchdnn性能测试

```bash
OMP_NUM_THREADS=8 ./benchdnn --mode=P --matmul --dt=s8:s8:s32 --stag=ab --wtag=ba --dtag=ba 128x1536:1536x1536

# OMP_NUM_THREADS环境变量表示使用的线程数
# --mode=P 表示只进行性能测试
# --dt=s8:s8:s3表示数据类型，int8*int8=int32
# --stag和--wtag 表示表示src和wei的数据布局
# 128x1536:1536x1536 表示src和dst矩阵大小
```
更多benchdnn用法参考[benchdnn](https://github.com/uxlfoundation/oneDNN/blob/main/tests/benchdnn/README.md)

## 📖学习教程

### 算子说明
KuDNN目前支持的算子如表 KuDNN支持的算子如下所示。

|算子名称|算子说明|
|:---:|:---|
|[GEMM & Linear](https://www.hikunpeng.com/document/detail/zh/kunpenghpcs/hpckit/devg/KunpengHPCKit_developer_163.html)|矩阵乘法算子|
|[Linear+postops](https://www.hikunpeng.com/document/detail/zh/kunpenghpcs/hpckit/devg/KunpengHPCKit_developer_166.html)|在Linear算子计算后进行relu、silu、gelu、resext、resmul及residential等操作|
|[Softmax](https://www.hikunpeng.com/document/detail/zh/kunpenghpcs/hpckit/devg/KunpengHPCKit_developer_172.html)|Softmax归一化算子|
|[Conv2d/Conv3d](https://www.hikunpeng.com/document/detail/zh/kunpenghpcs/hpckit/devg/KunpengHPCKit_developer_175.html)|2d/3d卷积算子|
|[Group Normalization](https://www.hikunpeng.com/document/detail/zh/kunpenghpcs/hpckit/devg/KunpengHPCKit_developer_178.html)|组归一化算子|
|[Layer Normalization](https://www.hikunpeng.com/document/detail/zh/kunpenghpcs/hpckit/devg/KunpengHPCKit_developer_181.html)|层归一化算子|
|[Embedding](https://www.hikunpeng.com/document/detail/zh/kunpenghpcs/hpckit/devg/KunpengHPCKit_developer_184.html)|映射|
|[Root Mean Square Normalization](https://www.hikunpeng.com/document/detail/zh/kunpenghpcs/hpckit/devg/KunpengHPCKit_developer_187.html)|均方根归一化算子|
|[Rotary Position Embedding](https://www.hikunpeng.com/document/detail/zh/kunpenghpcs/hpckit/devg/KunpengHPCKit_developer_190.html)|旋转位置编码|

更多学习教程见KuDNN开发指南。

[KuDNN开发指南](https://www.hikunpeng.com/document/detail/zh/kunpenghpcs/hpckit/devg/KunpengHPCKit_developer_158.html): 详细介绍了KuDNN的接口定义、使用说明和代码样例。

## 🔍目录结构

``` python
kuDNN/
├── build/
|  └── build_kudnn.sh            # kudnn编译脚本
├── src/dnn/
│  ├── include/                  # 头文件
│  └── src/                      # kudnn库源码
└── test/dnn/
   ├── llt/gtest                 # GoogleTets测试程序
   ├── oneDNN-3.4                # 适配了KuDNN的oneDNN-3.4
   └── oneDNN-3.7                # 适配了KuDNN的oneDNN-3.7
```

## 🤝联系我们
本项目功能和文档正在持续更新和完善中，建议您关注最新版本。

- **问题反馈**：通过 [【Issues】](https://atomgit.com/kunpengcompute/kudnn/issues) 提交问题。
- **社区互动**：通过 [【鲲鹏论坛（HPC专区）】](https://www.hikunpeng.com/forum/forum-0187135482144798003-1.html) 参与交流。
- **技术专栏**：通过 [【鲲鹏社区】](https://www.hikunpeng.com/developer/techArticles) 获取技术文章，如系列化教程、优秀实践等。
