# pytorch通过oneDNN对接KuDNN

## 依赖准备
为保持各组件编译器一致，全部采用HPCKit的gcc-12.3.1
```bash
module use /home/local/HPCKit/latest/modulefiles   # 加载HPCKit
module load gcc/compiler12.3.1/gccmodule           # 加载GCC 12.3.1
```
### pytorch
获取PyTorch源码，这里采用v2.5.1版本
```bash
git clone -b v2.5.1--depth=1 --recursive https://github.com/pytorch/pytorch.git
cd /path/to/pytorch
```
获取子仓依赖代码。
```bash
git submodule sync
git submodule update --init --recursive
```
安装Python依赖
```bash
pip install -r requirements.txt
```

### KuDNN
#### 下载KuDNN源码
git clone https://gitcode.com/kunpengcompute/kudnn.git
#### 编译KuDNN
```bash
sh build/build.sh --target_platform=KP920F --only_kudnn=on                  # 构建KuDNN
```
更多编译KuDNN相关文档见README.md

得到的kudnn位于out/目录下, 编译pytorch需要这里的lib和include，可以将其拷贝到任意需要的位置用于编译pytorch。
```bash
build  include  lib
```

### oneDNN
pytorch-2.5.1要求配套oneDNN版本大于3.5，所以用KuDNN提供的oneDNN-3.7

使用KuDNN里提供的oneDNN，位于KuDNN/test/dnn/oneDNN-3.7。

将oneDNN-3.7拷贝到pytorch/third_party/ideep/下



### ACL
#### 下载ACL
oneDNN-3.7支持的最低版本为24.11，可下载24.12版本的ACL： 
https://github.com/ARM-software/ComputeLibrary/archive/refs/tags/v24.12.zip

#### 编译ACL
```bash
cd ComputeLibrary-24.12 
pip install scons       # 安装构建工具scons
scons Werror=1 -j128 debug=0 neon=1 opencl=0 os=linux arch=armv8a build=native
```
编译成功后在build下可以看到libarm_compute_graph.so和libarm_compute.so
```bash
examples  libarm_compute_graph.so  libarm_compute_graph-static.a  libarm_compute.so  libarm_compute-static.a  src  tests  utils
```

## 构建pytorch+oneDNN+KuDNN

### 修改cmake配置
用KuDNN提供的oneDNN替换pytorch自带的mkl-dnn。
编辑pytorch的cmake/Modules/FindMKLDNN.cmake文件，修改如下内容
```cmake
将
SET(MKLDNN_ROOT "${PROJECT_SOURCE_DIR}/third_party/ideep/mkl-dnn")
改为
SET(MKLDNN_ROOT "${PROJECT_SOURCE_DIR}/third_party/ideep/oneDNN-3.7")

将
SET(LLGA_ROOT "${PROJECT_SOURCE_DIR}/third_party/ideep/mkl-dnn")
改为
SET(LLGA_ROOT "${PROJECT_SOURCE_DIR}/third_party/ideep/oneDNN-3.7")

新增一行
SET(DNNL_AARCH64_USE_KUDNN TRUE CACHE BOOL "" FORCE)
```

### 编译
```bash
KUDNN_ROOT_DIR=/path/to/KuDNN_SDK/out \        # 指定 kudnn的lib和include所在路径
_GLIBCXX_USE_CXX11_ABI=1 SVE_VECBITS=512 DEBUG=0 \                  # 设置C++ ABI兼容性
USE_MKLDNN=1 USE_MKLDNN_ACL=1 \                                     # 启用MKLDNN
ACL_ROOT_DIR=/path/to/ComputeLibrary-24.12 \     # 指定MKLDNN的依赖ACL 所在路径
USE_CUDA=0 BUILD_TEST=0 USE_FBGEMM=0 USE_NNPACK=0 USE_QNNPACK=0 \   # 关闭不用的功能
USE_XNNPACK=0 USE_LAPACK=0 USE_GLOO=0 \                             # 关闭不用的功能
python ./setup.py develop                                           # 构建pytorch
```
### 检查是否成功

查看pytorch是否启用mkddnn(也叫oneDNN)
```python
>>> import torch
>>> print(torch.backends.mkldnn.is_available()) 
True
```

查看oneDNN是否启用KuDNN方法：使用ldd命令查看libtorch_cpu.so的依赖库，是否有libkudnn.so，如果有说明对接KuDNN成功