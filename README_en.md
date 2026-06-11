# KuDNN

## 🔥Release Notes

- [2026/03] KuDNN went live, featuring support for Linear, Softmax, and Normal operators. The project utilizes KML BLAS and JIT-compiled kernels to deliver high-performance underlying computation.

## 🚀Overview

KuDNN is an operator library designed to optimize AI performance by leveraging Kunpeng CPU micro-architecture and advanced software optimization techniques. KuDNN can be integrated into the open-source oneDNN framework as a plugin, or deployed as a standalone backend to accelerate operators within TensorFlow and PyTorch.

## 📝Version Mapping

- Operating platform
    - Kunpeng 920 Pro
- System specifications
    - openEuler 20.03（LTS-SP3）AArch64
    - openEuler 22.03（LTS-SP2）AArch64
    - openEuler 22.03（LTS-SP3）AArch64
    - openEuler 22.03（LTS-SP4）AArch64
    - openEuler 24.03（LTS-SP3）AArch64

## ⚡️Build, Installation, and Testing

1. [Obtain the HPCKit software package](https://www.hikunpeng.com/document/detail/zh/kunpenghpcs/hpckit/instg/KunpengHPCKit_install_007.html).

    1. Download the HPCKit software package (replace the version number in the example with your actual version).
        ```
        wget https://mirrors.huaweicloud.com/kunpeng/archive/HPC/HPCKit/HPCKit_26.0.RC1_Linux-aarch64.tar.gz
        ```
2. [Install HPCKit](https://www.hikunpeng.com/document/detail/zh/kunpenghpcs/hpckit/instg/KunpengHPCKit_install_012.html).

   1. Extract the HPCKit software package (replace the version number in the example with your actual version).

        ```shell
        tar xvf HPCKit_26.0.RC1_Linux-aarch64.tar.gz
        ```

   2. Install HPCKit.

        ```shell
        sh HPCKit_26.0.RC1_Linux-aarch64/install.sh -y --prefix=[HPCKit_installation_directory]
        ```

3. Build and Install KuDNN.

    KuDNN depends on BiSheng Compiler, KML BLAS, and KUPL, all of which must be pre-loaded into the environment. The specific version numbers depend on the version of HPCKit in use. For this guide, HPCKit 26.0.RC1 is used as an example.

    - Using BiSheng Compiler
      
      Load `module`.

        ```shell
        module use [HPCKit_installation_directory]/HPCKit/latest/modulefiles            # Add HPCKit to the MODULEPATH environment variable.
        module avail                                            # View HPCKit components.
        ```
      Load the environment variable for the compiler.

        ```shell
        module load bisheng/compiler5.1.0/bishengmodule # Load BiSheng Compiler. The version number varies.
        ```
    
      Load environment variables for KML BLAS and KUPL.
    
        ```shell
        module load bisheng/kml26.0.RC1/kblas/multi     # Load KML BLAS.
        module load bisheng/kupl26.0.RC1/release        # Load KUPL.
        ```

      Compile KuDNN.

      1. Upload and extract the KuDNN source package (change the branch name as needed).

            ```shell
            unzip kudnn-main.zip
            ```

      2. Navigate to the root directory of the KuDNN source code (change the branch name as needed).

            ```shell
            cd kudnn-main
            ```

      3. Execute the following command:

            ```shell
            sh build/build_kudnn.sh --only_kudnn=on --build_type=Release --compiler=clang --install_path=[KuDNN_installation_directory]
            # --only_kudnn=on Compile KuDNN only, without building the test program.
            # --compiler=clang Specify BiSheng Compiler.
            # --install_path Specify the KuDNN installation directory.
            ```

    - Using GCC

      Load `module`.

        ```shell
        module use [HPCKit_installation_directory]/HPCKit/latest/modulefiles   # Add HPCKit to the MODULEPATH environment variable.
        module avail                                            # View HPCKit components.
        ```

      Load the environment variable for the compiler.

        ```shell
        module load gcc/compiler12.3.1/gccmodule    # Load GCC. The version number varies.
        ```

      Load environment variables for KML BLAS and KUPL.

        ```shell
        module load gcc/kml26.0.RC1/kblas/multi     # Load KML BLAS.
        module load gcc/kupl26.0.RC1/release        # Load KUPL.
        ```

      Compile KuDNN.

      1. Upload and extract the KuDNN source package (change the branch name as needed).

            ```shell
            unzip kudnn-main.zip
            ```

      2. Navigate to the root directory of the KuDNN source code (change the branch name as needed).

            ```shell
            cd kudnn-main
            ```

      3. Execute the following command:

            ```shell
            sh build/build_kudnn.sh --only_kudnn=on --build_type=Release --compiler=gcc --install_path=[KuDNN_installation_directory]
            # --only_kudnn=on Compile KuDNN only, without building the test program.
            # --compiler=gcc Specify GCC.
            # --install_path Specify the KuDNN installation directory.
            ```
    
4. Perform testing.

   1. Compile the test program.

        The gtest program of KuDNN depends on GoogleTest. Go to the KuDNN root directory and run the script to download and install GoogleTest.

        ```shell
        sh build/pre_env_dnn.sh
        ```

        If the `pre_env_kudnn.sh` script fails to download GoogleTest due to network issues, you can manually download [GoogleTest v1.15.2](https://github.com/google/googletest/tree/v1.15.2). If you cannot access GitHub, you can visit the domestic mirror [GoogleTest v1.15.2](https://gitcode.com/GitHub_Trending/go/googletest/tree/v1.15.2). If the extracted folder contains any version suffixes or extension names, rename it to `googletest` (without any suffixes) and upload it to the `opensource/` subdirectory within the KuDNN root path. Finally, execute the following command to skip the download phase and directly initiate build:

        ```shell
        sh build/pre_env_kudnn_compile.sh
        ```

        After the installation is complete, the gtest-related libraries are generated in `opensource/install/googletest/lib64`.

        ```shell
        libgmock.a  libgmock_main.a  libgtest.a  libgtest_main.a
        ```

        After building KuDNN, build the test program. Note that tests can only be built using the Clang-based BiSheng Compiler.

        ```shell
        sh build/build_kudnn.sh --only_tests=on --compiler=clang --install_path=[KuDNN_installation_directory]
        # --only_test=on: Build only the test program.
        # --compiler=clang: The test program supports only the Clang-based BiSheng Compiler.
        # --install_path: Specify the KuDNN installation directory.
        ```

        The generated test program `kudnn_gtest` is located in the `out/test/dnn/llt/gtest` directory.

        ##### *Tip: How do I test the GCC-compiled KuDNN using `kudnn_gtest`?*
        After compiling KuDNN and the test program with BiSheng Compiler, recompile KuDNN using GCC. This will automatically update the dynamic links of `kudnn_gtest` to point to the newly compiled GCC version of KuDNN, allowing you to proceed with the testing.

        ```shell
        sh build/build_kudnn.sh --only_kudnn=on --build_type=Release --compiler=clang --install_path=[KuDNN_installation_directory]
        ```

    2. Run the test program.
    
        Go to the `out/test/dnn/llt/gtest` directory. The test program kudnn_gtest is available and can be used to test specified cases. Example:

        ```shell
        ./kudnn_gtest --gtest_filter=LinearActivationTests.FWD_LINEAR_RELU_2D_FP32 # Test the float32 2D Linear API.
        ```

        Obtain the results.

        ```shell
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

        View all test cases.

        ```shell
        ./kudnn_gtest --gtest_list_tests
        ```

        For more usage, see the following:

        ```shell
        ./kudnn_gtest --help
        ```

## 📖Tutorials

### Operator Description

The following table lists the operators supported by KuDNN.

|Operator|Description|
|:---:|:---|
|GEMM & Linear|Matrix multiplication operator.|
|Linear+postops|Performs ReLU, SiLU, GeLU, ReSEXT, ResMul, and Residential operations following the Linear operator computation.|
|Softmax|Softmax normalization operator.|
|Conv2d/Conv3d|2D/3D convolution operator.|
|Group Normalization|Group normalization operator.|
|Layer Normalization|Layer normalization operator.|
|Embedding|Embedding.|
|Root Mean Square Normalization|Root mean square normalization operator.|
|Rotary Position Embedding|Rotary Positional Embedding (RoPE).|

For details about KuDNN API definitions, usage guidelines, and code examples, see the KuDNN section in the [HPCKit Developer Guide](https://www.hikunpeng.com/document/detail/zh/kunpenghpcs/hpckit/devg/KunpengHPCKit_developer_158.html).

## 🔍Directory Structure

``` python
kuDNN/
├── build/
│  ├── build_kudnn.sh              # KuDNN build script
│  ├── pre_env_kudnn_compile.sh    # Script for compiling the test program dependency GoogleTest
|  └── pre_env_kudnn.sh            # Script for downloading and compiling the test program dependency GoogleTest
├── src/dnn/
│  ├── include/                  # Header files
│  └── src/                      # Source code of the KUDNN library
└── test/dnn/llt/gtest
   ├── cmake/                    # CMake build configuration for the test program
   └── src/                      # Source code of the test program
```

## 🤝Contact Us

Features and documentation are updated regularly. Please follow the latest version for the most up-to-date information.

- **Issue feedback**: Submit queries or report bugs via [Issues](https://atomgit.com/kunpengcompute/kudnn/issues).
- **Community interaction**: Join discussions and share ideas via [Discussions](https://atomgit.com/kunpengcompute/kudnn/discussions).
- **Technical columns**: Access in-depth technical articles, serialized tutorials, and best practices through the [Kunpeng Community](https://www.hikunpeng.com/developer/techArticles).
