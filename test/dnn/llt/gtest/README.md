## Dependencies

### Compiler

gcc 10.3.1


### Google Test

https://github.com/google/googletest (2024 July 13 version)

```git
git clone https://github.com/google/googletest.git
```
### build gcc version lib
```bash
cd googletest
mkdir -p build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=your_path_to_install_lib
make -j
make install
```

### build clang version lib
```bash
cd googletest
mkdir -p build
cd build
export PATH=your_path/BiShengCompiler-3.2.0-aarch64-linux/bin:$PATH
export CC=clang
export CXX=clang++
cmake .. -DCMAKE_INSTALL_PREFIX=your_path_to_install_lib
make -j
make install
```
