import multiprocessing
import os
import platform
import subprocess
import sys
import torch
from setuptools import find_packages
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            _ = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                ", ".join(ext.name for ext in self.extensions))

        self.pytorch_dir = os.path.dirname(torch.utils.cmake_prefix_path)
        self.project_dir = os.path.dirname(os.path.abspath(__file__))
        self.build_type = os.getenv("HIKPEX_BUILD_TYPE", "release")
        self.target_platform = os.getenv("HIKPEX_TARGET_PLATFORM", "KP920F")
        self.enable_gcov = os.getenv("HIKPEX_ENABLE_GCOV", "off")
        self.enable_asan = os.getenv("HIKPEX_ENABLE_ASAN", "off")
        self.dependencies_dir = os.getenv("HIKPEX_DEPENDENCIES_DIR", self.project_dir + "/lib")
        self.output_dir = os.getenv("HIKPEX_OUTPUT_DIR", ".")

        for ext in self.extensions:
            self.build_cmake(ext)

    def build_cmake(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ["-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
                    "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={}".format(extdir),
                    "-DPROJECT_OUTPUT_DIR={}".format(self.output_dir),
                    "-DPROJECT_SOURCE_DIR={}".format(self.project_dir),
                    "-DKPL_BLAS_DIR={}/KPL_BLAS".format(self.project_dir),
                    "-DCMAKE_PREFIX_PATH={}/cmake/Torch".format(self.pytorch_dir),
                    "-DCMAKE_BUILD_TYPE={}".format(self.build_type),
                    "-DENABLE_GCOV={}".format(self.enable_gcov),
                    "-DENABLE_ASAN={}".format(self.enable_asan),
                    "-DTARGET_PLATFORM={}".format(self.target_platform),
                    "-DDEPENDENCIES_DIR={}".format(self.dependencies_dir)]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        cwd = os.getcwd()
        os.chdir(os.path.dirname(extdir))
        self.spawn(["cmake", ext.sourcedir] + cmake_args)
        if not self.dry_run:
            self.spawn(["cmake", "--build", ".", "--", "-j{}".format(multiprocessing.cpu_count())])
        os.chdir(cwd)


setup(
    name="hikp-pytorch-extension",
    version="0.0.2",
    author="Huawei Technologies Co., Ltd.",
    package_dir={"": "src"},
    packages=find_packages("src"),
    ext_modules=[CMakeExtension("hikp_pytorch_extension._C")],
    cmdclass={"build_ext": CMakeBuild, }
)