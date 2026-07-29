#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#------------------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# Description: Prepare pack environment for dnn
# Author: DNN
# Create: 2025-03-10
#
# Modified: 2025-07-29 — merge per-(exec,sw) files into per-(layout,kernel)
#   so that <arm_sve.h> and project headers are parsed once per group instead
#   of once per (exec_size, sw_size) pair. ~2000 TU → ~52 TU.  Product unchanged.
#-----------------------------------------------------------------------------------

import os
import shutil
import sys


def num_used_regs(es_num, sw_num, kernel_num, precision_type):
    return int(((es_num - 1) * sw_num + kernel_num + \
        (num_elems_in_128_bits.get(precision_type) - 1)) / (num_elems_in_128_bits.get(precision_type)))

precision_list = ["f32", "f16"]
precision_c_name = {"f32": "float", "f16": "__fp16"}
num_elems_in_128_bits = {"f32": 4, "f16": 8}

exec_size_map = {"f32": [1, 2, 3, 4, 5, 6, 7, 8], "f16": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]}
kernel_size_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
sw_size_list = [1, 2, 3, 4, 5]
layout_list = ["NCHW", "NHWC"]

LICENSE_NOTICE = ("/*\n * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.\n"
                    " * Description: Convolution kernel impl. Produced by convcodegen.py\n"
                    " * Author: KPL\n * Create: 2024-08-08\n * Notes: NA\n */\n\n")

INCLUDE_FILES = "#include \"vector_wrappers.hpp\"\n#include \"kernels/convolution/kudnn_conv_kernels.hpp\"\n\n"

OPENING_NAMESPACE = "namespace KuDNN {\nnamespace Detail {\n\n"
CLOSING_NAMESPACE = "\n} // Detail\n} // KuDNN\n"

path = os.path.join(os.getcwd(), "generated")

TABULATION = "    "

OPENING_BRACKET = "{\n"
CLOSING_BRACKET = "}\n"

precision = sys.argv[1]
current_path = os.path.join(path, precision)
if os.path.exists(current_path):
    shutil.rmtree(current_path)
os.makedirs(current_path)

# Build the merged file list for conv_gen.cmake
merged_files = []

for layout in layout_list:
    for kernel_size in kernel_size_list:
        filename = "kdnn_conv_kernel_2d_{}_{:d}_{:s}.cpp".format(
                    layout, kernel_size, precision)
        relative_path = "generated/{}/{}".format(precision, filename)
        merged_files.append(relative_path)
        full_path = os.path.join(current_path, filename)

        with open(full_path, "w") as f:
            # Write preamble once per merged file
            f.write(LICENSE_NOTICE)
            f.write(INCLUDE_FILES)
            f.write(OPENING_NAMESPACE)

            for exec_size in exec_size_map.get(precision):
                for sw_size in sw_size_list:
                    func_name = (("template<>\n"
                                "template<>\n"
                                "void ConvKernel<{}>::Do<Layout::{}, NUM_{}, NUM_{}, NUM_{}>(\n"
                                "{}const ConvKernel<{}>::CompType *src, const ConvKernel<{}>::CompType *wei,\n"
                                "{}ConvKernel<{}>::CompType *dst, IntType dstStride, "
                                "IntType dstIdx, IntType icBlock) noexcept\n").format(
                                    precision_c_name.get(precision), layout, kernel_size,
                                    exec_size, sw_size, TABULATION, precision_c_name.get(precision),
                                    precision_c_name.get(precision), TABULATION,
                                    precision_c_name.get(precision)))
                    function_body = ("{}svbool_t pgTrue = svptrue_b16();\n"
                                    "{}svbool_t pg0;\n"
                                    "{}Vector::WhileLt<CompType>::Do(pg0, NUM_0, dstIdx);\n"
                                    "{}svbool_t pg1;\n"
                                    "{}Vector::WhileLt<CompType>::Do(pg1, Vector::Length<CompType>(), dstIdx);\n"
                                    "{}svbool_t pg2;\n"
                                    "{}Vector::WhileLt<CompType>::Do(pg2, "
                                    "NUM_2 * Vector::Length<CompType>(), dstIdx);\n\n").format(
                                    TABULATION, TABULATION, TABULATION, TABULATION, TABULATION, TABULATION, TABULATION)
                    function_body += "{}typename Vector::Vec<CompType>::Type dst00, dst01, dst02".format(TABULATION)
                    for es in range(1, exec_size, 1):
                        if es % 3 == 0:
                            function_body += ",\n{}{}".format(TABULATION, TABULATION)
                        else:
                            function_body += ", "
                        function_body += f"dst{es}0, dst{es}1, dst{es}2"
                    function_body += ";\n\n"
                    if (layout == "NCHW"):
                        for es in range(0, exec_size, 1):
                            # nchw
                            function_body += f"{TABULATION}Vector::GatherLoad<CompType>::Do(dst + NUM_{es} + NUM_0 * Vector::Length<CompType>() * dstStride, pg0, dstStride, dst{es}0);\n"
                            function_body += f"{TABULATION}Vector::GatherLoad<CompType>::Do(dst + NUM_{es} + NUM_1 * Vector::Length<CompType>() * dstStride, pg1, dstStride, dst{es}1);\n"
                            function_body += f"{TABULATION}Vector::GatherLoad<CompType>::Do(dst + NUM_{es} + NUM_2 * Vector::Length<CompType>() * dstStride, pg2, dstStride, dst{es}2);\n\n"
                    else:
                        for es in range(0, exec_size, 1):
                            # nhwc
                            function_body += f"{TABULATION}dst{es}0 = svld1(pg0, dst + NUM_{es} * dstStride + NUM_0 * Vector::Length<CompType>());\n"
                            function_body += f"{TABULATION}dst{es}1 = svld1(pg1, dst + NUM_{es} * dstStride + NUM_1 * Vector::Length<CompType>());\n"
                            function_body += f"{TABULATION}dst{es}2 = svld1(pg2, dst + NUM_{es} * dstStride + NUM_2 * Vector::Length<CompType>());\n\n"

                    function_body += ("{}for (IntType ic = 0; ic < icBlock; ++ic) {{\n"
                                    "{}{}for (IntType kh = 0; kh < NUM_{}; ++kh) {{\n"
                                    "{}{}{}typename Vector::Vec<CompType>::Type wei0, wei1, wei2;\n".format(TABULATION, TABULATION, TABULATION, \
                                                                                                            kernel_size, TABULATION, TABULATION, TABULATION))
                    function_body += ("{}{}{}typename Vector::Vec<CompType>::Type src0 = svld1rq(pgTrue, "
                                        "src + NUM_{} * (kh + NUM_{} * ic));\n").format(TABULATION, TABULATION, TABULATION, \
                                            num_used_regs(exec_size, sw_size, kernel_size, precision) * num_elems_in_128_bits.get(precision), \
                                                kernel_size, kernel_size)

                    loaded_srcs = [False] * num_used_regs(exec_size, sw_size, kernel_size, precision)
                    loaded_srcs[0] = True
                    for kw in range(0, kernel_size, 1):
                        function_body += "\n{}{}{}// unroll iteration {}\n".format(TABULATION, TABULATION, TABULATION, kw)
                        function_body += ("{}{}{}wei0 = svld1(pg0, wei + NUM_0 * Vector::Length<CompType>() +\n"
                                            "{}{}{}{}OcSize() * (NUM_{} + NUM_{} * (kh + NUM_{} * ic)));\n".format(TABULATION, TABULATION, TABULATION, TABULATION,
                                                                                                            TABULATION, TABULATION, TABULATION, kw, kernel_size, kernel_size))

                        function_body += ("{}{}{}wei1 = svld1(pg1, wei + NUM_1 * Vector::Length<CompType>() +\n"
                                        "{}{}{}{}OcSize() * (NUM_{} + NUM_{} * (kh + NUM_{} * ic)));\n".format(TABULATION, TABULATION, TABULATION, TABULATION,
                                                                                                    TABULATION, TABULATION, TABULATION, kw, kernel_size, kernel_size))
                        function_body += ("{}{}{}wei2 = svld1(pg2, wei + NUM_2 * Vector::Length<CompType>() +\n"
                                        "{}{}{}{}OcSize() * (NUM_{} + NUM_{} * (kh + NUM_{} * ic)));\n".format(TABULATION, TABULATION, TABULATION, TABULATION,
                                                                                                    TABULATION, TABULATION, TABULATION, kw, kernel_size, kernel_size))

                        for es in range(0, exec_size, 1):
                            src_idx = es * sw_size + kw
                            if (not loaded_srcs[src_idx // num_elems_in_128_bits.get(precision)]):
                                function_body += ("\n{}typename Vector::Vec<CompType>::Type src{} =\n{}"
                                                    " svld1rq(pgTrue, src + NUM_{} + NUM_{} * (kh + NUM_{} * ic));\n").format(TABULATION * 3,
                                                        int(src_idx / num_elems_in_128_bits.get(precision)), TABULATION * 4,
                                                        int(src_idx / num_elems_in_128_bits.get(precision)) * num_elems_in_128_bits.get(precision),
                                                        num_used_regs(exec_size, sw_size, kernel_size, precision) * num_elems_in_128_bits.get(precision),
                                                        kernel_size)
                                loaded_srcs[src_idx // num_elems_in_128_bits.get(precision)] = True
                            function_body += "\n{}dst{}0 = svmla_lane(dst{}0, wei0, src{}, NUM_{});".format(TABULATION * 3, \
                                                es, es, int(src_idx / num_elems_in_128_bits.get(precision)), src_idx % num_elems_in_128_bits.get(precision))
                            function_body += "\n{}dst{}1 = svmla_lane(dst{}1, wei1, src{}, NUM_{});".format(TABULATION * 3, \
                                                es, es, int(src_idx / num_elems_in_128_bits.get(precision)), src_idx % num_elems_in_128_bits.get(precision))
                            function_body += "\n{}dst{}2 = svmla_lane(dst{}2, wei2, src{}, NUM_{});\n".format(TABULATION * 3, \
                                                es, es, int(src_idx / num_elems_in_128_bits.get(precision)), src_idx % num_elems_in_128_bits.get(precision))

                    function_body += "{}{}{}{}{}".format(TABULATION, TABULATION, CLOSING_BRACKET, TABULATION, CLOSING_BRACKET)

                    if layout == "NCHW":
                        for es in range(0, exec_size, 1):
                            # nchw
                            function_body += f"\n{TABULATION}Vector::ScatterStore<CompType>::Do(pg0, dstStride, dst{es}0, dst + NUM_{es} + NUM_0 * Vector::Length<CompType>() * dstStride);\n"
                            function_body += f"\n{TABULATION}Vector::ScatterStore<CompType>::Do(pg1, dstStride, dst{es}1, dst + NUM_{es} + NUM_1 * Vector::Length<CompType>() * dstStride);\n"
                            function_body += f"\n{TABULATION}Vector::ScatterStore<CompType>::Do(pg2, dstStride, dst{es}2, dst + NUM_{es} + NUM_2 * Vector::Length<CompType>() * dstStride);\n\n"
                    else:
                        for es in range(0, exec_size, 1):
                            # nhwc
                            function_body += f"\n{TABULATION}svst1(pg0, dst + NUM_{es} * dstStride + NUM_0 * Vector::Length<CompType>(), dst{es}0);"
                            function_body += f"\n{TABULATION}svst1(pg1, dst + NUM_{es} * dstStride + NUM_1 * Vector::Length<CompType>(), dst{es}1);"
                            function_body += f"\n{TABULATION}svst1(pg2, dst + NUM_{es} * dstStride + NUM_2 * Vector::Length<CompType>(), dst{es}2);\n"

                    function_content = "{}{}{}".format(func_name, OPENING_BRACKET, function_body)
                    function_content += "{}\n".format(CLOSING_BRACKET)
                    f.write(function_content)

            # Close namespace once per merged file
            f.write(CLOSING_NAMESPACE)

# Write conv_gen.cmake with the merged file list
# NOTE: conv_gen.cmake must be checked into the source tree because cmake
# reads it during the configure step (include()), before the custom commands
# that run this script. The Python script only generates the .cpp files;
# conv_gen.cmake is maintained separately and committed to git.
def write_conv_gen_cmake():
    cmake_dir = os.path.dirname(os.path.abspath(__file__))
    cmake_file = os.path.join(cmake_dir, "conv_gen.cmake")
    with open(cmake_file, "w") as cf:
        cf.write("# Auto-generated by convcodegen.py. DO NOT EDIT.\n")
        if precision == "f32":
            cf.write("set(CONVOLUTION_KERNELS_F32\n")
        else:
            cf.write("set(CONVOLUTION_KERNELS_F16\n")
        for mf in merged_files:
            cf.write("    {}\n".format(mf))
        cf.write(")\n")
