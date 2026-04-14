/* 
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 *
 * KuDNN is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *        http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#ifndef KUDNN_CONVOLUTION_BASE_COMMON_HPP
#define KUDNN_CONVOLUTION_BASE_COMMON_HPP

#include <cstdint>

#include "types/kudnn_tensor_info.hpp"

#include "kudnn.hpp"
#include "vector_wrappers.hpp"
#include "threading_impl.hpp"
#include "parallel_for.h"

namespace KuDNN {
namespace Detail {

template <typename T>
struct Img2colNN final { // wei - ncdhw, src - ncdhw
    static void Do(const T* dataIm, IntType channels,
        IntType depth, IntType height, IntType width,
        IntType outDept, IntType outHeight, IntType outWidth,
        IntType ksizeD, IntType ksizeH, IntType ksizeW,
        IntType padD, IntType padH, IntType padW,
        IntType strideD, IntType strideH, IntType strideW,
        IntType dilateD, IntType dilateH, IntType dilateW,
        T* dataCol, SizeType ndims, int numThreads) noexcept
    {
        bool is2d = (ndims >= DIM_4);
        bool is3d = (ndims == DIM_5);

        Shape shape(channels, ksizeD, ksizeH, ksizeW, outDept);
        Threading::ParallelForND<NUM_5>::run(
            [&](int c, int kd, int kh, int kw, int od) {
                for (IntType oh = 0; oh < outHeight; ++oh) {
                    for (IntType ow = 0; ow < outWidth; ++ow) {
                        IntType id = is3d ? (od * strideD - padD + kd * dilateD) : 0;
                        IntType ih = is2d ? (oh * strideH - padH + kh * dilateH) : 0;
                        IntType iw = ow * strideW - padW + kw * dilateW;
                        if ((id >= 0) && (id < depth) && (ih >= 0) && (ih < height) && (iw >= 0) &&
                            (iw < width)) {
                            dataCol[(((((c * ksizeD + kd) * ksizeH + kh) * ksizeW + kw)
                                * outDept + od) * outHeight + oh) * outWidth + ow] =
                                    dataIm[((c * depth + id) * height + ih) * width + iw];
                        }
                    }
                }
            },
            shape,
            numThreads);
    }
};

template <typename T>
struct Img2colNT final { // wei - ncdhw, src - ndhwc
    static void Do(const T* dataIm, IntType channels,
        IntType depth, IntType height, IntType width,
        IntType outDept, IntType outHeight, IntType outWidth,
        IntType ksizeD, IntType ksizeH, IntType ksizeW,
        IntType padD, IntType padH, IntType padW,
        IntType strideD, IntType strideH, IntType strideW,
        IntType dilateD, IntType dilateH, IntType dilateW,
        T* dataCol, SizeType ndims, int numThreads) noexcept
    {
        bool is2d = (ndims >= DIM_4);
        bool is3d = (ndims == DIM_5);

        Shape shape(channels, ksizeD, ksizeH, ksizeW, outDept);
        Threading::ParallelForND<NUM_5>::run(
            [&](int c, int kd, int kh, int kw, int od) {
                for (IntType oh = 0; oh < outHeight; ++oh) {
                    for (IntType ow = 0; ow < outWidth; ++ow) {
                        IntType id = is3d ? (od * strideD - padD + kd * dilateD) : 0;
                        IntType ih = is2d ? (oh * strideH - padH + kh * dilateH) : 0;
                        IntType iw = ow * strideW - padW + kw * dilateW;
                        if ((id >= 0) && (id < depth) && (ih >= 0) && (ih < height) && (iw >= 0) &&
                            (iw < width)) {
                            dataCol[(((((c * ksizeD + kd) * ksizeH + kh) * ksizeW + kw)
                                * outDept + od) * outHeight + oh) * outWidth + ow] =
                                    dataIm[((id * height + ih) * width + iw) * channels + c];
                        }
                    }
                }
            },
            shape,
            numThreads);
    }
};

template <typename T>
struct Img2colTN final { // wei - ndhwc, src - ncdhw
    static void Do(const T* dataIm, IntType channels,
        IntType depth, IntType height, IntType width,
        IntType outDept, IntType outHeight, IntType outWidth,
        IntType ksizeD, IntType ksizeH, IntType ksizeW,
        IntType padD, IntType padH, IntType padW,
        IntType strideD, IntType strideH, IntType strideW,
        IntType dilateD, IntType dilateH, IntType dilateW,
        T* dataCol, SizeType ndims, int numThreads) noexcept
    {
        bool is2d = (ndims >= DIM_4);
        bool is3d = (ndims == DIM_5);

        Shape shape(channels, ksizeD, ksizeH, ksizeW, outDept);
        Threading::ParallelForND<NUM_5>::run(
            [&](int c, int kd, int kh, int kw, int od) {
                for (IntType oh = 0; oh < outHeight; ++oh) {
                    for (IntType ow = 0; ow < outWidth; ++ow) {
                        IntType id = is3d ? (od * strideD - padD + kd * dilateD) : 0;
                        IntType ih = is2d ? (oh * strideH - padH + kh * dilateH) : 0;
                        IntType iw = ow * strideW - padW + kw * dilateW;
                        if ((id >= 0) && (id < depth) && (ih >= 0) && (ih < height) && (iw >= 0) &&
                            (iw < width)) {
                            dataCol[(((((kd * ksizeH + kh) * ksizeW + kw) * channels + c)
                                * outDept + od) * outHeight + oh) * outWidth + ow] =
                                    dataIm[((c * depth + id) * height + ih) * width + iw];
                        }
                    }
                }
            },
            shape,
            numThreads);
    }
};

template <typename T>
struct Img2colTT final { // wei - ndhwc, src - ndhwc
    static void Do(const T* dataIm, IntType channels,
        IntType depth, IntType height, IntType width,
        IntType outDept, IntType outHeight, IntType outWidth,
        IntType ksizeD, IntType ksizeH, IntType ksizeW,
        IntType padD, IntType padH, IntType padW,
        IntType strideD, IntType strideH, IntType strideW,
        IntType dilateD, IntType dilateH, IntType dilateW,
        T* dataCol, SizeType ndims, int numThreads) noexcept
    {
        bool is2d = (ndims >= DIM_4);
        bool is3d = (ndims == DIM_5);

        Shape shape(channels, ksizeD, ksizeH, ksizeW, outDept);
        Threading::ParallelForND<NUM_5>::run(
            [&](int kd, int kh, int kw, int c, int od) {
                for (IntType oh = 0; oh < outHeight; ++oh) {
                    for (IntType ow = 0; ow < outWidth; ++ow) {
                        IntType id = is3d ? (od * strideD - padD + kd * dilateD) : 0;
                        IntType ih = is2d ? (oh * strideH - padH + kh * dilateH) : 0;
                        IntType iw = ow * strideW - padW + kw * dilateW;
                        if ((id >= 0) && (id < depth) && (ih >= 0) && (ih < height) && (iw >= 0) &&
                            (iw < width)) {
                            dataCol[(((((kd * ksizeH + kh) * ksizeW + kw) * channels + c)
                                * outDept + od) * outHeight + oh) * outWidth + ow] =
                                    dataIm[((id * height + ih) * width + iw) * channels + c];
                        }
                    }
                }
            }, shape, numThreads);
    }
};

template <typename T1, typename T2>
struct Col2img final {
    static void Do(const T1* dataCol, IntType channels,
        IntType depth, IntType height, IntType width,
        IntType outDept, IntType outHeight, IntType outWidth,
        IntType ksizeD, IntType ksizeH, IntType ksizeW,
        IntType padD, IntType padH, IntType padW,
        IntType strideD, IntType strideH, IntType strideW,
        IntType dilateD, IntType dilateH, IntType dilateW,
        T2* dataIm, SizeType ndims) noexcept
    {
        bool is2d = (ndims >= DIM_4);
        bool is3d = (ndims == DIM_5);

        IntType channelSize = depth * height * width;
        std::fill(dataIm, dataIm + channels * channelSize, Vector::Convert<float, T2>::Do(0.0f));
        for (IntType c = 0; c < channels; ++c) {
            for (IntType kd = 0; kd < ksizeD; ++kd) {
                for (IntType kh = 0; kh < ksizeH; ++kh) {
                    for (IntType kw = 0; kw < ksizeW; ++kw) {
                        for (IntType od = 0; od < outDept; ++od) {
                            for (IntType oh = 0; oh < outHeight; ++oh) {
                                for (IntType ow = 0; ow < outWidth; ++ow) {
                                    IntType id = is3d ? (od * strideD - padD + kd * dilateD) : 0;
                                    IntType ih = is2d ? (oh * strideH - padH + kh * dilateH) : 0;
                                    IntType iw = ow * strideW - padW + kw * dilateW;
                                    if ((id >= 0) && (id < depth) && (ih >= 0) && (ih < height) &&
                                        (iw >= 0) && (iw < width)) {
                                        dataIm[c * channelSize + id * height * width + ih * width + iw] =
                                            Vector::Convert<float, T2>::Do(Vector::Convert<T2, float>::Do(dataIm[
                                                c * channelSize + id * height * width + ih * width + iw]) +
                                                Vector::Convert<T1, float>::Do(dataCol[(((c * ksizeD + kd) *
                                                    ksizeH + kh) * ksizeW + kw) * outDept * outHeight * outWidth +
                                                    od * outHeight * outWidth + oh * outWidth + ow]));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
};

} // Detail
} // KuDNN

#endif // KUDNN_CONVOLUTION_BASE_COMMON_HPP