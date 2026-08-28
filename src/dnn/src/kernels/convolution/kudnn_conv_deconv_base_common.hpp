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

namespace KuDNN {
namespace Detail {

enum class ImgLayout {
    NN,
    NT,
    TN,
    TT
};

template <typename T, ImgLayout layout>
struct Img2col final { // wei - ncdhw, src - ncdhw
    static void Do(const T* dataIm, IntType channels,
        IntType depth, IntType height, IntType width,
        IntType outDepth, IntType outHeight, IntType outWidth,
        IntType kSizeD, IntType kSizeH, IntType kSizeW,
        IntType padD, IntType padH, IntType padW,
        IntType strideD, IntType strideH, IntType strideW,
        IntType dilateD, IntType dilateH, IntType dilateW,
        T* dataCol, SizeType ndims, int numThreads) noexcept
    {
        bool is2d = (ndims >= DIM_4);
        bool is3d = (ndims == DIM_5);

#pragma omp parallel for num_threads(numThreads) collapse(NUM_5)
        for (IntType c = 0; c < channels; ++c) {
            for (IntType kd = 0; kd < kSizeD; ++kd) {
                for (IntType kh = 0; kh < kSizeH; ++kh) {
                    for (IntType kw = 0; kw < kSizeW; ++kw) {
                        for (IntType od = 0; od < outDepth; ++od) {
                            for (IntType oh = 0; oh < outHeight; ++oh) {
                                for (IntType ow = 0; ow < outWidth; ++ow) {
                                    IntType id = is3d ? (od * strideD - padD + kd * dilateD) : 0;
                                    IntType ih = is2d ? (oh * strideH - padH + kh * dilateH) : 0;
                                    IntType iw = ow * strideW - padW + kw * dilateW;
                                    bool inside = (id >= 0) && (id < depth) && (ih >= 0) &&
                                        (ih < height) && (iw >= 0) && (iw < width);
                                    if (inside) {
                                        if constexpr (layout == ImgLayout::NN) {
                                            dataCol[(((((c * kSizeD + kd) * kSizeH + kh) * kSizeW + kw)
                                                * outDepth + od) * outHeight + oh) * outWidth + ow] =
                                                    dataIm[((c * depth + id) * height + ih) * width + iw];
                                        } else if constexpr (layout == ImgLayout::NT) {
                                            dataCol[(((((c * kSizeD + kd) * kSizeH + kh) * kSizeW + kw)
                                                * outDepth + od) * outHeight + oh) * outWidth + ow] =
                                                    dataIm[((id * height + ih) * width + iw) *
                                                    channels + c];
                                        } else if constexpr (layout == ImgLayout::TN) {
                                            dataCol[(((((kd * kSizeH + kh) * kSizeW + kw) * channels + c)
                                                * outDepth + od) * outHeight + oh) * outWidth + ow] =
                                                    dataIm[((c * depth + id) * height + ih) * width + iw];
                                        } else if constexpr (layout == ImgLayout::TT) {
                                            dataCol[(((((kd * kSizeH + kh) * kSizeW + kw) * channels + c)
                                                * outDepth + od) * outHeight + oh) * outWidth + ow] =
                                                    dataIm[((id * height + ih) * width + iw) *
                                                    channels + c];
                                        }
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
#pragma omp parallel for collapse(NUM_1)
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