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

#include <cstdlib>
#include <iostream>
#include <cmath>
#include <ctime>
#include <random>
#include <map>
#include <string>

#include <arm_neon.h>

#include "gtest/gtest.h"
#include "kudnn.hpp"
#include "kudnn_tensor_data.hpp"

using Shape = KuDNN::Shape;
using SizeType = KuDNN::SizeType;


template <typename srcT, typename dstT>
static bool reorderFunc(const KuDNN::TensorInfo &srcTensor, const KuDNN::TensorInfo &dstTensor)
{
    srcT *src = (srcT *)malloc(sizeof(srcT) * srcTensor.GetTotalTensorSize());
    dstT *dst = (dstT *)malloc(sizeof(dstT) * dstTensor.GetTotalTensorSize());
    if (src == nullptr || dst == nullptr) {
        return false;
    }

    KuDNN::ReorderLayer reorderLayer(srcTensor, dstTensor);

    reorderLayer.Run(src, dst);

    return true;
}

TEST(ReorderGTests, F32ToF16_1)
{
    const KuDNN::TensorInfo srcTensor = {{12, 22, 12, 14}, KuDNN::Element::TypeT::F32, KuDNN::Layout::BCDA};
    const KuDNN::TensorInfo dstTensor = {{12, 22, 12, 14}, KuDNN::Element::TypeT::F16, KuDNN::Layout::CDAB};

    reorderFunc<float, __fp16>(srcTensor, dstTensor);
}

TEST(ReorderGTests, S32ToF16_1)
{
    const KuDNN::TensorInfo srcTensor = {{12, 22, 12, 144}, KuDNN::Element::TypeT::S32, KuDNN::Layout::DCAB};
    const KuDNN::TensorInfo dstTensor = {{12, 22, 12, 144}, KuDNN::Element::TypeT::F16, KuDNN::Layout::BACD};

    reorderFunc<int32_t, __fp16>(srcTensor, dstTensor);
}

TEST(ReorderGTests, S32ToBF16_1)
{
    const KuDNN::TensorInfo srcTensor = {{10, 12, 14, 81, 6}, KuDNN::Element::TypeT::S32, KuDNN::Layout::ADECB};
    const KuDNN::TensorInfo dstTensor = {{10, 12, 14, 81, 6}, KuDNN::Element::TypeT::F16, KuDNN::Layout::BACDE};

    reorderFunc<int32_t, __bf16>(srcTensor, dstTensor);
}

TEST(ReorderGTests, S32ToS8_1)
{
    const KuDNN::TensorInfo srcTensor = {{10, 12, 14}, KuDNN::Element::TypeT::S32, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo dstTensor = {{10, 12, 14}, KuDNN::Element::TypeT::S8, KuDNN::Layout::ABC};

    reorderFunc<int32_t, int8_t>(srcTensor, dstTensor);
}

TEST(ReorderGTests, S32ToU8_1)
{
    const KuDNN::TensorInfo srcTensor = {{10, 12, 14}, KuDNN::Element::TypeT::S32, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo dstTensor = {{10, 12, 14}, KuDNN::Element::TypeT::U8, KuDNN::Layout::ABC};

    reorderFunc<int32_t, uint8_t>(srcTensor, dstTensor);
}

TEST(ReorderGTests, S32ToF32_1)
{
    const KuDNN::TensorInfo srcTensor = {{10, 12, 14}, KuDNN::Element::TypeT::S32, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo dstTensor = {{10, 12, 14}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};

    reorderFunc<int32_t, float>(srcTensor, dstTensor);
}

TEST(ReorderGTests, S32ToS32_1)
{
    const KuDNN::TensorInfo srcTensor = {{10, 12, 14, 16}, KuDNN::Element::TypeT::S32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo dstTensor = {{10, 12, 14, 16}, KuDNN::Element::TypeT::S32, KuDNN::Layout::ACBD};

    reorderFunc<int32_t, int32_t>(srcTensor, dstTensor);
}

TEST(ReorderGTests, S32ToS32_2)
{
    const KuDNN::TensorInfo srcTensor = {{10, 12, 25, 14, 16}, KuDNN::Element::TypeT::S32, KuDNN::Layout::ACBDE};
    const KuDNN::TensorInfo dstTensor = {{10, 12, 25, 14, 16}, KuDNN::Element::TypeT::S32, KuDNN::Layout::ACDEB};

    reorderFunc<int32_t, int32_t>(srcTensor, dstTensor);
}

TEST(ReorderGTests, U8ToF16_1)
{
    const KuDNN::TensorInfo srcTensor = {{10, 12, 14}, KuDNN::Element::TypeT::U8, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo dstTensor = {{10, 12, 14}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABC};

    reorderFunc<uint8_t, __fp16>(srcTensor, dstTensor);
}

TEST(ReorderGTests, U8ToBF16_1)
{
    const KuDNN::TensorInfo srcTensor = {{10, 12, 14}, KuDNN::Element::TypeT::U8, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo dstTensor = {{10, 12, 14}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABC};

    reorderFunc<uint8_t, __bf16>(srcTensor, dstTensor);
}

TEST(ReorderGTests, U8ToS8_1)
{
    const KuDNN::TensorInfo srcTensor = {{10, 12, 14}, KuDNN::Element::TypeT::U8, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo dstTensor = {{10, 12, 14}, KuDNN::Element::TypeT::S8, KuDNN::Layout::ABC};

    reorderFunc<uint8_t, int8_t>(srcTensor, dstTensor);
}

TEST(ReorderGTests, U8ToU8_1)
{
    const KuDNN::TensorInfo srcTensor = {{10, 10, 12, 14}, KuDNN::Element::TypeT::U8, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo dstTensor = {{10, 10, 12, 14}, KuDNN::Element::TypeT::U8, KuDNN::Layout::ACBD};

    reorderFunc<uint8_t, uint8_t>(srcTensor, dstTensor);
}

TEST(ReorderGTests, U8ToS32_1)
{
    const KuDNN::TensorInfo srcTensor = {{10, 12, 14}, KuDNN::Element::TypeT::U8, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo dstTensor = {{10, 12, 14}, KuDNN::Element::TypeT::S32, KuDNN::Layout::ABC};

    reorderFunc<uint8_t, int32_t>(srcTensor, dstTensor);
}

TEST(ReorderGTests, U8ToF32_1)
{
    const KuDNN::TensorInfo srcTensor = {{10, 12, 14}, KuDNN::Element::TypeT::U8, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo dstTensor = {{10, 12, 14}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};

    reorderFunc<uint8_t, float>(srcTensor, dstTensor);
}


TEST(ReorderGTests, S8ToBF16_1)
{
    const KuDNN::TensorInfo srcTensor = {{10, 12, 14}, KuDNN::Element::TypeT::S8, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo dstTensor = {{10, 12, 14}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABC};

    reorderFunc<int8_t, __bf16>(srcTensor, dstTensor);
}

TEST(ReorderGTests, S8ToU8_1)
{
    const KuDNN::TensorInfo srcTensor = {{10, 12, 14}, KuDNN::Element::TypeT::S8, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo dstTensor = {{10, 12, 14}, KuDNN::Element::TypeT::U8, KuDNN::Layout::ABC};

    reorderFunc<int8_t, uint8_t>(srcTensor, dstTensor);
}

TEST(ReorderGTests, S8ToS32_1)
{
    const KuDNN::TensorInfo srcTensor = {{10, 12, 14}, KuDNN::Element::TypeT::S8, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo dstTensor = {{10, 12, 14}, KuDNN::Element::TypeT::S32, KuDNN::Layout::ABC};

    reorderFunc<int8_t, int32_t>(srcTensor, dstTensor);
}

TEST(ReorderGTests, MoveConstructor1)
{
    const KuDNN::TensorInfo srcTensor = {{10, 12, 14}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo dstTensor = {{10, 12, 14}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABC};

    KuDNN::ReorderLayer reorderLayer1(srcTensor, dstTensor);

    KuDNN::ReorderLayer reorderLayer2(std::move(reorderLayer1));
}

TEST(ReorderGTests, CopyConstructor2)
{
    const KuDNN::TensorInfo srcTensor = {{10, 12, 14}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo dstTensor = {{10, 12, 14}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABC};

    KuDNN::ReorderLayer reorderLayer1(srcTensor, dstTensor);
    KuDNN::ReorderLayer reorderLayer2(srcTensor, dstTensor);
    reorderLayer2 = std::move(reorderLayer1);
}

TEST(ReorderGTests, CopyConstructor1)
{
    const KuDNN::TensorInfo srcTensor = {{10, 12, 14}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo dstTensor = {{10, 12, 14}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABC};

    KuDNN::ReorderLayer reorderLayer1(srcTensor, dstTensor);

    KuDNN::ReorderLayer reorderLayer2(reorderLayer1);
}

TEST(ReorderGTests, MoveConstructor2)
{
    const KuDNN::TensorInfo srcTensor = {{10, 12, 14}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo dstTensor = {{10, 12, 14}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABC};

    KuDNN::ReorderLayer reorderLayer1(srcTensor, dstTensor);
    KuDNN::ReorderLayer reorderLayer2(srcTensor, dstTensor);
    reorderLayer2 = reorderLayer1;
}

TEST(ReorderGTests, S8_ACBToABC)
{
    const KuDNN::TensorInfo srcTensor = {{10, 12, 255}, KuDNN::Element::TypeT::S8, KuDNN::Layout::ACB};
    const KuDNN::TensorInfo dstTensor = {{10, 12, 255}, KuDNN::Element::TypeT::S8, KuDNN::Layout::ABC};

    reorderFunc<int8_t, int8_t>(srcTensor, dstTensor);
}
