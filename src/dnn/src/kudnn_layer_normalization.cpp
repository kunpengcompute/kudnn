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
#include <memory>
#include <cmath>
#include <cstddef>
#include <map>

#include <arm_sve.h>

#include "kudnn.hpp"

#include "kudnn_layer_normalization_impl.hpp"
#include "internal_funcs.hpp"
#include "threading_impl.hpp"
#include "vector_wrappers.hpp"

namespace KuDNN {

NormalizationFlags operator&(const NormalizationFlags &lhs, const NormalizationFlags &rhs)
{
    return static_cast<NormalizationFlags>(static_cast<std::uint32_t>(lhs) & static_cast<std::uint32_t>(rhs));
}

NormalizationFlags operator|(const NormalizationFlags &lhs, const NormalizationFlags &rhs)
{
    return static_cast<NormalizationFlags>(static_cast<std::uint32_t>(lhs) | static_cast<std::uint32_t>(rhs));
}

NormalizationFlags &operator|=(NormalizationFlags &lhs, const NormalizationFlags &rhs)
{
    lhs = lhs | rhs;
    return lhs;
}

NormalizationFlags &operator&=(NormalizationFlags &lhs, const NormalizationFlags &rhs)
{
    lhs = lhs & rhs;
    return lhs;
}

static const std::map<KuDNN::Layout, KuDNN::Layout> layoutMap = {
    {KuDNN::Layout::AB, KuDNN::Layout::A},       {KuDNN::Layout::BA, KuDNN::Layout::A},
    {KuDNN::Layout::ABC, KuDNN::Layout::AB},     {KuDNN::Layout::ACB, KuDNN::Layout::AB},
    {KuDNN::Layout::BAC, KuDNN::Layout::BA},     {KuDNN::Layout::BCA, KuDNN::Layout::BA},
    {KuDNN::Layout::CAB, KuDNN::Layout::AB},     {KuDNN::Layout::CBA, KuDNN::Layout::BA},
    {KuDNN::Layout::ABCD, KuDNN::Layout::ABC},   {KuDNN::Layout::ABDC, KuDNN::Layout::ABC},
    {KuDNN::Layout::ACBD, KuDNN::Layout::ACB},   {KuDNN::Layout::ACDB, KuDNN::Layout::ACB},
    {KuDNN::Layout::ADBC, KuDNN::Layout::ABC},   {KuDNN::Layout::ADCB, KuDNN::Layout::ACB},
    {KuDNN::Layout::BACD, KuDNN::Layout::BAC},   {KuDNN::Layout::BCDA, KuDNN::Layout::BCA},
    {KuDNN::Layout::CDAB, KuDNN::Layout::CAB},   {KuDNN::Layout::CDBA, KuDNN::Layout::CBA},
    {KuDNN::Layout::DCAB, KuDNN::Layout::CAB},   {KuDNN::Layout::ABCDE, KuDNN::Layout::ABCD},
    {KuDNN::Layout::ABCED, KuDNN::Layout::ABCD}, {KuDNN::Layout::ABDEC, KuDNN::Layout::ABDC},
    {KuDNN::Layout::ACBDE, KuDNN::Layout::ACBD}, {KuDNN::Layout::ACDEB, KuDNN::Layout::ACDB},
    {KuDNN::Layout::ADECB, KuDNN::Layout::ADCB}, {KuDNN::Layout::BACDE, KuDNN::Layout::BACD},
    {KuDNN::Layout::BCDEA, KuDNN::Layout::BCDA}, {KuDNN::Layout::CDEAB, KuDNN::Layout::CDAB},
    {KuDNN::Layout::CDEBA, KuDNN::Layout::CDBA}, {KuDNN::Layout::DECAB, KuDNN::Layout::DCAB}};

KuDNN::Layout Detail::GetSrcLayoutWithoutLastDim(const Layout &srcLayout)
{
    auto item = layoutMap.find(srcLayout);
    if (item != layoutMap.end()) {
        return item->second;
    }
    throw Service::LogicError{"LNorm: unknown src layout"};
}

} // namespace KuDNN
