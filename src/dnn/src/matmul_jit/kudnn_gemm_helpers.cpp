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

#include "kudnn.hpp"

#include "matmul_jit/kudnn_gemm_helpers.hpp"

namespace KuDNN {
namespace Detail {

// checks if layout has last logical dimension with fastest changing index
bool IsLayoutAlikeAB(Layout layout) noexcept
{
    switch (layout) {
        case Layout::A: {
            return true;
        }
        case Layout::AB: {
            return true;
        }
        case Layout::BA: {
            return false;
        }
        case Layout::ABC: {
            return true;
        }
        case Layout::ACB: {
            return false;
        }
        case Layout::BAC: {
            return true;
        }
        case Layout::BCA: {
            return false;
        }
        case Layout::CAB: {
            return false;
        }
        case Layout::CBA: {
            return false;
        }
        case Layout::ABCD: {
            return true;
        }
        case Layout::ABDC: {
            return false;
        }
        case Layout::ACBD: {
            return true;
        }
        case Layout::ACDB: {
            return false;
        }
        case Layout::ADBC: {
            return false;
        }
        case Layout::ADCB: {
            return false;
        }
        case Layout::BACD: {
            return true;
        }
        case Layout::BCDA: {
            return false;
        }
        case Layout::CDAB: {
            return false;
        }
        case Layout::CDBA: {
            return false;
        }
        case Layout::DCAB: {
            return false;
        }
        case Layout::ABCDE: {
            return true;
        }
        case Layout::ABCED: {
            return false;
        }
        case Layout::ABDEC: {
            return false;
        }
        case Layout::ACBDE: {
            return true;
        }
        case Layout::ACDEB: {
            return false;
        }
        case Layout::ADECB: {
            return false;
        }
        case Layout::BACDE: {
            return true;
        }
        case Layout::BCDEA: {
            return false;
        }
        case Layout::CDEAB: {
            return false;
        }
        case Layout::CDEBA: {
            return false;
        }
        case Layout::DECAB: {
            return false;
        }
        default: {
            return false;
        }
    }
}

// checks if layout has last but one logical dimension with fastest changing index
bool IsLayoutAlikeBA(Layout layout) noexcept
{
    switch (layout) {
        case Layout::A: {
            return true;
        }
        case Layout::AB: {
            return false;
        }
        case Layout::BA: {
            return true;
        }
        case Layout::ABC: {
            return false;
        }
        case Layout::ACB: {
            return true;
        }
        case Layout::BAC: {
            return false;
        }
        case Layout::BCA: {
            return false;
        }
        case Layout::CAB: {
            return true;
        }
        case Layout::CBA: {
            return false;
        }
        case Layout::ABCD: {
            return false;
        }
        case Layout::ABDC: {
            return true;
        }
        case Layout::ACBD: {
            return false;
        }
        case Layout::ACDB: {
            return false;
        }
        case Layout::ADBC: {
            return true;
        }
        case Layout::ADCB: {
            return false;
        }
        case Layout::BACD: {
            return false;
        }
        case Layout::BCDA: {
            return false;
        }
        case Layout::CDAB: {
            return false;
        }
        case Layout::CDBA: {
            return false;
        }
        case Layout::DCAB: {
            return false;
        }
        case Layout::ABCDE: {
            return false;
        }
        case Layout::ABCED: {
            return true;
        }
        case Layout::ABDEC: {
            return false;
        }
        case Layout::ACBDE: {
            return false;
        }
        case Layout::ACDEB: {
            return false;
        }
        case Layout::ADECB: {
            return false;
        }
        case Layout::BACDE: {
            return false;
        }
        case Layout::BCDEA: {
            return false;
        }
        case Layout::CDEAB: {
            return false;
        }
        case Layout::CDEBA: {
            return false;
        }
        case Layout::DECAB: {
            return false;
        }
        default: {
            return false;
        }
    }
}

SizeType GetLayoutDiemensionality(Layout layout) noexcept
{
    switch (layout) {
        case Layout::A: {
            return DIM_1;
        }
        case Layout::AB: {
            return DIM_2;
        }
        case Layout::BA: {
            return DIM_2;
        }
        case Layout::ABC: {
            return DIM_3;
        }
        case Layout::ACB: {
            return DIM_3;
        }
        case Layout::BAC: {
            return DIM_3;
        }
        case Layout::BCA: {
            return DIM_3;
        }
        case Layout::CAB: {
            return DIM_3;
        }
        case Layout::CBA: {
            return DIM_3;
        }
        case Layout::ABCD: {
            return DIM_4;
        }
        case Layout::ABDC: {
            return DIM_4;
        }
        case Layout::ACBD: {
            return DIM_4;
        }
        case Layout::ACDB: {
            return DIM_4;
        }
        case Layout::ADBC: {
            return DIM_4;
        }
        case Layout::ADCB: {
            return DIM_4;
        }
        case Layout::BACD: {
            return DIM_4;
        }
        case Layout::BCDA: {
            return DIM_4;
        }
        case Layout::CDAB: {
            return DIM_4;
        }
        case Layout::CDBA: {
            return DIM_4;
        }
        case Layout::DCAB: {
            return DIM_4;
        }
        case Layout::ABCDE: {
            return DIM_5;
        }
        case Layout::ABCED: {
            return DIM_5;
        }
        case Layout::ABDEC: {
            return DIM_5;
        }
        case Layout::ACBDE: {
            return DIM_5;
        }
        case Layout::ACDEB: {
            return DIM_5;
        }
        case Layout::ADECB: {
            return DIM_5;
        }
        case Layout::BACDE: {
            return DIM_5;
        }
        case Layout::BCDEA: {
            return DIM_5;
        }
        case Layout::CDEAB: {
            return DIM_5;
        }
        case Layout::CDEBA: {
            return DIM_5;
        }
        case Layout::DECAB: {
            return DIM_5;
        }
        default: {
            return 0;
        }
    }
}

Layout GetPlaneLayoutAB(Layout layout) noexcept
{
    switch (GetLayoutDiemensionality(layout)) {
        case DIM_1: {
            return Layout::A;
        }
        case DIM_2: {
            return Layout::AB;
        }
        case DIM_3: {
            return Layout::ABC;
        }
        case DIM_4: {
            return Layout::ABCD;
        }
        case DIM_5: {
            return Layout::ABCDE;
        }
        default: {
            return Layout::UNDEFINED;
        }
    }
}

Layout GetPlaneLayoutBA(Layout layout) noexcept
{
    switch (GetLayoutDiemensionality(layout)) {
        case DIM_1: {
            return Layout::A;
        }
        case DIM_2: {
            return Layout::BA;
        }
        case DIM_3: {
            return Layout::ACB;
        }
        case DIM_4: {
            return Layout::ABDC;
        }
        case DIM_5: {
            return Layout::ABCED;
        }
        default: {
            return Layout::UNDEFINED;
        }
    }
}

} // namespace Detail
} // namespace KuDNN
