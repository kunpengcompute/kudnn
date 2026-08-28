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

#ifndef KUDNN_GNORM_CALCULATE_STRUCT_HPP
#define KUDNN_GNORM_CALCULATE_STRUCT_HPP

#include "kudnn.hpp"
#include "vector_wrappers.hpp"

namespace KuDNN {
namespace Detail {
namespace GNorm {

template <typename CompType, typename InType, typename OutType, typename ScaleShiftType, ARCH Arch = ARCH::VC>
struct GroupNormCrossFWD final {
    static SizeType LoopInc() noexcept
    {
        if constexpr (Arch == ARCH::VC) {
            return Vector::UnrollLength<InType, CompType, OutType, ScaleShiftType, float>();
        } else {
            return NUM_2 * Vector::NeonLanes<CompType>();
        }
    }

    // for mean calculation. Mean is computed in float precision
    static void Do(SizeType loopIdx0, SizeType loopRange0, const InType *src0,
        typename Vector::Vec<float, Arch>::Type &vMean00, typename Vector::Vec<float, Arch>::Type &vMean01,
        typename Vector::Vec<float, Arch>::Type &vMean02, typename Vector::Vec<float, Arch>::Type &vMean03)
    {
        svbool_t ldPG00;
        svbool_t ldPG01;
        svbool_t ldPG02;
        svbool_t ldPG03;
        Vector::InitPGN<InType, Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>()>::Do(
            ldPG00, ldPG01, ldPG02, ldPG03, loopIdx0, loopRange0);
        typename Vector::Vec<float, Arch>::Type data00;
        typename Vector::Vec<float, Arch>::Type data01;
        typename Vector::Vec<float, Arch>::Type data02;
        typename Vector::Vec<float, Arch>::Type data03;
        Vector::LoadN<InType, float,
            Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>(), Arch>::Do(
            src0 + loopIdx0, ldPG00, ldPG01, ldPG02, ldPG03, data00, data01, data02, data03);
        svbool_t opPG00;
        svbool_t opPG01;
        svbool_t opPG02;
        svbool_t opPG03;
        Vector::InitPGN<float,
            Vector::GetUsedRegsN<CompType, InType, OutType, ScaleShiftType, float>()>::Do(
            opPG00, opPG01, opPG02, opPG03, loopIdx0, loopRange0);
        Vector::AddMN<float,
            Vector::GetUsedRegsN<CompType, InType, OutType, ScaleShiftType, float>(), Arch>::Do(
            opPG00, opPG01, opPG02, opPG03, vMean00, vMean01, vMean02, vMean03,
            data00, data01, data02, data03, vMean00, vMean01, vMean02, vMean03);
    }

    // for variance calculation. Variance is computed in float precision
    static void Do(SizeType loopIdx0, SizeType loopRange0, const InType *src0,
        typename Vector::Vec<float, Arch>::Type &vMean00, typename Vector::Vec<float, Arch>::Type &vMean01,
        typename Vector::Vec<float, Arch>::Type &vMean02, typename Vector::Vec<float, Arch>::Type &vMean03,
        typename Vector::Vec<float, Arch>::Type &vVariance00, typename Vector::Vec<float, Arch>::Type &vVariance01,
        typename Vector::Vec<float, Arch>::Type &vVariance02, typename Vector::Vec<float, Arch>::Type &vVariance03)
    {
        svbool_t ldPG00;
        svbool_t ldPG01;
        svbool_t ldPG02;
        svbool_t ldPG03;
        Vector::InitPGN<InType, Vector::GetUsedRegsN<InType, CompType, OutType, ScaleShiftType, float>()>::Do(
            ldPG00, ldPG01, ldPG02, ldPG03, loopIdx0, loopRange0);
        typename Vector::Vec<float, Arch>::Type data00;
        typename Vector::Vec<float, Arch>::Type data01;
        typename Vector::Vec<float, Arch>::Type data02;
        typename Vector::Vec<float, Arch>::Type data03;
        Vector::LoadN<InType, float,
            Vector::GetUsedRegsN<InType, CompType, OutType, ScaleShiftType, float>(), Arch>::Do(
            src0 + loopIdx0, ldPG00, ldPG01, ldPG02, ldPG03, data00, data01, data02, data03);
        svbool_t opPG00;
        svbool_t opPG01;
        svbool_t opPG02;
        svbool_t opPG03;
        Vector::InitPGN<float,
            Vector::GetUsedRegsN<float, InType, CompType, OutType, ScaleShiftType>()>::Do(
            opPG00, opPG01, opPG02, opPG03, loopIdx0, loopRange0);
        typename Vector::Vec<float, Arch>::Type m00;
        typename Vector::Vec<float, Arch>::Type m01;
        typename Vector::Vec<float, Arch>::Type m02;
        typename Vector::Vec<float, Arch>::Type m03;
        Vector::SubXN<float,
            Vector::GetUsedRegsN<float, InType, CompType, OutType, ScaleShiftType>(), Arch>::Do(
            opPG00, opPG01, opPG02, opPG03, data00, data01, data02, data03,
            vMean00, vMean01, vMean02, vMean03, m00, m01, m02, m03);
        Vector::MlaMN<float,
            Vector::GetUsedRegsN<float, InType, CompType, OutType, ScaleShiftType>(), Arch>::Do(
            opPG00, opPG01, opPG02, opPG03, vVariance00, vVariance01, vVariance02, vVariance03,
            m00, m01, m02, m03, m00, m01, m02, m03, vVariance00, vVariance01, vVariance02, vVariance03);
    }

    static void Do(SizeType loopIdx, SizeType loopRange, const InType *src,
        const ScaleShiftType *scale, const ScaleShiftType *shift, SizeType startGroup,
        bool useScale, bool useShift, SizeType innerSize, SizeType group, SizeType channelPerGroup, SizeType channel,
        typename Vector::Vec<CompType, Arch>::Type &invSqrtVarianceVec00,
        typename Vector::Vec<CompType, Arch>::Type &invSqrtVarianceVec01,
        typename Vector::Vec<CompType, Arch>::Type &invSqrtVarianceVec02,
        typename Vector::Vec<CompType, Arch>::Type &invSqrtVarianceVec03,
        typename Vector::Vec<CompType, Arch>::Type &constants00,
        typename Vector::Vec<CompType, Arch>::Type &constants01,
        typename Vector::Vec<CompType, Arch>::Type &constants02,
        typename Vector::Vec<CompType, Arch>::Type &constants03,
        typename Vector::Vec<CompType, Arch>::Type &vMean00,
        typename Vector::Vec<CompType, Arch>::Type &vMean01,
        typename Vector::Vec<CompType, Arch>::Type &vMean02,
        typename Vector::Vec<CompType, Arch>::Type &vMean03,
        OutType *dst) noexcept
    {
        Service::Unused(innerSize);
        Service::Unused(group);
        svbool_t opPG0;
        svbool_t opPG1;
        svbool_t opPG2;
        svbool_t opPG3;
        Vector::InitPGN<CompType, Vector::GetUsedRegsN<CompType, float, InType, OutType, ScaleShiftType>()>::Do(
            opPG0, opPG1, opPG2, opPG3, loopIdx, loopRange);

        SizeType scaleshiftIdx = channel + startGroup * channelPerGroup;
        if (useScale) {
            typename Vector::Vec<CompType, Arch>::Type scaleVal00;
            typename Vector::Vec<CompType, Arch>::Type scaleVal01;
            typename Vector::Vec<CompType, Arch>::Type scaleVal02;
            typename Vector::Vec<CompType, Arch>::Type scaleVal03;
            Vector::DupN<ScaleShiftType, CompType,
                Vector::GetUsedRegsN<ScaleShiftType, float, InType, CompType, OutType>(), Arch>::Do(
                scale[scaleshiftIdx], scaleVal00, scaleVal01, scaleVal02, scaleVal03);
            Vector::MulMN<CompType,
                Vector::GetUsedRegsN<CompType, ScaleShiftType, float, InType, OutType>(), Arch>::Do(
                opPG0, opPG1, opPG2, opPG3, scaleVal00, scaleVal01, scaleVal02, scaleVal03,
                invSqrtVarianceVec00, invSqrtVarianceVec01, invSqrtVarianceVec02, invSqrtVarianceVec03,
                constants00, constants01, constants02, constants03);
        }

        typename Vector::Vec<CompType, Arch>::Type shiftVal0;
        typename Vector::Vec<CompType, Arch>::Type shiftVal1;
        typename Vector::Vec<CompType, Arch>::Type shiftVal2;
        typename Vector::Vec<CompType, Arch>::Type shiftVal3;
        if (useShift) {
            Vector::DupN<ScaleShiftType, CompType, Vector::GetUsedRegsN<ScaleShiftType, float, InType,
                CompType, OutType>(), Arch>::Do(shift[scaleshiftIdx], shiftVal0, shiftVal1, shiftVal2, shiftVal3);
        } else {
            Vector::DupFloatN<ScaleShiftType, CompType, Vector::GetUsedRegsN<ScaleShiftType, CompType, float,
                InType, OutType>(), Arch>::Do(0.0f, shiftVal0, shiftVal1, shiftVal2, shiftVal3);
        }

        svbool_t ldPG00;
        svbool_t ldPG01;
        svbool_t ldPG02;
        svbool_t ldPG03;
        Vector::InitPGN<InType, Vector::GetUsedRegsN<InType, ScaleShiftType, CompType, float, OutType>()>::Do(
            ldPG00, ldPG01, ldPG02, ldPG03, loopIdx, loopRange);
        typename Vector::Vec<CompType, Arch>::Type data00;
        typename Vector::Vec<CompType, Arch>::Type data01;
        typename Vector::Vec<CompType, Arch>::Type data02;
        typename Vector::Vec<CompType, Arch>::Type data03;
        Vector::LoadN<InType, CompType,
            Vector::GetUsedRegsN<InType, ScaleShiftType, CompType, float, OutType>(), Arch>::Do(
            src + loopIdx, ldPG00, ldPG01, ldPG02, ldPG03, data00, data01, data02, data03);
        Vector::SubXN<CompType, Vector::GetUsedRegsN<CompType, InType, ScaleShiftType, float, OutType>(), Arch>::Do(
            opPG0, opPG1, opPG2, opPG3, data00, data01, data02, data03,
            vMean00, vMean01, vMean02, vMean03, data00, data01, data02, data03);
        Vector::MlaXN<CompType, Vector::GetUsedRegsN<CompType, InType, ScaleShiftType, float, OutType>(), Arch>::Do(
            opPG0, opPG1, opPG2, opPG3,
            shiftVal0, shiftVal1, shiftVal2, shiftVal3,
            constants00, constants01, constants02, constants03,
            data00, data01, data02, data03,
            data00, data01, data02, data03);
        svbool_t stPG00;
        svbool_t stPG01;
        svbool_t stPG02;
        svbool_t stPG03;
        Vector::InitPGN<OutType, Vector::GetUsedRegsN<OutType, CompType, InType, ScaleShiftType, float>()>::Do(
            stPG00, stPG01, stPG02, stPG03, loopIdx, loopRange);
        Vector::StoreN<OutType, CompType,
            Vector::GetUsedRegsN<OutType, CompType, InType, ScaleShiftType, float>(), Arch>::Do(
            stPG00, stPG01, stPG02, stPG03, data00, data01, data02, data03, dst + loopIdx);
    }
};

template <typename CompType, typename InType, typename OutType, typename ScaleShiftType, ARCH Arch = ARCH::VC>
struct GroupNormLastFWD final {
    static SizeType LoopInc() noexcept
    {
        if constexpr (Arch == ARCH::VC) {
            return Vector::UnrollLength<InType, CompType, OutType, ScaleShiftType, float>();
        } else {
            return NUM_2 * Vector::NeonLanes<CompType>();
        }
    }
    static void Do(SizeType loopIdx1, SizeType loopRange1, const InType *src1,
        typename Vector::Vec<float, Arch>::Type &vMean10, typename Vector::Vec<float, Arch>::Type &vMean11,
        typename Vector::Vec<float, Arch>::Type &vMean12, typename Vector::Vec<float, Arch>::Type &vMean13) noexcept
    {
        svbool_t ldPG10;
        svbool_t ldPG11;
        svbool_t ldPG12;
        svbool_t ldPG13;
        Vector::InitPGN<InType, Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>()>::Do(
            ldPG10, ldPG11, ldPG12, ldPG13, loopIdx1, loopRange1);
        typename Vector::Vec<float, Arch>::Type data10;
        typename Vector::Vec<float, Arch>::Type data11;
        typename Vector::Vec<float, Arch>::Type data12;
        typename Vector::Vec<float, Arch>::Type data13;
        Vector::LoadN<InType, float,
            Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>(), Arch>::Do(
            src1+ loopIdx1, ldPG10, ldPG11, ldPG12, ldPG13, data10, data11, data12, data13);
        svbool_t opPG10;
        svbool_t opPG11;
        svbool_t opPG12;
        svbool_t opPG13;
        Vector::InitPGN<float,
            Vector::GetUsedRegsN<CompType, InType, OutType, ScaleShiftType, float>()>::Do(
            opPG10, opPG11, opPG12, opPG13, loopIdx1, loopRange1);
        Vector::AddMN<float,
            Vector::GetUsedRegsN<CompType, InType, OutType, ScaleShiftType, float>(), Arch>::Do(
            opPG10, opPG11, opPG12, opPG13, vMean10, vMean11, vMean12, vMean13,
            data10, data11, data12, data13, vMean10, vMean11, vMean12, vMean13);
    }
    static void Do(SizeType loopIdx1, SizeType loopRange1, const InType *src1,
        typename Vector::Vec<float, Arch>::Type &vMean10,
        typename Vector::Vec<float, Arch>::Type &vMean11,
        typename Vector::Vec<float, Arch>::Type &vMean12,
        typename Vector::Vec<float, Arch>::Type &vMean13,
        typename Vector::Vec<float, Arch>::Type &vVariance10,
        typename Vector::Vec<float, Arch>::Type &vVariance11,
        typename Vector::Vec<float, Arch>::Type &vVariance12,
        typename Vector::Vec<float, Arch>::Type &vVariance13) noexcept
    {
        svbool_t ldPG10;
        svbool_t ldPG11;
        svbool_t ldPG12;
        svbool_t ldPG13;
        Vector::InitPGN<InType, Vector::GetUsedRegsN<InType, CompType, OutType, ScaleShiftType, float>()>::Do(
            ldPG10, ldPG11, ldPG12, ldPG13, loopIdx1, loopRange1);
        typename Vector::Vec<float, Arch>::Type data10;
        typename Vector::Vec<float, Arch>::Type data11;
        typename Vector::Vec<float, Arch>::Type data12;
        typename Vector::Vec<float, Arch>::Type data13;
        Vector::LoadN<InType, float,
            Vector::GetUsedRegsN<InType, CompType, OutType, ScaleShiftType, float>(), Arch>::Do(
            src1 + loopIdx1, ldPG10, ldPG11, ldPG12, ldPG13, data10, data11, data12, data13);
        svbool_t opPG10;
        svbool_t opPG11;
        svbool_t opPG12;
        svbool_t opPG13;
        Vector::InitPGN<float,
            Vector::GetUsedRegsN<float, InType, CompType, OutType, ScaleShiftType>()>::Do(
            opPG10, opPG11, opPG12, opPG13, loopIdx1, loopRange1);
        typename Vector::Vec<float, Arch>::Type m10;
        typename Vector::Vec<float, Arch>::Type m11;
        typename Vector::Vec<float, Arch>::Type m12;
        typename Vector::Vec<float, Arch>::Type m13;
        Vector::SubXN<float,
            Vector::GetUsedRegsN<float, InType, CompType, OutType, ScaleShiftType>(), Arch>::Do(
            opPG10, opPG11, opPG12, opPG13, data10, data11, data12, data13,
            vMean10, vMean11, vMean12, vMean13, m10, m11, m12, m13);
        Vector::MlaMN<float,
            Vector::GetUsedRegsN<float, InType, CompType, OutType, ScaleShiftType>(), Arch>::Do(
            opPG10, opPG11, opPG12, opPG13, vVariance10, vVariance11, vVariance12, vVariance13,
            m10, m11, m12, m13, m10, m11, m12, m13, vVariance10, vVariance11, vVariance12, vVariance13);
    }

    static void Do(SizeType loopIdx, SizeType loopRange, const InType *src,
        const ScaleShiftType *scale, const ScaleShiftType *shift,
        bool useScale, bool useShift, SizeType g,
        typename Vector::Vec<CompType, Arch>::Type &invSqrtVarianceVec10,
        typename Vector::Vec<CompType, Arch>::Type &invSqrtVarianceVec11,
        typename Vector::Vec<CompType, Arch>::Type &invSqrtVarianceVec12,
        typename Vector::Vec<CompType, Arch>::Type &invSqrtVarianceVec13,
        typename Vector::Vec<CompType, Arch>::Type &constants10,
        typename Vector::Vec<CompType, Arch>::Type &constants11,
        typename Vector::Vec<CompType, Arch>::Type &constants12,
        typename Vector::Vec<CompType, Arch>::Type &constants13,
        typename Vector::Vec<CompType, Arch>::Type &vMean10,
        typename Vector::Vec<CompType, Arch>::Type &vMean11,
        typename Vector::Vec<CompType, Arch>::Type &vMean12,
        typename Vector::Vec<CompType, Arch>::Type &vMean13,
        OutType *dst) noexcept
    {
        Service::Unused(g);
        svbool_t opPG0;
        svbool_t opPG1;
        svbool_t opPG2;
        svbool_t opPG3;
        Vector::InitPGN<CompType, Vector::GetUsedRegsN<CompType, float, InType, OutType, ScaleShiftType>()>::Do(
            opPG0, opPG1, opPG2, opPG3, loopIdx, loopRange);
        if (useScale) {
            typename Vector::Vec<CompType, Arch>::Type scaleVal10;
            typename Vector::Vec<CompType, Arch>::Type scaleVal11;
            typename Vector::Vec<CompType, Arch>::Type scaleVal12;
            typename Vector::Vec<CompType, Arch>::Type scaleVal13;
            svbool_t ldscalepg0;
            svbool_t ldscalepg1;
            svbool_t ldscalepg2;
            svbool_t ldscalepg3;
            Vector::InitPGN<ScaleShiftType,
                Vector::GetUsedRegsN<ScaleShiftType, CompType, float, InType, OutType>()>::Do(
                ldscalepg0, ldscalepg1, ldscalepg2, ldscalepg3, loopIdx, loopRange);
            Vector::LoadN<ScaleShiftType, CompType,
                Vector::GetUsedRegsN<ScaleShiftType, CompType, float, InType, OutType>(), Arch>::Do(
                scale + loopIdx, ldscalepg0, ldscalepg1, ldscalepg2, ldscalepg3,
                scaleVal10, scaleVal11, scaleVal12, scaleVal13);
            Vector::MulMN<CompType, Vector::GetUsedRegsN<CompType, ScaleShiftType, float, InType, OutType>(), Arch>::Do(
                opPG0, opPG1, opPG2, opPG3, scaleVal10, scaleVal11, scaleVal12, scaleVal13,
                invSqrtVarianceVec10, invSqrtVarianceVec11, invSqrtVarianceVec12, invSqrtVarianceVec13,
                constants10, constants11, constants12, constants13);
        }
        typename Vector::Vec<CompType, Arch>::Type shiftVal0;
        typename Vector::Vec<CompType, Arch>::Type shiftVal1;
        typename Vector::Vec<CompType, Arch>::Type shiftVal2;
        typename Vector::Vec<CompType, Arch>::Type shiftVal3;
        if (useShift) {
            svbool_t ldshiftpg0;
            svbool_t ldshiftpg1;
            svbool_t ldshiftpg2;
            svbool_t ldshiftpg3;
            Vector::InitPGN<ScaleShiftType,
                Vector::GetUsedRegsN<ScaleShiftType, CompType, float, InType, OutType>()>::Do(
                ldshiftpg0, ldshiftpg1, ldshiftpg2, ldshiftpg3, loopIdx, loopRange);
            Vector::LoadN<ScaleShiftType, CompType,
                Vector::GetUsedRegsN<ScaleShiftType, CompType, float, InType, OutType>(), Arch>::Do(
                shift + loopIdx, ldshiftpg0, ldshiftpg1, ldshiftpg2, ldshiftpg3,
                shiftVal0, shiftVal1, shiftVal2, shiftVal3);
        } else {
            Vector::DupFloatN<ScaleShiftType, CompType,
                Vector::GetUsedRegsN<ScaleShiftType, CompType, float, InType, OutType>(), Arch>::Do(
                0.0f, shiftVal0, shiftVal1, shiftVal2, shiftVal3);
        }
        svbool_t ldPG10;
        svbool_t ldPG11;
        svbool_t ldPG12;
        svbool_t ldPG13;
        Vector::InitPGN<InType, Vector::GetUsedRegsN<InType, ScaleShiftType, CompType, float, OutType>()>::Do(
            ldPG10, ldPG11, ldPG12, ldPG13, loopIdx, loopRange);
        typename Vector::Vec<CompType, Arch>::Type data10;
        typename Vector::Vec<CompType, Arch>::Type data11;
        typename Vector::Vec<CompType, Arch>::Type data12;
        typename Vector::Vec<CompType, Arch>::Type data13;
        Vector::LoadN<InType, CompType,
            Vector::GetUsedRegsN<InType, ScaleShiftType, CompType, float, OutType>(), Arch>::Do(
            src + loopIdx, ldPG10, ldPG11, ldPG12, ldPG13, data10, data11, data12, data13);
        Vector::SubXN<CompType, Vector::GetUsedRegsN<CompType, InType, ScaleShiftType, float, OutType>(), Arch>::Do(
            opPG0, opPG1, opPG2, opPG3, data10, data11, data12, data13,
            vMean10, vMean11, vMean12, vMean13, data10, data11, data12, data13);
        Vector::MlaXN<CompType, Vector::GetUsedRegsN<CompType, InType, ScaleShiftType, float, OutType>(), Arch>::Do(
            opPG0, opPG1, opPG2, opPG3, shiftVal0, shiftVal1, shiftVal2, shiftVal3,
            constants10, constants11, constants12, constants13,
            data10, data11, data12, data13,
            data10, data11, data12, data13);
        svbool_t stPG10;
        svbool_t stPG11;
        svbool_t stPG12;
        svbool_t stPG13;
        Vector::InitPGN<OutType, Vector::GetUsedRegsN<OutType, CompType, InType, ScaleShiftType, float>()>::Do(
            stPG10, stPG11, stPG12, stPG13, loopIdx, loopRange);
        Vector::StoreN<OutType, CompType,
            Vector::GetUsedRegsN<OutType, CompType, InType, ScaleShiftType, float>(), Arch>::Do(
            stPG10, stPG11, stPG12, stPG13, data10, data11, data12, data13, dst + loopIdx);
    }
};

} // GNorm
} // Detail
} // KuDNN

#endif //  KUDNN_GNORM_KERNELS_HPP
