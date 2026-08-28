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

#ifndef KUDNN_LNORM_FWD_STRUCT_HPP
#define KUDNN_LNORM_FWD_STRUCT_HPP

namespace KuDNN {
namespace Detail {
namespace LNorm {

template <typename CompType, typename InType, typename OutType, typename ScaleShiftType, ARCH Arch = ARCH::VC>
struct LayerNormLastFWD final {
    static SizeType LoopInc() noexcept
    {
        if constexpr (Arch == ARCH::VC) {
            return Vector::UnrollLength<InType, CompType, OutType, ScaleShiftType, float>();
        } else {
            return NUM_2 * Vector::NeonLanes<CompType>();
        }
    }
    static void Do(SizeType loopIdx, SizeType loopRange, const InType *src,
        typename Vector::Vec<float, Arch>::Type &vMean0, typename Vector::Vec<float, Arch>::Type &vMean1,
        typename Vector::Vec<float, Arch>::Type &vMean2, typename Vector::Vec<float, Arch>::Type &vMean3) noexcept
    {
        svbool_t dataPG0;
        svbool_t dataPG1;
        svbool_t dataPG2;
        svbool_t dataPG3;
        Vector::InitPGN<InType, Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>()>::Do(
            dataPG0, dataPG1, dataPG2, dataPG3, loopIdx, loopRange);
        typename Vector::Vec<float, Arch>::Type data0;
        typename Vector::Vec<float, Arch>::Type data1;
        typename Vector::Vec<float, Arch>::Type data2;
        typename Vector::Vec<float, Arch>::Type data3;
        Vector::LoadN<InType, float, Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>(),
            Arch>::Do(src + loopIdx, dataPG0, dataPG1, dataPG2, dataPG3, data0, data1, data2, data3);
        svbool_t opPG0;
        svbool_t opPG1;
        svbool_t opPG2;
        svbool_t opPG3;
        Vector::InitPGN<float, Vector::GetUsedRegsN<CompType, InType, OutType, ScaleShiftType, float>()>::Do(
            opPG0, opPG1, opPG2, opPG3, loopIdx, loopRange);
        Vector::AddMN<float, Vector::GetUsedRegsN<CompType, InType, OutType, ScaleShiftType, float>(), Arch>::Do(
            opPG0, opPG1, opPG2, opPG3, vMean0, vMean1, vMean2, vMean3,
            data0, data1, data2, data3, vMean0, vMean1, vMean2, vMean3);
    }
    static void Do(SizeType loopIdx, SizeType loopRange, const InType *src,
        typename Vector::Vec<float, Arch>::Type &vMean0,
        typename Vector::Vec<float, Arch>::Type &vMean1,
        typename Vector::Vec<float, Arch>::Type &vMean2,
        typename Vector::Vec<float, Arch>::Type &vMean3,
        typename Vector::Vec<float, Arch>::Type &vVariance0,
        typename Vector::Vec<float, Arch>::Type &vVariance1,
        typename Vector::Vec<float, Arch>::Type &vVariance2,
        typename Vector::Vec<float, Arch>::Type &vVariance3) noexcept
    {
        svbool_t pG0;
        svbool_t pG1;
        svbool_t pG2;
        svbool_t pG3;
        Vector::InitPGN<InType, Vector::GetUsedRegsN<InType, CompType, OutType, ScaleShiftType, float>()>::Do(
            pG0, pG1, pG2, pG3, loopIdx, loopRange);
        typename Vector::Vec<float, Arch>::Type data0;
        typename Vector::Vec<float, Arch>::Type data1;
        typename Vector::Vec<float, Arch>::Type data2;
        typename Vector::Vec<float, Arch>::Type data3;
        Vector::LoadN<InType, float, Vector::GetUsedRegsN<InType, CompType, OutType, ScaleShiftType, float>(),
            Arch>::Do(src + loopIdx, pG0, pG1, pG2, pG3, data0, data1, data2, data3);
        svbool_t opPG0;
        svbool_t opPG1;
        svbool_t opPG2;
        svbool_t opPG3;
        Vector::InitPGN<float, Vector::GetUsedRegsN<float, InType, CompType, OutType, ScaleShiftType>()>::Do(
            opPG0, opPG1, opPG2, opPG3, loopIdx, loopRange);
        typename Vector::Vec<float, Arch>::Type m0;
        typename Vector::Vec<float, Arch>::Type m1;
        typename Vector::Vec<float, Arch>::Type m2;
        typename Vector::Vec<float, Arch>::Type m3;
        Vector::SubXN<float, Vector::GetUsedRegsN<float, InType, CompType, OutType, ScaleShiftType>(), Arch>::Do(
            opPG0, opPG1, opPG2, opPG3, data0, data1, data2, data3, vMean0, vMean1, vMean2, vMean3, m0, m1, m2, m3);
        Vector::MlaMN<float, Vector::GetUsedRegsN<float, InType, CompType, OutType, ScaleShiftType>(), Arch>::Do(
            opPG0, opPG1, opPG2, opPG3, vVariance0, vVariance1, vVariance2, vVariance3,
            m0, m1, m2, m3, m0, m1, m2, m3, vVariance0, vVariance1, vVariance2, vVariance3);
    }

    static void Do(SizeType loopIdx, SizeType loopRange, const InType *src, const ScaleShiftType *scale,
        const ScaleShiftType *shift, bool useScale, bool useShift,
        typename Vector::Vec<CompType, Arch>::Type &invSqrtVarianceVec0,
        typename Vector::Vec<CompType, Arch>::Type &invSqrtVarianceVec1,
        typename Vector::Vec<CompType, Arch>::Type &invSqrtVarianceVec2,
        typename Vector::Vec<CompType, Arch>::Type &invSqrtVarianceVec3,
        typename Vector::Vec<CompType, Arch>::Type &constants0, typename Vector::Vec<CompType, Arch>::Type &constants1,
        typename Vector::Vec<CompType, Arch>::Type &constants2, typename Vector::Vec<CompType, Arch>::Type &constants3,
        typename Vector::Vec<CompType, Arch>::Type &vMean0, typename Vector::Vec<CompType, Arch>::Type &vMean1,
        typename Vector::Vec<CompType, Arch>::Type &vMean2, typename Vector::Vec<CompType, Arch>::Type &vMean3,
        OutType *dst) noexcept
    {
        svbool_t pG0;
        svbool_t pG1;
        svbool_t pG2;
        svbool_t pG3;
        Vector::InitPGN<CompType, Vector::GetUsedRegsN<CompType, float, InType, OutType, ScaleShiftType>()>::Do(
            pG0, pG1, pG2, pG3, loopIdx, loopRange);
        if (useScale) {
            typename Vector::Vec<CompType, Arch>::Type scaleVal0;
            typename Vector::Vec<CompType, Arch>::Type scaleVal1;
            typename Vector::Vec<CompType, Arch>::Type scaleVal2;
            typename Vector::Vec<CompType, Arch>::Type scaleVal3;
            svbool_t scalepg0;
            svbool_t scalepg1;
            svbool_t scalepg2;
            svbool_t scalepg3;
            Vector::InitPGN<ScaleShiftType, Vector::GetUsedRegsN<ScaleShiftType, CompType, float, InType,
                OutType>()>::Do(scalepg0, scalepg1, scalepg2, scalepg3, loopIdx, loopRange);
            Vector::LoadN<ScaleShiftType, CompType, Vector::GetUsedRegsN<ScaleShiftType, CompType, float, InType,
                OutType>(), Arch>::Do(scale + loopIdx, scalepg0, scalepg1, scalepg2, scalepg3,
                scaleVal0, scaleVal1, scaleVal2, scaleVal3);
            Vector::MulXN<CompType, Vector::GetUsedRegsN<CompType, ScaleShiftType, float, InType,
                OutType>(), Arch>::Do(pG0, pG1, pG2, pG3, scaleVal0, scaleVal1, scaleVal2, scaleVal3,
                invSqrtVarianceVec0, invSqrtVarianceVec1, invSqrtVarianceVec2, invSqrtVarianceVec3,
                constants0, constants1, constants2, constants3);
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
            Vector::InitPGN<ScaleShiftType, Vector::GetUsedRegsN<ScaleShiftType, CompType, float,
                InType, OutType>()>::Do(ldshiftpg0, ldshiftpg1, ldshiftpg2, ldshiftpg3, loopIdx, loopRange);
            Vector::LoadN<ScaleShiftType, CompType, Vector::GetUsedRegsN<ScaleShiftType, CompType, float, InType,
                OutType>(), Arch>::Do(shift + loopIdx, ldshiftpg0, ldshiftpg1, ldshiftpg2, ldshiftpg3,
                shiftVal0, shiftVal1, shiftVal2, shiftVal3);
        } else {
            Vector::DupFloatN<ScaleShiftType, CompType, Vector::GetUsedRegsN<ScaleShiftType, CompType, float, InType,
                OutType>(), Arch>::Do(0.0f, shiftVal0, shiftVal1, shiftVal2, shiftVal3);
        }
        svbool_t dataPG0;
        svbool_t dataPG1;
        svbool_t dataPG2;
        svbool_t dataPG3;
        Vector::InitPGN<InType, Vector::GetUsedRegsN<InType, ScaleShiftType, CompType, float, OutType>()>::Do(
            dataPG0, dataPG1, dataPG2, dataPG3, loopIdx, loopRange);
        typename Vector::Vec<CompType, Arch>::Type data0;
        typename Vector::Vec<CompType, Arch>::Type data1;
        typename Vector::Vec<CompType, Arch>::Type data2;
        typename Vector::Vec<CompType, Arch>::Type data3;
        Vector::LoadN<InType, CompType, Vector::GetUsedRegsN<InType, ScaleShiftType, CompType, float,
            OutType>(), Arch>::Do(src + loopIdx, dataPG0, dataPG1, dataPG2, dataPG3, data0, data1, data2, data3);
        Vector::SubXN<CompType, Vector::GetUsedRegsN<CompType, InType, ScaleShiftType, float, OutType>(), Arch>::Do(
            pG0, pG1, pG2, pG3, data0, data1, data2, data3,
            vMean0, vMean1, vMean2, vMean3, data0, data1, data2, data3);
        Vector::MlaXN<CompType, Vector::GetUsedRegsN<CompType, InType, ScaleShiftType, float, OutType>(), Arch>::Do(
            pG0, pG1, pG2, pG3, shiftVal0, shiftVal1, shiftVal2, shiftVal3,
            constants0, constants1, constants2, constants3,
            data0, data1, data2, data3, data0, data1, data2, data3);
        svbool_t savePG0;
        svbool_t savePG1;
        svbool_t savePG2;
        svbool_t savePG3;
        Vector::InitPGN<OutType, Vector::GetUsedRegsN<OutType, CompType, InType, ScaleShiftType, float>()>::Do(
            savePG0, savePG1, savePG2, savePG3, loopIdx, loopRange);
        Vector::StoreN<OutType, CompType, Vector::GetUsedRegsN<OutType, CompType, InType, ScaleShiftType, float>(),
            Arch>::Do(savePG0, savePG1, savePG2, savePG3, data0, data1, data2, data3, dst + loopIdx);
    }
};


} // LNorm
} // Detail
} // KuDNN

#endif //  KUDNN_LNORM_FWD_LAST_DEF_HPP