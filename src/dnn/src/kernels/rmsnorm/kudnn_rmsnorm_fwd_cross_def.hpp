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

#ifndef KUDNN_RMSNORM_FWD_CROSS_DEF_HPP
#define KUDNN_RMSNORM_FWD_CROSS_DEF_HPP

#include "kudnn.hpp"
#include "vector_wrappers.hpp"
#include "kernels/rmsnorm/kudnn_rmsnorm_kernels.hpp"

namespace KuDNN {
namespace Detail {
namespace RMSNorm {

template <typename CompType, typename InType, typename OutType, typename ScaleType>
struct RMSNormCrossFWD final {
    static SizeType LoopInc() noexcept
    {
        return 1;
    }
    
    // for Variance calculation accumulated in 1/axisSize. Variance is computed in float precision
    static void Do(SizeType loopIdx, SizeType loopRange, const InType *src, SizeType innerSize,
        svbool_t &ldStatspg0, svbool_t &ldStatspg1, svbool_t &ldStatspg2, svbool_t &ldStatspg3,
        svbool_t &opStatspg0, svbool_t &opStatspg1, svbool_t &opStatspg2, svbool_t &opStatspg3,
        typename Vector::Vec<float>::Type &vVariance0, typename Vector::Vec<float>::Type &vVariance1,
        typename Vector::Vec<float>::Type &vVariance2, typename Vector::Vec<float>::Type &vVariance3)
    {
        Service::Unused(loopRange);
        typename Vector::Vec<float>::Type data0;
        typename Vector::Vec<float>::Type data1;
        typename Vector::Vec<float>::Type data2;
        typename Vector::Vec<float>::Type data3;
        Vector::LoadN<InType, float, Vector::GetUsedRegsN<InType, CompType, OutType, ScaleType, float>()>::Do(
            src + loopIdx * innerSize, ldStatspg0, ldStatspg1, ldStatspg2, ldStatspg3, data0, data1, data2, data3);
        Vector::MlaXN<float,
            Vector::GetUsedRegsN<float, InType, CompType, OutType, ScaleType>()>::Do(
            opStatspg0, opStatspg1, opStatspg2, opStatspg3, vVariance0, vVariance1, vVariance2, vVariance3,
            data0, data1, data2, data3, data0, data1, data2, data3, vVariance0, vVariance1, vVariance2, vVariance3);
    }

    static void DoCalc(SizeType offset, SizeType axisSize,
        const InType *src, SizeType in, SizeType innerSize,
        typename Vector::Vec<float>::Type &vVariance0, typename Vector::Vec<float>::Type &vVariance1,
        typename Vector::Vec<float>::Type &vVariance2, typename Vector::Vec<float>::Type &vVariance3)
    {
        typename Vector::Vec<float>::Type axisSizeVec0;
        typename Vector::Vec<float>::Type axisSizeVec1;
        typename Vector::Vec<float>::Type axisSizeVec2;
        typename Vector::Vec<float>::Type axisSizeVec3;
        Vector::DupFloatN<InType, float, Vector::GetUsedRegsN<InType, CompType, OutType, ScaleType, float>()>::Do(
            axisSize, axisSizeVec0, axisSizeVec1, axisSizeVec2, axisSizeVec3);
        svbool_t ldStatspg0;
        svbool_t ldStatspg1;
        svbool_t ldStatspg2;
        svbool_t ldStatspg3;
        Vector::InitPGN<InType, Vector::GetUsedRegsN<InType, CompType, OutType, ScaleType, float>()>::Do(
            ldStatspg0, ldStatspg1, ldStatspg2, ldStatspg3, in, innerSize);
        svbool_t opStatspg0;
        svbool_t opStatspg1;
        svbool_t opStatspg2;
        svbool_t opStatspg3;
        Vector::InitPGN<float, Vector::GetUsedRegsN<float, InType, CompType, OutType, ScaleType>()>::Do(
            opStatspg0, opStatspg1, opStatspg2, opStatspg3, in, innerSize);
        Vector::DupFloatN<InType, float,
            Vector::GetUsedRegsN<InType, CompType, OutType, ScaleType, float>()>::Do(
            0.0f, vVariance0, vVariance1, vVariance2, vVariance3);
        Service::UnrollFor::Do<1>(axisSize, RMSNormCrossFWD<CompType, InType, OutType, ScaleType>{},
            src + offset, innerSize,
            ldStatspg0, ldStatspg1, ldStatspg2, ldStatspg3,
            opStatspg0, opStatspg1, opStatspg2, opStatspg3,
            vVariance0, vVariance1, vVariance2, vVariance3);
        Vector::DivXN<float,
            Vector::GetUsedRegsN<float, InType, CompType, OutType, ScaleType>()>::Do(
            opStatspg0, opStatspg1, opStatspg2, opStatspg3, vVariance0, vVariance1, vVariance2, vVariance3,
            axisSizeVec0, axisSizeVec1, axisSizeVec2, axisSizeVec3,
            vVariance0, vVariance1, vVariance2, vVariance3);
    }

    // for Variance calculation.
    static void DoStat(bool calculateStats, bool saveStats, SizeType offset, SizeType offsetStats, SizeType axisSize,
        const InType *src, SizeType in, SizeType innerSize, float *variance,
        typename Vector::Vec<float>::Type &vVariance0, typename Vector::Vec<float>::Type &vVariance1,
        typename Vector::Vec<float>::Type &vVariance2, typename Vector::Vec<float>::Type &vVariance3)
    {
        if (calculateStats) {
            DoCalc(offset, axisSize, src, in, innerSize, vVariance0, vVariance1, vVariance2, vVariance3);
            if (saveStats) {
                svbool_t stStatspg0;
                svbool_t stStatspg1;
                svbool_t stStatspg2;
                svbool_t stStatspg3;
                Vector::InitPGN<float,
                    Vector::GetUsedRegsN<float, InType, CompType, OutType, ScaleType>()>::Do(
                    stStatspg0, stStatspg1, stStatspg2, stStatspg3, in, innerSize);
                Vector::StoreN<float, float,
                    Vector::GetUsedRegsN<float, InType, CompType, OutType, ScaleType>()>::Do(
                    stStatspg0, stStatspg1, stStatspg2, stStatspg3, vVariance0, vVariance1, vVariance2, vVariance3,
                    variance + offsetStats);
            }
        } else {
            svbool_t ldStatspg0;
            svbool_t ldStatspg1;
            svbool_t ldStatspg2;
            svbool_t ldStatspg3;
            Vector::InitPGN<float,
                Vector::GetUsedRegsN<float, InType, CompType, OutType, ScaleType>()>::Do(
                ldStatspg0, ldStatspg1, ldStatspg2, ldStatspg3, in, innerSize);
            Vector::LoadN<float, float,
                Vector::GetUsedRegsN<float, InType, CompType, OutType, ScaleType>()>::Do(
                static_cast<const float *>(variance) + offsetStats, ldStatspg0, ldStatspg1,
                ldStatspg2, ldStatspg3, vVariance0, vVariance1, vVariance2, vVariance3);
        }
    }

    // for sqrtVariance
    static void DoSqrt(float eps, SizeType in, SizeType innerSize,
        typename Vector::Vec<float>::Type &vVariance0, typename Vector::Vec<float>::Type &vVariance1,
        typename Vector::Vec<float>::Type &vVariance2, typename Vector::Vec<float>::Type &vVariance3,
        typename Vector::Vec<CompType>::Type &sqrtVarianceComp0,
        typename Vector::Vec<CompType>::Type &sqrtVarianceComp1,
        typename Vector::Vec<CompType>::Type &sqrtVarianceComp2,
        typename Vector::Vec<CompType>::Type &sqrtVarianceComp3)
    {
        typename Vector::Vec<float>::Type eps0;
        typename Vector::Vec<float>::Type eps1;
        typename Vector::Vec<float>::Type eps2;
        typename Vector::Vec<float>::Type eps3;
        Vector::DupFloatN<float, float,
            Vector::GetUsedRegsN<float, InType, CompType, OutType, ScaleType>()>::Do(
            eps, eps0, eps1, eps2, eps3);
        svbool_t opPG0;
        svbool_t opPG1;
        svbool_t opPG2;
        svbool_t opPG3;
        Vector::InitPGN<float,
            Vector::GetUsedRegsN<float, InType, CompType, OutType, ScaleType>()>::Do(
            opPG0, opPG1, opPG2, opPG3, in, innerSize);
        Vector::AddXN<float,
            Vector::GetUsedRegsN<float, InType, CompType, OutType, ScaleType>()>::Do(
            opPG0, opPG1, opPG2, opPG3, eps0, eps1, eps2, eps3,
            vVariance0, vVariance1, vVariance2, vVariance3,
            vVariance0, vVariance1, vVariance2, vVariance3);
        typename Vector::Vec<float>::Type sqrtVariance0;
        typename Vector::Vec<float>::Type sqrtVariance1;
        typename Vector::Vec<float>::Type sqrtVariance2;
        typename Vector::Vec<float>::Type sqrtVariance3;
        Vector::SqrtXN<float,
            Vector::GetUsedRegsN<float, InType, CompType, OutType, ScaleType>()>::Do(
            opPG0, opPG1, opPG2, opPG3, vVariance0, vVariance1, vVariance2, vVariance3,
            sqrtVariance0, sqrtVariance1, sqrtVariance2, sqrtVariance3);
        Vector::ConvertN<float, CompType,
            Vector::GetUsedRegsN<float, InType, CompType, OutType, ScaleType>()>::Do(
            opPG0, opPG1, opPG2, opPG3, sqrtVariance0, sqrtVariance1, sqrtVariance2, sqrtVariance3,
            sqrtVarianceComp0, sqrtVarianceComp1, sqrtVarianceComp2, sqrtVarianceComp3);
    }

    // for norm
    static void Do(SizeType loopIdx, SizeType loopRange, const InType *src, SizeType in, SizeType innerSize,
        const ScaleType *scale, bool useScale,
        typename Vector::Vec<CompType>::Type &sqrtVariance0, typename Vector::Vec<CompType>::Type &sqrtVariance1,
        typename Vector::Vec<CompType>::Type &sqrtVariance2, typename Vector::Vec<CompType>::Type &sqrtVariance3,
        OutType *dst)
    {
        Service::Unused(loopRange);
        typename Vector::Vec<CompType>::Type scale0;
        typename Vector::Vec<CompType>::Type scale1;
        typename Vector::Vec<CompType>::Type scale2;
        typename Vector::Vec<CompType>::Type scale3;
        if (useScale) {
            Vector::DupN<ScaleType, CompType, Vector::GetUsedRegsN<ScaleType, float, InType,
                CompType, OutType>()>::Do(scale[loopIdx], scale0, scale1, scale2, scale3);
        } else {
            Vector::DupFloatN<ScaleType, CompType, Vector::GetUsedRegsN<ScaleType, float, InType,
                CompType, OutType>()>::Do(1.0f, scale0, scale1, scale2, scale3);
        }
        svbool_t ldPG0;
        svbool_t ldPG1;
        svbool_t ldPG2;
        svbool_t ldPG3;
        svbool_t opPG0;
        svbool_t opPG1;
        svbool_t opPG2;
        svbool_t opPG3;
        svbool_t stPG0;
        svbool_t stPG1;
        svbool_t stPG2;
        svbool_t stPG3;
        Vector::InitPGN<InType, Vector::GetUsedRegsN<InType, ScaleType, float, CompType, OutType>()>::Do(
            ldPG0, ldPG1, ldPG2, ldPG3, in, innerSize);
        Vector::InitPGN<CompType, Vector::GetUsedRegsN<CompType, InType, ScaleType, float, OutType>()>::Do(
            opPG0, opPG1, opPG2, opPG3, in, innerSize);
        Vector::DivXN<CompType, Vector::GetUsedRegsN<CompType, InType, ScaleType, float, OutType>()>::Do(
            opPG0, opPG1, opPG2, opPG3, scale0, scale1, scale2, scale3,
            sqrtVariance0, sqrtVariance1, sqrtVariance2, sqrtVariance3, scale0, scale1, scale2, scale3);
        typename Vector::Vec<CompType>::Type svSrc0;
        typename Vector::Vec<CompType>::Type svSrc1;
        typename Vector::Vec<CompType>::Type svSrc2;
        typename Vector::Vec<CompType>::Type svSrc3;
        Vector::LoadN<InType, CompType, Vector::GetUsedRegsN<InType, ScaleType, float, CompType, OutType>()>::Do(
            src + loopIdx * innerSize, ldPG0, ldPG1, ldPG2, ldPG3, svSrc0, svSrc1, svSrc2, svSrc3);
        Vector::MulXN<CompType, Vector::GetUsedRegsN<CompType, InType, ScaleType, float, OutType>()>::Do(
            opPG0, opPG1, opPG2, opPG3, svSrc0, svSrc1, svSrc2, svSrc3,
            scale0, scale1, scale2, scale3, svSrc0, svSrc1, svSrc2, svSrc3);
        Vector::InitPGN<OutType, Vector::GetUsedRegsN<OutType, CompType, InType, ScaleType, float>()>::Do(
            stPG0, stPG1, stPG2, stPG3, in, innerSize);
        Vector::StoreN<OutType, CompType, Vector::GetUsedRegsN<OutType, CompType, InType, ScaleType, float>()>::Do(
            stPG0, stPG1, stPG2, stPG3, svSrc0, svSrc1, svSrc2, svSrc3, dst + loopIdx * innerSize);
    }
};

template <typename CompType, typename InType, typename OutType, typename ScaleType>
void EvaluateCrossAxis(SizeType outerSize, const InType *src, OutType *dst,
    const ScaleType *scale, float *variance,
    bool saveStats, float eps, NormalizationFlags flags, SizeType axisSize, SizeType innerSize) noexcept
{
    bool calculateStats = !static_cast<bool>(flags & NormalizationFlags::USE_GLOBAL_STATS);
    bool useScale = static_cast<bool>(flags & NormalizationFlags::USE_SCALE);
    SizeType loopInc = Vector::UnrollLength<InType, CompType, OutType, ScaleType, float>();
    for (SizeType ou = 0; ou < outerSize; ++ou) {
        for (SizeType in = 0; in < innerSize; in += loopInc) {
            SizeType offset = ou * axisSize * innerSize + in;
            SizeType offsetStats = ou * innerSize + in;
            typename Vector::Vec<float>::Type vVariance0;
            typename Vector::Vec<float>::Type vVariance1;
            typename Vector::Vec<float>::Type vVariance2;
            typename Vector::Vec<float>::Type vVariance3;
            RMSNormCrossFWD<CompType, InType, OutType, ScaleType>::DoStat(calculateStats, saveStats,
                offset, offsetStats, axisSize, src, in, innerSize, variance,
                vVariance0, vVariance1, vVariance2, vVariance3);
            typename Vector::Vec<CompType>::Type sqrtVarianceComp0;
            typename Vector::Vec<CompType>::Type sqrtVarianceComp1;
            typename Vector::Vec<CompType>::Type sqrtVarianceComp2;
            typename Vector::Vec<CompType>::Type sqrtVarianceComp3;
            RMSNormCrossFWD<CompType, InType, OutType, ScaleType>::DoSqrt(eps, in, innerSize,
                vVariance0, vVariance1, vVariance2, vVariance3,
                sqrtVarianceComp0, sqrtVarianceComp1, sqrtVarianceComp2, sqrtVarianceComp3);
            Service::UnrollFor::Do<1>(axisSize, RMSNormCrossFWD<CompType, InType, OutType, ScaleType>{},
                src + offset, in, innerSize, scale, useScale,
                sqrtVarianceComp0, sqrtVarianceComp1, sqrtVarianceComp2, sqrtVarianceComp3,
                dst + offset);
        }
    }
}

} // RMSNorm
} // Detail
} // KuDNN

#endif //  KUDNN_RMSNORM_FWD_CROSS_DEF_HPP