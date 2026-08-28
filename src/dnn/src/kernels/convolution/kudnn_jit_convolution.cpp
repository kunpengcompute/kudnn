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
#include <climits>
#include <omp.h>
#include <algorithm>
#include <arm_sve.h>
#include "kudnn_convolution_base.hpp"
#include "tools/cg_cpu_isa_traits.hpp"
#include "brgemm/kudnn_brgemm_utils.hpp"
#include "kudnn_brgemm_1x1_conv.hpp"
#include "kudnn_conv_const.hpp"
#include "kudnn_jit_dw_conv_kernel.hpp"
#include "kudnn_jit_sve_conv_kernel.hpp"
#include "kudnn_jit_sme_conv_kernel.hpp"
#include "kudnn_jit_sve_x8s8s32x_conv_kernel.hpp"
#include "kudnn_primitive_conf.hpp"
#include "kudnn_wino_conv.hpp"
#include "internal_funcs.hpp"
#include "types/kudnn_tensor_info.hpp"

namespace KuDNN {
using KdnnConvJitKerT = void (*)(KuDNN::KdnnConvJitCallS *);
namespace {
template <typename T, typename U, typename V>
inline typename std::remove_reference<U>::type ThisBlockSize(
    const T offset, const U max, const V block_size)
{
    const T block_boundary = offset + block_size;
    if (block_boundary > max) {
        return max - offset;
    } else {
        return block_size;
    }
}

template <typename U, typename W, typename Y>
inline bool NdIteratorJump(U &cur, const U end, W &x1, const Y &x2)
{
    U max_jump = end - cur;
    U dim_jump = x2 - x1;
    if (dim_jump <= max_jump) {
        x1 = 0;
        cur += dim_jump;
        return true;
    } else {
        cur += max_jump;
        x1 += max_jump;
        return false;
    }
}

template <typename U, typename W, typename Y, typename... Args>
inline bool NdIteratorJump(U &cur, const U end, W &x1, const Y &x2, Args &&...tuple)
{
    if (NdIteratorJump(cur, end, std::forward<Args>(tuple)...)) {
        if (++x1 - x2 == 0) {
            x1 = 0;
            return true;
        }
    }
    return false;
}

inline void KdnnJitConvKerPipeline(KdnnConvJitKerT ker, KuDNN::KdnnConvJitCallS &p,
    const void *src, const void *dst, const void *filt, const void *bias,
    IntType channel, IntType khPadding, IntType reduceWork, IntType loadWork)
{
    p.src = p.srcPrf;
    p.srcPrf = src;
    p.dst = p.dstPrf;
    p.dstPrf = dst;
    p.filt = p.filtPrf;
    p.filtPrf = filt;
    p.bias = p.biasPrf;
    p.biasPrf = bias;
    p.channel = p.channelPrf;
    p.channelPrf = channel;
    // non-positive value of khPadding is allowed, in this case kernel must
    // skip computation part and initialize output by zeroes
    p.khPadding = p.khPaddingPrf;
    p.khPaddingPrf = khPadding;
    p.reduceWork = p.reduceWorkPrf;
    p.reduceWorkPrf = reduceWork;
    p.loadWork = p.loadWorkPrf;
    p.loadWorkPrf = loadWork;
    if (p.src) {
        ker(&p);
    }
}

inline void KdnnJitConvKerPipelineOwThr(KdnnConvJitKerT ker,
    KuDNN::KdnnConvJitCallS &p, const void *src, const void *dst,
    const void *filt, const void *bias, IntType channel, IntType khPadding,
    IntType owb, IntType reduceWork, IntType loadWork, IntType flags)
{
    p.owb = p.owbPrf;
    p.owbPrf = owb;
    p.flags = p.flagsPrf;
    p.flagsPrf = flags;
    KdnnJitConvKerPipeline(ker, p, src, dst, filt, bias, channel, khPadding, reduceWork, loadWork);
}

inline void KdnnJitVCConv3dKerPipeline(KdnnConvJitKerT ker,
    KuDNN::KdnnConvJitCallS &p, const void *src, const void *dst, const void *filt,
    const void *bias, IntType channel, IntType khPadding, IntType kdPadding, IntType reduceWork,
    IntType loadWork)
{
    p.kdPadding = p.kdPaddingPrf;
    p.kdPaddingPrf = kdPadding;
    KdnnJitConvKerPipeline(ker, p, src, dst, filt, bias, channel, khPadding, reduceWork, loadWork);
}

// The special case for the driver with ow-parallelization (FWD)
inline void KdnnJitVCConv3dKerPipelineOwThr(KdnnConvJitKerT ker,
    KuDNN::KdnnConvJitCallS &p, const void *src, const void *dst, const void *filt,
    const void *bias, IntType channel, IntType khPadding, IntType kdPadding, IntType owb,
    IntType reduceWork, IntType loadWork, IntType flags)
{
    p.owb = p.owbPrf;
    p.owbPrf = owb;
    p.flags = p.flagsPrf;
    p.flagsPrf = flags;
    KdnnJitVCConv3dKerPipeline(ker, p, src, dst, filt, bias, channel, khPadding, kdPadding, reduceWork, loadWork);
}

inline bool CheckDwConv(const TensorInfo &src, const TensorInfo &wei, const TensorInfo &dst, IntType kCpuIsa)
{
    if (!Service::EveryoneIs(KuDNN::Element::TypeT::F32, src.GetType(), wei.GetType(), dst.GetType())) {
        return false;
    }
    if (src.GetNumDims() != wei.GetNumDims() - 1) {
        return false;
    }
    if (src.GetDims()[1] != wei.GetDims()[0] || dst.GetDims()[1] != wei.GetDims()[0]) {
        return false;
    }
    // only support 2D case
    if (src.GetNumDims() != KERNEL_DIMS_4) {
        return false;
    }
    // layout is nxc and g/ic/oc with padding is unsupported
    IntType simdW = 0;
    ARMCG::CPUISAT isa = static_cast<ARMCG::CPUISAT>(kCpuIsa);
    if (isa == ARMCG::CPUISAT::VC_512) {
        simdW = ARMCG::CpuIsaTraits<ARMCG::CPUISAT::VC_512>::VLEN / sizeof(float);
    } else if (isa == ARMCG::CPUISAT::VC_256) {
        simdW = ARMCG::CpuIsaTraits<ARMCG::CPUISAT::VC_256>::VLEN / sizeof(float);
    } else {
        return false;
    }
    if (src.GetLayout() == src.GetStandardAXBLayout() && src.GetDims()[1] % simdW != 0) {
        return false;
    }
    return true;
}

inline void PickLoopOrder(KuDNN::KdnnConvJitConfT &jcp, const bool isIntTp)
{
    if (!isIntTp) {
        auto w = (jcp.propKind == KuDNN::Propagation::BACKWARD_DATA) ? jcp.iw : jcp.ow;
        auto h = (jcp.propKind == KuDNN::Propagation::BACKWARD_DATA) ? jcp.ih : jcp.oh;

        jcp.loopOrder = (w <= SMALL_SPATIAL && h <= SMALL_SPATIAL)
                         ? KuDNN::KdnnConvLoopOrderT::KUDNN_LOOP_CWGN
                         : KuDNN::KdnnConvLoopOrderT::KUDNN_LOOP_GNCW;
    } else {
        jcp.loopOrder = KuDNN::KdnnConvLoopOrderT::KUDNN_LOOP_CWGN;
        if (jcp.ngroups > 1) {
            jcp.loopOrder = KuDNN::KdnnConvLoopOrderT::KUDNN_LOOP_NGCW;
            if (jcp.mb < jcp.nThr) {
                jcp.loopOrder = jcp.ndims == KERNEL_DIMS_3 ? KuDNN::KdnnConvLoopOrderT::KUDNN_LOOP_NWCG :
                    KuDNN::KdnnConvLoopOrderT::KUDNN_LOOP_NHWCG;
            }
        }
    }
}

inline KuDNN::Shape GetPadR(const TensorInfo &src, const TensorInfo &weights,
    const TensorInfo &dst, KuDNN::Shape strides, KuDNN::Shape dilates, KuDNN::Shape paddingL)
{
    const bool withGroups = weights.GetNumDims() == src.GetNumDims() + 1;
    IntType ndims = src.GetNumDims();
    IntType id = (ndims == KERNEL_DIMS_5) ? src.GetDims()[KERNEL_DIMS_2] : 1;
    IntType ih = (ndims == KERNEL_DIMS_3) ? 1 : src.GetDims()[ndims - KERNEL_DIMS_2];
    IntType iw = src.GetDims()[ndims - 1];
    IntType od = (ndims == KERNEL_DIMS_5) ? dst.GetDims()[KERNEL_DIMS_2] : 1;
    IntType oh = (ndims == KERNEL_DIMS_3) ? 1 : dst.GetDims()[ndims - KERNEL_DIMS_2];
    IntType ow = dst.GetDims()[ndims - 1];
    IntType kd = (ndims == KERNEL_DIMS_5) ? weights.GetDims()[withGroups + KERNEL_DIMS_2] : 1;
    IntType kh = (ndims == KERNEL_DIMS_3) ? 1 : weights.GetDims()[withGroups + ndims - KERNEL_DIMS_2];
    IntType kw = weights.GetDims()[withGroups + ndims - 1];
    IntType fPad = (ndims == KERNEL_DIMS_5) ? paddingL[0] : 0;
    IntType tPad = (ndims == KERNEL_DIMS_3) ? 0 : paddingL[ndims - KERNEL_DIMS_4];
    IntType lPad = paddingL[ndims - KERNEL_DIMS_3];
    IntType strideD = (ndims == KERNEL_DIMS_5) ? strides[0] : 1;
    IntType strideH = (ndims == KERNEL_DIMS_3) ? 1 : strides[ndims - KERNEL_DIMS_4];
    IntType strideW = strides[ndims - KERNEL_DIMS_3];
 
    IntType dilateD = (ndims == KERNEL_DIMS_5) ? dilates[0] : 0;
    IntType dilateH = (ndims == KERNEL_DIMS_3) ? 0 : dilates[ndims - KERNEL_DIMS_4];
    IntType dilateW = dilates[ndims - KERNEL_DIMS_3];
    IntType extKw = CalculateExtendedFilterSize(kw, dilateW);
    IntType extKh = CalculateExtendedFilterSize(kh, dilateH);
    IntType extKd = CalculateExtendedFilterSize(kd, dilateD);
    IntType rPad = CalculateEndPadding(lPad, ow, iw, strideW, extKw);
    IntType bPad = CalculateEndPadding(tPad, oh, ih, strideH, extKh);
    IntType backPad = CalculateEndPadding(fPad, od, id, strideD, extKd);
    KuDNN::Shape paddingR;
    switch (ndims - KERNEL_DIMS_2) {
        case KERNEL_DIMS_1: {
            paddingR = {static_cast<KuDNN::SizeType>(rPad)};
            break;
        }
        case KERNEL_DIMS_2: {
            paddingR = {static_cast<KuDNN::SizeType>(bPad), static_cast<KuDNN::SizeType>(rPad)};
            break;
        }
        case KERNEL_DIMS_3: {
            paddingR = {static_cast<KuDNN::SizeType>(backPad), static_cast<KuDNN::SizeType>(bPad),
                static_cast<KuDNN::SizeType>(rPad)};
            break;
        }
        default: break;
    }
    return paddingR;
}

inline bool CheckKdnnConvInvalid(const TensorInfo &src, const TensorInfo &weights,
    const TensorInfo &dst, const TensorInfo &bias, KuDNN::Shape strides,
    KuDNN::Shape dilates, KuDNN::Shape paddingL)
{
    KuDNN::Shape paddingR = GetPadR(src, weights, dst, strides, dilates, paddingL);
    Status valSta = KuDNN::Detail::ValidateInput(src, weights, dst, bias,
        strides, dilates, paddingL, paddingR, ConvolutionAlgorithm::DIRECT);
    if (valSta != Status::SUCCESS) {
        return false;
    }
    return true;
}

inline IntType GetSimdW(const Element::TypeT& dTp)
{
#ifdef KUDNNL_920F
    IntType simdW = ARMCG::CpuIsaTraits<ARMCG::CPUISAT::VC_512>::VLEN / sizeof(float);
    if (dTp == Element::TypeT::F16) {
        simdW = ARMCG::CpuIsaTraits<ARMCG::CPUISAT::VC_512>::VLEN / sizeof(__fp16);
    }
#else
    IntType simdW = ARMCG::CpuIsaTraits<ARMCG::CPUISAT::VC_256>::VLEN / sizeof(float);
    if (dTp == Element::TypeT::F16) {
        simdW = ARMCG::CpuIsaTraits<ARMCG::CPUISAT::VC_256>::VLEN / sizeof(__fp16);
    }
#endif
    return simdW;
}

inline bool IsLayoutNxc(const TensorInfo &tensor)
{
    return Service::OneOf(tensor.GetLayout(), KuDNN::Layout::ACB, KuDNN::Layout::ACDB, KuDNN::Layout::ACDEB);
}

inline bool CheckDTypeIsSameF(const TensorInfo &src, const TensorInfo &weights,
    const TensorInfo &dst)
{
    bool res = Service::EveryoneIs(KuDNN::Element::TypeT::F16, src.GetType(), weights.GetType(), dst.GetType()) ||
        Service::EveryoneIs(KuDNN::Element::TypeT::BF16, src.GetType(), weights.GetType(), dst.GetType()) ||
        Service::EveryoneIs(KuDNN::Element::TypeT::F32, src.GetType(), weights.GetType(), dst.GetType());
    return res;
}
 
bool CheckIsaIsNotMatch(IntType cpuIsa, const IntType& simdw, const Element::Type& dTp)
{
    bool res = false;
    if (dTp == KuDNN::Element::TypeT::F32) {
        res = ((cpuIsa == static_cast<IntType>(ARMCG::CPUISAT::VC_256) && (simdw != IC_LEN8)) ||
            (cpuIsa == static_cast<IntType>(ARMCG::CPUISAT::VC_512) && (simdw != IC_LEN16)));
    } else if (dTp == KuDNN::Element::TypeT::F16) {
        res = ((cpuIsa == static_cast<IntType>(ARMCG::CPUISAT::VC_256) && (simdw != IC_LEN16)) ||
            (cpuIsa == static_cast<IntType>(ARMCG::CPUISAT::VC_512) && (simdw != IC_LEN32)));
    } else if (dTp == KuDNN::Element::TypeT::BF16) {
        res = ((cpuIsa == static_cast<IntType>(ARMCG::CPUISAT::VC_256) && (simdw != IC_LEN8)) ||
            (cpuIsa == static_cast<IntType>(ARMCG::CPUISAT::VC_512) && (simdw != IC_LEN16)));
    }
    return res;
}
 
Status ValidateInputDwConv(const TensorInfo &src, const TensorInfo &weights,
    const TensorInfo &dst, const Shape &strides, const Shape &dilates,
    const Shape &paddingL, IntType kCpuIsa)
{
    IntType ndims = src.GetNumDims();
    IntType ih = (ndims == KERNEL_DIMS_3) ? 1 : src.GetDims()[ndims - KERNEL_DIMS_2];
    IntType iw = src.GetDims()[ndims - 1];
    IntType kh = (ndims == KERNEL_DIMS_3) ? 1 : weights.GetDims()[ndims - KERNEL_DIMS_1];
    IntType kw = weights.GetDims()[ndims];
    IntType oh = (ndims == KERNEL_DIMS_3) ? 1 : dst.GetDims()[ndims - KERNEL_DIMS_2];
    IntType ow = dst.GetDims()[ndims - 1];
    IntType tPad = (ndims == KERNEL_DIMS_3) ? 0 : paddingL[ndims - KERNEL_DIMS_4];
    IntType lPad = paddingL[ndims - KERNEL_DIMS_3];
    IntType strideH = (ndims == KERNEL_DIMS_3) ? 1 : strides[ndims - KERNEL_DIMS_4];
    IntType strideW = strides[ndims - KERNEL_DIMS_3];
    IntType dilateH = (ndims == KERNEL_DIMS_3) ? 0 : dilates[ndims - KERNEL_DIMS_4];
    IntType dilateW = dilates[ndims - KERNEL_DIMS_3];
    IntType extKH = CalculateExtendedFilterSize(kh, dilateH);
    IntType extKW = CalculateExtendedFilterSize(kw, dilateW);
    IntType bPad = CalculateEndPadding(tPad, oh, ih, strideH, extKH);
    IntType rPad = CalculateEndPadding(lPad, ow, iw, strideW, extKW);
    bool unsupport = false || extKH <= tPad || extKH <= bPad || extKW <= lPad || extKW <= rPad;
    if (unsupport) {
        return Status::UNSUPPORTED;
    }
 
    IntType urW = 0;
    IntType chBlock = 0;
    IntType nbChBlocking = 0;
    ARMCG::CPUISAT isa = static_cast<ARMCG::CPUISAT>(kCpuIsa);
    if (isa == ARMCG::CPUISAT::VC_512) {
        urW = std::min(ow, UR_W_6);
        chBlock = ARMCG::CpuIsaTraits<ARMCG::CPUISAT::VC_512>::VLEN / sizeof(float);
        nbChBlocking = NB_CH_BLKING_4;
    } else if (isa == ARMCG::CPUISAT::VC_256) {
        urW = std::min(ow, UR_W_4);
        chBlock = ARMCG::CpuIsaTraits<ARMCG::CPUISAT::VC_256>::VLEN / sizeof(float);
        nbChBlocking = NB_CH_BLKING_3;
    } else {
        return Status::UNSUPPORTED;
    }
    IntType urWTail = ow % urW;
    IntType rPadNoTail = std::max<IntType>(CalculateEndPadding(lPad, ow - urWTail, iw, strideW, extKW), 0);
    if (rPadNoTail > urW || lPad > urW) {
        return Status::UNSUPPORTED;
    }
 
    if (src.GetLayout() != src.GetStandardAXBLayout()) {
        IntType typeSizeIn = sizeof(float);
        IntType typeSizeOut = sizeof(float);
        const SizeType maxChOff = static_cast<SizeType>(nbChBlocking - 1) * chBlock;
        constexpr SizeType maxExOff = 0;
        const SizeType maxICOff = maxChOff * ih * iw;
        const SizeType maxIWIdx = static_cast<SizeType>(urW - 1) * strideW + (extKW - 1);
        const SizeType maxIWOff = maxIWIdx * chBlock;
        const SizeType maxInpOff = (maxICOff + maxIWOff + maxExOff) * typeSizeIn;
 
        const SizeType maxOCOff = maxChOff * oh * ow;
        const SizeType maxOWOff = static_cast<SizeType>(urW - 1) * chBlock;
        const SizeType maxOutOff = (maxOCOff + maxOWOff + maxExOff) * typeSizeOut;
        if (maxOutOff > INT_MAX || maxInpOff > INT_MAX) {
            return Status::UNSUPPORTED;
        }
    }
 
    if (ARMCG::GetCpu().GetVCLen() == ARMCG::Util::VCLen::VC_512
        && isa == ARMCG::CPUISAT::VC_256) {
        return Status::UNSUPPORTED;
    }
    return Status::SUCCESS;
}
}

namespace Detail {
class KdnnJitConvFwdKerFactory {
public:
    explicit KdnnJitConvFwdKerFactory() {}
    ~KdnnJitConvFwdKerFactory() {}
public:
    static void GetFwdJitConvKer(std::unique_ptr<ARMCG::JitGenerator>& jitKerBase, Element::TypeT dtp,
        bool isDepthwise, bool isUseMC, const KdnnConvJitConfT &ajcp)
    {
#ifdef KUDNNL_920F
        constexpr ARMCG::CPUISAT isa = ARMCG::CPUISAT::VC_512;
        if (isUseMC) {
            jitKerBase.reset(new JitConvMCKernel(ajcp));
        }
#else
        constexpr ARMCG::CPUISAT isa = ARMCG::CPUISAT::VC_256;
#endif
        if (isDepthwise) {
            jitKerBase.reset(new KdnnJitDwConvFwdKernel<isa>(ajcp));
        } else if (!isUseMC) {
            if (dtp == Element::TypeT::F32) {
                jitKerBase.reset(new KdnnJitConvFwdKernel<isa, Element::TypeT::F32>(ajcp));
            } else if (dtp == Element::TypeT::F16) {
                jitKerBase.reset(new KdnnJitConvFwdKernel<isa, Element::TypeT::F16>(ajcp));
            } else if (dtp == Element::TypeT::BF16) {
                jitKerBase.reset(new KdnnJitConvFwdKernel<isa, Element::TypeT::BF16>(ajcp));
            } else if (Service::OneOf(dtp, Element::TypeT::S8, Element::TypeT::U8)) {
                jitKerBase.reset(new KdnnJitX8S8S32ConvFwdKernel(ajcp));
            } else {
                jitKerBase.reset(nullptr);
            }
        }
    }
};

class ConvolutionJITFWDImpl final : public ConvolutionFWDImplBase {
public:
    explicit ConvolutionJITFWDImpl(IntType isa, const TensorInfo &src, const TensorInfo &weights,
        const TensorInfo &dst, const TensorInfo &bias, KuDNN::Shape strides, KuDNN::Shape paddingL,
        KuDNN::Shape dilates, KuDNN::Propagation prop, IntType nthreads, bool isLayoutBlock) noexcept(false)
    {
        InitConf(isa, src, weights, dst, bias, strides, paddingL, dilates, prop, nthreads, isLayoutBlock);
    }

    explicit ConvolutionJITFWDImpl(const ConvolutionJITFWDImpl &other) noexcept(false)
        : jcp_(other.jcp_) {}

    explicit ConvolutionJITFWDImpl(ConvolutionJITFWDImpl &&) = delete;
    ConvolutionJITFWDImpl& operator=(const ConvolutionJITFWDImpl &other) = delete;
    ConvolutionJITFWDImpl &operator=(ConvolutionJITFWDImpl &&) = delete;

    struct ConvKerParam {
        Element::TypeT srcType;
        Element::TypeT weiType;
        Element::TypeT dstType;
        IntType ndims;
    };

    struct IntConvKerParamHash {
        SizeType operator()(const ConvKerParam& paras) const
        {
            return static_cast<SizeType>(paras.srcType) * NUM_10 * NUM_10 * NUM_10 * NUM_10 +
                static_cast<SizeType>(paras.weiType) * NUM_10 * NUM_10 * NUM_10 +
                static_cast<SizeType>(paras.dstType) * NUM_10 * NUM_10 + paras.ndims * NUM_10;
        }
    };

    struct IntConvKerParamEqual {
        bool operator()(const ConvKerParam& para0, const ConvKerParam& para1) const
        {
            return para0.srcType == para1.srcType && para0.dstType == para1.dstType &&
                para0.weiType == para1.weiType && para0.ndims == para1.ndims;
        }
    };
    using ConvRunFunc = void (ConvolutionJITFWDImpl::*)(const void*, const void*, void*, const void*);
    const std::unordered_map<ConvKerParam, ConvRunFunc, IntConvKerParamHash,
        IntConvKerParamEqual> CONV_KER_MAP = {
        {{Element::TypeT::S8, Element::TypeT::S8, Element::TypeT::S8, DIM_3},
            &ConvolutionJITFWDImpl::ExecuteForward1D4Int<int8_t, int8_t>},
        {{Element::TypeT::S8, Element::TypeT::S8, Element::TypeT::U8, DIM_3},
            &ConvolutionJITFWDImpl::ExecuteForward1D4Int<int8_t, uint8_t>},
        {{Element::TypeT::S8, Element::TypeT::S8, Element::TypeT::S32, DIM_3},
            &ConvolutionJITFWDImpl::ExecuteForward1D4Int<int8_t, int32_t>},
        {{Element::TypeT::S8, Element::TypeT::S8, Element::TypeT::F32, DIM_3},
            &ConvolutionJITFWDImpl::ExecuteForward1D4Int<int8_t, float>},
        {{Element::TypeT::U8, Element::TypeT::S8, Element::TypeT::S8, DIM_3},
            &ConvolutionJITFWDImpl::ExecuteForward1D4Int<uint8_t, int8_t>},
        {{Element::TypeT::U8, Element::TypeT::S8, Element::TypeT::U8, DIM_3},
            &ConvolutionJITFWDImpl::ExecuteForward1D4Int<uint8_t, uint8_t>},
        {{Element::TypeT::U8, Element::TypeT::S8, Element::TypeT::S32, DIM_3},
            &ConvolutionJITFWDImpl::ExecuteForward1D4Int<uint8_t, int32_t>},
        {{Element::TypeT::U8, Element::TypeT::S8, Element::TypeT::F32, DIM_3},
            &ConvolutionJITFWDImpl::ExecuteForward1D4Int<uint8_t, float>},
        {{Element::TypeT::S8, Element::TypeT::S8, Element::TypeT::S8, DIM_4},
            &ConvolutionJITFWDImpl::ExecuteForward2D4Int<int8_t, int8_t>},
        {{Element::TypeT::S8, Element::TypeT::S8, Element::TypeT::U8, DIM_4},
            &ConvolutionJITFWDImpl::ExecuteForward2D4Int<int8_t, uint8_t>},
        {{Element::TypeT::S8, Element::TypeT::S8, Element::TypeT::S32, DIM_4},
            &ConvolutionJITFWDImpl::ExecuteForward2D4Int<int8_t, int32_t>},
        {{Element::TypeT::S8, Element::TypeT::S8, Element::TypeT::F32, DIM_4},
            &ConvolutionJITFWDImpl::ExecuteForward2D4Int<int8_t, float>},
        {{Element::TypeT::U8, Element::TypeT::S8, Element::TypeT::S8, DIM_4},
            &ConvolutionJITFWDImpl::ExecuteForward2D4Int<uint8_t, int8_t>},
        {{Element::TypeT::U8, Element::TypeT::S8, Element::TypeT::U8, DIM_4},
            &ConvolutionJITFWDImpl::ExecuteForward2D4Int<uint8_t, uint8_t>},
        {{Element::TypeT::U8, Element::TypeT::S8, Element::TypeT::S32, DIM_4},
            &ConvolutionJITFWDImpl::ExecuteForward2D4Int<uint8_t, int32_t>},
        {{Element::TypeT::U8, Element::TypeT::S8, Element::TypeT::F32, DIM_4},
            &ConvolutionJITFWDImpl::ExecuteForward2D4Int<uint8_t, float>},
        {{Element::TypeT::S8, Element::TypeT::S8, Element::TypeT::S8, DIM_5},
            &ConvolutionJITFWDImpl::ExecuteForward3D4Int<int8_t, int8_t>},
        {{Element::TypeT::S8, Element::TypeT::S8, Element::TypeT::U8, DIM_5},
            &ConvolutionJITFWDImpl::ExecuteForward3D4Int<int8_t, uint8_t>},
        {{Element::TypeT::S8, Element::TypeT::S8, Element::TypeT::S32, DIM_5},
            &ConvolutionJITFWDImpl::ExecuteForward3D4Int<int8_t, int32_t>},
        {{Element::TypeT::S8, Element::TypeT::S8, Element::TypeT::F32, DIM_5},
            &ConvolutionJITFWDImpl::ExecuteForward3D4Int<int8_t, float>},
        {{Element::TypeT::U8, Element::TypeT::S8, Element::TypeT::S8, DIM_5},
            &ConvolutionJITFWDImpl::ExecuteForward3D4Int<uint8_t, int8_t>},
        {{Element::TypeT::U8, Element::TypeT::S8, Element::TypeT::U8, DIM_5},
            &ConvolutionJITFWDImpl::ExecuteForward3D4Int<uint8_t, uint8_t>},
        {{Element::TypeT::U8, Element::TypeT::S8, Element::TypeT::S32, DIM_5},
            &ConvolutionJITFWDImpl::ExecuteForward3D4Int<uint8_t, int32_t>},
        {{Element::TypeT::U8, Element::TypeT::S8, Element::TypeT::F32, DIM_5},
            &ConvolutionJITFWDImpl::ExecuteForward3D4Int<uint8_t, float>},
        {{Element::TypeT::F32, Element::TypeT::F32, Element::TypeT::F32, DIM_3},
            &ConvolutionJITFWDImpl::ExecuteForward1D<float>},
        {{Element::TypeT::F16, Element::TypeT::F16, Element::TypeT::F16, DIM_3},
            &ConvolutionJITFWDImpl::ExecuteForward1D<__fp16>},
        {{Element::TypeT::BF16, Element::TypeT::BF16, Element::TypeT::BF16, DIM_3},
            &ConvolutionJITFWDImpl::ExecuteForward1D<__bf16>},
        {{Element::TypeT::F32, Element::TypeT::F32, Element::TypeT::F32, DIM_4},
            &ConvolutionJITFWDImpl::ExecuteForward2D<float>},
        {{Element::TypeT::F16, Element::TypeT::F16, Element::TypeT::F16, DIM_4},
            &ConvolutionJITFWDImpl::ExecuteForward2D<__fp16>},
        {{Element::TypeT::BF16, Element::TypeT::BF16, Element::TypeT::BF16, DIM_4},
            &ConvolutionJITFWDImpl::ExecuteForward2D<__bf16>},
        {{Element::TypeT::F32, Element::TypeT::F32, Element::TypeT::F32, DIM_5},
            &ConvolutionJITFWDImpl::ExecuteForward3D<float>},
        {{Element::TypeT::F16, Element::TypeT::F16, Element::TypeT::F16, DIM_5},
            &ConvolutionJITFWDImpl::ExecuteForward3D<__fp16>},
        {{Element::TypeT::BF16, Element::TypeT::BF16, Element::TypeT::BF16, DIM_5},
            &ConvolutionJITFWDImpl::ExecuteForward3D<__bf16>}
    };

    void Run(const void *src, const void *wei,
        void *dst, const void *bias) noexcept(false) override
    {
        if (bcd_.useBrgconv) {
            if (bcd_.bcp.srcDt == Element::TypeT::F32 && bcd_.bcp.weiDt == Element::TypeT::F32
                && bcd_.bcp.dstDt == Element::TypeT::F32) {
                KdnnBrgconvExecute(bcd_, static_cast<const float *>(src),
                    static_cast<const float *>(wei), static_cast<float *>(dst),
                    static_cast<const float *>(bias));
            }
            return;
        }
        bool isUseMC = false;
#ifdef KUDNNL_920F
        isUseMC = UseMC();
        if (isUseMC) {
            ExecuteForwardMC(src, wei, dst, bias);
        }
#endif
        if (!isUseMC) {
            ConvKerParam convParam = {jcp_.srcDt, jcp_.weiDt, jcp_.dstDt, jcp_.ndims};
            if (CONV_KER_MAP.find(convParam) != CONV_KER_MAP.end()) {
                auto exeCuteForwardFunc = CONV_KER_MAP.at(convParam);
                (this->*exeCuteForwardFunc)(src, wei, dst, bias);
            } else {
                throw KuDNN::Service::Unsupported{"unsupported int conv ker param"};
            }
        }
    }

#ifdef KUDNNL_920F
    void Img2col(const float *src, float *dst);

    inline static void DataAssign(const float *src, float *dst,
        IntType mb, IntType ic, IntType id, IntType ih,
        IntType od, IntType oh, IntType kd, IntType kh, IntType kw,
        IntType sizeIn, IntType sizeOut, KdnnConvJitConfT &jcp);

    inline static void ContiguousCopy(const float *src, float *dst,
        IntType mb, IntType ic, IntType id, IntType ih,
        IntType od, IntType oh, IntType kd, IntType kh, IntType kw,
        IntType sizeIn, IntType sizeOut, KdnnConvJitConfT &jcp);

    inline static void StrideCopy(const float *src, float *dst,
        IntType mb, IntType ic, IntType id, IntType ih,
        IntType od, IntType oh, IntType kd, IntType kh, IntType kw,
        IntType sizeIn, IntType sizeOut, KdnnConvJitConfT &jcp);

    static void DataCopy(float *dst, const float *src, IntType len);
    static void DataCopyWithStride(float *dst, const float *src, IntType len, IntType stride);
    void ExecuteForwardMC(const void *src, const void *wei, void *dst, const void *bias) noexcept;
    bool AccelerateGemm();
    bool CanUseDirectMC();
    void GemmAccelarate(const float *src, const float *wei, float *dst, const float *bias) noexcept;
    void DirectMC(const float *src, const float *wei, float *dst, const float *bias) noexcept;
    void DirectMCWithStride(const float *src, const float *wei, float *dst, const float *bias) noexcept;
    void DoMCGemmPack(const float *src);
    void DoMCGemm(const float *src, const float *wei, float *dst,
        const float *bias, bool isNoImg2Col, bool isNeedDopad);
#endif
    static void DataDup(float *src, int32_t dupVal, IntType len);
    template<typename SrcDataType, typename DstDataType>
    void ExecuteForward1D4Int(const void *src, const void *wei, void *dst, const void *bias) noexcept(false);
    template<typename SrcDataType, typename DstDataType>
    void ExecuteForward2D4Int(const void *src, const void *wei, void *dst, const void *bias) noexcept(false);
    template<typename SrcDataType, typename DstDataType>
    void ExecuteForward3D4Int(const void *src, const void *wei, void *dst, const void *bias) noexcept(false);

    template<typename DataType>
    void ExecuteForward1D(const void *src, const void *wei, void *dst, const void *bias) noexcept(false);
    template<typename DataType>
    void ExecuteForward2D(const void *src, const void *wei, void *dst, const void *bias) noexcept(false);
    template<typename DataType>
    void ExecuteForward3D(const void *src, const void *wei, void *dst, const void *bias) noexcept(false);
    void ExecuteForwardDW(const void *src, const void *wei, void *dst, const void *bias) noexcept(false);

    bool CreateKernel(const Element::TypeT& srcDtp)
    {
        if (bcd_.useBrgconv) {
            Status s = KdnnBrgconvCreateKernel(bcd_);
            return s == Status::SUCCESS ? true : false;
        }
        bool isUseMC = false;
#ifdef KUDNNL_920F
        isUseMC = UseMC();
#endif
        KdnnJitConvFwdKerFactory::GetFwdJitConvKer(jit_conv_ker_base_, srcDtp, jcp_.isDepthwise, isUseMC, jcp_);
        return jit_conv_ker_base_->CreateKernel();
    }

    KdnnConvJitKerT GetKernel()
    {
        if (jit_conv_ker_base_.get()) {
            return jit_conv_ker_base_->template GetAsmCode<KdnnConvJitKerT>();
        } else {
            return nullptr;
        }
    }

#ifdef KUDNNL_920F
    bool UseMC()
    {
        return Service::EveryoneIs(Element::TypeT::F32, jcp_.srcDt, jcp_.weiDt, jcp_.dstDt)
            && jcp_.ngroups == 1 && Service::OneOf(jcp_.ndims, KERNEL_DIMS_4, KERNEL_DIMS_5)
            && ((jcp_.srcTag == KuDNN::Layout::ABCD && jcp_.dstTag == KuDNN::Layout::ABCD)
            || (jcp_.srcTag == KuDNN::Layout::ABCDE && jcp_.dstTag == KuDNN::Layout::ABCDE))
            && jcp_.ocWithoutPadding % NUM_16 == 0 && jcp_.dilateH == 0 && jcp_.dilateW == 0;
    }
#endif
    Element::TypeT GetTensorDtp(const KdnnConvJitConfT& jcp)
    {
        Element::TypeT dtp = Service::EveryoneIs(Element::TypeT::F32, jcp.srcDt, jcp.weiDt, jcp.dstDt) ?
            Element::TypeT::F32 : (((Service::EveryoneIs(Element::TypeT::F16, jcp.srcDt, jcp.weiDt, jcp.dstDt) ?
            Element::TypeT::F16 : ((Service::EveryoneIs(Element::TypeT::BF16, jcp.srcDt, jcp.weiDt, jcp.dstDt) ?
            Element::TypeT::BF16 : (Service::OneOf(jcp.srcDt, Element::TypeT::S8, Element::TypeT::U8) ?
            Element::TypeT::S8 : Element::TypeT::UNDEFINED))))));
        return dtp;
    }

    void GetBestCutWay(KdnnConvJitConfT& jcp)
    {
        IntType maxNbOc = KERNEL_C_LEN2;
        IntType tRegNum = REG_INDEX25;
        if (((jcp.kw == 1 && jcp.oc > KERNEL_C_LEN16) || (jcp.kw > 1 && jcp.lPad > 0 && jcp.strideW > 1)) &&
            (jcp.nbOc % KERNEL_C_LEN4 == 0)) {
            maxNbOc = KERNEL_C_LEN4;
            tRegNum = REG_INDEX31;
        } else if ((jcp.nbOc % KERNEL_C_LEN2 != 0) && (jcp.nbOc > maxNbOc)) {
            maxNbOc = jcp.nbOc > KERNEL_C_LEN8 ? KERNEL_C_LEN4 : jcp.nbOc;
            tRegNum = REG_INDEX31;
        }
        if (jcp.id > 1) {
            maxNbOc = KERNEL_C_LEN2;
            tRegNum = REG_INDEX31;
        }
        jcp.kernelKind = KuDNN::KdnnConvKernelKindT::KUDNN_EXPL_BCAST;
        jcp.nbIcBlocking = 1;
        IntType bestNbOcBlocking = 1;
        for (IntType i = std::min<IntType>(jcp.nbOc, maxNbOc); i > 0; i--) {
            if (jcp.nbOc % i == 0) {
                bestNbOcBlocking = i;
                break;
            }
        }
        jcp.nbOcBlocking = bestNbOcBlocking;
        if (jcp.srcDt == Element::TypeT::BF16) {
            jcp.nbOcBlocking = 1;
            tRegNum = REG_INDEX31;
        }
        jcp.urW = std::min<IntType>(jcp.ow, tRegNum / (jcp.nbOcBlocking + 1));
        if (jcp.lPad > jcp.urW) {
            jcp.nbOcBlocking = 1;
            jcp.urW = std::min<IntType>(jcp_.ow, tRegNum / (jcp.nbOcBlocking + 1));
        }
        if (jcp.nbOcBlocking == 1) {
            jcp.urW = std::min<IntType>(jcp.ow, KERNEL_UR_W_LEN8);
        }
        // KUDNN_CONT_BCAST: urW <= 8 && nbOcBlocking > 1
        // KUDNN_EXPL_BCAST: urW > 8 && nbOcBlocking == 1
        // perhaps some case get better performance: urW > 8 && nbOcBlocking == 2
        if (jcp.icBlock == IC_LEN4 && jcp.ocBlock == IC_LEN4) {
            jcp.kernelKind = KuDNN::KdnnConvKernelKindT::KUDNN_EMBD_BCAST;
        } else if (!jcp.is1stConv && jcp.urW <= KERNEL_UR_W_LEN8) {
            jcp.kernelKind = KuDNN::KdnnConvKernelKindT::KUDNN_CONT_BCAST;
        }
    }
 
    void DealDwInitConf(KdnnConvJitConfT& jcp, IntType cpuIsa, IntType fSimdW,
        const TensorInfo &dstTensor, bool isDLayoutNxc)
    {
        jcp.loopOrder = KuDNN::KdnnConvLoopOrderT::KUDNN_LOOP_NGCW;
        ARMCG::CPUISAT isa = static_cast<ARMCG::CPUISAT>(cpuIsa);
        jcp.urW = isa == ARMCG::CPUISAT::VC_512 ? UR_W_6 : isa == ARMCG::CPUISAT::VC_256 ? UR_W_4 : UR_W_3;
        jcp.urW = std::min<IntType>(jcp.urW, jcp.ow);
        jcp.chBlock = fSimdW;
        jcp.nbCh = Math::CeilDiv(dstTensor.GetDims()[1], jcp.chBlock);
        jcp.nbChBlocking = isa == ARMCG::CPUISAT::VC_512 ? NB_CH_BLKING_4 :
            isa == ARMCG::CPUISAT::VC_256 ? NB_CH_BLKING_3 : NB_CH_BLKING_2;
        if (jcp.nbCh < jcp.nbChBlocking) {
            jcp.nbChBlocking = jcp.nbCh;
        }
 
        if (isDLayoutNxc) {
            jcp.loopOrder = KuDNN::KdnnConvLoopOrderT::KUDNN_LOOP_NHWCG;
            bool cacheAliasing = (jcp.ngroups * jcp.iw * jcp.typeSizeIn) % CACHE_ALIAS_1024 == 0;
            if (cacheAliasing) {
                const IntType limit = jcp.ow > LIMIT_7 ? LIMIT_7 : LIMIT_4;
                jcp.urW = std::min<IntType>(jcp.urW, limit);
            }
        }
        jcp.urWTail = jcp.ow % jcp.urW;
        jcp.ngroups = RndUp(jcp.ngroups, fSimdW);
    }
 
    void SetJcpTypeSize(KdnnConvJitConfT& jcp)
    {
        jcp.typeSizeIn = sizeof(float);
        jcp.typeSizeOut = sizeof(float);
        jcp.typeSizeBia = sizeof(float);
        if (Service::OneOf(jcp.srcDt, Element::TypeT::F16, Element::TypeT::BF16) &&
            Service::OneOf(jcp.dstDt, Element::TypeT::F16, Element::TypeT::BF16)) {
            jcp.typeSizeIn = sizeof(__fp16);
            jcp.typeSizeOut = sizeof(__fp16);
        } else if (Service::OneOf(jcp.srcDt, Element::TypeT::S8, Element::TypeT::U8) &&
            Service::OneOf(jcp.dstDt, Element::TypeT::S8, Element::TypeT::U8)) {
            jcp.typeSizeIn = sizeof(int8_t);
            jcp.typeSizeOut = sizeof(int8_t);
            jcp.typeSizeBia = sizeof(float);
        } else if (Service::OneOf(jcp.srcDt, Element::TypeT::S8, Element::TypeT::U8) &&
            Service::OneOf(jcp.dstDt, Element::TypeT::S32, Element::TypeT::F32)) {
            jcp.typeSizeIn = sizeof(int8_t);
            jcp.typeSizeOut = sizeof(int32_t);
            jcp.typeSizeBia = sizeof(int32_t);
        }
    }

    void SetJcpInfoFromTensor(KdnnConvJitConfT& jcp, IntType kCpuIsa,
        const TensorInfo &src, const TensorInfo &weights, const TensorInfo &dst,
        const TensorInfo &bias, KuDNN::Shape strides, KuDNN::Shape paddingL,
        KuDNN::Shape dilates, KuDNN::Propagation prop, IntType nthreads)
    {
        jcp.withBias = bias.GetDims()[0] != 0;
        const bool withGroups = weights.GetNumDims() == src.GetNumDims() + 1;
        jcp.isWithGroups = withGroups;
        IntType ndims = src.GetNumDims();
        jcp.nThr = jcp.alignedThreads = nthreads;
        jcp.ndims = ndims;
        jcp.propKind = prop;
        jcp.ngroups = withGroups ? weights.GetDims()[0] : 1;
        const bool isDepthwise = CheckDwConv(src, weights, dst, kCpuIsa);
        jcp.isDepthwise = isDepthwise;
        jcp.mb = src.GetDims()[0];
        jcp.oc = isDepthwise ? dst.GetDims()[1] : dst.GetDims()[1] / jcp.ngroups;
        jcp.ocWithoutPadding = jcp.oc;
        jcp.ic = isDepthwise ? src.GetDims()[1] : src.GetDims()[1] / jcp.ngroups;
        jcp.icWithoutPadding = jcp.ic;
        jcp.id = (ndims == KERNEL_DIMS_5) ? src.GetDims()[KERNEL_DIMS_2] : 1;
        jcp.ih = (ndims == KERNEL_DIMS_3) ? 1 : src.GetDims()[ndims - KERNEL_DIMS_2];
        jcp.iw = src.GetDims()[ndims - 1];
        jcp.od = (ndims == KERNEL_DIMS_5) ? dst.GetDims()[KERNEL_DIMS_2] : 1;
        jcp.oh = (ndims == KERNEL_DIMS_3) ? 1 : dst.GetDims()[ndims - KERNEL_DIMS_2];
        jcp.ow = dst.GetDims()[ndims - 1];
        jcp.kd = (ndims == KERNEL_DIMS_5) ? weights.GetDims()[withGroups + KERNEL_DIMS_2] : 1;
        jcp.kh = (ndims == KERNEL_DIMS_3) ? 1 : weights.GetDims()[withGroups + ndims - KERNEL_DIMS_2];
        jcp.kw = weights.GetDims()[withGroups + ndims - 1];
 
        jcp.fPad = (ndims == KERNEL_DIMS_5) ? paddingL[0] : 0;
        jcp.tPad = (ndims == KERNEL_DIMS_3) ? 0 : paddingL[ndims - KERNEL_DIMS_4];
        jcp.lPad = paddingL[ndims - KERNEL_DIMS_3];

        jcp.strideD = (ndims == KERNEL_DIMS_5) ? strides[0] : 1;
        jcp.strideH = (ndims == KERNEL_DIMS_3) ? 1 : strides[ndims - KERNEL_DIMS_4];
        jcp.strideW = strides[ndims - KERNEL_DIMS_3];
        jcp.dilateD = (ndims == KERNEL_DIMS_5) ? dilates[0] : 0;
        jcp.dilateH = (ndims == KERNEL_DIMS_3) ? 0 : dilates[ndims - KERNEL_DIMS_4];
        jcp.dilateW = dilates[ndims - KERNEL_DIMS_3];

        IntType extKw = CalculateExtendedFilterSize(jcp.kw, jcp.dilateW);
        IntType extKh = CalculateExtendedFilterSize(jcp.kh, jcp.dilateH);
        IntType extKd = CalculateExtendedFilterSize(jcp.kd, jcp.dilateD);
        jcp.rPad = CalculateEndPadding(jcp.lPad, jcp.ow, jcp.iw, jcp.strideW, extKw);
        jcp.bPad = CalculateEndPadding(jcp.tPad, jcp.oh, jcp.ih, jcp.strideH, extKh);
        jcp.backPad = CalculateEndPadding(jcp.fPad, jcp.od, jcp.id, jcp.strideD, extKd);
        // post-op not supported
        jcp.withSum = 0;
        jcp.srcTag = src.GetLayout();
        jcp.dstTag = dst.GetLayout();
        jcp.srcDt = src.GetType();
        jcp.weiDt = weights.GetType();
        jcp.dstDt = dst.GetType();
        jcp.biaDt = bias.GetType();
    }

    Status InitConf(IntType kCpuIsa,
        const TensorInfo &src, const TensorInfo &weights, const TensorInfo &dst,
        const TensorInfo &bias, KuDNN::Shape strides, KuDNN::Shape paddingL,
        KuDNN::Shape dilates, KuDNN::Propagation prop, IntType nthreads, bool isLayoutBlock) noexcept
    {
        using namespace ARMCG;
        if (isLayoutBlock) {
            Status s = KdnnBrgconvInitConf(bcd_, static_cast<ARMCG::CPUISAT>(kCpuIsa), src, weights, dst, bias,
                strides, dilates, paddingL, prop, nthreads);
            if (s == Status::SUCCESS) {
                CreateKernel(bcd_.bcp.srcDt);
                return Status::SUCCESS;
            }
        }
        const IntType regs = REG_INDEX28;
        SetJcpInfoFromTensor(jcp_, kCpuIsa, src, weights, dst, bias, strides, paddingL, dilates, prop, nthreads);
#ifdef KUDNNL_920F
        if (UseMC()) {
            if (jcp_.ndims == KERNEL_DIMS_4 && jcp_.strideH == 1 && jcp_.lPad == 0 &&
                jcp_.dilateH == 0 && jcp_.kh == 1) {
                IntType realOh = jcp_.ih - jcp_.kh + 1;
                if (realOh != jcp_.oh) {
                    jcp_.lPad = realOh - jcp_.oh;
                }
                IntType extKw = CalculateExtendedFilterSize(jcp_.kw, jcp_.dilateW);
                jcp_.rPad = CalculateEndPadding(jcp_.lPad, jcp_.ow, jcp_.iw, jcp_.strideW, extKw);
            }
            IntType sizeD = jcp_.id + jcp_.fPad + jcp_.backPad;
            IntType sizeH = jcp_.ih + jcp_.tPad + jcp_.bPad;
            IntType sizeW = jcp_.iw + jcp_.lPad + jcp_.rPad;
            padSrc.resize(jcp_.mb * sizeD * sizeH * sizeW * jcp_.ic, 0.0f);
            if (CanUseDirectMC()) {
                jcp_.ver = KuDNN::KdnnConvVersionT::KUDNN_VER_MC_DIRECT;
            } else {
                colTmp.resize(jcp_.mb * jcp_.od * jcp_.oh * jcp_.ow * jcp_.kd * jcp_.kh * jcp_.kw * jcp_.ic, 0.0f);
                jcp_.ver = KuDNN::KdnnConvVersionT::KUDNN_VER_MC_GEMM;
            }
            Element::TypeT dtp = GetTensorDtp(jcp_);
            CreateKernel(dtp);
            return Status::SUCCESS;
        }
#endif
        bool isSrcF16 = jcp_.srcDt == Element::TypeT::F16;
        bool isGroupsOne = jcp_.ngroups == 1;
#ifdef KUDNNL_920F
        IntType tIcLen = IC_LEN16;
        IntType fullSimdW = CpuIsaTraits<CPUISAT::VC_512>::VLEN / sizeof(float);
        if (isSrcF16) {
            fullSimdW = CpuIsaTraits<CPUISAT::VC_512>::VLEN / sizeof(__fp16);
        }
#else
        IntType tIcLen = IC_LEN8;
        IntType fullSimdW = CpuIsaTraits<CPUISAT::VC_256>::VLEN / sizeof(float);
        if (isSrcF16) {
            tIcLen = IC_LEN16;
            fullSimdW = CpuIsaTraits<CPUISAT::VC_256>::VLEN / sizeof(__fp16);
        }
#endif
        /* 1st convolution check */
        jcp_.is1stConv = (jcp_.ic < tIcLen) && isGroupsOne;
        jcp_.simdW = fullSimdW;
        bool isIntType = Service::OneOf(jcp_.srcDt, Element::TypeT::U8, Element::TypeT::S8) &&
            Service::OneOf(jcp_.dstDt, Element::TypeT::U8, Element::TypeT::S8, Element::TypeT::S32,
            Element::TypeT::F32) && jcp_.weiDt == Element::TypeT::S8;
        if (!isIntType && jcp_.ngroups > 1 && (jcp_.ic % jcp_.simdW != 0 || jcp_.oc % jcp_.simdW != 0)) {
            jcp_.icBlock = IC_LEN4;
            jcp_.ocBlock = IC_LEN4;
        } else if (!isIntType) {
            jcp_.ocBlock = jcp_.simdW;
            jcp_.icBlock = jcp_.is1stConv ? jcp_.ic : jcp_.simdW;
        }
#ifdef KUDNNL_920F
        if (isIntType) {
            jcp_.chBlock = 1;
            jcp_.icBlock = IC_LEN16;
            jcp_.ocBlock = IC_LEN16;
            jcp_.maxRegsUr = REG_INDEX31;
        }
#else
        if (isIntType) {
            jcp_.chBlock = 1;
            jcp_.icBlock = IC_LEN8;
            jcp_.ocBlock = IC_LEN8;
            jcp_.maxRegsUr = REG_INDEX31;
        }
#endif
        if (isIntType) {
            if (isGroupsOne) {
                /* For non grouped convolutions, pad channels by 16 if needed */
                jcp_.oc = RndUp(jcp_.oc, jcp_.ocBlock);
                jcp_.ic = RndUp(jcp_.ic, jcp_.icBlock);
            } else if (jcp_.ngroups != 1 && ((jcp_.ic % jcp_.icBlock != 0) || (jcp_.oc % jcp_.ocBlock != 0))) {
                /* For grouped convolutions, oneDNN doesn't support padding.
                   When channels per group is not multiple of 4, 8, 16, return unimplemented. */
                jcp_.icBlock = (jcp_.ic % IC_LEN8 == 0) && (jcp_.oc % IC_LEN8 == 0) ? IC_LEN8 : IC_LEN4;
                jcp_.ocBlock = jcp_.icBlock;
            }
            jcp_.nbCh = Math::CeilDiv(jcp_.ngroups, jcp_.chBlock);
        }
        if (isSrcF16 && kCpuIsa == static_cast<IntType>(ARMCG::CPUISAT::VC_512)) {
            jcp_.icBlock = jcp_.is1stConv ? jcp_.ic : IC_LEN16;
        }
        /* Channel padding */
        jcp_.oc = RndUp(jcp_.oc, jcp_.ocBlock);
        jcp_.ic = RndUp(jcp_.ic, jcp_.icBlock);
        jcp_.nbIc = Math::CeilDiv(jcp_.ic, jcp_.icBlock);
        jcp_.nbOc = Math::CeilDiv(jcp_.oc, jcp_.ocBlock);
        jcp_.nbIcBlocking = jcp_.nbOcBlocking = 1;
        if (isIntType) {
            auto isOcBlockingOk = [this](IntType block) {
                IntType urW = std::min<IntType>(jcp_.ow, jcp_.maxRegsUr / (block + 1));
                return jcp_.nbOc % block == 0 && jcp_.lPad <= urW && jcp_.ow % urW != 1;
            };
            IntType maxThreadingNbOcChunk = IC_LEN4;
            jcp_.nbOcBlockingThrChunk = std::min<IntType>(maxThreadingNbOcChunk, jcp_.nbOc);
            for (; jcp_.nbOcBlockingThrChunk > 1; jcp_.nbOcBlockingThrChunk--) {
                if (isOcBlockingOk(jcp_.nbOcBlockingThrChunk)) {
                    break;
                }
            }
            jcp_.nbOcBlocking = jcp_.nbOcBlockingThrChunk;
        }
        jcp_.ver = KuDNN::KdnnConvVersionT::KUDNN_VER_FMA;
        SetJcpTypeSize(jcp_);
        jcp_.urW = std::min<IntType>(jcp_.ow, regs);
        /* Grouped channel offset to support 'non-blocked data' format for
        * convolution sizes with '(input_channel / ngroups) < simd' */
        jcp_.nonblkGroupOff = 1;
        if (!isIntType && jcp_.ver == KuDNN::KdnnConvVersionT::KUDNN_VER_FMA) {
            GetBestCutWay(jcp_);
        }
        
        PickLoopOrder(jcp_, isIntType);
        jcp_.nbIcL2 = jcp_.nbIc;
        bool isDataLayoutNxc = IsLayoutNxc(src) && IsLayoutNxc(dst);
        if (!isIntType && jcp_.isDepthwise) {
            DealDwInitConf(jcp_, kCpuIsa, fullSimdW, dst, isDataLayoutNxc);
        }
        
        jcp_.nbChBlocking = kCpuIsa == static_cast<IntType>(ARMCG::CPUISAT::VC_512) ? NB_CH_BLKING_4 :
            kCpuIsa == static_cast<IntType>(ARMCG::CPUISAT::VC_256) ? NB_CH_BLKING_3 : NB_CH_BLKING_2;
        if (!isIntType && jcp_.nbCh < jcp_.nbChBlocking) {
            jcp_.nbChBlocking = jcp_.nbCh;
        }
        if (isIntType) {
            IntType nbChBlocking = kCpuIsa == static_cast<IntType>(ARMCG::CPUISAT::VC_512) ? NB_CH_BLKING_4 :
                kCpuIsa == static_cast<IntType>(ARMCG::CPUISAT::VC_256) ? NB_CH_BLKING_3 : NB_CH_BLKING_2;
            for (; nbChBlocking > 1; nbChBlocking--) {
                if (jcp_.nbCh % nbChBlocking == 0) {
                    break;
                }
            }
            jcp_.nbChBlocking = jcp_.isDepthwise ? nbChBlocking : 1;
            jcp_.urW = jcp_.maxRegsUr / (jcp_.isDepthwise ? jcp_.nbChBlocking : jcp_.nbOcBlocking + 1);
            jcp_.urW = jcp_.ow < jcp_.urW ? jcp_.ow : jcp_.urW;
            if (!jcp_.isDepthwise && jcp_.urW < jcp_.ow) {
                const IntType broadcastSize = IC_LEN4;
                if (jcp_.icWithoutPadding % broadcastSize != 0) {
                    while (jcp_.urW > 0) {
                        IntType lastBlockSize = (jcp_.ow % jcp_.urW == 0) ? jcp_.urW : jcp_.ow % jcp_.urW;
                        IntType penultimateIwIindex = (jcp_.ow - 1 - lastBlockSize) * jcp_.strideW
                            + (jcp_.kw - 1) * (jcp_.dilateW + 1) - jcp_.lPad;
                        IntType penultimateIwLeeway = (jcp_.iw - 1 - penultimateIwIindex) * jcp_.icWithoutPadding
                            + jcp_.icWithoutPadding % broadcastSize;
                        if (penultimateIwLeeway >= broadcastSize) {
                            break;
                        }
                        --jcp_.urW;
                    }
                }
            }
        }
        jcp_.signedInput = (src.GetType() == Element::TypeT::S8) ? true : false;
        jcp_.urWTail = jcp_.ow % jcp_.urW;
        // currently kernel handle ow as a whole
        jcp_.owBlock = jcp_.ow;
        jcp_.nbOw = 1;
        if (isIntType) {
            IntType baseWorkAmount = jcp_.mb * jcp_.nbCh * jcp_.od * jcp_.oh * (jcp_.nbOc / jcp_.nbOcBlockingThrChunk);
            float bestThrEff = static_cast<float>(baseWorkAmount / RndUp(baseWorkAmount, jcp_.nThr));
            IntType maxNbOw = Math::CeilDiv(jcp_.ow, NB_OW_STEP_2 * jcp_.urW);
            for (IntType nbOw = 1; nbOw <= maxNbOw; nbOw++) {
                IntType owBlock = std::min<IntType>(RndUp(Math::CeilDiv(jcp_.ow, nbOw), jcp_.urW), jcp_.ow);
                if (owBlock < jcp_.nbOcBlockingThrChunk * jcp_.ocBlock && bestThrEff > 0.8f) {
                    break;
                }
                if (Math::CeilDiv(jcp_.ow, owBlock) != nbOw) {
                    continue;
                }
                auto workAmount = baseWorkAmount * nbOw;
                float thrEff = static_cast<float>(workAmount / RndUp(workAmount, jcp_.nThr));
                if (owBlock >= NB_OW_STEP_2 * jcp_.urW && thrEff > 1.1f * bestThrEff) {
                    jcp_.owBlock = owBlock;
                    bestThrEff = thrEff;
                }
                if (bestThrEff > 0.9f) {
                    break;
                }
            }
        }
        IntType L2_size = L2_CORE_CACHE_SIZE / sizeof(float);
        if (isSrcF16) {
            L2_size = L2_CORE_CACHE_SIZE / sizeof(__fp16);
        }
        // Source and output data needs to fit in L2,
        // leaving some space for weights and prefetching.
        IntType hL2Zero = 0;
        IntType hL2 = IntType(((0.6f * L2_size) / jcp_.simdW - std::min<IntType>(hL2Zero, jcp_.kh -
            jcp_.strideH) * jcp_.iw) / (jcp_.strideH * jcp_.iw + jcp_.ow));
        IntType tL2 = 1;
        jcp_.hBlocking = std::max<IntType>(tL2, std::min<IntType>(jcp_.oh, hL2));
        if (isDataLayoutNxc) {
            const IntType nbIcThesholdL2 = KERNEL_C_LEN32;
            if ((jcp_.nbIc > nbIcThesholdL2) && (jcp_.nbIc < KERNEL_C_LEN2 * nbIcThesholdL2)) {
                jcp_.nbIcL2 = jcp_.nbIc;
            } else {
                jcp_.nbIcL2 = std::min<IntType>(jcp_.nbIc, nbIcThesholdL2);
            }
        }
        Element::TypeT dtp = GetTensorDtp(jcp_);
        CreateKernel(dtp);
        return Status::SUCCESS;
    }
    KdnnBrgconvDataT bcd_ = KdnnBrgconvDataT();
    KdnnConvJitConfT jcp_ = KdnnConvJitConfT();
    std::unique_ptr<ARMCG::JitGenerator> jit_conv_ker_base_;
#ifdef KUDNNL_920F
    std::vector<float, Service::AlignedAllocator<float>> padSrc;
    std::vector<float, Service::AlignedAllocator<float>> colTmp;
#endif
};

void ConvolutionJITFWDImpl::ExecuteForwardDW(const void *src, const void *wei,
    void *dst, const void *bias) noexcept(false)
{
    KdnnConvJitConfT& jcp = jcp_;
    KdnnConvJitKerT jitKer = GetKernel();
    const IntType dilateH = jcp.dilateH + 1;
    const IntType strideH = jcp.strideH;
    const IntType chStep = jcp.nbChBlocking;
    const IntType iw = 0;
    const IntType kw = 0;
    const IntType ow = 0;
    const IntType chbWork = Math::CeilDiv(jcp.nbCh, chStep);
    const auto isSrcLayoutNXC = jcp.srcTag == KuDNN::Layout::ACDB;
    const auto isDstLayoutNXC = jcp.dstTag == KuDNN::Layout::ACDB;
    const IntType workAmount = jcp.mb * chbWork * jcp.oh;
    const auto nThr = jcp.nThr;

    Parallel(nThr, [&](const IntType ithr, const IntType subNthr) {
        IntType start {0};
        IntType end {0};
        Balance211(workAmount, subNthr, ithr, start, end);
        IntType n {0};
        IntType chb {0};
        IntType oh {0};
        if (jcp.loopOrder == KdnnConvLoopOrderT::KUDNN_LOOP_NGCW) {
            NdIteratorInit(start, n, jcp.mb, chb, chbWork, oh, jcp.oh);
        } else if (jcp.loopOrder == KdnnConvLoopOrderT::KUDNN_LOOP_NHWCG) {
            NdIteratorInit(start, n, jcp.mb, oh, jcp.oh, chb, chbWork);
        } else {
            throw KuDNN::Service::Unsupported{"unsupported loop order"};
        }

        auto iWork = start;
        while (iWork < end) {
            IntType ch = chb * chStep;
            const IntType itOverflow =
                std::max(jcp.tPad - oh * strideH, static_cast<IntType>(0));
            const IntType ibOverflow =
                std::max(oh * strideH + (jcp.kh - 1) * dilateH - jcp.tPad + 1, jcp.ih) - jcp.ih;
            const IntType ih =
                std::max((oh * strideH - jcp.tPad + Math::CeilDiv(itOverflow, dilateH) * dilateH),
                static_cast<IntType>(0));
            const IntType kh = Math::CeilDiv(itOverflow, dilateH);
            const IntType khPadding =
                jcp.kh - Math::CeilDiv(itOverflow, dilateH) - Math::CeilDiv(ibOverflow, dilateH);
            const auto icOffIdx = isSrcLayoutNXC ? ch * jcp.chBlock : ch;
            const auto ocOffIdx = isDstLayoutNXC ? ch * jcp.chBlock : ch;
            const IntType srcBlkOff = n * jcp.ngroups * jcp.ih * jcp.iw + icOffIdx * jcp.chBlock * jcp.ih * jcp.iw
                + jcp.chBlock * ih * jcp.iw + jcp.chBlock * iw;
            const IntType srcNXCOff = n * jcp.ngroups * jcp.ih * jcp.iw + jcp.ic * ih * jcp.iw
                + jcp.ic * iw + icOffIdx;
            const IntType weiBlkOff = ch * jcp.chBlock * jcp.kh * jcp.kw
                + jcp.chBlock * kh * jcp.kw + jcp.chBlock * kw;
            const IntType dstBlkOff = n * jcp.ngroups * jcp.oh * jcp.ow + ocOffIdx * jcp.chBlock * jcp.oh * jcp.ow
                + jcp.chBlock * oh * jcp.ow + jcp.chBlock * ow;
            const IntType dstNXCOff = n * jcp.ngroups * jcp.oh * jcp.ow
                + jcp.oc * oh * jcp.ow + jcp.oc * ow + ocOffIdx;

            auto paraConv = KdnnConvJitCallS();
            paraConv.src = jcp.isFusedConv ? static_cast<const float*>(src) :
                static_cast<const float*>(src) + (isSrcLayoutNXC ? srcNXCOff : srcBlkOff);
            paraConv.filt = static_cast<const float*>(wei) + weiBlkOff;
            paraConv.dst = static_cast<float*>(dst) + (isDstLayoutNXC ? dstNXCOff : dstBlkOff);
            paraConv.bias = bias ? static_cast<const float*>(bias) + (ch * jcp.chBlock) : nullptr;
            paraConv.khPadding = std::max(khPadding, static_cast<IntType>(0));
            if (isSrcLayoutNXC) {
                IntType workRemain = end - iWork;
                paraConv.chBlocks = ch + workRemain * chStep >= jcp.nbCh ? jcp.nbCh - ch : workRemain * chStep;
            } else {
                paraConv.chBlocks = ThisBlockSize(ch, jcp.nbCh, chStep);
            }
            jitKer(&paraConv);
            if (jcp.loopOrder == KdnnConvLoopOrderT::KUDNN_LOOP_NGCW) {
                iWork++;
                NdIteratorStep(n, jcp.mb, chb, chbWork, oh, jcp.oh);
            } else if (jcp.loopOrder == KdnnConvLoopOrderT::KUDNN_LOOP_NHWCG) {
                NdIteratorJump(iWork, end, n, jcp.mb, oh, jcp.oh, chb, chbWork);
            } else {
                throw KuDNN::Service::Unsupported{"unsupported loop order"};
            }
        }
    });
}

#ifdef KUDNNL_920F
inline void ConvolutionJITFWDImpl::DataAssign(const float *src, float *dst,
    IntType mb, IntType ic, IntType id, IntType ih,
    IntType od, IntType oh, IntType kd, IntType kh, IntType kw,
    IntType sizeIn, IntType sizeOut, KdnnConvJitConfT &jcp)
{
    for (IntType ow = 0; ow < jcp.ow; ++ow) {
        IntType iw = ow * jcp.strideW - jcp.lPad + kw * (jcp.dilateW + 1);
        if ((iw >= 0) && (iw < jcp.iw)) {
            dst[mb * sizeOut + (((((kd * jcp.kh + kh) * jcp.kw + kw) * jcp.ic + ic)
                * jcp.od + od) * jcp.oh + oh) * jcp.ow + ow] =
                src[mb * sizeIn + ((ic * jcp.id + id) * jcp.ih + ih) * jcp.iw + iw];
        }
    }
}

inline void ConvolutionJITFWDImpl::ContiguousCopy(const float *src, float *dst,
    IntType mb, IntType ic, IntType id, IntType ih,
    IntType od, IntType oh, IntType kd, IntType kh, IntType kw,
    IntType sizeIn, IntType sizeOut, KdnnConvJitConfT &jcp)
{
    IntType sSVal = jcp.lPad - kw;
    IntType sMVal = -jcp.lPad + kw;
    IntType sMidVal = jcp.ow - jcp.lPad + kw;
    IntType sStart = sSVal > 0 ? sSVal : 0;
    IntType sMinIw = sMVal > 0 ? sMVal : 0;
    IntType sMaxIw = sMidVal > jcp.iw ? jcp.iw : sMidVal;
    float *out = dst + mb * sizeOut + (((((kd * jcp.kh + kh) * jcp.kw + kw)
        * jcp.ic + ic) * jcp.od + od) * jcp.oh + oh) * jcp.ow + sStart;
    const float *in = src + mb * sizeIn + ((ic * jcp.id + id) * jcp.ih + ih) * jcp.iw + sMinIw;
    DataCopy(out, in, sMaxIw - sMinIw);
}

inline void ConvolutionJITFWDImpl::StrideCopy(const float *src, float *dst,
    IntType mb, IntType ic, IntType id, IntType ih,
    IntType od, IntType oh, IntType kd, IntType kh, IntType kw,
    IntType sizeIn, IntType sizeOut, KdnnConvJitConfT &jcp)
{
    float *out = dst + mb * sizeOut + (((((kd * jcp.kh + kh) * jcp.kw + kw) *
        jcp.ic + ic) * jcp.od + od) * jcp.oh + oh) * jcp.ow;
    const float *in = src + mb * sizeIn + ((ic * jcp.id + id) * jcp.ih + ih) * jcp.iw + kw;
    DataCopyWithStride(out, in, jcp.ow, jcp.strideW);
}

void ConvolutionJITFWDImpl::Img2col(const float *src, float *dst)
{
    using DataReorder = void (*) (const float *, float *, IntType, IntType, IntType, IntType, IntType, IntType,
        IntType, IntType, IntType, IntType, IntType, KdnnConvJitConfT &);
    DataReorder handler = nullptr;
    bool isContiguous = jcp_.dilateW == 0 && jcp_.strideW == 1;
    bool isStride = jcp_.dilateW == 0 && jcp_.lPad == 0 && jcp_.rPad == 0;
    if (isContiguous) {
        handler = ContiguousCopy;
    } else if (isStride) {
        handler = StrideCopy;
    } else {
        handler = DataAssign;
    }
    IntType sizeIn = jcp_.id * jcp_.ih * jcp_.iw * jcp_.ic;
    IntType sizeOut = jcp_.od * jcp_.oh * jcp_.ow * jcp_.kd * jcp_.kh * jcp_.kw * jcp_.ic;
#pragma omp parallel for num_threads(jcp_.alignedThreads) collapse(7)
    for (auto mb = 0; mb < jcp_.mb; mb++) {
        for (auto kd = 0; kd < jcp_.kd; ++kd) {
            for (auto kh = 0; kh < jcp_.kh; ++kh) {
                for (auto kw = 0; kw < jcp_.kw; ++kw) {
                    for (auto ic = 0; ic < jcp_.ic; ++ic) {
                        for (auto od = 0; od < jcp_.od; ++od) {
                            for (auto oh = 0; oh < jcp_.oh; ++oh) {
                                auto id = jcp_.ndims == KERNEL_DIMS_5
                                    ? (od * jcp_.strideD - jcp_.fPad + kd * (jcp_.dilateD + 1)) : 0;
                                auto ih = jcp_.ndims >= KERNEL_DIMS_4
                                    ? (oh * jcp_.strideH - jcp_.tPad + kh * (jcp_.dilateH + 1)) : 0;
                                if ((id >= 0) && (id < jcp_.id) && (ih >= 0) && (ih < jcp_.ih)) {
                                    handler(src, dst, mb, ic, id, ih, od, oh, kd, kh, kw, sizeIn, sizeOut, jcp_);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

bool ConvolutionJITFWDImpl::CanUseDirectMC()
{
    if (jcp_.ngroups != 1 || !Service::OneOf(jcp_.ndims, KERNEL_DIMS_4, KERNEL_DIMS_5)
        || jcp_.dilateH != 0 || jcp_.dilateW != 0 || jcp_.dilateD != 0) {
        return false;
    }
    if (jcp_.strideD == 1 && jcp_.strideH == 1 && jcp_.strideW == 1 && jcp_.ow >= NUM_24) {
        return true;
    }
    if (jcp_.strideW != 1 && jcp_.kw <= NUM_3 && jcp_.ow > NUM_32
        && jcp_.lPad == 0 && jcp_.rPad == 0 && jcp_.tPad == 0 && jcp_.bPad == 0
        && jcp_.fPad == 0 && jcp_.backPad == 0) {
        return true;
    }
    return false;
}

void ConvolutionJITFWDImpl::DataCopyWithStride(float *dst, const float *src, IntType len, IntType stride)
{
    svuint32_t idx = svindex_u32(IDX_0, stride);
    for (IntType i = 0; i < len / NUM_16; i++) {
        svbool_t pg = svptrue_b32();
        svfloat32_t row = svld1_gather_u32index_f32(pg, src + i * NUM_16 * stride, idx);
        svst1_f32(pg, dst + i * NUM_16, row);
    }
    IntType tail = len % NUM_16;
    if (tail != 0) {
        svbool_t pgTail = svwhilelt_b32_s64(0, tail);
        svfloat32_t row = svld1_gather_u32index_f32(pgTail, src + len / NUM_16 * NUM_16 * stride, idx);
        svst1_f32(pgTail, dst + len / NUM_16 * NUM_16, row);
    }
}

void ConvolutionJITFWDImpl::DataCopy(float *dst, const float *src, IntType len)
{
    for (IntType i = 0; i < len / NUM_16; i++) {
        svbool_t pg = svptrue_b32();
        svfloat32_t row = svld1_f32(pg, src + i * NUM_16);
        svst1_f32(pg, dst + i * NUM_16, row);
    }
    IntType tail = len % NUM_16;
    if (tail != 0) {
        svbool_t pgTail = svwhilelt_b32_s64(0, tail);
        svfloat32_t row = svld1_f32(pgTail, src + len / NUM_16 * NUM_16);
        svst1_f32(pgTail, dst + len / NUM_16 * NUM_16, row);
    }
}

void ConvolutionJITFWDImpl::DirectMC(const float *src, const float *wei,
    float *dst, const float *bias) noexcept
{
    KdnnConvJitConfT& jcp = jcp_;
    if (jcp.strideW != 1 && jcp.kw <= NUM_3) {
        DirectMCWithStride(src, wei, dst, bias);
        return;
    }
    KdnnConvJitKerT jitKer = GetKernel();
    auto nThr = jcp.alignedThreads;
    IntType sizeD = jcp.id + jcp.fPad + jcp.backPad;
    IntType sizeH = jcp.ih + jcp.tPad + jcp.bPad;
    IntType sizeW = jcp.iw + jcp.lPad + jcp.rPad;
    bool isTpadBigerZero = jcp.tPad > 0;
    bool isBPadBigerZero = jcp.bPad > 0;
    bool isLPadBigerZero = jcp.lPad > 0;
    bool isRPadBigerZero = jcp.rPad > 0;
    bool isFpadBigerZero = jcp.fPad > 0;
    bool isBackpadBigerZero = jcp.backPad > 0;
    bool isAllPadSmallerZero = (!isTpadBigerZero) && (!isBPadBigerZero) &&
        (!isLPadBigerZero) && (!isRPadBigerZero) && (!isFpadBigerZero) && (!isBackpadBigerZero);
    if (!isAllPadSmallerZero) {
#pragma omp parallel for num_threads(nThr) collapse(4)
        for (auto mb = 0; mb < jcp.mb; mb++) {
            for (auto ic = 0; ic < jcp.ic; ic++) {
                for (auto od = 0; od < jcp.od; od++) {
                    for (auto oh = 0; oh < jcp.oh; oh++) {
                        DataCopy(padSrc.data() + mb * jcp.ic * sizeD * sizeH * sizeW
                            + ic * sizeD * sizeH * sizeW + (jcp.fPad + od) * sizeH * sizeW +
                            (jcp.tPad + oh) * sizeW + jcp.lPad, src + mb * jcp.ic * jcp.id * jcp.ih * jcp.iw +
                            ic * jcp.id * jcp.ih * jcp.iw + od * jcp.ih * jcp.iw + oh * jcp.iw, jcp.iw);
                    }
                }
            }
        }
    }
#pragma omp parallel for num_threads(nThr) collapse(4)
    for (auto mb = 0; mb < jcp.mb; mb++) {
        for (auto od = 0; od < jcp.od; od++) {
            for (auto oh = 0; oh < jcp.oh; oh++) {
                for (auto oc = 0; oc < jcp.oc; oc += NUM_16) {
                    KdnnConvJitCallS p = KdnnConvJitCallS();
                    p.matrixM = jcp.od * jcp.oh * jcp.ow;
                    p.matrixN = oc + NUM_16 > jcp.oc ? jcp.oc - oc : NUM_16;
                    p.matrixK = sizeD * sizeH * sizeW;
                    if (isAllPadSmallerZero) {
                        p.matrixA = src + mb * jcp.ic * p.matrixK + od * jcp.ih * jcp.iw + oh * jcp.iw;
                    } else {
                        p.matrixA = padSrc.data() + mb * jcp.ic * p.matrixK + od * sizeH * sizeW + oh * sizeW;
                    }
                    p.matrixB = wei + oc * jcp.kd * jcp.kh * jcp.kw * jcp.ic;
                    p.matrixC = dst + mb * p.matrixM * jcp.oc + oc * p.matrixM + od * jcp.oh * jcp.ow + oh * jcp.ow;
                    p.bias = bias + oc;
                    if (jitKer) jitKer(&p);
                }
            }
        }
    }
}

void ConvolutionJITFWDImpl::GemmAccelarate(const float *src, const float *wei,
    float *dst, const float *bias) noexcept
{
    KdnnConvJitConfT& jcp = jcp_;
    KdnnConvJitKerT jitKer = GetKernel();
    auto nThr = jcp.alignedThreads;
    IntType sizeIn = jcp.id * jcp.ih * jcp.iw * jcp.ic;
    IntType sizeOut = jcp.od * jcp.oh * jcp.ow * jcp.kd * jcp.kh * jcp.kw * jcp.ic;
#pragma omp parallel for num_threads(nThr) collapse(7)
    for (auto mb = 0; mb < jcp.mb; ++mb) {
        for (auto ic = 0; ic < jcp.ic; ++ic) {
            for (auto od = 0; od < jcp.od; od++) {
                for (auto oh = 0; oh < jcp.oh; ++oh) {
                    for (auto kd = 0; kd < jcp.kd; ++kd) {
                        for (auto kh = 0; kh < jcp.kh; ++kh) {
                            for (auto kw = 0; kw < jcp.kw; ++kw) {
                                auto id = (od * jcp.strideD - jcp.fPad + kd * (jcp.dilateD + 1));
                                auto ih = (oh * jcp.strideH - jcp.tPad + kh * (jcp.dilateH + 1));
                                if ((ih >= 0) && (ih < jcp.ih) && (id >= 0) && (id < jcp.id)) {
                                    float *out = colTmp.data() + mb * sizeOut +
                                        od * jcp.oh * jcp.kd * jcp.kh * jcp.kw * jcp.ic * jcp.ow +
                                        oh * jcp.kd * jcp.kh * jcp.kw * jcp.ic * jcp.ow +
                                        kd * jcp.kh * jcp.kw * jcp.ic * jcp.ow +
                                        kh * jcp.kw * jcp.ic * jcp.ow + kw * jcp.ic * jcp.ow + ic * jcp.ow;
                                    const float *in = src + mb * sizeIn + ic * jcp.id * jcp.ih * jcp.iw +
                                        id * jcp.ih * jcp.iw + ih * jcp.iw + kw;
                                    DataCopyWithStride(out, in, jcp.ow, jcp.strideW);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
#pragma omp parallel for num_threads(nThr) collapse(4)
    for (auto mb = 0; mb < jcp.mb; mb++) {
        for (auto od = 0; od < jcp.od; od++) {
            for (auto oh = 0; oh < jcp.oh; oh++) {
                for (auto oc = 0; oc < jcp.oc; oc += NUM_16) {
                    KdnnConvJitCallS p = KdnnConvJitCallS();
                    p.matrixM = jcp.ow;
                    p.matrixN = oc + NUM_16 > jcp.oc ? jcp.oc - oc : NUM_16;
                    p.matrixK = jcp.kd * jcp.kh * jcp.kw * jcp.ic;
                    p.matrixA = colTmp.data() + mb * sizeOut +
                        od * jcp.oh * jcp.ow * jcp.kd * jcp.kh * jcp.kw * jcp.ic +
                        oh * jcp.ow * jcp.kd * jcp.kh * jcp.kw * jcp.ic;
                    p.matrixB = wei + oc * jcp.kd * jcp.kh * jcp.kw * jcp.ic;
                    p.matrixC = dst + mb * jcp.od * jcp.oh * jcp.ow * jcp.oc +
                        oc * jcp.od * jcp.oh * jcp.ow + od * jcp.oh * jcp.ow +
                        oh * jcp.ow;
                    p.bias = bias + oc;
                    if (jitKer) jitKer(&p);
                }
            }
        }
    }
}
void ConvolutionJITFWDImpl::DirectMCWithStride(const float *src, const float *wei,
    float *dst, const float *bias) noexcept
{
    KdnnConvJitConfT& jcp = jcp_;
    KdnnConvJitKerT jitKer = GetKernel();
    IntType nThr = jcp.alignedThreads;
    IntType sizeD = jcp.id + jcp.fPad + (jcp.backPad > 0 ? jcp.backPad : 0);
    IntType sizeH = jcp.ih + jcp.tPad + (jcp.bPad > 0 ? jcp.bPad : 0);
    IntType sizeW = jcp.iw + jcp.lPad + (jcp.rPad > 0 ? jcp.rPad : 0);
    // pack stride src in order
#pragma omp parallel for num_threads(nThr) collapse(4)
    for (auto mb = 0; mb < jcp.mb; mb++) {
        for (auto ic = 0; ic < jcp.ic; ic++) {
            for (auto id = 0; id < jcp.id; id++) {
                for (auto ih = 0; ih < jcp.ih; ih++) {
                    IntType off = 0;
                    for (auto stride = 0; stride < jcp.strideW; stride++) {
                        IntType len = (sizeW + 1 - stride) / jcp.strideW;
                        DataCopyWithStride(padSrc.data() + mb * jcp.ic * sizeD * sizeH * sizeW
                            + ic * sizeD * sizeH * sizeW + (id + jcp.fPad) * sizeH * sizeW +
                            (jcp.tPad + ih) * sizeW + jcp.lPad + off, src + mb * jcp.ic * jcp.id * jcp.ih * jcp.iw +
                            ic * jcp.id * jcp.ih * jcp.iw + id * jcp.ih * jcp.iw +
                            ih * jcp.iw + stride, len, jcp.strideW);
                            off += len;
                    }
                }
            }
        }
    }

#pragma omp parallel for num_threads(nThr) collapse(4)
    for (auto mb = 0; mb < jcp.mb; mb++) {
        for (auto od = 0; od < jcp.od; od++) {
            for (auto oh = 0; oh < jcp.oh; oh++) {
                for (auto oc = 0; oc < jcp.oc; oc += NUM_16) {
                    KdnnConvJitCallS p = KdnnConvJitCallS();
                    p.matrixM = jcp.od * jcp.oh * jcp.ow;
                    p.matrixN = oc + NUM_16 > jcp.oc ? jcp.oc - oc : NUM_16;
                    p.matrixK = sizeD * sizeH * sizeW;
                    p.matrixA = padSrc.data() + mb * jcp.ic * p.matrixK +
                        od * jcp.strideD * sizeH * sizeW +  oh * jcp.strideH * sizeW;
                    p.matrixB = wei + oc * jcp.kd * jcp.kh * jcp.kw * jcp.ic;
                    p.matrixC = dst + mb * p.matrixM * jcp.oc + oc * p.matrixM +
                        od * jcp.oh * jcp.ow + oh * jcp.ow;
                    p.bias = bias + oc;
                    if (jitKer) jitKer(&p);
                }
            }
        }
    }
}

void ConvolutionJITFWDImpl::DoMCGemmPack(const float *src)
{
    KdnnConvJitConfT& jcp = jcp_;
    IntType sizeD = jcp.id + jcp.fPad + jcp.backPad;
    IntType sizeH = jcp.ih + jcp.tPad + jcp.bPad;
    IntType sizeW = jcp.iw + jcp.lPad + jcp.rPad;
    auto nThr = jcp.alignedThreads;
#pragma omp parallel for num_threads(nThr) collapse(4)
    for (auto mb = 0; mb < jcp.mb; mb++) {
        for (auto ic = 0; ic < jcp.ic; ic++) {
            for (auto od = 0; od < jcp.od; od++) {
                for (auto oh = 0; oh < jcp.oh; oh++) {
                    if (oh >= jcp.bPad && oh < jcp.ih + jcp.bPad) {
                        DataCopy(padSrc.data() + mb * jcp.ic * sizeD * sizeH * sizeW
                            + ic * sizeD * sizeH * sizeW + od * jcp.oh * sizeW + oh * sizeW + jcp.lPad,
                            src + mb * jcp.ic * jcp.id * jcp.ih * jcp.iw + ic * jcp.id * jcp.ih *
                            jcp.iw + (od - jcp.fPad) * jcp.oh * jcp.iw + (oh - jcp.bPad) * jcp.iw, jcp.iw);
                    }
                }
            }
        }
    }
}

void ConvolutionJITFWDImpl::DoMCGemm(const float *src, const float *wei, float *dst,
    const float *bias, bool isNoImg2Col, bool isNeedDopad)
{
    KdnnConvJitConfT& jcp = jcp_;
    IntType nThr = jcp.alignedThreads;
    IntType sizeOut = jcp.od * jcp.oh * jcp.ow * jcp.kd * jcp.kh * jcp.kw * jcp.ic;
    KdnnConvJitKerT jitKer = GetKernel();
#pragma omp parallel for num_threads(nThr) collapse(2)
    for (auto mb = 0; mb < jcp.mb; mb++) {
        for (auto oc = 0; oc < jcp.oc; oc += NUM_16) {
            KdnnConvJitCallS p = KdnnConvJitCallS();
            p.matrixM = jcp.od * jcp.oh * jcp.ow;
            p.matrixN = oc + NUM_16 > jcp.oc ? jcp.oc - oc : NUM_16;
            p.matrixK = jcp.kd * jcp.kh * jcp.kw * jcp.ic;
            if (isNoImg2Col) {
                p.matrixA = isNeedDopad ? padSrc.data() + mb * sizeOut : src + mb * sizeOut;
            } else {
                p.matrixA = colTmp.data() + mb * sizeOut;
            }
            p.matrixB = wei + oc * jcp.kd * jcp.kh * jcp.kw * jcp.ic;
            p.matrixC = dst + mb * jcp.od * jcp.oh * jcp.ow * jcp.oc + oc * p.matrixM;
            p.bias = bias + oc;
            if (jitKer) jitKer(&p);
        }
    }
}

bool ConvolutionJITFWDImpl::AccelerateGemm()
{
    return jcp_.ngroups == 1 && Service::OneOf(jcp_.ndims, KERNEL_DIMS_4, KERNEL_DIMS_5) && jcp_.strideW != 1 &&
        jcp_.lPad == 0 && jcp_.rPad == 0 && jcp_.fPad == 0;
}

void ConvolutionJITFWDImpl::ExecuteForwardMC(const void *src, const void *wei,
    void *dst, const void *bias) noexcept
{
    KdnnConvJitConfT& jcp = jcp_;
    bool isCanUseDirectMC = CanUseDirectMC();
    const float* srcPtr = static_cast<const float*>(src);
    const float* weiPtr = static_cast<const float*>(wei);
    const float* biasPtr = static_cast<const float*>(bias);
    float* dstPtr = static_cast<float*>(dst);
    if (isCanUseDirectMC) {
        DirectMC(srcPtr, weiPtr, dstPtr, biasPtr);
        return;
    }
    bool isUseAcclerateGemm = AccelerateGemm();
    if (isUseAcclerateGemm) {
        GemmAccelarate(srcPtr, weiPtr, dstPtr, biasPtr);
        return;
    }
    bool noImg2Col = jcp.kd == 1 && jcp.kh == 1 && jcp.kw == 1 && jcp.strideD == 1 &&
        jcp.strideH == 1 && jcp.strideW == 1 && jcp.dilateD == 0 && jcp.dilateH == 0 && jcp.dilateW == 0;
    if (!noImg2Col) {
        Img2col(srcPtr, colTmp.data());
    }

    bool isNeedPad = jcp.fPad > 0 || jcp.backPad > 0 || jcp.tPad > 0 ||
        jcp.bPad > 0 || jcp.lPad > 0 || jcp.rPad > 0 ? true : false;
    if (isNeedPad) {
        DoMCGemmPack(srcPtr);
    }
    DoMCGemm(srcPtr, weiPtr, dstPtr, biasPtr, noImg2Col, isNeedPad);
}
#endif

void ConvolutionJITFWDImpl::DataDup(float *src, int32_t dupVal, IntType len)
{
#ifdef KUDNNL_920F
    constexpr IntType vLen = NUM_16;
#else
    constexpr IntType vLen = NUM_8;
#endif
    for (IntType i = 0; i < len / vLen; i++) {
        svbool_t pg = svptrue_b32();
        svfloat32_t row = svdup_n_f32(dupVal);
        svst1_f32(pg, src + i * vLen, row);
    }
    IntType tail = len % vLen;
    if (tail != 0) {
        svbool_t pgTail = svwhilelt_b32_s64(0, tail);
        svfloat32_t row = svdup_n_f32(dupVal);
        svst1_f32(pgTail, src + len / vLen * vLen, row);
    }
}

template<typename SrcDataType, typename DstDataType>
void ConvolutionJITFWDImpl::ExecuteForward1D4Int(const void *src, const void *wei,
    void *dst, const void *bias)
{
    KdnnConvJitConfT& jcp = jcp_;
    KdnnConvJitKerT jitKer = GetKernel();
    if (jitKer == nullptr) {
        return;
    }
#ifdef KUDNNL_920F
    constexpr IntType oscaleLen = OSCALE_LEN_16;
#else
    constexpr IntType oscaleLen = OSCALE_LEN_8;
#endif
    void* oscales = Service::AlignedAlloc(oscaleLen * sizeof(float));
    DataDup(static_cast<float*>(oscales), 1, oscaleLen);
    IntType offset = jcp.ic * jcp.oc * jcp.kw * jcp.ngroups;
    auto w = SafeCast<int8_t*>(static_cast<const int8_t*>(wei));
    int32_t* compensation = (!jcp.signedInput) ? reinterpret_cast<int32_t*>(&w[offset]) : nullptr;
    IntType ocChunks = jcp.nbOc / jcp.nbOcBlocking;
    IntType nbGroups = jcp.nbCh / jcp.nbChBlocking;
    IntType groupBlock = jcp.chBlock;
    IntType workAmount = jcp.mb * nbGroups * ocChunks * jcp.nbOw;
    IntType nThr = jcp.alignedThreads;
    Parallel(nThr, [&](const IntType ithr, const IntType subNThr) {
        IntType start {0};
        IntType end {0};
        Balance211(workAmount, subNThr, ithr, start, end);
        auto parConv = KuDNN::KdnnConvJitCallS();
        IntType n {0};
        IntType gg {0};
        IntType occ {0};
        IntType owb {0};
        if (jcp.loopOrder == KuDNN::KdnnConvLoopOrderT::KUDNN_LOOP_CWGN) {
            NdIteratorInit(start, occ, ocChunks, owb, jcp.nbOw, gg, nbGroups, n, jcp.mb);
        } else if (jcp.loopOrder == KuDNN::KdnnConvLoopOrderT::KUDNN_LOOP_GNCW) {
            NdIteratorInit(start, gg, nbGroups, n, jcp.mb, occ, ocChunks, owb, jcp.nbOw);
        } else if (jcp.loopOrder == KuDNN::KdnnConvLoopOrderT::KUDNN_LOOP_NGCW) {
            NdIteratorInit(start, n, jcp.mb, gg, nbGroups, occ, ocChunks, owb, jcp.nbOw);
        } else if (jcp.loopOrder == KuDNN::KdnnConvLoopOrderT::KUDNN_LOOP_NWCG) {
            NdIteratorInit(start, n, jcp.mb, owb, jcp.nbOw, occ, ocChunks, gg, nbGroups);
        } else {
            throw KuDNN::Service::Unsupported{"unsupported loop order"};
        }
        while (start < end) {
            IntType ocb = occ * jcp.nbOcBlocking;
            IntType gb = gg * jcp.nbChBlocking;
            IntType g = gb * groupBlock;
            IntType gOc = (g * jcp.nbOc + ocb) * jcp.ocBlock;
            IntType gIc = g * jcp.nbIc * jcp.icBlock;
            IntType owS = owb * jcp.owBlock;
            IntType iwS = owS * jcp.strideW;
            auto scales = &static_cast<float*>(oscales)[jcp.isOcScale * gOc];
            parConv.bias = bias ? static_cast<const char*>(bias) + gOc * jcp.typeSizeBia : nullptr;
            parConv.compensation = (!jcp.signedInput) ? (compensation + gOc) : nullptr;
            parConv.dst = static_cast<DstDataType*>(dst) + (n * jcp.ocWithoutPadding * jcp.ow * jcp.ngroups
                + gOc + owS * jcp.ocWithoutPadding * jcp.ngroups);
            parConv.src = static_cast<const SrcDataType*>(src) + (n * jcp.icWithoutPadding * jcp.iw * jcp.ngroups
                + gIc + iwS * jcp.icWithoutPadding * jcp.ngroups);
            parConv.scales = scales;
            parConv.filt = static_cast<const int8_t*>(wei) + jcp.ic * jcp.oc * gb * jcp.kw +
                jcp.ocBlock * ocb * jcp.ic * jcp.kw;
            parConv.ocBlocks = jcp.isDepthwise ? gb : ocb;
            parConv.khPadding = jcp.kh;
            parConv.tOverflow = 0;
            parConv.bOverflow = 0;
            parConv.owb = owb;
            jitKer(&parConv);
            ++start;
            if (jcp.loopOrder == KuDNN::KdnnConvLoopOrderT::KUDNN_LOOP_CWGN) {
                NdIteratorStep(occ, ocChunks, owb, jcp.nbOw, gg, nbGroups, n, jcp.mb);
            } else if (jcp.loopOrder == KuDNN::KdnnConvLoopOrderT::KUDNN_LOOP_GNCW) {
                NdIteratorStep(gg, nbGroups, n, jcp.mb, occ, ocChunks, owb, jcp.nbOw);
            } else if (jcp.loopOrder == KuDNN::KdnnConvLoopOrderT::KUDNN_LOOP_NGCW) {
                NdIteratorStep(n, jcp.mb, gg, nbGroups, occ, ocChunks, owb, jcp.nbOw);
            } else if (jcp.loopOrder == KuDNN::KdnnConvLoopOrderT::KUDNN_LOOP_NWCG) {
                NdIteratorStep(n, jcp.mb, owb, jcp.nbOw, occ, ocChunks, gg, nbGroups);
            } else {
                throw KuDNN::Service::Unsupported{"unsupported loop order"};
            }
        }
    });
    if (oscales) {
        Service::Deallocate(oscales);
    }
}

template <typename DataType>
void ConvolutionJITFWDImpl::ExecuteForward1D(const void *src, const void *wei,
    void *dst, const void *bias) noexcept(false)
{
    KdnnConvJitConfT& jcp = jcp_;
    KdnnConvJitKerT jitKer = GetKernel();
    if (jitKer == nullptr) {
        return;
    }
    const DataType* srcPtr = static_cast<const DataType*>(src);
    const DataType* weiPtr = static_cast<const DataType*>(wei);
    const DataType* biasPtr = static_cast<const DataType*>(bias);
    DataType* dstPtr = static_cast<DataType*>(dst);
    IntType ocChunks = jcp.nbOc / jcp.nbOcBlocking;
    IntType nbGroups = jcp.ngroups;
    IntType workAmount = jcp.mb * nbGroups * ocChunks * jcp.nbOw;
    IntType nThr = jcp.alignedThreads;
    Parallel(nThr, [&](const IntType ithr, const IntType subNThr) {
        IntType start {0};
        IntType end {0};
        IntType startCopy;
        Balance211(workAmount, subNThr, ithr, start, end);
        startCopy = start;
        auto parConv = KuDNN::KdnnConvJitCallS();
        const bool isSrcLayoutNxc = jcp.srcTag == KuDNN::Layout::ACB;
        const bool isDstLayoutNxc = jcp.dstTag == KuDNN::Layout::ACB;
        IntType srcCstride = 1;
        if (!isSrcLayoutNxc && jcp_.is1stConv) {
            srcCstride = jcp.iw;
        } else if (!isSrcLayoutNxc && !jcp_.is1stConv) {
            srcCstride = jcp.iw * jcp.icBlock;
        }
        IntType whtIcStride = jcp.ngroups * jcp.ic < jcp.ocBlock ? jcp.ocBlock :
            jcp.icBlock * jcp.ocBlock * jcp.kw;
        for (IntType icbL2 = 0; icbL2 < jcp.nbIc; icbL2 += jcp.nbIcL2) {
            start = startCopy;
            IntType n {0};
            IntType gg {0};
            IntType occ {0};
            IntType owb {0};
            IntType dummy {0};
            if (jcp.loopOrder == KuDNN::KdnnConvLoopOrderT::KUDNN_LOOP_CWGN) {
                NdIteratorInit(start, occ, ocChunks, owb, jcp.nbOw, gg, nbGroups, n, jcp.mb, dummy, 1);
            } else if (jcp.loopOrder == KuDNN::KdnnConvLoopOrderT::KUDNN_LOOP_GNCW) {
                NdIteratorInit(start, gg, nbGroups, n, jcp.mb, occ, ocChunks, owb, jcp.nbOw, dummy, 1);
            } else {
                throw KuDNN::Service::Unsupported{"unsupported loop order"};
            }
            
            while (start < end) {
                IntType ocb = occ * jcp.nbOcBlocking;
                IntType g = gg;
                IntType gOcb = g * jcp.nbOc + ocb;
                IntType gIcb = g * jcp.nbIc * jcp.nonblkGroupOff;
                IntType owS = owb * jcp.owBlock;
                IntType iwS = owS * jcp.strideW;
                const IntType ocOffIdx = isDstLayoutNxc ? g * jcp.oc + ocb * jcp.ocBlock : gOcb;
                auto dstW = dstPtr + (n * jcp.oc * jcp.ow * jcp.ngroups + ocOffIdx + owS * jcp.oc * jcp.ngroups);
                if (!isDstLayoutNxc) {
                    dstW = dstPtr + (n * jcp.oc * jcp.ow * jcp.ngroups + ocOffIdx *
                        jcp.ocBlock * jcp.ow + owS * jcp.ocBlock);
                }
                const IntType icOffIdx = isSrcLayoutNxc ? g * jcp.ic + icbL2 * jcp.icBlock : gIcb + icbL2;
                auto srcW = srcPtr + (n * jcp.ic * jcp.iw * jcp.ngroups + icOffIdx + iwS * jcp.ic * jcp.ngroups);
                if (!isSrcLayoutNxc && jcp_.is1stConv) {
                    srcW = srcPtr + (n * jcp.ic * jcp.iw + icOffIdx * jcp.iw + iwS);
                } else if (!isSrcLayoutNxc && !jcp_.is1stConv) {
                    srcW = srcPtr + (n * jcp.ic * jcp.iw * jcp.ngroups +
                        icOffIdx * jcp.icBlock * jcp.iw + iwS * jcp.icBlock);
                }
                auto whtW = weiPtr + (jcp.oc * jcp.ic * g +
                    jcp.ocBlock * (ocb * jcp.ic + icbL2 * jcp.icBlock)) * jcp.kw;
                auto biasW = bias ? biasPtr + ocOffIdx * (isDstLayoutNxc ? 1 : jcp.ocBlock) : nullptr;
                IntType icbStep = isSrcLayoutNxc ? jcp.nbIcL2 : 1;
                IntType icbL2Sum = icbL2 + jcp.nbIcL2;
                IntType icbEnd = jcp.nbIc > icbL2Sum ? icbL2Sum : jcp.nbIc;
                const IntType ocWork = ThisBlockSize(ocb * jcp.ocBlock, jcp.oc, jcp.nbOcBlocking * jcp.ocBlock);
                IntType icWork = icbStep * jcp.icBlock;
                for (IntType icb = icbL2; icb < icbEnd; icb += icbStep) {
                    IntType currNbIc = std::min(icbStep, icbEnd - icb);
                    IntType flags = 0;
                    if (icb == 0) {
                        flags |= static_cast<int32_t>(KuDNN::KdnnConvFlag::KUDNN_FLAG_IC_FIRST);
                    }
                    if (icb + currNbIc >= jcp.nbIc) {
                        flags |= static_cast<int32_t>(KuDNN::KdnnConvFlag::KUDNN_FLAG_IC_LAST);
                        icWork = ThisBlockSize(icb * jcp.icBlock, jcp.ic, icbStep * jcp.icBlock);
                    }
                    KdnnJitConvKerPipelineOwThr(jitKer, parConv, srcW,
                        dstW, whtW, biasW, icb, 1, owb, icWork, ocWork, flags);
                    srcW += srcCstride;
                    whtW += whtIcStride;
                }
                if (jcp.loopOrder == KuDNN::KdnnConvLoopOrderT::KUDNN_LOOP_CWGN) {
                    NdIteratorJump(start, end, occ, ocChunks, owb, jcp.nbOw, gg, nbGroups, n, jcp.mb, dummy, 1);
                } else if (jcp.loopOrder == KuDNN::KdnnConvLoopOrderT::KUDNN_LOOP_GNCW) {
                    NdIteratorJump(start, end, gg, nbGroups, n, jcp.mb, occ, ocChunks, owb, jcp.nbOw, dummy, 1);
                } else {
                    throw KuDNN::Service::Unsupported{"unsupported loop order"};
                }
            }
        }
        KdnnJitConvKerPipelineOwThr(jitKer, parConv, srcPtr, dstPtr, weiPtr, biasPtr, 0, 0, 0, 0, 0, 0);
    });
}

template<typename SrcDataType, typename DstDataType>
void ConvolutionJITFWDImpl::ExecuteForward2D4Int(const void *src, const void *wei,
    void *dst, const void *bias) noexcept(false)
{
    KdnnConvJitConfT& jcp = jcp_;
    KdnnConvJitKerT jitKer = GetKernel();
    if (jitKer == nullptr) {
        return;
    }
#ifdef KUDNNL_920F
    constexpr IntType oscaleLen = OSCALE_LEN_16;
#else
    constexpr IntType oscaleLen = OSCALE_LEN_8;
#endif
    void* oscales = Service::AlignedAlloc(oscaleLen * sizeof(float));
    DataDup(static_cast<float*>(oscales), 1, oscaleLen);
    IntType offset = jcp.ic * jcp.oc * jcp.kw * jcp.kh * jcp.ngroups;
    auto w = SafeCast<int8_t*>(static_cast<const int8_t*>(wei));
    int32_t* compensation = (!jcp.signedInput) ? reinterpret_cast<int32_t *>(&w[offset]) : nullptr;
    IntType ocChunks = jcp.nbOc / jcp.nbOcBlocking;
    IntType nbGroups = jcp.nbCh;
    IntType workAmount = jcp.mb * nbGroups * ocChunks * jcp.nbOw * jcp.oh;
    IntType nThr = jcp.alignedThreads;
    Parallel(nThr, [&](const IntType& ithr, const IntType& subNThr) {
        IntType start {0};
        IntType end {0};
        Balance211(workAmount, subNThr, ithr, start, end);
        auto parConv = KuDNN::KdnnConvJitCallS();
        IntType srcHstride = jcp.ngroups * jcp.iw * jcp.icWithoutPadding;
        IntType dstHstride = jcp.ngroups * jcp.ow * jcp.ocWithoutPadding;
        IntType whtHstride = jcp.icBlock * jcp.ocBlock * jcp.kw;
        IntType n {0};
        IntType g {0};
        IntType occ {0};
        IntType ohS {0};
        IntType owb {0};
        if (jcp.loopOrder == KuDNN::KdnnConvLoopOrderT::KUDNN_LOOP_CWGN) {
            NdIteratorInit(start, occ, ocChunks, owb, jcp.nbOw, g, nbGroups, n, jcp.mb, ohS, jcp.oh);
        } else if (jcp.loopOrder == KuDNN::KdnnConvLoopOrderT::KUDNN_LOOP_NGCW) {
            NdIteratorInit(start, n, jcp.mb, g, nbGroups, occ, ocChunks, owb, jcp.nbOw, ohS, jcp.oh);
        } else if (jcp.loopOrder == KuDNN::KdnnConvLoopOrderT::KUDNN_LOOP_NHWCG) {
            NdIteratorInit(start, n, jcp.mb, ohS, jcp.oh, owb, jcp.nbOw, occ, ocChunks, g, nbGroups);
        } else {
            throw KuDNN::Service::Unsupported{"unsupported loop order"};
        }
        while (start < end) {
            for (IntType occ1 = 0; occ1 < jcp.nbOcBlockingThrChunk; occ1 += jcp.nbOcBlocking) {
                IntType ocb = occ * jcp.nbOcBlockingThrChunk + occ1;
                IntType gOc = (g * jcp.nbOc + ocb) * jcp.ocBlock;
                IntType gIc = g * jcp.nbIc * jcp.icBlock;
                IntType workRem = end - start;
                IntType ihS = -jcp.tPad + ohS * jcp.strideH;
                IntType ohE = ohS + workRem > jcp.oh ? jcp.oh : ohS + workRem;
                if (jcp.loopOrder == KuDNN::KdnnConvLoopOrderT::KUDNN_LOOP_NHWCG) {
                    ohE = ohS + 1; // step instead
                }
                IntType owS = owb * jcp.owBlock;
                IntType iwS = owS * jcp.strideW;
                auto biasW = bias ? static_cast<const char*>(bias) + (gOc * jcp.typeSizeBia) : nullptr;
                int32_t *compensationW = (!jcp.signedInput) ? compensation + gOc : nullptr;

                auto dstW = static_cast<DstDataType*>(dst) +
                    (n * jcp.ocWithoutPadding * jcp.oh * jcp.ow * jcp.ngroups +
                    gOc + ohS * jcp.ow * jcp.ocWithoutPadding * jcp.ngroups +
                    owS * jcp.ocWithoutPadding * jcp.ngroups);
                auto srcW = static_cast<const SrcDataType*>(src) +
                    (n * jcp.icWithoutPadding * jcp.iw * jcp.ih * jcp.ngroups +
                    gIc + ihS * jcp.iw * jcp.icWithoutPadding * jcp.ngroups +
                    iwS * jcp.icWithoutPadding * jcp.ngroups);
                auto whtW = static_cast<const int8_t*>(wei) + (g * jcp.oc * jcp.ic * jcp.kh * jcp.kw +
                    ocb * jcp.ocBlock * jcp.ic * jcp.kw * jcp.kh);
                auto scales = &static_cast<float*>(oscales)[jcp.isOcScale * gOc];
                for (IntType oj = ohS, ij = ihS; oj < ohE; ++oj, ij += jcp.strideH) {
                    IntType dilateH = jcp.dilateH + 1;
                    IntType iTOverflow = std::min<IntType>(jcp.kh, Math::CeilDiv(std::max<IntType>(0, -ij), dilateH));
                    IntType iBOverflow = std::min<IntType>(jcp.kh,
                        Math::CeilDiv(std::max<IntType>(0, ij - jcp.ih + (jcp.kh - 1) * dilateH + 1), dilateH));
                    IntType khPadding = std::max<IntType>(0, jcp.kh - iTOverflow - iBOverflow);
                    IntType weiStride = (!jcp.signedInput) ? 0 : (iTOverflow * whtHstride);
                    parConv.src = srcW + iTOverflow * dilateH * srcHstride;
                    parConv.dst = dstW;
                    parConv.filt = whtW + weiStride;
                    parConv.bias = biasW;
                    parConv.compensation = compensationW;
                    parConv.ocBlocks = ocb;
                    parConv.khPadding = khPadding;
                    parConv.scales = scales;
                    parConv.tOverflow = iTOverflow;
                    parConv.bOverflow = iBOverflow;
                    parConv.owb = owb;
                    jitKer(&parConv);
                    srcW += srcHstride * jcp.strideH;
                    dstW += dstHstride;
                }
            }
            if (jcp.loopOrder == KuDNN::KdnnConvLoopOrderT::KUDNN_LOOP_CWGN) {
                NdIteratorJump(start, end, occ, ocChunks, owb, jcp.nbOw, g, nbGroups, n, jcp.mb, ohS, jcp.oh);
            } else if (jcp.loopOrder == KuDNN::KdnnConvLoopOrderT::KUDNN_LOOP_NGCW) {
                NdIteratorJump(start, end, n, jcp.mb, g, nbGroups, occ, ocChunks, owb, jcp.nbOw, ohS, jcp.oh);
            } else if (jcp.loopOrder == KuDNN::KdnnConvLoopOrderT::KUDNN_LOOP_NHWCG) {
                ++start;
                NdIteratorStep(n, jcp.mb, ohS, jcp.oh, owb, jcp.nbOw, occ, ocChunks, g, nbGroups);
            } else {
                throw KuDNN::Service::Unsupported{"unsupported loop order"};
            }
        }
    });

    if (oscales) {
        Service::Deallocate(oscales);
    }
}

template <typename DataType>
void ConvolutionJITFWDImpl::ExecuteForward2D(const void *src, const void *wei,
    void *dst, const void *bias) noexcept(false)
{
    KdnnConvJitConfT& jcp = jcp_;
    if constexpr (std::is_same_v<DataType, float>) {
        if (jcp.isDepthwise) {
            ExecuteForwardDW(src, wei, dst, bias);
            return;
        }
    }
    KdnnConvJitKerT jitKer = GetKernel();
    if (jitKer == nullptr) {
        return;
    }
    const DataType* srcPtr = static_cast<const DataType*>(src);
    const DataType* weiPtr = static_cast<const DataType*>(wei);
    const DataType* biasPtr = static_cast<const DataType*>(bias);
    DataType* dstPtr = static_cast<DataType*>(dst);
    IntType ocChunks = jcp.nbOc / jcp.nbOcBlocking;
    IntType nbGroups = jcp.ngroups;
    IntType workAmount = jcp.mb * nbGroups * ocChunks * jcp.oh * jcp.nbOw;
    IntType nThr = jcp.alignedThreads;
    Parallel(nThr, [&](const IntType& ithr, const IntType& subNThr) {
        IntType start {0};
        IntType end {0};
        IntType startCopy;
        Balance211(workAmount, subNThr, ithr, start, end);
        startCopy = start;
        auto parConv = KuDNN::KdnnConvJitCallS();
        const bool isDstLayoutNxc = jcp.dstTag == KuDNN::Layout::ACDB;
        const bool isSrcLayoutNxc = jcp.srcTag == KuDNN::Layout::ACDB;
        size_t srcHstride = jcp.ngroups * jcp.iw * jcp.ic;
        size_t srcCstride = 1;
        size_t dstHstride = jcp.ngroups * jcp.ow * jcp.oc;
        if (!isSrcLayoutNxc && jcp_.is1stConv) {
            srcHstride = jcp.iw;
            srcCstride = jcp.iw * jcp.ih;
        } else if (!isSrcLayoutNxc && !jcp_.is1stConv) {
            srcHstride = jcp.iw * jcp.icBlock;
            srcCstride = jcp.iw * jcp.ih * jcp.icBlock;
        }
        if (!isDstLayoutNxc) {
            dstHstride = jcp.ow * jcp.ocBlock;
        }
        size_t whtHstride = jcp.icBlock * jcp.ocBlock * jcp.kw;
        size_t whtIcStride = jcp.ngroups * jcp.ic < jcp.ocBlock ? jcp.ocBlock :
            (jcp.icBlock * jcp.ocBlock * jcp.kh * jcp.kw);
        for (IntType icbL2 = 0; icbL2 < jcp.nbIc; icbL2 += jcp.nbIcL2) {
            start = startCopy;
            IntType n {0};
            IntType gg {0};
            IntType occ {0};
            IntType ohS {0};
            IntType owb {0};
            if (jcp.loopOrder == KuDNN::KdnnConvLoopOrderT::KUDNN_LOOP_CWGN) {
                NdIteratorInit(start, occ, ocChunks, owb, jcp.nbOw, gg, nbGroups, n, jcp.mb, ohS, jcp.oh);
            } else if (jcp.loopOrder == KuDNN::KdnnConvLoopOrderT::KUDNN_LOOP_GNCW) {
                NdIteratorInit(start, gg, nbGroups, n, jcp.mb, occ, ocChunks, owb, jcp.nbOw, ohS, jcp.oh);
            } else {
                throw KuDNN::Service::Unsupported{"unsupported loop order"};
            }
            while (start < end) {
                IntType ocb = occ * jcp.nbOcBlocking;
                IntType g = gg;
                IntType gOcb = g * jcp.nbOc + ocb;
                IntType gIcb = g * jcp.nbIc * jcp.nonblkGroupOff;
                IntType workRem = end - start;
                IntType owS = owb * jcp.owBlock;
                IntType iwS = owS * jcp.strideW;
                IntType ohE = ohS + workRem > jcp.oh ? jcp.oh : ohS + workRem;
                for (IntType ohB = ohS; ohB < ohE; ohB += jcp.hBlocking) {
                    IntType ihB = -jcp.tPad + ohB * jcp.strideH;
                    const IntType ocOffIdx = isDstLayoutNxc ? g * jcp.oc + ocb * jcp.ocBlock : gOcb;
                    auto dstW = dstPtr + (n * jcp.oc * jcp.ow * jcp.oh * jcp.ngroups +
                        ocOffIdx + ohB * jcp.oc * jcp.ow * jcp.ngroups + iwS * jcp.oc * jcp.ngroups);
                    if (!isDstLayoutNxc) {
                        dstW = dstPtr + (n * jcp.oc * jcp.ow * jcp.oh * jcp.ngroups + ocOffIdx *
                            jcp.ocBlock * jcp.ow * jcp.oh + ohB * jcp.ocBlock * jcp.ow + iwS * jcp.ocBlock);
                    }
                    const IntType icOffIdx = isSrcLayoutNxc ? g * jcp.ic + icbL2 * jcp.icBlock : gIcb + icbL2;
                    auto srcW = srcPtr + (n * jcp.ic * jcp.iw * jcp.ih * jcp.ngroups +
                        icOffIdx + ihB * jcp.ic * jcp.iw * jcp.ngroups + iwS * jcp.ic * jcp.ngroups);
                    if (!isSrcLayoutNxc && jcp_.is1stConv) {
                        srcW = srcPtr + (n * jcp.ic * jcp.iw * jcp.ih * jcp.ngroups +
                            icOffIdx * jcp.iw * jcp.ih + ihB * jcp.iw + iwS);
                    } else if (!isSrcLayoutNxc && !jcp_.is1stConv) {
                        srcW = srcPtr + (n * jcp.ic * jcp.iw * jcp.ih * jcp.ngroups + icOffIdx *
                            jcp.icBlock * jcp.iw * jcp.ih + ihB * jcp.icBlock * jcp.iw + iwS * jcp.icBlock);
                    }
                    auto whtW = weiPtr + (g * jcp.oc * jcp.ic * jcp.kh * jcp.kw +
                        ocb * jcp.ocBlock * jcp.ic * jcp.kw * jcp.kh +
                        icbL2 * jcp.icBlock * jcp.ocBlock * jcp.kw * jcp.kh);
                    IntType icbStep = isSrcLayoutNxc ? jcp.nbIcL2 : 1;
                    IntType icbL2Sum = icbL2 + jcp.nbIcL2;
                    IntType icbEnd = jcp.nbIc > icbL2Sum ? icbL2Sum : jcp.nbIc;
                    auto biasW = bias ? static_cast<const DataType*>(bias)
                        + ocOffIdx * (isDstLayoutNxc ? 1 : jcp.ocBlock) : nullptr;
                    const IntType ocWork = ThisBlockSize(ocb * jcp.ocBlock, jcp.oc, jcp.nbOcBlocking * jcp.ocBlock);
                    IntType icWork = icbStep * jcp.icBlock;
                    for (IntType icb = icbL2; icb < icbEnd; icb += icbStep) {
                        IntType midIcbVal = icbEnd - icb;
                        IntType icStep = icbStep > midIcbVal ? midIcbVal : icbStep;
                        IntType flags = 0;
                        if (icb == 0) {
                            flags |= static_cast<int32_t>(KuDNN::KdnnConvFlag::KUDNN_FLAG_IC_FIRST);
                        }
                        if (icb + icStep >= jcp.nbIc) {
                            flags |= static_cast<int32_t>(KuDNN::KdnnConvFlag::KUDNN_FLAG_IC_LAST);
                            icWork = ThisBlockSize(icb * jcp.icBlock, jcp.ic, icbStep * jcp.icBlock);
                        }
                        auto srcC = srcW;
                        auto dstC = dstW;
                        IntType ohEMinVal = ohE > (ohB + jcp.hBlocking) ? (ohB + jcp.hBlocking) : ohE;
                        for (IntType oj = ohB, ij = ihB; oj < ohEMinVal; ++oj, ij += jcp.strideH) {
                            IntType dilateH = jcp.dilateH + 1;
                            IntType maxIj = -ij > 0 ? -ij : 0;
                            IntType maxIhVal = ij - jcp.ih + (jcp.kh - 1) * dilateH + 1;
                            IntType realMaxIhVal = maxIhVal > 0 ? maxIhVal : 0;
                            IntType iToverflow = Math::CeilDiv(maxIj, dilateH);
                            IntType iBoverflow = Math::CeilDiv(realMaxIhVal, dilateH);
                            IntType midBOverFlow = jcp.kh - iToverflow - iBoverflow;
                            IntType khPadding = midBOverFlow > 0 ? midBOverFlow : 0;
                            auto auxSrc = srcC + iToverflow * dilateH * srcHstride;
                            auto auxWht = whtW + iToverflow * whtHstride;
                            KdnnJitConvKerPipelineOwThr(jitKer, parConv, auxSrc, dstC, auxWht, biasW, icb,
                                khPadding, owb, icWork, ocWork, flags);
                            srcC += srcHstride * jcp.strideH;
                            dstC += dstHstride;
                        }
                        srcW += srcCstride;
                        whtW += whtIcStride;
                    }
                }
                if (jcp.loopOrder == KuDNN::KdnnConvLoopOrderT::KUDNN_LOOP_CWGN) {
                    NdIteratorJump(start, end, occ, ocChunks, owb, jcp.nbOw, gg, nbGroups, n, jcp.mb, ohS, jcp.oh);
                } else if (jcp.loopOrder == KuDNN::KdnnConvLoopOrderT::KUDNN_LOOP_GNCW) {
                    NdIteratorJump(start, end, gg, nbGroups, n, jcp.mb, occ, ocChunks, owb, jcp.nbOw, ohS, jcp.oh);
                } else {
                    throw KuDNN::Service::Unsupported{"unsupported loop order"};
                }
            }
        }
        KdnnJitConvKerPipelineOwThr(jitKer, parConv, srcPtr, dstPtr, weiPtr, biasPtr, 0, 0, 0, 0, 0, 0);
    });
}

template<typename SrcDataType, typename DstDataType>
void ConvolutionJITFWDImpl::ExecuteForward3D4Int(const void *src, const void *wei,
    void *dst, const void *bias) noexcept(false)
{
    KdnnConvJitConfT& jcp = jcp_;
    KdnnConvJitKerT jitKer = GetKernel();
    if (jitKer == nullptr) {
        return;
    }
#ifdef KUDNNL_920F
    constexpr IntType oscaleLen = OSCALE_LEN_16;
#else
    constexpr IntType oscaleLen = OSCALE_LEN_8;
#endif
    void* oscales = Service::AlignedAlloc(oscaleLen * sizeof(float));
    DataDup(static_cast<float*>(oscales), 1, oscaleLen);
    IntType offset = jcp.ic * jcp.oc * jcp.kd * jcp.kw * jcp.kh * jcp.ngroups;
    auto w = SafeCast<int8_t*>(static_cast<const int8_t*>(wei));
    int32_t* compensation = (!jcp.signedInput) ? reinterpret_cast<int32_t*>(&w[offset]) : nullptr;

    IntType ocChunks = jcp.nbOc / jcp.nbOcBlocking;
    IntType nbGroups = jcp.nbCh;
    IntType workAmount = jcp.mb * nbGroups * ocChunks * jcp.od * jcp.oh * jcp.nbOw;
    IntType nThr = jcp.alignedThreads;
    Parallel(nThr, [&](const IntType& ithr, const IntType& subNThr) {
        IntType start {0};
        IntType end {0};
        Balance211(workAmount, subNThr, ithr, start, end);
        auto parConv = KuDNN::KdnnConvJitCallS();
        IntType srcDstride = jcp.ih * jcp.iw * jcp.icWithoutPadding * jcp.ngroups;
        IntType srcHstride = jcp.iw * jcp.icWithoutPadding * jcp.ngroups;
        IntType dstHstride = jcp.ow * jcp.ocWithoutPadding * jcp.ngroups;
        IntType whtDstride = jcp.icBlock * jcp.ocBlock * jcp.kh * jcp.kw;
        IntType whtHstride = jcp.icBlock * jcp.ocBlock * jcp.kw;
        IntType n {0};
        IntType g {0};
        IntType occ {0};
        IntType odS {0};
        IntType ohS {0};
        IntType owb {0};
        
        if (jcp.loopOrder == KuDNN::KdnnConvLoopOrderT::KUDNN_LOOP_CWGN) {
            NdIteratorInit(start, occ, ocChunks, owb, jcp.nbOw, g, nbGroups, n, jcp.mb, odS, jcp.od, ohS, jcp.oh);
        } else if (jcp.loopOrder == KuDNN::KdnnConvLoopOrderT::KUDNN_LOOP_NGCW) {
            NdIteratorInit(start, n, jcp.mb, g, nbGroups, occ, ocChunks, owb, jcp.nbOw, odS, jcp.od, ohS, jcp.oh);
        } else if (jcp.loopOrder == KuDNN::KdnnConvLoopOrderT::KUDNN_LOOP_NHWCG) {
            NdIteratorInit(start, n, jcp.mb, odS, jcp.od, ohS, jcp.oh, owb, jcp.nbOw, occ, ocChunks, g, nbGroups);
        } else {
            throw KuDNN::Service::Unsupported{"unsupported loop order"};
        }
        while (start < end) {
            for (IntType occ1 = 0; occ1 < jcp.nbOcBlockingThrChunk; occ1 += jcp.nbOcBlocking) {
                IntType ocb = occ * jcp.nbOcBlockingThrChunk + occ1;
                IntType gOc = (g * jcp.nbOc + ocb) * jcp.ocBlock;
                IntType gIc = g * jcp.nbIc * jcp.icBlock;
                IntType workRem = end - start;
                IntType ihS = -jcp.tPad + ohS * jcp.strideH;
                IntType ohE = ohS + workRem > jcp.oh ? jcp.oh : ohS + workRem;
                if (jcp.loopOrder == KuDNN::KdnnConvLoopOrderT::KUDNN_LOOP_NHWCG) {
                    ohE = ohS + 1; // step instead
                }
                IntType owS = owb * jcp.owBlock;
                IntType iwS = owS * jcp.strideW;
                IntType idS = -jcp.fPad + odS * jcp.strideD;
                IntType dilateD = jcp.dilateD + 1;
                IntType dFOverflow = std::min<IntType>(jcp.kd, Math::CeilDiv(std::max<IntType>(0, -idS), dilateD));
                IntType dBackOverflow = std::min<IntType>(jcp.kd,
                    Math::CeilDiv(std::max<IntType>(0, idS - jcp.id + (jcp.kd - 1) * dilateD + 1), dilateD));
                auto scales = &static_cast<float*>(oscales)[jcp.isOcScale * gOc];
                IntType kdPadding = std::max<IntType>(0, jcp.kd - dFOverflow - dBackOverflow);
                int32_t *compensationW = (!jcp.signedInput) ? compensation + gOc : nullptr;

                auto dstW = static_cast<DstDataType*>(dst) +
                    (n * jcp.ocWithoutPadding * jcp.ow * jcp.oh * jcp.od * jcp.ngroups +
                    gOc + odS * jcp.ngroups * jcp.ocWithoutPadding * jcp.ow * jcp.oh +
                    ohS * jcp.ngroups * jcp.ocWithoutPadding * jcp.ow + owS * jcp.ngroups * jcp.ocWithoutPadding);
                auto srcW = static_cast<const SrcDataType*>(src) + (n * jcp.icWithoutPadding * jcp.iw * jcp.ih *
                    jcp.id * jcp.ngroups + gIc + idS * jcp.ngroups * jcp.icWithoutPadding * jcp.iw * jcp.ih +
                    ihS * jcp.ngroups * jcp.icWithoutPadding * jcp.iw +
                    iwS * jcp.ngroups * jcp.icWithoutPadding + dFOverflow * dilateD * srcDstride);
                auto whtW = static_cast<const int8_t*>(wei) + (ocb * jcp.ocBlock * jcp.ic *
                    jcp.kh * jcp.kw * jcp.kd + (!jcp.signedInput ? 0 : dFOverflow) * whtDstride);
                auto biasW = bias ? (static_cast<const char*>(bias) + gOc * jcp.typeSizeBia) : nullptr;
                for (IntType oj = ohS, ij = ihS; oj < ohE; ++oj, ij += jcp.strideH) {
                    IntType iTOverflow = std::min<IntType>(jcp.kh,
                        Math::CeilDiv(std::max<IntType>(0, -ij), jcp.dilateH + 1));
                    IntType iBOverflow = std::min<IntType>(jcp.kh, Math::CeilDiv(std::max<IntType>(0,
                        ij - jcp.ih + (jcp.kh - 1) * (jcp.dilateH + 1) + 1), jcp.dilateH + 1));
                    IntType khPadding = std::max<IntType>(0, jcp.kh - iTOverflow - iBOverflow);
                    IntType weiStride = !jcp.signedInput ? 0 : whtHstride * iTOverflow;
                    parConv.src = srcW + iTOverflow * (jcp.dilateH + 1) * srcHstride;
                    parConv.dst = dstW;
                    parConv.filt = whtW + weiStride;
                    parConv.bias = biasW;
                    parConv.compensation = compensationW;
                    parConv.ocBlocks = ocb;
                    parConv.khPadding = khPadding;
                    parConv.kdPadding = kdPadding;
                    parConv.scales = scales;
                    parConv.tOverflow = iTOverflow;
                    parConv.bOverflow = iBOverflow;
                    parConv.fOverflow = dFOverflow;
                    parConv.backOverflow = dBackOverflow;
                    parConv.owb = owb;
                    jitKer(&parConv);
                    srcW += srcHstride * jcp.strideH;
                    dstW += dstHstride;
                }
            }
            if (jcp.loopOrder == KuDNN::KdnnConvLoopOrderT::KUDNN_LOOP_CWGN) {
                NdIteratorJump(start, end, occ, ocChunks, owb, jcp.nbOw, g, nbGroups, n,
                    jcp.mb, odS, jcp.od, ohS, jcp.oh);
            } else if (jcp.loopOrder == KuDNN::KdnnConvLoopOrderT::KUDNN_LOOP_NGCW) {
                NdIteratorJump(start, end, n, jcp.mb, g, nbGroups, occ, ocChunks, owb,
                    jcp.nbOw, odS, jcp.od, ohS, jcp.oh);
            } else if (jcp.loopOrder == KuDNN::KdnnConvLoopOrderT::KUDNN_LOOP_NHWCG) {
                ++start;
                NdIteratorStep(n, jcp.mb, odS, jcp.od, ohS, jcp.oh, owb, jcp.nbOw, occ, ocChunks, g, nbGroups);
            } else {
                throw KuDNN::Service::Unsupported{"unsupported loop order"};
            }
        }
    });
    if (oscales) {
        Service::Deallocate(oscales);
    }
}

template <typename DataType>
void ConvolutionJITFWDImpl::ExecuteForward3D(const void *src, const void *wei,
    void *dst, const void *bias) noexcept(false)
{
    KdnnConvJitConfT& jcp = jcp_;
    KdnnConvJitKerT jitKer = GetKernel();
    if (jitKer == nullptr) {
        return;
    }
    IntType ocChunks = jcp.nbOc / jcp.nbOcBlocking;
    IntType nGroups = jcp.ngroups;
    IntType workAmount = jcp.mb * nGroups * ocChunks * jcp.od * jcp.oh * jcp.nbOw;
    IntType nThr = jcp.alignedThreads;
    const DataType* srcPtr = static_cast<const DataType*>(src);
    const DataType* weiPtr = static_cast<const DataType*>(wei);
    const DataType* biasPtr = static_cast<const DataType*>(bias);
    DataType* dstPtr = static_cast<DataType*>(dst);
    Parallel(nThr, [&](const IntType& ithr, const IntType& subNThr) {
        IntType start {0};
        IntType end {0};
        IntType startCopy;
        Balance211(workAmount, subNThr, ithr, start, end);
        startCopy = start;
        auto parConv = KuDNN::KdnnConvJitCallS();
        const bool isDstLayoutNxc = jcp.dstTag == KuDNN::Layout::ACDEB;
        const bool isSrcLayoutNxc = jcp.srcTag == KuDNN::Layout::ACDEB;
        size_t srcDstride = jcp.ngroups * jcp.ih * jcp.iw * jcp.ic;
        size_t srcHstride = jcp.ngroups * jcp.iw * jcp.ic;
        size_t srcCstride = 1;
        size_t dstHstride = jcp.oc * jcp.ow;
        if (!isSrcLayoutNxc && jcp_.is1stConv) { // when src layout is not axb / aBcde16b, and is abx
            srcDstride = jcp.ih * jcp.iw;
            srcHstride = jcp.iw;
            srcCstride = jcp.id * jcp.iw * jcp.ih;
        } else if (!isSrcLayoutNxc && !jcp_.is1stConv) { // when src layout is not axb, and is aBcde16b
            srcDstride = jcp.ih * jcp.iw * jcp.icBlock;
            srcHstride = jcp.iw * jcp.icBlock;
            srcCstride = jcp.id * jcp.iw * jcp.ih * jcp.icBlock;
        }
        if (!isDstLayoutNxc) {
            dstHstride = jcp.ocBlock * jcp.ow;
        }
        size_t whtDstride = jcp.icBlock * jcp.ocBlock * jcp.kh * jcp.kw;
        size_t whtHstride = jcp.icBlock * jcp.ocBlock * jcp.kw;
        size_t whtIcStride = jcp.ngroups * jcp.ic < jcp.ocBlock ? jcp.ocBlock :
            (jcp.icBlock * jcp.ocBlock * jcp.kh * jcp.kw * jcp.kd);
        for (IntType icbL2 = 0; icbL2 < jcp.nbIc; icbL2 += jcp.nbIcL2) {
            start = startCopy;
            IntType n {0};
            IntType gg {0};
            IntType occ {0};
            IntType ohS {0};
            IntType odS {0};
            IntType owb {0};
            if (jcp.loopOrder == KuDNN::KdnnConvLoopOrderT::KUDNN_LOOP_CWGN) {
                NdIteratorInit(start, occ, ocChunks, owb, jcp.nbOw, gg, nGroups, n, jcp.mb, odS, jcp.od, ohS, jcp.oh);
            } else if (jcp.loopOrder == KuDNN::KdnnConvLoopOrderT::KUDNN_LOOP_GNCW) {
                NdIteratorInit(start, gg, nGroups, n, jcp.mb, occ, ocChunks, owb, jcp.nbOw, odS, jcp.od, ohS, jcp.oh);
            } else {
                throw KuDNN::Service::Unsupported{"unsupported loop order"};
            }

            while (start < end) {
                IntType ocb = occ * jcp.nbOcBlocking;
                IntType g = gg;
                IntType gOcb = g * jcp.nbOc + ocb;
                IntType gIcb = g * jcp.nbIc * jcp.nonblkGroupOff;
                IntType workRem = end - start;
                IntType ihS = -jcp.tPad + ohS * jcp.strideH;
                IntType owS = owb * jcp.owBlock;
                IntType iwS = owS * jcp.strideW;
                IntType ohE = ohS + workRem > jcp.oh ? jcp.oh : ohS + workRem;
                IntType idS = -jcp.fPad + odS * jcp.strideD;
                IntType dilateD = jcp.dilateD + 1;
                IntType maxIds = -idS > 0 ? -idS : 0;
                IntType midIdVal = idS - jcp.id + (jcp.kd - 1) * dilateD + 1;
                IntType maxIdVal = midIdVal > 0 ? midIdVal : 0;
                IntType dToverflow = Math::CeilDiv(maxIds, dilateD);
                IntType dBoverflow = Math::CeilDiv(maxIdVal, dilateD);
                IntType midKdVal = jcp.kd - dToverflow - dBoverflow;
                IntType kdPadding = midKdVal > 0 ? midKdVal : 0;
                const IntType ocOffIdx = isDstLayoutNxc ? (g * jcp.oc + ocb * jcp.ocBlock) : gOcb;
                auto dstW = dstPtr + (n * jcp.oc * jcp.ow * jcp.oh * jcp.od * jcp.ngroups +
                    ocOffIdx + odS * jcp.ngroups * jcp.oc * jcp.ow * jcp.oh +
                    ohS * jcp.ngroups * jcp.oc * jcp.ow + owS * jcp.ngroups * jcp.oc);
                if (!isDstLayoutNxc) {
                    dstW = dstPtr + (n * jcp.oc * jcp.ow * jcp.oh * jcp.od * jcp.ngroups +
                        ocOffIdx * jcp.od * jcp.ow * jcp.oh * jcp.ocBlock +
                        odS * jcp.ocBlock * jcp.ow * jcp.oh + ohS * jcp.ocBlock * jcp.ow + owS * jcp.ocBlock);
                }
                const IntType icOffIdx = isSrcLayoutNxc ? g * jcp.ic + icbL2 * jcp.icBlock : gIcb + icbL2;
                auto srcW = srcPtr + (n * jcp.ic * jcp.iw * jcp.ih * jcp.id * jcp.ngroups +
                    icOffIdx + idS * jcp.ngroups * jcp.ic * jcp.iw * jcp.ih +
                    ihS * jcp.ngroups * jcp.ic * jcp.iw + iwS * jcp.ngroups * jcp.ic +
                    dToverflow * dilateD * srcDstride);
                if (!isSrcLayoutNxc && jcp_.is1stConv) { // when src layout is not axb / aBcde16b, and is abx
                    srcW = srcPtr + (n * jcp.ic * jcp.iw * jcp.ih * jcp.id * jcp.ngroups +
                        icOffIdx * jcp.id * jcp.iw * jcp.ih + idS * jcp.iw * jcp.ih +
                        ihS * jcp.iw + iwS + dToverflow * dilateD * srcDstride);
                } else if (!isSrcLayoutNxc && !jcp_.is1stConv) { // when src layout is not axb, and is aBcde16b
                    srcW = srcPtr + (n * jcp.ic * jcp.iw * jcp.ih * jcp.id * jcp.ngroups +
                        icOffIdx * jcp.id * jcp.iw * jcp.ih * jcp.icBlock +
                        idS * jcp.icBlock * jcp.iw * jcp.ih + ihS * jcp.icBlock * jcp.iw +
                        iwS * jcp.icBlock + dToverflow * dilateD * srcDstride);
                }
                auto whtW = weiPtr + (ocb * jcp.ocBlock * jcp.ic * jcp.kh * jcp.kw * jcp.kd +
                    icbL2 * jcp.ocBlock * jcp.ocBlock * jcp.kw * jcp.kh * jcp.kd + dToverflow * whtDstride);
                auto biasW = bias ? biasPtr + ocOffIdx * (isDstLayoutNxc ? 1 : jcp.ocBlock) : nullptr;
                const IntType icbStep = isSrcLayoutNxc ? jcp.nbIcL2 : 1;
                IntType icbEnd = std::min<IntType>(jcp.nbIc, icbL2 + jcp.nbIcL2);
                const IntType ocWork = ThisBlockSize(ocb * jcp.ocBlock, jcp.oc, jcp.nbOcBlocking * jcp.ocBlock);
                IntType icWork = icbStep * jcp.icBlock;
                for (IntType icb = icbL2; icb < icbEnd; icb += icbStep) {
                    IntType currNbIc = std::min<IntType>(icbStep, icbEnd - icb);
                    IntType flags = 0;
                    if (icb == 0) {
                        flags |= static_cast<int32_t>(KuDNN::KdnnConvFlag::KUDNN_FLAG_IC_FIRST);
                    }
                    if (icb + currNbIc >= jcp.nbIc) {
                        flags |= static_cast<int32_t>(KuDNN::KdnnConvFlag::KUDNN_FLAG_IC_LAST);
                        icWork = ThisBlockSize(icb * jcp.icBlock, jcp.ic, icbStep * jcp.icBlock);
                    }
                    auto srcC = srcW;
                    auto dstC = dstW;
                    for (IntType oj = ohS, ij = ihS; oj < ohE;
                            ++oj, ij += jcp.strideH) {
                        IntType dilateH = jcp.dilateH + 1;
                        IntType midIh = ij - jcp.ih + (jcp.kh - 1) * dilateH + 1;
                        IntType iToverflow = Math::CeilDiv((-ij > 0 ? -ij : 0), dilateH);
                        IntType iBoverflow = Math::CeilDiv((midIh > 0 ? midIh : 0), dilateH);
                        IntType midKh = jcp.kh - iToverflow - iBoverflow;
                        IntType khPadding = midKh > 0 ? midKh : 0;
                        KdnnJitVCConv3dKerPipelineOwThr(jitKer, parConv,
                            srcC + iToverflow * dilateH * srcHstride,
                            dstC, whtW + iToverflow * whtHstride,
                            biasW, icb, khPadding, kdPadding, owb,
                            icWork, ocWork, flags);
                        srcC += srcHstride * jcp.strideH;
                        dstC += dstHstride;
                    }
                    srcW += srcCstride;
                    whtW += whtIcStride;
                }
                if (jcp.loopOrder == KuDNN::KdnnConvLoopOrderT::KUDNN_LOOP_CWGN) {
                    NdIteratorJump(start, end, occ, ocChunks, owb, jcp.nbOw,
                        gg, nGroups, n, jcp.mb, odS, jcp.od, ohS, jcp.oh);
                } else if (jcp.loopOrder == KuDNN::KdnnConvLoopOrderT::KUDNN_LOOP_GNCW) {
                    NdIteratorJump(start, end, gg, nGroups, n, jcp.mb, occ,
                        ocChunks, owb, jcp.nbOw, odS, jcp.od, ohS, jcp.oh);
                } else {
                    throw KuDNN::Service::Unsupported{"unsupported loop order"};
                }
            }
        }
        KdnnJitVCConv3dKerPipelineOwThr(jitKer, parConv, srcPtr, dstPtr, weiPtr, biasPtr, 0, 0, 0, 0, 0, 0, 0);
    });
}
} // Detail

ConvolutionLayerFWD::~ConvolutionLayerFWD() noexcept {}

ConvolutionLayerFWD::ConvolutionLayerFWD(int isa, const TensorInfo &src, const TensorInfo &weights,
    const TensorInfo &dst, const TensorInfo &bias, KuDNN::Shape strides, KuDNN::Shape paddingL,
    KuDNN::Shape dilates, KuDNN::Propagation prop, int nthreads, bool isLayoutBlock, bool bUseWino) noexcept(false)
{
    bool isUseJitConv = false;
    int brgBlk = 0;
    Service::ThrowOnStatus(ConvolutionLayerFWD::ValidateInput(src, weights, dst, bias,
        strides, dilates, paddingL, isa, isLayoutBlock, isUseJitConv, bUseWino,
        brgBlk, nthreads), "ConvolutionJIT layer");
    if (isUseJitConv) {
        pImpl = std::make_unique<Detail::ConvolutionJITFWDImpl>(isa, src, weights, dst, bias, strides,
            paddingL, dilates, prop, nthreads, isLayoutBlock);
    } else {
        KuDNN::Shape paddingR = GetPadR(src, weights, dst, strides, dilates, paddingL);
        if (bUseWino) {
            pImpl = std::make_unique<Detail::WinoConvolutionLayerFWDImpl>(src, weights, dst, bias, strides,
                dilates, paddingL, paddingR, ConvolutionAlgorithm::DIRECT, nthreads);
        } else {
            pImpl = std::make_unique<Detail::ConvolutionLayerFWDImpl>(src, weights, dst, bias, strides,
                dilates, paddingL, paddingR, ConvolutionAlgorithm::DIRECT);
        }
    }
}

ConvolutionLayerFWD::ConvolutionLayerFWD(const TensorInfo &src, const TensorInfo &weights,
    const TensorInfo &dst, const TensorInfo &bias, const Shape &strides,
    const Shape &paddingL, const Shape &paddingR, ConvolutionAlgorithm alg) noexcept(false)
{
    pImpl = std::make_unique<Detail::ConvolutionLayerFWDImpl>(src, weights, dst, bias, strides, Shape(0, 0, 0),
        paddingL, paddingR, alg);
}

ConvolutionLayerFWD::ConvolutionLayerFWD(const TensorInfo &src, const TensorInfo &weights,
    const TensorInfo &dst, const TensorInfo &bias, const Shape &strides, const Shape &dilates,
    const Shape &paddingL, const Shape &paddingR, ConvolutionAlgorithm alg) noexcept(false)
{
    pImpl = std::make_unique<Detail::ConvolutionLayerFWDImpl>(src, weights, dst, bias, strides, dilates,
        paddingL, paddingR, alg);
}

ConvolutionLayerFWD::ConvolutionLayerFWD(const ConvolutionLayerFWD &other) noexcept(false)
{
    pImpl = std::make_unique<Detail::ConvolutionFWDImplBase>(*(other.pImpl.get()));
}

ConvolutionLayerFWD::ConvolutionLayerFWD(ConvolutionLayerFWD &&other) noexcept
    : pImpl(std::move(other.pImpl)) {}

ConvolutionLayerFWD& ConvolutionLayerFWD::operator=(const ConvolutionLayerFWD &other) noexcept(false)
{
    if (this != &other) {
        this->pImpl.reset(new Detail::ConvolutionFWDImplBase{*(other.pImpl.get())});
    }
    return *this;
}

ConvolutionLayerFWD& ConvolutionLayerFWD::operator=(ConvolutionLayerFWD &&other) noexcept
{
    if (this != &other) {
        this->pImpl = std::move(other.pImpl);
    }
    return *this;
}

void ConvolutionLayerFWD::Run(const void *src, const void *wei,
    void *dst, const void *bias) noexcept(false)
{
    pImpl->Run(src, wei, dst, bias);
}

Status ConvolutionLayerFWD::ValidateInput(const TensorInfo &src, const TensorInfo &weights,
    const TensorInfo &dst, const TensorInfo &bias, const Shape &strides, const Shape &dilates,
    const Shape &paddingL, int kCpuIsa, bool isLayoutBlock, bool &isJitConv, bool bUseWino) noexcept
{
    int brgBlock = 0;
    int nthr = 1;
    return ValidateInput(src, weights, dst, bias, strides, dilates, paddingL, kCpuIsa,
        isLayoutBlock, isJitConv, bUseWino, brgBlock, nthr);
}

Status ConvolutionLayerFWD::ValidateInput(const TensorInfo &src, const TensorInfo &weights,
    const TensorInfo &dst, const TensorInfo &bias, const Shape &strides,
    const Shape &dilates, const Shape &paddingL, int kCpuIsa,
    bool isLayoutBlock, bool &isJitConv, bool bUseWino, int &brgBlock, int nthreads) noexcept
{
    isJitConv = false;
    brgBlock = 0;
    bool isKdnnConvSupport = CheckKdnnConvInvalid(src, weights, dst, bias, strides, dilates, paddingL);
    if (kCpuIsa == static_cast<IntType>(ARMCG::CPUISAT::VC_256)) {
        if (bUseWino) {
            return Status::SUCCESS;
        }
    }

    auto dealKdnnIsOrNotSupport = [isKdnnConvSupport, &isJitConv] () -> Status {
        if (!isKdnnConvSupport) {
            return Status::UNSUPPORTED;
        }
        isJitConv = false;
        return Status::SUCCESS;
    };
    if (!isLayoutBlock) {
        return dealKdnnIsOrNotSupport();
    }

    if (isLayoutBlock) {
        KdnnBrgconvDataT bcd = KdnnBrgconvDataT();
        Status s = KdnnBrgconvInitConf(bcd, static_cast<ARMCG::CPUISAT>(kCpuIsa), src, weights, dst, bias, strides,
            dilates, paddingL, Propagation::FORWARD, nthreads);
        if (s == Status::SUCCESS) {
            isJitConv = true;
            brgBlock = bcd.bcp.ocBlock;
            return Status::SUCCESS;
        }
    }

    if (CheckDwConv(src, weights, dst, kCpuIsa)) {
        if (ValidateInputDwConv(src, weights, dst, strides, dilates, paddingL, kCpuIsa) == Status::SUCCESS) {
            isJitConv = true;
            return Status::SUCCESS;
        } else {
            return dealKdnnIsOrNotSupport();
        }
    }
    const bool withGroups = weights.GetNumDims() == src.GetNumDims() + 1;
    IntType ngroups = withGroups ? weights.GetDims()[0] : 1;
    bool isGroupsOne = ngroups == 1;
    IntType oc = dst.GetDims()[1] / ngroups;
    IntType ocWithoutPadding = oc;
    IntType ic = src.GetDims()[1] / ngroups;
    IntType icWithoutPadding = ic;
    IntType simdW = GetSimdW(src.GetType());

    bool isDataLayoutNxc = IsLayoutNxc(src) && IsLayoutNxc(dst);
    bool is1stConv = (ic < simdW) && isGroupsOne;
    IntType icBlock = is1stConv ? ic : simdW;
    IntType ocBlock = simdW;
    // for generate sve512 kernel, but user try to use sve256 kernel
    if (CheckIsaIsNotMatch(kCpuIsa, simdW, src.GetType())) {
        return dealKdnnIsOrNotSupport();
    }
    
    bool isIntType = Service::OneOf(src.GetType(), Element::TypeT::U8, Element::TypeT::S8) &&
            Service::OneOf(dst.GetType(), Element::TypeT::U8, Element::TypeT::S8, Element::TypeT::S32,
            Element::TypeT::F32) && (weights.GetType() == Element::TypeT::S8);
#ifdef KUDNNL_920B
    if (isIntType) {
        icBlock = IC_LEN8;
        ocBlock = IC_LEN8;
    }
#else
    if (isIntType) {
        icBlock = IC_LEN16;
        ocBlock = IC_LEN16;
    }
#endif
    if (isIntType && isGroupsOne) {
        /* For non grouped convolutions, pad channels by 16 if needed */
        oc = RndUp(oc, ocBlock);
        ic = RndUp(ic, icBlock);
    } else if (isIntType && !isGroupsOne && ((ic % icBlock != 0) || (oc % ocBlock != 0))) {
        icBlock = (ic % IC_LEN8 == 0) && (oc % IC_LEN8 == 0) ? IC_LEN8 : IC_LEN4;
        ocBlock = icBlock;
    }
#ifdef KUDNNL_920B
    if (isIntType && ((icWithoutPadding < IC_LEN8 || ocWithoutPadding < IC_LEN8) ||
        (ic % icBlock != 0 || oc % ocBlock != 0))) {
        return dealKdnnIsOrNotSupport();
    }
#else
    if (isIntType && ((icWithoutPadding < IC_LEN16 || ocWithoutPadding < IC_LEN16) ||
        (ic % icBlock != 0 || oc % ocBlock != 0))) {
        return dealKdnnIsOrNotSupport();
    }
#endif
    IntType ndims = src.GetNumDims();
    IntType fPad = (ndims == KERNEL_DIMS_5) ? paddingL[0] : 0;
    IntType tPad = (ndims == KERNEL_DIMS_3) ? 0 : paddingL[ndims - KERNEL_DIMS_4];
    IntType lPad = paddingL[ndims - KERNEL_DIMS_3];
    IntType id = (ndims == KERNEL_DIMS_5) ? src.GetDims()[KERNEL_DIMS_2] : 1;
    IntType ih = (ndims == KERNEL_DIMS_3) ? 1 : src.GetDims()[ndims - KERNEL_DIMS_2];
    IntType iw = src.GetDims()[ndims - 1];
    IntType ow = dst.GetDims()[ndims - 1];
    IntType od = (ndims == KERNEL_DIMS_5) ? dst.GetDims()[KERNEL_DIMS_2] : 1;
    IntType oh = (ndims == KERNEL_DIMS_3) ? 1 : dst.GetDims()[ndims - KERNEL_DIMS_2];
    IntType kh = (ndims == KERNEL_DIMS_3) ? 1 : weights.GetDims()[withGroups + ndims - KERNEL_DIMS_2];
    IntType kw = weights.GetDims()[withGroups + ndims - 1];
    IntType kd = (ndims == KERNEL_DIMS_5) ? weights.GetDims()[withGroups + KERNEL_DIMS_2] : 1;
    IntType strideH = (ndims == KERNEL_DIMS_3) ? 1 : strides[ndims - KERNEL_DIMS_4];
    IntType strideW = strides[ndims - KERNEL_DIMS_3];
    IntType strideD = (ndims == KERNEL_DIMS_5) ? strides[0] : 1;
    bool isPadNotEqSupport = (ndims == KERNEL_DIMS_4) ? (ih < oh && tPad != lPad) : (id < od && (fPad != lPad));
    bool isPadEqSupport = (ndims == KERNEL_DIMS_4) ? (ih < oh && iw < ow) : (id < od && ih < oh && iw < ow);
    bool isSwKwSupport = ndims == KERNEL_DIMS_5 && strideD != 1 && kd > NUM_3;
    if (isPadNotEqSupport || isPadEqSupport || isSwKwSupport) {
        return dealKdnnIsOrNotSupport();
    }
    if (isIntType) {
        IntType dilateD = (ndims == KERNEL_DIMS_5) ? dilates[0] : 0;
        IntType dilateH = (ndims == KERNEL_DIMS_3) ? 0 : dilates[ndims - KERNEL_DIMS_4];
        IntType dilateW = dilates[ndims - KERNEL_DIMS_3];

        IntType extKw = CalculateExtendedFilterSize(kw, dilateW);
        IntType extKh = CalculateExtendedFilterSize(kh, dilateH);
        IntType extKd = CalculateExtendedFilterSize(kd, dilateD);
        IntType rPad = CalculateEndPadding(lPad, ow, iw, strideW, extKw);
        IntType bPad = CalculateEndPadding(tPad, oh, ih, strideH, extKh);
        IntType backPad = CalculateEndPadding(fPad, od, id, strideD, extKd);
        bool kernelOutsideSrc = false || extKw <= lPad || extKw <= rPad || extKh <= tPad || extKh <= bPad
            || extKd <= fPad || extKd <= backPad;
        if (kernelOutsideSrc) {
            return dealKdnnIsOrNotSupport();
        }
        const IntType regs = REG_INDEX28;
        IntType nbOcBlocking = 1;
        IntType nbOc = Math::CeilDiv(oc, ocBlock);
        IntType urW = std::min<IntType>(ow, regs);
        auto isOcBlockingOk = [ow, nbOc, lPad](IntType block) {
            IntType tmpUrW = std::min<IntType>(ow, REG_INDEX31 / (block + 1));
            return nbOc % block == 0 && lPad <= tmpUrW && ow % tmpUrW != 1;
        };
        IntType nbOcBlockingThrChunk = std::min<IntType>(IC_LEN4, nbOc);
        for (; nbOcBlockingThrChunk > 1; nbOcBlockingThrChunk--) {
            if (isOcBlockingOk(nbOcBlockingThrChunk)) {
                break;
            }
        }
        nbOcBlocking = nbOcBlockingThrChunk;
        urW = REG_INDEX31 / (nbOcBlocking + 1);
        urW = ow < urW ? ow : urW;
        if (urW < ow) {
            const IntType broadcastSize = IC_LEN4;
            if (icWithoutPadding % broadcastSize != 0) {
                while (urW > 0) {
                    IntType lastBlockSize = (ow % urW == 0) ? urW : ow % urW;
                    IntType penultimateIwIindex = (ow - 1 - lastBlockSize) * strideW
                        + (kw - 1) * (dilateW + 1) - lPad;
                    IntType penultimateIwLeeway = (iw - 1 - penultimateIwIindex) * icWithoutPadding
                        + icWithoutPadding % broadcastSize;
                    if (penultimateIwLeeway >= broadcastSize) {
                        break;
                    }
                    --urW;
                }
            }
        }
        if (oc % ocBlock != 0 || lPad > urW) {
            return dealKdnnIsOrNotSupport();
        }
        IntType urwTail = ow % urW;
        IntType rPadNoTail = std::max<IntType>(0, CalculateEndPadding(lPad, ow - urwTail, iw, strideW, extKw));
        if (rPadNoTail > urW) {
            return dealKdnnIsOrNotSupport();
        }
    }
    if (CheckDTypeIsSameF(src, weights, dst) && src.GetType() == Element::TypeT::F32) {
#ifdef KUDNNL_920B
        if (kCpuIsa == static_cast<IntType>(ARMCG::CPUISAT::VC_256) &&
            !isGroupsOne && ic % IC_LEN4 == 0 && oc % IC_LEN4 == 0) {
            isJitConv = true;
            return Status::SUCCESS;
        }
#endif
        IntType ocPaded = RndUp(oc, ocBlock);
        IntType icPaded = RndUp(ic, icBlock);
        /* Input and output channels must be multiples of simdW */
        if (isGroupsOne && !isDataLayoutNxc) {
            oc = RndUp(oc, ocBlock);
            ic = RndUp(ic, icBlock);
        }

        bool argsOk = true && (ic <= icPaded) && (oc <= ocPaded);
        if (!argsOk) {
            return dealKdnnIsOrNotSupport();
        }
    } else if (CheckDTypeIsSameF(src, weights, dst)) {
        if (simdW == IC_LEN16 && isGroupsOne && !isDataLayoutNxc) {
            oc = RndUp(oc, ocBlock);
            ic = RndUp(ic, icBlock);
        } else {
            icBlock = simdW;
            ocBlock = simdW;
        }
    } else if (isIntType) {
        isJitConv = true;
        return Status::SUCCESS;
    } else {
        return dealKdnnIsOrNotSupport();
    }
    if (!(oc % ocBlock == 0 && ic % icBlock == 0)) {
        return dealKdnnIsOrNotSupport();
    }
    isJitConv = true;
    return Status::SUCCESS;
}
} // KuDNN
