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

#include <cassert>
#include <climits>
#include <omp.h>
#include <algorithm>
#include <arm_sve.h>
#include "kudnn.hpp"
#include "tools/cg_cpu_isa_traits.hpp"
#include "kernels/convolution/kudnn_conv_const.hpp"
#include "kernels/convolution/kudnn_jit_dw_conv_kernel.hpp"
#include "kernels/convolution/kudnn_jit_sve_conv_kernel.hpp"
#include "kernels/convolution/kudnn_jit_MatrixComputation_conv_kernel.hpp"
#include "kernels/convolution/kudnn_primitive_conf.hpp"
#include "internal_funcs.hpp"
#include "types/kudnn_tensor_info.hpp"
#include "parallel_for.h"

namespace KuDNN {

using KudnnConvJitKerT = void (*)(KuDNN::KudnnConvJitCallS *);
using KudnnConvMatrixComputationJitKerT = void (*)(KuDNN::JitGemmCallS *);
namespace {
constexpr IntType CACHE_ALIAS_1024 = 1024;
constexpr IntType LIMIT_4 = 4;
constexpr IntType LIMIT_7 = 7;
constexpr IntType NB_CH_BLKING_2 = 2;
constexpr IntType NB_CH_BLKING_3 = 3;
constexpr IntType NB_CH_BLKING_4 = 4;
constexpr IntType UR_W_3 = 3;
constexpr IntType UR_W_4 = 4;
constexpr IntType UR_W_6 = 6;

inline void KudnnJitConvKerPipeline(KudnnConvJitKerT ker, KuDNN::KudnnConvJitCallS &p,
    const void *src, const void *dst, const void *filt, const void *bias,
    int channel, int khPadding, int reduceWork, int loadWork)
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

inline void KudnnJitConvKerPipelineOwThr(KudnnConvJitKerT ker,
    KuDNN::KudnnConvJitCallS &p, const void *src, const void *dst,
    const void *filt, const void *bias, int channel, int khPadding,
    int owb, int reduceWork, int loadWork, int flags)
{
    p.owb = p.owbPrf;
    p.owbPrf = owb;
    p.flags = p.flagsPrf;
    p.flagsPrf = flags;
    KudnnJitConvKerPipeline(ker, p, src, dst, filt, bias, channel, khPadding,
        reduceWork, loadWork);
}

inline void KudnnJitSveConv3dKerPipeline(KudnnConvJitKerT ker,
    KuDNN::KudnnConvJitCallS &p, const void *src, const void *dst, const void *filt,
    const void *bias, int channel, int khPadding, int kdPadding, int reduceWork,
    int loadWork)
{
    p.kdPadding = p.kdPaddingPrf;
    p.kdPaddingPrf = kdPadding;
    KudnnJitConvKerPipeline(ker, p, src, dst, filt, bias, channel, khPadding,
        reduceWork, loadWork);
}

// The special case for the driver with ow-parallelization (FWD)
inline void KudnnJitSveConv3dKerPipelineOwThr(KudnnConvJitKerT ker,
    KuDNN::KudnnConvJitCallS &p, const void *src, const void *dst, const void *filt,
    const void *bias, int channel, int khPadding, int kdPadding, int owb,
    int reduceWork, int loadWork, int flags)
{
    p.owb = p.owbPrf;
    p.owbPrf = owb;
    p.flags = p.flagsPrf;
    p.flagsPrf = flags;
    KudnnJitSveConv3dKerPipeline(ker, p, src, dst, filt, bias, channel,
        khPadding, kdPadding, reduceWork, loadWork);
}

inline bool CheckDwConv(const TensorInfo &src, const TensorInfo &wei,
    const TensorInfo &dst, int kCpuIsa)
{
    if (src.GetType() != KuDNN::Element::TypeT::F32 || wei.GetType() != KuDNN::Element::TypeT::F32
        || dst.GetType() != KuDNN::Element::TypeT::F32) {
        return false;
    }
    if (src.GetNumDims() != wei.GetNumDims() - 1) {
        return false;
    }
    if (src.GetDims()[1] != wei.GetDims()[0] ||
        dst.GetDims()[1] != wei.GetDims()[0]) {
        return false;
    }
    // only support 2D case
    if (src.GetNumDims() != KERNEL_DIMS_4) {
        return false;
    }
    // layout is nxc and g/ic/oc with padding is unsupported
    IntType simdW = 0;
    ARMCG::CPUISAT isa = static_cast<ARMCG::CPUISAT>(kCpuIsa);
    if (isa == ARMCG::CPUISAT::VectorComputation_512) {
        simdW = ARMCG::CpuIsaTraits<ARMCG::CPUISAT::VectorComputation_512>::VLEN / sizeof(float);
    } else if (isa == ARMCG::CPUISAT::VectorComputation_256) {
        simdW = ARMCG::CpuIsaTraits<ARMCG::CPUISAT::VectorComputation_256>::VLEN / sizeof(float);
    } else {
        return false;
    }
    if (src.GetLayout() == src.GetStandardAXBLayout() && src.GetDims()[1] % simdW != 0) {
        return false;
    }
    return true;
}

template <typename T, typename P>
constexpr bool EveryoneIs(T val, P item)
{
    return val == item;
}

template <typename T, typename P, typename... Args>
constexpr bool EveryoneIs(T val, P item, Args... itemOthers)
{
    return val == item && EveryoneIs(val, itemOthers...);
}

template <typename T>
constexpr const T &Max(const T &a, const T &b)
{
    return a > b ? a : b;
}

template <typename T>
constexpr const T &Min(const T &a, const T &b)
{
    return a < b ? a : b;
}

template <typename T>
void Swap(T &t1, T &t2)
{
    T tmp(t1);
    t1 = t2;
    t2 = tmp;
}

template <typename T>
struct RemoveReference {
    using type = T;
};

template <typename T>
struct RemoveReference<T &> {
    using type = T;
};

template <typename T>
struct RemoveReference<T &&> {
    using type = T;
};

template <typename T>
inline T &&Forward(typename RemoveReference<T>::type &&t)
{
    return static_cast<T &&>(t);
}

template <typename T>
inline T &&Forward(typename RemoveReference<T>::type &t)
{
    return static_cast<T &&>(t);
}

template <typename T>
constexpr T Pick(SizeType i, const T &x0)
{
    static_cast<void>(i);
    return x0;
}

template <typename T, typename... Args>
constexpr T Pick(SizeType i, const T &x0, Args &&...args)
{
    return i == 0 ? x0 : Pick(i - 1, Forward<Args>(args)...);
}

template <typename T, typename U, typename V>
inline typename RemoveReference<U>::type ThisBlockSize(
    const T offset, const U max, const V block_size)
{
    assert(offset < max);
    const T block_boundary = offset + block_size;
    if (block_boundary > max) {
        return max - offset;
    } else {
        return block_size;
    }
}

template <typename T>
inline T NdIteratorInit(T start)
{
    return start;
}

template <typename T, typename U, typename W, typename... Args>
inline T NdIteratorInit(T start, U &x1, const W &x2, Args &&...tuple)
{
    start = NdIteratorInit(start, Forward<Args>(tuple)...);
    x1 = start % x2;
    return start / x2;
}

inline bool NdIteratorStep()
{
    return true;
}

template <typename U, typename W, typename... Args>
inline bool NdIteratorStep(U &x1, const W &x2, Args &&...tuple)
{
    if (NdIteratorStep(std::forward<Args>(tuple)...)) {
        if (++x1 - x2 == 0) {
            x1 = 0;
            return true;
        }
    }
    return false;
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
    if (NdIteratorJump(cur, end, Forward<Args>(tuple)...)) {
        if (++x1 - x2 == 0) {
            x1 = 0;
            return true;
        }
    }
    return false;
}

template <typename T, typename U>
inline T DivUp(const T a, const U b)
{
    assert(b);
    return static_cast<T>((a + b - 1) / b);
}

template <typename T, typename U>
inline T RndUp(const T a, const U b)
{
    return static_cast<T>(DivUp(a, b) * b);
}

template <typename T, typename P>
constexpr bool OneOf(T val, P item)
{
    return val == item;
}

template <typename T, typename P, typename... Args>
constexpr bool OneOf(T val, P item, Args... itemOthers)
{
    return val == item || OneOf(val, itemOthers...);
}

template <typename T, typename U>
inline void Balance211(T n, U team, U tid, T &nStart, T &nEnd)
{
    T nMin = 1;
    T &nMy = nEnd;
    if (team <= 1 || n == 0) {
        nStart = 0;
        nMy = n;
    } else if (nMin == 1) {
        T n1 = DivUp(n, static_cast<T>(team));
        T n2 = n1 - 1;
        T T1 = n - n2 * static_cast<T>(team);
        nMy = static_cast<T>(tid) < T1 ? n1 : n2;
        nStart = static_cast<T>(tid) <= T1 ? static_cast<T>(tid) * n1 : T1 * n1 + (static_cast<T>(tid - T1)) * n2;
    }
    nEnd += nStart;
}

inline int AdjustNumThreads(int nthr, int workAmount)
{
    if (nthr == 0) {
        nthr = KuDNN::Threading::GetMaxNumThreads();
    }
    return std::min(nthr, workAmount);
}

static inline void Parallel(int nthr, const std::function<void(int, int)> &f)
{
    nthr = AdjustNumThreads(nthr, MAX_THREAD_NUM);
    if (nthr == 1) {
        f(0, 1);
        return;
    }
#pragma omp parallel num_threads(nthr)
    {
        int lNThr = omp_get_num_threads();
        int lIThr = omp_get_thread_num();
        assert(lNThr == nthr);
        f(lIThr, lNThr);
    }
}

// calculates filter size taking into account dilation
inline int CalculateExtendedFilterSize(int filterSize, int dilation)
{
    return (filterSize - 1) * (dilation + 1) + 1;
}

inline int CalculateEndPadding(int startPadding, int dstSize, int srcSize,
    int spatialStride, int dilatedFilterSize)
{
    return (dstSize - 1) * spatialStride + dilatedFilterSize
            - (srcSize + startPadding);
}

inline void PickLoopOrder(KuDNN::KudnnConvJitConfT &jcp)
{
    auto w = (jcp.propKind == KuDNN::Propagation::BACKWARD_DATA) ? jcp.iw : jcp.ow;
    auto h = (jcp.propKind == KuDNN::Propagation::BACKWARD_DATA) ? jcp.ih : jcp.oh;

    jcp.loopOrder = (w <= SMALL_SPATIAL && h <= SMALL_SPATIAL)
                     ? KuDNN::KudnnConvLoopOrderT::KUDNN_LOOP_CWGN
                     : KuDNN::KudnnConvLoopOrderT::KUDNN_LOOP_GNCW;
}
}

namespace Detail {
class ConvolutionJITFWDImpl final {
public:
    explicit ConvolutionJITFWDImpl(int isa, const TensorInfo &src, const TensorInfo &weights,
        const TensorInfo &dst, const TensorInfo &bias, KuDNN::Shape strides, KuDNN::Shape paddingL,
        KuDNN::Shape dilates, KuDNN::Propagation prop, int nthreads, bool isDataLayoutNxc) noexcept(false)
    {
        Service::ThrowOnStatus(ConvolutionJITFWD::ValidateInput(src, weights, dst, bias,
            strides, dilates, paddingL, isa), "ConvolutionJIT layer");
        InitConf(isa, src, weights, dst, bias, strides, paddingL, dilates,
            prop, nthreads, isDataLayoutNxc);
    }

    explicit ConvolutionJITFWDImpl(const ConvolutionJITFWDImpl &other) noexcept(false)
        : jcp_(other.jcp_) {}

    explicit ConvolutionJITFWDImpl(ConvolutionJITFWDImpl &&) = delete;

    ConvolutionJITFWDImpl& operator=(const ConvolutionJITFWDImpl &other) = delete;

    ConvolutionJITFWDImpl &operator=(ConvolutionJITFWDImpl &&) = delete;

    void ExecuteForward(const void *src, const void *wei,
        void *dst, const void *bias) noexcept
    {
        if (jcp_.srcDt == Element::TypeT::F32 && jcp_.weiDt == Element::TypeT::F32
            && jcp_.dstDt == Element::TypeT::F32) {
            if (jcp_.ndims == DIM_3) {
                ExecuteForward1D<float>(static_cast<const float *>(src), static_cast<const float *>(wei),
                    static_cast<float *>(dst), static_cast<const float *>(bias));
            } else if (jcp_.ndims == DIM_4) {
                ExecuteForward2D<float>(static_cast<const float *>(src), static_cast<const float *>(wei),
                    static_cast<float *>(dst), static_cast<const float *>(bias));
            } else if (jcp_.ndims == DIM_5) {
                ExecuteForward3D<float>(static_cast<const float *>(src), static_cast<const float *>(wei),
                    static_cast<float *>(dst), static_cast<const float *>(bias));
            }
        } else if (jcp_.srcDt == Element::TypeT::F16 && jcp_.weiDt == Element::TypeT::F16
            && jcp_.dstDt == Element::TypeT::F16) {
            if (jcp_.ndims == DIM_3) {
                ExecuteForward1D<__fp16>(static_cast<const __fp16 *>(src), static_cast<const __fp16 *>(wei),
                    static_cast<__fp16 *>(dst), static_cast<const __fp16 *>(bias));
            } else if (jcp_.ndims == DIM_4) {
                ExecuteForward2D<__fp16>(static_cast<const __fp16 *>(src), static_cast<const __fp16 *>(wei),
                    static_cast<__fp16 *>(dst), static_cast<const __fp16 *>(bias));
            } else if (jcp_.ndims == DIM_5) {
                ExecuteForward3D<__fp16>(static_cast<const __fp16 *>(src), static_cast<const __fp16 *>(wei),
                    static_cast<__fp16 *>(dst), static_cast<const __fp16 *>(bias));
            }
        } else if (jcp_.srcDt == Element::TypeT::BF16 && jcp_.weiDt == Element::TypeT::BF16
            && jcp_.dstDt == Element::TypeT::BF16) {
            if (jcp_.ndims == DIM_3) {
                ExecuteForward1D<__bf16>(static_cast<const __bf16 *>(src), static_cast<const __bf16 *>(wei),
                    static_cast<__bf16 *>(dst), static_cast<const __bf16 *>(bias));
            } else if (jcp_.ndims == DIM_4) {
                ExecuteForward2D<__bf16>(static_cast<const __bf16 *>(src), static_cast<const __bf16 *>(wei),
                    static_cast<__bf16 *>(dst), static_cast<const __bf16 *>(bias));
            } else if (jcp_.ndims == DIM_5) {
                ExecuteForward3D<__bf16>(static_cast<const __bf16 *>(src), static_cast<const __bf16 *>(wei),
                    static_cast<__bf16 *>(dst), static_cast<const __bf16 *>(bias));
            }
        }
    }

#ifdef KUDNNL_920Pro
    void Img2col(const float *src, float *dst);

    void Img2colS(const float *src, float *dst);

    void Img2colSS(const float *src, float *dst);

    void DataCopy(float *dst, const float *src, IntType len);

    void DataCopyWithStride(float *dst, const float *src, IntType len, IntType stride);

    void ExecuteForwardMatrixComputation(const float *src, const float *wei,
        float *dst, const float *bias) noexcept;

    bool Accelerate2dGemm();

    void GemmAccelarate(const float *src, const float *wei,
        float *dst, const float *bias) noexcept;

    bool CanUseDirectMatrixComputation();

    void DirectMatrixComputation(const float *src, const float *wei,
        float *dst, const float *bias) noexcept;

    void DirectMatrixComputationWithStride(const float *src, const float *wei,
        float *dst, const float *bias) noexcept;
#endif

    template<typename DataType>
    void ExecuteForward1D(const DataType *src, const DataType *wei,
        DataType *dst, const DataType *bias) noexcept;

    template<typename DataType>
    void ExecuteForward2D(const DataType *src, const DataType *wei,
        DataType *dst, const DataType *bias) noexcept;

    template<typename DataType>
    void ExecuteForward3D(const DataType *src, const DataType *wei,
        DataType *dst, const DataType *bias) noexcept;

    void ExecuteForwardDW(const float *src, const float *wei,
        float *dst, const float *bias) noexcept;

    bool CreateKernel(const Element::TypeT& dtp)
    {
#ifdef KUDNNL_920Pro
        if (jcp_.isDepthwise) {
            dw_kernel_.reset(new KudnnJitDwConvFwdKernel<ARMCG::CPUISAT::VectorComputation_512>(jcp_));
            kernel_f32_.reset(nullptr);
            kernel_f16_.reset(nullptr);
            kernel_bf16_.reset(nullptr);
        } else {
            dw_kernel_.reset(nullptr);
            if (dtp == Element::TypeT::F32) {
                kernel_f32_.reset(new KudnnJitConvFwdKernel<ARMCG::CPUISAT::VectorComputation_512, Element::TypeT::F32>(jcp_));
            } else if (dtp == Element::TypeT::F16) {
                kernel_f16_.reset(new KudnnJitConvFwdKernel<ARMCG::CPUISAT::VectorComputation_512, Element::TypeT::F16>(jcp_));
            } else if (dtp == Element::TypeT::BF16) {
                kernel_bf16_.reset(new KudnnJitConvFwdKernel<ARMCG::CPUISAT::VectorComputation_512, Element::TypeT::BF16>(jcp_));
            }
        }
        
        if (UseMatrixComputation()) {
            MatrixComputationKernel_.reset(new JitConvMatrixComputationKernel(jcp_));
            return MatrixComputationKernel_->CreateKernel();
        }
#else
        if (jcp_.isDepthwise) {
            dw_kernel_.reset(new KudnnJitDwConvFwdKernel<ARMCG::CPUISAT::VectorComputation_256>(jcp_));
            kernel_f32_.reset(nullptr);
            kernel_f16_.reset(nullptr);
            kernel_bf16_.reset(nullptr);
        } else {
            dw_kernel_.reset(nullptr);
            if (dtp == Element::TypeT::F32) {
                kernel_f32_.reset(new KudnnJitConvFwdKernel<ARMCG::CPUISAT::VectorComputation_256, Element::TypeT::F32>(jcp_));
            } else if (dtp == Element::TypeT::F16) {
                kernel_f16_.reset(new KudnnJitConvFwdKernel<ARMCG::CPUISAT::VectorComputation_256, Element::TypeT::F16>(jcp_));
            } else if (dtp == Element::TypeT::BF16) {
                kernel_bf16_.reset(new KudnnJitConvFwdKernel<ARMCG::CPUISAT::VectorComputation_256, Element::TypeT::BF16>(jcp_));
            }
        }
#endif
        if (dtp == Element::TypeT::F32) {
            return jcp_.isDepthwise ? dw_kernel_->CreateKernel() : kernel_f32_->CreateKernel();
        } else if (dtp == Element::TypeT::F16) {
            return jcp_.isDepthwise ? dw_kernel_->CreateKernel() : kernel_f16_->CreateKernel();
        } else {
            return jcp_.isDepthwise ? dw_kernel_->CreateKernel() : kernel_bf16_->CreateKernel();
        }
    }

    const uint8_t* GetKernel(const Element::TypeT& dtp)
    {
#ifdef KUDNNL_920Pro
        if (dtp == Element::TypeT::F32) {
            return jcp_.isDepthwise ? dw_kernel_->JitKer()
                                : (UseMatrixComputation() ? MatrixComputationKernel_->JitKer() : kernel_f32_->JitKer());
        } else if (dtp == Element::TypeT::F16) {
            return jcp_.isDepthwise ? dw_kernel_->JitKer()
                                : (UseMatrixComputation() ? MatrixComputationKernel_->JitKer() : kernel_f16_->JitKer());
        } else if (dtp == Element::TypeT::BF16) {
            return jcp_.isDepthwise ? dw_kernel_->JitKer()
                                : (UseMatrixComputation() ? MatrixComputationKernel_->JitKer() : kernel_bf16_->JitKer());
        }
#else
        if (dtp == Element::TypeT::F32) {
            return jcp_.isDepthwise ? dw_kernel_->JitKer() : kernel_f32_->JitKer();
        } else if (dtp == Element::TypeT::F16) {
            return jcp_.isDepthwise ? dw_kernel_->JitKer() : kernel_f16_->JitKer();
        } else if (dtp == Element::TypeT::BF16) {
            return jcp_.isDepthwise ? dw_kernel_->JitKer() : kernel_bf16_->JitKer();
        }
#endif
        return nullptr;
    }

#ifdef KUDNNL_920Pro
    bool UseMatrixComputation()
    {
        return jcp_.srcDt == Element::TypeT::F32 && jcp_.weiDt == Element::TypeT::F32
            && jcp_.dstDt == Element::TypeT::F32
            && jcp_.ngroups == 1 && jcp_.ndims == KERNEL_DIMS_4
            && jcp_.srcTag == KuDNN::Layout::ABCD && jcp_.dstTag == KuDNN::Layout::ABCD
            && jcp_.ocWithoutPadding % NUM_16 == 0
            && jcp_.dilateH == 0 && jcp_.dilateW == 0;
    }
#endif
    Element::TypeT GetTensorDtp(const KudnnConvJitConfT& jcp)
    {
        Element::TypeT dtp = jcp.srcDt == Element::TypeT::F32 && jcp.weiDt == Element::TypeT::F32
            && jcp.dstDt == Element::TypeT::F32 ? Element::TypeT::F32 : (jcp.srcDt == Element::TypeT::F16 &&
            jcp.weiDt == Element::TypeT::F16 && jcp.dstDt == Element::TypeT::F16 ?
            Element::TypeT::F16 : Element::TypeT::BF16);
        return dtp;
    }

    Status InitConf(int kCpuIsa,
        const TensorInfo &src, const TensorInfo &weights, const TensorInfo &dst,
        const TensorInfo &bias, KuDNN::Shape strides, KuDNN::Shape paddingL,
        KuDNN::Shape dilates, KuDNN::Propagation prop, int nthreads, bool isDataLayoutNxc) noexcept
    {
        using namespace ARMCG;
        Service::Unused(bias);
        Service::Unused(kCpuIsa);
        const IntType regs = REG_INDEX28;
        jcp_.withBias = bias.GetDims()[0] != 0;
        const bool withGroups = weights.GetNumDims() == src.GetNumDims() + 1;
        jcp_.isWithGroups = withGroups;
        int ndims = src.GetNumDims();
        jcp_.nThr = jcp_.alignedThreads = nthreads;
        jcp_.ndims = ndims;
        jcp_.propKind = prop;
        jcp_.ngroups = withGroups ? weights.GetDims()[0] : 1;
        const bool isDepthwise = CheckDwConv(src, weights, dst, kCpuIsa);
        jcp_.isDepthwise = isDepthwise;
        jcp_.mb = src.GetDims()[0];
        jcp_.oc = isDepthwise ? dst.GetDims()[1] : dst.GetDims()[1] / jcp_.ngroups;
        jcp_.ocWithoutPadding = jcp_.oc;
        jcp_.ic = isDepthwise ? src.GetDims()[1] : src.GetDims()[1] / jcp_.ngroups;
        jcp_.icWithoutPadding = jcp_.ic;
        jcp_.id = (ndims == KERNEL_DIMS_5) ? src.GetDims()[KERNEL_DIMS_2] : 1;
        jcp_.ih = (ndims == KERNEL_DIMS_3) ? 1 : src.GetDims()[ndims - KERNEL_DIMS_2];
        jcp_.iw = src.GetDims()[ndims - 1];
        jcp_.od = (ndims == KERNEL_DIMS_5) ? dst.GetDims()[KERNEL_DIMS_2] : 1;
        jcp_.oh = (ndims == KERNEL_DIMS_3) ? 1 : dst.GetDims()[ndims - KERNEL_DIMS_2];
        jcp_.ow = dst.GetDims()[ndims - 1];
        jcp_.kd = (ndims == KERNEL_DIMS_5) ? weights.GetDims()[withGroups + KERNEL_DIMS_2] : 1;
        jcp_.kh = (ndims == KERNEL_DIMS_3) ? 1 : weights.GetDims()[withGroups + ndims - KERNEL_DIMS_2];
        jcp_.kw = weights.GetDims()[withGroups + ndims - 1];

        jcp_.fPad = (ndims == KERNEL_DIMS_5) ? paddingL[0] : 0;
        jcp_.tPad = (ndims == KERNEL_DIMS_3) ? 0 : paddingL[ndims - KERNEL_DIMS_4];
        jcp_.lPad = paddingL[ndims - KERNEL_DIMS_3];
        jcp_.strideD = (ndims == KERNEL_DIMS_5) ? strides[0] : 1;
        jcp_.strideH = (ndims == KERNEL_DIMS_3) ? 1 : strides[ndims - KERNEL_DIMS_4];
        jcp_.strideW = strides[ndims - KERNEL_DIMS_3];

        jcp_.dilateD = (ndims == KERNEL_DIMS_5) ? dilates[0] : 0;
        jcp_.dilateH = (ndims == KERNEL_DIMS_3) ? 0 : dilates[ndims - KERNEL_DIMS_4];
        jcp_.dilateW = dilates[ndims - KERNEL_DIMS_3];

        int extKw = CalculateExtendedFilterSize(jcp_.kw, jcp_.dilateW);
        int extKh = CalculateExtendedFilterSize(jcp_.kh, jcp_.dilateH);
        int extKd = CalculateExtendedFilterSize(jcp_.kd, jcp_.dilateD);
        jcp_.rPad = CalculateEndPadding(
            jcp_.lPad, jcp_.ow, jcp_.iw, jcp_.strideW, extKw);
        jcp_.bPad = CalculateEndPadding(
            jcp_.tPad, jcp_.oh, jcp_.ih, jcp_.strideH, extKh);
        jcp_.backPad = CalculateEndPadding(
            jcp_.fPad, jcp_.od, jcp_.id, jcp_.strideD, extKd);

        // post-op not supported
        jcp_.withSum = 0;
        jcp_.srcTag = src.GetLayout();
        jcp_.dstTag = dst.GetLayout();
        jcp_.srcDt = src.GetType();
        jcp_.weiDt = weights.GetType();
        jcp_.dstDt = dst.GetType();
        jcp_.biaDt = bias.GetType();

#ifdef KUDNNL_920Pro
        if (UseMatrixComputation()) {
            IntType sizeH = jcp_.ih + jcp_.tPad + jcp_.bPad;
            IntType sizeW = jcp_.iw + jcp_.lPad + jcp_.rPad;
            padSrc.resize(jcp_.mb * sizeH * sizeW * jcp_.ic, 0.0f);
            if (CanUseDirectMatrixComputation()) {
                jcp_.ver = KuDNN::KudnnConvVersionT::KUDNN_VER_MatrixComputation_DIRECT;
            } else {
                colTmp.resize(jcp_.mb * jcp_.od * jcp_.oh * jcp_.ow * jcp_.kd * jcp_.kh * jcp_.kw * jcp_.ic, 0.0f);
                jcp_.ver = KuDNN::KudnnConvVersionT::KUDNN_VER_MatrixComputation_GEMM;
            }
            Element::TypeT dtp = GetTensorDtp(jcp_);
            CreateKernel(dtp);
            return Status::SUCCESS;
        }
#endif

#ifdef KUDNNL_920Pro
        int tIcLen = IC_LEN16;
        int fullSimdW = CpuIsaTraits<CPUISAT::VectorComputation_512>::VLEN / sizeof(float);
        if (jcp_.srcDt == Element::TypeT::F16) {
            fullSimdW = CpuIsaTraits<CPUISAT::VectorComputation_512>::VLEN / sizeof(__fp16);
        }
#else
        int tIcLen = IC_LEN8;
        int fullSimdW = CpuIsaTraits<CPUISAT::VectorComputation_256>::VLEN / sizeof(float);
        if (jcp_.srcDt == Element::TypeT::F16) {
            tIcLen = IC_LEN16;
            fullSimdW = CpuIsaTraits<CPUISAT::VectorComputation_256>::VLEN / sizeof(__fp16);
        }
#endif
        /* 1st convolution check */
        jcp_.is1stConv = (jcp_.ic < tIcLen) && (jcp_.ngroups == 1);
        jcp_.simdW = fullSimdW;
        jcp_.ocBlock = jcp_.simdW;
        jcp_.icBlock = jcp_.is1stConv ? jcp_.ic : jcp_.simdW;
        if (jcp_.srcDt == Element::TypeT::F16 && kCpuIsa == static_cast<IntType>(ARMCG::CPUISAT::VectorComputation_512)) {
            jcp_.icBlock = jcp_.is1stConv ? jcp_.ic : IC_LEN16;
        }
        /* Channel padding */
        jcp_.oc = RndUp(jcp_.oc, jcp_.ocBlock);
        jcp_.ic = RndUp(jcp_.ic, jcp_.icBlock);
        jcp_.nbIc = DivUp(jcp_.ic, jcp_.icBlock);
        jcp_.nbOc = DivUp(jcp_.oc, jcp_.ocBlock);
        jcp_.nbIcBlocking = jcp_.nbOcBlocking = 1;
        jcp_.icTail = 0;
        jcp_.ocTail = 0;

        jcp_.ver = KuDNN::KudnnConvVersionT::KUDNN_VER_FMA;
        jcp_.typeSizeIn = sizeof(float);
        jcp_.typeSizeOut = sizeof(float);
        jcp_.typeSizeBia = sizeof(float);
        if (jcp_.srcDt == Element::TypeT::F16) {
            jcp_.typeSizeIn = sizeof(__fp16);
            jcp_.typeSizeOut = sizeof(__fp16);
        } else if (jcp_.srcDt == Element::TypeT::BF16) {
            jcp_.typeSizeIn = sizeof(__bf16);
            jcp_.typeSizeOut = sizeof(__bf16);
        }

        jcp_.urW = Min(jcp_.ow, regs);
        /* Grouped channel offset to support 'non-blocked data' format for
        * convolution sizes with '(input_channel / ngroups) < simd' */
        jcp_.nonblkGroupOff = 1;

        if (jcp_.ver == KuDNN::KudnnConvVersionT::KUDNN_VER_FMA) {
            IntType maxNbOc = KERNEL_C_LEN2;
            IntType tRegNum = REG_INDEX25;
            if (((jcp_.kw == 1 && jcp_.oc > KERNEL_C_LEN16) ||
                (jcp_.kw > 1 && jcp_.lPad > 0 && jcp_.strideW > 1)) &&
                (jcp_.nbOc % KERNEL_C_LEN4 == 0)) {
                maxNbOc = KERNEL_C_LEN4;
                tRegNum = REG_INDEX31;
            } else if ((jcp_.nbOc % KERNEL_C_LEN2 != 0) && (jcp_.nbOc > maxNbOc)) {
                maxNbOc = jcp_.nbOc > KERNEL_C_LEN8 ? KERNEL_C_LEN4 : jcp_.nbOc;
                tRegNum = REG_INDEX31;
            }
            if (jcp_.id > 1) {
                maxNbOc = KERNEL_C_LEN2;
                tRegNum = REG_INDEX31;
            }
            jcp_.kernelKind = KuDNN::KudnnConvKernelKindT::KUDNN_EXPL_BCAST;
            jcp_.nbIcBlocking = 1;
            int bestNbOcBlocking = 1;
            for (int i = Min(jcp_.nbOc, maxNbOc); i > 0; i--) {
                if (jcp_.nbOc % i == 0) {
                    bestNbOcBlocking = i;
                    break;
                }
            }
            jcp_.nbOcBlocking = bestNbOcBlocking;
            if (jcp_.srcDt == Element::TypeT::BF16) {
                jcp_.nbOcBlocking = 1;
                tRegNum = REG_INDEX31;
            }
            jcp_.urW = Min(jcp_.ow, tRegNum / (jcp_.nbOcBlocking + 1));
            if (jcp_.lPad > jcp_.urW) {
                jcp_.nbOcBlocking = 1;
                jcp_.urW = Min(jcp_.ow, tRegNum / (jcp_.nbOcBlocking + 1));
            }
            // KUDNN_CONT_BCAST: urW <= 8 && nbOcBlocking > 1
            // KUDNN_EXPL_BCAST: urW > 8 && nbOcBlocking == 1
            // perhaps some case get better performance: urW > 8 && nbOcBlocking == 2
            if (!jcp_.is1stConv && jcp_.urW <= KERNEL_UR_W_LEN8) {
                jcp_.kernelKind = KuDNN::KudnnConvKernelKindT::KUDNN_CONT_BCAST;
            }
        }
        jcp_.urWTail = jcp_.ow % jcp_.urW;
        PickLoopOrder(jcp_);
        jcp_.nbIcL2 = jcp_.nbIc;

        if (isDepthwise) {
            jcp_.loopOrder = KuDNN::KudnnConvLoopOrderT::KUDNN_LOOP_NGCW;

            ARMCG::CPUISAT isa = static_cast<ARMCG::CPUISAT>(kCpuIsa);
            jcp_.urW = isa == ARMCG::CPUISAT::VectorComputation_512 ? UR_W_6 :
                isa == ARMCG::CPUISAT::VectorComputation_256 ? UR_W_4 : UR_W_3;
            jcp_.urW = Min(jcp_.urW, jcp_.ow);
            jcp_.chBlock = fullSimdW;
            jcp_.nbCh = DivUp(dst.GetDims()[1], jcp_.chBlock);
            jcp_.nbChBlocking = isa == ARMCG::CPUISAT::VectorComputation_512 ? NB_CH_BLKING_4 :
                isa == ARMCG::CPUISAT::VectorComputation_256 ? NB_CH_BLKING_3 : NB_CH_BLKING_2;
            if (jcp_.nbCh < jcp_.nbChBlocking) {
                jcp_.nbChBlocking = jcp_.nbCh;
            }

            if (isDataLayoutNxc) {
                jcp_.loopOrder = KuDNN::KudnnConvLoopOrderT::KUDNN_LOOP_NHWCG;
                bool cacheAliasing =
                    (jcp_.ngroups * jcp_.iw * jcp_.typeSizeIn) % CACHE_ALIAS_1024 == 0;
                if (cacheAliasing) {
                    const IntType limit = jcp_.ow > LIMIT_7 ? LIMIT_7 : LIMIT_4;
                    jcp_.urW = Min(jcp_.urW, limit);
                }
            }

            jcp_.urWTail = jcp_.ow % jcp_.urW;
            jcp_.ngroups = RndUp(jcp_.ngroups, fullSimdW);
        }

        // currently kernel handle ow as a whole
        jcp_.owBlock = jcp_.ow;
        jcp_.nbOw = 1;
        IntType L2_size = L2_CORE_CACHE_SIZE / sizeof(float);
        if (jcp_.srcDt == KuDNN::Element::TypeT::F16) {
            L2_size = L2_CORE_CACHE_SIZE / sizeof(__fp16);
        }

        // Source and output data needs to fit in L2,
        // leaving some space for weights and prefetching.
        IntType hL2Zero = 0;
        IntType hL2 = int(((0.6f * L2_size) / jcp_.simdW
                           - Min(hL2Zero, jcp_.kh - jcp_.strideH) * jcp_.iw)
            / (jcp_.strideH * jcp_.iw + jcp_.ow));
        IntType tL2 = 1;
        jcp_.hBlocking = Max(tL2, Min(jcp_.oh, hL2));

        if (isDataLayoutNxc) {
            const int nbIcThesholdL2 = KERNEL_C_LEN32;
            if ((jcp_.nbIc > nbIcThesholdL2) && (jcp_.nbIc < KERNEL_C_LEN2 * nbIcThesholdL2)) {
                jcp_.nbIcL2 = jcp_.nbIc;
            } else {
                jcp_.nbIcL2 = jcp_.nbIc < nbIcThesholdL2 ? jcp_.nbIc : nbIcThesholdL2;
            }
        }
        Element::TypeT dtp = GetTensorDtp(jcp_);
        CreateKernel(dtp);
        return Status::SUCCESS;
    }

    KudnnConvJitConfT jcp_ = KudnnConvJitConfT();
#ifdef KUDNNL_920Pro
    std::unique_ptr<KudnnJitDwConvFwdKernel<ARMCG::CPUISAT::VectorComputation_512>> dw_kernel_;
    std::unique_ptr<KudnnJitConvFwdKernel<ARMCG::CPUISAT::VectorComputation_512, Element::TypeT::F32>> kernel_f32_;
    std::unique_ptr<KudnnJitConvFwdKernel<ARMCG::CPUISAT::VectorComputation_512, Element::TypeT::F16>> kernel_f16_;
    std::unique_ptr<KudnnJitConvFwdKernel<ARMCG::CPUISAT::VectorComputation_512, Element::TypeT::BF16>> kernel_bf16_;
    std::unique_ptr<JitConvMatrixComputationKernel> MatrixComputationKernel_;
    std::vector<float, Service::AlignedAllocator<float>> padSrc;
    std::vector<float, Service::AlignedAllocator<float>> colTmp;
#else
    std::unique_ptr<KudnnJitDwConvFwdKernel<ARMCG::CPUISAT::VectorComputation_256>> dw_kernel_;
    std::unique_ptr<KudnnJitConvFwdKernel<ARMCG::CPUISAT::VectorComputation_256, Element::TypeT::F32>> kernel_f32_;
    std::unique_ptr<KudnnJitConvFwdKernel<ARMCG::CPUISAT::VectorComputation_256, Element::TypeT::F16>> kernel_f16_;
    std::unique_ptr<KudnnJitConvFwdKernel<ARMCG::CPUISAT::VectorComputation_256, Element::TypeT::BF16>> kernel_bf16_;
#endif
};

void ConvolutionJITFWDImpl::ExecuteForwardDW(const float *src, const float *wei,
    float *dst, const float *bias) noexcept
{
    KudnnConvJitConfT& jcp = jcp_;
    Element::TypeT dtp = GetTensorDtp(jcp);
    KudnnConvJitKerT jitKer = (decltype(jitKer))(GetKernel(dtp));
    const IntType dilateH = jcp.dilateH + 1;
    const IntType strideH = jcp.strideH;
    const IntType chStep = jcp.nbChBlocking;
    const IntType iw = 0;
    const IntType kw = 0;
    const IntType ow = 0;
    const IntType chbWork = DivUp(jcp.nbCh, chStep);
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
        if (jcp.loopOrder == KudnnConvLoopOrderT::KUDNN_LOOP_NGCW) {
            NdIteratorInit(start, n, jcp.mb, chb, chbWork, oh, jcp.oh);
        } else if (jcp.loopOrder == KudnnConvLoopOrderT::KUDNN_LOOP_NHWCG) {
            NdIteratorInit(start, n, jcp.mb, oh, jcp.oh, chb, chbWork);
        } else {
            assert(!"unsupported loop order");
        }

        auto iWork = start;
        while (iWork < end) {
            IntType ch = chb * chStep;
            const IntType itOverflow =
                std::max(jcp.tPad - oh * strideH, static_cast<IntType>(0));
            const IntType ibOverflow =
                std::max(oh * strideH + (jcp.kh - 1) * dilateH - jcp.tPad + 1, jcp.ih) - jcp.ih;
            const IntType ih =
                std::max((oh * strideH - jcp.tPad + DivUp(itOverflow, dilateH) * dilateH),
                static_cast<IntType>(0));
            const IntType kh = DivUp(itOverflow, dilateH);
            const IntType khPadding =
                jcp.kh - DivUp(itOverflow, dilateH) - DivUp(ibOverflow, dilateH);
            const auto icOffIdx = isSrcLayoutNXC ? ch * jcp.chBlock : ch;
            const auto ocOffIdx = isDstLayoutNXC ? ch * jcp.chBlock : ch;

            const IntType srcBlkOff = n * jcp.ngroups * jcp.ih * jcp.iw
                + icOffIdx * jcp.chBlock * jcp.ih * jcp.iw
                + jcp.chBlock * ih * jcp.iw
                + jcp.chBlock * iw;
            const IntType srcNXCOff = n * jcp.ngroups * jcp.ih * jcp.iw
                + jcp.ic * ih * jcp.iw
                + jcp.ic * iw
                + icOffIdx;
            const IntType weiBlkOff = ch * jcp.chBlock * jcp.kh * jcp.kw
                + jcp.chBlock * kh * jcp.kw
                + jcp.chBlock * kw;
            const IntType dstBlkOff = n * jcp.ngroups * jcp.oh * jcp.ow
                + ocOffIdx * jcp.chBlock * jcp.oh * jcp.ow
                + jcp.chBlock * oh * jcp.ow
                + jcp.chBlock * ow;
            const IntType dstNXCOff = n * jcp.ngroups * jcp.oh * jcp.ow
                + jcp.oc * oh * jcp.ow
                + jcp.oc * ow
                + ocOffIdx;

            auto paraConv = KudnnConvJitCallS();
            paraConv.src = jcp.isFusedConv ? src : src + (isSrcLayoutNXC ? srcNXCOff : srcBlkOff);
            paraConv.filt = wei + weiBlkOff;
            paraConv.dst = dst + (isDstLayoutNXC ? dstNXCOff : dstBlkOff);
            paraConv.bias = bias ? bias + (ch * jcp.chBlock) : nullptr;
            paraConv.khPadding = std::max(khPadding, static_cast<IntType>(0));
            if (isSrcLayoutNXC) {
                IntType workRemain = end - iWork;
                paraConv.chBlocks = ch + workRemain * chStep >= jcp.nbCh ?
                    jcp.nbCh - ch : workRemain * chStep;
                assert(jcp.loopOrder == KudnnConvLoopOrderT::KUDNN_LOOP_NHWCG);
            } else {
                paraConv.chBlocks = ThisBlockSize(ch, jcp.nbCh, chStep);
                assert(jcp.loopOrder != KudnnConvLoopOrderT::KUDNN_LOOP_NHWCG);
            }

            jitKer(&paraConv);

            if (jcp.loopOrder == KudnnConvLoopOrderT::KUDNN_LOOP_NGCW) {
                iWork++;
                NdIteratorStep(n, jcp.mb, chb, chbWork, oh, jcp.oh);
            } else if (jcp.loopOrder == KudnnConvLoopOrderT::KUDNN_LOOP_NHWCG) {
                NdIteratorJump(iWork, end, n, jcp.mb, oh, jcp.oh, chb, chbWork);
            } else {
                assert(!"unsupported loop order");
            }
        }
    });
}

#ifdef KUDNNL_920Pro
void ConvolutionJITFWDImpl::Img2col(const float *src, float *dst)
{
    IntType sizeIn = jcp_.id * jcp_.ih * jcp_.iw * jcp_.ic;
    IntType sizeOut = jcp_.od * jcp_.oh * jcp_.ow * jcp_.kd * jcp_.kh * jcp_.kw * jcp_.ic;

    Shape shape(jcp_.mb, jcp_.kd, jcp_.kh, jcp_.kw, jcp_.ic);
    Threading::ParallelForND<5>::run(
        [&](int mb, int kd, int kh, int kw, int ic) {
            for (int od = 0; od < jcp_.od; ++od) {
                for (int oh = 0; oh < jcp_.oh; ++oh) {
                    for (int ow = 0; ow < jcp_.ow; ++ow) {
                        int id = jcp_.ndims == 5
                            ? (od * jcp_.strideD - jcp_.fPad + kd * (jcp_.dilateD + 1))
                            : 0;
                        int ih = jcp_.ndims >= 4
                            ? (oh * jcp_.strideH - jcp_.tPad + kh * (jcp_.dilateH + 1))
                            : 0;
                        int iw = ow * jcp_.strideW - jcp_.lPad + kw * (jcp_.dilateW + 1);
                        if ((id >= 0) && (id < jcp_.id) && (ih >= 0) && (ih < jcp_.ih)
                            && (iw >= 0) && (iw < jcp_.iw)) {
                            dst[mb * sizeOut + (((((kd * jcp_.kh + kh) * jcp_.kw + kw) * jcp_.ic + ic)
                                * jcp_.od + od) * jcp_.oh + oh) * jcp_.ow + ow] =
                                src[mb * sizeIn + ((ic * jcp_.id + id) * jcp_.ih + ih) * jcp_.iw + iw];
                        }
                    }
                }
            }
        },
        shape,
        jcp_.alignedThreads);
}

void ConvolutionJITFWDImpl::Img2colS(const float *src, float *dst)
{
    IntType sizeIn = jcp_.id * jcp_.ih * jcp_.iw * jcp_.ic;
    IntType sizeOut = jcp_.od * jcp_.oh * jcp_.ow * jcp_.kd * jcp_.kh * jcp_.kw * jcp_.ic;

    Shape shape(jcp_.mb, jcp_.kd, jcp_.kh, jcp_.kw, jcp_.ic);
    Threading::ParallelForND<5>::run(
        [&](int mb, int kd, int kh, int kw, int ic) {
            for (int od = 0; od < jcp_.od; ++od) {
                for (int oh = 0; oh < jcp_.oh; ++oh) {
                    int id = jcp_.ndims == 5 ? (od * jcp_.strideD - jcp_.fPad
                        + kd * (jcp_.dilateD + 1)) : 0;
                    int ih = jcp_.ndims >= 4 ? (oh * jcp_.strideH - jcp_.tPad
                        + kh * (jcp_.dilateH + 1)) : 0;
                    if ((id >= 0) && (id < jcp_.id) && (ih >= 0) && (ih < jcp_.ih)) {
                        int start = std::max(0, static_cast<int>(jcp_.lPad - kw));
                        int minIw = std::max(0, static_cast<int>(-jcp_.lPad + kw));
                        int maxIw = std::min(jcp_.iw, jcp_.ow - jcp_.lPad + kw);
                        float *out = dst + mb * sizeOut + (((((kd * jcp_.kh + kh) * jcp_.kw + kw)
                            * jcp_.ic + ic) * jcp_.od + od) * jcp_.oh + oh) * jcp_.ow + start;
                        const float *in = src + mb * sizeIn + ((ic * jcp_.id + id) * jcp_.ih + ih)
                            * jcp_.iw + minIw;
                        DataCopy(out, in, maxIw - minIw);
                    }
                }
            }
        },
        shape,
        jcp_.alignedThreads);
}

void ConvolutionJITFWDImpl::Img2colSS(const float *src, float *dst)
{
    IntType sizeIn = jcp_.id * jcp_.ih * jcp_.iw * jcp_.ic;
    IntType sizeOut = jcp_.od * jcp_.oh * jcp_.ow * jcp_.kd * jcp_.kh * jcp_.kw * jcp_.ic;

    Shape shape(jcp_.mb, jcp_.kd, jcp_.kh, jcp_.kw, jcp_.ic);
    Threading::ParallelForND<5>::run(
        [&](int mb, int kd, int kh, int kw, int ic) {
            for (int od = 0; od < jcp_.od; ++od) {
                for (int oh = 0; oh < jcp_.oh; ++oh) {
                    int id = jcp_.ndims == 5 ? (od * jcp_.strideD - jcp_.fPad
                        + kd * (jcp_.dilateD + 1)) : 0;
                    int ih = jcp_.ndims >= 4 ? (oh * jcp_.strideH - jcp_.tPad
                        + kh * (jcp_.dilateH + 1)) : 0;
                    if ((id >= 0) && (id < jcp_.id) && (ih >= 0) && (ih < jcp_.ih)) {
                        float *out = dst + mb * sizeOut + (((((kd * jcp_.kh + kh) * jcp_.kw + kw)
                            * jcp_.ic + ic) * jcp_.od + od) * jcp_.oh + oh) * jcp_.ow;
                        const float *in = src + mb * sizeIn + ((ic * jcp_.id + id) * jcp_.ih + ih)
                            * jcp_.iw + kw;
                        DataCopyWithStride(out, in, jcp_.ow, jcp_.strideW);
                    }
                }
            }
        }, shape, jcp_.alignedThreads);
}

bool ConvolutionJITFWDImpl::CanUseDirectMatrixComputation()
{
    if (jcp_.ngroups != 1 || jcp_.ndims != KERNEL_DIMS_4
        || jcp_.dilateH != 0 || jcp_.dilateW != 0) {
        return false;
    }
    if (jcp_.strideH == 1 && jcp_.strideW == 1 && jcp_.ow >= NUM_24) {
        return true;
    }
    if (jcp_.strideW != 1 && jcp_.kw <= NUM_3 && jcp_.ow > NUM_32
        && jcp_.lPad == 0 && jcp_.rPad == 0
        && jcp_.tPad == 0 && jcp_.bPad == 0) {
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

void ConvolutionJITFWDImpl::DirectMatrixComputationWithStride(const float *src, const float *wei,
    float *dst, const float *bias) noexcept
{
    KudnnConvJitConfT& jcp = jcp_;
    Element::TypeT dtp = GetTensorDtp(jcp);
    KudnnConvMatrixComputationJitKerT jitKer = (decltype(jitKer))(GetKernel(dtp));
    int nThr = jcp.alignedThreads;
    IntType sizeH = jcp.ih + jcp.tPad + (jcp.bPad > 0 ? jcp.bPad : 0);
    IntType sizeW = jcp.iw + jcp.lPad + (jcp.rPad > 0 ? jcp.rPad : 0);

    // pack stride src in order

    Threading::ParallelForND<3>::run(
        [&](int mb, int ic, int ih) {
            IntType off = 0;
            for (auto stride = 0; stride < jcp.strideW; stride++) {
                IntType len = (sizeW + 1 - stride) / jcp.strideW;
                DataCopyWithStride(padSrc.data() + mb * jcp.ic * sizeH * sizeW
                    + ic * sizeH * sizeW + (jcp.tPad + ih) * sizeW + jcp.lPad + off,
                    src + mb * jcp.ic * jcp.ih * jcp.iw + ic * jcp.ih * jcp.iw + ih * jcp.iw + stride,
                    len, jcp.strideW);
                    off += len;
            }
        },
        Shape(jcp.mb, jcp.ic, jcp.ih), nThr);

    Threading::ParallelForND<2>::run(
        [&](int mb, int oh) {
            for (int oc = 0; oc < jcp.oc; oc += NUM_16) {
                JitGemmCallS p = JitGemmCallS();
                p.matrixM = jcp.oh * jcp.ow;
                p.matrixN = oc + NUM_16 > jcp.oc ? jcp.oc - oc : NUM_16;
                p.matrixK = sizeH * sizeW;
                p.matrixA = padSrc.data() + mb * jcp.ic * p.matrixK + oh * jcp.strideH * sizeW;
                p.matrixB = wei + oc * jcp.kd * jcp.kh * jcp.kw * jcp.ic;
                p.matrixC = dst + mb * p.matrixM * jcp.oc + oc * p.matrixM + oh * jcp.ow;
                p.bias = bias + oc;
                if (jitKer) {
                    jitKer(&p);
                }
            }
        },
        Shape(jcp.mb, jcp.oh), nThr);
}

void ConvolutionJITFWDImpl::DirectMatrixComputation(const float *src, const float *wei,
    float *dst, const float *bias) noexcept
{
    KudnnConvJitConfT& jcp = jcp_;
    if (jcp.strideW != 1 && jcp.kw <= NUM_3) {
        DirectMatrixComputationWithStride(src, wei, dst, bias);
        return;
    }
    Element::TypeT dtp = GetTensorDtp(jcp);
    KudnnConvMatrixComputationJitKerT jitKer = (decltype(jitKer))(GetKernel(dtp));
    int nThr = jcp.alignedThreads;
    IntType sizeH = jcp.ih + jcp.tPad + jcp.bPad;
    IntType sizeW = jcp.iw + jcp.lPad + jcp.rPad;

    if (jcp.tPad > 0 || jcp.bPad > 0 || jcp.lPad > 0 || jcp.rPad > 0) {
        Threading::ParallelForND<3>::run(
            [&](int mb, int ic, int oh) {
                DataCopy(padSrc.data() + mb * jcp.ic * sizeH * sizeW
                    + ic * sizeH * sizeW + (jcp.tPad + oh) * sizeW + jcp.lPad,
                    src + mb * jcp.ic * jcp.ih * jcp.iw + ic * jcp.ih * jcp.iw + oh * jcp.iw, jcp.iw);
            },
            Shape(jcp.mb, jcp.ic, jcp.oh), nThr);
    }

    Threading::ParallelForND<2>::run(
        [&](int mb, int oh) {
            for (int oc = 0; oc < jcp.oc; oc += NUM_16) {
                JitGemmCallS p = JitGemmCallS();
                p.matrixM = jcp.oh * jcp.ow;
                p.matrixN = oc + NUM_16 > jcp.oc ? jcp.oc - oc : NUM_16;
                p.matrixK = sizeH * sizeW;
                if (jcp.tPad <= 0 && jcp.bPad <= 0 && jcp.lPad <= 0 && jcp.rPad <= 0) {
                    p.matrixA = src + mb * jcp.ic * p.matrixK + oh * jcp.iw;
                } else {
                    p.matrixA = padSrc.data() + mb * jcp.ic * p.matrixK + oh * sizeW;
                }
                p.matrixB = wei + oc * jcp.kd * jcp.kh * jcp.kw * jcp.ic;
                p.matrixC = dst + mb * p.matrixM * jcp.oc + oc * p.matrixM + oh * jcp.ow;
                p.bias = bias + oc;
                if (jitKer) {
                    jitKer(&p);
                }
            }
        },
        Shape(jcp.mb, jcp.oh), nThr);
}

bool ConvolutionJITFWDImpl::Accelerate2dGemm()
{
    return jcp_.ngroups == 1 && jcp_.ndims == KERNEL_DIMS_4 && jcp_.strideW != 1 && jcp_.lPad == 0 && jcp_.rPad == 0;
}

void ConvolutionJITFWDImpl::GemmAccelarate(const float *src, const float *wei,
    float *dst, const float *bias) noexcept
{
    KudnnConvJitConfT& jcp = jcp_;
    Element::TypeT dtp = GetTensorDtp(jcp);
    KudnnConvMatrixComputationJitKerT jitKer = (decltype(jitKer))(GetKernel(dtp));
    int nThr = jcp.alignedThreads;
    IntType sizeIn = jcp_.ih * jcp_.iw * jcp_.ic;
    IntType sizeOut = jcp.oh * jcp.ow * jcp.kh * jcp.kw * jcp.ic;

    Shape shape(jcp_.mb, jcp_.ic, jcp_.oh, jcp_.kh, jcp_.kw);
    Threading::ParallelForND<5>::run(
        [&](int mb, int ic, int oh, int kh, int kw) {
            int ih = (oh * jcp_.strideH - jcp_.tPad + kh * (jcp_.dilateH + 1));
            if ((ih >= 0) && (ih < jcp_.ih)) {
                float *out = colTmp.data() + mb * sizeOut + (((oh * jcp_.kh + kh)
                    * jcp_.kw + kw) * jcp_.ic + ic) * jcp_.ow;
                const float *in = src + mb * sizeIn + (ic * jcp_.ih + ih) * jcp_.iw + kw;
                DataCopyWithStride(out, in, jcp_.ow, jcp_.strideW);
            }
        }, shape, nThr);

    Threading::ParallelForND<2>::run(
        [&](int mb, int oh) {
            for (int oc = 0; oc < jcp.oc; oc += NUM_16) {
                JitGemmCallS p = JitGemmCallS();
                p.matrixM = jcp.ow;
                p.matrixN = oc + NUM_16 > jcp.oc ? jcp.oc - oc : NUM_16;
                p.matrixK = jcp.kh * jcp.kw * jcp.ic;
                p.matrixA = colTmp.data() + mb * sizeOut + oh * jcp.ow * jcp.kh * jcp.kw * jcp.ic;
                p.matrixB = wei + oc * jcp.kh * jcp.kw * jcp.ic;
                p.matrixC = dst + mb * jcp.oh * jcp.ow * jcp.oc + oc * jcp.oh * jcp.ow + oh * jcp.ow;
                p.bias = bias + oc;
                if (jitKer) {
                    jitKer(&p);
                }
            }
        },
        Shape(jcp.mb, jcp.oh, jcp.oc), nThr);
}

void ConvolutionJITFWDImpl::ExecuteForwardMatrixComputation(const float *src, const float *wei,
    float *dst, const float *bias) noexcept
{
    KudnnConvJitConfT& jcp = jcp_;
    if (CanUseDirectMatrixComputation()) {
        DirectMatrixComputation(src, wei, dst, bias);
        return;
    }
    if (Accelerate2dGemm()) {
        GemmAccelarate(src, wei, dst, bias);
        return;
    }
    Element::TypeT dtp = GetTensorDtp(jcp);
    KudnnConvMatrixComputationJitKerT jitKer = (decltype(jitKer))(GetKernel(dtp));
    int nThr = jcp.alignedThreads;

    IntType sizeOut = jcp.od * jcp.oh * jcp.ow * jcp.kd * jcp.kh * jcp.kw * jcp.ic;
    bool noImg2Col = jcp.kd == 1 && jcp.kh == 1 && jcp.kw == 1
        && jcp.strideD == 1 && jcp.strideH == 1 && jcp.strideW == 1
        && jcp.dilateD == 0 && jcp.dilateH == 0 && jcp.dilateW == 0;
    if (!noImg2Col) {
        if (jcp.dilateW == 0 && jcp.strideW == 1) {
            Img2colS(src, colTmp.data());
        } else if (jcp.dilateW == 0 && jcp.lPad == 0 && jcp.rPad == 0) {
            Img2colSS(src, colTmp.data());
        } else {
            Img2col(src, colTmp.data());
        }
    }
    bool isNeedPad = jcp.tPad > 0 || jcp.bPad > 0 || jcp.lPad > 0 || jcp.rPad > 0 ? true : false;
    if (isNeedPad) {
        IntType sizeH = jcp.ih + jcp.tPad + jcp.bPad;
        IntType sizeW = jcp.iw + jcp.lPad + jcp.rPad;

        Shape shape(jcp.mb, jcp.ic, jcp.oh);
        Threading::ParallelForND<3>::run(
            [&](int mb, int ic, int oh) {
                if (oh >= jcp.bPad && oh < jcp.ih + jcp.bPad) {
                    DataCopy(padSrc.data() + mb * jcp.ic * sizeH * sizeW
                        + ic * sizeH * sizeW + oh * sizeW + jcp.lPad,
                        src + mb * jcp.ic * jcp.ih * jcp.iw + ic * jcp.ih *
                        jcp.iw + (oh - jcp.bPad) * jcp.iw, jcp.iw);
                }
            }, shape, nThr);
        Threading::ParallelFor::instance().run(
            [&](int64_t mb) {
                for (int oc = 0; oc < jcp.oc; oc += NUM_16) {
                    JitGemmCallS p = JitGemmCallS();
                    p.matrixM = jcp.od * jcp.oh * jcp.ow;
                    p.matrixN = oc + NUM_16 > jcp.oc ? jcp.oc - oc : NUM_16;
                    p.matrixK = jcp.kd * jcp.kh * jcp.kw * jcp.ic;
                    if (noImg2Col) {
                        p.matrixA = isNeedPad ? padSrc.data() + mb * sizeOut : src + mb * sizeOut;
                    } else {
                        p.matrixA = colTmp.data() + mb * sizeOut;
                    }
                    p.matrixB = wei + oc * jcp.kd * jcp.kh * jcp.kw * jcp.ic;
                    p.matrixC = dst + mb * jcp.od * jcp.oh * jcp.ow * jcp.oc + oc * p.matrixM;
                    p.bias = bias + oc;
                    if (jitKer) {
                        jitKer(&p);
                    }
                }
            }, jcp.mb, nThr);
    }
}
#endif

template <typename DataType>
void ConvolutionJITFWDImpl::ExecuteForward1D(const DataType *src, const DataType *wei,
    DataType *dst, const DataType *bias) noexcept
{
#ifdef KUDNNL_920Pro
    if constexpr (std::is_same_v<DataType, float>) {
        if (UseMatrixComputation()) {
            ExecuteForwardMatrixComputation(src, wei, dst, bias);
            return;
        }
    }
#endif
    KudnnConvJitConfT& jcp = jcp_;
    Element::TypeT dtp = GetTensorDtp(jcp);
    assert(GetKernel(dtp) != nullptr);
    KudnnConvJitKerT jitKer = (decltype(jitKer))(GetKernel(dtp));
    assert(jcp.nbOc % jcp.nbOcBlocking == 0);
    IntType ocChunks = jcp.nbOc / jcp.nbOcBlocking;
    int gBlocking = 1;
    IntType nbGroups = jcp.ngroups / gBlocking;
    IntType workAmount = jcp.mb * nbGroups * ocChunks * jcp.nbOw;
    int nThr = jcp.alignedThreads;
    Parallel(nThr, [&](const int ithr, const int subNThr) {
        IntType start {0};
        IntType end {0};
        IntType startCopy;
        Balance211(workAmount, subNThr, ithr, start, end);
        startCopy = start;
        auto parConv = KuDNN::KudnnConvJitCallS();
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

            if (jcp.loopOrder == KuDNN::KudnnConvLoopOrderT::KUDNN_LOOP_CWGN) {
                IntType dummy {0};
                NdIteratorInit(start, occ, ocChunks, owb, jcp.nbOw, gg,
                    nbGroups, n, jcp.mb, dummy, 1);
            } else if (jcp.loopOrder == KuDNN::KudnnConvLoopOrderT::KUDNN_LOOP_GNCW) {
                IntType dummy {0};
                NdIteratorInit(start, gg, nbGroups, n, jcp.mb, occ,
                    ocChunks, owb, jcp.nbOw, dummy, 1);
            } else {
                assert(!"unsupported loop order");
            }
            
            while (start < end) {
                IntType ocb = occ * jcp.nbOcBlocking;
                IntType g = gg * gBlocking;
                IntType gOcb = g * jcp.nbOc + ocb;
                IntType gIcb = g * jcp.nbIc * jcp.nonblkGroupOff;

                int owS = owb * jcp.owBlock;
                int iwS = owS * jcp.strideW;
                
                const IntType ocOffIdx = isDstLayoutNxc
                        ? g * jcp.oc + ocb * jcp.ocBlock
                        : gOcb;
                auto dstW = dst + (n * jcp.oc * jcp.ow * jcp.ngroups +
                            ocOffIdx +
                            owS * jcp.oc * jcp.ngroups);
                if (!isDstLayoutNxc) {
                    dstW = dst + (n * jcp.oc * jcp.ow * jcp.ngroups +
                            ocOffIdx * jcp.ocBlock * jcp.ow +
                            owS * jcp.ocBlock);
                }
                const IntType icOffIdx = isSrcLayoutNxc
                        ? g * jcp.ic + icbL2 * jcp.icBlock
                        : gIcb + icbL2;
                auto srcW = src + (n * jcp.ic * jcp.iw * jcp.ngroups + icOffIdx + iwS * jcp.ic * jcp.ngroups);
                if (!isSrcLayoutNxc && jcp_.is1stConv) {
                    srcW = src + (n * jcp.ic * jcp.iw +
                            icOffIdx * jcp.iw +
                            iwS);
                } else if (!isSrcLayoutNxc && !jcp_.is1stConv) {
                    srcW = src + (n * jcp.ic * jcp.iw * jcp.ngroups +
                            icOffIdx * jcp.icBlock * jcp.iw +
                            iwS * jcp.icBlock);
                }
                auto whtW = wei + (jcp.oc * jcp.ic * g + jcp.ocBlock * (ocb * jcp.ic + icbL2 * jcp.icBlock)) * jcp.kw;
                auto biasW = bias ? bias
                                + ocOffIdx
                                        * (isDstLayoutNxc ? 1 : jcp.ocBlock)
                                   : nullptr;

                IntType icbStep = isSrcLayoutNxc ? jcp.nbIcL2 : 1;
                IntType icbEnd = std::min(static_cast<int>(jcp.nbIc), static_cast<int>(icbL2 + jcp.nbIcL2));
                const IntType ocWork = ThisBlockSize(ocb * jcp.ocBlock,
                    jcp.oc, jcp.nbOcBlocking * jcp.ocBlock);
                IntType icWork = icbStep * jcp.icBlock;
                
                for (IntType icb = icbL2; icb < icbEnd; icb += icbStep) {
                    IntType currNbIc = std::min(icbStep, icbEnd - icb);
                    IntType flags = 0;
                    if (icb == 0) {
                        flags |= static_cast<int32_t>(KuDNN::KudnnConvFlag::KUDNN_FLAG_IC_FIRST);
                    }
                    if (icb + currNbIc >= jcp.nbIc) {
                        flags |= static_cast<int32_t>(KuDNN::KudnnConvFlag::KUDNN_FLAG_IC_LAST);
                        icWork = ThisBlockSize(icb * jcp.icBlock,
                            jcp.ic, icbStep * jcp.icBlock);
                    }
                    KudnnJitConvKerPipelineOwThr(jitKer, parConv, srcW,
                        dstW, whtW, biasW, icb, 1, owb, icWork, ocWork, flags);
                    srcW += srcCstride;
                    whtW += whtIcStride;
                }
                if (jcp.loopOrder == KuDNN::KudnnConvLoopOrderT::KUDNN_LOOP_CWGN) {
                    IntType dummy {0};
                    NdIteratorJump(start, end, occ, ocChunks, owb, jcp.nbOw,
                        gg, nbGroups, n, jcp.mb, dummy, 1);
                } else if (jcp.loopOrder == KuDNN::KudnnConvLoopOrderT::KUDNN_LOOP_GNCW) {
                    IntType dummy {0};
                    NdIteratorJump(start, end, gg, nbGroups, n, jcp.mb, occ,
                        ocChunks, owb, jcp.nbOw, dummy, 1);
                } else {
                    assert(!"unsupported loop order");
                }
            }
        }
        // This call is required only to finalize pipeline with paramaters set
        // on the last iteration of loop above. Only valid pointers make sense
        // here as call parameters to avoid execution of prefetch instructions
        // with nullptr, other parameters are not used in real jit call here
        KudnnJitConvKerPipelineOwThr(jitKer, parConv, src, dst, wei, bias, 0, 0, 0, 0, 0, 0);
    });
}

template <typename DataType>
void ConvolutionJITFWDImpl::ExecuteForward2D(const DataType *src, const DataType *wei,
    DataType *dst, const DataType *bias) noexcept
{
#ifdef KUDNNL_920Pro
    if constexpr (std::is_same_v<DataType, float>) {
        if (UseMatrixComputation()) {
            ExecuteForwardMatrixComputation(src, wei, dst, bias);
            return;
        }
    }
#endif
    KudnnConvJitConfT& jcp = jcp_;
    if constexpr (std::is_same_v<DataType, float>) {
        if (jcp.isDepthwise) {
            ExecuteForwardDW(src, wei, dst, bias);
            return;
        }
    }
    Element::TypeT dtp = GetTensorDtp(jcp);
    assert(GetKernel(dtp) != nullptr);
    KudnnConvJitKerT jitKer = (decltype(jitKer))(GetKernel(dtp));
    assert(jcp.nbOc % jcp.nbOcBlocking == 0);
    IntType ocChunks = jcp.nbOc / jcp.nbOcBlocking;
    int gBlocking = 1;
    IntType nbGroups = jcp.ngroups / gBlocking;
    IntType workAmount = jcp.mb * nbGroups * ocChunks * jcp.oh * jcp.nbOw;
    int nThr = jcp.alignedThreads;
    Parallel(nThr, [&](const int ithr, const int subNThr) {
        IntType start {0};
        IntType end {0};
        IntType startCopy;
        Balance211(workAmount, subNThr, ithr, start, end);
        startCopy = start;
        auto parConv = KuDNN::KudnnConvJitCallS();
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
            jcp.icBlock * jcp.ocBlock * jcp.kh * jcp.kw;
        for (IntType icbL2 = 0; icbL2 < jcp.nbIc; icbL2 += jcp.nbIcL2) {
            start = startCopy;
            IntType n {0};
            IntType gg {0};
            IntType occ {0};
            IntType ohS {0};
            IntType owb {0};
            if (jcp.loopOrder == KuDNN::KudnnConvLoopOrderT::KUDNN_LOOP_CWGN) {
                NdIteratorInit(start, occ, ocChunks, owb, jcp.nbOw, gg,
                    nbGroups, n, jcp.mb, ohS, jcp.oh);
            } else if (jcp.loopOrder == KuDNN::KudnnConvLoopOrderT::KUDNN_LOOP_GNCW) {
                NdIteratorInit(start, gg, nbGroups, n, jcp.mb, occ,
                    ocChunks, owb, jcp.nbOw, ohS, jcp.oh);
            } else {
                assert(!"unsupported loop order");
            }
            while (start < end) {
                IntType ocb = occ * jcp.nbOcBlocking;
                IntType g = gg * gBlocking;
                IntType gOcb = g * jcp.nbOc + ocb;
                IntType gIcb = g * jcp.nbIc * jcp.nonblkGroupOff;

                IntType workRem = end - start;

                IntType owS = owb * jcp.owBlock;
                IntType iwS = owS * jcp.strideW;
                IntType ohE = ohS + workRem > jcp.oh ? jcp.oh : ohS + workRem;

                for (IntType ohB = ohS; ohB < ohE; ohB += jcp.hBlocking) {
                    IntType ihB = -jcp.tPad + ohB * jcp.strideH;
                    const IntType ocOffIdx = isDstLayoutNxc
                            ? g * jcp.oc + ocb * jcp.ocBlock
                            : gOcb;
                    auto dstW = dst + (n * jcp.oc * jcp.ow * jcp.oh * jcp.ngroups +
                        ocOffIdx + ohB * jcp.oc * jcp.ow * jcp.ngroups + iwS * jcp.oc * jcp.ngroups);
                    if (!isDstLayoutNxc) {
                        dstW = dst + (n * jcp.oc * jcp.ow * jcp.oh * jcp.ngroups +
                            ocOffIdx * jcp.ocBlock * jcp.ow * jcp.oh + ohB * jcp.ocBlock * jcp.ow + iwS * jcp.ocBlock);
                    }
                    const IntType icOffIdx = isSrcLayoutNxc
                            ? g * jcp.ic + icbL2 * jcp.icBlock
                            : gIcb + icbL2;
                    auto srcW = src + (n * jcp.ic * jcp.iw * jcp.ih * jcp.ngroups +
                        icOffIdx + ihB * jcp.ic * jcp.iw * jcp.ngroups + iwS * jcp.ic * jcp.ngroups);
                    if (!isSrcLayoutNxc && jcp_.is1stConv) {
                        srcW = src + (n * jcp.ic * jcp.iw * jcp.ih * jcp.ngroups +
                            icOffIdx * jcp.iw * jcp.ih + ihB * jcp.iw + iwS);
                    } else if (!isSrcLayoutNxc && !jcp_.is1stConv) {
                        srcW = src + (n * jcp.ic * jcp.iw * jcp.ih * jcp.ngroups +
                            icOffIdx * jcp.icBlock * jcp.iw * jcp.ih + ihB * jcp.icBlock * jcp.iw + iwS * jcp.icBlock);
                    }
                    auto whtW = wei + (g * jcp.oc * jcp.ic * jcp.kh * jcp.kw +
                        ocb * jcp.ocBlock * jcp.ic * jcp.kw * jcp.kh +
                        icbL2 * jcp.icBlock * jcp.ocBlock * jcp.kw * jcp.kh);
                    IntType icbStep = isSrcLayoutNxc ? jcp.nbIcL2 : 1;
                    IntType icbEnd = std::min(static_cast<int>(jcp.nbIc), static_cast<int>(icbL2 + jcp.nbIcL2));
                    auto biasW = bias ? bias
                                    + ocOffIdx
                                            * (isDstLayoutNxc ? 1
                                                                 : jcp.ocBlock)
                                       : nullptr;
                    const IntType ocWork
                            = ThisBlockSize(ocb * jcp.ocBlock, jcp.oc,
                                jcp.nbOcBlocking * jcp.ocBlock);
                    IntType icWork = icbStep * jcp.icBlock;
                    for (IntType icb = icbL2; icb < icbEnd; icb += icbStep) {
                        IntType currNbIc = std::min(static_cast<int>(icbStep), static_cast<int>(icbEnd - icb));
                        IntType flags = 0;
                        if (icb == 0) {
                            flags |= static_cast<int32_t>(KuDNN::KudnnConvFlag::KUDNN_FLAG_IC_FIRST);
                        }
                        if (icb + currNbIc >= jcp.nbIc) {
                            flags |= static_cast<int32_t>(KuDNN::KudnnConvFlag::KUDNN_FLAG_IC_LAST);
                            icWork = ThisBlockSize(icb * jcp.icBlock, jcp.ic, icbStep * jcp.icBlock);
                        }
                        auto srcC = srcW;
                        auto dstC = dstW;
                        for (IntType oj = ohB, ij = ihB;
                                oj < std::min(static_cast<int>(ohE), static_cast<int>(ohB + jcp.hBlocking));
                                ++oj, ij += jcp.strideH) {
                            IntType dilateH = jcp.dilateH + 1;
                            IntType iToverflow = DivUp(std::max(0, static_cast<int>(-ij)), static_cast<int>(dilateH));
                            IntType iBoverflow = DivUp(
                                std::max(0, static_cast<int>(ij - jcp.ih + (jcp.kh - 1) * dilateH) + 1), dilateH);
                            IntType khPadding = std::max(0, static_cast<int>(jcp.kh - iToverflow - iBoverflow));
                            auto auxSrc = srcC + iToverflow * dilateH * srcHstride;
                            auto auxWht = whtW + iToverflow * whtHstride;
                            KudnnJitConvKerPipelineOwThr(jitKer, parConv, auxSrc, dstC, auxWht, biasW, icb,
                                khPadding, owb, icWork, ocWork, flags);
                            srcC += srcHstride * jcp.strideH;
                            dstC += dstHstride;
                        }
                        srcW += srcCstride;
                        whtW += whtIcStride;
                    }
                }
                if (jcp.loopOrder == KuDNN::KudnnConvLoopOrderT::KUDNN_LOOP_CWGN) {
                    NdIteratorJump(start, end, occ, ocChunks, owb, jcp.nbOw,
                        gg, nbGroups, n, jcp.mb, ohS, jcp.oh);
                } else if (jcp.loopOrder == KuDNN::KudnnConvLoopOrderT::KUDNN_LOOP_GNCW) {
                    NdIteratorJump(start, end, gg, nbGroups, n, jcp.mb, occ,
                        ocChunks, owb, jcp.nbOw, ohS, jcp.oh);
                } else {
                    assert(!"unsupported loop order");
                }
            }
        }
        // This call is required only to finalize pipeline with paramaters set
        // on the last iteration of loop above. Only valid pointers make sense
        // here as call parameters to avoid execution of prefetch instructions
        // with nullptr, other parameters are not used in real jit call here
        KudnnJitConvKerPipelineOwThr(jitKer, parConv, src, dst, wei, bias, 0, 0, 0, 0, 0, 0);
    });
}

template <typename DataType>
void ConvolutionJITFWDImpl::ExecuteForward3D(const DataType *src, const DataType *wei,
    DataType *dst, const DataType *bias) noexcept
{
#ifdef KUDNNL_920Pro
    if constexpr (std::is_same_v<DataType, float>) {
        if (UseMatrixComputation()) {
            ExecuteForwardMatrixComputation(src, wei, dst, bias);
            return;
        }
    }
#endif
    KudnnConvJitConfT& jcp = jcp_;
    Element::TypeT dtp = GetTensorDtp(jcp);
    assert(GetKernel(dtp) != nullptr);
    KudnnConvJitKerT jitKer = (decltype(jitKer))(GetKernel(dtp));
    assert(jcp.nbOc % jcp.nbOcBlocking == 0);
    IntType ocChunks = jcp.nbOc / jcp.nbOcBlocking;
    int gBlocking = 1;
    IntType nbGroups = jcp.ngroups / gBlocking;
    IntType workAmount = jcp.mb * nbGroups * ocChunks * jcp.od * jcp.oh * jcp.nbOw;
    int nThr = jcp.alignedThreads;
    
    Parallel(nThr, [&](const int ithr, const int subNThr) {
        IntType start {0};
        IntType end {0};
        IntType startCopy;
        Balance211(workAmount, subNThr, ithr, start, end);
        startCopy = start;
        auto parConv = KuDNN::KudnnConvJitCallS();
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
            jcp.icBlock * jcp.ocBlock * jcp.kh * jcp.kw * jcp.kd;
        for (IntType icbL2 = 0; icbL2 < jcp.nbIc; icbL2 += jcp.nbIcL2) {
            start = startCopy;
            IntType n {0};
            IntType gg {0};
            IntType occ {0};
            IntType ohS {0};
            IntType odS {0};
            IntType owb {0};
            if (jcp.loopOrder == KuDNN::KudnnConvLoopOrderT::KUDNN_LOOP_CWGN) {
                NdIteratorInit(start, occ, ocChunks, owb, jcp.nbOw, gg,
                    nbGroups, n, jcp.mb, odS, jcp.od, ohS, jcp.oh);
            } else if (jcp.loopOrder == KuDNN::KudnnConvLoopOrderT::KUDNN_LOOP_GNCW) {
                NdIteratorInit(start, gg, nbGroups, n, jcp.mb, occ,
                    ocChunks, owb, jcp.nbOw, odS, jcp.od, ohS, jcp.oh);
            } else {
                assert(!"unsupported loop order");
            }

            while (start < end) {
                IntType ocb = occ * jcp.nbOcBlocking;
                IntType g = gg * gBlocking;
                IntType gOcb = g * jcp.nbOc + ocb;
                IntType gIcb = g * jcp.nbIc * jcp.nonblkGroupOff;

                IntType workRem = end - start;
                IntType ihS = -jcp.tPad + ohS * jcp.strideH;
                IntType owS = owb * jcp.owBlock;
                IntType iwS = owS * jcp.strideW;
                IntType ohE = ohS + workRem > jcp.oh ? jcp.oh : ohS + workRem;

                IntType idS = -jcp.fPad + odS * jcp.strideD;
                IntType dilateD = jcp.dilateD + 1;
                IntType dToverflow = DivUp(std::max(0, static_cast<int>(-idS)), dilateD);
                IntType dBoverflow = DivUp(
                    std::max(0, static_cast<int>(idS - jcp.id + (jcp.kd - 1) * dilateD + 1)),
                    dilateD);
                IntType kdPadding = std::max(0, static_cast<int>(jcp.kd - dToverflow - dBoverflow));

                const IntType ocOffIdx = isDstLayoutNxc
                        ? g * jcp.oc + ocb * jcp.ocBlock
                        : gOcb;
                auto dstW = dst + (n * jcp.oc * jcp.ow * jcp.oh * jcp.od * jcp.ngroups +
                    ocOffIdx + odS * jcp.ngroups * jcp.oc * jcp.ow * jcp.oh +
                    ohS * jcp.ngroups * jcp.oc * jcp.ow +
                    owS * jcp.ngroups * jcp.oc);
                if (!isDstLayoutNxc) {
                    dstW = dst + (n * jcp.oc * jcp.ow * jcp.oh * jcp.od * jcp.ngroups +
                        ocOffIdx * jcp.od * jcp.ow * jcp.oh * jcp.ocBlock +
                        odS * jcp.ocBlock * jcp.ow * jcp.oh +
                        ohS * jcp.ocBlock * jcp.ow +
                        owS * jcp.ocBlock);
                }
                const IntType icOffIdx = isSrcLayoutNxc
                        ? g * jcp.ic + icbL2 * jcp.icBlock
                        : gIcb + icbL2;
                auto srcW = src + (n * jcp.ic * jcp.iw * jcp.ih * jcp.id * jcp.ngroups +
                    icOffIdx + idS * jcp.ngroups * jcp.ic * jcp.iw * jcp.ih +
                    ihS * jcp.ngroups * jcp.ic * jcp.iw +
                    iwS * jcp.ngroups * jcp.ic +
                    dToverflow * dilateD * srcDstride);
                if (!isSrcLayoutNxc && jcp_.is1stConv) { // when src layout is not axb / aBcde16b, and is abx
                    srcW = src + (n * jcp.ic * jcp.iw * jcp.ih * jcp.id * jcp.ngroups +
                        icOffIdx * jcp.id * jcp.iw * jcp.ih +
                        idS * jcp.iw * jcp.ih +
                        ihS * jcp.iw +
                        iwS +
                        dToverflow * dilateD * srcDstride);
                } else if (!isSrcLayoutNxc && !jcp_.is1stConv) { // when src layout is not axb, and is aBcde16b
                    srcW = src + (n * jcp.ic * jcp.iw * jcp.ih * jcp.id * jcp.ngroups +
                        icOffIdx * jcp.id * jcp.iw * jcp.ih * jcp.icBlock +
                        idS * jcp.icBlock * jcp.iw * jcp.ih +
                        ihS * jcp.icBlock * jcp.iw +
                        iwS * jcp.icBlock +
                        dToverflow * dilateD * srcDstride);
                }
                auto whtW = wei + (ocb * jcp.ocBlock * jcp.ic * jcp.kh * jcp.kw * jcp.kd +
                    icbL2 * jcp.ocBlock * jcp.ocBlock * jcp.kw * jcp.kh * jcp.kd + dToverflow * whtDstride);
                auto biasW = bias ? bias + ocOffIdx * (isDstLayoutNxc ? 1 : jcp.ocBlock) : nullptr;
                const IntType icbStep = isSrcLayoutNxc ? jcp.nbIcL2 : 1;
                IntType icbEnd = std::min(static_cast<int>(jcp.nbIc), static_cast<int>(icbL2 + jcp.nbIcL2));
                const IntType ocWork = ThisBlockSize(ocb * jcp.ocBlock, jcp.oc, jcp.nbOcBlocking * jcp.ocBlock);
                IntType icWork = icbStep * jcp.icBlock;
                for (IntType icb = icbL2; icb < icbEnd; icb += icbStep) {
                    IntType currNbIc = std::min(static_cast<int>(icbStep), static_cast<int>(icbEnd - icb));
                    IntType flags = 0;
                    if (icb == 0) {
                        flags |= static_cast<int32_t>(KuDNN::KudnnConvFlag::KUDNN_FLAG_IC_FIRST);
                    }
                    if (icb + currNbIc >= jcp.nbIc) {
                        flags |= static_cast<int32_t>(KuDNN::KudnnConvFlag::KUDNN_FLAG_IC_LAST);
                        icWork = ThisBlockSize(icb * jcp.icBlock, jcp.ic, icbStep * jcp.icBlock);
                    }
                    auto srcC = srcW;
                    auto dstC = dstW;
                    for (IntType oj = ohS, ij = ihS; oj < ohE;
                            ++oj, ij += jcp.strideH) {
                        IntType dilateH = jcp.dilateH + 1;
                        IntType iToverflow = DivUp(std::max(0, static_cast<int>(-ij)), dilateH);
                        IntType iBoverflow = DivUp(std::max(0,
                            static_cast<int>(ij - jcp.ih + (jcp.kh - 1) * dilateH + 1)), dilateH);
                        IntType khPadding = std::max(
                            0, static_cast<int>(jcp.kh - iToverflow - iBoverflow));
                        KudnnJitSveConv3dKerPipelineOwThr(jitKer, parConv,
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

                if (jcp.loopOrder == KuDNN::KudnnConvLoopOrderT::KUDNN_LOOP_CWGN) {
                    NdIteratorJump(start, end, occ, ocChunks, owb, jcp.nbOw,
                        gg, nbGroups, n, jcp.mb, odS, jcp.od, ohS,
                        jcp.oh);
                } else if (jcp.loopOrder == KuDNN::KudnnConvLoopOrderT::KUDNN_LOOP_GNCW) {
                    NdIteratorJump(start, end, gg, nbGroups, n, jcp.mb, occ,
                        ocChunks, owb, jcp.nbOw, odS, jcp.od, ohS,
                        jcp.oh);
                } else {
                    assert(!"unsupported loop order");
                }
            }
        }

        // This call is required only to finalize pipeline with paramaters set
        // on the last iteration of loop above. Only valid pointers make sense
        // here as call parameters to avoid execution of prefetch instructions
        // with nullptr, other parameters are not used in real jit call here
        KudnnJitSveConv3dKerPipelineOwThr(jitKer, parConv, src, dst,
            wei, bias, 0, 0, 0, 0, 0, 0, 0);
    });
}
} // Detail

ConvolutionJITFWD::~ConvolutionJITFWD() noexcept {}

ConvolutionJITFWD::ConvolutionJITFWD(int isa, const TensorInfo &src, const TensorInfo &weights,
    const TensorInfo &dst, const TensorInfo &bias, KuDNN::Shape strides, KuDNN::Shape paddingL,
    KuDNN::Shape dilates, KuDNN::Propagation prop, int nthreads, bool isDataLayoutNxc) noexcept(false)
    : pImpl(new Detail::ConvolutionJITFWDImpl(isa, src, weights, dst, bias, strides,
        paddingL, dilates, prop, nthreads, isDataLayoutNxc)) {}

ConvolutionJITFWD::ConvolutionJITFWD(const ConvolutionJITFWD &other) noexcept(false)
    : pImpl(new Detail::ConvolutionJITFWDImpl{*(other.pImpl.get())}) {}

ConvolutionJITFWD::ConvolutionJITFWD(ConvolutionJITFWD &&other) noexcept
    : pImpl(std::move(other.pImpl)) {}

ConvolutionJITFWD& ConvolutionJITFWD::operator=(const ConvolutionJITFWD &other) noexcept(false)
{
    if (this != &other) {
        this->pImpl.reset(new Detail::ConvolutionJITFWDImpl{*(other.pImpl.get())});
    }
    return *this;
}

ConvolutionJITFWD& ConvolutionJITFWD::operator=(ConvolutionJITFWD &&other) noexcept
{
    if (this != &other) {
        this->pImpl = std::move(other.pImpl);
    }
    return *this;
}

void ConvolutionJITFWD::ExecuteForward(const void *src, const void *wei,
    void *dst, const void *bias) noexcept
{
    pImpl->ExecuteForward(src, wei, dst, bias);
}

Status ConvolutionJITFWD::ValidateInput(const TensorInfo &src, const TensorInfo &weights,
    const TensorInfo &dst, const TensorInfo &bias,
    const Shape &strides, const Shape &dilates, const Shape &paddingL, int kCpuIsa) noexcept
{
    Service::Unused(bias);
    if (CheckDwConv(src, weights, dst, kCpuIsa)) {
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
        if (false || extKH <= tPad || extKH <= bPad || extKW <= lPad || extKW <= rPad) {
            return Status::UNSUPPORTED;
        }

        IntType urW = 0;
        IntType chBlock = 0;
        IntType nbChBlocking = 0;
        ARMCG::CPUISAT isa = static_cast<ARMCG::CPUISAT>(kCpuIsa);
        if (isa == ARMCG::CPUISAT::VectorComputation_512) {
            urW = std::min(ow, UR_W_6);
            chBlock = ARMCG::CpuIsaTraits<ARMCG::CPUISAT::VectorComputation_512>::VLEN / sizeof(float);
            nbChBlocking = NB_CH_BLKING_4;
        } else if (isa == ARMCG::CPUISAT::VectorComputation_256) {
            urW = std::min(ow, UR_W_4);
            chBlock = ARMCG::CpuIsaTraits<ARMCG::CPUISAT::VectorComputation_256>::VLEN / sizeof(float);
            nbChBlocking = NB_CH_BLKING_3;
        } else {
            return Status::UNSUPPORTED;
        }
        IntType urWTail = ow % urW;
        IntType rPadNoTail = std::max(CalculateEndPadding(lPad, ow - urWTail, iw, strideW, extKW),
            static_cast<int>(0));
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
            if (maxInpOff > INT_MAX) {
                return Status::UNSUPPORTED;
            }

            const SizeType maxOCOff = maxChOff * oh * ow;
            const SizeType maxOWOff = static_cast<SizeType>(urW - 1) * chBlock;
            const SizeType maxOutOff = (maxOCOff + maxOWOff + maxExOff) * typeSizeOut;
            if (maxOutOff > INT_MAX) {
                return Status::UNSUPPORTED;
            }
        }

        if (ARMCG::GetCpu().GetSveLen() == ARMCG::Util::SveLen::VectorComputation_512
            && isa == ARMCG::CPUISAT::VectorComputation_256) {
            return Status::UNSUPPORTED;
        }

        return Status::SUCCESS;
    }
    const bool withGroups = weights.GetNumDims() == src.GetNumDims() + 1;
    IntType ngroups = withGroups ? weights.GetDims()[0] : 1;
    IntType oc = dst.GetDims()[1] / ngroups;
    IntType ic = src.GetDims()[1] / ngroups;
    const bool isSrcLayoutNxc = src.GetLayout() == KuDNN::Layout::ACB ||
        src.GetLayout() == KuDNN::Layout::ACDB || src.GetLayout() == KuDNN::Layout::ACDEB;
    const bool isDstLayoutNxc = dst.GetLayout() == KuDNN::Layout::ACB ||
        dst.GetLayout() == KuDNN::Layout::ACDB || dst.GetLayout() == KuDNN::Layout::ACDEB;
#ifdef KUDNNL_920Pro
    IntType simdW = ARMCG::CpuIsaTraits<ARMCG::CPUISAT::VectorComputation_512>::VLEN / sizeof(float);
    if (src.GetType() == Element::TypeT::F16) {
        simdW = ARMCG::CpuIsaTraits<ARMCG::CPUISAT::VectorComputation_512>::VLEN / sizeof(__fp16);
    }
#else
    IntType simdW = ARMCG::CpuIsaTraits<ARMCG::CPUISAT::VectorComputation_256>::VLEN / sizeof(float);
    if (src.GetType() == Element::TypeT::F16) {
        simdW = ARMCG::CpuIsaTraits<ARMCG::CPUISAT::VectorComputation_256>::VLEN / sizeof(__fp16);
    }
#endif
    bool isDataLayoutNxc = isSrcLayoutNxc && isDstLayoutNxc;
    bool is1stConv = (ic < simdW) && (ngroups == 1);
    IntType icBlock = is1stConv ? ic : simdW;
    IntType ocBlock = simdW;
    if (src.GetType() == KuDNN::Element::TypeT::F32 && weights.GetType() == KuDNN::Element::TypeT::F32
        && dst.GetType() == KuDNN::Element::TypeT::F32) {
        // for generate sve512 kernel, but user try to use sve256 kernel
        if ((kCpuIsa == static_cast<IntType>(ARMCG::CPUISAT::VectorComputation_256) && (simdW != IC_LEN8)) ||
            (kCpuIsa == static_cast<IntType>(ARMCG::CPUISAT::VectorComputation_512) && (simdW != IC_LEN16))) {
            return Status::UNSUPPORTED;
        }
        IntType ocPaded = RndUp(oc, ocBlock);
        IntType icPaded = RndUp(ic, icBlock);
        /* Input and output channels must be multiples of simdW */
        if (ngroups == 1 && !isDataLayoutNxc) {
            oc = RndUp(oc, ocBlock);
            ic = RndUp(ic, icBlock);
        }

        bool argsOk = true
            && (ic <= icPaded)
            && (oc <= ocPaded);
        if (!argsOk) {
            return Status::UNSUPPORTED;
        }
    } else if (src.GetType() == KuDNN::Element::TypeT::F16 && weights.GetType() == KuDNN::Element::TypeT::F16
        && dst.GetType() == KuDNN::Element::TypeT::F16) {
        // for generate sve512 kernel, but user try to use sve256 kernel
        if ((kCpuIsa == static_cast<IntType>(ARMCG::CPUISAT::VectorComputation_256) && (simdW != IC_LEN16)) ||
            (kCpuIsa == static_cast<IntType>(ARMCG::CPUISAT::VectorComputation_512) && (simdW != IC_LEN32))) {
            return Status::UNSUPPORTED;
        }
        icBlock = simdW;
        ocBlock = simdW;
    } else if (src.GetType() == KuDNN::Element::TypeT::BF16 && weights.GetType() == KuDNN::Element::TypeT::BF16
        && dst.GetType() == KuDNN::Element::TypeT::BF16) {
        // for generate sve512 kernel, but user try to use sve256 kernel
        if ((kCpuIsa == static_cast<IntType>(ARMCG::CPUISAT::VectorComputation_256) && (simdW != IC_LEN8)) ||
            (kCpuIsa == static_cast<IntType>(ARMCG::CPUISAT::VectorComputation_512) && (simdW != IC_LEN16))) {
            return Status::UNSUPPORTED;
        }
        icBlock = simdW;
        ocBlock = simdW;
    } else {
        return Status::UNSUPPORTED;
    }

    if (!(oc % ocBlock == 0 && ic % icBlock == 0)) {
        return Status::UNSUPPORTED;
    }
    return Status::SUCCESS;
}

} // KuDNN
