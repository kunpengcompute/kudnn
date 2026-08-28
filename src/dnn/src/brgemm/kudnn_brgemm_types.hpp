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

#ifndef KUDNN_BRGEMM_TYPES_HPP
#define KUDNN_BRGEMM_TYPES_HPP

#include <climits>
#include <cstdint>
#include <cstring>

#include "cg_jit_generator.hpp"
#include "service/kudnn_err_codes.hpp"
#include "service/kudnn_exception.hpp"
#include "types/kudnn_data_type.hpp"
#include "types/kudnn_tensor_info.hpp"

namespace KuDNN {

inline SizeType DataTypeVnniGranularity(Element::TypeT type)
{
    SizeType granularity = static_cast<SizeType>(0);
    switch (type) {
        case Element::TypeT::F32:
        case Element::TypeT::S32: granularity = static_cast<SizeType>(1); break;
        default: throw Service::Unsupported(""); break;
    }

    return granularity;
}

enum class FpmathModeT {
    STRICT = 0,
    BF16,
    F16,
    ANY,
    TF32
};

enum class BrgemmBatchKindT {
    UNDEFINED = 0,
    ADDRESS,
    OFFSET,
    STRIDE,
    STATIC
};

enum class BrgemmLayoutT {
    UNDEFINED = 0,
    COL_MAJOR,
    ROW_MAJOR
};

enum class BrgemmBroadcastT {
    UNDEFINED = 0,
    TENSOR,
    M,
    N,
    K
};

struct BrgemmStridesT {
    IntType strideA;
    IntType strideB;
};

enum class BrgemmKernelLoopOrderT {
    UNDEFINED = 0,
    LOAD,
    BCST
};

enum class BrgemmKernelPrefetchingT {
    UNDEFINED = 1,
    PRF1,
    PRF2
};

enum class BrgemmKernelInnermostLoopT {
    UNDEFINED = 0,
    BD,
    LD
};

enum class BrgemmKernelHintNtT {
    UNDEFINED = 0,
    FALSE,
    TRUE
};

struct BrgemmPrfT {
    IntType dist1 = -1;
    IntType dist2 = -1;
};

struct BrgemmBatchElementT {
    BrgemmBatchElementT()
    {
        ptr.a = nullptr;
        ptr.b = nullptr;
        offset.a = 0;
        offset.b = 0;
        vvpad.top = 0;
        vvpad.bottom = 0;
        hvpad.left = 0;
        hvpad.right = 0;
    }

    struct {
        const void *a;
        const void *b;
    } ptr;
    struct {
        IntType a;
        IntType b;
    } offset;
    struct {
        IntType top;
        IntType bottom;
    } vvpad;
    struct {
        IntType left;
        IntType right;
    } hvpad;
};

struct BrgemmAttrT {
    BrgemmAttrT() : maxBs(INT_MAX), maxTopVpad(0), maxBottomVpad(0), hintExpectedASize(ARMCG::GetCoreCacheSize(1)),
        hintExpectedBSize(ARMCG::GetCoreCacheSize(1)), hintExpectedCSize(ARMCG::GetCoreCacheSize(1)),
        hintInnermostLoop(BrgemmKernelInnermostLoopT::LD), hintLoopOrder(BrgemmKernelLoopOrderT::UNDEFINED),
        hintPrefetching(BrgemmKernelPrefetchingT::UNDEFINED), waryAKTailRead(true), extendableK(false),
        generateSkipAccumulation(false), bdMaskLevel(0), useUker(false), useInterleaveStores(false), ldA2(0), ldB2(0),
        ldC2M(0), ldC2N(0), bdMask(nullptr), staticOffsets(nullptr) {}

    // if unrolled kernel is used, maxBs is the the only batch size that can be used on kernel call
    // else maxBs is the maximum batch size that can be used
    IntType maxBs;
    IntType maxTopVpad;
    IntType maxBottomVpad;
    IntType hintExpectedASize;
    IntType hintExpectedBSize;
    IntType hintExpectedCSize;
    BrgemmKernelInnermostLoopT hintInnermostLoop = BrgemmKernelInnermostLoopT::LD;
    BrgemmKernelLoopOrderT hintLoopOrder;
    BrgemmKernelPrefetchingT hintPrefetching = BrgemmKernelPrefetchingT::UNDEFINED;
    BrgemmPrfT hintPrfA;
    BrgemmPrfT hintPrfB;
    BrgemmPrfT hintPrfC;

    bool waryAKTailRead {false};
    bool extendableK {false};
    bool generateSkipAccumulation;
    // bdMaskLevel specifies how bdMask is used in brgemm kernel
    // 0 – bd mask is not used
    // 1 – bd mask is used on storing stage only
    // 2 – bd mask used both on reading and storing stages
    IntType bdMaskLevel;
    // useUker is a boolean value that determines whether to use the unrolled
    // kernel or not
    bool useUker;
    // useInterleaveStores is a value that determines whether to use the
    // interleave stores or not
    bool useInterleaveStores;
    FpmathModeT fpmathMode = FpmathModeT::STRICT;
    bool bIsVnni {false};
    // second level leading dimension describing distance between 16-line
    // blocks in case of blocked layout. Used to calculate address of next
    // bdBlock. By default are equal to regular leading dimension parameters
    // specified on brgemm creation. supported by brgemm unrolled kernel for now
    IntType ldA2 {0};
    IntType ldB2 {0};
    IntType ldC2M {0};
    IntType ldC2N {0};
    // if true then batchsize is allowed to change on each kernel call
    // and there is no unrolling by batchsize in kernel
    bool varBs {false};
    bool postopsOnly {false};

    IntType hintBdBlock {0};
    IntType hintLdBlock {0};
    IntType hintBdBlock2 {0};
    IntType hintLdBlock2 {0};
    bool hintUnunrollBdLoop {false};

    BrgemmKernelHintNtT hintLoadNtA {BrgemmKernelHintNtT::UNDEFINED};
    BrgemmKernelHintNtT hintLoadNtB {BrgemmKernelHintNtT::UNDEFINED};
    // this variable is used in tile decomposition heuristics
    // to calculate "effective" K value which may be different from just
    // K * batchsize for non-1x1 convolutions due to data overlap
    float kKoef = 1.0f;
    // this flag may be used to omit some actions on calling from blocking
    // in convolutions InitConf
    bool testCall = false;
    // broad mask is char array in which each element is a boolean value that
    // determines whether to write this row to the result matrix or skip
    const char *bdMask;
    // static offset is a static array of fixed offsets used for STATIC batch kind
    const BrgemmBatchElementT *staticOffsets;
};

struct BrgemmT {
    // new added parameters must be taken into account in the brgemm comparison function
    IntType bcastDim = 0;  // M
    IntType loadDim = 0;   // N
    IntType reduceDim = 0; // K
    IntType ldA = 0;
    IntType ldB = 0;
    IntType ldC = 0;
    IntType ldD = 0;
    // isaUser store the user provided isa
    // isaImpl store actual implementation
    ARMCG::CPUISAT isaUser = ARMCG::CPUISAT::ISA_UNDEF;
    ARMCG::CPUISAT isaImpl = ARMCG::CPUISAT::ISA_UNDEF;
    float alpha = 0.0f;
    float beta = 0.0f;

    Element::TypeT dtA = Element::TypeT::UNDEFINED;
    Element::TypeT dtB = Element::TypeT::UNDEFINED;
    Element::TypeT dtC = Element::TypeT::UNDEFINED;
    Element::TypeT dtD = Element::TypeT::UNDEFINED;
    Element::TypeT dtBias = Element::TypeT::UNDEFINED;

    IntType strideA = 0;
    IntType strideB = 0;

    BrgemmLayoutT layout = BrgemmLayoutT::UNDEFINED;
    BrgemmBatchKindT type;
    bool isDgmm = false;
    bool withSum = false;
    bool reqCalCompPads = false;

    float sumScale = 0.0f;
    int32_t sumZp = 0;
    Element::TypeT sumDt;
    bool withEltwise = false;
    bool withBinary = false;
    bool withScales = false;

    BrgemmBroadcastT zpTypeA = BrgemmBroadcastT::UNDEFINED;
    BrgemmBroadcastT zpTypeB = BrgemmBroadcastT::UNDEFINED;
    BrgemmBroadcastT zpTypeC = BrgemmBroadcastT::UNDEFINED;

    IntType isOcScale = 0;
    bool withDstScales = false;

    BrgemmAttrT brgattr;

    IntType ldA2 {0};
    IntType ldB2 {0};
    IntType ldC2M {0};
    IntType ldC2N {0};
    bool isBlocked = false;

    IntType bdb = 0;
    IntType bdBlock = 0;
    IntType bdbTail = 0;
    IntType bdb2 = 0;
    IntType bdBlock2 = 0;
    IntType bdb2Tail = 0;
    IntType ldb = 0;
    IntType ldBlock = 0;
    IntType ldbTail = 0;
    IntType ldb2 = 0;
    IntType ldBlock2 = 0;
    IntType ldb2Tail = 0;
    IntType rdb = 0;
    IntType rdBlock = 0;
    IntType rdbTail = 0;
    IntType rdStep = 0;
    IntType ldStep = 0;

    IntType typeSizeA = 0;
    IntType typeSizeB = 0;
    IntType typeSizeC = 0;
    IntType typeSizeD = 0;
    IntType typeSizeBias = 0;

    bool isYmm = false;
    bool isZmm = false;

    bool isInt8 = false;
    bool isBF16 = false;
    bool isBF16Emu = false;
    bool isF16 = false;

    bool isF32 = false;
    bool isBF32 = false;

    bool hasInt8Vnni = false;

    bool loadNtA = false;
    bool loadNtB = false;

    bool embdBcst = false;
    bool withBias = false;
    bool reqS8S8Compensation = false;
    bool withWeightsScaleAdjust = false;
    BrgemmKernelInnermostLoopT innermostLoop = BrgemmKernelInnermostLoopT::LD;
    IntType isMTail;
    bool interleaveTilestores = false;
    BrgemmPrfT prfA;
    BrgemmPrfT prfB;
    BrgemmPrfT prfC;

    static constexpr IntType MAX_VPAD = 100;

    const void *attr = nullptr;
    const TensorInfo *dst = nullptr;

    bool IsRowMajor() const
    {
        return layout == BrgemmLayoutT::ROW_MAJOR;
    }
    IntType GetWspBufferSize() const noexcept
    {
        return static_cast<IntType>(0);
    }
    bool IsBDataLayoutVnni()
    {
        return true;
    }

    bool operator<(const BrgemmT &rhs) const
    {
        return std::memcmp(this, &rhs, sizeof(BrgemmT)) < 0;
    }
};

struct BrgemmKernelParamsT {
    const void *ptrA;
    const void *ptrB;
    const BrgemmBatchElementT *batch;
    void *ptrC;

    const void *ptrBias;
    void *ptrD;

    // kernel takes single pointer scales, but configuration relies on a
    // combination of arg scales. This helps to reuse attributes from
    // primitives, but requires them to pre-compute
    // scales = src_scale * wei_scale[:]
    const void *ptrScales;
    void *ptrBuf;

    SizeType doPostOps;
    SizeType doApplyComp;
    SizeType bs;

    // ptr to table of void * elements that are pointers to postop binary
    const void *postOpsBinaryRhsArgVec;
    SizeType ocLogicalOff;
    SizeType firstMbMatrixAddrOff;
    SizeType dstRowLogicalOff;

    const char *dataCPtr;

    const void *aZpCompensations = nullptr;
    const void *bZpCompensations = nullptr;
    const void *cZpValues = nullptr;
    SizeType skipAccm = 0;
    int32_t zpAVal = 1;
    const void *ptrDstScales = nullptr;
};

struct BrgemmBaseKernelT {
public:
    BrgemmBaseKernelT() {};
    virtual ~BrgemmBaseKernelT() {};
    virtual void operator()(BrgemmKernelParamsT *) const = 0;

    virtual Status CreateKernel() = 0;
};

struct BrgemmPostOpsDataT {
    BrgemmPostOpsDataT() = default;
    BrgemmPostOpsDataT(const void *biasIn, const float *scalesIn, const void *binaryPostOpsRhsIn,
        SizeType ocLogicalOffIn, const SizeType dstRowLogicalOffIn = 0, const char *dataCPtrIn = nullptr,
        const SizeType firstMbMatrixAddrOffIn = 0, const void *aZpCompensationsIn = nullptr,
        const void *bZpCompensationsIn = nullptr, const void *cZpValuesIn = nullptr, bool skipAccumulationIn = false,
        int32_t zpAValIn = 1, bool doOnlyCompIn = false, bool doOnlyZpAValIn = false,
        const float *dstScalesIn = nullptr) : bias(biasIn), scales(scalesIn), binaryPostOpsRhs(binaryPostOpsRhsIn),
        ocLogicalOff(ocLogicalOffIn), dstRowLogicalOff(dstRowLogicalOffIn), dataCPtr(dataCPtrIn),
        firstMbMatrixAddrOff(firstMbMatrixAddrOffIn), aZpCompensations(aZpCompensationsIn),
        bZpCompensations(bZpCompensationsIn), cZpValues(cZpValuesIn), skipAccumulation(skipAccumulationIn),
        zpAVal(zpAValIn), doOnlyComp(doOnlyCompIn), doOnlyZpAVal(doOnlyZpAValIn), dstScales(dstScalesIn) {}

    const void *bias = nullptr;
    const float *scales = nullptr;
    const void *binaryPostOpsRhs = nullptr;
    SizeType ocLogicalOff = 0;
    SizeType dstRowLogicalOff = 0;
    const char *dataCPtr = nullptr;
    SizeType firstMbMatrixAddrOff = 0;
    const void *aZpCompensations = nullptr;
    const void *bZpCompensations = nullptr;
    const void *cZpValues = nullptr;
    const bool skipAccumulation = false;
    int32_t zpAVal = 1;
    const bool doOnlyComp = false;
    const bool doOnlyZpAVal = false;
    const float *dstScales = nullptr;
};

} // namespace KuDNN

#endif // KUDNN_BRGEMM_TYPES_HPP
