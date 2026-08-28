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
#ifndef KUDNN_WINO_CONV_HPP
#define KUDNN_WINO_CONV_HPP
#include "internal_funcs.hpp"
#include "kudnn_convolution_base.hpp"

namespace KuDNN {
namespace Detail {
class ITransFormKernel {
public:
    ITransFormKernel() noexcept(false) {};
    virtual ~ITransFormKernel() {};
public:
    virtual void DoTransForm(const IntType& channel, const float* input, float* output,
        const IntType& rowStride, const IntType& colStride, IntType matrixStride,
        const float* bias) noexcept(false) = 0;
};

class OutputTransFormKer16x9 : public ITransFormKernel {
public:
    void DoTransForm(const IntType& channel, const float* input, float* output,
        const IntType& rowStride, const IntType& colStride, IntType matrixStride = 0,
        const float* bias = nullptr) override;
private:
    void ArmVCFloat16x9(const IntType& outChannel, const float* input, const IntType& matrixStride,
        const float* bias, float* output, const IntType& outRowStride, const IntType& outColStride);
};

class InputTransFormKer6x6 : public ITransFormKernel {
public:
    void DoTransForm(const IntType& channel, const float* input, float* output,
        const IntType& rowStride, const IntType& colStride, IntType matrixStride = 0,
        const float* bias = nullptr) override;
private:
    void ArmVCFloat6x6(const IntType& inChannel, const float* input, const IntType& inRowStride,
        const IntType& inColStride, float* output, const IntType matStride);
};

class WeightTransFormKer16x9 : public ITransFormKernel {
public:
    void DoTransForm(const IntType& channel, const float* input, float* output,
        const IntType& rowStride, const IntType& colStride, IntType matrixStride = 0,
        const float* bias = nullptr) override;
private:
    void ArmVCFloat16x9(const IntType& outChannel, const float* input, const IntType& ldWeiRow,
        const IntType& ldWeiCol, float* output, const IntType& matrixStride);
};

class GemmForWinoImpl final {
public:
    explicit GemmForWinoImpl() noexcept(false) {}
    ~GemmForWinoImpl() {}
public:
    void DoGemmForWino(const IntType& ic, const IntType& oc, const IntType& weiMatStride,
        const float* inPtr, const float* weiPtr, float* outPtr);
};

class WinoConvolutionLayerFWDImpl final : public ConvolutionBase, public ConvolutionFWDImplBase {
public:
    explicit WinoConvolutionLayerFWDImpl(const TensorInfo &src, const TensorInfo &weights,
        const TensorInfo &dst, const TensorInfo &bias, const Shape &strides, const Shape &dilates,
        const Shape &paddingL, const Shape &paddingR, ConvolutionAlgorithm convFWDAlg,
        const IntType& thr) noexcept(false) : ConvolutionBase(src, weights, dst, strides, dilates,
        paddingL, paddingR), alg(convFWDAlg), nThr(thr), inputTensor(src), kernelTensor(weights),
        outputTensor(dst), biaTensor(bias), padL(paddingL), isWeiTransformed(false), isDstGemmed(false),
        isAllNeedDealTail(false), isSrcNeedDealTail(false), isDstNeededCp(false), isSrcNeededCp(false)
    {
        Service::ThrowOnStatus(ValidateInput(
            src, weights, dst, bias, strides, dilates, paddingL, paddingR, alg), "Wino Convolution layer");
        CheckIntMultOverflow(std::make_tuple(params.inputShape[NUM_1],
            params.kernelShape[NUM_0], params.kernelShape[NUM_1], params.kernelShape[NUM_2],
            params.outputShape[NUM_2], params.outputShape[NUM_3], params.outputShape[NUM_4]),
            std::make_tuple((params.inputShape[NUM_0] - NUM_1),
                params.inputShape[NUM_2], params.inputShape[NUM_3], params.inputShape[NUM_4]),
            std::make_tuple((params.outputShape[NUM_0] - NUM_1),
                params.outputShape[NUM_2], params.outputShape[NUM_3], params.outputShape[NUM_4]));

        SizeType totalSize = (params.inputShape[NUM_1] / params.groups) *
            params.kernelShape[NUM_2] * params.kernelShape[NUM_3] * params.kernelShape[NUM_4] *
            params.outputShape[NUM_2] * params.outputShape[NUM_3] * params.outputShape[NUM_4] *
            src.GetType().GetSize();
        totalSize += params.outputShape[NUM_2] * params.outputShape[NUM_3] *
            params.outputShape[NUM_4] * dst.GetType().GetSize();
        if (totalSize > Detail::BYTES_AVAILABLE) {
            throw Service::LogicError {"Wino Convolution: exceeded available memory limit"};
        }
        bool isWeiNcx = kernelTensor.GetLayout() == Layout::ABCD ? true : false;
        bool isWeiNxc = kernelTensor.GetLayout() == Layout::ACDB ? true : false;
        if (!isWeiNxc && !isWeiNcx) {
            Service::ThrowOnStatus(Status::UNSUPPORTED, "kdnn wino is only unsuport ncx wei layout");
        }
        inputTransformKer_.reset(new InputTransFormKer6x6());
        outputTransformKer_.reset(new OutputTransFormKer16x9());
        weiTransformKer_.reset(new WeightTransFormKer16x9());
        gemmForWinoKer_.reset(new GemmForWinoImpl());
        
        const bool withGroups = kernelTensor.GetNumDims() == inputTensor.GetNumDims() + NUM_1;
        IntType ndims = inputTensor.GetNumDims();
        IntType nGroups = withGroups ? kernelTensor.GetDims()[0] : NUM_1;
        IntType mb = inputTensor.GetDims()[0];
        IntType ic = inputTensor.GetDims()[NUM_1] / nGroups;
        IntType ih = inputTensor.GetDims()[ndims - NUM_2];
        IntType oh = outputTensor.GetDims()[ndims - KERNEL_DIMS_2];
        IntType outScaleNum = oh % VLEN_NEON_F32 == 0 ? (oh / VLEN_NEON_F32) * (oh / VLEN_NEON_F32) :
            ((oh / VLEN_NEON_F32 + NUM_1) * (oh / VLEN_NEON_F32 + NUM_1));
        IntType oc = outputTensor.GetDims()[NUM_1] / nGroups;
        IntType kh = (ndims == KERNEL_DIMS_3) ? 1 : kernelTensor.GetDims()[withGroups + ndims - KERNEL_DIMS_2];
        IntType kw = kernelTensor.GetDims()[withGroups + ndims - 1];
        isAllNeedDealTail = oh % VLEN_NEON_F32 ? true : false;
        isSrcNeedDealTail = (oh - NUM_1 + kh) == ih ? false : true;

        IntType ohPadded = oh = oh % VLEN_NEON_F32 == 0 ? oh : (oh / VLEN_NEON_F32 + NUM_1) * VLEN_NEON_F32;
        IntType ihPadded = ohPadded - NUM_1 + kh;
        IntType srcNewSize = mb * ic * ihPadded * ihPadded * TYPE_SIZE;
        if (isAllNeedDealTail) {
            isDstNeededCp = true;
            isSrcNeededCp = true;
            oh = ihPadded - kh + NUM_1;
            srcPadedPtr = Service::AlignedAlloc(srcNewSize);
            if (!srcPadedPtr) {
                Service::ThrowOnStatus(Status::UNSUPPORTED,
                    "all need deal tail, kdnn wino padded src AlignedAlloc failed");
            }
            DataDup(static_cast<float*>(srcPadedPtr), 0, srcNewSize / TYPE_SIZE);
            IntType dstNewSize = mb * oc * oh * oh * TYPE_SIZE;
            dstPadedPtr = Service::AlignedAlloc(dstNewSize);
            if (!dstPadedPtr) {
                Service::Deallocate(srcPadedPtr);
                Service::ThrowOnStatus(Status::UNSUPPORTED,
                    "all need deal tail, kdnn wino padded dst AlignedAlloc failed");
            }
            DataDup(static_cast<float*>(dstPadedPtr), 0, dstNewSize / TYPE_SIZE);
        } else if (isSrcNeedDealTail) {
            isSrcNeededCp = true;
            srcPadedPtr = Service::AlignedAlloc(srcNewSize);
            if (!srcPadedPtr) {
                Service::ThrowOnStatus(Status::UNSUPPORTED,
                    "src need deal tail, kdnn wino padded src AlignedAlloc failed");
            }
            DataDup(static_cast<float*>(srcPadedPtr), 0, srcNewSize / TYPE_SIZE);
            ih = ihPadded;
        }

        IntType weiMatriStride = ic * (oc % VLEN_NEON_F32 == 0 ? oc :
            ((oc / VLEN_NEON_F32 + NUM_1) * VLEN_NEON_F32));
        IntType inStride = (ic % VLEN_NEON_F32) == 0 ? ic : ((ic / VLEN_NEON_F32 + NUM_1) * VLEN_NEON_F32);
        IntType ocAfterScale = (oc % VLEN_NEON_F32 == 0 ? oc : ((oc / VLEN_NEON_F32 + NUM_1) * VLEN_NEON_F32));
        transformWeiBuf = Service::AlignedAlloc(kh * kw * weiMatriStride *
            WEI_ENLARGE_NUM * TYPE_SIZE);
        if (!transformWeiBuf) {
            Service::ThrowOnStatus(Status::UNSUPPORTED, "kdnn wino transformWeiBuf AlignedAlloc failed");
        }
        DataDup(static_cast<float*>(transformWeiBuf), 0, kh * kw * weiMatriStride *
            WEI_ENLARGE_NUM);
        transformedInput = Service::AlignedAlloc(mb * ic * WINO_TRANS_MIN_MATRIX * inStride *
            VLEN_NEON_F32 * TYPE_SIZE);
        if (!transformedInput) {
            Service::ThrowOnStatus(Status::UNSUPPORTED, "kdnn wino transformedInput AlignedAlloc failed");
        }
        DataDup(static_cast<float*>(transformedInput), 0, mb * ic * WINO_TRANS_MIN_MATRIX * inStride * VLEN_NEON_F32);
        transformedOutput = Service::AlignedAlloc(mb * WINO_TRANS_MIN_MATRIX * ocAfterScale * TYPE_SIZE * outScaleNum);
        if (!transformedOutput) {
            Service::ThrowOnStatus(Status::UNSUPPORTED, "kdnn wino transformedOutput AlignedAlloc failed");
        }
        DataDup(static_cast<float*>(transformedOutput), 0, mb * WINO_TRANS_MIN_MATRIX * ocAfterScale * outScaleNum);
    }

    ~WinoConvolutionLayerFWDImpl() override
    {
        if (isSrcNeededCp && srcPadedPtr) {
            Service::Deallocate(srcPadedPtr);
        }
        if (isDstNeededCp && dstPadedPtr) {
            Service::Deallocate(dstPadedPtr);
        }
        if (transformWeiBuf) {
            Service::Deallocate(transformWeiBuf);
        }
        if (transformedOutput) {
            Service::Deallocate(transformedOutput);
        }
        if (transformedInput) {
            Service::Deallocate(transformedInput);
        }
    }

    void DataDup(float *src, int32_t dupVal, IntType len);
    void DataCopy(float *dst, const float *src, IntType len);

    void Run(const void *src, const void *wei,
        void *dst, const void *bias) noexcept(false) override
    {
        IntType ndims = inputTensor.GetNumDims();
        const bool withGroups = kernelTensor.GetNumDims() == inputTensor.GetNumDims() + NUM_1;
        IntType mb = inputTensor.GetDims()[0];
        IntType nGroups = withGroups ? kernelTensor.GetDims()[0] : NUM_1;
        IntType ic = inputTensor.GetDims()[NUM_1] / nGroups;
        IntType ih = inputTensor.GetDims()[ndims - NUM_2];
        IntType oc = outputTensor.GetDims()[NUM_1] / nGroups;
        IntType oh = outputTensor.GetDims()[ndims - NUM_2];
        IntType kh = kernelTensor.GetDims()[withGroups + ndims - NUM_2];
        IntType lPad = padL[ndims - NUM_3];

        void *hwioWei = nullptr;
        IntType weiMatriStride = ic * (oc % VLEN_NEON_F32 == 0 ? oc :
            ((oc / VLEN_NEON_F32 + NUM_1) * VLEN_NEON_F32));
        if (!isWeiTransformed) { // weight only do transform onece
            const float* weiPtr = static_cast<const float *>(wei);
            hwioWei = Service::AlignedAlloc(
                kernelTensor.GetTotalTensorSize() * kernelTensor.GetType().GetSize());
            ReorderLayer reorderWei({kernelTensor.GetDims(), kernelTensor.GetType(),
                kernelTensor.GetLayout()}, {kernelTensor.GetDims(),
                kernelTensor.GetType(), Layout::CDBA});
            reorderWei.Run(wei, hwioWei);
            weiPtr = static_cast<const float *>(hwioWei);
            
            IntType weiRowStride = kh * ic * oc;
            IntType weiColStride = ic * oc;
#pragma omp parallel for num_threads(nThr) collapse(1)
            for (auto inc = 0; inc < ic; inc++) {
                weiTransformKer_->DoTransForm(oc, weiPtr + inc * oc, static_cast<float *>(transformWeiBuf) +
                    inc * (weiMatriStride / ic), weiRowStride, weiColStride, weiMatriStride, nullptr);
            }
            isWeiTransformed = true;
        }

        auto doDataCp = [ic, src, mb, this] (const IntType &leftPad, const IntType &ihPadded, const IntType &iHei,
            void* srcPtr) {
            if (leftPad > 0) { // upper set pad val, need deal left and right pad
#pragma omp parallel for num_threads(nThr) collapse(3)
                for (IntType b = 0; b < mb; b++) {
                    for (IntType i = 0; i < iHei; i++) {
                        for (IntType j = 0; j < iHei; j++) {
                            DataCopy(static_cast<float*>(srcPtr) + (b * ihPadded * ihPadded * ic +
                                i * ihPadded * ic + j * ic) + leftPad * ihPadded * ic + ic * leftPad,
                                static_cast<float*>(const_cast<void*>(src)) + b * iHei * iHei * ic +
                                (i * iHei * ic + j * ic), ic);
                        }
                    }
                }
            } else { // upper unset pad val, only deal right pad
#pragma omp parallel for num_threads(nThr) collapse(3)
                for (IntType b = 0; b < mb; b++) {
                    for (IntType i = 0; i < iHei; i++) {
                        for (IntType j = 0; j < iHei; j++) {
                            DataCopy(static_cast<float*>(srcPtr) + (b * ihPadded * ihPadded * ic +
                                i * ihPadded * ic + j * ic),
                                static_cast<float*>(const_cast<void*>(src)) + b * iHei * iHei * ic +
                                (i * iHei * ic + j * ic), ic);
                        }
                    }
                }
            }
        };
        
        // do padding start
        IntType ohPadded = oh = oh % VLEN_NEON_F32 == 0 ? oh : (oh / VLEN_NEON_F32 + NUM_1) * VLEN_NEON_F32;
        IntType ihPadded = ohPadded - NUM_1 + kh;
        if (isAllNeedDealTail) {
            isDstNeededCp = true;
            isSrcNeededCp = true;
            oh = ihPadded - kh + NUM_1;
            doDataCp(lPad, ihPadded, ih, srcPadedPtr);
            ih = ihPadded;
        } else if (isSrcNeedDealTail) {
            isSrcNeededCp = true;
            doDataCp(lPad, ihPadded, ih, srcPadedPtr);
            ih = ihPadded;
            if (!isDstNeededCp) {
                dstPadedPtr = dst;
            }
        } else {
            if (!isSrcNeededCp) {
                srcPadedPtr = SafeCast<void*>(src);
            }
            if (!isDstNeededCp) {
                dstPadedPtr = dst;
            }
        }
        // do padding end
        IntType inStride = (ic % VLEN_NEON_F32) == 0 ? ic : ((ic / VLEN_NEON_F32 + NUM_1) * VLEN_NEON_F32);
        // prepare transformed weight
        IntType ocAfterScale = (oc % VLEN_NEON_F32 == 0 ? oc : ((oc / VLEN_NEON_F32 + NUM_1) * VLEN_NEON_F32));
        IntType inRowStride = ic * ih;
        IntType inColStride = ic;
        IntType inMatrixStride = inStride;

        IntType dstRowStride = oh * oc;
        IntType dstColStride = oc;
        IntType tmpDiffIhOh = ih - NUM_2;
        IntType pow2TmpDiffIhOh = tmpDiffIhOh * tmpDiffIhOh;
        IntType pow2MinMatirx = WINO_OUT_MIN_MATRIX_WEI * WINO_OUT_MIN_MATRIX_WEI;
        IntType srcOffset = ic * WINO_OUT_MIN_MATRIX_HEI;
        if (!isDstGemmed) {
            for (IntType b = 0; b < mb; b++) {
                for (IntType i = 0; i <= ih - WINO_MIN_MATRIX_WID; i += WINO_OUT_MIN_MATRIX_WEI) {
                    for (IntType j = 0; j <= ih - WINO_MIN_MATRIX_WID; j += WINO_OUT_MIN_MATRIX_HEI) {
                            inputTransformKer_->DoTransForm(ic, static_cast<const float *>(srcPadedPtr) + b * ih * ih *
                            ic + i * ih * ic + j / WINO_OUT_MIN_MATRIX_HEI * srcOffset,
                            static_cast<float *>(transformedInput), inRowStride, inColStride, inMatrixStride, nullptr);
                            gemmForWinoKer_->DoGemmForWino(ic, oc, weiMatriStride,
                                static_cast<const float *>(transformedInput),
                                static_cast<const float *>(transformWeiBuf), static_cast<float *>(transformedOutput) +
                                (WINO_TRANS_MIN_MATRIX * ocAfterScale * (b * pow2TmpDiffIhOh / pow2MinMatirx + (i *
                                tmpDiffIhOh / WINO_OUT_MIN_MATRIX_WEI + j) / WINO_OUT_MIN_MATRIX_HEI)));
                    }
                }
            }
            isDstGemmed = true;
        }
#pragma omp parallel for num_threads(nThr) collapse(3)
        for (IntType b = 0; b < mb; b++) {
            for (IntType i = 0; i <= (ih - WINO_MIN_MATRIX_WID); i += WINO_OUT_MIN_MATRIX_WEI) {
                for (IntType j = 0; j <= (ih - WINO_MIN_MATRIX_HEI); j += WINO_OUT_MIN_MATRIX_HEI) {
                    outputTransformKer_->DoTransForm(oc, static_cast<const float *>(transformedOutput) +
                        (WINO_TRANS_MIN_MATRIX * ocAfterScale * (b * pow2TmpDiffIhOh /
                        pow2MinMatirx + (i * tmpDiffIhOh / WINO_OUT_MIN_MATRIX_WEI + j) / WINO_OUT_MIN_MATRIX_HEI)),
                        (static_cast<float *>(dstPadedPtr)) + b * oh * oh * oc + (i * oh * oc) + j * oc,
                        dstRowStride, dstColStride, ocAfterScale, static_cast<const float *>(bias));
                }
            }
        }
        if (isDstNeededCp) {
            oh = outputTensor.GetDims()[ndims - KERNEL_DIMS_2];
            IntType ohPaded = ih + NUM_1 - kh;
#pragma omp parallel for num_threads(nThr) collapse(3)
            for (IntType b = 0; b < mb; b++) {
                for (IntType i = 0; i < oh; i++) {
                    for (IntType j = 0; j < oh; j++) {
                        DataCopy(static_cast<float*>(dst) + b * oh * oh * oc + (i * oh * oc + j * oc),
                            static_cast<float*>(dstPadedPtr) + b * ohPaded * ohPaded * oc +
                            (i * ohPaded * oc + j * oc), oc);
                    }
                }
            }
        }
        if (hwioWei) {
            Service::Deallocate(hwioWei);
        }
    }
private:
    std::unique_ptr<ITransFormKernel> weiTransformKer_;
    std::unique_ptr<ITransFormKernel> inputTransformKer_;
    std::unique_ptr<ITransFormKernel> outputTransformKer_;
    std::unique_ptr<GemmForWinoImpl> gemmForWinoKer_;
    ConvolutionAlgorithm alg;
    IntType nThr;
    TensorInfo inputTensor;
    TensorInfo kernelTensor;
    TensorInfo outputTensor;
    TensorInfo biaTensor;
    Shape padL;
    void *transformWeiBuf = nullptr;
    void *transformedInput = nullptr;
    void *transformedOutput = nullptr;
    void* srcPadedPtr = nullptr;
    void* dstPadedPtr = nullptr;
    bool isWeiTransformed;
    bool isDstGemmed;
    bool isAllNeedDealTail;
    bool isSrcNeedDealTail;
    bool isDstNeededCp;
    bool isSrcNeededCp;
};

} // Detail
} // KuDNN

#endif // KUDNN_WINO_CONV_HPP