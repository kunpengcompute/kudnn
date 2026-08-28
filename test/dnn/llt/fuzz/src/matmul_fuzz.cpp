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

#include "fuzz_test_common.h"

namespace KuDNN::Detail {
    void ClearImpls();
}

static std::map<int, std::pair<std::string, KuDNN::Element::TypeT>> g_fltPrecisionMap = {
    {INDEX0, {"F32", KuDNN::Element::TypeT::F32}},
    {INDEX1, {"F16", KuDNN::Element::TypeT::F16}},
    {INDEX2, {"BF16", KuDNN::Element::TypeT::BF16}},
};

static std::map<int, std::pair<std::string, KuDNN::Element::TypeT>> g_intPrecisionMap = {
    {INDEX0, {"U8", KuDNN::Element::TypeT::U8}},
    {INDEX1, {"S8", KuDNN::Element::TypeT::S8}},
    {INDEX2, {"S32", KuDNN::Element::TypeT::S32}},
};

#define NON_MIX_PRECISION_INDEX     INDEX0
#define MIX_PRECISION_INDEX         INDEX1

static std::map<int, std::string> g_mixPrecisionMap = {
    {NON_MIX_PRECISION_INDEX, "NON_MIX_PRECISION"},
    {MIX_PRECISION_INDEX, "MIX_PRECISION"},
};


static std::map<int, std::pair<std::string, KuDNN::Layout>> g_layoutMap = {
    {INDEX0, {"ABCDE", KuDNN::Layout::ABCDE}},
    {INDEX1, {"ABCED", KuDNN::Layout::ABCED}},
};

#define NON_BROADCAST_BIAS_INDEX     INDEX0
#define BROADCAST_BIAS_M_INDEX       INDEX1
#define BROADCAST_BIAS_N_INDEX       INDEX2
#define BROADCAST_BIAS_MN_INDEX      INDEX3

static std::map<int, std::string> g_broadCastBiasMap = {
    {NON_BROADCAST_BIAS_INDEX, "NON_BROADCAST_BIAS"},
    {BROADCAST_BIAS_M_INDEX, "BROADCAST_BIAS_M"},
    {BROADCAST_BIAS_N_INDEX, "BROADCAST_BIAS_N"},
    {BROADCAST_BIAS_MN_INDEX, "BROADCAST_BIAS_MN"},
};

static constexpr int g_dimRangeInitVal = 1;
static constexpr int g_dimRangeLowerBound = 1;
static constexpr int g_dimRangeUpperBound = 30;

static void DtFuzzMatmulFltCase(int counts)
{
    char funcName[] = "DtFuzzMatmulFltCase";
    DT_FUZZ_START(0, counts, funcName, 0)
    {
        int eleIndex = 0;
        int matAElemTypeIdx = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_fltPrecisionMap.size() - 1);
        int matBElemTypeIdx = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_fltPrecisionMap.size() - 1);
        int matCElemTypeIdx = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_fltPrecisionMap.size() - 1);
        int biasElemTypeIdx = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_fltPrecisionMap.size() - 1);
        int matALayoutIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_layoutMap.size() - 1);
        int matBLayoutIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_layoutMap.size() - 1);
        int matCLayoutIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_layoutMap.size() - 1);
        int matBiasLayoutIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_layoutMap.size() - 1);
        int broadCastBiasIdx = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0,
            g_broadCastBiasMap.size() - 1);

        size_t dim0 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dim1 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dim2 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dimM = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dimN = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dimK = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dimBiasM = dimM;
        size_t dimBiasN = dimN;

        // broadcast type
        switch (broadCastBiasIdx) {
            case BROADCAST_BIAS_M_INDEX:
                dimBiasM = 1;
                break;
            case BROADCAST_BIAS_N_INDEX:
                dimBiasN = 1;
                break;
            case BROADCAST_BIAS_MN_INDEX:
                dimBiasM = 1;
                dimBiasN = 1;
                break;
            default:
                break;
        }
        KuDNN::Element::Type matAType(g_fltPrecisionMap[matAElemTypeIdx].second);
        KuDNN::Element::Type matBType(g_fltPrecisionMap[matBElemTypeIdx].second);
        KuDNN::Element::Type matCType(g_fltPrecisionMap[matCElemTypeIdx].second);
        KuDNN::Element::Type biasType(g_fltPrecisionMap[biasElemTypeIdx].second);

        size_t matAElemSz = matAType.GetSize();
        size_t matBElemSz = matBType.GetSize();
        size_t matCElemSz = matCType.GetSize();
        size_t biasElemSz = biasType.GetSize();

        const KuDNN::TensorInfo matATensorInfo = {{dim0, dim1, dim2, dimM, dimK}, matAType,
            g_layoutMap[matALayoutIndex].second};
        const KuDNN::TensorInfo matBTensorInfo = {{dim0, dim1, dim2, dimK, dimN}, matBType,
            g_layoutMap[matBLayoutIndex].second};
        const KuDNN::TensorInfo matCTensorInfo = {{dim0, dim1, dim2, dimM, dimN}, matCType,
            g_layoutMap[matCLayoutIndex].second};
        const KuDNN::TensorInfo biasTensorInfo = {{dim0, dim1, dim2, dimBiasM, dimBiasN}, biasType,
            g_layoutMap[matBiasLayoutIndex].second};

        char format[] = "Matmul, Precision(A:B:C:bias): %s:%s:%s:%s, Shape: %zux%zux%zux%zux%zu X "
        "%zux%zux%zux%zux%zu = %zux%zux%zux%zux%zu + %zux%zux%zux%zux%zu, layout(A:B:C:bias): %s:%s:%s:%s ";
        printf(format, g_fltPrecisionMap[matAElemTypeIdx].first.c_str(), g_fltPrecisionMap[matBElemTypeIdx].first.c_str(),
            g_fltPrecisionMap[matCElemTypeIdx].first.c_str(), g_fltPrecisionMap[biasElemTypeIdx].first.c_str(),
            dim0, dim1, dim2, dimM, dimK, dim0, dim1, dim2, dimK, dimN, dim0, dim1, dim2, dimM, dimN,
            dim0, dim1, dim2, dimM, dimN, g_layoutMap[matALayoutIndex].first.c_str(),
            g_layoutMap[matBLayoutIndex].first.c_str(), g_layoutMap[matCLayoutIndex].first.c_str(),
            g_layoutMap[matBiasLayoutIndex].first.c_str());
        void *matA = NULL;
        void *matB = NULL;
        void *matC = NULL;
        void *matBias = NULL;
        try {
            KuDNN::Gemm gemmLayer(matATensorInfo, matBTensorInfo, matCTensorInfo, biasTensorInfo);
            matA = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                std::max((size_t)1, dimM) * std::max((size_t)1, dimK) * matAElemSz);
            if (matA == NULL) {
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }
            matB = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                std::max((size_t)1, dimN) * std::max((size_t)1, dimK) * matBElemSz);
            if (matB == NULL) {
                std::cerr<<"memory allocation failed"<<std::endl;
                free(matA);
                return;
            }
            matC = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                std::max((size_t)1, dimM) * std::max((size_t)1, dimN) * matCElemSz);
            if (matC == NULL) {
                std::cerr<<"memory allocation failed"<<std::endl;
                free(matA);
                free(matB);
                return;
            }
            matBias = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                std::max((size_t)1, dimBiasM) * std::max((size_t)1, dimBiasN) * biasElemSz);
            if (matBias == NULL) {
                std::cerr<<"memory allocation failed"<<std::endl;
                free(matA);
                free(matB);
                free(matC);
                return;
            }
            gemmLayer.Run(matA, matB, matC, matBias);
            KuDNN::Detail::ClearImpls();
            free(matA);
            free(matB);
            free(matC);
            free(matBias);
        } catch (const KuDNN::Service::Unsupported& e) {
            if (matA != NULL) {
                free(matA);
            }
            if (matB != NULL) {
                free(matB);
            }
            if (matC != NULL) {
                free(matC);
            }
            if (matBias != NULL) {
                free(matBias);
            }
            printf("Unsupported Configuration");
        } catch (const KuDNN::Service::LogicError& e) {
            if (matA != NULL) {
                free(matA);
            }
            if (matB != NULL) {
                free(matB);
            }
            if (matC != NULL) {
                free(matC);
            }
            if (matBias != NULL) {
                free(matBias);
            }
            printf("Bad shapes' arguments");
        }
        printf("\n");
    }
    DT_FUZZ_END()
}

static void DtFuzzMatmulIntCase(int counts)
{
    char funcName[] = "DtFuzzMatmulIntCase";
    DT_FUZZ_START(0, counts, funcName, 0)
    {
        int eleIndex = 0;
        int matAElemTypeIdx = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, INDEX1);
        int matBElemTypeIdx = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], INDEX1, 0, INDEX1);
        int matCElemTypeIdx = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], INDEX2, INDEX2, INDEX2);
        int biasElemTypeIdx = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_intPrecisionMap.size() - 1);
        int matALayoutIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_layoutMap.size() - 1);
        int matBLayoutIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_layoutMap.size() - 1);
        int matCLayoutIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_layoutMap.size() - 1);
        int matBiasLayoutIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_layoutMap.size() - 1);
        int broadCastBiasIdx = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0,
            g_broadCastBiasMap.size() - 1);

        size_t dim0 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dim1 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dim2 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dimM = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dimN = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dimK = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dimBiasM = dimM;
        size_t dimBiasN = dimN;

        // broadcast type
        switch (broadCastBiasIdx) {
            case BROADCAST_BIAS_M_INDEX:
                dimBiasM = 1;
                break;
            case BROADCAST_BIAS_N_INDEX:
                dimBiasN = 1;
                break;
            case BROADCAST_BIAS_MN_INDEX:
                dimBiasM = 1;
                dimBiasN = 1;
                break;
            default:
                break;
        }
        KuDNN::Element::Type matAType(g_intPrecisionMap[matAElemTypeIdx].second);
        KuDNN::Element::Type matBType(g_intPrecisionMap[matBElemTypeIdx].second);
        KuDNN::Element::Type matCType(g_intPrecisionMap[matCElemTypeIdx].second);
        KuDNN::Element::Type biasType(g_intPrecisionMap[biasElemTypeIdx].second);

        size_t matAElemSz = matAType.GetSize();
        size_t matBElemSz = matBType.GetSize();
        size_t matCElemSz = matCType.GetSize();
        size_t biasElemSz = biasType.GetSize();

        const KuDNN::TensorInfo matATensorInfo = {{dim0, dim1, dim2, dimM, dimK}, matAType,
            g_layoutMap[matALayoutIndex].second};
        const KuDNN::TensorInfo matBTensorInfo = {{dim0, dim1, dim2, dimK, dimN}, matBType,
            g_layoutMap[matBLayoutIndex].second};
        const KuDNN::TensorInfo matCTensorInfo = {{dim0, dim1, dim2, dimM, dimN}, matCType,
            g_layoutMap[matCLayoutIndex].second};
        const KuDNN::TensorInfo biasTensorInfo = {{dim0, dim1, dim2, dimBiasM, dimBiasN}, biasType,
            g_layoutMap[matBiasLayoutIndex].second};

        char format[] = "Matmul, Precision(A:B:C:bias): %s:%s:%s:%s, Shape: %zux%zux%zux%zux%zu X "
        "%zux%zux%zux%zux%zu = %zux%zux%zux%zux%zu + %zux%zux%zux%zux%zu, layout(A:B:C:bias): %s:%s:%s:%s ";
        printf(format, g_intPrecisionMap[matAElemTypeIdx].first.c_str(), g_intPrecisionMap[matBElemTypeIdx].first.c_str(),
            g_intPrecisionMap[matCElemTypeIdx].first.c_str(), g_intPrecisionMap[biasElemTypeIdx].first.c_str(),
            dim0, dim1, dim2, dimM, dimK, dim0, dim1, dim2, dimK, dimN, dim0, dim1, dim2, dimM, dimN,
            dim0, dim1, dim2, dimM, dimN, g_layoutMap[matALayoutIndex].first.c_str(),
            g_layoutMap[matBLayoutIndex].first.c_str(), g_layoutMap[matCLayoutIndex].first.c_str(),
            g_layoutMap[matBiasLayoutIndex].first.c_str());
        void *matA = NULL;
        void *matB = NULL;
        void *matC = NULL;
        void *matBias = NULL;
        try {
            KuDNN::Gemm gemmLayer(matATensorInfo, matBTensorInfo, matCTensorInfo,
                biasTensorInfo);
            matA = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                std::max((size_t)1, dimM) * std::max((size_t)1, dimK) * matAElemSz);
            if (matA == NULL) {
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }
            matB = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                std::max((size_t)1, dimN) * std::max((size_t)1, dimK) * matBElemSz);
            if (matB == NULL) {
                std::cerr<<"memory allocation failed"<<std::endl;
                free(matA);
                return;
            }
            matC = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                std::max((size_t)1, dimM) * std::max((size_t)1, dimN) * matCElemSz);
            if (matC == NULL) {
                std::cerr<<"memory allocation failed"<<std::endl;
                free(matA);
                free(matB);
                return;
            }
            matBias = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                std::max((size_t)1, dimBiasM) * std::max((size_t)1, dimBiasN) * biasElemSz);
            if (matBias == NULL) {
                std::cerr<<"memory allocation failed"<<std::endl;
                free(matA);
                free(matB);
                free(matC);
                return;
            }
            gemmLayer.Run(matA, matB, matC, matBias);
            KuDNN::Detail::ClearImpls();
            free(matA);
            free(matB);
            free(matC);
            free(matBias);
        } catch (const KuDNN::Service::Unsupported& e) {
            if (matA != NULL) {
                free(matA);
            }
            if (matB != NULL) {
                free(matB);
            }
            if (matC != NULL) {
                free(matC);
            }
            if (matBias != NULL) {
                free(matBias);
            }
            printf("Unsupported Configuration");
        } catch (const KuDNN::Service::LogicError& e) {
            if (matA != NULL) {
                free(matA);
            }
            if (matB != NULL) {
                free(matB);
            }
            if (matC != NULL) {
                free(matC);
            }
            if (matBias != NULL) {
                free(matBias);
            }
            printf("Bad shapes' arguments");
        }
        printf("\n");
    }
    DT_FUZZ_END()
}

// should set env of KUDNN_FORCE_NEON=1 before use and unset KUDNN_FORCE_NEON after use
static void DtFuzzMatmulNEONF32Case(int counts)
{
    char funcName[] = "DtFuzzMatmulNEONF32Case";
    DT_FUZZ_START(0, counts, funcName, 0)
    {
        int eleIndex = 0;
        int matAElemTypeIdx = 0;
        int matBElemTypeIdx = 0;
        int matCElemTypeIdx = 0;
        int biasElemTypeIdx = 0;
        int matALayoutIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_layoutMap.size() - 1);
        int matBLayoutIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_layoutMap.size() - 1);
        int matCLayoutIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_layoutMap.size() - 1);
        int matBiasLayoutIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_layoutMap.size() - 1);
        int broadCastBiasIdx = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0,
            g_broadCastBiasMap.size() - 1);

        size_t dim0 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dim1 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dim2 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dimM = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dimN = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dimK = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dimBiasM = dimM;
        size_t dimBiasN = dimN;

        // broadcast type
        switch (broadCastBiasIdx) {
            case BROADCAST_BIAS_M_INDEX:
                dimBiasM = 1;
                break;
            case BROADCAST_BIAS_N_INDEX:
                dimBiasN = 1;
                break;
            case BROADCAST_BIAS_MN_INDEX:
                dimBiasM = 1;
                dimBiasN = 1;
                break;
            default:
                break;
        }
        KuDNN::Element::Type matAType(g_fltPrecisionMap[matAElemTypeIdx].second);
        KuDNN::Element::Type matBType(g_fltPrecisionMap[matBElemTypeIdx].second);
        KuDNN::Element::Type matCType(g_fltPrecisionMap[matCElemTypeIdx].second);
        KuDNN::Element::Type biasType(g_fltPrecisionMap[biasElemTypeIdx].second);

        size_t matAElemSz = matAType.GetSize();
        size_t matBElemSz = matBType.GetSize();
        size_t matCElemSz = matCType.GetSize();
        size_t biasElemSz = biasType.GetSize();

        const KuDNN::TensorInfo matATensorInfo = {{dim0, dim1, dim2, dimM, dimK}, matAType,
            g_layoutMap[matALayoutIndex].second};
        const KuDNN::TensorInfo matBTensorInfo = {{dim0, dim1, dim2, dimK, dimN}, matBType,
            g_layoutMap[matBLayoutIndex].second};
        const KuDNN::TensorInfo matCTensorInfo = {{dim0, dim1, dim2, dimM, dimN}, matCType,
            g_layoutMap[matCLayoutIndex].second};
        const KuDNN::TensorInfo biasTensorInfo = {{dim0, dim1, dim2, dimBiasM, dimBiasN}, biasType,
            g_layoutMap[matBiasLayoutIndex].second};

        char format[] = "Matmul_NEON, Precision(A:B:C:bias): %s:%s:%s:%s, Shape: %zux%zux%zux%zux%zu X "
        "%zux%zux%zux%zux%zu = %zux%zux%zux%zux%zu + %zux%zux%zux%zux%zu, layout(A:B:C:bias): %s:%s:%s:%s ";
        printf(format, g_fltPrecisionMap[matAElemTypeIdx].first.c_str(), g_fltPrecisionMap[matBElemTypeIdx].first.c_str(),
            g_fltPrecisionMap[matCElemTypeIdx].first.c_str(), g_fltPrecisionMap[biasElemTypeIdx].first.c_str(),
            dim0, dim1, dim2, dimM, dimK, dim0, dim1, dim2, dimK, dimN, dim0, dim1, dim2, dimM, dimN,
            dim0, dim1, dim2, dimM, dimN, g_layoutMap[matALayoutIndex].first.c_str(),
            g_layoutMap[matBLayoutIndex].first.c_str(), g_layoutMap[matCLayoutIndex].first.c_str(),
            g_layoutMap[matBiasLayoutIndex].first.c_str());
        void *matA = NULL;
        void *matB = NULL;
        void *matC = NULL;
        void *matBias = NULL;
        try {
            KuDNN::Gemm gemmLayer(matATensorInfo, matBTensorInfo, matCTensorInfo, biasTensorInfo);
            matA = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                std::max((size_t)1, dimM) * std::max((size_t)1, dimK) * matAElemSz);
            if (matA == NULL) {
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }
            matB = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                std::max((size_t)1, dimN) * std::max((size_t)1, dimK) * matBElemSz);
            if (matB == NULL) {
                std::cerr<<"memory allocation failed"<<std::endl;
                free(matA);
                return;
            }
            matC = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                std::max((size_t)1, dimM) * std::max((size_t)1, dimN) * matCElemSz);
            if (matC == NULL) {
                std::cerr<<"memory allocation failed"<<std::endl;
                free(matA);
                free(matB);
                return;
            }
            matBias = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                std::max((size_t)1, dimBiasM) * std::max((size_t)1, dimBiasN) * biasElemSz);
            if (matBias == NULL) {
                std::cerr<<"memory allocation failed"<<std::endl;
                free(matA);
                free(matB);
                free(matC);
                return;
            }
            gemmLayer.Run(matA, matB, matC, matBias);
            KuDNN::Detail::ClearImpls();
            free(matA);
            free(matB);
            free(matC);
            free(matBias);
        } catch (const KuDNN::Service::Unsupported& e) {
            if (matA != NULL) {
                free(matA);
            }
            if (matB != NULL) {
                free(matB);
            }
            if (matC != NULL) {
                free(matC);
            }
            if (matBias != NULL) {
                free(matBias);
            }
            printf("Unsupported Configuration");
        } catch (const KuDNN::Service::LogicError& e) {
            if (matA != NULL) {
                free(matA);
            }
            if (matB != NULL) {
                free(matB);
            }
            if (matC != NULL) {
                free(matC);
            }
            if (matBias != NULL) {
                free(matBias);
            }
            printf("Bad shapes' arguments");
        }
        printf("\n");
    }
    DT_FUZZ_END()
}

static void DtFuzzMatmulPostOpsSumFltCase(int counts)
{
    char funcName[] = "DtFuzzMatmulPostOpsSumFltCase";
    DT_FUZZ_START(0, counts, funcName, 0)
    {
        int eleIndex = 0;
        int matAElemTypeIdx = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_fltPrecisionMap.size() - 1);
        int matBElemTypeIdx = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_fltPrecisionMap.size() - 1);
        int matCElemTypeIdx = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_fltPrecisionMap.size() - 1);
        int biasElemTypeIdx = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_fltPrecisionMap.size() - 1);
        int matALayoutIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_layoutMap.size() - 1);
        int matBLayoutIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_layoutMap.size() - 1);
        int matCLayoutIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_layoutMap.size() - 1);
        int matBiasLayoutIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_layoutMap.size() - 1);
        int broadCastBiasIdx = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0,
            g_broadCastBiasMap.size() - 1);

        size_t dim0 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dim1 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dim2 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dimM = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dimN = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dimK = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dimBiasM = dimM;
        size_t dimBiasN = dimN;

        // broadcast type
        switch (broadCastBiasIdx) {
            case BROADCAST_BIAS_M_INDEX:
                dimBiasM = 1;
                break;
            case BROADCAST_BIAS_N_INDEX:
                dimBiasN = 1;
                break;
            case BROADCAST_BIAS_MN_INDEX:
                dimBiasM = 1;
                dimBiasN = 1;
                break;
            default:
                break;
        }
        KuDNN::Element::Type matAType(g_fltPrecisionMap[matAElemTypeIdx].second);
        KuDNN::Element::Type matBType(g_fltPrecisionMap[matBElemTypeIdx].second);
        KuDNN::Element::Type matCType(g_fltPrecisionMap[matCElemTypeIdx].second);
        KuDNN::Element::Type biasType(g_fltPrecisionMap[biasElemTypeIdx].second);

        size_t matAElemSz = matAType.GetSize();
        size_t matBElemSz = matBType.GetSize();
        size_t matCElemSz = matCType.GetSize();
        size_t biasElemSz = biasType.GetSize();

        const KuDNN::TensorInfo matATensorInfo = {{dim0, dim1, dim2, dimM, dimK}, matAType,
            g_layoutMap[matALayoutIndex].second};
        const KuDNN::TensorInfo matBTensorInfo = {{dim0, dim1, dim2, dimK, dimN}, matBType,
            g_layoutMap[matBLayoutIndex].second};
        const KuDNN::TensorInfo matCTensorInfo = {{dim0, dim1, dim2, dimM, dimN}, matCType,
            g_layoutMap[matCLayoutIndex].second};
        const KuDNN::TensorInfo biasTensorInfo = {{dim0, dim1, dim2, dimBiasM, dimBiasN}, biasType,
            g_layoutMap[matBiasLayoutIndex].second};

        float alpha = 1.0f;
        float beta = 2.0f;

        KuDNN::PostOps post_ops;
        KuDNN::Attributes attributesInfo;

        post_ops.AppendSum(beta);
        attributesInfo.SetPostOps(post_ops);
        KuDNN::PostOpsDataPtrs poPtrs = {nullptr};

        char format[] = "Matmul, Precision(A:B:C:bias:post-ops): %s:%s:%s:%s, Shape: %zux%zux%zux%zux%zu X "
        "%zux%zux%zux%zux%zu = %zux%zux%zux%zux%zu + %zux%zux%zux%zux%zu, layout(A:B:C:bias): %s:%s:%s:%s, Post-Ops: sum: %.2f ";
        printf(format, g_fltPrecisionMap[matAElemTypeIdx].first.c_str(), g_fltPrecisionMap[matBElemTypeIdx].first.c_str(),
            g_fltPrecisionMap[matCElemTypeIdx].first.c_str(), g_fltPrecisionMap[biasElemTypeIdx].first.c_str(),
            dim0, dim1, dim2, dimM, dimK, dim0, dim1, dim2, dimK, dimN, dim0, dim1, dim2, dimM, dimN,
            dim0, dim1, dim2, dimM, dimN, g_layoutMap[matALayoutIndex].first.c_str(),
            g_layoutMap[matBLayoutIndex].first.c_str(), g_layoutMap[matCLayoutIndex].first.c_str(),
            g_layoutMap[matBiasLayoutIndex].first.c_str(), beta);
        void *matA = NULL;
        void *matB = NULL;
        void *matC = NULL;
        void *matBias = NULL;
        try {
            KuDNN::Gemm gemmLayer(matATensorInfo, matBTensorInfo, matCTensorInfo, biasTensorInfo, attributesInfo);
            matA = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                std::max((size_t)1, dimM) * std::max((size_t)1, dimK) * matAElemSz);
            if (matA == NULL) {
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }
            matB = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                std::max((size_t)1, dimN) * std::max((size_t)1, dimK) * matBElemSz);
            if (matB == NULL) {
                std::cerr<<"memory allocation failed"<<std::endl;
                free(matA);
                return;
            }
            matC = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                std::max((size_t)1, dimM) * std::max((size_t)1, dimN) * matCElemSz);
            if (matC == NULL) {
                std::cerr<<"memory allocation failed"<<std::endl;
                free(matA);
                free(matB);
                return;
            }
            matBias = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                std::max((size_t)1, dimBiasM) * std::max((size_t)1, dimBiasN) * biasElemSz);
            if (matBias == NULL) {
                std::cerr<<"memory allocation failed"<<std::endl;
                free(matA);
                free(matB);
                free(matC);
                return;
            }
            gemmLayer.Run(matA, matB, matC, matBias, poPtrs, alpha, beta);
            KuDNN::Detail::ClearImpls();
            free(matA);
            free(matB);
            free(matC);
            free(matBias);
        } catch (const KuDNN::Service::Unsupported& e) {
            if (matA != NULL) {
                free(matA);
            }
            if (matB != NULL) {
                free(matB);
            }
            if (matC != NULL) {
                free(matC);
            }
            if (matBias != NULL) {
                free(matBias);
            }
            printf("Unsupported Configuration");
        } catch (const KuDNN::Service::LogicError& e) {
            if (matA != NULL) {
                free(matA);
            }
            if (matB != NULL) {
                free(matB);
            }
            if (matC != NULL) {
                free(matC);
            }
            if (matBias != NULL) {
                free(matBias);
            }
            printf("Bad shapes' arguments");
        }
        printf("\n");
    }
    DT_FUZZ_END()
}

static void DtFuzzMatmulPostOpsSumIntCase(int counts)
{
    char funcName[] = "DtFuzzMatmulPostOpsSumIntCase";
    DT_FUZZ_START(0, counts, funcName, 0)
    {
        int eleIndex = 0;
        int matAElemTypeIdx = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, INDEX1);
        int matBElemTypeIdx = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], INDEX1, 0, INDEX1);
        int matCElemTypeIdx = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], INDEX2, INDEX2, INDEX2);
        int biasElemTypeIdx = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_intPrecisionMap.size() - 1);
        int matALayoutIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_layoutMap.size() - 1);
        int matBLayoutIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_layoutMap.size() - 1);
        int matCLayoutIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_layoutMap.size() - 1);
        int matBiasLayoutIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_layoutMap.size() - 1);
        int broadCastBiasIdx = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0,
            g_broadCastBiasMap.size() - 1);

        size_t dim0 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dim1 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dim2 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dimM = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dimN = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dimK = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dimBiasM = dimM;
        size_t dimBiasN = dimN;

        // broadcast type
        switch (broadCastBiasIdx) {
            case BROADCAST_BIAS_M_INDEX:
                dimBiasM = 1;
                break;
            case BROADCAST_BIAS_N_INDEX:
                dimBiasN = 1;
                break;
            case BROADCAST_BIAS_MN_INDEX:
                dimBiasM = 1;
                dimBiasN = 1;
                break;
            default:
                break;
        }
        KuDNN::Element::Type matAType(g_intPrecisionMap[matAElemTypeIdx].second);
        KuDNN::Element::Type matBType(g_intPrecisionMap[matBElemTypeIdx].second);
        KuDNN::Element::Type matCType(g_intPrecisionMap[matCElemTypeIdx].second);
        KuDNN::Element::Type biasType(g_intPrecisionMap[biasElemTypeIdx].second);

        size_t matAElemSz = matAType.GetSize();
        size_t matBElemSz = matBType.GetSize();
        size_t matCElemSz = matCType.GetSize();
        size_t biasElemSz = biasType.GetSize();

        const KuDNN::TensorInfo matATensorInfo = {{dim0, dim1, dim2, dimM, dimK}, matAType,
            g_layoutMap[matALayoutIndex].second};
        const KuDNN::TensorInfo matBTensorInfo = {{dim0, dim1, dim2, dimK, dimN}, matBType,
            g_layoutMap[matBLayoutIndex].second};
        const KuDNN::TensorInfo matCTensorInfo = {{dim0, dim1, dim2, dimM, dimN}, matCType,
            g_layoutMap[matCLayoutIndex].second};
        const KuDNN::TensorInfo biasTensorInfo = {{dim0, dim1, dim2, dimBiasM, dimBiasN}, biasType,
            g_layoutMap[matBiasLayoutIndex].second};

        float alpha = 1.0f;
        float beta = 2.0f;

        KuDNN::PostOps post_ops;
        KuDNN::Attributes attributesInfo;

        post_ops.AppendSum(beta);
        attributesInfo.SetPostOps(post_ops);
        KuDNN::PostOpsDataPtrs poPtrs = {nullptr};

        char format[] = "Matmul, Precision(A:B:C:bias:post-ops): %s:%s:%s:%s, Shape: %zux%zux%zux%zux%zu X "
        "%zux%zux%zux%zux%zu = %zux%zux%zux%zux%zu + %zux%zux%zux%zux%zu, layout(A:B:C:bias): %s:%s:%s:%s, Post-Ops: sum: %.2f ";
        printf(format, g_intPrecisionMap[matAElemTypeIdx].first.c_str(), g_intPrecisionMap[matBElemTypeIdx].first.c_str(),
            g_intPrecisionMap[matCElemTypeIdx].first.c_str(), g_intPrecisionMap[biasElemTypeIdx].first.c_str(),
            dim0, dim1, dim2, dimM, dimK, dim0, dim1, dim2, dimK, dimN, dim0, dim1, dim2, dimM, dimN,
            dim0, dim1, dim2, dimM, dimN, g_layoutMap[matALayoutIndex].first.c_str(),
            g_layoutMap[matBLayoutIndex].first.c_str(), g_layoutMap[matCLayoutIndex].first.c_str(),
            g_layoutMap[matBiasLayoutIndex].first.c_str(), beta);
        void *matA = NULL;
        void *matB = NULL;
        void *matC = NULL;
        void *matBias = NULL;
        try {
            KuDNN::Gemm gemmLayer(matATensorInfo, matBTensorInfo, matCTensorInfo,
                biasTensorInfo, attributesInfo);
            matA = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                std::max((size_t)1, dimM) * std::max((size_t)1, dimK) * matAElemSz);
            if (matA == NULL) {
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }
            matB = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                std::max((size_t)1, dimN) * std::max((size_t)1, dimK) * matBElemSz);
            if (matB == NULL) {
                std::cerr<<"memory allocation failed"<<std::endl;
                free(matA);
                return;
            }
            matC = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                std::max((size_t)1, dimM) * std::max((size_t)1, dimN) * matCElemSz);
            if (matC == NULL) {
                std::cerr<<"memory allocation failed"<<std::endl;
                free(matA);
                free(matB);
                return;
            }
            matBias = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                std::max((size_t)1, dimBiasM) * std::max((size_t)1, dimBiasN) * biasElemSz);
            if (matBias == NULL) {
                std::cerr<<"memory allocation failed"<<std::endl;
                free(matA);
                free(matB);
                free(matC);
                return;
            }
            gemmLayer.Run(matA, matB, matC, matBias, poPtrs, alpha, beta);
            KuDNN::Detail::ClearImpls();
            free(matA);
            free(matB);
            free(matC);
            free(matBias);
        } catch (const KuDNN::Service::Unsupported& e) {
            if (matA != NULL) {
                free(matA);
            }
            if (matB != NULL) {
                free(matB);
            }
            if (matC != NULL) {
                free(matC);
            }
            if (matBias != NULL) {
                free(matBias);
            }
            printf("Unsupported Configuration");
        } catch (const KuDNN::Service::LogicError& e) {
            if (matA != NULL) {
                free(matA);
            }
            if (matB != NULL) {
                free(matB);
            }
            if (matC != NULL) {
                free(matC);
            }
            if (matBias != NULL) {
                free(matBias);
            }
            printf("Bad shapes' arguments");
        }
        printf("\n");
    }
    DT_FUZZ_END()
}

// should set env of KUDNN_FORCE_NEON=1 before use and unset KUDNN_FORCE_NEON after use
static void DtFuzzMatmulPostOpsSumNEONF32Case(int counts)
{
    char funcName[] = "DtFuzzMatmulPostOpsSumNEONF32Case";
    DT_FUZZ_START(0, counts, funcName, 0)
    {
        int eleIndex = 0;
        int matAElemTypeIdx = 0;
        int matBElemTypeIdx = 0;
        int matCElemTypeIdx = 0;
        int biasElemTypeIdx = 0;
        int matALayoutIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_layoutMap.size() - 1);
        int matBLayoutIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_layoutMap.size() - 1);
        int matCLayoutIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_layoutMap.size() - 1);
        int matBiasLayoutIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_layoutMap.size() - 1);
        int broadCastBiasIdx = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0,
            g_broadCastBiasMap.size() - 1);

        size_t dim0 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dim1 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dim2 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dimM = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dimN = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dimK = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dimBiasM = dimM;
        size_t dimBiasN = dimN;

        // broadcast type
        switch (broadCastBiasIdx) {
            case BROADCAST_BIAS_M_INDEX:
                dimBiasM = 1;
                break;
            case BROADCAST_BIAS_N_INDEX:
                dimBiasN = 1;
                break;
            case BROADCAST_BIAS_MN_INDEX:
                dimBiasM = 1;
                dimBiasN = 1;
                break;
            default:
                break;
        }
        KuDNN::Element::Type matAType(g_fltPrecisionMap[matAElemTypeIdx].second);
        KuDNN::Element::Type matBType(g_fltPrecisionMap[matBElemTypeIdx].second);
        KuDNN::Element::Type matCType(g_fltPrecisionMap[matCElemTypeIdx].second);
        KuDNN::Element::Type biasType(g_fltPrecisionMap[biasElemTypeIdx].second);

        size_t matAElemSz = matAType.GetSize();
        size_t matBElemSz = matBType.GetSize();
        size_t matCElemSz = matCType.GetSize();
        size_t biasElemSz = biasType.GetSize();

        const KuDNN::TensorInfo matATensorInfo = {{dim0, dim1, dim2, dimM, dimK}, matAType,
            g_layoutMap[matALayoutIndex].second};
        const KuDNN::TensorInfo matBTensorInfo = {{dim0, dim1, dim2, dimK, dimN}, matBType,
            g_layoutMap[matBLayoutIndex].second};
        const KuDNN::TensorInfo matCTensorInfo = {{dim0, dim1, dim2, dimM, dimN}, matCType,
            g_layoutMap[matCLayoutIndex].second};
        const KuDNN::TensorInfo biasTensorInfo = {{dim0, dim1, dim2, dimBiasM, dimBiasN}, biasType,
            g_layoutMap[matBiasLayoutIndex].second};

        float alpha = 1.0f;
        float beta = 2.0f;

        KuDNN::PostOps post_ops;
        KuDNN::Attributes attributesInfo;

        post_ops.AppendSum(beta);
        attributesInfo.SetPostOps(post_ops);
        KuDNN::PostOpsDataPtrs poPtrs = {nullptr};

        char format[] = "Matmul_NEON, Precision(A:B:C:bias): %s:%s:%s:%s, Shape: %zux%zux%zux%zux%zu X "
        "%zux%zux%zux%zux%zu = %zux%zux%zux%zux%zu + %zux%zux%zux%zux%zu, layout(A:B:C:bias): %s:%s:%s:%s, Post-Ops: sum: %.2f ";
        printf(format, g_fltPrecisionMap[matAElemTypeIdx].first.c_str(), g_fltPrecisionMap[matBElemTypeIdx].first.c_str(),
            g_fltPrecisionMap[matCElemTypeIdx].first.c_str(), g_fltPrecisionMap[biasElemTypeIdx].first.c_str(),
            dim0, dim1, dim2, dimM, dimK, dim0, dim1, dim2, dimK, dimN, dim0, dim1, dim2, dimM, dimN,
            dim0, dim1, dim2, dimM, dimN, g_layoutMap[matALayoutIndex].first.c_str(),
            g_layoutMap[matBLayoutIndex].first.c_str(), g_layoutMap[matCLayoutIndex].first.c_str(),
            g_layoutMap[matBiasLayoutIndex].first.c_str(), beta);
        void *matA = NULL;
        void *matB = NULL;
        void *matC = NULL;
        void *matBias = NULL;
        try {
            KuDNN::Gemm gemmLayer(matATensorInfo, matBTensorInfo, matCTensorInfo, biasTensorInfo);
            matA = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                std::max((size_t)1, dimM) * std::max((size_t)1, dimK) * matAElemSz);
            if (matA == NULL) {
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }
            matB = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                std::max((size_t)1, dimN) * std::max((size_t)1, dimK) * matBElemSz);
            if (matB == NULL) {
                std::cerr<<"memory allocation failed"<<std::endl;
                free(matA);
                return;
            }
            matC = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                std::max((size_t)1, dimM) * std::max((size_t)1, dimN) * matCElemSz);
            if (matC == NULL) {
                std::cerr<<"memory allocation failed"<<std::endl;
                free(matA);
                free(matB);
                return;
            }
            matBias = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                std::max((size_t)1, dimBiasM) * std::max((size_t)1, dimBiasN) * biasElemSz);
            if (matBias == NULL) {
                std::cerr<<"memory allocation failed"<<std::endl;
                free(matA);
                free(matB);
                free(matC);
                return;
            }
            gemmLayer.Run(matA, matB, matC, matBias, poPtrs, alpha, beta);
            KuDNN::Detail::ClearImpls();
            free(matA);
            free(matB);
            free(matC);
            free(matBias);
        } catch (const KuDNN::Service::Unsupported& e) {
            if (matA != NULL) {
                free(matA);
            }
            if (matB != NULL) {
                free(matB);
            }
            if (matC != NULL) {
                free(matC);
            }
            if (matBias != NULL) {
                free(matBias);
            }
            printf("Unsupported Configuration");
        } catch (const KuDNN::Service::LogicError& e) {
            if (matA != NULL) {
                free(matA);
            }
            if (matB != NULL) {
                free(matB);
            }
            if (matC != NULL) {
                free(matC);
            }
            if (matBias != NULL) {
                free(matBias);
            }
            printf("Bad shapes' arguments");
        }
        printf("\n");
    }
    DT_FUZZ_END()
}

static void DtFuzzMatmulPostOpsReluFltCase(int counts)
{
    char funcName[] = "DtFuzzMatmulPostOpsReluFltCase";
    DT_FUZZ_START(0, counts, funcName, 0)
    {
        int eleIndex = 0;
        int matAElemTypeIdx = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_fltPrecisionMap.size() - 1);
        int matBElemTypeIdx = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_fltPrecisionMap.size() - 1);
        int matCElemTypeIdx = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_fltPrecisionMap.size() - 1);
        int biasElemTypeIdx = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_fltPrecisionMap.size() - 1);
        int matALayoutIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_layoutMap.size() - 1);
        int matBLayoutIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_layoutMap.size() - 1);
        int matCLayoutIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_layoutMap.size() - 1);
        int matBiasLayoutIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_layoutMap.size() - 1);
        int broadCastBiasIdx = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0,
            g_broadCastBiasMap.size() - 1);

        size_t dim0 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dim1 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dim2 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dimM = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dimN = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dimK = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dimBiasM = dimM;
        size_t dimBiasN = dimN;

        // broadcast type
        switch (broadCastBiasIdx) {
            case BROADCAST_BIAS_M_INDEX:
                dimBiasM = 1;
                break;
            case BROADCAST_BIAS_N_INDEX:
                dimBiasN = 1;
                break;
            case BROADCAST_BIAS_MN_INDEX:
                dimBiasM = 1;
                dimBiasN = 1;
                break;
            default:
                break;
        }
        KuDNN::Element::Type matAType(g_fltPrecisionMap[matAElemTypeIdx].second);
        KuDNN::Element::Type matBType(g_fltPrecisionMap[matBElemTypeIdx].second);
        KuDNN::Element::Type matCType(g_fltPrecisionMap[matCElemTypeIdx].second);
        KuDNN::Element::Type biasType(g_fltPrecisionMap[biasElemTypeIdx].second);

        size_t matAElemSz = matAType.GetSize();
        size_t matBElemSz = matBType.GetSize();
        size_t matCElemSz = matCType.GetSize();
        size_t biasElemSz = biasType.GetSize();

        const KuDNN::TensorInfo matATensorInfo = {{dim0, dim1, dim2, dimM, dimK}, matAType,
            g_layoutMap[matALayoutIndex].second};
        const KuDNN::TensorInfo matBTensorInfo = {{dim0, dim1, dim2, dimK, dimN}, matBType,
            g_layoutMap[matBLayoutIndex].second};
        const KuDNN::TensorInfo matCTensorInfo = {{dim0, dim1, dim2, dimM, dimN}, matCType,
            g_layoutMap[matCLayoutIndex].second};
        const KuDNN::TensorInfo biasTensorInfo = {{dim0, dim1, dim2, dimBiasM, dimBiasN}, biasType,
            g_layoutMap[matBiasLayoutIndex].second};

        float alpha = 3.5f;
        float beta = 2.0f;

        KuDNN::PostOps post_ops;
        KuDNN::Attributes attributesInfo;

        post_ops.AppendEltwise(KuDNN::ActivationFunction::RELU, alpha, beta);
        attributesInfo.SetPostOps(post_ops);
        KuDNN::PostOpsDataPtrs poPtrs = {nullptr};

        char format[] = "Matmul, Precision(A:B:C:bias:post-ops): %s:%s:%s:%s, Shape: %zux%zux%zux%zux%zu X "
        "%zux%zux%zux%zux%zu = %zux%zux%zux%zux%zu + %zux%zux%zux%zux%zu, layout(A:B:C:bias): %s:%s:%s:%s, Post-Ops: relu: %.2f ";
        printf(format, g_fltPrecisionMap[matAElemTypeIdx].first.c_str(), g_fltPrecisionMap[matBElemTypeIdx].first.c_str(),
            g_fltPrecisionMap[matCElemTypeIdx].first.c_str(), g_fltPrecisionMap[biasElemTypeIdx].first.c_str(),
            dim0, dim1, dim2, dimM, dimK, dim0, dim1, dim2, dimK, dimN, dim0, dim1, dim2, dimM, dimN,
            dim0, dim1, dim2, dimM, dimN, g_layoutMap[matALayoutIndex].first.c_str(),
            g_layoutMap[matBLayoutIndex].first.c_str(), g_layoutMap[matCLayoutIndex].first.c_str(),
            g_layoutMap[matBiasLayoutIndex].first.c_str(), alpha);
        void *matA = NULL;
        void *matB = NULL;
        void *matC = NULL;
        void *matBias = NULL;
        try {
            KuDNN::Gemm gemmLayer(matATensorInfo, matBTensorInfo, matCTensorInfo, biasTensorInfo, attributesInfo);
            matA = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                std::max((size_t)1, dimM) * std::max((size_t)1, dimK) * matAElemSz);
            if (matA == NULL) {
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }
            matB = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                std::max((size_t)1, dimN) * std::max((size_t)1, dimK) * matBElemSz);
            if (matB == NULL) {
                std::cerr<<"memory allocation failed"<<std::endl;
                free(matA);
                return;
            }
            matC = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                std::max((size_t)1, dimM) * std::max((size_t)1, dimN) * matCElemSz);
            if (matC == NULL) {
                std::cerr<<"memory allocation failed"<<std::endl;
                free(matA);
                free(matB);
                return;
            }
            matBias = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                std::max((size_t)1, dimBiasM) * std::max((size_t)1, dimBiasN) * biasElemSz);
            if (matBias == NULL) {
                std::cerr<<"memory allocation failed"<<std::endl;
                free(matA);
                free(matB);
                free(matC);
                return;
            }
            gemmLayer.Run(matA, matB, matC, matBias, poPtrs, alpha, beta);
            KuDNN::Detail::ClearImpls();
            free(matA);
            free(matB);
            free(matC);
            free(matBias);
        } catch (const KuDNN::Service::Unsupported& e) {
            if (matA != NULL) {
                free(matA);
            }
            if (matB != NULL) {
                free(matB);
            }
            if (matC != NULL) {
                free(matC);
            }
            if (matBias != NULL) {
                free(matBias);
            }
            printf("Unsupported Configuration");
        } catch (const KuDNN::Service::LogicError& e) {
            if (matA != NULL) {
                free(matA);
            }
            if (matB != NULL) {
                free(matB);
            }
            if (matC != NULL) {
                free(matC);
            }
            if (matBias != NULL) {
                free(matBias);
            }
            printf("Bad shapes' arguments");
        }
        printf("\n");
    }
    DT_FUZZ_END()
}

static void DtFuzzMatmulPostOpsReluIntCase(int counts)
{
    char funcName[] = "DtFuzzMatmulPostOpsReluIntCase";
    DT_FUZZ_START(0, counts, funcName, 0)
    {
        int eleIndex = 0;
        int matAElemTypeIdx = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, INDEX1);
        int matBElemTypeIdx = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], INDEX1, 0, INDEX1);
        int matCElemTypeIdx = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], INDEX2, INDEX2, INDEX2);
        int biasElemTypeIdx = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_intPrecisionMap.size() - 1);
        int matALayoutIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_layoutMap.size() - 1);
        int matBLayoutIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_layoutMap.size() - 1);
        int matCLayoutIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_layoutMap.size() - 1);
        int matBiasLayoutIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_layoutMap.size() - 1);
        int broadCastBiasIdx = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0,
            g_broadCastBiasMap.size() - 1);

        size_t dim0 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dim1 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dim2 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dimM = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dimN = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dimK = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dimBiasM = dimM;
        size_t dimBiasN = dimN;

        // broadcast type
        switch (broadCastBiasIdx) {
            case BROADCAST_BIAS_M_INDEX:
                dimBiasM = 1;
                break;
            case BROADCAST_BIAS_N_INDEX:
                dimBiasN = 1;
                break;
            case BROADCAST_BIAS_MN_INDEX:
                dimBiasM = 1;
                dimBiasN = 1;
                break;
            default:
                break;
        }
        KuDNN::Element::Type matAType(g_intPrecisionMap[matAElemTypeIdx].second);
        KuDNN::Element::Type matBType(g_intPrecisionMap[matBElemTypeIdx].second);
        KuDNN::Element::Type matCType(g_intPrecisionMap[matCElemTypeIdx].second);
        KuDNN::Element::Type biasType(g_intPrecisionMap[biasElemTypeIdx].second);

        size_t matAElemSz = matAType.GetSize();
        size_t matBElemSz = matBType.GetSize();
        size_t matCElemSz = matCType.GetSize();
        size_t biasElemSz = biasType.GetSize();

        const KuDNN::TensorInfo matATensorInfo = {{dim0, dim1, dim2, dimM, dimK}, matAType,
            g_layoutMap[matALayoutIndex].second};
        const KuDNN::TensorInfo matBTensorInfo = {{dim0, dim1, dim2, dimK, dimN}, matBType,
            g_layoutMap[matBLayoutIndex].second};
        const KuDNN::TensorInfo matCTensorInfo = {{dim0, dim1, dim2, dimM, dimN}, matCType,
            g_layoutMap[matCLayoutIndex].second};
        const KuDNN::TensorInfo biasTensorInfo = {{dim0, dim1, dim2, dimBiasM, dimBiasN}, biasType,
            g_layoutMap[matBiasLayoutIndex].second};

        float alpha = 3.5f;
        float beta = 2.0f;

        KuDNN::PostOps post_ops;
        KuDNN::Attributes attributesInfo;

        post_ops.AppendEltwise(KuDNN::ActivationFunction::RELU, alpha, beta);
        attributesInfo.SetPostOps(post_ops);
        KuDNN::PostOpsDataPtrs poPtrs = {nullptr};

        char format[] = "Matmul, Precision(A:B:C:bias:post-ops): %s:%s:%s:%s, Shape: %zux%zux%zux%zux%zu X "
        "%zux%zux%zux%zux%zu = %zux%zux%zux%zux%zu + %zux%zux%zux%zux%zu, layout(A:B:C:bias): %s:%s:%s:%s, Post-Ops: relu: %.2f ";
        printf(format, g_intPrecisionMap[matAElemTypeIdx].first.c_str(), g_intPrecisionMap[matBElemTypeIdx].first.c_str(),
            g_intPrecisionMap[matCElemTypeIdx].first.c_str(), g_intPrecisionMap[biasElemTypeIdx].first.c_str(),
            dim0, dim1, dim2, dimM, dimK, dim0, dim1, dim2, dimK, dimN, dim0, dim1, dim2, dimM, dimN,
            dim0, dim1, dim2, dimM, dimN, g_layoutMap[matALayoutIndex].first.c_str(),
            g_layoutMap[matBLayoutIndex].first.c_str(), g_layoutMap[matCLayoutIndex].first.c_str(),
            g_layoutMap[matBiasLayoutIndex].first.c_str(), alpha);
        void *matA = NULL;
        void *matB = NULL;
        void *matC = NULL;
        void *matBias = NULL;
        try {
            KuDNN::Gemm gemmLayer(matATensorInfo, matBTensorInfo, matCTensorInfo,
                biasTensorInfo, attributesInfo);
            matA = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                std::max((size_t)1, dimM) * std::max((size_t)1, dimK) * matAElemSz);
            if (matA == NULL) {
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }
            matB = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                std::max((size_t)1, dimN) * std::max((size_t)1, dimK) * matBElemSz);
            if (matB == NULL) {
                std::cerr<<"memory allocation failed"<<std::endl;
                free(matA);
                return;
            }
            matC = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                std::max((size_t)1, dimM) * std::max((size_t)1, dimN) * matCElemSz);
            if (matC == NULL) {
                std::cerr<<"memory allocation failed"<<std::endl;
                free(matA);
                free(matB);
                return;
            }
            matBias = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                std::max((size_t)1, dimBiasM) * std::max((size_t)1, dimBiasN) * biasElemSz);
            if (matBias == NULL) {
                std::cerr<<"memory allocation failed"<<std::endl;
                free(matA);
                free(matB);
                free(matC);
                return;
            }
            gemmLayer.Run(matA, matB, matC, matBias, poPtrs, alpha, beta);
            KuDNN::Detail::ClearImpls();
            free(matA);
            free(matB);
            free(matC);
            free(matBias);
        } catch (const KuDNN::Service::Unsupported& e) {
            if (matA != NULL) {
                free(matA);
            }
            if (matB != NULL) {
                free(matB);
            }
            if (matC != NULL) {
                free(matC);
            }
            if (matBias != NULL) {
                free(matBias);
            }
            printf("Unsupported Configuration");
        } catch (const KuDNN::Service::LogicError& e) {
            if (matA != NULL) {
                free(matA);
            }
            if (matB != NULL) {
                free(matB);
            }
            if (matC != NULL) {
                free(matC);
            }
            if (matBias != NULL) {
                free(matBias);
            }
            printf("Bad shapes' arguments");
        }
        printf("\n");
    }
    DT_FUZZ_END()
}

// should set env of KUDNN_FORCE_NEON=1 before use and unset KUDNN_FORCE_NEON after use
static void DtFuzzMatmulPostOpsReluNEONF32Case(int counts)
{
    char funcName[] = "DtFuzzMatmulPostOpsReluNEONF32Case";
    DT_FUZZ_START(0, counts, funcName, 0)
    {
        int eleIndex = 0;
        int matAElemTypeIdx = 0;
        int matBElemTypeIdx = 0;
        int matCElemTypeIdx = 0;
        int biasElemTypeIdx = 0;
        int matALayoutIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_layoutMap.size() - 1);
        int matBLayoutIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_layoutMap.size() - 1);
        int matCLayoutIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_layoutMap.size() - 1);
        int matBiasLayoutIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_layoutMap.size() - 1);
        int broadCastBiasIdx = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0,
            g_broadCastBiasMap.size() - 1);

        size_t dim0 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dim1 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dim2 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dimM = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dimN = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dimK = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dimBiasM = dimM;
        size_t dimBiasN = dimN;

        // broadcast type
        switch (broadCastBiasIdx) {
            case BROADCAST_BIAS_M_INDEX:
                dimBiasM = 1;
                break;
            case BROADCAST_BIAS_N_INDEX:
                dimBiasN = 1;
                break;
            case BROADCAST_BIAS_MN_INDEX:
                dimBiasM = 1;
                dimBiasN = 1;
                break;
            default:
                break;
        }
        KuDNN::Element::Type matAType(g_fltPrecisionMap[matAElemTypeIdx].second);
        KuDNN::Element::Type matBType(g_fltPrecisionMap[matBElemTypeIdx].second);
        KuDNN::Element::Type matCType(g_fltPrecisionMap[matCElemTypeIdx].second);
        KuDNN::Element::Type biasType(g_fltPrecisionMap[biasElemTypeIdx].second);

        size_t matAElemSz = matAType.GetSize();
        size_t matBElemSz = matBType.GetSize();
        size_t matCElemSz = matCType.GetSize();
        size_t biasElemSz = biasType.GetSize();

        const KuDNN::TensorInfo matATensorInfo = {{dim0, dim1, dim2, dimM, dimK}, matAType,
            g_layoutMap[matALayoutIndex].second};
        const KuDNN::TensorInfo matBTensorInfo = {{dim0, dim1, dim2, dimK, dimN}, matBType,
            g_layoutMap[matBLayoutIndex].second};
        const KuDNN::TensorInfo matCTensorInfo = {{dim0, dim1, dim2, dimM, dimN}, matCType,
            g_layoutMap[matCLayoutIndex].second};
        const KuDNN::TensorInfo biasTensorInfo = {{dim0, dim1, dim2, dimBiasM, dimBiasN}, biasType,
            g_layoutMap[matBiasLayoutIndex].second};

        float alpha = 3.5f;
        float beta = 2.0f;

        KuDNN::PostOps post_ops;
        KuDNN::Attributes attributesInfo;

        post_ops.AppendEltwise(KuDNN::ActivationFunction::RELU, alpha, beta);
        attributesInfo.SetPostOps(post_ops);
        KuDNN::PostOpsDataPtrs poPtrs = {nullptr};

        char format[] = "Matmul_NEON, Precision(A:B:C:bias): %s:%s:%s:%s, Shape: %zux%zux%zux%zux%zu X "
        "%zux%zux%zux%zux%zu = %zux%zux%zux%zux%zu + %zux%zux%zux%zux%zu, layout(A:B:C:bias): %s:%s:%s:%s, Post-Ops: relu: %.2f ";
        printf(format, g_fltPrecisionMap[matAElemTypeIdx].first.c_str(), g_fltPrecisionMap[matBElemTypeIdx].first.c_str(),
            g_fltPrecisionMap[matCElemTypeIdx].first.c_str(), g_fltPrecisionMap[biasElemTypeIdx].first.c_str(),
            dim0, dim1, dim2, dimM, dimK, dim0, dim1, dim2, dimK, dimN, dim0, dim1, dim2, dimM, dimN,
            dim0, dim1, dim2, dimM, dimN, g_layoutMap[matALayoutIndex].first.c_str(),
            g_layoutMap[matBLayoutIndex].first.c_str(), g_layoutMap[matCLayoutIndex].first.c_str(),
            g_layoutMap[matBiasLayoutIndex].first.c_str(), alpha);
        void *matA = NULL;
        void *matB = NULL;
        void *matC = NULL;
        void *matBias = NULL;
        try {
            KuDNN::Gemm gemmLayer(matATensorInfo, matBTensorInfo, matCTensorInfo, biasTensorInfo);
            matA = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                std::max((size_t)1, dimM) * std::max((size_t)1, dimK) * matAElemSz);
            if (matA == NULL) {
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }
            matB = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                std::max((size_t)1, dimN) * std::max((size_t)1, dimK) * matBElemSz);
            if (matB == NULL) {
                std::cerr<<"memory allocation failed"<<std::endl;
                free(matA);
                return;
            }
            matC = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                std::max((size_t)1, dimM) * std::max((size_t)1, dimN) * matCElemSz);
            if (matC == NULL) {
                std::cerr<<"memory allocation failed"<<std::endl;
                free(matA);
                free(matB);
                return;
            }
            matBias = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                std::max((size_t)1, dimBiasM) * std::max((size_t)1, dimBiasN) * biasElemSz);
            if (matBias == NULL) {
                std::cerr<<"memory allocation failed"<<std::endl;
                free(matA);
                free(matB);
                free(matC);
                return;
            }
            gemmLayer.Run(matA, matB, matC, matBias, poPtrs, alpha, beta);
            KuDNN::Detail::ClearImpls();
            free(matA);
            free(matB);
            free(matC);
            free(matBias);
        } catch (const KuDNN::Service::Unsupported& e) {
            if (matA != NULL) {
                free(matA);
            }
            if (matB != NULL) {
                free(matB);
            }
            if (matC != NULL) {
                free(matC);
            }
            if (matBias != NULL) {
                free(matBias);
            }
            printf("Unsupported Configuration");
        } catch (const KuDNN::Service::LogicError& e) {
            if (matA != NULL) {
                free(matA);
            }
            if (matB != NULL) {
                free(matB);
            }
            if (matC != NULL) {
                free(matC);
            }
            if (matBias != NULL) {
                free(matBias);
            }
            printf("Bad shapes' arguments");
        }
        printf("\n");
    }
    DT_FUZZ_END()
}

static Cmdinfo g_matmulCmdTable[] = {
    {"MatmulFltCase", DtFuzzMatmulFltCase, 20000}, // full random cases amount : 20000
    {"MatmulIntCase", DtFuzzMatmulIntCase, 20000}, // full random cases amount : 20000
    {"MatmulNEONF32Case", DtFuzzMatmulNEONF32Case, 5000}, // full random cases amount : 5000
    {"MatmulPostOpsSumFltCase", DtFuzzMatmulPostOpsSumFltCase, 5000}, // full random cases amount : 5000
    {"MatmulPostOpsSumIntCase", DtFuzzMatmulPostOpsSumIntCase, 5000}, // full random cases amount : 5000
    {"MatmulPostOpsSumNEONF32Case", DtFuzzMatmulPostOpsSumNEONF32Case, 5000}, // full random cases amount : 5000
    {"MatmulPostOpsReluFltCase", DtFuzzMatmulPostOpsReluFltCase, 5000}, // full random cases amount : 5000
    {"MatmulPostOpsReluIntCase", DtFuzzMatmulPostOpsReluIntCase, 5000}, // full random cases amount : 5000
    {"MatmulPostOpsReluNEONF32Case", DtFuzzMatmulPostOpsReluNEONF32Case, 5000}, // full random cases amount : 5000
};

int main(int argc, char *argv[])
{
    // fuzz test times control, ./matmul_fuzz 1000 (default: 20000)
    DT_Enable_Leak_Check(0, 0);
    if (argc > 1) {
        g_matmulCmdTable[0] = {"MatmulFltCase", DtFuzzMatmulFltCase, atoi(argv[1])};
        g_matmulCmdTable[1] = {"MatmulIntCase", DtFuzzMatmulIntCase, atoi(argv[1])};
        g_matmulCmdTable[2] = {"MatmulNEONF32Case", DtFuzzMatmulNEONF32Case, atoi(argv[1])};
        g_matmulCmdTable[3] = {"MatmulPostOpsSumFltCase", DtFuzzMatmulPostOpsSumFltCase, atoi(argv[1])};
        g_matmulCmdTable[4] = {"MatmulPostOpsIntSumCase", DtFuzzMatmulPostOpsSumIntCase, atoi(argv[1])};
        g_matmulCmdTable[5] = {"MatmulPostOpsSumNEONF32Case", DtFuzzMatmulPostOpsSumNEONF32Case, atoi(argv[1])};
        g_matmulCmdTable[6] = {"MatmulPostOpsReluFltCase", DtFuzzMatmulPostOpsReluFltCase, atoi(argv[1])};
        g_matmulCmdTable[7] = {"MatmulPostOpsReluIntCase", DtFuzzMatmulPostOpsReluIntCase, atoi(argv[1])};
        g_matmulCmdTable[8] = {"MatmulPostOpsReluNEONF32Case", DtFuzzMatmulPostOpsReluNEONF32Case, atoi(argv[1])};
    }

    int numOfTestItem = sizeof(g_matmulCmdTable) / sizeof(g_matmulCmdTable[0]);
    int testCount = 0;
    if (argc > 2) {
        const char* inputCmd = argv[2];
        for (int testCaseIdx = 0; testCaseIdx < numOfTestItem; testCaseIdx++) {
            if (strcmp(g_matmulCmdTable[testCaseIdx].cmd, inputCmd) == 0) {
                testCount = g_matmulCmdTable[testCaseIdx].testCount;
                printf("g_matmulCmdTable[%d] = %s test %d time\n", testCaseIdx, g_matmulCmdTable[testCaseIdx].cmd,
                    testCount);
                g_matmulCmdTable[testCaseIdx].func(testCount);
                break;
            }
        }
    } else {

        for (int testCaseIdx = 0; testCaseIdx < numOfTestItem; testCaseIdx++) {
            testCount = g_matmulCmdTable[testCaseIdx].testCount;
            printf("g_matmulCmdTable[%d] = %s test %d time\n", testCaseIdx, g_matmulCmdTable[testCaseIdx].cmd,
                testCount);
            g_matmulCmdTable[testCaseIdx].func(testCount);
        }
    }

    return 0;
}