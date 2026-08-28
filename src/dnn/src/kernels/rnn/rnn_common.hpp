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

#ifndef KUDNN_RNN_COMMON_HPP
#define KUDNN_RNN_COMMON_HPP

#include "service/kudnn_threading.hpp"
#include "internal_funcs.hpp"

namespace KuDNN {
namespace Detail {
namespace Rnn {

inline IntType NdIteratorInit(IntType start)
{
    return start;
}
template <typename... Args>
inline IntType NdIteratorInit(IntType start, IntType &index, const IntType &length, Args &&...tuple)
{
    start = NdIteratorInit(start, std::forward<Args>(tuple)...);
    index = start % length;
    return start / length;
}

inline bool NdIteratorStep()
{
    return true;
}
template <typename... Args>
inline bool NdIteratorStep(IntType &index, const IntType &length, Args &&...tuple)
{
    if (NdIteratorStep(std::forward<Args>(tuple)...)) {
        if (++index - length == 0) {
            index = 0;
            return true;
        }
    }
    return false;
}

inline std::vector<std::pair<IntType, IntType>> SplitLen(const IntType len, IntType &optNumThreads)
{
    optNumThreads = std::min(len, optNumThreads);
    IntType *splitLen = new IntType[optNumThreads + 1];

    IntType singleThreadLen = len / optNumThreads;
    IntType remainderLen = len - optNumThreads * singleThreadLen;

    splitLen[0] = 0;
    IntType i = 0;
    for (; i < remainderLen; i++) {
        splitLen[i + 1] = splitLen[i] + singleThreadLen + 1;
    }
    for (; i < optNumThreads; i++) {
        splitLen[i + 1] = splitLen[i] + singleThreadLen;
    }

    std::vector<std::pair<IntType, IntType>> out;
    out.reserve(optNumThreads);
    for (i = 0; i < optNumThreads; i++) {
        out.push_back({splitLen[i], splitLen[i + 1]});
    }

    delete[] splitLen;
    return out;
}

template <typename FuncPtr>
inline void MultiThreader(IntType num, FuncPtr func) noexcept
{
    IntType optNumThreads = std::min(int{NUM_80}, KuDNN::Threading::GetMaxNumThreads());
    std::vector<std::pair<IntType, IntType>> &&split = SplitLen(num, optNumThreads);
    const auto exec = [func](IntType start, IntType end) {
        for (IntType i = start; i < end; i++) {
            func(i);
        }
    };
    if (optNumThreads > 1) {
#pragma omp parallel for num_threads(optNumThreads)
        for (SizeType i = 0; i < split.size(); i++) {
            exec(split[i].first, split[i].second);
        }
    } else {
        for (SizeType i = 0; i < static_cast<SizeType>(num); ++i) {
            func(i);
        }
    }
}

template <typename FuncPtr>
inline void MultiThreader(IntType d0, IntType d1, FuncPtr func) noexcept
{
    IntType num = d0 * d1;
    IntType optNumThreads = std::min(int{NUM_80}, KuDNN::Threading::GetMaxNumThreads());
    std::vector<std::pair<IntType, IntType>> split = SplitLen(num, optNumThreads);
    const auto exec = [d0, d1, func](IntType start, IntType end) {
        IntType i0;
        IntType i1;
        NdIteratorInit(start, i0, d0, i1, d1);
        for (IntType i = start; i < end; i++) {
            func(i0, i1);
            NdIteratorStep(i0, d0, i1, d1);
        }
    };
    if (optNumThreads > 1) {
#pragma omp parallel for num_threads(optNumThreads)
        for (SizeType i = 0; i < split.size(); i++) {
            exec(split[i].first, split[i].second);
        }
    } else {
        for (SizeType i = 0; i < static_cast<SizeType>(num); ++i) {
            IntType i0;
            IntType i1;
            NdIteratorInit(i, i0, d0, i1, d1);
            func(i0, i1);
        }
    }
}

template <typename FuncPtr>
inline void MultiThreader(IntType d0, IntType d1, IntType d2, FuncPtr func) noexcept
{
    IntType num = d0 * d1 * d2;
    IntType optNumThreads = std::min(int{NUM_80}, KuDNN::Threading::GetMaxNumThreads());
    std::vector<std::pair<IntType, IntType>> split = SplitLen(num, optNumThreads);
    const auto exec = [d0, d1, d2, func](IntType start, IntType end) {
        IntType i0;
        IntType i1;
        IntType i2;
        NdIteratorInit(start, i0, d0, i1, d1, i2, d2);
        for (IntType i = start; i < end; i++) {
            func(i0, i1, i2);
            NdIteratorStep(i0, d0, i1, d1, i2, d2);
        }
    };
    if (optNumThreads > 1) {
#pragma omp parallel for num_threads(optNumThreads)
        for (SizeType i = 0; i < split.size(); i++) {
            exec(split[i].first, split[i].second);
        }
    } else {
        for (SizeType i = 0; i < static_cast<SizeType>(num); ++i) {
            IntType i2;
            IntType i0;
            IntType i1;
            NdIteratorInit(i, i0, d0, i1, d1, i2, d2);
            func(i0, i1, i2);
        }
    }
}

template <typename T>
void ArrayCopy(T *dst, const T *src, SizeType size)
{
    for (SizeType i = 0; i < size; ++i) {
        dst[i] = src[i];
    }
}

template <typename T>
void ArrayAdd(T *dst, const T *src, SizeType len)
{
    for (SizeType i = 0; i < len; ++i) {
        dst[i] += src[i];
    }
}

inline float ToFloat(__bf16 data)
{
    union {
        uint16_t u16;
        __bf16 bf16;
    } s;
    s.bf16 = data;
    union Converter {
        uint32_t u32;
        float f32;
    } converter;
    converter.u32 = static_cast<uint32_t>(s.u16) << 0x10;
    return converter.f32;
}

inline float ToFloat(float data)
{
    return data;
}

// Calculates cell position flags by combining iteration/layer position conditions
inline CellPositionT CalculateCellPosition(IntType iter, IntType lay, IntType nIter, IntType nLayer)
{
    // CellPositionT::MIDDLE_CELL = 0x0
    return (iter == 0 ? CellPositionT::FIRST_ITER : CellPositionT::MIDDLE_CELL) |
           (lay == 0 ? CellPositionT::FIRST_LAYER : CellPositionT::MIDDLE_CELL) |
           (iter == nIter - 1 ? CellPositionT::LAST_ITER : CellPositionT::MIDDLE_CELL) |
           (lay == nLayer - 1 ? CellPositionT::LAST_LAYER : CellPositionT::MIDDLE_CELL);
}

template <typename SrcType>
void CopyInitLayer(const RnnCfg &rnn, SrcType *wsDiffStatesLayer, SrcType *wsDiffDstLayer)
{
    switch (rnn.execDir) {
        case ExecutionDirectionT::L2R: {
            auto copy = [&](IntType iter, IntType b) {
                SrcType *currentDiffStateLayer = wsDiffStatesLayer + (((rnn.nLayer * rnn.nDir) *
                    (rnn.nIter + 1) + iter) * rnn.mb + b) * rnn.wsDiffStatesLayerLd;
                SrcType *currentDiffDstLayer = wsDiffDstLayer + (iter * rnn.mb + b) * rnn.dstLayerLd;
                ArrayCopy(currentDiffStateLayer, currentDiffDstLayer, rnn.dlc);
            };
            MultiThreader(rnn.nIter, rnn.mb, copy);
            break;
        }
        case ExecutionDirectionT::R2L: {
            auto copy = [&](IntType iter, IntType b) {
                IntType tmpIter = rnn.nIter - iter - 1;
                SrcType *currentDiffStateLayer = wsDiffStatesLayer + (((rnn.nLayer * rnn.nDir) *
                    (rnn.nIter + 1) + iter) * rnn.mb + b) * rnn.wsDiffStatesLayerLd;
                SrcType *currentDiffDstLayer = wsDiffDstLayer + (tmpIter * rnn.mb + b) * rnn.dstLayerLd;
                ArrayCopy(currentDiffStateLayer, currentDiffDstLayer, rnn.dlc);
            };
            for (IntType iter = 0; iter < rnn.nIter; iter++) {
                for (IntType b = 0; b < rnn.mb; b++) {
                    copy(iter, b);
                }
            }
            break;
        }
        case ExecutionDirectionT::BI_SUM: {
            auto copy = [&](IntType iter, IntType b) {
                IntType tmpIter = rnn.nIter - iter - 1;
                SrcType *currentDiffStateLayer1 = wsDiffStatesLayer + (((rnn.nLayer * rnn.nDir) *
                    (rnn.nIter + 1) + iter) * rnn.mb + b) * rnn.wsDiffStatesLayerLd;
                SrcType *currentDiffStateLayer2 = wsDiffStatesLayer + (((rnn.nLayer * rnn.nDir + 1) *
                    (rnn.nIter + 1) + tmpIter) * rnn.mb + b) * rnn.wsDiffStatesLayerLd;
                SrcType *currentDiffDstLayer = wsDiffDstLayer + (iter * rnn.mb + b) * rnn.dstLayerLd;
                ArrayCopy(currentDiffStateLayer1, currentDiffDstLayer, rnn.dlc);
                ArrayCopy(currentDiffStateLayer2, currentDiffDstLayer, rnn.dlc);
            };
            for (IntType iter = 0; iter < rnn.nIter; iter++) {
                for (IntType b = 0; b < rnn.mb; b++) {
                    copy(iter, b);
                }
            }
            break;
        }
        case ExecutionDirectionT::BI_CONCAT: {
            auto copy = [&](IntType iter, IntType b) {
                IntType tmpIter = rnn.nIter - iter - 1;
                SrcType *currentDiffStateLayer1 = wsDiffStatesLayer + (((rnn.nLayer * rnn.nDir) *
                    (rnn.nIter + 1) + iter) * rnn.mb + b) * rnn.wsDiffStatesLayerLd;
                SrcType *currentDiffStateLayer2 = wsDiffStatesLayer + (((rnn.nLayer * rnn.nDir + 1) *
                    (rnn.nIter + 1) + tmpIter) * rnn.mb + b) * rnn.wsDiffStatesLayerLd;
                SrcType *currentDiffDstLayer = wsDiffDstLayer + (iter * rnn.mb + b) * rnn.dstLayerLd;
                ArrayCopy(currentDiffStateLayer1, currentDiffDstLayer, rnn.dlc);
                ArrayCopy(currentDiffStateLayer2, currentDiffDstLayer + rnn.dlc, rnn.dlc);
            };
            for (IntType iter = 0; iter < rnn.nIter; iter++) {
                for (IntType b = 0; b < rnn.mb; b++) {
                    copy(iter, b);
                }
            }
            break;
        }
        default: throw Service::LogicError {"Unsupported direction"};
    }
}

template <typename SrcType>
void CopyInitIter(const RnnCfg &rnn, SrcType *wsDiffStatesIter, SrcType *wsDiffStatesIterC,
    SrcType *wsDiffDstIter, SrcType *wsDiffDstIterC)
{
    if (wsDiffDstIter) {
        auto copy = [&](IntType lay, IntType dir, IntType b) {
            SrcType *currentDiffStateIter = wsDiffStatesIter + (((lay * rnn.nDir + dir) *
                (rnn.nIter + 1) + rnn.nIter) * rnn.mb + b) * rnn.wsDiffStatesIterLd;
            SrcType *currentDiffDstIter = wsDiffDstIter + ((lay * rnn.nDir + dir) * rnn.mb + b) * rnn.dic;
            ArrayCopy(currentDiffStateIter, currentDiffDstIter, rnn.dic);
            if (wsDiffDstIterC) {
                SrcType *currentDiffStateIterC = wsDiffStatesIterC + (((lay * rnn.nDir + dir) *
                    (rnn.nIter + 1) + rnn.nIter) * rnn.mb + b) * rnn.wsDiffStatesIterCLd;
                SrcType *currentDiffDstIterC = wsDiffDstIterC + ((lay * rnn.nDir + dir) * rnn.mb + b) * rnn.dhc;
                ArrayCopy(currentDiffStateIterC, currentDiffDstIterC, rnn.dhc);
            }
        };
        MultiThreader(rnn.nLayer, rnn.nDir, rnn.mb, copy);
    }
}

template <typename SrcType>
void CopyResLayer(const RnnCfg &rnn, SrcType *wsDiffSrcLayer, SrcType *wsDiffStatesLayer)
{
    auto copy = [&](IntType iter, IntType b) {
        IntType tmpIter = (rnn.execDir == ExecutionDirectionT::R2L) ? rnn.nIter - iter - 1 : iter;
        // iter = (0 * rnn.nDir + 0) * (rnn.nIter + 1) + iter
        SrcType *currentDiffStateLayer1 = wsDiffStatesLayer + (iter * rnn.mb + b) * rnn.wsDiffStatesLayerLd;
        SrcType *currentDiffSrcLayer = wsDiffSrcLayer +  (tmpIter * rnn.mb + b) * rnn.srcLayerLd;
        ArrayCopy(currentDiffSrcLayer, currentDiffStateLayer1, rnn.slc);
        if (rnn.nDir - 1) {
            // 2 * rnn.nIter - iter = (0 * rnn.nDir + 1) * (rnn.nIter + 1) + rnn.nIter - 1 - iter
            SrcType *currentDiffStateLayer2 = wsDiffStatesLayer +
                                              ((2 * rnn.nIter - iter) * rnn.mb + b) * rnn.wsDiffStatesLayerLd;
            ArrayAdd(currentDiffSrcLayer, currentDiffStateLayer2, rnn.slc);
        }
    };
    MultiThreader(rnn.nIter, rnn.mb, copy);
}

template <typename SrcType>
void CopyResIter(const RnnCfg &rnn, SrcType *wsDiffSrcIter, SrcType *wsDiffSrcIterC,
    SrcType *wsDiffStatesIter, SrcType *wsDiffStatesIterC)
{
    if (wsDiffSrcIter) {
        auto copy = [&](IntType lay, IntType dir, IntType b) {
            SrcType *currentDiffStateIter = wsDiffStatesIter + (((lay * rnn.nDir + dir) *
                (rnn.nIter + 1)) * rnn.mb + b) * rnn.wsDiffStatesIterLd;
            SrcType *currentDiffStateIterC = wsDiffStatesIterC + (((lay * rnn.nDir + dir) *
                (rnn.nIter + 1)) * rnn.mb + b) * rnn.wsDiffStatesIterCLd;
            SrcType *currentDiffDstIter = wsDiffSrcIter + ((lay * rnn.nDir + dir) * rnn.mb + b) * rnn.sic;
            SrcType *currentDiffDstIterC = wsDiffSrcIterC + ((lay * rnn.nDir + dir) * rnn.mb + b) * rnn.dhc;
            ArrayCopy(currentDiffDstIter, currentDiffStateIter, rnn.sic);
            if (wsDiffSrcIterC) {
                ArrayCopy(currentDiffDstIterC, currentDiffStateIterC, rnn.dhc);
            }
        };
        MultiThreader(rnn.nLayer, rnn.nDir, rnn.mb, copy);
    }
}

} // Rnn
} // Detail
} // KuDNN

#endif // KUDNN_RNN_COMMON_HPP
