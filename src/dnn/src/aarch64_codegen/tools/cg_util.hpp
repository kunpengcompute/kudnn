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

#ifndef CG_UTIL_HPP
#define CG_UTIL_HPP

#ifndef __linux__
#error "Unsupported OS"
#endif

#include <cstdint>
#include <cstddef>
#include <utility>

namespace ARMCG {
namespace Util {

template <typename T>
inline void Unused(T &&t) noexcept
{
    static_cast<void>(t);
}

template <typename T, typename ... Args>
inline void Unused(T &&t, Args && ... args) noexcept
{
    Unused(std::forward<T>(t));
    Unused(std::forward<Args>(args)...);
}

using Type = uint64_t;

constexpr uint32_t MAX_CACHE_LEVEL = 7; // armv9
constexpr uint32_t MAX_TOPOLOGY_LEVEL = 2;
constexpr uint32_t MAX_PATH_LEN = 1024;

enum class Arm64CpuTopologyLevel {
    SMT = 1,
    CORE = 2
};

enum class Arm64CacheType {
    NO_CACHE = 0,
    INST_ONLY = 1,
    DATA_ONLY = 2,
    SEPARATE = 3,
    UNIFIED = 4,
    OTHER = 5
};

enum class Arm64CacheLevel {
    L1 = 1,
    L2,
    L3,
    L4,
    L5,
    L6,
    L7
};

enum class CacheType {
    ICACHE = 0,
    DCACHE = 1,
    UCACHE = 2
};

enum class VCLen {
    VC_NONE = 0,
    VC_128 = 16 * 1,
    VC_256 = 16 * 2,
    VC_384 = 16 * 3,
    VC_512 = 16 * 4,
    VC_640 = 16 * 5,
    VC_768 = 16 * 6,
    VC_896 = 16 * 7,
    VC_1024 = 16 * 8,
    VC_1152 = 16 * 9,
    VC_1280 = 16 * 10,
    VC_1408 = 16 * 11,
    VC_1536 = 16 * 12,
    VC_1664 = 16 * 13,
    VC_1792 = 16 * 14,
    VC_1920 = 16 * 15,
    VC_2048 = 16 * 16,
};

enum class HwCap {
    CG_HWCAP_NONE = 0,
    CG_HWCAP_ADVSIMD = 1 << 1,
    CG_HWCAP_FP = 1 << 2,
    CG_HWCAP_VC = 1 << 3,
    CG_HWCAP_ATOMIC = 1 << 4,
    CG_HWCAP_BF16 = 1 << 5,
};

struct Implementer {
    uint32_t id;
    const char *implementer;
};

struct LevelCacheInfo {
    Arm64CacheType type;
    // I cache size, D cache size and Unified cache size.
    // Example1: some cache level of a CPU core has separate caches of
    // 32KB I$ and 64KB D$,
    // size is 1024 * 32, 1024 * 64, 0 sharingCores is 1, 1, 0
    // Example2: some cache level of CPU cores has a
    // 1MB unified cache, and it is shared by 12 CPU cores,
    // size is 0, 0, 1024 * 1024, sharingCores is 0, 0, 12.
    uint32_t size[3];         // I cache, D cache, Unified cache
    uint32_t sharingCores[3]; // I cache, D cache, Unified cache
};

struct CacheInfoV2 {
    // used as table index
    uint64_t midrEL1;
    LevelCacheInfo levelCache[MAX_CACHE_LEVEL];
};

// CPU detection class
class CpuInfo {
public:
    CpuInfo() {}
    virtual ~CpuInfo() {}
    void DumpCacheInfo() const;
    int GetCacheSize(CacheType type, uint32_t level) const;
    Arm64CacheType GetCacheType(int level) const;
    int GetCodeCacheSize(int level) const;
    int GetLastDataCacheLevel() const;
    int GetNumCores(Arm64CpuTopologyLevel level = Arm64CpuTopologyLevel::CORE) const;
    uint64_t GetVCLen() const;
    Type GetType() const;
    uint32_t GetUnifiedCacheSize(int level) const;
    void Put() const;

protected:
    void Init();

    int numCores_[2] = {}; // [0]:SmtLevel, [1], CoreLevel
    CacheInfoV2 cacheInfo_ = {};
    uint32_t lastDataCacheLevel_ = 0;
    Type type_ = 0;
    uint64_t sveLen_ = 0;
};

class Cpu {
public:
    Cpu();
    ~Cpu();
    void DumpCacheInfo() const;
    Arm64CacheType GetCacheType(const Arm64CacheLevel i) const;
    uint32_t GetLastDataCacheLevel() const;
    uint32_t GetNumCores(Arm64CpuTopologyLevel level) const;
    Type GetType() const;
    VCLen GetVCLen() const;
    bool Has(HwCap type) const;
    bool IsAtomicSupported() const;
    bool IsBf16Supported() const;

private:
    CpuInfo *info;
};

// Returns a value of type T by reinterpretting the representation of the input
// value (part of C++20).
//
// Provides a safe implementation of type punning.
//
// Constraints:
// - U and T must have the same size
// - U and T must be trivially copyable
template <typename T, typename U>
inline T BitCast(const U &u)
{
    T t;
    // Since bit_cast is used in SYCL kernels it cannot use std::memcpy as it
    // can be implemented as @llvm.objectsize.* + __memcpy_chk for Release
    // builds which cannot be translated to SPIR-V.
    void* pT = static_cast<void*>(&t);
    uint8_t *tPtr = static_cast<uint8_t *>(pT);
    U tempU = u;
    void* pU = static_cast<void*>(&tempU);
    const uint8_t *uPtr = static_cast<const uint8_t *>(pU);
    for (size_t i = 0; i < sizeof(U); i++) {
        tPtr[i] = uPtr[i];
    }
    return t;
}

inline int Float2Int(float x)
{
    return BitCast<int>(x);
}

inline float Int2Float(int x)
{
    return BitCast<float>(x);
}

} // namespace Util
} // namespace ARMCG

#endif // CG_UTIL_HPP
