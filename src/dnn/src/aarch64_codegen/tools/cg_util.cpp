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

#include "cg_util.hpp"
#include <string>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <dirent.h>
#include <regex.h>
#include <sys/auxv.h>
#include <sys/prctl.h>
#include <unistd.h>

#include "src/cg_error.hpp"
#include "src/cg_arm_const.hpp"

#ifndef __linux__
#error "Something wrong"
#endif
// In old Linux such as Ubuntu 16.04, HWCAP_ATOMICS, HWCAP_FP, HWCAP_ASIMD
// can not be found in <bits/hwcap.h> which is included from <sys/auxv.h>.
// cg_codegen uses <asm/hwcap.h> as an alternative.
#ifndef HWCAP_FP
#include <asm/hwcap.h>
#endif

namespace ARMCG {
namespace Util {

// Linux kernel used in Ubuntu 20.04 does not have HWCAP2_BF16 definition.
#ifdef AT_HWCAP2
#ifndef HWCAP2_BF16
constexpr unsigned long HWCAP2_BF16 = (1UL << 14);
#endif
#endif

constexpr inline uint64_t CG_MIDR_EL1(uint64_t i, uint64_t v, uint64_t a, uint64_t p, uint64_t r)
{
    return (i << BITFIELD_24) | (v << BITFIELD_20) | (a << BITFIELD_16) | (p << BITFIELD_4) | (r << BITFIELD_0);
}

static char *MovNumBytes(char *dest, const char *src, int n)
{
    if (dest == nullptr || src == nullptr) {
        return nullptr;
    }
    int srcNum = strlen(src);
    int num = n < srcNum ? n : srcNum;
    for (int i = 0; i < num; ++i) {
        if (i < srcNum) {
            dest[i] = src[i];
        } else {
            dest[i] = '\0';
        }
    }
    return dest;
}

class CpuInfoLinux : public CpuInfo {
public:
    CpuInfoLinux()
    {
        Init();
        SetSysRegVal();
        SetNumCores();
        SetHwCap();
        SetImplementer();
    }

private:
    int GetFilePathMaxTailNumPlus1(const char *path)
    {
        char dirPath[MAX_PATH_LEN];
        char filePattern[MAX_PATH_LEN];
        int retVal = 0;
        GetRegEx(dirPath, path, "/([^/]+/)+");
        // Remove last '/'
        dirPath[strlen(dirPath) - 1] = '\0';
        GetRegEx(filePattern, path, "[^/]+$");
        std::string pattern = filePattern;
        pattern += "[0-9]+";
        DIR *dir = opendir(dirPath);
        if (dir != nullptr) {
            struct dirent *dp = readdir(dir);
            while (dp != nullptr) {
                if (GetRegEx(dirPath, dp->d_name, pattern.c_str()) == 0) {
                    retVal++;
                }
                dp = readdir(dir);
            }
            closedir(dir);
        }
        return retVal;
    }
    void GetLineInFile(char *buf, const char *path, const int num)
    {
        auto chomp = [](char *ptr, const int size) {
            for (int i = 0; i < size; i++) {
                if (*(ptr + i) == '\n') {
                    *(ptr + i) = '\0';
                } else if (*(ptr + i) == '\0') {
                    break;
                }
            }
        };
        char resolvedPath[PATH_MAX];
        if (realpath(path, resolvedPath) == nullptr) {
            printf("%s path is not real path\n", __FUNCTION__);
            return;
        }
        FILE *fp = fopen(resolvedPath, "r");
        if (!(fp && fread(buf, sizeof(char), num, fp))) {
            buf[0] = '\0';
        }
        chomp(buf, BUF_SIZE);
        if (fp != nullptr) {
            fclose(fp);
        }
    }
    // Return directory path
    // @param[in] path ex. /sys/devices/system/node/node
    // @param[out] buf ex. /sys/devices/system/node
    int GetRegEx(char *buf, const char *path, const char *regex)
    {
        regex_t regexBuf;
        regmatch_t match[1];
        if (regcomp(&regexBuf, regex, REG_EXTENDED) != 0) {
            throw Error(Err::INTERNAL);
        }
        const int retVal = regexec(&regexBuf, path, 1, match, 0);
        regfree(&regexBuf);
        if (retVal != 0) {
            return -1;
        }
        const int startIdx = match[0].rm_so;
        const int endIdx = match[0].rm_eo;
        // Something wrong (multiple match or not match)
        if (startIdx == -1 || endIdx == -1 || (endIdx - startIdx - 1) < 1) {
            return -1;
        }
        MovNumBytes(buf, path + startIdx, endIdx - startIdx);
        buf[endIdx - startIdx] = '\0';
        return 0;
    }

    void SetHwCap()
    {
        const unsigned long hwcap = getauxval(AT_HWCAP);
        if (hwcap & HWCAP_ATOMICS) {
            type_ |= static_cast<Type>(HwCap::CG_HWCAP_ATOMIC);
        }
        if (hwcap & HWCAP_FP) {
            type_ |= static_cast<Type>(HwCap::CG_HWCAP_FP);
        }
        if (hwcap & HWCAP_ASIMD) {
            type_ |= static_cast<Type>(HwCap::CG_HWCAP_ADVSIMD);
        }
#ifdef AT_HWCAP2
        const unsigned long hwcap2 = getauxval(AT_HWCAP2);
        if (hwcap2 & HWCAP2_BF16) {
            type_ |= static_cast<Type>(HwCap::CG_HWCAP_BF16);
        }
#endif
#ifdef HWCAP_VectorComputation
        // Some old <sys/auxv.h> may not define HWCAP_VectorComputation.
        // In that case, VectorComputation is treated as if it were not supported.
        if (hwcap & HWCAP_VectorComputation) {
            type_ |= static_cast<Type>(HwCap::CG_HWCAP_VectorComputation);
            // svcntb(); if arm_sve.h is available
            // PR_VectorComputation_GET_VL
            sveLen_ = static_cast<uint64_t>(static_cast<SveLen>(prctl(VAL_51)));
        }
#endif
    }
    void SetNumCores()
    {
        // There are some methods to Get # of cores.
        // Considering various kernel versions and CPUs, a combination of
        // multiple methods may be required.
        // 1) sysconf(_SC_NPROCESSORS_ONLN)
        // 2) /sys/devices/system/cpu/online
        // 3) std::thread::hardware_concurrency()
        numCores_[0] = numCores_[1] = sysconf(_SC_NPROCESSORS_ONLN);
    }
    void SetSysRegVal()
    {
        // Some Linux does not permit to access MIDR_EL1 register
        // from user program. The register value is disclosed
        // in /sys/devices/system/cpu/cpu?/regs/identification/midrEL1
        std::string realPath(PATH_MAX, '\0');
        if (realpath(pathMidrEL1, realPath.data()) == nullptr) {
            return;
        }
        FILE *file = fopen(realPath.c_str(), "r");
        if (file == nullptr) {
            throw Error(Err::INTERNAL);
            return;
        }
        char buf[64];
        if (fread(buf, sizeof(char), VAL_64, file) == 0) {
            fclose(file);
            throw Error(Err::INTERNAL);
            return;
        }
        char *stop;
        cacheInfo_.midrEL1 = strtoull(buf, &stop, VAL_16);
        fclose(file);
    }

    static constexpr int MAX_PATH_LEN = 1024;
    static constexpr int BUF_SIZE = 1024;
    const char *pathMidrEL1 = "/sys/devices/system/cpu/cpu0/regs/identification/midr_el1";
    const struct CacheInfoV2 cacheInfoDict[2] = {
        {CG_MIDR_EL1(0x46, 0x1, 0xf, 0x1, 0x0), // A64FX
         {
             {Arm64CacheType::SEPARATE, {1024 * 64, 1024 * 64, 0}, {1, 1, 0}}, // L1
             {Arm64CacheType::UNIFIED, {0, 0, 1024 * 1024 * 8}, {0, 0, 12}},   // L2
             {Arm64CacheType::NO_CACHE, {0, 0, 0}, {0, 0, 0}},                 // L3
             {Arm64CacheType::NO_CACHE, {0, 0, 0}, {0, 0, 0}},                 // L4
             {Arm64CacheType::NO_CACHE, {0, 0, 0}, {0, 0, 0}},                 // L5
             {Arm64CacheType::NO_CACHE, {0, 0, 0}, {0, 0, 0}},                 // L6
             {Arm64CacheType::NO_CACHE, {0, 0, 0}, {0, 0, 0}}                  // L7
         }},
        {CG_MIDR_EL1(0x46, 0x2, 0xf, 0x1, 0x0), // A64FX
         {
             {Arm64CacheType::SEPARATE, {1024 * 64, 1024 * 64, 0}, {1, 1, 0}}, // L1
             {Arm64CacheType::UNIFIED, {0, 0, 1024 * 1024 * 8}, {0, 0, 12}},   // L2
             {Arm64CacheType::NO_CACHE, {0, 0, 0}, {0, 0, 0}},                 // L3
             {Arm64CacheType::NO_CACHE, {0, 0, 0}, {0, 0, 0}},                 // L4
             {Arm64CacheType::NO_CACHE, {0, 0, 0}, {0, 0, 0}},                 // L5
             {Arm64CacheType::NO_CACHE, {0, 0, 0}, {0, 0, 0}},                 // L6
             {Arm64CacheType::NO_CACHE, {0, 0, 0}, {0, 0, 0}}                  // L7
         }}};
};

void CpuInfo::DumpCacheInfo() const
{
    printf("numCores=%d\n", numCores_[1]); // phisical cores
    for (size_t i = 0; i < MAX_CACHE_LEVEL; i++) {
        auto cache = &cacheInfo_.levelCache[i];
        printf("L%zu, %d, %u, %u, %u, %u, %u, %u\n", i + 1, static_cast<int>(cache->type), cache->size[0],
               cache->size[1], cache->size[OFFSET_TWO], cache->sharingCores[0], cache->sharingCores[1],
               cache->sharingCores[OFFSET_TWO]);
    }
}

int CpuInfo::GetCacheSize(CacheType type, uint32_t level) const
{
    if (level <= MAX_CACHE_LEVEL) {
        auto cache = &cacheInfo_.levelCache[level - 1];
        switch (type) {
            case CacheType::ICACHE:
                return cache->size[0];
            case CacheType::DCACHE:
                return cache->size[1];
            case CacheType::UCACHE:
                return cache->size[OFFSET_TWO];
            default:
                throw Error(Err::BAD_PARAMETER);
                break;
        }
    } else {
        throw Error(Err::BAD_PARAMETER);
    }
}

Arm64CacheType CpuInfo::GetCacheType(int level) const
{
    return cacheInfo_.levelCache[level - 1].type;
}

int CpuInfo::GetCodeCacheSize(int level) const
{
    return cacheInfo_.levelCache[level - 1].size[0];
}

const char *CpuInfo::GetImplementer() const
{
    return implementer_;
}

int CpuInfo::GetLastDataCacheLevel() const
{
    return lastDataCacheLevel_;
}

int CpuInfo::GetNumCores(Arm64CpuTopologyLevel level) const
{
    switch (level) {
        case Arm64CpuTopologyLevel::SMT:
            return numCores_[0];
        case Arm64CpuTopologyLevel::CORE:
            return numCores_[1];
        default:
            return 0;
    }
}

uint64_t CpuInfo::GetSveLen() const
{
    return sveLen_;
}

Type CpuInfo::GetType() const
{
    return type_;
}

uint32_t CpuInfo::GetUnifiedCacheSize(int level) const
{
    return cacheInfo_.levelCache[level - 1].size[OFFSET_TWO];
}

void CpuInfo::Init()
{
    for (size_t i = 0; i < MAX_CACHE_LEVEL; i++) {
        auto cache = &cacheInfo_.levelCache[i];
        cache->type = Arm64CacheType::NO_CACHE;
        cache->size[0] = cache->size[1] = cache->size[OFFSET_TWO] = 0;
        cache->sharingCores[0] = cache->sharingCores[1] = cache->sharingCores[OFFSET_TWO] = 0;
    }
}

void CpuInfo::Put() const
{
    printf("numCores=%d\n", numCores_[0]);
    for (int level = 1; level <= VAL_3; level++) {
        printf("L%d unified size = %u\n", level, GetUnifiedCacheSize(level));
        printf("L%d code size = %d\n", level, GetCodeCacheSize(level));
    }
}

void CpuInfo::SetImplementer()
{
    const uint32_t id = (cacheInfo_.midrEL1 >> BITFIELD_24) & 0xff;
    const int lastId = sizeof(IMPLEMENTERS) / sizeof(Implementer);
    for (int i = 0; i < lastId; i++) {
        if (IMPLEMENTERS[i].id == id) {
            implementer_ = IMPLEMENTERS[i].implementer;
            return;
        }
    }
    implementer_ = static_cast<const char *>(IMPLEMENTERS[lastId - 1].implementer);
}

Cpu::Cpu() : info(new CpuInfoLinux()) {}

void Cpu::DumpCacheInfo() const
{
    return info->DumpCacheInfo();
}

Arm64CacheType Cpu::GetCacheType(const Arm64CacheLevel i) const
{
    return info->GetCacheType(static_cast<int>(i));
}

const char *Cpu::GetImplementer() const
{
    return info->GetImplementer();
}

uint32_t Cpu::GetLastDataCacheLevel() const
{
    return info->GetLastDataCacheLevel();
}

uint32_t Cpu::GetNumCores(Arm64CpuTopologyLevel level) const
{
    return info->GetNumCores(level);
}

SveLen Cpu::GetSveLen() const
{
    return static_cast<SveLen>(info->GetSveLen());
}

Type Cpu::GetType() const
{
    return info->GetType();
}

bool Cpu::Has(HwCap type) const
{
    return (static_cast<Type>(type) & info->GetType()) != 0;
}

bool Cpu::IsAtomicSupported() const
{
    return info->GetType() & static_cast<Type>(HwCap::CG_HWCAP_ATOMIC);
}

bool Cpu::IsBf16Supported() const
{
    return info->GetType() & static_cast<Type>(HwCap::CG_HWCAP_BF16);
}

} // namespace Util
} // namespace ARMCG
