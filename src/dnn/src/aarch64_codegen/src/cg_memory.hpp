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

#ifndef CG_MEMORY_HPP
#define CG_MEMORY_HPP

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <list>
#include <iostream>

#include <sys/mman.h>

#include "cg_inner.hpp"
#include "service/kudnn_service.hpp"

namespace ARMCG {

const size_t CSIZE = sizeof(uint32_t);

inline void *AlignedMalloc(size_t size, size_t alignment)
{
    void *p = ::KuDNN::Service::AlignedAlloc(size, alignment);
    // no p == nullptr check since in Service::AlignedAlloc it throw a bad_alloc exception instead of return nullptr
    return p;
}

inline void AlignedFree(void *p)
{
    ::KuDNN::Service::Deallocate(p);
}

template <class To, class From>
inline const To CastTo(From p)
{
    return static_cast<const To>(static_cast<size_t>(p));
}

class MmapAllocator {
public:
    uint32_t *Alloc(size_t size)
    {
        const size_t alignedSizeM1 = Inner::GetPageSize() - 1;
        size = (size + alignedSizeM1) & ~alignedSizeM1;

        int mode = MAP_PRIVATE | MAP_ANONYMOUS;

        void *p = mmap(nullptr, size, PROT_READ | PROT_WRITE, mode, -1, 0);
        if (p == MAP_FAILED) {
            throw Error(Err::CANT_ALLOC);
        }
        sizeList_[p] = size;
        return static_cast<uint32_t *>(p);
    }
    void Free(uint32_t *p)
    {
        if (p == nullptr) {
            return;
        }
        SizeList::iterator i = sizeList_.find(p);
        if (i == sizeList_.end()) {
            throw Error(Err::BAD_PARAMETER);
        }
        if (munmap(i->first, i->second) < 0) {
            throw Error(Err::MUNMAP);
        }
        sizeList_.erase(i);
    }
    bool UseProtect() const
    {
        return true;
    }

private:
    using SizeList = std::unordered_map<void *, size_t>;
    SizeList sizeList_;
};

// 2nd parameter for constructor of CodeArray(maxSize, userPtr, alloc)
void *const AUTO_GROW = reinterpret_cast<void*>(AUTO_GROW_VAL);
void *const DONT_SET_PROTECT_RWE = reinterpret_cast<void*>(DONT_SET_PROTECT_RWE_VAL);

class CodeArray {
public:
    // type of partially applied function for encoding
    using EncFunc = std::function<uint32_t(int64_t)>;
    struct AddrInfo {
        AddrInfo(size_t offset, size_t addr, EncFunc func)
            : codeOffset(offset),
              jmpAddr(addr),
              encFunc(func) {}
        uint32_t GetVal() const
        {
            return encFunc(static_cast<int64_t>(jmpAddr - codeOffset) * CSIZE);
        }
        size_t codeOffset; // position to write
        size_t jmpAddr;    // value to write
        EncFunc encFunc;   // encoding function
    };
    using AddrInfoList = std::list<AddrInfo>;
    enum class Type {
        BUF_USER = 1, // use userPtr(non alignment, non protect)
        BUF_ALLOC,    // use new(alignment, protect)
        MEM_AUTO_GROW // automatically move and grow memory if necessary
    };
    enum class ProtectMode {
        RW = 0,  // read/write
        RWE = 1, // read/write/exec
        RE = 2   // read/exec
    };

    CodeArray(const CodeArray &rhs);
    explicit CodeArray(size_t maxSize, void *userPtr = nullptr, MmapAllocator *allocator = nullptr)
        : type_(userPtr == AUTO_GROW ? Type::MEM_AUTO_GROW :
             (userPtr == nullptr || userPtr == DONT_SET_PROTECT_RWE) ? Type::BUF_ALLOC : Type::BUF_USER),
          alloc_(allocator ? allocator : &defaultAllocator_),
          maxSize_(maxSize / CSIZE),
          top_(type_ == Type::BUF_USER ? static_cast<uint32_t *>(userPtr) :
              alloc_->Alloc((std::max<size_t>)(maxSize, CSIZE))),
          size_(0),
          isCalledCalcJmpAddress_(false)
    {
        if (maxSize_ > 0 && top_ == nullptr) {
            throw Error(Err::CANT_ALLOC);
        }
        if ((type_ == Type::BUF_ALLOC && userPtr != DONT_SET_PROTECT_RWE &&
                UseProtect()) && !SetProtectMode(ProtectMode::RWE, false)) {
            alloc_->Free(top_);
            throw Error(Err::CANT_PROTECT);
        }
    }
    void operator=(const CodeArray &);
    virtual ~CodeArray() noexcept
    {
        if (IsAllocType()) {
            bool isNeedFree = true;
            if (UseProtect()) {
                try {
                    SetProtectModeRW(false);
                } catch (ARMCG::Error& err) {
                    std::cout << __FUNCTION__ << " err:" << err.What() << std::endl;
                    isNeedFree = false;
                }
            }
            if (isNeedFree) {
                alloc_->Free(top_);
            }
        }
    }

    bool IsAllocType() const
    {
        return type_ == Type::BUF_ALLOC || type_ == Type::MEM_AUTO_GROW;
    }
    bool SetProtectMode(ProtectMode mode, bool throwException = true)
    {
        bool isOK = Protect(top_, GetMaxSize(), mode);
        if (isOK) {
            return true;
        }
        if (throwException) {
            throw Error(Err::CANT_PROTECT);
        }
        return false;
    }
    bool SetProtectModeRE(bool throwException = true)
    {
        return SetProtectMode(ProtectMode::RE, throwException);
    }
    bool SetProtectModeRW(bool throwException = true)
    {
        return SetProtectMode(ProtectMode::RW, throwException);
    }
    void ResetSize()
    {
        size_ = 0;
        addrInfoList_.clear();
        isCalledCalcJmpAddress_ = false;
    }
    void ClearCodeArray()
    {
        for (size_t i = 0; i < size_; i++) {
            top_[i] = 0;
        }
        size_ = 0;
    }
    // write 4 byte data
    void WriteData(uint32_t code)
    {
        if (size_ >= maxSize_) {
            if (type_ == Type::MEM_AUTO_GROW) {
                GrowMemory();
            } else {
                throw Error(Err::CODE_IS_TOO_BIG);
            }
        }
        top_[size_++] = code;
    }
    uint8_t *GetCode() const
    {
        return static_cast<uint8_t *>(static_cast<void*>(top_));
    }
    template <class F>
    const F GetCode() const
    {
        return reinterpret_cast<F>(top_);
    }
    uint8_t *GetCurr() const
    {
        return static_cast<uint8_t *>(static_cast<void*>(&top_[size_]));
    }
    template <class F>
    const F GetCurr() const
    {
        return static_cast<F>(static_cast<void*>(&top_[size_]));
    }
    // return byte size
    size_t GetSize() const
    {
        return size_ * CSIZE;
    }
    size_t GetMaxSize() const
    {
        return maxSize_ * CSIZE;
    }
    // set byte size
    void SetSize(size_t size)
    {
        if (size > GetMaxSize()) {
            throw Error(Err::OFFSET_IS_TOO_BIG);
        }
        size_ = size / CSIZE;
    }
    void Dump() const
    {
        for (size_t i = 0; i < size_; ++i) {
            printf("%08X\n", top_[i]);
        }
    }
    // @param offset [in] offset from top
    // @param disp [in] offset from the next of jmp
    void Rewrite(size_t offset, uint32_t disp)
    {
        uint32_t *const data = top_ + offset;
        *data = disp;
    }
    void Save(size_t offset, size_t addr, const EncFunc &func)
    {
        addrInfoList_.push_back(AddrInfo(offset, addr, func));
    }
    bool IsAutoGrow() const
    {
        return type_ == Type::MEM_AUTO_GROW;
    }
    bool IsCalledCalcJmpAddress() const
    {
        return isCalledCalcJmpAddress_;
    }
    // change exec permission of memory
    // @param addr [in] buffer address
    // @param size [in] buffer size
    // @param protectMode [in] mode(RW/RWE/RE)
    // @return true(success), false(failure)
    static inline bool Protect(void *addr, size_t size, ProtectMode protectMode)
    {
        const int rw = PROT_READ | PROT_WRITE;
        const int rwe = PROT_READ | PROT_WRITE | PROT_EXEC;
        const int re = PROT_READ | PROT_EXEC;
        int mode;
        switch (protectMode) {
            case ProtectMode::RW:
                mode = rw;
                break;
            case ProtectMode::RWE:
                mode = rwe;
                break;
            case ProtectMode::RE:
                mode = re;
                break;
            default:
                return false;
        }
        size_t pageSize = Inner::GetPageSize();
        uintptr_t addrVal = (uintptr_t)addr;
        size_t iaddr = static_cast<size_t>(addrVal);
        size_t roundAddr = iaddr & ~(pageSize - static_cast<size_t>(1));
        return mprotect(reinterpret_cast<void *>(roundAddr), size + (iaddr - roundAddr), mode) == 0;
    }
    // get aligned memory pointer
    // @param addr [in] address
    // @param alignedSize [in] power of two
    // @return aligned addr by alingedSize
    static inline uint32_t *GetAlignedAddress(uint32_t *addr, size_t alignedSize = 16)
    {
        uintptr_t addrVal = (uintptr_t)addr;
        return reinterpret_cast<uint32_t *>((static_cast<size_t>(addrVal) + alignedSize - 1) &
            ~(alignedSize - static_cast<size_t>(1)));
    }

protected:
    bool UseProtect() const
    {
        return alloc_->UseProtect();
    }
    // allocate new memory and copy old data to the new area
    void GrowMemory()
    {
        const size_t newSize = (std::max<size_t>)(static_cast<size_t>(MaxCodeSize::DEFAULT),
            GetMaxSize() * 2);
        uint32_t *newTop = alloc_->Alloc(newSize);
        if (newTop == nullptr) {
            throw Error(Err::CANT_ALLOC);
        }
        for (size_t i = 0; i < size_; i++) {
            newTop[i] = top_[i];
        }
        alloc_->Free(top_);
        top_ = newTop;
        maxSize_ = newSize / CSIZE;
    }
    // calc jmp address for AutoGrow mode
    void CalcJmpAddress()
    {
        if (isCalledCalcJmpAddress_) {
            return;
        }
        for (AddrInfoList::const_iterator i = addrInfoList_.begin(), ie = addrInfoList_.end();
                i != ie; ++i) {
            uint32_t disp = i->GetVal();
            Rewrite(i->codeOffset, disp);
        }
        isCalledCalcJmpAddress_ = true;
    }

    friend class LabelManager;

    AddrInfoList addrInfoList_;
    const Type type_;
    MmapAllocator *alloc_;
    MmapAllocator defaultAllocator_;
    size_t maxSize_; // max size of code size (per uint32_t)
    uint32_t *top_;
    size_t size_; // code size
    bool isCalledCalcJmpAddress_;
};

} // namespace ARMCG

#endif // CG_MEMORY_HPP
