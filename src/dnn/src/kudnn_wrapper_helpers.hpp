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

#ifndef KUDNN_WRAPPER_HELPERS_HPP
#define KUDNN_WRAPPER_HELPERS_HPP

#include <utility>

#include "kudnn.hpp"

namespace KuDNN {
namespace Detail {

struct ReorderWrapper;

template <typename CodeGenKind>
struct CodeGenWrapper;

template <typename T>
struct TypeHelper final {
    using Type = T;
};

template <>
struct TypeHelper<ReorderWrapper> final {
    using Type = ReorderLayer;
};

template <typename CodeGenKind>
struct TypeHelper<CodeGenWrapper<CodeGenKind>> final {
    using Type = CodeGenKind;
};

template <typename T>
struct Wrapper {
    using UnderlyingType = typename TypeHelper<T>::Type;
    Wrapper() noexcept : obj(nullptr) {}
    Wrapper(const Wrapper &other) = delete;
    Wrapper(Wrapper &&other) noexcept
    {
        obj = other.obj;
        other.obj = nullptr;
    }
    Wrapper &operator=(const Wrapper &other) = delete;
    Wrapper &operator=(Wrapper &&other) noexcept
    {
        if (this != &other) {
            obj = other.obj;
            other.obj = nullptr;
        }
        return *this;
    }
    template <typename... Args>
    void Initialize(Args &&...args)
    {
        obj = new UnderlyingType{std::forward<Args>(args)...};
        static_cast<T *>(this)->InitPostamble(std::forward<Args>(args)...);
    }
    bool IsInitialized() const noexcept
    {
        return (obj != nullptr);
    }
    void Deinitialize() noexcept
    {
        delete obj;
        obj = nullptr;
    }
    virtual ~Wrapper()
    {
        Deinitialize();
    }

protected:
    UnderlyingType *obj;
};

struct ReorderWrapper final : Wrapper<ReorderWrapper> {
    ReorderWrapper() noexcept : p(nullptr) {}
    ReorderWrapper(ReorderWrapper &&other) noexcept
        : Wrapper<ReorderWrapper>(static_cast<Wrapper<ReorderWrapper> &&>(other))
    {
        p = other.p;
        other.p = nullptr;
    }
    ReorderWrapper &operator=(ReorderWrapper &&other) = delete;
    void InitPostamble(const TensorInfo &tensorInfo, const TensorInfo &desiredInfo)
    {
        if (WillReorder(tensorInfo, desiredInfo)) {
            p = Service::AlignedAlloc(desiredInfo.GetTotalTensorSize() * desiredInfo.GetType().GetSize());
        }
    }
    static bool WillReorder(const TensorInfo &tensorInfo, const TensorInfo &desiredInfo) noexcept
    {
        return ((tensorInfo.GetType() != desiredInfo.GetType()) ||
                ((tensorInfo.GetLayout() != desiredInfo.GetLayout()) &&
                 (tensorInfo.GetStrides() != desiredInfo.GetStrides())));
    }
    const void *Get(const void *data) const
    {
        if ((p == nullptr) || (data == nullptr)) {
            return data;
        } else {
            obj->Run(data, p);
            return p;
        }
    }
    void *Get(void *data) const
    {
        if ((p == nullptr) || (data == nullptr)) {
            return data;
        } else {
            obj->Run(data, p);
            return p;
        }
    }
    ~ReorderWrapper() noexcept
    {
        Service::Deallocate(p);
    }

protected:
    void *p;
};

template <typename CodeGenKind>
struct CodeGenWrapper final : Wrapper<CodeGenWrapper<CodeGenKind>> {
    CodeGenWrapper() noexcept
    {
        VERBOSE_PRINT;
    }
    CodeGenWrapper(CodeGenWrapper &&other) noexcept
        : Wrapper<CodeGenWrapper>(static_cast<Wrapper<CodeGenWrapper> &&>(other))
    {
        VERBOSE_PRINT;
    }
    CodeGenWrapper &operator=(CodeGenWrapper &&other) = delete;
    template <typename... Args>
    void InitPostamble(Args &&...args) const noexcept
    {
        VERBOSE_PRINT;
        Service::Unused(std::forward<Args>(args)...);
        this->obj->Ready();
    }
    template <typename F>
    const F GetCode() const
    {
        VERBOSE_PRINT;
        return this->obj->template GetCode<F>();
    }
    ~CodeGenWrapper()
    {
        VERBOSE_PRINT;
    }
};

} // namespace Detail
} // namespace KuDNN

#endif // KUDNN_WRAPPER_HELPERS_HPP
