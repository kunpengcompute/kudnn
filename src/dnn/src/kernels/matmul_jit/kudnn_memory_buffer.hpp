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

#ifndef KUDNN_MEMORY_BUFFER_HPP
#define KUDNN_MEMORY_BUFFER_HPP

#include "kudnn.hpp"

namespace KuDNN::Service {

enum class MemoryType {
    DEFAULT,
    FAST
};

struct MemoryBuffer {
    void* ptr;
    MemoryType type;
    bool owner;

    // type here is just a request
    MemoryBuffer(SizeType size, MemoryType requestedType, SizeType alignment = ALIGNMENT);
    MemoryBuffer& operator=(MemoryBuffer&& other);
    MemoryBuffer(MemoryBuffer&& other);
    MemoryBuffer(const MemoryBuffer& other);
    MemoryBuffer& operator=(const MemoryBuffer& other);
    ~MemoryBuffer();
    void Free();
    MemoryBuffer();
};

}
#endif // KUDNN_MEMORY_BUFFER_HPP