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

#ifndef KUDNN_EXCEPTION_HPP
#define KUDNN_EXCEPTION_HPP

#include <exception>
#include <new>
#include <stdexcept>

namespace KuDNN {
namespace Service {

// Error handling in KuDNN is performed via exceptions.
// KuDNN has its own exceptions which are nested from std::exception standard class
struct BadAlloc : public std::bad_alloc {
    const char* what() const noexcept override
    {
        return "KuDNN allocation failed";
    }
};

struct BadArrayNewLength : public std::bad_array_new_length {
    const char* what() const noexcept override
    {
        return "KuDNN bad array new length";
    }
};

struct LogicError : public std::logic_error {
    explicit LogicError(const std::string& whatArg) : std::logic_error(whatArg) {}
    explicit LogicError(const char* whatArg) : std::logic_error(whatArg) {}
    LogicError(const LogicError& other) noexcept : std::logic_error(other) {}
    LogicError& operator=(const LogicError& other) noexcept = default;
};

struct Unsupported : public std::invalid_argument {
    explicit Unsupported(const std::string& whatArg) : std::invalid_argument(whatArg) {}
    explicit Unsupported(const char* whatArg) : std::invalid_argument(whatArg) {}
    Unsupported(const Unsupported& other) noexcept : std::invalid_argument(other) {}
    Unsupported& operator=(const Unsupported& other) noexcept = default;
};

} // Service
} // KuDNN

#endif // KUDNN_EXCEPTION_HPP
