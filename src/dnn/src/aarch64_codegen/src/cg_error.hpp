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

#ifndef CG_ERROR_HPP
#define CG_ERROR_HPP

#include <stdexcept>

namespace ARMCG {

enum class Err {
    NONE = 0,
    CODE_IS_TOO_BIG,           // use at CodeArray
    LABEL_IS_REDEFINED,        // use at LabelMgr
    LABEL_IS_TOO_FAR,          // use at CodeGenerator
    LABEL_IS_NOT_FOUND,        // use at LabelMgr
    BAD_PARAMETER,             // use at CodeArray
    CANT_PROTECT,              // use at CodeArray
    OFFSET_IS_TOO_BIG,         // use at CodeArray
    CANT_ALLOC,                // use at CodeArray
    LABEL_ISNOT_SET_BY_L,      // use at LabelMgr
    LABEL_IS_ALREADY_SET_BY_L, // use at LabelMgr
    INTERNAL,                  // use at Error
    ILLEGAL_REG_IDX,           // use at CodeGenerator
    ILLEGAL_REG_ELEM_IDX,      // use at CodeGenerator
    ILLEGAL_PREDICATE_TYPE,    // use at CodeGenerator
    ILLEGAL_IMM_RANGE,         // use at CodeGenerator
    ILLEGAL_IMM_VALUE,         // use at CodeGenerator
    ILLEGAL_IMM_COND,          // use at CodeGenerator
    ILLEGAL_SHMOD,             // use at CodeGenerator
    ILLEGAL_EXTMOD,            // use at CodeGenerator
    ILLEGAL_COND,              // use at CodeGenerator
    ILLEGAL_BARRIER_OPT,       // use at CodeGenerator
    ILLEGAL_CONST_RANGE,       // use at CodeGenerator
    ILLEGAL_CONST_VALUE,       // use at CodeGenerator
    ILLEGAL_CONST_COND,        // use at CodeGenerator
    ILLEGAL_TYPE,
    BAD_ALIGN,
    BAD_ADDRESSING,
    BAD_SCALE,
    MUNMAP
};

class Error : public std::runtime_error {
    static constexpr int ERR_NUM = 30;
    static constexpr const char *ERR_DESC[ERR_NUM] = {"none",
                                                      "code is too big",
                                                      "label is redefined",
                                                      "label is too far",
                                                      "label is not found",
                                                      "bad parameter",
                                                      "can't protect",
                                                      "offset is too big",
                                                      "can't alloc",
                                                      "label is not set by L()",
                                                      "label is already set by L()",
                                                      "internal error",
                                                      "illegal register index (can not encoding register index)",
                                                      "illegal register element index (can not encoding element index)",
                                                      "illegal predicate register type",
                                                      "illegal immediate parameter (range error)",
                                                      "illegal immediate parameter (unavailable value error)",
                                                      "illegal immediate parameter (condition error)",
                                                      "illegal shift-mode paramater",
                                                      "illegal extend-mode parameter",
                                                      "illegal condition parameter",
                                                      "illegal barrier option",
                                                      "illegal const parameter (range error)",
                                                      "illegal const parameter (unavailable error)",
                                                      "illegal const parameter (condition error)",
                                                      "illegal type",
                                                      "bad align",
                                                      "bad addressing",
                                                      "bad scale",
                                                      "Munmap"};

public:
    static const char *GetErrorDescription(Err err)
    {
        int item = static_cast<int>(err);
        if (item < 0 || item >= ERR_NUM) {
            return "bad err num";
        } else {
            return ERR_DESC[item];
        }
    }

    static const char *ConvertErrorToString(const Error &err)
    {
        return err.What();
    }

    explicit Error(Err err) : runtime_error(GetErrorDescription(err)), err_(err) {}

    operator int() const
    {
        return static_cast<int>(err_);
    }

    const char *What() const
    {
        return what();
    }

private:
    Err err_;
};

} // namespace ARMCG

#endif // CG_ERROR_HPP
