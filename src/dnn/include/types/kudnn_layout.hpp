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

#ifndef KUDNN_LAYOUT_HPP
#define KUDNN_LAYOUT_HPP

namespace KuDNN {

// Enumeration of common data types which are supported in KuDNN.
enum class Layout {
    UNDEFINED = 0,
    A, // single row
    AB,
    BA,
    ABC,
    ACB,
    BAC,
    BCA,
    CAB,
    CBA,
    ABCD,
    ABDC,
    ACBD,
    ACDB,
    ADBC,
    ADCB,
    BACD,
    BCDA,
    CDAB,
    CDBA,
    DCAB,
    ABCDE,
    ABCED,
    ABDEC,
    ACBDE,
    ACDEB,
    ADECB,
    BACDE,
    BCDEA,
    CDEAB,
    CDEBA,
    DECAB,
    ROW_MAJOR = AB,
    COL_MAJOR = BA,
    NCHW = ABCD,
    NHWC = ACDB,
    NCDHW = ABCDE,
    NDHWC = ACDEB,
    OIHW = ABCD,
    HWIO = CDBA,
    HWOI = CDAB,
    OHWI = ACDB,
    OHWO = BCDA,
    IOHW = BACD,
};

} // KuDNN

#endif // KUDNN_LAYOUT_HPP
