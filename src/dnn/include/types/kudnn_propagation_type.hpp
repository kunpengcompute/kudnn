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


#ifndef KUDNN_PROPAGATION_TYPE_HPP
#define KUDNN_PROPAGATION_TYPE_HPP

namespace KuDNN {

// Enumeration of common propagation types which are supported in KuDNN.
enum class Propagation {
    UNDEFINED = 0,
    FORWARD_TRAINING,
    FORWARD_INFERENCE,
    BACKWARD_DATA,
    BACKWARD_WEIGHTS,
    BACKWARD_BIAS,
    BACKWARD,
    FORWARD = FORWARD_TRAINING,
};

} // KuDNN

#endif // KUDNN_PROPAGATION_TYPE_HPP
