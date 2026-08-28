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

#include "gtest/gtest.h"

#include "service/kudnn_err_codes.hpp"
#include "tools/cg_cpu_isa_traits.hpp"
#include "types/kudnn_data_type.hpp"

#include "kudnn_brgemm.hpp"
#include "internal_funcs.hpp"

using namespace KuDNN;

class BrgemmTest : public ::testing::Test {
public:
    void SetUp() override {}
    void TearDown() override {}
};

TEST(BrgemmTest, TestBrgemmUtilsBalance211)
{
    IntType n = 1;
    int team = 1;
    int tid = 1;
    IntType nStart = 1;
    IntType nEnd = 1;
    EXPECT_NO_THROW(Balance211(n, team, tid, nStart, nEnd));
}

TEST(BrgemmTest, TestBrgemmUtilsCHECK)
{
    EXPECT_ANY_THROW(CHECK(Status::UNSUPPORTED));
}
