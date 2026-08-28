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


#ifndef BLOCKING_COUNTER_HPP
#define BLOCKING_COUNTER_HPP

struct CountingBarrier {
    CountingBarrier(unsigned size = 0)
    {
        Init(size);
    }

    void Init(unsigned size)
    {
        assert(size < KUDNNWaiterMask);
        notified = false;
        state = size;
    }

    void Notify()
    {
        auto s = state.fetch_sub(1) - 1;
        if (s != KUDNNWaiterMask) {
            assert(((s + 1) & ~KUDNNWaiterMask) != 0);
            return;
        }
        std::unique_lock<std::mutex> l(mutex);
        notified = true;
        cv.notify_all();
    }

    void Wait()
    {
        auto s = state.fetch_or(KUDNNWaiterMask);
        if (s == 0) {
            return;
        }
        std::unique_lock<std::mutex> l(mutex);
        cv.wait(l, [this]() { return notified; });
    }

private:
    static constexpr unsigned KUDNNWaiterMask = 1u << (sizeof(unsigned) * 8 - 1);

    std::atomic<unsigned> state;
    bool notified;

    std::condition_variable cv;
    std::mutex mutex;
};

#endif