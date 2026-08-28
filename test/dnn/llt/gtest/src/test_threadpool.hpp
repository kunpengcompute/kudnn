
#include <memory>
#include "Eigen/Core"
#include "unsupported/Eigen/CXX11/ThreadPool"

#include "kudnn.hpp"

template <typename T, typename U>
inline void Balance211(T n, U team, U tid, T& nStart, T& nEnd)
{
    if (team <= 1 || n == 0) {
        nStart = 0;
        nEnd = n;
        return;
    }
    T minPerTeam = n / team;
    T remainder = n - minPerTeam * team;  // i.e., n % teams.
    nStart = tid * minPerTeam + std::min(tid, remainder);
    nEnd = nStart + minPerTeam + (tid < remainder);
}

// detailed impl of abstract threadpool
using EigenThreadPool = Eigen::NonBlockingThreadPool;

class ThreadPoolImpl : public KuDNN::Threading::ThreadpoolIface {
public:
    explicit ThreadPoolImpl(int numThreads = 0) {
        tp.reset(new EigenThreadPool(numThreads > 0 ? numThreads : 1));
    }

    int GetNumThreads() const override {
        return tp->NumThreads();
    }

    uint64_t GetFlags() const override {
        return KUDNN_ASYNCHRONOUS;
    }

    void ParallelFor(int n, const std::function<void(int, int)> &fn) override {
        int nthr = GetNumThreads();
        int njobs = std::min(n, nthr);

        for (int i = 0; i < njobs; i++) {
            tp->Schedule([i, n, njobs, fn]() {
                int start, end;
                Balance211(n, njobs, i, start, end);
                for (int j = start; j < end; j++) {
                    fn(j, n);
                }
            });
        }
    };

    bool IsInParallel() const override {
        return tp->CurrentThreadId() != -1;
    }
private:
    std::unique_ptr<EigenThreadPool> tp;
};