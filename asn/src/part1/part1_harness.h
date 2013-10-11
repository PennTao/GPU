#ifndef PART1_HARNESS_H
#define PART1_HARNESS_H

#include "part1.h"

class Run {
public:
    virtual bool checkOp() = 0;
    virtual void performOp() = 0;
    virtual void setupOp() = 0;
    virtual void teardownOp() = 0;
};

class RunDriver {
public:
    RunDriver(Run * run, 
            int warmup_iters, 
            int test_iters);
    ~RunDriver();
    float timeRun();
    bool testRun();
private:
    Run * mRun;
    int mWarmupIters;
    int mTestIters;
    unsigned int timer;
};

enum SortingRoutine {
    Thrust,
    Stl,
    Stdlib
};


template <typename K>
class SortRun : public Run {
public:
    SortRun(SortingRoutine sort,
            Processor src,
            Processor dst,
            int num) :
        mIdata(NULL),
        mOdata(NULL),
        mSrc(src),
        mDst(dst),
        mSort(sort),
        mNum(num) {}
    virtual bool checkOp();
    virtual void performOp();
    virtual void setupOp();
    virtual void teardownOp();
private:
    K * mIdata;
    K * mOdata;
    Processor mSrc;
    Processor mDst;
    SortingRoutine mSort;
    int mNum;
};

#include "part1_harness.inl"

#endif //PART1_HARNESS_H
