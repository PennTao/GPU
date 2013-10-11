#ifndef PART2_HARNESS_H
#define PART2_HARNESS_H

#include "part2_kernel.h"

#include "part2.h"

#include <vector>
#include <deque>

#define CHECK(p) (p ? "PASSED" : "FAILED")

work_item_t alloc_item(int len);

void free_item(work_item_t item);

bool check_item(work_item_t item);

class ItemGenerator {
public:
    ItemGenerator(matrix_palette_t palette, 
            int num, int len) : 
    mPalette(palette),
    mTotal(num),
    mMade(0),
    mLen(len) {}
    work_item_t yield();
    bool hasNext();
private:
    matrix_palette_t mPalette;
    int mTotal;
    int mMade;
    int mLen;
    float4 goldTransform(const float4 & v,
            const float4 & bw); 
};

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

class ItemRun : public Run {
public:    
    ItemRun(ItemProcessor * proc);
    virtual void setupOp();
    virtual void performOp(); 
    virtual bool checkOp();
    virtual void teardownOp();
private:
    std::deque<work_item_t> mTodo;
    std::vector<work_item_t> mTotal;
    ItemProcessor * mProc;
};

#endif //PART2_HARNESS_H
