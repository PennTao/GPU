#include "part1_harness.h"
#include "part1.h"

#include <cutil_inline.h>

RunDriver::RunDriver(Run * run,
        int warmup_iters,
        int test_iters) : mRun(run),
    mWarmupIters(warmup_iters),
    mTestIters(test_iters)
{
    CUT_SAFE_CALL(cutCreateTimer(&timer));
}

RunDriver::~RunDriver() 
{
    CUT_SAFE_CALL(cutDeleteTimer(timer));
}

float
RunDriver::timeRun()
{
    for (int i = 0; i < mWarmupIters; ++i) {
        mRun->setupOp();
        mRun->performOp();
        mRun->teardownOp();
    }
    
    float totalTime = 0.0f;

    for (int i = 0; i < mTestIters; ++i) {
        mRun->setupOp();
        CUT_SAFE_CALL(cutResetTimer(timer));
        CUT_SAFE_CALL(cutStartTimer(timer));
        mRun->performOp();
        cutilSafeCall(cudaThreadSynchronize());
        CUT_SAFE_CALL(cutStopTimer(timer));
        mRun->teardownOp();
        totalTime += cutGetTimerValue(timer);
    }

    float avgTime = totalTime / mTestIters;
    return avgTime;
}

bool
RunDriver::testRun()
{
    mRun->setupOp();
    mRun->performOp();
    bool ret = mRun->checkOp();
    mRun->teardownOp();
    return ret;
}

const char * thr_nm = "Thrust";
const char * stl_nm = "Stl";
const char * std_nm = "Stdlib";

const char * 
sortName(SortingRoutine sort) {
    switch(sort) {
        case(Thrust):
            return thr_nm;
        case(Stdlib):
            return std_nm;
        case(Stl):
            return stl_nm;
        default:
            return NULL;
    }
}

const char * us_nm = "unsigned short";
const char * i_nm = "int";
const char * f_nm = "float";

template<typename K>
const char * typeName() 
{
    return NULL;
}

template<>
const char * typeName<unsigned short>()
{
    return us_nm;
}

template <>
const char * typeName<float>()
{
    return f_nm;
}

template <>
const char * typeName<int>()
{
    return i_nm;
}

#define CHECK(x) x ? "PASSED" : "FAILED"

template<typename K>
void runType(SortingRoutine sort)
{
    printf("Sort: %s\n", sortName(sort));
    printf("Datatype: %s\n", typeName<K>());
    printf("%14s, %14s, %14s, %14s, %14s\n",
            "Length",
            "Hst2Hst",
            "Hst2Dev",
            "Dev2Hst",
            "Dev2Dev");
    fflush(stdout);
    SortRun<K> testh2h(sort,
        Host,
        Host,
        1 << 10);
    SortRun<K> testh2d(sort,
        Host,
        Device,
        1 << 10);
    SortRun<K> testd2h(sort,
        Device,
        Host,
        1 << 10);
    SortRun<K> testd2d(sort,
        Device,
        Device,
        1 << 10);

    RunDriver dr_tsth2h(&testh2h,0,1);
    printf("%14s", "Correct");
    printf("%14s, ", CHECK(dr_tsth2h.testRun()));
    fflush(stdout);
    
    RunDriver dr_tsth2d(&testh2d,0,1);
    printf("%14s, ", CHECK(dr_tsth2d.testRun()));
    fflush(stdout);
 
    RunDriver dr_tstd2h(&testd2h,0,1);
    printf("%14s, ", CHECK(dr_tstd2h.testRun()));
    fflush(stdout);
 
    RunDriver dr_tstd2d(&testd2d,0,1);
    printf("%14s\n", CHECK(dr_tstd2d.testRun()));
    fflush(stdout);
    
    for (int sz = 10; sz < 24; ++sz) {
        SortRun<K> h2h(sort,
                Host,
                Host,
                1 << sz);
        SortRun<K> h2d(sort,
                Host,
                Device,
                1 << sz);
        SortRun<K> d2h(sort,
                Device,
                Host,
                1 << sz);
        SortRun<K> d2d(sort,
                Device,
                Device,
                1 << sz);

        RunDriver dr_h2h(&h2h,0,1);
        printf("%14d", 1 << sz);
        printf("%12.4fms, ", (dr_h2h.timeRun()));
        fflush(stdout);

        RunDriver dr_h2d(&h2d,0,1);
        printf("%12.4fms, ", (dr_h2d.timeRun()));
        fflush(stdout);

        RunDriver dr_d2h(&d2h,0,1);
        printf("%12.4fms, ", (dr_d2h.timeRun()));
        fflush(stdout);

        RunDriver dr_d2d(&d2d,0,1);
        printf("%12.4fms\n", (dr_d2d.timeRun()));
        fflush(stdout);

    }


    printf("\n\n");
}

int main(int argc, char ** argv)
{
   SortingRoutine sorts [] = {Thrust, Stdlib, Stl};
   for (int s = 0; s < sizeof(sorts)/sizeof(SortingRoutine); ++s) {
//	int s = 0;
        runType<unsigned short>(sorts[s]);
        runType<int>(sorts[s]);
        runType<float>(sorts[s]);
   }
}
