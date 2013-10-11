#include "part3.h"

#include "part3_harness.h"
#include "part3_kernel.h"


///////////////////////////////////////
////////      GENERAL      ////////////
///////////////////////////////////////
void 
MatrixPerfTest::warmup(int iters)
{
    for (int i = 0; i < iters; i++) {
        setupOp();
        performOp();
        teardownOp();
    }
}

float 
MatrixPerfTest::timeOp()
{
    warmup(mWarmupIters);
    float accum_time = 0.0f;
    for (int it = 0; it < mTestIters; it++) {
        accum_time += timeIter();
    }
    accum_time /= mTestIters;
    return accum_time;
}

float
MatrixPerfTest::timeIter() 
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    setupOp();

    cudaEventRecord(start, 0);
    performOp();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    teardownOp();
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    return elapsedTime;
}





/////////////////////////////////////////
/////////       SETUP      //////////////
/////////////////////////////////////////
void
BasicTest::setupOp()
{
    cutilSafeCall(cudaMalloc((void**)&mDIData, sizeof(float)*mRows*mCols));
    cutilSafeCall(cudaMalloc((void**)&mDOData, sizeof(float)*mRows*mCols));
    randomInit(mDIData, mRows, mCols);
    zeroInit(mDOData, mRows, mCols);
}

void
BasicTest::teardownOp() 
{
    cutilSafeCall(cudaFree(mDIData));
    cutilSafeCall(cudaFree(mDOData));
}

void IndexedTest::setupOp() 
{
    cutilSafeCall(cudaMalloc((void**)&mDIData, sizeof(float)*mRows*mCols));
    cutilSafeCall(cudaMalloc((void**)&mDOData, sizeof(float)*mRows*mCols));
    cutilSafeCall(cudaMalloc((void**)&mDWriteInds, sizeof(int)*mCols*mRows));
    randomInit(mDIData, mRows, mCols);
    zeroInit(mDOData, mRows, mCols);
    randomPermInit(mDWriteInds, mRows, mCols);
}

void IndexedTest::teardownOp()
{
    cutilSafeCall(cudaFree(mDIData));
    cutilSafeCall(cudaFree(mDOData));
    cutilSafeCall(cudaFree(mDWriteInds));
}





////////////////////////////////////////////
////////     CORRECTNESS TESTS    //////////
////////////////////////////////////////////
bool CopyTest::testOp()
{
    setupOp();
    gold_copy(mDIData,
        mDOData,
        mRows, mCols);
    float * h_gold = uploadToHost(mDOData,mRows,mCols);
    zeroInit(mDOData, mRows, mCols);
    performOp();
    float * h_test = uploadToHost(mDOData,mRows,mCols);
    teardownOp();
    bool ret = check_matrix_eq(h_gold, h_test, mRows, mCols, 1e-3);
    freeHost(h_gold);
    freeHost(h_test);
    return ret;
}

bool TransposeTest::testOp()
{
    setupOp();
    gold_transpose(mDIData,
        mDOData,
        mRows, mCols);
    float * h_gold = uploadToHost(mDOData,mRows,mCols);
    zeroInit(mDOData, mRows, mCols);
    performOp();
    float * h_test = uploadToHost(mDOData,mRows,mCols);
    teardownOp();
    bool ret = check_matrix_eq(h_gold, h_test, mRows, mCols, 1e-3);
    freeHost(h_gold);
    freeHost(h_test);
    return ret;
}

bool ScatterTest::testOp()
{
    setupOp();
    gold_scatter(mDIData,
        mDOData,
        mDWriteInds,
        mRows, mCols);
    float * h_gold = uploadToHost(mDOData,mRows,mCols);
    zeroInit(mDOData, mRows, mCols);
    performOp();
    float * h_test = uploadToHost(mDOData,mRows,mCols);
    teardownOp();
    bool ret = check_matrix_eq(h_gold, h_test, mRows, mCols, 1e-3);
    freeHost(h_gold);
    freeHost(h_test);
    return ret;
}



///////////////////////////////////////
///////    LAUNCH WRAPPERS    /////////
///////////////////////////////////////
void 
NaiveCopyTest::performOp()
{
    launch_naive_copy(mDIData,
            mDOData,
            mRows, mCols);
}

void 
OptimizedCopyTest::performOp()
{
    launch_optimized_copy(mDIData,
            mDOData,
            mRows, mCols);
}

void 
NaiveTransposeTest::performOp()
{
    launch_naive_transpose(mDIData,
            mDOData,
            mRows, mCols);
}


void 
OptimizedTransposeTest::performOp()
{
    launch_optimized_transpose(mDIData,
            mDOData,
            mRows, mCols);
}

void 
NaiveScatterTest::performOp()
{
    launch_naive_scatter(mDIData,
            mDOData,
            mDWriteInds,
            mRows, mCols);
}

void 
OptimizedScatterTest::performOp()
{
    launch_optimized_scatter(mDIData,
            mDOData,
            mDWriteInds,
            mRows, mCols);
}


#define UNUSED(x) (void)x

////////////////////////////////////
///////      ENTRY POINT    ////////
////////////////////////////////////
int main(int argc, char ** argv) 
{
    UNUSED(argc);
    UNUSED(argv);
    int warmup_iters = 1;
    int test_iters = 3;
    int rows = 2048;
    int cols = 2048;
    
    NaiveCopyTest nc(warmup_iters, test_iters, rows, cols);
    NaiveTransposeTest nt(warmup_iters, test_iters, rows, cols);
    NaiveScatterTest ns(warmup_iters, test_iters, rows, cols);
    OptimizedCopyTest oc(warmup_iters, test_iters, rows, cols);
    OptimizedScatterTest os(warmup_iters, test_iters, rows, cols);
    OptimizedTransposeTest ot(warmup_iters, test_iters, rows, cols);

    printf("Part 3 Correctness:\n");
    printf("%10s, %10s, %10s, %10s\n", "Version", "Copy", "Transpose", "Scatter");
    printf("%10s, %10s, %10s, %10s\n", 
            "Naive",
            CHECK(nc.testOp()),
            CHECK(nt.testOp()),
            CHECK(ns.testOp()));
    printf("%10s, %10s, %10s, %10s\n",
            "Optim",
            CHECK(oc.testOp()),
            CHECK(ot.testOp()),
            CHECK(os.testOp()));
    printf("\n");

    printf("Part 3 Timings:\n");
    printf("%10s, %10s, %10s, %10s\n", "Version", "Copy", "Transpose", "Scatter");
    printf("%10s, %8.4fms, %8.4fms, %8.4fms\n", "Naive", 
            nc.timeOp(), nt.timeOp(), ns.timeOp());

    printf("%10s, %8.4fms, %8.4fms, %8.4fms\n", "Optim", 
            oc.timeOp(), ot.timeOp(), os.timeOp());
}