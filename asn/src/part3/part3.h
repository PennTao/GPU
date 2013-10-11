#ifndef PART3_H
#define PART3_H

#include <stdlib.h>

class MatrixPerfTest {
public:
    float timeOp();
    virtual bool testOp() = 0;
    MatrixPerfTest(int warmup_iters,
            int test_iters) :
        mWarmupIters(warmup_iters),
        mTestIters(test_iters) {}
private:
    //test run the kernel a number of times to 
    //clear up any timing problems due to unitialized hardware
    //CUDA contexts, JIT compilation, etc.
    void warmup(int iters);
    float timeIter();
    int mWarmupIters;
    int mTestIters;
protected:
    virtual void setupOp() = 0;
    virtual void performOp() = 0;
    virtual void teardownOp() = 0;
    //virtual void fakeOp() = 0;
};

class BasicTest : public MatrixPerfTest {
public:
    BasicTest(int warmup_iters,
            int test_iters,
            int rows, int cols) :
        MatrixPerfTest(warmup_iters, test_iters),
        mRows(rows),mCols(cols),
        mDIData(NULL),
        mDOData(NULL) {}
protected:
    int mRows;
    int mCols;
    float * mDIData;
    float * mDOData;
    virtual void setupOp();
    virtual void teardownOp();
};

class IndexedTest : public MatrixPerfTest {
public:
    IndexedTest(int warmup_iters,
            int test_iters,
            int rows, int cols):
        MatrixPerfTest(warmup_iters, test_iters),
        mRows(rows), mCols(cols),
        mDIData(NULL),
        mDOData(NULL),
        mDWriteInds(NULL) {}
protected:
    int mRows;
    int mCols;
    float * mDIData;
    float * mDOData;
    int * mDWriteInds;
    virtual void setupOp();
    virtual void teardownOp();
};

class CopyTest : public BasicTest {
public:
    CopyTest(int warmup_iters,
            int test_iters,
            int rows, int cols): 
        BasicTest(warmup_iters,
                test_iters,
                rows, cols) {}
    virtual bool testOp();
}; 

class TransposeTest : public BasicTest {
public:
     TransposeTest(int warmup_iters,
            int test_iters,
            int rows, int cols): 
        BasicTest(warmup_iters,
                test_iters,
                rows, cols) {}   
     virtual bool testOp();
};

class ScatterTest : public IndexedTest {
public:
     ScatterTest(int warmup_iters,
            int test_iters,
            int rows, int cols): 
        IndexedTest(warmup_iters,
                test_iters,
                rows, cols) {}   
    virtual bool testOp();
}; 

class NaiveCopyTest : public CopyTest {
public:
     NaiveCopyTest(int warmup_iters,
            int test_iters,
            int rows, int cols): 
        CopyTest(warmup_iters,
                test_iters,
                rows, cols) {}   
protected:

    virtual void performOp();
}; 

class OptimizedCopyTest : public CopyTest {
public:
     OptimizedCopyTest(int warmup_iters,
            int test_iters,
            int rows, int cols): 
        CopyTest(warmup_iters,
                test_iters,
                rows, cols) {}   
protected:
    virtual void performOp();
};

class NaiveTransposeTest : public TransposeTest {
public:
     NaiveTransposeTest(int warmup_iters,
            int test_iters,
            int rows, int cols): 
        TransposeTest(warmup_iters,
                test_iters,
                rows, cols) {}   

protected:
    virtual void performOp();
};

class OptimizedTransposeTest : public TransposeTest {
public:
     OptimizedTransposeTest(int warmup_iters,
            int test_iters,
            int rows, int cols): 
        TransposeTest(warmup_iters,
                test_iters,
                rows, cols) {}   

protected:
    virtual void performOp();
};

class NaiveScatterTest : public ScatterTest {
public:
     NaiveScatterTest(int warmup_iters,
            int test_iters,
            int rows, int cols): 
        ScatterTest(warmup_iters,
                test_iters,
                rows, cols) {}   

protected:
    virtual void performOp();
};

class OptimizedScatterTest : public ScatterTest {
public:
     OptimizedScatterTest(int warmup_iters,
            int test_iters,
            int rows, int cols): 
        ScatterTest(warmup_iters,
                test_iters,
                rows, cols) {}   
protected:
    virtual void performOp();
};

#endif //PART3_H
