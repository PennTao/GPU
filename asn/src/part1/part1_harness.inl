#ifndef PART1_HARNESS_INL
#define PART1_HARNESS_INL

#include <limits>
#include <cutil_inline.h>
#include <iostream>

template<typename K>
K randomSingle();

template<>
int randomSingle<int>()
{
    return rand();
}

template<>
unsigned short randomSingle<unsigned short>()
{
    return rand() % std::numeric_limits<unsigned short>::max();
}

template<>
float randomSingle<float>()
{
    return ((float)rand()) / RAND_MAX;
}

template <typename K>
void randomInit(K * h_data, int num)
{
    for (int i = 0; i < num; ++i) {
        h_data[i] = randomSingle<K>();
    }
}

template <typename K>
void zeroInit(K * h_data, int num)
{
    for (int i = 0; i < num; ++i) {
        h_data[i] = 0; //Works due to numerical conversion
    }
}

template <typename K>
K * downloadToDev(K * h_data, int num)
{
    K * dev;
    cutilSafeCall(cudaMalloc(&dev, sizeof(K)*num));
    cutilSafeCall(cudaMemcpy(dev,h_data,
                sizeof(K)*num,
                cudaMemcpyHostToDevice));
    return dev;
}

template <typename K>
void SortRun<K>::performOp()
{
    switch(mSort) {
        case(Thrust):
            thrustWrapper(mIdata,
                    mOdata,
                    mNum,
                    mSrc,
                    mDst);
            break;
        case(Stl):
            stlWrapper(mIdata,
                    mOdata,
                    mNum,
                    mSrc,
                    mDst);
            break;
        case(Stdlib):
            qsortWrapper(mIdata,
                    mOdata,
                    mNum,
                    mSrc,
                    mDst);
            break;
        default:
            throw "Bad Sort Routine!";
    }
}

template <typename K>
void SortRun<K>::setupOp()
{
    if (mSrc == Host) {
        mIdata = (K*)malloc(sizeof(K)*mNum);
        randomInit(mIdata, mNum);
    } else if (mSrc == Device) {
        K * tmp = (K*)malloc(sizeof(K)*mNum);
        randomInit(tmp, mNum);
        mIdata = downloadToDev(tmp, mNum);
        free(tmp);
    }
    

    if (mDst == Host) {
        mOdata = (K*)malloc(sizeof(K)*mNum);
        randomInit(mOdata,mNum);
    } else if (mDst == Device) {
        K * tmp = (K*)malloc(sizeof(K)*mNum);
        randomInit(tmp, mNum);
        mOdata = downloadToDev(tmp, mNum);
        free(tmp);
    }
}

template <typename K>
bool SortRun<K>::checkOp()
{
    K * tmp;
    if (mDst == Host) {
        tmp = mOdata;
    } else if (mDst == Device) {
        tmp = (K*)malloc(sizeof(K)*mNum);
        cutilSafeCall(cudaMemcpy(tmp,
                    mOdata,
                    sizeof(K)*mNum,
                    cudaMemcpyDeviceToHost));
    }
    bool pass = true;
    for (int i = 0; i < mNum - 1; ++i) {
        pass = pass && (tmp[i] <= tmp[i + 1]);
    }
    if (mDst == Device) {
        free(tmp);
    }
    return pass;
}

template<typename K>
void SortRun<K>::teardownOp()
{
    if (mSrc == Host) {
        free(mIdata);
    } else if (mSrc == Device) {
        cutilSafeCall(cudaFree(mIdata));
    }

    if (mDst == Host) {
        free(mOdata);
    } else if (mDst == Device) {
        cutilSafeCall(cudaFree(mOdata));
    }
    
    mIdata = NULL;
    mOdata = NULL;
}

#endif //PART1_HARNESS_INL