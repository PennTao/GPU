#ifndef PART3_HARNESS_INL
#define PART3_HARNESS_INL

#include <cutil_inline.h>

template<typename K>
K * uploadToHost(K * dev_data,
        int rows, int cols)
{
    K * hst_data = (K*)malloc(sizeof(K)*rows*cols);
    cutilSafeCall(cudaMemcpy(hst_data,
                dev_data,
                sizeof(K)*rows*cols, 
                cudaMemcpyDeviceToHost));
    return hst_data;
}

//Unnecesary if you used downloadToDev
template<typename K>
void freeHost(K * hst_data)
{
    free(hst_data);
}

template<typename K>
void downloadToDev(K * hst_data,
        K * dev_data,
        int rows, int cols)
{
    cutilSafeCall(cudaMemcpy(dev_data, 
                hst_data,
                sizeof(K)*rows*cols, 
                cudaMemcpyHostToDevice));
    freeHost(hst_data);
}

#endif //PART3_HARNESS_INL
