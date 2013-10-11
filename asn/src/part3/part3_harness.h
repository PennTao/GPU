#ifndef PART3_HARNESS_H
#define PART3_HARNESS_H

#define CHECK(p) (p ? "PASSED" : "FAILED")

void gold_copy(float * dev_i_data,
        float * dev_o_data,
        int rows, int cols);

void gold_transpose(float * dev_i_data,
        float * dev_o_data,
        int rows, int cols);

void gold_scatter(float * dev_i_data,
        float * dev_o_data,
        int * write_inds,
        int rows, int cols);

void randomInit(float * dev_data,
        int rows, int cols);

void zeroInit(float * dev_data,
        int rows, int cols);

void randomPermInit(int * dev_data,
        int rows, int cols);

bool check_matrix_eq(float * m1,
        float * m2, int rows, 
        int cols, float tol);

template<typename K>
K * uploadToHost(K * dev_data,
        int rows, int cols);

template<typename K>
void freeHost(K * hst_data);

template<typename K>
void downloadToDev(K * hst_data,
        K * dev_data,
        int rows, int cols);


#include "part3_harness.inl"

#endif //PART3_HARNESS_H
