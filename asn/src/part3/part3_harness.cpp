#include "part3_harness.h"

#include "curand.h"

bool check_matrix_eq(float * m1, 
        float * m2,
        int rows, int cols,
        float tol) 
{
    int num = rows*cols;
    bool pass = true;
    for (int i = 0; i < num; ++i) {
        pass = pass && (fabs(m1[i] - m2[i]) < tol);
    }
    return pass;
}

void gold_copy(float * dev_i_data,
        float * dev_o_data,
        int rows, int cols)
{
    float * h_i_data = uploadToHost(dev_i_data,
            rows,cols);
    float * h_o_data = uploadToHost(dev_o_data,
            rows,cols);
    memcpy(h_o_data, h_i_data, sizeof(float)*rows*cols);
    downloadToDev(h_i_data, dev_i_data,
            rows,cols);
    downloadToDev(h_o_data, dev_o_data,
            rows,cols);
}

void gold_transpose(float * dev_i_data,
        float * dev_o_data,
        int rows, int cols)
{
    float * h_i_data = uploadToHost(dev_i_data,
            rows,cols);
    float * h_o_data = uploadToHost(dev_o_data,
            rows,cols);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            h_o_data[c*cols + r] = h_i_data[r*cols + c];
        }
    }
    downloadToDev(h_i_data, dev_i_data,
            rows,cols);
    downloadToDev(h_o_data, dev_o_data,
            rows,cols);
}

void gold_scatter(float * dev_i_data,
        float * dev_o_data,
        int * dev_write_inds,
        int rows, int cols)
{
    float * h_i_data = uploadToHost(dev_i_data,
            rows,cols);
    float * h_o_data = uploadToHost(dev_o_data,
            rows,cols);
    int * h_wr_inds =  uploadToHost(dev_write_inds,
            rows,cols);
    for (int i = 0; i < rows*cols; ++i) {
        h_o_data[h_wr_inds[i]] = h_i_data[i];
    }
    downloadToDev(h_i_data, dev_i_data,
            rows,cols);
    downloadToDev(h_o_data, dev_o_data,
            rows,cols);
    downloadToDev(h_wr_inds, dev_write_inds,
            rows, cols);
}

#define CURAND_CALL

void randomInit(float * dev_data,
        int rows, int cols)
{
    int num = rows*cols;
    float * tmp = (float*) malloc(sizeof(float)*num);
    for (int i = 0; i < num; ++i) {
        tmp[i] = (float)rand() / RAND_MAX;
    }
    downloadToDev(tmp, dev_data,
            rows, cols);
}

void zeroInit(float * dev_data,
        int rows, int cols)
{
    cutilSafeCall(cudaMemset(dev_data, 0, rows*cols*sizeof(float)));
}

void swap(int * hst_data, int i, int j)
{
    int temp = hst_data[i];
    hst_data[i] = hst_data[j];
    hst_data[j] = temp;
}

void knuth_shuffle(int * hst_data, int num)
{
    for (int i = 0; i < num; ++i) {
        hst_data[i] = i;
    }

    for (int i = 0; i < num - 2; ++i) {
        int excl_top = (num - 1) - i;
        int off = rand() % excl_top;
        int targ = i + off;
        swap(hst_data, i, targ);
    }
}

void randomPermInit(int * dev_data,
        int rows, int cols) {
    int * hst_data = uploadToHost(dev_data,
            rows, cols);
    knuth_shuffle(hst_data, rows*cols);
    downloadToDev(hst_data, dev_data,
            rows, cols);
}

