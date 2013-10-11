#ifndef PART3_KERNEL_H
#define PART3_KERNEL_H

extern "C" {

void launch_naive_copy(float * dev_i_data, 
        float * dev_o_data,
        int rows, int cols);

void launch_optimized_copy(float * dev_i_data,
        float * dev_o_data,
        int rows, int cols);

void launch_naive_transpose(float * dev_i_data, 
        float * dev_o_data,
        int rows, int cols);

void launch_optimized_transpose(float * dev_i_data,
        float * dev_o_data,
        int rows, int cols); 

void launch_naive_scatter(float * dev_i_data, 
        float * dev_o_data,
        int * write_inds,
        int rows, int cols);

void launch_optimized_scatter(float * dev_i_data,
        float * dev_o_data,
        int * write_inds,
        int rows, int cols);

}

#endif //PART3_KERNEL_H
