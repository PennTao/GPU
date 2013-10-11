#include "part3_kernel.h"

__global__ void naive_copy(float * i_data,
		float * o_data,
		int rows, int cols)
{
	unsigned int x = blockDim.x*blockIdx.x + threadIdx.x;
	unsigned int y = blockDim.y*blockIdx.y + threadIdx.y;
	o_data[y*rows + x] = i_data[y*rows + x];
}

__global__ void optimized_copy(float * i_data,
		float * o_data,int rows, int cols)			//no need to change kernel, change block size.
{
	//////////// TODO //////////////
//	unsigned int x = blockDim.x*blockIdx.x + threadIdx.x;
	
//	unsigned int y = blockDim.y*blockIdx.y + threadIdx.y;
//	o_data[]= i_data[]
	
	unsigned int x = blockDim.x*blockIdx.x + threadIdx.x;
	unsigned int y = blockDim.y*blockIdx.y + threadIdx.y;

	o_data[y*rows + x] = i_data[y*rows + x];
}

__global__ void naive_transpose(float * i_data,
		float * o_data,
		int rows, int cols)
{
	//////////// TODO ///////////////
	unsigned int x = blockDim.x*blockIdx.x + threadIdx.x;
	unsigned int y = blockDim.y*blockIdx.y + threadIdx.y;
	o_data[x*cols + y] = i_data[y*rows + x];

}


__global__ void optimized_transpose(float * i_data,
		float * o_data,
		int rows, int cols)					//mem coalescing + block size modification
{
	//////////// TODO ///////////////
	unsigned int x = blockDim.x*blockIdx.x + threadIdx.x;
	unsigned int y = blockDim.y*blockIdx.y + threadIdx.y;
	o_data[y*rows + x] = i_data[x*cols + y];

}

__global__ void naive_scatter(float * i_data,
		float * o_data,
		int * write_inds,
		int rows, int cols)
{
	//////////// TODO ///////////////	
	unsigned int x = blockDim.x*blockIdx.x + threadIdx.x;
	unsigned int y = blockDim.y*blockIdx.y + threadIdx.y;
	unsigned int i = y*rows + x;
	o_data[write_inds[i]] = i_data[i];
		//////////// TODO ///////////////
//	int excl_top = (rows*cols-1) - i;
//	int off = rand()% excl_top;
//	int targ = i +off;
//	float old = atomicExch(&i_data[targ],i_data[i]);
//	float temp = i_data[i];
//	i_data[i] = i_data[targ];
//	i_data[targ] = temp;
//	o_data[i] = old;
}


__global__ void optimized_scatter(float * i_data,
		float * o_data,
		int * write_inds,
		int rows, int cols)				//unable to optimized much
{
	//////////// TODO ///////////////
	unsigned int x = blockDim.x*blockIdx.x + threadIdx.x;
	unsigned int y = blockDim.y*blockIdx.y + threadIdx.y;
	unsigned int i = y*rows + x;
	o_data[write_inds[i]] = i_data[i];
}

extern "C" {

	void launch_naive_copy(float * dev_i_data, 
			float * dev_o_data,
			int rows, int cols)
	{
		//NAIVE MATRIX COPY 
		//straight copy of dev_i_data to dev_o_data
		dim3 block(8,8,1);
		dim3 grid(cols/block.x, rows/block.y,1);
		naive_copy<<<grid,block>>>(dev_i_data,
				dev_o_data,
				rows,cols);
	}

	void launch_optimized_copy(float * dev_i_data,
			float * dev_o_data,
			int rows, int cols) 
	{
		//OPTIMIZE MATRIX COPY 
		//straight copy of dev_i_data to dev_o_data
		//////////// TODO ///////////////
		
		dim3 block(16,16,1);
		dim3 grid(cols/block.x, rows/block.y,1);
		optimized_copy<<<grid,block>>>(dev_i_data,
				dev_o_data,rows,cols);
	}

	void launch_naive_transpose(float * dev_i_data, 
			float * dev_o_data,
			int rows, int cols)
	{
		//NAIVE MATRIX TRANPOSE 
		//rows of dev_i_data into columns of dev_o_data
		//////////// TODO ///////////////
		dim3 block(8,8,1);
		dim3 grid(cols/block.x, rows/block.y,1);
		naive_transpose<<<grid,block>>>(dev_i_data,
				dev_o_data,rows,cols);
	}

	void launch_optimized_transpose(float * dev_i_data,
			float * dev_o_data,
			int rows, int cols) 
	{
		//OPTIMIZE MATRIX TRANSPOSE
		//rows of dev_i_data into columns of dev_o_data
		//////////// TODO ///////////////
		dim3 block(4,4,1);
		dim3 grid(cols/block.x, rows/block.y,1);
		optimized_transpose<<<grid,block>>>(dev_i_data,
				dev_o_data,rows,cols);
	}

	void launch_naive_scatter(float * dev_i_data, 
			float * dev_o_data,
			int * write_inds,
			int rows, int cols)
	{
		//NAIVE SCATTERED WRITES
		//Pseudocode: dev_o_data[write_inds[index]] = dev_i_data[index]
		//////////// TODO ///////////////
		
		dim3 block(8,8,1);
		dim3 grid(cols/block.x, rows/block.y,1);
		naive_scatter<<<grid,block>>>(dev_i_data,
				dev_o_data,write_inds,rows,cols);

//		cudaMemcpy(dev_o_data,dev_i_data,sizeof(dev_i_data[0],cudaMemcpyDeviceToDevice));
	}

	void launch_optimized_scatter(float * dev_i_data,
			float * dev_o_data,
			int * write_inds,
			int rows, int cols) 
	{
		//OPTIMIZE SCATTERED WRITES 
		//pseudocode: dev_o_data[write_inds[index]] = dev_i_data[index]
		//////////// TODO ///////////////
		dim3 block(16,16,1);
		dim3 grid(cols/block.x, rows/block.y,1);
		optimized_scatter<<<grid,block>>>(dev_i_data,
				dev_o_data,write_inds,rows,cols);
	}

}
