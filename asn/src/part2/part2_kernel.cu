#include "part2_kernel.h"

__constant__ matrix_palette_t dev_palette;

__device__ float4
mat4_mult(mat4 mult, float4 in) 
{
    return make_float4(dot(mult.rows[0], in),
        dot(mult.rows[1], in),
        dot(mult.rows[2], in),
        dot(mult.rows[3], in));
}

__device__ float4
transp_mat4_mult(transp_mat4 mult, float4 in) 
{
    return mult.cols[0]*in.x +
        mult.cols[1]*in.y + 
        mult.cols[2]*in.z + 
        mult.cols[3]*in.w;
}


__global__ void soft_skinning(dev_attributes_t dev_attr, 
        float4 * dev_output, int start_ind)
{
    unsigned int off_set = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int ind = start_ind + off_set;

    float4 pos = dev_attr.position[ind];
    float4 weights = dev_attr.blendweights[ind];
    transp_mat4 transed;
    for (int i = 0; i < 4; ++i) {
        mat4 trans = dev_palette.transforms[i];
        transed.cols[i] =  mat4_mult(trans, pos);
    }
    dev_output[ind] = transp_mat4_mult(transed, weights);
}


extern "C"
void load_palette(matrix_palette_t palette)
{
    cutilSafeCall(cudaMemcpyToSymbol("dev_palette", 
                &palette, 
                sizeof(matrix_palette_t),
                0, cudaMemcpyHostToDevice));
}

extern "C"
void launch_basic(basic_state_t * self, work_item_t item)
{
    int len = item.len;
    float4 * dev_out = item.dev_output;
    dev_attributes_t dev_attr = self->dev_attrs[self->num_run++];
    hst_attributes_t hst_attr = item.attr;
    cutilSafeCall(cudaMemcpy(dev_attr.position,
        hst_attr.pinned_position,
        sizeof(float4)*len,
        cudaMemcpyHostToDevice));
    cutilSafeCall(cudaMemcpy(dev_attr.blendweights,
        hst_attr.pinned_blendweights,
        sizeof(float4)*len,
        cudaMemcpyHostToDevice));
    dim3 block(128,1,1);
    dim3 grid(len / block.x,1,1);
    soft_skinning<<<grid,block,0>>>(dev_attr, dev_out, 0);
}

extern "C"
void launch_async(async_state_t * self, work_item_t item)
{
	///////////////// TODO /////////////////
	int len = item.len;
    float4 * dev_out = item.dev_output;
    dev_attributes_t dev_attr = self->dev_attrs[self->num_run++];
    hst_attributes_t hst_attr = item.attr;

	cutilSafeCall(cudaMemcpyAsync(dev_attr.position,hst_attr.pinned_position,sizeof(float4)*len,cudaMemcpyHostToDevice,*self->streams));
	cutilSafeCall(cudaMemcpyAsync(dev_attr.blendweights,hst_attr.pinned_blendweights,sizeof(float4)*len,cudaMemcpyHostToDevice,*self->streams));
	dim3 block(128,1,1);
	dim3 grid(len / block.x,1,1);
	soft_skinning<<<grid,block,0,*self->streams>>>(dev_attr, dev_out, 0);
	
}

extern "C"
void launch_streaming(streaming_state_t * self, work_item_t item)
{
	///////////////// TODO /////////////////
	int len = item.len;
    float4 * dev_out = item.dev_output;
    dev_attributes_t dev_attr = self->dev_attrs[self->num_run++];
    hst_attributes_t hst_attr = item.attr;
	for(int i = 0; i<NUM_STREAMS; i++)
	{
		cutilSafeCall(cudaMemcpyAsync(dev_attr.position,hst_attr.pinned_position,sizeof(float4)*len,cudaMemcpyHostToDevice,self->streams[i]));
	}
	for(int i = 0; i<NUM_STREAMS; i++)
	{
		cutilSafeCall(cudaMemcpyAsync(dev_attr.blendweights,hst_attr.pinned_blendweights,sizeof(float4)*len,cudaMemcpyHostToDevice,self->streams[i]));
	}
		dim3 block(128,1,1);
		dim3 grid(len / block.x,1,1);
	for(int i = 0; i<NUM_STREAMS; i++)
	{
		soft_skinning<<<grid,block,0,self->streams[i]>>>(dev_attr, dev_out, 0);
	}
}

extern "C"
void launch_mapped(mapped_state_t * self, work_item_t item)
{
	///////////////// TODO /////////////////
	int len = item.len;
	dev_attributes_t dev_attr;
//	float4 * dev_attr_pos;
//	float4 * dev_attr_blendw;
	//hst_attributes_t hst_attr = item.attr;
	cudaHostGetDevicePointer((void**)&dev_attr.position,(void*)item.attr.pinned_position,0);
	cudaHostGetDevicePointer((void**)&dev_attr.blendweights,(void*)item.attr.pinned_blendweights,0);
	
	dim3 block(128,1,1);
	dim3 grid(len/block.x,1,1);
	soft_skinning<<<grid,block,0>>>(dev_attr, item.dev_output,0);
}
