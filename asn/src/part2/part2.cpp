#include "part2.h"
//#define NUM_STREAMS 3

///////////////BASIC PROCESSOR///////
void
BasicProcessor::push(work_item_t item) 
{
    launch_basic(&state, item);
}

void
BasicProcessor::clear()
{
    for (int i = 0; i < state.num; ++i) {
        cutilSafeCall(cudaFree(state.dev_attrs[i].position));
        cutilSafeCall(cudaFree(state.dev_attrs[i].blendweights));
    }
    free(state.dev_attrs);
    state.dev_attrs = NULL;
    state.num = 0;
}

void
BasicProcessor::prepare(int num_to_expect, int len) //num_to_expect = num_items =16 in part2_kernel.cu
{
    state.num = num_to_expect;
    state.num_run = 0;
    state.dev_attrs = (dev_attributes_t*)
        malloc(sizeof(dev_attributes_t)*num_to_expect);
    for (int i = 0; i < num_to_expect; ++i) {
        cutilSafeCall(cudaMalloc(
            &(state.dev_attrs[i].position),
            sizeof(float4)*len));
        cutilSafeCall(cudaMalloc(
            &(state.dev_attrs[i].blendweights),
            sizeof(float4)*len));
    }
}

#define UNUSED(x) (void)(x)


///////////////ASYNC PROCESSOR///////
void
AsyncProcessor::push(work_item_t item) 
{
    launch_async(&state, item);
}

void
AsyncProcessor::clear()
{
////////////////// TODO (maybe) ////////////////////
	for (int i = 0; i < state.num; ++i) {
        cutilSafeCall(cudaFree(state.dev_attrs[i].position));
        cutilSafeCall(cudaFree(state.dev_attrs[i].blendweights));
    }
    free(state.dev_attrs);
    state.dev_attrs = NULL;
    state.num = 0;
	
	cudaStreamDestroy(*state.streams);
	


}

void
AsyncProcessor::prepare(int num_to_expect, int len)
{
////////////////// TODO ////////////////////

	// Pragma to shut up compiler in this stub:
	// DELETE THIS
	state.num = num_to_expect;
    state.num_run = 0;
    state.dev_attrs = (dev_attributes_t*)
        malloc(sizeof(dev_attributes_t)*num_to_expect);
    for (int i = 0; i < num_to_expect; ++i) {
        cutilSafeCall(cudaMalloc(
            &(state.dev_attrs[i].position),
            sizeof(float4)*len));
        cutilSafeCall(cudaMalloc(
            &(state.dev_attrs[i].blendweights),
            sizeof(float4)*len));
	}
	state.streams = (cudaStream_t *)malloc(sizeof(cudaStream_t));
	
	cudaStreamCreate(state.streams);

}



///////////////STREAMING PROCESSOR///////
void
StreamingProcessor::push(work_item_t item) 
{
    launch_streaming(&state, item);
}

void
StreamingProcessor::clear()
{
////////////////// TODO (maybe) ////////////////////
	for (int i = 0; i < state.num; ++i) {
        cutilSafeCall(cudaFree(state.dev_attrs[i].position));
        cutilSafeCall(cudaFree(state.dev_attrs[i].blendweights));
    }
    free(state.dev_attrs);
    state.dev_attrs = NULL;
    state.num = 0;

	for (int i = 0; i < state.num_streams; ++i)
	{
		cudaStreamDestroy(state.streams[i]);
	}
	state.num_streams = 1;
	
}

void
StreamingProcessor::prepare(int num_to_expect, int len)
{
////////////////// TODO ////////////////////
	state.num = num_to_expect;
    state.num_run = 0;
	state.stream_width = 1<<13;
	state.num_streams = len / state.stream_width;
	if(state.num_streams == 0)
		state.num_streams = 1;
    state.dev_attrs = (dev_attributes_t*)
        malloc(sizeof(dev_attributes_t)*num_to_expect);
    for (int i = 0; i < num_to_expect; ++i) {
        cutilSafeCall(cudaMalloc(
            &(state.dev_attrs[i].position),
            sizeof(float4)*len));
        cutilSafeCall(cudaMalloc(
            &(state.dev_attrs[i].blendweights),
            sizeof(float4)*len));
	}
	state.streams = (cudaStream_t *)malloc(sizeof(cudaStream_t)*state.num_streams);
	for (int i = 0; i<state.num_streams; i++)
	{
		cudaStreamCreate(&state.streams[i]);
	}

}


///////////////MAPPED PROCESSOR////////
void
MappedProcessor::push(work_item_t item) 
{
    launch_mapped(&state, item);
}

void
MappedProcessor::clear()
{
////////////////// TODO (maybe) ////////////////////
}

void
MappedProcessor::prepare(int num_to_expect, int len)
{
////////////////// TODO ////////////////////
	// Pragma to shut up compiler in this stub:
	// DELETE THIS
	UNUSED(num_to_expect);
	UNUSED(len);
}
