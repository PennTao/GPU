
#ifndef PART2_KERNEL_H
#define PART2_KERNEL_H
#define NUM_STREAMS 3
#include <cutil_inline.h>
#include <cutil_math.h>
#include <nvMatrix.h>

typedef struct {
    float4 * pinned_position;
    float4 * pinned_blendweights; 
} hst_attributes_t;

typedef struct {
    float4 rows[4];
} mat4;

typedef struct {
    float4 cols[4];
} transp_mat4;

typedef struct {
    float4 * position;
    float4 * blendweights; 
} dev_attributes_t;

typedef struct {
    hst_attributes_t attr;
    float4 * dev_output;
    int len;
    float4 * answer_key;
} work_item_t;

typedef struct {
    mat4 transforms[4];
} matrix_palette_t;

//Note on extern "C"
//I like to keep my CUDA files as small as possible
//Since they take forever to build
//disagree with syntax highlighting/debuggers 
//etc.
//However, to call CUDA code from plain-jane C++,
//We need some kind of common ground, since different C++
//compilers have very different name conventions
//Hence the C interface

extern "C"
void load_palette(matrix_palette_t palette);

//To allow the kernels to be called from Cpp,
//All the relevant state of the Processors should
//be wrapped in a basic C struct.
//This allows it to be manipulated on the
//far side of an cdecl function
typedef struct {
    int num;
    int num_run;
    dev_attributes_t * dev_attrs;
} basic_state_t;

extern "C"
void launch_basic(basic_state_t * self, work_item_t item);

//See basic_state_t note
typedef struct {
    int num;
    int num_run;
    dev_attributes_t * dev_attrs;
    cudaStream_t * streams;
} async_state_t;

extern "C"
void launch_async(async_state_t * self, work_item_t item);

//See basic_state_t note
typedef struct {
    int num;
    int num_run;
    int num_streams;
    int stream_width;
    dev_attributes_t * dev_attrs;
    cudaStream_t * streams;
} streaming_state_t;

extern "C"
void launch_streaming(streaming_state_t * self, work_item_t item);

//See basic_state_t note
typedef struct {

} mapped_state_t;

extern "C"
void launch_mapped(mapped_state_t * self, work_item_t item);

#endif //PART2_KERNEL_H
