#include "part2_harness.h"

work_item_t alloc_item(int len)
{
    work_item_t mk;
    mk.len = len;
    cutilSafeCall(cudaHostAlloc(
       &(mk.attr.pinned_position),
       sizeof(float4)*len,
       cudaHostAllocMapped));
    //Enables this memory for memory mapping
    cutilSafeCall(cudaHostAlloc(
       &(mk.attr.pinned_blendweights),
       sizeof(float4)*len,
       cudaHostAllocMapped));
    cutilSafeCall(cudaMalloc(
        &(mk.dev_output),
        sizeof(float4)*len));
    mk.answer_key = (float4*)
        malloc(sizeof(float4)*len);
    return mk;
}

void free_item(work_item_t item)
{
    cutilSafeCall(cudaFreeHost(
        item.attr.pinned_position));
    cutilSafeCall(cudaFreeHost(
        item.attr.pinned_blendweights));
    cutilSafeCall(cudaFree(
        item.dev_output));
    free(item.answer_key);
}

bool check_f4_eq(const float4 & v1,
        const float4 & v2,
        float tolerance)
{
    return abs(v1.x - v2.x) < tolerance &&
        abs(v1.y - v2.y) < tolerance &&
        abs(v1.z - v2.z) < tolerance &&
        abs(v1.w - v2.w) < tolerance;
}

bool check_item(work_item_t item) 
{
    float4 * tmp = (float4*)
        malloc(item.len*sizeof(float4));
    cutilSafeCall(cudaMemcpy(
        tmp, item.dev_output,
        item.len*sizeof(float4),
        cudaMemcpyDeviceToHost));
    bool pass = true;
    for (int i = 0; i < item.len; ++i) {
        pass = pass && check_f4_eq(tmp[i], item.answer_key[i], 0.001f);
    }
    free(tmp);
    return pass;
}

bool
ItemGenerator::hasNext()
{
    return mMade < mTotal;
}

work_item_t
ItemGenerator::yield()
{
    work_item_t mk = alloc_item(mLen);
    for (int i = 0; i < mLen; ++i) {
        float4 next_vec = make_float4(
                ((float)rand()) / RAND_MAX,
                ((float)rand()) / RAND_MAX,
                ((float)rand()) / RAND_MAX,
                1.0f);
         float4 next_bw = make_float4(
                ((float)rand()) / RAND_MAX,
                ((float)rand()) / RAND_MAX,
                ((float)rand()) / RAND_MAX,
                ((float)rand()) / RAND_MAX);
         float4 answer = goldTransform(next_vec, 
                 next_bw);
         mk.attr.pinned_position[i] = next_vec;
         mk.attr.pinned_blendweights[i] = next_bw;
         mk.answer_key[i] = answer;
    }
    ++mMade;
    return mk;
}

float4
h_mat4_mult(mat4 mult, float4 in) 
{
    return make_float4(dot(mult.rows[0], in),
        dot(mult.rows[1], in),
        dot(mult.rows[2], in),
        dot(mult.rows[3], in));
}

float4
h_transp_mat4_mult(transp_mat4 mult, float4 in) 
{
    return mult.cols[0]*in.x +
        mult.cols[1]*in.y + 
        mult.cols[2]*in.z + 
        mult.cols[3]*in.w;
}

float4 
ItemGenerator::goldTransform(const float4 & v,
        const float4 & bw) 
{
    transp_mat4 transed;
    for (int i = 0; i < 4; ++i) {
        transed.cols[i] = h_mat4_mult(mPalette.transforms[i], v);
    }
    return h_transp_mat4_mult(transed, bw);
}


RunDriver::RunDriver(Run * run,
        int warmup_iters,
        int test_iters) :
        mRun(run),
        mWarmupIters(warmup_iters),
        mTestIters(test_iters),
        timer(0)
{
        CUT_SAFE_CALL(cutCreateTimer(&timer));
}


RunDriver::~RunDriver() 
{
    CUT_SAFE_CALL(cutDeleteTimer(timer));
}

float
RunDriver::timeRun()
{
    for ( int i = 0; i < mWarmupIters; ++i) {
        mRun->setupOp();
        mRun->performOp();
        mRun->teardownOp();
    }
    
    float totalTime = 0.0f;

    for (int i = 0; i < mTestIters; ++i) {
        mRun->setupOp();
        CUT_SAFE_CALL(cutResetTimer(timer));
        CUT_SAFE_CALL(cutStartTimer(timer));
        mRun->performOp();
        CUT_SAFE_CALL(cutStopTimer(timer));
        mRun->teardownOp();
        totalTime += cutGetTimerValue(timer);
    }

    float avgTime = totalTime / mTestIters;
    return avgTime;
}

bool
RunDriver::testRun()
{
    mRun->setupOp();
    mRun->performOp();
    bool ret = mRun->checkOp();
    mRun->teardownOp();
    return ret;
}

ItemRun::ItemRun(ItemProcessor * proc) :
    mProc(proc) {}

void
ItemRun::setupOp()
{
    matrix_palette_t palette;
		// Initialize random palette
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				palette.transforms[i].rows[j] = make_float4(
										(float)rand()/(float)RAND_MAX,
										(float)rand()/(float)RAND_MAX,
										(float)rand()/(float)RAND_MAX,
										(float)rand()/(float)RAND_MAX);
			}
		}
    int num_items = 16;
    int length = 1 << 15;
    mProc->prepare(num_items, length);
    ItemGenerator gen(palette, num_items, length);
    load_palette(palette);
    while (gen.hasNext()) {
       work_item_t next = gen.yield();
       mTodo.push_back(next);
       mTotal.push_back(next);
    }
}

void
ItemRun::teardownOp()
{
    mProc->clear();
    for (int i = 0; i < (int)mTotal.size(); ++i) {
        free_item(mTotal[i]);
    }
    mTotal.clear();
    mTodo.clear();
}

void
ItemRun::performOp()
{
    while (mTodo.size() > 0) {
        mProc->push(mTodo[0]);
        mTodo.pop_front();
    }
    cutilSafeCall(cudaThreadSynchronize());
}

bool
ItemRun::checkOp()
{
    for (int i = 0; i < (int)mTotal.size(); ++i) {
       if (!check_item(mTotal[i])) {
            return false; }
    }
    return true;
}

void
cudaSetup() 
{
    cudaSetDeviceFlags(cudaDeviceMapHost);
}

int main(int , char ** )
{

    cudaSetup();

    int warmup_iters = 0;
    int test_iters = 1;

    BasicProcessor bp;
    AsyncProcessor ap;
    StreamingProcessor sp;
    MappedProcessor mp;
    
    ItemRun br(&bp);
    ItemRun ar(&ap);
    ItemRun sr(&sp);
    ItemRun mr(&mp);

    RunDriver bd(&br, warmup_iters, test_iters);
    RunDriver ad(&ar, warmup_iters, test_iters);
    RunDriver sd(&sr, warmup_iters, test_iters);
    RunDriver md(&mr, warmup_iters, test_iters);
        
    printf("Part 2 Correctness\n");
    printf("%10s, %10s, %10s, %10s\n",
            "Basic",
            "Async",
            "Streaming",
            "Mapped");

    fflush(stdout);
    printf("%10s, ",  CHECK(bd.testRun()));
    fflush(stdout);
    printf("%10s, ",  CHECK(ad.testRun()));
    fflush(stdout);
    printf("%10s, ",  CHECK(sd.testRun()));
    fflush(stdout);
    printf("%10s\n",  CHECK(md.testRun()));
    fflush(stdout);

    printf("Part 2 Timings\n");
    printf("%10s, %10s, %10s, %10s\n",
            "Basic",
            "Async",
            "Streaming",
            "Mapped");
    fflush(stdout);

    printf("%10.4fms, ",  bd.timeRun());
    fflush(stdout);
    printf("%10.4fms, ",  ad.timeRun());
    fflush(stdout);
    printf("%10.4fms, ",  sd.timeRun());
    fflush(stdout);
    printf("%10.4fms\n",  md.timeRun());
    fflush(stdout);

}
