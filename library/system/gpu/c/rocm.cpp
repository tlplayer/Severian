// Synchronous HIP system-call boundary. No Severian expressions or kernel code.

#include <hip/hip_runtime_api.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
static thread_local int64_t kernels, allocations, peak;
static thread_local double seconds;
extern "C" void __sev_gpu_begin() { kernels=allocations=peak=0; seconds=0; }
extern "C" int64_t __sev_gpu_kernels() { return kernels; }
extern "C" int64_t __sev_gpu_allocations() { return allocations; }
extern "C" double __sev_gpu_peak() { return (double)peak * 8.0; }
extern "C" double __sev_gpu_time() { return seconds; }
extern "C" int32_t __sev_gpu_execute(const unsigned char *image, void *input,
    void *output, int64_t n, int64_t width, void **captures, int64_t capture_count, int64_t input_count) {
    if (!n) return 0;
    void *a=nullptr, *b=nullptr;
    hipModule_t module=nullptr; hipFunction_t kernel=nullptr;
    hipEvent_t start=nullptr, stop=nullptr;
    hipStream_t stream=nullptr;
    hipError_t error=hipSuccess;
    float elapsed=0; size_t bytes=0; void **args=nullptr;
    const char *stage="validate extent";
    if(n<0 || (width!=4 && width!=8) || capture_count<0 ||
       (uint64_t)capture_count>SIZE_MAX/sizeof(void*)-3 || (capture_count && !captures) ||
       (input_count!=1 && input_count!=2) || !image || !input || !output ||
       (uint64_t)n>(uint64_t)INT64_MAX/(uint64_t)width/(uint64_t)(input_count+1) ||
       (uint64_t)n/256+((uint64_t)n%256!=0)>UINT_MAX) {
        return -1;
    }
    bytes=(size_t)n*(size_t)width;
    args=(void**)calloc((size_t)capture_count+3,sizeof(void*));
    if(!args) return -2;
#define CHECK(expr) do { stage=#expr; error=(expr); if(error!=hipSuccess) goto cleanup; } while(0)
    CHECK(hipInit(0));
    CHECK(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));
    CHECK(hipModuleLoadData(&module,image));
    CHECK(hipModuleGetFunction(&kernel,module,"map"));
    CHECK(hipMalloc(&a,bytes*input_count)); allocations++;
    CHECK(hipMalloc(&b,bytes)); allocations++;
    if ((uint64_t)peak < (input_count+1)*(uint64_t)bytes) peak=(input_count+1)*bytes;
    CHECK(hipMemcpyAsync(a,input,bytes*input_count,hipMemcpyHostToDevice,stream));
    CHECK(hipEventCreate(&start)); CHECK(hipEventCreate(&stop));
    args[0]=&a; args[1]=&b; args[2]=&n;
    for(int64_t i=0;i<capture_count;i++) args[i+3]=captures[i];
    CHECK(hipEventRecord(start,stream));
    CHECK(hipModuleLaunchKernel(kernel,(unsigned)((n-1)/256+1),1,1,256,1,1,0,stream,args,nullptr));
    CHECK(hipEventRecord(stop,stream)); CHECK(hipEventSynchronize(stop));
    CHECK(hipEventElapsedTime(&elapsed,start,stop));
    kernels++; seconds+=(double)elapsed/1000.0;
    CHECK(hipMemcpyAsync(output,b,bytes,hipMemcpyDeviceToHost,stream));
    CHECK(hipStreamSynchronize(stream));
cleanup:
    // Synchronize before freeing buffers even after a failed launch/event call.
    if(stream) { hipError_t e=hipStreamSynchronize(stream); if(error==hipSuccess && e!=hipSuccess) { error=e; stage="hipStreamSynchronize"; } }
    #define RELEASE(expr) do { hipError_t e=(expr); if(error==hipSuccess && e!=hipSuccess) { error=e; stage=#expr; } } while(0)
    if(stop) RELEASE(hipEventDestroy(stop));
    if(start) RELEASE(hipEventDestroy(start));
    if(b) RELEASE(hipFree(b));
    if(a) RELEASE(hipFree(a));
    if(module) RELEASE(hipModuleUnload(module));
    if(stream) RELEASE(hipStreamDestroy(stream));
#undef RELEASE
    free(args);
    if(error!=hipSuccess) { fprintf(stderr,"Severian GPU: %s: %s\n",stage,hipGetErrorString(error)); }
#undef CHECK
    return (int32_t)error;
}
