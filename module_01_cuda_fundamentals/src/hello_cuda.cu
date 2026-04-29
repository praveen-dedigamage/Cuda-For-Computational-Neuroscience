/*
 * hello_cuda.cu
 * Module 01 — GPU Architecture & CUDA Fundamentals
 *
 * Prints thread identity from each GPU thread.
 * Demonstrates: kernel launch syntax, thread/block indexing, printf from GPU.
 *
 * Compile:  nvcc -o hello_cuda hello_cuda.cu
 * Run:      ./hello_cuda
 */

#include <stdio.h>
#include <cuda_runtime.h>

// ─────────────────────────────────────────────────────────────────────────────
// Error-checking macro — wraps every CUDA API call
// ─────────────────────────────────────────────────────────────────────────────
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));                \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// ─────────────────────────────────────────────────────────────────────────────
// Kernel: runs on the GPU, once per thread
// ─────────────────────────────────────────────────────────────────────────────
__global__ void hello_kernel()
{
    // Each thread computes its unique global index
    int block_id  = blockIdx.x;
    int thread_id = threadIdx.x;
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    printf("Hello from block %2d, thread %2d  →  global id = %3d\n",
           block_id, thread_id, global_id);
}

// ─────────────────────────────────────────────────────────────────────────────
// Host (CPU) code
// ─────────────────────────────────────────────────────────────────────────────
int main()
{
    // Query and print GPU information
    int device_id;
    CUDA_CHECK(cudaGetDevice(&device_id));

    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, device_id));

    printf("GPU: %s\n", props.name);
    printf("Streaming Multiprocessors: %d\n", props.multiProcessorCount);
    printf("Max threads per block:     %d\n", props.maxThreadsPerBlock);
    printf("Warp size:                 %d\n", props.warpSize);
    printf("\n");

    // Launch configuration
    int num_blocks  = 4;
    int threads_per_block = 8;    // total = 32 threads (one warp)

    printf("Launching %d blocks × %d threads = %d total threads\n\n",
           num_blocks, threads_per_block, num_blocks * threads_per_block);

    // <<<grid_size, block_size>>> — the kernel launch syntax
    hello_kernel<<<num_blocks, threads_per_block>>>();

    // Wait for all GPU threads to finish before reading output
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("\nDone.\n");
    return 0;
}
