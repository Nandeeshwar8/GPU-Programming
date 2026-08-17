

#include <stdio.h>
#include <cuda_runtime.h>

__device__ void customBarrier(int *counter,
                              int *sense,
                              int *localSense)
{
    *localSense = 1 - *localSense;

    int arrived = atomicAdd(counter, 1);

    if (arrived == blockDim.x - 1)
    {
        atomicExch(counter, 0);
        atomicExch(sense, *localSense);
    }
    else
    {
        while (*sense != *localSense)
        {
        }
    }

    __syncthreads();
}


__global__ void barrierKernel(int *counter, int *sense)
{
    int tid = threadIdx.x;

    int localSense = 0;

    printf("Thread %d reached first part\n", tid);

    if (tid % 2 == 0)
    {
        printf("Thread %d is doing work before barrier\n", tid);
    }

    customBarrier(counter, sense, &localSense);

    printf("Thread %d passed the barrier\n", tid);

    if (tid % 2 == 1)
    {
        printf("Thread %d is doing work after barrier\n", tid);
    }

    customBarrier(counter, sense, &localSense);

    printf("Thread %d completed execution\n", tid);
}


int main()
{
    int *d_counter;
    int *d_sense;

    int counter = 0;
    int sense = 0;

    cudaMalloc((void **)&d_counter, sizeof(int));
    cudaMalloc((void **)&d_sense, sizeof(int));

    cudaMemcpy(d_counter,
               &counter,
               sizeof(int),
               cudaMemcpyHostToDevice);

    cudaMemcpy(d_sense,
               &sense,
               sizeof(int),
               cudaMemcpyHostToDevice);

    barrierKernel<<<1, 8>>>(d_counter, d_sense);

    cudaDeviceSynchronize();

    cudaFree(d_counter);
    cudaFree(d_sense);

    return 0;
}