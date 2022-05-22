#include <cuda.h>
#include <iostream>
#include <stdio.h>

__global__ void KernelMul(int N, float* x, float* y, float* result) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N; i += stride) {
        result[i] = x[i] * y[i];
    }
}

__global__ void Reduce(float* in_data, float* out_data) {
    extern  __shared__ float shared[];

    unsigned int tid = threadIdx.x;
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    shared[tid] = in_data[index];
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out_data[blockIdx.x] = shared[0];
    }
}

float KernelScalarMul(int N, float* vector1, float* vector2, int blockSize) {
    int numBlocks = (N + blockSize - 1) / blockSize;
    int size = N * sizeof(float);

    float *vec1_d = nullptr;
    float *vec2_d = nullptr;
    float *result_d = nullptr;
    float *reduce1_d = nullptr;
    float *out_d = nullptr;

    cudaMalloc(&vec1_d, size);
    cudaMalloc(&vec2_d, size);
    cudaMalloc(&result_d, size);
    cudaMalloc(&reduce1_d, numBlocks * sizeof(float));
    cudaMalloc(&out_d, sizeof(float));
    
    cudaMemcpy(vec1_d, vector1, size, cudaMemcpyHostToDevice);
    cudaMemcpy(vec2_d, vector2, size, cudaMemcpyHostToDevice);

        KernelMul<<<numBlocks, blockSize>>>(N, vec1_d, vec2_d, result_d);
        cudaDeviceSynchronize();

    Reduce<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(result_d, reduce1_d);
        cudaDeviceSynchronize();

    const int numBlocksReduce = (numBlocks + blockSize - 1) / blockSize;
    Reduce<<<numBlocksReduce, blockSize, blockSize * sizeof(float)>>>(reduce1_d, out_d);
        cudaDeviceSynchronize();

    float result = 0;
    cudaMemcpy(&result, out_d, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(vec1_d);
    cudaFree(vec2_d);
    cudaFree(result_d);
    cudaFree(reduce1_d);
    cudaFree(out_d);

    return result;
}

int main() {
    float *vec1_h = nullptr, *vec2_h = nullptr;
    int N = 10;

    vec1_h = (float*)malloc(N * sizeof(float));
    vec2_h = (float*)malloc(N * sizeof(float));


    for (int i = 0; i < N; i++) {
        vec1_h[i] = 1;
        vec2_h[i] = -1;
    }

    float ret_val, first, second;

    first = sqrt(KernelScalarMul(N, vec1_h, vec1_h, 256));
    second = sqrt(KernelScalarMul(N, vec2_h, vec2_h, 256));
    ret_val = KernelScalarMul(N, vec1_h, vec2_h, 256) / (first * second);

    std::cout << "ret_val" << " : " << ret_val << std::endl;

    free(vec1_h);
    free(vec2_h);

    return(0);
}
