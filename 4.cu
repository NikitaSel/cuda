#include <iostream>
#include <iomanip>

void FillMatrix(float* matrix, int height, int width) {
        for (int i = 0; i < height; ++i) {
                for (int j = 0; j < width; ++j) {
                        if (i == j) {
                                matrix[i * width + j] = 1;
                        } else {
                                matrix[i * width + j] = 0;
                        }
                }
        }
}

void FillVector(float* vector, int length) {
        for (int i = 0; i < length; ++i) {
                vector[i] = i;
        }
}


void PrintMatrix(const float *matrix, int height, int width, char* text) {
        std::cout << text << std::endl;
        for (int i = 0; i < height; ++i) {
                for (int j = 0; j < width; ++j) {
                        std::cout << std::setw(2) << matrix[i * width + j];
                }
                std::cout << std::endl;
        }
}

void PrintVector(const float *vector, int length, char* text) {
        std::cout << text << std::endl;
        for (int i = 0; i < length; ++i) {
                std::cout << std::setw(2) << vector[i];
        }
        std::cout << std::endl;
}

__global__
void KernelMatrixAdd(float* A, float* B, float* C, int num_rows, int num_cols) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
           
        int height = blockDim.x * gridDim.x;
        int width = blockDim.y * gridDim.y;
                    
        int n_l = (num_rows + height - 1) / height;
        int n_m = (num_cols + width - 1) / width;  
    
        int curr_l_shift = 0, curr_m_shift = 0;
        for (int l = 0; l < n_l; ++l) {
                curr_m_shift = 0;
                for (int m = 0; m < n_m; ++m) {
                        int curr_ind = (i + curr_l_shift) * width + (j + curr_m_shift);
                        atomicAdd(&C[i + curr_l_shift], A[curr_ind] * B[j + curr_m_shift]);
                        curr_m_shift += blockDim.y * gridDim.y;
                }
                curr_l_shift += blockDim.x * gridDim.x;
        }
}

int main() {

        float *h_A;
        float *h_B;
        float *h_C;

        int num_rows = 5, num_cols = 5;

        h_A = new float[num_rows * num_cols];
        h_B = new float[num_cols];
        h_C = new float[num_rows];

        FillMatrix(h_A, num_rows, num_cols);
        FillVector(h_B, num_cols);

        float* d_A;
        float* d_B;
        float* d_C;

        cudaMalloc(&d_A, sizeof(float) * num_rows * num_cols);
        cudaMalloc(&d_B, sizeof(float) * num_cols);
        cudaMalloc(&d_C, sizeof(float) * num_rows);

        cudaMemcpy(d_A, h_A, sizeof(float) * num_rows * num_cols, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, sizeof(float) * num_cols, cudaMemcpyHostToDevice);

        dim3 num_blocks(1, 1);
        dim3 block_size(5, 5);

        KernelMatrixAdd<<<num_blocks, block_size>>>(d_A, d_B, d_C, num_rows, num_cols);

        cudaMemcpy(h_C, d_C, sizeof(float) * num_rows, cudaMemcpyDeviceToHost);

        PrintMatrix(h_A, num_rows, num_cols, "M");
        PrintVector(h_B, num_cols, "V");
        PrintVector(h_C, num_rows, "Vres");

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);

        delete[] h_A;
        delete[] h_B;
        delete[] h_C;

        return 0;
}