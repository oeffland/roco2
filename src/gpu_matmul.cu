#include <roco2/kernels/base_kernel.hpp>
#include <roco2/memory/thread_local.hpp>
#include <roco2/metrics/utility.hpp>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

/*#if __has_include(<mkl_cblas.h>)
#include <mkl_cblas.h>
#define USE_MKL
#elif __has_include(<cblas.h>)
#include <cblas.h>
#define USE_BLAS
#elif __has_include(<acml.h>)
#include <acml.h>
#define USE_ACML
#endif
*/

#include <chrono>

//#define ROW_TILE_WIDTH 32
//#define COL_TILE_WIDTH 32

//N = matrix_size
__global__ void matrixMulNaive(const double* A, const double* B, double* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        double sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "Error: " << msg << " (" << cudaGetErrorString(err) << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

void runGPUMatrix(const double* d_A, const double* d_B, double* d_C, int N, dim3 gridDim, dim3 blockDim) {

	
//        std::cerr << "calling matrixMulNaive" << " N:" << N << std::endl;
	matrixMulNaive<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
//        std::cerr << "matrixMulNaive successfully run" << std::endl;
       	checkCudaError(cudaGetLastError(), "Kernel launch");
       	//checkCudaError(cudaDeviceSynchronize(), "Device synchronize");

}

void runDevSync(){
       	checkCudaError(cudaDeviceSynchronize(), "Device synchronize");
}

// Allocate device memory
//void allocGPUMemory(double** d_Mem, size_t bytes){
void allocGPUMemory(double*& d_Mem, size_t bytes){
	//checkCudaError(cudaMalloc(&d_Mem, bytes), "Allocating d_Mem");
    checkCudaError(cudaMallocManaged(&d_Mem, bytes), "Allocating d_Mem");
}

// Copy data to device
void copyDataToGPU(double* d_Mem, double* h_Mem, size_t bytes){
    checkCudaError(cudaMemcpy(d_Mem, h_Mem, bytes, cudaMemcpyHostToDevice), "Copying h_Mem to d_Mem");
//    cudaGetLastError();
}

void copyDataToHost(double* h_Mem, double* d_Mem, size_t bytes){
    checkCudaError(cudaMemcpy(h_Mem, d_Mem, bytes, cudaMemcpyDeviceToHost), "Copying d_Mem to h_Mem");
}

// Launch parameters
void initDimensions(int block_size, int matrix_size){
	dim3 blockDim(block_size, block_size);				// threads per block
	dim3 gridDim((matrix_size + blockDim.x - 1) / blockDim.x,
                 (matrix_size + blockDim.y - 1) / blockDim.y);		// number of blocks
}

//Cleanup
void cleanGPUMemory(double* d_Mem){
       checkCudaError(cudaFree(d_Mem), "Cleanup of d_Mem");
}

/*void getDevInfo(){
	getDeviceInformation();
}
*/
