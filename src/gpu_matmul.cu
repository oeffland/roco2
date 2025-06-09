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
__global__ void matrixMulNaive(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
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

void runGPUMatrix(const float* d_A, const float* d_B, float* d_C, int N, dim3 gridDim, dim3 blockDim) {
	//for (int i = 0; i < 100; i++){
		matrixMulNaive<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
	//}
	//cudaDeviceSynchronize();
//      checkCudaError(cudaGetLastError(), "Kernel launch");
       	//checkCudaError(cudaDeviceSynchronize(), "Device synchronize");

}

void runDevSync(){
	cudaDeviceSynchronize();
	//       	checkCudaError(cudaDeviceSynchronize(), "Device synchronize");
}

void setGPUDevices(int device_id){
//    int numDevices;
//    checkCudaError(cudaGetDeviceCount(&numDevices), "getDeviceCount");
//	std::cerr << "Device id: " << device_id << " of " << numDevices << "devices." << std::endl;
//    int device = numDevices;
    checkCudaError(cudaSetDevice(device_id), "set Deviceid");
	std::cerr << "Device id: " << device_id << std::endl;
}

// Allocate device memory
//void allocGPUMemory(float** d_Mem, size_t bytes){
void allocGPUMemory(float*& d_Mem, size_t bytes){
	//checkCudaError(cudaMalloc(&d_Mem, bytes), "Allocating d_Mem with cudaMalloc"); ???
	checkCudaError(cudaMallocManaged(&d_Mem, bytes), "Allocating d_Mem with cudaMallocManaged");
}

void allocAllGPUs(){
/*	std::cerr << "alloc Device Memory" << std::endl;

    	int numDevices;
	checkCudaError(cudaGetDeviceCount(&numDevices), "getDeviceCount");
	int device_id=0;
	std::cerr << "Device id: " << device_id << " of " << numDevices << "devices." << std::endl;

   for(i=0;i<numDevices;i++){
	
	checkCudaError(cudaSetDevice(device_id), "set Deviceid");
    allocGPUMemory(d_A, matrix_size);
    allocGPUMemory(d_B, matrix_size);
    allocGPUMemory(d_C, matrix_size);
   }
  std::cerr << "synchronizing device" << std::endl;
 runDevSync();


//  std::cerr << "trying to init matrices with size:" << matrix_size << std::endl;
    // Initialize memory
  for( int row = 0; row < N; ++row )
    for( int col = 0; col < N; ++col )
    {
//      std::cerr << "init mem: " << row*N+col << " with: " << row << " " <<col+2 << std::endl;
      d_A[row*N + col] = row;
//      std::cerr << "a" << std::endl;
      d_B[row*N + col] = col+2;
//      std::cerr << "b" << std::endl;
      d_C[row*N + col] = 0;
//      std::cerr << "c" << std::endl;
      
    }
  std::cerr << "matrices initialized" << std::endl;

    checkCudaError(cudaMallocManaged(&d_Mem, bytes), "Allocating d_Mem");
*/
}

// Copy data to device
void copyDataToGPU(float* d_Mem, float* h_Mem, size_t bytes){
    checkCudaError(cudaMemcpy(d_Mem, h_Mem, bytes, cudaMemcpyHostToDevice), "Copying h_Mem to d_Mem");
//    cudaGetLastError();
}

void copyDataToHost(float* h_Mem, float* d_Mem, size_t bytes){
    checkCudaError(cudaMemcpy(h_Mem, d_Mem, bytes, cudaMemcpyDeviceToHost), "Copying d_Mem to h_Mem");
}

// Launch parameters
void initDimensions(int block_size, int matrix_size){
	dim3 blockDim(block_size, block_size);				// threads per block
	dim3 gridDim((matrix_size + blockDim.x - 1) / blockDim.x,
                 (matrix_size + blockDim.y - 1) / blockDim.y);		// number of blocks
}

//Cleanup
void cleanGPUMemory(float*& d_Mem){
	if(d_Mem)
       checkCudaError(cudaFree(d_Mem), "Cleanup of d_Mem");
}

void resetGPU()
{
	cudaDeviceReset();
}
/*void getDevInfo(){
	getDeviceInformation();
}
*/
