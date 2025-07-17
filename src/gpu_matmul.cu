#include <roco2/kernels/base_kernel.hpp>
#include <roco2/memory/thread_local.hpp>
#include <roco2/metrics/utility.hpp>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

#include <chrono>


/*__global__ void trig_muladd(double *result, int iterations, volatile bool *cudaStopFlag){
    while (true) {
        if (atomicAdd((int*)cudaStopFlag, 0) != 0) {
            return;
        }
        double sum = 0.0;

    for (int i = 0; i < 1000; ++i) {
        sum += sin(i) * tan(i) + cos(i);
    }
    result[idx] = sum;
        
    }
*/

__global__ void matrixMulNaive(const float* A, const float* B, float* C, int N, volatile bool* cudaStopFlag) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    while (true) {
        if(atomicAdd((int*) cudaStopFlag, 0) != 0){
            return; //exit kernel
        }
        for (int i=0; i<100; i++) {
    	    if (row < N && col < N) {
        	    double sum = 0.0;
        	    for (int k = 0; k < N; ++k) {
            		sum += A[row * N + k] * B[k * N + col];
        	    }
        	    C[row * N + col] = sum;
    	    }
        }
    }

}
/*
void runTrigKernel(double* result, int iterations, volatile bool *cudaStopFlag, dim3 gridDim, dim3 blockDim){
    trig_muladd<<<gridDim,blockDim>>>(result, iterations);
}*/

void runGPUMatrix(const float* d_A, const float* d_B, float* d_C, volatile bool* cudaStopFlag, int N, dim3 gridDim, dim3 blockDim) {
		matrixMulNaive<<<gridDim, blockDim>>>(d_A, d_B, d_C, N, cudaStopFlag);
}


/*void runGPUMatrix(const float* d_A, const float* d_B, float* d_C, int N, dim3 gridDim, dim3 blockDim) {
		matrixMulNaive<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
}
*/

void launchKernelStream(const std::vector<int>& devices, std::vector<cudaStream_t>& streams, int grid_size, int block_size, const float* d_A, const float* d_B, float* d_C, int N, volatile bool* cudaStopFlag) {
    int numDevices = devices.size();
    // Ensure streams vector has the correct size
    if (streams.size() != numDevices) {
        streams.resize(numDevices);
    }

    for (int i = 0; i < numDevices; ++i) {
        cudaSetDevice(devices[i]);
//        // Create stream if not already created
//        if (streams[i] == nullptr) {
//            cudaStreamCreate(&streams[i]);
//        }
        // Launch kernel asynchronously on each device
	//for (int j = 0; j < streams.size(); j++) {
        	matrixMulNaive<<<dim3(grid_size), dim3(block_size), 0, streams[i]>>>(d_A, d_B, d_C, N, cudaStopFlag);
	//}
    }
}

void runDevSync(){
	cudaDeviceSynchronize();
}

void setGPUDevices(int device_id){
    cudaSetDevice(device_id);
	std::cerr << "Device id: " << device_id << std::endl;
}

// Allocate device memory
/*template <typename T>
void allocGPUMemory(T*& d_Mem, size_t count){
	cudaMallocManaged(&d_Mem, count * sizeof(T));
}
*/
void allocGPUMemory(float*& d_Mem, size_t bytes){
	cudaMallocManaged(&d_Mem, bytes);
}

void allocGPUMemory(volatile bool*& d_Mem, size_t bytes){
    cudaMallocManaged(&d_Mem, bytes);
}

void prefetchGPUMemory(float* d_Mem, size_t bytes, int deviceId){
	cudaMemPrefetchAsync(d_Mem, bytes, deviceId);
}

void createStream(cudaStream_t* stream) {
	cudaStreamCreate(stream);
}

void syncStream(cudaStream_t stream){
	cudaStreamSynchronize(stream);
}

void destroyStream(cudaStream_t stream) {
	cudaStreamDestroy(stream);
}

void allocAllGPUs(){
}

// Copy data to device
void copyDataToGPU(float* d_Mem, float* h_Mem, size_t bytes){
    cudaMemcpy(d_Mem, h_Mem, bytes, cudaMemcpyHostToDevice);
}

void copyDataToHost(float* h_Mem, float* d_Mem, size_t bytes){
    cudaMemcpy(h_Mem, d_Mem, bytes, cudaMemcpyDeviceToHost);
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
       cudaFree(d_Mem);
}

void resetGPU()
{
	cudaDeviceReset();
}
