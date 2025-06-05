#ifndef INCLUDE_ROCO2_KERNELS_GPU_MATMUL_HPP
#define INCLUDE_ROCO2_KERNELS_GPU_MATMUL_HPP

#include <roco2/kernels/base_kernel.hpp>
#include <roco2/memory/thread_local.hpp>
#include <roco2/metrics/utility.hpp>

#include <chrono>

#include <cuda_runtime.h>
//__global__ void matrixMulNaive(const float* A, const float* B, float* C, int N);

void allocGPUMemory(double*& d_Mem, size_t bytes);
void copyDataToGPU(double* d_Mem, double* h_Mem, size_t bytes);
void copyDataToHost(double* h_Mem, double* d_Mem, size_t bytes);
void initDimensions(int block_size, int matrix_size);
void cleanGPUMemory(double* d_Mem);
void runGPUMatrix(const double* d_A, const double* d_B, double* d_C, int N, dim3 gridDim, dim3 blockDim);
void runDevSync();
//void runGPUMatrix(const float* d_A, const float* d_B, float* d_C, int N, int block_size);


/*inline void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "Error: " << msg << " (" << cudaGetErrorString(err) << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}
*/

//void initialize_matrix(double* M, int rows, int cols, std::function<float()> F) {

/*----------------------------------------------------------*/

namespace roco2
{
namespace kernels
{
    class gpu_matmul : public base_kernel
    {
        virtual experiment_tag tag() const override
        {
            return 100;
        }

        virtual void run_kernel(chrono::time_point until) override
        {
#ifdef HAS_SCOREP
            SCOREP_USER_REGION("gpu_matmul_kernel", SCOREP_USER_REGION_TYPE_FUNCTION)
#endif

/*            double* A = roco2::thread_local_memory().mat_A.data();
            double* B = roco2::thread_local_memory().mat_B.data();
            double* C = roco2::thread_local_memory().mat_C.data();

            uint64_t m = roco2::thread_local_memory().mat_size;
*/
    std::cerr << "trying to setup vars" << std::endl;
    const int N = 64; // Matrix size
    const int block_size = 32; // Block size
    size_t bytes = N * N * sizeof(double);

        std::cerr << "vars setup" << std::endl;
 
	auto& mem = roco2::thread_local_memory();
	mem.mat_A.resize(N * N);

        std::cerr << "alloc h_A" << std::endl;
//	    double* h_A = roco2::thread_local_memory().mat_A.data();
	    double* h_A = mem.mat_A.data();
        std::cerr << "alloc h_B" << std::endl;
//            double* h_B = roco2::thread_local_memory().mat_B.data();
            double* h_B = mem.mat_B.data();
        std::cerr << "alloc h_C" << std::endl;
//            double* h_C = roco2::thread_local_memory().mat_C.data();
            double* h_C = mem.mat_C.data();

        std::cerr << "alloc m" << std::endl;
            uint64_t m = roco2::thread_local_memory().mat_size;

        std::cerr << "allocated h_mems m=" << m << std::endl;

    size_t m_bytes = m * sizeof(double);

            std::size_t loops = 0;

   // Allocate host memory
/*    float *h_A = new float[N * N];
    float *h_B = new float[N * N];
    float *h_C = new float[N * N];
*/
/*
        std::cerr << "trying to init matrices" << std::endl;
    // Initialize matrices
//    for (int i = 0; i < N * N; ++i) {
    for (int i = 0; i < m; ++i) {
//std::cerr << "Debug h_A:" << h_A[i] << std::endl;
	h_A[i] = static_cast<double>(rand()) / RAND_MAX;
        h_B[i] = static_cast<double>(rand()) / RAND_MAX;
        //h_A[i] = static_cast<double>(2);
        //h_B[i] = static_cast<double>(3);
    }
        std::cerr << "init matrices" << std::endl;
*/

    // Allocate device memory
//    double *d_A, *d_B, *d_C;
/*	
        std::cerr << "alloc d_A" << std::endl;
    double* d_A = roco2::thread_local_memory().mat_A.data();
        std::cerr << "alloc d_B" << std::endl;
    double* d_B = roco2::thread_local_memory().mat_B.data();
        std::cerr << "alloc d_C" << std::endl;
    double* d_C = roco2::thread_local_memory().mat_C.data();
*/
//    uint64_t m = roco2::thread_local_memory().mat_size;



    double *d_A, *d_B, *d_C;
    int matrix_size = N * N * sizeof(double);


  std::cerr << "alloc Device Memory" << std::endl;

        std::cerr << "before d_A" << std::endl;
    allocGPUMemory(d_A, matrix_size);
        std::cerr << "d_A" << std::endl;
    allocGPUMemory(d_B, matrix_size);
        std::cerr << "d_B" << std::endl;
    allocGPUMemory(d_C, matrix_size);
        std::cerr << "d_C" << std::endl;

  std::cerr << "synchronizing device" << std::endl;
 runDevSync();

  std::cerr << "trying to init matrices with size:" << matrix_size << std::endl;
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


/*    checkCudaError(cudaMalloc(&d_A, bytes), "Allocating d_A");
    checkCudaError(cudaMalloc(&d_B, bytes), "Allocating d_B");
    checkCudaError(cudaMalloc(&d_C, bytes), "Allocating d_C");
*/

/*
    // Copy data to device
    std::cerr << "before_copy h_A" << std::endl;
   copyDataToGPU(d_A, h_A, m_bytes);
    std::cerr << "h_A copied" << std::endl;
   copyDataToGPU(d_B, h_B, m_bytes);
    std::cerr << "h_B copied" << std::endl;
*/
/*    checkCudaError(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice), "Copying h_A to d_A");
    checkCudaError(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice), "Copying h_B to d_B");
*/
    // Launch parameters
//	initDimensions(32, N);
    std::cerr << "setting up dims with m="<< m << std::endl;
    dim3 blockDim(32, 32); // 1024 threads per block for 32,32 
    dim3 gridDim((m + blockDim.x - 1) / blockDim.x,
                 (m + blockDim.y - 1) / blockDim.y);
/*    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (N + blockDim.y - 1) / blockDim.y);
*/

    std::cerr << "dims setup" << std::endl;

    std::cerr << "run GPU Matrix" << std::endl;
            do
            {
		//cudaDeviceSynchronize();

/*#ifdef HAS_SCOREP
                // SCOREP_USER_REGION("matmul_kernel_loop", SCOREP_USER_REGION_TYPE_FUNCTION)
#endif

#if defined USE_MKL || defined USE_BLAS
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, m, m, 1.0, A, m, B, m,
                            1.0, C, m);
#else
                // ACML
                dgemm('N', 'N', m, m, m, 1.0, A, m, B, m, 1.0, C, m);
#endif
*/

//		matrixMulNaive<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
//        	checkCudaError(cudaGetLastError(), "Kernel launch");
//        	checkCudaError(cudaDeviceSynchronize(), "Device synchronize");

//    std::cerr << "run GPU Matrix" << std::endl;
		//runGPUMatrix(d_A, d_B, d_C, m, blockDim, gridDim);
		runGPUMatrix(d_A, d_B, d_C, N, 1, 1);
//    std::cerr << "GPU Matrix successfully run. Loop counter:" << loops << std::endl;
                loops++;
            } while (std::chrono::high_resolution_clock::now() < until);

    std::cerr << "GPU Matrix successfully run. Loop counter:" << loops << std::endl;
//	    runDevSync();
//    std::cerr << "Device Synchronized" << loops << std::endl;

	    roco2::metrics::utility::instance().write(loops);

    	    // Copy result back
    std::cerr << "d_C before copy" << std::endl;
	    copyDataToHost(h_C, d_C, m_bytes);
    std::cerr << "d_C copied to host" << std::endl;
//    	    checkCudaError(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost), "Copying d_C to h_C");

//    std::cerr << "DEBUG: no clean" << std::endl;

    /*
    std::cerr << " cleaning up d_A" << std::endl;
	    // Cleanup
	    cleanGPUMemory(d_A);
    std::cerr << " cleaning up d_B" << std::endl;
	    cleanGPUMemory(d_B);
    std::cerr << " cleaning up d_C" << std::endl;
	    cleanGPUMemory(d_C);

*/
/*    	    cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_C);
*/
/*    std::cerr << " delete d_A" << std::endl;
            delete[] d_A;
    std::cerr << " delete d_B" << std::endl;
            delete[] d_B;
    std::cerr << " delete d_C" << std::endl;
            delete[] d_C;
    std::cerr << " cleaning up h_A" << std::endl;
            delete[] h_A;
    std::cerr << " cleaning up h_B" << std::endl;
            delete[] h_B;
    std::cerr << " cleaning up h_C" << std::endl;
            delete[] h_C;
*/
	std::cerr << " all cleaned." << std::endl;

        }
    };
} // namespace kernels
} // namespace roco2

#endif // INCLUDE_ROCO2_KERNELS_MATMUL_HPP
