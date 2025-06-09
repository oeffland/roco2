#ifndef INCLUDE_ROCO2_KERNELS_GPU_MATMUL_HPP
#define INCLUDE_ROCO2_KERNELS_GPU_MATMUL_HPP

#include <roco2/kernels/base_kernel.hpp>
#include <roco2/memory/thread_local.hpp>
#include <roco2/metrics/utility.hpp>

#include <chrono>

#include <cuda_runtime.h>

void allocGPUMemory(float*& d_Mem, size_t bytes);
void copyDataToGPU(float* d_Mem, float* h_Mem, size_t bytes);
void copyDataToHost(float* h_Mem, float* d_Mem, size_t bytes);
void initDimensions(int block_size, int matrix_size);
void cleanGPUMemory(float*& d_Mem);
void runGPUMatrix(const float* d_A, const float* d_B, float* d_C, int N, dim3 gridDim, dim3 blockDim);
void runDevSync();
void setGPUDevices(int device_id);
void allocAllGPUs();
void resetGPU();

/*----------------------------------------------------------*/

namespace roco2
{
namespace kernels
{
    class gpu_matmul : public base_kernel
    {    
	float *d_A, *d_B, *d_C;
        
	const int N = 16384; // Matrix size 56816 for 14GB
        int block_size = 256;
	int grid_size = (N + block_size - 1) / block_size;

        std::size_t loops = 0;
        int tid = omp_get_thread_num();

	public: gpu_matmul()
        {
//        int tid_ctor = omp_get_thread_num();
	#pragma omp master
	    {
            size_t bytes = N * N * sizeof(float);


            int cpu_id = omp_get_thread_num();
            int device_id = 1;
            setGPUDevices(device_id);

            int matrix_size = N * N * sizeof(float);

            std::cerr << "alloc Device Memory" << std::endl;

            allocGPUMemory(d_A, matrix_size);
            allocGPUMemory(d_B, matrix_size);
            allocGPUMemory(d_C, matrix_size);

            for (int row = 0; row < N; ++row)
            {
                for (int col = 0; col < N; ++col)
                {
                    d_A[row * N + col] = row;
                    d_B[row * N + col] = col + 2;
                    d_C[row * N + col] = 0;
                }
            }

            runDevSync();
	    }
        }

	public: virtual ~gpu_matmul()
 	{
/*	#pragma omp master
	    {
	    // Cleanup
            std::cerr << " cleaning up d_A" << std::endl;
	    cleanGPUMemory(d_A);
            std::cerr << " cleaning up d_B" << std::endl;
	    cleanGPUMemory(d_B);
            std::cerr << " cleaning up d_C" << std::endl;
	    cleanGPUMemory(d_C);
	
            std::cerr << " all cleaned." << std::endl;
	    cudaDeviceReset();
	    }
*/
	#pragma omp master
	{
	    resetGPU();
	    std::cerr << " Device reset." << std::endl;
	}
	}

        virtual experiment_tag tag() const override
        {
            return 100;
        }

        virtual void run_kernel(chrono::time_point until) override
        {

#ifdef HAS_SCOREP
            SCOREP_USER_REGION("gpu_matmul_kernel", SCOREP_USER_REGION_TYPE_FUNCTION)
#endif

	 

#pragma omp master
    {
//	std::cerr << "PRAGMA tid: " << tid << std::endl;
    std::cerr << "synchronizing device" << std::endl;

    runDevSync();

    std::cerr << "run GPU Matrix with gridsize: " << grid_size << " blocksize: "<< block_size << std::endl;

    do
            {
		for (int i = 0; i < 1000; i++)
		{
		    runGPUMatrix(d_A, d_B, d_C, N, grid_size, block_size);
		}
		runDevSync();
                loops++;
            } while (std::chrono::high_resolution_clock::now() < until);

	    roco2::metrics::utility::instance().write(loops);
	    std::cerr << "GPU Matrix successfully run. Loop counter:" << loops << std::endl;

	    loops = 0;
     }
     
	    std::this_thread::sleep_until(until);
     }
    };
} // namespace kernels
} // namespace roco2

#endif // INCLUDE_ROCO2_KERNELS_MATMUL_HPP
