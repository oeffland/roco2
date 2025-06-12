#ifndef INCLUDE_ROCO2_KERNELS_GPU_MATMUL_HPP
#define INCLUDE_ROCO2_KERNELS_GPU_MATMUL_HPP

#include <roco2/kernels/base_kernel.hpp>
#include <roco2/memory/thread_local.hpp>
#include <roco2/metrics/utility.hpp>

#include <chrono>

#include <vector>
#include <cuda_runtime.h>

void allocGPUMemory(float*& d_Mem, size_t bytes);
void prefetchGPUMemory(float* d_Mem, size_t bytes, int deviceId);
void copyDataToGPU(float* d_Mem, float* h_Mem, size_t bytes);
void copyDataToHost(float* h_Mem, float* d_Mem, size_t bytes);
void initDimensions(int block_size, int matrix_size);
void cleanGPUMemory(float*& d_Mem);
void runGPUMatrix(const float* d_A, const float* d_B, float* d_C, int N, dim3 gridDim, dim3 blockDim);
void runDevSync();
void setGPUDevices(int device_id);
void createStream(cudaStream_t* stream);
void syncStream(cudaStream_t stream);
void destroyStream(cudaStream_t stream);
void launchKernelStream(const std::vector<int>& devices, std::vector<cudaStream_t>& streams, int grid_size, int block_size, const float* d_A, const float* d_B, float* d_C, int N);
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
	cudaStream_t stream1, stream2, stream3, stream4;
        
	const int N = 16384; // Matrix size 56816 for 14GB
        int block_size = 256;
	int grid_size = (N + block_size - 1) / block_size;

        std::size_t loops = 0;
        int tid = omp_get_thread_num();
	int t_num = omp_get_num_threads();
	int t_num_max = omp_get_max_threads();

	int thread_counter=0;
	int device_counter=1;
	int device_count=4;
	size_t NUM_DEVICES = 4;
    	std::vector<int> devices = {0, 1, 2, 3}; // Specify your device IDs
    	std::vector<cudaStream_t> streams;
    	//std::vector<cudaStream_t> streams(NUM_DEVICES, 0);

	public: gpu_matmul()
        {
//        int tid_ctor = omp_get_thread_num();
	#pragma omp master
	    {
            size_t bytes = N * N * sizeof(float);


            int cpu_id = omp_get_thread_num();
	//    std::cerr << "t_id=" << t_id << " t_num=" << t_num << " t_num_max" << t_num_max << std::endl;
            int device_id = 0;
            setGPUDevices(device_id);

            int matrix_size = N * N * sizeof(float);

            std::cerr << "alloc Device Memory" << std::endl;

            allocGPUMemory(d_A, matrix_size);
            allocGPUMemory(d_B, matrix_size);
            allocGPUMemory(d_C, matrix_size);

	    prefetchGPUMemory(d_A, matrix_size, device_id);
	    prefetchGPUMemory(d_B, matrix_size, device_id);
	    prefetchGPUMemory(d_C, matrix_size, device_id);

	    createStream(&stream1);
	    createStream(&stream2);
	    createStream(&stream3);
	    createStream(&stream4);

            for (int row = 0; row < N; ++row)
            {
                for (int col = 0; col < N; ++col)
                {
                    d_A[row * N + col] = row;
                    d_B[row * N + col] = col + 2;
                    d_C[row * N + col] = 0;
                }
            }

            streams.resize(NUM_DEVICES, 0);
	
//	if (err != cudaSuccess) {
    	    // handle error
//	     std::cerr << "adding stream to vector streams failed" << std::endl;
//	} else {
	    streams.push_back(stream1);
	    streams.push_back(stream2);
	    streams.push_back(stream3);
	    streams.push_back(stream4);
//	}



            runDevSync(); //works
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
	    destroyStream(stream1);
	    destroyStream(stream2);
	    destroyStream(stream3);
	    destroyStream(stream4);

	    for (int i = 0; i < device_count; ++i) {
      		setGPUDevices(i);
		runDevSync();
	    	resetGPU();
    	    }
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

	 

//	std::cerr << "t_id=" << t_id << " t_num=" << t_num << " t_num_max" << t_num_max << std::endl;
//#pragma omp single nowait
//if (tid == 0 || tid == 14)
#pragma omp master
{
	//std::cerr << "t_id=" << tid << " t_num=" << t_num << " t_num_max" << t_num_max << std::endl;
//if (tid == 0 || tid == 14 || tid == 28 || tid == 42)
//	{

//	std::cerr << "t_id=" << tid << " survived." << std::endl;
//	std::cerr << "PRAGMA tid: " << tid << std::endl;
//    std::cerr << "synchronizing device" << std::endl;

    if(thread_counter<3){
    std::cerr << "A" << thread_counter << std::endl;
//	setGPUDevices(0);
	NUM_DEVICES=1;
    	devices = {0};
    } else if(thread_counter<8) {
    std::cerr << "B" << thread_counter << std::endl;
//	setGPUDevices(1);
	NUM_DEVICES=2;
    	devices = {0,1};
    } else if(thread_counter<12) {
    std::cerr << "C" << thread_counter << std::endl;
//	setGPUDevices(2);
	NUM_DEVICES=3;
    	devices = {0,1,2};
//	device_counter=3;
    } else if(thread_counter<16) {
    std::cerr << "D" << thread_counter << std::endl;
//	setGPUDevices(3);
	//device_counter=2;
	NUM_DEVICES=4;
    	devices = {0,1,2,3};
	//device_counter=4;
    } else {
    std::cerr << "E" << thread_counter << std::endl;
//    	device_counter=4;
//	setGPUDevices(0);
	NUM_DEVICES=4;
    	devices = {0,1,2,3};
    }


//      setGPUDevices(tid%13);
      runDevSync();	//works


    std::cerr << "run GPU Matrix with gridsize: " << grid_size << " blocksize: "<< block_size << std::endl;
//    std::cerr << "thread_counter=" << thread_counter << std::endl;
   
    do
            {
		/*for (int i = 0; i < 1000; i++)
		{
		    runGPUMatrix(d_A, d_B, d_C, N, grid_size, block_size);
		}*/
		launchKernelStream(devices, streams, grid_size, block_size, d_A, d_B, d_C, N);
		
		runDevSync();
                loops++;
            } while (std::chrono::high_resolution_clock::now() < until);

	    roco2::metrics::utility::instance().write(loops);
	    std::cerr << "GPU Matrix successfully run. Loop counter:" << loops << std::endl;
		
	    loops = 0;
	    thread_counter++;

	    syncStream(stream1);
	    syncStream(stream2);
	    syncStream(stream3);
	    syncStream(stream4);
     } //else{
     
//	    std::cerr << "tid=" << tid << " going to sleep" << std::endl;
	    std::this_thread::sleep_until(until);
	//}
     } //run_kernel
    }; //gpu_matmul
} // namespace kernels
} // namespace roco2

#endif // INCLUDE_ROCO2_KERNELS_MATMUL_HPP
