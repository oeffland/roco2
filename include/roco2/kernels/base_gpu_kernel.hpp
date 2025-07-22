#pragma once

#include <roco2/chrono/util.hpp>
#include <roco2/experiments/gpu_sets/gpu_set.hpp>
#include <roco2/metrics/experiment.hpp>
#include <roco2/metrics/metric_guard.hpp>
#include <roco2/metrics/threads.hpp>
#include <roco2/metrics/utility.hpp>
#include <roco2/scorep.hpp>

#include <vector>

#include <cassert>

#include <omp.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

namespace roco2
{
namespace kernels
{

    class base_gpu_kernel
    {
    public:
        using experiment_tag_t = std::size_t;

        void setup_streams(const roco2::experiments::gpu_sets::gpu_set& on)
        {
            if (omp_get_thread_num() == 0)
            {
                assert(streams_.empty());

                for (std::size_t gpu_id = 0; gpu_id < on.max(); gpu_id++)
                {
                    if (!on.contains(gpu_id))
                    {
                        streams_.push_back(nullptr);
                        continue;
                    }

                    cudaSetDevice(gpu_id);

                    cudaStream_t stream;
                    cudaStreamCreate(&stream);

                    std::cout << "schedule_task from setup_streams for device: " << gpu_id << std::endl;

                    schedule_task(gpu_id, stream);

                    streams_.push_back(stream);
                }
            }
        }

        virtual void schedule_task(int gpu_id, cudaStream_t s) = 0;

        void progress()
        {
            #ifdef HAS_SCOREP
                SCOREP_USER_REGION("progress", SCOREP_USER_REGION_TYPE_FUNCTION)
            #endif

            for (int gpu_id = 0; (unsigned)gpu_id < streams_.size(); gpu_id++)
            {
                auto& stream = streams_[gpu_id];

                if (stream == nullptr)
                    continue;

                cudaSetDevice(gpu_id);

                if (cudaStreamQuery(stream) == cudaSuccess)
                {
                    std::cout << "schedule_task from progress for device: " << gpu_id << std::endl;
                    schedule_task(gpu_id, stream);
                }
            }
        }

        virtual experiment_tag_t tag() const = 0;

        void stop()
        {
            for (auto stream : streams_)
            {
                cudaStreamSynchronize(stream);
                cudaStreamDestroy(stream);
            }

            streams_.clear();
        }

    private:
        std::vector<cudaStream_t> streams_;

    public:
        virtual ~base_gpu_kernel()
        {
        }
    };
} // namespace kernels
} // namespace roco2
