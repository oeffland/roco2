#ifndef INCLUDE_ROCO2_KERNELS_KERNEL_HPP
#define INCLUDE_ROCO2_KERNELS_KERNEL_HPP

#include <roco2/chrono/util.hpp>
#include <roco2/cpu/info.hpp>
#include <roco2/experiments/cpu_sets/cpu_set.hpp>
#include <roco2/metrics/experiment.hpp>
#include <roco2/metrics/metric_guard.hpp>
#include <roco2/metrics/threads.hpp>
#include <roco2/metrics/utility.hpp>
#include <roco2/scorep.hpp>

#include <algorithm>
#include <chrono>
#include <thread>
#include <vector>

namespace roco2
{
namespace kernels
{
    template <typename return_functor>
    class base_kernel
    {
    public:
        using experiment_tag = std::size_t;

        void run(return_functor& cond, roco2::experiments::cpu_sets::cpu_set on)
        {
            if (std::find(on.begin(), on.end(), roco2::cpu::info::current_thread()) != on.end())
            {
                roco2::metrics::metric_guard<roco2::metrics::experiment> guard(this->tag());
                roco2::metrics::threads::instance().write(on.num_threads());

                this->run_kernel(cond);

                cond.sync_working();
            }
            else
            {
                roco2::metrics::metric_guard<roco2::metrics::experiment> guard(2);
                roco2::metrics::threads::instance().write(on.num_threads());

                SCOREP_USER_REGION("idle_sleep", SCOREP_USER_REGION_TYPE_FUNCTION)

                cond.sync_idle();

                roco2::metrics::utility::instance().write(1);
            }
        }

        virtual experiment_tag tag() const = 0;

    private:
        virtual void run_kernel(return_functor& cond) = 0;

    public:
        virtual ~base_kernel()
        {
        }
    };
}
}

#endif // INCLUDE_ROCO2_KERNELS_KERNEL_HPP
