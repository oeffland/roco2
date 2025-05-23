#!/bin/bash
module load release/24.04  GCC/13.2.0
module load OpenBLAS/0.3.24
module load protobuf/25.3

module load OpenMPI/4.1.6
module load Score-P/8.4-CUDA-12.4.0


module load CMake
#module spider OpenBLAS/0.3.24
#module spider Score-P/8.4-CUDA-12.4.0


#module load scorep
#module load scorep_metricq

echo "Test."

export SCOREP_ENABLE_PROFILING=false
export SCOREP_ENABLE_TRACING=true
export SCOREP_TOTAL_MEMORY=4095M

export OMP_NUM_THREADS=128
export GOMP_CPU_AFFINITY=0-127
#export SCOREP_METRIC_PLUGINS=nvml_plugin
#export SCOREP_METRIC_NVML_PLUGIN="utilization_gpu,power_usage"

#cd $HOME/roco2/build/src/configurations/ariel

#export SCOREP_METRIC_METRICQ_PLUGIN_TIMEOUT="24h"
export SCOREP_METRIC_METRICQ_PLUGIN="hpc.capella.*.cpu.power,hpc.capella.*.gpu.power,hpc.capella.*.mem.power,hpc.capella.*.sys.power"
#export SCOREP_METRIC_METRICQ_PLUGIN="elab.ariel.power,elab.ariel.s0.package.power.100Hz,elab.ariel.s1.package.power.100Hz,elab.ariel.s0.dram.power.100Hz,elab.ariel.s1.dram.power.100Hz"

#make -j || exit 1

#ulimit -n unlimited
#sudo perf probe -d roco2:metrics
#sudo perf probe -x ./roco2_ariel roco2:metrics=_ZN5roco27metrics4meta5writeEmmlm experiment frequency shell threads || exit 1
#LD_PRELOAD=/opt/global/18.04/fftw/3.3.8/lib/libfftw3.so GOMP_CPU_AFFINITY=0-71 lo2s -X -t roco2:metrics -o "/fastfs/tilsche/roco2/ariel/lo2s_trace_{DATE}" ./roco2_ariel

