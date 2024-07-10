#!/bin/bash

module load toolchain/system
#module load toolchain/gcc-2023.2
module load scorep
module load scorep_metricq
module load scorep_x86_energy
#module use $HOME/privatemodules

FAST_HOME=/fastfs/s5770874
HOME=/home/s5770874


export SCOREP_METRIC_METRICQ_PLUGIN_TIMEOUT="10000s"
export SCOREP_METRIC_METRICQ_PLUGIN="elab.ariel.power,elab.ariel.s0.package.power.100Hz,elab.ariel.s1.package.power.100Hz,elab.ariel.s0.dram.power.100Hz,elab.ariel.s1.dram.power.100Hz,elab.ariel.board.3V.power.100Hz,elab.ariel.board.5V.power.100Hz,elab.ariel.fan.power.100Hz,elab.ariel.sum.power.100Hz"
#export SCOREP_METRIC_METRICQ_PLUGIN="elab.ariel.power,elab.ariel.s0.dram.power.100Hz,elab.ariel.s1.dram.power.100Hz"

cd $HOME/roco2/build/src/configurations/ariel


make -j SCOREP_WRAPPER_INSTRUMENTER_FLAGS="--user --openmp --thread=omp --nocompiler" || exit 1

cd $FAST_HOME/roco2/ariel/traces

#export SCOREP_MACHINE_NAME='Ariel'

export SCOREP_ENABLE_PROFILING=false
export SCOREP_ENABLE_TRACING=true
#export SCOREP_METRIC_PLUGINS=x86_energy_plugin,metricq_plugin  #automatically loaded
export SCOREP_TOTAL_MEMORY=4G #MAX 4G - PAGE_SIZE (8k)

export GOMP_CPU_AFFINITY=0-71

#echo $LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/scorep_plugin_x86_energy/build
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/tilsche/github/scorep_plugin_x86_energy/BUILD
#echo $LD_LIBRARY_PATH


/home/s5770874/roco2/build/src/configurations/ariel/roco2_ariel -d # DEBUG
#/home/s5770874/roco2/build/src/configurations/ariel/roco2_ariel


ls /fastfs/s5770874/roco2/ariel/traces/




##Permission problems (LEGACY)
#ulimit -n unlimited
#sudo perf probe -d roco2:metrics
#sudo perf probe -x ./roco2_ariel roco2:metrics=_ZN5roco27metrics4meta5writeEmmlm experiment frequency shell threads || exit 1
#printenv

#LD_PRELOAD=/opt/global/22.04/amd64/fftw/3.3.8/lib/libfftw3.so GOMP_CPU_AFFINITY=0-71 lo2s -X -t roco2:metrics -o "/fastfs/s5770874/roco2/ariel/lo2s_trace_{DATE}" ./roco2_ariel

#lo2s -A system monitor mode


