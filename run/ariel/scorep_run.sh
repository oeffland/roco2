#!/bin/bash
#module load lo2s/2019-05-06
#module load lo2s

FAST_HOME=/fastfs/s5770874
HOME=/home/s5770874

module load toolchain/gcc-2023.2
module load scorep
module load scorep_metricq
#module use $HOME/privatemodules
module load scorep_x86_energy

export SCOREP_METRIC_METRICQ_PLUGIN_TIMEOUT="1h"
export SCOREP_METRIC_METRICQ_PLUGIN="elab.ariel.power,elab.ariel.s0.package.power.100Hz,elab.ariel.s1.package.power.100Hz,elab.ariel.s0.dram.power.100Hz,elab.ariel.s1.dram.power.100Hz,elab.ariel.board.3V.power.100Hz,elab.ariel.board.5V.power.100Hz,elab.ariel.fan.power.100Hz,elab.ariel.sum.power.100Hz"
export 

cd $HOME/roco2/build/src/configurations/ariel


# in build dir CC=scorep-gcc ..cmake
make -j SCOREP_WRAPPER_INSTRUMENTER_FLAGS="--user --thread=omp" || exit 1

cd $FAST_HOME/roco2/ariel/traces

export SCOREP_MACHINE_NAME='Ariel'

export SCOREP_ENABLE_TRACING=true
#export SCOREP_METRIC_PLUGINS=x86_energy_plugin
export SCOREP_TOTAL_MEMORY=360G #MAX 4G - 8k
export GOMP_CPU_AFFINITY=0-71

echo $LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/scorep_plugin_x86_energy/build
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/tilsche/github/scorep_plugin_x86_energy/BUILD

echo $LD_LIBRARY_PATH


/home/s5770874/roco2/build/src/configurations/ariel/roco2_ariel





#LD_LIBRARY_PATH=/home/s5770874/scorep_plugin_x86_energy/build GOMP_CPU_AFFINITY=0-71 SCOREP_ENABLE_TRACING=true scorep ./roco2_ariel

#printenv LD_LIBRARY_PATH

##Permission problems
#ulimit -n unlimited
#sudo perf probe -d roco2:metrics
#sudo perf probe -x ./roco2_ariel roco2:metrics=_ZN5roco27metrics4meta5writeEmmlm experiment frequency shell threads || exit 1
#printenv

#LD_PRELOAD=/opt/global/22.04/amd64/fftw/3.3.8/lib/libfftw3.so GOMP_CPU_AFFINITY=0-71 lo2s -X -t roco2:metrics -o "/fastfs/s5770874/roco2/ariel/lo2s_trace_{DATE}" ./roco2_ariel

#lo2s -A system monitor mode


