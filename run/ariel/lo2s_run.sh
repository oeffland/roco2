#!/bin/bash
#module load lo2s/2019-05-06
module load lo2s
module load scorep_metricq

module load oneapi/2024.1
module load tbb/2021.12
module load compiler-rt/2024.1.0
module load mkl/2024.1

export SCOREP_METRIC_METRICQ_PLUGIN_TIMEOUT="1h"
#export SCOREP_METRIC_METRICQ_PLUGIN="elab.ariel.power,elab.ariel.s0.package.power.100Hz,elab.ariel.s1.package.power.100Hz,elab.ariel.s0.dram.power.100Hz,elab.ariel.s1.dram.power.100Hz"
export SCOREP_METRIC_METRICQ_PLUGIN="elab.ariel.power,elab.ariel.s0.package.power.100Hz,elab.ariel.s1.package.power.100Hz,elab.ariel.s0.dram.power.100Hz,elab.ariel.s1.dram.power.100Hz,elab.ariel.board.3V.power.100Hz,elab.ariel.board.5V.power.100Hz,elab.ariel.fan.power.100Hz,elab.ariel.sum.power.100Hz"

cd /home/s5770874/roco2/build/src/configurations/ariel

make -j || exit 1

##Permission problems
#ulimit -n unlimited
#sudo perf probe -d roco2:metrics
#sudo perf probe -x ./roco2_ariel roco2:metrics=_ZN5roco27metrics4meta5writeEmmlm experiment frequency shell threads || exit 1

#printenv

LD_PRELOAD=/opt/global/22.04/amd64/fftw/3.3.8/lib/libfftw3.so GOMP_CPU_AFFINITY=0-71 lo2s -X -t roco2:metrics -o "/fastfs/s5770874/roco2/ariel/lo2s_trace_{DATE}" ./roco2_ariel

#-A system monitor mode


