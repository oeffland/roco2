#!/bin/bash

TRACE_PATH=/fastfs/s5770874/roco2/ariel/traces
#TRACE_DIR=scorep-20240705_1354_7485447833175980
TRACE_DIR=scorep-20240705_1416_7489463016118988

module load toolchain/system
#module load toolchain/gcc-2023.2
#module load scorep

ulimit -n 999999

mpirun --mca mpi_yield_when_idle 1 -oversubscribe -np 85 \
	/home/s9242987/Software/haec-sim/build-lo2s/main/haec_sim \
		-c /home/s5770874/roco2/src/configurations/ariel/haec_sim.conf \
		-m phase_profile \
		-V debug \
	$TRACE_PATH/$TRACE_DIR/traces.otf2

#	/fastfs/s5770874/roco2/ariel/traces/scorep-20240625_1347_4898415570676158/traces.otf2

#mpirun --mca mpi_yield_when_idle 1 -oversubscribe -np 101 \
#    /home/s9242987/Software/haec-sim/build-lo2s/main/haec_sim \
#    -c ./haec_sim.json -m phase_profile -V info \
#    /home/s9242987/roco2-kallisto/build/src/configurations/kallisto/lo2s_trace_2022-06-23T16-19-08/traces.otf2
