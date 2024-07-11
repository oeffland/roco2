#!/bin/bash

TRACE_PATH=/fastfs/s5770874/roco2/ariel/traces
#TRACE_DIR=scorep-20240625_1347_4898415570676158
#TRACE_DIR=scorep-20240705_1416_7489463016118988
#TRACE_DIR=scorep-20240708_1425_8266808195885830
#TRACE_DIR=scorep-20240708_1631_8289445284276060 #long run
#TRACE_DIR=scorep-20240708_1733_8300613610854800 #ddcm, pattern changed #works
#TRACE_DIR=scorep-20240708_2035_8333297766291720 #long run

#TRACE_DIR=scorep-20240709_1338_8516977145884998 #debug
#TRACE_DIR=scorep-20240709_1347_8518504960119530 #debug with different subblocks
#TRACE_DIR=scorep-20240709_1725_8557755467099280
#TRACE_DIR=scorep-20240709_1745_8561283425740418
#TRACE_DIR=scorep-20240709_1801_8564179377166020
#TRACE_DIR=scorep-20240709_1838_8570728764053368
#TRACE_DIR=scorep-20240709_2159_8606890814827778	#long run with +10s? #works
#TRACE_DIR=scorep-20240710_1337_8775290426116378	
#TRACE_DIR=scorep-20240710_1719_8815261660553388
#TRACE_DIR=scorep-20240710_1831_8828214792004398
#TRACE_DIR=scorep-20240710_1853_8832052445479878	#1s 4,12 works
#TRACE_DIR=scorep-20240710_1923_8837546615891018	#1s 2,18 fails
#TRACE_DIR=scorep-20240710_2257_8875851842018490	#1s 2,18 fails
#TRACE_DIR=scorep-20240710_1853_8832052445479878		#1s 4,12 works
#TRACE_DIR=scorep-20240710_2350_8885357817028168		#100ms 2,18

#TRACE_DIR=scorep-20240711_1633_9065453631901528		#1s 2,18 debug fails
TRACE_DIR=scorep-20240711_1653_9069107088749808			#1s 4,12 debug fails differently

module load toolchain/system
#module load toolchain/gcc-2023.2
#module load scorep

ulimit -n 999999

mpirun --mca mpi_yield_when_idle 1 -oversubscribe -np 85 \
	/home/s9242987/Software/haec-sim/build-lo2s/main/haec_sim \
		-c /home/s5770874/roco2/src/configurations/ariel/haec_sim.conf \
		-m phase_profile \
		-V info \
	$TRACE_PATH/$TRACE_DIR/traces.otf2

#	/fastfs/s5770874/roco2/ariel/traces/scorep-20240625_1347_4898415570676158/traces.otf2

#mpirun --mca mpi_yield_when_idle 1 -oversubscribe -np 101 \
#    /home/s9242987/Software/haec-sim/build-lo2s/main/haec_sim \
#    -c ./haec_sim.json -m phase_profile -V info \
#    /home/s9242987/roco2-kallisto/build/src/configurations/kallisto/lo2s_trace_2022-06-23T16-19-08/traces.otf2
