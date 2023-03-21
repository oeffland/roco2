#!/bin/sh

source /etc/profile.d/lmod.sh
source /etc/profile.d/zih-a-lmod.sh

module purge --force
module load toolchain/system scorep_metricq lo2s elab

echo "it is $(date)"

export GOMP_CPU_AFFINITY=0-23
export OPENBLAS_NUM_THREADS=1

export SCOREP_ENABLE_TRACING=1
export SCOREP_ENABLE_PROFILING=0
export SCOREP_TOTAL_MEMORY=4095M
export SCOREP_METRIC_METRICQ_PLUGIN_TIMEOUT=12h

echo "environment variables:"
echo "  GOMP_CPU_AFFINITY                    = $GOMP_CPU_AFFINITY"
echo "  SCOREP_ENABLE_TRACING                = $SCOREP_ENABLE_TRACING"
echo "  SCOREP_ENABLE_PROFILING              = $SCOREP_ENABLE_PROFILING"
echo "  SCOREP_TOTAL_MEMORY                  = $SCOREP_TOTAL_MEMORY"
echo "  SCOREP_METRIC_PLUGINS                = $SCOREP_METRIC_PLUGINS"
echo "  SCOREP_METRIC_METRICQ_PLUGIN_TIMEOUT = $SCOREP_METRIC_METRICQ_PLUGIN_TIMEOUT"

echo "executing test..."

ulimit -n 999999
elab frequency turbo

perf probe -d roco2:metrics

sudo perf probe -x ./roco2_kallisto roco2:metrics=_ZN5roco27metrics4meta5writeEmmlmmmm experiment frequency shell threads utility || exit 1

GOMP_CPU_AFFINITY=0-23 lo2s \
-X -t roco2:metrics \
-- /home/s9242987/roco2-kallisto/build/src/configurations/kallisto/roco2_kallisto
echo "done"