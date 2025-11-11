#!/bin/bash
CURRENT_DIR=$(pwd)

export CUDA_DEVICE_MAX_CONNECTIONS=1

if [ $# -eq 0 ]; then
   echo "$0 {start|stop}"
   exit 1
fi	

# $2 set configuration name

if [ "$2" == "llama" ]; then
	case $1 in
		start)
		echo "Start a training"
		python $CURRENT_DIR/FlagScale/run.py --config-path $CURRENT_DIR/configs/OpenSeek-Small-v1-Baseline --config-name config_deepseek_v3_1_4b.yaml
		;;
		profile)
		echo "Start a profiling"
		nsys profile --wait=all -s none -t nvtx,cuda -o $CURRENT_DIR/build/profile_aquila --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop \
		python $CURRENT_DIR/FlagScale/run.py --config-path $CURRENT_DIR/examples/llama/conf --config-name config
		;;
		stop)
		echo "Stop a training"
		python $CURRENT_DIR/FlagScale/run.py --config-path $CURRENT_DIR/examples/llama/conf --config-name config action=stop
		;;
	esac
else
	case $1 in
		start)
		echo "Start a training in 1"
		python3 $CURRENT_DIR/FlagScale/run.py --config-path $CURRENT_DIR/configs/OpenSeek-Small-v1-Baseline --config-name config_deepseek_v3_1_4b.yaml
		;;
		profile)
		echo "Start a profiling"
		nsys profile --wait=all -s none -t nvtx,cuda -o $CURRENT_DIR/build/profile_aquila --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop \
		python3 $CURRENT_DIR/FlagScale/run.py --config-path configs --config-name $2
		;;
		stop)
		echo "Stop a training"
		python3 $CURRENT_DIR/FlagScale/run.py --config-path configs --config-name $2 action=stop
		;;
	esac
fi

