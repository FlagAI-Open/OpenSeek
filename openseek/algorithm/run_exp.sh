#!/bin/bash

if [ $# -eq 0 ]; then
   echo "$0 {start|stop}"
   exit 1
fi	

# $2 set configuration name

if [ "$2" == "llama" ]; then
	case $1 in
		start)
		echo "Start a traing"
		python run.py --config-path ./examples/llama/conf --config-name config
		;;
		profile)
		echo "Start a profiling"
		nsys profile --wait=all -s none -t nvtx,cuda -o ./build/profile_aquila --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop \
		python run.py --config-path ./examples/llama/conf --config-name config
		;;
		stop)
		echo "Stop a traing"
		python run.py --config-path ./examples/llama/conf --config-name config action=stop
		;;
	esac
else
	case $1 in
		start)
		echo "Start a traing"
		python3 run.py --config-path configs --config-name $2
		;;
		profile)
		echo "Start a profiling"
		nsys profile --wait=all -s none -t nvtx,cuda -o ./build/profile_aquila --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop \
		python3 run.py --config-path configs --config-name $2
		;;
		stop)
		echo "Stop a traing"
		python3 run.py --config-path configs --config-name $2 action=stop
		;;
	esac
fi

