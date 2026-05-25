#!/bin/bash

pid=$(ps -ef | grep src/main | grep -v grep | awk '{print $2}')
if [ -z "$pid" ]; then
    echo 'not need stop for none runing task'

else
    kill -9 $pid

    pid=$(ps -ef | grep src/main | grep -v grep | awk '{print $2}')
    if [ -n "$pid" ]; then
        echo 'stop fail ...'
        ps -ef | grep src/main | grep -v grep
    else
        echo 'stop success.'
    fi

fi
