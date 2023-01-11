#!/bin/bash
if [ $# -eq 0 ];
then
    echo "$0: Missing arguments: $@"
    exit 1
elif [ $# -gt 1 ];
then
    echo "$0: Too many arguments: $@"
    exit 1
else
    if [ $1=='dp' ];
    then 
        echo "Enter data processing step"
        python3 data_processing.py
    elif [ $1=='ht' ];
    then 
         echo "Enter hyperparameter tunning step"
         python3 hyperparameter_tuning.py
    elif [ $1=='p' ];
    then 
         echo "Enter prediction step"
         python3 prediction.py
    fi

fi