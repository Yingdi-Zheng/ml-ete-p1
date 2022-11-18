if [ $# -eq 0 ];
then
    python3 data_processing.py
    python3 hyperparameter_tuning.py
    # todo: add all functions
elif [ $# -gt 1 ];
then
    echo "$0: Too many arguments: $@"
    exit 1
else
    if [ $1 == 'dp' ];
    then 
        echo "Enter data processing step"
        python3 data_processing.py
    elif [ $1 == 'ht' ];
    then 
         echo "Enter hyperparameter tunning step"
         python3 hyperparameter_tuning.py
    # todo: add all functions
    fi

fi