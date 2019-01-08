#!/usr/bin/env bash

POSITIONAL=()
while [[ $# -gt 0 ]]
do
    key="$1"

    case $key in
        -d|--delete)
        DELETE_SESSIONS=True
        shift # past argument
        ;;
        -l|--logpath)
        LOGPATH="$2"
        shift # past argument
        shift # past value
        ;;
        *)    # unknown option
        POSITIONAL+=("$1") # save it in an array for later
        shift # past argument
        ;;
    esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

functions=("01" "02" "03" "04" "05" "06" "07" "08" "09" "10" "11" "12" "13" "14" "15" "16" "17" "18")

if [[ -z ${LOGPATH+x} ]]
then
    echo Log path not set
else
    echo Logging job to ${LOGPATH}
fi

for i in ${functions[*]}
do
    if [[ -z ${DELETE_SESSIONS+x} ]]
    then
        echo Creating ${i}
        tmux new-session -d -s C${i} ./run.sh -r 25 -f C${i} -c ./config.ini -l ./logs/run2/C${i}
    else
        echo Deleting ${i}
        tmux kill-session -t C${i}
    fi
done