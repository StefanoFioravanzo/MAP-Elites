#!/usr/bin/env bash

POSITIONAL=()
while [[ $# -gt 0 ]]
do
    key="$1"

    case $key in
        -p|--parallel)
        PARALLEL="$2"
        shift # past argument
        shift # past value
        ;;
        -c|--configfile)
        CONFIGFILE="$2"
        shift # past argument
        shift # past value
        ;;
        -l|--logdir)
        LOGDIR="$2"
        shift # past argument
        shift # past value
        ;;
#        --default)
#        DEFAULT=YES
#        shift # past argument
#        ;;
        *)    # unknown option
        POSITIONAL+=("$1") # save it in an array for later
        shift # past argument
        ;;
    esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

for i in $(seq 1 ${PARALLEL})
do
    echo Run ${i}
    mkdir -p ${LOGDIR}
    python mapelites_continuous_opt.py --conf ${CONFIGFILE} --logdir ${LOGDIR}/${i}
    echo
done