#!/usr/bin/env bash

functions=("01" "02" "03" "04" "05" "06" "07" "08" "09" "10" "11" "12" "13" "14" "15" "16" "17" "18")

for i in ${functions[*]}
do
    tmux kill-session -t C${i}
    tmux new-session -d -s C${i} ./run.sh -r 25 -f C${i} -c ./config.ini -l ./logs/run2/C${i}
done