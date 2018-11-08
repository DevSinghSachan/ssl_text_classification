#!/usr/bin/env bash

unit=$1
path_=$2

for i in 5 10 15 20 25
do
    echo $i
    num_lines=$(($i * $unit))
    echo $num_lines
    sort -R $path_ | head -${num_lines} > ${path_}.${i}
done