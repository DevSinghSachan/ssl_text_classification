#!/usr/bin/env bash

unit=$1
path_=$2

for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14
do
    echo $i
    num_lines=$(($i * $unit))
    echo $num_lines
    sort -R $path_ | head -${num_lines} > ${path_}.${i}
done

