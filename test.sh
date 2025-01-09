#!/bin/bash

start=0
end=10
step=0.25

value=$start
while (( $(echo "$value <= $end" | bc -l) )); do
    echo "111$value"
    value=$(echo "$value + $step" | bc)
    echo "222$value"
done
