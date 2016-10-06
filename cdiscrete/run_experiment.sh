#!/bin/bash

R=25
E=32

make
for J in 0 10 20 30
do
    for B in 0 1
    do
        ./minop_flow_refine -o ~/data/minop/flow_refine_$J_$B \
                            -j$J -r$R -E$E -b$B
    done
done
