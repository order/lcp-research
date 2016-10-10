#!/bin/bash

R=25
E=30

make
for J in 0 10 15 20 25
do
    for B in false
    do
        echo ./minop_flow_refine -o ~/data/minop/flow_refine_${J}_$B \
                            -j$J -r$R -E$E -b$B -e 0.15
        ./minop_flow_refine -o ~/data/minop/flow_refine_${J}_$B \
                            -j$J -r$R -E$E -b$B -e 0.15
    done
done
