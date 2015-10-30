#!/bin/bash
./split -p 3 -e 0.001 train.txt
./sendtotrain.py machinefile train.txt temp_dir
mpiexec -n 3 --machinefile machinefile ./nomad-q -t 4 /home/jing/dis_data/train.txt.sub
