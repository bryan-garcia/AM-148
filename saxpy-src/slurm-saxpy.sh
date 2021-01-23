#!/bin/bash
#SBATCH --job-name=saxpy
#SBATCH --output=saxpy_test.log
#SBATCH --partition=am-148-s20
#SBATCH --mail-type=ALL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=bgarci26@ucsc.edu   # Where to send mail
#SBATCH --nodes=1                    # Number of nodes

srun make
srun ./saxpy
