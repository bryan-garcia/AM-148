#!/bin/bash
#SBATCH --job-name=matmult     # Job name
#SBATCH --output=matmult.log
#SBATCH --partition=am-148-s20
#SBATCH --mail-type=END      # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=bgarci26@ucsc.edu   # Where to send mail
#SBATCH --nodes=1                    # Number of nodes

srun make
srun ./matmult
